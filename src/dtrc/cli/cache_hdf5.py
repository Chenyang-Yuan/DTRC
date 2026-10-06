"""Encode HDF5 RGB observations with a frozen LeWM encoder."""

from __future__ import annotations
import argparse
import json
import math
import os
import importlib
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    importlib.import_module("hdf5plugin")
except Exception:
    pass
from dtrc.backends.lewm import load_lewm_backend, preprocess_uint8
from dtrc.environments.lewm import get_task


def resolve_h5(dataset: str, cache_dir: str | None) -> Path:
    try:
        import stable_worldmodel as swm

        source = swm.data.load_dataset(dataset, cache_dir=cache_dir)
        if hasattr(source, "h5_path"):
            return Path(source.h5_path).resolve()
    except Exception as error:
        print(
            f"[cache] stable_worldmodel lookup failed ({error}); using STABLEWM_HOME",
            flush=True,
        )
    home = cache_dir or os.environ.get("STABLEWM_HOME")
    if home is None:
        raise FileNotFoundError("set STABLEWM_HOME or pass --cache-dir")
    path = Path(home) / "datasets" / dataset
    if not path.exists():
        raise FileNotFoundError(path)
    return path.resolve()


class H5PixelChunks(Dataset):

    def __init__(self, h5_path, start, stop, chunk_size):
        self.h5_path = str(h5_path)
        (self.start, self.stop, self.chunk_size) = (
            int(start),
            int(stop),
            int(chunk_size),
        )
        self._h5 = None

    def __len__(self):
        return math.ceil(max(self.stop - self.start, 0) / self.chunk_size)

    def __getitem__(self, item):
        if self._h5 is None:
            self._h5 = h5py.File(
                self.h5_path, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024
            )
        begin = self.start + int(item) * self.chunk_size
        end = min(begin + self.chunk_size, self.stop)
        return (
            begin,
            torch.from_numpy(np.asarray(self._h5["pixels"][begin:end]).copy()),
        )


def atomic_json(path: Path, payload) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp, path)


def episode_bounds(episodes: np.ndarray, steps: np.ndarray):
    rows = len(episodes)
    episode_start = np.arange(rows, dtype=np.int64) - steps
    boundaries = np.flatnonzero(np.r_[episodes[1:] != episodes[:-1], True])
    starts = np.r_[0, boundaries[:-1] + 1]
    episode_end = np.empty(rows, dtype=np.int64)
    for start, end in zip(starts, boundaries):
        episode_end[start : end + 1] = end
    if np.any(episode_start != np.repeat(starts, boundaries - starts + 1)):
        raise ValueError("step_idx is inconsistent with episode boundaries")
    return (episode_start, episode_end)


def write_index(h5_path: Path, output_dir: Path, max_episodes=None) -> dict:
    with h5py.File(h5_path, "r", swmr=True) as handle:
        actions = np.asarray(handle["action"], dtype=np.float32)
        steps = np.asarray(handle["step_idx"], dtype=np.int64).reshape(-1)
        ep_key = "ep_idx" if "ep_idx" in handle else "episode_idx"
        episodes = np.asarray(handle[ep_key], dtype=np.int64).reshape(-1)
        pixel_shape = tuple((int(x) for x in handle["pixels"].shape[1:]))
        columns = sorted(handle.keys())
    if max_episodes is not None:
        if max_episodes < 1:
            raise ValueError("max-episodes must be positive")
        ends = np.flatnonzero(np.r_[episodes[1:] != episodes[:-1], True]) + 1
        stop = int(ends[min(max_episodes, len(ends)) - 1])
        actions, steps, episodes = actions[:stop], steps[:stop], episodes[:stop]
    if actions.ndim == 1:
        actions = actions[:, None]
    actions = np.nan_to_num(actions, nan=0.0)
    (episode_start, episode_end) = episode_bounds(episodes, steps)
    np.savez(
        output_dir / "index.npz",
        action=actions,
        episode_idx=episodes,
        step_idx=steps,
        episode_start=episode_start,
        episode_end=episode_end,
    )
    return {
        "rows": int(len(actions)),
        "action_dim": int(actions.shape[1]),
        "action_mean": actions.mean(axis=0, dtype=np.float64)
        .astype(np.float32)
        .tolist(),
        "action_std": np.maximum(actions.std(axis=0, dtype=np.float64), 1e-06)
        .astype(np.float32)
        .tolist(),
        "num_episodes": int(len(np.unique(episodes))),
        "pixel_shape": list(pixel_shape),
        "h5_columns": columns,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--dataset", default=None, help="override the task's training dataset"
    )
    parser.add_argument(
        "--lewm-checkpoint",
        required=True,
        help="HF directory (weights.pt + config.json) or SWM checkpoint name",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--read-chunk-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--progress-every", type=int, default=20000)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Encode only the first complete episodes for a short pipeline check",
    )
    parser.add_argument(
        "--h5", default=None, help="explicit HDF5 path (skips SWM lookup)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = get_task(args.task)
    dataset = args.dataset or spec["train_dataset"]
    h5_path = (
        Path(args.h5).resolve() if args.h5 else resolve_h5(dataset, args.cache_dir)
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    latents_path = output_dir / "latents.npy"
    progress_path = output_dir / "progress.json"
    meta_path = output_dir / "meta.json"
    if meta_path.exists() and (not progress_path.exists()) and latents_path.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            previous.get("h5_path"),
            previous.get("max_episodes"),
            previous.get("lewm_checkpoint"),
        ) != (str(h5_path), args.max_episodes, str(args.lewm_checkpoint)):
            raise ValueError(
                "Existing cache uses different inputs; choose a new output directory"
            )
        print(f"[cache] already complete: {output_dir}")
        return
    index_meta = write_index(h5_path, output_dir, args.max_episodes)
    row_count = index_meta["rows"]
    backend = load_lewm_backend(args.lewm_checkpoint, device=args.device)
    feature_dim = backend.latent_dim
    torch.backends.cuda.matmul.allow_tf32 = True
    start_row = 0
    total = np.zeros(feature_dim, dtype=np.float64)
    square = np.zeros(feature_dim, dtype=np.float64)
    if latents_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            progress.get("rows"),
            progress.get("feature_dim"),
            progress.get("dataset"),
        ) != (row_count, feature_dim, dataset):
            raise ValueError("resume metadata mismatch; delete the cache dir")
        start_row = int(progress["completed_rows"])
        total = np.asarray(progress["sum"], dtype=np.float64)
        square = np.asarray(progress["square"], dtype=np.float64)
        latents = np.lib.format.open_memmap(latents_path, mode="r+")
        print(f"[cache] resuming at {start_row}/{row_count}", flush=True)
    else:
        latents = np.lib.format.open_memmap(
            latents_path, mode="w+", dtype=np.float32, shape=(row_count, feature_dim)
        )
    loader_kwargs = dict(
        dataset=H5PixelChunks(h5_path, start_row, row_count, args.read_chunk_size),
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(**loader_kwargs)
    started = time.time()
    last_report = start_row
    with torch.inference_mode():
        for begin, raw_chunk in loader:
            begin = int(begin)
            encoded = []
            for offset in range(0, len(raw_chunk), args.batch_size):
                pixels = preprocess_uint8(
                    raw_chunk[offset : offset + args.batch_size].to(backend.device)
                )
                encoded.append(
                    backend.encode_pixels(pixels, batch_size=args.batch_size)
                    .cpu()
                    .numpy()
                )
            encoded = np.concatenate(encoded, axis=0)
            end = begin + len(encoded)
            latents[begin:end] = encoded
            values = encoded.astype(np.float64)
            total += values.sum(axis=0)
            square += np.square(values).sum(axis=0)
            if end == row_count or end - last_report >= args.progress_every:
                latents.flush()
                atomic_json(
                    progress_path,
                    {
                        "dataset": dataset,
                        "rows": row_count,
                        "feature_dim": feature_dim,
                        "completed_rows": end,
                        "sum": total.tolist(),
                        "square": square.tolist(),
                    },
                )
                rate = (end - start_row) / max(time.time() - started, 1e-06)
                print(f"[cache] {end}/{row_count} ({rate:.1f} rows/s)", flush=True)
                last_report = end
    latents.flush()
    mean = total / row_count
    variance = np.maximum(square / row_count - np.square(mean), 1e-12)
    meta = {
        "format": "lewm_latent_frame_cache_v1",
        "task": args.task,
        "dataset": dataset,
        "h5_path": str(h5_path),
        "max_episodes": args.max_episodes,
        "lewm_checkpoint": str(args.lewm_checkpoint),
        "lewm_weights_sha256": getattr(backend, "weights_sha256", None),
        "lewm": backend.describe(),
        "image_size": 224,
        "normalization": "imagenet",
        "row_count": row_count,
        "feature_dim": feature_dim,
        "feature_mean": mean.astype(np.float32).tolist(),
        "feature_std": np.sqrt(variance).astype(np.float32).tolist(),
        "source_columns_read_by_encoder": ["pixels"],
        "privileged_state_model_access": False,
        **index_meta,
    }
    atomic_json(meta_path, meta)
    progress_path.unlink(missing_ok=True)
    print(
        f"[cache] saved {latents_path} rows={row_count} dim={feature_dim}", flush=True
    )


if __name__ == "__main__":
    main()
