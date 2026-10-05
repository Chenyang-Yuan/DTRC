"""Encode visual OGBench Lance observations and preserve trajectory alignment."""

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dtrc.backends.lewm import load_lewm_backend
from dtrc.environments.registry import OGBENCH_TASKS, TASKS


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def episode_bounds(episodes: np.ndarray, steps: np.ndarray):
    episodes = np.asarray(episodes, dtype=np.int64).reshape(-1)
    steps = np.asarray(steps, dtype=np.int64).reshape(-1)
    if len(episodes) == 0 or len(episodes) != len(steps):
        raise ValueError("invalid Lance episode/step index columns")
    boundaries = np.flatnonzero(np.r_[episodes[1:] != episodes[:-1], True])
    starts = np.r_[0, boundaries[:-1] + 1]
    lengths = boundaries - starts + 1
    if np.any(steps[starts] != 0):
        raise ValueError("every Lance episode must start at step_idx=0")
    for start, length in zip(starts, lengths):
        expected = np.arange(length, dtype=np.int64)
        if not np.array_equal(steps[start : start + length], expected):
            raise ValueError("Lance step_idx is not contiguous inside an episode")
    episode_start = np.repeat(starts, lengths)
    episode_end = np.repeat(boundaries, lengths)
    return (episode_start, episode_end, int(len(starts)))


def compact_episode_lengths(terminals: np.ndarray) -> np.ndarray:
    terminals = np.asarray(terminals, dtype=bool).reshape(-1)
    run_end = terminals & np.r_[~terminals[1:], True]
    ends = np.flatnonzero(run_end)
    if len(ends) == 0 or ends[-1] != len(terminals) - 1:
        raise ValueError("state NPZ terminals do not close the final episode")
    starts = np.r_[0, ends[:-1] + 1]
    return ends - starts + 1


def pixels_nhwc(batch: dict) -> torch.Tensor:
    value = batch["pixels"]
    if value.ndim == 5 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 4:
        raise ValueError(
            f"expected batched images, got pixels shape {tuple(value.shape)}"
        )
    if value.shape[1] in (3, 4):
        value = value[:, :3].permute(0, 2, 3, 1)
    elif value.shape[-1] in (3, 4):
        value = value[..., :3]
    else:
        raise ValueError(f"expected RGB images, got pixels shape {tuple(value.shape)}")
    if value.dtype != torch.uint8:
        raise ValueError(f"raw uint8 Lance pixels required, got {value.dtype}")
    return value.contiguous()


def latent_diagnostics(latents_path: Path, rows: int, sample_size: int = 10000):
    latent = np.load(latents_path, mmap_mode="r")
    count = min(int(sample_size), rows)
    indices = np.linspace(0, rows - 1, count, dtype=np.int64)
    sample = np.asarray(latent[indices], dtype=np.float64)
    if not np.isfinite(sample).all():
        raise ValueError("cached latents contain non-finite values")
    centered = sample - sample.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(count - 1, 1)
    variance = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    probability = variance / max(float(variance.sum()), 1e-12)
    effective_rank = float(np.exp(-(probability * np.log(probability + 1e-12)).sum()))
    per_dim_std = sample.std(axis=0)
    return {
        "sample_rows": count,
        "mean_feature_std": float(per_dim_std.mean()),
        "min_feature_std": float(per_dim_std.min()),
        "active_dims_std_gt_1e-5": int((per_dim_std > 1e-05).sum()),
        "effective_rank": effective_rank,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=OGBENCH_TASKS, required=True)

    parser.add_argument("--lance", required=True)
    parser.add_argument("--state-npz", default=None)
    parser.add_argument("--lewm-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = TASKS[args.task]
    lance_path = Path(args.lance).resolve()
    state_path = Path(args.state_npz).resolve() if args.state_npz else None
    checkpoint = Path(args.lewm_checkpoint).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not lance_path.is_dir():
        raise FileNotFoundError(lance_path)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    metadata_path = lance_path.parent / f"{lance_path.name}.metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Lance provenance metadata is required: {metadata_path}"
        )
    lance_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_source = Path(lance_metadata.get("source", "")).resolve()
    if state_path is not None and metadata_source != state_path:
        raise ValueError(
            f"training state NPZ must be the exact source used to build Lance: metadata source={metadata_source}, supplied={state_path}"
        )
    import stable_worldmodel as swm

    dataset = swm.data.LanceDataset(
        path=str(lance_path), frameskip=1, num_steps=1, keys_to_load=["pixels"]
    )
    actions = np.asarray(dataset.get_col_data("action"), dtype=np.float32)
    episodes = np.asarray(dataset.get_col_data("episode_idx"), dtype=np.int64)
    steps = np.asarray(dataset.get_col_data("step_idx"), dtype=np.int64)
    if actions.ndim == 1:
        actions = actions[:, None]
    if len(dataset) != len(actions):
        raise ValueError(
            f"Lance clip/row mismatch: clips={len(dataset)} rows={len(actions)}"
        )
    if actions.shape[1] != spec["action_dim"]:
        raise ValueError(
            f"{args.task} action_dim={actions.shape[1]} != expected {spec['action_dim']}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("Lance actions contain non-finite values")
    (episode_start, episode_end, num_episodes) = episode_bounds(episodes, steps)
    if int(lance_metadata.get("num_rows", -1)) != len(actions):
        raise ValueError(
            f"Lance row count disagrees with its provenance metadata: {len(actions)} != {lance_metadata.get('num_rows')}"
        )
    if int(lance_metadata.get("action_dim", -1)) != actions.shape[1]:
        raise ValueError(
            f"Lance action dimension disagrees with its provenance metadata: {actions.shape[1]} != {lance_metadata.get('action_dim')}"
        )
    np.savez(
        output / "index.npz",
        action=actions,
        episode_idx=episodes,
        step_idx=steps,
        episode_start=episode_start,
        episode_end=episode_end,
    )
    backend = load_lewm_backend(str(checkpoint), device=args.device)
    if backend.latent_dim != 192:
        raise ValueError(f"expected LeWM latent_dim=192, got {backend.latent_dim}")
    latents_path = output / "latents.npy"
    progress_path = output / "progress.json"
    meta_path = output / "meta.json"
    (rows, dim) = (len(actions), backend.latent_dim)
    total = np.zeros(dim, dtype=np.float64)
    square = np.zeros(dim, dtype=np.float64)
    start_row = 0
    if latents_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        identity = (
            progress.get("task"),
            progress.get("lance"),
            progress.get("lance_source_npz"),
            progress.get("checkpoint"),
        )
        expected = (args.task, str(lance_path), str(metadata_source), str(checkpoint))
        if identity != expected or progress.get("rows") != rows:
            raise ValueError(
                "cache resume provenance mismatch; use a new output directory"
            )
        start_row = int(progress["completed_rows"])
        total = np.asarray(progress["sum"], dtype=np.float64)
        square = np.asarray(progress["square"], dtype=np.float64)
        latents = np.lib.format.open_memmap(latents_path, mode="r+")
        print(f"[cache-ogbench] resuming {start_row}/{rows}", flush=True)
    elif meta_path.exists() and latents_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        identity = (
            existing.get("task"),
            existing.get("lance_path"),
            existing.get("lance_source_npz", existing.get("state_npz")),
            existing.get("lewm_checkpoint"),
        )
        expected = (args.task, str(lance_path), str(metadata_source), str(checkpoint))
        if identity != expected:
            raise ValueError(
                f"completed cache provenance mismatch: {identity} != {expected}"
            )
        recorded_hash = existing.get("lewm_weights_sha256")
        actual_hash = getattr(backend, "weights_sha256", None)
        if recorded_hash and actual_hash and (recorded_hash != actual_hash):
            raise ValueError("completed cache LeWM checkpoint hash mismatch")
        if not (output / "index.npz").is_file():
            raise FileNotFoundError(output / "index.npz")
        print(f"[cache-ogbench] already complete and verified: {output}", flush=True)
        return
    else:
        latents = np.lib.format.open_memmap(
            latents_path, mode="w+", dtype=np.float32, shape=(rows, dim)
        )
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        sampler=range(start_row, rows),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(**loader_kwargs)
    completed = start_row
    last_report = start_row
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    with torch.inference_mode():
        for batch in loader:
            encoded = (
                backend.encode_uint8(pixels_nhwc(batch), batch_size=args.batch_size)
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            end = completed + len(encoded)
            latents[completed:end] = encoded
            values = encoded.astype(np.float64)
            total += values.sum(axis=0)
            square += np.square(values).sum(axis=0)
            completed = end
            if completed == rows or completed - last_report >= args.progress_every:
                latents.flush()
                atomic_json(
                    progress_path,
                    {
                        "task": args.task,
                        "lance": str(lance_path),
                        "lance_source_npz": str(metadata_source),
                        "checkpoint": str(checkpoint),
                        "rows": rows,
                        "feature_dim": dim,
                        "completed_rows": completed,
                        "sum": total.tolist(),
                        "square": square.tolist(),
                    },
                )
                rate = (completed - start_row) / max(time.time() - started, 1e-06)
                print(
                    f"[cache-ogbench] {completed}/{rows} ({rate:.1f} frames/s)",
                    flush=True,
                )
                last_report = completed
    if completed != rows:
        raise RuntimeError(f"cache stopped early: {completed}/{rows}")
    latents.flush()
    mean = total / rows
    variance = np.maximum(square / rows - np.square(mean), 1e-12)
    diagnostics = latent_diagnostics(latents_path, rows)
    if diagnostics["active_dims_std_gt_1e-5"] < max(8, dim // 10):
        raise RuntimeError(f"LeWM latent appears collapsed: {diagnostics}")
    meta = {
        "format": "lewm_latent_frame_cache_v1",
        "cache_extension": "ogbench_visual_lance_v1",
        "task": args.task,
        "dataset": str(lance_path),
        "lance_path": str(lance_path),
        "state_npz": str(metadata_source),
        "lance_source_npz": str(metadata_source),
        "lewm_checkpoint": str(checkpoint),
        "lewm_weights_sha256": getattr(backend, "weights_sha256", None),
        "lewm": backend.describe(),
        "image_size": 224,
        "normalization": "imagenet",
        "row_count": rows,
        "feature_dim": dim,
        "feature_mean": mean.astype(np.float32).tolist(),
        "feature_std": np.sqrt(variance).astype(np.float32).tolist(),
        "action_dim": int(actions.shape[1]),
        "action_mean": actions.mean(axis=0, dtype=np.float64)
        .astype(np.float32)
        .tolist(),
        "action_std": np.maximum(actions.std(axis=0, dtype=np.float64), 1e-06)
        .astype(np.float32)
        .tolist(),
        "num_episodes": num_episodes,
        "source_columns_read_by_encoder": ["pixels"],
        "index_columns": ["action", "episode_idx", "step_idx"],
        "privileged_state_model_access": False,
        "latent_diagnostics": diagnostics,
    }
    atomic_json(meta_path, meta)
    progress_path.unlink(missing_ok=True)
    print(f"[cache-ogbench] complete: {output} rows={rows} dim={dim}", flush=True)
    print(f"[cache-ogbench] diagnostics: {diagnostics}", flush=True)


if __name__ == "__main__":
    main()
