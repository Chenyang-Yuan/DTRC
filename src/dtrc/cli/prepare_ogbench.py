"""Convert official OGBench trajectories to image-based Lance datasets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import zipfile

import numpy as np

from dtrc.environments.ogbench import episode_layout
from dtrc.environments.registry import OGBENCH_TASKS, TASKS


def read_arrays(source, task):
    with np.load(source, allow_pickle=False) as data:
        required = {"observations", "actions", "terminals", "qpos", "qvel"}
        if task in {"scene", "puzzle-3x3"}:
            required.add("button_states")
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Source NPZ is missing {sorted(missing)}")
        arrays = {key: np.asarray(data[key]) for key in required - {"observations"}}
        if task == "pointmaze":
            arrays["observations"] = np.asarray(data["observations"])
    rows = len(arrays["terminals"])
    for key, value in arrays.items():
        if len(value) != rows or not np.isfinite(value).all():
            raise ValueError(f"Invalid values or row count in {key}")
    if arrays["actions"].shape != (rows, TASKS[task]["action_dim"]):
        raise ValueError("Action dimensions do not match the selected task")
    return arrays


def extract_pixels(source, destination, rows):
    """Read a prefix of the compressed pixel member without loading it into RAM."""
    with zipfile.ZipFile(source) as archive:
        with archive.open("observations.npy") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"Unsupported NPY version: {version}")
            if fortran or dtype != np.uint8 or len(shape) != 4 or shape[-1] != 3:
                raise ValueError(
                    f"Expected C-order uint8 RGB observations, got {shape}, {dtype}"
                )
            if rows > shape[0]:
                raise ValueError("Pixels have fewer rows than the trajectory index")
            pixels = np.lib.format.open_memmap(
                destination, mode="w+", dtype=dtype, shape=(rows, *shape[1:])
            )
            bytes_per_row = int(np.prod(shape[1:]))
            for start in range(0, rows, 512):
                count = min(512, rows - start)
                expected = count * bytes_per_row
                buffer = bytearray()
                while len(buffer) < expected:
                    block = stream.read(expected - len(buffer))
                    if not block:
                        raise ValueError("Truncated observations.npy")
                    buffer.extend(block)
                pixels[start : start + count] = np.frombuffer(
                    buffer, dtype=dtype
                ).reshape(count, *shape[1:])
            pixels.flush()
    return pixels, int(shape[0])


def render_pointmaze(arrays, starts, lengths, destination):
    """Restore each recorded state before rendering; do not replay whole episodes."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    from stable_worldmodel.envs.ogbench.maze_env import MazeEnv

    pixels = np.lib.format.open_memmap(
        destination, mode="w+", dtype=np.uint8, shape=(int(lengths.sum()), 64, 64, 3)
    )
    env = MazeEnv(
        loco_env_type="point",
        maze_env_type="maze",
        maze_type="large",
        ob_type="pixels",
        width=64,
        height=64,
        terminate_at_goal=False,
    )
    try:
        for episode, (start, length) in enumerate(zip(starts, lengths)):
            env.reset(seed=episode)
            for row in range(int(start), int(start + length)):
                env.set_state(
                    arrays["qpos"][row].astype(np.float64),
                    arrays["qvel"][row].astype(np.float64),
                )
                frame = np.asarray(env.unwrapped.get_ob())
                if frame.shape != (64, 64, 3) or frame.dtype != np.uint8:
                    raise ValueError(
                        f"Unexpected PointMaze rendering: {frame.shape}, {frame.dtype}"
                    )
                pixels[row] = frame
            print(f"[prepare] rendered episode {episode + 1}/{len(starts)}", flush=True)
    finally:
        env.close()
    pixels.flush()
    return pixels


def prepare(task, source, output, work_dir, max_episodes=None, jpeg_quality=95):
    from stable_worldmodel.data import LanceWriter

    source, output, work_dir = (
        Path(path).resolve() for path in (source, output, work_dir)
    )
    metadata_path = output.with_name(output.name + ".metadata.json")
    if output.suffix != ".lance":
        raise ValueError("Output must end in .lance")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("Output already exists; choose a new output path")
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max-episodes must be positive")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg-quality must be between 1 and 100")
    arrays = read_arrays(source, task)
    starts, lengths = episode_layout(arrays["terminals"])
    if max_episodes is not None:
        starts, lengths = starts[:max_episodes], lengths[:max_episodes]
    rows = int(lengths.sum())
    work_dir.mkdir(parents=True, exist_ok=True)
    pixels_path = work_dir / (output.stem + "_pixels.npy")
    if pixels_path.exists():
        raise FileExistsError(
            f"Pixel work file already exists: {pixels_path}; choose a new work directory"
        )
    if task == "pointmaze":
        pixels = render_pointmaze(arrays, starts, lengths, pixels_path)
    else:
        pixels, source_rows = extract_pixels(source, pixels_path, rows)
        if source_rows != len(arrays["terminals"]):
            raise ValueError("Pixel and trajectory arrays have different row counts")

    def episodes():
        for episode, (start, length) in enumerate(zip(starts, lengths)):
            begin, end = int(start), int(start + length)
            values = {
                "pixels": pixels[begin:end],
                "action": list(arrays["actions"][begin:end].astype(np.float32)),
            }
            for key in ("qpos", "qvel", "button_states"):
                if key in arrays:
                    values[key] = list(arrays[key][begin:end])
            yield values
            if episode % 25 == 0:
                print(
                    f"[prepare] encoded episode {episode + 1}/{len(starts)}", flush=True
                )

    # Lance writes on a worker thread. Rendering above stays on the EGL owner thread.
    output.parent.mkdir(parents=True, exist_ok=True)
    with LanceWriter(output, jpeg_quality=jpeg_quality, mode="error") as writer:
        writer.write_episodes(episodes())
    metadata = {
        "format": "dtrc_ogbench_visual_lance_v1",
        "task": task,
        "source": str(source),
        "num_rows": rows,
        "num_episodes": len(starts),
        "action_dim": TASKS[task]["action_dim"],
        "image_shape": list(pixels.shape[1:]),
        "jpeg_quality": jpeg_quality,
        "source_rows": len(arrays["terminals"]),
        "source_bytes": source.stat().st_size,
        "pixel_source": (
            "per_row_state_render"
            if task == "pointmaze"
            else "official_visual_observations"
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"[prepare] complete: {output}; {rows} rows, {len(starts)} episodes", flush=True
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=OGBENCH_TASKS, required=True)
    parser.add_argument(
        "--source", required=True, help="Official training NPZ; see docs/data.md"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--work-dir", required=True, help="Disk space for uncompressed RGB pixels"
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional prefix for a short pipeline check",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()
    prepare(
        args.task,
        args.source,
        args.output,
        args.work_dir,
        args.max_episodes,
        args.jpeg_quality,
    )


if __name__ == "__main__":
    main()
