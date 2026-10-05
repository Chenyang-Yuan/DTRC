"""Evaluate a pixel-conditioned DTRC policy on matched future-goal problems."""

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch
from dtrc.environments.ogbench import (
    TASKS,
    current_button_states,
    goal_distance,
    is_success,
    load_manifest,
    make_env,
    restore,
)
from dtrc.backends.lewm import load_lewm_backend
from dtrc.backends.frontend import LatentFrontend
from dtrc.agent import load_agent
from dtrc.environments.registry import OGBENCH_TASKS
from dtrc.policy import DTRCPolicy


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for (key, item) in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def trajectory_metrics(actions, distances):
    action_array = np.asarray(actions, dtype=np.float64)
    distance_array = np.asarray(distances, dtype=np.float64)
    if len(action_array) >= 3:
        second_difference = (
            action_array[2:] - 2.0 * action_array[1:-1] + action_array[:-2]
        )
        action_jerk = float(np.linalg.norm(second_difference, axis=-1).mean())
    else:
        action_jerk = None
    changes = np.diff(distance_array)
    step_monotonicity = float(np.mean(changes <= 1e-08)) if len(changes) else None
    episode_monotonic = bool(np.all(changes <= 1e-08)) if len(changes) else True
    total_variation = float(np.abs(changes).sum())
    net_progress = float(max(distance_array[0] - distance_array[-1], 0.0))
    path_efficiency = (
        float(net_progress / total_variation) if total_variation > 1e-12 else 0.0
    )
    return {
        "action_jerk": action_jerk,
        "step_monotonicity": step_monotonicity,
        "episode_monotonic": episode_monotonic,
        "path_efficiency": path_efficiency,
    }


def lance_pixel_at(dataset, index: int) -> np.ndarray:
    value = dataset[int(index)]["pixels"]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    while value.ndim > 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 3:
        raise ValueError(f"expected one Lance image at row {index}, got {value.shape}")
    if value.shape[0] in (3, 4) and value.shape[-1] not in (3, 4):
        value = np.moveaxis(value[:3], 0, -1)
    if value.shape[-1] not in (3, 4):
        raise ValueError(f"expected RGB(A) image at row {index}, got {value.shape}")
    if np.issubdtype(value.dtype, np.floating):
        if not np.isfinite(value).all() or value.min() < 0:
            raise ValueError("Lance goal pixels must be finite and nonnegative")
        if value.max() <= 1.0 + 1e-06:
            value = value * 255.0
    return np.clip(value[..., :3], 0, 255).astype(np.uint8, copy=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=OGBENCH_TASKS, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--state-npz", required=True)
    parser.add_argument("--lewm-checkpoint", default=None)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--act-mode", default="single", choices=["single"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--record-success-dir",
        default=None,
        help="Optional directory for lossless per-step RGB recordings. Only successful episodes are written, as env_XX.npz files containing frames and the exact rendered goal frame.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_path = Path(args.state_npz).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    (agent, meta) = load_agent(checkpoint, args.device, act_mode="single")
    if meta.get("task") != args.task:
        raise ValueError(f"DTRC task mismatch: {meta.get('task')!r} != {args.task!r}")
    if meta.get("privileged_state_model_access") is not False:
        raise RuntimeError("checkpoint lacks pixels-only provenance")
    lewm_source = str(Path(args.lewm_checkpoint or meta["lewm_checkpoint"]).resolve())
    backend = load_lewm_backend(lewm_source, device=args.device)
    recorded_hash = meta.get("lewm_weights_sha256")
    actual_hash = getattr(backend, "weights_sha256", None)
    if recorded_hash and actual_hash and (recorded_hash != actual_hash):
        raise RuntimeError("LeWM checkpoint differs from the DTRC cache encoder")
    frontend = LatentFrontend(backend, meta)
    policy = DTRCPolicy(agent, frontend, meta, budget=args.budget)
    agent.generator.manual_seed(args.seed)
    with np.load(state_path, allow_pickle=False) as raw:
        required = {"observations", "terminals", "qpos", "qvel"}
        missing = required - set(raw.files)
        if missing:
            raise KeyError(f"state NPZ missing fields: {sorted(missing)}")
        observations = np.asarray(raw["observations"], dtype=np.float32)
        terminals = np.asarray(raw["terminals"], dtype=bool)
        qpos = np.asarray(raw["qpos"], dtype=np.float64)
        qvel = np.asarray(raw["qvel"], dtype=np.float64)
        buttons = (
            np.asarray(raw["button_states"]) if "button_states" in raw.files else None
        )
    (episode_starts, episode_ids, local_starts) = load_manifest(
        args.manifest, terminals, args.num_eval, args.goal_offset
    )
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if manifest.get("task") != args.task:
        raise ValueError("manifest task mismatch")
    env = make_env(args.task)
    policy.set_env(env)
    record_root = (
        Path(args.record_success_dir).resolve() if args.record_success_dir else None
    )
    if record_root is not None:
        record_root.mkdir(parents=True, exist_ok=True)
    rows = []
    decision_seconds = 0.0
    decision_calls = 0
    started = time.perf_counter()
    for eval_index, (episode, local_start) in enumerate(zip(episode_ids, local_starts)):
        start = int(episode_starts[episode] + local_start)
        goal_index = start + args.goal_offset
        restore(
            env,
            args.task,
            qpos[goal_index],
            qvel[goal_index],
            None if buttons is None else buttons[goal_index],
            args.seed * 1000000 + eval_index,
        )
        goal_pixels = np.asarray(env.render(), dtype=np.uint8)
        observation = restore(
            env,
            args.task,
            qpos[start],
            qvel[start],
            None if buttons is None else buttons[start],
            args.seed * 1000000 + eval_index,
        )
        goal_state = observations[goal_index]
        goal_buttons = None if buttons is None else buttons[goal_index]
        current_buttons = (
            current_button_states(env) if TASKS[args.task]["kind"] == "puzzle" else None
        )
        initial_distance = goal_distance(
            args.task, observation, goal_state, current_buttons, goal_buttons
        )
        initial_success = is_success(args.task, initial_distance)
        (success, success_step) = (False, None)
        final_distance = initial_distance
        step_count = 0
        executed_actions = []
        distance_trace = [float(initial_distance)]
        recorded_frames = []
        while step_count < args.budget and (not success):
            current_pixels = np.asarray(env.render(), dtype=np.uint8)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            tick = time.perf_counter()
            action = policy.get_action(
                {
                    "pixels": current_pixels[None],
                    "goal": goal_pixels[None],
                    "_needs_flush": np.asarray([step_count == 0]),
                }
            )[0]
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            decision_seconds += time.perf_counter() - tick
            decision_calls += 1
            action = np.clip(action, env.action_space.low, env.action_space.high)
            executed_actions.append(np.asarray(action, dtype=np.float32).copy())
            (observation, _, terminated, truncated, _) = env.step(action)
            observation = np.asarray(observation, dtype=np.float32)
            step_count += 1
            if record_root is not None:
                recorded_frames.append(np.asarray(env.render(), dtype=np.uint8).copy())
            if TASKS[args.task]["kind"] == "puzzle":
                current_buttons = current_button_states(env)
            final_distance = goal_distance(
                args.task, observation, goal_state, current_buttons, goal_buttons
            )
            distance_trace.append(float(final_distance))
            success = is_success(args.task, final_distance)
            if success:
                success_step = step_count
            if terminated or truncated:
                break
        raw_recording = None
        if record_root is not None and success:
            raw_recording = record_root / f"env_{eval_index:02d}.npz"
            np.savez_compressed(
                raw_recording,
                frames=np.asarray(recorded_frames[:success_step], dtype=np.uint8),
                goal_frame=np.asarray(goal_pixels, dtype=np.uint8),
            )
        quality = trajectory_metrics(executed_actions, distance_trace)
        rows.append(
            {
                "evaluation_index": eval_index,
                "episode": int(episode),
                "start": int(local_start),
                "goal": int(local_start + args.goal_offset),
                "start_dataset_index": start,
                "goal_dataset_index": goal_index,
                "success": bool(success),
                "initial_within_goal": bool(initial_success),
                "nontrivial_success": bool(success and (not initial_success)),
                "success_step": success_step,
                "steps": step_count,
                "initial_goal_distance": float(initial_distance),
                "final_goal_distance": float(final_distance),
                "actions": [action.tolist() for action in executed_actions],
                "goal_distance_trace": distance_trace,
                "raw_success_recording": (
                    str(raw_recording) if raw_recording is not None else None
                ),
                **quality,
            }
        )
        print(
            f"[{eval_index + 1:03d}/{args.num_eval}] success={int(success)} steps={step_count} distance={initial_distance:.3f}->{final_distance:.3f}",
            flush=True,
        )
    env.close()
    successes = sum((row["success"] for row in rows))
    initial = sum((row["initial_within_goal"] for row in rows))
    nontrivial = args.num_eval - initial
    nontrivial_successes = sum((row["nontrivial_success"] for row in rows))
    jerk_values = [row["action_jerk"] for row in rows if row["action_jerk"] is not None]
    step_monotonicity_values = [
        row["step_monotonicity"] for row in rows if row["step_monotonicity"] is not None
    ]
    successful_steps = [
        row["success_step"]
        for row in rows
        if row["nontrivial_success"] and row["success_step"] is not None
    ]
    result = {
        "protocol": "dtrc_same_trajectory_future_goal_ogbench_visual_v1",
        "method": "dtrc",
        "task": args.task,
        "checkpoint": str(checkpoint_path),
        "lewm_checkpoint": lewm_source,
        "lewm_weights_sha256": actual_hash or recorded_hash,
        "state_npz": str(state_path),
        "goal_pixel_source": "simulator_render_of_exact_evaluation_state",
        "manifest": str(Path(args.manifest).resolve()),
        "num_eval": args.num_eval,
        "goal_offset": args.goal_offset,
        "budget": args.budget,
        "seed": args.seed,
        "act_mode": args.act_mode,
        "chunk": int(agent.cfg.chunk),
        "observation_protocol": "pixels_only",
        "model_input_keys": ["pixels", "goal"],
        "privileged_state_model_access": False,
        "evaluator_privileged_state_use": "reset_and_success_only",
        "encoder_frozen": True,
        "successes": successes,
        "success_rate": 100.0 * successes / args.num_eval,
        "initial_successes": initial,
        "num_nontrivial": nontrivial,
        "nontrivial_successes": nontrivial_successes,
        "nontrivial_success_rate": (
            100.0 * nontrivial_successes / nontrivial if nontrivial else None
        ),
        "evaluation_seconds": time.perf_counter() - started,
        "record_success_dir": str(record_root) if record_root is not None else None,
        "trajectory_quality": {
            "distance_space": "shared_native_task_goal_distance",
            "action_jerk_mean": float(np.mean(jerk_values)) if jerk_values else None,
            "step_monotonicity_mean": (
                float(np.mean(step_monotonicity_values))
                if step_monotonicity_values
                else None
            ),
            "episode_monotonicity_rate": float(
                np.mean([row["episode_monotonic"] for row in rows])
            ),
            "path_efficiency_mean": float(
                np.mean([row["path_efficiency"] for row in rows])
            ),
            "successful_steps_mean": (
                float(np.mean(successful_steps)) if successful_steps else None
            ),
            "decision_latency_ms": 1000.0 * decision_seconds / max(decision_calls, 1),
            "decision_calls": decision_calls,
        },
        "episodes": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jsonable(result), indent=2), encoding="utf-8")
    print(json.dumps({k: v for (k, v) in result.items() if k != "episodes"}, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
