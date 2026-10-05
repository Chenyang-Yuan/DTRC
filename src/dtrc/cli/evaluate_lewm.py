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
import torch.nn.functional as F
from dtrc.backends.lewm import load_lewm_backend, preprocess_uint8
from dtrc.backends.frontend import LatentFrontend
from dtrc.environments.lewm import get_task
from dtrc.agent import load_agent
from dtrc.policy import DTRCPolicy
from dtrc.metrics import ParallelTrajectoryRecorder


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for (key, item) in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--lewm-checkpoint", default=None)
    p.add_argument("--budget", type=int, default=50)
    p.add_argument("--act-mode", default="single", choices=["single"])
    p.add_argument("--seed", type=int, default=42, help="flow sampling seed")
    p.add_argument("--device", default="cuda")
    p.add_argument("--video", default=None)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import stable_worldmodel as swm

    spec = get_task(args.task)
    dataset_name = args.dataset or spec["eval_dataset"]
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "stable_worldmodel_visual_manifest_v1":
        raise ValueError("strict visual manifest required")
    if manifest["task"] != args.task or manifest["dataset"] != dataset_name:
        raise ValueError("task/dataset does not match manifest")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    (agent, meta) = load_agent(checkpoint, args.device, act_mode="single")
    agent.generator.manual_seed(args.seed)
    if meta.get("privileged_state_model_access") is not False:
        raise RuntimeError("checkpoint lacks pixels-only provenance")
    lewm_source = args.lewm_checkpoint or meta["lewm_checkpoint"]
    backend = load_lewm_backend(lewm_source, device=args.device)
    recorded = meta.get("lewm_weights_sha256")
    actual = getattr(backend, "weights_sha256", None)
    if recorded and actual and (recorded != actual):
        raise RuntimeError(
            "LeWM weights differ from the ones used to build the latent cache"
        )
    frontend = LatentFrontend(backend, meta)
    policy = DTRCPolicy(agent, frontend, meta, budget=args.budget)
    n = int(manifest["num_eval"])
    world_kwargs = dict(spec["world"])
    world_kwargs.update(num_envs=n, max_episode_steps=2 * args.budget)
    world = swm.World(**world_kwargs, image_shape=(224, 224), render_mode="rgb_array")
    dataset = swm.data.load_dataset(dataset_name)
    recorder = ParallelTrajectoryRecorder(n)
    original_get_action = policy.get_action
    raw_goal_latent = None

    def raw_lewm_latent(images):
        tensor = torch.as_tensor(np.ascontiguousarray(images)).to(backend.device)
        return backend.encode_pixels(preprocess_uint8(tensor))

    def traced_get_action(info_dict, **kwargs):
        nonlocal raw_goal_latent
        pixels = np.asarray(info_dict["pixels"])
        goals = np.asarray(info_dict["goal"])
        current = pixels[:, -1] if pixels.ndim == 5 else pixels
        goal_images = goals[:, -1] if goals.ndim == 5 else goals
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        tick = time.perf_counter()
        actions = original_get_action(info_dict, **kwargs)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        decision_seconds = time.perf_counter() - tick
        with torch.no_grad():
            current_z = F.normalize(raw_lewm_latent(current), dim=-1)
            if raw_goal_latent is None:
                raw_goal_latent = F.normalize(raw_lewm_latent(goal_images), dim=-1)
            distances = (1.0 - (current_z * raw_goal_latent).sum(dim=-1)).cpu().numpy()
        recorder.record_decision(actions, distances, decision_seconds)
        return actions

    policy.get_action = traced_get_action
    world.set_policy(policy)
    original_step = world.envs.step

    def traced_step(actions, mask=None):
        output = original_step(actions, mask=mask)
        recorder.mark_done(output[2], output[3])
        return output

    world.envs.step = traced_step
    started = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        episodes_idx=manifest["episodes"],
        start_steps=manifest["starts"],
        goal_offset=int(manifest["goal_offset"]),
        eval_budget=args.budget,
        callables=spec["callables"],
        video=args.video,
    )
    variant = meta.get("variant")
    result = {
        "format": "stable_worldmodel_visual_result_v1",
        "method": "dtrc",
        "variant": variant,
        "act_mode": args.act_mode,
        "controller": meta.get("controller"),
        "chunk": int(agent.cfg.chunk),
        "task": args.task,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset": dataset_name,
        "normalization_dataset": meta["dataset"],
        "lewm_checkpoint": str(lewm_source),
        "lewm_weights_sha256": actual or recorded,
        "frameskip": 1,
        "history_size": 1,
        "horizon_max": int(agent.cfg.horizon_max),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "num_eval": n,
        "goal_offset": int(manifest["goal_offset"]),
        "budget": args.budget,
        "sampling_seed": args.seed,
        "observation_protocol": "pixels_only",
        "model_input_keys": ["goal", "pixels"],
        "privileged_state_model_access": False,
        "evaluator_privileged_state_use": "reset_and_success_only",
        "encoder": "official_lewm_vit_tiny14_cls_projector",
        "encoder_frozen": True,
        "metrics": jsonable(metrics),
        "evaluation_seconds": time.time() - started,
        "trajectory_quality": recorder.summary(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
