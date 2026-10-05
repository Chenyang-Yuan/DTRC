"""Train DTRC or a component variant on cached frozen visual features."""

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from dtrc.cache import LatentCache
from dtrc.agent import VARIANTS, build_agent
from dtrc.data import TemporalBuffer


def atomic_save(agent, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(agent.state_dict(), temporary)
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_overrides(items):
    overrides = {}
    for item in items or []:
        (key, _, raw) = item.partition("=")
        if not key or not raw:
            raise ValueError(f"override must be key=value, got {item!r}")
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw
    return overrides


def agent_settings(args):
    """Resolve model settings without disagreeing with the temporal sampler."""
    overrides = dict(args.agent_settings)
    overrides.update(parse_overrides(args.override))
    reserved = {"obs_dim", "action_dim", "device"} & overrides.keys()
    if reserved:
        raise ValueError(
            f"agent settings {sorted(reserved)} are determined by the cache or --device"
        )
    for name in ("horizon_max", "chunk", "span"):
        value = getattr(args, name)
        if name in overrides and overrides[name] != value:
            raise ValueError(
                f"agent.{name}={overrides[name]} conflicts with training.{name}={value}; "
                f"set --{name.replace('_', '-')} instead of an agent override"
            )
        overrides[name] = value
    overrides.setdefault("n_step", min(args.span, 8))
    overrides.setdefault("dyn_rollout", min(args.span, 4))
    return overrides


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", default=None, help="JSON with training and agent sections"
    )
    p.add_argument("--latent-cache", required=True, help="latent cache directory")
    p.add_argument("--out", required=True)
    p.add_argument("--variant", default="dtrc", choices=sorted(VARIANTS))
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-steps", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument(
        "--chunk", type=int, default=1, help="primitive actions per policy output"
    )
    p.add_argument(
        "--span",
        type=int,
        default=8,
        help="window length (>= n_step, dyn_rollout, chunk)",
    )
    p.add_argument("--horizon-max", type=int, default=50)
    p.add_argument("--train-split", type=float, default=0.9)
    p.add_argument(
        "--split-mode",
        choices=("row", "episode"),
        default="row",
        help="unit used to partition eligible training starts",
    )
    p.add_argument("--goal-discount", type=float, default=0.98)
    p.add_argument("--p-current", type=float, default=0.1)
    p.add_argument("--p-traj", type=float, default=0.6)
    p.add_argument("--p-random", type=float, default=0.3)
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="DTRCConfig field override, e.g. --override n_candidates=32",
    )
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=20000)
    p.add_argument("--holdout-every", type=int, default=5000)
    p.add_argument("--resume", default=None)
    p.add_argument("--mmap", action="store_true")
    preliminary, _ = p.parse_known_args()
    settings = {}
    if preliminary.config:
        settings = json.loads(Path(preliminary.config).read_text(encoding="utf-8"))
        if set(settings) - {"training", "agent"}:
            p.error("configuration sections must be training and agent")
        training = settings.get("training", {})
        allowed = {action.dest for action in p._actions} - {
            "help",
            "config",
            "latent_cache",
            "out",
        }
        if set(training) - allowed:
            p.error(f"unknown training settings: {sorted(set(training) - allowed)}")
        p.set_defaults(**training)
    args = p.parse_args()
    args.agent_settings = settings.get("agent", {})
    if args.variant not in VARIANTS:
        p.error("unknown variant")
    if (
        min(
            args.batch_size,
            args.train_steps,
            args.log_every,
            args.save_every,
            args.horizon_max,
        )
        < 1
    ):
        p.error(
            "batch size, iterations, logging intervals, and horizon must be positive"
        )
    if args.resume and not Path(args.resume).is_file():
        p.error("resume checkpoint does not exist")
    return args


@torch.no_grad()
def holdout_metrics(agent, buffer, batch_size: int) -> dict:
    batch = buffer.sample_holdout(batch_size)
    k = agent.cfg.chunk
    (z, g, h) = (batch["seq_obs"][:, 0], batch["goals"], batch["h_norm"])
    pred = agent.act(z, g, h).reshape(len(z), -1)
    target = batch["seq_actions"][:, :k].reshape(len(z), -1)
    out = {"pi_chunk_mse": float(torch.nn.functional.mse_loss(pred, target))}
    if agent.cfg.use_critic:
        d = agent.distance(z, g)
        gaps = batch["gaps"]
        known = gaps > 0
        out["d_goal_mean"] = float(d.mean())
        if known.any():
            (dk, gk) = (d[known], gaps[known].float())
            out["d_vs_gap_mae"] = float((dk - gk).abs().mean())
            if known.sum() > 2:
                out["d_gap_corr"] = float(torch.corrcoef(torch.stack([dk, gk]))[0, 1])
        d_next = agent.distance(batch["seq_obs"][:, 1], g)
        out["d_progress_frac"] = float((d_next < d).float().mean())
    if agent.cfg.use_model:
        (z_pred, dis) = agent.dyn.rollout(z, batch["seq_actions"][:, :k])
        out["dyn_chunk_mse"] = float(
            torch.nn.functional.mse_loss(z_pred, batch["seq_obs"][:, k])
        )
        out["dyn_disagreement"] = float(dis.mean())
        out["support_score_data"] = float(agent.support.score(target, z).mean())
    return out


def main() -> None:
    args = parse_args()
    overrides = agent_settings(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    if (out / "config.json").exists() and not args.resume:
        raise FileExistsError(
            "run directory already contains a configuration; use a new output directory or --resume"
        )
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    cache = LatentCache(args.latent_cache, mmap=args.mmap)
    buffer = TemporalBuffer(
        cache,
        span=args.span,
        chunk=args.chunk,
        device=args.device,
        seed=args.seed,
        train_split=args.train_split,
        split_mode=args.split_mode,
        goal_discount=args.goal_discount,
        p_current=args.p_current,
        p_traj=args.p_traj,
        p_random=args.p_random,
        horizon_max=args.horizon_max,
    )
    agent = build_agent(
        args.variant, buffer.obs_dim, buffer.action_dim, args.device, overrides
    )
    agent.generator.manual_seed(args.seed)
    meta = buffer.metadata()
    meta.update(
        {
            "variant": args.variant,
            "train_steps": args.train_steps,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "controller": f"closed_loop_chunk{args.chunk}",
        }
    )
    agent.buffer_meta = meta
    start_step = 0
    if args.resume and Path(args.resume).is_file():
        row = torch.load(args.resume, map_location="cpu", weights_only=False)
        old = row.get("buffer_meta") or {}
        for key in (
            "dataset",
            "latent_cache",
            "variant",
            "obs_dim",
            "action_dim",
            "chunk",
            "span",
            "horizon_max",
            "split_mode",
        ):
            old_value = old.get(key, "row" if key == "split_mode" else None)
            if old_value != meta.get(key):
                raise ValueError(
                    f"resume mismatch for {key}: {old_value!r} != {meta.get(key)!r}"
                )
        agent.load_state_dict(row)
        agent.buffer_meta = meta
        start_step = int(old.get("completed_steps", 0))
        if args.train_steps <= start_step:
            raise ValueError(f"--train-steps must exceed the resumed step {start_step}")
        agent.step = start_step
        print(f"[train] resuming at step {start_step}", flush=True)
    (out / "config.json").write_text(
        json.dumps(
            {
                "method": "dtrc",
                "train": vars(args),
                "agent": agent.state_dict()["cfg"],
                "buffer_meta": meta,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    n_params = sum((p.numel() for p in agent.policy.parameters()))
    print(
        f"[train] variant={args.variant} rows={len(buffer.valid_indices)} obs_dim={buffer.obs_dim} action_dim={buffer.action_dim} chunk={args.chunk} span={args.span} hmax={args.horizon_max} policy_params={n_params / 1000000.0:.2f}M steps={args.train_steps}",
        flush=True,
    )
    log_path = out / "train.jsonl"
    (started, last_log) = (time.time(), start_step)
    for step in range(start_step + 1, args.train_steps + 1):
        info = agent.update(buffer.sample(args.batch_size))
        if step == 1 or step % args.log_every == 0:
            elapsed = max(time.time() - started, 1e-06)
            parts = [
                f"[train] step {step}/{args.train_steps}",
                f"pi {info['loss/pi']:.4f}",
            ]
            if "loss/critic_goal" in info:
                parts += [
                    f"dgoal {info['loss/critic_goal']:.3f}",
                    f"dloc {info['loss/critic_local']:.3f}",
                    f"d {info['value/d_goal_mean']:.2f}",
                    f"adv {info['value/adv_data_mean']:.3f}",
                    f"w {info['value/awr_weight_mean']:.2f}",
                ]
            if "loss/dyn" in info:
                parts += [
                    f"dyn {info['loss/dyn']:.4f}",
                    f"sup {info['loss/support']:.3f}",
                ]
            if "im/use_frac" in info:
                parts += [
                    f"im_valid {info['im/valid_frac']:.2f}",
                    f"im_use {info['im/use_frac']:.2f}",
                    f"im_adv {info['im/adv_mean_used']:.2f}",
                ]
            parts.append(f"{(step - last_log) / elapsed:.1f} it/s")
            print(" | ".join(parts), flush=True)
            (started, last_log) = (time.time(), step)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, **info}) + "\n")
        if (
            args.holdout_every > 0
            and step % args.holdout_every == 0
            and len(buffer.holdout_indices)
        ):
            hold = holdout_metrics(agent, buffer, args.batch_size)
            print(
                f"[holdout] step {step} "
                + " ".join((f"{k}={v:.4f}" for (k, v) in hold.items())),
                flush=True,
            )
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, "holdout": hold}) + "\n")
        if step % args.save_every == 0 or step == args.train_steps:
            meta["completed_steps"] = step
            agent.buffer_meta = meta
            atomic_save(agent, checkpoints / "latest.pt")
    meta["completed_steps"] = args.train_steps
    agent.buffer_meta = meta
    atomic_save(agent, checkpoints / "final.pt")
    (out / "DTRC_TRAIN_COMPLETE").touch()
    print(f"[train] DTRC TRAIN COMPLETE: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
