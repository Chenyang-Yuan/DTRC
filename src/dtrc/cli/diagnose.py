"""Temporal geometry and model diagnostics on a sampled episode subset."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from dtrc.cache import LatentCache
from dtrc.agent import load_agent


def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    result = np.empty(len(x), dtype=np.float64)
    result[order] = np.arange(len(x), dtype=np.float64)
    return result


def corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def auroc(negative: np.ndarray, positive: np.ndarray) -> float:
    scores = np.concatenate([negative, positive])
    labels = np.concatenate([np.zeros(len(negative)), np.ones(len(positive))])
    r = ranks(scores)
    (n_pos, n_neg) = (len(positive), len(negative))
    return float((r[labels == 1].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--latent-cache", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--pairs", type=int, default=4096)
    p.add_argument("--gaps", default="1,2,4,8,16,25,50")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.pairs < 1:
        p.error("pairs must be positive")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cache = LatentCache(args.latent_cache, mmap=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    (agent, meta) = load_agent(checkpoint, args.device, act_mode="single")
    if not agent.cfg.use_critic:
        p.error("temporal diagnostics require a trained critic")
    device = torch.device(args.device)
    episodes = np.unique(cache.episode_idx)
    sampled_episodes = rng.choice(
        episodes, size=max(1, int(round(0.1 * len(episodes)))), replace=False
    )
    pool = np.flatnonzero(np.isin(cache.episode_idx, sampled_episodes))
    gaps_requested = [int(x) for x in args.gaps.split(",") if x.strip()]
    if not gaps_requested or min(gaps_requested) < 1:
        p.error("gaps must contain positive recorded offsets")
    geometry = []
    (all_d, all_gap) = ([], [])
    with torch.no_grad():
        for gap in gaps_requested:
            valid = pool[pool + gap <= cache.episode_end[pool]]
            if not len(valid):
                continue
            rows = rng.choice(
                valid, size=min(args.pairs, len(valid)), replace=len(valid) < args.pairs
            )
            goals = rows + gap
            z = torch.as_tensor(
                cache.normalized(rows), dtype=torch.float32, device=device
            )
            z1 = torch.as_tensor(
                cache.normalized(rows + 1), dtype=torch.float32, device=device
            )
            g = torch.as_tensor(
                cache.normalized(goals), dtype=torch.float32, device=device
            )
            d = agent.distance(z, g).cpu().numpy()
            d1 = agent.distance(z1, g).cpu().numpy()
            reverse = agent.distance(g, z).cpu().numpy()
            geometry.append(
                {
                    "gap": gap,
                    "n": len(rows),
                    "distance_mean": float(d.mean()),
                    "distance_mae_to_gap": float(np.abs(d - gap).mean()),
                    "one_step_progress_rate": float(np.mean(d1 < d)),
                    "one_step_progress_mean": float(np.mean(d - d1)),
                    "reverse_minus_forward_mean": float(np.mean(reverse - d)),
                }
            )
            all_d.append(d)
            all_gap.append(np.full(len(d), gap))
        dynamics = []
        support = None
        if agent.cfg.use_model:
            max_h = max(min(max(gaps_requested), agent.cfg.span), 1)
            valid = pool[pool + max_h <= cache.episode_end[pool]]
            rows = rng.choice(
                valid, size=min(args.pairs, len(valid)), replace=len(valid) < args.pairs
            )
            z0 = torch.as_tensor(
                cache.normalized(rows), dtype=torch.float32, device=device
            )
            actions_np = (cache.actions[rows] - cache.action_mean) / cache.action_std
            data_action = torch.as_tensor(
                actions_np, dtype=torch.float32, device=device
            )
            ood_action = (data_action + 2.0 * torch.randn_like(data_action)).clamp(
                -3, 3
            )
            data_chunk = (
                data_action[:, None, :]
                .expand(-1, agent.cfg.chunk, -1)
                .reshape(len(rows), -1)
            )
            ood_chunk = (
                ood_action[:, None, :]
                .expand(-1, agent.cfg.chunk, -1)
                .reshape(len(rows), -1)
            )
            in_score = agent.support.score(data_chunk, z0).cpu().numpy()
            out_score = agent.support.score(ood_chunk, z0).cpu().numpy()
            support = {
                "data_score_mean": float(in_score.mean()),
                "noisy_score_mean": float(out_score.mean()),
                "noisy_vs_data_auroc": auroc(in_score, out_score),
                "noise_definition": "standardized data action plus N(0, 2), clipped to [-3,3]",
            }
            z_pred = z0
            dis_sum = torch.zeros(len(rows), device=device)
            for h in range(1, max_h + 1):
                a_np = (
                    cache.actions[rows + h - 1] - cache.action_mean
                ) / cache.action_std
                a = torch.as_tensor(a_np, dtype=torch.float32, device=device)
                (z_pred, dis) = agent.dyn.rollout(z_pred, a[:, None, :])
                dis_sum += dis
                if h in gaps_requested:
                    target = torch.as_tensor(
                        cache.normalized(rows + h), dtype=torch.float32, device=device
                    )
                    per = (z_pred - target).pow(2).mean(-1).cpu().numpy()
                    uncertainty = (dis_sum / h).cpu().numpy()
                    dynamics.append(
                        {
                            "horizon": h,
                            "mse_mean": float(per.mean()),
                            "disagreement_mean": float(uncertainty.mean()),
                            "error_disagreement_pearson": corr(per, uncertainty),
                        }
                    )
    distances = np.concatenate(all_d) if all_d else np.empty(0)
    gap_values = np.concatenate(all_gap) if all_gap else np.empty(0)
    result = {
        "protocol": "dtrc_episode_diagnostics_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "latent_cache": str(Path(args.latent_cache).resolve()),
        "variant": meta.get("variant"),
        "sampled_episode_count": int(len(sampled_episodes)),
        "geometry": geometry,
        "geometry_summary": {
            "ordinal_rank_distance_gap": (
                corr(ranks(distances), ranks(gap_values)) if len(distances) else None
            ),
            "pearson_distance_gap": (
                corr(distances, gap_values) if len(distances) else None
            ),
        },
        "dynamics": dynamics,
        "support": support,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
