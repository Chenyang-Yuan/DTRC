"""Construct matched visual start--goal evaluation pairs."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from dtrc.environments.lewm import get_task


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    import stable_worldmodel as swm

    if args.num_eval <= 0 or args.goal_offset <= 0:
        raise ValueError("num-eval and goal-offset must be positive")
    spec = get_task(args.task)
    dataset_name = args.dataset or spec["eval_dataset"]
    dataset = swm.data.load_dataset(dataset_name, cache_dir=args.cache_dir)
    ep_key = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep = np.asarray(dataset.get_col_data(ep_key), dtype=np.int64)
    step = np.asarray(dataset.get_col_data("step_idx"), dtype=np.int64)
    index_hash = hashlib.sha256()
    index_hash.update(ep.tobytes())
    index_hash.update(step.tobytes())
    (episodes, first) = np.unique(ep, return_index=True)
    order = np.argsort(first)
    episodes = episodes[order]
    lengths = np.asarray([step[ep == item].max() + 1 for item in episodes])
    max_start = {
        int(item): int(length - args.goal_offset - 1)
        for (item, length) in zip(episodes, lengths)
    }
    valid = np.flatnonzero(
        np.asarray([step[i] <= max_start[int(ep[i])] for i in range(len(ep))])
    )
    if len(valid) < args.num_eval:
        raise ValueError(f"only {len(valid)} valid starts for N={args.num_eval}")
    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(valid, size=args.num_eval, replace=False))
    selected_ep = ep[rows]
    starts = step[rows]
    goals = starts + args.goal_offset
    problems = [
        {
            "evaluation_index": int(i),
            "episode": int(episode),
            "start": int(start),
            "goal": int(goal),
            "start_dataset_index": int(row),
        }
        for (i, (episode, start, goal, row)) in enumerate(
            zip(selected_ep, starts, goals, rows)
        )
    ]
    identity = {
        "task": args.task,
        "dataset": dataset_name,
        "num_eval": args.num_eval,
        "goal_offset": args.goal_offset,
        "seed": args.seed,
        "num_rows": int(len(ep)),
        "dataset_index_sha256": index_hash.hexdigest(),
        "episodes": selected_ep.tolist(),
        "starts": starts.tolist(),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    payload = {
        "format": "stable_worldmodel_visual_manifest_v1",
        **identity,
        "manifest_sha256": digest,
        "num_episodes": int(len(episodes)),
        "global_indices": rows.tolist(),
        "goals": goals.tolist(),
        "problems": problems,
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("manifest already exists; use a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved visual manifest: {output.resolve()}")
    print(
        f"task={args.task} dataset={dataset_name} N={args.num_eval} offset={args.goal_offset} seed={args.seed} valid={len(valid)}"
    )
    print(f"manifest_sha256={digest}")


if __name__ == "__main__":
    main()
