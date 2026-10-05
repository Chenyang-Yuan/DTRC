"""Construct matched start--goal pairs from recorded OGBench episodes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dtrc.environments.ogbench import sample_problems
from dtrc.environments.registry import OGBENCH_TASKS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=OGBENCH_TASKS, required=True)
    parser.add_argument("--state-npz", required=True)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.num_eval, args.goal_offset) < 1:
        parser.error("num-eval and goal-offset must be positive")
    with np.load(args.state_npz, allow_pickle=False) as data:
        terminals = np.asarray(data["terminals"], dtype=bool)
    _, episodes, starts = sample_problems(
        terminals, args.num_eval, args.goal_offset, args.seed
    )
    result = {
        "format": "dtrc_paired_manifest_v1",
        "task": args.task,
        "num_eval": args.num_eval,
        "goal_offset": args.goal_offset,
        "seed": args.seed,
        "terminal_index_sha256": hashlib.sha256(terminals.tobytes()).hexdigest(),
        "episodes": episodes.tolist(),
        "starts": starts.tolist(),
    }
    result["manifest_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists; use a new manifest path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.num_eval} start--goal pairs.")


if __name__ == "__main__":
    main()
