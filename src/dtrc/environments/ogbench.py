"""Future-goal evaluation, simulator restoration, and task success predicates."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

TASKS = {
    "pointmaze": {
        "kind": "maze",
        "loco": "point",
        "maze_type": "large",
        "tolerance": 1.0,
    },
    "antmaze": {"kind": "maze", "loco": "ant", "maze_type": "large", "tolerance": 0.5},
    "antmaze-medium": {
        "kind": "maze",
        "loco": "ant",
        "maze_type": "medium",
        "tolerance": 0.5,
    },
    "humanoidmaze-medium": {
        "kind": "maze",
        "loco": "humanoid",
        "maze_type": "medium",
        "tolerance": 0.5,
    },
    "cube-double": {"kind": "cube"},
    "scene": {"kind": "scene"},
    "puzzle-3x3": {"kind": "puzzle"},
}


def episode_layout(terminals: np.ndarray):
    terminals = np.asarray(terminals, dtype=bool).reshape(-1)
    run_end = terminals & np.concatenate([~terminals[1:], np.ones(1, dtype=bool)])
    ends = np.flatnonzero(run_end)
    if len(ends) == 0 or ends[-1] != len(terminals) - 1:
        raise ValueError("could not reconstruct compact OGBench episodes")
    starts = np.concatenate([np.zeros(1, dtype=np.int64), ends[:-1] + 1])
    lengths = ends - starts + 1
    return (starts, lengths)


def sample_problems(terminals, num_eval, goal_offset, seed):
    (starts, lengths) = episode_layout(terminals)
    (episode_ids, local_starts) = ([], [])
    for episode, length in enumerate(lengths):
        max_start = int(length) - int(goal_offset) - 1
        if max_start < 0:
            continue
        episode_ids.extend([episode] * (max_start + 1))
        local_starts.extend(range(max_start + 1))
    if len(episode_ids) < num_eval:
        raise ValueError(
            f"only {len(episode_ids)} valid problems for num_eval={num_eval}"
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(episode_ids), size=num_eval, replace=False)
    return (
        starts,
        np.asarray(episode_ids, dtype=np.int64)[chosen],
        np.asarray(local_starts, dtype=np.int64)[chosen],
    )


def load_manifest(path, terminals, num_eval, goal_offset):
    row = json.loads(Path(path).read_text(encoding="utf-8"))
    if not str(row.get("format", "")).endswith("_paired_manifest_v1"):
        raise ValueError(f"unsupported manifest format: {row.get('format')}")
    if int(row["num_eval"]) != int(num_eval):
        raise ValueError("manifest num_eval does not match --num_eval")
    if int(row["goal_offset"]) != int(goal_offset):
        raise ValueError("manifest goal_offset does not match evaluator")
    (layout_starts, lengths) = episode_layout(terminals)
    episodes = np.asarray(row["episodes"], dtype=np.int64)
    starts = np.asarray(row["starts"], dtype=np.int64)
    if len(episodes) != num_eval or len(starts) != num_eval:
        raise ValueError("manifest arrays do not have num_eval entries")
    if np.any(episodes < 0) or np.any(episodes >= len(lengths)):
        raise ValueError("manifest contains invalid episode indices")
    if np.any(starts < 0) or np.any(starts + goal_offset >= lengths[episodes]):
        raise ValueError("manifest contains invalid start/goal indices")
    return (layout_starts, episodes, starts)


def make_env(task):
    spec = TASKS[task]
    if spec["kind"] == "maze":
        from stable_worldmodel.envs.ogbench.maze_env import MazeEnv

        return MazeEnv(
            loco_env_type=spec["loco"],
            maze_env_type="maze",
            maze_type=spec["maze_type"],
            ob_type="states",
            terminate_at_goal=False,
        )
    if spec["kind"] == "cube":
        from stable_worldmodel.envs.ogbench.cube_env import CubeEnv

        return CubeEnv(
            env_type="double",
            ob_type="states",
            permute_blocks=False,
            mode="task",
            terminate_at_goal=False,
            visualize_info=False,
        )
    if spec["kind"] == "puzzle":
        from ogbench.manipspace.envs.puzzle_env import PuzzleEnv

        return PuzzleEnv(
            env_type="3x3",
            ob_type="states",
            mode="task",
            terminate_at_goal=False,
            visualize_info=False,
        )
    from stable_worldmodel.envs.ogbench.scene_env import SceneEnv

    return SceneEnv(
        ob_type="states",
        permute_blocks=False,
        mode="task",
        terminate_at_goal=False,
        visualize_info=False,
    )


def restore(env, task, qpos, qvel, button_states, seed):
    state = np.concatenate([qpos, qvel]).astype(np.float64, copy=False)
    kind = TASKS[task]["kind"]
    if kind not in {"scene", "puzzle"}:
        (observation, _) = env.reset(seed=seed, options={"state": state})
        return np.asarray(observation, dtype=np.float32)
    env.reset(seed=seed)
    if button_states is None:
        raise KeyError(f"{task} evaluation requires NPZ key 'button_states'")
    if kind == "puzzle":
        env.set_state(
            np.asarray(qpos, dtype=np.float64),
            np.asarray(qvel, dtype=np.float64),
            np.asarray(button_states, dtype=np.int64),
        )
        return np.asarray(env.compute_observation(), dtype=np.float32)
    kwargs = {
        f"button_state_{index}": int(value)
        for (index, value) in enumerate(button_states)
    }
    env.set_state(
        np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64), **kwargs
    )
    return np.asarray(env.compute_observation(), dtype=np.float32)


def goal_distance(task, observation, goal, current_buttons=None, goal_buttons=None):
    observation = np.asarray(observation)
    goal = np.asarray(goal)
    kind = TASKS[task]["kind"]
    if kind == "maze":
        tolerance = TASKS[task]["tolerance"]
        return float(np.linalg.norm(observation[:2] - goal[:2]) / tolerance)
    if kind == "cube":
        errors = [
            np.linalg.norm(observation[start : start + 3] - goal[start : start + 3])
            / 0.4
            for start in (19, 28)
        ]
        return float(max(errors))
    if kind == "puzzle":
        if current_buttons is None or goal_buttons is None:
            raise ValueError("Puzzle goal distance requires logical button states")
        return float(
            np.count_nonzero(np.asarray(current_buttons) != np.asarray(goal_buttons))
        )
    errors = [np.linalg.norm(observation[19:22] - goal[19:22]) / 0.4]
    for start in (28, 32):
        same = np.argmax(observation[start : start + 2]) == np.argmax(
            goal[start : start + 2]
        )
        errors.append(0.0 if same else 2.0)
    errors.extend(
        [
            abs(float(observation[36] - goal[36])) / 0.72,
            abs(float(observation[38] - goal[38])) / 0.6,
        ]
    )
    return float(max(errors))


def current_button_states(env):
    return np.asarray(env.unwrapped._cur_button_states, dtype=np.int64).copy()


def is_success(task, distance):
    if TASKS[task]["kind"] == "puzzle":
        return bool(distance == 0.0)
    return bool(distance <= 1.0)
