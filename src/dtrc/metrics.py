"""Action smoothness, distance monotonicity, and decision latency."""

from __future__ import annotations
import numpy as np


def trajectory_metrics(actions, distances):
    action_array = np.asarray(actions, dtype=np.float64)
    distance_array = np.asarray(distances, dtype=np.float64)
    if len(action_array) >= 3:
        second = action_array[2:] - 2.0 * action_array[1:-1] + action_array[:-2]
        jerk = float(np.linalg.norm(second, axis=-1).mean())
    else:
        jerk = None
    changes = np.diff(distance_array)
    step_monotonicity = float(np.mean(changes <= 1e-08)) if len(changes) else None
    episode_monotonic = bool(np.all(changes <= 1e-08)) if len(changes) else True
    total_variation = float(np.abs(changes).sum())
    progress = float(max(distance_array[0] - distance_array[-1], 0.0))
    efficiency = float(progress / total_variation) if total_variation > 1e-12 else 0.0
    return (jerk, step_monotonicity, episode_monotonic, efficiency)


class ParallelTrajectoryRecorder:

    def __init__(self, num_envs):
        self.active = np.ones(int(num_envs), dtype=bool)
        self.actions = [[] for _ in range(int(num_envs))]
        self.distances = [[] for _ in range(int(num_envs))]
        self.success_steps = [None for _ in range(int(num_envs))]
        self.decision_seconds = 0.0
        self.decision_calls = 0

    def record_decision(self, actions, distances, seconds):
        actions = np.asarray(actions)
        distances = np.asarray(distances).reshape(-1)
        for index in np.flatnonzero(self.active):
            self.actions[index].append(
                np.asarray(actions[index], dtype=np.float32).copy()
            )
            self.distances[index].append(float(distances[index]))
        self.decision_seconds += float(seconds)
        self.decision_calls += 1

    def mark_done(self, terminated, truncated):
        terminated = np.asarray(terminated, dtype=bool).reshape(-1)
        truncated = np.asarray(truncated, dtype=bool).reshape(-1)
        step = max((len(row) for row in self.actions), default=0)
        newly_successful = self.active & terminated
        for index in np.flatnonzero(newly_successful):
            self.success_steps[index] = step
        self.active &= ~(terminated | truncated)

    def summary(self):
        per_episode = []
        for actions, distances, success_step in zip(
            self.actions, self.distances, self.success_steps
        ):
            (jerk, step_mono, episode_mono, efficiency) = trajectory_metrics(
                actions, distances
            )
            per_episode.append(
                {
                    "action_jerk": jerk,
                    "step_monotonicity": step_mono,
                    "episode_monotonic": episode_mono,
                    "path_efficiency": efficiency,
                    "success_step": success_step,
                    "actions": [action.tolist() for action in actions],
                    "latent_distance_trace": distances,
                }
            )

        def avg(key):
            values = [row[key] for row in per_episode if row[key] is not None]
            return float(np.mean(values)) if values else None

        return {
            "distance_space": "shared_frozen_lewm_cosine_distance",
            "action_jerk_mean": avg("action_jerk"),
            "step_monotonicity_mean": avg("step_monotonicity"),
            "episode_monotonicity_rate": float(
                np.mean([row["episode_monotonic"] for row in per_episode])
            ),
            "path_efficiency_mean": avg("path_efficiency"),
            "successful_steps_mean": avg("success_step"),
            "decision_latency_ms": 1000.0
            * self.decision_seconds
            / max(self.decision_calls, 1),
            "decision_calls": self.decision_calls,
            "episodes": per_episode,
        }
