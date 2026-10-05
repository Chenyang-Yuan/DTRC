"""Episode-bounded temporal sampling and hindsight goal relabeling."""

from __future__ import annotations
import numpy as np
import torch


class TemporalBuffer:
    """Sample normalized temporal windows without crossing episode boundaries."""

    def __init__(
        self,
        cache,
        span: int = 8,
        chunk: int = 1,
        device: str = "cuda",
        seed: int = 0,
        train_split: float = 0.9,
        split_mode: str = "row",
        goal_discount: float = 0.98,
        p_current: float = 0.1,
        p_traj: float = 0.6,
        p_random: float = 0.3,
        horizon_max: int = 50,
    ):
        if not np.isclose(p_current + p_traj + p_random, 1.0):
            raise ValueError("goal probabilities must sum to 1")
        if not 0 < goal_discount < 1:
            raise ValueError("goal_discount must be in (0, 1)")
        if span < chunk or chunk < 1:
            raise ValueError("need span >= chunk >= 1")
        self.cache = cache
        (self.span, self.chunk) = (int(span), int(chunk))
        self.device = torch.device(device)
        self.goal_discount = float(goal_discount)
        (self.p_current, self.p_traj, self.p_random) = (
            float(p_current),
            float(p_traj),
            float(p_random),
        )
        self.horizon_max = int(horizon_max)
        self.rng = np.random.default_rng(seed)
        rows = np.arange(cache.row_count, dtype=np.int64)
        valid = np.flatnonzero(rows + self.span <= cache.episode_end)
        if len(valid) == 0:
            raise ValueError("no window of `span` steps fits inside an episode")
        if not 0.0 < train_split <= 1.0:
            raise ValueError("train_split must lie in (0, 1]")
        if split_mode not in {"row", "episode"}:
            raise ValueError("split_mode must be 'row' or 'episode'")
        if train_split < 1.0:
            if split_mode == "episode":
                episodes = np.unique(cache.episode_idx[valid])
                if len(episodes) < 2:
                    raise ValueError(
                        "episode splitting requires at least two eligible episodes; "
                        "use train_split=1.0 to train on the only available episode"
                    )
                order = self.rng.permutation(episodes)
                cut = min(max(int(len(order) * train_split), 1), len(order) - 1)
                train_episodes = order[:cut]
                holdout_episodes = order[cut:]
                train_mask = np.isin(cache.episode_idx[valid], train_episodes)
                holdout_mask = np.isin(cache.episode_idx[valid], holdout_episodes)
                self.holdout_indices = np.sort(valid[holdout_mask])
                valid = valid[train_mask]
            else:
                order = self.rng.permutation(len(valid))
                cut = int(len(order) * train_split)
                self.holdout_indices = np.sort(valid[order[cut:]])
                valid = valid[order[:cut]]
        else:
            self.holdout_indices = np.zeros(0, dtype=np.int64)
        if len(valid) == 0:
            raise ValueError("training split contains no eligible starts")
        self.valid_indices = np.sort(valid)
        self.train_split = float(train_split)
        self.split_mode = split_mode
        self.obs_dim = cache.feature_dim
        self.action_dim = cache.action_dim

    def _sample_goals(self, rows: np.ndarray):
        batch = len(rows)
        remaining = self.cache.episode_end[rows] - rows
        u = self.rng.random(batch)
        offsets = np.ceil(
            np.log(1 - self.rng.random(batch)) / np.log(self.goal_discount)
        ).astype(np.int64)
        traj_goals = rows + np.clip(offsets, 1, remaining)
        goals = np.where(u < self.p_current, rows + self.chunk, traj_goals)
        use_random = u >= self.p_current + self.p_traj
        random_goals = self.rng.integers(0, self.cache.row_count, size=batch)
        goals = np.where(use_random, random_goals, goals)
        same = (self.cache.episode_idx[goals] == self.cache.episode_idx[rows]) & (
            goals > rows
        )
        gaps = np.where(same, goals - rows, -1).astype(np.int64)
        return (goals, gaps)

    def sample_rows(self, rows: np.ndarray) -> dict:
        rows = np.asarray(rows, dtype=np.int64)
        (goal_rows, gaps) = self._sample_goals(rows)
        offsets = np.arange(self.span + 1, dtype=np.int64)
        seq_rows = rows[:, None] + offsets[None]
        seq_obs = self.cache.normalized(seq_rows.reshape(-1)).reshape(
            len(rows), self.span + 1, -1
        )
        act_rows = seq_rows[:, :-1]
        actions = (
            self.cache.actions[act_rows.reshape(-1)] - self.cache.action_mean
        ) / self.cache.action_std
        actions = actions.reshape(len(rows), self.span, -1)
        random_gap = 1 + np.floor(self.rng.random(len(rows)) * self.horizon_max).astype(
            np.int64
        )
        h_steps = np.where(gaps > 0, gaps, random_gap)
        h_norm = (np.clip(h_steps, 1, self.horizon_max) / self.horizon_max).astype(
            np.float32
        )

        def floats(value):
            return torch.as_tensor(
                np.ascontiguousarray(value), dtype=torch.float32, device=self.device
            )

        return {
            "seq_obs": floats(seq_obs),
            "seq_actions": floats(actions),
            "goals": floats(self.cache.normalized(goal_rows)),
            "gaps": torch.as_tensor(gaps, dtype=torch.long, device=self.device),
            "h_norm": floats(h_norm),
        }

    def sample(self, batch_size: int) -> dict:
        positions = self.rng.integers(0, len(self.valid_indices), size=batch_size)
        return self.sample_rows(self.valid_indices[positions])

    def sample_holdout(self, batch_size: int) -> dict:
        pool = self.holdout_indices if len(self.holdout_indices) else self.valid_indices
        positions = self.rng.integers(0, len(pool), size=batch_size)
        return self.sample_rows(pool[positions])

    def metadata(self) -> dict:
        meta = self.cache.meta
        return {
            "format": "dtrc_frozen_lewm_latent_v1",
            "method": "dtrc",
            "task": meta.get("task"),
            "dataset": meta.get("dataset"),
            "latent_cache": str(self.cache.cache_dir),
            "lewm_checkpoint": meta.get("lewm_checkpoint"),
            "lewm_weights_sha256": meta.get("lewm_weights_sha256"),
            "lewm": meta.get("lewm"),
            "encoder_frozen": True,
            "frameskip": 1,
            "history_size": 1,
            "chunk": self.chunk,
            "span": self.span,
            "feature_dim": self.cache.feature_dim,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "feature_mean": self.cache.feature_mean.tolist(),
            "feature_std": self.cache.feature_std.tolist(),
            "action_normalization": "training_dataset_standard_scaler",
            "action_scaler_mean": self.cache.action_mean.tolist(),
            "action_scaler_scale": self.cache.action_std.tolist(),
            "goal_discount": self.goal_discount,
            "goal_probs": [self.p_current, self.p_traj, self.p_random],
            "horizon_max": self.horizon_max,
            "train_split": self.train_split,
            "split_mode": self.split_mode,
            "train_rows": int(len(self.valid_indices)),
            "holdout_rows": int(len(self.holdout_indices)),
            "model_observation_keys": ["pixels", "goal"],
            "privileged_state_model_access": False,
        }
