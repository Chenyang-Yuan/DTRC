"""Pixel-conditioned action execution with a fixed remaining-horizon budget."""

from __future__ import annotations
from collections import deque
import numpy as np
import torch

try:
    import stable_worldmodel as swm

    _BasePolicy = swm.policy.BasePolicy
except ImportError:

    class _BasePolicy:

        def __init__(self, **kwargs):
            self.env = None

        def set_env(self, env):
            self.env = env


class DTRCPolicy(_BasePolicy):

    def __init__(self, agent, encoder, meta: dict, budget: int):
        super().__init__()
        self.agent = agent
        self.encoder = encoder
        self.meta = meta
        self.budget = int(budget)
        self.chunk = int(agent.cfg.chunk)
        self.horizon_max = int(agent.cfg.horizon_max)
        self.device = agent.device
        self.action_mean = torch.as_tensor(
            meta["action_scaler_mean"], dtype=torch.float32, device=self.device
        )
        self.action_scale = torch.as_tensor(
            meta["action_scaler_scale"], dtype=torch.float32, device=self.device
        ).clamp_min(1e-06)
        self.low = self.high = None
        self.steps = None
        self.queues = None
        self.goal_latent = None
        self.model_input_keys_seen = {"pixels", "goal"}

    def set_env(self, env) -> None:
        super().set_env(env)
        n = int(getattr(env, "num_envs", 1))
        self.steps = np.zeros(n, dtype=np.int64)
        self.queues = [deque() for _ in range(n)]
        self.goal_latent = None
        space = getattr(env, "single_action_space", None)
        if space is not None and hasattr(space, "low"):
            self.low = torch.as_tensor(
                np.array(space.low, dtype=np.float32), device=self.device
            )
            self.high = torch.as_tensor(
                np.array(space.high, dtype=np.float32), device=self.device
            )

    @torch.no_grad()
    def get_action(self, info_dict: dict, **kwargs) -> np.ndarray:
        if "pixels" not in info_dict or "goal" not in info_dict:
            raise KeyError("DTRC requires pixels and goal observations")
        pixels = np.asarray(info_dict["pixels"])
        goals = np.asarray(info_dict["goal"])
        current = pixels[:, -1] if pixels.ndim == 5 else pixels
        goal_images = goals[:, -1] if goals.ndim == 5 else goals
        n = len(current)
        flush = np.asarray(info_dict.get("_needs_flush", np.zeros(n, bool))).reshape(-1)
        if flush.any():
            self.steps[flush] = 0
            for index in np.flatnonzero(flush):
                self.queues[index].clear()
        if self.goal_latent is None or flush.any():
            self.goal_latent = self.encoder.encode_frames(goal_images)
        need = [i for i in range(n) if len(self.queues[i]) == 0]
        if need:
            z = self.encoder.encode_frames(current[need])
            remaining = np.clip(self.budget - self.steps[need], 1, self.horizon_max)
            h_norm = torch.as_tensor(
                remaining / self.horizon_max, dtype=torch.float32, device=self.device
            )
            chunk = self.agent.act(z, self.goal_latent[need], h_norm)
            chunk = chunk * self.action_scale + self.action_mean
            if self.low is not None:
                chunk = torch.maximum(torch.minimum(chunk, self.high), self.low)
            chunk = chunk.cpu().numpy().astype(np.float32)
            for row, index in enumerate(need):
                self.queues[index].extend(chunk[row])
        self.steps += 1
        return np.stack([self.queues[i].popleft() for i in range(n)], axis=0)
