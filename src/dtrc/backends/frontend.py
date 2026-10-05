"""Pixel preprocessing and feature normalization for closed-loop control."""

from __future__ import annotations
from collections import deque
import numpy as np
import torch
from .lewm import LeWMBackend, preprocess_uint8


class LatentFrontend:

    def __init__(self, backend: LeWMBackend, meta: dict):
        self.backend = backend
        self.history_size = int(meta["history_size"])
        self.frameskip = int(meta["frameskip"])
        device = backend.device
        self.feature_mean = torch.as_tensor(
            meta["feature_mean"], dtype=torch.float32, device=device
        )
        self.feature_std = torch.as_tensor(
            meta["feature_std"], dtype=torch.float32, device=device
        ).clamp_min(1e-06)
        expected = int(meta["feature_dim"])
        if backend.latent_dim != expected:
            raise ValueError(
                f"LeWM latent dim {backend.latent_dim} != cache dim {expected}"
            )

    @torch.no_grad()
    def encode_frames(self, frames: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(np.ascontiguousarray(frames)).to(self.backend.device)
        latent = self.backend.encode_pixels(preprocess_uint8(tensor))
        return (latent - self.feature_mean) / self.feature_std

    def stack_history(self, history: deque) -> torch.Tensor:
        frames = list(history)
        while len(frames) < self.history_size:
            frames.insert(0, frames[0])
        return torch.cat(frames[-self.history_size :], dim=-1)

    def goal_observation(self, goal_latent: torch.Tensor) -> torch.Tensor:
        return goal_latent.repeat(1, self.history_size)
