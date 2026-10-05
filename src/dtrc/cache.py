"""Normalized per-frame LeWM features and recorded transition indices."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch


class LatentCache:

    def __init__(self, cache_dir: str | Path, mmap: bool = False):
        self.cache_dir = Path(cache_dir).resolve()
        self.meta = json.loads(
            (self.cache_dir / "meta.json").read_text(encoding="utf-8")
        )
        if self.meta.get("format") != "lewm_latent_frame_cache_v1":
            raise ValueError(f"not a LeWM latent cache: {self.cache_dir}")
        raw = np.load(self.cache_dir / "latents.npy", mmap_mode="r" if mmap else None)
        self.feature_mean = np.asarray(self.meta["feature_mean"], dtype=np.float32)
        self.feature_std = np.maximum(
            np.asarray(self.meta["feature_std"], dtype=np.float32), 1e-06
        )
        if mmap:
            self.raw_latents = raw
            self.latents = None
        else:
            self.raw_latents = None
            self.latents = (
                (np.asarray(raw, dtype=np.float32) - self.feature_mean)
                / self.feature_std
            ).astype(np.float32)
        index = np.load(self.cache_dir / "index.npz")
        self.actions = np.asarray(index["action"], dtype=np.float32)
        self.episode_idx = np.asarray(index["episode_idx"], dtype=np.int64)
        self.step_idx = np.asarray(index["step_idx"], dtype=np.int64)
        self.episode_start = np.asarray(index["episode_start"], dtype=np.int64)
        self.episode_end = np.asarray(index["episode_end"], dtype=np.int64)
        self.row_count = int(self.meta["row_count"])
        self.feature_dim = int(self.meta["feature_dim"])
        self.action_dim = int(self.actions.shape[1])
        self.action_mean = np.asarray(self.meta["action_mean"], dtype=np.float32)
        self.action_std = np.maximum(
            np.asarray(self.meta["action_std"], dtype=np.float32), 1e-06
        )
        if len(self.actions) != self.row_count:
            raise ValueError("index/latent row mismatch")

    def normalized(self, rows: np.ndarray) -> np.ndarray:
        if self.latents is not None:
            return self.latents[rows]
        values = np.asarray(self.raw_latents[rows], dtype=np.float32)
        return (values - self.feature_mean) / self.feature_std

    def normalize_torch(self, latent: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(
            self.feature_mean, device=latent.device, dtype=latent.dtype
        )
        std = torch.as_tensor(
            self.feature_std, device=latent.device, dtype=latent.dtype
        )
        return (latent - mean) / std
