"""Small deterministic trajectory data for numerical tests."""

import json
from pathlib import Path

import numpy as np


def make_cache(directory: Path, feature_dim=12, action_dim=2, episodes=8, length=31):
    rng = np.random.default_rng(7)
    count = episodes * length
    actions = rng.normal(0, 0.5, (count, action_dim)).astype(np.float32)
    features = np.empty((count, feature_dim), dtype=np.float32)
    for episode in range(episodes):
        state = rng.normal(size=feature_dim).astype(np.float32)
        for step in range(length):
            row = episode * length + step
            features[row] = state
            state = state + 0.1 * np.resize(actions[row], feature_dim)
    episode_ids = np.repeat(np.arange(episodes), length)
    starts = episode_ids * length
    directory.mkdir(parents=True)
    np.save(directory / "latents.npy", features)
    np.savez(
        directory / "index.npz",
        action=actions,
        episode_idx=episode_ids,
        step_idx=np.tile(np.arange(length), episodes),
        episode_start=starts,
        episode_end=starts + length - 1,
    )
    meta = {
        "format": "lewm_latent_frame_cache_v1",
        "task": "pusht",
        "dataset": "test_trajectories.h5",
        "lewm_checkpoint": "test_encoder",
        "lewm_weights_sha256": None,
        "lewm": {},
        "row_count": count,
        "feature_dim": feature_dim,
        "feature_mean": features.mean(0).tolist(),
        "feature_std": np.maximum(features.std(0), 1e-6).tolist(),
        "action_mean": actions.mean(0).tolist(),
        "action_std": np.maximum(actions.std(0), 1e-6).tolist(),
        "privileged_state_model_access": False,
    }
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


SMALL = dict(
    hidden=32,
    rep_dim=8,
    sym_dim=4,
    asym_dim=4,
    depth=1,
    span=4,
    n_step=4,
    dyn_rollout=3,
    dyn_depth=1,
    dyn_heads=3,
    n_candidates=3,
    flow_steps=2,
    improve_start=2,
)
