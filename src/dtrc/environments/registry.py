"""Visual OGBench task names, datasets, and action dimensions."""

TASKS = {
    "pointmaze": {
        "visual_dataset": "visual-pointmaze-large-navigate-v0.lance",
        "train_state_dataset": "pointmaze-large-navigate-v0.npz",
        "eval_state_dataset": "pointmaze-large-navigate-v0.npz",
        "action_dim": 2,
    },
    "cube-double": {
        "visual_dataset": "visual-cube-double-play-v0.lance",
        "train_state_dataset": "visual-cube-double-play-v0.npz",
        "eval_state_dataset": "cube-double-play-v0.npz",
        "action_dim": 5,
    },
    "scene": {
        "visual_dataset": "visual-scene-play-v0.lance",
        "train_state_dataset": "visual-scene-play-v0.npz",
        "eval_state_dataset": "scene-play-v0.npz",
        "action_dim": 5,
    },
    "antmaze-medium": {
        "visual_dataset": "visual-antmaze-medium-navigate-v0.lance",
        "train_state_dataset": "visual-antmaze-medium-navigate-v0.npz",
        "eval_state_dataset": "antmaze-medium-navigate-v0.npz",
        "action_dim": 8,
    },
    "humanoidmaze-medium": {
        "visual_dataset": "visual-humanoidmaze-medium-navigate-v0.lance",
        "train_state_dataset": "visual-humanoidmaze-medium-navigate-v0.npz",
        "eval_state_dataset": "humanoidmaze-medium-navigate-v0.npz",
        "action_dim": 21,
    },
    "puzzle-3x3": {
        "visual_dataset": "visual-puzzle-3x3-play-v0.lance",
        "train_state_dataset": "visual-puzzle-3x3-play-v0.npz",
        "eval_state_dataset": "puzzle-3x3-play-v0.npz",
        "action_dim": 5,
    },
}

OGBENCH_TASKS = tuple(TASKS)
