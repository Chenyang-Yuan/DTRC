"""LeWM benchmark datasets and simulator reset specifications."""

from __future__ import annotations
from copy import deepcopy

TASKS = {
    "pusht": {
        "train_dataset": "pusht_expert_train.h5",
        "eval_dataset": "pusht_expert_train.h5",
        "data_config": "pusht",
        "action_dim": 2,
        "world": {"env_name": "swm/PushT-v1"},
        "callables": [
            {"method": "_set_state", "args": {"state": {"value": "state"}}},
            {
                "method": "_set_goal_state",
                "args": {"goal_state": {"value": "goal_state"}},
            },
        ],
        "lewm_config": "pusht.yaml",
        "lewm_checkpoint": "pusht/lewm",
    },
    "tworoom": {
        "train_dataset": "tworoom.h5",
        "eval_dataset": "tworoom.h5",
        "data_config": "tworoom",
        "action_dim": 2,
        "world": {"env_name": "swm/TwoRoom-v1"},
        "callables": [
            {"method": "_set_state", "args": {"state": {"value": "proprio"}}},
            {
                "method": "_set_goal_state",
                "args": {"goal_state": {"value": "goal_proprio"}},
            },
        ],
        "lewm_config": "tworoom.yaml",
        "lewm_checkpoint": "tworoom/lewm",
    },
    "reacher": {
        "train_dataset": "reacher.h5",
        "eval_dataset": "dmc/reacher_random.h5",
        "data_config": "dmc",
        "action_dim": 2,
        "world": {"env_name": "swm/ReacherDMControl-v0", "task": "qpos_match"},
        "callables": [
            {
                "method": "set_state",
                "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}},
            },
            {
                "method": "set_target_qpos",
                "args": {"target_qpos": {"value": "goal_qpos"}},
            },
        ],
        "lewm_config": "reacher.yaml",
        "lewm_checkpoint": "reacher/lewm",
    },
    "cube": {
        "train_dataset": "ogbench/cube_single_expert.h5",
        "eval_dataset": "ogbench/cube_single_expert.h5",
        "data_config": "ogb",
        "action_dim": 5,
        "world": {
            "env_name": "swm/OGBCube-v0",
            "env_type": "single",
            "ob_type": "states",
            "multiview": False,
            "width": 224,
            "height": 224,
            "visualize_info": False,
            "terminate_at_goal": True,
        },
        "callables": [
            {
                "method": "set_state",
                "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}},
            },
            {
                "method": "set_target_pos",
                "args": {
                    "cube_id": {"value": 0, "in_dataset": False},
                    "target_pos": {"value": "goal_privileged_block_0_pos"},
                    "target_quat": {"value": "goal_privileged_block_0_quat"},
                },
            },
        ],
        "lewm_config": "cube.yaml",
        "lewm_checkpoint": "cube/lewm",
    },
}


def get_task(name: str) -> dict:
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; choices={sorted(TASKS)}")
    return deepcopy(TASKS[name])
