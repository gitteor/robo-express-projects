# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""E0509 Cube Stack 환경 패키지"""

import gymnasium as gym
from .e0509_cube_stack_env import E0509CubeStackEnv, E0509CubeStackEnvCfg

gym.register(
    id="Isaac-E0509-CubeStack-Direct-v0",
    entry_point="isaaclab_tasks.direct.e0509_cube_stack.e0509_cube_stack_env:E0509CubeStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": E0509CubeStackEnvCfg,
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.direct.e0509_cube_stack.agents.rsl_rl_ppo_cfg:E0509CubeStackPPORunnerCfg",
    },
)

print("[E0509CubeStack] Registered: Isaac-E0509-CubeStack-Direct-v0")
