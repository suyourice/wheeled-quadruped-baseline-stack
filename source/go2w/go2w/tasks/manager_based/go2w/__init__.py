# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

# =============================================================================
# Baseline locomotion (PPO, proprioception only)
# =============================================================================

gym.register(
    id="Flat-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2w_env_cfg:Go2wEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Flat-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2w_env_cfg:Go2wEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# =============================================================================
# 2 m/s flat pre-training (obstacle-env compatible, for checkpoint transfer)
# =============================================================================

gym.register(
    id="Fast-Flat-2m-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2w_env_cfg:Go2wFastFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FastFlatRunnerCfg",
    },
)

gym.register(
    id="Fast-Flat-2m-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2w_env_cfg:Go2wFastFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FastFlatRunnerCfg",
    },
)

# =============================================================================
# RL navigation teacher (PPO, proprio + obstacle positions)
# =============================================================================

gym.register(
    id="Nav-Teacher-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2w_obstacle_env_cfg:Go2wNavTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavTeacherRunnerCfg",
    },
)

gym.register(
    id="Nav-Teacher-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2w_obstacle_env_cfg:Go2wNavTeacherEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavTeacherRunnerCfg",
    },
)

# =============================================================================
# RL navigation distillation (teacher privileged + obstacle, student LiDAR)
# =============================================================================

gym.register(
    id="Navigation-RL-Distill-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2w_obstacle_env_cfg:Go2wNavRLDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-RL-Distill-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2w_obstacle_env_cfg:Go2wNavRLDistillEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavRLDistillRunnerCfg",
    },
)

# =============================================================================
# RL navigation distillation (teacher privileged + obstacle, student depth camera)
# =============================================================================

gym.register(
    id="Navigation-Depth-Distill-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2w_obstacle_env_cfg:Go2wNavDepthRLDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Distill-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2w_obstacle_env_cfg:Go2wNavDepthRLDistillEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthRLDistillRunnerCfg",
    },
)

# =============================================================================
# Hospital play environment (test Nav Teacher in hospital-style corridors)
# =============================================================================

gym.register(
    id="Nav-Teacher-Hospital-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2w_obstacle_env_cfg:Go2wHospitalPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavTeacherRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Distill-Hospital-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2w_obstacle_env_cfg:Go2wHospitalDepthPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Distill-Hospital-Ward-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2w_obstacle_env_cfg:Go2wHospitalWardDepthPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Distill-Hospital-Floor-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2w_obstacle_env_cfg:Go2wHospitalFloorDepthPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthRLDistillRunnerCfg",
    },
)
