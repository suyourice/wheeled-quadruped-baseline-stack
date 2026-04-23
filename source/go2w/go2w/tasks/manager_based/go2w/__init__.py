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
# Teacher (PPO, proprioception + privileged observations)
# =============================================================================

gym.register(
    id="Teacher-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2w_teacher_env_cfg:Go2wTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:TeacherRunnerCfg",
    },
)

gym.register(
    id="Teacher-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2w_teacher_env_cfg:Go2wTeacherEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:TeacherRunnerCfg",
    },
)

# =============================================================================
# Distillation (student learns to mimic teacher)
# =============================================================================

gym.register(
    id="Distill-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2w_teacher_env_cfg:Go2wDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:DistillRunnerCfg",
    },
)

gym.register(
    id="Distill-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2w_teacher_env_cfg:Go2wDistillEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:DistillRunnerCfg",
    },
)

# =============================================================================
# Obstacle avoidance — Teacher (PPO, privileged obstacle positions)
# =============================================================================

gym.register(
    id="Obstacle-Teacher-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2w_obstacle_env_cfg:Go2wObstacleTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:ObstacleTeacherRunnerCfg",
    },
)

gym.register(
    id="Obstacle-Teacher-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2w_obstacle_env_cfg:Go2wObstacleTeacherEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:ObstacleTeacherRunnerCfg",
    },
)

# =============================================================================
# Obstacle avoidance — Distillation (student LiDAR, teacher privileged)
# =============================================================================

gym.register(
    id="Obstacle-Distill-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2w_obstacle_env_cfg:Go2wObstacleDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:ObstacleDistillRunnerCfg",
    },
)

gym.register(
    id="Obstacle-Distill-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2w_obstacle_env_cfg:Go2wObstacleDistillEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:ObstacleDistillRunnerCfg",
    },
)
