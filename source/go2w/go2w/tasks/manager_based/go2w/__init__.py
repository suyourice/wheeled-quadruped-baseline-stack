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
        "env_cfg_entry_point": f"{__name__}.cfg.locomotion.env:Go2wEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Flat-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.locomotion.env:Go2wEnvCfg_PLAY",
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
        "env_cfg_entry_point": f"{__name__}.cfg.locomotion.env:Go2wFastFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FastFlatRunnerCfg",
    },
)

gym.register(
    id="Fast-Flat-2m-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.locomotion.env:Go2wFastFlatEnvCfg_PLAY",
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
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wNavTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavTeacherRunnerCfg",
    },
)

gym.register(
    id="Nav-Teacher-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wNavTeacherEnvCfg_PLAY",
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
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wNavRLDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-RL-Distill-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wNavRLDistillEnvCfg_PLAY",
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
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wNavDepthRLDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Distill-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wNavDepthRLDistillEnvCfg_PLAY",
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
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavTeacherRunnerCfg",
    },
)

gym.register(
    id="Nav-Teacher-Hospital-Floor-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalFloorPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavTeacherRunnerCfg",
    },
)

gym.register(
    id="Nav-Hospital-Teacher-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wHospitalTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavHospitalTeacherRunnerCfg",
    },
)

gym.register(
    id="Nav-Hospital-Teacher-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalTeacherPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavHospitalTeacherRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Distill-Hospital-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalDepthPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Distill-Hospital-Ward-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalWardDepthPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Distill-Hospital-Floor-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalFloorDepthPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthRLDistillRunnerCfg",
    },
)

# =============================================================================
# Hospital Teacher structured corridor play environments (v0)
# =============================================================================

gym.register(
    id="Nav-Hospital-Teacher-LCorridor-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalTeacherLCorridorPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavHospitalTeacherRunnerCfg",
    },
)

gym.register(
    id="Nav-Hospital-Teacher-Ward-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalTeacherWardPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavHospitalTeacherRunnerCfg",
    },
)

gym.register(
    id="Nav-Hospital-Teacher-Floor-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalTeacherFloorPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavHospitalTeacherRunnerCfg",
    },
)

# Hospital Teacher structured corridor play (v1 training lidar).
gym.register(
    id="Nav-Hospital-Teacher-LCorridor-Go2w-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalTeacherLCorridorPlayEnvCfgV1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavHospitalTeacherRunnerCfg",
    },
)

gym.register(
    id="Nav-Hospital-Teacher-Ward-Go2w-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalTeacherWardPlayEnvCfgV1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavHospitalTeacherRunnerCfg",
    },
)

gym.register(
    id="Nav-Hospital-Teacher-Floor-Go2w-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalTeacherFloorPlayEnvCfgV1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavHospitalTeacherRunnerCfg",
    },
)

# =============================================================================
# Depth student ablation experiments
# =============================================================================

gym.register(
    id="Navigation-Depth-Distill-LongHist-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wNavDepthLongHistRLDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthLongHistRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Distill-LongHist-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wNavDepthLongHistRLDistillEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthLongHistRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Distill-Sparse-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wNavDepthSparseRLDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthSparseRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Distill-Sparse-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wNavDepthSparseRLDistillEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthSparseRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Distill-4Cam-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wNavDepthMultiCamRLDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthMultiCamRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Distill-4Cam-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wNavDepthMultiCamRLDistillEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthMultiCamRLDistillRunnerCfg",
    },
)

# =============================================================================
# Depth student ablation hospital floor play environments
# =============================================================================

gym.register(
    id="Navigation-Depth-Distill-Hospital-Floor-LongHist-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalFloorLongHistDepthPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthLongHistRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Distill-Hospital-Floor-Sparse-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalFloorSparseDepthPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthSparseRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Distill-Hospital-Floor-4Cam-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalFloorMultiCamDepthPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthMultiCamRLDistillRunnerCfg",
    },
)

# =============================================================================
# Hospital Maze Depth Distillation
# =============================================================================

gym.register(
    id="Navigation-Depth-Hospital-Distill-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wHospitalDepthRLDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Hospital-Distill-LongHist-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wHospitalDepthLongHistRLDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalLongHistRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Hospital-Distill-Sparse-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wHospitalDepthSparseRLDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalSparseRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Hospital-Distill-4Cam-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wHospitalDepthMultiCamRLDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalMultiCamRLDistillRunnerCfg",
    },
)

# =============================================================================
# Hospital Maze Eval (20 obstacles, 220 s, static + dynamic, 5 policies)
# =============================================================================

gym.register(
    id="Nav-Hospital-Maze-Eval-Teacher-Static-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeStaticEvalTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavHospitalTeacherRunnerCfg",
    },
)

gym.register(
    id="Nav-Hospital-Maze-Eval-Teacher-Dynamic-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeDynamicEvalTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavHospitalTeacherRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Hospital-Maze-Eval-Static-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeStaticEvalBaselineEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Hospital-Maze-Eval-Dynamic-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeDynamicEvalBaselineEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Hospital-Maze-Eval-LongHist-Static-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeStaticEvalLongHistEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalLongHistRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Hospital-Maze-Eval-LongHist-Dynamic-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeDynamicEvalLongHistEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalLongHistRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Hospital-Maze-Eval-Sparse-Static-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeStaticEvalSparseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalSparseRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Hospital-Maze-Eval-Sparse-Dynamic-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeDynamicEvalSparseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalSparseRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Hospital-Maze-Eval-4Cam-Static-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeStaticEvalMultiCamEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalMultiCamRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Hospital-Maze-Eval-4Cam-Dynamic-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeDynamicEvalMultiCamEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalMultiCamRLDistillRunnerCfg",
    },
)

# Training-distribution maze eval (16-slot scene, last curriculum phase, max 12 obstacles)

gym.register(
    id="Nav-Hospital-Maze-Eval-Teacher-TrainDist-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeTrainDistEvalTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavHospitalTeacherRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Hospital-Maze-Eval-TrainDist-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeTrainDistEvalBaselineEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Hospital-Maze-Eval-LongHist-TrainDist-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeTrainDistEvalLongHistEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalLongHistRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Hospital-Maze-Eval-Sparse-TrainDist-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeTrainDistEvalSparseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalSparseRLDistillRunnerCfg",
    },
)

gym.register(
    id="Navigation-Depth-Hospital-Maze-Eval-4Cam-TrainDist-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeTrainDistEvalMultiCamEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalMultiCamRLDistillRunnerCfg",
    },
)
