# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register Go2w tasks under the unified task-ID grammar.
IDs start with Loco or Nav, followed by the training scene and policy.
Training scenes are Flat, ObstacleFlat, or HospitalMaze.
Optional LCorridor, Ward, or Hospital tokens select a play venue.
TrainLidar marks HospitalMaze teacher play with the training LiDAR layout.
Play marks rollout tasks; Eval-* marks TrainDist, Static, or Dynamic evaluation.
Every registration includes Go2w and ends in v0.
"""

import gymnasium as gym

from . import agents

# =============================================================================
# [1] LLC locomotion (Loco-*)
# =============================================================================

# Train the legacy locomotion baseline on a flat scene.
gym.register(
    id="Loco-Flat-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.locomotion.env:Go2wEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# Play the legacy locomotion baseline on its flat scene.
gym.register(
    id="Loco-Flat-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.locomotion.env:Go2wEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# Train the FastFlat LLC on a flat scene, producing frozen LLC model_1999.
gym.register(
    id="Loco-FastFlat-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.locomotion.env:Go2wFastFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FastFlatRunnerCfg",
    },
)

# Play frozen LLC model_1999 on its FastFlat scene.
gym.register(
    id="Loco-FastFlat-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.locomotion.env:Go2wFastFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FastFlatRunnerCfg",
    },
)

# =============================================================================
# [2] ObstacleFlat navigation: train and play
# =============================================================================

# Train the legacy privileged teacher in the ObstacleFlat scene.
gym.register(
    id="Nav-ObstacleFlat-Teacher-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wNavTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavTeacherRunnerCfg",
    },
)

# Play the legacy privileged teacher in the ObstacleFlat scene.
gym.register(
    id="Nav-ObstacleFlat-Teacher-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wNavTeacherEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavTeacherRunnerCfg",
    },
)

# Distill the legacy LiDAR student from a privileged teacher in ObstacleFlat.
gym.register(
    id="Nav-ObstacleFlat-Distill-Lidar-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wNavRLDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavRLDistillRunnerCfg",
    },
)

# Play the legacy LiDAR student in the ObstacleFlat scene.
gym.register(
    id="Nav-ObstacleFlat-Distill-Lidar-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wNavRLDistillEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavRLDistillRunnerCfg",
    },
)

# Distill the legacy baseline depth student in the ObstacleFlat scene.
gym.register(
    id="Nav-ObstacleFlat-Distill-Depth-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wNavDepthRLDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthRLDistillRunnerCfg",
    },
)

# Play the legacy baseline depth student in the ObstacleFlat scene.
gym.register(
    id="Nav-ObstacleFlat-Distill-Depth-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wNavDepthRLDistillEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthRLDistillRunnerCfg",
    },
)

# Distill the legacy LongHist depth student in the ObstacleFlat scene.
gym.register(
    id="Nav-ObstacleFlat-Distill-Depth-LongHist-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wNavDepthLongHistRLDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthLongHistRLDistillRunnerCfg",
    },
)

# Play the legacy LongHist depth student in the ObstacleFlat scene.
gym.register(
    id="Nav-ObstacleFlat-Distill-Depth-LongHist-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wNavDepthLongHistRLDistillEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthLongHistRLDistillRunnerCfg",
    },
)

# Distill the legacy Sparse depth student in the ObstacleFlat scene.
gym.register(
    id="Nav-ObstacleFlat-Distill-Depth-Sparse-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wNavDepthSparseRLDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthSparseRLDistillRunnerCfg",
    },
)

# Play the legacy Sparse depth student in the ObstacleFlat scene.
gym.register(
    id="Nav-ObstacleFlat-Distill-Depth-Sparse-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wNavDepthSparseRLDistillEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthSparseRLDistillRunnerCfg",
    },
)

# Distill the legacy 4Cam depth student in the ObstacleFlat scene.
gym.register(
    id="Nav-ObstacleFlat-Distill-Depth-4Cam-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wNavDepthMultiCamRLDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthMultiCamRLDistillRunnerCfg",
    },
)

# Play the legacy 4Cam depth student in the ObstacleFlat scene.
gym.register(
    id="Nav-ObstacleFlat-Distill-Depth-4Cam-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wNavDepthMultiCamRLDistillEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthMultiCamRLDistillRunnerCfg",
    },
)

# =============================================================================
# [3] ObstacleFlat policies played in hospital venues
# =============================================================================

# Play the ObstacleFlat teacher in a hospital L-corridor venue.
gym.register(
    id="Nav-ObstacleFlat-Teacher-LCorridor-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavTeacherRunnerCfg",
    },
)

# Play the ObstacleFlat teacher in the full hospital venue.
gym.register(
    id="Nav-ObstacleFlat-Teacher-Hospital-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalFloorPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavTeacherRunnerCfg",
    },
)

# Play the ObstacleFlat baseline depth student in a hospital L-corridor venue.
gym.register(
    id="Nav-ObstacleFlat-Distill-Depth-LCorridor-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalDepthPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthRLDistillRunnerCfg",
    },
)

# Play the ObstacleFlat baseline depth student in a hospital ward venue.
gym.register(
    id="Nav-ObstacleFlat-Distill-Depth-Ward-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalWardDepthPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthRLDistillRunnerCfg",
    },
)

# Play the ObstacleFlat depth policy or student model_599 in the full hospital venue.
gym.register(
    id="Nav-ObstacleFlat-Distill-Depth-Hospital-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalFloorDepthPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthRLDistillRunnerCfg",
    },
)

# Play LongHist student model_599 in the full hospital venue.
gym.register(
    id="Nav-ObstacleFlat-Distill-Depth-LongHist-Hospital-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalFloorLongHistDepthPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthLongHistRLDistillRunnerCfg",
    },
)

# Play Sparse student model_599 in the full hospital venue.
gym.register(
    id="Nav-ObstacleFlat-Distill-Depth-Sparse-Hospital-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalFloorSparseDepthPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthSparseRLDistillRunnerCfg",
    },
)

# Play 4Cam student model_599 in the full hospital venue.
gym.register(
    id="Nav-ObstacleFlat-Distill-Depth-4Cam-Hospital-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalFloorMultiCamDepthPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthMultiCamRLDistillRunnerCfg",
    },
)

# =============================================================================
# [4] HospitalMaze teacher: train and play, including TrainLidar variants
# =============================================================================

# Train the HospitalMaze teacher with frozen LLC model_1999, producing teacher model_1100.
gym.register(
    id="Nav-HospitalMaze-Teacher-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wHospitalTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavHospitalTeacherRunnerCfg",
    },
)

# Play teacher model_1100 in its HospitalMaze training scene.
gym.register(
    id="Nav-HospitalMaze-Teacher-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalTeacherPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavHospitalTeacherRunnerCfg",
    },
)

# Play the HospitalMaze teacher with the legacy play LiDAR in an L-corridor venue.
gym.register(
    id="Nav-HospitalMaze-Teacher-LCorridor-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalTeacherLCorridorPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavHospitalTeacherRunnerCfg",
    },
)

# Play the HospitalMaze teacher with the legacy play LiDAR in a ward venue.
gym.register(
    id="Nav-HospitalMaze-Teacher-Ward-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalTeacherWardPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavHospitalTeacherRunnerCfg",
    },
)

# Play the HospitalMaze teacher with the legacy play LiDAR in the full hospital venue.
gym.register(
    id="Nav-HospitalMaze-Teacher-Hospital-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalTeacherFloorPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavHospitalTeacherRunnerCfg",
    },
)

# Play teacher model_1100 with TrainLidar in an L-corridor venue.
gym.register(
    id="Nav-HospitalMaze-Teacher-TrainLidar-LCorridor-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalTeacherLCorridorPlayEnvCfgV1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavHospitalTeacherRunnerCfg",
    },
)

# Play teacher model_1100 with TrainLidar in a ward venue.
gym.register(
    id="Nav-HospitalMaze-Teacher-TrainLidar-Ward-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalTeacherWardPlayEnvCfgV1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavHospitalTeacherRunnerCfg",
    },
)

# Play teacher model_1100 with TrainLidar in the full hospital venue.
gym.register(
    id="Nav-HospitalMaze-Teacher-TrainLidar-Hospital-Go2w-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalTeacherFloorPlayEnvCfgV1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavHospitalTeacherRunnerCfg",
    },
)

# =============================================================================
# [5] HospitalMaze depth distillation
# =============================================================================

# Distill the baseline depth student from teacher model_1100 in HospitalMaze, producing model_599.
gym.register(
    id="Nav-HospitalMaze-Distill-Depth-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wHospitalDepthRLDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalRLDistillRunnerCfg",
    },
)

# Distill the LongHist depth student from teacher model_1100 in HospitalMaze, producing model_599.
gym.register(
    id="Nav-HospitalMaze-Distill-Depth-LongHist-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wHospitalDepthLongHistRLDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalLongHistRLDistillRunnerCfg",
    },
)

# Distill the Sparse depth student from teacher model_1100 in HospitalMaze, producing model_599.
gym.register(
    id="Nav-HospitalMaze-Distill-Depth-Sparse-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wHospitalDepthSparseRLDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalSparseRLDistillRunnerCfg",
    },
)

# Distill the 4Cam depth student from teacher model_1100 in HospitalMaze, producing model_599.
gym.register(
    id="Nav-HospitalMaze-Distill-Depth-4Cam-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.navigation.env:Go2wHospitalDepthMultiCamRLDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalMultiCamRLDistillRunnerCfg",
    },
)

# =============================================================================
# [6] Evaluation suite (Eval-*)
# =============================================================================

# Evaluate teacher model_1100 on the static HospitalMaze suite.
gym.register(
    id="Nav-HospitalMaze-Teacher-Eval-Static-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeStaticEvalTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavHospitalTeacherRunnerCfg",
    },
)

# Evaluate teacher model_1100 on the dynamic HospitalMaze suite.
gym.register(
    id="Nav-HospitalMaze-Teacher-Eval-Dynamic-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeDynamicEvalTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavHospitalTeacherRunnerCfg",
    },
)

# Evaluate baseline student model_599 on the static HospitalMaze suite.
gym.register(
    id="Nav-HospitalMaze-Distill-Depth-Eval-Static-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeStaticEvalBaselineEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalRLDistillRunnerCfg",
    },
)

# Evaluate baseline student model_599 on the dynamic HospitalMaze suite.
gym.register(
    id="Nav-HospitalMaze-Distill-Depth-Eval-Dynamic-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeDynamicEvalBaselineEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalRLDistillRunnerCfg",
    },
)

# Evaluate LongHist student model_599 on the static HospitalMaze suite.
gym.register(
    id="Nav-HospitalMaze-Distill-Depth-LongHist-Eval-Static-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeStaticEvalLongHistEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalLongHistRLDistillRunnerCfg",
    },
)

# Evaluate LongHist student model_599 on the dynamic HospitalMaze suite.
gym.register(
    id="Nav-HospitalMaze-Distill-Depth-LongHist-Eval-Dynamic-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeDynamicEvalLongHistEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalLongHistRLDistillRunnerCfg",
    },
)

# Evaluate Sparse student model_599 on the static HospitalMaze suite.
gym.register(
    id="Nav-HospitalMaze-Distill-Depth-Sparse-Eval-Static-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeStaticEvalSparseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalSparseRLDistillRunnerCfg",
    },
)

# Evaluate Sparse student model_599 on the dynamic HospitalMaze suite.
gym.register(
    id="Nav-HospitalMaze-Distill-Depth-Sparse-Eval-Dynamic-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeDynamicEvalSparseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalSparseRLDistillRunnerCfg",
    },
)

# Evaluate 4Cam student model_599 on the static HospitalMaze suite.
gym.register(
    id="Nav-HospitalMaze-Distill-Depth-4Cam-Eval-Static-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeStaticEvalMultiCamEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalMultiCamRLDistillRunnerCfg",
    },
)

# Evaluate 4Cam student model_599 on the dynamic HospitalMaze suite.
gym.register(
    id="Nav-HospitalMaze-Distill-Depth-4Cam-Eval-Dynamic-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeDynamicEvalMultiCamEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalMultiCamRLDistillRunnerCfg",
    },
)

# Training-distribution maze eval (16-slot scene, last curriculum phase, max 12 obstacles)

# Evaluate teacher model_1100 on the HospitalMaze training distribution.
gym.register(
    id="Nav-HospitalMaze-Teacher-Eval-TrainDist-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeTrainDistEvalTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavHospitalTeacherRunnerCfg",
    },
)

# Evaluate baseline student model_599 on the HospitalMaze training distribution.
gym.register(
    id="Nav-HospitalMaze-Distill-Depth-Eval-TrainDist-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeTrainDistEvalBaselineEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalRLDistillRunnerCfg",
    },
)

# Evaluate LongHist student model_599 on the HospitalMaze training distribution.
gym.register(
    id="Nav-HospitalMaze-Distill-Depth-LongHist-Eval-TrainDist-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeTrainDistEvalLongHistEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalLongHistRLDistillRunnerCfg",
    },
)

# Evaluate Sparse student model_599 on the HospitalMaze training distribution.
gym.register(
    id="Nav-HospitalMaze-Distill-Depth-Sparse-Eval-TrainDist-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeTrainDistEvalSparseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalSparseRLDistillRunnerCfg",
    },
)

# Evaluate 4Cam student model_599 on the HospitalMaze training distribution.
gym.register(
    id="Nav-HospitalMaze-Distill-Depth-4Cam-Eval-TrainDist-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.hospital.env:Go2wHospitalMazeTrainDistEvalMultiCamEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_obstacle_cfg:NavDepthHospitalMultiCamRLDistillRunnerCfg",
    },
)
