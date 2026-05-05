# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agent configs for obstacle avoidance Teacher and Student training."""

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class FrozenCommandResidualActorCfg(RslRlMLPModelCfg):
    """Frozen fast-flat actor with trainable obstacle-aware command residual."""

    class_name: str = "go2w.tasks.manager_based.go2w.residual_models:FrozenCommandResidualActor"

    frozen_hidden_dims: list[int] = MISSING
    frozen_activation: str = MISSING
    frozen_obs_normalization: bool = False

    command_obs_start: int = 9
    command_obs_dim: int = 3
    obstacle_obs_start: int = 60
    obstacle_obs_dim: int = 30
    state_obs_dim: int = 12
    obstacle_max_distance: float = 8.0

    residual_vy_scale: float = 0.9
    residual_yaw_scale: float = 1.1
    lateral_command_clip: float = 2.0
    yaw_command_clip: float = 2.0

    gate_forward_distance: float = 3.5
    gate_min_forward_distance: float = 0.2
    gate_path_width: float = 1.2


@configclass
class GeometricSteeringTeacherCfg(RslRlMLPModelCfg):
    """Rule-based steering teacher on top of a frozen fast-flat LLC."""

    class_name: str = "go2w.tasks.manager_based.go2w.residual_models:GeometricSteeringTeacher"

    frozen_hidden_dims: list[int] = MISSING
    frozen_activation: str = MISSING
    frozen_obs_normalization: bool = False

    command_obs_start: int = 9
    command_obs_dim: int = 3
    obstacle_obs_start: int = 60
    obstacle_obs_dim: int = 30
    obstacle_max_distance: float = 8.0

    min_command_speed: float = 0.15
    speed_reference: float = 2.0
    safe_distance: float = 2.8
    min_forward_distance: float = 0.15
    corridor_half_width: float = 0.75
    center_deadband: float = 0.05
    side_gain: float = 1.00
    brake_gain: float = 0.60
    yaw_gain: float = 1.10
    heading_align_gain: float = 1.10
    forward_bias_gain: float = 0.85
    local_avoid_distance: float = 1.18
    local_avoid_gain: float = 0.78
    local_brake_gain: float = 0.52
    sideways_body_bias_gain: float = 0.70
    sideways_heading_boost_gain: float = 0.75
    diagonal_sweep_gain: float = 1.12
    diagonal_blocked_gain: float = 0.68
    turning_sweep_clearance_gain: float = 0.36
    turning_speed_reduction_gain: float = 0.30
    turn_commitment_gain: float = 0.85
    turn_commitment_activation: float = 0.20
    narrow_gap_speed_reduction_gain: float = 0.65
    narrow_gap_forward_bias_gain: float = 0.45
    narrow_gap_heading_gain: float = 1.40
    obstacle_width: float = 0.30
    robot_forward_clearance: float = 0.42
    robot_side_clearance: float = 0.80
    gap_pair_forward_tolerance: float = 0.9
    blocked_detour_gain: float = 1.30
    blocked_brake_gain: float = 1.00
    blocked_yaw_gain: float = 1.70
    guidance_blend_gain: float = 0.80
    high_speed_safe_distance_gain: float = 0.22
    high_speed_local_distance_gain: float = 0.12
    high_speed_brake_gain: float = 0.22
    high_speed_yaw_gain: float = 0.18
    high_speed_turn_reduction_gain: float = 0.15
    high_speed_smoothing_reduction: float = 0.12
    max_delta_vx: float = 0.8
    max_delta_vy: float = 0.8
    max_delta_yaw: float = 1.5
    smoothing_alpha: float = 0.55
    lateral_command_clip: float = 2.0
    yaw_command_clip: float = 2.0


@configclass
class ObstacleTeacherRunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for training the obstacle-avoidance teacher.

    Uses privileged observations (obstacle positions relative to robot).
    """

    num_steps_per_env = 128
    max_iterations    = 3000
    save_interval     = 100
    experiment_name   = "go2w_obstacle_teacher"
    logger            = "wandb"
    wandb_project     = "go2w_obstacle_teacher"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.30,  # scale=28: 0.30×28×0.086≈0.72 m/s
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class ObstacleTeacherFastRunnerCfg(ObstacleTeacherRunnerCfg):
    """PPO config for obstacle teacher training from a pre-trained flat locomotion checkpoint.

    Shorter training budget (1500 vs 3000 iter) — no locomotion warmup phase needed.
    Network architecture [512,256,128] matches the flat fast checkpoint exactly.
    Uses bounded exploration: the simplified transfer run kept locomotion stable
    but collapsed action std to 0.10 and did not reduce obstacle contacts.
    """

    max_iterations  = 1500
    experiment_name = "go2w_obstacle_teacher_fast"
    wandb_project   = "go2w_obstacle_teacher_fast"

    policy = MISSING
    actor = FrozenCommandResidualActorCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.25,
        ),
        frozen_hidden_dims=[512, 256, 128],
        frozen_activation="elu",
        frozen_obs_normalization=False,
        command_obs_start=9,
        command_obs_dim=3,
        obstacle_obs_start=60,
        obstacle_obs_dim=30,
        state_obs_dim=12,
        obstacle_max_distance=8.0,
        residual_vy_scale=0.9,
        residual_yaw_scale=1.1,
        lateral_command_clip=2.0,
        yaw_command_clip=2.0,
        gate_forward_distance=3.5,
        gate_min_forward_distance=0.2,
        gate_path_width=1.2,
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class ObstacleDistillRunnerCfg(RslRlDistillationRunnerCfg):
    """Distillation config for a LiDAR student with a rule-based privileged teacher.

    The teacher is no longer learned with PPO. It is a deterministic geometric
    steering layer that rewrites `(vx, vy, yaw)` using privileged obstacle
    positions and then delegates final action generation to the frozen fast-flat
    LLC. Student distillation still uses the same action-cloning pipeline.
    """

    num_steps_per_env = 128
    max_iterations    = 1000
    save_interval     = 100
    experiment_name   = "go2w_obstacle_distill"
    logger            = "wandb"
    wandb_project     = "go2w_obstacle_distill"

    load_run        = ".*"
    load_checkpoint = "model_.*.pt"

    student = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.8,
        ),
    )

    teacher = GeometricSteeringTeacherCfg(
        hidden_dims=[1],  # unused; config kept only for RSL-RL model-schema compatibility
        activation="identity",  # unused
        obs_normalization=False,  # unused
        distribution_cfg=None,  # deterministic teacher
        frozen_hidden_dims=[512, 256, 128],
        frozen_activation="elu",
        frozen_obs_normalization=False,
        command_obs_start=9,
        command_obs_dim=3,
        obstacle_obs_start=60,
        obstacle_obs_dim=30,
        obstacle_max_distance=8.0,
        min_command_speed=0.15,
        speed_reference=2.0,
        safe_distance=2.8,
        min_forward_distance=0.15,
        corridor_half_width=0.75,
        center_deadband=0.05,
        side_gain=1.00,
        brake_gain=0.60,
        yaw_gain=1.10,
        heading_align_gain=1.10,
        forward_bias_gain=0.85,
        local_avoid_distance=1.18,
        local_avoid_gain=0.78,
        local_brake_gain=0.52,
        sideways_body_bias_gain=0.70,
        sideways_heading_boost_gain=0.75,
        diagonal_sweep_gain=1.12,
        diagonal_blocked_gain=0.68,
        turning_sweep_clearance_gain=0.36,
        turning_speed_reduction_gain=0.30,
        turn_commitment_gain=0.85,
        turn_commitment_activation=0.20,
        narrow_gap_speed_reduction_gain=0.65,
        narrow_gap_forward_bias_gain=0.45,
        narrow_gap_heading_gain=1.40,
        obstacle_width=0.30,
        robot_forward_clearance=0.42,
        robot_side_clearance=0.80,
        gap_pair_forward_tolerance=0.9,
        blocked_detour_gain=1.30,
        blocked_brake_gain=1.00,
        blocked_yaw_gain=1.70,
        guidance_blend_gain=0.80,
        high_speed_safe_distance_gain=0.22,
        high_speed_local_distance_gain=0.12,
        high_speed_brake_gain=0.22,
        high_speed_yaw_gain=0.18,
        high_speed_turn_reduction_gain=0.15,
        high_speed_smoothing_reduction=0.12,
        max_delta_vx=0.8,
        max_delta_vy=0.8,
        max_delta_yaw=1.5,
        smoothing_alpha=0.55,
        lateral_command_clip=2.0,
        yaw_command_clip=2.0,
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1.0e-3,
        gradient_length=15,
        max_grad_norm=1.0,
        loss_type="mse",
    )
