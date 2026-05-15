# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Active RSL-RL configs for the current local-navigation distillation path."""

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)

from go2w.tasks.manager_based.go2w.observation_layout import (
    GOAL_COMMAND_DIM,
    GOAL_COMMAND_START,
    PRIVILEGED_OBSTACLE_START,
)


@configclass
class GeometricSteeringTeacherCfg(RslRlMLPModelCfg):
    """Rule-based privileged teacher that outputs obstacle-aware short-horizon targets."""

    class_name: str = "go2w.tasks.manager_based.go2w.residual_models:GeometricSteeringTeacher"

    frozen_hidden_dims: list[int] = MISSING
    frozen_activation: str = MISSING
    frozen_obs_normalization: bool = False

    command_obs_start: int = GOAL_COMMAND_START
    command_obs_dim: int = GOAL_COMMAND_DIM
    obstacle_obs_start: int = PRIVILEGED_OBSTACLE_START
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
    obstacle_present_risk_floor: float = 0.30
    target_pose_horizon: float = 0.75
    target_pose_yaw_horizon: float = 0.6
    target_pose_x_clip: float = 1.5
    target_pose_y_clip: float = 1.5
    target_pose_yaw_clip: float = 1.2


@configclass
class NavigationCommandStudentCfg(RslRlMLPModelCfg):
    """Student that predicts a local target pose and executes it through the LLC."""

    class_name: str = "go2w.tasks.manager_based.go2w.navigation_models:NavigationCommandStudent"

    frozen_hidden_dims: list[int] = MISSING
    frozen_activation: str = MISSING
    frozen_obs_normalization: bool = False

    command_obs_start: int = GOAL_COMMAND_START
    command_obs_dim: int = GOAL_COMMAND_DIM
    representation_dim: int = 8
    target_pose_horizon: float = 0.75
    target_pose_yaw_horizon: float = 0.6
    target_pose_x_clip: float = 1.5
    target_pose_y_clip: float = 1.5
    target_pose_yaw_clip: float = 1.2
    target_pose_to_vx_gain: float = 1.0
    target_pose_to_vy_gain: float = 1.0
    target_pose_to_yaw_gain: float = 1.0
    side_guidance_lateral_gain: float = 0.25
    side_guidance_yaw_gain: float = 0.45
    command_clip_xy: float = 2.0
    command_clip_yaw: float = 2.0


@configclass
class NavigationCommandDistillationAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    """Distill short-horizon local target poses instead of low-level actions."""

    class_name: str = "go2w.tasks.manager_based.go2w.distillation_algorithms:NavigationCommandDistillation"

    command_loss_weight: float = 1.0
    target_pose_loss_weight: float = 1.0
    representation_loss_weight: float = 0.5
    base_anchor_weight: float = 0.1
    yaw_loss_weight: float = 0.1
    delta_norm_loss_weight: float = 0.15
    delta_norm_margin: float = 0.10
    near_waypoint_command_discount: float = 0.5
    near_waypoint_anchor_bonus: float = 0.25
    blocked_lateral_yaw_weight: float = 1.0
    blocked_vx_weight: float = 1.0
    hard_case_delta_norm_threshold: float = 0.25
    blocked_risk_threshold: float = 0.06
    near_waypoint_distance_threshold: float = 0.9
    near_waypoint_heading_threshold: float = 0.7
    near_goal_distance_threshold: float = 0.9
    side_target_vy_scale: float = 0.8
    side_target_yaw_scale: float = 1.5
    side_target_deadband: float = 0.05
    debug_obstacle_print_interval: int = 10
    debug_obstacle_print_count: int = 6
    debug_rollout_print_interval: int = 128


@configclass
class NavigationDistillRunnerCfg(RslRlDistillationRunnerCfg):
    """Current active local-navigation distillation runner."""

    num_steps_per_env = 128
    max_iterations = 400
    save_interval = 50
    experiment_name = "go2w_navigation_distill"
    logger = "wandb"
    wandb_project = "go2w_navigation_distill"

    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    student = NavigationCommandStudentCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.35),
        frozen_hidden_dims=[512, 256, 128],
        frozen_activation="elu",
        frozen_obs_normalization=False,
        command_obs_start=GOAL_COMMAND_START,
        command_obs_dim=GOAL_COMMAND_DIM,
        representation_dim=8,
        target_pose_horizon=0.75,
        target_pose_yaw_horizon=0.6,
        target_pose_x_clip=1.5,
        target_pose_y_clip=1.5,
        target_pose_yaw_clip=1.2,
        target_pose_to_vx_gain=1.0,
        target_pose_to_vy_gain=1.0,
        target_pose_to_yaw_gain=1.0,
        side_guidance_lateral_gain=0.25,
        side_guidance_yaw_gain=0.45,
        command_clip_xy=2.0,
        command_clip_yaw=2.0,
    )

    teacher = GeometricSteeringTeacherCfg(
        hidden_dims=[1],
        activation="identity",
        obs_normalization=False,
        distribution_cfg=None,
        frozen_hidden_dims=[512, 256, 128],
        frozen_activation="elu",
        frozen_obs_normalization=False,
        command_obs_start=GOAL_COMMAND_START,
        command_obs_dim=GOAL_COMMAND_DIM,
        obstacle_obs_start=PRIVILEGED_OBSTACLE_START,
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
        obstacle_present_risk_floor=0.30,
        target_pose_horizon=0.75,
        target_pose_yaw_horizon=0.6,
        target_pose_x_clip=1.5,
        target_pose_y_clip=1.5,
        target_pose_yaw_clip=1.2,
    )

    algorithm = NavigationCommandDistillationAlgorithmCfg(
        num_learning_epochs=4,
        learning_rate=5.0e-4,
        gradient_length=8,
        max_grad_norm=1.0,
        loss_type="mse",
        command_loss_weight=1.0,
        target_pose_loss_weight=1.0,
        representation_loss_weight=0.5,
        base_anchor_weight=0.1,
        yaw_loss_weight=0.1,
        delta_norm_loss_weight=0.15,
        delta_norm_margin=0.10,
        near_waypoint_command_discount=0.5,
        near_waypoint_anchor_bonus=0.25,
        blocked_lateral_yaw_weight=1.0,
        blocked_vx_weight=1.0,
        hard_case_delta_norm_threshold=0.25,
        blocked_risk_threshold=0.06,
        near_waypoint_distance_threshold=0.9,
        near_waypoint_heading_threshold=0.7,
        near_goal_distance_threshold=0.9,
        side_target_vy_scale=0.8,
        side_target_yaw_scale=1.5,
        side_target_deadband=0.05,
        debug_obstacle_print_interval=10,
        debug_obstacle_print_count=6,
        debug_rollout_print_interval=128,
    )
