# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions for the Go2-W locomotion task.

Functions here supplement the standard rewards from isaaclab.envs.mdp.
They are adapted from the Dodo locomotion project (IsaacLab_Dodo).
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

from .obstacle_geometry import (
    DEFAULT_OBSTACLE_EFFECTIVE_RADIUS,
    footprint_clearance,
    obstacle_active_mask,
    obstacle_risk_radius,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ensure_navigation_goal_buffers(env: ManagerBasedRLEnv) -> None:
    """Create goal-navigation buffers on demand for reward/termination helpers."""
    if not hasattr(env, "_go2w_goal_pos_w"):
        env._go2w_goal_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._go2w_goal_heading_w = torch.zeros(env.num_envs, device=env.device)
        env._go2w_start_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._go2w_start_heading_w = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_go2w_scenario_template_id"):
        env._go2w_scenario_template_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_go2w_initial_scenario_template_id"):
        env._go2w_initial_scenario_template_id = env._go2w_scenario_template_id.clone()
    # Passable narrow-gap metadata (mirrors events._ensure_navigation_goal_buffers).
    if not hasattr(env, "_go2w_gap_center_w"):
        env._go2w_gap_center_w = torch.zeros(env.num_envs, 2, device=env.device)
        env._go2w_gap_dir_w = torch.zeros(env.num_envs, 2, device=env.device)
        env._go2w_gap_half_width = torch.zeros(env.num_envs, device=env.device)
        env._go2w_gap_passable = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_go2w_gap_free_half_width"):
        env._go2w_gap_free_half_width = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_go2w_gap_center_tolerance"):
        env._go2w_gap_center_tolerance = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_go2w_stuck_counter"):
        env._go2w_stuck_counter = torch.zeros(env.num_envs, device=env.device)


def _goal_command_from_buffers(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return goal distance and heading error from the sampled task buffers."""
    _ensure_navigation_goal_buffers(env)
    asset = env.scene[asset_cfg.name]
    goal_vec_w = env._go2w_goal_pos_w - asset.data.root_pos_w[:, :3]
    goal_vec_b = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), goal_vec_w)
    goal_distance = torch.norm(goal_vec_b[:, :2], dim=1)
    goal_heading_error = wrap_to_pi(env._go2w_goal_heading_w - asset.data.heading_w).abs()
    return goal_distance, goal_heading_error


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track commanded linear velocity in the yaw-aligned frame.

    Unlike the body-frame version (track_lin_vel_xy_exp), this projects
    velocity onto the **horizontal plane** using only the yaw component
    of the robot orientation. This avoids measurement distortion when
    the robot is tilted in roll or pitch.

        r = exp( -||v_cmd_xy - v_yaw_xy||^2 / std^2 )

    Args:
        std: Gaussian kernel width (smaller = stricter tracking).
        command_name: Name of the velocity command term.
        asset_cfg: Robot scene entity.
    """
    asset = env.scene[asset_cfg.name]
    # Project world-frame velocity into yaw-only frame (removes roll/pitch)
    vel_yaw = quat_apply_inverse(
        yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track commanded yaw rate in the world frame.

    Uses world-frame angular velocity z-component directly, which is
    always aligned with the gravity vector regardless of robot tilt.

        r = exp( -(w_cmd_z - w_world_z)^2 / std^2 )

    Args:
        std: Gaussian kernel width (smaller = stricter tracking).
        command_name: Name of the velocity command term.
        asset_cfg: Robot scene entity.
    """
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


def wheel_vel_zero_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise wheel angular velocity only when the velocity command is exactly zero.

    Has no effect during locomotion so wheels can spin freely.

        penalty = sum(w_wheel_i^2)  if cmd == 0, else 0
    """
    command = env.command_manager.get_command(command_name)
    is_zero_cmd = (command[:, :3].abs().sum(dim=1) == 0.0).float()

    asset = env.scene[asset_cfg.name]
    wheel_vel_sq = torch.sum(
        torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1
    )
    return is_zero_cmd * wheel_vel_sq


def wheel_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalise each wheel that loses ground contact.

    Returns the count of wheels not in contact. A wheeled robot should
    keep all four wheels on the ground at all times.

    Args:
        sensor_cfg: ContactSensor pointing at wheel (foot) bodies.
        threshold: Minimum contact force [N] to consider a wheel grounded.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
    )
    no_contact = (forces < threshold).float()
    return no_contact.sum(dim=1)


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from target base height."""
    height = env.scene[robot_cfg.name].data.root_pos_w[:, 2]
    return (height - target_height).pow(2)


def _curriculum_scale(
    env: ManagerBasedRLEnv,
    start_steps: int = 0,
    warmup_steps: int = 0,
    start_iteration: int | None = None,
    warmup_iterations: int | None = None,
    steps_per_iteration: int = 128,
) -> float:
    """Return a scalar in [0, 1] for step- or iteration-based reward curricula."""
    if start_iteration is not None:
        start_steps = start_iteration * steps_per_iteration
    if warmup_iterations is not None:
        warmup_steps = warmup_iterations * steps_per_iteration
    if env.common_step_counter < start_steps:
        return 0.0
    if warmup_steps <= 0:
        return 1.0
    return max(min((env.common_step_counter - start_steps) / warmup_steps, 1.0), 0.0)


def joint_deviation_l1_command_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_abs_lin_x: float = 0.0,
    max_abs_lin_x: float | None = None,
    max_abs_lin_y: float | None = None,
    max_abs_ang_z: float | None = None,
) -> torch.Tensor:
    """Joint deviation penalty active only for selected command regions."""
    command = env.command_manager.get_command(command_name)
    mask = torch.ones(command.shape[0], device=command.device, dtype=command.dtype)
    if min_abs_lin_x > 0.0:
        mask = mask * (command[:, 0].abs() >= min_abs_lin_x).to(command.dtype)
    if max_abs_lin_x is not None:
        mask = mask * (command[:, 0].abs() <= max_abs_lin_x).to(command.dtype)
    if max_abs_lin_y is not None:
        mask = mask * (command[:, 1].abs() <= max_abs_lin_y).to(command.dtype)
    if max_abs_ang_z is not None:
        mask = mask * (command[:, 2].abs() <= max_abs_ang_z).to(command.dtype)

    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return mask * torch.sum(torch.abs(joint_pos - default_joint_pos), dim=1)


def goal_progress_dense(
    env: ManagerBasedRLEnv,
    clip: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense reward for velocity component toward the local navigation goal.

    Returns a value in [-1, 1]: +1 when moving toward the goal at speed >= clip,
    -1 when moving directly away. Zero when stationary or perpendicular.
    """
    _ensure_navigation_goal_buffers(env)
    asset = env.scene[asset_cfg.name]
    goal_vec_w = env._go2w_goal_pos_w[:, :2] - asset.data.root_pos_w[:, :2]
    goal_dist = goal_vec_w.norm(dim=-1).clamp(min=0.01)
    goal_dir_w = goal_vec_w / goal_dist.unsqueeze(-1)
    vel_w = asset.data.root_lin_vel_w[:, :2]
    progress = (vel_w * goal_dir_w).sum(dim=-1)
    result = progress.clamp(-clip, clip) / clip

    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["goal_progress_activation_mean"] = result.mean()
    env.extras["log"]["goal_progress_positive_rate"] = (progress > 0.0).float().mean()
    hlc_action = env.action_manager.get_term("llc_cmd").processed_actions
    env.extras["log"]["mean_action_vx"] = hlc_action[:, 0].mean()
    env.extras["log"]["mean_action_speed_norm"] = hlc_action[:, :2].norm(dim=-1).mean()
    return result


def obstacle_nav_ttc_penalty(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    safe_ttc: float = 1.5,
    command_name: str = "base_velocity",
    obstacle_radius: float = 0.22,
    robot_half_width: float = 0.30,
    safety_margin: float = 0.05,
    robot_front_margin: float = 0.20,
    lookahead_distance: float = 2.2,
    sum_clip: float = 1.5,
    min_command_speed: float = 0.05,
    passable_gap_relief: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Soft corridor TTC penalty for the navigation teacher.

    The HLC command defines the intended path in the robot yaw frame. Obstacles
    near the edge of that path receive a small penalty, while obstacles deeply
    inside the swept corridor receive a strong penalty. This keeps narrow-gap
    entries possible while still discouraging direct base collisions.
    ``obstacle_radius`` is a compatibility fallback only; configured navigation
    environments use per-slot physical footprint metadata.

        corridor_half_width = robot_half_width + obstacle_risk_radius + safety_margin
        lateral_risk = smoothstep(clamp((corridor_half_width - lateral) / corridor_half_width))
        ttc = (forward - obstacle_risk_radius - robot_front_margin) / command_speed
        penalty = sum(lateral_risk * clamp((safe_ttc - ttc) / safe_ttc, 0, 1))
    """
    asset = env.scene[asset_cfg.name]
    command_xy = env.command_manager.get_command(command_name)[:, :2]      # (N, 2), yaw frame
    command_speed = command_xy.norm(dim=-1).clamp(min=0.0)                 # (N,)
    command_dir = command_xy / command_speed.clamp(min=min_command_speed).unsqueeze(-1)
    moving = command_speed > min_command_speed
    robot_pos_w = asset.data.root_pos_w[:, :2]                             # (N, 2)

    # Cached (N, K, 3) world-frame obstacle positions, sliced to xy.
    obs_pos_all = _obstacle_positions_w(env, obstacle_names)[..., :2]      # (N, K, 2)
    rel_w = obs_pos_all - robot_pos_w.unsqueeze(1)                         # (N, K, 2)

    heading = asset.data.heading_w
    cos_h = torch.cos(heading).unsqueeze(-1)
    sin_h = torch.sin(heading).unsqueeze(-1)
    rel_b_x = cos_h * rel_w[..., 0] + sin_h * rel_w[..., 1]
    rel_b_y = -sin_h * rel_w[..., 0] + cos_h * rel_w[..., 1]
    rel_b = torch.stack((rel_b_x, rel_b_y), dim=-1)                        # (N, K, 2)

    forward = (rel_b * command_dir.unsqueeze(1)).sum(dim=-1)               # (N, K)
    lateral = torch.abs(
        command_dir[:, 0:1] * rel_b[..., 1] - command_dir[:, 1:2] * rel_b[..., 0]
    )                                                                       # (N, K)

    center_distance = rel_w.norm(dim=-1)
    active_obstacles = obstacle_active_mask(env, obstacle_names, center_distance, lookahead_distance + 10.0)
    radii = obstacle_risk_radius(env, obstacle_names, center_distance, fallback_radius=obstacle_radius)
    corridor_half_width = robot_half_width + radii + safety_margin
    intrusion = torch.minimum((corridor_half_width - lateral).clamp(min=0.0), corridor_half_width)
    lateral_alpha = intrusion / corridor_half_width
    lateral_risk = lateral_alpha * lateral_alpha * (3.0 - 2.0 * lateral_alpha)

    forward_clearance = forward - radii - robot_front_margin
    ttc = forward_clearance / command_speed.clamp(min=min_command_speed).unsqueeze(-1)
    ttc_risk = ((safe_ttc - ttc) / safe_ttc).clamp(min=0.0, max=1.0)

    active = (
        moving.unsqueeze(-1)
        & (forward > 0.0)
        & (forward_clearance < lookahead_distance)
        & active_obstacles
    )
    penalty = lateral_risk * ttc_risk * active.to(lateral_risk.dtype)
    result = penalty.sum(dim=1).clamp(max=sum_clip)

    if passable_gap_relief > 0.0:
        relief = _passable_gap_relief(env, asset_cfg, passable_gap_relief)
        result = result * (1.0 - relief)
    return result


def obstacle_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    start_steps: int = 0,
    warmup_steps: int = 0,
    start_iteration: int | None = None,
    warmup_iterations: int | None = None,
    steps_per_iteration: int = 128,
) -> torch.Tensor:
    """Count obstacle bodies with contact, with an optional curriculum scale.

    Obstacles are floated above the floor and parked separately when inactive, so
    their net contact force reflects robot-to-obstacle contacts rather than
    floor or parked-obstacle contacts.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history
    if sensor_cfg.body_ids is not None:
        forces = forces[:, :, sensor_cfg.body_ids, :]
    max_forces = forces.norm(dim=-1).max(dim=1)[0]
    contacts = (max_forces > threshold).float().sum(dim=1)

    # A mid-episode obstacle pose write leaves pre-resample forces in sensor
    # history. Discard that finite history before attributing contact to the new layout.
    if not hasattr(env, "_go2w_ignore_obstacle_contact_history_steps"):
        env._go2w_ignore_obstacle_contact_history_steps = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )
    if hasattr(env, "_go2w_obstacle_pose_changed_midstep"):
        changed = env._go2w_obstacle_pose_changed_midstep
        if changed.any():
            history_length = max(int(contact_sensor.cfg.history_length), 1)
            env._go2w_ignore_obstacle_contact_history_steps[changed] = history_length
            env._go2w_obstacle_pose_changed_midstep[changed] = False
    ignore_history = env._go2w_ignore_obstacle_contact_history_steps > 0
    contacts = torch.where(ignore_history, torch.zeros_like(contacts), contacts)
    env._go2w_ignore_obstacle_contact_history_steps.sub_(1).clamp_(min=0)

    # Track contact since the current sampled scenario began for compatibility logs.
    if not hasattr(env, "_go2w_had_collision_episode"):
        env._go2w_had_collision_episode = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    reset_mask = env.episode_length_buf == 0
    env._go2w_had_collision_episode[reset_mask] = False
    env._go2w_had_collision_episode |= contacts > 0

    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["obstacle_contact_activation_rate"] = (contacts > 0).float().mean()
    env.extras["log"]["obstacle_contact_force_max_mean"] = max_forces.max(dim=1).values.mean()
    if hasattr(env, "_go2w_scenario_template_id"):
        for scenario_id, scenario_name in _NAV_SCENARIO_NAMES.items():
            scenario_mask = env._go2w_scenario_template_id == scenario_id
            if scenario_mask.any():
                env.extras["log"][f"contact_activation_rate/{scenario_name}"] = (
                    (contacts[scenario_mask] > 0).float().mean()
                )

    return _curriculum_scale(
        env, start_steps, warmup_steps, start_iteration, warmup_iterations, steps_per_iteration
    ) * contacts


def obstacle_contact_termination(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    start_steps: int = 0,
    start_iteration: int | None = None,
    steps_per_iteration: int = 128,
) -> torch.Tensor:
    """Terminate an episode when any obstacle contact force exceeds threshold.

    Obstacles are floated above the floor and parked separately when inactive, so
    the contact force reflects robot-to-obstacle contacts.
    """
    if start_iteration is not None:
        start_steps = start_iteration * steps_per_iteration
    if env.common_step_counter < start_steps:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history
    if sensor_cfg.body_ids is not None:
        forces = forces[:, :, sensor_cfg.body_ids, :]
    max_forces = forces.norm(dim=-1).max(dim=1)[0]
    return torch.any(max_forces > threshold, dim=1)


def navigation_path_final_goal_reached(
    env: ManagerBasedRLEnv,
    position_threshold: float = 0.70,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot reaches the final waypoint of a stored navigation path."""
    if not hasattr(env, "_go2w_navigation_path_w") or not hasattr(env, "_go2w_navigation_path_count"):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    robot = env.scene[asset_cfg.name]
    path_count = env._go2w_navigation_path_count.clamp(min=1)
    final_idx = path_count - 1
    row_idx = torch.arange(env.num_envs, device=env.device)
    final_xy = env._go2w_navigation_path_w[row_idx, final_idx, :2]
    final_distance = (robot.data.root_pos_w[:, :2] - final_xy).norm(dim=-1)

    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["structured_final_goal_distance_mean"] = final_distance.mean()
    env.extras["log"]["structured_final_goal_reached_rate"] = (final_distance <= position_threshold).float().mean()
    return final_distance <= position_threshold


def goal_heading_tanh_reward(
    env: ManagerBasedRLEnv,
    std: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward aligning the robot heading with the sampled goal heading."""
    _, heading_error = _goal_command_from_buffers(env, asset_cfg)
    return 1.0 - torch.tanh(heading_error / max(std, 1.0e-6))


def _resample_goal_positions_only(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    goal_forward_range: tuple[float, float],
    goal_lateral_range: tuple[float, float],
    goal_heading_jitter_range: tuple[float, float],
    min_goal_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Sample a new goal position for envs that just reached their goal.

    Obstacles are NOT moved — only the goal buffer is updated so the
    episode can continue toward a fresh target without a full reset.
    """
    _ensure_navigation_goal_buffers(env)
    robot = env.scene[asset_cfg.name]

    curr_pos_w = robot.data.root_pos_w[env_ids, :3].clone()
    yaw_list = robot.data.heading_w[env_ids].cpu().tolist()

    goal_pos_w = curr_pos_w.clone()
    goal_heading_w = robot.data.heading_w[env_ids].clone()

    for idx in range(len(env_ids)):
        yaw_i = yaw_list[idx]
        cos_yaw = math.cos(yaw_i)
        sin_yaw = math.sin(yaw_i)

        forward = goal_forward_range[0]
        lateral = 0.0
        for _ in range(50):
            forward = random.uniform(*goal_forward_range)
            lateral = random.uniform(*goal_lateral_range)
            if math.hypot(forward, lateral) >= min_goal_distance:
                break

        goal_dx = forward * cos_yaw - lateral * sin_yaw
        goal_dy = forward * sin_yaw + lateral * cos_yaw
        goal_pos_w[idx, 0] += goal_dx
        goal_pos_w[idx, 1] += goal_dy

        path_heading = math.atan2(goal_dy, goal_dx)
        jitter = random.uniform(*goal_heading_jitter_range)
        goal_heading_w[idx] = math.atan2(
            math.sin(path_heading + jitter),
            math.cos(path_heading + jitter),
        )

    env._go2w_goal_pos_w[env_ids] = goal_pos_w
    env._go2w_goal_heading_w[env_ids] = goal_heading_w


def goal_reached_and_resample(
    env: ManagerBasedRLEnv,
    position_threshold: float = 0.35,
    heading_threshold: float = 0.6,
    goal_forward_range: tuple[float, float] = (2.5, 4.5),
    goal_lateral_range: tuple[float, float] = (-1.5, 1.5),
    goal_heading_jitter_range: tuple[float, float] = (-0.35, 0.35),
    min_goal_distance: float = 2.0,
    near_goal_threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Success bonus that immediately samples a new navigation segment.

    When the robot enters the goal zone, it receives a +1 reward (scaled by weight
    in the config). If a navigation resampler is installed, both the goal and
    obstacle layout are replaced relative to the robot's current pose.
    """
    if not hasattr(env, "_go2w_goals_reached_episode"):
        env._go2w_goals_reached_episode = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_go2w_first_goal_reached_episode"):
        env._go2w_first_goal_reached_episode = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
    if not hasattr(env, "_go2w_min_goal_distance_episode"):
        env._go2w_min_goal_distance_episode = torch.full(
            (env.num_envs,), float("inf"), device=env.device
        )

    goal_distance, heading_error = _goal_command_from_buffers(env, asset_cfg)
    path_direct_goal = (
        bool(getattr(env, "_go2w_navigation_path_direct_goal", False))
        and hasattr(env, "_go2w_navigation_path_w")
        and hasattr(env, "_go2w_navigation_path_count")
    )
    if path_direct_goal:
        robot = env.scene[asset_cfg.name]
        path_count = env._go2w_navigation_path_count.clamp(min=1)
        final_idx = path_count - 1
        row_idx = torch.arange(env.num_envs, device=env.device)
        final_xy = env._go2w_navigation_path_w[row_idx, final_idx, :2]
        goal_distance = (robot.data.root_pos_w[:, :2] - final_xy).norm(dim=-1)
        heading_error = torch.zeros_like(goal_distance)
    reset_mask = env.episode_length_buf == 0
    env._go2w_first_goal_reached_episode[reset_mask] = False
    env._go2w_min_goal_distance_episode[reset_mask] = goal_distance[reset_mask]
    env._go2w_min_goal_distance_episode = torch.minimum(
        env._go2w_min_goal_distance_episode, goal_distance
    )

    position_candidate = goal_distance <= position_threshold
    heading_candidate = heading_error <= heading_threshold
    if path_direct_goal:
        # Structured path following updates intermediate waypoints separately.
        # This reward/resample hook should only consider the final stored path goal,
        # otherwise reaching a local waypoint mutates path state during reward eval.
        reached = position_candidate
    else:
        reached = position_candidate & heading_candidate
    first_reached = reached & (env._go2w_goals_reached_episode <= 0.0)
    env._go2w_first_goal_reached_episode |= first_reached

    env_ids = reached.nonzero(as_tuple=False).squeeze(-1)
    if len(env_ids) > 0:
        if hasattr(env, "_nav_resample_on_goal"):
            # Full resample: new goal + new obstacle layout around the robot's
            # current position.  Save/restore the per-env goal counter so the
            # reset function's zero-out doesn't erase in-episode progress.
            saved = env._go2w_goals_reached_episode[env_ids].clone()
            env._nav_resample_on_goal(env_ids)
            env._go2w_goals_reached_episode[env_ids] = saved + 1.0
        else:
            _resample_goal_positions_only(
                env,
                env_ids,
                goal_forward_range,
                goal_lateral_range,
                goal_heading_jitter_range,
                min_goal_distance,
                asset_cfg,
            )
            env._go2w_goals_reached_episode[env_ids] += 1.0
        if hasattr(env, "_go2w_had_collision_episode"):
            # The next sampled layout starts a new collision-accounting segment.
            env._go2w_had_collision_episode[env_ids] = False

    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["goals_per_episode"] = env._go2w_goals_reached_episode.mean()
    env.extras["log"]["mean_goal_distance"] = goal_distance.mean()
    env.extras["log"]["min_goal_distance_per_episode"] = env._go2w_min_goal_distance_episode.mean()
    env.extras["log"]["near_goal_but_not_reached_rate"] = (
        (goal_distance <= near_goal_threshold) & ~reached
    ).float().mean()
    env.extras["log"]["goal_reached_candidate_rate"] = position_candidate.float().mean()
    env.extras["log"]["goal_heading_blocked_at_candidate_rate"] = (
        position_candidate & ~heading_candidate
    ).float().mean()

    # Attribute total goal throughput to the episode-start template. Successful
    # envs are retargeted to random_fallback for the next segment, so using the
    # current template here would leave only zero-goal envs under head_on/gap/etc.
    # Log first-goal success separately for debugging the initial segment.
    has_scenario = hasattr(env, "_go2w_scenario_template_id")
    has_collision = hasattr(env, "_go2w_had_collision_episode")
    if has_scenario:
        initial_template_ids = env._go2w_initial_scenario_template_id
        current_template_ids = env._go2w_scenario_template_id
        for tid, tname in _NAV_SCENARIO_NAMES.items():
            initial_mask = initial_template_ids == tid
            if initial_mask.any():
                env.extras["log"][f"goals_per_ep/{tname}"] = (
                    env._go2w_goals_reached_episode[initial_mask].mean()
                )
                env.extras["log"][f"first_goal_success_per_ep/{tname}"] = (
                    env._go2w_first_goal_reached_episode[initial_mask].float().mean()
                )
            current_mask = current_template_ids == tid
            if current_mask.any():
                if has_collision:
                    env.extras["log"][f"collision_per_ep/{tname}"] = (
                        env._go2w_had_collision_episode[current_mask].float().mean()
                    )
    return reached.float()


# =============================================================================
# Navigation-specific reward functions (Phase 1: static obstacle teacher)
# =============================================================================

# Scenario id → name mapping shared by goal-reached and collision logging.
_NAV_SCENARIO_NAMES: dict[int, str] = {
    0: "empty", 1: "head_on", 2: "left_edge", 3: "right_edge",
    4: "diag_left", 5: "diag_right", 6: "off_left", 7: "off_right",
    8: "narrow_gap", 9: "random_fallback",
    10: "partial_blockage_left_open", 11: "partial_blockage_right_open",
    12: "cluttered", 13: "narrow_gap_wide", 14: "narrow_gap_barely",
}


def nav_clearance_penalty(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    min_safe_dist: float = 0.8,
    robot_safety_radius: float = 0.30,
    passable_gap_relief: float = 0.0,
    max_logged_clearance: float = 8.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise proximity to any obstacle with a smooth gradient.

    Returns a value in [0, 1]:
      0   when nearest safety-inflated clearance >= min_safe_dist
      1   when the safety envelopes overlap (clearance <= 0)

    Intended use: weight should be negative (e.g. −1.5) so this acts as a
    penalty. Complements obstacle_ttc which is command-direction-specific;
    this term penalises closeness regardless of movement direction.

    When passable_gap_relief > 0, the penalty is softened (up to that fraction)
    while the robot is aligned and threading a passable narrow gap, so the policy
    is not afraid of the close side obstacles of a corridor it can pass.
    """
    if len(obstacle_names) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene[asset_cfg.name]
    robot_pos = robot.data.root_pos_w[:, :2]

    obs_pos = _obstacle_positions_w(env, obstacle_names)[..., :2]  # (N, K, 2)
    center_dists = (obs_pos - robot_pos.unsqueeze(1)).norm(dim=-1)  # (N, K)
    active = obstacle_active_mask(env, obstacle_names, center_dists, min_safe_dist + 100.0)
    clearances = footprint_clearance(env, obstacle_names, center_dists, robot_safety_radius)
    nearest_clearance = torch.where(
        active, clearances, torch.full_like(clearances, max_logged_clearance)
    ).min(dim=1).values
    nearest_dist = nearest_clearance.clamp(min=0.0)

    # Smooth proximity penalty: 0 when safe, saturates at 1 when very close
    intrusion = (min_safe_dist - nearest_dist).clamp(min=0.0, max=min_safe_dist)
    penalty = (intrusion / min_safe_dist) ** 2  # quadratic near contact, zero far away

    if passable_gap_relief > 0.0:
        relief = _passable_gap_relief(env, asset_cfg, passable_gap_relief)
        penalty = penalty * (1.0 - relief)
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["footprint_clearance_mean"] = nearest_clearance.clamp(
        min=0.0, max=max_logged_clearance
    ).mean()
    if hasattr(env, "_go2w_obstacle_effective_radius"):
        radii = env._go2w_obstacle_effective_radius
        mask = env._go2w_obstacle_active_mask.float()
        env.extras["log"]["obstacle_effective_radius_mean"] = (radii * mask).sum() / mask.sum().clamp(min=1.0)
        # Active layouts remain local to the robot; parked physical assets are about 1000 m away.
        physical_active = center_dists < 100.0
        env.extras["log"]["active_obstacle_count_mean"] = mask.sum(dim=1).mean()
        env.extras["log"]["obstacle_active_pose_mismatch_rate"] = (
            physical_active != env._go2w_obstacle_active_mask
        ).float().mean()
    return penalty


def _nav_step_cache(env: ManagerBasedRLEnv) -> dict:
    """Return a per-step memo dict, cleared when the global step counter advances.

    All reward terms in a navigation step are evaluated against identical
    robot/obstacle poses, so the obstacle-geometry helpers (frontal geometry,
    goal-path blockage, passable-gap geometry) can safely share their results
    within the same step instead of recomputing the same GPU kernels several
    times. The cache is keyed by env.common_step_counter so it self-invalidates
    on the next step.
    """
    step = env.common_step_counter
    if getattr(env, "_nav_reward_cache_step", None) != step:
        env._nav_reward_cache = {}
        env._nav_reward_cache_step = step
    return env._nav_reward_cache


def _obstacle_positions_w(env: ManagerBasedRLEnv, obstacle_names: list[str]) -> torch.Tensor:
    """Cached (N, K, 3) world-frame obstacle positions for the current step.

    Gathering K scene entities and stacking them is a Python-side loop that runs
    once per obstacle reward term; memoising it removes most of the per-step
    scene-access overhead. All navigation reward terms use the same obstacle
    list, so the per-step length-keyed cache never collides.
    """
    cache = _nav_step_cache(env)
    cache_key = ("obstacle_pos_w", len(obstacle_names))
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    pos = torch.stack(
        [env.scene[n].data.root_pos_w[:, :3] for n in obstacle_names], dim=1
    )  # (N, K, 3)
    cache[cache_key] = pos
    return pos


def _compute_nav_frontal_geometry(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    robot_cfg: SceneEntityCfg,
    frontal_half_angle_deg: float,
    max_distance: float,
    robot_safety_radius: float = 0.30,
    reference_slot_count: int = 15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared obstacle geometry for frontal-blockage reward functions.

    Returns (frontal_blockage, left_blockage, right_blockage, vel_yaw,
             closeness, angles, active) — all (N, ...) on env.device.
    N = num_envs, K = num_obstacles in obstacle_names.

    Result is memoised per step (shared by lateral-escape, impossible-gap, and
    dense-recovery terms which all call this with the same arguments).
    """
    cache = _nav_step_cache(env)
    cache_key = (
        "frontal_geom",
        frontal_half_angle_deg,
        max_distance,
        robot_safety_radius,
        reference_slot_count,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    robot = env.scene[robot_cfg.name]
    robot_yaw_quat = yaw_quat(robot.data.root_quat_w)

    obs_pos_all = _obstacle_positions_w(env, obstacle_names)  # (N, K, 3)
    rel_w = obs_pos_all - robot.data.root_pos_w[:, :3].unsqueeze(1)
    N, K = rel_w.shape[:2]

    quat_exp = robot_yaw_quat.unsqueeze(1).expand(-1, K, -1).reshape(N * K, 4)
    rel_xy = quat_apply_inverse(quat_exp, rel_w.reshape(N * K, 3)).reshape(N, K, 3)[:, :, :2]
    center_dists = rel_xy.norm(dim=-1)
    angles = torch.atan2(rel_xy[..., 1], rel_xy[..., 0])

    active = obstacle_active_mask(env, obstacle_names, center_dists, max_distance)
    risk_radii = obstacle_risk_radius(env, obstacle_names, center_dists)
    nominal_radius = DEFAULT_OBSTACLE_EFFECTIVE_RADIUS + float(
        getattr(env, "_go2w_obstacle_radius_margin", 0.0)
    )
    radius_delta = risk_radii - nominal_radius
    # Recovery gates should keep the baseline response for familiar boxes; only
    # the footprint difference from that nominal box changes blockage strength.
    blockage_dists = (center_dists - radius_delta).clamp(min=0.0, max=max_distance)
    closeness = (1.0 - blockage_dists / max_distance) * active.float()

    frontal_rad = math.radians(frontal_half_angle_deg)
    blockage_scale = 1.0 / max(reference_slot_count, 1)
    frontal_blockage = (
        (closeness * ((angles.abs() < frontal_rad) & active).float()).sum(dim=1) * blockage_scale
    ).clamp(max=1.0)
    left_blockage = (
        (closeness * ((angles > frontal_rad) & (angles <= math.pi) & active).float()).sum(dim=1)
        * blockage_scale
    ).clamp(max=1.0)
    right_blockage = (
        (closeness * ((angles < -frontal_rad) & (angles >= -math.pi) & active).float()).sum(dim=1)
        * blockage_scale
    ).clamp(max=1.0)

    vel_yaw = quat_apply_inverse(robot_yaw_quat, robot.data.root_lin_vel_w[:, :3])
    result = (frontal_blockage, left_blockage, right_blockage, vel_yaw, closeness, angles, active)
    cache[cache_key] = result
    return result


def _compute_goal_path_blockage(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    robot_cfg: SceneEntityCfg,
    corridor_half_width: float = 0.7,
    max_distance: float = 8.0,
    robot_safety_radius: float = 0.30,
) -> torch.Tensor:
    """Measure how much obstacles intrude into the robot→goal straight-line corridor.

    Projects each obstacle onto the axis from the robot to its current goal and
    measures lateral deviation from that axis.  Returns a value in [0, 1] per
    environment — 0 when the corridor is clear, approaching 1 when blocked.

    Result is memoised per step (shared by lateral-escape, open-path, and
    dense-recovery terms which all call this with the same arguments).
    """
    if len(obstacle_names) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    cache = _nav_step_cache(env)
    cache_key = ("goal_path_blockage", corridor_half_width, max_distance, robot_safety_radius)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    _ensure_navigation_goal_buffers(env)
    robot = env.scene[robot_cfg.name]
    robot_yaw_quat = yaw_quat(robot.data.root_quat_w)

    goal_vec_w = env._go2w_goal_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    goal_dist = goal_vec_w.norm(dim=-1).clamp(min=0.01)
    goal_vec_w_3d = torch.cat(
        [goal_vec_w, torch.zeros(env.num_envs, 1, device=env.device)], dim=-1
    )
    goal_vec_b = quat_apply_inverse(robot_yaw_quat, goal_vec_w_3d)[:, :2]
    goal_dir_b = goal_vec_b / goal_dist.unsqueeze(-1)  # (N, 2) unit vector toward goal

    obs_pos_all = _obstacle_positions_w(env, obstacle_names)  # (N, K, 3)
    rel_w = obs_pos_all - robot.data.root_pos_w[:, :3].unsqueeze(1)
    N, K = rel_w.shape[:2]

    quat_exp = robot_yaw_quat.unsqueeze(1).expand(-1, K, -1).reshape(N * K, 4)
    rel_b = quat_apply_inverse(quat_exp, rel_w.reshape(N * K, 3)).reshape(N, K, 3)[:, :, :2]

    goal_dir_exp = goal_dir_b.unsqueeze(1).expand(-1, K, -1)
    forward_goal = (rel_b * goal_dir_exp).sum(dim=-1)
    lateral_goal = (
        goal_dir_b[:, 0:1] * rel_b[..., 1] - goal_dir_b[:, 1:2] * rel_b[..., 0]
    ).abs()
    center_dists = rel_b.norm(dim=-1)
    radii = obstacle_risk_radius(env, obstacle_names, center_dists)
    active_slots = obstacle_active_mask(env, obstacle_names, center_dists, max_distance)

    # The original corridor width was tuned around the standard 0.30 m box.
    # Expand or contract it only by the footprint difference from that nominal
    # obstacle so familiar boxes preserve the original blockage calibration.
    nominal_radius = DEFAULT_OBSTACLE_EFFECTIVE_RADIUS + float(
        getattr(env, "_go2w_obstacle_radius_margin", 0.0)
    )
    radius_delta = radii - nominal_radius
    corridor_extent = (corridor_half_width + radius_delta).clamp(min=1.0e-3)
    active = (
        (forward_goal > -radius_delta)
        & (forward_goal - radius_delta < goal_dist.unsqueeze(-1) + 0.3)
        & (lateral_goal < corridor_extent)
        & active_slots
    )
    closeness = 1.0 - (center_dists / max_distance).clamp(0.0, 1.0)
    intrusion = (
        (corridor_extent - lateral_goal).clamp(min=0.0) / corridor_extent.clamp(min=1.0e-6)
    ).clamp(max=1.0)

    # Use the strongest obstacle intrusion instead of averaging over all slots.
    # Averaging by K made a single obstacle on the direct path nearly invisible
    # when many play/training slots were parked far away.
    blockage = (closeness * intrusion * active.float()).max(dim=1).values
    result = blockage.clamp(0.0, 1.0)
    cache[cache_key] = result
    return result


def _compute_passable_gap_geometry(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    half_width_margin: float = 0.15,
    approach_max_forward: float = 3.5,
    approach_back_tol: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Relate the robot to the stored passable-gap centerline.

    Reads the per-env gap buffers populated at reset (only set for the
    narrow_gap / narrow_gap_wide / narrow_gap_barely scenarios). For every
    other scenario the passable flag is False so callers see an inactive gap.

    Returns (passable, approaching, forward_to_gap, lateral_err, align,
             forward_vel, speed) — all (N,) on env.device.
      passable     : bool, scenario is a passable narrow gap
      approaching  : bool, gap is ahead/just-passed and within range
      forward_to_gap: signed distance to gap center along the gap direction (>0 ahead)
      lateral_err  : absolute lateral offset from the gap centerline [m]
      align        : 1 on the centerline -> 0 at the gap edge
      forward_vel  : world velocity projected onto the gap direction [m/s]
      speed        : planar speed [m/s]

    Result is memoised per step (shared by the passable-gap reward and the
    clearance/TTC/grazing relief which all call this with the same arguments).
    """
    if not hasattr(env, "_go2w_gap_passable"):
        z = torch.zeros(env.num_envs, device=env.device)
        b = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        return b, b, z, z, z, z, z

    cache = _nav_step_cache(env)
    cache_key = (
        "passable_gap_geom",
        half_width_margin,
        approach_max_forward,
        approach_back_tol,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    robot = env.scene[robot_cfg.name]
    robot_pos = robot.data.root_pos_w[:, :2]
    gap_center = env._go2w_gap_center_w
    gap_dir = env._go2w_gap_dir_w
    half_w = env._go2w_gap_half_width
    passable = env._go2w_gap_passable

    to_gap = gap_center - robot_pos                                    # (N, 2)
    forward_to_gap = (to_gap * gap_dir).sum(dim=-1)                    # (N,)
    perp = torch.stack([-gap_dir[:, 1], gap_dir[:, 0]], dim=-1)        # left normal
    lateral_err = (to_gap * perp).sum(dim=-1).abs()                    # (N,)
    align = 1.0 - (lateral_err / (half_w + half_width_margin).clamp(min=1.0e-3)).clamp(0.0, 1.0)

    vel_w = robot.data.root_lin_vel_w[:, :2]
    forward_vel = (vel_w * gap_dir).sum(dim=-1)
    speed = vel_w.norm(dim=-1)

    approaching = (
        passable
        & (forward_to_gap > -approach_back_tol)
        & (forward_to_gap < approach_max_forward)
    )
    result = (passable, approaching, forward_to_gap, lateral_err, align, forward_vel, speed)
    cache[cache_key] = result
    return result


def _passable_gap_relief(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_relief: float = 0.0,
) -> torch.Tensor:
    """Return a per-env relief factor in [0, max_relief] for passable narrow gaps.

    Relief is highest when the robot is aligned with the gap centerline and is
    actively approaching/threading a passable gap. Used to soften clearance and
    TTC penalties only in passable corridors, never elsewhere. Collision penalty
    is intentionally left untouched by this helper.
    """
    if max_relief <= 0.0 or not hasattr(env, "_go2w_gap_passable"):
        return torch.zeros(env.num_envs, device=env.device)
    _, approaching, _, _, align, _, _ = _compute_passable_gap_geometry(env, robot_cfg)
    return max_relief * approaching.float() * align


def nav_frontal_blocked_lateral_escape_reward(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    frontal_half_angle_deg: float = 45.0,
    min_blockage_for_reward: float = 0.20,
    side_diff_deadband: float = 0.04,
    max_distance: float = 8.0,
    goal_path_min_blockage: float = 0.10,
    goal_path_corridor_half_width: float = 0.7,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward lateral escape velocity when both the frontal corridor and the direct
    goal path are blocked.

    Adds a goal_path_blockage gate on top of the original frontal-only gating so
    that lateral escape is suppressed when the straight robot→goal path is clear —
    preventing the policy from swerving sideways on open ground.

    Convention:
      - preferred_side_hint > 0 → go right (left has more obstacles)
      - preferred_side_hint < 0 → go left  (right has more obstacles)
      - reward = frontal_gate × goal_path_gate × clamp(0, aligned_lateral / 2.0)

    Returns a value in [0, 1].
    """
    if len(obstacle_names) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    frontal_blockage, left_blockage, right_blockage, vel_yaw, _, _, _ = (
        _compute_nav_frontal_geometry(env, obstacle_names, robot_cfg, frontal_half_angle_deg, max_distance)
    )

    goal_path_blockage = _compute_goal_path_blockage(
        env, obstacle_names, robot_cfg, goal_path_corridor_half_width, max_distance
    )
    goal_path_gate = (
        (goal_path_blockage - goal_path_min_blockage)
        / (1.0 - goal_path_min_blockage + 1.0e-6)
    ).clamp(0.0, 1.0)

    side_diff = left_blockage - right_blockage
    preferred_sign = torch.where(
        side_diff.abs() > side_diff_deadband,
        torch.sign(side_diff),
        torch.zeros_like(side_diff),
    )
    aligned_lateral = -preferred_sign * vel_yaw[:, 1]

    frontal_gate = (
        (frontal_blockage - min_blockage_for_reward)
        / (1.0 - min_blockage_for_reward + 1.0e-6)
    ).clamp(0.0, 1.0)

    reward = frontal_gate * goal_path_gate * (aligned_lateral / 2.0).clamp(0.0, 1.0)

    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["lateral_escape_activation_rate"] = (
        (frontal_gate * goal_path_gate > 0.05).float().mean()
    )
    env.extras["log"]["goal_path_blockage_mean"] = goal_path_blockage.mean()
    return reward


def nav_open_path_straightness_reward(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    goal_path_corridor_half_width: float = 0.7,
    open_blockage_threshold: float = 0.12,
    max_distance: float = 8.0,
    min_speed: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward straight-line progress along the goal path when it is unobstructed.

    Gated by goal_path_blockage: suppressed when obstacles are in the robot→goal
    corridor so the policy is free to deviate laterally during avoidance.

    Returns a value in [-1, 1]: positive when velocity is aligned with the goal
    direction and has low lateral component; negative when the robot is moving
    strongly sideways on an otherwise clear path.  Exactly zero when the path
    is blocked or the robot is stationary.
    """
    if len(obstacle_names) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    goal_path_blockage = _compute_goal_path_blockage(
        env, obstacle_names, robot_cfg, goal_path_corridor_half_width, max_distance
    )
    # Strongly suppress this shaping as soon as the direct path is blocked.
    x = ((open_blockage_threshold - goal_path_blockage) / max(open_blockage_threshold, 1.0e-6)).clamp(0.0, 1.0)
    open_gate = x * x * (3.0 - 2.0 * x)

    _ensure_navigation_goal_buffers(env)
    robot = env.scene[robot_cfg.name]
    robot_yaw_quat = yaw_quat(robot.data.root_quat_w)

    goal_vec_w = env._go2w_goal_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    goal_dist = goal_vec_w.norm(dim=-1).clamp(min=0.01)
    goal_vec_w_3d = torch.cat(
        [goal_vec_w, torch.zeros(env.num_envs, 1, device=env.device)], dim=-1
    )
    goal_dir_b = quat_apply_inverse(robot_yaw_quat, goal_vec_w_3d)[:, :2]
    goal_dir_b = goal_dir_b / goal_dist.unsqueeze(-1)

    vel_b = quat_apply_inverse(robot_yaw_quat, robot.data.root_lin_vel_w[:, :3])[:, :2]
    speed = vel_b.norm(dim=-1)
    moving = (speed > min_speed).float()

    vel_norm = vel_b / speed.clamp(min=min_speed).unsqueeze(-1)
    alignment = (vel_norm * goal_dir_b).sum(dim=-1)                                              # cos θ
    lateral_frac = (goal_dir_b[:, 0] * vel_norm[:, 1] - goal_dir_b[:, 1] * vel_norm[:, 0]).abs()  # |sin θ|

    score = (0.6 * alignment - 0.4 * lateral_frac).clamp(-1.0, 1.0)
    result = open_gate * moving * score

    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["open_path_straightness_mean"] = result.mean()
    env.extras["log"]["path_efficiency"] = (open_gate * moving * alignment.clamp(min=0.0)).mean()
    env.extras["log"]["stuck_rate"] = ((speed < 0.15) & (goal_dist > 1.0)).float().mean()
    return result


def nav_open_path_goal_heading_reward(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    goal_path_corridor_half_width: float = 0.7,
    open_blockage_threshold: float = 0.12,
    max_distance: float = 8.0,
    heading_std: float = 0.7,
    min_goal_distance: float = 0.7,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward facing the goal direction when the direct path is open.

    Goal progress alone can be satisfied by strafing toward the goal. This term
    nudges the high-level policy to point the body toward the goal on open paths
    while staying inactive when the corridor is blocked or near the final pose.
    """
    if len(obstacle_names) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    goal_path_blockage = _compute_goal_path_blockage(
        env, obstacle_names, robot_cfg, goal_path_corridor_half_width, max_distance
    )
    x = ((open_blockage_threshold - goal_path_blockage) / max(open_blockage_threshold, 1.0e-6)).clamp(0.0, 1.0)
    open_gate = x * x * (3.0 - 2.0 * x)

    _ensure_navigation_goal_buffers(env)
    robot = env.scene[robot_cfg.name]
    goal_vec_w = env._go2w_goal_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    goal_dist = goal_vec_w.norm(dim=-1)
    path_heading_w = torch.atan2(goal_vec_w[:, 1], goal_vec_w[:, 0])
    heading_error = wrap_to_pi(path_heading_w - robot.data.heading_w).abs()

    far_from_goal = (goal_dist > min_goal_distance).float()
    heading_score = 1.0 - torch.tanh(heading_error / max(heading_std, 1.0e-6))
    result = open_gate * far_from_goal * heading_score

    if "log" not in env.extras:
        env.extras["log"] = {}
    denom = open_gate.sum().clamp(min=1.0)
    env.extras["log"]["open_path_goal_heading_mean"] = result.mean()
    env.extras["log"]["open_path_heading_error_mean"] = (heading_error * open_gate).sum() / denom
    return result


def nav_near_goal_settling_reward(
    env: ManagerBasedRLEnv,
    settling_distance: float = 0.5,
    max_command_norm: float = 0.6,
    max_yaw_rate: float = 0.8,
    max_action_rate: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward calm behaviour near the goal.

    Uses the actual HLC action norm (policy output [vx, vy, yaw]), actual body
    yaw rate, and action-rate change to encourage the policy to reduce commands
    and oscillations once within settling_distance of the target.

    nav tasks zero out base_velocity commands, so command_manager is not used.
    """
    _ensure_navigation_goal_buffers(env)
    robot = env.scene[asset_cfg.name]

    goal_dist = (env._go2w_goal_pos_w[:, :2] - robot.data.root_pos_w[:, :2]).norm(dim=-1)
    near_goal_gate = (1.0 - (goal_dist / max(settling_distance, 1e-6)).clamp(0.0, 1.0))

    # HLC action is [vx, vy, yaw_rate] - the policy's actual output, not the zero nav command.
    hlc_cmd = env.action_manager.action
    command_norm = hlc_cmd.norm(dim=-1)

    yaw_rate = robot.data.root_ang_vel_w[:, 2].abs()
    try:
        action_rate = (env.action_manager.action - env.action_manager.prev_action).norm(dim=-1)
    except AttributeError:
        action_rate = torch.zeros(env.num_envs, device=env.device)

    command_quality = (1.0 - command_norm / max(max_command_norm, 1.0e-6)).clamp(0.0, 1.0)
    yaw_quality = (1.0 - yaw_rate / max(max_yaw_rate, 1.0e-6)).clamp(0.0, 1.0)
    action_rate_quality = (1.0 - action_rate / max(max_action_rate, 1.0e-6)).clamp(0.0, 1.0)
    settling_quality = command_quality * yaw_quality * action_rate_quality

    result = near_goal_gate * settling_quality

    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["near_goal_settling_activation_rate"] = (near_goal_gate > 0.1).float().mean()
    env.extras["log"]["near_goal_settling_mean"] = result.mean()
    return result


def nav_impossible_gap_penalty(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    frontal_half_angle_deg: float = 45.0,
    high_frontal_threshold: float = 0.40,
    side_blocked_threshold: float = 0.15,
    min_gap_available: float = 0.35,
    min_gap_width_norm: float = 0.45,
    gap_reference_width: float = 0.7,
    max_distance: float = 8.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Shape recovery only when the frontal gap is effectively impossible.

    With a negative config weight, positive return values penalize pushing
    forward. Negative return values become a small reward for backing off or
    turning away, but only under this impossible-gap gate.

    Returns roughly [-1, 1].  Use a negative weight in the reward config.
    """
    if len(obstacle_names) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    frontal_blockage, left_blockage, right_blockage, vel_yaw, closeness, angles, active = (
        _compute_nav_frontal_geometry(env, obstacle_names, robot_cfg, frontal_half_angle_deg, max_distance)
    )

    frontal_gate = (
        (frontal_blockage - high_frontal_threshold) / (1.0 - high_frontal_threshold + 1.0e-6)
    ).clamp(0.0, 1.0)

    frontal_rad = math.radians(frontal_half_angle_deg)
    frontal_mask = (angles.abs() < frontal_rad) & active
    dists = (1.0 - closeness).clamp(0.0, 1.0) * max_distance
    frontal_min = torch.where(frontal_mask, dists, torch.full_like(dists, max_distance)).min(dim=1).values
    gap_available = torch.sigmoid(
        (frontal_min - gap_reference_width) / max(gap_reference_width * 0.3, 1.0e-6)
    )
    gap_width_norm = (frontal_min / max(gap_reference_width * 2.0, 1.0e-6)).clamp(0.0, 1.0)
    gap_gate = (
        ((min_gap_available - gap_available) / max(min_gap_available, 1.0e-6)).clamp(0.0, 1.0)
        * ((min_gap_width_norm - gap_width_norm) / max(min_gap_width_norm, 1.0e-6)).clamp(0.0, 1.0)
    )

    # Both sides blocked → no lateral escape available → truly stuck.
    min_side = torch.minimum(left_blockage, right_blockage)
    side_gate = (
        (min_side - side_blocked_threshold) / (1.0 - side_blocked_threshold + 1.0e-6)
    ).clamp(0.0, 1.0)

    impossible_gap_gate = frontal_gate * side_gate * gap_gate
    positive_vx = vel_yaw[:, 0].clamp(0.0, 2.0) / 2.0
    small_negative_vx = (-vel_yaw[:, 0]).clamp(0.0, 0.5) / 0.5

    # Use HLC yaw command (policy output index 2) rather than vel_yaw[:, 2] which is
    # vertical linear velocity, not yaw rate.
    side_diff = left_blockage - right_blockage
    preferred_turn = -torch.sign(side_diff)
    hlc_yaw_cmd = env.action_manager.action[:, 2]
    turn_away = (preferred_turn * hlc_yaw_cmd).clamp(0.0, 1.5) / 1.5

    # Negative components become positive reward because the config weight is negative.
    result = impossible_gap_gate * (positive_vx - 0.35 * small_negative_vx - 0.25 * turn_away)

    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["impossible_gap_activation_rate"] = (impossible_gap_gate > 0.1).float().mean()
    env.extras["log"]["impossible_gap_backward_activation_rate"] = (
        (impossible_gap_gate > 0.1) & (small_negative_vx > 0.05)
    ).float().mean()
    env.extras["log"]["impossible_gap_turnaway_activation_rate"] = (
        (impossible_gap_gate > 0.1) & (turn_away > 0.05)
    ).float().mean()
    return result


def nav_passable_gap_traversal_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    forward_vel_ref: float = 1.0,
    align_weight: float = 0.5,
    progress_weight: float = 0.5,
    stop_speed: float = 0.1,
) -> torch.Tensor:
    """Encourage decisive traversal of passable narrow gaps.

    Active only for the narrow_gap / narrow_gap_wide / narrow_gap_barely scenarios
    (gated by the per-env _go2w_gap_passable flag set at reset) and only while the
    gap is ahead of, or just behind, the robot. It rewards:
      - alignment with the gap centerline (always, while approaching),
      - forward progress through the gap, scaled by alignment so the policy is
        only encouraged to push forward when it is actually lined up with the gap.

    The forward-progress term naturally rewards not stopping in front of a passable
    gap (a stationary robot earns only the alignment part). It returns roughly
    [0, 1] and is intended to be used with a moderate positive weight so it never
    forces unsafe squeezing.  It does NOT activate for impossible-gap/dead-end
    layouts because those scenarios never set the passable flag.
    """
    passable, approaching, _, lateral_err, align, forward_vel, speed = (
        _compute_passable_gap_geometry(env, robot_cfg)
    )
    active = approaching.float()
    progress = (forward_vel / max(forward_vel_ref, 1.0e-6)).clamp(0.0, 1.0)
    reward = active * (align_weight * align + progress_weight * align * progress)

    if "log" not in env.extras:
        env.extras["log"] = {}
    denom = active.sum().clamp(min=1.0)
    env.extras["log"]["passable_gap_activation_rate"] = active.mean()
    env.extras["log"]["passable_gap_alignment_error"] = (lateral_err * active).sum() / denom
    env.extras["log"]["passable_gap_progress_mean"] = (forward_vel.clamp(min=0.0) * active).sum() / denom
    env.extras["log"]["passable_gap_stop_rate"] = (
        ((speed < stop_speed).float()) * active
    ).sum() / denom
    env.extras["log"]["frozen_despite_passable_rate"] = (
        (speed < stop_speed).float() * active
    ).sum() / denom
    env.extras["log"]["mean_vx_when_passable"] = (forward_vel * active).sum() / denom
    if hasattr(env, "_go2w_scenario_template_id"):
        sid = env._go2w_scenario_template_id
        candidate = (sid == 8) | (sid == 13) | (sid == 14)
        candidate_count = candidate.float().sum().clamp(min=1.0)
        rejected = candidate & ~passable
        env.extras["log"]["passable_gap_geometry_rejection_rate"] = rejected.float().sum() / candidate_count
    return reward


def nav_dense_recovery_reward(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frontal_half_angle_deg: float = 45.0,
    max_distance: float = 8.0,
    frontal_block_threshold: float = 0.25,
    goal_path_block_threshold: float = 0.20,
    goal_path_corridor_half_width: float = 0.7,
    generic_block_threshold: float = 0.45,
    min_goal_dist: float = 1.0,
    side_blocked_threshold: float = 0.15,
    side_diff_deadband: float = 0.04,
    vel_ref: float = 1.0,
    stop_speed: float = 0.12,
    stuck_steps_for_penalty: int = 10,
    stop_penalty: float = 0.5,
) -> torch.Tensor:
    """Encourage productive recovery (not stopping) in cluttered/blocked layouts.

    Active only when the goal is still far AND the path is genuinely blocked AND
    the scenario is a blocked one (cluttered / partial-blockage) or the direct
    goal corridor is strongly blocked. It rewards:
      - lateral velocity toward the more open side,
      - small backward motion when both sides are blocked,
      - turn-away yaw command when both sides are blocked,
    and penalises sustained stopping (tracked via a per-env stuck counter) while
    the goal is unreached and the path is blocked.

    Returns roughly [-stop_penalty, 1]; use a positive weight. It never activates
    on empty/open-path layouts, so it is not a global movement reward.
    """
    if len(obstacle_names) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    _ensure_navigation_goal_buffers(env)

    frontal_blockage, left_blockage, right_blockage, vel_yaw, _, _, _ = (
        _compute_nav_frontal_geometry(env, obstacle_names, robot_cfg, frontal_half_angle_deg, max_distance)
    )
    goal_path_blockage = _compute_goal_path_blockage(
        env, obstacle_names, robot_cfg, goal_path_corridor_half_width, max_distance
    )

    robot = env.scene[robot_cfg.name]
    goal_dist = (env._go2w_goal_pos_w[:, :2] - robot.data.root_pos_w[:, :2]).norm(dim=-1)
    speed = robot.data.root_lin_vel_w[:, :2].norm(dim=-1)

    fr_gate = (
        (frontal_blockage - frontal_block_threshold) / (1.0 - frontal_block_threshold + 1.0e-6)
    ).clamp(0.0, 1.0)
    gp_gate = (
        (goal_path_blockage - goal_path_block_threshold) / (1.0 - goal_path_block_threshold + 1.0e-6)
    ).clamp(0.0, 1.0)
    blocked_gate = torch.maximum(fr_gate, gp_gate)
    goal_far = (goal_dist > min_goal_dist).float()

    # Scenario gate: blocked/cluttered templates, or any strongly blocked corridor.
    sid = env._go2w_scenario_template_id
    in_blocked_scenario = ((sid == 10) | (sid == 11) | (sid == 12)).float()
    generic_blocked = (goal_path_blockage > generic_block_threshold).float()
    scenario_gate = torch.maximum(in_blocked_scenario, generic_blocked)
    recovery_active = blocked_gate * goal_far * scenario_gate

    # Productive recovery directions.
    side_diff = left_blockage - right_blockage
    preferred_sign = torch.where(
        side_diff.abs() > side_diff_deadband,
        torch.sign(side_diff),
        torch.zeros_like(side_diff),
    )
    aligned_lateral = -preferred_sign * vel_yaw[:, 1]
    lateral_score = (aligned_lateral / max(vel_ref, 1.0e-6)).clamp(0.0, 1.0)

    both_sides_blocked = (torch.minimum(left_blockage, right_blockage) > side_blocked_threshold).float()
    backward_score = both_sides_blocked * (-vel_yaw[:, 0] / max(vel_ref, 1.0e-6)).clamp(0.0, 1.0)

    preferred_turn = -preferred_sign
    hlc_yaw_cmd = env.action_manager.action[:, 2]
    turn_score = both_sides_blocked * (preferred_turn * hlc_yaw_cmd / 1.5).clamp(0.0, 1.0)

    move_score = (lateral_score + 0.5 * backward_score + 0.3 * turn_score).clamp(0.0, 1.0)

    # Per-env stuck counter: increment while slow and far from goal, reset otherwise
    # and on episode start. Penalise only sustained (not momentary) stopping.
    if not hasattr(env, "_go2w_stuck_counter"):
        env._go2w_stuck_counter = torch.zeros(env.num_envs, device=env.device)
    stuck_now = (speed < stop_speed) & (goal_dist > min_goal_dist)
    env._go2w_stuck_counter = torch.where(
        stuck_now, env._go2w_stuck_counter + 1.0, torch.zeros_like(env._go2w_stuck_counter)
    )
    env._go2w_stuck_counter[env.episode_length_buf == 0] = 0.0
    sustained_stuck = (env._go2w_stuck_counter >= stuck_steps_for_penalty).float()

    result = recovery_active * (move_score - stop_penalty * sustained_stuck)

    if "log" not in env.extras:
        env.extras["log"] = {}
    active_mask = (recovery_active > 0.05).float()
    denom = active_mask.sum().clamp(min=1.0)
    env.extras["log"]["dense_recovery_activation_rate"] = active_mask.mean()
    env.extras["log"]["dense_recovery_move_mean"] = (move_score * active_mask).sum() / denom
    env.extras["log"]["dense_recovery_stuck_rate"] = sustained_stuck.mean()
    env.extras["log"]["dense_recovery_generic_blocked_rate"] = generic_blocked.mean()
    env.extras["log"]["dense_recovery_blocked_scenario_rate"] = in_blocked_scenario.mean()
    cluttered_mask = (sid == 12).float()
    cdenom = cluttered_mask.sum().clamp(min=1.0)
    env.extras["log"]["cluttered_stuck_rate"] = (sustained_stuck * cluttered_mask).sum() / cdenom
    return result


def nav_grazing_penalty(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    graze_distance: float = 0.65,
    contact_distance: float = 0.50,
    robot_safety_radius: float = 0.30,
    passable_gap_relief: float = 0.0,
    max_distance: float = 8.0,
) -> torch.Tensor:
    """Mild near-contact (grazing) penalty separate from the collision penalty.

    Penalises being very close to the nearest safety-inflated obstacle footprint
    in the [contact_distance, graze_distance] clearance band without requiring a full
    contact event. It is intentionally weak so it nudges the policy to leave a
    slightly larger margin and reduce leg/wheel scraping without making it timid.
    In passable narrow gaps the penalty is relieved in proportion to centerline
    alignment, so threading a passable corridor is not discouraged. The full
    collision penalty (separate term) remains stronger and unchanged.

    Returns a value in [0, 1]; use a small negative weight.
    """
    if len(obstacle_names) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene[robot_cfg.name]
    robot_pos = robot.data.root_pos_w[:, :2]
    obs_pos = _obstacle_positions_w(env, obstacle_names)[..., :2]  # (N, K, 2)
    center_dists = (obs_pos - robot_pos.unsqueeze(1)).norm(dim=-1)  # (N, K)
    active = obstacle_active_mask(env, obstacle_names, center_dists, max_distance)
    clearances = footprint_clearance(env, obstacle_names, center_dists, robot_safety_radius)
    nearest = torch.where(active, clearances, torch.full_like(clearances, max_distance)).min(dim=1).values

    band = max(graze_distance - contact_distance, 1.0e-6)
    graze = ((graze_distance - nearest) / band).clamp(0.0, 1.0)

    relief = _passable_gap_relief(env, robot_cfg, passable_gap_relief)
    penalty = graze * (1.0 - relief)

    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["grazing_penalty_mean"] = penalty.mean()
    env.extras["log"]["near_contact_activation_rate"] = (graze > 0.05).float().mean()
    env.extras["log"]["min_footprint_clearance_mean"] = nearest.clamp(max=max_distance).mean()
    nearest_center = torch.where(active, center_dists, torch.full_like(center_dists, max_distance)).min(dim=1).values
    env.extras["log"]["min_obstacle_distance_mean"] = nearest_center.clamp(max=max_distance).mean()
    if hasattr(env, "_go2w_scenario_template_id"):
        sid = env._go2w_scenario_template_id
        narrow_mask = ((sid == 8) | (sid == 13) | (sid == 14)).float()
        cluttered_mask = (sid == 12).float()
        graze_active = (graze > 0.05).float()
        env.extras["log"]["narrow_gap_grazing_rate"] = (
            (graze_active * narrow_mask).sum() / narrow_mask.sum().clamp(min=1.0)
        )
        env.extras["log"]["cluttered_grazing_rate"] = (
            (graze_active * cluttered_mask).sum() / cluttered_mask.sum().clamp(min=1.0)
        )
    return penalty
