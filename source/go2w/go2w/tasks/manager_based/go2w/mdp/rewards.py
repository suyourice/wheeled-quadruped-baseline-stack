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

from .events import ensure_navigation_goal_buffers
from .nav_scenarios import NAV_SCENARIO_NAMES as _NAV_SCENARIO_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _goal_command_from_buffers(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return goal distance and heading error from the sampled task buffers."""
    ensure_navigation_goal_buffers(env)
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
    ensure_navigation_goal_buffers(env)
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
    ensure_navigation_goal_buffers(env)
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
