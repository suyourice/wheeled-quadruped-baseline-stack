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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ensure_navigation_goal_buffers(env: ManagerBasedRLEnv) -> None:
    """Create goal-navigation buffers on demand for reward/termination helpers."""
    if not hasattr(env, "_go2w_goal_pos_w"):
        env._go2w_goal_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._go2w_goal_heading_w = torch.zeros(env.num_envs, device=env.device)
        env._go2w_start_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._go2w_start_heading_w = torch.zeros(env.num_envs, device=env.device)


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


def joint_deviation_l1_curriculum(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    start_steps: int = 0,
    warmup_steps: int = 0,
    start_iteration: int | None = None,
    warmup_iterations: int | None = None,
    steps_per_iteration: int = 128,
) -> torch.Tensor:
    """Joint deviation from the default pose with an optional curriculum scale."""
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return _curriculum_scale(
        env, start_steps, warmup_steps, start_iteration, warmup_iterations, steps_per_iteration
    ) * torch.sum(
        torch.abs(joint_pos - default_joint_pos), dim=1
    )


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
    return progress.clamp(-clip, clip) / clip


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
    """Count obstacle bodies with contact, with an optional curriculum scale."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history
    if sensor_cfg.body_ids is not None:
        forces = forces[:, :, sensor_cfg.body_ids, :]
    max_forces = forces.norm(dim=-1).max(dim=1)[0]
    contacts = (max_forces > threshold).float().sum(dim=1)
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
    """Terminate an episode when any obstacle contact force exceeds threshold."""
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


def goal_distance_tanh_reward(
    env: ManagerBasedRLEnv,
    std: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward being close to the sampled local-navigation goal."""
    goal_distance, _ = _goal_command_from_buffers(env, asset_cfg)
    return 1.0 - torch.tanh(goal_distance / max(std, 1.0e-6))


def goal_heading_tanh_reward(
    env: ManagerBasedRLEnv,
    std: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward aligning the robot heading with the sampled goal heading."""
    _, heading_error = _goal_command_from_buffers(env, asset_cfg)
    return 1.0 - torch.tanh(heading_error / max(std, 1.0e-6))


def goal_reached_bonus(
    env: ManagerBasedRLEnv,
    position_threshold: float = 0.35,
    heading_threshold: float = 0.6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Binary success bonus when the robot reaches the local-navigation goal."""
    goal_distance, heading_error = _goal_command_from_buffers(env, asset_cfg)
    return ((goal_distance <= position_threshold) & (heading_error <= heading_threshold)).float()


def goal_reached_termination(
    env: ManagerBasedRLEnv,
    position_threshold: float = 0.35,
    heading_threshold: float = 0.6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate successful local-navigation episodes once the goal is reached."""
    goal_distance, heading_error = _goal_command_from_buffers(env, asset_cfg)
    return (goal_distance <= position_threshold) & (heading_error <= heading_threshold)


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
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Success bonus that immediately samples a new goal without ending the episode.

    When the robot enters the goal zone, it receives a +1 reward (scaled by weight
    in the config) and the goal is moved to a fresh position relative to the
    robot's current pose.  Obstacles remain in place so the robot continues
    navigating through the same cluttered environment.
    """
    if not hasattr(env, "_go2w_goals_reached_episode"):
        env._go2w_goals_reached_episode = torch.zeros(env.num_envs, device=env.device)

    goal_distance, heading_error = _goal_command_from_buffers(env, asset_cfg)
    reached = (goal_distance <= position_threshold) & (heading_error <= heading_threshold)

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

    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["goals_per_episode"] = env._go2w_goals_reached_episode.mean()

    return reached.float()
