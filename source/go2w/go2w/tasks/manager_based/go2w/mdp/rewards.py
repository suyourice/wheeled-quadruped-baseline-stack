# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions for the Go2-W locomotion task.

Functions here supplement the standard rewards from isaaclab.envs.mdp.
They are adapted from the Dodo locomotion project (IsaacLab_Dodo).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import yaw_quat, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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


def _obstacle_path_geometry(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    path_length: float = 1.6,
    path_width: float = 0.55,
    min_command_speed: float = 0.2,
    score_power: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return commanded-path risk scores plus signed avoidance geometry."""
    if len(obstacle_names) == 0:
        empty = torch.zeros((env.num_envs, 0), device=env.device)
        zeros = torch.zeros(env.num_envs, device=env.device)
        zeros_2 = torch.zeros((env.num_envs, 2), device=env.device)
        return empty, empty, zeros_2, zeros_2, zeros

    robot = env.scene[robot_cfg.name]
    robot_pos = robot.data.root_pos_w[:, :2]
    obs_positions = torch.stack(
        [env.scene[name].data.root_pos_w[:, :2] for name in obstacle_names], dim=1
    )

    rel_xy_w = obs_positions - robot_pos.unsqueeze(1)
    rel_w = torch.zeros((*rel_xy_w.shape[:2], 3), device=env.device)
    rel_w[..., :2] = rel_xy_w
    rel_yaw = quat_apply_inverse(
        yaw_quat(robot.data.root_quat_w)
        .unsqueeze(1)
        .expand(-1, len(obstacle_names), -1)
        .reshape(-1, 4),
        rel_w.reshape(-1, 3),
    ).reshape(env.num_envs, len(obstacle_names), 3)[..., :2]

    command_xy = env.command_manager.get_command(command_name)[:, :2]
    command_speed = command_xy.norm(dim=1)
    command_dir = command_xy / command_speed.clamp(min=1.0e-6).unsqueeze(1)

    forward = (rel_yaw * command_dir.unsqueeze(1)).sum(dim=-1)
    signed_lateral = (
        command_dir[:, 0].unsqueeze(1) * rel_yaw[..., 1]
        - command_dir[:, 1].unsqueeze(1) * rel_yaw[..., 0]
    )
    lateral = signed_lateral.abs()

    in_path = (
        (command_speed > min_command_speed).unsqueeze(1)
        & (forward > 0.0)
        & (forward < path_length)
    )
    lateral_score = ((path_width - lateral).clamp(min=0.0) / max(path_width, 1.0e-6)).pow(
        score_power
    )
    forward_score = ((path_length - forward).clamp(min=0.0) / max(path_length, 1.0e-6)).pow(
        score_power
    )
    per_obstacle = torch.where(in_path, lateral_score * forward_score, torch.zeros_like(forward))
    left_dir = torch.stack((-command_dir[:, 1], command_dir[:, 0]), dim=1)
    return per_obstacle, signed_lateral, command_dir, left_dir, command_speed


def _obstacle_path_scores(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    path_length: float = 1.6,
    path_width: float = 0.55,
    min_command_speed: float = 0.2,
    score_power: float = 2.0,
) -> torch.Tensor:
    """Return per-obstacle commanded-path risk scores before aggregation."""
    per_obstacle, _, _, _, _ = _obstacle_path_geometry(
        env,
        obstacle_names,
        command_name,
        robot_cfg,
        path_length,
        path_width,
        min_command_speed,
        score_power,
    )
    return per_obstacle


def obstacle_path_clearance_penalty(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    path_length: float = 1.6,
    path_width: float = 0.55,
    min_command_speed: float = 0.2,
    score_power: float = 2.0,
    aggregation: str = "max",
    sum_clip: float = 1.5,
    start_steps: int = 0,
    warmup_steps: int = 0,
    start_iteration: int | None = None,
    warmup_iterations: int | None = None,
    steps_per_iteration: int = 128,
) -> torch.Tensor:
    """Penalize obstacles inside the commanded path corridor.

    Unlike TTC, this does not use the robot's current velocity, so the policy
    cannot reduce the penalty by destabilizing locomotion or simply slowing the
    root velocity. It only asks the robot to keep the commanded path clear.
    """
    if len(obstacle_names) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    per_obstacle = _obstacle_path_scores(
        env,
        obstacle_names,
        command_name,
        robot_cfg,
        path_length,
        path_width,
        min_command_speed,
        score_power,
    )
    if aggregation == "max":
        total = per_obstacle.max(dim=1).values
    elif aggregation == "sum":
        total = per_obstacle.sum(dim=1)
    elif aggregation == "sum_clamped":
        total = per_obstacle.sum(dim=1).clamp(max=sum_clip)
    else:
        raise ValueError(f"Unsupported obstacle path-clearance aggregation: {aggregation}")

    scale = _curriculum_scale(
        env, start_steps, warmup_steps, start_iteration, warmup_iterations, steps_per_iteration
    )
    return total * scale


def _obstacle_avoidance_terms(
    env: ManagerBasedRLEnv,
    command_name: str,
    obstacle_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    path_length: float = 1.6,
    path_width: float = 0.55,
    min_command_speed: float = 0.2,
    score_power: float = 1.0,
    risk_clip: float = 1.5,
    center_deadband: float = 0.05,
    min_progress: float = 0.25,
    start_steps: int = 0,
    warmup_steps: int = 0,
    start_iteration: int | None = None,
    warmup_iterations: int | None = None,
    steps_per_iteration: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return shared gates and signed motion terms for direct obstacle avoidance."""
    zeros = torch.zeros(env.num_envs, device=env.device)
    if len(obstacle_names) == 0:
        return zeros, zeros, zeros, zeros

    per_obstacle, signed_lateral, command_dir, left_dir, command_speed = _obstacle_path_geometry(
        env,
        obstacle_names,
        command_name,
        asset_cfg,
        path_length,
        path_width,
        min_command_speed,
        score_power,
    )
    path_mass = per_obstacle.sum(dim=1)
    risk = path_mass.clamp(max=risk_clip) / max(risk_clip, 1.0e-6)

    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(
        yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )[:, :2]
    lateral_vel = (vel_yaw * left_dir).sum(dim=1)

    weighted_lateral = (per_obstacle * signed_lateral).sum(dim=1) / path_mass.clamp(
        min=1.0e-6
    )
    obstacle_side = -torch.sign(weighted_lateral)
    lateral_sign = torch.sign(lateral_vel)
    command_lateral_sign = torch.sign(env.command_manager.get_command(command_name)[:, 1])
    fallback_side = torch.where(
        lateral_sign.abs() > 0.0,
        lateral_sign,
        torch.where(
            command_lateral_sign.abs() > 0.0,
            command_lateral_sign,
            torch.ones_like(lateral_sign),
        ),
    )
    desired_side = torch.where(
        weighted_lateral.abs() > center_deadband, obstacle_side, fallback_side
    )

    progress = (vel_yaw * command_dir).sum(dim=1)
    progress_gate = (progress.clamp(min=0.0) / max(min_progress, 1.0e-6)).clamp(max=1.0)
    curriculum_scale = _curriculum_scale(
        env, start_steps, warmup_steps, start_iteration, warmup_iterations, steps_per_iteration
    )
    active = (command_speed > min_command_speed) & (path_mass > 0.0)
    gate = torch.where(active, risk * progress_gate * curriculum_scale, zeros)
    yaw_extra = asset.data.root_ang_vel_w[:, 2] - env.command_manager.get_command(command_name)[
        :, 2
    ]

    return gate, desired_side, lateral_vel, yaw_extra


def obstacle_lateral_avoidance_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    obstacle_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    path_length: float = 1.6,
    path_width: float = 0.55,
    min_command_speed: float = 0.2,
    score_power: float = 1.0,
    risk_clip: float = 1.5,
    target_lateral_speed: float = 0.35,
    center_deadband: float = 0.05,
    min_progress: float = 0.25,
    start_steps: int = 0,
    warmup_steps: int = 0,
    start_iteration: int | None = None,
    warmup_iterations: int | None = None,
    steps_per_iteration: int = 128,
) -> torch.Tensor:
    """Reward lateral motion to the side opposite the weighted path obstacle."""
    gate, desired_side, lateral_vel, _ = _obstacle_avoidance_terms(
        env,
        command_name,
        obstacle_names,
        asset_cfg,
        path_length,
        path_width,
        min_command_speed,
        score_power,
        risk_clip,
        center_deadband,
        min_progress,
        start_steps,
        warmup_steps,
        start_iteration,
        warmup_iterations,
        steps_per_iteration,
    )
    lateral_score = (desired_side * lateral_vel / max(target_lateral_speed, 1.0e-6)).clamp(
        min=0.0, max=1.0
    )
    return gate * lateral_score


def obstacle_yaw_avoidance_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    obstacle_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    path_length: float = 1.6,
    path_width: float = 0.55,
    min_command_speed: float = 0.2,
    score_power: float = 1.0,
    risk_clip: float = 1.5,
    target_yaw_rate: float = 0.8,
    center_deadband: float = 0.05,
    min_progress: float = 0.25,
    start_steps: int = 0,
    warmup_steps: int = 0,
    start_iteration: int | None = None,
    warmup_iterations: int | None = None,
    steps_per_iteration: int = 128,
) -> torch.Tensor:
    """Reward extra yaw rate toward the avoidance side when a path obstacle exists."""
    gate, desired_side, _, yaw_extra = _obstacle_avoidance_terms(
        env,
        command_name,
        obstacle_names,
        asset_cfg,
        path_length,
        path_width,
        min_command_speed,
        score_power,
        risk_clip,
        center_deadband,
        min_progress,
        start_steps,
        warmup_steps,
        start_iteration,
        warmup_iterations,
        steps_per_iteration,
    )
    yaw_score = (desired_side * yaw_extra / max(target_yaw_rate, 1.0e-6)).clamp(
        min=0.0, max=1.0
    )
    return gate * yaw_score


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
