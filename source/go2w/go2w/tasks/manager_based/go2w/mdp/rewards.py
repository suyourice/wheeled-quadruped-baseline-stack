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
