# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for the Go2-W obstacle avoidance task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lidar_distances(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_distance: float = 5.0,
) -> torch.Tensor:
    """Compute per-ray distances from a RayCaster sensor.

    Returns normalized distances in [0, 1] where 0 = at sensor and 1 = max_distance or no hit.
    Rays that miss all meshes return max_distance.

    Args:
        sensor_cfg: SceneEntityCfg pointing to a RayCaster sensor.
        max_distance: Distance used for normalization and clamping.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # ray_hits_w: (N, B, 3), pos_w: (N, 3)
    hit_positions = sensor.data.ray_hits_w  # (N, B, 3): world positions of ray hits; far point at max_distance if no hit
    sensor_pos = sensor.data.pos_w  # (N, 3)
    # Compute distance from sensor to each hit point
    diff = hit_positions - sensor_pos.unsqueeze(1)  # (N, B, 3); sensor -> hit vector
    distances = torch.norm(diff, dim=-1)  # (N, B) distance to each hit; max_distance if no hit (since hit_positions=0)
    # Clamp and normalize distances to [0, 1]
    distances = distances.clamp(max=max_distance) / max_distance # (N, B) in [0, 1], where 1 means no hit or hit at max_distance
    return distances


def obstacle_positions_rel(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Obstacle positions relative to the robot base in the robot's local frame.

    Returns a flattened tensor of shape (N, num_obstacles * 2) containing
    (x, y) relative positions for each obstacle.

    Args:
        obstacle_names: List of scene entity names for each obstacle.
        robot_cfg: SceneEntityCfg for the robot.
    """
    robot = env.scene[robot_cfg.name]
    robot_pos_w = robot.data.root_pos_w[:, :3]  # (N, 3)
    robot_quat_w = robot.data.root_quat_w  # (N, 4)

    rel_positions = []
    for name in obstacle_names:
        obstacle: RigidObject = env.scene[name]
        obs_pos_w = obstacle.data.root_pos_w[:, :3]  # (N, 3)
        rel_pos_w = obs_pos_w - robot_pos_w  # (N, 3)
        rel_pos_b = quat_apply_inverse(robot_quat_w, rel_pos_w)  # (N, 3)
        rel_positions.append(rel_pos_b[:, :2])  # only x, y

    return torch.cat(rel_positions, dim=-1)  # (N, num_obstacles * 2)
