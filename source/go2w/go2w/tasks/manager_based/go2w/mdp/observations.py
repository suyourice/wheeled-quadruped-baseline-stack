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
    """Compute log-scaled per-ray distances from a RayCaster sensor.

    Returns log(distance) after clamping to [0.05, max_distance]. This keeps
    close obstacles prominent while compressing far/no-hit rays.

    Args:
        sensor_cfg: SceneEntityCfg pointing to a RayCaster sensor.
        max_distance: Distance used for normalization and clamping.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # ray_hits_w: (N, B, 3), pos_w: (N, 3)
    hit_positions = sensor.data.ray_hits_w  # (N, B, 3): world positions returned by the ray caster
    sensor_pos = sensor.data.pos_w  # (N, 3)
    # Compute distance from sensor to each hit point
    diff = hit_positions - sensor_pos.unsqueeze(1)  # (N, B, 3); sensor -> hit vector
    distances = torch.norm(diff, dim=-1)  # (N, B) distance to each reported hit point
    # Log-scale: sensitive to close obstacles, range ≈ [log(0.05), log(max_distance)]
    # no-hit rays return max_distance → log(20) ≈ 3.0
    # 0.1 m away              → log(0.1) ≈ -2.3
    return torch.log(distances.clamp(min=0.05, max=max_distance))


def obstacle_positions_rel(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_distance: float | None = None,
    normalize: bool = False,
    zero_beyond_max_distance: bool = True,
    num_closest: int | None = None,
) -> torch.Tensor:
    """Obstacle positions relative to the robot base in the robot's local frame.

    Returns a flattened tensor containing (x, y) relative positions for each
    obstacle. If num_closest is set, the output is sorted by current distance
    every call and padded with zeros when fewer candidates exist.

    Args:
        obstacle_names: List of scene entity names for each obstacle.
        robot_cfg: SceneEntityCfg for the robot.
        max_distance: Optional clipping/masking radius in meters.
        normalize: If True, divide relative positions by max_distance.
        zero_beyond_max_distance: If True, obstacles beyond max_distance return zero.
        num_closest: Optional fixed number of closest obstacles to return.
    """
    if len(obstacle_names) == 0:
        k = 0 if num_closest is None else num_closest
        return torch.zeros(env.num_envs, k * 2, device=env.device)

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

    rel_positions = torch.stack(rel_positions, dim=1)  # (N, num_obstacles, 2)
    dists = torch.norm(rel_positions, dim=-1)

    if num_closest is not None:
        k = min(num_closest, rel_positions.shape[1])
        closest_idx = torch.topk(dists, k=k, dim=1, largest=False, sorted=True).indices
        gather_idx = closest_idx.unsqueeze(-1).expand(-1, -1, 2)
        rel_positions = torch.gather(rel_positions, dim=1, index=gather_idx)
        dists = torch.gather(dists, dim=1, index=closest_idx)

        if k < num_closest:
            pad_shape = (rel_positions.shape[0], num_closest - k, 2)
            rel_positions = torch.cat(
                [rel_positions, torch.zeros(pad_shape, device=rel_positions.device, dtype=rel_positions.dtype)],
                dim=1,
            )
            dists = torch.cat(
                [dists, torch.full((dists.shape[0], num_closest - k), float("inf"), device=dists.device)],
                dim=1,
            )

    if max_distance is not None:
        rel_positions = rel_positions.clamp(min=-max_distance, max=max_distance)
        if zero_beyond_max_distance:
            rel_positions = torch.where(
                dists.unsqueeze(-1) <= max_distance,
                rel_positions,
                torch.zeros_like(rel_positions),
            )
        if normalize:
            rel_positions = rel_positions / max_distance

    return rel_positions.flatten(start_dim=1)  # (N, num_obstacles * 2)
