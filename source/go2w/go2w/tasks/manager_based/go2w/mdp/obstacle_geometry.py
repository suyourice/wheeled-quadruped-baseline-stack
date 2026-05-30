# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared footprint metadata helpers for Go2-W navigation obstacles."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


OBSTACLE_SHAPE_CUBOID = 0
OBSTACLE_SHAPE_CYLINDER = 1
OBSTACLE_SHAPE_CONE = 2
DEFAULT_OBSTACLE_WIDTH = 0.30
DEFAULT_OBSTACLE_DEPTH = 0.30
DEFAULT_OBSTACLE_EFFECTIVE_RADIUS = math.sqrt(
    (DEFAULT_OBSTACLE_WIDTH * 0.5) ** 2 + (DEFAULT_OBSTACLE_DEPTH * 0.5) ** 2
)


def _effective_radius(
    shape_ids: torch.Tensor,
    widths: torch.Tensor,
    depths: torch.Tensor,
) -> torch.Tensor:
    """Return footprint radius for cuboids, cylinders, and cones."""
    cuboid_radius = torch.sqrt((widths * 0.5).square() + (depths * 0.5).square())
    round_radius = widths * 0.5
    return torch.where(shape_ids == OBSTACLE_SHAPE_CUBOID, cuboid_radius, round_radius)


def _ensure_obstacle_metadata_buffers(env: ManagerBasedRLEnv, obstacle_names: list[str]) -> None:
    """Allocate per-environment footprint metadata buffers if required."""
    shape = (env.num_envs, len(obstacle_names))
    if (
        not hasattr(env, "_go2w_obstacle_active_mask")
        or env._go2w_obstacle_active_mask.shape != shape
    ):
        env._go2w_obstacle_active_mask = torch.zeros(shape, dtype=torch.bool, device=env.device)
        env._go2w_obstacle_shape_id = torch.full(
            shape, OBSTACLE_SHAPE_CUBOID, dtype=torch.long, device=env.device
        )
        env._go2w_obstacle_width = torch.full(shape, DEFAULT_OBSTACLE_WIDTH, device=env.device)
        env._go2w_obstacle_depth = torch.full(shape, DEFAULT_OBSTACLE_DEPTH, device=env.device)
        env._go2w_obstacle_effective_radius = torch.full(
            shape, DEFAULT_OBSTACLE_EFFECTIVE_RADIUS, device=env.device
        )
        env._go2w_obstacle_yaw = torch.zeros(shape, device=env.device)
    if not hasattr(env, "_go2w_obstacle_yaw") or env._go2w_obstacle_yaw.shape != shape:
        env._go2w_obstacle_yaw = torch.zeros(shape, device=env.device)
    if not hasattr(env, "_go2w_obstacle_radius_margin"):
        env._go2w_obstacle_radius_margin = 0.0


def set_obstacle_metadata(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    obstacle_names: list[str],
    active_mask: torch.Tensor,
    *,
    obstacle_radius_margin: float = 0.0,
    fixed_obstacle_shape_ids: tuple[int, ...] | None = None,
    fixed_obstacle_widths: tuple[float, ...] | None = None,
    fixed_obstacle_depths: tuple[float, ...] | None = None,
    obstacle_yaws: torch.Tensor | None = None,
) -> None:
    """Write active and footprint metadata for newly placed obstacle slots."""
    _ensure_obstacle_metadata_buffers(env, obstacle_names)
    n = len(env_ids)
    k = len(obstacle_names)
    device = env.device
    shape_ids = torch.full((n, k), OBSTACLE_SHAPE_CUBOID, dtype=torch.long, device=device)
    widths = torch.full((n, k), DEFAULT_OBSTACLE_WIDTH, device=device)
    depths = torch.full((n, k), DEFAULT_OBSTACLE_DEPTH, device=device)

    if fixed_obstacle_shape_ids is not None:
        if (
            fixed_obstacle_widths is None
            or fixed_obstacle_depths is None
            or len(fixed_obstacle_shape_ids) != k
            or len(fixed_obstacle_widths) != k
            or len(fixed_obstacle_depths) != k
        ):
            raise ValueError("Fixed obstacle metadata must contain one shape, width, and depth per obstacle slot.")
        shape_ids[:] = torch.tensor(fixed_obstacle_shape_ids, dtype=torch.long, device=device).unsqueeze(0)
        widths[:] = torch.tensor(fixed_obstacle_widths, device=device).unsqueeze(0)
        depths[:] = torch.tensor(fixed_obstacle_depths, device=device).unsqueeze(0)

    env._go2w_obstacle_active_mask[env_ids] = active_mask
    env._go2w_obstacle_shape_id[env_ids] = shape_ids
    env._go2w_obstacle_width[env_ids] = widths
    env._go2w_obstacle_depth[env_ids] = depths
    env._go2w_obstacle_effective_radius[env_ids] = _effective_radius(shape_ids, widths, depths)
    if obstacle_yaws is None:
        env._go2w_obstacle_yaw[env_ids] = torch.zeros((n, k), device=device)
    else:
        if obstacle_yaws.shape != (n, k):
            raise ValueError("Obstacle yaw metadata must contain one yaw per env and obstacle slot.")
        env._go2w_obstacle_yaw[env_ids] = obstacle_yaws
    env._go2w_obstacle_radius_margin = max(0.0, obstacle_radius_margin)


def obstacle_active_mask(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    center_distances: torch.Tensor,
    max_distance: float,
) -> torch.Tensor:
    """Return metadata activity when available, with parked-position fallback."""
    if (
        hasattr(env, "_go2w_obstacle_active_mask")
        and env._go2w_obstacle_active_mask.shape == center_distances.shape
    ):
        return env._go2w_obstacle_active_mask
    return center_distances < max_distance


def obstacle_effective_radius(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    reference: torch.Tensor,
    fallback_radius: float = DEFAULT_OBSTACLE_EFFECTIVE_RADIUS,
) -> torch.Tensor:
    """Return physical footprint radii aligned with an obstacle tensor."""
    if (
        hasattr(env, "_go2w_obstacle_effective_radius")
        and env._go2w_obstacle_effective_radius.shape == reference.shape
    ):
        return env._go2w_obstacle_effective_radius
    return torch.full_like(reference, fallback_radius)


def obstacle_risk_radius(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    reference: torch.Tensor,
    fallback_radius: float = DEFAULT_OBSTACLE_EFFECTIVE_RADIUS,
) -> torch.Tensor:
    """Return footprint radius expanded by the configured robustness margin."""
    radii = obstacle_effective_radius(env, obstacle_names, reference, fallback_radius)
    margin = float(getattr(env, "_go2w_obstacle_radius_margin", 0.0))
    return radii + margin


def footprint_clearance(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    center_distances: torch.Tensor,
    robot_safety_radius: float = 0.0,
    fallback_radius: float = DEFAULT_OBSTACLE_EFFECTIVE_RADIUS,
) -> torch.Tensor:
    """Return center distance minus obstacle footprint and robot safety radii."""
    radii = obstacle_risk_radius(env, obstacle_names, center_distances, fallback_radius)
    return center_distances - radii - robot_safety_radius
