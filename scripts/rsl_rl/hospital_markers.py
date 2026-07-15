# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared visualization-marker helpers for the hospital teacher play/visualize scripts.

Imports Isaac Lab modules at module level — import this only after AppLauncher
has started the simulation app.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_from_angle_axis


def line_marker_transforms(points: torch.Tensor, z_offset: float = 0.10):
    """Return cylinder marker transforms for a connected 3D polyline."""
    if points.shape[0] < 2:
        return None
    points = points.clone()
    points[:, 2] += z_offset
    start = points[:-1]
    end = points[1:]
    direction = end - start
    lengths = direction.norm(dim=-1)
    valid = lengths > 0.03
    if not bool(valid.any()):
        return None

    start = start[valid]
    end = end[valid]
    direction = direction[valid]
    lengths = lengths[valid]
    positions = (start + end) * 0.5

    direction_norm = direction / lengths.unsqueeze(-1).clamp(min=1.0e-6)
    default_axis = torch.zeros_like(direction_norm)
    default_axis[:, 2] = 1.0
    rotation_axis = torch.linalg.cross(default_axis, direction_norm, dim=-1)
    rotation_axis_norm = rotation_axis.norm(dim=-1)
    fallback_axis = torch.zeros_like(rotation_axis)
    fallback_axis[:, 0] = 1.0
    rotation_axis = torch.where(
        (rotation_axis_norm > 1.0e-6).unsqueeze(-1),
        rotation_axis / rotation_axis_norm.unsqueeze(-1).clamp(min=1.0e-6),
        fallback_axis,
    )
    cos_angle = (default_axis * direction_norm).sum(dim=-1).clamp(-1.0, 1.0)
    orientations = quat_from_angle_axis(torch.acos(cos_angle), rotation_axis)
    scales = torch.ones(positions.shape[0], 3, device=positions.device, dtype=positions.dtype)
    scales[:, 2] = lengths
    return positions, orientations, scales


def make_sphere_marker(prim_path: str, radius: float, color: tuple[float, float, float]) -> VisualizationMarkers:
    return VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
            },
        )
    )


def make_line_marker(prim_path: str, radius: float, color: tuple[float, float, float]) -> VisualizationMarkers:
    return VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "line": sim_utils.CylinderCfg(
                    radius=radius,
                    height=1.0,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=1.0),
                ),
            },
        )
    )
