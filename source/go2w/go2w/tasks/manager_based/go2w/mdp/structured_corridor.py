# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Structured navigation scene helpers."""

from __future__ import annotations

import math

import torch

from .astar import GridMap2D, plan_astar_path


WorldPoint = tuple[float, float]


def _bezier_arc(
    A: WorldPoint, B: WorldPoint, C: WorldPoint, corner_radius: float, arc_steps: int
) -> list[WorldPoint] | None:
    """Return quadratic Bezier arc points for corner B, or None if the corner is too shallow.

    Returns None when the turn angle is below ~11 deg or segments are degenerate.
    The returned list starts at P0 (cut-back from B along A->B) and ends at P2
    (cut-forward from B along B->C), inclusive.
    """
    ab = (B[0] - A[0], B[1] - A[1])
    bc = (C[0] - B[0], C[1] - B[1])
    len_ab = math.hypot(*ab)
    len_bc = math.hypot(*bc)
    if len_ab < 1.0e-6 or len_bc < 1.0e-6:
        return None
    cos_a = (ab[0] * bc[0] + ab[1] * bc[1]) / (len_ab * len_bc)
    cos_a = max(-1.0, min(1.0, cos_a))
    if math.acos(cos_a) < 0.2:
        return None
    d = min(corner_radius, 0.4 * len_ab, 0.4 * len_bc)
    uab = (ab[0] / len_ab, ab[1] / len_ab)
    ubc = (bc[0] / len_bc, bc[1] / len_bc)
    P0: WorldPoint = (B[0] - uab[0] * d, B[1] - uab[1] * d)
    P2: WorldPoint = (B[0] + ubc[0] * d, B[1] + ubc[1] * d)
    arc: list[WorldPoint] = []
    for k in range(arc_steps + 1):
        t = k / arc_steps
        mt = 1.0 - t
        arc.append((
            mt * mt * P0[0] + 2.0 * mt * t * B[0] + t * t * P2[0],
            mt * mt * P0[1] + 2.0 * mt * t * B[1] + t * t * P2[1],
        ))
    return arc


def _arc_is_free(arc: list[WorldPoint], grid: GridMap2D) -> bool:
    """Return True if every arc point lies in a free grid cell."""
    for pt in arc:
        cell = grid.world_to_grid(pt, clamp=False)
        if not grid.is_free(cell):
            return False
    return True


def _round_corners_safe(
    points: list[WorldPoint],
    corner_radius: float,
    grid: GridMap2D | None = None,
    arc_steps: int = 8,
) -> list[WorldPoint]:
    """Replace sharp interior corners with quadratic Bezier arcs.

    Each interior waypoint B is replaced by an arc only when:
      1. The turn angle exceeds ~11 deg.
      2. All arc points are free in ``grid`` (if provided).
    If a corner fails the safety check, the original sharp waypoint is kept.
    """
    if len(points) <= 2 or corner_radius <= 0.0:
        return list(points)

    result: list[WorldPoint] = [points[0]]
    for i in range(1, len(points) - 1):
        arc = _bezier_arc(points[i - 1], points[i], points[i + 1], corner_radius, arc_steps)
        if arc is None:
            result.append(points[i])
            continue
        if grid is not None and not _arc_is_free(arc, grid):
            result.append(points[i])
            continue
        result.extend(arc)
    result.append(points[-1])
    return result


def _sparsify_preserve_turns(points: list[WorldPoint], spacing: float) -> list[WorldPoint]:
    """Downsample a path without deleting explicit turn points."""
    if len(points) <= 2 or spacing <= 0.0:
        return list(points)

    sparse: list[WorldPoint] = [points[0]]
    last_kept = points[0]
    for idx in range(1, len(points) - 1):
        prev = points[idx - 1]
        curr = points[idx]
        nxt = points[idx + 1]
        v0 = (curr[0] - prev[0], curr[1] - prev[1])
        v1 = (nxt[0] - curr[0], nxt[1] - curr[1])
        turn = abs(v0[0] * v1[1] - v0[1] * v1[0]) > 1.0e-8
        far_enough = math.hypot(curr[0] - last_kept[0], curr[1] - last_kept[1]) >= spacing
        if turn or far_enough:
            sparse.append(curr)
            last_kept = curr
    if sparse[-1] != points[-1]:
        sparse.append(points[-1])
    return sparse


def l_corridor_wall_specs(
    leg_length: float,
    corridor_width: float,
    wall_thickness: float,
) -> tuple[tuple[float, float, float, float, float], ...]:
    """Return wall specs as (x, y, yaw, length, thickness) in corridor-local frame."""
    half_width = corridor_width * 0.5
    return (
        (leg_length * 0.5, -half_width - wall_thickness * 0.5, 0.0, leg_length + wall_thickness, wall_thickness),
        ((leg_length - half_width) * 0.5, half_width + wall_thickness * 0.5, 0.0, leg_length - half_width, wall_thickness),
        (leg_length - half_width - wall_thickness * 0.5, (leg_length + half_width) * 0.5, math.pi * 0.5, leg_length - half_width, wall_thickness),
        (leg_length + half_width + wall_thickness * 0.5, leg_length * 0.5, math.pi * 0.5, leg_length + wall_thickness, wall_thickness),
    )




def structured_corridor_centerline(
    corridor_kind: str,
    leg_length: float,
    corridor_width: float,
    turn_length: float | None = None,
) -> tuple[WorldPoint, ...]:
    """Return a corridor-local centerline for a named structured scene."""
    kind = corridor_kind.lower()
    if kind == "l_corridor":
        return ((0.0, 0.0), (leg_length, 0.0), (leg_length, leg_length))
    if kind == "serpentine_corridor":
        step = turn_length if turn_length is not None else max(corridor_width * 1.7, leg_length * 0.38)
        return (
            (0.0, 0.0),
            (leg_length, 0.0),
            (leg_length, step),
            (0.0, step),
            (0.0, 2.0 * step),
            (leg_length, 2.0 * step),
        )
    raise ValueError(f"Unsupported structured corridor kind: {corridor_kind!r}.")


def densify_polyline(points: tuple[WorldPoint, ...], spacing: float = 0.35) -> list[WorldPoint]:
    """Return evenly spaced points along a polyline."""
    if len(points) < 2:
        return list(points)
    spacing = max(spacing, 1.0e-3)
    dense: list[WorldPoint] = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = math.hypot(dx, dy)
        if length <= 1.0e-6:
            continue
        steps = max(1, int(math.ceil(length / spacing)))
        for step in range(1, steps + 1):
            t = step / steps
            dense.append((a[0] + dx * t, a[1] + dy * t))
    return dense


def _corridor_wall_specs_from_centerline(
    centerline: tuple[WorldPoint, ...],
    corridor_width: float,
    wall_thickness: float,
) -> tuple[tuple[float, float, float, float, float], ...]:
    """Return wall cuboids for a polyline corridor centerline."""
    specs: list[tuple[float, float, float, float, float]] = []
    wall_offset = corridor_width * 0.5 + wall_thickness * 0.5
    for a, b in zip(centerline[:-1], centerline[1:]):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = math.hypot(dx, dy)
        if length <= 1.0e-6:
            continue
        ux = dx / length
        uy = dy / length
        nx = -uy
        ny = ux
        mx = (a[0] + b[0]) * 0.5
        my = (a[1] + b[1]) * 0.5
        yaw = math.atan2(dy, dx)
        wall_length = length + wall_thickness
        specs.append((mx + nx * wall_offset, my + ny * wall_offset, yaw, wall_length, wall_thickness))
        specs.append((mx - nx * wall_offset, my - ny * wall_offset, yaw, wall_length, wall_thickness))
    return tuple(specs)


def _line_wall_spec(a: WorldPoint, b: WorldPoint, wall_thickness: float) -> tuple[float, float, float, float, float]:
    """Return one wall cuboid spec from endpoint coordinates."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = math.hypot(dx, dy)
    return (
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        math.atan2(dy, dx),
        length + wall_thickness,
        wall_thickness,
    )


def serpentine_corridor_wall_specs(
    leg_length: float,
    corridor_width: float,
    wall_thickness: float,
    turn_length: float | None = None,
) -> tuple[tuple[float, float, float, float, float], ...]:
    """Return boundary wall specs for a three-lane serpentine corridor."""
    step = turn_length if turn_length is not None else max(corridor_width * 1.7, leg_length * 0.38)
    h = corridor_width * 0.5
    l = leg_length
    s = step
    end_cap_offset = max(0.8, corridor_width * 0.30)
    boundary = (
        (-end_cap_offset, -h),
        (l + h, -h),
        (l + h, s + h),
        (h, s + h),
        (h, 2.0 * s - h),
        (l + end_cap_offset, 2.0 * s - h),
        (l + end_cap_offset, 2.0 * s + h),
        (-h, 2.0 * s + h),
        (-h, s - h),
        (l - h, s - h),
        (l - h, h),
        (-end_cap_offset, h),
        (-end_cap_offset, -h),
    )
    cap_size = max(wall_thickness * 2.5, 0.35)
    wall_lines = tuple(zip(boundary[:-1], boundary[1:]))
    cap_points = set(boundary[:-1])
    caps = tuple((x, y, 0.0, cap_size, cap_size) for x, y in sorted(cap_points))
    return tuple(_line_wall_spec(a, b, wall_thickness) for a, b in wall_lines) + caps


def structured_corridor_wall_specs(
    corridor_kind: str,
    leg_length: float,
    corridor_width: float,
    wall_thickness: float,
    turn_length: float | None = None,
) -> tuple[tuple[float, float, float, float, float], ...]:
    """Return wall specs for a named structured corridor."""
    kind = corridor_kind.lower()
    if kind == "l_corridor":
        return l_corridor_wall_specs(leg_length, corridor_width, wall_thickness)
    if kind == "serpentine_corridor":
        return serpentine_corridor_wall_specs(leg_length, corridor_width, wall_thickness, turn_length)
    centerline = structured_corridor_centerline(corridor_kind, leg_length, corridor_width, turn_length)
    return _corridor_wall_specs_from_centerline(centerline, corridor_width, wall_thickness)


def _distance_to_segment(x: float, y: float, a: WorldPoint, b: WorldPoint) -> float:
    """Return point-to-segment distance in corridor-local 2D coordinates."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 1.0e-12:
        return math.hypot(x - a[0], y - a[1])
    t = ((x - a[0]) * dx + (y - a[1]) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    px = a[0] + t * dx
    py = a[1] + t * dy
    return math.hypot(x - px, y - py)


def _polyline_corridor_is_free(
    x: float,
    y: float,
    centerline: tuple[WorldPoint, ...],
    corridor_width: float,
    robot_inflation: float,
) -> bool:
    """Return whether a point lies inside the inflated polyline corridor."""
    half_width = max(0.05, corridor_width * 0.5 - robot_inflation)
    return any(_distance_to_segment(x, y, a, b) <= half_width for a, b in zip(centerline[:-1], centerline[1:]))


def build_polyline_corridor_grid(
    centerline: tuple[WorldPoint, ...],
    corridor_width: float,
    robot_inflation: float,
    grid_resolution: float,
) -> tuple[GridMap2D, WorldPoint, WorldPoint]:
    """Build an occupancy grid around any polyline corridor."""
    xs = [p[0] for p in centerline]
    ys = [p[1] for p in centerline]
    margin = corridor_width
    x_min = min(xs) - margin
    y_min = min(ys) - margin
    x_max = max(xs) + margin
    y_max = max(ys) + margin
    nx = int(math.ceil((x_max - x_min) / grid_resolution)) + 1
    ny = int(math.ceil((y_max - y_min) / grid_resolution)) + 1

    free_grid = [[False for _ in range(ny)] for _ in range(nx)]
    for gx in range(nx):
        for gy in range(ny):
            x = x_min + gx * grid_resolution
            y = y_min + gy * grid_resolution
            free_grid[gx][gy] = _polyline_corridor_is_free(
                x, y, centerline, corridor_width, robot_inflation
            )

    first = centerline[0]
    second = centerline[1]
    last = centerline[-1]
    prev = centerline[-2]
    start_len = max(math.hypot(second[0] - first[0], second[1] - first[1]), 1.0e-6)
    goal_len = max(math.hypot(last[0] - prev[0], last[1] - prev[1]), 1.0e-6)
    start_xy = (
        first[0] + (second[0] - first[0]) / start_len * 0.35,
        first[1] + (second[1] - first[1]) / start_len * 0.35,
    )
    goal_xy = (
        last[0] - (last[0] - prev[0]) / goal_len * 0.35,
        last[1] - (last[1] - prev[1]) / goal_len * 0.35,
    )
    return GridMap2D(free_grid, origin=(x_min, y_min), resolution=grid_resolution), start_xy, goal_xy


def plan_structured_corridor_path(
    corridor_kind: str,
    leg_length: float,
    corridor_width: float,
    robot_inflation: float,
    grid_resolution: float,
    turn_length: float | None = None,
    clearance_cost_weight: float = 2.0,
    clearance_cost_sigma: float = 0.4,
    corner_rounding: bool = False,
    corner_radius: float = 0.5,
) -> list[WorldPoint]:
    """Return a cost-aware A*-planned path for a named structured corridor.

    Uses a clearance map so cells near walls cost more, naturally steering the
    path toward the corridor centre without hardcoding per-leg sub-problems.
    When corner_rounding is True, sharp sparse waypoints are replaced by
    quadratic Bezier arcs; each arc is validated against the occupancy grid
    so collision safety is preserved.
    """
    centerline = structured_corridor_centerline(corridor_kind, leg_length, corridor_width, turn_length)
    grid, start_xy, goal_xy = build_polyline_corridor_grid(
        centerline, corridor_width, robot_inflation, grid_resolution
    )
    result = plan_astar_path(
        grid,
        start_xy,
        goal_xy,
        allow_diagonal=True,
        prevent_corner_cutting=True,
        clearance_cost_weight=clearance_cost_weight,
        clearance_cost_sigma=clearance_cost_sigma,
    )
    sparse = _sparsify_preserve_turns(result.points, 0.35)
    if corner_rounding:
        return _round_corners_safe(sparse, corner_radius, grid=grid)
    return sparse


def project_polyline_corridor_local(
    local_xy: torch.Tensor,
    centerline_xy: torch.Tensor,
    corridor_width: torch.Tensor,
) -> torch.Tensor:
    """Project points into the union of corridor rectangles around a polyline."""
    if centerline_xy.dim() == 2:
        centerline_xy = centerline_xy.unsqueeze(0).expand(local_xy.shape[0], -1, -1)
    half_width = corridor_width.view(-1, 1) * 0.5
    best_error = torch.full(local_xy.shape[:-1], float("inf"), device=local_xy.device, dtype=local_xy.dtype)
    best_projected = local_xy
    for segment_idx in range(centerline_xy.shape[1] - 1):
        a = centerline_xy[:, segment_idx].unsqueeze(1)
        b = centerline_xy[:, segment_idx + 1].unsqueeze(1)
        seg = b - a
        seg_len_sq = (seg * seg).sum(dim=-1).clamp(min=1.0e-8)
        t = ((local_xy - a) * seg).sum(dim=-1) / seg_len_sq
        t = t.clamp(0.0, 1.0).unsqueeze(-1)
        closest = a + t * seg
        lateral = local_xy - closest
        dist = lateral.norm(dim=-1).clamp(min=1.0e-8)
        clamped_dist = torch.minimum(dist, half_width)
        projected = closest + lateral / dist.unsqueeze(-1) * clamped_dist.unsqueeze(-1)
        error = (projected - local_xy).norm(dim=-1)
        update = error < best_error
        best_error = torch.where(update, error, best_error)
        best_projected = torch.where(update.unsqueeze(-1), projected, best_projected)
    return best_projected


def nearest_polyline_tangent_local(local_xy: torch.Tensor, centerline_xy: torch.Tensor) -> torch.Tensor:
    """Return the nearest centerline segment tangent for each local point."""
    if centerline_xy.dim() == 2:
        centerline_xy = centerline_xy.unsqueeze(0).expand(local_xy.shape[0], -1, -1)
    best_dist = torch.full(local_xy.shape[:-1], float("inf"), device=local_xy.device, dtype=local_xy.dtype)
    best_tangent = torch.zeros_like(local_xy)
    for segment_idx in range(centerline_xy.shape[1] - 1):
        a = centerline_xy[:, segment_idx].unsqueeze(1)
        b = centerline_xy[:, segment_idx + 1].unsqueeze(1)
        seg = b - a
        seg_len = seg.norm(dim=-1).clamp(min=1.0e-8)
        tangent = seg / seg_len.unsqueeze(-1)
        t = ((local_xy - a) * seg).sum(dim=-1) / (seg_len * seg_len)
        t = t.clamp(0.0, 1.0).unsqueeze(-1)
        closest = a + t * seg
        dist = (local_xy - closest).norm(dim=-1)
        update = dist < best_dist
        best_dist = torch.where(update, dist, best_dist)
        best_tangent = torch.where(update.unsqueeze(-1), tangent.expand_as(local_xy), best_tangent)
    return best_tangent
