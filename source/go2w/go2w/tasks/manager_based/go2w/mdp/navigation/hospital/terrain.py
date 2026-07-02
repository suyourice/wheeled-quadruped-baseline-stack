# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-terrain geometry for hospital teacher training and play environments."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

from ..global_planning.corridors import structured_corridor_wall_specs
from .specs import (
    HOSPITAL_TRAIN_LEG_LENGTH,
    HOSPITAL_TRAIN_WALL_HEIGHT,
    HOSPITAL_TRAIN_WALL_THICKNESS,
)

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------

HOSPITAL_PATH_POINT_COUNT = 16
# A* paths in a 5×5 grid span at most 9 junctions (8 steps, corner to corner).
HOSPITAL_CENTERLINE_POINT_COUNT = 9

# ---------------------------------------------------------------------------
# 5×5 junction grid
# ---------------------------------------------------------------------------
# 25 junctions at x,y ∈ {-20,-10,0,+10,+20}.
# Columns left→right: L(x=-20), CL(x=-10), C(x=0), CR(x=+10), R(x=+20).
# Rows top→bottom:    T(y=+20),  U(y=+10), M(y=0),  L(y=-10),  B(y=-20).

MAZE_JUNCTIONS: dict[str, tuple[float, float]] = {
    "TL":  (-20.0, +20.0), "TCL": (-10.0, +20.0), "TC": (0.0, +20.0), "TCR": (+10.0, +20.0), "TR":  (+20.0, +20.0),
    "UL":  (-20.0, +10.0), "UCL": (-10.0, +10.0), "UC": (0.0, +10.0), "UCR": (+10.0, +10.0), "UR":  (+20.0, +10.0),
    "ML":  (-20.0,   0.0), "MCL": (-10.0,   0.0), "MC": (0.0,  0.0),  "MCR": (+10.0,  0.0),  "MR":  (+20.0,  0.0),
    "LL":  (-20.0, -10.0), "LCL": (-10.0, -10.0), "LC": (0.0, -10.0), "LCR": (+10.0, -10.0), "LR":  (+20.0, -10.0),
    "BL":  (-20.0, -20.0), "BCL": (-10.0, -20.0), "BC": (0.0, -20.0), "BCR": (+10.0, -20.0), "BR":  (+20.0, -20.0),
}

# Stable ordering: index ↔ name for layout_id encoding (start_idx * 25 + end_idx).
MAZE_JUNCTION_NAMES: list[str] = list(MAZE_JUNCTIONS.keys())


# ---------------------------------------------------------------------------
# A* path planning infrastructure
# ---------------------------------------------------------------------------

# (GridMap2D, clearance_map) — built once on first call to cached_maze_path.
_MAZE_GRID_DATA: tuple | None = None
# Keyed by (start_idx, end_idx); path geometry does not depend on corridor width.
_MAZE_PATH_CACHE: dict[tuple[int, int], dict[str, object]] = {}


def _get_maze_grid_and_clearance() -> tuple:
    """Build and cache the maze occupancy grid + clearance map for A* planning.

    Uses half_width = 1.05 m (narrowest corridor 1.4 m minus robot inflation 0.35 m)
    so planned paths fit in any tile regardless of per-corridor width randomisation.
    Clearance map is pre-computed once so each cached_maze_path call is fast.
    """
    global _MAZE_GRID_DATA
    if _MAZE_GRID_DATA is not None:
        return _MAZE_GRID_DATA
    from ..global_planning.astar import GridMap2D, compute_clearance_map

    eff_hw = 1.05          # 1.4 m (min half_width) − 0.35 m robot inflation
    res = 0.20
    col_xs = [-20.0, -10.0, 0.0, 10.0, 20.0]
    row_ys = [20.0, 10.0, 0.0, -10.0, -20.0]
    margin = 0.5
    x_min = col_xs[0] - eff_hw - margin
    x_max = col_xs[-1] + eff_hw + margin
    y_min = row_ys[-1] - eff_hw - margin
    y_max = row_ys[0] + eff_hw + margin
    nx = int(math.ceil((x_max - x_min) / res)) + 1
    ny = int(math.ceil((y_max - y_min) / res)) + 1

    free_grid: list[list[bool]] = [[False] * ny for _ in range(nx)]
    for gx in range(nx):
        x = x_min + gx * res
        for gy in range(ny):
            y = y_min + gy * res
            free_grid[gx][gy] = (
                any(abs(y - ry) <= eff_hw for ry in row_ys)
                or any(abs(x - cx) <= eff_hw for cx in col_xs)
            )

    grid = GridMap2D(free_grid, origin=(x_min, y_min), resolution=res)
    clearance = compute_clearance_map(grid)
    _MAZE_GRID_DATA = (grid, clearance)
    return _MAZE_GRID_DATA


def cached_maze_path(start_idx: int, end_idx: int) -> dict[str, object]:
    """Return cached A*-planned corridor path between two junction indices.

    Uses the maze occupancy grid with a clearance cost to keep waypoints centred.
    Navigation waypoints, compact centerline observations, and actor placement
    segments are all derived from this same A* polyline.
    """
    key = (start_idx, end_idx)
    if (cached := _MAZE_PATH_CACHE.get(key)) is not None:
        return cached

    from ..global_planning.astar import astar_search

    start_name = MAZE_JUNCTION_NAMES[start_idx]
    end_name = MAZE_JUNCTION_NAMES[end_idx]

    grid, clearance = _get_maze_grid_and_clearance()
    astar_result = astar_search(
        grid,
        grid.world_to_grid(MAZE_JUNCTIONS[start_name]),
        grid.world_to_grid(MAZE_JUNCTIONS[end_name]),
        allow_diagonal=True,
        prevent_corner_cutting=True,
        clearance_map=clearance,
        clearance_cost_weight=1.5,
        clearance_cost_sigma=0.3,
    )

    # Dense A* path → 16 evenly-spaced waypoints for the navigation path buffer.
    fixed_path_local = _resample_polyline(astar_result.points, HOSPITAL_PATH_POINT_COUNT)

    # Use the same A* route for navigation, corridor observations, and actor
    # placement.  Separate shortest-path tie breakers can otherwise put actors
    # on a different but equally short maze corridor.
    centerline_tensor = _resample_polyline(astar_result.points, HOSPITAL_CENTERLINE_POINT_COUNT)
    centerline_pts = tuple((float(p[0]), float(p[1])) for p in centerline_tensor.tolist())

    path_points = [(float(p[0]), float(p[1])) for p in fixed_path_local.tolist()]
    segments: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    for a, b in zip(path_points[:-1], path_points[1:]):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if length > 1.0e-6:
            segments.append((a, b, length))
    total_length = max(sum(length for _, _, length in segments), 1.0e-6)

    result: dict[str, object] = {
        "centerline": centerline_pts,
        "centerline_tensor": centerline_tensor,
        "fixed_path_local": fixed_path_local,
        "segments": segments,
        "total_length": total_length,
    }
    _MAZE_PATH_CACHE[key] = result
    return result

# ---------------------------------------------------------------------------
# Polyline utilities
# ---------------------------------------------------------------------------


def _rand_corridor_widths(seed: float, n: int, lo: float, hi: float) -> list[float]:
    """Return n pseudo-random values in [lo, hi] from a float seed (LCG, deterministic)."""
    x = int(seed * (1 << 24)) & 0xFFFFFF
    result = []
    for _ in range(n):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        result.append(lo + (x / 0xFFFFFFFF) * (hi - lo))
    return result


def _resample_polyline(points: list[tuple[float, float]], count: int) -> torch.Tensor:
    """Resample a 2D polyline to a fixed point count."""
    if len(points) < 2:
        raise ValueError("Hospital training paths require at least two points.")
    distances = [0.0]
    for a, b in zip(points[:-1], points[1:]):
        distances.append(distances[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    total = max(distances[-1], 1.0e-6)
    result: list[tuple[float, float]] = []
    seg_idx = 0
    for sample_idx in range(count):
        target_s = total * sample_idx / (count - 1)
        while seg_idx < len(points) - 2 and distances[seg_idx + 1] < target_s:
            seg_idx += 1
        s0 = distances[seg_idx]
        s1 = max(distances[seg_idx + 1], s0 + 1.0e-6)
        u = (target_s - s0) / (s1 - s0)
        a = points[seg_idx]
        b = points[seg_idx + 1]
        result.append((a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u))
    return torch.tensor(result, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Maze terrain geometry
# ---------------------------------------------------------------------------


def _box_mesh(extents: tuple[float, float, float], center: tuple[float, float, float], yaw: float):
    """Create one yawed trimesh box."""
    import trimesh
    transform = trimesh.transformations.translation_matrix(center)
    transform = transform @ trimesh.transformations.rotation_matrix(yaw, (0.0, 0.0, 1.0))
    return trimesh.creation.box(extents=extents, transform=transform)


def hospital_maze_sub_terrain(
    difficulty: float, cfg: "HospitalMazeSubTerrainCfg"
) -> tuple[list, "np.ndarray"]:
    """Generate a 5×5 junction-grid maze as a sub-terrain tile.

    25 junctions connected by 40 corridor segments (20 vertical + 20 horizontal).
    Per-column and per-row half-widths are drawn from a LCG seeded by ``difficulty``
    so every tile gets a distinct corridor-width mix within ``half_width_range``.
    """
    import numpy as np
    import trimesh

    tile_w, tile_h = cfg.size
    tile_cx, tile_cy = tile_w * 0.5, tile_h * 0.5
    H = cfg.wall_height
    h = H * 0.5
    rx = cfg.rail_x    # = 20.0
    ry = cfg.rail_y    # = 20.0
    sp = cfg.spacing   # = 10.0

    hw_min, hw_max = cfg.half_width_range   # (1.4, 1.7) → corridor 2.8–3.4 m

    n_junc = round(2 * rx / sp) + 1            # = 5
    col_hws = _rand_corridor_widths(difficulty,        n_junc, hw_min, hw_max)
    row_hws = _rand_corridor_widths(difficulty + 0.5, n_junc, hw_min, hw_max)

    col_xs = [-rx + j * sp for j in range(n_junc)]   # [-20,-10, 0,+10,+20]
    row_ys = [ ry - i * sp for i in range(n_junc)]   # [+20,+10, 0,-10,-20]
    n_inner = n_junc - 1                              # = 4

    left_bx  = tile_cx - (rx + col_hws[0])
    right_bx = tile_cx - (rx + col_hws[-1])
    top_by   = tile_cy - (ry + row_hws[0])
    bot_by   = tile_cy - (ry + row_hws[-1])
    side_ey  = 2.0 * ry + row_hws[0] + row_hws[-1]
    side_cy  = (row_hws[0] - row_hws[-1]) * 0.5

    # (ex, ey, cx_rel, cy_rel) — relative to tile centre
    wall_defs: list[tuple[float, float, float, float]] = [
        (tile_w,   top_by,   0.0,                                   ry + row_hws[0]  + top_by  * 0.5),
        (tile_w,   bot_by,   0.0,                                 -(ry + row_hws[-1] + bot_by  * 0.5)),
        (left_bx,  side_ey, -(rx + col_hws[0]  + left_bx  * 0.5), side_cy),
        (right_bx, side_ey,   rx + col_hws[-1] + right_bx * 0.5,  side_cy),
    ]
    for j in range(n_inner):
        for i in range(n_inner):
            ex = sp - col_hws[j] - col_hws[j + 1]
            ey = sp - row_hws[i] - row_hws[i + 1]
            cx = (col_xs[j] + col_xs[j + 1]) * 0.5 + (col_hws[j] - col_hws[j + 1]) * 0.5
            cy = (row_ys[i] + row_ys[i + 1]) * 0.5 + (row_hws[i + 1] - row_hws[i]) * 0.5
            wall_defs.append((ex, ey, cx, cy))

    meshes = [
        _box_mesh((ex, ey, H), (tile_cx + cx_r, tile_cy + cy_r, h), 0.0)
        for ex, ey, cx_r, cy_r in wall_defs
    ]
    mesh = trimesh.util.concatenate(meshes)
    return [mesh], np.array([tile_cx, tile_cy, 0.0])


@configclass
class HospitalMazeSubTerrainCfg(SubTerrainBaseCfg):
    """Sub-terrain config for the 5×5 junction-grid hospital training maze."""

    function: Callable = hospital_maze_sub_terrain
    size: tuple[float, float] = (48.0, 48.0)
    wall_height: float = HOSPITAL_TRAIN_WALL_HEIGHT
    half_width_range: tuple[float, float] = (1.4, 1.7)   # hw range → corridor 2.8–3.4 m
    rail_x: float = 20.0
    rail_y: float = 20.0
    spacing: float = 10.0

# ---------------------------------------------------------------------------
# Play-environment wall terrain
# ---------------------------------------------------------------------------


def hospital_wall_sub_terrain(
    difficulty: float, cfg: "HospitalWallSubTerrainCfg"
) -> tuple[list, "np.ndarray"]:
    """Generate wall geometry for a single structured corridor as a sub-terrain tile.

    The corridor is centred at (tile_cx, tile_cy) so env_origins align with the
    corridor reference point used by the reset function.
    """
    import numpy as np
    import trimesh

    tile_cx = cfg.size[0] * 0.5
    tile_cy = cfg.size[1] * 0.5
    H = cfg.wall_height

    wall_specs = structured_corridor_wall_specs(
        cfg.corridor_kind,
        cfg.leg_length,
        cfg.corridor_width,
        cfg.wall_thickness,
        cfg.corridor_turn_length,
    )
    meshes = [
        _box_mesh((length, thickness, H), (tile_cx + x, tile_cy + y, H * 0.5), yaw)
        for x, y, yaw, length, thickness in wall_specs
    ]
    mesh = trimesh.util.concatenate(meshes)
    return [mesh], np.array([tile_cx, tile_cy, 0.0])


@configclass
class HospitalWallSubTerrainCfg(SubTerrainBaseCfg):
    """Sub-terrain config for a single structured corridor (for play envs)."""

    function: Callable = hospital_wall_sub_terrain
    size: tuple[float, float] = (30.0, 30.0)
    corridor_kind: str = "l_corridor"
    leg_length: float = HOSPITAL_TRAIN_LEG_LENGTH
    corridor_width: float = 2.6
    wall_thickness: float = HOSPITAL_TRAIN_WALL_THICKNESS
    wall_height: float = HOSPITAL_TRAIN_WALL_HEIGHT
    corridor_turn_length: float | None = None
