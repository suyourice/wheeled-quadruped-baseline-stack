# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable 2D occupancy-grid A* planner utilities."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from collections.abc import Sequence


GridCell = tuple[int, int]
WorldPoint = tuple[float, float]


@dataclass(frozen=True)
class GridMap2D:
    """A 2D free-space grid with metric world-coordinate conversion."""

    free: Sequence[Sequence[bool]]
    origin: WorldPoint = (0.0, 0.0)
    resolution: float = 1.0

    def __post_init__(self) -> None:
        if self.resolution <= 0.0:
            raise ValueError("Grid resolution must be positive.")
        if len(self.free) == 0:
            raise ValueError("GridMap2D requires a non-empty grid.")
        height = len(self.free[0])
        if height == 0:
            raise ValueError("GridMap2D requires a non-empty grid.")
        if any(len(column) != height for column in self.free):
            raise ValueError("GridMap2D requires a rectangular grid.")

    @property
    def width(self) -> int:
        """Return the number of x cells."""
        return len(self.free)

    @property
    def height(self) -> int:
        """Return the number of y cells."""
        return len(self.free[0])

    def in_bounds(self, cell: GridCell) -> bool:
        """Return whether a grid cell lies inside the map."""
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def is_free(self, cell: GridCell) -> bool:
        """Return whether a grid cell is traversable."""
        return self.in_bounds(cell) and bool(self.free[cell[0]][cell[1]])

    def world_to_grid(self, point: WorldPoint, *, clamp: bool = False) -> GridCell:
        """Convert a world point to the nearest grid cell."""
        gx = int(round((point[0] - self.origin[0]) / self.resolution))
        gy = int(round((point[1] - self.origin[1]) / self.resolution))
        if clamp:
            gx = max(0, min(self.width - 1, gx))
            gy = max(0, min(self.height - 1, gy))
        return gx, gy

    def grid_to_world(self, cell: GridCell) -> WorldPoint:
        """Convert a grid cell to the world point at its center."""
        return (
            self.origin[0] + cell[0] * self.resolution,
            self.origin[1] + cell[1] * self.resolution,
        )


@dataclass(frozen=True)
class AStarResult:
    """Result returned by the generic A* planner."""

    cells: list[GridCell]
    points: list[WorldPoint]
    cost: float
    expanded: int


def _heuristic(a: GridCell, b: GridCell) -> float:
    """Return the Euclidean grid-distance heuristic."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _neighbors(allow_diagonal: bool) -> tuple[tuple[int, int, float], ...]:
    """Return cardinal or 8-connected grid neighbors."""
    cardinal = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0))
    if not allow_diagonal:
        return cardinal
    diagonal = (
        (1, 1, 2.0**0.5),
        (1, -1, 2.0**0.5),
        (-1, 1, 2.0**0.5),
        (-1, -1, 2.0**0.5),
    )
    return cardinal + diagonal


def compute_clearance_map(grid: GridMap2D) -> list[list[float]]:
    """Compute the distance-to-nearest-obstacle for every cell, in metres.

    Multi-source Dijkstra outward from all occupied cells.  Occupied cells get
    clearance 0; free cells receive the shortest metric distance to any
    occupied cell through 8-connected propagation.
    """
    width, height = grid.width, grid.height
    res = grid.resolution
    clearance: list[list[float]] = [[math.inf] * height for _ in range(width)]
    pq: list[tuple[float, int, int]] = []

    for gx in range(width):
        for gy in range(height):
            if not grid.is_free((gx, gy)):
                clearance[gx][gy] = 0.0
                heapq.heappush(pq, (0.0, gx, gy))

    directions = (
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    )
    while pq:
        dist, cx, cy = heapq.heappop(pq)
        if dist > clearance[cx][cy]:
            continue
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            new_dist = clearance[cx][cy] + math.hypot(dx, dy) * res
            if new_dist < clearance[nx][ny]:
                clearance[nx][ny] = new_dist
                heapq.heappush(pq, (new_dist, nx, ny))

    return clearance


def astar_search(
    grid: GridMap2D,
    start: GridCell,
    goal: GridCell,
    *,
    allow_diagonal: bool = True,
    prevent_corner_cutting: bool = True,
    heuristic_weight: float = 1.0,
    clearance_map: list[list[float]] | None = None,
    clearance_cost_weight: float = 0.0,
    clearance_cost_sigma: float = 0.4,
) -> AStarResult:
    """Run A* on any 2D occupancy grid and return the reconstructed path.

    When clearance_map is provided and clearance_cost_weight > 0, each step
    adds a wall-proximity penalty that steers the path away from obstacles:

        wall_penalty = clearance_cost_weight * exp(-clearance / clearance_cost_sigma)
    """
    if heuristic_weight < 0.0:
        raise ValueError("heuristic_weight must be non-negative.")
    if not grid.is_free(start):
        raise ValueError(f"A* start cell is occupied or out of bounds: {start}.")
    if not grid.is_free(goal):
        raise ValueError(f"A* goal cell is occupied or out of bounds: {goal}.")

    use_clearance = (clearance_map is not None) and (clearance_cost_weight > 0.0)

    frontier: list[tuple[float, int, GridCell]] = [(heuristic_weight * _heuristic(start, goal), 0, start)]
    came_from: dict[GridCell, GridCell | None] = {start: None}
    cost_so_far: dict[GridCell, float] = {start: 0.0}
    closed: set[GridCell] = set()
    push_count = 1
    expanded = 0

    neighbor_steps = _neighbors(allow_diagonal)

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current in closed:
            continue
        if current == goal:
            break

        closed.add(current)
        expanded += 1

        for dx, dy, step_cost in neighbor_steps:
            nxt = (current[0] + dx, current[1] + dy)
            if not grid.is_free(nxt):
                continue
            if dx != 0 and dy != 0 and prevent_corner_cutting:
                if not (
                    grid.is_free((current[0] + dx, current[1]))
                    and grid.is_free((current[0], current[1] + dy))
                ):
                    continue

            wall_penalty = 0.0
            if use_clearance:
                c = clearance_map[nxt[0]][nxt[1]]  # type: ignore[index]
                wall_penalty = clearance_cost_weight * math.exp(-c / clearance_cost_sigma)

            new_cost = cost_so_far[current] + step_cost + wall_penalty
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                came_from[nxt] = current
                priority = new_cost + heuristic_weight * _heuristic(nxt, goal)
                heapq.heappush(frontier, (priority, push_count, nxt))
                push_count += 1

    if goal not in came_from:
        raise RuntimeError("A* failed to find a path through the occupancy grid.")

    cells: list[GridCell] = []
    current: GridCell | None = goal
    while current is not None:
        cells.append(current)
        current = came_from[current]
    cells.reverse()
    points = [grid.grid_to_world(cell) for cell in cells]
    return AStarResult(cells=cells, points=points, cost=cost_so_far[goal] * grid.resolution, expanded=expanded)


def sparsify_path_by_distance(
    points: Sequence[WorldPoint],
    spacing: float,
    *,
    keep_first: bool = True,
    keep_last: bool = True,
) -> list[WorldPoint]:
    """Downsample a metric path while preserving path order."""
    if spacing <= 0.0:
        return list(points)
    if len(points) == 0:
        return []

    sparse: list[WorldPoint] = []
    last_kept: WorldPoint | None = None
    start_index = 0
    if keep_first:
        sparse.append(points[0])
        last_kept = points[0]
        start_index = 1

    for point in points[start_index:]:
        if last_kept is None or math.hypot(point[0] - last_kept[0], point[1] - last_kept[1]) >= spacing:
            sparse.append(point)
            last_kept = point

    if keep_last and (not sparse or sparse[-1] != points[-1]):
        sparse.append(points[-1])
    return sparse


def plan_astar_path(
    grid: GridMap2D,
    start_xy: WorldPoint,
    goal_xy: WorldPoint,
    *,
    allow_diagonal: bool = True,
    prevent_corner_cutting: bool = True,
    heuristic_weight: float = 1.0,
    waypoint_spacing: float | None = None,
    clearance_cost_weight: float = 0.0,
    clearance_cost_sigma: float = 0.4,
) -> AStarResult:
    """Plan from world start/goal coordinates on any GridMap2D.

    When clearance_cost_weight > 0, computes a clearance map once before
    search so the planner naturally prefers paths away from obstacles.
    """
    start = grid.world_to_grid(start_xy)
    goal = grid.world_to_grid(goal_xy)

    clearance_map = None
    if clearance_cost_weight > 0.0:
        clearance_map = compute_clearance_map(grid)

    result = astar_search(
        grid,
        start,
        goal,
        allow_diagonal=allow_diagonal,
        prevent_corner_cutting=prevent_corner_cutting,
        heuristic_weight=heuristic_weight,
        clearance_map=clearance_map,
        clearance_cost_weight=clearance_cost_weight,
        clearance_cost_sigma=clearance_cost_sigma,
    )
    if waypoint_spacing is None:
        return result
    sparse_points = sparsify_path_by_distance(result.points, waypoint_spacing)
    point_to_idx: dict[WorldPoint, int] = {p: i for i, p in enumerate(result.points)}
    sparse_cells = [result.cells[point_to_idx[p]] for p in sparse_points]
    return AStarResult(
        cells=sparse_cells,
        points=sparse_points,
        cost=result.cost,
        expanded=result.expanded,
    )
