#!/usr/bin/env python3
"""
Standalone A* planner validation with debug plots.

Compares three planner variants per scenario:
  baseline   — binary A* (weight=0, no rounding)
  clearance  — cost-aware A* (weight=alpha, no rounding)
  rounded    — cost-aware A* + Bezier arc corner rounding

No Isaac Lab or GPU required.

Usage:
    python scripts/validate_astar.py [--output_dir PATH] [--alpha FLOAT] [--sigma FLOAT]
                                     [--corner_radius FLOAT] [--show]

Outputs:
    logs/astar_validation/<scenario>.png  — 6-panel debug plot per scenario
    logs/astar_validation/metrics.txt     — metrics table (also printed to stdout)

Plots per scenario:
    [0,0] Occupancy grid (free/occupied with robot inflation)
    [0,1] Clearance map (distance to nearest obstacle, metres)
    [0,2] Wall penalty map  alpha * exp(-clearance / sigma)
    [1,0] Baseline path  (weight=0)
    [1,1] Clearance-aware path  (weight=alpha)
    [1,2] Rounded path  (weight=alpha + Bezier rounding)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import importlib.util as _ilu
import sys
import types

# Load mdp modules without triggering the full Isaac Lab package chain.
# structured_corridor uses `from .astar import ...` so we register a fake
# "mdp" package and both submodules in sys.modules first.
_MDP = Path(__file__).resolve().parent.parent / "source/go2w/go2w/tasks/manager_based/go2w/mdp"

_mdp_pkg = types.ModuleType("mdp")
_mdp_pkg.__path__ = [str(_MDP)]
_mdp_pkg.__package__ = "mdp"
sys.modules.setdefault("mdp", _mdp_pkg)


def _load_mdp(stem: str):
    name = f"mdp.{stem}"
    p = _MDP / f"{stem}.py"
    spec = _ilu.spec_from_file_location(name, p, submodule_search_locations=[])
    assert spec is not None and spec.loader is not None
    mod = _ilu.module_from_spec(spec)
    mod.__package__ = "mdp"
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_astar_mod    = _load_mdp("astar")
_corridor_mod = _load_mdp("structured_corridor")

GridMap2D                      = _astar_mod.GridMap2D
AStarResult                    = _astar_mod.AStarResult
compute_clearance_map          = _astar_mod.compute_clearance_map
plan_astar_path                = _astar_mod.plan_astar_path

build_polyline_corridor_grid   = _corridor_mod.build_polyline_corridor_grid
structured_corridor_centerline = _corridor_mod.structured_corridor_centerline
_sparsify_preserve_turns       = _corridor_mod._sparsify_preserve_turns
_round_corners_safe            = _corridor_mod._round_corners_safe

WorldPoint = tuple[float, float]

# ── scenario builders ─────────────────────────────────────────────────────────

def _straight_grid(
    length: float = 10.0,
    width: float = 2.5,
    robot_inflation: float = 0.5,
    resolution: float = 0.12,
) -> tuple[GridMap2D, WorldPoint, WorldPoint]:
    cl = ((0.0, 0.0), (length, 0.0))
    return build_polyline_corridor_grid(cl, width, robot_inflation, resolution)


def _l_corridor_grid(
    leg: float = 6.0,
    width: float = 2.5,
    robot_inflation: float = 0.5,
    resolution: float = 0.12,
) -> tuple[GridMap2D, WorldPoint, WorldPoint]:
    cl = structured_corridor_centerline("l_corridor", leg, width)
    return build_polyline_corridor_grid(cl, width, robot_inflation, resolution)


def _serpentine_grid(
    leg: float = 7.0,
    width: float = 2.5,
    robot_inflation: float = 0.5,
    resolution: float = 0.14,
) -> tuple[GridMap2D, WorldPoint, WorldPoint]:
    cl = structured_corridor_centerline("serpentine_corridor", leg, width)
    return build_polyline_corridor_grid(cl, width, robot_inflation, resolution)


def _narrow_passage_grid(
    length: float = 8.0,
    width: float = 1.4,
    robot_inflation: float = 0.35,
    resolution: float = 0.08,
) -> tuple[GridMap2D, WorldPoint, WorldPoint]:
    """Narrow straight corridor — clearance cost should have visible centering effect."""
    cl = ((0.0, 0.0), (length, 0.0))
    return build_polyline_corridor_grid(cl, width, robot_inflation, resolution)


def _obstacle_corridor_grid(
    length: float = 12.0,
    width: float = 3.5,
    robot_inflation: float = 0.5,
    resolution: float = 0.12,
) -> tuple[GridMap2D, WorldPoint, WorldPoint]:
    """Wide corridor with two static box obstacles (inflation already baked in)."""
    cl = ((0.0, 0.0), (length, 0.0))
    base_grid, start, goal = build_polyline_corridor_grid(cl, width, robot_inflation, resolution)

    box_half = 0.30 + robot_inflation  # physical 0.3 m box + inflation
    obstacles: list[WorldPoint] = [(3.5, -0.5), (8.5, 0.5)]

    free = [list(col) for col in base_grid.free]
    for ox, oy in obstacles:
        gx0 = int(round((ox - box_half - base_grid.origin[0]) / base_grid.resolution))
        gx1 = int(round((ox + box_half - base_grid.origin[0]) / base_grid.resolution))
        gy0 = int(round((oy - box_half - base_grid.origin[1]) / base_grid.resolution))
        gy1 = int(round((oy + box_half - base_grid.origin[1]) / base_grid.resolution))
        for gx in range(max(0, gx0), min(base_grid.width, gx1 + 1)):
            for gy in range(max(0, gy0), min(base_grid.height, gy1 + 1)):
                free[gx][gy] = False

    return GridMap2D(free, origin=base_grid.origin, resolution=base_grid.resolution), start, goal


SCENARIOS = {
    "1_straight":          _straight_grid,
    "2_l_corridor":        _l_corridor_grid,
    "3_serpentine":        _serpentine_grid,
    "4_narrow_passage":    _narrow_passage_grid,
    "5_obstacle_corridor": _obstacle_corridor_grid,
}

# ── planning ──────────────────────────────────────────────────────────────────

class PlanResult(NamedTuple):
    success: bool
    path_raw: list[WorldPoint]
    path_sparse: list[WorldPoint]
    path_length: float
    min_clearance: float
    mean_clearance: float
    num_waypoints: int
    path_in_free: bool
    max_turn_angle: float   # max interior turn angle in sparse path (rad)
    error_msg: str


def _path_length(pts: list[WorldPoint]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts[:-1], pts[1:]))


def _path_clearances(pts: list[WorldPoint], cmap: list[list[float]], grid: GridMap2D) -> list[float]:
    out = []
    for p in pts:
        cell = grid.world_to_grid(p, clamp=True)
        v = cmap[cell[0]][cell[1]]
        out.append(v if v != math.inf else 0.0)
    return out


def _max_turn_angle(pts: list[WorldPoint]) -> float:
    """Return max absolute interior turn angle (rad) in a sparse path."""
    max_a = 0.0
    for i in range(1, len(pts) - 1):
        ab = (pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
        bc = (pts[i+1][0] - pts[i][0], pts[i+1][1] - pts[i][1])
        la = math.hypot(*ab); lb = math.hypot(*bc)
        if la < 1e-6 or lb < 1e-6:
            continue
        cos_a = (ab[0]*bc[0] + ab[1]*bc[1]) / (la * lb)
        a = math.acos(max(-1.0, min(1.0, cos_a)))
        if a > max_a:
            max_a = a
    return max_a


def run_plan(
    grid: GridMap2D,
    start: WorldPoint,
    goal: WorldPoint,
    clearance_cost_weight: float,
    clearance_cost_sigma: float,
    corner_radius: float = 0.0,
) -> PlanResult:
    try:
        result = plan_astar_path(
            grid, start, goal,
            allow_diagonal=True,
            prevent_corner_cutting=True,
            clearance_cost_weight=clearance_cost_weight,
            clearance_cost_sigma=clearance_cost_sigma,
        )
    except Exception as exc:
        return PlanResult(False, [], [], 0.0, 0.0, 0.0, 0, False, 0.0, str(exc))

    raw = result.points
    sparse = _sparsify_preserve_turns(raw, 0.35)
    if corner_radius > 0.0:
        sparse = _round_corners_safe(sparse, corner_radius, grid=grid)
    cmap = compute_clearance_map(grid)
    cl = _path_clearances(raw, cmap, grid)
    sparse_cl = _path_clearances(sparse, cmap, grid)
    min_cl = min(sparse_cl) if sparse_cl else 0.0
    mean_cl = sum(cl) / len(cl) if cl else 0.0
    return PlanResult(
        True, raw, sparse, _path_length(raw), min_cl, mean_cl,
        len(sparse), min_cl > 0.0, _max_turn_angle(sparse), "",
    )

# ── grid image helpers ────────────────────────────────────────────────────────

def _occ_image(grid: GridMap2D) -> np.ndarray:
    arr = np.zeros((grid.height, grid.width), dtype=float)
    for gx in range(grid.width):
        for gy in range(grid.height):
            arr[gy, gx] = 1.0 if grid.is_free((gx, gy)) else 0.0
    return arr


def _cl_image(cmap: list[list[float]], grid: GridMap2D) -> np.ndarray:
    arr = np.full((grid.height, grid.width), float("nan"), dtype=float)
    for gx in range(grid.width):
        for gy in range(grid.height):
            v = cmap[gx][gy]
            if v != math.inf:
                arr[gy, gx] = v
    return arr


def _penalty_image(cmap: list[list[float]], grid: GridMap2D, alpha: float, sigma: float) -> np.ndarray:
    arr = np.full((grid.height, grid.width), float("nan"), dtype=float)
    for gx in range(grid.width):
        for gy in range(grid.height):
            v = cmap[gx][gy]
            if v != math.inf and grid.is_free((gx, gy)):
                arr[gy, gx] = alpha * math.exp(-v / sigma)
    return arr


def _extent(grid: GridMap2D) -> list[float]:
    x0, y0 = grid.origin
    return [x0, x0 + (grid.width - 1) * grid.resolution,
            y0, y0 + (grid.height - 1) * grid.resolution]

# ── plotting ──────────────────────────────────────────────────────────────────

def _draw_path(ax, plan: PlanResult, raw_color: str, sparse_color: str, start: WorldPoint, goal: WorldPoint) -> None:
    if not plan.success:
        ax.text(0.5, 0.5, f"FAILED\n{plan.error_msg[:60]}", ha="center", va="center",
                transform=ax.transAxes, color="red", fontsize=8)
        return
    if plan.path_raw:
        xs, ys = zip(*plan.path_raw)
        ax.plot(xs, ys, color=raw_color, linewidth=0.8, alpha=0.45, label="raw A*")
    if plan.path_sparse:
        xs, ys = zip(*plan.path_sparse)
        ax.plot(xs, ys, color=sparse_color, linestyle="--", linewidth=1.5,
                marker="o", markersize=4, label=f"sparse ({plan.num_waypoints} wpts)")
    ax.plot(*start, "go", markersize=9, zorder=5, label="start")
    ax.plot(*goal, "r*", markersize=11, zorder=5, label="goal")
    ax.legend(fontsize=7, loc="best")


def plot_scenario(
    name: str,
    grid: GridMap2D,
    start: WorldPoint,
    goal: WorldPoint,
    baseline: PlanResult,
    clearance: PlanResult,
    rounded: PlanResult,
    alpha: float,
    sigma: float,
    out_dir: Path,
    show: bool,
) -> None:
    cmap_data = compute_clearance_map(grid)
    occ   = _occ_image(grid)
    cl    = _cl_image(cmap_data, grid)
    pen   = _penalty_image(cmap_data, grid, alpha, sigma)
    ext   = _extent(grid)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(f"Scenario: {name}   (α={alpha}, σ={sigma})", fontsize=13)

    def _bg(ax):
        ax.imshow(occ, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1, alpha=0.45)
        ax.set_xlabel("x (m)")

    # [0,0] Occupancy
    ax = axes[0, 0]
    ax.imshow(occ, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Occupancy grid  (white = free, robot-inflated)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.plot(*start, "go", markersize=9); ax.plot(*goal, "r*", markersize=11)

    # [0,1] Clearance map
    ax = axes[0, 1]
    im = ax.imshow(cl, origin="lower", extent=ext, cmap="viridis")
    plt.colorbar(im, ax=ax, label="clearance (m)")
    ax.set_title("Clearance map  (distance to nearest wall)")
    ax.set_xlabel("x (m)")

    # [0,2] Wall penalty map
    ax = axes[0, 2]
    im = ax.imshow(pen, origin="lower", extent=ext, cmap="hot_r")
    plt.colorbar(im, ax=ax, label="α·exp(−c/σ)")
    ax.set_title(f"Wall penalty map  α={alpha} σ={sigma}")
    ax.set_xlabel("x (m)")

    # [1,0] Baseline A*
    ax = axes[1, 0]
    _bg(ax)
    ax.set_title(
        f"Baseline (weight=0)  min_cl={baseline.min_clearance:.3f}m  "
        f"max_turn={math.degrees(baseline.max_turn_angle):.0f}°"
    )
    ax.set_ylabel("y (m)")
    _draw_path(ax, baseline, "#2244cc", "#0022ff", start, goal)

    # [1,1] Clearance-aware A*
    ax = axes[1, 1]
    _bg(ax)
    ax.set_title(
        f"Clearance (α={alpha})  min_cl={clearance.min_clearance:.3f}m  "
        f"max_turn={math.degrees(clearance.max_turn_angle):.0f}°"
    )
    _draw_path(ax, clearance, "#cc2222", "#ff0000", start, goal)

    # [1,2] Rounded path
    ax = axes[1, 2]
    _bg(ax)
    ax.set_title(
        f"Rounded (α={alpha})  min_cl={rounded.min_clearance:.3f}m  "
        f"max_turn={math.degrees(rounded.max_turn_angle):.0f}°"
    )
    _draw_path(ax, rounded, "#228822", "#00bb00", start, goal)

    plt.tight_layout()
    out_path = out_dir / f"{name}.png"
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"  → {out_path}")

# ── metrics table ─────────────────────────────────────────────────────────────

def metrics_table(results: dict[str, tuple[PlanResult, PlanResult, PlanResult]]) -> str:
    hdr = (
        f"{'Scenario':<26} {'Variant':<10} {'OK':>4} {'Len(m)':>8} "
        f"{'MinCl':>7} {'MeanCl':>8} {'Wpts':>5} {'MaxTurn':>9} {'ΔMinCl':>8}"
    )
    sep = "─" * len(hdr)
    lines = [hdr, sep]
    for name, (base, cl, rnd) in results.items():
        for label, r in (("baseline", base), ("clearance", cl), ("rounded", rnd)):
            ok = "YES" if r.success else " NO"
            lv = f"{r.path_length:.2f}" if r.success else "N/A"
            mc = f"{r.min_clearance:.3f}" if r.success else "N/A"
            ac = f"{r.mean_clearance:.3f}" if r.success else "N/A"
            wp = str(r.num_waypoints) if r.success else "N/A"
            mt = f"{math.degrees(r.max_turn_angle):.1f}°" if r.success else "N/A"
            delta = ""
            if label != "baseline" and base.success and r.success:
                d = r.min_clearance - base.min_clearance
                delta = f"{d:+.3f}"
            lines.append(
                f"{name:<26} {label:<10} {ok:>4} {lv:>8} {mc:>7} {ac:>8} {wp:>5} {mt:>9} {delta:>8}"
            )
        lines.append("")
    return "\n".join(lines)

# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Validate A* planner variants.")
    ap.add_argument("--output_dir", default="logs/astar_validation")
    ap.add_argument("--alpha", type=float, default=2.0, help="clearance_cost_weight")
    ap.add_argument("--sigma", type=float, default=0.4, help="clearance_cost_sigma")
    ap.add_argument("--corner_radius", type=float, default=0.5, help="Bezier arc radius for rounding")
    ap.add_argument("--show", action="store_true", help="open plots interactively")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: dict[str, tuple[PlanResult, PlanResult, PlanResult]] = {}
    print(f"\nA* validation  α={args.alpha}  σ={args.sigma}  corner_radius={args.corner_radius}  → {out}\n")

    for name, build_fn in SCENARIOS.items():
        print(f"[{name}]")
        try:
            grid, start, goal = build_fn()
        except Exception as exc:
            print(f"  grid build failed: {exc}\n")
            continue

        base = run_plan(grid, start, goal, 0.0, args.sigma)
        cl   = run_plan(grid, start, goal, args.alpha, args.sigma)
        rnd  = run_plan(grid, start, goal, args.alpha, args.sigma, corner_radius=args.corner_radius)
        results[name] = (base, cl, rnd)

        for label, r in (("baseline", base), ("clearance", cl), ("rounded", rnd)):
            if r.success:
                print(
                    f"  {label:<10}: len={r.path_length:.2f}m  min_cl={r.min_clearance:.3f}m  "
                    f"mean_cl={r.mean_clearance:.3f}m  wpts={r.num_waypoints}  "
                    f"max_turn={math.degrees(r.max_turn_angle):.1f}°"
                )
            else:
                print(f"  {label:<10}: FAIL  {r.error_msg}")

        plot_scenario(name, grid, start, goal, base, cl, rnd, args.alpha, args.sigma, out, args.show)
        print()

    table = metrics_table(results)
    print(table)
    metrics_path = out / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(
            f"A* Validation  α={args.alpha}  σ={args.sigma}  corner_radius={args.corner_radius}\n\n{table}\n"
        )
    print(f"Metrics saved → {metrics_path}")


if __name__ == "__main__":
    main()
