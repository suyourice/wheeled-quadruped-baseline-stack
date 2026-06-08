#!/usr/bin/env python3
"""
Combined navigation debug visualization.

Shows in one figure:
  Main map  : corridor occupancy grid + A* planned path + robot XY trajectory
              + active lookahead target positions (color = time)
  progress_s: arc-length progress over steps  (check: always increasing)
  fwd_b     : target forward component in body frame  (check: no large negatives = reverse)
  head_err  : path heading error in body frame  (check: small = no crab-walk)
  lookahead_dist: euclidean robot→target distance  (check: ≈1.25m, no wall-jump spikes)

Usage:
    # 1. capture debug log during play
    GO2W_NAV_DEBUG=1 GO2W_NAV_DEBUG_INTERVAL=5 \\
      python scripts/rsl_rl/play.py ... 2>&1 | tee nav_debug.log

    # 2. plot (corridor grid reconstructed from these args)
    python scripts/plot_nav_debug.py nav_debug.log \\
        --corridor serpentine_corridor --leg_length 6.0 --corridor_width 1.8

Output: nav_debug_env0.png  (same dir as log file, or --output_dir)
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import math
import re
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# ── mdp module loader (no Isaac Lab needed) ───────────────────────────────────

_MDP = Path(__file__).resolve().parent.parent / "source/go2w/go2w/tasks/manager_based/go2w/mdp"

_mdp_pkg = types.ModuleType("mdp")
_mdp_pkg.__path__ = [str(_MDP)]
_mdp_pkg.__package__ = "mdp"
sys.modules.setdefault("mdp", _mdp_pkg)


def _load_mdp(stem: str):
    name = f"mdp.{stem}"
    if name in sys.modules:
        return sys.modules[name]
    p = _MDP / f"{stem}.py"
    spec = _ilu.spec_from_file_location(name, p, submodule_search_locations=[])
    assert spec is not None and spec.loader is not None
    mod = _ilu.module_from_spec(spec)
    mod.__package__ = "mdp"
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ── log parsing ───────────────────────────────────────────────────────────────

_LINE_RE = re.compile(
    r"\[GO2W_NAV_PATH\]"
    r"\s+step=(\d+)"
    r"\s+env=(\d+)"
    r".*?"
    r"progress_s=([\d.]+)"
    r"\s+target_s=([\d.]+)"
    r".*?"
    r"target_fwd_b=([+\-\d.]+)"
    r"\s+path_head_b=([+\-\d.]+)"
    r"\s+robot=\(([+\-\d.]+),([+\-\d.]+)\)"
    r"\s+target=\(([+\-\d.]+),([+\-\d.]+)\)"
    r"\s+final=\(([+\-\d.]+),([+\-\d.]+)\)"
)


def parse_log(path: str, env_filter: int) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            m = _LINE_RE.search(line)
            if not m:
                continue
            if int(m.group(2)) != env_filter:
                continue
            records.append({
                "step":       int(m.group(1)),
                "progress_s": float(m.group(3)),
                "target_s":   float(m.group(4)),
                "fwd_b":      float(m.group(5)),
                "head_b":     float(m.group(6)),
                "robot_x":    float(m.group(7)),
                "robot_y":    float(m.group(8)),
                "target_x":   float(m.group(9)),
                "target_y":   float(m.group(10)),
                "final_x":    float(m.group(11)),
                "final_y":    float(m.group(12)),
            })
    records.sort(key=lambda r: r["step"])
    return records


# ── corridor grid + A* path reconstruction ───────────────────────────────────

def build_corridor_and_path(
    corridor_kind: str,
    leg_length: float,
    corridor_width: float,
    robot_inflation: float,
    grid_resolution: float,
    clearance_cost_weight: float,
    clearance_cost_sigma: float,
):
    """Return (occ_image, extent, astar_raw_path, astar_sparse_path) or raise."""
    _astar = _load_mdp("astar")
    _corr  = _load_mdp("structured_corridor")

    cl = _corr.structured_corridor_centerline(corridor_kind, leg_length, corridor_width)
    grid, start, goal = _corr.build_polyline_corridor_grid(cl, corridor_width, robot_inflation, grid_resolution)

    result = _astar.plan_astar_path(
        grid, start, goal,
        allow_diagonal=True,
        prevent_corner_cutting=True,
        clearance_cost_weight=clearance_cost_weight,
        clearance_cost_sigma=clearance_cost_sigma,
    )
    raw    = result.points
    sparse = _corr._sparsify_preserve_turns(raw, 0.35)

    # occupancy image
    arr = np.zeros((grid.height, grid.width), dtype=float)
    for gx in range(grid.width):
        for gy in range(grid.height):
            arr[gy, gx] = 1.0 if grid.is_free((gx, gy)) else 0.0

    x0, y0 = grid.origin
    extent = [x0, x0 + (grid.width - 1) * grid.resolution,
              y0, y0 + (grid.height - 1) * grid.resolution]

    return arr, extent, raw, sparse


# ── combined plot ─────────────────────────────────────────────────────────────

def plot_combined(
    records: list[dict],
    env_id: int,
    occ: np.ndarray | None,
    extent: list[float] | None,
    astar_raw: list | None,
    astar_sparse: list | None,
    out_path: Path,
    show: bool,
) -> None:
    steps      = [r["step"]       for r in records]
    robot_x    = [r["robot_x"]    for r in records]
    robot_y    = [r["robot_y"]    for r in records]
    target_x   = [r["target_x"]  for r in records]
    target_y   = [r["target_y"]  for r in records]
    progress_s = [r["progress_s"] for r in records]
    fwd_b      = [r["fwd_b"]      for r in records]
    head_b     = [r["head_b"]     for r in records]
    lookahead_dist = [
        math.hypot(r["target_x"] - r["robot_x"], r["target_y"] - r["robot_y"])
        for r in records
    ]
    final_x = records[0]["final_x"]
    final_y = records[0]["final_y"]

    fig = plt.figure(figsize=(20, 11))
    fig.suptitle(
        f"Navigation debug  env={env_id}  {len(records)} entries",
        fontsize=14,
    )

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        width_ratios=[2.5, 1, 1],
        hspace=0.38,
        wspace=0.32,
    )
    ax_map  = fig.add_subplot(gs[:, 0])   # spans both rows
    ax_prog = fig.add_subplot(gs[0, 1])
    ax_fwd  = fig.add_subplot(gs[1, 1])
    ax_head = fig.add_subplot(gs[0, 2])
    ax_look = fig.add_subplot(gs[1, 2])

    # ── Map ──────────────────────────────────────────────────────────────────
    if occ is not None and extent is not None:
        ax_map.imshow(occ, origin="lower", extent=extent, cmap="gray", vmin=0, vmax=1, alpha=0.40)

    # A* planned path
    if astar_raw:
        xs, ys = zip(*astar_raw)
        ax_map.plot(xs, ys, color="#888800", linewidth=1.2, alpha=0.55, label="A* path (raw)", zorder=2)
    if astar_sparse:
        xs, ys = zip(*astar_sparse)
        ax_map.plot(xs, ys, "o--", color="#aaaa00", markersize=5, linewidth=1.5,
                    alpha=0.85, label=f"A* sparse ({len(astar_sparse)} wpts)", zorder=3)

    # robot trajectory (color = time)
    sc_robot = ax_map.scatter(robot_x, robot_y, c=steps, cmap="plasma",
                               s=10, zorder=4, label="robot trajectory")
    plt.colorbar(sc_robot, ax=ax_map, label="step", fraction=0.03, pad=0.02)

    # lookahead target (color = time)
    sc_look = ax_map.scatter(target_x, target_y, c=steps, cmap="cool",
                              s=8, marker="x", linewidths=1.0, zorder=5,
                              label="lookahead target")

    ax_map.plot(robot_x[0], robot_y[0], "g^", markersize=11, zorder=6, label="start")
    ax_map.plot(final_x, final_y, "r*", markersize=13, zorder=6, label="final goal")
    ax_map.set_title("Map: A* path + robot trajectory + lookahead target\n(plasma=robot, cool=target, yellow=planned path)")
    ax_map.set_xlabel("x (m)")
    ax_map.set_ylabel("y (m)")
    ax_map.set_aspect("equal")
    ax_map.legend(fontsize=8, loc="upper left")

    # ── progress_s ───────────────────────────────────────────────────────────
    ax_prog.plot(steps, progress_s, "b-", linewidth=1.5, label="progress_s")
    ax_prog.fill_between(steps, progress_s, alpha=0.15, color="blue")
    ax_prog.set_title("progress_s\n(check: always increasing)")
    ax_prog.set_xlabel("step")
    ax_prog.set_ylabel("arc-length (m)")
    ax_prog.grid(True, alpha=0.3)

    # ── target_fwd_b ─────────────────────────────────────────────────────────
    ax_fwd.axhline(0.0, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_fwd.axhline(-0.3, color="orange", linewidth=0.8, linestyle=":", alpha=0.6, label="reverse threshold")
    neg_mask = [v < -0.1 for v in fwd_b]
    ax_fwd.plot(steps, fwd_b, color="#333333", linewidth=1.0)
    neg_steps  = [s for s, m in zip(steps, neg_mask) if m]
    neg_values = [v for v, m in zip(fwd_b, neg_mask) if m]
    if neg_steps:
        ax_fwd.scatter(neg_steps, neg_values, color="red", s=15, zorder=5, label=f"reverse ({len(neg_steps)})")
    ax_fwd.set_title("target_fwd_b\n(check: < 0 = reverse risk)")
    ax_fwd.set_xlabel("step")
    ax_fwd.set_ylabel("fwd component (m)")
    ax_fwd.legend(fontsize=7)
    ax_fwd.grid(True, alpha=0.3)

    # ── path heading error ────────────────────────────────────────────────────
    ax_head.axhline(0.0, color="gray", linewidth=0.6, alpha=0.5)
    large_mask = [abs(v) > 0.5 for v in head_b]
    ax_head.plot(steps, head_b, color="#555555", linewidth=1.0)
    lg_steps  = [s for s, m in zip(steps, large_mask) if m]
    lg_values = [v for v, m in zip(head_b, large_mask) if m]
    if lg_steps:
        ax_head.scatter(lg_steps, lg_values, color="darkorange", s=12, zorder=5,
                        label=f"|err|>0.5 ({len(lg_steps)})")
    ax_head.set_title("path_head_b (heading error)\n(check: large = crab-walk)")
    ax_head.set_xlabel("step")
    ax_head.set_ylabel("heading error (rad)")
    ax_head.legend(fontsize=7)
    ax_head.grid(True, alpha=0.3)

    # ── lookahead distance ────────────────────────────────────────────────────
    ax_look.axhline(1.25, color="green", linewidth=1.0, linestyle="--", alpha=0.7, label="nominal 1.25m")
    spike_mask = [d > 2.5 for d in lookahead_dist]
    ax_look.plot(steps, lookahead_dist, color="#222222", linewidth=1.0)
    sp_steps  = [s for s, m in zip(steps, spike_mask) if m]
    sp_values = [d for d, m in zip(lookahead_dist, spike_mask) if m]
    if sp_steps:
        ax_look.scatter(sp_steps, sp_values, color="red", s=15, zorder=5,
                        label=f"spike >2.5m ({len(sp_steps)})")
    ax_look.set_title("lookahead distance\n(check: spike = target jumped)")
    ax_look.set_xlabel("step")
    ax_look.set_ylabel("dist robot→target (m)")
    ax_look.legend(fontsize=7)
    ax_look.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"  → {out_path}")


# ── summary stats ─────────────────────────────────────────────────────────────

def print_summary(records: list[dict]) -> None:
    steps      = [r["step"]       for r in records]
    progress_s = [r["progress_s"] for r in records]
    fwd_b      = [r["fwd_b"]      for r in records]
    head_b     = [r["head_b"]     for r in records]
    lookahead_dist = [
        math.hypot(r["target_x"] - r["robot_x"], r["target_y"] - r["robot_y"])
        for r in records
    ]

    n = len(records)
    n_reverse   = sum(1 for v in fwd_b if v < -0.1)
    n_crab      = sum(1 for v in head_b if abs(v) > 0.5)
    n_spike     = sum(1 for d in lookahead_dist if d > 2.5)
    monotone    = all(a <= b for a, b in zip(progress_s[:-1], progress_s[1:]))
    final_prog  = progress_s[-1] if progress_s else 0.0

    print(f"  records       : {n}")
    print(f"  steps         : {steps[0]} → {steps[-1]}")
    print(f"  progress_s    : 0 → {final_prog:.2f}m  (monotone={monotone})")
    print(f"  reverse steps : {n_reverse}/{n}  ({100*n_reverse/max(n,1):.1f}%)")
    print(f"  crab steps    : {n_crab}/{n}  ({100*n_crab/max(n,1):.1f}%)  [|head_err|>0.5rad]")
    print(f"  target spikes : {n_spike}/{n}  ({100*n_spike/max(n,1):.1f}%)  [dist>2.5m]")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Combined navigation debug plot.")
    ap.add_argument("log", help="Path to nav_debug.log")
    ap.add_argument("--env", type=int, default=0, help="Env ID to plot (default: 0)")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--show", action="store_true")
    # corridor grid + A* reconstruction
    ap.add_argument("--corridor", default=None,
                    choices=["l_corridor", "serpentine_corridor"],
                    help="Reconstruct A* path and corridor grid for overlay")
    ap.add_argument("--leg_length", type=float, default=6.0)
    ap.add_argument("--corridor_width", type=float, default=1.8)
    ap.add_argument("--robot_inflation", type=float, default=0.50)
    ap.add_argument("--grid_resolution", type=float, default=0.14)
    ap.add_argument("--alpha", type=float, default=2.0, help="clearance_cost_weight")
    ap.add_argument("--sigma", type=float, default=0.4, help="clearance_cost_sigma")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Log not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else log_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    records = parse_log(str(log_path), env_filter=args.env)
    if not records:
        print("No [GO2W_NAV_PATH] lines found. Run with GO2W_NAV_DEBUG=1.", file=sys.stderr)
        sys.exit(1)

    print(f"\n[env {args.env}] parsed {len(records)} records from {log_path.name}")
    print_summary(records)

    occ = extent = astar_raw = astar_sparse = None
    if args.corridor:
        print(f"\nReconstructing A* ({args.corridor}, leg={args.leg_length}, width={args.corridor_width}) ...")
        try:
            occ, extent, astar_raw, astar_sparse = build_corridor_and_path(
                args.corridor, args.leg_length, args.corridor_width,
                args.robot_inflation, args.grid_resolution,
                args.alpha, args.sigma,
            )
            print(f"  A* path: {len(astar_raw)} raw, {len(astar_sparse)} sparse waypoints")
        except Exception as exc:
            print(f"  [warn] A* reconstruction failed: {exc}")

    out_path = out_dir / f"{log_path.stem}_env{args.env}.png"
    plot_combined(records, args.env, occ, extent, astar_raw, astar_sparse, out_path, args.show)
    print(f"\nDone.")


if __name__ == "__main__":
    main()
