"""Parse session_manifest.json files from a validation run and generate comparison plots.

Usage:
    python scripts/plot_validation.py logs/nav_play/validation_<timestamp>

Outputs (in the same directory):
    summary.csv          — per-run metrics table
    comparison_bar.png   — grouped bar chart: 5 policies × 3 scenarios

Scenarios:
    maze_train   — training distribution (16-slot scene, max 12 obstacles)
    maze_static  — extended obstacles static  (20-slot scene, ~20 obstacles)
    maze_dynamic — extended obstacles dynamic (20-slot scene, ~20 obstacles moving)
"""

import argparse
import csv
import json
import os
import sys


POLICY_NAMES = ["teacher", "baseline", "longhist", "sparse", "4cam"]
SCENARIO_NAMES = ["maze_train", "maze_static", "maze_dynamic"]

METRICS = [
    "stuck_timeout_frac",
    "avg_episode_length",
    "goals_per_episode",
    "path_progress_mean",
    "avg_obstacle_collisions_per_ep",
    "avg_low_obstacle_collisions_per_ep",
    "avg_wall_collisions_per_ep",
]
METRIC_LABELS = {
    "stuck_timeout_frac": "Stuck Fraction",
    "avg_episode_length": "Avg Episode Length (steps)",
    "goals_per_episode": "Goals / Episode",
    "path_progress_mean": "Cumulative Nav Distance (m)",
    "avg_obstacle_collisions_per_ep": "Obstacle Collisions / Episode",
    "avg_low_obstacle_collisions_per_ep": "Low-Obs Collisions / Episode",
    "avg_wall_collisions_per_ep": "Wall Collisions / Episode",
}


def load_manifest(base_dir, scenario, policy):
    path = os.path.join(base_dir, f"{scenario}_{policy}", "session_manifest.json")
    if not os.path.exists(path):
        print(f"[plot_validation] WARNING: missing {path}", file=sys.stderr)
        return None
    with open(path) as f:
        return json.load(f)


def extract_metrics(manifest):
    if manifest is None:
        return {m: None for m in METRICS}
    n = max(manifest.get("completed_episodes", 0), 1)
    tc = manifest.get("termination_counts", {})
    stuck = tc.get("stuck_timeout", 0)
    return {
        "stuck_timeout_frac": stuck / n,
        "avg_episode_length": manifest.get("avg_episode_length"),
        "goals_per_episode": manifest.get("goals_per_episode"),
        "path_progress_mean": manifest.get("path_progress_mean"),
        "avg_obstacle_collisions_per_ep": manifest.get("avg_obstacle_collisions_per_ep"),
        "avg_low_obstacle_collisions_per_ep": manifest.get("avg_low_obstacle_collisions_per_ep"),
        "avg_wall_collisions_per_ep": manifest.get("avg_wall_collisions_per_ep"),
    }


def write_csv(rows, out_path):
    fieldnames = ["scenario", "policy"] + METRICS
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[plot_validation] CSV saved: {out_path}")


def plot_bars(data, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[plot_validation] matplotlib not available — skipping plot", file=sys.stderr)
        return

    plot_metrics = [
        "goals_per_episode",
        "avg_obstacle_collisions_per_ep",
        "avg_low_obstacle_collisions_per_ep",
        "avg_wall_collisions_per_ep",
        "stuck_timeout_frac",
        "path_progress_mean",
    ]
    fixed_ylim_metrics = {"stuck_timeout_frac"}
    n_metrics = len(plot_metrics)
    n_policies = len(POLICY_NAMES)
    bar_width = 0.18
    x = np.arange(n_policies)

    scenario_colors = {
        "maze_train": "#DD8452",
        "maze_static": "#4C72B0",
        "maze_dynamic": "#55A868",
    }

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, plot_metrics):
        all_vals = []
        for si, scenario in enumerate(SCENARIO_NAMES):
            vals = []
            for policy in POLICY_NAMES:
                v = data.get((scenario, policy), {}).get(metric)
                vals.append(v if v is not None else 0.0)
            offset = (si - 1) * bar_width
            bars = ax.bar(x + offset, vals, bar_width,
                          label=scenario.replace("_", " ").capitalize(),
                          color=scenario_colors[scenario], alpha=0.85)
            fmt = ".2f" if metric in fixed_ylim_metrics else ".1f"
            for bar, val in zip(bars, vals):
                if val is not None and val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{val:{fmt}}", ha="center", va="bottom", fontsize=7)
            all_vals.extend(vals)

        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(POLICY_NAMES, rotation=20, ha="right", fontsize=8)
        if metric in fixed_ylim_metrics:
            ax.set_ylim(0, 1.15)
        else:
            ax.set_ylim(0, max(all_vals) * 1.2 if any(v > 0 for v in all_vals) else 1)
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Hospital Maze Ablation Validation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot_validation] Bar chart saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", help="Validation output dir (contains maze_train_*/maze_static_*/maze_dynamic_* sub-dirs)")
    args = parser.parse_args()

    base_dir = args.base_dir
    if not os.path.isdir(base_dir):
        print(f"[plot_validation] ERROR: directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    data = {}
    for scenario in SCENARIO_NAMES:
        for policy in POLICY_NAMES:
            manifest = load_manifest(base_dir, scenario, policy)
            metrics = extract_metrics(manifest)
            data[(scenario, policy)] = metrics
            row = {"scenario": scenario, "policy": policy}
            row.update({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in metrics.items()})
            rows.append(row)

    write_csv(rows, os.path.join(base_dir, "summary.csv"))

    # Print table
    header = (f"{'Scenario':<14} {'Policy':<12} {'Goals/Ep':>8} {'Obs/Ep':>7} "
              f"{'LowObs/Ep':>10} {'Wall/Ep':>8} {'Stuck':>6} {'Progress':>9}")
    print(f"\n{header}")
    print("-" * 88)
    for row in rows:
        m = data[(row["scenario"], row["policy"])]
        goals  = m["goals_per_episode"] or 0
        obs_c  = m["avg_obstacle_collisions_per_ep"]
        low_c  = m["avg_low_obstacle_collisions_per_ep"]
        wall_c = m["avg_wall_collisions_per_ep"]
        stuck  = m["stuck_timeout_frac"] or 0
        prog   = m["path_progress_mean"]
        obs_str  = f"{obs_c:>7.2f}"  if obs_c  is not None else f"{'N/A':>7}"
        low_str  = f"{low_c:>10.2f}" if low_c  is not None else f"{'N/A':>10}"
        wall_str = f"{wall_c:>8.2f}" if wall_c is not None else f"{'N/A':>8}"
        prog_str = f"{prog:>9.1f}"   if prog   is not None else f"{'N/A':>9}"
        print(f"{row['scenario']:<14} {row['policy']:<12} "
              f"{goals:>8.2f} {obs_str} {low_str} {wall_str} {stuck:>6.3f} {prog_str}")

    plot_bars(data, os.path.join(base_dir, "comparison_bar.png"))


if __name__ == "__main__":
    main()
