"""Parse session_manifest.json files from a validation run and generate comparison plots.

Usage:
    python scripts/plot_validation.py logs/nav_play/validation_<timestamp>

Outputs (in the same directory):
    summary.csv          — per-run metrics table
    comparison_bar.png   — grouped bar chart: 4 ablations × maze/floor
"""

import argparse
import csv
import json
import os
import sys


ABLATION_NAMES = ["baseline", "longhist", "sparse", "4cam"]
SCENARIO_NAMES = ["maze", "floor"]

METRICS = ["success_rate", "spl", "stuck_timeout_frac", "avg_episode_length",
           "goals_per_episode", "path_progress_mean"]
METRIC_LABELS = {
    "success_rate": "Success Rate",
    "spl": "SPL",
    "stuck_timeout_frac": "Stuck Timeout Fraction",
    "avg_episode_length": "Avg Episode Length (steps)",
    "goals_per_episode": "Goals per Episode",
    "path_progress_mean": "Path Progress (m)",
}


def load_manifest(base_dir, scenario, ablation):
    path = os.path.join(base_dir, f"{scenario}_{ablation}", "session_manifest.json")
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
        "success_rate": manifest.get("success_rate"),
        "spl": manifest.get("spl"),
        "stuck_timeout_frac": stuck / n,
        "avg_episode_length": manifest.get("avg_episode_length"),
        "goals_per_episode": manifest.get("goals_per_episode"),
        "path_progress_mean": manifest.get("path_progress_mean"),
    }


def write_csv(rows, out_path):
    fieldnames = ["scenario", "ablation"] + METRICS
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

    # maze: success_rate/spl are meaningless; path_progress_mean and stuck_timeout_frac matter
    # floor: success_rate/spl/stuck are the key metrics
    plot_metrics = ["success_rate", "spl", "stuck_timeout_frac", "path_progress_mean"]
    fixed_ylim_metrics = {"success_rate", "spl", "stuck_timeout_frac"}
    n_metrics = len(plot_metrics)
    n_ablations = len(ABLATION_NAMES)
    bar_width = 0.35
    x = np.arange(n_ablations)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors = {"maze": "#4C72B0", "floor": "#DD8452"}

    for ax, metric in zip(axes, plot_metrics):
        all_vals = []
        for si, scenario in enumerate(SCENARIO_NAMES):
            vals = []
            for abl in ABLATION_NAMES:
                v = data.get((scenario, abl), {}).get(metric)
                vals.append(v if v is not None else 0.0)
            offset = (si - 0.5) * bar_width
            bars = ax.bar(x + offset, vals, bar_width, label=scenario.capitalize(),
                          color=colors[scenario], alpha=0.85)
            fmt = ".2f" if metric in fixed_ylim_metrics else ".1f"
            for bar, val in zip(bars, vals):
                if val is not None:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{val:{fmt}}", ha="center", va="bottom", fontsize=8)
            all_vals.extend(vals)

        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_xticks(x)
        ax.set_xticklabels(ABLATION_NAMES, rotation=15)
        if metric in fixed_ylim_metrics:
            ax.set_ylim(0, 1.15)
        else:
            ax.set_ylim(0, max(all_vals) * 1.2 if any(v > 0 for v in all_vals) else 1)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Depth Student Ablation Validation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot_validation] Bar chart saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", help="Validation output directory (contains maze_*/floor_* sub-dirs)")
    args = parser.parse_args()

    base_dir = args.base_dir
    if not os.path.isdir(base_dir):
        print(f"[plot_validation] ERROR: directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    data = {}
    for scenario in SCENARIO_NAMES:
        for ablation in ABLATION_NAMES:
            manifest = load_manifest(base_dir, scenario, ablation)
            metrics = extract_metrics(manifest)
            data[(scenario, ablation)] = metrics
            row = {"scenario": scenario, "ablation": ablation}
            row.update({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in metrics.items()})
            rows.append(row)

    write_csv(rows, os.path.join(base_dir, "summary.csv"))

    # Print table
    print(f"\n{'Scenario':<8} {'Ablation':<12} {'SuccRate':>8} {'SPL':>6} {'StuckFrac':>10} {'AvgEpLen':>10} {'Progress':>9}")
    print("-" * 70)
    for row in rows:
        m = data[(row["scenario"], row["ablation"])]
        sr = m["success_rate"] or 0
        spl = m["spl"] or 0
        stuck = m["stuck_timeout_frac"] or 0
        ep = m["avg_episode_length"] or 0
        prog = m["path_progress_mean"]
        prog_str = f"{prog:>9.1f}" if prog is not None else f"{'N/A':>9}"
        print(f"{row['scenario']:<8} {row['ablation']:<12} "
              f"{sr:>8.3f} {spl:>6.3f} {stuck:>10.3f} {ep:>10.1f} {prog_str}")

    plot_bars(data, os.path.join(base_dir, "comparison_bar.png"))


if __name__ == "__main__":
    main()
