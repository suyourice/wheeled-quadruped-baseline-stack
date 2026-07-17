"""Parse session_manifest.json files from a validation run and generate comparison plots.

Usage:
    python scripts/plot_validation.py logs/nav_play/validation_<timestamp>

Accepts either layout:
    <base_dir>/<scenario>_<policy>/session_manifest.json          — single seed
    <base_dir>/seed<N>/<scenario>_<policy>/session_manifest.json  — multi-seed
                                                                    (mean ± std across seeds)

Outputs (in base_dir):
    summary.csv          — per-run metrics table (mean, plus _std columns when multi-seed)
    comparison_bar.png   — grouped bar chart: policies × scenarios (error bars = seed std)
    seed_points.png      — individual seed estimates overlaid on their means
    contact_peak_ecdf.png— empirical CDF of contact-event peak force
    safety_progress.png  — per-episode progress/contact trade-off
    termination_mix.png  — mutually exclusive episode termination composition

Scenarios:
    maze_train   — training distribution (16-slot scene, max 12 obstacles)
    maze_static  — extended obstacles static  (20-slot scene, ~20 obstacles)
    maze_dynamic — extended obstacles dynamic (20-slot scene, ~20 obstacles moving)
    maze_success — short static route used only for success/SPL measurement
"""

import argparse
import csv
import json
import math
import os
import sys


POLICY_NAMES = ["teacher", "baseline", "longhist", "sparse", "4cam"]
SCENARIO_NAMES = ["maze_train", "maze_static", "maze_dynamic"]

METRICS = [
    "success_rate",
    "spl",
    "stuck_timeout_frac",
    "avg_episode_length",
    "goals_per_episode",
    "path_progress_mean",
    "avg_obstacle_contacts_per_ep",
    "avg_low_obstacle_contacts_per_ep",
    "avg_wall_contacts_per_ep",
    "obstacle_contacts_per_path_progress_meter",
]
METRIC_LABELS = {
    "success_rate": "Success Rate",
    "spl": "SPL",
    "stuck_timeout_frac": "Stuck Fraction",
    "avg_episode_length": "Avg Episode Length (steps)",
    "goals_per_episode": "Goals / Episode",
    "path_progress_mean": "Cumulative Nav Distance (m)",
    "avg_obstacle_contacts_per_ep": "Obstacle Contacts / Episode",
    "avg_low_obstacle_contacts_per_ep": "Low-Obs Contacts / Episode",
    "avg_wall_contacts_per_ep": "Wall Contacts / Episode",
    "obstacle_contacts_per_path_progress_meter": "Obstacle Contacts / Path-progress m",
}
# Pre-2026-07-17 manifests used "collision" naming; fall back so old runs still plot.
LEGACY_KEYS = {
    "avg_obstacle_contacts_per_ep": "avg_obstacle_collisions_per_ep",
    "avg_low_obstacle_contacts_per_ep": "avg_low_obstacle_collisions_per_ep",
    "avg_wall_contacts_per_ep": "avg_wall_collisions_per_ep",
    "obstacle_contacts_per_path_progress_meter": "obstacle_contacts_per_meter",
}


def load_manifest(run_dir, scenario, policy):
    path = os.path.join(run_dir, f"{scenario}_{policy}", "session_manifest.json")
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
    out = {
        "success_rate": manifest.get("success_rate"),
        "spl": manifest.get("spl"),
        "stuck_timeout_frac": tc.get("stuck_timeout", 0) / n,
        "avg_episode_length": manifest.get("avg_episode_length"),
        "goals_per_episode": manifest.get("goals_per_episode"),
        "path_progress_mean": manifest.get("path_progress_mean"),
    }
    for key in METRICS:
        if key in out:
            continue
        value = manifest.get(key)
        if value is None and key in LEGACY_KEYS:
            value = manifest.get(LEGACY_KEYS[key])
        out[key] = value
    return out


def aggregate_metrics(metric_dicts):
    """Mean and std per metric across seeds, ignoring None entries."""
    mean, std = {}, {}
    for metric in METRICS:
        vals = [d[metric] for d in metric_dicts if d.get(metric) is not None]
        if not vals:
            mean[metric], std[metric] = None, None
            continue
        mu = sum(vals) / len(vals)
        mean[metric] = mu
        std[metric] = (
            math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else None
        )
    return mean, std


def write_csv(rows, multi_seed, out_path):
    fieldnames = ["scenario", "policy"] + METRICS
    if multi_seed:
        fieldnames += [f"{m}_std" for m in METRICS]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[plot_validation] CSV saved: {out_path}")


def plot_bars(data, out_path, scenarios):
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
        "path_progress_mean",
        "obstacle_contacts_per_path_progress_meter",
        "avg_low_obstacle_contacts_per_ep",
        "stuck_timeout_frac",
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
        "maze_success": "#C44E52",
    }

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, plot_metrics):
        all_vals = []
        for si, scenario in enumerate(scenarios):
            vals, errs = [], []
            for policy in POLICY_NAMES:
                mean, std = data.get((scenario, policy), ({}, {}))
                v = mean.get(metric)
                e = std.get(metric)
                vals.append(v if v is not None else 0.0)
                errs.append(e if e is not None else 0.0)
            offset = (si - (len(scenarios) - 1) / 2) * bar_width
            yerr = errs if any(e > 0 for e in errs) else None
            bars = ax.bar(x + offset, vals, bar_width,
                          label=scenario.replace("_", " ").capitalize(),
                          color=scenario_colors[scenario], alpha=0.85,
                          yerr=yerr, capsize=2)
            fmt = ".2f" if metric in fixed_ylim_metrics else ".2f"
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


def _load_csv_rows(path, **labels):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row.update(labels)
    return rows


def _float(row, key):
    try:
        value = row.get(key)
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _plot_imports():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        return plt, np
    except ImportError:
        print("[plot_validation] matplotlib not available — skipping plot", file=sys.stderr)
        return None, None


def plot_seed_points(per_seed_data, out_path, scenarios):
    """Show each independent seed, rather than only a mean±std bar."""
    plt, np = _plot_imports()
    if plt is None:
        return
    metrics = ["success_rate", "spl", "obstacle_contacts_per_path_progress_meter", "stuck_timeout_frac"]
    fig, axes = plt.subplots(len(scenarios), len(metrics), figsize=(4 * len(metrics), 3 * len(scenarios)), squeeze=False)
    for row_i, scenario in enumerate(scenarios):
        for col_i, metric in enumerate(metrics):
            ax = axes[row_i, col_i]
            any_values = False
            for policy_i, policy in enumerate(POLICY_NAMES):
                vals = [extract_metrics(m).get(metric) for m in per_seed_data.get((scenario, policy), [])]
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue
                any_values = True
                jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else [0.0]
                ax.scatter(np.full(len(vals), policy_i) + jitter, vals, color="#333333", s=24, zorder=3)
                ax.scatter([policy_i], [sum(vals) / len(vals)], color="#C44E52", marker="_", s=360, linewidths=2, zorder=4)
            ax.set_title(METRIC_LABELS[metric], fontsize=9)
            ax.set_xticks(range(len(POLICY_NAMES)))
            ax.set_xticklabels(POLICY_NAMES, rotation=20, ha="right", fontsize=8)
            if metric in {"success_rate", "spl", "stuck_timeout_frac"}:
                ax.set_ylim(0, 1.05)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            if col_i == 0:
                ax.set_ylabel(scenario.replace("_", " "))
            if not any_values:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    fig.suptitle("Independent evaluation seeds (dot) and mean (red bar)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot_validation] Seed plot saved: {out_path}")


def plot_contact_peak_ecdf(contact_rows, out_path, scenarios):
    plt, _ = _plot_imports()
    if plt is None or not contact_rows:
        return
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.2 * len(scenarios), 3.8), squeeze=False)
    for ax, scenario in zip(axes[0], scenarios):
        for policy in POLICY_NAMES:
            vals = sorted(
                value for row in contact_rows
                if row["scenario"] == scenario and row["policy"] == policy and row.get("kind") == "obstacle"
                for value in [_float(row, "peak_force_n")] if value is not None
            )
            if vals:
                ax.step(vals, [(i + 1) / len(vals) for i in range(len(vals))], where="post", label=policy)
        ax.set_title(scenario.replace("_", " "))
        ax.set_xlabel("Obstacle-event peak force (N)")
        ax.set_ylabel("Empirical CDF")
        ax.set_ylim(0, 1.02)
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(fontsize=7)
    fig.suptitle("Contact-event peak-force distribution (no preset graze/collision cutoff)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot_validation] Contact ECDF saved: {out_path}")


def plot_safety_progress(episode_rows, out_path, scenarios):
    plt, _ = _plot_imports()
    if plt is None or not episode_rows:
        return
    colors = {policy: color for policy, color in zip(POLICY_NAMES, ["#4C72B0", "#DD8452", "#55A868", "#8172B2", "#C44E52"])}
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.2 * len(scenarios), 3.8), squeeze=False)
    for ax, scenario in zip(axes[0], scenarios):
        for policy in POLICY_NAMES:
            points = []
            for row in episode_rows:
                if row["scenario"] != scenario or row["policy"] != policy:
                    continue
                progress, contacts = _float(row, "path_progress_m"), _float(row, "obstacle_contact_events")
                if progress is not None and contacts is not None:
                    points.append((progress, contacts))
            if points:
                x, y = zip(*points)
                ax.scatter(x, y, s=12, alpha=0.32, color=colors[policy], label=policy)
        ax.set_title(scenario.replace("_", " "))
        ax.set_xlabel("Route progress (m)")
        ax.set_ylabel("Obstacle contact events / episode")
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(fontsize=7)
    fig.suptitle("Episode-level safety/progress trade-off", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot_validation] Safety/progress plot saved: {out_path}")


def plot_termination_mix(episode_rows, out_path, scenarios):
    plt, np = _plot_imports()
    if plt is None or not episode_rows:
        return
    categories = ["success", "stuck_timeout", "obstacle_contact", "time_out", "other"]
    colors = {"success": "#55A868", "stuck_timeout": "#C44E52", "obstacle_contact": "#DD8452", "time_out": "#8172B2", "other": "#999999"}
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.2 * len(scenarios), 3.8), squeeze=False)
    for ax, scenario in zip(axes[0], scenarios):
        bottoms = np.zeros(len(POLICY_NAMES))
        for category in categories:
            vals = []
            for policy in POLICY_NAMES:
                rows = [row for row in episode_rows if row["scenario"] == scenario and row["policy"] == policy]
                n = len(rows)
                count = 0
                for row in rows:
                    term = row.get("termination", "")
                    success = row.get("success") == "1"
                    primary = "success" if success else (
                        "stuck_timeout" if "stuck_timeout" in term else
                        "obstacle_contact" if "obstacle_contact" in term else
                        "time_out" if "time_out" in term else "other"
                    )
                    count += primary == category
                vals.append(count / n if n else 0.0)
            ax.bar(POLICY_NAMES, vals, bottom=bottoms, color=colors[category], label=category)
            bottoms += np.asarray(vals)
        ax.set_title(scenario.replace("_", " "))
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=7)
    fig.suptitle("Mutually exclusive episode outcome composition", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot_validation] Termination plot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", help="Validation output dir (or parent of seed<N>/ sub-dirs)")
    parser.add_argument("--scenarios", nargs="+", choices=[*SCENARIO_NAMES, "maze_success"],
                        default=SCENARIO_NAMES,
                        help="Scenarios present in this run (default: historical three-scenario protocol).")
    parser.add_argument("--out_prefix", type=str, default="",
                        help="Prefix for output filenames, e.g. 'success_' -> success_summary.csv. "
                        "Required when plotting maze_success separately from the long-horizon "
                        "scenarios against the same base_dir, since they measure different things "
                        "(single-route success vs. sustained multi-route progress) and must not "
                        "share axes or overwrite each other's files.")
    args = parser.parse_args()
    if "maze_success" in args.scenarios and len(args.scenarios) > 1 and not args.out_prefix:
        parser.error(
            "--scenarios includes maze_success alongside long-horizon scenarios; "
            "pass --out_prefix (or plot maze_success separately) so path_progress_mean etc. "
            "are not shown on the same axes as single-route success runs."
        )

    base_dir = args.base_dir
    if not os.path.isdir(base_dir):
        print(f"[plot_validation] ERROR: directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)

    seed_dirs = sorted(
        d for d in os.listdir(base_dir)
        if d.startswith("seed") and os.path.isdir(os.path.join(base_dir, d))
    )
    run_dirs = [os.path.join(base_dir, d) for d in seed_dirs] if seed_dirs else [base_dir]
    multi_seed = len(run_dirs) > 1
    if seed_dirs:
        print(f"[plot_validation] Aggregating {len(run_dirs)} seed dirs: {seed_dirs}")

    rows = []
    data = {}
    per_seed_data = {}
    episode_rows = []
    contact_rows = []
    for scenario in args.scenarios:
        for policy in POLICY_NAMES:
            manifests = [load_manifest(run_dir, scenario, policy) for run_dir in run_dirs]
            per_seed_data[(scenario, policy)] = manifests
            per_seed = [extract_metrics(manifest) for manifest in manifests]
            mean, std = aggregate_metrics(per_seed)
            data[(scenario, policy)] = (mean, std)
            row = {"scenario": scenario, "policy": policy}
            row.update({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in mean.items()})
            if multi_seed:
                row.update({
                    f"{k}_std": (f"{v:.4f}" if isinstance(v, float) else v) for k, v in std.items()
                })
            rows.append(row)
            for seed_index, run_dir in enumerate(run_dirs):
                labels = {
                    "scenario": scenario,
                    "policy": policy,
                    "seed_dir": os.path.basename(run_dir) if seed_dirs else "single_seed",
                    "seed_index": str(seed_index),
                }
                run_path = os.path.join(run_dir, f"{scenario}_{policy}")
                episode_rows.extend(_load_csv_rows(os.path.join(run_path, "episode_metrics.csv"), **labels))
                contact_rows.extend(_load_csv_rows(os.path.join(run_path, "contact_events.csv"), **labels))

    write_csv(rows, multi_seed, os.path.join(base_dir, f"{args.out_prefix}summary.csv"))

    # Print table
    header = (f"{'Scenario':<14} {'Policy':<12} {'Goals/Ep':>8} {'Obs/Prog-m':>10} "
              f"{'LowObs/Ep':>10} {'Stuck':>6} {'Progress':>9}")
    print(f"\n{header}")
    print("-" * 88)
    for row in rows:
        mean, _ = data[(row["scenario"], row["policy"])]
        goals  = mean["goals_per_episode"] or 0
        obs_m  = mean["obstacle_contacts_per_path_progress_meter"]
        low_c  = mean["avg_low_obstacle_contacts_per_ep"]
        stuck  = mean["stuck_timeout_frac"] or 0
        prog   = mean["path_progress_mean"]
        obs_str  = f"{obs_m:>7.3f}"  if obs_m  is not None else f"{'N/A':>7}"
        low_str  = f"{low_c:>10.2f}" if low_c  is not None else f"{'N/A':>10}"
        prog_str = f"{prog:>9.1f}"   if prog   is not None else f"{'N/A':>9}"
        print(f"{row['scenario']:<14} {row['policy']:<12} "
              f"{goals:>8.2f} {obs_str} {low_str} {stuck:>6.3f} {prog_str}")

    p = args.out_prefix
    plot_bars(data, os.path.join(base_dir, f"{p}comparison_bar.png"), args.scenarios)
    plot_seed_points(per_seed_data, os.path.join(base_dir, f"{p}seed_points.png"), args.scenarios)
    if contact_rows:
        plot_contact_peak_ecdf(contact_rows, os.path.join(base_dir, f"{p}contact_peak_ecdf.png"), args.scenarios)
    else:
        print("[plot_validation] No contact_events.csv found; skipping contact ECDF.")
    if episode_rows:
        plot_safety_progress(episode_rows, os.path.join(base_dir, f"{p}safety_progress.png"), args.scenarios)
        plot_termination_mix(episode_rows, os.path.join(base_dir, f"{p}termination_mix.png"), args.scenarios)
    else:
        print("[plot_validation] No episode_metrics.csv found; skipping episode-level plots.")


if __name__ == "__main__":
    main()
