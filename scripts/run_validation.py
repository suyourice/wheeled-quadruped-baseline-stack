"""Depth student ablation validation runner.

Runs all 4 ablations (baseline, longhist, sparse, 4cam) on both maze and floor
sequentially in headless mode, then generates comparison plots.

Usage:
    python scripts/run_validation.py [options]

Key options:
    --out_name NAME       Sub-directory under logs/nav_play/ (default: validation_<timestamp>)
    --ablation NAME       Run only one ablation (baseline/longhist/sparse/4cam); default: all
    --maze_episodes N     Episodes per ablation for maze (default: 200)
    --floor_episodes N    Episodes per ablation for floor (default: 50)
    --seed N              Base seed; each episode gets seed+k via --seed_per_episode (default: 42)
    --stuck_timeout N     Steps without movement before forced reset (default: 300)
    --num_envs N          Parallel envs per run (default: 1)
    --dry_run             Print commands without executing
"""

import argparse
import datetime
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Ablation definitions
# ---------------------------------------------------------------------------

ABLATIONS = [
    {
        "name": "baseline",
        "maze_task": "Navigation-Depth-Hospital-Distill-Go2w-v0",
        "floor_task": "Navigation-Depth-Distill-Hospital-Floor-Go2w-Play-v0",
        "checkpoint": "logs/rsl_rl/go2w_nav_depth_hospital_distill/2026-07-06_17-47-02/model_599.pt",
    },
    {
        "name": "longhist",
        "maze_task": "Navigation-Depth-Hospital-Distill-LongHist-Go2w-v0",
        "floor_task": "Navigation-Depth-Distill-Hospital-Floor-LongHist-Go2w-Play-v0",
        "checkpoint": "logs/rsl_rl/go2w_nav_depth_hospital_longhist_distill/2026-07-06_17-47-02/model_599.pt",
    },
    {
        "name": "sparse",
        "maze_task": "Navigation-Depth-Hospital-Distill-Sparse-Go2w-v0",
        "floor_task": "Navigation-Depth-Distill-Hospital-Floor-Sparse-Go2w-Play-v0",
        "checkpoint": "logs/rsl_rl/go2w_nav_depth_hospital_sparse_distill/2026-07-06_17-47-04/model_599.pt",
    },
    {
        "name": "4cam",
        "maze_task": "Navigation-Depth-Hospital-Distill-4Cam-Go2w-v0",
        "floor_task": "Navigation-Depth-Distill-Hospital-Floor-4Cam-Go2w-Play-v0",
        "checkpoint": "logs/rsl_rl/go2w_nav_depth_hospital_multicam_distill/2026-07-06_17-47-03/model_599.pt",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PLAY_SCRIPT = os.path.join("scripts", "rsl_rl", "play.py")
PLOT_SCRIPT = os.path.join("scripts", "plot_validation.py")
DEFAULT_LLC_CHECKPOINT = "logs/rsl_rl/go2w_fast_flat/2026-04-29_18-17-48/model_1999.pt"


def build_cmd(task, checkpoint, play_name, episodes, seed, stuck_timeout, num_envs,
              locomotion_checkpoint, depth_video_steps=2000, extra_args=None):
    cmd = [
        sys.executable, PLAY_SCRIPT,
        "--task", task,
        "--checkpoint", checkpoint,
        "--locomotion_checkpoint", locomotion_checkpoint,
        "--headless",
        "--num_envs", str(num_envs),
        "--seed", str(seed),
        "--seed_per_episode",
        "--nav_eval_episodes", str(episodes),
        "--stuck_timeout_steps", str(stuck_timeout),
        "--depth_video_steps", str(depth_video_steps),
        "--play_name", play_name,
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def run_cmd(cmd, dry_run=False):
    print("\n" + "=" * 70)
    print("[run_validation] " + " ".join(cmd))
    print("=" * 70)
    if dry_run:
        return 0
    result = subprocess.run(cmd)
    return result.returncode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Depth student ablation validation")
    parser.add_argument("--out_name", type=str, default=None,
                        help="Output sub-dir under logs/nav_play/ (default: validation_<timestamp>)")
    parser.add_argument("--maze_episodes", type=int, default=200)
    parser.add_argument("--floor_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stuck_timeout", type=int, default=300,
                        help="Steps without movement before forced reset (default: 300)")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--locomotion_checkpoint", type=str, default=DEFAULT_LLC_CHECKPOINT,
                        help="Frozen LLC checkpoint path (default: go2w_fast_flat model_1999.pt)")
    parser.add_argument("--depth_video_steps", type=int, default=2000,
                        help="Max steps to record for depth video per run. 0=entire run. "
                        "Default 2000 ≈ first 2 episodes (~40s at 50Hz)")
    parser.add_argument("--ablation", type=str, default=None,
                        help="Run only this ablation (baseline/longhist/sparse/4cam). Default: all.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_plot", action="store_true",
                        help="Skip plot_validation.py after runs")
    args = parser.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = args.out_name or f"validation_{ts}"
    out_base = os.path.join("logs", "nav_play", out_name)

    ablations = ABLATIONS
    if args.ablation:
        ablations = [a for a in ABLATIONS if a["name"] == args.ablation]
        if not ablations:
            print(f"[run_validation] ERROR: unknown ablation '{args.ablation}'. "
                  f"Valid: {[a['name'] for a in ABLATIONS]}", file=sys.stderr)
            sys.exit(1)

    print(f"[run_validation] Output base: {out_base}")
    print(f"[run_validation] Ablations: {[a['name'] for a in ablations]}")
    print(f"[run_validation] Maze episodes: {args.maze_episodes}  Floor episodes: {args.floor_episodes}")
    print(f"[run_validation] Seed: {args.seed} (per-episode increment enabled)")
    print(f"[run_validation] Stuck timeout: {args.stuck_timeout} steps")
    print(f"[run_validation] Num envs: {args.num_envs}")

    failures = []

    for abl in ablations:
        name = abl["name"]
        ckpt = abl["checkpoint"]

        # Maze run
        maze_play_name = f"{out_name}/maze_{name}"
        cmd = build_cmd(
            task=abl["maze_task"],
            checkpoint=ckpt,
            play_name=maze_play_name,
            episodes=args.maze_episodes,
            seed=args.seed,
            stuck_timeout=args.stuck_timeout,
            num_envs=args.num_envs,
            locomotion_checkpoint=args.locomotion_checkpoint,
            depth_video_steps=args.depth_video_steps,
        )
        rc = run_cmd(cmd, dry_run=args.dry_run)
        if rc != 0:
            print(f"[run_validation] WARNING: maze_{name} exited with code {rc}")
            failures.append(f"maze_{name}")

        # Floor run
        floor_play_name = f"{out_name}/floor_{name}"
        cmd = build_cmd(
            task=abl["floor_task"],
            checkpoint=ckpt,
            play_name=floor_play_name,
            episodes=args.floor_episodes,
            seed=args.seed,
            stuck_timeout=args.stuck_timeout,
            num_envs=args.num_envs,
            locomotion_checkpoint=args.locomotion_checkpoint,
            depth_video_steps=args.depth_video_steps,
        )
        rc = run_cmd(cmd, dry_run=args.dry_run)
        if rc != 0:
            print(f"[run_validation] WARNING: floor_{name} exited with code {rc}")
            failures.append(f"floor_{name}")

    # Plot
    if not args.skip_plot:
        plot_cmd = [sys.executable, PLOT_SCRIPT, out_base]
        rc = run_cmd(plot_cmd, dry_run=args.dry_run)
        if rc != 0:
            print(f"[run_validation] WARNING: plot_validation.py exited with code {rc}")

    print("\n" + "=" * 70)
    if failures:
        print(f"[run_validation] DONE (with failures: {failures})")
    else:
        print("[run_validation] DONE (all runs succeeded)")
    print(f"[run_validation] Results: {out_base}")
    print("=" * 70)


if __name__ == "__main__":
    main()
