"""Depth student ablation validation runner.

Runs all 5 policies (teacher + 4 ablations) on three maze scenarios:
  maze_train  — training distribution (16-slot scene, last curriculum phase, max 12 obs)
  maze_static — extended obstacles static  (20-slot scene, 20 obs)
  maze_dynamic— extended obstacles dynamic (20-slot scene, 20 obs moving)

Usage:
    python scripts/run_validation.py [options]

Key options:
    --out_name NAME       Sub-directory under logs/nav_play/ (default: validation_<timestamp>)
    --ablation NAME       Run only one policy (teacher/baseline/longhist/sparse/4cam); default: all
    --maze_episodes N     Episodes per policy per scenario (default: 200)
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
# Policy definitions (teacher + 4 depth student ablations)
# ---------------------------------------------------------------------------

_LLC = "logs/rsl_rl/go2w_fast_flat/2026-04-29_18-17-48/model_1999.pt"
_TEACHER_CKPT = "logs/rsl_rl/go2w_nav_hospital_teacher_rl/2026-07-03_23-18-48/model_1100.pt"

POLICIES = [
    {
        "name": "teacher",
        "maze_train_task":   "Nav-Hospital-Maze-Eval-Teacher-TrainDist-Go2w-v0",
        "maze_static_task":  "Nav-Hospital-Maze-Eval-Teacher-Static-Go2w-v0",
        "maze_dynamic_task": "Nav-Hospital-Maze-Eval-Teacher-Dynamic-Go2w-v0",
        "checkpoint": _TEACHER_CKPT,
    },
    {
        "name": "baseline",
        "maze_train_task":   "Navigation-Depth-Hospital-Maze-Eval-TrainDist-Go2w-v0",
        "maze_static_task":  "Navigation-Depth-Hospital-Maze-Eval-Static-Go2w-v0",
        "maze_dynamic_task": "Navigation-Depth-Hospital-Maze-Eval-Dynamic-Go2w-v0",
        "checkpoint": "logs/rsl_rl/go2w_nav_depth_hospital_distill/2026-07-06_17-47-02/model_599.pt",
    },
    {
        "name": "longhist",
        "maze_train_task":   "Navigation-Depth-Hospital-Maze-Eval-LongHist-TrainDist-Go2w-v0",
        "maze_static_task":  "Navigation-Depth-Hospital-Maze-Eval-LongHist-Static-Go2w-v0",
        "maze_dynamic_task": "Navigation-Depth-Hospital-Maze-Eval-LongHist-Dynamic-Go2w-v0",
        "checkpoint": "logs/rsl_rl/go2w_nav_depth_hospital_longhist_distill/2026-07-06_17-47-02/model_599.pt",
    },
    {
        "name": "sparse",
        "maze_train_task":   "Navigation-Depth-Hospital-Maze-Eval-Sparse-TrainDist-Go2w-v0",
        "maze_static_task":  "Navigation-Depth-Hospital-Maze-Eval-Sparse-Static-Go2w-v0",
        "maze_dynamic_task": "Navigation-Depth-Hospital-Maze-Eval-Sparse-Dynamic-Go2w-v0",
        "checkpoint": "logs/rsl_rl/go2w_nav_depth_hospital_sparse_distill/2026-07-06_17-47-04/model_599.pt",
    },
    {
        "name": "4cam",
        "maze_train_task":   "Navigation-Depth-Hospital-Maze-Eval-4Cam-TrainDist-Go2w-v0",
        "maze_static_task":  "Navigation-Depth-Hospital-Maze-Eval-4Cam-Static-Go2w-v0",
        "maze_dynamic_task": "Navigation-Depth-Hospital-Maze-Eval-4Cam-Dynamic-Go2w-v0",
        "checkpoint": "logs/rsl_rl/go2w_nav_depth_hospital_multicam_distill/2026-07-06_17-47-03/model_599.pt",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PLAY_SCRIPT = os.path.join("scripts", "rsl_rl", "play.py")
PLOT_SCRIPT = os.path.join("scripts", "plot_validation.py")
DEFAULT_LLC_CHECKPOINT = _LLC


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
    parser = argparse.ArgumentParser(description="Hospital maze ablation validation")
    parser.add_argument("--out_name", type=str, default=None,
                        help="Output sub-dir under logs/nav_play/ (default: validation_<timestamp>)")
    parser.add_argument("--maze_episodes", type=int, default=200,
                        help="Episodes per policy per scenario (default: 200)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stuck_timeout", type=int, default=300,
                        help="Steps without movement before forced reset (default: 300)")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--locomotion_checkpoint", type=str, default=DEFAULT_LLC_CHECKPOINT,
                        help="Frozen LLC checkpoint path (default: go2w_fast_flat model_1999.pt)")
    parser.add_argument("--depth_video_steps", type=int, default=2000,
                        help="Max steps to record for depth video per run. 0=entire run. "
                        "Default 2000 ≈ first 2 episodes (~40s at 50Hz). "
                        "Teacher has no depth camera so no video is saved regardless.")
    parser.add_argument("--ablation", type=str, default=None,
                        help="Run only this policy (teacher/baseline/longhist/sparse/4cam). Default: all.")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Run only this scenario (maze_train/maze_static/maze_dynamic). Default: all.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_plot", action="store_true",
                        help="Skip plot_validation.py after runs")
    args = parser.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = args.out_name or f"validation_{ts}"
    out_base = os.path.join("logs", "nav_play", out_name)

    policies = POLICIES
    if args.ablation:
        policies = [p for p in POLICIES if p["name"] == args.ablation]
        if not policies:
            print(f"[run_validation] ERROR: unknown policy '{args.ablation}'. "
                  f"Valid: {[p['name'] for p in POLICIES]}", file=sys.stderr)
            sys.exit(1)

    _ALL_SCENARIOS = ["maze_train", "maze_static", "maze_dynamic"]
    run_scenarios: list[str]
    if args.scenario:
        if args.scenario not in _ALL_SCENARIOS:
            print(f"[run_validation] ERROR: unknown scenario '{args.scenario}'. "
                  f"Valid: {_ALL_SCENARIOS}", file=sys.stderr)
            sys.exit(1)
        run_scenarios = [args.scenario]
    else:
        run_scenarios = _ALL_SCENARIOS

    print(f"[run_validation] Output base: {out_base}")
    print(f"[run_validation] Policies: {[p['name'] for p in policies]}")
    print(f"[run_validation] Scenarios: {run_scenarios}")
    print(f"[run_validation] Episodes per scenario: {args.maze_episodes}")
    print(f"[run_validation] Seed: {args.seed} (per-episode increment enabled)")
    print(f"[run_validation] Stuck timeout: {args.stuck_timeout} steps")
    print(f"[run_validation] Num envs: {args.num_envs}")

    _SCENARIO_TASK_KEY = {
        "maze_train":   "maze_train_task",
        "maze_static":  "maze_static_task",
        "maze_dynamic": "maze_dynamic_task",
    }

    failures = []

    for policy in policies:
        name = policy["name"]
        ckpt = policy["checkpoint"]

        for scenario in run_scenarios:
            play_name = f"{out_name}/{scenario}_{name}"
            cmd = build_cmd(
                task=policy[_SCENARIO_TASK_KEY[scenario]],
                checkpoint=ckpt,
                play_name=play_name,
                episodes=args.maze_episodes,
                seed=args.seed,
                stuck_timeout=args.stuck_timeout,
                num_envs=args.num_envs,
                locomotion_checkpoint=args.locomotion_checkpoint,
                depth_video_steps=args.depth_video_steps,
            )
            rc = run_cmd(cmd, dry_run=args.dry_run)
            if rc != 0:
                print(f"[run_validation] WARNING: {scenario}_{name} exited with code {rc}")
                failures.append(f"{scenario}_{name}")

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
