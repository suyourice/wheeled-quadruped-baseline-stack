"""Depth student ablation validation runner.

Runs all 5 policies (teacher + 4 ablations) on four maze scenarios:
  maze_train  — training distribution (16-slot scene, last curriculum phase, max 12 obs).
                Long-horizon: never terminates on goal reached, same as training —
                goal_reached_and_resample keeps sampling new routes for the full episode.
  maze_static — extended obstacles static  (20-slot scene, 20 obs). Long-horizon (see above).
  maze_dynamic— extended obstacles dynamic (20-slot scene, 20 obs moving). Long-horizon (see above).
  maze_success— short static route (20-slot scene, 2-3 junctions). The ONLY scenario that
                terminates on first route completion (--terminate_on_final_goal), so it is
                the only one with meaningful success_rate/SPL. Do not compare its
                path_progress_mean/avg_episode_length against the long-horizon scenarios —
                they measure different things (one route vs. sustained multi-route exposure).

Usage:
    python scripts/run_validation.py [options]

Key options:
    --config PATH         Validation YAML (default: scripts/configs/validation.yaml)
    --out_name NAME       Sub-directory under logs/nav_play/ (default: validation_<timestamp>)
    --ablation NAME       Run only one policy (teacher/baseline/longhist/sparse/4cam); default: all
    --maze_episodes N     Completion-order-independent trajectories per policy/scenario (default: 100)
    --seed N              Run only this seed; otherwise YAML evaluation_seeds are used.
    --stuck_timeout N     Steps without movement before forced reset (default: 300)
    --num_envs N          Parallel envs per run (default: 1)
    --dry_run             Print commands without executing
"""

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PLAY_SCRIPT = os.path.join("scripts", "rsl_rl", "play.py")
PLOT_SCRIPT = os.path.join("scripts", "plot_validation.py")
PREFLIGHT_SCRIPT = os.path.join("scripts", "preflight_validation.py")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "configs" / "validation.yaml"

_DEFAULT_KEYS = (
    "maze_episodes", "stuck_timeout", "num_envs", "depth_video_steps",
    "success_route_min_steps", "success_route_max_steps",
)
_CONFIG_DEFAULT_KEYS = _DEFAULT_KEYS + ("evaluation_seeds",)
_SCENARIO_TASK_KEYS = {
    "maze_train": "maze_train_task",
    "static": "maze_static_task",
    "dynamic": "maze_dynamic_task",
}


def resolve_config_path(config_arg: str | None) -> Path:
    path = Path(config_arg).expanduser() if config_arg else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def _require_mapping(value, label: str, required_keys: tuple[str, ...]) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    missing = [key for key in required_keys if key not in value]
    if missing:
        raise ValueError(f"{label} is missing keys: {missing}")
    return value


def load_validation_config(path: Path) -> tuple[str, dict, list[dict[str, str]]]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Validation config not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid validation YAML in {path}: {exc}") from exc

    data = _require_mapping(data, "Validation config", ("llc_checkpoint", "defaults", "policies"))

    llc_checkpoint = data["llc_checkpoint"]
    if not isinstance(llc_checkpoint, str) or not llc_checkpoint:
        raise ValueError("Validation config 'llc_checkpoint' must be a non-empty string.")

    defaults = _require_mapping(data["defaults"], "Validation config defaults", _CONFIG_DEFAULT_KEYS)
    invalid_defaults = [
        key for key in _DEFAULT_KEYS if isinstance(defaults[key], bool) or not isinstance(defaults[key], int)
    ]
    if invalid_defaults:
        raise ValueError(f"Validation defaults must be integers: {invalid_defaults}")
    evaluation_seeds = defaults["evaluation_seeds"]
    if (
        not isinstance(evaluation_seeds, list)
        or not evaluation_seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in evaluation_seeds)
        or len(set(evaluation_seeds)) != len(evaluation_seeds)
    ):
        raise ValueError("Validation default 'evaluation_seeds' must be a non-empty list of unique integers.")
    validated_defaults = {key: defaults[key] for key in _DEFAULT_KEYS}
    validated_defaults["evaluation_seeds"] = list(evaluation_seeds)

    raw_policies = data["policies"]
    if not isinstance(raw_policies, list) or not raw_policies:
        raise ValueError("Validation config 'policies' must be a non-empty list.")

    policies = []
    policy_names = set()
    for index, raw_policy in enumerate(raw_policies):
        label = f"Validation policy at index {index}"
        raw_policy = _require_mapping(raw_policy, label, ("name", "checkpoint", "tasks"))
        tasks = _require_mapping(raw_policy["tasks"], f"{label} tasks", tuple(_SCENARIO_TASK_KEYS))
        string_values = {
            "name": raw_policy["name"],
            "checkpoint": raw_policy["checkpoint"],
            **{key: tasks[key] for key in _SCENARIO_TASK_KEYS},
        }
        invalid_strings = [key for key, value in string_values.items() if not isinstance(value, str) or not value]
        if invalid_strings:
            raise ValueError(f"{label} must have non-empty string values: {invalid_strings}")

        name = string_values["name"]
        if name in policy_names:
            raise ValueError(f"Duplicate validation policy name: {name}")

        policy = {"name": name, "checkpoint": string_values["checkpoint"]}
        policy.update({runtime_key: tasks[config_key] for config_key, runtime_key in _SCENARIO_TASK_KEYS.items()})
        policy_names.add(name)
        policies.append(policy)

    return llc_checkpoint, validated_defaults, policies


def build_cmd(task, checkpoint, play_name, episodes, seed, stuck_timeout, num_envs,
              locomotion_checkpoint, depth_video_steps=2000, maze_route_steps=None,
              terminate_on_final_goal=False, extra_args=None):
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
    if maze_route_steps is not None:
        cmd.extend(["--hospital_maze_route_steps", str(maze_route_steps[0]), str(maze_route_steps[1])])
    if terminate_on_final_goal:
        # Only maze_success wants single-route termination — see play.py's
        # --terminate_on_final_goal docstring for why the long-horizon
        # scenarios (train/static/dynamic) must never pass this.
        cmd.append("--terminate_on_final_goal")
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
    parser.add_argument("--config", type=str, default=None,
                        help="Validation YAML path (relative paths are resolved from this script)")
    parser.add_argument("--out_name", type=str, default=None,
                        help="Output sub-dir under logs/nav_play/ (default: validation_<timestamp>)")
    parser.add_argument("--maze_episodes", type=int, default=None,
                        help="Trajectories per policy per scenario (default: 100). Each admitted "
                             "trajectory is run to its own terminal event; completion order cannot "
                             "truncate survivors.")
    parser.add_argument("--maze_route_steps", type=int, nargs=2, default=None, metavar=("MIN", "MAX"),
                        help="Override the hospital-maze route range for every selected scenario. "
                             "By default, train/static/dynamic retain their task-native route distribution; "
                             "maze_success uses the short range in validation YAML.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Run the full policy×scenario grid once per seed "
                        "(outputs under <out_name>/seed<N>/); overrides --seed.")
    parser.add_argument("--stuck_timeout", type=int, default=None,
                        help="Steps without movement before forced reset (default: 300)")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--locomotion_checkpoint", type=str, default=None,
                        help="Frozen LLC checkpoint path (default: go2w_fast_flat model_1999.pt)")
    parser.add_argument("--depth_video_steps", type=int, default=None,
                        help="Max steps to record for depth video per run. 0=entire run. "
                        "Default 2000 ≈ first 2 episodes (~40s at 50Hz). "
                        "Teacher has no depth camera so no video is saved regardless.")
    parser.add_argument("--ablation", type=str, default=None,
                        help="Run only this policy (teacher/baseline/longhist/sparse/4cam). Default: all.")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Run only this scenario (maze_train/maze_static/maze_dynamic/maze_success). Default: all.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_plot", action="store_true",
                        help="Skip plot_validation.py after runs")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    try:
        llc_checkpoint, config_defaults, all_policies = load_validation_config(config_path)
    except ValueError as exc:
        parser.error(str(exc))

    for key in ("maze_episodes", "stuck_timeout", "num_envs", "depth_video_steps"):
        if getattr(args, key) is None:
            setattr(args, key, config_defaults[key])
    if args.maze_episodes <= 0:
        parser.error("--maze_episodes must be positive.")
    if args.num_envs <= 0:
        parser.error("--num_envs must be positive.")
    if args.locomotion_checkpoint is None:
        args.locomotion_checkpoint = llc_checkpoint
    if args.maze_route_steps is not None and (
        args.maze_route_steps[0] < 1 or args.maze_route_steps[1] < args.maze_route_steps[0]
    ):
        parser.error("--maze_route_steps requires 1 <= MIN <= MAX.")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = args.out_name or f"validation_{ts}"
    out_base = os.path.join("logs", "nav_play", out_name)

    policies = all_policies
    if args.ablation:
        policies = [p for p in all_policies if p["name"] == args.ablation]
        if not policies:
            print(f"[run_validation] ERROR: unknown policy '{args.ablation}'. "
                  f"Valid: {[p['name'] for p in all_policies]}", file=sys.stderr)
            sys.exit(1)

    _ALL_SCENARIOS = ["maze_train", "maze_static", "maze_dynamic", "maze_success"]
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
    effective_num_envs = min(args.num_envs, args.maze_episodes)
    print(f"[run_validation] Trajectories per scenario: {args.maze_episodes}")
    if args.maze_route_steps is not None:
        print(f"[run_validation] Route override (all scenarios): {args.maze_route_steps[0]}-{args.maze_route_steps[1]}")
    else:
        print("[run_validation] Routes: task-native for train/static/dynamic; "
              f"maze_success={config_defaults['success_route_min_steps']}-"
              f"{config_defaults['success_route_max_steps']} junctions")
    run_seeds = args.seeds if args.seeds is not None else (
        [args.seed] if args.seed is not None else config_defaults["evaluation_seeds"]
    )
    if len(set(run_seeds)) != len(run_seeds):
        parser.error("Evaluation seeds must be unique.")
    multi_seed = len(run_seeds) > 1
    print(f"[run_validation] Seeds: {run_seeds} (per-episode increment enabled)")
    print(f"[run_validation] Stuck timeout: {args.stuck_timeout} steps")
    print(f"[run_validation] Num envs: {effective_num_envs} (requested {args.num_envs})")

    _SCENARIO_TASK_KEY = {
        "maze_train":   "maze_train_task",
        "maze_static":  "maze_static_task",
        "maze_dynamic": "maze_dynamic_task",
        "maze_success": "maze_static_task",
    }

    # Do this before any simulator is launched.  It instantiates each selected
    # cfg only and rejects a stale obstacle_names list (the 16-vs-20 teacher
    # oracle failure) rather than wasting a Slurm allocation.
    preflight_cmd = [sys.executable, PREFLIGHT_SCRIPT, "--config", str(config_path)]
    if args.ablation:
        preflight_cmd.extend(["--ablation", args.ablation])
    if args.scenario:
        preflight_cmd.extend(["--scenario", args.scenario])
    if run_cmd(preflight_cmd, dry_run=args.dry_run) != 0:
        print("[run_validation] ERROR: cfg preflight failed; no evaluation was launched.", file=sys.stderr)
        sys.exit(2)

    failures = []

    for seed in run_seeds:
        seed_prefix = f"{out_name}/seed{seed}" if multi_seed else out_name
        for policy in policies:
            name = policy["name"]
            ckpt = policy["checkpoint"]

            for scenario in run_scenarios:
                route_steps = args.maze_route_steps
                if route_steps is None and scenario == "maze_success":
                    route_steps = (
                        config_defaults["success_route_min_steps"],
                        config_defaults["success_route_max_steps"],
                    )
                play_name = f"{seed_prefix}/{scenario}_{name}"
                cmd = build_cmd(
                    task=policy[_SCENARIO_TASK_KEY[scenario]],
                    checkpoint=ckpt,
                    play_name=play_name,
                    episodes=args.maze_episodes,
                    seed=seed,
                    stuck_timeout=args.stuck_timeout,
                    num_envs=effective_num_envs,
                    locomotion_checkpoint=args.locomotion_checkpoint,
                    depth_video_steps=args.depth_video_steps,
                    maze_route_steps=route_steps,
                    terminate_on_final_goal=(scenario == "maze_success"),
                )
                rc = run_cmd(cmd, dry_run=args.dry_run)
                if rc != 0:
                    print(f"[run_validation] WARNING: seed{seed} {scenario}_{name} exited with code {rc}")
                    failures.append(f"seed{seed}/{scenario}_{name}")

    # Plot. maze_success (single-route success/SPL) and the long-horizon
    # scenarios (sustained multi-route progress) measure different things and
    # must not share a chart's axes or a summary.csv's rows — see
    # plot_validation.py's --out_prefix guard. Plot each group separately.
    if not args.skip_plot:
        long_horizon_scenarios = [s for s in run_scenarios if s != "maze_success"]
        plot_groups = []
        if long_horizon_scenarios:
            plot_groups.append((long_horizon_scenarios, ""))
        if "maze_success" in run_scenarios:
            plot_groups.append((["maze_success"], "success_"))
        for scenarios, prefix in plot_groups:
            plot_cmd = [sys.executable, PLOT_SCRIPT, out_base, "--scenarios", *scenarios]
            if prefix:
                plot_cmd += ["--out_prefix", prefix]
            rc = run_cmd(plot_cmd, dry_run=args.dry_run)
            if rc != 0:
                print(f"[run_validation] WARNING: plot_validation.py exited with code {rc} for {scenarios}")

    print("\n" + "=" * 70)
    if failures:
        print(f"[run_validation] DONE (with failures: {failures})")
    else:
        print("[run_validation] DONE (all runs succeeded)")
    print(f"[run_validation] Results: {out_base}")
    print("=" * 70)


if __name__ == "__main__":
    main()
