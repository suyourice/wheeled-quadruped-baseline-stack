"""Check validation task configurations before launching Isaac Sim rollouts.

This is a cfg-only check: no environment, renderer, or GPU allocation is
created.  It verifies that every observation/reward/event/termination term
with ``obstacle_names`` covers exactly the obstacle slots in that task scene.

Run with the Isaac Sim Python executable, for example:
    ./isaaclab.sh -p scripts/preflight_validation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "rsl_rl"))

from play_common import preflight_check_cfg_obstacle_slots  # noqa: E402
from run_validation import load_validation_config, resolve_config_path  # noqa: E402


SCENARIO_TASK_KEYS = {
    "maze_train": "maze_train_task",
    "maze_static": "maze_static_task",
    "maze_dynamic": "maze_dynamic_task",
    # Short-route completion protocol reuses the static 20-slot task cfg.
    "maze_success": "maze_static_task",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight hospital validation cfgs")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ablation", type=str, default=None)
    parser.add_argument("--scenario", choices=tuple(SCENARIO_TASK_KEYS), default=None)
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    try:
        _, _, policies = load_validation_config(config_path)
    except ValueError as exc:
        parser.error(str(exc))

    if args.ablation:
        policies = [policy for policy in policies if policy["name"] == args.ablation]
        if not policies:
            parser.error(f"Unknown policy '{args.ablation}'.")
    scenarios = [args.scenario] if args.scenario else list(SCENARIO_TASK_KEYS)

    # Task registration and config loading require Isaac Sim's Python runtime,
    # but do not launch the simulator or instantiate an environment.
    import go2w.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    checked = 0
    for policy in policies:
        for scenario in scenarios:
            task = policy[SCENARIO_TASK_KEYS[scenario]]
            try:
                cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
                slots = preflight_check_cfg_obstacle_slots(cfg)
            except Exception as exc:
                print(
                    f"[preflight_validation] FAIL {scenario}/{policy['name']} ({task}): {exc}",
                    file=sys.stderr,
                )
                return 1
            print(
                f"[preflight_validation] OK {scenario}/{policy['name']}: "
                f"{slots} obstacle slots ({task})"
            )
            checked += 1

    print(f"[preflight_validation] PASS: {checked} task cfgs checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
