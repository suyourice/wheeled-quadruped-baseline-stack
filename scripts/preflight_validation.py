"""Check validation task configurations before launching Isaac Sim rollouts.

This checks configs only: no Gym environment, scene, or physics step is ever
created. It still needs a running (headless) Kit app, though, because
``import go2w.tasks`` pulls in ``isaaclab.envs`` / ``isaaclab_tasks``, which
in turn import Kit extension modules (e.g. ``omni.timeline``) that only
become importable once ``AppLauncher`` has bootstrapped the Omniverse Kit
runtime. Skipping that bootstrap raises ``ModuleNotFoundError: No module
named 'omni.timeline'`` before a single cfg is even loaded.

Run with the Isaac Sim Python executable, for example:
    ./isaaclab.sh -p scripts/preflight_validation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Preflight hospital validation cfgs")
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--ablation", type=str, default=None)
parser.add_argument(
    "--scenario",
    choices=("maze_train", "maze_static", "maze_dynamic", "maze_success"),
    default=None,
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True  # cfg check only; never needs a display

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
    config_path = resolve_config_path(args_cli.config)
    try:
        _, _, policies = load_validation_config(config_path)
    except ValueError as exc:
        parser.error(str(exc))

    if args_cli.ablation:
        policies = [policy for policy in policies if policy["name"] == args_cli.ablation]
        if not policies:
            parser.error(f"Unknown policy '{args_cli.ablation}'.")
    scenarios = [args_cli.scenario] if args_cli.scenario else list(SCENARIO_TASK_KEYS)

    # Task registration and cfg loading do not instantiate an environment.
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
    exit_code = main()
    # simulation_app.close() tears the Kit process down abruptly rather than
    # through Python's normal interpreter shutdown, which would otherwise
    # flush stdio; without an explicit flush here the last print()s above
    # (in particular "OK ..."/"PASS ...", which unlike the debug prints
    # elsewhere in this file are not flush=True) are silently dropped when
    # stdout isn't a TTY (e.g. piped to a Slurm log), making a successful
    # run look like it hung or produced no output.
    sys.stdout.flush()
    sys.stderr.flush()
    simulation_app.close()
    raise SystemExit(exit_code)
