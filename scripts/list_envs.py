# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to list all Go2-W environments registered in the go2w package.

Usage:
    ./isaaclab.sh -p scripts/list_envs.py
    ./isaaclab.sh -p scripts/list_envs.py --keyword Flat
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="List registered Go2-W environments.")
parser.add_argument("--keyword", type=str, default=None, help="Optional keyword to filter environment names.")
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
from prettytable import PrettyTable

import go2w.tasks  # noqa: F401


def main():
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Go2-W Registered Environments"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    index = 0
    for task_spec in gym.registry.values():
        if "go2w" in task_spec.id.lower() or "go2w" in str(task_spec.kwargs.get("env_cfg_entry_point", "")).lower():
            if args_cli.keyword is None or args_cli.keyword in task_spec.id:
                table.add_row([index + 1, task_spec.id, task_spec.entry_point, task_spec.kwargs["env_cfg_entry_point"]])
                index += 1

    print(table)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
    finally:
        simulation_app.close()
