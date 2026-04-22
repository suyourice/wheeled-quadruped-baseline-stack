# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained policy with fixed velocity commands specified from the CLI.

Unlike play.py (which uses random commands sampled from the training distribution),
this script locks the velocity command to user-specified values so you can
evaluate a specific motion (e.g. forward drive, pure yaw, lateral slide).

Usage examples:
    # Forward at 0.5 m/s
    python scripts/rsl_rl/play_cmd.py --task Flat-Go2w-Play-v0 --cmd_vx 0.5

    # Lateral slide
    python scripts/rsl_rl/play_cmd.py --task Flat-Go2w-Play-v0 --cmd_vy 0.3

    # Spin in place
    python scripts/rsl_rl/play_cmd.py --task Flat-Go2w-Play-v0 --cmd_wz 1.0

    # Diagonal + yaw
    python scripts/rsl_rl/play_cmd.py --task Flat-Go2w-Play-v0 --cmd_vx 0.5 --cmd_vy 0.3 --cmd_wz 0.5

    # Stand still
    python scripts/rsl_rl/play_cmd.py --task Flat-Go2w-Play-v0
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play a Go2-W policy with fixed velocity commands.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task",     type=str, default=None)
parser.add_argument("--seed",     type=int, default=None)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False,
    help="Disable fabric and use USD I/O operations.",
)
# Fixed velocity command arguments
parser.add_argument("--cmd_vx", type=float, default=0.0, help="Linear velocity x [m/s]  (default: 0.0)")
parser.add_argument("--cmd_vy", type=float, default=0.0, help="Linear velocity y [m/s]  (default: 0.0)")
parser.add_argument("--cmd_wz", type=float, default=0.0, help="Angular velocity z [rad/s] (default: 0.0)")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata
import os
import time

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import go2w.tasks  # noqa: F401

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with fixed velocity commands."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ------------------------------------------------------------------
    # Fix velocity commands to CLI values — never resampled.
    # ------------------------------------------------------------------
    vx = args_cli.cmd_vx
    vy = args_cli.cmd_vy
    wz = args_cli.cmd_wz
    print(f"[INFO] Fixed command: vx={vx:.2f} m/s  vy={vy:.2f} m/s  wz={wz:.2f} rad/s")

    cmd = env_cfg.commands.base_velocity
    cmd.ranges.lin_vel_x = (vx, vx)
    cmd.ranges.lin_vel_y = (vy, vy)
    cmd.ranges.ang_vel_z = (wz, wz)
    cmd.ranges.heading   = (0.0, 0.0)
    cmd.resampling_time_range = (1e9, 1e9)  # effectively never resample
    cmd.rel_standing_envs = 0.0
    # ------------------------------------------------------------------

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[INFO] Loading checkpoint: {resume_path}")
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    obs = env.get_observations()

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
