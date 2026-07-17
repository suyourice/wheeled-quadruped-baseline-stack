# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained policy with fixed velocity commands specified from the CLI.

Unlike play.py (which uses random commands sampled from the training distribution),
this script locks the velocity command to user-specified values so you can
evaluate a specific motion (e.g. forward drive, pure yaw, lateral slide).
Pass ``--random_commands`` to keep the environment's native random command
sampler instead.

Usage examples:
    # Forward at 0.5 m/s
    python scripts/rsl_rl/play_cmd.py --task Loco-Flat-Go2w-Play-v0 --cmd_vx 0.5

    # Lateral slide
    python scripts/rsl_rl/play_cmd.py --task Loco-Flat-Go2w-Play-v0 --cmd_vy 0.3

    # Spin in place
    python scripts/rsl_rl/play_cmd.py --task Loco-Flat-Go2w-Play-v0 --cmd_wz 1.0

    # Diagonal + yaw
    python scripts/rsl_rl/play_cmd.py --task Loco-Flat-Go2w-Play-v0 --cmd_vx 0.5 --cmd_vy 0.3 --cmd_wz 0.5

    # Stand still
    python scripts/rsl_rl/play_cmd.py --task Loco-Flat-Go2w-Play-v0

    # Evaluate the RL navigation teacher
    python scripts/rsl_rl/play_cmd.py \
        --task Nav-ObstacleFlat-Teacher-Go2w-Play-v0 \
        --checkpoint <path> \
        --cmd_vx 1.0 \
        --num_obstacles 5

    # Evaluate the LiDAR distillation student
    python scripts/rsl_rl/play_cmd.py \
        --task Nav-ObstacleFlat-Distill-Lidar-Go2w-Play-v0 \
        --checkpoint <path> \
        --num_obstacles 5
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from checkpoint_utils import configure_frozen_llc_action  # isort: skip
from play_common import (  # isort: skip
    build_teacher_policy,
    override_play_obstacle_count,
    override_play_command_path_spawn,
    resolve_play_seed,
)

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
parser.add_argument("--cmd_wz", "--cmd_yaw", dest="cmd_wz", type=float, default=0.0, help="Angular velocity z [rad/s] (default: 0.0)")
parser.add_argument(
    "--random_commands",
    action="store_true",
    default=False,
    help="Keep the env's native random command sampler instead of locking to --cmd_vx/--cmd_vy/--cmd_wz.",
)
parser.add_argument("--num_obstacles", "--num-obstacles", dest="num_obstacles", type=int, default=None,
                    help="Override active obstacle count for obstacle play envs.")
parser.add_argument(
    "--command_path_obstacles",
    "--command-path-obstacles",
    dest="command_path_obstacles",
    type=int,
    default=None,
    help="Force this many active obstacle slots to spawn in front of the current command direction.",
)
parser.add_argument(
    "--command_path_forward_range",
    "--command-path-forward-range",
    dest="command_path_forward_range",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=None,
    help="Forward spawn range [m] for command-path obstacles. Default: 1.6 2.4",
)
parser.add_argument(
    "--command_path_lateral_range",
    "--command-path-lateral-range",
    dest="command_path_lateral_range",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=None,
    help="Lateral spawn range [m] for command-path obstacles. Default: -0.35 0.35",
)
parser.add_argument(
    "--command_path_min_speed",
    "--command-path-min-speed",
    dest="command_path_min_speed",
    type=float,
    default=None,
    help="Below this command speed, command-path spawn falls back to random placement. Default: 0.2",
)
parser.add_argument(
    "--teacher_steering",
    action="store_true",
    default=False,
    help="Run the rule-based steering teacher directly instead of loading a trained policy checkpoint.",
)
parser.add_argument(
    "--locomotion_checkpoint",
    type=str,
    default=None,
    help="Flat locomotion checkpoint used to initialize the frozen LLC in --teacher_steering mode.",
)
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
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_seed = resolve_play_seed(args_cli, agent_cfg.seed)
    agent_cfg.seed = env_seed
    env_cfg.seed = env_seed
    print(f"[INFO] Play seed: {env_seed}")
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    override_play_obstacle_count(env_cfg, args_cli.num_obstacles)
    override_play_command_path_spawn(
        env_cfg,
        args_cli.command_path_obstacles,
        args_cli.command_path_forward_range,
        args_cli.command_path_lateral_range,
        args_cli.command_path_min_speed,
        command_path_reference_xy=None if args_cli.random_commands else (args_cli.cmd_vx, args_cli.cmd_vy),
    )

    # ------------------------------------------------------------------
    # Either fix velocity commands to CLI values or keep native random sampling.
    # ------------------------------------------------------------------
    cmd = env_cfg.commands.base_velocity
    if args_cli.random_commands:
        print(
            "[INFO] Random command mode: "
            f"vx={cmd.ranges.lin_vel_x}, vy={cmd.ranges.lin_vel_y}, wz={cmd.ranges.ang_vel_z}, "
            f"resample={cmd.resampling_time_range}, standing={cmd.rel_standing_envs:.2f}"
        )
    else:
        vx = args_cli.cmd_vx
        vy = args_cli.cmd_vy
        wz = args_cli.cmd_wz
        print(f"[INFO] Fixed command: vx={vx:.2f} m/s  vy={vy:.2f} m/s  wz={wz:.2f} rad/s")

        cmd.ranges.lin_vel_x = (vx, vx)
        cmd.ranges.lin_vel_y = (vy, vy)
        cmd.ranges.ang_vel_z = (wz, wz)
        cmd.ranges.heading = (0.0, 0.0)
        cmd.resampling_time_range = (1e9, 1e9)  # effectively never resample
        cmd.rel_standing_envs = 0.0
    # ------------------------------------------------------------------

    resume_path = None
    if not args_cli.teacher_steering:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        print(f"[INFO] Loading checkpoint: {resume_path}")
        log_dir = os.path.dirname(resume_path)
        env_cfg.log_dir = log_dir

    configure_frozen_llc_action(env_cfg, args_cli.locomotion_checkpoint, args_cli.task)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    teacher = None

    if args_cli.teacher_steering:
        teacher = build_teacher_policy(
            env, obs, agent_cfg, env.unwrapped.device, args_cli.locomotion_checkpoint
        )
        print("[INFO] Running direct teacher evaluation: geometric steering + frozen LLC")
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

    step_count = 0

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            if teacher is not None:
                actions = teacher(obs)
            else:
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if teacher is not None:
                teacher.reset(dones.bool())
            elif version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
        step_count += 1

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
