# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import random
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

DEFAULT_COMMAND_PATH_FORWARD_RANGE = (1.6, 2.4)
DEFAULT_COMMAND_PATH_LATERAL_RANGE = (-0.35, 0.35)
DEFAULT_COMMAND_PATH_MIN_SPEED = 0.2

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--num_obstacles",
    "--num-obstacles",
    dest="num_obstacles",
    type=int,
    default=None,
    help="Obstacle play tasks only: force this many active boxes in the scene.",
)
parser.add_argument(
    "--command_path_obstacles",
    "--command-path-obstacles",
    dest="command_path_obstacles",
    type=int,
    default=None,
    help="Obstacle play tasks only: force this many active slots to spawn in front of the current command direction.",
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
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import go2w.tasks  # noqa: F401


def _override_play_obstacle_count(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    num_obstacles: int | None,
):
    """Override active obstacle count for obstacle play configs."""
    if num_obstacles is None:
        return
    if num_obstacles < 0:
        raise ValueError("--num_obstacles must be >= 0.")

    events_cfg = getattr(env_cfg, "events", None)
    reset_obstacles = getattr(events_cfg, "reset_obstacles", None) if events_cfg is not None else None
    if reset_obstacles is None:
        raise ValueError("--num_obstacles requires an obstacle play task with a reset_obstacles event.")

    params = reset_obstacles.params
    obstacle_names = params.get("obstacle_names", [])
    max_available = len(obstacle_names)
    if num_obstacles > max_available:
        raise ValueError(
            f"--num_obstacles={num_obstacles} exceeds the play scene capacity ({max_available}). "
            "Increase PLAY_MAX_OBSTACLES in go2w_obstacle_env_cfg.py if you need more."
        )

    params["start_iteration"] = 0
    params["warmup_iterations"] = 0
    params["min_obstacles"] = num_obstacles
    params["max_obstacles"] = num_obstacles
    print(f"[INFO] Active play obstacles: {num_obstacles}/{max_available}")


def _override_play_command_path_spawn(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    command_path_obstacles: int | None,
    command_path_forward_range: tuple[float, float] | list[float] | None,
    command_path_lateral_range: tuple[float, float] | list[float] | None,
    command_path_min_speed: float | None,
):
    """Override command-direction obstacle spawn for obstacle play configs."""
    if (
        command_path_obstacles is None
        and command_path_forward_range is None
        and command_path_lateral_range is None
        and command_path_min_speed is None
    ):
        return

    events_cfg = getattr(env_cfg, "events", None)
    reset_obstacles = getattr(events_cfg, "reset_obstacles", None) if events_cfg is not None else None
    if reset_obstacles is None:
        raise ValueError("--command_path_obstacles requires an obstacle play task with a reset_obstacles event.")

    params = reset_obstacles.params
    obstacle_names = params.get("obstacle_names", [])
    max_available = len(obstacle_names)
    active_obstacles = int(params.get("max_obstacles", max_available))

    count = params.get("command_path_obstacles", 0) if command_path_obstacles is None else command_path_obstacles
    if count < 0:
        raise ValueError("--command_path_obstacles must be >= 0.")
    if count > active_obstacles:
        raise ValueError(
            f"--command_path_obstacles={count} exceeds active obstacle count ({active_obstacles}). "
            "Increase --num_obstacles or lower the command-path count."
        )

    forward_range = tuple(command_path_forward_range) if command_path_forward_range is not None else params.get(
        "command_path_forward_range", DEFAULT_COMMAND_PATH_FORWARD_RANGE
    )
    lateral_range = tuple(command_path_lateral_range) if command_path_lateral_range is not None else params.get(
        "command_path_lateral_range", DEFAULT_COMMAND_PATH_LATERAL_RANGE
    )
    min_speed = command_path_min_speed
    if min_speed is None:
        min_speed = params.get("command_path_min_speed", DEFAULT_COMMAND_PATH_MIN_SPEED)

    params["command_path_obstacles"] = count
    params["command_name"] = "base_velocity"
    params["command_path_forward_range"] = forward_range
    params["command_path_lateral_range"] = lateral_range
    params["command_path_min_speed"] = min_speed
    print(
        "[INFO] Command-path obstacle spawn: "
        f"count={count}, forward={forward_range}, lateral={lateral_range}, min_speed={min_speed:.2f}"
    )


def _resolve_play_seed(args_cli, default_seed: int | None) -> int:
    if args_cli.seed is not None:
        return args_cli.seed
    base_seed = default_seed if default_seed is not None else 0
    return (base_seed + random.SystemRandom().randrange(1, 2_147_483_647)) % 2_147_483_647


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    _override_play_obstacle_count(env_cfg, args_cli.num_obstacles)
    _override_play_command_path_spawn(
        env_cfg,
        args_cli.command_path_obstacles,
        args_cli.command_path_forward_range,
        args_cli.command_path_lateral_range,
        args_cli.command_path_min_speed,
    )

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_seed = _resolve_play_seed(args_cli, agent_cfg.seed)
    agent_cfg.seed = env_seed
    env_cfg.seed = env_seed
    print(f"[INFO] Play seed: {env_seed}")
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
