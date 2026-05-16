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
    "--nav_case",
    "--nav-case",
    dest="nav_case",
    choices=[
        "random",
        "empty",
        "head_on",
        "left_edge",
        "right_edge",
        "diag_left",
        "diag_right",
        "off_left",
        "off_right",
        "narrow_gap",
    ],
    default="random",
    help="Navigation play tasks only: force the sampled obstacle template.",
)
parser.add_argument(
    "--nav_fixed_start",
    "--nav-fixed-start",
    dest="nav_fixed_start",
    action="store_true",
    default=False,
    help="Navigation play tasks only: reset every episode from a fixed local start pose.",
)
parser.add_argument("--nav_start_x", "--nav-start-x", dest="nav_start_x", type=float, default=None)
parser.add_argument("--nav_start_y", "--nav-start-y", dest="nav_start_y", type=float, default=None)
parser.add_argument("--nav_start_yaw", "--nav-start-yaw", dest="nav_start_yaw", type=float, default=None)
parser.add_argument("--nav_goal_forward", "--nav-goal-forward", dest="nav_goal_forward", type=float, default=None)
parser.add_argument("--nav_goal_lateral", "--nav-goal-lateral", dest="nav_goal_lateral", type=float, default=None)
parser.add_argument(
    "--nav_goal_heading_jitter",
    "--nav-goal-heading-jitter",
    dest="nav_goal_heading_jitter",
    type=float,
    default=None,
)
parser.add_argument(
    "--nav_log_interval",
    "--nav-log-interval",
    dest="nav_log_interval",
    type=int,
    default=120,
    help="Navigation play tasks only: print env debug every N sim steps. Use 0 to disable.",
)
parser.add_argument("--nav_log_env", "--nav-log-env", dest="nav_log_env", type=int, default=0)
parser.add_argument(
    "--nav_eval_episodes",
    "--nav-eval-episodes",
    dest="nav_eval_episodes",
    type=int,
    default=0,
    help="Navigation play tasks only: aggregate this many completed episodes, then exit. Use 0 for endless play.",
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
parser.add_argument(
    "--disable_export",
    "--disable-export",
    dest="disable_export",
    action="store_true",
    default=False,
    help="Skip JIT/ONNX policy export before play/eval.",
)
parser.add_argument(
    "--teacher_steering",
    "--teacher-steering",
    dest="teacher_steering",
    action="store_true",
    default=False,
    help="Run the rule-based navigation teacher directly instead of loading a student checkpoint.",
)
parser.add_argument(
    "--locomotion_checkpoint",
    type=str,
    default=None,
    help="Flat locomotion checkpoint used to initialize the teacher frozen LLC in --teacher_steering mode.",
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

from collections import defaultdict
import copy
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
from rsl_rl.utils import resolve_callable

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
from go2w.tasks.manager_based.go2w.observation_layout import POLICY_OBS


def _find_state_dict(ckpt: dict, candidates: tuple[str, ...], label: str) -> tuple[str, dict]:
    """Return the first matching state dict from a checkpoint."""
    for key in candidates:
        if key in ckpt and isinstance(ckpt[key], dict):
            return key, ckpt[key]
    raise ValueError(f"No {label} state dict found. Keys found: {list(ckpt.keys())}")


def _load_padded_state_dict(model, src_sd: dict, device: str, label: str, strip_distribution: bool = False) -> None:
    """Load a state dict, zero-padding first-layer inputs when obs dims grow."""
    src_sd = {key: value.to(device) for key, value in src_sd.items()}
    if strip_distribution:
        src_sd = {key: value for key, value in src_sd.items() if not key.startswith("distribution.")}

    current_sd = model.state_dict()
    new_sd = {}
    for key, target in current_sd.items():
        if key not in src_sd:
            new_sd[key] = target
            continue

        source = src_sd[key]
        if source.shape == target.shape:
            new_sd[key] = source
        elif len(source.shape) == 2 and source.shape[0] == target.shape[0] and source.shape[1] < target.shape[1]:
            pad = torch.zeros(source.shape[0], target.shape[1] - source.shape[1], dtype=source.dtype, device=device)
            new_sd[key] = torch.cat([source, pad], dim=1)
            print(f"[INFO] Zero-padded {label} '{key}': {tuple(source.shape)} -> {tuple(new_sd[key].shape)}")
        else:
            print(
                f"[WARN] Shape mismatch in {label} '{key}': "
                f"src={tuple(source.shape)}, tgt={tuple(target.shape)}; keeping current init"
            )
            new_sd[key] = target

    model.load_state_dict(new_sd)


def _load_teacher_locomotion_checkpoint(teacher, ckpt_path: str, device: str) -> None:
    """Initialize the teacher frozen LLC from a flat locomotion checkpoint."""
    frozen_actor = getattr(teacher, "frozen_actor", None)
    if frozen_actor is None:
        raise ValueError("--teacher_steering requires a teacher with a frozen_actor.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor_key, actor_sd = _find_state_dict(
        ckpt,
        ("actor_state_dict", "model_state_dict", "policy_state_dict"),
        "actor",
    )
    _load_padded_state_dict(frozen_actor, actor_sd, device, "teacher frozen actor", strip_distribution=True)
    print(f"[INFO] Loaded teacher frozen LLC from '{actor_key}' in: {ckpt_path}")


def _configure_frozen_llc_action(env_cfg, ckpt_path: str | None) -> bool:
    """Inject the fast-flat LLC checkpoint into HLC action configs before env creation."""
    actions_cfg = getattr(env_cfg, "actions", None)
    llc_cmd_cfg = getattr(actions_cfg, "llc_cmd", None)
    if llc_cmd_cfg is None:
        return False
    if not ckpt_path:
        raise ValueError(
            f"Task '{args_cli.task}' uses FrozenLLCActionTerm and requires --locomotion_checkpoint "
            "before gym.make() so the frozen fast-flat LLC can be loaded."
        )
    llc_cmd_cfg.llc_checkpoint_path = ckpt_path
    print(f"[INFO] Frozen LLC checkpoint injected into action term: {ckpt_path}")
    return True


def _build_teacher_policy(env, obs, agent_cfg: RslRlBaseRunnerCfg, device: str):
    """Instantiate the rule-based navigation teacher for direct play/evaluation."""
    teacher_cfg = getattr(agent_cfg, "teacher", None)
    if teacher_cfg is None:
        raise ValueError("--teacher_steering requires a distillation runner config with a teacher model.")
    if "teacher" not in obs:
        raise ValueError("--teacher_steering requires an env exposing the 'teacher' observation group.")
    if args_cli.locomotion_checkpoint is None:
        raise ValueError("--teacher_steering requires --locomotion_checkpoint for the frozen LLC.")

    teacher_cfg_dict = copy.deepcopy(teacher_cfg.to_dict())
    teacher_class = resolve_callable(teacher_cfg_dict.pop("class_name"))
    teacher = teacher_class(obs, {"teacher": ["teacher"]}, "teacher", env.num_actions, **teacher_cfg_dict)
    teacher = teacher.to(device)
    teacher.eval()
    _load_teacher_locomotion_checkpoint(teacher, args_cli.locomotion_checkpoint, device)
    return teacher


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

    if "start_iteration" in params:
        params["start_iteration"] = 0
    if "warmup_iterations" in params:
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


def _override_navigation_play_case(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    args_cli: argparse.Namespace,
):
    """Override start/goal/template sampling for navigation play tasks."""
    events_cfg = getattr(env_cfg, "events", None)
    reset_obstacles = getattr(events_cfg, "reset_obstacles", None) if events_cfg is not None else None
    reset_base = getattr(events_cfg, "reset_base", None) if events_cfg is not None else None
    reset_params = getattr(reset_obstacles, "params", None)

    has_nav_override = (
        args_cli.nav_case != "random"
        or args_cli.nav_goal_forward is not None
        or args_cli.nav_goal_lateral is not None
        or args_cli.nav_goal_heading_jitter is not None
        or args_cli.nav_fixed_start
        or args_cli.nav_start_x is not None
        or args_cli.nav_start_y is not None
        or args_cli.nav_start_yaw is not None
    )
    if not has_nav_override:
        return
    if reset_params is None or "fixed_scenario_template" not in reset_params:
        raise ValueError("Navigation play overrides require a navigation-distillation play task.")

    if args_cli.nav_case != "random":
        reset_params["fixed_scenario_template"] = args_cli.nav_case
    if args_cli.nav_goal_forward is not None:
        reset_params["fixed_goal_forward"] = args_cli.nav_goal_forward
    if args_cli.nav_goal_lateral is not None:
        reset_params["fixed_goal_lateral"] = args_cli.nav_goal_lateral
    if args_cli.nav_goal_heading_jitter is not None:
        reset_params["fixed_goal_heading_jitter"] = args_cli.nav_goal_heading_jitter

    if (
        args_cli.nav_fixed_start
        or args_cli.nav_start_x is not None
        or args_cli.nav_start_y is not None
        or args_cli.nav_start_yaw is not None
    ):
        if reset_base is None:
            raise ValueError("Navigation fixed-start play requires a reset_base event.")
        pose_range = reset_base.params.setdefault("pose_range", {})
        velocity_range = reset_base.params.setdefault("velocity_range", {})
        start_x = 0.0 if args_cli.nav_start_x is None else args_cli.nav_start_x
        start_y = 0.0 if args_cli.nav_start_y is None else args_cli.nav_start_y
        start_yaw = 0.0 if args_cli.nav_start_yaw is None else args_cli.nav_start_yaw
        pose_range["x"] = (start_x, start_x)
        pose_range["y"] = (start_y, start_y)
        pose_range["yaw"] = (start_yaw, start_yaw)
        for key in ("x", "y", "z", "roll", "pitch", "yaw"):
            velocity_range[key] = (0.0, 0.0)
        print(f"[INFO] Navigation fixed start: x={start_x:.2f}, y={start_y:.2f}, yaw={start_yaw:.2f}")

    print(
        "[INFO] Navigation play case: "
        f"case={reset_params.get('fixed_scenario_template') or 'random'}, "
        f"goal_forward={reset_params.get('fixed_goal_forward')}, "
        f"goal_lateral={reset_params.get('fixed_goal_lateral')}, "
        f"goal_heading_jitter={reset_params.get('fixed_goal_heading_jitter')}"
    )


def _resolve_play_seed(args_cli, default_seed: int | None) -> int:
    if args_cli.seed is not None:
        return args_cli.seed
    base_seed = default_seed if default_seed is not None else 0
    return (base_seed + random.SystemRandom().randrange(1, 2_147_483_647)) % 2_147_483_647


def _format_vector(values: torch.Tensor, precision: int = 3) -> str:
    return ", ".join(f"{value.item():+.{precision}f}" for value in values)


def _format_eval_metrics(metrics: dict[str, float], completed_episodes: int, avg_episode_length: float) -> str:
    preferred_keys = [
        "goal_reached_rate",
        "time_out_rate",
        "base_contact_rate",
        "root_height_below_minimum_rate",
        "multi_term_fraction",
    ]
    parts = [f"episodes={completed_episodes}", f"avg_episode_len={avg_episode_length:.2f}"]
    for key in preferred_keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    return " ".join(parts)


def _print_navigation_play_log(obs, dones: torch.Tensor, step_count: int, env_index: int) -> None:
    """Print navigation task diagnostics from the policy obs group."""
    # Works for both PPO (policy group) and distillation (student/teacher groups).
    group_key = "policy" if isinstance(obs, dict) and "policy" in obs else None
    if group_key is None and isinstance(obs, dict) and "student" in obs:
        group_key = "student"
    if group_key is None:
        return
    data = obs[group_key]
    if data.ndim != 2 or data.shape[0] == 0:
        return
    env_index = max(0, min(env_index, data.shape[0] - 1))
    row = data[env_index]

    if row.numel() == 189:
        base_lin_vel = row[0:3]
        projected_gravity = row[3:6]
        goal_command = row[6:9]
        state_text = f"projected_gravity=[{_format_vector(projected_gravity)}]"
    else:
        base_lin_vel = row[POLICY_OBS["base_lin_vel"].as_slice()]
        base_ang_vel = row[POLICY_OBS["base_ang_vel"].as_slice()]
        goal_command = row[POLICY_OBS["goal_command"].as_slice()]
        state_text = f"base_ang_vel=[{_format_vector(base_ang_vel)}]"
    done = int(dones[env_index].item()) if dones.numel() > env_index else 0

    print(
        "[nav-play] "
        f"step={step_count} env={env_index} done={done} "
        f"goal_cmd=[{_format_vector(goal_command)}] "
        f"base_lin_vel=[{_format_vector(base_lin_vel)}] "
        f"{state_text}"
    )


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
    _override_navigation_play_case(env_cfg, args_cli)

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
    resume_path = None
    if args_cli.teacher_steering:
        log_dir = os.path.join(log_root_path, "teacher_play")
    elif args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
        log_dir = os.path.dirname(resume_path)
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        log_dir = os.path.dirname(resume_path)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir
    _configure_frozen_llc_action(env_cfg, args_cli.locomotion_checkpoint)

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

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    teacher = None
    policy_nn = None
    if args_cli.teacher_steering:
        teacher = _build_teacher_policy(env, obs, agent_cfg, env.unwrapped.device)
        print("[INFO] Running direct teacher steering: geometric navigation teacher + frozen LLC")
    else:
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
        if args_cli.disable_export or agent_cfg.class_name == "DistillationRunner":
            print("[INFO] Skipping policy export before play/eval.")
        else:
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

    timestep = 0
    step_count = 0
    episode_lengths = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device, dtype=torch.long)
    completed_episodes = 0
    total_episode_length = 0.0
    termination_manager = getattr(env.unwrapped, "termination_manager", None)
    termination_names = list(termination_manager.active_terms) if termination_manager is not None else []
    termination_counts: dict[str, int] = defaultdict(int)
    multi_term_episodes = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = teacher(obs) if teacher is not None else policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if teacher is not None:
                teacher.reset(dones.bool())
            elif version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

        episode_lengths += 1
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        num_done = int(done_ids.numel())
        if num_done > 0:
            total_episode_length += float(episode_lengths[done_ids].sum().item())
            episode_lengths[done_ids] = 0
            completed_episodes += num_done
            done_terms = (
                termination_manager._last_episode_dones[done_ids] if termination_manager is not None else None
            )
            if done_terms is not None and done_terms.numel() > 0:
                multi_term_episodes += int((done_terms.sum(dim=1) > 1).sum().item())
                for idx, term_name in enumerate(termination_names):
                    termination_counts[term_name] += int(done_terms[:, idx].sum().item())

        if args_cli.nav_log_interval > 0 and step_count % args_cli.nav_log_interval == 0:
            _print_navigation_play_log(obs, dones, step_count, args_cli.nav_log_env)
        if args_cli.nav_eval_episodes > 0 and (
            completed_episodes >= args_cli.nav_eval_episodes
            or (num_done > 0 and completed_episodes % max(args_cli.nav_eval_episodes // 4, 1) < num_done)
        ):
            averaged = {
                f"{term_name}_rate": termination_counts[term_name] / max(completed_episodes, 1)
                for term_name in termination_names
            }
            averaged["multi_term_fraction"] = multi_term_episodes / max(completed_episodes, 1)
            avg_episode_length = total_episode_length / max(completed_episodes, 1)
            print("[PLAY-EVAL] " + _format_eval_metrics(averaged, completed_episodes, avg_episode_length))
            if completed_episodes >= args_cli.nav_eval_episodes:
                break

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        step_count += 1

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
