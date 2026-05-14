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
    python scripts/rsl_rl/play_cmd.py --task Flat-Go2w-Play-v0 --cmd_vx 0.5

    # Lateral slide
    python scripts/rsl_rl/play_cmd.py --task Flat-Go2w-Play-v0 --cmd_vy 0.3

    # Spin in place
    python scripts/rsl_rl/play_cmd.py --task Flat-Go2w-Play-v0 --cmd_wz 1.0

    # Diagonal + yaw
    python scripts/rsl_rl/play_cmd.py --task Flat-Go2w-Play-v0 --cmd_vx 0.5 --cmd_vy 0.3 --cmd_wz 0.5

    # Stand still
    python scripts/rsl_rl/play_cmd.py --task Flat-Go2w-Play-v0

    # Evaluate the rule-based navigation-distillation teacher directly
    python scripts/rsl_rl/play_cmd.py \
        --task Navigation-Distill-Go2w-Play-v0 \
        --teacher_steering \
        --locomotion_checkpoint logs/rsl_rl/go2w_fast_flat/2026-04-29_18-17-48/model_1999.pt \
        --cmd_vx 1.0 \
        --num_obstacles 2

    # Evaluate the rule-based navigation-distillation teacher on random commands
    python scripts/rsl_rl/play_cmd.py \
        --task Navigation-Distill-Go2w-Play-v0 \
        --teacher_steering \
        --random_commands \
        --locomotion_checkpoint logs/rsl_rl/go2w_fast_flat/2026-04-29_18-17-48/model_1999.pt \
        --num_obstacles 2

    # Spawn two obstacles directly ahead of the commanded motion ray
    python scripts/rsl_rl/play_cmd.py \
        --task Navigation-Distill-Go2w-Play-v0 \
        --teacher_steering \
        --locomotion_checkpoint logs/rsl_rl/go2w_fast_flat/2026-04-29_18-17-48/model_1999.pt \
        --cmd_vx 0.8 --cmd_vy -0.6 \
        --num_obstacles 3 \
        --command_path_obstacles 2
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import random
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

DEFAULT_COMMAND_PATH_FORWARD_RANGE = (1.6, 2.4)
DEFAULT_COMMAND_PATH_LATERAL_RANGE = (-0.35, 0.35)
DEFAULT_COMMAND_PATH_MIN_SPEED = 0.2

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
parser.add_argument(
    "--teacher_debug_interval",
    type=int,
    default=120,
    help="Simulation steps between teacher debug prints in --teacher_steering mode.",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata
import copy
import os
import time

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_callable

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


def _find_state_dict(ckpt: dict, candidates: tuple[str, ...], label: str) -> tuple[str, dict]:
    """Return the first matching state dict from a checkpoint."""
    for key in candidates:
        if key in ckpt and isinstance(ckpt[key], dict):
            return key, ckpt[key]
    raise ValueError(f"No {label} state dict found. Keys found: {list(ckpt.keys())}")


def _load_padded_state_dict(model, src_sd: dict, device: str, label: str, strip_distribution: bool = False) -> None:
    """Load a state dict, zero-padding first-layer inputs when obs dims grow."""
    src_sd = {k: v.to(device) for k, v in src_sd.items()}
    if strip_distribution:
        src_sd = {k: v for k, v in src_sd.items() if not k.startswith("distribution.")}

    current_sd = model.state_dict()
    new_sd = {}
    for key, tgt in current_sd.items():
        if key not in src_sd:
            new_sd[key] = tgt
            continue

        src = src_sd[key]
        if src.shape == tgt.shape:
            new_sd[key] = src
        elif len(src.shape) == 2 and src.shape[0] == tgt.shape[0] and src.shape[1] < tgt.shape[1]:
            n_pad = tgt.shape[1] - src.shape[1]
            pad = torch.zeros(src.shape[0], n_pad, dtype=src.dtype, device=device)
            new_sd[key] = torch.cat([src, pad], dim=1)
            print(f"[INFO] Zero-padded {label} '{key}': {tuple(src.shape)} -> {tuple(new_sd[key].shape)}")
        else:
            print(
                f"[WARN] Shape mismatch in {label} '{key}': "
                f"src={tuple(src.shape)}, tgt={tuple(tgt.shape)}; keeping current init"
            )
            new_sd[key] = tgt

    model.load_state_dict(new_sd)


def _load_teacher_locomotion_checkpoint(teacher, ckpt_path: str, device: str) -> None:
    """Initialize the teacher frozen LLC from a flat locomotion checkpoint."""
    teacher_target = getattr(teacher, "frozen_actor", None)
    if teacher_target is None:
        raise ValueError("Teacher-steering mode requires a teacher with a frozen_actor.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor_key, actor_sd = _find_state_dict(
        ckpt,
        ("actor_state_dict", "model_state_dict", "policy_state_dict"),
        "actor",
    )
    _load_padded_state_dict(teacher_target, actor_sd, device, "teacher frozen actor", strip_distribution=True)
    print(f"[INFO] Loaded teacher frozen LLC from '{actor_key}' in: {ckpt_path}")


def _override_play_obstacle_count(env_cfg, num_obstacles):
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
    max_available = len(params.get("obstacle_names", []))
    if num_obstacles > max_available:
        raise ValueError(
            f"--num_obstacles={num_obstacles} exceeds play scene capacity ({max_available})."
        )
    params["start_iteration"] = 0
    params["warmup_iterations"] = 0
    params["min_obstacles"] = num_obstacles
    params["max_obstacles"] = num_obstacles
    print(f"[INFO] Active play obstacles: {num_obstacles}/{max_available}")


def _override_play_command_path_spawn(
    env_cfg,
    command_path_obstacles,
    command_path_forward_range,
    command_path_lateral_range,
    command_path_min_speed,
    command_path_reference_xy=None,
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
    params["command_path_reference_xy"] = command_path_reference_xy
    params["command_path_forward_range"] = forward_range
    params["command_path_lateral_range"] = lateral_range
    params["command_path_min_speed"] = min_speed
    print(
        "[INFO] Command-path obstacle spawn: "
        f"count={count}, forward={forward_range}, lateral={lateral_range}, min_speed={min_speed:.2f}, "
        f"reference={command_path_reference_xy}"
    )


def _build_teacher_policy(env, obs, agent_cfg, device: str):
    """Instantiate the rule-based steering teacher for direct play/evaluation."""
    teacher_cfg = getattr(agent_cfg, "teacher", None)
    if teacher_cfg is None:
        raise ValueError(
            "--teacher_steering requires a distillation runner config with a 'teacher' model config."
        )
    if "teacher" not in obs.keys():
        raise ValueError(
            "--teacher_steering requires an environment exposing a 'teacher' observation group. "
            "Use a distillation play task such as Navigation-Distill-Go2w-Play-v0."
        )
    if args_cli.locomotion_checkpoint is None:
        raise ValueError("--teacher_steering requires --locomotion_checkpoint for the frozen LLC.")

    teacher_cfg_dict = copy.deepcopy(teacher_cfg.to_dict())
    teacher_class = resolve_callable(teacher_cfg_dict.pop("class_name"))
    teacher = teacher_class(obs, {"teacher": ["teacher"]}, "teacher", env.num_actions, **teacher_cfg_dict)
    teacher = teacher.to(device)
    teacher.eval()
    _load_teacher_locomotion_checkpoint(teacher, args_cli.locomotion_checkpoint, device)
    return teacher


def _compute_teacher_debug_line(obs, teacher, env_index: int = 0) -> str:
    """Summarize the teacher command correction and nearest obstacle geometry."""
    teacher_obs = obs["teacher"]
    obstacle_positions = teacher_obs[
        :, teacher.obstacle_obs_start : teacher.obstacle_obs_start + teacher.obstacle_obs_dim
    ].view(teacher_obs.shape[0], -1, 2)
    obstacle_positions = obstacle_positions * teacher.obstacle_max_distance

    command = teacher.last_base_command
    cmd_xy = command[:, :2]
    cmd_speed = cmd_xy.norm(dim=1)

    cmd_dir = torch.zeros_like(cmd_xy)
    moving = cmd_speed > teacher.min_command_speed
    cmd_dir[moving] = cmd_xy[moving] / cmd_speed[moving].unsqueeze(1)
    cmd_dir[~moving, 0] = 1.0

    obs_x = obstacle_positions[..., 0]
    obs_y = obstacle_positions[..., 1]
    valid = obstacle_positions.abs().sum(dim=-1) > 1.0e-6
    distance = torch.sqrt(torch.clamp(obs_x.square() + obs_y.square(), min=1.0e-6))
    forward = (obstacle_positions * cmd_dir.unsqueeze(1)).sum(dim=-1)
    lateral = cmd_dir[:, 0].unsqueeze(1) * obs_y - cmd_dir[:, 1].unsqueeze(1) * obs_x

    valid_distance = torch.where(valid, distance, torch.full_like(distance, float("inf")))
    closest_idx = torch.argmin(valid_distance, dim=1)
    batch_indices = torch.arange(valid_distance.shape[0], device=valid_distance.device)
    closest_dist = valid_distance[batch_indices, closest_idx]
    closest_forward = forward[batch_indices, closest_idx]
    closest_lateral = lateral[batch_indices, closest_idx]

    base_cmd = teacher.last_base_command[env_index]
    guide_cmd = teacher.last_guidance_command[env_index]
    delta_cmd = teacher.last_delta_command[env_index]
    adjusted_cmd = teacher.last_adjusted_command[env_index]
    gap_width = teacher.last_gap_width[env_index]
    gap_turn_need = teacher.last_gap_turn_need[env_index]
    gap_blocked = teacher.last_gap_blocked[env_index]
    turn_side = teacher.last_turn_side[env_index]
    return (
        f"[TEACHER] env={env_index} "
        f"base=({base_cmd[0]:+.2f},{base_cmd[1]:+.2f},{base_cmd[2]:+.2f}) "
        f"guide=({guide_cmd[0]:+.2f},{guide_cmd[1]:+.2f},{guide_cmd[2]:+.2f}) "
        f"delta=(vx {delta_cmd[0]:+.2f}, vy {delta_cmd[1]:+.2f}, yaw {delta_cmd[2]:+.2f}) "
        f"adj=({adjusted_cmd[0]:+.2f},{adjusted_cmd[1]:+.2f},{adjusted_cmd[2]:+.2f}) "
        f"closest=(dist {closest_dist[env_index]:.2f} m, fwd {closest_forward[env_index]:+.2f}, lat {closest_lateral[env_index]:+.2f}) "
        f"gap=(w {gap_width:.2f}, turn {gap_turn_need:.2f}, blocked {gap_blocked:.2f}, commit {turn_side:+.0f})"
    )


def _resolve_play_seed(args_cli, default_seed: int | None) -> int:
    if args_cli.seed is not None:
        return args_cli.seed
    base_seed = default_seed if default_seed is not None else 0
    return (base_seed + random.SystemRandom().randrange(1, 2_147_483_647)) % 2_147_483_647


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with fixed velocity commands."""
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_seed = _resolve_play_seed(args_cli, agent_cfg.seed)
    agent_cfg.seed = env_seed
    env_cfg.seed = env_seed
    print(f"[INFO] Play seed: {env_seed}")
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    _override_play_obstacle_count(env_cfg, args_cli.num_obstacles)
    _override_play_command_path_spawn(
        env_cfg,
        args_cli.command_path_obstacles,
        args_cli.command_path_forward_range,
        args_cli.command_path_lateral_range,
        args_cli.command_path_min_speed,
        None if args_cli.random_commands else (args_cli.cmd_vx, args_cli.cmd_vy),
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

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    teacher = None

    if args_cli.teacher_steering:
        teacher = _build_teacher_policy(env, obs, agent_cfg, env.unwrapped.device)
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
                if args_cli.teacher_debug_interval > 0 and step_count % args_cli.teacher_debug_interval == 0:
                    print(_compute_teacher_debug_line(obs, teacher, env_index=0))
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
