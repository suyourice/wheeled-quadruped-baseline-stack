# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visual sanity check for the hospital-teacher training layout.

This script creates the real ``Nav-Hospital-Teacher-Go2w-v0`` environment, forces
one curriculum phase/layout, and overlays start/current-goal/final-goal/path
markers.  It is intentionally separate from train.py so the training task can
stay lean.
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize the hospital teacher training environment.")
parser.add_argument("--task", type=str, default="Nav-Hospital-Teacher-Go2w-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_index", type=int, default=0, help="Environment index used for path/console diagnostics.")
parser.add_argument(
    "--layout",
    type=str,
    default="serpentine_corridor",
    choices=("straight_corridor", "l_corridor", "t_corridor", "serpentine_corridor"),
)
parser.add_argument("--phase", type=int, default=0, help="Hospital width/actor phase index to show.")
parser.add_argument("--locomotion_checkpoint", type=str, required=True)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--camera_height", type=float, default=18.0, help="Top-down camera height above the selected env.")
parser.add_argument(
    "--drive_mode",
    type=str,
    default="zero",
    choices=("zero", "path", "forward", "random"),
    help="Action source for visual checks. Use 'path' to make the robot move toward the rolling goal.",
)
parser.add_argument("--drive_speed", type=float, default=0.45, help="Forward speed used by path/forward drive modes.")
parser.add_argument("--max_lateral_speed", type=float, default=0.35, help="Lateral speed clamp for path/random drive modes.")
parser.add_argument("--yaw_gain", type=float, default=1.8, help="Yaw-rate gain for path drive mode.")
parser.add_argument("--max_yaw_rate", type=float, default=0.9, help="Yaw-rate clamp for path/random drive modes.")
parser.add_argument("--print_interval", type=int, default=120, help="Step interval for drive diagnostics. Set 0 to disable.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_from_angle_axis
from isaaclab_tasks.utils import parse_env_cfg

from checkpoint_utils import configure_frozen_llc_action

import go2w.tasks  # noqa: F401
from go2w.tasks.manager_based.go2w.mdp.navigation.hospital.specs import (
    CURRICULUM_STEPS_PER_ITERATION,
    HOSPITAL_TRAIN_OBSTACLE_LABELS,
    HOSPITAL_TRAIN_WIDTH_SCHEDULE,
)


def _phase_schedule(phase: int) -> tuple[int, float, int]:
    """Return a validated hospital curriculum phase tuple."""
    if phase < 0 or phase >= len(HOSPITAL_TRAIN_WIDTH_SCHEDULE):
        raise ValueError(
            f"--phase must be in [0, {len(HOSPITAL_TRAIN_WIDTH_SCHEDULE) - 1}], got {phase}."
        )
    return HOSPITAL_TRAIN_WIDTH_SCHEDULE[phase]


def _line_marker_transforms(points: torch.Tensor, z_offset: float = 0.10):
    """Return cylinder marker transforms for a connected 3D polyline."""
    if points.shape[0] < 2:
        return None
    points = points.clone()
    points[:, 2] += z_offset
    start = points[:-1]
    end = points[1:]
    direction = end - start
    lengths = direction.norm(dim=-1)
    valid = lengths > 0.03
    if not bool(valid.any()):
        return None

    start = start[valid]
    end = end[valid]
    direction = direction[valid]
    lengths = lengths[valid]
    positions = (start + end) * 0.5

    direction_norm = direction / lengths.unsqueeze(-1).clamp(min=1.0e-6)
    default_axis = torch.zeros_like(direction_norm)
    default_axis[:, 2] = 1.0
    rotation_axis = torch.linalg.cross(default_axis, direction_norm, dim=-1)
    rotation_axis_norm = rotation_axis.norm(dim=-1)
    fallback_axis = torch.zeros_like(rotation_axis)
    fallback_axis[:, 0] = 1.0
    rotation_axis = torch.where(
        (rotation_axis_norm > 1.0e-6).unsqueeze(-1),
        rotation_axis / rotation_axis_norm.unsqueeze(-1).clamp(min=1.0e-6),
        fallback_axis,
    )
    cos_angle = (default_axis * direction_norm).sum(dim=-1).clamp(-1.0, 1.0)
    orientations = quat_from_angle_axis(torch.acos(cos_angle), rotation_axis)
    scales = torch.ones(positions.shape[0], 3, device=positions.device, dtype=positions.dtype)
    scales[:, 2] = lengths
    return positions, orientations, scales


def _make_sphere_marker(prim_path: str, radius: float, color: tuple[float, float, float]) -> VisualizationMarkers:
    return VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
            },
        )
    )


def _make_line_marker(prim_path: str, radius: float, color: tuple[float, float, float]) -> VisualizationMarkers:
    return VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "line": sim_utils.CylinderCfg(
                    radius=radius,
                    height=1.0,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=1.0),
                ),
            },
        )
    )


def _set_camera(base_env, env_index: int) -> None:
    """Place the viewport above the selected env cell."""
    env_index = max(0, min(env_index, base_env.num_envs - 1))
    robot_pos = base_env.scene["robot"].data.root_pos_w[env_index].detach().cpu()
    eye = (float(robot_pos[0]), float(robot_pos[1]) - 0.5, float(args_cli.camera_height))
    target = (float(robot_pos[0]), float(robot_pos[1]), 0.0)
    base_env.sim.set_camera_view(eye=eye, target=target)


def _print_summary(base_env, env_index: int) -> None:
    """Print layout, phase, path, and active actor details for one environment."""
    env_index = max(0, min(env_index, base_env.num_envs - 1))
    schedule = _phase_schedule(args_cli.phase)
    path_count = int(base_env._go2w_navigation_path_count[env_index].item())
    final_idx = max(0, path_count - 1)
    path_len = float(base_env._go2w_navigation_path_s[env_index, final_idx].item())
    start = base_env._go2w_start_pos_w[env_index, :3].detach().cpu().tolist()
    current_goal = base_env._go2w_goal_pos_w[env_index, :3].detach().cpu().tolist()
    final_goal = base_env._go2w_navigation_path_w[env_index, final_idx, :3].detach().cpu().tolist()
    actor_count = int(getattr(base_env, "_go2w_hospital_actor_count", torch.zeros(1, device=base_env.device))[env_index].item())

    print("\n[HOSPITAL-VIS]")
    print(f"  task:          {args_cli.task}")
    print(f"  env_index:     {env_index}")
    print(f"  layout:        {args_cli.layout}")
    print(f"  phase:         {args_cli.phase}")
    print(f"  drive mode:    {args_cli.drive_mode}")
    print(f"  width:         {schedule[1]:.2f} m")
    print(f"  active actors: {actor_count} / {int(schedule[2])} max")
    print(f"  path length:   {path_len:.2f} m")
    print(f"  start xyz:     [{start[0]:+.2f}, {start[1]:+.2f}, {start[2]:+.2f}]")
    print(f"  current goal:  [{current_goal[0]:+.2f}, {current_goal[1]:+.2f}, {current_goal[2]:+.2f}]")
    print(f"  final goal:    [{final_goal[0]:+.2f}, {final_goal[1]:+.2f}, {final_goal[2]:+.2f}]")

    active = getattr(base_env, "_go2w_obstacle_active_mask", None)
    if active is None:
        return
    active_ids = active[env_index].nonzero(as_tuple=False).flatten().detach().cpu().tolist()
    if not active_ids:
        print("  actors:        none")
        return
    print("  actors:")
    for idx in active_ids:
        obstacle = base_env.scene[f"obstacle_{idx}"]
        pos = obstacle.data.root_pos_w[env_index, :3].detach().cpu().tolist()
        width = float(base_env._go2w_obstacle_width[env_index, idx].item())
        depth = float(base_env._go2w_obstacle_depth[env_index, idx].item())
        height = float(base_env._go2w_obstacle_height[env_index, idx].item())
        yaw = float(base_env._go2w_obstacle_yaw[env_index, idx].item())
        label = HOSPITAL_TRAIN_OBSTACLE_LABELS[idx]
        print(
            f"    {idx:02d} {label:<20} "
            f"pos=[{pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}] "
            f"size=({width:.2f}, {depth:.2f}, {height:.2f}) yaw={math.degrees(yaw):+.1f} deg"
        )


def _compute_drive_actions(base_env) -> torch.Tensor:
    """Return simple visual-check HLC actions for all environments."""
    actions = torch.zeros(base_env.num_envs, 3, device=base_env.device)
    mode = args_cli.drive_mode
    if mode == "zero":
        return actions
    if mode == "forward":
        actions[:, 0] = args_cli.drive_speed
        return actions
    if mode == "random":
        actions[:, 0] = (torch.rand(base_env.num_envs, device=base_env.device) * 2.0 - 1.0) * args_cli.drive_speed
        actions[:, 1] = (
            torch.rand(base_env.num_envs, device=base_env.device) * 2.0 - 1.0
        ) * args_cli.max_lateral_speed
        actions[:, 2] = (
            torch.rand(base_env.num_envs, device=base_env.device) * 2.0 - 1.0
        ) * args_cli.max_yaw_rate
        return actions

    robot = base_env.scene["robot"]
    robot_xy = robot.data.root_pos_w[:, :2]
    goal_xy = base_env._go2w_goal_pos_w[:, :2]
    rel_w = goal_xy - robot_xy
    heading = robot.data.heading_w
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    rel_x_b = cos_h * rel_w[:, 0] + sin_h * rel_w[:, 1]
    rel_y_b = -sin_h * rel_w[:, 0] + cos_h * rel_w[:, 1]
    distance = torch.hypot(rel_x_b, rel_y_b).clamp(min=1.0e-4)
    bearing = torch.atan2(rel_y_b, rel_x_b)

    speed_scale = (distance / 0.9).clamp(0.15, 1.0)
    forward_gate = torch.cos(bearing).clamp(min=0.0, max=1.0)
    actions[:, 0] = args_cli.drive_speed * speed_scale * forward_gate
    actions[:, 1] = (args_cli.drive_speed * 0.65 * torch.sin(bearing)).clamp(
        min=-args_cli.max_lateral_speed,
        max=args_cli.max_lateral_speed,
    )
    actions[:, 2] = (args_cli.yaw_gain * bearing).clamp(min=-args_cli.max_yaw_rate, max=args_cli.max_yaw_rate)
    return actions


def _print_drive_state(base_env, env_index: int, actions: torch.Tensor, step_count: int) -> None:
    """Print compact motion diagnostics for the selected environment."""
    env_index = max(0, min(env_index, base_env.num_envs - 1))
    robot_xy = base_env.scene["robot"].data.root_pos_w[env_index, :2]
    goal_xy = base_env._go2w_goal_pos_w[env_index, :2]
    distance = torch.linalg.norm(goal_xy - robot_xy)
    action = actions[env_index]
    print(
        "[HOSPITAL-VIS-DRIVE] "
        f"step={step_count} env={env_index} "
        f"goal_dist={float(distance.item()):.2f} "
        f"action=({float(action[0].item()):+.2f}, {float(action[1].item()):+.2f}, {float(action[2].item()):+.2f})"
    )


def _update_markers(base_env, env_index: int, markers: dict[str, VisualizationMarkers]) -> None:
    """Refresh start/goal/path/centerline/actor markers."""
    env_index = max(0, min(env_index, base_env.num_envs - 1))
    device = base_env.device
    row = torch.tensor([env_index], device=device)

    start = base_env._go2w_start_pos_w[row, :3].clone()
    start[:, 2] += 0.35
    markers["start"].visualize(translations=start)

    goal = base_env._go2w_goal_pos_w[row, :3].clone()
    goal[:, 2] += 0.35
    markers["current_goal"].visualize(translations=goal)

    path_count = int(base_env._go2w_navigation_path_count[env_index].item())
    final_idx = max(0, path_count - 1)
    final_goal = base_env._go2w_navigation_path_w[row, final_idx, :3].clone()
    final_goal[:, 2] += 0.45
    markers["final_goal"].visualize(translations=final_goal)

    path = base_env._go2w_navigation_path_w[env_index, :path_count, :3]
    path_markers = _line_marker_transforms(path, z_offset=0.18)
    if path_markers is not None:
        positions, orientations, scales = path_markers
        markers["path"].visualize(translations=positions, orientations=orientations, scales=scales)

    centerline_count = int(base_env._go2w_structured_corridor_centerline_count[env_index].item())
    centerline_local = base_env._go2w_structured_corridor_centerline_local[env_index, :centerline_count]
    origin = base_env._go2w_structured_corridor_start_xy[env_index]
    centerline = torch.zeros(centerline_count, 3, device=device)
    centerline[:, :2] = centerline_local + origin.unsqueeze(0)
    centerline[:, 2] = base_env.scene["robot"].data.root_pos_w[env_index, 2]
    centerline_markers = _line_marker_transforms(centerline, z_offset=0.10)
    if centerline_markers is not None:
        positions, orientations, scales = centerline_markers
        markers["centerline"].visualize(translations=positions, orientations=orientations, scales=scales)

    active = getattr(base_env, "_go2w_obstacle_active_mask", None)
    if active is not None:
        active_ids = active[env_index].nonzero(as_tuple=False).flatten()
        if active_ids.numel() > 0:
            actor_pos = torch.stack(
                [base_env.scene[f"obstacle_{int(idx)}"].data.root_pos_w[env_index, :3] for idx in active_ids],
                dim=0,
            )
            actor_pos[:, 2] += 0.20
            markers["actors"].visualize(translations=actor_pos)


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    configure_frozen_llc_action(env_cfg, args_cli.locomotion_checkpoint, args_cli.task)
    env_cfg.commands.base_velocity.debug_vis = True

    env_cfg.scene.num_envs = args_cli.num_envs
    # Visual checks are easiest with one environment, but keep the normal spacing
    # for multiple envs so atlas cells never overlap.
    if args_cli.num_envs == 1:
        env_cfg.scene.env_spacing = 1.0
    env_cfg.events.reset_obstacles.params = {
        **env_cfg.events.reset_obstacles.params,
        "force_layout_kind": args_cli.layout,
    }

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    base_env.common_step_counter = _phase_schedule(args_cli.phase)[0] * CURRICULUM_STEPS_PER_ITERATION
    env.reset()

    markers = {
        "start": _make_sphere_marker("/Visuals/HospitalStart", 0.22, (0.10, 0.45, 1.0)),
        "current_goal": _make_sphere_marker("/Visuals/HospitalCurrentGoal", 0.28, (0.0, 0.9, 0.10)),
        "final_goal": _make_sphere_marker("/Visuals/HospitalFinalGoal", 0.24, (1.0, 0.05, 0.02)),
        "actors": _make_sphere_marker("/Visuals/HospitalActorCenters", 0.10, (1.0, 1.0, 1.0)),
        "path": _make_line_marker("/Visuals/HospitalPath", 0.03, (1.0, 0.85, 0.0)),
        "centerline": _make_line_marker("/Visuals/HospitalCenterline", 0.018, (0.0, 0.85, 1.0)),
    }

    _set_camera(base_env, args_cli.env_index)
    _print_summary(base_env, args_cli.env_index)
    _update_markers(base_env, args_cli.env_index, markers)

    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = _compute_drive_actions(base_env)
            env.step(actions)
            _update_markers(base_env, args_cli.env_index, markers)
            if args_cli.print_interval > 0 and step_count % args_cli.print_interval == 0:
                _print_drive_state(base_env, args_cli.env_index, actions, step_count)
            step_count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
