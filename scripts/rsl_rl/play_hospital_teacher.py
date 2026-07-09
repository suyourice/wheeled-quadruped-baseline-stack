# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained hospital-teacher checkpoint with corridor/debug markers."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time

from isaaclab.app import AppLauncher

from checkpoint_utils import configure_frozen_llc_action  # isort: skip

parser = argparse.ArgumentParser(description="Play a hospital teacher checkpoint.")
parser.add_argument("--task", type=str, default="Nav-Hospital-Teacher-Go2w-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--locomotion_checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--env_index", type=int, default=0, help="Environment index used for camera and markers.")
parser.add_argument(
    "--hospital_curriculum_iteration_offset",
    type=int,
    default=None,
    help="Curriculum iteration to play. Defaults to the number in model_<N>.pt.",
)
parser.add_argument("--camera_height", type=float, default=10.0)
parser.add_argument("--camera_mode", type=str, default="topdown", choices=("chase", "topdown"))
parser.add_argument("--camera_distance", type=float, default=4.5)
parser.add_argument("--camera_target_ahead", type=float, default=1.2)
parser.add_argument(
    "--follow_camera",
    action="store_true",
    default=False,
    help="Continuously reset the camera to follow the robot. Off by default so mouse camera control works.",
)
parser.add_argument("--print_interval", type=int, default=120)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument(
    "--zero_policy",
    action="store_true",
    default=False,
    help="Skip checkpoint loading and play zero HLC actions. Useful for visual reset/obstacle smoke tests.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--show_markers", action="store_true", default=True)
parser.add_argument("--hide_markers", action="store_false", dest="show_markers")
parser.add_argument(
    "--marker_envs",
    choices=("all", "selected"),
    default="all",
    help="Draw play markers/labels for all envs or only --env_index. Camera still starts at --env_index.",
)
parser.add_argument(
    "--label_max_per_env",
    type=int,
    default=40,
    help="Maximum live obstacle labels per displayed env.",
)
parser.add_argument(
    "--show_centerline",
    action="store_true",
    default=False,
    help="Also draw the compact centerline observation. Hidden by default to keep visual smoke uncluttered.",
)
parser.add_argument("--large_markers", action="store_true", default=False)
parser.add_argument("--video", action="store_true", default=False, help="Record viewport video during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--play_name",
    "--play-name",
    dest="play_name",
    type=str,
    default=None,
    help="Name for this play run. Saves nav_debug.log and nav_debug_env0.png to logs/nav_play/<name>/.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import subprocess

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_from_angle_axis
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
import go2w.tasks  # noqa: F401
from play_nav_debug import (  # isort: skip
    NAV_LIVE_LABEL_INTERVAL,
    NAV_LIVE_LABEL_SCALE,
    NAV_LIVE_LABEL_MAX,
    _LiveObstacleLabelDrawer,
    _get_hospital_live_annotations,
    _print_nav_obstacle_label_log,
)

installed_version = metadata.version("rsl-rl-lib")


class _TeeStream:
    """Mirrors writes to multiple streams (stdout + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def __getattr__(self, name):
        return getattr(self._streams[0], name)


def _infer_iteration_from_checkpoint(path: str) -> int:
    match = re.search(r"model_(\d+)\.pt$", os.path.basename(path))
    if match is None:
        raise ValueError(
            "Could not infer hospital curriculum iteration from checkpoint name. "
            "Pass --hospital_curriculum_iteration_offset explicitly."
        )
    return int(match.group(1))


def _apply_hospital_curriculum_offset(env_cfg: ManagerBasedRLEnvCfg, iteration_offset: int) -> None:
    if iteration_offset < 0:
        raise ValueError("--hospital_curriculum_iteration_offset must be non-negative.")

    reset_obstacles = getattr(getattr(env_cfg, "events", None), "reset_obstacles", None)
    params = getattr(reset_obstacles, "params", None)
    if not isinstance(params, dict) or "curriculum_iteration_offset" not in params:
        print(
            "[INFO] reset_obstacles has no 'curriculum_iteration_offset' — skipping curriculum offset "
            "(play env uses a fixed layout, no curriculum needed)."
        )
        return
    params["curriculum_iteration_offset"] = int(iteration_offset)
    print(f"[INFO] Hospital curriculum iteration offset: {iteration_offset}")


def _selected_env(base_env) -> int:
    return max(0, min(int(args_cli.env_index), base_env.num_envs - 1))


def _marker_env_indices(base_env) -> list[int]:
    if args_cli.marker_envs == "selected":
        return [_selected_env(base_env)]
    return list(range(base_env.num_envs))


def _line_marker_transforms(points: torch.Tensor, z_offset: float = 0.10):
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


def _place_ward_junction_markers(base_env) -> list[VisualizationMarkers]:
    """Colored sphere markers at T-junction inner corners for geometry debugging.

    Works for both the isolated ward corridor and the full hospital floor tasks.
    RED   = suspect inner corner (wall box overlap, potential phantom contact)
    YELLOW = reference outer corner

    Prints a legend to the console so the user can identify which sphere is which.
    """
    if not args_cli.show_markers:
        return []
    task = args_cli.task or ""

    if "Floor" in task or "floor" in task:
        from go2w.tasks.manager_based.go2w.mdp.navigation.hospital.floor import (
            HOSPITAL_FLOOR_CORRIDOR_WIDTH,
            HOSPITAL_FLOOR_LEG_LENGTH,
            HOSPITAL_FLOOR_WALL_THICKNESS,
        )
        hw = HOSPITAL_FLOOR_CORRIDOR_WIDTH * 0.5
        L = HOSPITAL_FLOOR_LEG_LENGTH
        wt = HOSPITAL_FLOOR_WALL_THICKNESS
        # (local_x, local_y, label, color)
        named_pts = [
            (2.0 * L - hw, hw, "floor-LEFT-inner (suspect, x={:.1f} y={:.1f})".format(2.0 * L - hw, hw), (1.0, 0.05, 0.0)),
            (2.0 * L + hw, hw, "floor-RIGHT-outer (ref,     x={:.1f} y={:.1f})".format(2.0 * L + hw, hw), (1.0, 0.90, 0.0)),
        ]
    elif "Ward" in task or "ward" in task:
        from go2w.tasks.manager_based.go2w.cfg.hospital.env import (
            HOSPITAL_WARD_CORRIDOR_WIDTH,
            HOSPITAL_WARD_LEG_LENGTH,
            HOSPITAL_WARD_WALL_THICKNESS,
        )
        hw = HOSPITAL_WARD_CORRIDOR_WIDTH * 0.5
        L = HOSPITAL_WARD_LEG_LENGTH
        wt = HOSPITAL_WARD_WALL_THICKNESS
        B1, B2 = L, 2.0 * L
        named_pts = [
            (B1 - hw, hw, "B1-LEFT  (suspect, x={:.1f} y={:.1f})".format(B1 - hw, hw), (1.0, 0.05, 0.0)),
            (B1 + hw, hw, "B1-RIGHT (ref,     x={:.1f} y={:.1f})".format(B1 + hw, hw), (1.0, 0.90, 0.0)),
            (B2 - hw, hw, "B2-LEFT  (suspect, x={:.1f} y={:.1f})".format(B2 - hw, hw), (1.0, 0.05, 0.0)),
            (B2 + hw, hw, "B2-RIGHT (ref,     x={:.1f} y={:.1f})".format(B2 + hw, hw), (1.0, 0.90, 0.0)),
        ]
    else:
        return []

    origin = base_env.scene.env_origins[_selected_env(base_env), :3].clone()
    origin[2] = 0.0
    radius = 0.20 + wt * 0.5

    print("[DEBUG] Junction markers (all coords are local to env origin):")
    placed: list[VisualizationMarkers] = []
    for i, (lx, ly, label, color) in enumerate(named_pts):
        world_pt = torch.tensor([[lx, ly, 0.6]], dtype=torch.float32, device=base_env.device) + origin.unsqueeze(0)
        m = _make_sphere_marker(f"/Visuals/JunctionMarker_{i}", radius, color)
        m.visualize(translations=world_pt)
        placed.append(m)
        color_name = "RED   " if color[1] < 0.5 else "YELLOW"
        print(f"  [{color_name}] {label}")

    return placed


def _init_markers() -> dict[str, VisualizationMarkers]:
    if not args_cli.show_markers:
        return {}
    goal_radius = 0.10 if not args_cli.large_markers else 0.28
    final_radius = 0.09 if not args_cli.large_markers else 0.24
    start_radius = 0.08 if not args_cli.large_markers else 0.22
    markers = {
        "robot": _make_sphere_marker("/Visuals/HospitalTeacherRobot", start_radius, (1.0, 1.0, 1.0)),
        "start": _make_sphere_marker("/Visuals/HospitalTeacherStart", start_radius, (0.15, 0.90, 1.0)),
        "current_goal": _make_sphere_marker("/Visuals/HospitalTeacherCurrentGoal", goal_radius, (0.0, 0.9, 0.10)),
        "final_goal": _make_sphere_marker("/Visuals/HospitalTeacherFinalGoal", final_radius, (1.0, 0.05, 0.02)),
        # actor footprint boxes removed — obstacle geometry + text labels are sufficient
        "path": _make_line_marker("/Visuals/HospitalTeacherPath", 0.03, (1.0, 0.85, 0.0)),
    }
    if args_cli.show_centerline:
        markers["centerline"] = _make_line_marker("/Visuals/HospitalTeacherCenterline", 0.018, (0.0, 0.85, 1.0))
    return markers


def _update_markers(base_env, markers: dict[str, VisualizationMarkers]) -> None:
    if not markers:
        return
    env_indices = _marker_env_indices(base_env)
    if not env_indices:
        return
    device = base_env.device
    rows = torch.tensor(env_indices, dtype=torch.long, device=device)

    robot = base_env.scene["robot"].data.root_pos_w[rows, :3].clone()
    robot[:, 2] += 1.05
    markers["robot"].visualize(translations=robot)

    start = base_env._go2w_start_pos_w[rows, :3].clone()
    start[:, 2] += 0.85
    markers["start"].visualize(translations=start)

    goal = base_env._go2w_goal_pos_w[rows, :3].clone()
    goal[:, 2] += 0.85
    markers["current_goal"].visualize(translations=goal)

    final_goals = []
    path_positions = []
    path_orientations = []
    path_scales = []
    center_positions = []
    center_orientations = []
    center_scales = []

    _centerline_local_buf = getattr(base_env, "_go2w_structured_corridor_centerline_local", None)
    _centerline_count_buf = getattr(base_env, "_go2w_structured_corridor_centerline_count", None)

    for env_index in env_indices:
        path_count = int(base_env._go2w_navigation_path_count[env_index].item())
        final_idx = max(0, path_count - 1)
        final_goal = base_env._go2w_navigation_path_w[env_index, final_idx, :3].clone()
        final_goal[2] += 0.95
        final_goals.append(final_goal)

        path_full = base_env._go2w_navigation_path_w[env_index, :path_count, :3]
        _PATH_MARKER_MAX = 32
        if path_count > _PATH_MARKER_MAX:
            idx = torch.linspace(0, path_count - 1, _PATH_MARKER_MAX, device=device).long()
            path = path_full[idx]
        else:
            path = path_full
        path_markers = _line_marker_transforms(path, z_offset=0.18)
        if path_markers is not None:
            positions, orientations, scales = path_markers
            path_positions.append(positions)
            path_orientations.append(orientations)
            path_scales.append(scales)

        if _centerline_local_buf is not None and "centerline" in markers:
            origin = base_env._go2w_structured_corridor_start_xy[env_index]
            centerline_count = (
                int(_centerline_count_buf[env_index].item())
                if _centerline_count_buf is not None
                else _centerline_local_buf.shape[1]
            )
            centerline_local = _centerline_local_buf[env_index, :centerline_count]
            centerline = torch.zeros(centerline_count, 3, device=device)
            centerline[:, :2] = centerline_local + origin.unsqueeze(0)
            centerline[:, 2] = base_env.scene["robot"].data.root_pos_w[env_index, 2]
            centerline_markers = _line_marker_transforms(centerline, z_offset=0.10)
            if centerline_markers is not None:
                positions, orientations, scales = centerline_markers
                center_positions.append(positions)
                center_orientations.append(orientations)
                center_scales.append(scales)

    if final_goals:
        markers["final_goal"].visualize(translations=torch.stack(final_goals, dim=0))
    if path_positions:
        markers["path"].visualize(
            translations=torch.cat(path_positions, dim=0),
            orientations=torch.cat(path_orientations, dim=0),
            scales=torch.cat(path_scales, dim=0),
        )
    if center_positions:
        markers["centerline"].visualize(
            translations=torch.cat(center_positions, dim=0),
            orientations=torch.cat(center_orientations, dim=0),
            scales=torch.cat(center_scales, dim=0),
        )



def _set_camera(base_env) -> None:
    if getattr(args_cli, "headless", False):
        return
    env_index = _selected_env(base_env)
    robot_pos = base_env.scene["robot"].data.root_pos_w[env_index].detach().cpu()
    if args_cli.camera_mode == "topdown":
        eye = (float(robot_pos[0]), float(robot_pos[1]) - 0.5, float(args_cli.camera_height))
        target = (float(robot_pos[0]), float(robot_pos[1]), 0.0)
    else:
        heading = float(base_env.scene["robot"].data.heading_w[env_index].detach().cpu().item())
        forward_x = math.cos(heading)
        forward_y = math.sin(heading)
        eye = (
            float(robot_pos[0]) - forward_x * args_cli.camera_distance,
            float(robot_pos[1]) - forward_y * args_cli.camera_distance,
            float(robot_pos[2]) + float(args_cli.camera_height),
        )
        target = (
            float(robot_pos[0]) + forward_x * args_cli.camera_target_ahead,
            float(robot_pos[1]) + forward_y * args_cli.camera_target_ahead,
            float(robot_pos[2]) + 0.35,
        )
    base_env.sim.set_camera_view(eye=eye, target=target)


def _print_state(base_env, step_count: int) -> None:
    if args_cli.print_interval <= 0 or step_count % args_cli.print_interval != 0:
        return
    env_index = _selected_env(base_env)
    robot_xy = base_env.scene["robot"].data.root_pos_w[env_index, :2]
    goal_xy = base_env._go2w_goal_pos_w[env_index, :2]
    final_idx = int((base_env._go2w_navigation_path_count[env_index] - 1).clamp(min=0).item())
    final_s = float(base_env._go2w_navigation_path_s[env_index, final_idx].item())
    progress_s = float(base_env._go2w_navigation_path_progress_s[env_index].item())
    actor_count = int(getattr(base_env, "_go2w_hospital_actor_count", torch.zeros(1, device=base_env.device))[env_index].item())
    phase = int(getattr(base_env, "_go2w_hospital_phase_id", torch.zeros(1, device=base_env.device))[env_index].item())
    layout = int(getattr(base_env, "_go2w_hospital_layout_id", torch.zeros(1, device=base_env.device))[env_index].item())
    goal_dist = float(torch.linalg.norm(goal_xy - robot_xy).item())
    print(
        "[HOSPITAL-PLAY] "
        f"step={step_count} env={env_index} phase={phase} layout={layout} actors={actor_count} "
        f"progress={progress_s:.2f}/{final_s:.2f} goal_dist={goal_dist:.2f} "
        f"robot=({float(robot_xy[0]):+.2f},{float(robot_xy[1]):+.2f})"
    )


def _print_robot_runtime_state(base_env) -> None:
    env_index = _selected_env(base_env)
    root_pos = base_env.scene["robot"].data.root_pos_w[env_index, :3].detach().cpu()
    root_quat = base_env.scene["robot"].data.root_quat_w[env_index, :4].detach().cpu()
    print(
        "[INFO] Robot runtime state: "
        f"env={env_index} pos=({float(root_pos[0]):+.3f},{float(root_pos[1]):+.3f},{float(root_pos[2]):+.3f}) "
        f"quat=({float(root_quat[0]):+.3f},{float(root_quat[1]):+.3f},"
        f"{float(root_quat[2]):+.3f},{float(root_quat[3]):+.3f})"
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs
    terrain_generator = getattr(getattr(env_cfg.scene, "terrain", None), "terrain_generator", None)
    if terrain_generator is not None:
        terrain_generator.num_rows = 1
        terrain_generator.num_cols = max(1, int(args_cli.num_envs))
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    env_cfg.seed = agent_cfg.seed if args_cli.seed is None else args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    print(
        "[INFO] Hospital teacher play visualization: "
        f"use_fabric={env_cfg.sim.use_fabric} replicate_physics={getattr(env_cfg.scene, 'replicate_physics', None)}"
    )

    # play output directory setup
    _out_dir: str | None = None
    _out_log_file = None
    if args_cli.play_name:
        _out_dir = os.path.abspath(os.path.join("logs", "nav_play", args_cli.play_name))
        os.makedirs(_out_dir, exist_ok=True)
        _out_log_file = open(os.path.join(_out_dir, "nav_debug.log"), "w", buffering=1)
        sys.stdout = _TeeStream(sys.__stdout__, _out_log_file)
        os.environ["GO2W_NAV_DEBUG"] = "1"
        os.environ["GO2W_NAV_DEBUG_ENV"] = str(args_cli.env_index)
        print(f"[INFO] Play output dir: {_out_dir}")

    if args_cli.checkpoint is None and not args_cli.zero_policy:
        raise ValueError("Pass --checkpoint for policy play, or use --zero_policy for visual reset smoke tests.")

    checkpoint_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint is not None else None
    iteration_offset = (
        (1000 if checkpoint_path is None else _infer_iteration_from_checkpoint(checkpoint_path))
        if args_cli.hospital_curriculum_iteration_offset is None
        else args_cli.hospital_curriculum_iteration_offset
    )
    env_cfg.log_dir = os.path.dirname(checkpoint_path) if checkpoint_path is not None else os.path.abspath("logs/nav_play")
    _apply_hospital_curriculum_offset(env_cfg, iteration_offset)
    configure_frozen_llc_action(env_cfg, args_cli.locomotion_checkpoint, args_cli.task)

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if args_cli.video:
        video_dir = _out_dir if _out_dir else os.path.join("logs", "videos", "hospital_teacher")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)  # type: ignore[arg-type]
    obs = env.get_observations()

    if args_cli.zero_policy:
        policy = None
    else:
        if agent_cfg.class_name != "OnPolicyRunner":
            raise ValueError(f"Hospital teacher play expects OnPolicyRunner, got {agent_cfg.class_name!r}.")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
    base_env = env.unwrapped
    _print_robot_runtime_state(base_env)
    markers = _init_markers()
    _update_markers(base_env, markers)
    _place_ward_junction_markers(base_env)
    _set_camera(base_env)

    marker_env_indices = _marker_env_indices(base_env)
    label_drawer = _LiveObstacleLabelDrawer(
        scale=NAV_LIVE_LABEL_SCALE,
        max_labels=max(0, int(args_cli.label_max_per_env)) * max(1, len(marker_env_indices)),
    )
    if label_drawer.enabled:
        print(
            f"[INFO] Live obstacle labels enabled "
            f"(scale={NAV_LIVE_LABEL_SCALE:.3f}, max={label_drawer.max_labels})"
        )

    def _save_plot_and_log():
        sys.stdout = sys.__stdout__
        if _out_log_file is not None:
            _out_log_file.flush()
        if _out_dir is not None:
            _nav_log = os.path.join(_out_dir, "nav_debug.log")
            _plot_cmd = [sys.executable, "scripts/plot_nav_debug.py", _nav_log, "--output_dir", _out_dir]
            sys.__stdout__.write("[INFO] Generating nav debug plot ...\n")
            sys.__stdout__.flush()
            result = subprocess.run(_plot_cmd, capture_output=True, text=True)
            if result.stdout:
                sys.__stdout__.write(result.stdout.strip() + "\n")
            if result.returncode != 0 and result.stderr:
                sys.__stdout__.write(f"[WARN] plot_nav_debug: {result.stderr.strip()}\n")
        if _out_log_file is not None:
            _out_log_file.close()

    def _finalize():
        label_drawer.clear()
        env.close()

    if args_cli.zero_policy:
        print("[INFO] Playing hospital teacher with zero policy for visual reset smoke.")
    else:
        print(f"[INFO] Playing hospital teacher checkpoint: {checkpoint_path}")
    step_count = 0
    dt = base_env.step_dt
    try:
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = (
                    torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                    if policy is None
                    else policy(obs)
                )
                obs, _, dones, _ = env.step(actions)
                if policy is not None and hasattr(policy, "reset"):
                    policy.reset(dones)
            _update_markers(base_env, markers)
            if args_cli.follow_camera:
                _set_camera(base_env)
            _print_state(base_env, step_count)
            if args_cli.print_interval > 0 and step_count % args_cli.print_interval == 0:
                _print_nav_obstacle_label_log(base_env, step_count, args_cli.env_index)
            if step_count % NAV_LIVE_LABEL_INTERVAL == 0:
                label_drawer.update_many(base_env, _marker_env_indices(base_env))
            step_count += 1

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt — saving logs before exit.")
    finally:
        _save_plot_and_log()
        _finalize()


if __name__ == "__main__":
    main()
    simulation_app.close()
