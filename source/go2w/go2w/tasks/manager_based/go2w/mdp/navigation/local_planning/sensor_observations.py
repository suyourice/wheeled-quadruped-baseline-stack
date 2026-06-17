# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for the Go2-W obstacle avoidance task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

from ...common.debug import fmt_xy, nav_debug_enabled, nav_debug_env_id, nav_debug_interval
from ...common.orientation import quat_yaw_wxyz
from ..goals import ensure_navigation_goal_buffers
from .obstacle_geometry import (
    DEFAULT_OBSTACLE_EFFECTIVE_RADIUS,
    DEFAULT_OBSTACLE_WIDTH,
    DEFAULT_OBSTACLE_DEPTH,
    OBSTACLE_SHAPE_CONE,
    OBSTACLE_SHAPE_CUBOID,
    OBSTACLE_SHAPE_CYLINDER,
    obstacle_active_mask,
    obstacle_risk_radius,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lidar_distances(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_distance: float = 5.0,
) -> torch.Tensor:
    """Compute per-ray LiDAR closeness from a RayCaster sensor.

    Raw LiDAR should represent occupancy/free-space geometry. Steering features
    below provide navigation-oriented summaries, so this term stays simple:
    close obstacle -> 1, far/no hit -> 0.

    Args:
        sensor_cfg: SceneEntityCfg pointing to a RayCaster sensor.
        max_distance: Distance used for normalization and clamping.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    hit_positions = sensor.data.ray_hits_w
    sensor_pos = sensor.data.pos_w
    distances = torch.norm(hit_positions - sensor_pos.unsqueeze(1), dim=-1)
    distances = torch.where(
        torch.isfinite(distances),
        distances,
        torch.full_like(distances, max_distance),
    )
    distances = distances.clamp(min=0.05, max=max_distance)
    return 1.0 - distances / max_distance


def depth_closeness_image(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    data_type: str = "distance_to_image_plane",
    min_depth: float = 0.60,
    max_depth: float = 6.0,
) -> torch.Tensor:
    """Return a normalized depth image for the student policy.

    The output keeps the image layout as (N, H, W). Observation history then
    turns it into (N, T, H, W), which the CNN student treats as T channels.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    depth = sensor.data.output[data_type]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)

    depth = torch.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=max_depth)
    depth = depth.clamp(min=min_depth, max=max_depth)
    return (1.0 - depth / max_depth).clamp(0.0, 1.0)


def depth_closeness_multicam_image(
    env: ManagerBasedRLEnv,
    sensor_cfgs: list,
    data_type: str = "distance_to_image_plane",
    min_depth: float = 0.60,
    max_depth: float = 6.0,
    history_length: int = 3,
) -> torch.Tensor:
    """Multi-camera depth history for the CNN student.

    Reads `len(sensor_cfgs)` depth cameras and maintains a rolling buffer with
    `history_length` timesteps.  Returns (N, n_cams * history_length, H, W).
    Buffer layout: [cam0_t, cam1_t, ..., cam0_t-1, cam1_t-1, ...].
    """
    n_cams = len(sensor_cfgs)
    buf_key = "_mcam_depth_buf"

    # Compute current closeness frame for each camera.
    frames = []
    for sc in sensor_cfgs:
        sensor = env.scene.sensors[sc.name]
        depth = sensor.data.output[data_type]
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        depth = torch.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=max_depth)
        frames.append((1.0 - depth.clamp(min_depth, max_depth) / max_depth).clamp(0.0, 1.0))

    N, H, W = frames[0].shape

    # Lazy init.
    if not hasattr(env, buf_key):
        buf = torch.zeros(N, n_cams * history_length, H, W, device=frames[0].device)
        setattr(env, buf_key, buf)
    buf = getattr(env, buf_key)

    # Reset buffer rows for envs that terminated this step.
    # Guard: termination_manager is not yet available during obs shape probing at init.
    if hasattr(env, "termination_manager"):
        terminated = env.termination_manager.terminated
        if terminated.any():
            buf[terminated] = 0.0

    # Shift history back by one slot (n_cams channels) and insert current frames.
    buf[:, n_cams:] = buf[:, :-n_cams].clone()
    for i, frame in enumerate(frames):
        buf[:, i] = frame

    return buf


def obstacle_polar_depth(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    num_bins: int = 180,
    max_distance: float = 20.0,
    robot_safety_radius: float = 0.30,
    angular_chunk_size: int = 30,
) -> torch.Tensor:
    """Convert privileged obstacle positions to a 180-bin polar closeness map.

    Encodes close=1 and far/empty=0. The familiar 0.30 m box keeps the original
    center-distance encoding; larger or smaller randomized footprints shift the
    encoded distance by their radius difference from that nominal box.

    Each bin covers 360°/num_bins (default 2°). Each obstacle contributes to
    its center-bearing bin, preserving the teacher observation structure that
    the baseline PPO setting learned from.
    Inactive obstacles parked at ≫max_distance contribute 0 closeness via
    the clamp and do not pollute occupied bins.

    Args:
        obstacle_names: List of scene entity names for the obstacle rigid bodies.
        robot_cfg: SceneEntityCfg identifying the robot.
        num_bins: Angular resolution (default 180 → 2° per bin).
        max_distance: Normalization distance; obstacles beyond this map to 0.
        robot_safety_radius: Kept for API compatibility; risk rewards apply it.
        angular_chunk_size: Kept for API compatibility.

    Returns:
        Tensor of shape (num_envs, num_bins) with values in [0, 1].
    """
    if len(obstacle_names) == 0:
        return torch.zeros(env.num_envs, num_bins, device=env.device)

    robot = env.scene[robot_cfg.name]
    robot_pos_w = robot.data.root_pos_w[:, :3]       # (N, 3)
    robot_yaw_quat_w = yaw_quat(robot.data.root_quat_w)  # (N, 4)

    # Stack all obstacle world positions, then apply quat_apply_inverse in one
    # batched call instead of K separate calls.
    obs_pos_all = torch.stack(
        [env.scene[n].data.root_pos_w[:, :3] for n in obstacle_names], dim=1
    )  # (N, K, 3)
    rel_pos_w_all = obs_pos_all - robot_pos_w.unsqueeze(1)  # (N, K, 3)
    N, K = rel_pos_w_all.shape[:2]
    quat_exp = robot_yaw_quat_w.unsqueeze(1).expand(-1, K, -1).reshape(N * K, 4)
    rel_pos_b_flat = quat_apply_inverse(quat_exp, rel_pos_w_all.reshape(N * K, 3))
    rel_xy = rel_pos_b_flat.reshape(N, K, 3)[:, :, :2]  # (N, K, 2)
    center_dists = torch.norm(rel_xy, dim=-1)  # (N, K)
    active = obstacle_active_mask(env, obstacle_names, center_dists, max_distance)
    risk_radii = obstacle_risk_radius(env, obstacle_names, center_dists)
    nominal_radius = DEFAULT_OBSTACLE_EFFECTIVE_RADIUS + float(
        getattr(env, "_go2w_obstacle_radius_margin", 0.0)
    )
    radius_delta = risk_radii - nominal_radius
    encoded_dists = (center_dists - radius_delta).clamp(min=0.05, max=max_distance)

    angles = torch.atan2(rel_xy[..., 1], rel_xy[..., 0])  # (N, K), range [-π, π]
    bin_idx = ((angles + torch.pi) / (2.0 * torch.pi) * num_bins).long().clamp(0, num_bins - 1)
    dist_map = torch.full((env.num_envs, num_bins), max_distance, device=env.device)
    candidates = torch.where(active, encoded_dists, torch.full_like(encoded_dists, max_distance))
    dist_map.scatter_reduce_(1, bin_idx, candidates, reduce="amin", include_self=True)

    return (1.0 - dist_map / max_distance).clamp(0.0, 1.0)


def _compute_navigation_waypoint_world(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lookahead_distance: float = 1.25,
    goal_snap_distance: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project robot onto start-goal segment, look ahead by lookahead_distance, snap to goal when close."""
    ensure_navigation_goal_buffers(env)
    robot = env.scene[robot_cfg.name]

    start_pos_w = env._go2w_start_pos_w
    goal_pos_w = env._go2w_goal_pos_w
    goal_heading_w = env._go2w_goal_heading_w
    root_pos_w = robot.data.root_pos_w[:, :3]

    if bool(getattr(env, "_go2w_navigation_path_direct_goal", False)) and hasattr(env, "_go2w_navigation_path_w"):
        return goal_pos_w.clone(), goal_heading_w.clone()

    path_vec_w = goal_pos_w[:, :2] - start_pos_w[:, :2]
    path_len = torch.norm(path_vec_w, dim=1, keepdim=True).clamp(min=1.0e-6)
    path_dir_w = path_vec_w / path_len

    rel_from_start_w = root_pos_w[:, :2] - start_pos_w[:, :2]
    progress_along_path = (rel_from_start_w * path_dir_w).sum(dim=1, keepdim=True).clamp(min=0.0)
    progress_along_path = torch.minimum(progress_along_path, path_len)
    waypoint_progress = torch.minimum(progress_along_path + lookahead_distance, path_len)
    waypoint_xy_w = start_pos_w[:, :2] + waypoint_progress * path_dir_w

    waypoint_pos_w = goal_pos_w.clone()
    waypoint_pos_w[:, :2] = waypoint_xy_w

    path_heading_w = torch.atan2(path_dir_w[:, 1], path_dir_w[:, 0])
    remaining_goal_distance = torch.norm(goal_pos_w[:, :2] - root_pos_w[:, :2], dim=1)
    waypoint_heading_w = torch.where(
        remaining_goal_distance <= goal_snap_distance,
        goal_heading_w,
        path_heading_w,
    )
    return waypoint_pos_w, waypoint_heading_w


def local_goal_command_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lookahead_distance: float = 1.25,
    goal_snap_distance: float = 1.0,
    command_min_forward: float = 0.45,
    command_max_lateral: float = 0.85,
    command_max_heading: float = 0.6,
    command_turn_slowdown_heading: float = math.inf,
    command_turn_slowdown_min_forward: float = 0.0,
) -> torch.Tensor:
    """Return rolling local-waypoint command in robot yaw frame as [cmd_x, cmd_y, cmd_heading]."""
    robot = env.scene[robot_cfg.name]
    waypoint_pos_w, waypoint_heading_w = _compute_navigation_waypoint_world(
        env,
        robot_cfg=robot_cfg,
        lookahead_distance=lookahead_distance,
        goal_snap_distance=goal_snap_distance,
    )

    ensure_navigation_goal_buffers(env)
    remaining_goal_distance = torch.norm(env._go2w_goal_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1)
    path_mode = bool(getattr(env, "_go2w_navigation_path_direct_goal", False)) and hasattr(
        env, "_go2w_navigation_path_w"
    )
    if path_mode and hasattr(env, "_go2w_navigation_path_final_distance"):
        remaining_goal_distance = env._go2w_navigation_path_final_distance
        # In path-following mode only snap to final-goal heading/direction when the
        # waypoint tracker has already advanced to the last path segment.  Without
        # this gate the robot would commit to pointing at a potentially obstacle-
        # blocked final waypoint as soon as it enters the 1 m snap radius, even
        # while earlier corners still need to be navigated.
        if (
            hasattr(env, "_go2w_navigation_path_target_index")
            and hasattr(env, "_go2w_navigation_path_count")
        ):
            final_idx = (env._go2w_navigation_path_count - 1).clamp(min=0)
            at_final_waypoint = env._go2w_navigation_path_target_index >= final_idx
            near_goal = at_final_waypoint & (remaining_goal_distance <= goal_snap_distance)
        else:
            near_goal = remaining_goal_distance <= goal_snap_distance
    else:
        near_goal = remaining_goal_distance <= goal_snap_distance

    target_vec_w = waypoint_pos_w - robot.data.root_pos_w[:, :3]
    target_vec_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), target_vec_w)
    heading_b = wrap_to_pi(waypoint_heading_w - robot.data.heading_w)

    raw_xy_b = target_vec_b[:, :2]
    command_x = torch.where(
        near_goal,
        raw_xy_b[:, 0],
        raw_xy_b[:, 0].clamp(min=command_min_forward, max=lookahead_distance),
    )
    command_y = raw_xy_b[:, 1].clamp(min=-command_max_lateral, max=command_max_lateral)
    bearing = torch.atan2(raw_xy_b[:, 1], raw_xy_b[:, 0])
    raw_heading = torch.where(near_goal, heading_b, bearing)
    command_heading = raw_heading.clamp(min=-command_max_heading, max=command_max_heading)
    if math.isfinite(command_turn_slowdown_heading):
        turn_denom = max(command_max_heading - command_turn_slowdown_heading, 1.0e-6)
        turn_factor = ((raw_heading.abs() - command_turn_slowdown_heading) / turn_denom).clamp(0.0, 1.0)
        min_turn_forward = max(command_min_forward, command_turn_slowdown_min_forward)
        max_forward = lookahead_distance + turn_factor * (min_turn_forward - lookahead_distance)
        command_x = torch.where(near_goal, command_x, torch.minimum(command_x, max_forward))

    command = torch.stack((command_x, command_y, command_heading), dim=-1)
    if nav_debug_enabled():
        step = int(getattr(env, "common_step_counter", 0))
        backward_raw = raw_xy_b[:, 0] < -0.05
        backward_cmd = command[:, 0] < -0.05
        debug_interval = nav_debug_interval()
        event_interval = max(1, debug_interval // 4)
        should_print = (
            (step % debug_interval == 0)
            or (
                step % event_interval == 0
                and (bool(backward_raw.any().item()) or bool(backward_cmd.any().item()))
            )
        )
        if should_print:
            debug_env = nav_debug_env_id()
            row = debug_env if 0 <= debug_env < env.num_envs else 0
            problem_rows = (backward_raw | backward_cmd).nonzero(as_tuple=False).flatten()
            if problem_rows.numel() > 0:
                row = int(problem_rows[0].item())
            target_idx = -1
            nearest_idx = -1
            final_idx = -1
            final_dist = float("nan")
            if (
                hasattr(env, "_go2w_navigation_path_target_index")
                and hasattr(env, "_go2w_navigation_path_nearest_index")
                and hasattr(env, "_go2w_navigation_path_count")
            ):
                target_idx = int(env._go2w_navigation_path_target_index[row].item())
                nearest_idx = int(env._go2w_navigation_path_nearest_index[row].item())
                final_idx = int((env._go2w_navigation_path_count[row] - 1).clamp(min=0).item())
            if hasattr(env, "_go2w_navigation_path_final_distance"):
                final_dist = float(env._go2w_navigation_path_final_distance[row].item())
            print(
                "[GO2W_GOAL_CMD] "
                f"step={step} env={row} nearest={nearest_idx} target={target_idx} final={final_idx} "
                f"final_dist={final_dist:.2f} near_goal={bool(near_goal[row].item())} "
                f"raw_b={fmt_xy(raw_xy_b[row])} cmd=({float(command[row, 0].item()):+.2f},"
                f"{float(command[row, 1].item()):+.2f},{float(command[row, 2].item()):+.2f}) "
                f"heading_raw={float(raw_heading[row].item()):+.2f} "
                f"heading_clipped={float(command_heading[row].item()):+.2f} "
                f"heading_b={float(heading_b[row].item()):+.2f} path_mode={path_mode} "
                f"robot={fmt_xy(robot.data.root_pos_w[row, :2])} waypoint={fmt_xy(waypoint_pos_w[row, :2])}"
            )

    return command
