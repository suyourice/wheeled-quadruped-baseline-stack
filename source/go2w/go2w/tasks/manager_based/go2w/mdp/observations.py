# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for the Go2-W obstacle avoidance task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

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


def obstacle_positions_rel(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_distance: float | None = None,
    normalize: bool = False,
    zero_beyond_max_distance: bool = True,
    num_closest: int | None = None,
) -> torch.Tensor:
    """Obstacle positions relative to the robot base in the robot's local frame.

    Returns a flattened tensor containing (x, y) relative positions for each
    obstacle. If num_closest is set, the output is sorted by current distance
    every call and padded with zeros when fewer candidates exist.

    Args:
        obstacle_names: List of scene entity names for each obstacle.
        robot_cfg: SceneEntityCfg for the robot.
        max_distance: Optional clipping/masking radius in meters.
        normalize: If True, divide relative positions by max_distance.
        zero_beyond_max_distance: If True, obstacles beyond max_distance return zero.
        num_closest: Optional fixed number of closest obstacles to return.
    """
    if len(obstacle_names) == 0:
        k = 0 if num_closest is None else num_closest
        return torch.zeros(env.num_envs, k * 2, device=env.device)

    robot = env.scene[robot_cfg.name]
    robot_pos_w = robot.data.root_pos_w[:, :3]  # (N, 3)
    robot_quat_w = robot.data.root_quat_w  # (N, 4)

    rel_positions = []
    for name in obstacle_names:
        obstacle: RigidObject = env.scene[name]
        obs_pos_w = obstacle.data.root_pos_w[:, :3]  # (N, 3)
        rel_pos_w = obs_pos_w - robot_pos_w  # (N, 3)
        rel_pos_b = quat_apply_inverse(robot_quat_w, rel_pos_w)  # (N, 3)
        rel_positions.append(rel_pos_b[:, :2])  # only x, y

    rel_positions = torch.stack(rel_positions, dim=1)  # (N, num_obstacles, 2)
    dists = torch.norm(rel_positions, dim=-1)

    if num_closest is not None:
        k = min(num_closest, rel_positions.shape[1])
        closest_idx = torch.topk(dists, k=k, dim=1, largest=False, sorted=True).indices
        gather_idx = closest_idx.unsqueeze(-1).expand(-1, -1, 2)
        rel_positions = torch.gather(rel_positions, dim=1, index=gather_idx)
        dists = torch.gather(dists, dim=1, index=closest_idx)

        if k < num_closest:
            pad_shape = (rel_positions.shape[0], num_closest - k, 2)
            rel_positions = torch.cat(
                [rel_positions, torch.zeros(pad_shape, device=rel_positions.device, dtype=rel_positions.dtype)],
                dim=1,
            )
            dists = torch.cat(
                [dists, torch.full((dists.shape[0], num_closest - k), float("inf"), device=dists.device)],
                dim=1,
            )

    if max_distance is not None:
        rel_positions = rel_positions.clamp(min=-max_distance, max=max_distance)
        if zero_beyond_max_distance:
            rel_positions = torch.where(
                dists.unsqueeze(-1) <= max_distance,
                rel_positions,
                torch.zeros_like(rel_positions),
            )
        if normalize:
            rel_positions = rel_positions / max_distance

    return rel_positions.flatten(start_dim=1)  # (N, num_obstacles * 2)


def root_position_w(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the robot root position in world coordinates for debug logging."""
    robot = env.scene[robot_cfg.name]
    return robot.data.root_pos_w[:, :3]


def _ensure_navigation_goal_buffers(env: ManagerBasedRLEnv) -> None:
    """Create goal-navigation buffers on the env the first time they are requested."""
    if not hasattr(env, "_go2w_goal_pos_w"):
        env._go2w_goal_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._go2w_goal_heading_w = torch.zeros(env.num_envs, device=env.device)
        env._go2w_start_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._go2w_start_heading_w = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_go2w_scenario_template_id"):
        env._go2w_scenario_template_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)


def _compute_navigation_waypoint_world(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    lookahead_distance: float = 1.25,
    goal_snap_distance: float = 1.0,
    use_lidar_refinement: bool = True,
    lidar_max_distance: float = 20.0,
    local_planner_forward_padding: float = 0.35,
    local_planner_corridor_half_width: float = 0.65,
    local_planner_candidate_offsets: tuple[float, ...] = (0.0, 0.55, -0.55, 0.9, -0.9),
    local_planner_min_forward_distance: float = 0.1,
    local_planner_min_hit_height: float = -0.15,
    local_planner_activation_threshold: float = 0.12,
    local_planner_lateral_penalty: float = 0.08,
    local_planner_min_improvement: float = 0.03,
    local_planner_max_blend: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a rolling local waypoint and optionally refine it with LiDAR.

    The active task keeps a single final goal for success bookkeeping, but the
    policy should react to a shorter-horizon subgoal. We therefore project the
    current robot position onto the sampled start-goal segment and look ahead by
    a fixed distance along that segment. The lightweight local planner then
    scores straight/side detour waypoint candidates using LiDAR corridor
    occupancy. Close to the end, the waypoint snaps to the true goal pose so
    heading alignment is still learned.
    """
    _ensure_navigation_goal_buffers(env)
    robot = env.scene[robot_cfg.name]

    start_pos_w = env._go2w_start_pos_w
    goal_pos_w = env._go2w_goal_pos_w
    goal_heading_w = env._go2w_goal_heading_w
    root_pos_w = robot.data.root_pos_w[:, :3]

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
    if use_lidar_refinement:
        waypoint_pos_w, waypoint_heading_w = _refine_waypoint_with_lidar(
            env=env,
            robot=robot,
            waypoint_pos_w=waypoint_pos_w,
            waypoint_heading_w=waypoint_heading_w,
            remaining_goal_distance=remaining_goal_distance,
            sensor_cfg=sensor_cfg,
            max_distance=lidar_max_distance,
            forward_padding=local_planner_forward_padding,
            corridor_half_width=local_planner_corridor_half_width,
            candidate_offsets=local_planner_candidate_offsets,
            min_forward_distance=local_planner_min_forward_distance,
            min_hit_height=local_planner_min_hit_height,
            activation_threshold=local_planner_activation_threshold,
            lateral_penalty=local_planner_lateral_penalty,
            min_improvement=local_planner_min_improvement,
            max_blend=local_planner_max_blend,
            goal_snap_distance=goal_snap_distance,
        )
    return waypoint_pos_w, waypoint_heading_w


def _refine_waypoint_with_lidar(
    *,
    env: ManagerBasedRLEnv,
    robot,
    waypoint_pos_w: torch.Tensor,
    waypoint_heading_w: torch.Tensor,
    remaining_goal_distance: torch.Tensor,
    sensor_cfg: SceneEntityCfg,
    max_distance: float,
    forward_padding: float,
    corridor_half_width: float,
    candidate_offsets: tuple[float, ...],
    min_forward_distance: float,
    min_hit_height: float,
    activation_threshold: float,
    lateral_penalty: float,
    min_improvement: float,
    max_blend: float,
    goal_snap_distance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose a LiDAR-scored local detour waypoint when the straight corridor is blocked."""
    try:
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    except KeyError:
        return waypoint_pos_w, waypoint_heading_w

    if len(candidate_offsets) == 0:
        return waypoint_pos_w, waypoint_heading_w

    root_pos_w = robot.data.root_pos_w[:, :3]
    target_vec_w = waypoint_pos_w - root_pos_w
    target_vec_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), target_vec_w)
    target_xy_b = target_vec_b[:, :2]
    target_dist = torch.norm(target_xy_b, dim=-1).clamp_min(1.0e-6)

    default_dir = torch.zeros_like(target_xy_b)
    default_dir[:, 0] = 1.0
    target_dir = torch.where(
        target_dist.unsqueeze(-1) > 0.2,
        target_xy_b / target_dist.unsqueeze(-1),
        default_dir,
    )
    target_left = torch.stack((-target_dir[:, 1], target_dir[:, 0]), dim=-1)

    offsets = torch.as_tensor(candidate_offsets, device=target_xy_b.device, dtype=target_xy_b.dtype)
    candidate_xy_b = target_xy_b.unsqueeze(1) + offsets.view(1, -1, 1) * target_left.unsqueeze(1)
    candidate_dist = torch.norm(candidate_xy_b, dim=-1).clamp_min(1.0e-6)
    candidate_dir = candidate_xy_b / candidate_dist.unsqueeze(-1)
    candidate_left = torch.stack((-candidate_dir[..., 1], candidate_dir[..., 0]), dim=-1)

    hit_positions = sensor.data.ray_hits_w
    sensor_pos = sensor.data.pos_w
    rel_hit_w = hit_positions - sensor_pos.unsqueeze(1)
    finite_hits = torch.isfinite(rel_hit_w).all(dim=-1)
    safe_rel_hit_w = torch.where(finite_hits.unsqueeze(-1), rel_hit_w, torch.zeros_like(rel_hit_w))

    num_envs, num_rays, _ = safe_rel_hit_w.shape
    rel_hit_yaw = quat_apply_inverse(
        yaw_quat(robot.data.root_quat_w).unsqueeze(1).expand(-1, num_rays, -1).reshape(-1, 4),
        safe_rel_hit_w.reshape(-1, 3),
    ).view(num_envs, num_rays, 3)

    planar_xy = rel_hit_yaw[..., :2]
    planar_dist = torch.norm(planar_xy, dim=-1)
    valid = (
        finite_hits
        & (planar_dist > 1.0e-4)
        & (planar_dist < (max_distance - 1.0e-4))
        & (rel_hit_yaw[..., 2] > min_hit_height)
    )

    forward = torch.einsum("nrd,nkd->nkr", planar_xy, candidate_dir)
    lateral = torch.einsum("nrd,nkd->nkr", planar_xy, candidate_left)
    corridor_length = candidate_dist + forward_padding
    corridor_mask = (
        valid.unsqueeze(1)
        & (forward > min_forward_distance)
        & (forward < corridor_length.unsqueeze(-1))
        & (lateral.abs() < corridor_half_width)
    )

    forward_score = (1.0 - forward / corridor_length.unsqueeze(-1).clamp_min(1.0e-6)).clamp(0.0, 1.0)
    lateral_score = (1.0 - lateral.abs() / corridor_half_width).clamp(0.0, 1.0)
    candidate_risk = (corridor_mask.float() * forward_score * lateral_score).amax(dim=-1)

    max_offset = offsets.abs().amax().clamp_min(1.0e-6)
    candidate_score = candidate_risk + lateral_penalty * (offsets.abs() / max_offset).view(1, -1)
    best_idx = torch.argmin(candidate_score, dim=1)
    gather_idx = best_idx.view(-1, 1, 1).expand(-1, 1, 2)
    best_xy_b = torch.gather(candidate_xy_b, dim=1, index=gather_idx).squeeze(1)

    straight_idx = int(torch.argmin(offsets.abs()).item())
    straight_risk = candidate_risk[:, straight_idx]
    best_risk = torch.gather(candidate_risk, dim=1, index=best_idx.view(-1, 1)).squeeze(1)
    improvement = straight_risk - best_risk
    should_refine = (
        (remaining_goal_distance > goal_snap_distance)
        & (straight_risk > activation_threshold)
        & (improvement > min_improvement)
    )
    blend = torch.where(
        should_refine,
        ((straight_risk - activation_threshold) / (1.0 - activation_threshold)).clamp(0.0, max_blend),
        torch.zeros_like(straight_risk),
    )
    refined_xy_b = (1.0 - blend).unsqueeze(-1) * target_xy_b + blend.unsqueeze(-1) * best_xy_b

    yaw = robot.data.heading_w
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    refined_xy_w = torch.stack(
        (
            cos_yaw * refined_xy_b[:, 0] - sin_yaw * refined_xy_b[:, 1],
            sin_yaw * refined_xy_b[:, 0] + cos_yaw * refined_xy_b[:, 1],
        ),
        dim=-1,
    )

    refined_pos_w = waypoint_pos_w.clone()
    refined_pos_w[:, :2] = root_pos_w[:, :2] + refined_xy_w
    refined_heading_w = torch.atan2(refined_xy_w[:, 1], refined_xy_w[:, 0])
    refined_heading_w = torch.where(should_refine, refined_heading_w, waypoint_heading_w)
    return refined_pos_w, refined_heading_w


def local_goal_command_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    lookahead_distance: float = 1.25,
    goal_snap_distance: float = 1.0,
    use_lidar_refinement: bool = True,
    lidar_max_distance: float = 20.0,
    local_planner_forward_padding: float = 0.35,
    local_planner_corridor_half_width: float = 0.65,
    local_planner_candidate_offsets: tuple[float, ...] = (0.0, 0.55, -0.55, 0.9, -0.9),
    local_planner_min_forward_distance: float = 0.1,
    local_planner_min_hit_height: float = -0.15,
    local_planner_activation_threshold: float = 0.12,
    local_planner_lateral_penalty: float = 0.08,
    local_planner_min_improvement: float = 0.03,
    local_planner_max_blend: float = 1.0,
) -> torch.Tensor:
    """Return the current rolling local-waypoint command in the robot yaw frame.

    The command is a compact 3-vector:
      [waypoint_x_b, waypoint_y_b, waypoint_heading_b]

    It starts from the sampled world-frame start/goal task buffers, then the
    lightweight local planner can move the waypoint sideways when the straight
    LiDAR corridor is blocked.
    """
    robot = env.scene[robot_cfg.name]
    waypoint_pos_w, waypoint_heading_w = _compute_navigation_waypoint_world(
        env,
        robot_cfg=robot_cfg,
        sensor_cfg=sensor_cfg,
        lookahead_distance=lookahead_distance,
        goal_snap_distance=goal_snap_distance,
        use_lidar_refinement=use_lidar_refinement,
        lidar_max_distance=lidar_max_distance,
        local_planner_forward_padding=local_planner_forward_padding,
        local_planner_corridor_half_width=local_planner_corridor_half_width,
        local_planner_candidate_offsets=local_planner_candidate_offsets,
        local_planner_min_forward_distance=local_planner_min_forward_distance,
        local_planner_min_hit_height=local_planner_min_hit_height,
        local_planner_activation_threshold=local_planner_activation_threshold,
        local_planner_lateral_penalty=local_planner_lateral_penalty,
        local_planner_min_improvement=local_planner_min_improvement,
        local_planner_max_blend=local_planner_max_blend,
    )

    target_vec_w = waypoint_pos_w - robot.data.root_pos_w[:, :3]
    target_vec_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), target_vec_w)
    heading_b = wrap_to_pi(waypoint_heading_w - robot.data.heading_w)
    return torch.cat((target_vec_b[:, :2], heading_b.unsqueeze(1)), dim=-1)


def goal_position_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the sampled navigation goal in world coordinates."""
    _ensure_navigation_goal_buffers(env)
    return env._go2w_goal_pos_w


def waypoint_position_w(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    lookahead_distance: float = 1.25,
    goal_snap_distance: float = 1.0,
    use_lidar_refinement: bool = True,
    lidar_max_distance: float = 20.0,
    local_planner_forward_padding: float = 0.35,
    local_planner_corridor_half_width: float = 0.65,
    local_planner_candidate_offsets: tuple[float, ...] = (0.0, 0.55, -0.55, 0.9, -0.9),
    local_planner_min_forward_distance: float = 0.1,
    local_planner_min_hit_height: float = -0.15,
    local_planner_activation_threshold: float = 0.12,
    local_planner_lateral_penalty: float = 0.08,
    local_planner_min_improvement: float = 0.03,
    local_planner_max_blend: float = 1.0,
) -> torch.Tensor:
    """Return the current rolling local waypoint in world coordinates."""
    waypoint_pos_w, _ = _compute_navigation_waypoint_world(
        env,
        robot_cfg=robot_cfg,
        sensor_cfg=sensor_cfg,
        lookahead_distance=lookahead_distance,
        goal_snap_distance=goal_snap_distance,
        use_lidar_refinement=use_lidar_refinement,
        lidar_max_distance=lidar_max_distance,
        local_planner_forward_padding=local_planner_forward_padding,
        local_planner_corridor_half_width=local_planner_corridor_half_width,
        local_planner_candidate_offsets=local_planner_candidate_offsets,
        local_planner_min_forward_distance=local_planner_min_forward_distance,
        local_planner_min_hit_height=local_planner_min_hit_height,
        local_planner_activation_threshold=local_planner_activation_threshold,
        local_planner_lateral_penalty=local_planner_lateral_penalty,
        local_planner_min_improvement=local_planner_min_improvement,
        local_planner_max_blend=local_planner_max_blend,
    )
    return waypoint_pos_w


def start_position_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the sampled episode start pose origin in world coordinates."""
    _ensure_navigation_goal_buffers(env)
    return env._go2w_start_pos_w


def navigation_scenario_code(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the current scenario template code for debugging.

    Codes are assigned during obstacle reset and identify the dominant encounter
    family for an episode:
      0 empty, 1 head_on, 2 left_edge, 3 right_edge, 4 diag_left,
      5 diag_right, 6 off_left, 7 off_right, 8 narrow_gap, 9 random_fallback.
    """
    _ensure_navigation_goal_buffers(env)
    return env._go2w_scenario_template_id.to(dtype=torch.float32).unsqueeze(-1)


def lidar_steering_features(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    max_distance: float = 20.0,
    close_distance: float = 2.5,
    min_hit_height: float = -0.15,
    min_forward_distance: float = 0.1,
    min_command_speed: float = 0.15,
    front_center_half_angle_deg: float = 20.0,
    front_side_inner_angle_deg: float = 10.0,
    front_side_outer_angle_deg: float = 60.0,
    side_inner_angle_deg: float = 60.0,
    side_outer_angle_deg: float = 120.0,
    corridor_forward_distance: float = 2.8,
    corridor_half_width: float = 0.75,
    detour_lane_outer_width: float = 1.6,
) -> torch.Tensor:
    """Summarize LiDAR geometry into compact steering-friendly sector features.

    The student already receives the full raw scan. This term adds a compact set of
    local geometry summaries that are easier to map to avoidance decisions:
      - closeness in front-left / front-center / front-right / left / right sectors
      - close-hit ratios in the three front sectors
      - command-aligned corridor blockage and side-lane openness summaries

    Features are derived from LiDAR hit geometry in the robot yaw frame rather
    than from privileged obstacle positions, so the representation stays
    perception-based and sim2real-friendly.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    robot = env.scene[robot_cfg.name]

    hit_positions = sensor.data.ray_hits_w
    sensor_pos = sensor.data.pos_w
    rel_hit_w = hit_positions - sensor_pos.unsqueeze(1)
    finite_hits = torch.isfinite(rel_hit_w).all(dim=-1)
    safe_rel_hit_w = torch.where(finite_hits.unsqueeze(-1), rel_hit_w, torch.zeros_like(rel_hit_w))

    num_envs, num_rays, _ = safe_rel_hit_w.shape
    rel_hit_yaw = quat_apply_inverse(
        yaw_quat(robot.data.root_quat_w).unsqueeze(1).expand(-1, num_rays, -1).reshape(-1, 4),
        safe_rel_hit_w.reshape(-1, 3),
    ).view(num_envs, num_rays, 3)

    planar_xy = rel_hit_yaw[..., :2]
    planar_dist = torch.norm(planar_xy, dim=-1)
    bearing = torch.atan2(planar_xy[..., 1], planar_xy[..., 0])

    if command_name == "local_goal":
        command_xy = local_goal_command_b(env, robot_cfg=robot_cfg, use_lidar_refinement=False)[:, :2]
    else:
        command_xy = env.command_manager.get_command(command_name)[:, :2]
    command_speed = torch.norm(command_xy, dim=-1, keepdim=True)
    default_dir = torch.zeros_like(command_xy)
    default_dir[:, 0] = 1.0
    cmd_dir = torch.where(
        command_speed > min_command_speed,
        command_xy / command_speed.clamp(min=1.0e-6),
        default_dir,
    )
    left_dir = torch.stack((-cmd_dir[:, 1], cmd_dir[:, 0]), dim=-1)
    cmd_forward = (planar_xy * cmd_dir.unsqueeze(1)).sum(dim=-1)
    cmd_lateral = (planar_xy * left_dir.unsqueeze(1)).sum(dim=-1)

    # Ignore rays that effectively report "no obstacle nearby" and suppress
    # obvious ground returns from the lower ring by requiring a minimum hit height.
    valid = (
        finite_hits
        & (planar_dist > 1.0e-4)
        & (planar_dist < (max_distance - 1.0e-4))
        & (rel_hit_yaw[..., 2] > min_hit_height)
    )

    deg = torch.pi / 180.0
    front_center_half_angle = front_center_half_angle_deg * deg
    front_side_inner_angle = front_side_inner_angle_deg * deg
    front_side_outer_angle = front_side_outer_angle_deg * deg
    side_inner_angle = side_inner_angle_deg * deg
    side_outer_angle = side_outer_angle_deg * deg

    is_forward = planar_xy[..., 0] > min_forward_distance
    abs_bearing = bearing.abs()

    sector_masks = (
        valid & is_forward & (bearing >= front_side_inner_angle) & (bearing <= front_side_outer_angle),   # front-left
        valid & is_forward & (abs_bearing <= front_center_half_angle),                                     # front-center
        valid & is_forward & (bearing <= -front_side_inner_angle) & (bearing >= -front_side_outer_angle), # front-right
        valid & (bearing >= side_inner_angle) & (bearing <= side_outer_angle),                             # left
        valid & (bearing <= -side_inner_angle) & (bearing >= -side_outer_angle),                           # right
    )

    def _masked_min_distance(mask: torch.Tensor) -> torch.Tensor:
        masked_dist = torch.where(mask, planar_dist, torch.full_like(planar_dist, max_distance))
        return masked_dist.amin(dim=1)

    def _close_ratio(mask: torch.Tensor) -> torch.Tensor:
        count = mask.sum(dim=1).float()
        close_hits = (mask & (planar_dist <= close_distance)).sum(dim=1).float()
        return torch.where(count > 0.0, close_hits / count.clamp(min=1.0), torch.zeros_like(count))

    sector_min_dist = [_masked_min_distance(mask) for mask in sector_masks]
    sector_closeness = [1.0 - (dist / max_distance).clamp(0.0, 1.0) for dist in sector_min_dist]

    front_close_ratios = [_close_ratio(mask) for mask in sector_masks[:3]]

    corridor_forward_score = ((corridor_forward_distance - cmd_forward) / corridor_forward_distance).clamp(0.0, 1.0)

    center_lateral_score = ((corridor_half_width - cmd_lateral.abs()) / corridor_half_width).clamp(0.0, 1.0)
    left_lateral_score = ((corridor_half_width - cmd_lateral) / corridor_half_width).clamp(0.0, 1.0)
    right_lateral_score = ((corridor_half_width + cmd_lateral) / corridor_half_width).clamp(0.0, 1.0)

    center_corridor_score = (
        valid
        & (cmd_forward > min_forward_distance)
        & (cmd_forward < corridor_forward_distance)
        & (cmd_lateral.abs() <= corridor_half_width)
    ).float() * corridor_forward_score * center_lateral_score
    left_corridor_score = (
        valid
        & (cmd_forward > min_forward_distance)
        & (cmd_forward < corridor_forward_distance)
        & (cmd_lateral >= 0.0)
        & (cmd_lateral <= corridor_half_width)
    ).float() * corridor_forward_score * left_lateral_score
    right_corridor_score = (
        valid
        & (cmd_forward > min_forward_distance)
        & (cmd_forward < corridor_forward_distance)
        & (cmd_lateral <= 0.0)
        & (-cmd_lateral <= corridor_half_width)
    ).float() * corridor_forward_score * right_lateral_score

    center_corridor_blockage = center_corridor_score.amax(dim=1)
    left_corridor_blockage = left_corridor_score.amax(dim=1)
    right_corridor_blockage = right_corridor_score.amax(dim=1)

    side_lane_inner = corridor_half_width
    side_lane_outer = detour_lane_outer_width

    def _lane_openness(mask: torch.Tensor) -> torch.Tensor:
        lane_min_dist = _masked_min_distance(mask)
        return (lane_min_dist / max_distance).clamp(0.0, 1.0)

    left_lane_mask = (
        valid
        & (cmd_forward > min_forward_distance)
        & (cmd_forward < corridor_forward_distance)
        & (cmd_lateral >= side_lane_inner)
        & (cmd_lateral <= side_lane_outer)
    )
    right_lane_mask = (
        valid
        & (cmd_forward > min_forward_distance)
        & (cmd_forward < corridor_forward_distance)
        & (cmd_lateral <= -side_lane_inner)
        & (cmd_lateral >= -side_lane_outer)
    )

    left_lane_open = _lane_openness(left_lane_mask)
    right_lane_open = _lane_openness(right_lane_mask)

    weighted_side = (center_corridor_score * cmd_lateral).sum(dim=1) / (
        center_corridor_score.sum(dim=1) + 1.0e-6
    )
    weighted_side = (weighted_side / corridor_half_width).clamp(-1.0, 1.0)

    return torch.stack(
        (
            *sector_closeness,
            *front_close_ratios,
            center_corridor_blockage,
            left_corridor_blockage,
            right_corridor_blockage,
            left_lane_open,
            right_lane_open,
            weighted_side,
        ),
        dim=-1,
    )
