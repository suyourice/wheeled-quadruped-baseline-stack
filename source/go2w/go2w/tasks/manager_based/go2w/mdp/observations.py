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
    robot_quat_w = robot.data.root_quat_w        # (N, 4)

    # Stack all obstacle positions and apply quat_apply_inverse in one batched
    # call instead of K separate calls.
    obs_pos_all = torch.stack(
        [env.scene[n].data.root_pos_w[:, :3] for n in obstacle_names], dim=1
    )  # (N, K, 3)
    rel_pos_w_all = obs_pos_all - robot_pos_w.unsqueeze(1)  # (N, K, 3)
    N, K = rel_pos_w_all.shape[:2]
    quat_exp = robot_quat_w.unsqueeze(1).expand(-1, K, -1).reshape(N * K, 4)
    rel_pos_b_flat = quat_apply_inverse(quat_exp, rel_pos_w_all.reshape(N * K, 3))
    rel_positions = rel_pos_b_flat.reshape(N, K, 3)[:, :, :2]  # (N, K, 2)
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


def obstacle_polar_depth(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    num_bins: int = 180,
    max_distance: float = 20.0,
) -> torch.Tensor:
    """Convert privileged obstacle positions to a 180-bin polar closeness map.

    Encodes the same format as :func:`lidar_distances` (close=1, far/empty=0)
    so teacher (uses this) and student (uses lidar_distances) can be trained
    with direct MSE loss without any bridging transform.

    Each bin covers 360°/num_bins (default 2°). For each bin, the minimum
    distance among obstacles whose yaw-frame bearing falls in that bin is used.
    Inactive obstacles parked at ≫max_distance contribute 0 closeness via
    the clamp and do not pollute occupied bins.

    Args:
        obstacle_names: List of scene entity names for the obstacle rigid bodies.
        robot_cfg: SceneEntityCfg identifying the robot.
        num_bins: Angular resolution (default 180 → 2° per bin).
        max_distance: Normalization distance; obstacles beyond this map to 0.

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
    dists = torch.norm(rel_xy, dim=-1)  # (N, K)

    # Map bearing angle → bin index [0, num_bins)
    angles = torch.atan2(rel_xy[..., 1], rel_xy[..., 0])  # (N, K), range [-π, π]
    bin_idx = ((angles + torch.pi) / (2.0 * torch.pi) * num_bins).long().clamp(0, num_bins - 1)

    # Start with max_distance in every bin (= no obstacle = closeness 0)
    dist_map = torch.full((env.num_envs, num_bins), max_distance, device=env.device)

    # Scatter minimum distance per bin; inactive obstacles at 1000m stay ≫max_distance
    dist_map.scatter_reduce_(1, bin_idx, dists, reduce="amin", include_self=True)

    return (1.0 - dist_map / max_distance).clamp(0.0, 1.0)


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
    command_min_forward: float = 0.45,
    command_max_lateral: float = 0.85,
    command_max_heading: float = 0.9,
) -> torch.Tensor:
    """Return the current rolling local-waypoint command in the robot yaw frame.

    The command is a compact 3-vector:
      [command_x_b, command_y_b, command_heading_b]

    It starts from the sampled world-frame start/goal task buffers, then the
    lightweight local planner can move the waypoint sideways when the straight
    LiDAR corridor is blocked. The final observation is shaped to stay
    forward-biased so the frozen locomotion controller is not driven by mostly
    lateral or backward local subgoals.
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

    _ensure_navigation_goal_buffers(env)
    remaining_goal_distance = torch.norm(env._go2w_goal_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1)
    near_goal = remaining_goal_distance <= goal_snap_distance

    raw_xy_b = target_vec_b[:, :2]
    command_x = torch.where(
        near_goal,
        raw_xy_b[:, 0],
        raw_xy_b[:, 0].clamp(min=command_min_forward, max=lookahead_distance),
    )
    command_y = raw_xy_b[:, 1].clamp(min=-command_max_lateral, max=command_max_lateral)
    shaped_heading = torch.atan2(raw_xy_b[:, 1], raw_xy_b[:, 0])
    command_heading = torch.where(near_goal, heading_b, shaped_heading)
    command_heading = command_heading.clamp(min=-command_max_heading, max=command_max_heading)

    return torch.stack((command_x, command_y, command_heading), dim=-1)


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
