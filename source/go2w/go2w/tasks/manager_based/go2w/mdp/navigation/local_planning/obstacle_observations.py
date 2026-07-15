# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Privileged obstacle geometry observation functions for the Go2-W teacher policy."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

from ...common.orientation import quat_yaw_wxyz
from ..hospital.specs import (
    HOSPITAL_CLASS_COUNT,
    HOSPITAL_CORRIDOR_FEATURE_DIM,
    HOSPITAL_MAZE_JUNCTION_COUNT,
    HOSPITAL_PATH_FEATURE_DIM,
    HOSPITAL_SEMANTIC_FEATURE_DIM,
    HOSPITAL_TRAIN_ACTOR_SLOTS,
    HOSPITAL_TRAIN_MAX_ROUTE_LENGTH,
    HOSPITAL_TRAIN_OBSTACLE_DENSITY_SCHEDULE,
)
from ..goals import ensure_navigation_goal_buffers
from .obstacle_geometry import (
    DEFAULT_OBSTACLE_DEPTH,
    DEFAULT_OBSTACLE_EFFECTIVE_RADIUS,
    DEFAULT_OBSTACLE_WIDTH,
    OBSTACLE_SHAPE_CONE,
    OBSTACLE_SHAPE_CUBOID,
    OBSTACLE_SHAPE_CYLINDER,
    _meta_slice,
    obstacle_active_mask,
    obstacle_risk_radius,
)
from .hospital_metrics import hospital_centerline_metrics

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# =============================================================================
# Privileged navigation geometry features (teacher-only)
# =============================================================================


def _obstacle_obs_step_cache(env: ManagerBasedRLEnv) -> dict:
    """Per-step observation cache for shared obstacle geometry tensors."""
    step = int(getattr(env, "common_step_counter", 0))
    if getattr(env, "_go2w_obstacle_obs_cache_step", None) != step:
        env._go2w_obstacle_obs_cache = {}
        env._go2w_obstacle_obs_cache_step = step
    return env._go2w_obstacle_obs_cache


def _get_obstacle_relative_xy(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Obstacle positions in the robot yaw frame.

    Returns:
        rel_xy:  (N, K, 2)  - obstacle x/y in robot yaw frame
        dists:   (N, K)     - Euclidean distance to each obstacle
        angles:  (N, K)     - bearing angle ∈ (−π, π]
    """
    if len(obstacle_names) == 0:
        N = env.num_envs
        z = torch.zeros(N, 0, device=env.device)
        return z.unsqueeze(-1).expand(-1, -1, 2), z, z

    cache = _obstacle_obs_step_cache(env)
    cache_key = ("relative_xy", tuple(obstacle_names), robot_cfg.name)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    robot = env.scene[robot_cfg.name]
    robot_pos_w = robot.data.root_pos_w[:, :3]
    robot_yaw_quat = yaw_quat(robot.data.root_quat_w)

    obs_pos_all = torch.stack(
        [env.scene[n].data.root_pos_w[:, :3] for n in obstacle_names], dim=1
    )  # (N, K, 3)
    rel_w = obs_pos_all - robot_pos_w.unsqueeze(1)  # (N, K, 3)
    N, K = rel_w.shape[:2]

    quat_exp = robot_yaw_quat.unsqueeze(1).expand(-1, K, -1).reshape(N * K, 4)
    rel_b = quat_apply_inverse(quat_exp, rel_w.reshape(N * K, 3))
    rel_xy = rel_b.reshape(N, K, 3)[:, :, :2]            # (N, K, 2)
    dists = torch.norm(rel_xy, dim=-1)                    # (N, K)
    angles = torch.atan2(rel_xy[..., 1], rel_xy[..., 0])  # (N, K) ∈ (−π, π]
    result = (rel_xy, dists, angles)
    cache[cache_key] = result
    return result


def obstacle_full_geometry_features(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    num_slots: int = 15,
    max_distance: float = 8.0,
    max_footprint_size: float = 1.0,
    max_area: float = 1.0,
    max_radius: float = 1.0,
    robot_safety_radius: float = 0.30,
) -> torch.Tensor:
    """Return sorted privileged obstacle geometry for the teacher policy.

    Per obstacle layout (16D):
      active, rel_x, rel_y, center_dist, bearing_sin, bearing_cos,
      robot_view_width, robot_view_depth, area, effective_radius, clearance,
      cuboid, cylinder, cone, relative_yaw_sin, relative_yaw_cos.

    The output is sorted nearest-first and padded with zeros so train and play
    keep the same observation dimension even when play has many physical slots.
    """
    feature_dim = 16
    if num_slots <= 0 or len(obstacle_names) == 0:
        return torch.zeros(env.num_envs, max(num_slots, 0) * feature_dim, device=env.device)

    robot = env.scene[robot_cfg.name]
    rel_xy, center_dists, bearings = _get_obstacle_relative_xy(env, obstacle_names, robot_cfg)
    N, K = center_dists.shape
    device = env.device

    active = obstacle_active_mask(env, obstacle_names, center_dists, max_distance)
    sort_dists = torch.where(active, center_dists, torch.full_like(center_dists, max_distance * 10.0))
    selected_slots = min(K, num_slots)
    nearest_idx = torch.topk(sort_dists, k=selected_slots, dim=1, largest=False, sorted=True).indices

    def gather_slots(values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 2:
            return torch.gather(values, dim=1, index=nearest_idx)
        index = nearest_idx.unsqueeze(-1).expand(-1, -1, values.shape[-1])
        return torch.gather(values, dim=1, index=index)

    rel_xy = gather_slots(rel_xy)
    center_dists = gather_slots(center_dists)
    bearings = gather_slots(bearings)
    active = gather_slots(active).float()

    meta_shape = (env.num_envs, len(obstacle_names))

    def gather_meta(attr_name: str, fallback: torch.Tensor) -> torch.Tensor:
        stored = getattr(env, attr_name, None)
        sliced = _meta_slice(stored, meta_shape) if stored is not None else None
        return gather_slots(sliced) if sliced is not None else fallback

    shape_ids = gather_meta(
        "_go2w_obstacle_shape_id",
        torch.full((N, selected_slots), OBSTACLE_SHAPE_CUBOID, dtype=torch.long, device=device),
    )
    widths = gather_meta(
        "_go2w_obstacle_width",
        torch.full((N, selected_slots), DEFAULT_OBSTACLE_WIDTH, device=device),
    )
    depths = gather_meta(
        "_go2w_obstacle_depth",
        torch.full((N, selected_slots), DEFAULT_OBSTACLE_DEPTH, device=device),
    )
    effective_radius = gather_meta(
        "_go2w_obstacle_effective_radius",
        torch.full((N, selected_slots), DEFAULT_OBSTACLE_EFFECTIVE_RADIUS, device=device),
    )
    yaw_stored = getattr(env, "_go2w_obstacle_yaw", None)
    yaw_sliced = _meta_slice(yaw_stored, meta_shape) if yaw_stored is not None else None
    if yaw_sliced is not None:
        obstacle_yaw_w = gather_slots(yaw_sliced)
    else:
        obs_quat_all = torch.stack([env.scene[n].data.root_quat_w for n in obstacle_names], dim=1)
        obstacle_yaw_w = gather_slots(quat_yaw_wxyz(obs_quat_all.reshape(N * K, 4)).reshape(N, K))

    robot_yaw_w = robot.data.heading_w.unsqueeze(1)
    rel_yaw = wrap_to_pi(obstacle_yaw_w - robot_yaw_w)
    abs_cos_yaw = torch.cos(rel_yaw).abs()
    abs_sin_yaw = torch.sin(rel_yaw).abs()
    is_cuboid = shape_ids == OBSTACLE_SHAPE_CUBOID
    is_cylinder = shape_ids == OBSTACLE_SHAPE_CYLINDER
    is_cone = shape_ids == OBSTACLE_SHAPE_CONE

    cuboid_view_depth = abs_cos_yaw * widths + abs_sin_yaw * depths
    cuboid_view_width = abs_sin_yaw * widths + abs_cos_yaw * depths
    robot_view_depth = torch.where(is_cuboid, cuboid_view_depth, depths)
    robot_view_width = torch.where(is_cuboid, cuboid_view_width, widths)

    bearing_in_obstacle = wrap_to_pi(bearings - rel_yaw)
    cuboid_support_radius = (
        torch.cos(bearing_in_obstacle).abs() * widths * 0.5
        + torch.sin(bearing_in_obstacle).abs() * depths * 0.5
    )
    margin = float(getattr(env, "_go2w_obstacle_radius_margin", 0.0))
    surface_radius = torch.where(is_cuboid, cuboid_support_radius, effective_radius) + margin
    clearance = center_dists - surface_radius - robot_safety_radius

    cuboid_area = widths * depths
    round_area = math.pi * (widths * 0.5).square()
    area = torch.where(is_cuboid, cuboid_area, round_area)

    rel_xy_norm = rel_xy.clamp(min=-max_distance, max=max_distance) / max_distance
    center_dist_norm = center_dists.clamp(max=max_distance) / max_distance
    view_width_norm = robot_view_width.clamp(max=max_footprint_size) / max_footprint_size
    view_depth_norm = robot_view_depth.clamp(max=max_footprint_size) / max_footprint_size
    area_norm = area.clamp(max=max_area) / max_area
    effective_radius_norm = effective_radius.clamp(max=max_radius) / max_radius
    clearance_norm = clearance.clamp(min=-1.0, max=max_distance) / max_distance

    features = torch.stack(
        [
            active,
            rel_xy_norm[..., 0],
            rel_xy_norm[..., 1],
            center_dist_norm,
            bearings.sin(),
            bearings.cos(),
            view_width_norm,
            view_depth_norm,
            area_norm,
            effective_radius_norm,
            clearance_norm,
            is_cuboid.float(),
            is_cylinder.float(),
            is_cone.float(),
            rel_yaw.sin(),
            rel_yaw.cos(),
        ],
        dim=-1,
    )
    features = features * active.unsqueeze(-1)

    if selected_slots < num_slots:
        pad_shape = (N, num_slots - selected_slots, feature_dim)
        features = torch.cat([features, torch.zeros(pad_shape, device=device)], dim=1)

    return features.flatten(start_dim=1)


def obstacle_navigation_features(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    max_distance: float = 8.0,
    frontal_half_angle_deg: float = 45.0,
    corridor_half_width: float = 0.65,
    min_command_speed: float = 0.05,
    robot_safety_radius: float = 0.30,
    reference_slot_count: int = 15,
) -> torch.Tensor:
    """Compact privileged navigation geometry features for the nav teacher (16D).

    Converts privileged obstacle positions into explicit local-navigation inductive
    bias features: frontal blockage, side clearance, gap availability, preferred
    escape direction, and TTC proxy.  These are designed to accelerate PPO learning
    by exposing structure that would otherwise need to be inferred from the raw
    180-bin polar depth map.

    Teacher-only (privileged).  The student learns equivalent information from
    LiDAR/depth observations at distillation time.

    Feature layout (16D):
      [0]  nearest_dist_norm     - nearest baseline-compatible obstacle distance / max
      [1]  nearest_sin           - sin(bearing to nearest obstacle)     ∈ [−1, 1]
      [2]  nearest_cos           - cos(bearing to nearest obstacle)     ∈ [−1, 1]
      [3]  frontal_blockage      - density in ±frontal_angle sector     ∈ [0, 1]
      [4]  left_blockage         - density in left 90° sector           ∈ [0, 1]
      [5]  right_blockage        - density in right 90° sector          ∈ [0, 1]
      [6]  frontal_min_dist_norm - min baseline-compatible distance in frontal sector / max
      [7]  left_min_dist_norm    - min baseline-compatible distance in left hemisphere / max
      [8]  right_min_dist_norm   - min baseline-compatible distance in right hemisphere / max
      [9]  preferred_side_hint   - (left_blockage − right_blockage)     ∈ [−1, 1]
                                    positive = right preferred (left more blocked)
                                    negative = left preferred (right more blocked)
      [10] gap_available         - soft: passable frontal gap exists    ∈ [0, 1]
      [11] gap_width_norm        - frontal gap width / (2×corridor_hw)  ∈ [0, 1]
      [12] goal_path_blockage    - obstacle density along goal direction ∈ [0, 1]
      [13] ttc_proxy             - time-to-collision risk from HLC cmd  ∈ [0, 1]
      [14] obstacle_count_norm   - active count / training reference slots
      [15] rear_clearance        - min baseline-compatible rear distance / max
    """
    if len(obstacle_names) == 0:
        return torch.zeros(env.num_envs, 16, device=env.device)

    rel_xy, center_dists, angles = _get_obstacle_relative_xy(env, obstacle_names, robot_cfg)
    N, _ = center_dists.shape
    device = env.device
    frontal_rad = math.radians(frontal_half_angle_deg)

    active = obstacle_active_mask(env, obstacle_names, center_dists, max_distance)
    risk_radii = obstacle_risk_radius(env, obstacle_names, center_dists)
    nominal_radius = DEFAULT_OBSTACLE_EFFECTIVE_RADIUS + float(
        getattr(env, "_go2w_obstacle_radius_margin", 0.0)
    )
    radius_delta = risk_radii - nominal_radius
    # Preserve baseline feature values for the familiar 0.30 m box while making
    # larger randomized footprints appear closer and smaller ones appear farther.
    feature_dists = (center_dists - radius_delta).clamp(min=0.0, max=max_distance)

    # ------- nearest obstacle -------
    big = max_distance * 10.0
    masked_dists = torch.where(active, feature_dists, torch.full_like(center_dists, big))
    nearest_vals, nearest_idx = masked_dists.min(dim=1)  # (N,)
    nearest_dist_norm = nearest_vals.clamp(max=max_distance) / max_distance

    arange_n = torch.arange(N, device=device)
    nearest_sin = angles[arange_n, nearest_idx].sin()
    nearest_cos = angles[arange_n, nearest_idx].cos()
    has_active = active.any(dim=1)
    nearest_sin = torch.where(has_active, nearest_sin, torch.zeros_like(nearest_sin))
    nearest_cos = torch.where(has_active, nearest_cos, torch.zeros_like(nearest_cos))

    # ------- per-obstacle closeness -------
    closeness = ((1.0 - feature_dists / max_distance) * active.float()).clamp(0.0, 1.0)

    # ------- sector masks -------
    frontal_mask = (angles.abs() < frontal_rad) & active
    left_mask    = (angles > frontal_rad) & (angles <= math.pi) & active
    right_mask   = (angles < -frontal_rad) & (angles >= -math.pi) & active
    rear_mask    = (angles.abs() > math.pi - frontal_rad) & active

    blockage_scale = 1.0 / max(reference_slot_count, 1)
    frontal_blockage = ((closeness * frontal_mask.float()).sum(dim=1) * blockage_scale).clamp(max=1.0)
    left_blockage = ((closeness * left_mask.float()).sum(dim=1) * blockage_scale).clamp(max=1.0)
    right_blockage = ((closeness * right_mask.float()).sum(dim=1) * blockage_scale).clamp(max=1.0)

    # ------- min distance per sector -------
    inf_val = max_distance * 2.0
    fill = torch.full_like(center_dists, inf_val)
    frontal_min = torch.where(frontal_mask, feature_dists, fill).min(dim=1).values.clamp(max=max_distance)
    left_min    = torch.where(left_mask,    feature_dists, fill).min(dim=1).values.clamp(max=max_distance)
    right_min   = torch.where(right_mask,   feature_dists, fill).min(dim=1).values.clamp(max=max_distance)
    rear_min    = torch.where(rear_mask,    feature_dists, fill).min(dim=1).values.clamp(max=max_distance)

    frontal_min_dist_norm = frontal_min / max_distance
    left_min_dist_norm    = left_min    / max_distance
    right_min_dist_norm   = right_min   / max_distance
    rear_clearance        = rear_min    / max_distance

    # ------- preferred side hint -------
    # +1 → prefer right (left has more obstacles), −1 → prefer left
    preferred_side_hint = (left_blockage - right_blockage).clamp(-1.0, 1.0)

    # ------- gap availability (simplified) -------
    # Frontal minimum distance > corridor_half_width → passable gap
    gap_available = torch.sigmoid(
        (frontal_min - corridor_half_width) / (corridor_half_width * 0.3)
    )
    gap_width_norm = (frontal_min / (corridor_half_width * 2.0)).clamp(0.0, 1.0)

    # ------- goal path blockage -------
    ensure_navigation_goal_buffers(env)
    robot = env.scene[robot_cfg.name]
    robot_pos_2d = robot.data.root_pos_w[:, :2]
    goal_dir_w = env._go2w_goal_pos_w[:, :2] - robot_pos_2d    # (N, 2) world frame
    goal_dist_w = goal_dir_w.norm(dim=-1).clamp(min=0.01)
    goal_dir_w = goal_dir_w / goal_dist_w.unsqueeze(-1)

    # Rotate goal direction to robot yaw frame
    h = robot.data.heading_w   # (N,)
    cos_h, sin_h = torch.cos(h), torch.sin(h)
    goal_dir_b_x = cos_h * goal_dir_w[:, 0] + sin_h * goal_dir_w[:, 1]
    goal_dir_b_y = -sin_h * goal_dir_w[:, 0] + cos_h * goal_dir_w[:, 1]
    goal_dir_b = torch.stack([goal_dir_b_x, goal_dir_b_y], dim=-1)  # (N, 2)

    goal_forward = (rel_xy * goal_dir_b.unsqueeze(1)).sum(dim=-1)   # (N, K)
    # Perpendicular (left normal of goal direction)
    perp_b = torch.stack([-goal_dir_b[:, 1], goal_dir_b[:, 0]], dim=-1)
    goal_lateral = (rel_xy * perp_b.unsqueeze(1)).sum(dim=-1).abs() # (N, K)
    goal_corridor = corridor_half_width * 1.5
    # Keep the baseline corridor meaning for the familiar box and expose only
    # additional footprint width from larger or smaller randomized assets.
    goal_corridor_extent = (goal_corridor + radius_delta).clamp(min=1.0e-3)
    goal_path_mask = (
        (goal_forward > -radius_delta)
        & (goal_forward - radius_delta < goal_dist_w.unsqueeze(-1) + 0.3)
        & (goal_lateral < goal_corridor_extent)
        & active
    )
    goal_intrusion = (
        (goal_corridor_extent - goal_lateral).clamp(min=0.0) / goal_corridor_extent.clamp(min=1.0e-6)
    ).clamp(max=1.0)
    goal_closeness = ((1.0 - center_dists / max_distance) * active.float()).clamp(0.0, 1.0)
    goal_path_blockage = (goal_closeness * goal_intrusion * goal_path_mask.float()).max(dim=1).values

    # ------- TTC proxy from HLC command -------
    # FrozenLLCActionTerm mirrors the HLC velocity into base_velocity.
    cmd_xy = env.command_manager.get_command(command_name)[:, :2]   # (N, 2)
    cmd_speed = cmd_xy.norm(dim=-1)
    moving = cmd_speed > min_command_speed
    clearance_m = (frontal_min - 0.20).clamp(min=0.0)  # front margin 0.20 m
    ttc_val = clearance_m / cmd_speed.clamp(min=min_command_speed)
    safe_ttc = 1.5
    ttc_risk = ((safe_ttc - ttc_val) / safe_ttc).clamp(0.0, 1.0)
    ttc_proxy = torch.where(moving, ttc_risk, torch.zeros_like(ttc_risk))

    # ------- active obstacle count -------
    obstacle_count_norm = (active.float().sum(dim=1) / max(reference_slot_count, 1)).clamp(max=1.0)

    return torch.stack([
        nearest_dist_norm,     # [0]
        nearest_sin,           # [1]
        nearest_cos,           # [2]
        frontal_blockage,      # [3]
        left_blockage,         # [4]
        right_blockage,        # [5]
        frontal_min_dist_norm, # [6]
        left_min_dist_norm,    # [7]
        right_min_dist_norm,   # [8]
        preferred_side_hint,   # [9]
        gap_available,         # [10]
        gap_width_norm,        # [11]
        goal_path_blockage,    # [12]
        ttc_proxy,             # [13]
        obstacle_count_norm,   # [14]
        rear_clearance,        # [15]
    ], dim=-1)  # (N, 16)


def hospital_path_features(
    env: ManagerBasedRLEnv,
    max_path_length: float = HOSPITAL_TRAIN_MAX_ROUTE_LENGTH,
) -> torch.Tensor:
    """Return compact A*/waypoint progress features for the hospital teacher."""
    if not hasattr(env, "_go2w_navigation_path_w"):
        return torch.zeros(env.num_envs, HOSPITAL_PATH_FEATURE_DIM, device=env.device)

    N = env.num_envs
    ids = torch.arange(N, device=env.device)
    count = env._go2w_navigation_path_count.clamp(min=2)
    final_idx = (count - 1).clamp(max=env._go2w_navigation_path_s.shape[1] - 1)
    final_s = env._go2w_navigation_path_s[ids, final_idx].clamp(min=1.0e-6)
    progress_s = env._go2w_navigation_path_progress_s.clamp(min=0.0)
    target_s = env._go2w_navigation_path_target_s.clamp(min=0.0)
    final_dist = env._go2w_navigation_path_final_distance.clamp(min=0.0, max=max_path_length)
    target_idx = env._go2w_navigation_path_target_index.float()

    layout_count = max(HOSPITAL_MAZE_JUNCTION_COUNT * HOSPITAL_MAZE_JUNCTION_COUNT - 1, 1)
    layout_norm = getattr(
        env, "_go2w_hospital_layout_id", torch.zeros(N, dtype=torch.long, device=env.device)
    ).float() / layout_count
    phase_norm = getattr(
        env, "_go2w_hospital_phase_id", torch.zeros(N, dtype=torch.long, device=env.device)
    ).float() / max(len(HOSPITAL_TRAIN_OBSTACLE_DENSITY_SCHEDULE) - 1, 1)
    actor_norm = getattr(env, "_go2w_hospital_actor_count", torch.zeros(N, device=env.device)).float()
    actor_norm = actor_norm / max(HOSPITAL_TRAIN_ACTOR_SLOTS, 1)

    return torch.stack(
        [
            (progress_s / final_s).clamp(0.0, 1.0),
            (target_s / final_s).clamp(0.0, 1.0),
            final_dist / max_path_length,
            final_s.clamp(max=max_path_length) / max_path_length,
            ((target_s - progress_s).clamp(min=0.0, max=4.0) / 4.0),
            (target_idx / (count.float() - 1.0).clamp(min=1.0)).clamp(0.0, 1.0),
            (count.float() / max(env._go2w_navigation_path_s.shape[1], 1)).clamp(0.0, 1.0),
            layout_norm.clamp(0.0, 1.0),
            phase_norm.clamp(0.0, 1.0),
            actor_norm.clamp(0.0, 1.0),
        ],
        dim=-1,
    )


def hospital_corridor_features(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_lateral_error: float = 2.0,
    max_front_distance: float = 8.0,
) -> torch.Tensor:
    """Return corridor-following features for the hospital teacher."""
    if not hasattr(env, "_go2w_structured_corridor_width"):
        return torch.zeros(env.num_envs, HOSPITAL_CORRIDOR_FEATURE_DIM, device=env.device)

    lateral_error, heading_error, curvature, left_clearance, right_clearance = hospital_centerline_metrics(
        env, robot_cfg
    )
    corridor_width = env._go2w_structured_corridor_width.clamp(min=0.1)  # (N,)
    final_dist = torch.zeros(env.num_envs, device=env.device)
    if hasattr(env, "_go2w_navigation_path_final_distance"):
        final_dist = env._go2w_navigation_path_final_distance.clamp(min=0.0, max=max_front_distance)

    return torch.stack(
        [
            (lateral_error / max_lateral_error).clamp(-1.0, 1.0),
            (lateral_error.abs() / max_lateral_error).clamp(0.0, 1.0),
            heading_error.sin(),
            heading_error.cos(),
            (curvature / math.pi).clamp(0.0, 1.0),
            (left_clearance / corridor_width).clamp(0.0, 1.0),
            (right_clearance / corridor_width).clamp(0.0, 1.0),
            final_dist / max_front_distance,
        ],
        dim=-1,
    )


def hospital_semantic_obstacle_features(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    num_slots: int = 15,
    max_distance: float = 8.0,
    max_height: float = 2.5,
    max_relative_speed: float = 2.5,
) -> torch.Tensor:
    """Return nearest hospital obstacle semantics, priority, and height metadata."""
    if num_slots <= 0 or len(obstacle_names) == 0:
        return torch.zeros(env.num_envs, max(num_slots, 0) * HOSPITAL_SEMANTIC_FEATURE_DIM, device=env.device)

    robot = env.scene[robot_cfg.name]
    _, center_dists, _ = _get_obstacle_relative_xy(env, obstacle_names, robot_cfg)
    N, K = center_dists.shape
    device = env.device
    active = obstacle_active_mask(env, obstacle_names, center_dists, max_distance)
    sort_dists = torch.where(active, center_dists, torch.full_like(center_dists, max_distance * 10.0))
    selected_slots = min(K, num_slots)
    nearest_idx = torch.topk(sort_dists, k=selected_slots, dim=1, largest=False, sorted=True).indices

    def gather_slots(values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 2:
            return torch.gather(values, dim=1, index=nearest_idx)
        index = nearest_idx.unsqueeze(-1).expand(-1, -1, values.shape[-1])
        return torch.gather(values, dim=1, index=index)

    meta_shape = (env.num_envs, len(obstacle_names))
    active_s = gather_slots(active).float()

    def gather_meta(attr_name: str, fallback: torch.Tensor) -> torch.Tensor:
        stored = getattr(env, attr_name, None)
        sliced = _meta_slice(stored, meta_shape) if stored is not None else None
        return gather_slots(sliced) if sliced is not None else fallback

    class_ids = gather_meta(
        "_go2w_obstacle_class_id",
        torch.zeros((N, selected_slots), dtype=torch.long, device=device),
    ).clamp(min=0, max=HOSPITAL_CLASS_COUNT - 1)
    priority = gather_meta(
        "_go2w_obstacle_priority",
        torch.full((N, selected_slots), 0.5, device=device),
    )
    height = gather_meta("_go2w_obstacle_height", torch.zeros((N, selected_slots), device=device))
    top_z = gather_meta("_go2w_obstacle_top_z", height)
    low_flag = gather_meta("_go2w_obstacle_low_flag", torch.zeros_like(height)).float()
    dynamic_flag = gather_meta("_go2w_obstacle_dynamic_mask", torch.zeros_like(height)).float()

    obs_vel_w = torch.stack(
        [env.scene[n].data.root_lin_vel_w[:, :3] for n in obstacle_names], dim=1
    )
    rel_vel_w = obs_vel_w - robot.data.root_lin_vel_w[:, :3].unsqueeze(1)
    robot_yaw_quat = yaw_quat(robot.data.root_quat_w)
    quat_exp = robot_yaw_quat.unsqueeze(1).expand(-1, K, -1).reshape(N * K, 4)
    rel_vel_b = quat_apply_inverse(quat_exp, rel_vel_w.reshape(N * K, 3)).reshape(N, K, 3)[:, :, :2]
    rel_vel_b = gather_slots(rel_vel_b) * dynamic_flag.unsqueeze(-1)

    class_one_hot = torch.nn.functional.one_hot(class_ids, num_classes=HOSPITAL_CLASS_COUNT).float()
    actor_flag = (class_ids != 0).float()
    rel_vx = (rel_vel_b[..., 0] / max(max_relative_speed, 1.0e-6)).clamp(-1.0, 1.0)
    rel_vy = (rel_vel_b[..., 1] / max(max_relative_speed, 1.0e-6)).clamp(-1.0, 1.0)
    features = torch.cat(
        (
            active_s.unsqueeze(-1),
            class_one_hot,
            priority.clamp(0.0, 1.0).unsqueeze(-1),
            (height.clamp(0.0, max_height) / max_height).unsqueeze(-1),
            (top_z.clamp(0.0, max_height) / max_height).unsqueeze(-1),
            low_flag.unsqueeze(-1),
            dynamic_flag.unsqueeze(-1),
            rel_vx.unsqueeze(-1),
            rel_vy.unsqueeze(-1),
            actor_flag.unsqueeze(-1),
        ),
        dim=-1,
    )
    features = features * active_s.unsqueeze(-1)

    if selected_slots < num_slots:
        pad_shape = (N, num_slots - selected_slots, HOSPITAL_SEMANTIC_FEATURE_DIM)
        features = torch.cat([features, torch.zeros(pad_shape, device=device)], dim=1)

    return features.flatten(start_dim=1)


def prev_hlc_actions(
    env: ManagerBasedRLEnv,
    num_frames: int = 2,
    action_term_name: str = "llc_cmd",
) -> torch.Tensor:
    """Return the previous N frames of HLC velocity commands (N, num_frames × 3).

    Maintains env._hlc_action_history of shape (num_envs, num_frames, 3).  On
    each call it reads raw_actions from the named action term (which holds the
    velocity command from the PREVIOUS physics step) and:
      1. Returns the current buffer contents (steps t−1, t−2, ...) as the obs.
      2. Shifts the buffer left and inserts the latest action at position 0.

    The buffer is zeroed for newly-reset environments so episodes start clean.
    """
    buf_key = "_hlc_action_history"
    if not hasattr(env, buf_key):
        setattr(env, buf_key, torch.zeros(env.num_envs, num_frames, 3, device=env.device))
    history: torch.Tensor = getattr(env, buf_key)

    # Zero out history for episodes that just started
    reset_ids = (env.episode_length_buf == 0).nonzero(as_tuple=False).squeeze(-1)
    if reset_ids.numel() > 0:
        history[reset_ids] = 0.0

    # Snapshot previous history to return as observation
    result = history.flatten(start_dim=1).clone()  # (N, num_frames*3)

    # Read the last completed HLC action and shift into buffer
    try:
        term = env.action_manager.get_term(action_term_name)
        latest = term.raw_actions.detach().clamp(-2.0, 2.0)  # (N, 3)
    except (AttributeError, KeyError):
        latest = torch.zeros(env.num_envs, 3, device=env.device)

    if num_frames > 1:
        history[:, 1:] = history[:, :-1].clone()
    history[:, 0] = latest

    return result  # (N, num_frames*3)
