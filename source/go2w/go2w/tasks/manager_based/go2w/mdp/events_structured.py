# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Structured-corridor A* play reset for the Go2-W navigation environment."""

from __future__ import annotations

import functools
import math
import random
from typing import TYPE_CHECKING

import torch

from .events import (
    ensure_navigation_goal_buffers,
    quat_yaw_wxyz,
    yaw_to_quat_wxyz,
    yaw_pitch_roll_to_quat_wxyz,
)
from .nav_scenarios import NAV_RANDOM_FALLBACK_SCENARIO_ID as _NAV_RANDOM_FALLBACK_SCENARIO_ID
from .nav_slotting import _separated_parked_positions
from .navigation_path import set_navigation_path_w, update_navigation_path_waypoint
from .obstacle_geometry import set_obstacle_metadata
from .structured_corridor import (
    plan_structured_corridor_path,
    structured_corridor_centerline,
    structured_corridor_extra_polylines,
    structured_corridor_wall_specs,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_structured_astar_corridor(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    obstacle_names: list[str],
    fixed_scenario_template: str | None = None,
    corridor_kind: str | None = None,
    dynamic_obstacle_names: list[str] | None = None,
    dynamic_obstacle_indices: list[int] | None = None,
    dynamic_obstacle_count: int = 6,
    corridor_width: float = 1.8,
    leg_length: float = 6.0,
    corridor_turn_length: float | None = None,
    wall_thickness: float = 0.20,
    grid_resolution: float = 0.20,
    robot_inflation: float = 0.35,
    lookahead_distance: float = 1.25,
    waypoint_reach_radius: float = 0.45,
    obstacle_z: float = 0.30,
    park_distance: float = 1000.0,
    min_inter_obstacle_dist: float = 0.75,
    dynamic_start_exclusion_radius: float = 1.8,
    goal_exclusion_radius: float = 0.9,
    dynamic_robot_keepout_radius: float = 1.25,
    obstacle_radius_margin: float = 0.0,
    fixed_obstacle_shape_ids: tuple[int, ...] | None = None,
    fixed_obstacle_widths: tuple[float, ...] | None = None,
    fixed_obstacle_depths: tuple[float, ...] | None = None,
    fixed_obstacle_center_zs: tuple[float, ...] | None = None,
    obstacle_labels: tuple[str, ...] | list[str] | None = None,
    fixed_obstacle_local_poses: tuple[tuple[int, float, float, float], ...] | None = None,
    robot_start_local_xy: tuple[float, float] = (0.0, 0.0),
    ramp_asset_name: str | None = None,
    ramp_local_pose: tuple[float, ...] | None = None,
    ramp_b_asset_name: str | None = None,
    ramp_b_local_pose: tuple[float, ...] | None = None,
    randomize_obstacle_yaw: bool = True,
    obstacle_yaw_range: tuple[float, float] = (-math.pi, math.pi),
    clearance_cost_weight: float = 2.0,
    clearance_cost_sigma: float = 0.4,
    corner_rounding: bool = False,
    corner_radius: float = 0.5,
    adaptive_lookahead: bool = True,
    lookahead_min: float = 0.6,
    curvature_scan_horizon: float = 2.5,
    curvature_threshold: float = 0.3,
) -> None:
    """Reset play envs into a known structured corridor with an A* waypoint path."""
    del dynamic_obstacle_names, dynamic_obstacle_indices, obstacle_labels
    if len(obstacle_names) == 0:
        return
    ensure_navigation_goal_buffers(env)
    structured_kind = (corridor_kind or fixed_scenario_template or "l_corridor").lower()
    centerline = structured_corridor_centerline(
        structured_kind,
        leg_length,
        corridor_width,
        corridor_turn_length,
    )
    wall_specs = structured_corridor_wall_specs(
        structured_kind,
        leg_length,
        corridor_width,
        wall_thickness,
        corridor_turn_length,
    )
    if len(obstacle_names) < len(wall_specs):
        raise ValueError(
            f"Structured {structured_kind} requires at least {len(wall_specs)} obstacle slots for wall segments."
        )

    if hasattr(env, "_go2w_goals_reached_episode"):
        env._go2w_goals_reached_episode[env_ids] = 0.0
    if hasattr(env, "_go2w_first_goal_reached_episode"):
        env._go2w_first_goal_reached_episode[env_ids] = False
    if hasattr(env, "_go2w_min_goal_distance_episode"):
        env._go2w_min_goal_distance_episode[env_ids] = float("inf")
    if hasattr(env, "_go2w_had_collision_episode"):
        env._go2w_had_collision_episode[env_ids] = False
    if hasattr(env, "_go2w_obstacle_pose_changed_midstep"):
        env._go2w_obstacle_pose_changed_midstep[env_ids] = False
    if hasattr(env, "_go2w_ignore_obstacle_contact_history_steps"):
        env._go2w_ignore_obstacle_contact_history_steps[env_ids] = 0
    if hasattr(env, "_go2w_dynamic_obstacle_initialized"):
        env._go2w_dynamic_obstacle_initialized[env_ids] = False

    n = len(env_ids)
    k = len(obstacle_names)
    device = env.device
    robot = env.scene["robot"]
    start_pos_w = robot.data.root_pos_w[env_ids, :3].clone()
    start_heading_w = robot.data.heading_w[env_ids].clone()
    yaw = quat_yaw_wxyz(robot.data.root_quat_w[env_ids])
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    robot_start_local = torch.tensor(robot_start_local_xy, dtype=start_pos_w.dtype, device=device)
    layout_origin_xy = torch.zeros(n, 2, dtype=start_pos_w.dtype, device=device)
    layout_origin_xy[:, 0] = (
        start_pos_w[:, 0]
        - robot_start_local[0] * cos_yaw
        + robot_start_local[1] * sin_yaw
    )
    layout_origin_xy[:, 1] = (
        start_pos_w[:, 1]
        - robot_start_local[0] * sin_yaw
        - robot_start_local[1] * cos_yaw
    )

    env._go2w_start_pos_w[env_ids] = start_pos_w
    env._go2w_start_heading_w[env_ids] = start_heading_w
    env._go2w_gap_passable[env_ids] = False
    env._go2w_scenario_template_id[env_ids] = _NAV_RANDOM_FALLBACK_SCENARIO_ID
    env._go2w_initial_scenario_template_id[env_ids] = _NAV_RANDOM_FALLBACK_SCENARIO_ID
    if (
        not hasattr(env, "_go2w_structured_corridor_start_xy")
        or env._go2w_structured_corridor_start_xy.shape != (env.num_envs, 2)
    ):
        env._go2w_structured_corridor_start_xy = torch.zeros(env.num_envs, 2, device=device)
        env._go2w_structured_corridor_yaw = torch.zeros(env.num_envs, device=device)
        env._go2w_structured_corridor_leg_length = torch.zeros(env.num_envs, device=device)
        env._go2w_structured_corridor_width = torch.zeros(env.num_envs, device=device)
        env._go2w_structured_corridor_waypoint_clearance = torch.zeros(env.num_envs, device=device)
    if not hasattr(env, "_go2w_structured_corridor_waypoint_clearance"):
        env._go2w_structured_corridor_waypoint_clearance = torch.zeros(env.num_envs, device=device)
    centerline_count = len(centerline)
    if (
        not hasattr(env, "_go2w_structured_corridor_centerline_local")
        or env._go2w_structured_corridor_centerline_local.shape != (env.num_envs, centerline_count, 2)
    ):
        env._go2w_structured_corridor_centerline_local = torch.zeros(
            env.num_envs, centerline_count, 2, device=device
        )
    env._go2w_structured_corridor_start_xy[env_ids] = layout_origin_xy
    env._go2w_structured_corridor_yaw[env_ids] = yaw
    env._go2w_structured_corridor_leg_length[env_ids] = leg_length
    env._go2w_structured_corridor_width[env_ids] = corridor_width
    env._go2w_structured_corridor_waypoint_clearance[env_ids] = robot_inflation
    centerline_tensor = torch.tensor(centerline, dtype=start_pos_w.dtype, device=device)
    env._go2w_structured_corridor_centerline_local[env_ids] = centerline_tensor.unsqueeze(0).expand(n, -1, -1)
    extra_polylines = structured_corridor_extra_polylines(structured_kind, leg_length, corridor_width)
    if (
        not hasattr(env, "_go2w_structured_corridor_extra_polyline_count")
        or env._go2w_structured_corridor_extra_polyline_count.shape != (env.num_envs,)
    ):
        env._go2w_structured_corridor_extra_polyline_count = torch.zeros(
            env.num_envs, dtype=torch.long, device=device
        )
    env._go2w_structured_corridor_extra_polyline_count[env_ids] = len(extra_polylines)
    if len(extra_polylines) > 0:
        extra_tensor = torch.tensor(extra_polylines, dtype=start_pos_w.dtype, device=device)
        if (
            not hasattr(env, "_go2w_structured_corridor_extra_polylines_local")
            or env._go2w_structured_corridor_extra_polylines_local.shape
            != (env.num_envs, len(extra_polylines), extra_tensor.shape[1], 2)
        ):
            env._go2w_structured_corridor_extra_polylines_local = torch.zeros(
                env.num_envs, len(extra_polylines), extra_tensor.shape[1], 2, dtype=start_pos_w.dtype, device=device
            )
        env._go2w_structured_corridor_extra_polylines_local[env_ids] = extra_tensor.unsqueeze(0).expand(n, -1, -1, -1)

    local_path = plan_structured_corridor_path(
        structured_kind,
        leg_length,
        corridor_width,
        robot_inflation,
        grid_resolution,
        corridor_turn_length,
        clearance_cost_weight=clearance_cost_weight,
        clearance_cost_sigma=clearance_cost_sigma,
        corner_rounding=corner_rounding,
        corner_radius=corner_radius,
    )
    path_count = len(local_path)
    path_local = torch.tensor(local_path, dtype=start_pos_w.dtype, device=device)
    dx = path_local[:, 0].unsqueeze(0)
    dy = path_local[:, 1].unsqueeze(0)
    path_w = torch.zeros(n, path_count, 3, dtype=start_pos_w.dtype, device=device)
    path_w[:, :, 0] = layout_origin_xy[:, 0:1] + dx * cos_yaw.unsqueeze(1) - dy * sin_yaw.unsqueeze(1)
    path_w[:, :, 1] = layout_origin_xy[:, 1:2] + dx * sin_yaw.unsqueeze(1) + dy * cos_yaw.unsqueeze(1)
    path_w[:, :, 2] = start_pos_w[:, 2:3]
    env._go2w_navigation_path_direct_goal = True
    set_navigation_path_w(env, env_ids, path_w)

    wall_count = len(wall_specs)
    dynamic_count = max(0, min(dynamic_obstacle_count, k - wall_count))
    active_mask = torch.zeros(n, k, dtype=torch.bool, device=device)
    active_mask[:, : wall_count + dynamic_count] = True
    obstacle_yaws = torch.zeros(n, k, device=device)
    slot_radii = torch.full((k,), 0.25, dtype=start_pos_w.dtype, device=device)
    if fixed_obstacle_center_zs is not None:
        if len(fixed_obstacle_center_zs) != k:
            raise ValueError("Fixed obstacle center z table must contain one value per obstacle slot.")
        center_z_values = torch.tensor(fixed_obstacle_center_zs, dtype=start_pos_w.dtype, device=device)
    else:
        center_z_values = torch.full((k,), obstacle_z, dtype=start_pos_w.dtype, device=device)
    if (
        fixed_obstacle_shape_ids is not None
        and fixed_obstacle_widths is not None
        and fixed_obstacle_depths is not None
    ):
        shape_ids_t = torch.tensor(fixed_obstacle_shape_ids, dtype=torch.long, device=device)
        widths_t = torch.tensor(fixed_obstacle_widths, dtype=start_pos_w.dtype, device=device)
        depths_t = torch.tensor(fixed_obstacle_depths, dtype=start_pos_w.dtype, device=device)
        cuboid_r = torch.sqrt((widths_t * 0.5).square() + (depths_t * 0.5).square())
        round_r = torch.maximum(widths_t, depths_t) * 0.5
        slot_radii = torch.where(shape_ids_t == 0, cuboid_r, round_r)

    parked_world = start_pos_w.clone()
    parked_world[:, 0] += park_distance
    parked_positions = _separated_parked_positions(parked_world, k)
    positions = parked_positions.clone()

    for slot_idx, (local_x, local_y, local_yaw, _, _) in enumerate(wall_specs):
        wx = layout_origin_xy[:, 0] + local_x * cos_yaw - local_y * sin_yaw
        wy = layout_origin_xy[:, 1] + local_x * sin_yaw + local_y * cos_yaw
        positions[:, slot_idx, 0] = wx
        positions[:, slot_idx, 1] = wy
        positions[:, slot_idx, 2] = center_z_values[slot_idx]
        obstacle_yaws[:, slot_idx] = yaw + local_yaw

    # Spawn dynamic obstacles along all corridor segments (nav path + dead-end branches).
    placed_dynamic: list[tuple[float, float]] = []
    rng = random
    half_width = corridor_width * 0.5
    all_centerlines = (centerline,) + extra_polylines
    segments = [seg for cl in all_centerlines for seg in zip(cl[:-1], cl[1:])]
    segment_lengths = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in segments]
    total_length = max(sum(segment_lengths), 1.0e-6)
    goal_local_x, goal_local_y = local_path[-1]

    def sample_corridor_local_point(slot_idx: int) -> tuple[float, float]:
        s = rng.uniform(0.0, total_length)
        slot_radius = float(slot_radii[slot_idx].item())
        lateral_limit = max(0.05, half_width - slot_radius - 0.15)
        for (a, b), length in zip(segments, segment_lengths):
            if s <= length:
                t = 0.0 if length <= 1.0e-6 else s / length
                x = a[0] + (b[0] - a[0]) * t
                y = a[1] + (b[1] - a[1]) * t
                ux = 0.0 if length <= 1.0e-6 else (b[0] - a[0]) / length
                uy = 0.0 if length <= 1.0e-6 else (b[1] - a[1]) / length
                lateral = rng.uniform(-lateral_limit, lateral_limit)
                return x - uy * lateral, y + ux * lateral
            s -= length
        return centerline[-1]

    for dyn_idx in range(dynamic_count):
        slot_idx = wall_count + dyn_idx
        chosen = None
        for _ in range(80):
            local_x, local_y = sample_corridor_local_point(slot_idx)
            if math.hypot(local_x, local_y) < dynamic_start_exclusion_radius:
                continue
            if math.hypot(local_x - goal_local_x, local_y - goal_local_y) < goal_exclusion_radius:
                continue
            if any(math.hypot(local_x - px, local_y - py) < min_inter_obstacle_dist for px, py in placed_dynamic):
                continue
            chosen = (local_x, local_y)
            placed_dynamic.append(chosen)
            break
        if chosen is None:
            chosen = (leg_length * 0.5, 0.0)
        local_x, local_y = chosen
        positions[:, slot_idx, 0] = layout_origin_xy[:, 0] + local_x * cos_yaw - local_y * sin_yaw
        positions[:, slot_idx, 1] = layout_origin_xy[:, 1] + local_x * sin_yaw + local_y * cos_yaw
        positions[:, slot_idx, 2] = center_z_values[slot_idx]
        if randomize_obstacle_yaw:
            obstacle_yaws[:, slot_idx] = torch.empty(n, device=device).uniform_(*obstacle_yaw_range)

    if fixed_obstacle_local_poses is not None:
        for slot_idx, local_x, local_y, local_yaw in fixed_obstacle_local_poses:
            if slot_idx < 0 or slot_idx >= k:
                raise ValueError(f"Fixed obstacle local pose slot {slot_idx} is outside the obstacle table.")
            positions[:, slot_idx, 0] = layout_origin_xy[:, 0] + local_x * cos_yaw - local_y * sin_yaw
            positions[:, slot_idx, 1] = layout_origin_xy[:, 1] + local_x * sin_yaw + local_y * cos_yaw
            positions[:, slot_idx, 2] = center_z_values[slot_idx]
            obstacle_yaws[:, slot_idx] = yaw + local_yaw
            active_mask[:, slot_idx] = True

    set_obstacle_metadata(
        env,
        env_ids,
        obstacle_names,
        active_mask,
        obstacle_radius_margin=obstacle_radius_margin,
        fixed_obstacle_shape_ids=fixed_obstacle_shape_ids,
        fixed_obstacle_widths=fixed_obstacle_widths,
        fixed_obstacle_depths=fixed_obstacle_depths,
        obstacle_yaws=obstacle_yaws,
    )
    if hasattr(env, "_go2w_hospital_dynamic_initialized"):
        env._go2w_hospital_dynamic_initialized[env_ids] = False

    zero_vel = torch.zeros(n, 6, device=device)
    for slot_idx, name in enumerate(obstacle_names):
        pose = torch.zeros(n, 7, device=device)
        pose[:, :3] = positions[:, slot_idx]
        pose[:, 3:7] = yaw_to_quat_wxyz(obstacle_yaws[:, slot_idx])
        env.scene[name].write_root_pose_to_sim(pose, env_ids=env_ids)
        env.scene[name].write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

    if ramp_asset_name is not None:
        ramp = env.scene[ramp_asset_name]
        pose = torch.zeros(n, 7, device=device)
        if ramp_local_pose is None:
            pose[:, 0] = start_pos_w[:, 0] + park_distance
            pose[:, 1] = start_pos_w[:, 1] + park_distance
            pose[:, 2] = 0.20
            pose[:, 3] = 1.0
        else:
            # ramp_local_pose is (x, y, yaw, pitch, z) or (x, y, yaw, pitch, z, roll).
            ramp_pose_ext = tuple(ramp_local_pose) + (0.0,) if len(ramp_local_pose) == 5 else tuple(ramp_local_pose)
            local_x, local_y, local_yaw, local_pitch, local_z, local_roll = ramp_pose_ext
            pose[:, 0] = layout_origin_xy[:, 0] + local_x * cos_yaw - local_y * sin_yaw
            pose[:, 1] = layout_origin_xy[:, 1] + local_x * sin_yaw + local_y * cos_yaw
            pose[:, 2] = local_z
            dtype_r = start_pos_w.dtype
            pitch_t = torch.full((n,), local_pitch, dtype=dtype_r, device=device)
            roll_t = torch.full((n,), local_roll, dtype=dtype_r, device=device)
            pose[:, 3:7] = yaw_pitch_roll_to_quat_wxyz(yaw + local_yaw, pitch_t, roll_t)
        ramp.write_root_pose_to_sim(pose, env_ids=env_ids)
        ramp.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

    if ramp_b_asset_name is not None:
        ramp_b = env.scene[ramp_b_asset_name]
        pose = torch.zeros(n, 7, device=device)
        if ramp_b_local_pose is None:
            pose[:, 0] = start_pos_w[:, 0] + park_distance
            pose[:, 1] = start_pos_w[:, 1] + park_distance + 5.0
            pose[:, 2] = 0.20
            pose[:, 3] = 1.0
        else:
            ramp_b_pose_ext = tuple(ramp_b_local_pose) + (0.0,) if len(ramp_b_local_pose) == 5 else tuple(ramp_b_local_pose)
            local_x, local_y, local_yaw, local_pitch, local_z, local_roll = ramp_b_pose_ext
            pose[:, 0] = layout_origin_xy[:, 0] + local_x * cos_yaw - local_y * sin_yaw
            pose[:, 1] = layout_origin_xy[:, 1] + local_x * sin_yaw + local_y * cos_yaw
            pose[:, 2] = local_z
            dtype_r = start_pos_w.dtype
            pitch_t = torch.full((n,), local_pitch, dtype=dtype_r, device=device)
            roll_t = torch.full((n,), local_roll, dtype=dtype_r, device=device)
            pose[:, 3:7] = yaw_pitch_roll_to_quat_wxyz(yaw + local_yaw, pitch_t, roll_t)
        ramp_b.write_root_pose_to_sim(pose, env_ids=env_ids)
        ramp_b.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

    env._nav_resample_on_goal = functools.partial(
        update_navigation_path_waypoint,
        env,
        lookahead_distance=lookahead_distance,
        waypoint_reach_radius=waypoint_reach_radius,
        adaptive_lookahead=adaptive_lookahead,
        lookahead_min=lookahead_min,
        curvature_scan_horizon=curvature_scan_horizon,
        curvature_threshold=curvature_threshold,
    )
    update_navigation_path_waypoint(
        env,
        env_ids,
        lookahead_distance=lookahead_distance,
        waypoint_reach_radius=waypoint_reach_radius,
        adaptive_lookahead=adaptive_lookahead,
        lookahead_min=lookahead_min,
        curvature_scan_horizon=curvature_scan_horizon,
        curvature_threshold=curvature_threshold,
    )
