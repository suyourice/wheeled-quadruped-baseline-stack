# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actor-only reset for H-maze hospital teacher training."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from ...common.orientation import yaw_to_quat_wxyz
from ..global_planning.path_state import set_navigation_path_w, update_navigation_path_waypoint
from ..goals import ensure_navigation_goal_buffers
from ..hospital.specs import (
    CURRICULUM_STEPS_PER_ITERATION,
    HOSPITAL_LOW_OBSTACLE_FLAG_HEIGHT,
    HOSPITAL_TRAIN_ACTOR_SLOTS,
    HOSPITAL_TRAIN_ACTOR_LATERAL_MARGIN,
    HOSPITAL_TRAIN_ACTOR_LONGITUDINAL_MARGIN,
    HOSPITAL_TRAIN_CENTER_BLOCK_LATERAL_RANGE,
    HOSPITAL_TRAIN_CENTER_BLOCK_PROBABILITY,
    HOSPITAL_TRAIN_CORRIDOR_WIDTH,
    HOSPITAL_TRAIN_GOAL_EXCLUSION_RADIUS,
    HOSPITAL_TRAIN_LONG_PATH_MAX_STEPS,
    HOSPITAL_TRAIN_LONG_PATH_PROBABILITY,
    HOSPITAL_TRAIN_MAX_PATH_STEPS,
    HOSPITAL_TRAIN_MIN_INTER_ACTOR_DIST,
    HOSPITAL_TRAIN_MIN_PATH_STEPS,
    HOSPITAL_TRAIN_OBSTACLE_CENTER_ZS,
    HOSPITAL_TRAIN_OBSTACLE_CLASS_IDS,
    HOSPITAL_TRAIN_OBSTACLE_DENSITY_SCHEDULE,
    HOSPITAL_TRAIN_OBSTACLE_DEPTHS,
    HOSPITAL_TRAIN_OBSTACLE_HEIGHTS,
    HOSPITAL_TRAIN_OBSTACLE_PRIORITIES,
    HOSPITAL_TRAIN_OBSTACLE_SHAPE_IDS,
    HOSPITAL_TRAIN_OBSTACLE_WIDTHS,
    HOSPITAL_TRAIN_PASSAGE_WIDTH,
    HOSPITAL_TRAIN_START_EXCLUSION_RADIUS,
    NAV_OBSTACLE_RADIUS_MARGIN,
    NAV_WAYPOINT_LOOKAHEAD_DISTANCE,
)
from ..hospital.terrain import (
    HOSPITAL_CENTERLINE_POINT_COUNT,
    MAZE_JUNCTION_NAMES,
    MAZE_JUNCTIONS,
    cached_maze_path,
)
from ..local_planning.obstacle_geometry import set_obstacle_metadata
from ..scenarios import NAV_RANDOM_FALLBACK_SCENARIO_ID as _NAV_RANDOM_FALLBACK_SCENARIO_ID
from ..slotting import _separated_parked_positions

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_nav_path_markers = None
_VIS_Z_LIFT = 2.5  # lift markers above 2.2 m walls so they are visible from overhead
_MAZE_VIS_PATHS: dict[int, torch.Tensor] = {}  # env_id → current path (world coords, (M,3))


def _get_nav_markers():
    """Lazily create orange sphere markers for navigation path waypoints."""
    global _nav_path_markers
    if _nav_path_markers is not None:
        return _nav_path_markers
    try:
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        import isaaclab.sim as sim_utils

        cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/HospNavPath",
            markers={
                "waypoint": sim_utils.SphereCfg(
                    radius=0.30,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
                ),
            },
        )
        _nav_path_markers = VisualizationMarkers(cfg)
    except Exception:
        pass
    return _nav_path_markers


def _visualize_nav_paths(paths: list[torch.Tensor]) -> None:
    """Draw waypoint spheres and connecting lines for every env path.

    Each element of ``paths`` is a (M, 3) tensor (one env).  Markers are lifted
    _VIS_Z_LIFT metres above the floor so they are visible from overview cameras.
    VisualizationMarkers (USD prim) persist until the next call; debug-draw lines
    also persist until cleared on the next reset.
    """
    if not paths:
        return

    raised = []
    for p in paths:
        r = p.clone()
        r[:, 2] = r[:, 2] + _VIS_Z_LIFT
        raised.append(r)

    # Spheres at all waypoints across all envs.
    all_pts = torch.cat(raised, dim=0)
    markers = _get_nav_markers()
    if markers is not None:
        try:
            markers.visualize(translations=all_pts)
        except Exception:
            pass

    # Lines connecting consecutive waypoints within each env path.
    try:
        from isaacsim.util.debug_draw import _debug_draw

        draw = _debug_draw.acquire_debug_draw_interface()
        draw.clear_lines()
        for p_raised in raised:
            pts = p_raised.cpu().tolist()
            n = len(pts) - 1
            if n > 0:
                draw.draw_lines(pts[:-1], pts[1:], [(1.0, 0.55, 0.0, 1.0)] * n, [3.0] * n)
    except Exception:
        pass


def _update_maze_vis(env_id_list: list[int], new_paths: list[torch.Tensor]) -> None:
    """Update per-env path registry and redraw all env visualizations.

    Keeps a global registry so that when only a subset of envs reset, the other
    envs' paths are not lost when clear_lines() wipes the debug-draw canvas.
    """
    global _MAZE_VIS_PATHS
    for eid, p in zip(env_id_list, new_paths):
        _MAZE_VIS_PATHS[eid] = p
    if _MAZE_VIS_PATHS:
        _visualize_nav_paths(list(_MAZE_VIS_PATHS.values()))


def _phase_settings(step: int, steps_per_iteration: int) -> tuple[float, int, int]:
    """Return (target_spacing_m, max_actor_count, phase_index) for the global iteration."""
    iteration = step // max(1, steps_per_iteration)
    phase_index = 0
    for idx, (start_iteration, _, _) in enumerate(HOSPITAL_TRAIN_OBSTACLE_DENSITY_SCHEDULE):
        if iteration >= start_iteration:
            phase_index = idx
    _, spacing, actor_count = HOSPITAL_TRAIN_OBSTACLE_DENSITY_SCHEDULE[phase_index]
    return max(0.0, float(spacing)), min(actor_count, HOSPITAL_TRAIN_ACTOR_SLOTS), phase_index


def _phase_settings_by_id(phase_id: int) -> tuple[float, int]:
    """Return (target_spacing_m, max_actor_count) for a stored curriculum phase id."""
    idx = max(0, min(int(phase_id), len(HOSPITAL_TRAIN_OBSTACLE_DENSITY_SCHEDULE) - 1))
    _, spacing, actor_count = HOSPITAL_TRAIN_OBSTACLE_DENSITY_SCHEDULE[idx]
    return max(0.0, float(spacing)), min(actor_count, HOSPITAL_TRAIN_ACTOR_SLOTS)


def _points_from_corridor_samples(
    segment_starts: torch.Tensor,
    segment_ends: torch.Tensor,
    segment_lengths: torch.Tensor,
    segment_end_s: torch.Tensor,
    s: torch.Tensor,
    lateral: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map batched arc-length/lateral samples to local corridor points and tangents.

    Accepts per-env segment data: segment_* are (n, S, *) with padding, s/lateral are (n,).
    """
    n = s.shape[0]
    device = s.device
    # segment_end_s: (n, S); last column holds total path length per env
    max_s = (segment_end_s[:, -1] - 1.0e-6).clamp(min=0.0)  # (n,)
    s = torch.minimum(s.clamp(min=0.0), max_s)

    # Batched searchsorted: input (n, S), query (n, 1) → result (n, 1)
    seg_idx = torch.searchsorted(
        segment_end_s.contiguous(), s.unsqueeze(1).contiguous(), right=False
    ).squeeze(1)  # (n,)
    seg_idx = seg_idx.clamp(max=segment_lengths.shape[1] - 1)

    row = torch.arange(n, device=device)
    seg_len_sel = segment_lengths[row, seg_idx]                     # (n,)
    seg_start_s = segment_end_s[row, seg_idx] - seg_len_sel         # (n,)
    t = ((s - seg_start_s) / seg_len_sel.clamp(min=1.0e-6)).clamp(0.0, 1.0)
    a = segment_starts[row, seg_idx]                                 # (n, 2)
    b = segment_ends[row, seg_idx]                                   # (n, 2)
    seg = b - a                                                      # (n, 2)
    seg_len = seg_len_sel.clamp(min=1.0e-6)
    center = a + t.unsqueeze(-1) * seg
    ux = seg[:, 0] / seg_len
    uy = seg[:, 1] / seg_len
    points = torch.stack((center[:, 0] - uy * lateral, center[:, 1] + ux * lateral), dim=-1)
    yaws = torch.atan2(seg[:, 1], seg[:, 0])
    return points, yaws


def _has_passable_lateral_gap(
    half_width: float,
    candidate_s: torch.Tensor,
    candidate_lateral: torch.Tensor,
    candidate_cross_half_width: torch.Tensor,
    candidate_forward_half_length: torch.Tensor,
    placed_s: torch.Tensor,
    placed_lateral: torch.Tensor,
    placed_active: torch.Tensor,
    placed_cross_half_widths: torch.Tensor,
    placed_forward_half_lengths: torch.Tensor,
    actor_idx: int,
) -> torch.Tensor:
    """Return True when any lateral interval remains passable at the candidate station.

    candidate_cross_half_width / candidate_forward_half_length: (n,) per-env tensors.
    placed_cross_half_widths / placed_forward_half_lengths: (n, actor_idx) per-env tensors.
    """
    n = candidate_s.shape[0]
    device = candidate_s.device
    dtype = candidate_s.dtype
    count = actor_idx + 1
    left = -float(half_width)
    right = float(half_width)
    inactive_start = torch.full((n,), right, dtype=dtype, device=device)

    starts = torch.empty(n, count, dtype=dtype, device=device)
    ends = torch.empty(n, count, dtype=dtype, device=device)
    starts[:, 0] = candidate_lateral - candidate_cross_half_width - HOSPITAL_TRAIN_ACTOR_LATERAL_MARGIN
    ends[:, 0] = candidate_lateral + candidate_cross_half_width + HOSPITAL_TRAIN_ACTOR_LATERAL_MARGIN

    if actor_idx > 0:
        prev_s = placed_s[:, :actor_idx]                              # (n, actor_idx)
        prev_lateral = placed_lateral[:, :actor_idx]                  # (n, actor_idx)
        prev_active = placed_active[:, :actor_idx]                    # (n, actor_idx)
        prev_cross = placed_cross_half_widths[:, :actor_idx]          # (n, actor_idx)
        prev_forward = placed_forward_half_lengths[:, :actor_idx]     # (n, actor_idx)
        overlap = (
            (candidate_s.unsqueeze(1) - prev_s).abs()
            <= candidate_forward_half_length.unsqueeze(1) + prev_forward + HOSPITAL_TRAIN_ACTOR_LONGITUDINAL_MARGIN
        ) & prev_active
        prev_starts = prev_lateral - prev_cross - HOSPITAL_TRAIN_ACTOR_LATERAL_MARGIN
        prev_ends = prev_lateral + prev_cross + HOSPITAL_TRAIN_ACTOR_LATERAL_MARGIN
        starts[:, 1:] = torch.where(overlap, prev_starts, inactive_start.unsqueeze(1))
        ends[:, 1:] = torch.where(overlap, prev_ends, inactive_start.unsqueeze(1))

    starts = starts.clamp(min=left, max=right)
    ends = ends.clamp(min=left, max=right)
    starts, order = starts.sort(dim=1)
    ends = ends.gather(1, order)

    current_end = torch.full((n,), left, dtype=dtype, device=device)
    max_gap = torch.zeros(n, dtype=dtype, device=device)
    for slot_idx in range(count):
        max_gap = torch.maximum(max_gap, starts[:, slot_idx] - current_end)
        current_end = torch.maximum(current_end, ends[:, slot_idx])
    max_gap = torch.maximum(max_gap, torch.full((n,), right, dtype=dtype, device=device) - current_end)
    return max_gap >= HOSPITAL_TRAIN_PASSAGE_WIDTH


def _sample_corridor_points_batch(
    segment_starts: torch.Tensor,
    segment_ends: torch.Tensor,
    segment_lengths: torch.Tensor,
    segment_end_s: torch.Tensor,
    half_width: float,
    slot_cross_half_width: torch.Tensor,
    slot_forward_half_length: torch.Tensor,
    start_xy: torch.Tensor,
    goal_xy: torch.Tensor,
    placed_xy: torch.Tensor,
    placed_s: torch.Tensor,
    placed_lateral: torch.Tensor,
    placed_active: torch.Tensor,
    placed_cross_half_widths: torch.Tensor,
    placed_forward_half_lengths: torch.Tensor,
    actor_idx: int,
    actor_counts: torch.Tensor,
    attempts: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one actor slot for all resetting envs in a single batched GPU call.

    All segment_* tensors are (n, S, *) with padding for differing path lengths.
    slot_cross_half_width / slot_forward_half_length are (n,) per-env tensors because
    each env uses a different slot permutation.
    start_xy / goal_xy are (n, 2).
    actor_counts is (n,) — per-env total actor count for bucket sizing.
    """
    n = placed_xy.shape[0]
    device = placed_xy.device
    dtype = placed_xy.dtype

    # Per-env total path length from the last (non-padded) segment cumsum value.
    total_length = segment_end_s[:, -1].clamp(min=1.0e-6)                 # (n,)
    lateral_limit = (half_width - slot_cross_half_width - 0.12).clamp(min=0.05)  # (n,)

    result = torch.zeros(n, 2, dtype=dtype, device=device)
    result_yaw = torch.zeros(n, dtype=dtype, device=device)
    result_s = torch.zeros(n, dtype=dtype, device=device)
    result_lateral = torch.zeros(n, dtype=dtype, device=device)
    unresolved = torch.ones(n, dtype=torch.bool, device=device)

    start_cap = HOSPITAL_TRAIN_START_EXCLUSION_RADIUS + 0.2
    goal_margin = HOSPITAL_TRAIN_GOAL_EXCLUSION_RADIUS + 0.2
    usable_start = torch.minimum(total_length * 0.45, total_length.new_full((n,), start_cap))
    usable_end = torch.maximum(usable_start + 1.0e-3, total_length - goal_margin)
    span = (usable_end - usable_start).clamp(min=1.0e-3)
    bucket = span / actor_counts.to(dtype=dtype).clamp(min=1.0)           # (n,)
    bucket_center = usable_start + bucket * (actor_idx + 0.5)             # (n,)

    center_span = torch.minimum(lateral_limit, lateral_limit.new_full((n,), HOSPITAL_TRAIN_CENTER_BLOCK_LATERAL_RANGE))

    for _ in range(attempts):
        s = bucket_center + (torch.rand(n, dtype=dtype, device=device) - 0.5) * bucket * 0.85
        s = torch.maximum(torch.minimum(s, usable_end), usable_start)
        lateral = (torch.rand(n, dtype=dtype, device=device) * 2.0 - 1.0) * lateral_limit
        center_lateral = (torch.rand(n, dtype=dtype, device=device) * 2.0 - 1.0) * center_span
        center_mask = torch.rand(n, dtype=dtype, device=device) < HOSPITAL_TRAIN_CENTER_BLOCK_PROBABILITY
        lateral = torch.where(center_mask, center_lateral, lateral)
        candidate, candidate_yaw = _points_from_corridor_samples(
            segment_starts, segment_ends, segment_lengths, segment_end_s, s, lateral
        )
        valid = (candidate - start_xy).norm(dim=-1) >= HOSPITAL_TRAIN_START_EXCLUSION_RADIUS
        valid &= (candidate - goal_xy).norm(dim=-1) >= HOSPITAL_TRAIN_GOAL_EXCLUSION_RADIUS
        valid &= _has_passable_lateral_gap(
            half_width,
            s,
            lateral,
            slot_cross_half_width,
            slot_forward_half_length,
            placed_s,
            placed_lateral,
            placed_active,
            placed_cross_half_widths,
            placed_forward_half_lengths,
            actor_idx,
        )
        if actor_idx > 0:
            sep = (candidate.unsqueeze(1) - placed_xy[:, :actor_idx]).norm(dim=-1)
            sep = torch.where(placed_active[:, :actor_idx], sep, sep.new_full(sep.shape, 1.0e6))
            min_sep = sep.min(dim=1).values
            valid &= min_sep >= HOSPITAL_TRAIN_MIN_INTER_ACTOR_DIST
        take = unresolved & valid
        result = torch.where(take.unsqueeze(-1), candidate, result)
        result_yaw = torch.where(take, candidate_yaw, result_yaw)
        result_s = torch.where(take, s, result_s)
        result_lateral = torch.where(take, lateral, result_lateral)
        unresolved &= ~take

    fraction = (actor_idx + 0.5) / actor_counts.to(dtype=dtype).clamp(min=1.0)
    fallback_s = usable_start + (usable_end - usable_start) * fraction
    fallback_lateral = (1.0 if actor_idx % 2 == 0 else -1.0) * torch.minimum(
        lateral_limit * 0.40, lateral_limit.new_full((n,), 0.30)
    )
    fallback, fallback_yaw = _points_from_corridor_samples(
        segment_starts, segment_ends, segment_lengths, segment_end_s, fallback_s, fallback_lateral
    )
    fallback_valid = _has_passable_lateral_gap(
        half_width,
        fallback_s,
        fallback_lateral,
        slot_cross_half_width,
        slot_forward_half_length,
        placed_s,
        placed_lateral,
        placed_active,
        placed_cross_half_widths,
        placed_forward_half_lengths,
        actor_idx,
    )
    take_fallback = unresolved & fallback_valid
    result = torch.where(take_fallback.unsqueeze(-1), fallback, result)
    result_yaw = torch.where(take_fallback, fallback_yaw, result_yaw)
    result_s = torch.where(take_fallback, fallback_s, result_s)
    result_lateral = torch.where(take_fallback, fallback_lateral, result_lateral)
    return result, result_yaw, result_s, result_lateral, ~unresolved | take_fallback


def _ensure_hospital_buffers(env: ManagerBasedRLEnv) -> None:
    """Allocate hospital-layout diagnostic and observation buffers."""
    device = env.device
    if (
        not hasattr(env, "_go2w_structured_corridor_start_xy")
        or env._go2w_structured_corridor_start_xy.shape != (env.num_envs, 2)
    ):
        env._go2w_structured_corridor_start_xy = torch.zeros(env.num_envs, 2, device=device)
        env._go2w_structured_corridor_yaw = torch.zeros(env.num_envs, device=device)
        env._go2w_structured_corridor_width = torch.zeros(env.num_envs, device=device)
        env._go2w_structured_corridor_leg_length = torch.zeros(env.num_envs, device=device)
        env._go2w_structured_corridor_centerline_count = torch.zeros(
            env.num_envs, dtype=torch.long, device=device
        )
        env._go2w_structured_corridor_centerline_local = torch.zeros(
            env.num_envs, HOSPITAL_CENTERLINE_POINT_COUNT, 2, device=device
        )
        env._go2w_hospital_layout_id = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        env._go2w_hospital_phase_id = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        env._go2w_hospital_path_reversed = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        env._go2w_hospital_actor_count = torch.zeros(env.num_envs, device=device)


def reset_hospital_maze_training(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    obstacle_names: list[str],
    *,
    lookahead_distance: float = NAV_WAYPOINT_LOOKAHEAD_DISTANCE,
    waypoint_reach_radius: float = 0.45,
    steps_per_iteration: int = CURRICULUM_STEPS_PER_ITERATION,
    obstacle_radius_margin: float = NAV_OBSTACLE_RADIUS_MARGIN,
    park_distance: float = 1000.0,
    reset_robot_pose: bool = True,
    force_path_id: int | None = None,
    curriculum_iteration_offset: int = 0,
    debug_vis: bool = False,
) -> None:
    """Reset envs into the 5×5 junction-grid maze with actor-only corridor obstacles."""
    if len(env_ids) == 0 or len(obstacle_names) == 0:
        return
    if len(obstacle_names) != len(HOSPITAL_TRAIN_OBSTACLE_SHAPE_IDS):
        raise ValueError("Hospital teacher reset requires the actor-only hospital obstacle table.")

    ensure_navigation_goal_buffers(env)
    _ensure_hospital_buffers(env)
    if reset_robot_pose:
        if hasattr(env, "_go2w_goals_reached_episode"):
            env._go2w_goals_reached_episode[env_ids] = 0.0
        if hasattr(env, "_go2w_first_goal_reached_episode"):
            env._go2w_first_goal_reached_episode[env_ids] = False
        if hasattr(env, "_go2w_min_goal_distance_episode"):
            env._go2w_min_goal_distance_episode[env_ids] = float("inf")
        if hasattr(env, "_go2w_had_collision_episode"):
            env._go2w_had_collision_episode[env_ids] = False
    if hasattr(env, "_go2w_ignore_obstacle_contact_history_steps"):
        env._go2w_ignore_obstacle_contact_history_steps[env_ids] = 0
    if hasattr(env, "_go2w_hospital_prev_progress_s"):
        env._go2w_hospital_prev_progress_s[env_ids] = 0.0
    if hasattr(env, "_go2w_hospital_stuck_counter"):
        env._go2w_hospital_stuck_counter[env_ids] = 0.0
    if hasattr(env, "_go2w_hospital_centerline_cache_key"):
        env._go2w_hospital_centerline_cache_key = None

    device = env.device
    n = len(env_ids)
    k = len(obstacle_names)
    robot = env.scene["robot"]
    env_origins = env.scene.terrain.terrain_origins.reshape(-1, 3)[env_ids]
    current_root_pos = robot.data.root_pos_w[env_ids, :3].clone()
    current_heading = robot.data.heading_w[env_ids].clone()

    curriculum_step = int(getattr(env, "common_step_counter", 0)) + int(curriculum_iteration_offset) * max(
        1, steps_per_iteration
    )
    target_spacing, actor_count_cap, phase_id = _phase_settings(curriculum_step, steps_per_iteration)

    _NUM_J = len(MAZE_JUNCTION_NAMES)
    _MIN_PATH_STEPS = HOSPITAL_TRAIN_MIN_PATH_STEPS
    _j_xs = [MAZE_JUNCTIONS[name][0] for name in MAZE_JUNCTION_NAMES]
    _j_ys = [MAZE_JUNCTIONS[name][1] for name in MAZE_JUNCTION_NAMES]

    if reset_robot_pose:
        if force_path_id is None:
            layout_indices = torch.zeros(n, dtype=torch.long, device=device)
            for _li in range(n):
                _s_i, _e_i = 0, 0
                max_path_steps = (
                    HOSPITAL_TRAIN_LONG_PATH_MAX_STEPS
                    if torch.rand((), device=device).item() < HOSPITAL_TRAIN_LONG_PATH_PROBABILITY
                    else HOSPITAL_TRAIN_MAX_PATH_STEPS
                )
                chosen = False
                for _ in range(80):
                    _s_i = int(torch.randint(0, _NUM_J, (1,)).item())
                    _e_i = int(torch.randint(0, _NUM_J, (1,)).item())
                    if _s_i != _e_i:
                        _steps = (abs(_j_xs[_s_i] - _j_xs[_e_i]) + abs(_j_ys[_s_i] - _j_ys[_e_i])) / 10.0
                        if _MIN_PATH_STEPS <= _steps <= max_path_steps:
                            chosen = True
                            break
                if not chosen:
                    _s_i, _e_i = 0, _MIN_PATH_STEPS
                layout_indices[_li] = _s_i * _NUM_J + _e_i
        else:
            layout_indices = torch.full((n,), int(force_path_id) % (_NUM_J * _NUM_J), dtype=torch.long, device=device)
        phase_indices = torch.full((n,), phase_id, dtype=torch.long, device=device)
        reversed_flags = torch.zeros(n, dtype=torch.bool, device=device)
    else:
        layout_indices = env._go2w_hospital_layout_id[env_ids].clone()
        phase_indices = env._go2w_hospital_phase_id[env_ids].clone()
        reversed_flags = ~env._go2w_hospital_path_reversed[env_ids]

    active_mask = torch.zeros(n, k, dtype=torch.bool, device=device)
    obstacle_yaws = torch.zeros(n, k, device=device)
    parked_world = env_origins.clone()
    parked_world[:, 0] += park_distance
    positions = _separated_parked_positions(parked_world, k)
    robot_pose = torch.zeros(n, 7, device=device)
    robot_pose[:, :3] = current_root_pos
    robot_pose[:, 3:7] = yaw_to_quat_wxyz(current_heading)

    env._go2w_gap_passable[env_ids] = False
    env._go2w_scenario_template_id[env_ids] = _NAV_RANDOM_FALLBACK_SCENARIO_ID
    env._go2w_initial_scenario_template_id[env_ids] = _NAV_RANDOM_FALLBACK_SCENARIO_ID
    env._go2w_hospital_layout_id[env_ids] = layout_indices
    env._go2w_hospital_phase_id[env_ids] = phase_indices
    env._go2w_hospital_path_reversed[env_ids] = reversed_flags

    dtype = current_root_pos.dtype
    actor_cross_half_widths_t = torch.tensor(
        [HOSPITAL_TRAIN_OBSTACLE_DEPTHS[i] * 0.5 for i in range(HOSPITAL_TRAIN_ACTOR_SLOTS)],
        dtype=dtype, device=device,
    )
    actor_forward_half_lengths_t = torch.tensor(
        [HOSPITAL_TRAIN_OBSTACLE_WIDTHS[i] * 0.5 for i in range(HOSPITAL_TRAIN_ACTOR_SLOTS)],
        dtype=dtype, device=device,
    )
    center_zs_t = torch.tensor(HOSPITAL_TRAIN_OBSTACLE_CENTER_ZS, dtype=dtype, device=device)

    # -----------------------------------------------------------------------
    # Phase 1: Python loop — collect per-env path specs (no GPU actor placement)
    # -----------------------------------------------------------------------
    layout_specs: list[dict] = []
    layout_actor_counts: list[int] = []
    slot_perms: list[torch.Tensor] = []
    origin_xys: list[torch.Tensor] = []
    _vis_new: list[tuple[int, torch.Tensor]] = []

    for _li in range(n):
        env_id_single = env_ids[_li:_li + 1]
        local_phase = int(phase_indices[_li].item())
        if reset_robot_pose:
            env_target_spacing = target_spacing
            env_actor_count_cap = actor_count_cap
        else:
            env_target_spacing, env_actor_count_cap = _phase_settings_by_id(local_phase)
        env_width = HOSPITAL_TRAIN_CORRIDOR_WIDTH

        layout_id = int(layout_indices[_li].item())
        start_idx = layout_id // _NUM_J
        end_idx = layout_id % _NUM_J

        layout_spec = cached_maze_path(start_idx, end_idx)
        total_length = float(layout_spec["total_length"])
        layout_actor_count = 0
        if env_target_spacing > 1.0e-6:
            layout_actor_count = min(env_actor_count_cap, max(0, int(total_length / env_target_spacing)), k)

        fixed_path_local = layout_spec["fixed_path_local"].to(device=device, dtype=dtype)
        if reversed_flags[_li]:
            fixed_path_local = torch.flip(fixed_path_local, dims=(0,))

        origin_xy = env_origins[_li, :2]   # (2,)
        _N = fixed_path_local.shape[0]
        path_w = torch.zeros(1, _N, 3, dtype=dtype, device=device)
        path_w[0, :, :2] = origin_xy + fixed_path_local[:, :2]
        path_w[0, :, 2] = current_root_pos[_li, 2]

        if reset_robot_pose:
            start_xy = path_w[0, 0, :2]
            path_delta = path_w[0, 1, :2] - path_w[0, 0, :2]
            start_yaw_val = torch.atan2(path_delta[1], path_delta[0])
            robot_pose[_li, :2] = start_xy
            robot_pose[_li, 3:7] = yaw_to_quat_wxyz(start_yaw_val)
            current_heading[_li] = start_yaw_val

        if debug_vis:
            _vis_new.append((int(env_ids[_li].item()), path_w[0].detach()))

        env._go2w_navigation_path_direct_goal = True
        set_navigation_path_w(env, env_id_single, path_w)

        centerline_tensor = layout_spec["centerline_tensor"].to(device=device, dtype=dtype)
        centerline = layout_spec["centerline"]
        env._go2w_structured_corridor_start_xy[env_id_single] = origin_xy.unsqueeze(0)
        env._go2w_structured_corridor_yaw[env_id_single] = 0.0
        env._go2w_structured_corridor_width[env_id_single] = env_width
        env._go2w_structured_corridor_leg_length[env_id_single] = total_length
        env._go2w_structured_corridor_centerline_count[env_id_single] = len(centerline)
        env._go2w_structured_corridor_centerline_local[env_id_single] = centerline_tensor.unsqueeze(0)

        layout_specs.append(layout_spec)
        layout_actor_counts.append(layout_actor_count)
        slot_perms.append(torch.randperm(k, device=device))
        origin_xys.append(origin_xy)

    # -----------------------------------------------------------------------
    # Phase 2: build padded batched segment tensors
    # -----------------------------------------------------------------------
    max_actor_count = max(layout_actor_counts) if layout_actor_counts else 0

    if max_actor_count > 0:
        max_seg = max(len(spec["segments"]) for spec in layout_specs)

        seg_starts_list: list[torch.Tensor] = []
        seg_ends_list: list[torch.Tensor] = []
        seg_lengths_list: list[torch.Tensor] = []
        seg_end_s_list: list[torch.Tensor] = []
        start_xy_list: list[torch.Tensor] = []
        goal_xy_list: list[torch.Tensor] = []

        for _li, spec in enumerate(layout_specs):
            segs = spec["segments"]
            S = len(segs)
            seg_t = torch.tensor(
                [(a[0], a[1], b[0], b[1], length) for a, b, length in segs],
                dtype=dtype, device=device,
            )  # (S, 5)
            lengths = seg_t[:, 4].clamp(min=1.0e-6)
            end_s = torch.cumsum(lengths, dim=0)
            total_len = end_s[-1].item()

            pad = max_seg - S
            seg_starts_list.append(torch.nn.functional.pad(seg_t[:, 0:2], (0, 0, 0, pad)))
            seg_ends_list.append(torch.nn.functional.pad(seg_t[:, 2:4], (0, 0, 0, pad)))
            # Pad lengths with a small non-zero value to avoid division by zero in lookup.
            seg_lengths_list.append(torch.nn.functional.pad(lengths, (0, pad), value=1.0e-6))
            # Pad end_s with total_length so searchsorted always resolves to the last real segment.
            seg_end_s_list.append(torch.nn.functional.pad(end_s, (0, pad), value=total_len))

            fixed_path = spec["fixed_path_local"]
            if reversed_flags[_li]:
                fixed_path = torch.flip(fixed_path, dims=(0,))
            start_xy_list.append(fixed_path[0, :2].to(dtype=dtype, device=device))
            goal_xy_list.append(fixed_path[-1, :2].to(dtype=dtype, device=device))

        stacked_seg_starts = torch.stack(seg_starts_list)   # (n, max_seg, 2)
        stacked_seg_ends = torch.stack(seg_ends_list)        # (n, max_seg, 2)
        stacked_seg_lengths = torch.stack(seg_lengths_list)  # (n, max_seg)
        stacked_seg_end_s = torch.stack(seg_end_s_list)      # (n, max_seg)
        stacked_start_xy = torch.stack(start_xy_list)        # (n, 2)
        stacked_goal_xy = torch.stack(goal_xy_list)          # (n, 2)
        stacked_origin_xy = torch.stack(origin_xys)          # (n, 2)
        slot_perms_t = torch.stack(slot_perms)               # (n, k)
        layout_actor_counts_t = torch.tensor(layout_actor_counts, dtype=torch.long, device=device)

        # -----------------------------------------------------------------------
        # Phase 3: batched GPU actor placement — max_actor_count calls total
        # -----------------------------------------------------------------------
        half_width = HOSPITAL_TRAIN_CORRIDOR_WIDTH * 0.5

        actor_pos_local = torch.zeros(n, max_actor_count, 2, dtype=dtype, device=device)
        actor_s_local = torch.zeros(n, max_actor_count, dtype=dtype, device=device)
        actor_lat_local = torch.zeros(n, max_actor_count, dtype=dtype, device=device)
        actor_active_local = torch.zeros(n, max_actor_count, dtype=torch.bool, device=device)
        # Track per-env placed slot dimensions for passable-gap checks.
        placed_cross_hw = torch.zeros(n, max_actor_count, dtype=dtype, device=device)
        placed_forward_hl = torch.zeros(n, max_actor_count, dtype=dtype, device=device)

        row_idx = torch.arange(n, device=device)

        for actor_idx in range(max_actor_count):
            needs_actor = layout_actor_counts_t > actor_idx  # (n,) bool mask

            # Per-env slot index for this actor position (from each env's permutation).
            slot_idx_per_env = slot_perms_t[:, actor_idx]    # (n,)

            # Per-env actor footprint dimensions.
            cross_hw = actor_cross_half_widths_t[slot_idx_per_env]    # (n,)
            forward_hl = actor_forward_half_lengths_t[slot_idx_per_env]  # (n,)

            actor_xy, actor_yaw, actor_s, actor_lat, actor_valid = _sample_corridor_points_batch(
                stacked_seg_starts, stacked_seg_ends, stacked_seg_lengths, stacked_seg_end_s,
                half_width,
                cross_hw, forward_hl,
                stacked_start_xy, stacked_goal_xy,
                actor_pos_local, actor_s_local, actor_lat_local, actor_active_local,
                placed_cross_hw, placed_forward_hl,
                actor_idx, layout_actor_counts_t,
            )

            # Mask out envs that don't need this actor slot.
            actor_valid = actor_valid & needs_actor

            actor_pos_local[:, actor_idx] = actor_xy
            actor_s_local[:, actor_idx] = actor_s
            actor_lat_local[:, actor_idx] = actor_lat
            actor_active_local[:, actor_idx] = actor_valid
            placed_cross_hw[:, actor_idx] = cross_hw
            placed_forward_hl[:, actor_idx] = forward_hl

            # Scatter valid placements into the world position / yaw buffers.
            valid_rows = row_idx[actor_valid]
            valid_slots = slot_idx_per_env[actor_valid]
            if valid_rows.numel() > 0:
                positions[valid_rows, valid_slots, :2] = stacked_origin_xy[valid_rows] + actor_xy[actor_valid]
                positions[valid_rows, valid_slots, 2] = center_zs_t[valid_slots]
                obstacle_yaws[valid_rows, valid_slots] = actor_yaw[actor_valid]
                active_mask[valid_rows, valid_slots] = True

        env._go2w_hospital_actor_count[env_ids] = active_mask.sum(dim=1).float()
    else:
        env._go2w_hospital_actor_count[env_ids] = 0.0

    if _vis_new:
        _update_maze_vis([eid for eid, _ in _vis_new], [p for _, p in _vis_new])

    if reset_robot_pose:
        robot.write_root_pose_to_sim(robot_pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)
        current_root_pos = robot_pose[:, :3]

    env._go2w_start_pos_w[env_ids] = current_root_pos
    env._go2w_start_heading_w[env_ids] = current_heading

    set_obstacle_metadata(
        env,
        env_ids,
        obstacle_names,
        active_mask,
        obstacle_radius_margin=obstacle_radius_margin,
        fixed_obstacle_shape_ids=HOSPITAL_TRAIN_OBSTACLE_SHAPE_IDS,
        fixed_obstacle_widths=HOSPITAL_TRAIN_OBSTACLE_WIDTHS,
        fixed_obstacle_depths=HOSPITAL_TRAIN_OBSTACLE_DEPTHS,
        fixed_obstacle_heights=HOSPITAL_TRAIN_OBSTACLE_HEIGHTS,
        fixed_obstacle_center_zs=HOSPITAL_TRAIN_OBSTACLE_CENTER_ZS,
        fixed_obstacle_class_ids=HOSPITAL_TRAIN_OBSTACLE_CLASS_IDS,
        fixed_obstacle_priorities=HOSPITAL_TRAIN_OBSTACLE_PRIORITIES,
        dynamic_mask=(False,) * k,
        obstacle_yaws=obstacle_yaws,
        low_obstacle_height_threshold=HOSPITAL_LOW_OBSTACLE_FLAG_HEIGHT,
    )

    zero_vel = torch.zeros(n, 6, device=device)
    for slot_idx, name in enumerate(obstacle_names):
        pose = torch.zeros(n, 7, device=device)
        pose[:, :3] = positions[:, slot_idx]
        pose[:, 3:7] = yaw_to_quat_wxyz(obstacle_yaws[:, slot_idx])
        env.scene[name].write_root_pose_to_sim(pose, env_ids=env_ids)
        env.scene[name].write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

    env._nav_resample_on_goal = functools.partial(
        reset_hospital_maze_training,
        env,
        obstacle_names=obstacle_names,
        lookahead_distance=lookahead_distance,
        waypoint_reach_radius=waypoint_reach_radius,
        steps_per_iteration=steps_per_iteration,
        obstacle_radius_margin=obstacle_radius_margin,
        park_distance=park_distance,
        reset_robot_pose=False,
        force_path_id=force_path_id,
        debug_vis=debug_vis,
    )
    update_navigation_path_waypoint(
        env,
        env_ids,
        lookahead_distance=lookahead_distance,
        waypoint_reach_radius=waypoint_reach_radius,
        adaptive_lookahead=True,
        lookahead_min=0.55,
        curvature_scan_horizon=2.5,
        curvature_threshold=0.3,
    )
