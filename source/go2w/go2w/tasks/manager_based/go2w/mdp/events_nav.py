# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation goal / obstacle reset functions for the Go2-W environment."""

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
)
from .nav_scenarios import (
    NAV_RANDOM_FALLBACK_SCENARIO_ID as _NAV_RANDOM_FALLBACK_SCENARIO_ID,
    NAV_SCENARIO_CODES as _NAV_SCENARIO_CODES,
)
from .nav_slotting import (
    _assign_logical_positions_to_physical_slots,
    _physical_slot_randomization_mask,
    _separated_parked_positions,
)
from .obstacle_geometry import obstacle_effective_radius, set_obstacle_metadata

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resample_nav_on_goal_reached(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    obstacle_names: list[str],
    num_obstacles: int,
    spawn_half_x: float,
    spawn_half_y: float,
    min_inter_obstacle_dist: float,
    goal_forward_range: tuple[float, float],
    goal_lateral_range: tuple[float, float],
    goal_heading_jitter_range: tuple[float, float],
    min_goal_distance: float,
    start_exclusion_radius: float,
    goal_exclusion_radius: float,
    obstacle_z: float,
    park_distance: float,
    obstacle_radius_margin: float = 0.0,
    fixed_obstacle_shape_ids: tuple[int, ...] | None = None,
    fixed_obstacle_widths: tuple[float, ...] | None = None,
    fixed_obstacle_depths: tuple[float, ...] | None = None,
    randomize_physical_obstacle_slots: bool = False,
    physical_slot_randomization_start_iteration: int = 500,
    physical_slot_randomization_warmup_iterations: int = 500,
    steps_per_iteration: int = 128,
    randomize_obstacle_yaw: bool = False,
    obstacle_yaw_range: tuple[float, float] = (-math.pi, math.pi),
) -> None:
    """Resample goal and obstacles mid-episode, centered on the robot's current world position.

    Unlike reset_navigation_goals_and_obstacles (which uses env_origin-relative
    spawn ranges), this function places obstacles in a box centered on the
    midpoint between the robot and the new goal in world coordinates.  This
    ensures obstacles appear along the path regardless of how far the robot has
    travelled from the env origin.

    Fully vectorized on GPU: all n envs are processed in parallel using tensor
    operations instead of Python for-loops, avoiding CPU bottlenecks when many
    envs reach the goal simultaneously.
    """
    ensure_navigation_goal_buffers(env)
    robot = env.scene["robot"]
    n = len(env_ids)
    device = env.device

    # --- sample new goal (vectorized over n envs) ---
    curr_pos_w = robot.data.root_pos_w[env_ids, :3].clone()
    yaw_t = robot.data.heading_w[env_ids]  # (n,)

    T_GOAL = 50
    fwd_lo, fwd_hi = goal_forward_range
    lat_lo, lat_hi = goal_lateral_range
    fwd_cands = torch.empty(n, T_GOAL, device=device).uniform_(fwd_lo, fwd_hi)  # (n, T)
    lat_cands = torch.empty(n, T_GOAL, device=device).uniform_(lat_lo, lat_hi)  # (n, T)
    dist_cands = (fwd_cands**2 + lat_cands**2).sqrt()                            # (n, T)
    valid_goal = dist_cands >= min_goal_distance                                  # (n, T)
    # argmax picks first True; if none valid, picks 0 (forward is always >=fwd_lo
    # which is typically >= min_goal_distance, so this is safe in practice)
    best_idx = valid_goal.long().argmax(dim=1)                                    # (n,)
    arange_n = torch.arange(n, device=device)
    fwd_chosen = fwd_cands[arange_n, best_idx]                                    # (n,)
    lat_chosen = lat_cands[arange_n, best_idx]                                    # (n,)

    cos_y = yaw_t.cos()
    sin_y = yaw_t.sin()
    dx = fwd_chosen * cos_y - lat_chosen * sin_y  # (n,)
    dy = fwd_chosen * sin_y + lat_chosen * cos_y  # (n,)

    goal_pos_w = curr_pos_w.clone()
    goal_pos_w[:, 0] += dx
    goal_pos_w[:, 1] += dy

    path_heading = torch.atan2(dy, dx)  # (n,)
    jit_lo, jit_hi = goal_heading_jitter_range
    jitter = torch.empty(n, device=device).uniform_(jit_lo, jit_hi)
    ph_j = path_heading + jitter
    goal_heading_w = torch.atan2(ph_j.sin(), ph_j.cos())  # (n,)

    env._go2w_goal_pos_w[env_ids] = goal_pos_w
    env._go2w_goal_heading_w[env_ids] = goal_heading_w
    env._go2w_start_pos_w[env_ids] = curr_pos_w
    env._go2w_start_heading_w[env_ids] = yaw_t.clone()

    # Mid-episode resample produces a generic random layout (not a templated gap),
    # so disable the passable-gap shaping and reset the stuck counter for these envs.
    if hasattr(env, "_go2w_gap_passable"):
        env._go2w_gap_passable[env_ids] = False
    if hasattr(env, "_go2w_stuck_counter"):
        env._go2w_stuck_counter[env_ids] = 0.0

    # A mid-episode resample is a new generic scenario. Contact sensors retain
    # history across this pose write, so the reward term will discard stale frames.
    env._go2w_scenario_template_id[env_ids] = _NAV_RANDOM_FALLBACK_SCENARIO_ID
    if not hasattr(env, "_go2w_obstacle_pose_changed_midstep"):
        env._go2w_obstacle_pose_changed_midstep = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    env._go2w_obstacle_pose_changed_midstep[env_ids] = True

    # --- place obstacles (vectorized over n envs, sequential over obstacle slots) ---
    # Fallback position: park away from the arena so the obstacle is inactive.
    park_xy = curr_pos_w[:, :2].clone()
    park_xy[:, 0] += park_distance  # (n, 2)

    robot_xy = curr_pos_w[:, :2]   # (n, 2)
    goal_xy = goal_pos_w[:, :2]    # (n, 2)
    midpoint_xy = (robot_xy + goal_xy) * 0.5  # (n, 2)

    effective = min(num_obstacles, len(obstacle_names))
    T_OBS = 200  # candidates per obstacle per env (more attempts, same GPU cost)
    half = torch.tensor([spawn_half_x, spawn_half_y], device=device)  # (2,)

    # placed[:, i, :] = world-xy of the i-th placed obstacle, NaN until filled.
    placed = torch.full((n, effective, 2), float("nan"), device=device)
    placed_active = torch.zeros(n, len(obstacle_names), dtype=torch.bool, device=device)

    # Pre-allocate candidate buffer once; reuse in-place each obstacle iteration.
    cands_buf = torch.empty(n, T_OBS, 2, device=device)
    robot_xy_exp = robot_xy.unsqueeze(1)  # (n, 1, 2) — constant across iterations
    goal_xy_exp = goal_xy.unsqueeze(1)    # (n, 1, 2)
    mid_exp = midpoint_xy.unsqueeze(1)    # (n, 1, 2)

    for obs_i in range(effective):
        # Resample candidates in-place to avoid repeated allocation.
        cands_buf.uniform_(-1.0, 1.0).mul_(half)
        cands = mid_exp + cands_buf  # (n, T_OBS, 2)

        # Constraint 1: outside robot exclusion zone
        valid = (cands - robot_xy_exp).norm(dim=-1) >= start_exclusion_radius  # (n, T_OBS)

        # Constraint 2: outside goal exclusion zone
        valid &= (cands - goal_xy_exp).norm(dim=-1) >= goal_exclusion_radius

        # Constraint 3: away from all previously placed obstacles at once.
        # placed[:, :obs_i, :] is (n, obs_i, 2); broadcast to (n, T_OBS, obs_i).
        if obs_i > 0:
            prev_all = placed[:, :obs_i, :].unsqueeze(1)                        # (n, 1, obs_i, 2)
            cands_exp = cands.unsqueeze(2)                                       # (n, T_OBS, 1, 2)
            d_all = (cands_exp - prev_all).norm(dim=-1)                         # (n, T_OBS, obs_i)
            valid &= (d_all >= min_inter_obstacle_dist).all(dim=2)

        # Pick first valid candidate; fall back to park position if none found.
        has_valid = valid.any(dim=1)                                             # (n,)
        first_valid_idx = valid.long().argmax(dim=1)                             # (n,)
        chosen = cands[arange_n, first_valid_idx]                                # (n, 2)
        chosen = torch.where(has_valid.unsqueeze(1), chosen, park_xy)
        placed[:, obs_i, :] = chosen
        placed_active[:, obs_i] = has_valid

    parked_world = torch.zeros(n, 3, device=device)
    parked_world[:, :2] = park_xy
    parked_world[:, 2] = obstacle_z
    parked_positions = _separated_parked_positions(parked_world, len(obstacle_names))
    logical_positions = parked_positions.clone()
    sampled_positions = logical_positions[:, :effective].clone()
    sampled_positions[:, :, :2] = placed
    logical_positions[:, :effective] = torch.where(
        placed_active[:, :effective].unsqueeze(-1), sampled_positions, logical_positions[:, :effective]
    )
    logical_active = placed_active
    randomize_slot_mask = _physical_slot_randomization_mask(
        env,
        n,
        randomize_physical_obstacle_slots,
        physical_slot_randomization_start_iteration,
        physical_slot_randomization_warmup_iterations,
        steps_per_iteration,
        device,
    )
    physical_positions, placed_active, logical_to_physical = _assign_logical_positions_to_physical_slots(
        logical_positions, logical_active, parked_positions, randomize_slot_mask
    )
    logical_yaws = torch.zeros(n, len(obstacle_names), device=device)
    if randomize_obstacle_yaw:
        sampled_yaws = torch.empty(n, len(obstacle_names), device=device).uniform_(*obstacle_yaw_range)
        logical_yaws = torch.where(logical_active, sampled_yaws, logical_yaws)
    physical_yaws = torch.zeros_like(logical_yaws)
    physical_yaws.scatter_(1, logical_to_physical, logical_yaws)

    # Write all physical obstacle poses to sim.
    pose_buf = torch.zeros(n, 7, device=device)
    zero_vel = torch.zeros(n, 6, device=device)
    for slot_idx, name in enumerate(obstacle_names):
        obstacle = env.scene[name]
        pose_buf[:, :3] = physical_positions[:, slot_idx]
        pose_buf[:, 3:7] = yaw_to_quat_wxyz(physical_yaws[:, slot_idx])
        obstacle.write_root_pose_to_sim(pose_buf.clone(), env_ids=env_ids)
        obstacle.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

    set_obstacle_metadata(
        env,
        env_ids,
        obstacle_names,
        placed_active,
        obstacle_radius_margin=obstacle_radius_margin,
        fixed_obstacle_shape_ids=fixed_obstacle_shape_ids,
        fixed_obstacle_widths=fixed_obstacle_widths,
        fixed_obstacle_depths=fixed_obstacle_depths,
        obstacle_yaws=physical_yaws,
    )

    # A new corridor was sampled; dynamic-play motion must restart from these
    # freshly placed obstacles rather than continuing old anchor trajectories.
    if hasattr(env, "_go2w_dynamic_obstacle_initialized"):
        env._go2w_dynamic_obstacle_initialized[env_ids] = False
    if hasattr(env, "_go2w_hospital_dynamic_initialized"):
        env._go2w_hospital_dynamic_initialized[env_ids] = False



def reset_navigation_goals_and_obstacles(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    obstacle_names: list[str],
    min_obstacles: int = 5,
    max_obstacles: int | None = None,
    empty_env_fraction: float = 0.0,
    spawn_range_x: tuple[float, float] = (-3.5, 3.5),
    spawn_range_y: tuple[float, float] = (-2.5, 2.5),
    obstacle_z: float = 0.25,
    min_inter_obstacle_dist: float = 0.7,
    goal_forward_range: tuple[float, float] = (2.5, 4.5),
    goal_lateral_range: tuple[float, float] = (-1.5, 1.5),
    goal_heading_jitter_range: tuple[float, float] = (-0.35, 0.35),
    min_goal_distance: float = 2.0,
    start_exclusion_radius: float = 1.0,
    goal_exclusion_radius: float = 0.9,
    head_on_progress_range: tuple[float, float] = (0.2, 0.85),
    head_on_lateral_range: tuple[float, float] = (-0.25, 0.25),
    edge_progress_range: tuple[float, float] = (0.25, 0.8),
    edge_lateral_range: tuple[float, float] = (0.55, 1.1),
    diagonal_progress_range: tuple[float, float] = (0.15, 0.7),
    diagonal_lateral_range: tuple[float, float] = (0.8, 1.6),
    offpath_progress_range: tuple[float, float] = (0.3, 0.9),
    offpath_lateral_range: tuple[float, float] = (1.3, 2.2),
    narrow_gap_progress_range: tuple[float, float] = (0.35, 0.75),
    narrow_gap_center_lateral_range: tuple[float, float] = (-0.15, 0.15),
    narrow_gap_half_width_range: tuple[float, float] = (0.38, 0.55),
    narrow_gap_probability: float = 0.35,
    # New Phase-1 scenario parameters
    partial_blockage_progress_range: tuple[float, float] = (0.2, 0.75),
    partial_blockage_lateral_range: tuple[float, float] = (0.5, 1.15),
    partial_blockage_probability: float = 0.20,
    narrow_gap_wide_half_width_range: tuple[float, float] = (0.60, 0.80),
    narrow_gap_barely_half_width_range: tuple[float, float] = (0.40, 0.52),
    cluttered_progress_range: tuple[float, float] = (0.15, 0.85),
    cluttered_lateral_range: tuple[float, float] = (-1.2, 1.2),
    # Curriculum phase schedule: maps start_iteration → available template tuple.
    # None means "all templates always available".
    phase_schedule: dict[str, tuple[str, ...]] | None = None,
    steps_per_iteration: int = 128,
    fixed_goal_forward: float | None = None,
    fixed_goal_lateral: float | None = None,
    fixed_goal_heading_jitter: float | None = None,
    fixed_scenario_template: str | None = None,
    park_distance: float = 1000.0,
    fixed_layout_seed: int | None = None,
    obstacle_radius_margin: float = 0.0,
    fixed_obstacle_shape_ids: tuple[int, ...] | None = None,
    fixed_obstacle_widths: tuple[float, ...] | None = None,
    fixed_obstacle_depths: tuple[float, ...] | None = None,
    randomize_physical_obstacle_slots: bool = False,
    physical_slot_randomization_start_iteration: int = 500,
    physical_slot_randomization_warmup_iterations: int = 500,
    passable_gap_min_width: float = 0.50,
    passable_gap_robot_width: float = 0.44,
    randomize_obstacle_yaw: bool = False,
    obstacle_yaw_range: tuple[float, float] = (-math.pi, math.pi),
) -> None:
    """Sample explicit start-goal local-navigation tasks and place varied obstacles.

    This event is purpose-built for the active local-navigation distillation stage.
    It samples:
      1. an episode start pose (already set by reset_base),
      2. a goal pose ahead/laterally offset from that start,
      3. obstacle layouts distributed along the start-goal corridor with varied
         encounter types (head-on, edge graze, diagonal, off-path, narrow gap).

    The sampled start/goal are stored on the env so observation, reward, and
    termination helpers can derive a local goal command every step.
    """
    ensure_navigation_goal_buffers(env)
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

    if max_obstacles is None:
        max_obstacles = len(obstacle_names)
    fixed_template = None if fixed_scenario_template is None else fixed_scenario_template.lower()
    if fixed_template == "random":
        fixed_template = None

    # When fixed_layout_seed is set, use a seeded RNG so every reset produces
    # the same obstacle/goal layout.  Training always passes None → default
    # random module behavior is preserved.
    import random as _rm
    _rng = _rm.Random(fixed_layout_seed) if fixed_layout_seed is not None else _rm
    random = _rng  # noqa: F841 — shadows module import; all uses below (incl. closures) pick up _rng

    n = len(env_ids)
    device = env.device
    env_origins = env.scene.env_origins[env_ids]

    robot = env.scene["robot"]
    start_pos_w = robot.data.root_pos_w[env_ids, :3].clone()
    start_heading_w = robot.data.heading_w[env_ids].clone()
    yaw = quat_yaw_wxyz(robot.data.root_quat_w[env_ids])

    env._go2w_start_pos_w[env_ids] = start_pos_w
    env._go2w_start_heading_w[env_ids] = start_heading_w

    robot_local_xy = (start_pos_w[:, :2] - env_origins[:, :2]).cpu().tolist()
    env_origin_xy = env_origins[:, :2].cpu().tolist()
    yaw_list = yaw.cpu().tolist()

    goal_pos_w = start_pos_w.clone()
    goal_heading_w = start_heading_w.clone()

    for idx in range(n):
        start_x, start_y = robot_local_xy[idx]
        yaw_i = yaw_list[idx]
        cos_yaw = math.cos(yaw_i)
        sin_yaw = math.sin(yaw_i)

        if fixed_goal_forward is not None or fixed_goal_lateral is not None:
            forward = fixed_goal_forward if fixed_goal_forward is not None else sum(goal_forward_range) * 0.5
            lateral = fixed_goal_lateral if fixed_goal_lateral is not None else sum(goal_lateral_range) * 0.5
        else:
            forward = 0.0
            lateral = 0.0
            for _ in range(50):
                forward = random.uniform(*goal_forward_range)
                lateral = random.uniform(*goal_lateral_range)
                goal_distance = math.hypot(forward, lateral)
                if goal_distance >= min_goal_distance:
                    break

        goal_dx_world = forward * cos_yaw - lateral * sin_yaw
        goal_dy_world = forward * sin_yaw + lateral * cos_yaw
        goal_pos_w[idx, 0] = start_pos_w[idx, 0] + goal_dx_world
        goal_pos_w[idx, 1] = start_pos_w[idx, 1] + goal_dy_world
        path_heading_w = math.atan2(goal_dy_world, goal_dx_world)
        heading_jitter = (
            fixed_goal_heading_jitter
            if fixed_goal_heading_jitter is not None
            else random.uniform(*goal_heading_jitter_range)
        )
        goal_heading_w[idx] = math.atan2(
            math.sin(path_heading_w + heading_jitter), math.cos(path_heading_w + heading_jitter)
        )

    env._go2w_goal_pos_w[env_ids] = goal_pos_w
    env._go2w_goal_heading_w[env_ids] = goal_heading_w

    if len(obstacle_names) == 0:
        return

    goal_local_xy = (goal_pos_w[:, :2] - env_origins[:, :2]).cpu().tolist()

    if fixed_layout_seed is not None:
        active_counts = torch.tensor(
            [_rng.randint(min_obstacles, max_obstacles) for _ in range(n)],
            dtype=torch.long,
            device=device,
        )
    else:
        active_counts = torch.randint(
            low=min_obstacles,
            high=max_obstacles + 1,
            size=(n,),
            device=device,
        )
    if empty_env_fraction > 0.0:
        if fixed_layout_seed is not None:
            _frac = max(0.0, min(1.0, empty_env_fraction))
            empty_mask = torch.tensor(
                [_rng.random() < _frac for _ in range(n)], dtype=torch.bool, device=device
            )
        else:
            empty_mask = torch.rand(n, device=device) < max(0.0, min(1.0, empty_env_fraction))
        active_counts = torch.where(empty_mask, torch.zeros_like(active_counts), active_counts)
    if fixed_template == "empty":
        active_counts = torch.zeros_like(active_counts)
    elif fixed_template in ("narrow_gap", "narrow_gap_wide", "narrow_gap_barely"):
        active_counts = torch.clamp(active_counts, min=min(2, len(obstacle_names)))
    elif fixed_template is not None:
        active_counts = torch.clamp(active_counts, min=1)
    env._go2w_scenario_template_id[env_ids] = 0

    parked_world = env_origins.clone()
    parked_world[:, 0] += park_distance
    parked_world[:, 2] = obstacle_z
    parked_positions = _separated_parked_positions(parked_world, len(obstacle_names))
    # Keep parking references immutable while layout slots are populated in-place.
    world_positions_per_slot = list(parked_positions.clone().unbind(dim=1))

    # Per-call passable-gap metadata buffers, filled in the narrow-gap branch below.
    # Defaults (zeros / not passable) cover every non-gap scenario.
    gap_center_w_buf = torch.zeros(n, 2, device=device)
    gap_dir_w_buf = torch.zeros(n, 2, device=device)
    gap_half_w_buf = torch.zeros(n, device=device)
    gap_free_half_w_buf = torch.zeros(n, device=device)
    gap_center_tolerance_buf = torch.zeros(n, device=device)
    gap_passable_buf = torch.zeros(n, dtype=torch.bool, device=device)

    template_choices = (
        "head_on",
        "left_edge",
        "right_edge",
        "diag_left",
        "diag_right",
        "off_left",
        "off_right",
        "cluttered",
    )
    template_codes = _NAV_SCENARIO_CODES
    _special_fixed = {
        "empty", "narrow_gap", "narrow_gap_wide", "narrow_gap_barely",
        "partial_blockage_left_open", "partial_blockage_right_open",
    }
    valid_fixed_templates = set(template_choices) | _special_fixed | {None}
    if fixed_template not in valid_fixed_templates:
        _all_valid = sorted(set(template_choices) | _special_fixed)
        raise ValueError(
            f"Unsupported fixed_scenario_template={fixed_scenario_template!r}. "
            f"Expected one of: random, {', '.join(_all_valid)}."
        )

    # Determine active template pool from phase_schedule
    _phase_active_templates: tuple[str, ...] | None = None
    if phase_schedule is not None:
        _current_step = env.common_step_counter
        for _phase_key in sorted(phase_schedule.keys(), key=int):
            if _current_step >= int(_phase_key) * steps_per_iteration:
                _phase_active_templates = phase_schedule[_phase_key]

    for env_idx in range(n):
        active_count = int(active_counts[env_idx].item())
        if active_count <= 0:
            continue

        start_x, start_y = robot_local_xy[env_idx]
        goal_x, goal_y = goal_local_xy[env_idx]
        origin_x, origin_y = env_origin_xy[env_idx]
        path_dx = goal_x - start_x
        path_dy = goal_y - start_y
        path_len = math.hypot(path_dx, path_dy)
        if path_len < 1.0e-6:
            path_dx = 1.0
            path_dy = 0.0
            path_len = 1.0
        path_dir_x = path_dx / path_len
        path_dir_y = path_dy / path_len
        normal_x = -path_dir_y
        normal_y = path_dir_x
        scenario_code = template_codes["empty"]

        placed_positions: list[tuple[float, float]] = []

        def _valid_position(local_x: float, local_y: float) -> bool:
            if local_x < spawn_range_x[0] or local_x > spawn_range_x[1]:
                return False
            if local_y < spawn_range_y[0] or local_y > spawn_range_y[1]:
                return False
            if math.hypot(local_x - start_x, local_y - start_y) < start_exclusion_radius:
                return False
            if math.hypot(local_x - goal_x, local_y - goal_y) < goal_exclusion_radius:
                return False
            for prev_x, prev_y in placed_positions:
                if math.hypot(local_x - prev_x, local_y - prev_y) < min_inter_obstacle_dist:
                    return False
            return True

        def _place_from_path(progress: float, lateral_offset: float) -> tuple[float, float]:
            center_x = start_x + progress * path_dx
            center_y = start_y + progress * path_dy
            return center_x + lateral_offset * normal_x, center_y + lateral_offset * normal_y

        def _sample_template_position(template: str) -> tuple[float, float]:
            if template == "head_on":
                progress = random.uniform(*head_on_progress_range)
                lateral_offset = random.uniform(*head_on_lateral_range)
            elif template == "left_edge":
                progress = random.uniform(*edge_progress_range)
                lateral_offset = random.uniform(*edge_lateral_range)
            elif template == "right_edge":
                progress = random.uniform(*edge_progress_range)
                lateral_offset = -random.uniform(*edge_lateral_range)
            elif template == "diag_left":
                progress = random.uniform(*diagonal_progress_range)
                lateral_offset = random.uniform(*diagonal_lateral_range)
            elif template == "diag_right":
                progress = random.uniform(*diagonal_progress_range)
                lateral_offset = -random.uniform(*diagonal_lateral_range)
            elif template == "off_left":
                progress = random.uniform(*offpath_progress_range)
                lateral_offset = random.uniform(*offpath_lateral_range)
            elif template == "off_right":
                progress = random.uniform(*offpath_progress_range)
                lateral_offset = -random.uniform(*offpath_lateral_range)
            elif template == "partial_blockage_left_open":
                progress = random.uniform(*partial_blockage_progress_range)
                lateral_offset = random.uniform(*partial_blockage_lateral_range)
            elif template == "partial_blockage_right_open":
                progress = random.uniform(*partial_blockage_progress_range)
                lateral_offset = -random.uniform(*partial_blockage_lateral_range)
            elif template == "cluttered":
                progress = random.uniform(*cluttered_progress_range)
                lateral_offset = random.uniform(cluttered_lateral_range[0], cluttered_lateral_range[1])
            else:
                progress = random.uniform(*head_on_progress_range)
                lateral_offset = random.uniform(*head_on_lateral_range)
            return _place_from_path(progress, lateral_offset)

        next_slot = 0
        _ng_set = {"narrow_gap", "narrow_gap_wide", "narrow_gap_barely"}
        _pb_set = {"partial_blockage_left_open", "partial_blockage_right_open"}
        _ng_phase_ok = _phase_active_templates is None or bool(_ng_set & set(_phase_active_templates))
        _pb_phase_ok = _phase_active_templates is None or bool(_pb_set & set(_phase_active_templates))
        use_narrow_gap = fixed_template in _ng_set or (
            fixed_template is None and active_count >= 2
            and _ng_phase_ok and random.random() < narrow_gap_probability
        )
        use_partial_blockage = (
            not use_narrow_gap
            and (
                fixed_template in _pb_set
                or (
                    fixed_template is None and active_count >= 2
                    and _pb_phase_ok and random.random() < partial_blockage_probability
                )
            )
        )

        if active_count >= 2 and use_narrow_gap:
            # Determine narrow gap width variant
            if fixed_template == "narrow_gap_wide":
                _gap_hwrange = narrow_gap_wide_half_width_range
                scenario_code = template_codes["narrow_gap_wide"]
            elif fixed_template == "narrow_gap_barely":
                _gap_hwrange = narrow_gap_barely_half_width_range
                scenario_code = template_codes["narrow_gap_barely"]
            else:
                if _phase_active_templates is None:
                    _ng_avail = ["narrow_gap", "narrow_gap_wide", "narrow_gap_barely"]
                else:
                    _ng_avail = [t for t in _phase_active_templates if t in _ng_set] or ["narrow_gap"]
                _ng_pick = random.choice(_ng_avail)
                if _ng_pick == "narrow_gap_wide":
                    _gap_hwrange = narrow_gap_wide_half_width_range
                    scenario_code = template_codes["narrow_gap_wide"]
                elif _ng_pick == "narrow_gap_barely":
                    _gap_hwrange = narrow_gap_barely_half_width_range
                    scenario_code = template_codes["narrow_gap_barely"]
                else:
                    _gap_hwrange = narrow_gap_half_width_range
                    scenario_code = template_codes["narrow_gap"]
            progress = random.uniform(*narrow_gap_progress_range)
            center_lateral = random.uniform(*narrow_gap_center_lateral_range)
            gap_half_width = random.uniform(*_gap_hwrange)
            pair_offsets = (center_lateral + gap_half_width, center_lateral - gap_half_width)
            # Store the gap centerline (world frame) and half width so reward helpers
            # can score traversal. The path direction equals the world direction
            # because env origins are pure translations of the world frame.
            _gap_cx_local, _gap_cy_local = _place_from_path(progress, center_lateral)
            gap_center_w_buf[env_idx, 0] = origin_x + _gap_cx_local
            gap_center_w_buf[env_idx, 1] = origin_y + _gap_cy_local
            gap_dir_w_buf[env_idx, 0] = path_dir_x
            gap_dir_w_buf[env_idx, 1] = path_dir_y
            gap_half_w_buf[env_idx] = gap_half_width
            for pair_offset in pair_offsets:
                placed = False
                for _ in range(30):
                    local_x, local_y = _place_from_path(progress, pair_offset)
                    if _valid_position(local_x, local_y):
                        placed_positions.append((local_x, local_y))
                        world_positions_per_slot[next_slot][env_idx, 0] = origin_x + local_x
                        world_positions_per_slot[next_slot][env_idx, 1] = origin_y + local_y
                        world_positions_per_slot[next_slot][env_idx, 2] = obstacle_z
                        placed = True
                        break
                if not placed:
                    for _ in range(40):
                        local_x = random.uniform(*spawn_range_x)
                        local_y = random.uniform(*spawn_range_y)
                        if _valid_position(local_x, local_y):
                            scenario_code = template_codes["random_fallback"]
                            placed_positions.append((local_x, local_y))
                            world_positions_per_slot[next_slot][env_idx, 0] = origin_x + local_x
                            world_positions_per_slot[next_slot][env_idx, 1] = origin_y + local_y
                            world_positions_per_slot[next_slot][env_idx, 2] = obstacle_z
                            placed = True
                            break
                if not placed:
                    scenario_code = template_codes["random_fallback"]
                next_slot += 1

        elif active_count >= 2 and use_partial_blockage:
            # Partial blockage: 2 obstacles clustered on one side, opposite side open
            if fixed_template == "partial_blockage_left_open":
                _pb_sign = 1.0
                scenario_code = template_codes["partial_blockage_left_open"]
            elif fixed_template == "partial_blockage_right_open":
                _pb_sign = -1.0
                scenario_code = template_codes["partial_blockage_right_open"]
            else:
                _pb_sign = random.choice([-1.0, 1.0])
                scenario_code = template_codes[
                    "partial_blockage_left_open" if _pb_sign > 0 else "partial_blockage_right_open"
                ]
            progress = random.uniform(*partial_blockage_progress_range)
            lat_base = random.uniform(*partial_blockage_lateral_range)
            pb_offsets = (_pb_sign * lat_base, _pb_sign * (lat_base + 0.5))
            for pb_offset in pb_offsets:
                placed = False
                for _ in range(30):
                    _pb_prog = progress + random.uniform(-0.05, 0.05)
                    local_x, local_y = _place_from_path(_pb_prog, pb_offset)
                    if _valid_position(local_x, local_y):
                        placed_positions.append((local_x, local_y))
                        world_positions_per_slot[next_slot][env_idx, 0] = origin_x + local_x
                        world_positions_per_slot[next_slot][env_idx, 1] = origin_y + local_y
                        world_positions_per_slot[next_slot][env_idx, 2] = obstacle_z
                        placed = True
                        break
                if not placed:
                    for _ in range(40):
                        local_x = random.uniform(*spawn_range_x)
                        local_y = random.uniform(*spawn_range_y)
                        if _valid_position(local_x, local_y):
                            scenario_code = template_codes["random_fallback"]
                            placed_positions.append((local_x, local_y))
                            world_positions_per_slot[next_slot][env_idx, 0] = origin_x + local_x
                            world_positions_per_slot[next_slot][env_idx, 1] = origin_y + local_y
                            world_positions_per_slot[next_slot][env_idx, 2] = obstacle_z
                            placed = True
                            break
                next_slot += 1

        # Place first obstacle for non-special fixed templates (excluding cluttered, which
        # fills all slots uniformly rather than placing a single "anchor" obstacle).
        if fixed_template in template_choices and fixed_template != "cluttered" and next_slot < active_count:
            scenario_code = template_codes[fixed_template]
            placed = False
            for _ in range(40):
                local_x, local_y = _sample_template_position(fixed_template)
                if _valid_position(local_x, local_y):
                    placed_positions.append((local_x, local_y))
                    world_positions_per_slot[next_slot][env_idx, 0] = origin_x + local_x
                    world_positions_per_slot[next_slot][env_idx, 1] = origin_y + local_y
                    world_positions_per_slot[next_slot][env_idx, 2] = obstacle_z
                    placed = True
                    break
            if not placed:
                for _ in range(40):
                    local_x = random.uniform(*spawn_range_x)
                    local_y = random.uniform(*spawn_range_y)
                    if _valid_position(local_x, local_y):
                        scenario_code = template_codes["random_fallback"]
                        placed_positions.append((local_x, local_y))
                        world_positions_per_slot[next_slot][env_idx, 0] = origin_x + local_x
                        world_positions_per_slot[next_slot][env_idx, 1] = origin_y + local_y
                        world_positions_per_slot[next_slot][env_idx, 2] = obstacle_z
                        placed = True
                        break
            next_slot += 1

        # Build the fill-loop template pool, respecting fixed_template and phase.
        if fixed_template == "cluttered":
            scenario_code = template_codes["cluttered"]
            _fill_choices: tuple[str, ...] = ("cluttered",)
        elif _phase_active_templates is not None:
            _filtered = tuple(t for t in template_choices if t in _phase_active_templates)
            _fill_choices = _filtered if _filtered else template_choices
        else:
            _fill_choices = template_choices

        while next_slot < active_count:
            placed = False
            for _ in range(40):
                template = random.choice(_fill_choices)
                local_x, local_y = _sample_template_position(template)
                if _valid_position(local_x, local_y):
                    if scenario_code == template_codes["empty"]:
                        scenario_code = template_codes[template]
                    placed_positions.append((local_x, local_y))
                    world_positions_per_slot[next_slot][env_idx, 0] = origin_x + local_x
                    world_positions_per_slot[next_slot][env_idx, 1] = origin_y + local_y
                    world_positions_per_slot[next_slot][env_idx, 2] = obstacle_z
                    placed = True
                    break
            if not placed:
                for _ in range(40):
                    local_x = random.uniform(*spawn_range_x)
                    local_y = random.uniform(*spawn_range_y)
                    if _valid_position(local_x, local_y):
                        if scenario_code == template_codes["empty"]:
                            scenario_code = template_codes["random_fallback"]
                        placed_positions.append((local_x, local_y))
                        world_positions_per_slot[next_slot][env_idx, 0] = origin_x + local_x
                        world_positions_per_slot[next_slot][env_idx, 1] = origin_y + local_y
                        world_positions_per_slot[next_slot][env_idx, 2] = obstacle_z
                        placed = True
                        break
            next_slot += 1

        env._go2w_scenario_template_id[env_ids[env_idx]] = scenario_code
        # Only flag passable when both gap obstacles were actually placed at the
        # gap (scenario_code stays a gap type; it is downgraded to random_fallback
        # if placement fell back to uniform spawn).
        if scenario_code in (
            template_codes["narrow_gap"],
            template_codes["narrow_gap_wide"],
            template_codes["narrow_gap_barely"],
        ):
            gap_passable_buf[env_idx] = True

    # Preserve the sampled episode-start template before successful goals are
    # replaced with random fallback segments later in the same episode.
    env._go2w_initial_scenario_template_id[env_ids] = env._go2w_scenario_template_id[env_ids]

    logical_active_mask = torch.stack(
        [
            (slot_positions[:, :2] - parked_positions[:, slot_idx, :2]).norm(dim=1) > 1.0
            for slot_idx, slot_positions in enumerate(world_positions_per_slot)
        ],
        dim=1,
    )
    logical_positions = torch.stack(world_positions_per_slot, dim=1)
    randomize_slot_mask = (
        False
        if fixed_layout_seed is not None
        else _physical_slot_randomization_mask(
            env,
            n,
            randomize_physical_obstacle_slots,
            physical_slot_randomization_start_iteration,
            physical_slot_randomization_warmup_iterations,
            steps_per_iteration,
            device,
        )
    )
    physical_positions, active_mask, logical_to_physical = _assign_logical_positions_to_physical_slots(
        logical_positions, logical_active_mask, parked_positions, randomize_slot_mask
    )
    logical_yaws = torch.zeros(n, len(obstacle_names), device=device)
    if randomize_obstacle_yaw:
        if fixed_layout_seed is not None:
            sampled_yaws = torch.empty_like(logical_yaws)
            for env_idx in range(n):
                for slot_idx in range(len(obstacle_names)):
                    sampled_yaws[env_idx, slot_idx] = _rng.uniform(*obstacle_yaw_range)
        else:
            sampled_yaws = torch.empty(n, len(obstacle_names), device=device).uniform_(*obstacle_yaw_range)
        logical_yaws = torch.where(logical_active_mask, sampled_yaws, logical_yaws)
    physical_yaws = torch.zeros_like(logical_yaws)
    physical_yaws.scatter_(1, logical_to_physical, logical_yaws)
    world_positions_per_slot = list(physical_positions.unbind(dim=1))
    set_obstacle_metadata(
        env,
        env_ids,
        obstacle_names,
        active_mask,
        obstacle_radius_margin=obstacle_radius_margin,
        fixed_obstacle_shape_ids=fixed_obstacle_shape_ids,
        fixed_obstacle_widths=fixed_obstacle_widths,
        fixed_obstacle_depths=fixed_obstacle_depths,
        obstacle_yaws=physical_yaws,
    )
    if len(obstacle_names) >= 2:
        reference = torch.zeros(env.num_envs, len(obstacle_names), device=device)
        footprint_radii = obstacle_effective_radius(env, obstacle_names, reference)[env_ids]
        gap_slots = logical_to_physical[:, :2]
        gap_radii = footprint_radii.gather(1, gap_slots)
        gap_free_width = (2.0 * gap_half_w_buf - gap_radii[:, 0] - gap_radii[:, 1]).clamp(min=0.0)
        gap_free_half_w_buf = gap_free_width * 0.5
        gap_center_tolerance_buf = (gap_free_width - passable_gap_robot_width).clamp(min=0.0) * 0.5
        gap_passable_buf &= gap_free_width >= passable_gap_min_width

    env._go2w_gap_center_w[env_ids] = gap_center_w_buf
    env._go2w_gap_dir_w[env_ids] = gap_dir_w_buf
    env._go2w_gap_half_width[env_ids] = gap_half_w_buf
    env._go2w_gap_free_half_width[env_ids] = gap_free_half_w_buf
    env._go2w_gap_center_tolerance[env_ids] = gap_center_tolerance_buf
    env._go2w_gap_passable[env_ids] = gap_passable_buf

    for slot_idx, name in enumerate(obstacle_names):
        obstacle = env.scene[name]
        pose = torch.zeros(n, 7, device=device)
        pose[:, :3] = world_positions_per_slot[slot_idx]
        pose[:, 3:7] = yaw_to_quat_wxyz(physical_yaws[:, slot_idx])
        obstacle.write_root_pose_to_sim(pose, env_ids=env_ids)
        obstacle.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)

    if hasattr(env, "_go2w_dynamic_obstacle_initialized"):
        env._go2w_dynamic_obstacle_initialized[env_ids] = False

    # Store a callable so goal_reached_and_resample can trigger obstacle+goal
    # resample mid-episode.  Uses robot-world-centered placement so obstacles
    # appear correctly even after the robot has moved far from the env origin.
    _half_x = goal_forward_range[1] * 0.6 + 1.0
    _half_y = max(abs(goal_lateral_range[0]), abs(goal_lateral_range[1])) + 1.0
    _num_obs = min(
        max_obstacles if max_obstacles is not None else len(obstacle_names),
        len(obstacle_names),
    )
    env._nav_resample_on_goal = functools.partial(
        _resample_nav_on_goal_reached,
        env,
        obstacle_names=obstacle_names,
        num_obstacles=_num_obs,
        spawn_half_x=_half_x,
        spawn_half_y=_half_y,
        min_inter_obstacle_dist=min_inter_obstacle_dist,
        goal_forward_range=goal_forward_range,
        goal_lateral_range=goal_lateral_range,
        goal_heading_jitter_range=goal_heading_jitter_range,
        min_goal_distance=min_goal_distance,
        start_exclusion_radius=start_exclusion_radius,
        goal_exclusion_radius=goal_exclusion_radius,
        obstacle_z=obstacle_z,
        park_distance=park_distance,
        obstacle_radius_margin=obstacle_radius_margin,
        fixed_obstacle_shape_ids=fixed_obstacle_shape_ids,
        fixed_obstacle_widths=fixed_obstacle_widths,
        fixed_obstacle_depths=fixed_obstacle_depths,
        randomize_physical_obstacle_slots=randomize_physical_obstacle_slots,
        physical_slot_randomization_start_iteration=physical_slot_randomization_start_iteration,
        physical_slot_randomization_warmup_iterations=physical_slot_randomization_warmup_iterations,
        steps_per_iteration=steps_per_iteration,
        randomize_obstacle_yaw=randomize_obstacle_yaw,
        obstacle_yaw_range=obstacle_yaw_range,
    )
