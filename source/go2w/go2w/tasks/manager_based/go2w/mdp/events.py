# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event functions for the Go2-W obstacle avoidance environment."""

from __future__ import annotations

import functools
import math
import random
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _curriculum_progress(
    env: ManagerBasedRLEnv,
    start_iteration: int,
    warmup_iterations: int,
    steps_per_iteration: int,
) -> float:
    """Return curriculum progress t in [0, 1] based on training step counter."""
    start_steps = start_iteration * steps_per_iteration
    warmup_steps = warmup_iterations * steps_per_iteration
    step = env.common_step_counter
    if step < start_steps:
        return 0.0
    if warmup_steps <= 0:
        return 1.0
    return max(0.0, min(1.0, (step - start_steps) / warmup_steps))


def update_locomotion_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    start_iteration: int,
    warmup_iterations: int,
    steps_per_iteration: int,
    command_name: str,
    lin_vel_x_initial: tuple[float, float],
    lin_vel_x_final: tuple[float, float],
    lin_vel_y_initial: tuple[float, float],
    lin_vel_y_final: tuple[float, float],
    ang_vel_z_initial: tuple[float, float],
    ang_vel_z_final: tuple[float, float],
    min_survival_steps: int = 0,
) -> None:
    """Update velocity command ranges for speed curriculum with performance gate.

    Linearly interpolates command ranges from initial to final values over the warmup
    period. If min_survival_steps > 0, the range is scaled by the ratio of mean
    completed episode length to min_survival_steps, even after warmup. This lets
    the command range contract again when the policy starts falling.
    """
    t = _curriculum_progress(env, start_iteration, warmup_iterations, steps_per_iteration)

    if min_survival_steps > 0 and 0.0 < t and len(env_ids) > 0:
        mean_len = env.episode_length_buf[env_ids].float().mean().item()
        survival_factor = min(1.0, mean_len / min_survival_steps)
        t = t * survival_factor

    def lerp(a: float, b: float) -> float:
        return a + t * (b - a)

    cmd = env.command_manager.get_term(command_name)
    cmd.cfg.ranges.lin_vel_x = (lerp(lin_vel_x_initial[0], lin_vel_x_final[0]),  # type: ignore[attr-defined]
                                  lerp(lin_vel_x_initial[1], lin_vel_x_final[1]))
    cmd.cfg.ranges.lin_vel_y = (lerp(lin_vel_y_initial[0], lin_vel_y_final[0]),  # type: ignore[attr-defined]
                                  lerp(lin_vel_y_initial[1], lin_vel_y_final[1]))
    cmd.cfg.ranges.ang_vel_z = (lerp(ang_vel_z_initial[0], ang_vel_z_final[0]),  # type: ignore[attr-defined]
                                  lerp(ang_vel_z_initial[1], ang_vel_z_final[1]))


def _quat_yaw_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Return yaw angle from a wxyz quaternion tensor."""
    w, x, y, z = quat.unbind(dim=-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _ensure_navigation_goal_buffers(env: ManagerBasedRLEnv) -> None:
    """Create persistent start/goal buffers used by the navigation-distill task."""
    if not hasattr(env, "_go2w_goal_pos_w"):
        env._go2w_goal_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._go2w_goal_heading_w = torch.zeros(env.num_envs, device=env.device)
        env._go2w_start_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._go2w_start_heading_w = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_go2w_scenario_template_id"):
        env._go2w_scenario_template_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    # Passable narrow-gap metadata: gap centerline in world frame, half width, and
    # a passable flag set only for the narrow_gap / narrow_gap_wide / narrow_gap_barely
    # scenarios. Reward helpers use these to encourage decisive gap traversal.
    if not hasattr(env, "_go2w_gap_center_w"):
        env._go2w_gap_center_w = torch.zeros(env.num_envs, 2, device=env.device)
        env._go2w_gap_dir_w = torch.zeros(env.num_envs, 2, device=env.device)
        env._go2w_gap_half_width = torch.zeros(env.num_envs, device=env.device)
        env._go2w_gap_passable = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # Per-env stuck counter for cluttered/blocked recovery diagnostics and gating.
    if not hasattr(env, "_go2w_stuck_counter"):
        env._go2w_stuck_counter = torch.zeros(env.num_envs, device=env.device)


def reset_obstacles_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    obstacle_names: list[str],
    start_iteration: int,
    warmup_iterations: int,
    steps_per_iteration: int,
    min_obstacles: int = 1,
    max_obstacles: int | None = None,
    spawn_range_x: tuple[float, float] = (-3.5, 3.5),
    spawn_range_y: tuple[float, float] = (-2.5, 2.5),
    obstacle_z: float = 0.25,
    min_spawn_distance_from_robot: float = 1.2,
    min_spawn_distance_from_robot_initial: float | None = None,
    min_inter_obstacle_dist: float = 0.8,
    min_survival_steps: int = 0,
    empty_env_fraction: float = 0.0,
    command_path_obstacles: int = 0,
    command_name: str = "base_velocity",
    command_path_reference_xy: tuple[float, float] | None = None,
    command_path_forward_range: tuple[float, float] = (1.2, 2.0),
    command_path_lateral_range: tuple[float, float] = (-0.35, 0.35),
    command_path_min_speed: float = 0.2,
    near_field_obstacles: int = 0,
    near_field_radius_range: tuple[float, float] = (1.3, 1.9),
) -> None:
    """Reset obstacles with curriculum-controlled active count.

    Active obstacles are placed randomly within spawn_range, excluding a radius
    around the robot spawn position and around already-placed obstacles.
    Inactive obstacles are parked 1000 m away so obstacle_positions_rel
    returns zero for them (beyond the 8 m mask).

    Args:
        obstacle_names: All obstacle scene entity names.
        start_iteration: Iteration at which obstacle curriculum begins.
        warmup_iterations: Iterations to ramp from min_obstacles to max_obstacles.
        steps_per_iteration: Must match num_steps_per_env in the runner config.
        min_obstacles: Active obstacle count at curriculum start.
        max_obstacles: Active obstacle count at curriculum end (default: all).
        spawn_range_x: Local x spawn range [m] relative to env origin.
        spawn_range_y: Local y spawn range [m] relative to env origin.
        obstacle_z: Fixed z height of obstacle center [m].
        min_spawn_distance_from_robot: Final exclusion radius around robot spawn [m].
        min_spawn_distance_from_robot_initial: Optional initial exclusion radius
            ramped to min_spawn_distance_from_robot over obstacle warmup.
        min_inter_obstacle_dist: Minimum distance between any two obstacles [m].
        min_survival_steps: Gate obstacle ramp on episode survival length (0 = disabled).
        empty_env_fraction: Fraction of reset environments that keep all obstacles
            parked even after the obstacle curriculum starts. This preserves
            obstacle-free locomotion samples during obstacle fine-tuning.
        command_path_obstacles: Number of active obstacle slots sampled in front
            of the robot along the current commanded velocity direction. These
            slots are assigned before near_field_obstacles.
        command_name: Velocity command term used for command-path obstacle spawn.
        command_path_reference_xy: Optional explicit command xy override used for
            command-path spawn. This is mainly for play/eval with fixed commands,
            because reset events run before command_manager.reset(), so the live
            command buffer may still hold zeros during obstacle placement.
        command_path_forward_range: Forward distance range [m] for command-path
            obstacle slots in the commanded direction.
        command_path_lateral_range: Lateral offset range [m] from the commanded
            path centerline.
        command_path_min_speed: Below this command speed, command-path slots fall
            back to uniform random spawn so standing commands are not forced to
            solve an avoid-obstacle task.
        near_field_obstacles: Number of active obstacle slots sampled in an
            annulus around the robot after command_path_obstacles instead of
            uniformly over the whole scene. This increases encounter frequency
            while keeping obstacle count low.
        near_field_radius_range: Local annulus radius range [m] for near-field
            obstacle slots.
    """
    if max_obstacles is None:
        max_obstacles = len(obstacle_names)

    t = _curriculum_progress(env, start_iteration, warmup_iterations, steps_per_iteration)

    # Performance gate: slow obstacle ramp if robot is not surviving well enough.
    if min_survival_steps > 0 and 0.0 < t and len(env_ids) > 0:
        mean_len = env.episode_length_buf[env_ids].float().mean().item()
        survival_factor = min(1.0, mean_len / min_survival_steps)
        t = t * survival_factor

    start_steps = start_iteration * steps_per_iteration
    if env.common_step_counter < start_steps:
        num_active = 0
    else:
        num_active = round(min_obstacles + t * (max_obstacles - min_obstacles))
    num_active = max(0, min(num_active, len(obstacle_names)))

    if min_spawn_distance_from_robot_initial is not None:
        min_spawn_distance = (
            min_spawn_distance_from_robot_initial
            + t * (min_spawn_distance_from_robot - min_spawn_distance_from_robot_initial)
        )
    else:
        min_spawn_distance = min_spawn_distance_from_robot

    n = len(env_ids)
    device = env.device
    env_origins = env.scene.env_origins[env_ids]  # world-frame env origins (n, 3)

    active_counts = torch.full((n,), num_active, dtype=torch.long, device=device)
    if empty_env_fraction > 0.0 and num_active > 0 and n > 0:
        fraction = max(0.0, min(1.0, empty_env_fraction))
        empty_mask = torch.rand(n, device=device) < fraction
        active_counts = torch.where(empty_mask, torch.zeros_like(active_counts), active_counts)

    # Robot current/reset position in local env frame for exclusion zone and
    # command-path obstacle spawn. reset_base is defined before reset_obstacles,
    # so this normally reflects the just-sampled root pose.
    robot = env.scene["robot"]
    robot_world_pos = robot.data.root_pos_w[env_ids, :2]
    robot_local_pos = robot_world_pos - env_origins[:, :2]

    command_path_count = max(0, min(command_path_obstacles, len(obstacle_names)))
    if command_path_count > 0:
        if command_path_reference_xy is not None:
            command_xy = torch.tensor(command_path_reference_xy, device=device, dtype=robot_world_pos.dtype).repeat(n, 1)
        else:
            command_xy = env.command_manager.get_command(command_name)[env_ids, :2]
        command_speed = command_xy.norm(dim=1)
        command_dir_body = command_xy / command_speed.clamp(min=1.0e-6).unsqueeze(1)
        yaw = _quat_yaw_wxyz(robot.data.root_quat_w[env_ids])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        body_x_world = torch.stack([cos_yaw, sin_yaw], dim=-1)
        body_y_world = torch.stack([-sin_yaw, cos_yaw], dim=-1)
        command_dir_local = (
            command_dir_body[:, 0:1] * body_x_world
            + command_dir_body[:, 1:2] * body_y_world
        )
        command_dir_local = command_dir_local / command_dir_local.norm(dim=1, keepdim=True).clamp(min=1.0e-6)
        command_path_ok = command_speed >= command_path_min_speed
    else:
        command_dir_local = torch.zeros(n, 2, device=device)
        command_path_ok = torch.zeros(n, dtype=torch.bool, device=device)

    # Track placed local positions for inter-obstacle distance checks: list of (n, 2)
    placed_local_positions: list[torch.Tensor] = []

    for idx, name in enumerate(obstacle_names):
        obstacle = env.scene[name]
        active_mask = idx < active_counts
        use_command_path = idx < command_path_count
        use_near_field = command_path_count <= idx < command_path_count + near_field_obstacles

        if active_mask.any():
            # Sample valid positions: retry up to 20 times per env
            local_x = torch.zeros(n, device=device)
            local_y = torch.zeros(n, device=device)
            placed = ~active_mask

            for _ in range(20):
                if use_command_path:
                    forward = torch.empty(n, device=device).uniform_(*command_path_forward_range)
                    lateral = torch.empty(n, device=device).uniform_(*command_path_lateral_range)
                    normal = torch.stack([-command_dir_local[:, 1], command_dir_local[:, 0]], dim=-1)
                    path_xy = robot_local_pos + forward.unsqueeze(1) * command_dir_local + lateral.unsqueeze(1) * normal

                    rand_x = torch.empty(n, device=device).uniform_(*spawn_range_x)
                    rand_y = torch.empty(n, device=device).uniform_(*spawn_range_y)
                    cx = torch.where(command_path_ok, path_xy[:, 0], rand_x)
                    cy = torch.where(command_path_ok, path_xy[:, 1], rand_y)
                elif use_near_field:
                    radius = torch.empty(n, device=device).uniform_(*near_field_radius_range)
                    angle = torch.empty(n, device=device).uniform_(-3.141592653589793, 3.141592653589793)
                    cx = robot_local_pos[:, 0] + radius * torch.cos(angle)
                    cy = robot_local_pos[:, 1] + radius * torch.sin(angle)
                else:
                    cx = torch.empty(n, device=device).uniform_(*spawn_range_x)
                    cy = torch.empty(n, device=device).uniform_(*spawn_range_y)

                # Check distance from robot spawn
                dist_robot = (
                    (cx - robot_local_pos[:, 0]).pow(2)
                    + (cy - robot_local_pos[:, 1]).pow(2)
                ).sqrt()
                robot_ok = dist_robot >= min_spawn_distance
                range_ok = (
                    (cx >= spawn_range_x[0])
                    & (cx <= spawn_range_x[1])
                    & (cy >= spawn_range_y[0])
                    & (cy <= spawn_range_y[1])
                )

                # Check distance from already-placed obstacles
                if min_inter_obstacle_dist > 0 and len(placed_local_positions) > 0:
                    prev = torch.stack(placed_local_positions, dim=1)       # (n, k, 2)
                    cxy = torch.stack([cx, cy], dim=-1).unsqueeze(1)        # (n, 1, 2)
                    dist_others = (cxy - prev).norm(dim=-1).min(dim=1).values  # (n,)
                    others_ok = dist_others >= min_inter_obstacle_dist
                else:
                    others_ok = torch.ones(n, dtype=torch.bool, device=device)

                valid = active_mask & (~placed) & robot_ok & range_ok & others_ok
                local_x = torch.where(valid, cx, local_x)
                local_y = torch.where(valid, cy, local_y)
                placed = placed | valid
                if placed.all():
                    break

            # Dense play scenes can exhaust the inter-obstacle spacing budget.
            # For any remaining envs, keep the robot exclusion zone and relax
            # only the obstacle-obstacle spacing so boxes never fall back to
            # the origin under the robot.
            if not placed.all():
                for _ in range(20):
                    if use_command_path:
                        forward = torch.empty(n, device=device).uniform_(*command_path_forward_range)
                        lateral = torch.empty(n, device=device).uniform_(*command_path_lateral_range)
                        normal = torch.stack([-command_dir_local[:, 1], command_dir_local[:, 0]], dim=-1)
                        path_xy = robot_local_pos + forward.unsqueeze(1) * command_dir_local + lateral.unsqueeze(1) * normal

                        rand_x = torch.empty(n, device=device).uniform_(*spawn_range_x)
                        rand_y = torch.empty(n, device=device).uniform_(*spawn_range_y)
                        cx = torch.where(command_path_ok, path_xy[:, 0], rand_x)
                        cy = torch.where(command_path_ok, path_xy[:, 1], rand_y)
                    elif use_near_field:
                        radius = torch.empty(n, device=device).uniform_(*near_field_radius_range)
                        angle = torch.empty(n, device=device).uniform_(-3.141592653589793, 3.141592653589793)
                        cx = robot_local_pos[:, 0] + radius * torch.cos(angle)
                        cy = robot_local_pos[:, 1] + radius * torch.sin(angle)
                    else:
                        cx = torch.empty(n, device=device).uniform_(*spawn_range_x)
                        cy = torch.empty(n, device=device).uniform_(*spawn_range_y)
                    dist_robot = (
                        (cx - robot_local_pos[:, 0]).pow(2)
                        + (cy - robot_local_pos[:, 1]).pow(2)
                    ).sqrt()
                    range_ok = (
                        (cx >= spawn_range_x[0])
                        & (cx <= spawn_range_x[1])
                        & (cy >= spawn_range_y[0])
                        & (cy <= spawn_range_y[1])
                    )
                    valid = active_mask & (~placed) & (dist_robot >= min_spawn_distance) & range_ok
                    local_x = torch.where(valid, cx, local_x)
                    local_y = torch.where(valid, cy, local_y)
                    placed = placed | valid
                    if placed.all():
                        break

            if not placed.all():
                # Last-resort deterministic placement on the spawn boundary.
                fallback_x = torch.full((n,), spawn_range_x[1], device=device)
                fallback_y = torch.empty(n, device=device).uniform_(*spawn_range_y)
                fallback_mask = active_mask & (~placed)
                local_x = torch.where(fallback_mask, fallback_x, local_x)
                local_y = torch.where(fallback_mask, fallback_y, local_y)
                placed = torch.ones_like(placed)

            placed_xy = torch.stack([local_x, local_y], dim=-1)
            parked_xy = torch.full_like(placed_xy, 1000.0)
            placed_local_positions.append(torch.where(active_mask.unsqueeze(1), placed_xy, parked_xy))

            active_world_pos = env_origins.clone()
            active_world_pos[:, 0] += local_x
            active_world_pos[:, 1] += local_y
            active_world_pos[:, 2] = obstacle_z

            parked_world_pos = env_origins.clone()
            parked_world_pos[:, 0] += 1000.0
            parked_world_pos[:, 2] = obstacle_z
            world_pos = torch.where(active_mask.unsqueeze(1), active_world_pos, parked_world_pos)
        else:
            # Park inactive obstacle far away (beyond 8 m obs mask → reads as zero)
            world_pos = env_origins.clone()
            world_pos[:, 0] += 1000.0
            world_pos[:, 2] = obstacle_z

        # Pose: [x, y, z, qw, qx, qy, qz] with identity rotation
        pose = torch.zeros(n, 7, device=device)
        pose[:, :3] = world_pos
        pose[:, 3] = 1.0  # quaternion w component

        obstacle.write_root_pose_to_sim(pose, env_ids=env_ids)
        obstacle.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)


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
    _ensure_navigation_goal_buffers(env)
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

    # Write all obstacle poses to sim
    pose_buf = torch.zeros(n, 7, device=device)
    pose_buf[:, 3] = 1.0  # quaternion w
    zero_vel = torch.zeros(n, 6, device=device)
    for slot_idx, name in enumerate(obstacle_names):
        obstacle = env.scene[name]
        if slot_idx < effective:
            pose_buf[:, 0] = placed[:, slot_idx, 0]
            pose_buf[:, 1] = placed[:, slot_idx, 1]
            pose_buf[:, 2] = obstacle_z
        else:
            # remaining slots: park at park_distance
            pose_buf[:, 0] = park_xy[:, 0]
            pose_buf[:, 1] = park_xy[:, 1]
            pose_buf[:, 2] = obstacle_z
        obstacle.write_root_pose_to_sim(pose_buf.clone(), env_ids=env_ids)
        obstacle.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

    # A new corridor was sampled; dynamic-play motion must restart from these
    # freshly placed obstacles rather than continuing old anchor trajectories.
    if hasattr(env, "_go2w_dynamic_obstacle_initialized"):
        env._go2w_dynamic_obstacle_initialized[env_ids] = False


def move_dynamic_play_obstacles(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    obstacle_names: list[str],
    obstacle_z: float,
    longitudinal_speed_range: tuple[float, float] = (0.25, 0.70),
    lateral_speed_max: float = 0.12,
    longitudinal_extent: float = 2.0,
    lateral_extent: float = 0.30,
    min_inter_obstacle_dist: float = 0.7,
    active_distance: float = 100.0,
) -> None:
    """Move active play obstacles like pedestrians along the current corridor.

    This function is intentionally wired only by ``play.py`` when the dynamic
    play flag is enabled. Obstacles move predominantly along the start-to-goal
    corridor with a small lateral component. Their motion reflects at a bounded
    excursion from each sampled pose and rejects any proposed step that would
    violate the obstacle-to-obstacle separation constraint.
    """
    if len(obstacle_names) == 0:
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if len(env_ids) == 0:
        return

    _ensure_navigation_goal_buffers(env)
    n_envs = env.num_envs
    n_slots = len(obstacle_names)
    device = env.device

    if (
        not hasattr(env, "_go2w_dynamic_obstacle_initialized")
        or env._go2w_dynamic_anchor_xy.shape != (n_envs, n_slots, 2)
    ):
        env._go2w_dynamic_obstacle_initialized = torch.zeros(n_envs, dtype=torch.bool, device=device)
        env._go2w_dynamic_anchor_xy = torch.zeros(n_envs, n_slots, 2, device=device)
        env._go2w_dynamic_dir_xy = torch.zeros(n_envs, n_slots, 2, device=device)
        env._go2w_dynamic_long_speed = torch.zeros(n_envs, n_slots, device=device)
        env._go2w_dynamic_lat_speed = torch.zeros(n_envs, n_slots, device=device)

    positions_xy = torch.stack(
        [env.scene[name].data.root_pos_w[env_ids, :2] for name in obstacle_names], dim=1
    )
    robot_xy = env.scene["robot"].data.root_pos_w[env_ids, :2]
    active = (positions_xy - robot_xy.unsqueeze(1)).norm(dim=-1) < active_distance

    corridor = env._go2w_goal_pos_w[env_ids, :2] - env._go2w_start_pos_w[env_ids, :2]
    corridor_norm = corridor.norm(dim=-1, keepdim=True)
    fallback_yaw = env.scene["robot"].data.heading_w[env_ids]
    fallback_dir = torch.stack((fallback_yaw.cos(), fallback_yaw.sin()), dim=-1)
    corridor_dir = torch.where(
        corridor_norm > 1.0e-6,
        corridor / corridor_norm.clamp(min=1.0e-6),
        fallback_dir,
    )

    needs_init = ~env._go2w_dynamic_obstacle_initialized[env_ids]
    if needs_init.any():
        init_ids = env_ids[needs_init]
        init_active = active[needs_init]
        init_dir = corridor_dir[needs_init].unsqueeze(1).expand(-1, n_slots, -1)
        signs = torch.randint(0, 2, init_active.shape, device=device, dtype=torch.int64).float() * 2.0 - 1.0
        speed_lo, speed_hi = longitudinal_speed_range
        long_speed = torch.empty(init_active.shape, device=device).uniform_(speed_lo, speed_hi) * signs
        lat_speed = torch.empty(init_active.shape, device=device).uniform_(-lateral_speed_max, lateral_speed_max)
        long_speed = torch.where(init_active, long_speed, torch.zeros_like(long_speed))
        lat_speed = torch.where(init_active, lat_speed, torch.zeros_like(lat_speed))

        env._go2w_dynamic_anchor_xy[init_ids] = positions_xy[needs_init]
        env._go2w_dynamic_dir_xy[init_ids] = init_dir
        env._go2w_dynamic_long_speed[init_ids] = long_speed
        env._go2w_dynamic_lat_speed[init_ids] = lat_speed
        env._go2w_dynamic_obstacle_initialized[init_ids] = True
        # Static-gap shaping is no longer meaningful once the gap obstacles move.
        # This affects play reward logs only; the pretrained policy observation is unchanged.
        env._go2w_gap_passable[init_ids] = False

    anchor = env._go2w_dynamic_anchor_xy[env_ids]
    path_dir = env._go2w_dynamic_dir_xy[env_ids]
    normal = torch.stack((-path_dir[..., 1], path_dir[..., 0]), dim=-1)
    long_speed = env._go2w_dynamic_long_speed[env_ids]
    lat_speed = env._go2w_dynamic_lat_speed[env_ids]

    offset = positions_xy - anchor
    long_offset = (offset * path_dir).sum(dim=-1)
    lat_offset = (offset * normal).sum(dim=-1)
    dt = env.step_dt

    next_long = long_offset + long_speed * dt
    reflect_long = next_long.abs() > longitudinal_extent
    long_speed = torch.where(reflect_long, -long_speed, long_speed)
    next_long = (long_offset + long_speed * dt).clamp(-longitudinal_extent, longitudinal_extent)

    next_lat = lat_offset + lat_speed * dt
    reflect_lat = next_lat.abs() > lateral_extent
    lat_speed = torch.where(reflect_lat, -lat_speed, lat_speed)
    next_lat = (lat_offset + lat_speed * dt).clamp(-lateral_extent, lateral_extent)

    proposed = anchor + next_long.unsqueeze(-1) * path_dir + next_lat.unsqueeze(-1) * normal
    proposed = torch.where(active.unsqueeze(-1), proposed, positions_xy)

    # Resolve motion slot-by-slot against current or already accepted positions.
    # Rejected obstacles stay in place and reverse direction for the next step.
    accepted = positions_xy.clone()
    for slot_idx in range(n_slots):
        candidate = proposed[:, slot_idx]
        distances = (candidate.unsqueeze(1) - accepted).norm(dim=-1)
        others = torch.arange(n_slots, device=device) != slot_idx
        conflict = (
            (distances < min_inter_obstacle_dist)
            & active
            & others.unsqueeze(0)
        ).any(dim=1)
        conflict &= active[:, slot_idx]
        accepted[:, slot_idx] = torch.where(conflict.unsqueeze(-1), positions_xy[:, slot_idx], candidate)
        long_speed[:, slot_idx] = torch.where(conflict, -long_speed[:, slot_idx], long_speed[:, slot_idx])
        lat_speed[:, slot_idx] = torch.where(conflict, -lat_speed[:, slot_idx], lat_speed[:, slot_idx])

    env._go2w_dynamic_long_speed[env_ids] = long_speed
    env._go2w_dynamic_lat_speed[env_ids] = lat_speed

    zero_vel = torch.zeros(len(env_ids), 6, device=device)
    for slot_idx, name in enumerate(obstacle_names):
        pose = torch.zeros(len(env_ids), 7, device=device)
        pose[:, :2] = accepted[:, slot_idx]
        pose[:, 2] = obstacle_z
        pose[:, 3] = 1.0
        env.scene[name].write_root_pose_to_sim(pose, env_ids=env_ids)
        env.scene[name].write_root_velocity_to_sim(zero_vel, env_ids=env_ids)


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
    narrow_gap_barely_half_width_range: tuple[float, float] = (0.35, 0.48),
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
    if len(obstacle_names) == 0:
        return

    _ensure_navigation_goal_buffers(env)
    if hasattr(env, "_go2w_goals_reached_episode"):
        env._go2w_goals_reached_episode[env_ids] = 0.0

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
    yaw = _quat_yaw_wxyz(robot.data.root_quat_w[env_ids])

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
    elif fixed_template == "narrow_gap":
        active_counts = torch.clamp(active_counts, min=min(2, len(obstacle_names)))
    elif fixed_template is not None:
        active_counts = torch.clamp(active_counts, min=1)
    env._go2w_scenario_template_id[env_ids] = 0

    parked_world = env_origins.clone()
    parked_world[:, 0] += park_distance
    parked_world[:, 2] = obstacle_z
    world_positions_per_slot = [parked_world.clone() for _ in obstacle_names]

    # Per-call passable-gap metadata buffers, filled in the narrow-gap branch below.
    # Defaults (zeros / not passable) cover every non-gap scenario.
    gap_center_w_buf = torch.zeros(n, 2, device=device)
    gap_dir_w_buf = torch.zeros(n, 2, device=device)
    gap_half_w_buf = torch.zeros(n, device=device)
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
    template_codes = {
        "empty": 0,
        "head_on": 1,
        "left_edge": 2,
        "right_edge": 3,
        "diag_left": 4,
        "diag_right": 5,
        "off_left": 6,
        "off_right": 7,
        "narrow_gap": 8,
        "random_fallback": 9,
        "partial_blockage_left_open": 10,
        "partial_blockage_right_open": 11,
        "cluttered": 12,
        "narrow_gap_wide": 13,
        "narrow_gap_barely": 14,
    }
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

    env._go2w_gap_center_w[env_ids] = gap_center_w_buf
    env._go2w_gap_dir_w[env_ids] = gap_dir_w_buf
    env._go2w_gap_half_width[env_ids] = gap_half_w_buf
    env._go2w_gap_passable[env_ids] = gap_passable_buf

    for slot_idx, name in enumerate(obstacle_names):
        obstacle = env.scene[name]
        pose = torch.zeros(n, 7, device=device)
        pose[:, :3] = world_positions_per_slot[slot_idx]
        pose[:, 3] = 1.0
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
    )
