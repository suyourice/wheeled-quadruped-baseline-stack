# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event functions for the Go2-W obstacle avoidance environment."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from ...common.curriculum import _curriculum_progress
from ...common.orientation import (
    quat_yaw_wxyz,
    yaw_pitch_roll_to_quat_wxyz,
    yaw_pitch_to_quat_wxyz,
    yaw_to_quat_wxyz,
)
from ..local_planning.obstacle_geometry import set_obstacle_metadata
from ..global_planning.corridors import (
    nearest_polyline_tangent_local,
    project_polyline_corridor_local,
)
from ..scenarios import (
    NAV_RANDOM_FALLBACK_SCENARIO_ID as _NAV_RANDOM_FALLBACK_SCENARIO_ID,
    NAV_SCENARIO_CODES as _NAV_SCENARIO_CODES,
    NAV_SCENARIO_NAMES as _NAV_SCENARIO_NAMES,
)
from ..goals import ensure_navigation_goal_buffers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
    obstacle_radius_margin: float = 0.0,
    fixed_obstacle_shape_ids: tuple[int, ...] | None = None,
    fixed_obstacle_widths: tuple[float, ...] | None = None,
    fixed_obstacle_depths: tuple[float, ...] | None = None,
    randomize_obstacle_yaw: bool = False,
    obstacle_yaw_range: tuple[float, float] = (-math.pi, math.pi),
) -> None:
    """Reset obstacles with curriculum-controlled active count.

    Active obstacles are placed randomly within spawn_range, excluding a radius
    around the robot spawn position and around already-placed obstacles.
    Inactive obstacles are parked 1000 m away so obstacle observation and
    reward terms mask them out by distance.

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
        yaw = quat_yaw_wxyz(robot.data.root_quat_w[env_ids])
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
    obstacle_yaws = torch.zeros(n, len(obstacle_names), device=device)

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
                    angle = torch.empty(n, device=device).uniform_(-math.pi, math.pi)
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
            # only the obstacle-obstacle spacing so obstacles never fall back to
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
                        angle = torch.empty(n, device=device).uniform_(-math.pi, math.pi)
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
            parked_world_pos[:, 1] += float(idx)
            parked_world_pos[:, 2] = obstacle_z
            world_pos = torch.where(active_mask.unsqueeze(1), active_world_pos, parked_world_pos)
        else:
            # Park inactive obstacle far away (beyond 8 m obs mask → reads as zero)
            world_pos = env_origins.clone()
            world_pos[:, 0] += 1000.0
            world_pos[:, 1] += float(idx)
            world_pos[:, 2] = obstacle_z

        yaw = torch.zeros(n, device=device)
        if randomize_obstacle_yaw:
            sampled_yaw = torch.empty(n, device=device).uniform_(*obstacle_yaw_range)
            yaw = torch.where(active_mask, sampled_yaw, yaw)
        obstacle_yaws[:, idx] = yaw

        # Pose: [x, y, z, qw, qx, qy, qz]
        pose = torch.zeros(n, 7, device=device)
        pose[:, :3] = world_pos
        pose[:, 3:7] = yaw_to_quat_wxyz(yaw)

        obstacle.write_root_pose_to_sim(pose, env_ids=env_ids)
        obstacle.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)

    slot_ids = torch.arange(len(obstacle_names), device=device).unsqueeze(0)
    set_obstacle_metadata(
        env,
        env_ids,
        obstacle_names,
        slot_ids < active_counts.unsqueeze(1),
        obstacle_radius_margin=obstacle_radius_margin,
        fixed_obstacle_shape_ids=fixed_obstacle_shape_ids,
        fixed_obstacle_widths=fixed_obstacle_widths,
        fixed_obstacle_depths=fixed_obstacle_depths,
        obstacle_yaws=obstacle_yaws,
    )


def move_dynamic_play_obstacles(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    obstacle_names: list[str],
    obstacle_z: float,
    obstacle_indices: list[int] | None = None,
    longitudinal_speed_range: tuple[float, float] = (0.25, 0.70),
    lateral_speed_max: float = 0.12,
    longitudinal_extent: float = 2.0,
    lateral_extent: float = 0.30,
    min_inter_obstacle_dist: float = 0.7,
    active_distance: float = 100.0,
    velocity_resample_interval_range: tuple[float, float] | None = None,
    random_trajectory_fraction: float = 0.0,
    goal_exclusion_radius: float = 0.9,
    robot_keepout_radius: float = 1.25,
    start_iteration: int = 0,
    warmup_iterations: int = 0,
    steps_per_iteration: int = 128,
) -> None:
    """Move active play obstacles like pedestrians along the current corridor.

    This function is intentionally wired only by ``play.py`` when the dynamic
    play flag is enabled. Obstacles move predominantly along the start-to-goal
    corridor with a small lateral component. In mixed-motion mode, each obstacle
    periodically changes speed and a sampled subset wanders in arbitrary planar
    directions. Motion reflects at a bounded excursion from each sampled pose
    and rejects any proposed step that would violate separation constraints.
    """
    if len(obstacle_names) == 0:
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if len(env_ids) == 0:
        return
    current_iteration = env.common_step_counter / max(steps_per_iteration, 1)
    if current_iteration < start_iteration:
        return
    if warmup_iterations > 0:
        speed_scale = min(1.0, max(0.0, (current_iteration - start_iteration) / warmup_iterations))
    else:
        speed_scale = 1.0
    longitudinal_speed_range = (
        longitudinal_speed_range[0] * speed_scale,
        longitudinal_speed_range[1] * speed_scale,
    )
    lateral_speed_max *= speed_scale

    ensure_navigation_goal_buffers(env)
    n_envs = env.num_envs
    n_slots = len(obstacle_names)
    device = env.device
    metadata_indices = None
    if obstacle_indices is not None:
        if len(obstacle_indices) != n_slots:
            raise ValueError("obstacle_indices must have one entry per dynamic obstacle name.")
        metadata_indices = torch.tensor(obstacle_indices, dtype=torch.long, device=device)

    if (
        not hasattr(env, "_go2w_dynamic_obstacle_initialized")
        or env._go2w_dynamic_anchor_xy.shape != (n_envs, n_slots, 2)
    ):
        env._go2w_dynamic_obstacle_initialized = torch.zeros(n_envs, dtype=torch.bool, device=device)
        env._go2w_dynamic_anchor_xy = torch.zeros(n_envs, n_slots, 2, device=device)
        env._go2w_dynamic_dir_xy = torch.zeros(n_envs, n_slots, 2, device=device)
        env._go2w_dynamic_long_speed = torch.zeros(n_envs, n_slots, device=device)
        env._go2w_dynamic_lat_speed = torch.zeros(n_envs, n_slots, device=device)
        env._go2w_dynamic_wander = torch.zeros(n_envs, n_slots, dtype=torch.bool, device=device)
        env._go2w_dynamic_velocity_timer = torch.zeros(n_envs, n_slots, device=device)

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
        if (
            hasattr(env, "_go2w_structured_corridor_centerline_local")
            and hasattr(env, "_go2w_structured_corridor_start_xy")
            and hasattr(env, "_go2w_structured_corridor_yaw")
        ):
            start_xy = env._go2w_structured_corridor_start_xy[init_ids]
            corridor_yaw = env._go2w_structured_corridor_yaw[init_ids]
            cos_yaw = torch.cos(corridor_yaw).unsqueeze(1)
            sin_yaw = torch.sin(corridor_yaw).unsqueeze(1)
            local_positions = positions_xy[needs_init] - start_xy.unsqueeze(1)
            local_positions = torch.stack(
                (
                    local_positions[..., 0] * cos_yaw + local_positions[..., 1] * sin_yaw,
                    -local_positions[..., 0] * sin_yaw + local_positions[..., 1] * cos_yaw,
                ),
                dim=-1,
            )
            local_tangent = nearest_polyline_tangent_local(
                local_positions,
                env._go2w_structured_corridor_centerline_local[init_ids],
            )
            init_dir = torch.stack(
                (
                    local_tangent[..., 0] * cos_yaw - local_tangent[..., 1] * sin_yaw,
                    local_tangent[..., 0] * sin_yaw + local_tangent[..., 1] * cos_yaw,
                ),
                dim=-1,
            )
        signs = torch.randint(0, 2, init_active.shape, device=device, dtype=torch.int64).float() * 2.0 - 1.0
        speed_lo, speed_hi = longitudinal_speed_range
        speed = torch.empty(init_active.shape, device=device).uniform_(speed_lo, speed_hi)
        long_speed = speed * signs
        lat_speed = torch.empty(init_active.shape, device=device).uniform_(-lateral_speed_max, lateral_speed_max)
        wander = torch.zeros_like(init_active)
        velocity_timer = torch.zeros(init_active.shape, device=device)
        if velocity_resample_interval_range is not None:
            wander = (torch.rand(init_active.shape, device=device) < random_trajectory_fraction) & init_active
            if random_trajectory_fraction > 0.0:
                needs_wanderer = init_active.any(dim=1) & ~wander.any(dim=1)
                first_active = init_active.long().argmax(dim=1)
                row_ids = torch.arange(len(init_ids), device=device)
                wander[row_ids, first_active] = wander[row_ids, first_active] | needs_wanderer
            wander_heading = torch.empty(init_active.shape, device=device).uniform_(-math.pi, math.pi)
            long_speed = torch.where(wander, speed * wander_heading.cos(), long_speed)
            lat_speed = torch.where(wander, speed * wander_heading.sin(), lat_speed)
            velocity_timer.uniform_(*velocity_resample_interval_range)
            velocity_timer = torch.where(init_active, velocity_timer, torch.zeros_like(velocity_timer))
        long_speed = torch.where(init_active, long_speed, torch.zeros_like(long_speed))
        lat_speed = torch.where(init_active, lat_speed, torch.zeros_like(lat_speed))

        env._go2w_dynamic_anchor_xy[init_ids] = positions_xy[needs_init]
        env._go2w_dynamic_dir_xy[init_ids] = init_dir
        env._go2w_dynamic_long_speed[init_ids] = long_speed
        env._go2w_dynamic_lat_speed[init_ids] = lat_speed
        env._go2w_dynamic_wander[init_ids] = wander
        env._go2w_dynamic_velocity_timer[init_ids] = velocity_timer
        env._go2w_dynamic_obstacle_initialized[init_ids] = True
        # Static-gap shaping is no longer meaningful once the gap obstacles move.
        # This affects play reward logs only; the pretrained policy observation is unchanged.
        env._go2w_gap_passable[init_ids] = False

    anchor = env._go2w_dynamic_anchor_xy[env_ids]
    path_dir = env._go2w_dynamic_dir_xy[env_ids]
    normal = torch.stack((-path_dir[..., 1], path_dir[..., 0]), dim=-1)
    long_speed = env._go2w_dynamic_long_speed[env_ids]
    lat_speed = env._go2w_dynamic_lat_speed[env_ids]
    wander = env._go2w_dynamic_wander[env_ids]

    offset = positions_xy - anchor
    long_offset = (offset * path_dir).sum(dim=-1)
    lat_offset = (offset * normal).sum(dim=-1)
    dt = env.step_dt

    if velocity_resample_interval_range is not None:
        velocity_timer = env._go2w_dynamic_velocity_timer[env_ids] - dt
        change_velocity = active & (velocity_timer <= 0.0)
        if change_velocity.any():
            speed_lo, speed_hi = longitudinal_speed_range
            new_speed = torch.empty(active.shape, device=device).uniform_(speed_lo, speed_hi)
            direction = torch.where(long_speed < 0.0, -torch.ones_like(long_speed), torch.ones_like(long_speed))
            new_long_speed = new_speed * direction
            new_lat_speed = torch.empty(active.shape, device=device).uniform_(
                -lateral_speed_max, lateral_speed_max
            )
            wander_heading = torch.empty(active.shape, device=device).uniform_(-math.pi, math.pi)
            new_long_speed = torch.where(wander, new_speed * wander_heading.cos(), new_long_speed)
            new_lat_speed = torch.where(wander, new_speed * wander_heading.sin(), new_lat_speed)
            long_speed = torch.where(change_velocity, new_long_speed, long_speed)
            lat_speed = torch.where(change_velocity, new_lat_speed, lat_speed)
            next_timer = torch.empty(active.shape, device=device).uniform_(*velocity_resample_interval_range)
            velocity_timer = torch.where(change_velocity, next_timer, velocity_timer)
        env._go2w_dynamic_velocity_timer[env_ids] = torch.where(
            active, velocity_timer, torch.zeros_like(velocity_timer)
        )

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

    if (
        hasattr(env, "_go2w_structured_corridor_start_xy")
        and hasattr(env, "_go2w_structured_corridor_yaw")
        and hasattr(env, "_go2w_structured_corridor_width")
        and hasattr(env, "_go2w_structured_corridor_centerline_local")
    ):
        start_xy = env._go2w_structured_corridor_start_xy[env_ids]
        corridor_yaw = env._go2w_structured_corridor_yaw[env_ids]
        cos_yaw = torch.cos(corridor_yaw).unsqueeze(1)
        sin_yaw = torch.sin(corridor_yaw).unsqueeze(1)
        rel = proposed - start_xy.unsqueeze(1)
        local = torch.stack(
            (
                rel[..., 0] * cos_yaw + rel[..., 1] * sin_yaw,
                -rel[..., 0] * sin_yaw + rel[..., 1] * cos_yaw,
            ),
            dim=-1,
        )
        projected_local = project_polyline_corridor_local(
            local,
            env._go2w_structured_corridor_centerline_local[env_ids],
            env._go2w_structured_corridor_width[env_ids],
        )
        projected_world = torch.stack(
            (
                start_xy[:, 0:1] + projected_local[..., 0] * cos_yaw - projected_local[..., 1] * sin_yaw,
                start_xy[:, 1:2] + projected_local[..., 0] * sin_yaw + projected_local[..., 1] * cos_yaw,
            ),
            dim=-1,
        )
        proposed = torch.where(active.unsqueeze(-1), projected_world, proposed)

    if hasattr(env, "_go2w_navigation_path_w") and hasattr(env, "_go2w_navigation_path_count"):
        path_count = env._go2w_navigation_path_count[env_ids].clamp(min=1)
        final_idx = path_count - 1
        goal_xy = env._go2w_navigation_path_w[env_ids, final_idx, :2]
    else:
        goal_xy = env._go2w_goal_pos_w[env_ids, :2]
    goal_keepout_radius = torch.full(
        (len(env_ids), n_slots),
        max(0.0, goal_exclusion_radius),
        device=device,
        dtype=positions_xy.dtype,
    )
    if (
        hasattr(env, "_go2w_obstacle_effective_radius")
        and metadata_indices is not None
        and env._go2w_obstacle_effective_radius.shape[0] == n_envs
        and env._go2w_obstacle_effective_radius.shape[1] > int(metadata_indices.max().item())
    ):
        margin = float(getattr(env, "_go2w_obstacle_radius_margin", 0.0))
        goal_keepout_radius = (
            goal_keepout_radius
            + env._go2w_obstacle_effective_radius[env_ids][:, metadata_indices]
            + margin
        )
    elif (
        hasattr(env, "_go2w_obstacle_effective_radius")
        and env._go2w_obstacle_effective_radius.shape == (n_envs, n_slots)
    ):
        margin = float(getattr(env, "_go2w_obstacle_radius_margin", 0.0))
        goal_keepout_radius = goal_keepout_radius + env._go2w_obstacle_effective_radius[env_ids] + margin

    goal_dist = (proposed - goal_xy.unsqueeze(1)).norm(dim=-1)
    goal_keepout = active & (goal_dist < goal_keepout_radius)
    robot_keepout = active & ((proposed - robot_xy.unsqueeze(1)).norm(dim=-1) < max(0.0, robot_keepout_radius))

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
        conflict |= goal_keepout[:, slot_idx] | robot_keepout[:, slot_idx]
        accepted[:, slot_idx] = torch.where(conflict.unsqueeze(-1), positions_xy[:, slot_idx], candidate)
        reverse_motion = conflict
        long_speed[:, slot_idx] = torch.where(reverse_motion, -long_speed[:, slot_idx], long_speed[:, slot_idx])
        lat_speed[:, slot_idx] = torch.where(reverse_motion, -lat_speed[:, slot_idx], lat_speed[:, slot_idx])

    env._go2w_dynamic_long_speed[env_ids] = long_speed
    env._go2w_dynamic_lat_speed[env_ids] = lat_speed

    zero_vel = torch.zeros(len(env_ids), 6, device=device)
    if (
        hasattr(env, "_go2w_obstacle_yaw")
        and metadata_indices is not None
        and env._go2w_obstacle_yaw.shape[0] == n_envs
        and env._go2w_obstacle_yaw.shape[1] > int(metadata_indices.max().item())
    ):
        obstacle_yaws = env._go2w_obstacle_yaw[env_ids][:, metadata_indices]
    elif hasattr(env, "_go2w_obstacle_yaw") and env._go2w_obstacle_yaw.shape == (n_envs, n_slots):
        obstacle_yaws = env._go2w_obstacle_yaw[env_ids]
    else:
        obstacle_yaws = torch.zeros(len(env_ids), n_slots, device=device)
    for slot_idx, name in enumerate(obstacle_names):
        pose = torch.zeros(len(env_ids), 7, device=device)
        pose[:, :2] = accepted[:, slot_idx]
        pose[:, 2] = obstacle_z
        pose[:, 3:7] = yaw_to_quat_wxyz(obstacle_yaws[:, slot_idx])
        env.scene[name].write_root_pose_to_sim(pose, env_ids=env_ids)
        env.scene[name].write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
