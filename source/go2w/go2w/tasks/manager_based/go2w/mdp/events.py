# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event functions for the Go2-W obstacle avoidance environment."""

from __future__ import annotations

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
