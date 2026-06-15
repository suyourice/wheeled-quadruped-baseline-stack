# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hospital-specific event functions: label-aware dynamics, group movement, logging."""

from __future__ import annotations

import math
import torch

from isaaclab.envs import ManagerBasedRLEnv

from ..events import yaw_to_quat_wxyz as _yaw_to_quat_wxyz
from ..obstacle_geometry import obstacle_effective_radius
from ..structured_corridor import nearest_polyline_tangent_local, project_polyline_corridor_local
from .relations import HOSPITAL_RELATION_SPECS
from .specs import HOSPITAL_LABEL_SPECS

_MOTION_PROFILE_PARAMS: dict[str, tuple[float, float, float]] = {
    "burst_runner":   (0.45, 2.2, 0.85),
    "queue_wait":     (0.02, 0.35, 0.08),
    "door_crossing":  (0.35, 1.2, 0.90),
    "leashed_pet":    (0.30, 1.8, 0.65),
    "wander":         (0.30, 1.8, 0.65),
    "pushed_payload": (0.10, 1.2, 0.30),
    "careful_roll":   (0.10, 1.2, 0.30),
    "slow_walk":      (0.14, 1.4, 0.35),
    "careful_walk":   (0.14, 1.4, 0.35),
    "cleaning_pass":  (0.12, 1.6, 0.35),
}

_WANDER_FRACTION: dict[str, float] = {
    "burst_runner":  0.80,
    "queue_wait":    0.0,
    "door_crossing": 0.55,
    "leashed_pet":   0.45,
    "wander":        0.45,
    "careful_walk":  0.12,
    "slow_walk":     0.12,
    "cleaning_pass": 0.20,
}


def _motion_profile_params(label: str) -> tuple[float, float, float]:
    """Return (lateral_speed_max, longitudinal_extent, lateral_extent)."""
    spec = HOSPITAL_LABEL_SPECS.get(label)
    if spec is None or spec.motion_profile == "static":
        return 0.0, 0.0, 0.0
    return _MOTION_PROFILE_PARAMS.get(spec.motion_profile, (0.20, 1.6, 0.45))


def _wander_fraction(label: str) -> float:
    """Return per-label fraction of dynamic actors that wander off the tangent."""
    spec = HOSPITAL_LABEL_SPECS.get(label)
    if spec is None:
        return 0.0
    return _WANDER_FRACTION.get(spec.motion_profile, 0.08)


def _local_offset_to_world(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    local_offset_xy: tuple[float, float],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Rotate a corridor-local XY offset into world coordinates."""
    device = env.device
    ox, oy = local_offset_xy
    offset = torch.tensor((ox, oy), device=device, dtype=dtype).unsqueeze(0).expand(len(env_ids), -1)
    if hasattr(env, "_go2w_structured_corridor_yaw"):
        yaw = env._go2w_structured_corridor_yaw[env_ids]
    else:
        yaw = env.scene["robot"].data.heading_w[env_ids]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    return torch.stack(
        (
            offset[:, 0] * cos_yaw - offset[:, 1] * sin_yaw,
            offset[:, 0] * sin_yaw + offset[:, 1] * cos_yaw,
        ),
        dim=-1,
    )


# ---------------------------------------------------------------------------
# Label-aware dynamic obstacle velocity sampling
# ---------------------------------------------------------------------------

def resample_hospital_obstacle_velocities(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    obstacle_names: list[str],
    obstacle_labels: list[str],
    obstacle_z: float,  # noqa: ARG001 — kept for caller API compatibility
    speed_scale: float = 1.0,
) -> None:
    """Resample each obstacle's velocity according to its label's motion profile.

    Replaces the generic uniform velocity sampling with per-label speed and
    yaw-rate ranges drawn from ``HOSPITAL_LABEL_SPECS``.

    Args:
        env: The RL environment instance.
        env_ids: Environments to update (passed by the event manager).
        obstacle_names: Ordered list of obstacle asset names.
        obstacle_labels: Label string for each obstacle slot (same order).
        obstacle_z: Vertical spawn height (unused here; kept for API compat).
        speed_scale: Multiplier applied to both speed and yaw-rate ranges (default 1.0).
    """
    n = len(env_ids)
    device = env.device
    all_env_ids = env_ids
    zero_vel = torch.zeros(n, 6, device=device)

    for name, label in zip(obstacle_names, obstacle_labels):
        spec = HOSPITAL_LABEL_SPECS.get(label)
        if spec is None or spec.motion_profile == "static":
            continue

        speed_lo, speed_hi = spec.speed_range
        yaw_lo, yaw_hi = spec.yaw_rate_range

        heading = torch.empty(n, device=device).uniform_(0.0, 2.0 * math.pi)
        speed = torch.empty(n, device=device).uniform_(speed_lo * speed_scale, speed_hi * speed_scale)
        wz = torch.empty(n, device=device).uniform_(yaw_lo * speed_scale, yaw_hi * speed_scale)

        vel = zero_vel.clone()
        vel[:, 0] = speed * torch.cos(heading)   # vx
        vel[:, 1] = speed * torch.sin(heading)   # vy
        vel[:, 5] = wz                           # yaw rate

        env.scene[name].write_root_velocity_to_sim(vel, env_ids=all_env_ids)


# ---------------------------------------------------------------------------
# Label-aware constrained dynamic obstacle motion
# ---------------------------------------------------------------------------

def move_hospital_dynamic_obstacles(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    obstacle_names: list[str],
    obstacle_labels: list[str],
    obstacle_indices: list[int] | None = None,
    obstacle_center_zs: tuple[float, ...] | None = None,
    group_registry: list[dict] | None = None,
    queue_groups: list[dict] | None = None,
    seated_groups: list[dict] | None = None,
    speed_scale: float = 1.0,
    active_distance: float = 100.0,
    min_inter_obstacle_dist: float = 0.25,
    velocity_resample_interval_range: tuple[float, float] = (1.2, 3.0),
    goal_exclusion_radius: float = 0.9,
    robot_keepout_radius: float = 1.25,
) -> None:
    """Move hospital actors with label-specific motion inside the corridor.

    The function is evaluation-safe for an existing policy: labels determine
    obstacle motion and coupling, but the policy observation is unchanged.
    """
    if len(obstacle_names) == 0 or len(env_ids) == 0:
        return
    if len(obstacle_labels) != len(obstacle_names):
        raise ValueError("Hospital obstacle labels must align with obstacle names.")
    if obstacle_center_zs is not None and len(obstacle_center_zs) != len(obstacle_names):
        raise ValueError("Hospital obstacle center z values must align with obstacle names.")

    n_envs = env.num_envs
    n_slots = len(obstacle_names)
    device = env.device
    dtype = env.scene[obstacle_names[0]].data.root_pos_w.dtype
    metadata_indices = None
    if obstacle_indices is not None:
        if len(obstacle_indices) != n_slots:
            raise ValueError("obstacle_indices must align with obstacle_names.")
        metadata_indices = torch.tensor(obstacle_indices, dtype=torch.long, device=device)

    speed_lo = torch.zeros(n_slots, device=device, dtype=dtype)
    speed_hi = torch.zeros(n_slots, device=device, dtype=dtype)
    lat_speed_max = torch.zeros(n_slots, device=device, dtype=dtype)
    long_extent = torch.zeros(n_slots, device=device, dtype=dtype)
    lat_extent = torch.zeros(n_slots, device=device, dtype=dtype)
    wander_prob = torch.zeros(n_slots, device=device, dtype=dtype)
    dynamic_slot = torch.zeros(n_slots, device=device, dtype=torch.bool)
    writable_slot = [False] * n_slots
    for slot_idx, label in enumerate(obstacle_labels):
        spec = HOSPITAL_LABEL_SPECS.get(label)
        if spec is None or spec.motion_profile == "static":
            continue
        writable_slot[slot_idx] = True
        lo, hi = spec.speed_range
        lateral, longitudinal, lateral_extent = _motion_profile_params(label)
        speed_lo[slot_idx] = lo * speed_scale
        speed_hi[slot_idx] = hi * speed_scale
        lat_speed_max[slot_idx] = lateral * speed_scale
        long_extent[slot_idx] = longitudinal
        lat_extent[slot_idx] = lateral_extent
        wander_prob[slot_idx] = _wander_fraction(label)
        dynamic_slot[slot_idx] = True

    positions_xy = torch.stack(
        [env.scene[name].data.root_pos_w[env_ids, :2] for name in obstacle_names], dim=1
    )
    robot_xy = env.scene["robot"].data.root_pos_w[env_ids, :2]
    distance_active = (positions_xy - robot_xy.unsqueeze(1)).norm(dim=-1) < active_distance
    active = distance_active
    if (
        metadata_indices is not None
        and hasattr(env, "_go2w_obstacle_active_mask")
        and env._go2w_obstacle_active_mask.shape[0] == n_envs
        and env._go2w_obstacle_active_mask.shape[1] > int(metadata_indices.max().item())
    ):
        active = env._go2w_obstacle_active_mask[env_ids][:, metadata_indices] & distance_active
    present = active
    active = present & dynamic_slot.unsqueeze(0)
    name_to_slot = {name: idx for idx, name in enumerate(obstacle_names)}
    if (
        queue_groups
        and hasattr(env, "_go2w_hospital_queue_served")
        and env._go2w_hospital_queue_served.shape[0] == n_envs
        and env._go2w_hospital_queue_served.shape[2] == n_slots
    ):
        served_mask = torch.zeros_like(active)
        max_group_count = min(len(queue_groups), env._go2w_hospital_queue_served.shape[1])
        for group_idx in range(max_group_count):
            slots = [
                name_to_slot[name]
                for name in queue_groups[group_idx].get("names", ())
                if name in name_to_slot
            ]
            if slots:
                served_mask[:, slots] |= env._go2w_hospital_queue_served[env_ids, group_idx][:, slots]
        active = active & ~served_mask

    if (
        not hasattr(env, "_go2w_hospital_dynamic_initialized")
        or env._go2w_hospital_dynamic_anchor_xy.shape != (n_envs, n_slots, 2)
    ):
        env._go2w_hospital_dynamic_initialized = torch.zeros(n_envs, dtype=torch.bool, device=device)
        env._go2w_hospital_dynamic_anchor_xy = torch.zeros(n_envs, n_slots, 2, device=device)
        env._go2w_hospital_dynamic_dir_xy = torch.zeros(n_envs, n_slots, 2, device=device)
        env._go2w_hospital_dynamic_long_speed = torch.zeros(n_envs, n_slots, device=device)
        env._go2w_hospital_dynamic_lat_speed = torch.zeros(n_envs, n_slots, device=device)
        env._go2w_hospital_dynamic_wander = torch.zeros(n_envs, n_slots, dtype=torch.bool, device=device)
        env._go2w_hospital_dynamic_velocity_timer = torch.zeros(n_envs, n_slots, device=device)

    corridor_dir = torch.zeros(len(env_ids), 2, device=device, dtype=dtype)
    if hasattr(env, "_go2w_goal_pos_w") and hasattr(env, "_go2w_start_pos_w"):
        corridor = env._go2w_goal_pos_w[env_ids, :2] - env._go2w_start_pos_w[env_ids, :2]
        corridor_norm = corridor.norm(dim=-1, keepdim=True)
        fallback_yaw = env.scene["robot"].data.heading_w[env_ids]
        fallback = torch.stack((fallback_yaw.cos(), fallback_yaw.sin()), dim=-1)
        corridor_dir = torch.where(corridor_norm > 1.0e-6, corridor / corridor_norm.clamp(min=1.0e-6), fallback)
    else:
        fallback_yaw = env.scene["robot"].data.heading_w[env_ids]
        corridor_dir = torch.stack((fallback_yaw.cos(), fallback_yaw.sin()), dim=-1)

    needs_init = ~env._go2w_hospital_dynamic_initialized[env_ids]
    if needs_init.any():
        init_ids = env_ids[needs_init]
        init_active = active[needs_init]
        init_dir = corridor_dir[needs_init].unsqueeze(1).expand(-1, n_slots, -1).clone()
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

        rand = torch.rand(init_active.shape, device=device, dtype=dtype)
        speed = rand * (speed_hi - speed_lo).unsqueeze(0) + speed_lo.unsqueeze(0)
        signs = torch.randint(0, 2, init_active.shape, device=device, dtype=torch.int64).to(dtype) * 2.0 - 1.0
        long_speed = speed * signs
        lat_rand = torch.rand(init_active.shape, device=device, dtype=dtype) * 2.0 - 1.0
        lateral_speed = lat_rand * lat_speed_max.unsqueeze(0)
        wander = (torch.rand(init_active.shape, device=device, dtype=dtype) < wander_prob.unsqueeze(0)) & init_active
        wander_heading = torch.empty(init_active.shape, device=device, dtype=dtype).uniform_(-math.pi, math.pi)
        long_speed = torch.where(wander, speed * wander_heading.cos(), long_speed)
        lateral_speed = torch.where(wander, speed * wander_heading.sin(), lateral_speed)
        long_speed = torch.where(init_active, long_speed, torch.zeros_like(long_speed))
        lateral_speed = torch.where(init_active, lateral_speed, torch.zeros_like(lateral_speed))
        timer = torch.empty(init_active.shape, device=device, dtype=dtype).uniform_(*velocity_resample_interval_range)
        timer = torch.where(init_active, timer, torch.zeros_like(timer))

        env._go2w_hospital_dynamic_anchor_xy[init_ids] = positions_xy[needs_init]
        env._go2w_hospital_dynamic_dir_xy[init_ids] = init_dir
        env._go2w_hospital_dynamic_long_speed[init_ids] = long_speed
        env._go2w_hospital_dynamic_lat_speed[init_ids] = lateral_speed
        env._go2w_hospital_dynamic_wander[init_ids] = wander
        env._go2w_hospital_dynamic_velocity_timer[init_ids] = timer
        env._go2w_hospital_dynamic_initialized[init_ids] = True

    anchor = env._go2w_hospital_dynamic_anchor_xy[env_ids]
    path_dir = env._go2w_hospital_dynamic_dir_xy[env_ids]
    normal = torch.stack((-path_dir[..., 1], path_dir[..., 0]), dim=-1)
    long_speed = env._go2w_hospital_dynamic_long_speed[env_ids]
    lateral_speed = env._go2w_hospital_dynamic_lat_speed[env_ids]
    wander = env._go2w_hospital_dynamic_wander[env_ids]
    velocity_timer = env._go2w_hospital_dynamic_velocity_timer[env_ids] - env.step_dt

    change_velocity = active & (velocity_timer <= 0.0)
    if change_velocity.any():
        rand = torch.rand(active.shape, device=device, dtype=dtype)
        new_speed = rand * (speed_hi - speed_lo).unsqueeze(0) + speed_lo.unsqueeze(0)
        direction = torch.where(long_speed < 0.0, -torch.ones_like(long_speed), torch.ones_like(long_speed))
        new_long = new_speed * direction
        new_lat = (torch.rand(active.shape, device=device, dtype=dtype) * 2.0 - 1.0) * lat_speed_max.unsqueeze(0)
        wander_heading = torch.empty(active.shape, device=device, dtype=dtype).uniform_(-math.pi, math.pi)
        new_long = torch.where(wander, new_speed * wander_heading.cos(), new_long)
        new_lat = torch.where(wander, new_speed * wander_heading.sin(), new_lat)
        long_speed = torch.where(change_velocity, new_long, long_speed)
        lateral_speed = torch.where(change_velocity, new_lat, lateral_speed)
        next_timer = torch.empty(active.shape, device=device, dtype=dtype).uniform_(*velocity_resample_interval_range)
        velocity_timer = torch.where(change_velocity, next_timer, velocity_timer)
    env._go2w_hospital_dynamic_velocity_timer[env_ids] = torch.where(
        active, velocity_timer, torch.zeros_like(velocity_timer)
    )

    offset = positions_xy - anchor
    long_offset = (offset * path_dir).sum(dim=-1)
    lat_offset = (offset * normal).sum(dim=-1)
    next_long = long_offset + long_speed * env.step_dt
    reflect_long = next_long.abs() > long_extent.unsqueeze(0).clamp(min=1.0e-6)
    long_speed = torch.where(reflect_long, -long_speed, long_speed)
    next_long = (long_offset + long_speed * env.step_dt).clamp(
        -long_extent.unsqueeze(0),
        long_extent.unsqueeze(0),
    )
    next_lat = lat_offset + lateral_speed * env.step_dt
    reflect_lat = next_lat.abs() > lat_extent.unsqueeze(0).clamp(min=1.0e-6)
    lateral_speed = torch.where(reflect_lat, -lateral_speed, lateral_speed)
    next_lat = (lat_offset + lateral_speed * env.step_dt).clamp(
        -lat_extent.unsqueeze(0),
        lat_extent.unsqueeze(0),
    )
    proposed = anchor + next_long.unsqueeze(-1) * path_dir + next_lat.unsqueeze(-1) * normal
    proposed = torch.where(active.unsqueeze(-1), proposed, positions_xy)

    radii = torch.zeros(len(env_ids), n_slots, device=device, dtype=dtype)
    if (
        metadata_indices is not None
        and hasattr(env, "_go2w_obstacle_effective_radius")
        and env._go2w_obstacle_effective_radius.shape[0] == n_envs
        and env._go2w_obstacle_effective_radius.shape[1] > int(metadata_indices.max().item())
    ):
        margin = float(getattr(env, "_go2w_obstacle_radius_margin", 0.0))
        radii = env._go2w_obstacle_effective_radius[env_ids][:, metadata_indices] + margin
    elif hasattr(env, "_go2w_obstacle_effective_radius"):
        reference = torch.zeros(n_envs, n_slots, device=device)
        maybe_radii = obstacle_effective_radius(env, obstacle_names, reference)
        if maybe_radii.shape == (n_envs, n_slots):
            radii = maybe_radii[env_ids]

    structured_projection_ready = False
    start_xy = None
    cos_yaw = None
    sin_yaw = None
    centerline_local = None
    extra_polylines_local = None
    extra_polyline_count = None
    safe_corridor_width = None
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
        safe_corridor_width = (
            env._go2w_structured_corridor_width[env_ids].unsqueeze(1)
            - 2.0 * (radii + 0.10)
        ).clamp(min=0.30)
        centerline_local = env._go2w_structured_corridor_centerline_local[env_ids]
        if hasattr(env, "_go2w_structured_corridor_extra_polylines_local") and hasattr(
            env, "_go2w_structured_corridor_extra_polyline_count"
        ):
            extra_polylines_local = env._go2w_structured_corridor_extra_polylines_local[env_ids]
            extra_polyline_count = env._go2w_structured_corridor_extra_polyline_count[env_ids]
        structured_projection_ready = True

    def _project_xy_to_safe_corridor(world_xy: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
        """Project world XY positions back into the physical corridor envelope."""
        if not structured_projection_ready:
            return world_xy
        rel_xy = world_xy - start_xy.unsqueeze(1)
        local_xy = torch.stack(
            (
                rel_xy[..., 0] * cos_yaw + rel_xy[..., 1] * sin_yaw,
                -rel_xy[..., 0] * sin_yaw + rel_xy[..., 1] * cos_yaw,
            ),
            dim=-1,
        )
        projected_xy = project_polyline_corridor_local(local_xy, centerline_local, widths)
        best_error = (local_xy - projected_xy).norm(dim=-1)
        if extra_polylines_local is not None and extra_polyline_count is not None:
            for extra_idx in range(extra_polylines_local.shape[1]):
                extra_centerline = extra_polylines_local[:, extra_idx]
                candidate_xy = project_polyline_corridor_local(local_xy, extra_centerline, widths)
                candidate_error = (local_xy - candidate_xy).norm(dim=-1)
                valid = extra_polyline_count > extra_idx
                use_extra = (candidate_error < best_error) & valid.unsqueeze(1)
                projected_xy = torch.where(use_extra.unsqueeze(-1), candidate_xy, projected_xy)
                best_error = torch.where(use_extra, candidate_error, best_error)
        return torch.stack(
            (
                start_xy[:, 0:1] + projected_xy[..., 0] * cos_yaw - projected_xy[..., 1] * sin_yaw,
                start_xy[:, 1:2] + projected_xy[..., 0] * sin_yaw + projected_xy[..., 1] * cos_yaw,
            ),
            dim=-1,
        )

    if structured_projection_ready and safe_corridor_width is not None:
        projected = _project_xy_to_safe_corridor(proposed, safe_corridor_width)
        proposed = torch.where(active.unsqueeze(-1), projected, proposed)

    linked_pair_mask = torch.zeros(n_slots, n_slots, device=device, dtype=torch.bool)
    if group_registry:
        for entry in group_registry:
            leader_idx = name_to_slot.get(entry["leader_name"])
            follower_idx = name_to_slot.get(entry["follower_name"])
            if leader_idx is None or follower_idx is None:
                continue
            linked_pair_mask[leader_idx, follower_idx] = True
            linked_pair_mask[follower_idx, leader_idx] = True

    if hasattr(env, "_go2w_navigation_path_w") and hasattr(env, "_go2w_navigation_path_count"):
        path_count = env._go2w_navigation_path_count[env_ids].clamp(min=1)
        final_idx = path_count - 1
        goal_xy = env._go2w_navigation_path_w[env_ids, final_idx, :2]
    elif hasattr(env, "_go2w_goal_pos_w"):
        goal_xy = env._go2w_goal_pos_w[env_ids, :2]
    else:
        goal_xy = robot_xy + corridor_dir * 10.0
    goal_keepout = active & (
        (proposed - goal_xy.unsqueeze(1)).norm(dim=-1)
        < (goal_exclusion_radius + radii).clamp(min=0.0)
    )
    robot_keepout = active & (
        (proposed - robot_xy.unsqueeze(1)).norm(dim=-1)
        < (robot_keepout_radius + radii).clamp(min=0.0)
    )

    accepted = positions_xy.clone()
    for slot_idx in range(n_slots):
        candidate = proposed[:, slot_idx]
        distances = (candidate.unsqueeze(1) - accepted).norm(dim=-1)
        pair_min_dist = min_inter_obstacle_dist + radii[:, slot_idx:slot_idx + 1] + radii
        others = (torch.arange(n_slots, device=device) != slot_idx) & ~linked_pair_mask[slot_idx]
        conflict = ((distances < pair_min_dist) & present & others.unsqueeze(0)).any(dim=1)
        conflict &= active[:, slot_idx]
        conflict |= goal_keepout[:, slot_idx] | robot_keepout[:, slot_idx]
        accepted[:, slot_idx] = torch.where(conflict.unsqueeze(-1), positions_xy[:, slot_idx], candidate)
        long_speed[:, slot_idx] = torch.where(conflict, -long_speed[:, slot_idx], long_speed[:, slot_idx])
        lateral_speed[:, slot_idx] = torch.where(conflict, -lateral_speed[:, slot_idx], lateral_speed[:, slot_idx])

    if group_registry:
        for entry in group_registry:
            leader_name = entry["leader_name"]
            follower_name = entry["follower_name"]
            if leader_name not in name_to_slot or follower_name not in name_to_slot:
                continue
            relation_type = entry["relation_type"]
            spec = HOSPITAL_RELATION_SPECS.get(relation_type)
            if spec is None:
                continue
            leader_idx = name_to_slot[leader_name]
            follower_idx = name_to_slot[follower_name]
            rel_active = active[:, leader_idx] & active[:, follower_idx]
            if not rel_active.any():
                continue
            leader_dir = path_dir[:, leader_idx]
            leader_normal = torch.stack((-leader_dir[:, 1], leader_dir[:, 0]), dim=-1)
            ox, oy = spec.desired_offset_xy
            desired = accepted[:, leader_idx] + ox * leader_dir + oy * leader_normal
            follower = accepted[:, follower_idx]
            dist = (follower - accepted[:, leader_idx]).norm(dim=-1)
            if relation_type == "guardian_child":
                rejoin = rel_active & (dist > spec.max_separation)
                candidate = torch.where(rejoin.unsqueeze(-1), desired, follower)
            elif relation_type == "handler_dog":
                travel_sign = torch.where(
                    long_speed[:, leader_idx] < 0.0,
                    -torch.ones_like(long_speed[:, leader_idx]),
                    torch.ones_like(long_speed[:, leader_idx]),
                ).unsqueeze(-1)
                leash_dir = leader_dir * travel_sign
                leash_normal = torch.stack((-leash_dir[:, 1], leash_dir[:, 0]), dim=-1)
                desired = accepted[:, leader_idx] + abs(ox) * leash_dir + oy * leash_normal
                candidate = torch.where(rel_active.unsqueeze(-1), desired, follower)
            else:
                candidate = torch.where(rel_active.unsqueeze(-1), desired, follower)
            follow_speed = 1.0 if relation_type == "handler_dog" else 0.75
            max_step = follow_speed * env.step_dt
            follow_delta = candidate - follower
            follow_dist = follow_delta.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
            limited_candidate = follower + follow_delta / follow_dist * follow_dist.clamp(max=max_step)
            candidate = torch.where(rel_active.unsqueeze(-1), limited_candidate, follower)
            if structured_projection_ready and safe_corridor_width is not None:
                candidate = _project_xy_to_safe_corridor(
                    candidate.unsqueeze(1),
                    safe_corridor_width[:, follower_idx:follower_idx + 1],
                ).squeeze(1)
            relation_blocked = rel_active & (
                ((candidate - robot_xy).norm(dim=-1) < (robot_keepout_radius + radii[:, follower_idx]).clamp(min=0.0))
                | ((candidate - goal_xy).norm(dim=-1) < (goal_exclusion_radius + radii[:, follower_idx]).clamp(min=0.0))
            )
            accepted[:, follower_idx] = torch.where(relation_blocked.unsqueeze(-1), follower, candidate)

    if seated_groups:
        for group in seated_groups:
            slot_idx = name_to_slot.get(group.get("name"))
            if slot_idx is not None:
                writable_slot[slot_idx] = True

    if queue_groups:
        group_count = len(queue_groups)
        if (
            not hasattr(env, "_go2w_hospital_queue_phase")
            or env._go2w_hospital_queue_phase.shape != (n_envs, group_count)
            or not hasattr(env, "_go2w_hospital_queue_served")
            or env._go2w_hospital_queue_served.shape != (n_envs, group_count, n_slots)
        ):
            env._go2w_hospital_queue_phase = torch.zeros(n_envs, group_count, device=device, dtype=dtype)
            env._go2w_hospital_queue_timer = torch.zeros(n_envs, group_count, device=device, dtype=dtype)
            env._go2w_hospital_queue_moving = torch.zeros(n_envs, group_count, device=device, dtype=torch.bool)
            env._go2w_hospital_queue_served = torch.zeros(
                n_envs, group_count, n_slots, device=device, dtype=torch.bool
            )

        if needs_init.any():
            init_ids = env_ids[needs_init]
            for group_idx, group in enumerate(queue_groups):
                lo, hi = group.get("idle_interval_range", (4.0, 8.0))
                env._go2w_hospital_queue_phase[init_ids, group_idx] = 0.0
                env._go2w_hospital_queue_timer[init_ids, group_idx] = torch.empty(
                    len(init_ids), device=device, dtype=dtype
                ).uniform_(lo, hi)
                env._go2w_hospital_queue_moving[init_ids, group_idx] = False
                env._go2w_hospital_queue_served[init_ids, group_idx] = False

        for group_idx, group in enumerate(queue_groups):
            slots = [name_to_slot[name] for name in group.get("names", ()) if name in name_to_slot]
            explicit_direction = group.get("advance_direction_local_xy")
            if len(slots) < 2 and explicit_direction is None:
                continue
            spacing = float(group.get("spacing", 0.65))
            advance_speed = float(group.get("advance_speed", 0.18))
            shuffle_amplitude = float(group.get("shuffle_amplitude", 0.03))
            wave_delay = float(group.get("wave_delay_per_person", 0.0))
            idle_lo, idle_hi = group.get("idle_interval_range", (4.0, 8.0))
            exit_distance = float(group.get("exit_distance", spacing))

            phase = env._go2w_hospital_queue_phase[env_ids, group_idx]
            timer = env._go2w_hospital_queue_timer[env_ids, group_idx] - env.step_dt
            moving_queue = env._go2w_hospital_queue_moving[env_ids, group_idx]
            slot_tensor = torch.tensor(slots, device=device, dtype=torch.long)
            served = env._go2w_hospital_queue_served[env_ids, group_idx].clone()
            served_slots = served[:, slot_tensor]
            remaining = (~served_slots).any(dim=1)
            moving_queue = moving_queue & remaining
            start_move = (~moving_queue) & remaining & (timer <= 0.0)
            moving_queue = moving_queue | start_move
            phase = torch.where(start_move, torch.zeros_like(phase), phase)
            phase = torch.where(moving_queue, phase + advance_speed * env.step_dt, phase)
            cycle_length = max(2.0 * spacing + wave_delay * max(len(slots) - 1, 0), 1.0e-6)
            finished = moving_queue & (phase >= cycle_length)
            next_timer = torch.empty(len(env_ids), device=device, dtype=dtype).uniform_(idle_lo, idle_hi)
            timer = torch.where(finished, next_timer, timer)
            phase_for_pose = phase.clamp(max=cycle_length)
            phase = torch.where(finished, torch.zeros_like(phase), phase_for_pose)
            moving_queue = moving_queue & ~finished
            env._go2w_hospital_queue_phase[env_ids, group_idx] = phase
            env._go2w_hospital_queue_timer[env_ids, group_idx] = timer
            env._go2w_hospital_queue_moving[env_ids, group_idx] = moving_queue

            base = anchor[:, slots]
            if explicit_direction is not None:
                line_dir = _local_offset_to_world(env, env_ids, explicit_direction, dtype)
                line_dir = line_dir / line_dir.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
            elif len(slots) >= 2:
                line = base[:, 0] - base[:, -1]
                line_dir = line / line.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
            else:
                line_dir = path_dir[:, slots[0]]
            normal_dir = torch.stack((-line_dir[:, 1], line_dir[:, 0]), dim=-1)
            shuffle = torch.sin(phase_for_pose * 11.0 + float(group_idx)) * shuffle_amplitude
            shuffle = torch.where(finished, torch.zeros_like(shuffle), shuffle)
            projection = (base * line_dir.unsqueeze(1)).sum(dim=-1)
            projection = torch.where(served_slots, torch.full_like(projection, -1.0e9), projection)
            order = torch.argsort(projection, dim=1, descending=True)
            ordered_slots = slot_tensor.unsqueeze(0).expand(len(env_ids), -1).gather(1, order)
            ordered_base = base.gather(1, order.unsqueeze(-1).expand(-1, -1, 2))
            ordered_served = served_slots.gather(1, order)
            active_order = ~ordered_served
            front_available = active_order[:, 0] & remaining
            service_pos = ordered_base[:, 0] + line_dir * spacing
            exit_dir = normal_dir
            exit_pos = service_pos + exit_dir * exit_distance
            committed_anchor = anchor.clone()
            updated_served = served.clone()
            for order_idx, slot_idx in enumerate(slots):
                slot_per_env = ordered_slots[:, order_idx]
                order_active = active_order[:, order_idx] & remaining
                delayed_phase = (phase_for_pose - wave_delay * float(max(order_idx - 1, 0))).clamp(
                    min=0.0, max=2.0 * spacing
                )
                if order_idx == 0:
                    service_alpha = (delayed_phase / spacing).clamp(0.0, 1.0)
                    exit_alpha = ((delayed_phase - spacing) / spacing).clamp(0.0, 1.0)
                    service_leg = ordered_base[:, 0] + (service_pos - ordered_base[:, 0]) * service_alpha.unsqueeze(-1)
                    queue_pos = service_leg + (exit_pos - service_leg) * exit_alpha.unsqueeze(-1)
                    commit_pos = exit_pos
                else:
                    shift = ((phase_for_pose - spacing - wave_delay * float(order_idx - 1)) / spacing).clamp(0.0, 1.0)
                    commit_pos = ordered_base[:, order_idx - 1]
                    queue_pos = ordered_base[:, order_idx] + (
                        commit_pos - ordered_base[:, order_idx]
                    ) * shift.unsqueeze(-1)
                queue_pos = queue_pos + normal_dir * shuffle.unsqueeze(-1)
                queue_pos = torch.where(finished.unsqueeze(-1), commit_pos, queue_pos)
                slot_active = present.gather(1, slot_per_env.unsqueeze(1)).squeeze(1)
                slot_radius = radii.gather(1, slot_per_env.unsqueeze(1)).squeeze(1)
                current_pos = accepted.gather(1, slot_per_env.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
                safe = (
                    ((queue_pos - robot_xy).norm(dim=-1) > (robot_keepout_radius + slot_radius).clamp(min=0.0))
                    & ((queue_pos - goal_xy).norm(dim=-1) > (goal_exclusion_radius + slot_radius).clamp(min=0.0))
                    & slot_active
                    & order_active
                )
                next_pos = torch.where(safe.unsqueeze(-1), queue_pos, current_pos)
                accepted.scatter_(1, slot_per_env.view(-1, 1, 1).expand(-1, 1, 2), next_pos.unsqueeze(1))
                committed_anchor.scatter_(
                    1,
                    slot_per_env.view(-1, 1, 1).expand(-1, 1, 2),
                    torch.where(
                        order_active.view(-1, 1, 1),
                        commit_pos.unsqueeze(1),
                        anchor.gather(1, slot_per_env.view(-1, 1, 1).expand(-1, 1, 2)),
                    ),
                )

            finish_active = finished & front_available
            updated_anchor = torch.where(finish_active.view(-1, 1, 1), committed_anchor, anchor)
            env._go2w_hospital_dynamic_anchor_xy[env_ids] = updated_anchor
            anchor = updated_anchor
            front_slot = ordered_slots[:, 0]
            served_value = served.gather(1, front_slot.unsqueeze(1)).squeeze(1) | finish_active
            updated_served.scatter_(1, front_slot.unsqueeze(1), served_value.unsqueeze(1))
            env._go2w_hospital_queue_served[env_ids, group_idx] = updated_served

    if seated_groups:
        group_count = len(seated_groups)
        if (
            not hasattr(env, "_go2w_hospital_seated_phase")
            or env._go2w_hospital_seated_phase.shape != (n_envs, group_count)
        ):
            env._go2w_hospital_seated_phase = torch.zeros(n_envs, group_count, device=device, dtype=dtype)
            env._go2w_hospital_seated_timer = torch.zeros(n_envs, group_count, device=device, dtype=dtype)
            env._go2w_hospital_seated_moving = torch.zeros(n_envs, group_count, device=device, dtype=torch.bool)
            env._go2w_hospital_seated_complete = torch.zeros(n_envs, group_count, device=device, dtype=torch.bool)

        if needs_init.any():
            init_ids = env_ids[needs_init]
            for group_idx, group in enumerate(seated_groups):
                lo, hi = group.get("stand_delay_range", (7.0, 14.0))
                env._go2w_hospital_seated_phase[init_ids, group_idx] = 0.0
                env._go2w_hospital_seated_timer[init_ids, group_idx] = torch.empty(
                    len(init_ids), device=device, dtype=dtype
                ).uniform_(lo, hi)
                env._go2w_hospital_seated_moving[init_ids, group_idx] = False
                env._go2w_hospital_seated_complete[init_ids, group_idx] = False

        for group_idx, group in enumerate(seated_groups):
            name = group.get("name")
            if name not in name_to_slot:
                continue
            slot_idx = name_to_slot[name]
            duration = max(float(group.get("stand_duration", 2.5)), 1.0e-6)
            world_offset = _local_offset_to_world(
                env,
                env_ids,
                group.get("merge_offset_xy", (0.0, 0.9)),
                dtype,
            )
            phase = env._go2w_hospital_seated_phase[env_ids, group_idx]
            timer = env._go2w_hospital_seated_timer[env_ids, group_idx] - env.step_dt
            moving_seated = env._go2w_hospital_seated_moving[env_ids, group_idx]
            complete = env._go2w_hospital_seated_complete[env_ids, group_idx]
            start_move = (~moving_seated) & (~complete) & (timer <= 0.0)
            moving_seated = moving_seated | start_move
            phase = torch.where(start_move, torch.zeros_like(phase), phase)
            phase = torch.where(moving_seated, (phase + env.step_dt / duration).clamp(max=1.0), phase)
            finished = moving_seated & (phase >= 1.0)
            complete = complete | finished
            moving_seated = moving_seated & ~finished
            env._go2w_hospital_seated_phase[env_ids, group_idx] = phase
            env._go2w_hospital_seated_timer[env_ids, group_idx] = torch.where(complete, torch.zeros_like(timer), timer)
            env._go2w_hospital_seated_moving[env_ids, group_idx] = moving_seated
            env._go2w_hospital_seated_complete[env_ids, group_idx] = complete

            seated_pos = anchor[:, slot_idx] + world_offset * phase.unsqueeze(-1)
            slot_active = present[:, slot_idx]
            safe = (
                ((seated_pos - robot_xy).norm(dim=-1) > (robot_keepout_radius + radii[:, slot_idx]).clamp(min=0.0))
                & ((seated_pos - goal_xy).norm(dim=-1) > (goal_exclusion_radius + radii[:, slot_idx]).clamp(min=0.0))
                & slot_active
            )
            accepted[:, slot_idx] = torch.where(safe.unsqueeze(-1), seated_pos, accepted[:, slot_idx])

    if structured_projection_ready and safe_corridor_width is not None:
        projected_accepted = _project_xy_to_safe_corridor(accepted, safe_corridor_width)
        constrained = active.clone()
        if group_registry:
            for entry in group_registry:
                follower_idx = name_to_slot.get(entry.get("follower_name"))
                leader_idx = name_to_slot.get(entry.get("leader_name"))
                if follower_idx is not None:
                    constrained[:, follower_idx] |= active[:, follower_idx]
                if entry.get("relation_type") == "handler_dog" and leader_idx is not None:
                    constrained[:, leader_idx] |= active[:, leader_idx]
        if queue_groups:
            for group in queue_groups:
                for name in group.get("names", ()):
                    slot_idx = name_to_slot.get(name)
                    if slot_idx is not None:
                        constrained[:, slot_idx] |= active[:, slot_idx]
        if seated_groups:
            for group in seated_groups:
                slot_idx = name_to_slot.get(group.get("name"))
                if slot_idx is not None:
                    constrained[:, slot_idx] |= active[:, slot_idx]
        accepted = torch.where(constrained.unsqueeze(-1), projected_accepted, accepted)

    # Keep the lightweight first-pass conflict rejection above as the default.
    # Use only a bounded final correction so overlap is reduced without visible teleports.
    writable_mask = torch.tensor(writable_slot, device=device, dtype=torch.bool).unsqueeze(0)
    movable = present & writable_mask
    slot_ids = torch.arange(n_slots, device=device)
    not_self = slot_ids.unsqueeze(0) != slot_ids.unsqueeze(1)
    pair_present = present.unsqueeze(2) & present.unsqueeze(1)
    pair_movable = movable.unsqueeze(2) | movable.unsqueeze(1)
    linked_pair = linked_pair_mask.unsqueeze(0)
    pair_margin = torch.where(
        linked_pair,
        torch.full((1, n_slots, n_slots), 0.02, device=device, dtype=dtype),
        torch.full((1, n_slots, n_slots), min_inter_obstacle_dist, device=device, dtype=dtype),
    )
    pair_scale = torch.where(
        linked_pair,
        torch.full((1, n_slots, n_slots), 0.85, device=device, dtype=dtype),
        torch.ones((1, n_slots, n_slots), device=device, dtype=dtype),
    )
    min_dist = pair_scale * (radii.unsqueeze(2) + radii.unsqueeze(1)) + pair_margin
    pair_valid = pair_present & pair_movable & not_self.unsqueeze(0)
    move_weight = torch.where(
        movable.unsqueeze(2) & movable.unsqueeze(1),
        torch.full((len(env_ids), n_slots, n_slots), 0.5, device=device, dtype=dtype),
        torch.ones((len(env_ids), n_slots, n_slots), device=device, dtype=dtype),
    )
    move_weight = torch.where(movable.unsqueeze(2), move_weight, torch.zeros_like(move_weight))
    fallback_sign = torch.where(
        slot_ids.unsqueeze(0) > slot_ids.unsqueeze(1),
        torch.ones((n_slots, n_slots), device=device, dtype=dtype),
        -torch.ones((n_slots, n_slots), device=device, dtype=dtype),
    )
    fallback_dir = path_dir.unsqueeze(2) * fallback_sign.unsqueeze(0).unsqueeze(-1)
    delta = accepted.unsqueeze(2) - accepted.unsqueeze(1)
    dist = delta.norm(dim=-1).clamp(min=1.0e-6)
    sep_dir = torch.where((dist > 1.0e-5).unsqueeze(-1), delta / dist.unsqueeze(-1), fallback_dir)
    overlap_depth = (min_dist - dist).clamp(min=0.0)
    overlap_depth = torch.where(pair_valid, overlap_depth, torch.zeros_like(overlap_depth))
    correction = (overlap_depth * move_weight).unsqueeze(-1) * sep_dir
    correction_xy = correction.sum(dim=2)
    correction_norm = correction_xy.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
    correction_xy = correction_xy / correction_norm * correction_norm.clamp(max=0.035)
    correction_moved = movable & (correction_xy.norm(dim=-1) > 1.0e-6)
    accepted = torch.where(correction_moved.unsqueeze(-1), accepted + correction_xy, accepted)
    if structured_projection_ready and safe_corridor_width is not None:
        projected_accepted = _project_xy_to_safe_corridor(accepted, safe_corridor_width)
        accepted = torch.where(correction_moved.unsqueeze(-1), projected_accepted, accepted)

    env._go2w_hospital_dynamic_long_speed[env_ids] = long_speed
    env._go2w_hospital_dynamic_lat_speed[env_ids] = lateral_speed

    center_z = torch.full((n_slots,), 0.30, device=device, dtype=dtype)
    if obstacle_center_zs is not None:
        center_z = torch.tensor(obstacle_center_zs, device=device, dtype=dtype)
    obstacle_yaws = torch.zeros(len(env_ids), n_slots, device=device, dtype=dtype)
    if (
        metadata_indices is not None
        and hasattr(env, "_go2w_obstacle_yaw")
        and env._go2w_obstacle_yaw.shape[0] == n_envs
        and env._go2w_obstacle_yaw.shape[1] > int(metadata_indices.max().item())
    ):
        obstacle_yaws = env._go2w_obstacle_yaw[env_ids][:, metadata_indices]
    motion_delta = accepted - positions_xy
    moving = motion_delta.norm(dim=-1) > 1.0e-4
    motion_yaw = torch.atan2(motion_delta[..., 1], motion_delta[..., 0])
    tangent_labels = {
        "wheelchair_patient",
        "cart",
        "cleaning_machine",
        "gurney_patient",
    }
    free_yaw_labels = {
        "patient_ambulatory",
        "elderly",
        "adult",
        "child",
        "staff",
        "visitor",
        "dog",
        "patient_with_iv",
        "doorway_patient",
        "doorway_staff",
        "queue_patient",
        "queue_visitor",
        "seated_patient",
        "seated_visitor",
    }
    tangent_mask = torch.tensor(
        [label in tangent_labels for label in obstacle_labels],
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)
    free_yaw_mask = torch.tensor(
        [label in free_yaw_labels for label in obstacle_labels],
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)
    tangent_yaw = torch.atan2(path_dir[..., 1], path_dir[..., 0])
    obstacle_yaws = torch.where(active & tangent_mask, tangent_yaw, obstacle_yaws)
    obstacle_yaws = torch.where(moving & free_yaw_mask, motion_yaw, obstacle_yaws)

    zero_vel = torch.zeros(len(env_ids), 6, device=device)
    for slot_idx, name in enumerate(obstacle_names):
        if not writable_slot[slot_idx]:
            continue
        write_mask = moving[:, slot_idx]
        if not bool(write_mask.any().item()):
            continue
        write_env_ids = env_ids[write_mask]
        pose = torch.zeros(len(write_env_ids), 7, device=device)
        pose[:, :2] = accepted[write_mask, slot_idx]
        pose[:, 2] = center_z[slot_idx]
        pose[:, 3:7] = _yaw_to_quat_wxyz(obstacle_yaws[write_mask, slot_idx])
        env.scene[name].write_root_pose_to_sim(pose, env_ids=write_env_ids)
        env.scene[name].write_root_velocity_to_sim(zero_vel[write_mask], env_ids=write_env_ids)


# ---------------------------------------------------------------------------
# Episode manifest logging
# ---------------------------------------------------------------------------

def write_episode_manifest(
    env: ManagerBasedRLEnv,  # noqa: ARG001 — reserved for future env-state queries
    log_dir: str,
    episode_seed: int,
    template_name: str,
    layout_name: str,
    scenario_name: str,
    spawned_labels: list[str],
    dynamic_density: float,
) -> None:
    """Write a replayable episode manifest to ``<log_dir>/episode_manifest.jsonl``.

    Args:
        env: The RL environment instance.
        log_dir: Directory where the manifest file is written.
        episode_seed: RNG seed used for this episode.
        template_name: Hospital layout template key.
        layout_name: Human-readable layout variant name.
        scenario_name: Scenario identifier (e.g. ``"busy_shift_start"``).
        spawned_labels: Ordered list of labels for each spawned obstacle.
        dynamic_density: Fraction of obstacles that are dynamic (0–1).
    """
    from .logging import HospitalEpisodeManifest, write_jsonl
    from collections import Counter

    label_summary = dict(Counter(spawned_labels))
    relation_summary: dict[str, int] = {}
    for label in spawned_labels:
        spec = HOSPITAL_LABEL_SPECS.get(label)
        if spec and spec.relation_type != "independent":
            relation_summary[spec.relation_type] = relation_summary.get(spec.relation_type, 0) + 1

    manifest = HospitalEpisodeManifest(
        episode_seed=episode_seed,
        template_name=template_name,
        layout_name=layout_name,
        scenario_name=scenario_name,
        dynamic_density=dynamic_density,
        label_summary=label_summary,
        relation_summary=relation_summary,
    )
    write_jsonl(f"{log_dir}/episode_manifest.jsonl", [manifest])


__all__ = [
    "resample_hospital_obstacle_velocities",
    "move_hospital_dynamic_obstacles",
    "write_episode_manifest",
]
