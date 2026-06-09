# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simple A* polyline following helpers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import wrap_to_pi

from .debug_utils import fmt_xy, nav_debug_enabled, nav_debug_env_id, nav_debug_interval

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ensure_path_buffers(
    env: ManagerBasedRLEnv,
    path_w: torch.Tensor,
) -> None:
    path_count = path_w.shape[1]
    if (
        not hasattr(env, "_go2w_navigation_path_w")
        or env._go2w_navigation_path_w.shape != (env.num_envs, path_count, 3)
    ):
        env._go2w_navigation_path_w = torch.zeros(
            env.num_envs, path_count, 3, dtype=path_w.dtype, device=env.device
        )
        env._go2w_navigation_path_s = torch.zeros(
            env.num_envs, path_count, dtype=path_w.dtype, device=env.device
        )
        env._go2w_navigation_path_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._go2w_navigation_path_target_index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._go2w_navigation_path_nearest_index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._go2w_navigation_path_final_distance = torch.zeros(env.num_envs, dtype=path_w.dtype, device=env.device)
        env._go2w_navigation_path_progress_s = torch.zeros(env.num_envs, dtype=path_w.dtype, device=env.device)
        env._go2w_navigation_path_target_s = torch.zeros(env.num_envs, dtype=path_w.dtype, device=env.device)



def set_navigation_path_w(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    path_w: torch.Tensor | list[tuple[float, float]] | list[tuple[float, float, float]],
) -> None:
    """Store an A* path as a world-frame polyline.

    The follower treats the path as immutable.  Dynamic obstacle avoidance is left
    to the local policy; this helper only computes progress and lookahead along
    the A* polyline.
    """
    if len(env_ids) == 0:
        return
    if not torch.is_tensor(path_w):
        path_w = torch.tensor(path_w, dtype=torch.float32, device=env.device)
    else:
        path_w = path_w.to(device=env.device)

    if path_w.dim() == 2:
        path_w = path_w.unsqueeze(0).expand(len(env_ids), -1, -1).clone()
    elif path_w.dim() != 3:
        raise ValueError("path_w must have shape (P, 2/3) or (N, P, 2/3).")

    if path_w.shape[0] == 1 and len(env_ids) > 1:
        path_w = path_w.expand(len(env_ids), -1, -1).clone()
    if path_w.shape[0] != len(env_ids):
        raise ValueError("path_w batch dimension must match env_ids length.")
    if path_w.shape[1] < 2:
        raise ValueError("Navigation paths require at least two waypoints.")

    if path_w.shape[-1] == 2:
        robot_z = env.scene["robot"].data.root_pos_w[env_ids, 2].view(len(env_ids), 1, 1)
        path_w = torch.cat((path_w, robot_z.expand(-1, path_w.shape[1], -1)), dim=-1)
    elif path_w.shape[-1] != 3:
        raise ValueError("path_w last dimension must be 2 or 3.")

    segment_len = torch.zeros(path_w.shape[0], path_w.shape[1], dtype=path_w.dtype, device=env.device)
    segment_len[:, 1:] = (path_w[:, 1:, :2] - path_w[:, :-1, :2]).norm(dim=-1)
    path_s = torch.cumsum(segment_len, dim=1)

    _ensure_path_buffers(env, path_w)
    env._go2w_navigation_path_w[env_ids] = path_w
    env._go2w_navigation_path_s[env_ids] = path_s
    env._go2w_navigation_path_count[env_ids] = path_w.shape[1]
    env._go2w_navigation_path_target_index[env_ids] = 0
    env._go2w_navigation_path_nearest_index[env_ids] = 0
    env._go2w_navigation_path_final_distance[env_ids] = float("inf")
    env._go2w_navigation_path_progress_s[env_ids] = 0.0
    env._go2w_navigation_path_target_s[env_ids] = 0.0


def _path_max_angle_ahead(
    path_xy: torch.Tensor,
    path_s: torch.Tensor,
    nearest_idx: int,
    progress_s: float,
    horizon_s: float,
) -> float:
    """Return the max absolute turn angle (rad) at any waypoint within horizon_s of progress_s."""
    scan_end = progress_s + horizon_s
    n = int(path_xy.shape[0])
    max_angle = 0.0
    for i in range(max(0, nearest_idx), n - 2):
        if float(path_s[i].item()) > scan_end:
            break
        ab = path_xy[i + 1] - path_xy[i]
        bc = path_xy[min(i + 2, n - 1)] - path_xy[i + 1]
        len_ab = float(ab.norm().item())
        len_bc = float(bc.norm().item())
        if len_ab < 1.0e-6 or len_bc < 1.0e-6:
            continue
        cos_a = float((ab * bc).sum().item()) / (len_ab * len_bc)
        angle = math.acos(max(-1.0, min(1.0, cos_a)))
        if angle > max_angle:
            max_angle = angle
    return max_angle


def _adaptive_lookahead(
    max_angle: float,
    lookahead_max: float,
    lookahead_min: float,
    curvature_threshold: float,
) -> float:
    """Map path curvature angle to a reduced lookahead distance via smoothstep."""
    if max_angle <= curvature_threshold:
        return lookahead_max
    upper = max(curvature_threshold + 1.0e-6, math.pi / 2.0)
    t = min(1.0, (max_angle - curvature_threshold) / (upper - curvature_threshold))
    t = t * t * (3.0 - 2.0 * t)  # smoothstep
    return lookahead_max + (lookahead_min - lookahead_max) * t


def _project_robot_onto_path(
    path_xy: torch.Tensor,
    path_s: torch.Tensor,
    final_idx: int,
    robot_xy: torch.Tensor,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Project the robot onto the closest point of the path polyline."""
    best_dist_sq = None
    best_s = path_s[0]
    best_idx = 0
    best_point = path_xy[0]

    for idx in range(final_idx):
        a = path_xy[idx]
        b = path_xy[idx + 1]
        seg = b - a
        seg_len_sq = (seg * seg).sum()
        if float(seg_len_sq.item()) <= 1.0e-10:
            continue
        u = ((robot_xy - a) * seg).sum() / seg_len_sq
        u = u.clamp(0.0, 1.0)
        point = a + u * seg
        dist_sq = ((robot_xy - point) * (robot_xy - point)).sum()
        if best_dist_sq is None or float(dist_sq.item()) < float(best_dist_sq.item()):
            seg_len = seg_len_sq.sqrt()
            best_dist_sq = dist_sq
            best_s = path_s[idx] + u * seg_len
            best_idx = idx
            best_point = point

    return best_s, best_idx, best_point


def _sample_path_at_s(
    path_w: torch.Tensor,
    path_s: torch.Tensor,
    final_idx: int,
    target_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Sample a target position and tangent heading from the path polyline."""
    if target_s >= path_s[final_idx]:
        from_idx = max(0, final_idx - 1)
        to_idx = final_idx
        segment = path_w[to_idx, :2] - path_w[from_idx, :2]
        heading = torch.atan2(segment[1], segment[0])
        return path_w[final_idx].clone(), heading, to_idx

    for idx in range(final_idx):
        s0 = path_s[idx]
        s1 = path_s[idx + 1]
        if bool((target_s <= s1).item()):
            denom = (s1 - s0).clamp(min=1.0e-6)
            u = ((target_s - s0) / denom).clamp(0.0, 1.0)
            pos = path_w[idx].clone()
            pos[:2] = path_w[idx, :2] + u * (path_w[idx + 1, :2] - path_w[idx, :2])
            segment = path_w[idx + 1, :2] - path_w[idx, :2]
            heading = torch.atan2(segment[1], segment[0])
            return pos, heading, idx + 1

    segment = path_w[final_idx, :2] - path_w[max(0, final_idx - 1), :2]
    heading = torch.atan2(segment[1], segment[0])
    return path_w[final_idx].clone(), heading, final_idx


def update_navigation_path_waypoint(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
    lookahead_distance: float = 1.25,
    waypoint_reach_radius: float = 0.45,
    adaptive_lookahead: bool = True,
    lookahead_min: float = 0.6,
    curvature_scan_horizon: float = 2.5,
    curvature_threshold: float = 0.3,
) -> None:
    """Update the local navigation target from progress along the stored A* path.

    When adaptive_lookahead is enabled, the lookahead distance is reduced near turns
    so the robot never sees a far target across a corner.  The effective lookahead
    shrinks from lookahead_distance to lookahead_min as path curvature ahead (within
    curvature_scan_horizon) increases above curvature_threshold.
    """
    if not hasattr(env, "_go2w_navigation_path_w"):
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if len(env_ids) == 0:
        return

    path_w_all = env._go2w_navigation_path_w
    path_s_all = env._go2w_navigation_path_s
    path_count = env._go2w_navigation_path_count
    robot_xy_all = env.scene["robot"].data.root_pos_w[:, :2]
    heading_all = env.scene["robot"].data.heading_w

    previous_nearest = env._go2w_navigation_path_nearest_index[env_ids].clone()
    previous_target = env._go2w_navigation_path_target_index[env_ids].clone()

    for local_row, env_id_tensor in enumerate(env_ids):
        env_id = int(env_id_tensor.item())
        count = int(path_count[env_id].item())
        if count < 2:
            continue

        final_idx = count - 1
        path_w = path_w_all[env_id, :count]
        path_s = path_s_all[env_id, :count]
        robot_xy = robot_xy_all[env_id]
        final = path_w[final_idx]

        projected_s, nearest_idx, _ = _project_robot_onto_path(path_w[:, :2], path_s, final_idx, robot_xy)
        final_s = path_s[final_idx]
        progress_s = projected_s.clamp(max=final_s)

        if adaptive_lookahead:
            max_angle = _path_max_angle_ahead(
                path_w[:, :2], path_s, nearest_idx,
                float(progress_s.item()), curvature_scan_horizon,
            )
            effective_lookahead = _adaptive_lookahead(
                max_angle, lookahead_distance, lookahead_min, curvature_threshold
            )
        else:
            max_angle = 0.0
            effective_lookahead = lookahead_distance

        target_s = (progress_s + effective_lookahead).clamp(max=final_s)

        target, segment_heading, target_idx = _sample_path_at_s(path_w, path_s, final_idx, target_s)
        final_distance = (robot_xy - final[:2]).norm()
        if final_distance <= waypoint_reach_radius:
            target = final.clone()
            target_idx = final_idx
            segment = final[:2] - path_w[max(0, final_idx - 1), :2]
            if segment.norm() > 1.0e-4:
                segment_heading = torch.atan2(segment[1], segment[0])
            progress_s = final_s
            target_s = final_s

        env._go2w_goal_pos_w[env_id] = target
        env._go2w_goal_heading_w[env_id] = segment_heading
        env._go2w_navigation_path_nearest_index[env_id] = nearest_idx
        env._go2w_navigation_path_target_index[env_id] = target_idx
        env._go2w_navigation_path_final_distance[env_id] = final_distance
        env._go2w_navigation_path_progress_s[env_id] = progress_s
        env._go2w_navigation_path_target_s[env_id] = target_s

        if nav_debug_enabled():
            step = int(getattr(env, "common_step_counter", 0))
            debug_env = nav_debug_env_id()
            goal_vec = target[:2] - robot_xy
            target_forward_b = (
                torch.cos(heading_all[env_id]) * goal_vec[0]
                + torch.sin(heading_all[env_id]) * goal_vec[1]
            )
            should_print = (
                step % nav_debug_interval() == 0
                or (
                    env_id == debug_env
                    and (
                        target_idx < int(previous_target[local_row].item())
                        or nearest_idx < int(previous_nearest[local_row].item())
                        or target_forward_b < -0.05
                    )
                )
            )
            if should_print:
                heading_error_b = wrap_to_pi(segment_heading - heading_all[env_id])
                print(
                    "[GO2W_NAV_PATH] "
                    f"step={step} env={env_id} "
                    f"nearest={int(previous_nearest[local_row].item())}->{nearest_idx} "
                    f"target={int(previous_target[local_row].item())}->{target_idx} "
                    f"final={final_idx} final_dist={float(final_distance.item()):.2f} "
                    f"progress_s={float(progress_s.item()):.2f} target_s={float(target_s.item()):.2f} "
                    f"lookahead={effective_lookahead:.2f} curvature={max_angle:.2f} "
                    f"target_fwd_b={float(target_forward_b.item()):+.2f} "
                    f"path_head_b={float(heading_error_b.item()):+.2f} "
                    f"robot={fmt_xy(robot_xy)} target={fmt_xy(target[:2])} final={fmt_xy(final[:2])}"
                )
