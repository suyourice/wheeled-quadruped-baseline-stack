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

from ...common.debug import fmt_xy, nav_debug_enabled, nav_debug_env_id, nav_debug_interval

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
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if len(env_ids) == 0:
        return

    path_w = env._go2w_navigation_path_w[env_ids]
    path_s = env._go2w_navigation_path_s[env_ids]
    count = env._go2w_navigation_path_count[env_ids].clamp(min=2)
    robot_xy = env.scene["robot"].data.root_pos_w[env_ids, :2]
    heading = env.scene["robot"].data.heading_w[env_ids]
    batch = torch.arange(env_ids.numel(), device=env.device)

    segment_count = path_w.shape[1] - 1
    segment_idx = torch.arange(segment_count, device=env.device, dtype=torch.long)
    valid_segment = segment_idx.unsqueeze(0) < (count - 1).unsqueeze(1)

    seg_start = path_w[:, :-1, :2]
    seg_end = path_w[:, 1:, :2]
    seg = seg_end - seg_start
    seg_len_sq = (seg * seg).sum(dim=-1)
    valid_segment = valid_segment & (seg_len_sq > 1.0e-10)

    rel_robot = robot_xy.unsqueeze(1) - seg_start
    u = ((rel_robot * seg).sum(dim=-1) / seg_len_sq.clamp(min=1.0e-10)).clamp(0.0, 1.0)
    closest = seg_start + u.unsqueeze(-1) * seg
    dist_sq = ((robot_xy.unsqueeze(1) - closest) ** 2).sum(dim=-1)
    dist_sq = torch.where(valid_segment, dist_sq, torch.full_like(dist_sq, float("inf")))
    nearest_idx = dist_sq.argmin(dim=1)

    final_idx = count - 1
    final = path_w[batch, final_idx]
    final_s = path_s[batch, final_idx]
    nearest_u = u[batch, nearest_idx]
    nearest_seg_len = seg_len_sq[batch, nearest_idx].sqrt()
    projected_s = path_s[batch, nearest_idx] + nearest_u * nearest_seg_len
    progress_s = torch.minimum(projected_s.clamp(min=0.0), final_s)

    if adaptive_lookahead and segment_count > 1:
        heading_a = torch.atan2(seg[:, :-1, 1], seg[:, :-1, 0])
        heading_b = torch.atan2(seg[:, 1:, 1], seg[:, 1:, 0])
        turn = wrap_to_pi(heading_b - heading_a).abs()
        turn_idx = torch.arange(segment_count - 1, device=env.device, dtype=torch.long)
        scan_end = progress_s + curvature_scan_horizon
        turn_valid = (
            (turn_idx.unsqueeze(0) >= nearest_idx.unsqueeze(1))
            & (turn_idx.unsqueeze(0) < (count - 2).unsqueeze(1))
            & (path_s[:, :-2] <= scan_end.unsqueeze(1))
        )
        max_angle = torch.where(turn_valid, turn, torch.zeros_like(turn)).sum(dim=1)
        upper = max(curvature_threshold + 1.0e-6, math.pi / 2.0)
        blend = ((max_angle - curvature_threshold) / (upper - curvature_threshold)).clamp(0.0, 1.0)
        blend = blend * blend * (3.0 - 2.0 * blend)
        effective_lookahead = lookahead_distance + (lookahead_min - lookahead_distance) * blend
    else:
        max_angle = torch.zeros(env_ids.numel(), device=env.device)
        effective_lookahead = torch.full_like(progress_s, lookahead_distance)

    target_s = torch.minimum(progress_s + effective_lookahead, final_s)
    target_segment_mask = valid_segment & (target_s.unsqueeze(1) <= path_s[:, 1:])
    fallback_idx = (count - 2).clamp(min=0)
    target_from_idx = torch.where(
        target_segment_mask,
        segment_idx.unsqueeze(0).expand_as(target_segment_mask),
        torch.full(target_segment_mask.shape, segment_count, dtype=torch.long, device=env.device),
    ).min(dim=1).values
    target_from_idx = torch.where(target_from_idx < segment_count, target_from_idx, fallback_idx)
    target_to_idx = target_from_idx + 1

    p0 = path_w[batch, target_from_idx]
    p1 = path_w[batch, target_to_idx]
    s0 = path_s[batch, target_from_idx]
    s1 = path_s[batch, target_to_idx]
    target_u = ((target_s - s0) / (s1 - s0).clamp(min=1.0e-6)).clamp(0.0, 1.0)
    target = p0 + target_u.unsqueeze(-1) * (p1 - p0)
    target_segment = p1[:, :2] - p0[:, :2]
    segment_heading = torch.atan2(target_segment[:, 1], target_segment[:, 0])

    final_distance = (robot_xy - final[:, :2]).norm(dim=-1)
    reached = final_distance <= waypoint_reach_radius
    previous_final_idx = (final_idx - 1).clamp(min=0)
    previous_final = path_w[batch, previous_final_idx]
    final_segment = final[:, :2] - previous_final[:, :2]
    final_segment_valid = final_segment.norm(dim=-1) > 1.0e-4
    final_heading = torch.atan2(final_segment[:, 1], final_segment[:, 0])

    target = torch.where(reached.unsqueeze(-1), final, target)
    segment_heading = torch.where(reached & final_segment_valid, final_heading, segment_heading)
    target_to_idx = torch.where(reached, final_idx, target_to_idx)
    progress_s = torch.where(reached, final_s, progress_s)
    target_s = torch.where(reached, final_s, target_s)

    env._go2w_goal_pos_w[env_ids] = target
    env._go2w_goal_heading_w[env_ids] = segment_heading
    env._go2w_navigation_path_nearest_index[env_ids] = nearest_idx
    env._go2w_navigation_path_target_index[env_ids] = target_to_idx
    env._go2w_navigation_path_final_distance[env_ids] = final_distance
    env._go2w_navigation_path_progress_s[env_ids] = progress_s
    env._go2w_navigation_path_target_s[env_ids] = target_s

    if nav_debug_enabled():
        debug_env = nav_debug_env_id()
        debug_rows = (env_ids == debug_env).nonzero(as_tuple=False).flatten()
        if debug_rows.numel() > 0 and int(getattr(env, "common_step_counter", 0)) % nav_debug_interval() == 0:
            row = int(debug_rows[0].item())
            heading_error_b = wrap_to_pi(segment_heading[row] - heading[row])
            goal_vec = target[row, :2] - robot_xy[row]
            target_forward_b = torch.cos(heading[row]) * goal_vec[0] + torch.sin(heading[row]) * goal_vec[1]
            print(
                "[GO2W_NAV_PATH] "
                f"step={int(getattr(env, 'common_step_counter', 0))} env={debug_env} "
                f"nearest={int(nearest_idx[row].item())} target={int(target_to_idx[row].item())} "
                f"final={int(final_idx[row].item())} final_dist={float(final_distance[row].item()):.2f} "
                f"progress_s={float(progress_s[row].item()):.2f} target_s={float(target_s[row].item()):.2f} "
                f"lookahead={float(effective_lookahead[row].item()):.2f} "
                f"curvature={float(max_angle[row].item()):.2f} "
                f"target_fwd_b={float(target_forward_b.item()):+.2f} "
                f"path_head_b={float(heading_error_b.item()):+.2f} "
                f"robot={fmt_xy(robot_xy[row])} target={fmt_xy(target[row, :2])} final={fmt_xy(final[row, :2])}"
            )


def update_navigation_path_waypoint_event(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    lookahead_distance: float = 1.25,
    waypoint_reach_radius: float = 0.45,
    adaptive_lookahead: bool = True,
    lookahead_min: float = 0.6,
    curvature_scan_horizon: float = 2.5,
    curvature_threshold: float = 0.3,
) -> None:
    """Event-manager wrapper for rolling waypoint updates.

    IsaacLab validates event terms by treating the first two positional
    arguments as ``env`` and ``env_ids``. Keep ``env_ids`` mandatory here so
    optional-parameter validation stays aligned with interval events.
    """
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
