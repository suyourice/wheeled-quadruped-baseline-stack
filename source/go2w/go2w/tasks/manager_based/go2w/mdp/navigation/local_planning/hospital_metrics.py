# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared hospital-corridor geometry metrics for observations and rewards."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hospital_centerline_metrics(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project robot pose onto the stored hospital centerline."""
    robot = env.scene[robot_cfg.name]
    n_envs = env.num_envs
    device = env.device
    if not hasattr(env, "_go2w_structured_corridor_centerline_local"):
        zero = torch.zeros(n_envs, device=device)
        return zero, zero, zero, zero, zero

    cache_key = (int(getattr(env, "common_step_counter", 0)), robot_cfg.name)
    cached_key = getattr(env, "_go2w_hospital_centerline_cache_key", None)
    cached_value = getattr(env, "_go2w_hospital_centerline_cache", None)
    if cached_key == cache_key and cached_value is not None:
        return cached_value

    origin = env._go2w_structured_corridor_start_xy
    corridor_yaw = env._go2w_structured_corridor_yaw
    rel_w = robot.data.root_pos_w[:, :2] - origin
    cos_yaw = torch.cos(corridor_yaw)
    sin_yaw = torch.sin(corridor_yaw)
    local_xy = torch.stack(
        (
            cos_yaw * rel_w[:, 0] + sin_yaw * rel_w[:, 1],
            -sin_yaw * rel_w[:, 0] + cos_yaw * rel_w[:, 1],
        ),
        dim=-1,
    )

    centerline = env._go2w_structured_corridor_centerline_local
    count = getattr(
        env,
        "_go2w_structured_corridor_centerline_count",
        torch.full((n_envs,), centerline.shape[1], dtype=torch.long, device=device),
    )
    best_abs = torch.full((n_envs,), float("inf"), device=device)
    best_signed = torch.zeros(n_envs, device=device)
    best_heading = torch.zeros(n_envs, device=device)
    best_segment_idx = torch.zeros(n_envs, dtype=torch.long, device=device)

    for segment_idx in range(centerline.shape[1] - 1):
        valid = count > segment_idx + 1
        a = centerline[:, segment_idx]
        b = centerline[:, segment_idx + 1]
        seg = b - a
        seg_len_sq = (seg * seg).sum(dim=-1).clamp(min=1.0e-8)
        u = ((local_xy - a) * seg).sum(dim=-1) / seg_len_sq
        closest = a + u.clamp(0.0, 1.0).unsqueeze(-1) * seg
        delta = local_xy - closest
        seg_len = seg_len_sq.sqrt()
        nx = -seg[:, 1] / seg_len
        ny = seg[:, 0] / seg_len
        signed = delta[:, 0] * nx + delta[:, 1] * ny
        abs_signed = signed.abs()
        update = valid & (abs_signed < best_abs)
        best_abs = torch.where(update, abs_signed, best_abs)
        best_signed = torch.where(update, signed, best_signed)
        best_heading = torch.where(update, torch.atan2(seg[:, 1], seg[:, 0]), best_heading)
        best_segment_idx = torch.where(
            update,
            torch.full_like(best_segment_idx, segment_idx),
            best_segment_idx,
        )

    curvature = torch.zeros(n_envs, device=device)
    for segment_idx in range(centerline.shape[1] - 2):
        a = centerline[:, segment_idx]
        b = centerline[:, segment_idx + 1]
        c = centerline[:, segment_idx + 2]
        h0 = torch.atan2(b[:, 1] - a[:, 1], b[:, 0] - a[:, 0])
        h1 = torch.atan2(c[:, 1] - b[:, 1], c[:, 0] - b[:, 0])
        turn = wrap_to_pi(h1 - h0).abs()
        mask = (best_segment_idx == segment_idx) & (count > segment_idx + 2)
        curvature = torch.where(mask, turn, curvature)

    if hasattr(env, "_go2w_hospital_path_reversed"):
        best_heading = torch.where(
            env._go2w_hospital_path_reversed,
            wrap_to_pi(best_heading + math.pi),
            best_heading,
        )
    heading_error = wrap_to_pi(best_heading + corridor_yaw - robot.data.heading_w)
    half_width = env._go2w_structured_corridor_width.clamp(min=0.1) * 0.5
    left_clearance = (half_width - best_signed).clamp(min=0.0)
    right_clearance = (half_width + best_signed).clamp(min=0.0)
    result = (best_signed, heading_error, curvature, left_clearance, right_clearance)
    env._go2w_hospital_centerline_cache_key = cache_key
    env._go2w_hospital_centerline_cache = result
    return result
