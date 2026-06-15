# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation math helpers shared by reset, observations, and hospital events."""

from __future__ import annotations

import torch


def quat_yaw_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Return yaw angle from a wxyz quaternion tensor."""
    w, x, y, z = quat.unbind(dim=-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def yaw_to_quat_wxyz(yaw: torch.Tensor) -> torch.Tensor:
    """Return a wxyz quaternion for a planar yaw angle tensor."""
    half_yaw = yaw * 0.5
    quat = torch.zeros(*yaw.shape, 4, device=yaw.device, dtype=yaw.dtype)
    quat[..., 0] = torch.cos(half_yaw)
    quat[..., 3] = torch.sin(half_yaw)
    return quat


def yaw_pitch_to_quat_wxyz(yaw: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    """Return a wxyz quaternion for Rz(yaw) * Ry(pitch)."""
    half_yaw = yaw * 0.5
    half_pitch = pitch * 0.5
    cz = torch.cos(half_yaw)
    sz = torch.sin(half_yaw)
    cp = torch.cos(half_pitch)
    sp = torch.sin(half_pitch)
    quat = torch.zeros(*yaw.shape, 4, device=yaw.device, dtype=yaw.dtype)
    quat[..., 0] = cz * cp
    quat[..., 1] = -sz * sp
    quat[..., 2] = cz * sp
    quat[..., 3] = sz * cp
    return quat


def yaw_pitch_roll_to_quat_wxyz(
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    roll: torch.Tensor,
) -> torch.Tensor:
    """Return a wxyz quaternion for Rz(yaw) * Ry(pitch) * Rx(roll).

    Used for the ramp hump: with pitch=0 and roll=−π/2 the cylinder axis (Z)
    is rotated to point perpendicular to the corridor, creating a symmetric
    speed-hump profile where both entry and exit ends are at ground level.
    """
    hz, hp, hr = yaw * 0.5, pitch * 0.5, roll * 0.5
    cz, sz = torch.cos(hz), torch.sin(hz)
    cp, sp = torch.cos(hp), torch.sin(hp)
    cr, sr = torch.cos(hr), torch.sin(hr)
    quat = torch.zeros(*yaw.shape, 4, device=yaw.device, dtype=yaw.dtype)
    quat[..., 0] = cz * cp * cr + sz * sp * sr
    quat[..., 1] = cz * cp * sr - sz * sp * cr
    quat[..., 2] = cz * sp * cr + sz * cp * sr
    quat[..., 3] = -cz * sp * sr + sz * cp * cr
    return quat
