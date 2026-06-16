# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared observation slice definitions for Go2-W navigation tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObsSlice:
    """Half-open observation range."""

    start: int
    stop: int

    @property
    def dim(self) -> int:
        return self.stop - self.start

    def as_slice(self) -> slice:
        return slice(self.start, self.stop)


# LLC (fast-flat) policy obs - 60D.
# Reconstructed in FrozenLLCActionTerm._build_llc_obs; matches Go2wEnvCfg.PolicyCfg.
POLICY_OBS = {
    "base_lin_vel":     ObsSlice(0,  3),   # mdp.base_lin_vel
    "base_ang_vel":     ObsSlice(3,  6),   # mdp.base_ang_vel
    "projected_gravity": ObsSlice(6,  9),  # mdp.projected_gravity
    "goal_command":     ObsSlice(9,  12),  # velocity command injected by HLC
    "joint_pos":        ObsSlice(12, 28),  # mdp.joint_pos_rel (relative to default)
    "joint_vel":        ObsSlice(28, 44),  # mdp.joint_vel
    "actions":          ObsSlice(44, 60),  # raw LLC MLP output from previous step
}

# HLC teacher policy obs - 451D.
# Defined in NavTeacherObsCfg.PolicyCfg (cfg/navigation/env.py).
HLC_TEACHER_OBS = {
    "proprio":           ObsSlice(0,   9),    # base_lin_vel(3) + projected_gravity(3) + goal_command(3)
    "polar_depth":       ObsSlice(9,   189),  # mdp.obstacle_polar_depth (180 bins)
    "nav_features":      ObsSlice(189, 205),  # mdp.obstacle_navigation_features (16D)
    "full_geometry":     ObsSlice(205, 445),  # mdp.obstacle_full_geometry_features (15 slots x 16D)
    "prev_actions":      ObsSlice(445, 451),  # mdp.prev_hlc_actions (2 frames x 3D)
}

# HLC student obs - 189D.
# Defined in NavRLDistillObsCfg.StudentCfg (cfg/navigation/env.py).
HLC_STUDENT_OBS = {
    "proprio":      ObsSlice(0,   9),    # base_lin_vel(3) + projected_gravity(3) + goal_command(3)
    "lidar_scan":   ObsSlice(9,   189),  # mdp.lidar_distances (180 bins)
}
