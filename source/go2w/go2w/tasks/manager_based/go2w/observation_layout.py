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


POLICY_OBS = {
    "base_lin_vel": ObsSlice(0, 3),
    "base_ang_vel": ObsSlice(3, 6),
    "projected_gravity": ObsSlice(6, 9),
    "goal_command": ObsSlice(9, 12),
    "joint_pos": ObsSlice(12, 28),
    "joint_vel": ObsSlice(28, 44),
    "actions": ObsSlice(44, 60),
}

GOAL_COMMAND = POLICY_OBS["goal_command"]
GOAL_COMMAND_START = GOAL_COMMAND.start
GOAL_COMMAND_DIM = GOAL_COMMAND.dim
PRIVILEGED_OBSTACLE_START = POLICY_OBS["actions"].stop

DEBUG_OBS = {
    "root_position_w": ObsSlice(0, 3),
    "base_lin_vel": ObsSlice(3, 6),
    "base_ang_vel": ObsSlice(6, 9),
    "goal_command": ObsSlice(9, 12),
    "joint_pos": ObsSlice(12, 28),
    "joint_vel": ObsSlice(28, 44),
    "actions": ObsSlice(44, 60),
    "start_position_w": ObsSlice(60, 63),
    "waypoint_position_w": ObsSlice(63, 66),
    "goal_position_w": ObsSlice(66, 69),
    "scenario_template_code": ObsSlice(69, 70),
}

DEBUG_OBSTACLE_START = DEBUG_OBS["scenario_template_code"].stop
