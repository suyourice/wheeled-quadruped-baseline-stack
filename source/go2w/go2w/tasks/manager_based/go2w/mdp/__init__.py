# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP sub-module for the Go2-W locomotion task.

All symbols are imported explicitly so that static-analysis tools
(Pyright, pylance) can resolve them without tracing wildcard imports.
"""

# -- Actions -------------------------------------------------------------------
from isaaclab.envs.mdp import (
    JointPositionActionCfg,
    JointVelocityActionCfg,
)

# -- Commands ------------------------------------------------------------------
from isaaclab.envs.mdp import (
    UniformVelocityCommandCfg,
    generated_commands,
)

# -- Observations --------------------------------------------------------------
from isaaclab.envs.mdp import (
    base_ang_vel,
    base_lin_vel,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    projected_gravity,
)

# -- Events --------------------------------------------------------------------
from isaaclab.envs.mdp import (
    push_by_setting_velocity,
    randomize_rigid_body_mass,
    randomize_rigid_body_material,
    reset_joints_by_scale,
    reset_root_state_uniform,
)

# -- Rewards (isaaclab built-in) -----------------------------------------------
from isaaclab.envs.mdp import (
    action_rate_l2,
    ang_vel_xy_l2,
    flat_orientation_l2,
    is_terminated,
    joint_deviation_l1,
    joint_torques_l2,
    lin_vel_z_l2,
    undesired_contacts,
)

# -- Rewards (local custom) ----------------------------------------------------
from .rewards import track_ang_vel_z_world_exp, track_lin_vel_xy_yaw_frame_exp

# -- Terminations --------------------------------------------------------------
from isaaclab.envs.mdp import (
    illegal_contact,
    root_height_below_minimum,
    time_out,
)
