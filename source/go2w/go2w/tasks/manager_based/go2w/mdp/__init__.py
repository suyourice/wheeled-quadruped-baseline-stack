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
from .actions import FrozenLLCActionTermCfg

# -- Commands ------------------------------------------------------------------
from isaaclab.envs.mdp import (
    UniformVelocityCommandCfg,
    generated_commands,
)

# -- Observations --------------------------------------------------------------
from isaaclab.envs.mdp import (
    base_ang_vel,
    base_lin_vel,
    base_pos_z,
    body_incoming_wrench,
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
    is_terminated_term,
    joint_deviation_l1,
    joint_torques_l2,
    lin_vel_z_l2,
    undesired_contacts,
)

# -- Observations (local custom) -----------------------------------------------
from .observations import (
    goal_position_w,
    lidar_distances,
    local_goal_command_b,
    obstacle_navigation_features,
    obstacle_polar_depth,
    obstacle_positions_rel,
    prev_hlc_actions,
    root_position_w,
    start_position_w,
    waypoint_position_w,
)

# -- Events (local custom) -----------------------------------------------------
from .events import (
    move_dynamic_play_obstacles,
    reset_navigation_goals_and_obstacles,
    reset_obstacles_curriculum,
    update_locomotion_curriculum,
)

# -- Rewards (local custom) ----------------------------------------------------
from .rewards import (
    base_height_l2,
    goal_distance_tanh_reward,
    goal_heading_tanh_reward,
    goal_progress_dense,
    goal_reached_and_resample,
    goal_reached_bonus,
    goal_reached_termination,
    joint_deviation_l1_command_gated,
    joint_deviation_l1_curriculum,
    nav_clearance_penalty,
    nav_dense_recovery_reward,
    nav_frontal_blocked_lateral_escape_reward,
    nav_grazing_penalty,
    nav_impossible_gap_penalty,
    nav_near_goal_settling_reward,
    nav_open_path_straightness_reward,
    nav_passable_gap_traversal_reward,
    obstacle_contact_penalty,
    obstacle_contact_termination,
    obstacle_nav_ttc_penalty,
    track_ang_vel_z_world_exp,
    track_lin_vel_xy_yaw_frame_exp,
    wheel_contact_penalty,
    wheel_vel_zero_cmd,
)

# -- Terminations --------------------------------------------------------------
from isaaclab.envs.mdp import (
    illegal_contact,
    root_height_below_minimum,
    time_out,
)
