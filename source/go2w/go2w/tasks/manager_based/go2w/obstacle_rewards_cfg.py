# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward configuration for the Go2-W RL navigation teacher."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .mdp.hospital.specs import *  # noqa: F401, F403


@configclass
class NavTeacherRewardsCfg:
    """Rewards for the RL navigation teacher.

    Combines goal-conditioned navigation rewards with locomotion stability
    penalties. No velocity-command tracking — the goal buffers drive learning.
    """

    # -- Navigation (goal-conditioned) -----------------------------------------
    goal_progress = RewTerm(
        func=mdp.goal_progress_dense,
        weight=5.0,
        params={"clip": 1.0},
    )
    goal_heading = RewTerm(
        func=mdp.goal_heading_tanh_reward,
        weight=0.5,
        params={"std": NAV_GOAL_HEADING_STD},
    )
    goal_reached = RewTerm(
        func=mdp.goal_reached_and_resample,
        weight=100.0,
        params={
            "position_threshold": NAV_GOAL_SUCCESS_POSITION_THRESHOLD,
            "heading_threshold": NAV_GOAL_SUCCESS_HEADING_THRESHOLD,
            "goal_forward_range": NAV_GOAL_FORWARD_RANGE,
            "goal_lateral_range": NAV_GOAL_LATERAL_RANGE,
            "goal_heading_jitter_range": NAV_GOAL_HEADING_JITTER_RANGE,
            "min_goal_distance": NAV_MIN_GOAL_DISTANCE,
        },
    )

    # -- Obstacle avoidance ----------------------------------------------------
    obstacle_collision = RewTerm(
        func=mdp.obstacle_contact_penalty,
        weight=OBSTACLE_COLLISION_WEIGHT,
        params={
            "sensor_cfg": SceneEntityCfg("obstacle_contacts"),
            "threshold": 1.0,
            "start_iteration": NAV_CURRICULUM_COLLISION_START_ITERATION,
            "warmup_iterations": CURRICULUM_COLLISION_WARMUP_ITERATIONS,
            "steps_per_iteration": CURRICULUM_STEPS_PER_ITERATION,
        },
    )
    nav_clearance = RewTerm(
        func=mdp.nav_clearance_penalty,
        weight=-1.5,
        params={
            "obstacle_names": OBSTACLE_NAMES,
            "min_safe_dist": NAV_CLEARANCE_SURFACE_BUFFER,
            "robot_safety_radius": NAV_TTC_ROBOT_HALF_WIDTH,
            # Soften proximity penalty while threading a passable narrow gap.
            "passable_gap_relief": NAV_CLEARANCE_PASSABLE_GAP_RELIEF,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    nav_lateral_escape = RewTerm(
        func=mdp.nav_frontal_blocked_lateral_escape_reward,
        weight=2.5,
        params={
            "obstacle_names": OBSTACLE_NAMES,
            "robot_cfg": SceneEntityCfg("robot"),
            # Gate lateral escape on goal_path_blockage so the reward is
            # suppressed when the direct robot→goal corridor is clear.
            "goal_path_min_blockage": 0.10,
            "goal_path_corridor_half_width": 0.7,
        },
    )
    # Reward straight-line progress toward the goal when the direct path is open.
    # Suppressed when goal_path_blockage is high so avoidance maneuvers are free.
    nav_open_path_straightness = RewTerm(
        func=mdp.nav_open_path_straightness_reward,
        weight=1.2,
        params={
            "obstacle_names": OBSTACLE_NAMES,
            "robot_cfg": SceneEntityCfg("robot"),
            "goal_path_corridor_half_width": 0.7,
        },
    )
    nav_open_path_goal_heading = RewTerm(
        func=mdp.nav_open_path_goal_heading_reward,
        weight=0.6,
        params={
            "obstacle_names": OBSTACLE_NAMES,
            "robot_cfg": SceneEntityCfg("robot"),
            "goal_path_corridor_half_width": 0.7,
        },
    )
    # Encourage calm (low speed, low yaw rate) when the robot is very close to goal.
    nav_near_goal_settling = RewTerm(
        func=mdp.nav_near_goal_settling_reward,
        weight=0.8,
        params={"settling_distance": 0.5},
    )
    # Penalise pushing forward when hemmed in on all sides (impossible gap).
    nav_impossible_gap = RewTerm(
        func=mdp.nav_impossible_gap_penalty,
        weight=-2.0,
        params={
            "obstacle_names": OBSTACLE_NAMES,
            "robot_cfg": SceneEntityCfg("robot"),
            "high_frontal_threshold": 0.40,
            "side_blocked_threshold": 0.15,
            "min_gap_available": 0.35,
            "min_gap_width_norm": 0.45,
            "gap_reference_width": 0.7,
        },
    )
    obstacle_ttc = RewTerm(
        func=mdp.obstacle_nav_ttc_penalty,
        weight=-4.0,
        params={
            "obstacle_names": OBSTACLE_NAMES,
            "safe_ttc": 1.0,
            "command_name": "base_velocity",
            "obstacle_radius": NAV_TTC_FALLBACK_OBSTACLE_RADIUS,
            "robot_half_width": NAV_TTC_ROBOT_HALF_WIDTH,
            "safety_margin": NAV_TTC_SAFETY_MARGIN,
            "robot_front_margin": NAV_TTC_FRONT_MARGIN,
            "lookahead_distance": NAV_TTC_LOOKAHEAD_DISTANCE,
            "sum_clip": NAV_TTC_SUM_CLIP,
            # Soften corridor TTC while aligned in a passable narrow gap so the
            # robot is not blocked from entering barely-passable corridors.
            "passable_gap_relief": NAV_TTC_PASSABLE_GAP_RELIEF,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # Encourage decisive traversal of passable narrow gaps (scenarios 8/13/14).
    # Gated on the per-env _go2w_gap_passable flag set at reset; never fires for
    # impossible-gap/dead-end layouts.
    nav_passable_gap = RewTerm(
        func=mdp.nav_passable_gap_traversal_reward,
        weight=NAV_PASSABLE_GAP_REWARD_WEIGHT,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )
    # Encourage productive recovery (lateral/backward/turn) instead of stopping in
    # cluttered/partial-blockage layouts. Gated on blocked/cluttered states only.
    nav_dense_recovery = RewTerm(
        func=mdp.nav_dense_recovery_reward,
        weight=NAV_DENSE_RECOVERY_WEIGHT,
        params={
            "obstacle_names": OBSTACLE_NAMES,
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    # Mild near-contact penalty to reduce leg/wheel grazing; far weaker than the
    # collision penalty and relieved while aligned in a passable gap.
    nav_grazing = RewTerm(
        func=mdp.nav_grazing_penalty,
        weight=NAV_GRAZING_WEIGHT,
        params={
            "obstacle_names": OBSTACLE_NAMES,
            "robot_cfg": SceneEntityCfg("robot"),
            "graze_distance": NAV_GRAZING_DISTANCE,
            "contact_distance": NAV_GRAZING_CONTACT_DISTANCE,
            "robot_safety_radius": NAV_TTC_ROBOT_HALF_WIDTH,
            "passable_gap_relief": NAV_GRAZING_PASSABLE_GAP_RELIEF,
        },
    )

    # -- Stability -------------------------------------------------------------
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.5)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-3.0,
        params={"target_height": 0.45, "robot_cfg": SceneEntityCfg("robot")},
    )

    # -- Posture ---------------------------------------------------------------
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])},
    )
    joint_deviation_stance = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_joint", ".*_calf_joint"])},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )

    # -- Wheel contact / smoothness -------------------------------------------
    wheel_contact = RewTerm(
        func=mdp.wheel_contact_penalty,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"])},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
            "threshold": 1.0,
        },
    )

    # -- Termination -----------------------------------------------------------
    termination_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-200.0,
        params={"term_keys": ["base_contact", "root_height_below_minimum"]},
    )
