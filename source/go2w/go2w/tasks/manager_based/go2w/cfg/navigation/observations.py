# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation configurations for the Go2-W RL navigation teacher and distillation students."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ... import mdp
from ...mdp.navigation.hospital.specs import *  # noqa: F401, F403


@configclass
class NavTeacherObsCfg:
    """PPO observations for the RL navigation teacher (451D).

    proprio(9D) + obstacle_polar_depth(180D) + obstacle_nav_features(16D)
    + obstacle_full_geometry(240D) + prev_hlc_actions(6D).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel,      noise=Unoise(n_min=-0.1,  n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity,  noise=Unoise(n_min=-0.05, n_max=0.05))

        # Goal direction in body frame (3D); obstacle avoidance is learned by the HLC.
        goal_command = ObsTerm(
            func=mdp.local_goal_command_b,
            params={
                "lookahead_distance": NAV_WAYPOINT_LOOKAHEAD_DISTANCE,
                "goal_snap_distance": NAV_WAYPOINT_GOAL_SNAP_DISTANCE,
                "command_min_forward": NAV_WAYPOINT_COMMAND_MIN_FORWARD,
                "command_max_lateral": NAV_WAYPOINT_COMMAND_MAX_LATERAL,
                "command_max_heading": NAV_WAYPOINT_COMMAND_MAX_HEADING,
            },
        )

        obstacle_depth = ObsTerm(
            func=mdp.obstacle_polar_depth,
            params={
                "obstacle_names": OBSTACLE_NAMES,
                "num_bins": 180,
                "max_distance": LIDAR_MAX_DISTANCE,
                "robot_safety_radius": NAV_TTC_ROBOT_HALF_WIDTH,
            },
        )

        # Privileged geometry features (16D): nearest/frontal/side clearance,
        # preferred lateral escape direction, gap detection, TTC proxy.
        obstacle_nav_features = ObsTerm(
            func=mdp.obstacle_navigation_features,
            params={
                "obstacle_names": OBSTACLE_NAMES,
                "robot_cfg": SceneEntityCfg("robot"),
                "command_name": "base_velocity",
                "robot_safety_radius": NAV_TTC_ROBOT_HALF_WIDTH,
            },
        )

        # Teacher-only full geometry (15 slots x 16D): active flag, robot-frame
        # position, projected footprint, shape, clearance, and relative yaw.
        obstacle_full_geometry = ObsTerm(
            func=mdp.obstacle_full_geometry_features,
            params={
                "obstacle_names": OBSTACLE_NAMES,
                "robot_cfg": SceneEntityCfg("robot"),
                "num_slots": PRIVILEGED_OBSTACLE_SLOTS,
                "max_distance": 8.0,
                "robot_safety_radius": NAV_TTC_ROBOT_HALF_WIDTH,
            },
        )

        # Short temporal action history (2 frames × 3D = 6D).
        prev_actions = ObsTerm(
            func=mdp.prev_hlc_actions,
            params={"num_frames": 2, "action_term_name": "llc_cmd"},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class NavRLDistillObsCfg:
    """Distillation observations: student (LiDAR) and teacher (privileged).

    student: proprio(9D) + lidar_scan(180D) = 189D
    teacher: proprio(9D) + obstacle_polar_depth(180D) + obstacle_nav_features(16D)
             + obstacle_full_geometry(240D) + prev_hlc_actions(6D) = 451D
    """

    @configclass
    class StudentCfg(ObsGroup):
        """LiDAR student observations."""

        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel,      noise=Unoise(n_min=-0.1,  n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity,  noise=Unoise(n_min=-0.05, n_max=0.05))

        # Goal direction in body frame.  The policy learns avoidance from LiDAR.
        goal_command = ObsTerm(
            func=mdp.local_goal_command_b,
            params={
                "lookahead_distance": NAV_WAYPOINT_LOOKAHEAD_DISTANCE,
                "goal_snap_distance": NAV_WAYPOINT_GOAL_SNAP_DISTANCE,
                "command_min_forward": NAV_WAYPOINT_COMMAND_MIN_FORWARD,
                "command_max_lateral": NAV_WAYPOINT_COMMAND_MAX_LATERAL,
                "command_max_heading": NAV_WAYPOINT_COMMAND_MAX_HEADING,
            },
        )

        # 1 channel × 180 rays = 180D LiDAR scan.
        lidar_scan = ObsTerm(
            func=mdp.lidar_distances,
            params={"sensor_cfg": SceneEntityCfg("lidar"), "max_distance": LIDAR_MAX_DISTANCE},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class TeacherCfg(ObsGroup):
        """Privileged teacher observations (451D, must match NavTeacherObsCfg.PolicyCfg)."""

        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel,      noise=Unoise(n_min=-0.1,  n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity,  noise=Unoise(n_min=-0.05, n_max=0.05))

        goal_command = ObsTerm(
            func=mdp.local_goal_command_b,
            params={
                "lookahead_distance": NAV_WAYPOINT_LOOKAHEAD_DISTANCE,
                "goal_snap_distance": NAV_WAYPOINT_GOAL_SNAP_DISTANCE,
                "command_min_forward": NAV_WAYPOINT_COMMAND_MIN_FORWARD,
                "command_max_lateral": NAV_WAYPOINT_COMMAND_MAX_LATERAL,
                "command_max_heading": NAV_WAYPOINT_COMMAND_MAX_HEADING,
            },
        )

        obstacle_depth = ObsTerm(
            func=mdp.obstacle_polar_depth,
            params={
                "obstacle_names": OBSTACLE_NAMES,
                "num_bins": 180,
                "max_distance": LIDAR_MAX_DISTANCE,
                "robot_safety_radius": NAV_TTC_ROBOT_HALF_WIDTH,
            },
        )

        obstacle_nav_features = ObsTerm(
            func=mdp.obstacle_navigation_features,
            params={
                "obstacle_names": OBSTACLE_NAMES,
                "robot_cfg": SceneEntityCfg("robot"),
                "command_name": "base_velocity",
                "robot_safety_radius": NAV_TTC_ROBOT_HALF_WIDTH,
            },
        )

        obstacle_full_geometry = ObsTerm(
            func=mdp.obstacle_full_geometry_features,
            params={
                "obstacle_names": OBSTACLE_NAMES,
                "robot_cfg": SceneEntityCfg("robot"),
                "num_slots": PRIVILEGED_OBSTACLE_SLOTS,
                "max_distance": 8.0,
                "robot_safety_radius": NAV_TTC_ROBOT_HALF_WIDTH,
            },
        )

        prev_actions = ObsTerm(
            func=mdp.prev_hlc_actions,
            params={"num_frames": 2, "action_term_name": "llc_cmd"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    student: StudentCfg = StudentCfg()
    teacher: TeacherCfg = TeacherCfg()


@configclass
class NavDepthRLDistillObsCfg:
    """Depth distillation observations.

    student_state: proprio/goal/action history (15D)
    student_depth: depth history stack (T x H x W)
    teacher: privileged teacher observation matching NavTeacherObsCfg.PolicyCfg
    """

    @configclass
    class StudentStateCfg(ObsGroup):
        """Low-dimensional student state paired with depth images."""

        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel,      noise=Unoise(n_min=-0.1,  n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity,  noise=Unoise(n_min=-0.05, n_max=0.05))

        goal_command = ObsTerm(
            func=mdp.local_goal_command_b,
            params={
                "lookahead_distance": NAV_WAYPOINT_LOOKAHEAD_DISTANCE,
                "goal_snap_distance": NAV_WAYPOINT_GOAL_SNAP_DISTANCE,
                "command_min_forward": NAV_WAYPOINT_COMMAND_MIN_FORWARD,
                "command_max_lateral": NAV_WAYPOINT_COMMAND_MAX_LATERAL,
                "command_max_heading": NAV_WAYPOINT_COMMAND_MAX_HEADING,
            },
        )

        prev_actions = ObsTerm(
            func=mdp.prev_hlc_actions,
            params={"num_frames": 2, "action_term_name": "llc_cmd"},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class StudentDepthCfg(ObsGroup):
        """Head-mounted D456-like depth history for the CNN student."""

        depth_stack = ObsTerm(
            func=mdp.depth_closeness_image,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "data_type": "distance_to_image_plane",
                "min_depth": D456_DEPTH_MIN_DISTANCE,
                "max_depth": D456_DEPTH_MAX_DISTANCE,
            },
            history_length=DEPTH_HISTORY_LENGTH,
            flatten_history_dim=False,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    student_state: StudentStateCfg = StudentStateCfg()
    student_depth: StudentDepthCfg = StudentDepthCfg()
    teacher: NavRLDistillObsCfg.TeacherCfg = NavRLDistillObsCfg.TeacherCfg()


@configclass
class NavDepthRLDistillLongHistObsCfg(NavDepthRLDistillObsCfg):
    """Depth distillation observations with longer dense history (8 frames = 0.14s)."""

    @configclass
    class StudentDepthLongHistCfg(ObsGroup):
        depth_stack = ObsTerm(
            func=mdp.depth_closeness_image,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "data_type": "distance_to_image_plane",
                "min_depth": D456_DEPTH_MIN_DISTANCE,
                "max_depth": D456_DEPTH_MAX_DISTANCE,
            },
            history_length=DEPTH_HISTORY_LENGTH_LONG,
            flatten_history_dim=False,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    student_depth: StudentDepthLongHistCfg = StudentDepthLongHistCfg()


@configclass
class NavDepthRLDistillMultiCamObsCfg(NavDepthRLDistillObsCfg):
    """Depth distillation observations with 4 cameras (front/left/right/rear).

    Returns (N, 12, H, W) = 4 cams x 3 frames.  The multicam function manages
    its own rolling buffer so no ObsTerm history_length is needed.
    """

    @configclass
    class StudentDepthMultiCamCfg(ObsGroup):
        depth_stack = ObsTerm(
            func=mdp.depth_closeness_multicam_image,
            params={
                "sensor_cfgs": [
                    SceneEntityCfg("depth_camera"),
                    SceneEntityCfg("depth_camera_left"),
                    SceneEntityCfg("depth_camera_right"),
                    SceneEntityCfg("depth_camera_rear"),
                ],
                "data_type": "distance_to_image_plane",
                "min_depth": D456_DEPTH_MIN_DISTANCE,
                "max_depth": D456_DEPTH_MAX_DISTANCE,
                "history_length": 3,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    student_depth: StudentDepthMultiCamCfg = StudentDepthMultiCamCfg()
