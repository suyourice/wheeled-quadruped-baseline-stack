# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Go2-W obstacle avoidance environment.

Extends the baseline flat-terrain environment with:
  - Static box obstacles (kinematic rigid bodies) randomized per reset
  - A 2D LiDAR sensor (horizontal ray-cast) for obstacle detection
  - Collision penalty reward for obstacle contact
  - Teacher/Student/Distillation observation groups

Active training flow:
  1. Initialize the frozen fast-flat LLC from a locomotion checkpoint.
  2. Sample explicit start-goal local-navigation tasks with varied obstacle
     encounters between the start and the goal.
  3. Use a rule-based geometric steering teacher over (goal + privileged
     obstacle positions) to generate obstacle-aware local target poses online.
  4. Distill those target poses into a LiDAR student:
     train.py --task Navigation-Distill-Go2w-v0 --locomotion_checkpoint <flat_ckpt>
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .go2w_env_cfg import EventCfg, Go2wEnvCfg, Go2wSceneCfg, TerminationsCfg

# -- Constants ----------------------------------------------------------------
#
# Obstacle-count terminology:
#   physical slots   = rigid obstacle prims instantiated in the scene
#   active obstacles = subset of slots placed near the robot at reset
#   observed slots   = closest obstacles exposed to teacher privileged obs
#
# For distillation, these three counts matter separately:
#   1. teacher observation dim must stay stable
#   2. training physics load must stay manageable
#   3. play/eval can have extra obstacle capacity without changing obs dim

# Teacher/distillation observation capacity (K closest obstacles -> K * 2 dims)
TEACHER_OBS_OBSTACLE_SLOTS = 15

# Active train scene capacity
TRAIN_PHYSICAL_OBSTACLE_SLOTS = 15

# Play/eval scene capacity and default active clutter count
PLAY_PHYSICAL_OBSTACLE_SLOTS = 64
PLAY_DEFAULT_ACTIVE_OBSTACLES = 5

# Local aliases used throughout this file.
NUM_OBSTACLES = TRAIN_PHYSICAL_OBSTACLE_SLOTS
PRIVILEGED_OBSTACLE_SLOTS = TEACHER_OBS_OBSTACLE_SLOTS
PLAY_MAX_OBSTACLES = PLAY_PHYSICAL_OBSTACLE_SLOTS
PLAY_NUM_OBSTACLES = PLAY_DEFAULT_ACTIVE_OBSTACLES

OBSTACLE_SIZE = (0.3, 0.3, 0.5)  # (x, y, z) meters
OBSTACLE_SPAWN_RANGE = {"x": (-3.5, 3.5), "y": (-2.5, 2.5)}
OBSTACLE_NAMES = [f"obstacle_{i}" for i in range(TRAIN_PHYSICAL_OBSTACLE_SLOTS)]
PLAY_OBSTACLE_NAMES = [f"obstacle_{i}" for i in range(PLAY_PHYSICAL_OBSTACLE_SLOTS)]
CURRICULUM_STEPS_PER_ITERATION = 128  # Must match ObstacleTeacherRunnerCfg.num_steps_per_env.
CURRICULUM_OBSTACLE_START_ITERATION = 1700  # start obstacles after full-speed locomotion has stabilized
CURRICULUM_OBSTACLE_WARMUP_ITERATIONS = 1000  # reach full density at iteration 2700
CURRICULUM_CLEARANCE_WARMUP_ITERATIONS = 400  # reach full path-clearance weight at iteration 2100
CURRICULUM_COLLISION_WARMUP_ITERATIONS = 600  # ramp collision 0 -> -40 by iteration 2300
CURRICULUM_SPEED_START_ITERATION = 0          # ramp speed before obstacles appear
CURRICULUM_SPEED_WARMUP_ITERATIONS = 800      # reach ±2.0 m/s by iteration 800
OBSTACLE_MIN_SPAWN_DISTANCE_INITIAL = 2.2
OBSTACLE_MIN_SPAWN_DISTANCE_FROM_ROBOT = 1.2
OBSTACLE_WHEEL_VEL_SCALE = 28.0  # matches fast-flat transfer action scale
OBSTACLE_LIN_VEL_X = (-2.0, 2.0)
OBSTACLE_LIN_VEL_Y = (-2.0, 2.0)
OBSTACLE_ANG_VEL_Z = (-2.0, 2.0)
OBSTACLE_PATH_CLEARANCE_LENGTH = 1.6
OBSTACLE_PATH_CLEARANCE_WIDTH = 0.55
OBSTACLE_PATH_CLEARANCE_WEIGHT = -2.0
OBSTACLE_COLLISION_WEIGHT = -40.0
DISTILL_OBSTACLE_COLLISION_WEIGHT = -15.0
DISTILL_ACTIVE_OBSTACLES = 5
DISTILL_EMPTY_ENV_FRACTION = 0.05
NAV_GOAL_FORWARD_RANGE = (2.5, 4.5)
NAV_GOAL_LATERAL_RANGE = (-1.5, 1.5)
NAV_GOAL_HEADING_JITTER_RANGE = (-0.35, 0.35)
NAV_MIN_GOAL_DISTANCE = 2.0
NAV_START_EXCLUSION_RADIUS = 1.0
NAV_GOAL_EXCLUSION_RADIUS = 0.9
NAV_HEAD_ON_PROGRESS_RANGE = (0.2, 0.85)
NAV_HEAD_ON_LATERAL_RANGE = (-0.25, 0.25)
NAV_EDGE_PROGRESS_RANGE = (0.25, 0.8)
NAV_EDGE_LATERAL_RANGE = (0.55, 1.1)
NAV_DIAGONAL_PROGRESS_RANGE = (0.15, 0.7)
NAV_DIAGONAL_LATERAL_RANGE = (0.8, 1.6)
NAV_OFFPATH_PROGRESS_RANGE = (0.3, 0.9)
NAV_OFFPATH_LATERAL_RANGE = (1.3, 2.2)
NAV_NARROW_GAP_PROGRESS_RANGE = (0.35, 0.75)
NAV_NARROW_GAP_CENTER_LATERAL_RANGE = (-0.15, 0.15)
NAV_NARROW_GAP_HALF_WIDTH_RANGE = (0.45, 0.65)
NAV_NARROW_GAP_PROBABILITY = 0.25
NAV_GOAL_DISTANCE_STD = 1.2
NAV_GOAL_HEADING_STD = 0.8
NAV_GOAL_SUCCESS_POSITION_THRESHOLD = 0.35
NAV_GOAL_SUCCESS_HEADING_THRESHOLD = 0.6
NAV_WAYPOINT_LOOKAHEAD_DISTANCE = 1.25
NAV_WAYPOINT_GOAL_SNAP_DISTANCE = 1.0
NAV_WAYPOINT_USE_LIDAR_REFINEMENT = True
NAV_WAYPOINT_REFINEMENT_OFFSETS = (0.0, 0.45, -0.45, 0.70, -0.70)
NAV_LOCAL_PLANNER_ACTIVATION_THRESHOLD = 0.22
NAV_LOCAL_PLANNER_LATERAL_PENALTY = 0.16
NAV_LOCAL_PLANNER_MIN_IMPROVEMENT = 0.07
NAV_LOCAL_PLANNER_MAX_BLEND = 0.65
NAV_WAYPOINT_COMMAND_MIN_FORWARD = 0.45
NAV_WAYPOINT_COMMAND_MAX_LATERAL = 0.85
NAV_WAYPOINT_COMMAND_MAX_HEADING = 0.90

# Unitree L2 reference spec: 360 x 96 deg FoV, 0.05 m near blind spot,
# 30 m max range at high reflectivity, and 64k effective points/s. The
# training ray-caster intentionally uses a lightweight subset for runtime.
LIDAR_MAX_DISTANCE = 20.0  # meters; train below the hardware max range
LIDAR_HORIZONTAL_FOV = (0.0, 360.0)  # full 360 degrees
LIDAR_HORIZONTAL_RES = 2.0  # 2 deg resolution -> 180 rays
LIDAR_CHANNELS = 5  # lightweight subset of the 96-degree vertical FoV
LIDAR_VERTICAL_FOV = (-30.0, 6.0)  # stays within the L2 vertical range and keeps useful box-obstacle bands


def _local_waypoint_params(*, include_command_shape: bool = True) -> dict:
    """Build local waypoint params without sharing mutable config objects."""
    params = {
        "sensor_cfg": SceneEntityCfg("lidar"),
        "lookahead_distance": NAV_WAYPOINT_LOOKAHEAD_DISTANCE,
        "goal_snap_distance": NAV_WAYPOINT_GOAL_SNAP_DISTANCE,
        "use_lidar_refinement": NAV_WAYPOINT_USE_LIDAR_REFINEMENT,
        "lidar_max_distance": LIDAR_MAX_DISTANCE,
        "local_planner_candidate_offsets": NAV_WAYPOINT_REFINEMENT_OFFSETS,
        "local_planner_activation_threshold": NAV_LOCAL_PLANNER_ACTIVATION_THRESHOLD,
        "local_planner_lateral_penalty": NAV_LOCAL_PLANNER_LATERAL_PENALTY,
        "local_planner_min_improvement": NAV_LOCAL_PLANNER_MIN_IMPROVEMENT,
        "local_planner_max_blend": NAV_LOCAL_PLANNER_MAX_BLEND,
    }
    if include_command_shape:
        params.update(
            {
                "command_min_forward": NAV_WAYPOINT_COMMAND_MIN_FORWARD,
                "command_max_lateral": NAV_WAYPOINT_COMMAND_MAX_LATERAL,
                "command_max_heading": NAV_WAYPOINT_COMMAND_MAX_HEADING,
            }
        )
    return params


# =============================================================================
# Actions
# =============================================================================


@configclass
class ObstacleActionsCfg:
    """Obstacle env hybrid action space: wheel velocity + hip + stance.

    Matches fast-flat action scaling for checkpoint transfer while keeping the
    same 4+4+8 action split used by the obstacle tasks.
    Total action dim = 4 (wheel) + 4 (hip) + 8 (stance) = 16.
    """

    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[".*_foot_joint"],
        scale=OBSTACLE_WHEEL_VEL_SCALE,
    )
    hip_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hip_joint"],
        scale=0.35,
        use_default_offset=True,
    )
    stance_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_thigh_joint", ".*_calf_joint"],
        scale=0.35,
        use_default_offset=True,
    )


# =============================================================================
# Scene
# =============================================================================


def _make_obstacle_cfg(name: str, idx: int) -> RigidObjectCfg:
    """Create a kinematic box obstacle config."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=OBSTACLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2),
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.5 + idx * 0.5, 0.0, OBSTACLE_SIZE[2] / 2),
        ),
    )


@configclass
class ObstacleSceneCfg(Go2wSceneCfg):
    """Scene with Go2-W robot, flat ground, and static box obstacles."""

    # Obstacles must exist as independent prims in each environment so their
    # transforms/collisions can diverge after reset randomization.
    replicate_physics: bool = False  # randomized locations for each env

    # Static box obstacles
    for i in range(NUM_OBSTACLES):
        vars()[f"obstacle_{i}"] = _make_obstacle_cfg(f"obstacle_{i}", i)
    del i

    # Contact sensor on all obstacles — used only for reward, not observation.
    # Detects any robot body part touching an obstacle (wheels, legs, base, etc.)
    # regardless of contact direction, so ground-contact/obstacle-contact
    # ambiguity is avoided entirely.
    obstacle_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/obstacle_.*",
        history_length=3,
        track_air_time=False,
    )

    # Lightweight L2-like LiDAR subset. The real L2 supports 360 x 96 degrees,
    # but this task samples only a few vertical rings for box-obstacle navigation.
    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.28945, 0.0, -0.046825)),
        ray_alignment="yaw",  # rays track yaw only, not roll/pitch
        pattern_cfg=patterns.LidarPatternCfg(
            channels=LIDAR_CHANNELS,
            vertical_fov_range=LIDAR_VERTICAL_FOV,
            horizontal_fov_range=LIDAR_HORIZONTAL_FOV,
            horizontal_res=LIDAR_HORIZONTAL_RES,
        ),
        max_distance=LIDAR_MAX_DISTANCE,
        mesh_prim_paths=[
            "/World/ground",
            # Track obstacle transforms so ray-cast updates with randomized positions
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/obstacle_.*",
                track_mesh_transforms=True,
                is_shared=True,  # all envs use same mesh shape
            ),
        ],
        debug_vis=False,
    )


@configclass
class ObstaclePlaySceneCfg(ObstacleSceneCfg):
    """Play scene with extra obstacle slots for dense-clutter visual testing.

    The trained teacher observation still returns PRIVILEGED_OBSTACLE_SLOTS
    closest obstacles, so checkpoint input dimensions stay unchanged.
    """

    for i in range(NUM_OBSTACLES, PLAY_MAX_OBSTACLES):
        vars()[f"obstacle_{i}"] = _make_obstacle_cfg(f"obstacle_{i}", i)
    del i


# =============================================================================
# Events (randomisation)
# =============================================================================


@configclass
class ObstacleEventCfg(EventCfg):
    """Shared obstacle-environment events.

    These are the legacy obstacle randomizers and command curricula. The active
    local-navigation distillation task overrides the obstacle reset event and
    disables the speed curriculum in its `__post_init__`.
    """

    speed_curriculum: EventTerm | None = EventTerm(
        func=mdp.update_locomotion_curriculum,
        mode="reset",
        params={
            "start_iteration": CURRICULUM_SPEED_START_ITERATION,
            "warmup_iterations": CURRICULUM_SPEED_WARMUP_ITERATIONS,
            "steps_per_iteration": CURRICULUM_STEPS_PER_ITERATION,
            "command_name": "base_velocity",
            "lin_vel_x_initial": (-1.0, 1.0),
            "lin_vel_x_final": OBSTACLE_LIN_VEL_X,
            "lin_vel_y_initial": (-0.3, 0.3),
            "lin_vel_y_final": OBSTACLE_LIN_VEL_Y,
            "ang_vel_z_initial": (-0.3, 0.3),
            "ang_vel_z_final": OBSTACLE_ANG_VEL_Z,
            "min_survival_steps": 800,
        },
    )

    reset_obstacles = EventTerm(
        func=mdp.reset_obstacles_curriculum,
        mode="reset",
        params={
            "obstacle_names": OBSTACLE_NAMES,
            "start_iteration": CURRICULUM_OBSTACLE_START_ITERATION,
            "warmup_iterations": CURRICULUM_OBSTACLE_WARMUP_ITERATIONS,
            "steps_per_iteration": CURRICULUM_STEPS_PER_ITERATION,
            "min_obstacles": 5,
            "spawn_range_x": OBSTACLE_SPAWN_RANGE["x"],
            "spawn_range_y": OBSTACLE_SPAWN_RANGE["y"],
            "obstacle_z": OBSTACLE_SIZE[2] / 2,
            "min_spawn_distance_from_robot": OBSTACLE_MIN_SPAWN_DISTANCE_FROM_ROBOT,
            "min_spawn_distance_from_robot_initial": OBSTACLE_MIN_SPAWN_DISTANCE_INITIAL,
            "min_inter_obstacle_dist": 0.8,
            "min_survival_steps": 800,
        },
    )


# =============================================================================
# Rewards
# =============================================================================


@configclass
class ObstacleRewardsCfg:
    """Shared obstacle-task rewards.

    The active local-navigation task keeps the base stability penalties and
    sparse obstacle-collision penalty, then overrides command-tracking and
    goal-related terms in its `__post_init__`.
    """

    # -- Velocity tracking -----------------------------------------------------
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": 0.35},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    # -- Stability penalties ---------------------------------------------------
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.5)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # -- Height maintenance (prevents crouching) --------------------------------
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-3.0,
        params={"target_height": 0.45, "robot_cfg": SceneEntityCfg("robot")},
    )

    # -- Posture (strong from day 1 — no curriculum ramp needed) ---------------
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
    straight_hip_deviation = RewTerm(
        func=mdp.joint_deviation_l1_command_gated,
        weight=-0.30,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
            "min_abs_lin_x": 0.2,
            "max_abs_lin_y": 0.15,
            "max_abs_ang_z": 0.2,
        },
    )
    stand_joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1_command_gated,
        weight=-0.35,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]),
            "max_abs_lin_x": 0.1,
            "max_abs_lin_y": 0.1,
            "max_abs_ang_z": 0.1,
        },
    )

    # -- Wheel contact / spin --------------------------------------------------
    wheel_contact = RewTerm(
        func=mdp.wheel_contact_penalty,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"])},
    )
    wheel_vel_zero_cmd = RewTerm(
        func=mdp.wheel_vel_zero_cmd,
        weight=-0.01,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_foot_joint"]),
        },
    )

    # -- Action smoothness -----------------------------------------------------
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # -- Contact penalty -------------------------------------------------------
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
            "threshold": 1.0,
        },
    )

    # -- Termination penalty ---------------------------------------------------
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # -- Obstacle commanded-path clearance penalty (dense) ---------------------
    obstacle_path_clearance = RewTerm(
        func=mdp.obstacle_path_clearance_penalty,
        weight=OBSTACLE_PATH_CLEARANCE_WEIGHT,
        params={
            "obstacle_names": OBSTACLE_NAMES,
            "command_name": "base_velocity",
            "path_length": OBSTACLE_PATH_CLEARANCE_LENGTH,
            "path_width": OBSTACLE_PATH_CLEARANCE_WIDTH,
            "start_iteration": CURRICULUM_OBSTACLE_START_ITERATION,
            "warmup_iterations": CURRICULUM_CLEARANCE_WARMUP_ITERATIONS,
            "steps_per_iteration": CURRICULUM_STEPS_PER_ITERATION,
        },
    )
    obstacle_lateral_avoidance = RewTerm(
        func=mdp.obstacle_lateral_avoidance_reward,
        weight=0.0,
        params={
            "obstacle_names": OBSTACLE_NAMES,
            "command_name": "base_velocity",
            "path_length": OBSTACLE_PATH_CLEARANCE_LENGTH,
            "path_width": OBSTACLE_PATH_CLEARANCE_WIDTH,
            "start_iteration": CURRICULUM_OBSTACLE_START_ITERATION,
            "warmup_iterations": CURRICULUM_CLEARANCE_WARMUP_ITERATIONS,
            "steps_per_iteration": CURRICULUM_STEPS_PER_ITERATION,
        },
    )
    obstacle_yaw_avoidance = RewTerm(
        func=mdp.obstacle_yaw_avoidance_reward,
        weight=0.0,
        params={
            "obstacle_names": OBSTACLE_NAMES,
            "command_name": "base_velocity",
            "path_length": OBSTACLE_PATH_CLEARANCE_LENGTH,
            "path_width": OBSTACLE_PATH_CLEARANCE_WIDTH,
            "start_iteration": CURRICULUM_OBSTACLE_START_ITERATION,
            "warmup_iterations": CURRICULUM_CLEARANCE_WARMUP_ITERATIONS,
            "steps_per_iteration": CURRICULUM_STEPS_PER_ITERATION,
        },
    )

    # -- Obstacle collision penalty (sparse) -----------------------------------
    obstacle_collision = RewTerm(
        func=mdp.obstacle_contact_penalty,
        weight=OBSTACLE_COLLISION_WEIGHT,
        params={
            "sensor_cfg": SceneEntityCfg("obstacle_contacts"),
            "threshold": 1.0,
            "start_iteration": CURRICULUM_OBSTACLE_START_ITERATION,
            "warmup_iterations": CURRICULUM_COLLISION_WARMUP_ITERATIONS,
            "steps_per_iteration": CURRICULUM_STEPS_PER_ITERATION,
        },
    )


# =============================================================================
# Observations
# =============================================================================


@configclass
class ObstacleDistillObsCfg:
    """Distillation observations: student (LiDAR) and teacher (privileged) groups."""

    @configclass
    class StudentCfg(ObsGroup):
        """Student obs = proprioception + rolling local waypoint + raw LiDAR + steering summaries."""

        # -- Proprioception --
        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel,      noise=Unoise(n_min=-0.1,  n_max=0.1))
        base_ang_vel      = ObsTerm(func=mdp.base_ang_vel,      noise=Unoise(n_min=-0.2,  n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity,  noise=Unoise(n_min=-0.05, n_max=0.05))

        goal_command = ObsTerm(
            func=mdp.local_goal_command_b,
            params=_local_waypoint_params(),
        )

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5,  n_max=1.5))

        actions = ObsTerm(func=mdp.last_action)

        # -- LiDAR scan (5 rings x 180 rays at 2° resolution = 900 distances) --
        lidar_scan = ObsTerm(
            func=mdp.lidar_distances,
            params={
                "sensor_cfg": SceneEntityCfg("lidar"),
                "max_distance": LIDAR_MAX_DISTANCE,
            },
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )

        # Compact LiDAR-derived geometry cues for steering relative to the
        # current local-goal direction.
        # Output dims:
        #   5 sector closeness scores  = [front-left, front-center, front-right, left, right]
        #   3 front-sector close ratios = [front-left, front-center, front-right]
        #   3 command-aligned corridor blockage scores = [center, left, right]
        #   2 command-aligned detour openness scores = [left, right]
        #   1 command-aligned weighted lateral bias
        lidar_steering = ObsTerm(
            func=mdp.lidar_steering_features,
            params={
                "sensor_cfg": SceneEntityCfg("lidar"),
                "command_name": "local_goal",
                "max_distance": LIDAR_MAX_DISTANCE,
                "close_distance": 2.5,
                "min_hit_height": -0.15,
            },
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class TeacherCfg(ObsGroup):
        """Teacher obs = proprioception + rolling local waypoint + privileged obstacle positions."""

        # -- Proprioception --
        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel,      noise=Unoise(n_min=-0.1,  n_max=0.1))
        base_ang_vel      = ObsTerm(func=mdp.base_ang_vel,      noise=Unoise(n_min=-0.2,  n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity,  noise=Unoise(n_min=-0.05, n_max=0.05))

        goal_command = ObsTerm(
            func=mdp.local_goal_command_b,
            params=_local_waypoint_params(),
        )

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5,  n_max=1.5))

        actions = ObsTerm(func=mdp.last_action)

        # -- Privileged --
        obstacle_positions = ObsTerm(
            func=mdp.obstacle_positions_rel,
            params={
                "obstacle_names": OBSTACLE_NAMES,
                "max_distance": 8.0,
                "normalize": True,
                "num_closest": PRIVILEGED_OBSTACLE_SLOTS,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False  # Teacher sees clean data
            self.concatenate_terms = True

    student: StudentCfg = StudentCfg()
    teacher: TeacherCfg = TeacherCfg()

    @configclass
    class DebugCfg(ObsGroup):
        """Debug-only observations kept out of the student/teacher policy inputs."""

        root_position_w = ObsTerm(func=mdp.root_position_w)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        goal_command = ObsTerm(
            func=mdp.local_goal_command_b,
            params=_local_waypoint_params(),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        start_position_w = ObsTerm(func=mdp.start_position_w)
        waypoint_position_w = ObsTerm(
            func=mdp.waypoint_position_w,
            params=_local_waypoint_params(include_command_shape=False),
        )
        goal_position_w = ObsTerm(func=mdp.goal_position_w)
        scenario_template_code = ObsTerm(func=mdp.navigation_scenario_code)
        obstacle_positions = ObsTerm(
            func=mdp.obstacle_positions_rel,
            params={
                "obstacle_names": OBSTACLE_NAMES,
                "max_distance": 8.0,
                "normalize": False,
                "num_closest": PRIVILEGED_OBSTACLE_SLOTS,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    debug: DebugCfg = DebugCfg()


# =============================================================================
# Terminations
# =============================================================================


@configclass
class ObstacleTerminationsCfg(TerminationsCfg):
    """Shared obstacle-task terminations.

    The active local-navigation task disables `obstacle_contact` and adds
    `goal_reached` termination in its `__post_init__`.
    """

    obstacle_contact = DoneTerm(
        func=mdp.obstacle_contact_termination,
        params={
            "sensor_cfg": SceneEntityCfg("obstacle_contacts"),
            "threshold": 1.0,
            "start_iteration": 0,
            "steps_per_iteration": CURRICULUM_STEPS_PER_ITERATION,
        },
    )


# =============================================================================
# Active local-navigation environment configs
# =============================================================================


@configclass
class Go2wNavigationDistillEnvCfg(Go2wEnvCfg):
    """Static-box local-navigation distillation env.

    This stage no longer treats obstacle avoidance as "follow a velocity command
    while avoiding boxes". Instead, every episode samples:

      start pose -> explicit local goal pose -> varied obstacle field

    The active student predicts a short-horizon local target pose from
    (LiDAR + proprio + goal), converts that target into `(vx, vy, yaw)`, and
    sends the result to the frozen LLC. Distillation remains online:

      teacher(goal + privileged obstacles) -> target pose
      student(goal + LiDAR)                -> target pose
    """

    scene: ObstacleSceneCfg = ObstacleSceneCfg(num_envs=8192, env_spacing=8.0)
    actions: ObstacleActionsCfg = ObstacleActionsCfg()
    observations: ObstacleDistillObsCfg = ObstacleDistillObsCfg()
    rewards: ObstacleRewardsCfg = ObstacleRewardsCfg()
    events: ObstacleEventCfg = ObstacleEventCfg()
    terminations: ObstacleTerminationsCfg = ObstacleTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        if self.scene.lidar is not None:
            self.scene.lidar.update_period = self.decimation * self.sim.dt

        self.events.speed_curriculum = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0   # 100%
        self.commands.base_velocity.debug_vis = False

        self.events.reset_obstacles.func = mdp.reset_navigation_goals_and_obstacles
        self.events.reset_obstacles.params = {
            "obstacle_names": OBSTACLE_NAMES,
            "min_obstacles": DISTILL_ACTIVE_OBSTACLES,
            "max_obstacles": DISTILL_ACTIVE_OBSTACLES,
            "empty_env_fraction": DISTILL_EMPTY_ENV_FRACTION,
            "spawn_range_x": OBSTACLE_SPAWN_RANGE["x"],
            "spawn_range_y": OBSTACLE_SPAWN_RANGE["y"],
            "obstacle_z": OBSTACLE_SIZE[2] / 2,
            "min_inter_obstacle_dist": 0.7,
            "goal_forward_range": NAV_GOAL_FORWARD_RANGE,
            "goal_lateral_range": NAV_GOAL_LATERAL_RANGE,
            "goal_heading_jitter_range": NAV_GOAL_HEADING_JITTER_RANGE,
            "min_goal_distance": NAV_MIN_GOAL_DISTANCE,
            "start_exclusion_radius": NAV_START_EXCLUSION_RADIUS,
            "goal_exclusion_radius": NAV_GOAL_EXCLUSION_RADIUS,
            "head_on_progress_range": NAV_HEAD_ON_PROGRESS_RANGE,
            "head_on_lateral_range": NAV_HEAD_ON_LATERAL_RANGE,
            "edge_progress_range": NAV_EDGE_PROGRESS_RANGE,
            "edge_lateral_range": NAV_EDGE_LATERAL_RANGE,
            "diagonal_progress_range": NAV_DIAGONAL_PROGRESS_RANGE,
            "diagonal_lateral_range": NAV_DIAGONAL_LATERAL_RANGE,
            "offpath_progress_range": NAV_OFFPATH_PROGRESS_RANGE,
            "offpath_lateral_range": NAV_OFFPATH_LATERAL_RANGE,
            "narrow_gap_progress_range": NAV_NARROW_GAP_PROGRESS_RANGE,
            "narrow_gap_center_lateral_range": NAV_NARROW_GAP_CENTER_LATERAL_RANGE,
            "narrow_gap_half_width_range": NAV_NARROW_GAP_HALF_WIDTH_RANGE,
            "narrow_gap_probability": NAV_NARROW_GAP_PROBABILITY,
            "fixed_goal_forward": None,
            "fixed_goal_lateral": None,
            "fixed_goal_heading_jitter": None,
            "fixed_scenario_template": None,
        }

        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.straight_hip_deviation = None
        self.rewards.stand_joint_deviation = None
        self.rewards.wheel_vel_zero_cmd = None
        self.rewards.termination_penalty = None
        self.rewards.obstacle_path_clearance = None
        self.rewards.obstacle_lateral_avoidance = None
        self.rewards.obstacle_yaw_avoidance = None
        self.rewards.obstacle_collision.weight = DISTILL_OBSTACLE_COLLISION_WEIGHT
        self.rewards.obstacle_collision.params["start_iteration"] = 0
        self.rewards.obstacle_collision.params["warmup_iterations"] = 0
        self.rewards.goal_distance = RewTerm(
            func=mdp.goal_distance_tanh_reward,
            weight=6.0,
            params={"std": NAV_GOAL_DISTANCE_STD},
        )
        self.rewards.goal_heading = RewTerm(
            func=mdp.goal_heading_tanh_reward,
            weight=1.0,
            params={"std": NAV_GOAL_HEADING_STD},
        )
        self.rewards.goal_reached = RewTerm(
            func=mdp.goal_reached_bonus,
            weight=15.0,
            params={
                "position_threshold": NAV_GOAL_SUCCESS_POSITION_THRESHOLD,
                "heading_threshold": NAV_GOAL_SUCCESS_HEADING_THRESHOLD,
            },
        )
        self.terminations.obstacle_contact = None
        self.terminations.goal_reached = DoneTerm(
            func=mdp.goal_reached_termination,
            params={
                "position_threshold": NAV_GOAL_SUCCESS_POSITION_THRESHOLD,
                "heading_threshold": NAV_GOAL_SUCCESS_HEADING_THRESHOLD,
            },
        )


@configclass
class Go2wNavigationDistillEnvCfg_PLAY(Go2wNavigationDistillEnvCfg):
    """Play/eval env for the local-navigation distillation stage."""

    scene: ObstaclePlaySceneCfg = ObstaclePlaySceneCfg(num_envs=16, env_spacing=5.0)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 5.0
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.observations.student.enable_corruption = False
        self.observations.teacher.obstacle_positions.params["obstacle_names"] = PLAY_OBSTACLE_NAMES
        self.observations.teacher.obstacle_positions.params["num_closest"] = PRIVILEGED_OBSTACLE_SLOTS
        self.observations.debug.obstacle_positions.params["obstacle_names"] = PLAY_OBSTACLE_NAMES
        self.observations.debug.obstacle_positions.params["num_closest"] = PRIVILEGED_OBSTACLE_SLOTS

        # The original velocity command is disabled for the navigation-distillation task.
        # The active command sent to the frozen LLC is generated online from the local
        # goal by the teacher/student navigation policy, not sampled by the command manager.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.debug_vis = False

        self.events.reset_obstacles.func = mdp.reset_navigation_goals_and_obstacles
        self.events.reset_obstacles.params = {
            "obstacle_names": PLAY_OBSTACLE_NAMES,
            "min_obstacles": PLAY_NUM_OBSTACLES,
            "max_obstacles": PLAY_NUM_OBSTACLES,
            "empty_env_fraction": 0.0,
            "spawn_range_x": OBSTACLE_SPAWN_RANGE["x"],
            "spawn_range_y": OBSTACLE_SPAWN_RANGE["y"],
            "obstacle_z": OBSTACLE_SIZE[2] / 2,
            "min_inter_obstacle_dist": 0.45,
            "goal_forward_range": NAV_GOAL_FORWARD_RANGE,
            "goal_lateral_range": NAV_GOAL_LATERAL_RANGE,
            "goal_heading_jitter_range": NAV_GOAL_HEADING_JITTER_RANGE,
            "min_goal_distance": NAV_MIN_GOAL_DISTANCE,
            "start_exclusion_radius": NAV_START_EXCLUSION_RADIUS,
            "goal_exclusion_radius": NAV_GOAL_EXCLUSION_RADIUS,
            "head_on_progress_range": NAV_HEAD_ON_PROGRESS_RANGE,
            "head_on_lateral_range": NAV_HEAD_ON_LATERAL_RANGE,
            "edge_progress_range": NAV_EDGE_PROGRESS_RANGE,
            "edge_lateral_range": NAV_EDGE_LATERAL_RANGE,
            "diagonal_progress_range": NAV_DIAGONAL_PROGRESS_RANGE,
            "diagonal_lateral_range": NAV_DIAGONAL_LATERAL_RANGE,
            "offpath_progress_range": NAV_OFFPATH_PROGRESS_RANGE,
            "offpath_lateral_range": NAV_OFFPATH_LATERAL_RANGE,
            "narrow_gap_progress_range": NAV_NARROW_GAP_PROGRESS_RANGE,
            "narrow_gap_center_lateral_range": NAV_NARROW_GAP_CENTER_LATERAL_RANGE,
            "narrow_gap_half_width_range": NAV_NARROW_GAP_HALF_WIDTH_RANGE,
            "narrow_gap_probability": NAV_NARROW_GAP_PROBABILITY,
            "fixed_goal_forward": None,
            "fixed_goal_lateral": None,
            "fixed_goal_heading_jitter": None,
            "fixed_scenario_template": None,
        }

        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.straight_hip_deviation = None
        self.rewards.stand_joint_deviation = None
        self.rewards.wheel_vel_zero_cmd = None
        self.rewards.termination_penalty = None
        self.rewards.obstacle_path_clearance = None
        self.rewards.obstacle_lateral_avoidance = None
        self.rewards.obstacle_yaw_avoidance = None
        self.rewards.obstacle_collision.weight = DISTILL_OBSTACLE_COLLISION_WEIGHT
        self.rewards.obstacle_collision.params["start_iteration"] = 0
        self.rewards.obstacle_collision.params["warmup_iterations"] = 0
        self.rewards.goal_distance = RewTerm(
            func=mdp.goal_distance_tanh_reward,
            weight=6.0,
            params={"std": NAV_GOAL_DISTANCE_STD},
        )
        self.rewards.goal_heading = RewTerm(
            func=mdp.goal_heading_tanh_reward,
            weight=1.0,
            params={"std": NAV_GOAL_HEADING_STD},
        )
        self.rewards.goal_reached = RewTerm(
            func=mdp.goal_reached_bonus,
            weight=15.0,
            params={
                "position_threshold": NAV_GOAL_SUCCESS_POSITION_THRESHOLD,
                "heading_threshold": NAV_GOAL_SUCCESS_HEADING_THRESHOLD,
            },
        )

        # Keep obstacle contact as a penalty instead of an episode termination.
        # This allows the policy to experience and recover from minor contacts, while
        # collision frequency can still be monitored through rewards/eval metrics.
        self.terminations.obstacle_contact = None

        self.terminations.goal_reached = DoneTerm(
            func=mdp.goal_reached_termination,
            params={
                "position_threshold": NAV_GOAL_SUCCESS_POSITION_THRESHOLD,
                "heading_threshold": NAV_GOAL_SUCCESS_HEADING_THRESHOLD,
            },
        )
