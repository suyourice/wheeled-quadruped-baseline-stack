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

Training flow:
  1. Initialize the frozen fast-flat LLC from a locomotion checkpoint.
  2. Use a rule-based geometric steering teacher over privileged obstacle
     positions to generate final teacher actions online.
  3. Distill those teacher actions into a LiDAR student:
     train.py --task Obstacle-Distill-Go2w-v0 --locomotion_checkpoint <flat_ckpt>
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

# Standard obstacle-train scene capacity
TRAIN_PHYSICAL_OBSTACLE_SLOTS = 15

# Fast teacher-train scene capacity and active clutter count
FAST_TRAIN_PHYSICAL_OBSTACLE_SLOTS = 8
FAST_TRAIN_ACTIVE_OBSTACLES = 3

# Play/eval scene capacity and default active clutter count
PLAY_PHYSICAL_OBSTACLE_SLOTS = 64
PLAY_DEFAULT_ACTIVE_OBSTACLES = 5

# Backward-compatible aliases kept to avoid broad refactors while distillation
# is in progress. Prefer the role-based names above in new code/comments.
NUM_OBSTACLES = TRAIN_PHYSICAL_OBSTACLE_SLOTS
PRIVILEGED_OBSTACLE_SLOTS = TEACHER_OBS_OBSTACLE_SLOTS
NUM_OBSTACLES_FAST = FAST_TRAIN_PHYSICAL_OBSTACLE_SLOTS
FAST_ACTIVE_OBSTACLES = FAST_TRAIN_ACTIVE_OBSTACLES
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
FAST_COMMAND_PATH_OBSTACLES = 2  # Put the first slots directly in the commanded path.
FAST_COMMAND_PATH_FORWARD_RANGE = (1.6, 2.4)
FAST_COMMAND_PATH_LATERAL_RANGE = (-0.35, 0.35)
FAST_NEAR_FIELD_OBSTACLES = 0  # Avoid random near-field contacts; remaining active slots are random clutter.
FAST_NEAR_FIELD_RADIUS_RANGE = (1.3, 1.9)
FAST_PATH_CLEARANCE_LENGTH = 2.6
FAST_PATH_CLEARANCE_WIDTH = 0.75
FAST_PATH_CLEARANCE_WEIGHT = -7.0
FAST_PATH_CLEARANCE_SCORE_POWER = 1.0
FAST_PATH_CLEARANCE_AGGREGATION = "sum_clamped"
FAST_PATH_CLEARANCE_SUM_CLIP = 1.5
FAST_OBSTACLE_LATERAL_AVOIDANCE_WEIGHT = 4.5
FAST_OBSTACLE_YAW_AVOIDANCE_WEIGHT = 1.0
FAST_AVOIDANCE_RISK_CLIP = 1.5
FAST_AVOID_TARGET_LATERAL_SPEED = 0.35
FAST_AVOID_TARGET_YAW_RATE = 0.8
FAST_AVOID_CENTER_DEADBAND = 0.05
FAST_AVOID_MIN_PROGRESS = 0.25
FAST_OBSTACLE_COLLISION_WEIGHT = -100.0
FAST_OBSTACLE_TERMINATION_START_ITERATION = 300
FAST_OBSTACLE_WARMUP_ITERATIONS = 0
FAST_CLEARANCE_WARMUP_ITERATIONS = 0
FAST_COLLISION_WARMUP_ITERATIONS = 0
FAST_EMPTY_ENV_FRACTION = 0.35  # Keep no-obstacle rollouts in the PPO batch to preserve flat locomotion.
FAST_OBSTACLE_NAMES = [f"obstacle_{i}" for i in range(FAST_TRAIN_PHYSICAL_OBSTACLE_SLOTS)]

LIDAR_MAX_DISTANCE = 20.0  # meters
LIDAR_HORIZONTAL_FOV = (0.0, 360.0)  # full 360 degrees
LIDAR_HORIZONTAL_RES = 2.0  # 2 deg resolution -> 180 rays
LIDAR_CHANNELS = 1  # single horizontal ring (2D LiDAR)
LIDAR_VERTICAL_FOV = (0.0, 0.0)  # single horizontal plane


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
    replicate_physics: bool = False

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

    # 2D LiDAR sensor (horizontal ray-cast from robot base)
    # Uses MultiMeshRayCaster to ray-cast against ground + dynamic obstacle meshes.
    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, -0.2)),  # below base
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


@configclass
class ObstacleSceneFastCfg(ObstacleSceneCfg):
    """Lightweight scene for the Fast obstacle teacher (checkpoint-transfer variant).

    Drops the LiDAR sensor (teacher obs uses privileged positions, not LiDAR) and
    nullifies obstacle slots 8-14 (only 8 physical slots are used in Fast).
    This cuts Isaac Sim init from 8192 × (15 rigid bodies + 180 LiDAR rays) down
    to 8192 × 8 rigid bodies, which fits within the 4-hour HPC dev allocation.
    """

    lidar = None  # teacher doesn't consume LiDAR; dropping saves 8192×180 ray init

    # Nullify unused obstacle physics slots (Fast env uses 8 physical slots)
    for i in range(NUM_OBSTACLES_FAST, NUM_OBSTACLES):
        vars()[f"obstacle_{i}"] = None
    del i


# =============================================================================
# Events (randomisation)
# =============================================================================


@configclass
class ObstacleEventCfg(EventCfg):
    """Events for the obstacle environment.

    reset_obstacles: curriculum that grows active obstacle count from 5 to NUM_OBSTACLES.
    speed_curriculum: resets command range to ±1.0 m/s at obstacle start, ramps to ±2.0 m/s.
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
    """Rewards for obstacle avoidance.

    Matches the fast-flat locomotion objective, plus obstacle clearance/collision.
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
class ObstacleTeacherObsCfg:
    """Teacher observations: proprioception + privileged obstacle positions."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Teacher policy obs = proprioception + obstacle positions (privileged)."""

        # -- Proprioception (same as baseline) --
        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel,      noise=Unoise(n_min=-0.1,  n_max=0.1))
        base_ang_vel      = ObsTerm(func=mdp.base_ang_vel,      noise=Unoise(n_min=-0.2,  n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity,  noise=Unoise(n_min=-0.05, n_max=0.05))

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5,  n_max=1.5))

        actions = ObsTerm(func=mdp.last_action)

        # -- Privileged: obstacle positions relative to robot --
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
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ObstacleDistillObsCfg:
    """Distillation observations: student (LiDAR) and teacher (privileged) groups."""

    @configclass
    class StudentCfg(ObsGroup):
        """Student obs = proprioception + LiDAR distances."""

        # -- Proprioception --
        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel,      noise=Unoise(n_min=-0.1,  n_max=0.1))
        base_ang_vel      = ObsTerm(func=mdp.base_ang_vel,      noise=Unoise(n_min=-0.2,  n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity,  noise=Unoise(n_min=-0.05, n_max=0.05))

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5,  n_max=1.5))

        actions = ObsTerm(func=mdp.last_action)

        # -- LiDAR scan (180 rays at 2° resolution) --
        lidar_scan = ObsTerm(
            func=mdp.lidar_distances,
            params={
                "sensor_cfg": SceneEntityCfg("lidar"),
                "max_distance": LIDAR_MAX_DISTANCE,
            },
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class TeacherCfg(ObsGroup):
        """Teacher obs = proprioception + privileged obstacle positions."""

        # -- Proprioception --
        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel,      noise=Unoise(n_min=-0.1,  n_max=0.1))
        base_ang_vel      = ObsTerm(func=mdp.base_ang_vel,      noise=Unoise(n_min=-0.2,  n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity,  noise=Unoise(n_min=-0.05, n_max=0.05))

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

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


# =============================================================================
# Terminations
# =============================================================================


@configclass
class ObstacleTerminationsCfg(TerminationsCfg):
    """Terminate on falls and on any robot-obstacle contact."""

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
# Top-level environment configs
# =============================================================================


@configclass
class Go2wObstacleTeacherEnvCfg(Go2wEnvCfg):
    """Obstacle environment for Teacher PPO training with privileged obs."""

    scene:        ObstacleSceneCfg       = ObstacleSceneCfg(num_envs=8192, env_spacing=8.0)
    actions:      ObstacleActionsCfg     = ObstacleActionsCfg()
    observations: ObstacleTeacherObsCfg  = ObstacleTeacherObsCfg()
    rewards:      ObstacleRewardsCfg     = ObstacleRewardsCfg()
    events:       ObstacleEventCfg       = ObstacleEventCfg()
    terminations: ObstacleTerminationsCfg = ObstacleTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scene.lidar is not None:
            self.scene.lidar.update_period = self.decimation * self.sim.dt
        self.commands.base_velocity.ranges.lin_vel_x = OBSTACLE_LIN_VEL_X
        self.commands.base_velocity.ranges.lin_vel_y = OBSTACLE_LIN_VEL_Y
        self.commands.base_velocity.ranges.ang_vel_z = OBSTACLE_ANG_VEL_Z
        self.commands.base_velocity.rel_standing_envs = 0.2


@configclass
class Go2wObstacleTeacherEnvCfg_PLAY(Go2wObstacleTeacherEnvCfg):
    """Teacher evaluation environment."""

    scene: ObstaclePlaySceneCfg = ObstaclePlaySceneCfg(num_envs=16, env_spacing=5.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs    = 16
        self.scene.env_spacing = 5.0
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = OBSTACLE_LIN_VEL_X
        self.commands.base_velocity.ranges.lin_vel_y = OBSTACLE_LIN_VEL_Y
        self.commands.base_velocity.ranges.ang_vel_z = OBSTACLE_ANG_VEL_Z
        self.observations.policy.obstacle_positions.params["obstacle_names"] = PLAY_OBSTACLE_NAMES
        self.observations.policy.obstacle_positions.params["num_closest"] = PRIVILEGED_OBSTACLE_SLOTS
        self.rewards.obstacle_path_clearance.params["obstacle_names"] = PLAY_OBSTACLE_NAMES
        self.rewards.obstacle_lateral_avoidance.params["obstacle_names"] = PLAY_OBSTACLE_NAMES
        self.rewards.obstacle_yaw_avoidance.params["obstacle_names"] = PLAY_OBSTACLE_NAMES
        self.events.reset_obstacles.params["start_iteration"] = 0
        self.events.reset_obstacles.params["warmup_iterations"] = 0
        self.events.reset_obstacles.params["obstacle_names"] = PLAY_OBSTACLE_NAMES
        self.events.reset_obstacles.params["min_obstacles"] = PLAY_NUM_OBSTACLES
        self.events.reset_obstacles.params["max_obstacles"] = PLAY_NUM_OBSTACLES
        self.events.reset_obstacles.params["min_inter_obstacle_dist"] = 0.45
        self.events.reset_obstacles.params["empty_env_fraction"] = 0.0
        self.events.push_robot        = None
        self.events.add_base_mass     = None
        self.events.speed_curriculum  = None  # use fixed ±2.0 m/s during play


# =============================================================================
# Fast obstacle teacher (uses a pre-trained ±2.0 m/s flat locomotion checkpoint)
# =============================================================================


@configclass
class Go2wObstacleTeacherFastEnvCfg(Go2wObstacleTeacherEnvCfg):
    """Obstacle teacher env for use with a pre-trained ±2.0 m/s flat checkpoint.

    Uses the full fast-flat command range from the start. The obstacle setup is
    intentionally simple: 35% no-obstacle rollouts, otherwise three active boxes.
    Two boxes are sampled in the commanded path so the policy sees a consistent
    "continue command but avoid this box" problem; the remaining box is random clutter.
    Uses ObstacleSceneFastCfg (no LiDAR, 8 obstacle slots) to cut init time.
    """

    scene: ObstacleSceneFastCfg = ObstacleSceneFastCfg(num_envs=8192, env_spacing=8.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        # Disable push: fast-flat checkpoint has no push-recovery training.
        self.events.push_robot = None
        # The transferred checkpoint already learned the full ±2 m/s random command range.
        self.events.speed_curriculum = None
        # Start obstacle density, path-clearance, and collision at iteration 0 with no adaptive gate.
        self.events.reset_obstacles.params["start_iteration"] = 0
        self.events.reset_obstacles.params["warmup_iterations"] = FAST_OBSTACLE_WARMUP_ITERATIONS
        self.events.reset_obstacles.params["min_obstacles"] = FAST_ACTIVE_OBSTACLES
        self.events.reset_obstacles.params["max_obstacles"] = FAST_ACTIVE_OBSTACLES
        self.events.reset_obstacles.params["min_survival_steps"] = 0
        self.events.reset_obstacles.params["empty_env_fraction"] = FAST_EMPTY_ENV_FRACTION
        self.events.reset_obstacles.params["command_path_obstacles"] = FAST_COMMAND_PATH_OBSTACLES
        self.events.reset_obstacles.params["command_name"] = "base_velocity"
        self.events.reset_obstacles.params["command_path_forward_range"] = FAST_COMMAND_PATH_FORWARD_RANGE
        self.events.reset_obstacles.params["command_path_lateral_range"] = FAST_COMMAND_PATH_LATERAL_RANGE
        self.events.reset_obstacles.params["near_field_obstacles"] = FAST_NEAR_FIELD_OBSTACLES
        self.events.reset_obstacles.params["near_field_radius_range"] = FAST_NEAR_FIELD_RADIUS_RANGE
        self.events.reset_obstacles.params["obstacle_names"] = FAST_OBSTACLE_NAMES
        # Keep exact fast-flat locomotion tracking intact; avoidance is a separate risk-gated bonus.
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_lin_vel_xy_exp.params = {"command_name": "base_velocity", "std": 0.35}
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp
        self.rewards.track_ang_vel_z_exp.params = {"command_name": "base_velocity", "std": 0.25}
        fast_path_reward_params = {
            "start_iteration": 0,
            "warmup_iterations": FAST_CLEARANCE_WARMUP_ITERATIONS,
            "obstacle_names": FAST_OBSTACLE_NAMES,
            "path_length": FAST_PATH_CLEARANCE_LENGTH,
            "path_width": FAST_PATH_CLEARANCE_WIDTH,
            "score_power": FAST_PATH_CLEARANCE_SCORE_POWER,
        }
        self.rewards.obstacle_path_clearance.params.update(
            {
                **fast_path_reward_params,
                "aggregation": FAST_PATH_CLEARANCE_AGGREGATION,
                "sum_clip": FAST_PATH_CLEARANCE_SUM_CLIP,
            }
        )
        self.rewards.obstacle_path_clearance.weight = FAST_PATH_CLEARANCE_WEIGHT
        self.rewards.obstacle_lateral_avoidance.params.update(
            {
                **fast_path_reward_params,
                "risk_clip": FAST_AVOIDANCE_RISK_CLIP,
                "target_lateral_speed": FAST_AVOID_TARGET_LATERAL_SPEED,
                "center_deadband": FAST_AVOID_CENTER_DEADBAND,
                "min_progress": FAST_AVOID_MIN_PROGRESS,
            }
        )
        self.rewards.obstacle_lateral_avoidance.weight = FAST_OBSTACLE_LATERAL_AVOIDANCE_WEIGHT
        self.rewards.obstacle_yaw_avoidance.params.update(
            {
                **fast_path_reward_params,
                "risk_clip": FAST_AVOIDANCE_RISK_CLIP,
                "target_yaw_rate": FAST_AVOID_TARGET_YAW_RATE,
                "center_deadband": FAST_AVOID_CENTER_DEADBAND,
                "min_progress": FAST_AVOID_MIN_PROGRESS,
            }
        )
        self.rewards.obstacle_yaw_avoidance.weight = FAST_OBSTACLE_YAW_AVOIDANCE_WEIGHT
        self.rewards.obstacle_collision.params["start_iteration"] = 0
        self.rewards.obstacle_collision.params["warmup_iterations"] = FAST_COLLISION_WARMUP_ITERATIONS
        self.rewards.obstacle_collision.weight = FAST_OBSTACLE_COLLISION_WEIGHT
        self.terminations.obstacle_contact.params["start_iteration"] = FAST_OBSTACLE_TERMINATION_START_ITERATION
        self.terminations.obstacle_contact.params["steps_per_iteration"] = CURRICULUM_STEPS_PER_ITERATION
        self.observations.policy.obstacle_positions.params["obstacle_names"] = FAST_OBSTACLE_NAMES


@configclass
class Go2wObstacleTeacherFastEnvCfg_PLAY(Go2wObstacleTeacherFastEnvCfg):
    """Play variant of the fast obstacle teacher environment."""

    scene: ObstaclePlaySceneCfg = ObstaclePlaySceneCfg(num_envs=16, env_spacing=5.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs    = 16
        self.scene.env_spacing = 5.0
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = OBSTACLE_LIN_VEL_X
        self.commands.base_velocity.ranges.lin_vel_y = OBSTACLE_LIN_VEL_Y
        self.commands.base_velocity.ranges.ang_vel_z = OBSTACLE_ANG_VEL_Z
        self.observations.policy.obstacle_positions.params["obstacle_names"] = PLAY_OBSTACLE_NAMES
        self.observations.policy.obstacle_positions.params["num_closest"] = PRIVILEGED_OBSTACLE_SLOTS
        self.rewards.obstacle_path_clearance.params["obstacle_names"] = PLAY_OBSTACLE_NAMES
        self.rewards.obstacle_lateral_avoidance.params["obstacle_names"] = PLAY_OBSTACLE_NAMES
        self.rewards.obstacle_yaw_avoidance.params["obstacle_names"] = PLAY_OBSTACLE_NAMES
        self.events.reset_obstacles.params["start_iteration"] = 0
        self.events.reset_obstacles.params["warmup_iterations"] = 0
        self.events.reset_obstacles.params["obstacle_names"] = PLAY_OBSTACLE_NAMES
        self.events.reset_obstacles.params["min_obstacles"] = PLAY_NUM_OBSTACLES
        self.events.reset_obstacles.params["max_obstacles"] = PLAY_NUM_OBSTACLES
        self.events.reset_obstacles.params["min_inter_obstacle_dist"] = 0.45
        self.events.reset_obstacles.params["empty_env_fraction"] = 0.0
        self.terminations.obstacle_contact.params["start_iteration"] = 0
        self.events.push_robot        = None
        self.events.add_base_mass     = None
        self.events.speed_curriculum  = None  # use fixed ±2.0 m/s during play


@configclass
class Go2wObstacleDistillEnvCfg(Go2wObstacleTeacherEnvCfg):
    """Obstacle distillation environment: inherits scene/rewards/events from Teacher.

    Only observations differ: two groups (student LiDAR + teacher privileged)
    instead of the single teacher policy group. The teacher actions are now
    produced by a geometric steering module plus the frozen fast-flat LLC.
    """

    observations: ObstacleDistillObsCfg = ObstacleDistillObsCfg()


@configclass
class Go2wObstacleDistillEnvCfg_PLAY(Go2wObstacleDistillEnvCfg):
    """Distillation evaluation environment (uses student obs only)."""

    scene: ObstaclePlaySceneCfg = ObstaclePlaySceneCfg(num_envs=16, env_spacing=5.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs    = 16
        self.scene.env_spacing = 5.0
        self.observations.student.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = OBSTACLE_LIN_VEL_X
        self.commands.base_velocity.ranges.lin_vel_y = OBSTACLE_LIN_VEL_Y
        self.commands.base_velocity.ranges.ang_vel_z = OBSTACLE_ANG_VEL_Z
        self.observations.teacher.obstacle_positions.params["obstacle_names"] = PLAY_OBSTACLE_NAMES
        self.observations.teacher.obstacle_positions.params["num_closest"] = PRIVILEGED_OBSTACLE_SLOTS
        self.rewards.obstacle_path_clearance.params["obstacle_names"] = PLAY_OBSTACLE_NAMES
        self.rewards.obstacle_lateral_avoidance.params["obstacle_names"] = PLAY_OBSTACLE_NAMES
        self.rewards.obstacle_yaw_avoidance.params["obstacle_names"] = PLAY_OBSTACLE_NAMES
        self.events.reset_obstacles.params["start_iteration"] = 0
        self.events.reset_obstacles.params["warmup_iterations"] = 0
        self.events.reset_obstacles.params["obstacle_names"] = PLAY_OBSTACLE_NAMES
        self.events.reset_obstacles.params["min_obstacles"] = PLAY_NUM_OBSTACLES
        self.events.reset_obstacles.params["max_obstacles"] = PLAY_NUM_OBSTACLES
        self.events.reset_obstacles.params["min_inter_obstacle_dist"] = 0.45
        self.events.reset_obstacles.params["empty_env_fraction"] = 0.0
        self.events.push_robot    = None
        self.events.add_base_mass = None
        self.events.speed_curriculum = None  # use fixed ±2.0 m/s during play
