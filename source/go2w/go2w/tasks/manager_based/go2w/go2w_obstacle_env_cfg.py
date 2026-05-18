# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Go2-W HLC/LLC navigation teacher and LiDAR student distillation environment.

Training flow:
  1. Train RL navigation teacher (PPO) on goal-conditioned task:
       obs  = base_lin_vel(3) + projected_gravity(3) + goal_command(3) + obstacle_polar_depth(180) = 189D
       acts = FrozenLLCActionTerm: 3D velocity [vx,vy,yaw] -> frozen fast-flat LLC -> 16D joints
  2. Distill teacher into LiDAR student (action MSE, both 189D so direct MSE works):
       student obs = base_lin_vel(3) + projected_gravity(3) + goal_command(3) + lidar_scan(180) = 189D

  train.py --task Nav-Teacher-Go2w-v0 --locomotion_checkpoint <fast-flat-ckpt>
  train.py --task Navigation-RL-Distill-Go2w-v0 --teacher_checkpoint <teacher-ckpt> --locomotion_checkpoint <fast-flat-ckpt>
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
from .go2w_env_cfg import EventCfg, Go2wEnvCfg, Go2wSceneCfg

# =============================================================================
# Constants
# =============================================================================

# Physical obstacle slots
TRAIN_PHYSICAL_OBSTACLE_SLOTS = 15
PLAY_PHYSICAL_OBSTACLE_SLOTS = 64
PLAY_DEFAULT_ACTIVE_OBSTACLES = 5

# Convenience aliases
NUM_OBSTACLES = TRAIN_PHYSICAL_OBSTACLE_SLOTS
PRIVILEGED_OBSTACLE_SLOTS = 15
PLAY_MAX_OBSTACLES = PLAY_PHYSICAL_OBSTACLE_SLOTS
PLAY_NUM_OBSTACLES = PLAY_DEFAULT_ACTIVE_OBSTACLES
PLAY_MIN_INTER_OBSTACLE_DIST = 0.7

OBSTACLE_SIZE = (0.3, 0.3, 0.5)
OBSTACLE_SPAWN_RANGE = {"x": (-3.5, 3.5), "y": (-2.5, 2.5)}
OBSTACLE_NAMES = [f"obstacle_{i}" for i in range(TRAIN_PHYSICAL_OBSTACLE_SLOTS)]
PLAY_OBSTACLE_NAMES = [f"obstacle_{i}" for i in range(PLAY_PHYSICAL_OBSTACLE_SLOTS)]

CURRICULUM_STEPS_PER_ITERATION = 128
CURRICULUM_OBSTACLE_START_ITERATION = 1700
CURRICULUM_OBSTACLE_WARMUP_ITERATIONS = 1000
CURRICULUM_COLLISION_WARMUP_ITERATIONS = 600
CURRICULUM_SPEED_START_ITERATION = 0
CURRICULUM_SPEED_WARMUP_ITERATIONS = 800
NAV_CURRICULUM_COLLISION_START_ITERATION = 0
OBSTACLE_MIN_SPAWN_DISTANCE_INITIAL = 2.2
OBSTACLE_MIN_SPAWN_DISTANCE_FROM_ROBOT = 1.2
OBSTACLE_WHEEL_VEL_SCALE = 28.0
OBSTACLE_LIN_VEL_X = (-2.0, 2.0)
OBSTACLE_LIN_VEL_Y = (-2.0, 2.0)
OBSTACLE_ANG_VEL_Z = (-2.0, 2.0)
OBSTACLE_COLLISION_WEIGHT = -40.0

# Navigation task geometry
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

# Local waypoint shaping for obs terms
NAV_WAYPOINT_LOOKAHEAD_DISTANCE = 1.25
NAV_WAYPOINT_GOAL_SNAP_DISTANCE = 1.0
NAV_WAYPOINT_REFINEMENT_OFFSETS = (0.0, 0.45, -0.45, 0.70, -0.70)
NAV_LOCAL_PLANNER_ACTIVATION_THRESHOLD = 0.22
NAV_LOCAL_PLANNER_LATERAL_PENALTY = 0.16
NAV_LOCAL_PLANNER_MIN_IMPROVEMENT = 0.07
NAV_LOCAL_PLANNER_MAX_BLEND = 0.65
NAV_WAYPOINT_COMMAND_MIN_FORWARD = 0.0
NAV_WAYPOINT_COMMAND_MAX_LATERAL = 1.5
NAV_WAYPOINT_COMMAND_MAX_HEADING = 0.90

# Unitree L2 reference spec: 360 x 96 deg FoV, 30 m max range.
# Training uses a lightweight subset.
LIDAR_MAX_DISTANCE = 20.0
LIDAR_HORIZONTAL_FOV = (0.0, 360.0)
LIDAR_HORIZONTAL_RES = 2.0   # 180 rays
LIDAR_CHANNELS = 1            # single horizontal ring for 180D HLC student obs
LIDAR_VERTICAL_FOV = (0.0, 0.0)


# =============================================================================
# Actions
# =============================================================================


@configclass
class HLCNavActionsCfg:
    """High-level navigation action.

    The policy outputs only a 3D velocity command [vx, vy, yaw].  The frozen
    fast-flat LLC is executed inside the action term and sends the final 16D
    wheel/leg targets to the robot.
    """

    llc_cmd = mdp.FrozenLLCActionTermCfg(asset_name="robot")


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
    """Scene with Go2-W robot, flat ground, static box obstacles, and LiDAR."""

    replicate_physics: bool = False  # each env needs independent physics

    for i in range(NUM_OBSTACLES):
        vars()[f"obstacle_{i}"] = _make_obstacle_cfg(f"obstacle_{i}", i)
    del i

    obstacle_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/obstacle_.*",
        history_length=3,
        track_air_time=False,
    )

    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.28945, 0.0, -0.046825)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=LIDAR_CHANNELS,
            vertical_fov_range=LIDAR_VERTICAL_FOV,
            horizontal_fov_range=LIDAR_HORIZONTAL_FOV,
            horizontal_res=LIDAR_HORIZONTAL_RES,
        ),
        max_distance=LIDAR_MAX_DISTANCE,
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/obstacle_.*",
                track_mesh_transforms=True,
                is_shared=True,
            ),
        ],
        debug_vis=False,
    )


@configclass
class ObstaclePlaySceneCfg(ObstacleSceneCfg):
    """Play scene with extra obstacle slots for dense-clutter visual testing."""

    for i in range(NUM_OBSTACLES, PLAY_MAX_OBSTACLES):
        vars()[f"obstacle_{i}"] = _make_obstacle_cfg(f"obstacle_{i}", i)
    del i


# =============================================================================
# Events
# =============================================================================

# Shared nav task reset parameters passed to reset_navigation_goals_and_obstacles.
_NAV_RESET_PARAMS_BASE = {
    "spawn_range_x": OBSTACLE_SPAWN_RANGE["x"],
    "spawn_range_y": OBSTACLE_SPAWN_RANGE["y"],
    "obstacle_z": OBSTACLE_SIZE[2] / 2,
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


@configclass
class ObstacleEventCfg(EventCfg):
    """Base obstacle-environment events (legacy obstacle curriculum for compatibility)."""

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
class NavTeacherRewardsCfg:
    """Rewards for the RL navigation teacher.

    Combines goal-conditioned navigation rewards with locomotion stability
    penalties. No velocity-command tracking — the goal buffers drive learning.
    """

    # -- Navigation (goal-conditioned) -----------------------------------------
    goal_progress = RewTerm(
        func=mdp.goal_progress_dense,
        weight=6.0,
        params={"clip": 1.5},
    )
    goal_heading = RewTerm(
        func=mdp.goal_heading_tanh_reward,
        weight=0.5,
        params={"std": NAV_GOAL_HEADING_STD},
    )
    goal_reached = RewTerm(
        func=mdp.goal_reached_and_resample,
        weight=50.0,
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
    obstacle_ttc = RewTerm(
        func=mdp.obstacle_nav_ttc_penalty,
        weight=-3.0,
        params={
            "obstacle_names": OBSTACLE_NAMES,
            "safe_ttc": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
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


# =============================================================================
# Observations
# =============================================================================


@configclass
class NavTeacherObsCfg:
    """PPO observations for the RL navigation teacher (189D).

    proprio(9D) + privileged obstacle polar depth(180D).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel,      noise=Unoise(n_min=-0.1,  n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity,  noise=Unoise(n_min=-0.05, n_max=0.05))

        # Goal direction in body frame (3D); obstacle avoidance is learned by the HLC.
        goal_command = ObsTerm(
            func=mdp.local_goal_command_b,
            params={
                "use_lidar_refinement": False,
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
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class NavRLDistillObsCfg:
    """Distillation observations: student (LiDAR) and teacher (privileged).

    student: proprio(9D) + lidar_scan(180D) = 189D
    teacher: proprio(9D) + obstacle_polar_depth(180D) = 189D
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
                "use_lidar_refinement": False,
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
        """Privileged teacher observations (189D, matches NavTeacherObsCfg.PolicyCfg)."""

        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel,      noise=Unoise(n_min=-0.1,  n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity,  noise=Unoise(n_min=-0.05, n_max=0.05))

        goal_command = ObsTerm(
            func=mdp.local_goal_command_b,
            params={
                "use_lidar_refinement": False,
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
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    student: StudentCfg = StudentCfg()
    teacher: TeacherCfg = TeacherCfg()


# =============================================================================
# Environment configs
# =============================================================================


@configclass
class Go2wNavTeacherEnvCfg(Go2wEnvCfg):
    """RL navigation teacher environment.

    Inherits locomotion infrastructure from Go2wEnvCfg and adds:
    - Obstacle scene (boxes + LiDAR)
    - Goal-conditioned reward structure
    - Navigation reset event
    """

    scene: ObstacleSceneCfg = ObstacleSceneCfg(num_envs=8192, env_spacing=8.0)
    actions: HLCNavActionsCfg = HLCNavActionsCfg()
    observations: NavTeacherObsCfg = NavTeacherObsCfg()
    rewards: NavTeacherRewardsCfg = NavTeacherRewardsCfg()
    events: ObstacleEventCfg = ObstacleEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.lidar.update_period = self.decimation * self.sim.dt

        # Navigation task: goals are set by the reset event, not velocity commands.
        self.events.speed_curriculum = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.debug_vis = False

        # Replace the legacy obstacle curriculum with the navigation reset.
        self.events.reset_obstacles.func = mdp.reset_navigation_goals_and_obstacles
        self.events.reset_obstacles.params = {
            **_NAV_RESET_PARAMS_BASE,
            "obstacle_names": OBSTACLE_NAMES,
            "min_obstacles": 5,
            "max_obstacles": 12,
            "empty_env_fraction": 0.1,
            "min_inter_obstacle_dist": 0.7,
        }

        # Episode never terminates on goal reached — goal_reached_and_resample
        # resamples the target in-place so navigation continues uninterrupted.
        self.terminations.goal_reached = None


@configclass
class Go2wNavTeacherEnvCfg_PLAY(Go2wNavTeacherEnvCfg):
    """Play/eval env for the RL navigation teacher."""

    scene: ObstaclePlaySceneCfg = ObstaclePlaySceneCfg(num_envs=16, env_spacing=5.0)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 5.0
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.observations.policy.enable_corruption = False
        self.observations.policy.obstacle_depth.params["obstacle_names"] = PLAY_OBSTACLE_NAMES
        # Show velocity arrows driven by the HLC output (synced in FrozenLLCActionTerm).
        self.commands.base_velocity.debug_vis = True

        self.events.reset_obstacles.params = {
            **_NAV_RESET_PARAMS_BASE,
            "obstacle_names": PLAY_OBSTACLE_NAMES,
            "min_obstacles": PLAY_NUM_OBSTACLES,
            "max_obstacles": PLAY_NUM_OBSTACLES,
            "empty_env_fraction": 0.0,
            "min_inter_obstacle_dist": PLAY_MIN_INTER_OBSTACLE_DIST,
        }


@configclass
class Go2wNavRLDistillEnvCfg(Go2wNavTeacherEnvCfg):
    """LiDAR student distillation environment.

    Inherits the teacher env (same scene, rewards, events, terminations) and
    replaces observations with the student/teacher distillation obs groups.
    """

    observations: NavRLDistillObsCfg = NavRLDistillObsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Student acts in the env, so use more obstacles to diversify experience.
        self.events.reset_obstacles.params = {
            **_NAV_RESET_PARAMS_BASE,
            "obstacle_names": OBSTACLE_NAMES,
            "min_obstacles": 5,
            "max_obstacles": 5,
            "empty_env_fraction": 0.05,
            "min_inter_obstacle_dist": 0.7,
        }


@configclass
class Go2wNavRLDistillEnvCfg_PLAY(Go2wNavRLDistillEnvCfg):
    """Play/eval env for the LiDAR student distillation."""

    scene: ObstaclePlaySceneCfg = ObstaclePlaySceneCfg(num_envs=16, env_spacing=5.0)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 5.0
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.observations.student.enable_corruption = False
        self.observations.teacher.obstacle_depth.params["obstacle_names"] = PLAY_OBSTACLE_NAMES
        # Show velocity arrows driven by the HLC output (synced in FrozenLLCActionTerm).
        self.commands.base_velocity.debug_vis = True

        self.events.reset_obstacles.params = {
            **_NAV_RESET_PARAMS_BASE,
            "obstacle_names": PLAY_OBSTACLE_NAMES,
            "min_obstacles": PLAY_NUM_OBSTACLES,
            "max_obstacles": PLAY_NUM_OBSTACLES,
            "empty_env_fraction": 0.0,
            "min_inter_obstacle_dist": PLAY_MIN_INTER_OBSTACLE_DIST,
        }
