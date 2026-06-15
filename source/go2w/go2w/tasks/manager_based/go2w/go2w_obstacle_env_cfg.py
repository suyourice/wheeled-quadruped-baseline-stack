# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Go2-W HLC/LLC navigation teacher and student distillation environments.

Training flow:
  1. Train RL navigation teacher (PPO) on goal-conditioned task:
       obs  = base_lin_vel(3) + projected_gravity(3) + goal_command(3)
              + obstacle_polar_depth(180) + obstacle_nav_features(16)
              + obstacle_full_geometry(240) + prev_hlc_actions(6) = 451D
       acts = FrozenLLCActionTerm: 3D velocity [vx,vy,yaw] -> frozen fast-flat LLC -> 16D joints
  2. Distill teacher into LiDAR student (action MSE, teacher=451D, student=189D):
       student obs = base_lin_vel(3) + projected_gravity(3) + goal_command(3) + lidar_scan(180) = 189D
  3. Distill teacher into depth student:
       student obs = state(15D) + depth history stack from a head-mounted D456-like camera

  train.py --task Nav-Teacher-Go2w-v0 --locomotion_checkpoint <fast-flat-ckpt>
  train.py --task Navigation-RL-Distill-Go2w-v0 --teacher_checkpoint <teacher-ckpt> --locomotion_checkpoint <fast-flat-ckpt>
  train.py --task Navigation-Depth-Distill-Go2w-v0 --teacher_checkpoint <teacher-ckpt> --locomotion_checkpoint <fast-flat-ckpt>
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCameraCfg, MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .mdp.hospital import events as _hospital_events
from .mdp.hospital import specs as _hospital_specs
from .mdp.hospital.floor import (
    HOSPITAL_RAMP_ASSET_NAME,
    HOSPITAL_RAMP_B_ASSET_NAME,
    make_hospital_ramp_cfg as _make_hospital_ramp_cfg,
    make_hospital_ramp_b_cfg as _make_hospital_ramp_b_cfg,
    HOSPITAL_FLOOR_CORRIDOR_KIND,
    HOSPITAL_FLOOR_LEG_LENGTH,
    HOSPITAL_FLOOR_CORRIDOR_WIDTH,
    HOSPITAL_FLOOR_WALL_THICKNESS,
    HOSPITAL_FLOOR_DYNAMIC_OBSTACLE_COUNT,
    HOSPITAL_FLOOR_EPISODE_LENGTH_S,
    HOSPITAL_FLOOR_ENV_SPACING,
    HOSPITAL_FLOOR_ROBOT_START_LOCAL_XY,
    HOSPITAL_FLOOR_GOAL_DONE_RADIUS,
    HOSPITAL_FLOOR_RAMP_LOCAL_POSE,
    HOSPITAL_FLOOR_RAMP_B_LOCAL_POSE,
    hospital_floor_semantic_local_poses as _hospital_floor_semantic_local_poses,
    hospital_floor_queue_groups as _hospital_floor_queue_groups,
    hospital_floor_seated_groups as _hospital_floor_seated_groups,
)
from .go2w_env_cfg import EventCfg, Go2wEnvCfg, Go2wSceneCfg

TRAIN_PHYSICAL_OBSTACLE_SLOTS = _hospital_specs.TRAIN_PHYSICAL_OBSTACLE_SLOTS
PLAY_PHYSICAL_OBSTACLE_SLOTS = _hospital_specs.PLAY_PHYSICAL_OBSTACLE_SLOTS
PLAY_DEFAULT_ACTIVE_OBSTACLES = _hospital_specs.PLAY_DEFAULT_ACTIVE_OBSTACLES
NUM_OBSTACLES = _hospital_specs.NUM_OBSTACLES
PRIVILEGED_OBSTACLE_SLOTS = _hospital_specs.PRIVILEGED_OBSTACLE_SLOTS
PLAY_MAX_OBSTACLES = _hospital_specs.PLAY_MAX_OBSTACLES
PLAY_NUM_OBSTACLES = _hospital_specs.PLAY_NUM_OBSTACLES
PLAY_MIN_INTER_OBSTACLE_DIST = _hospital_specs.PLAY_MIN_INTER_OBSTACLE_DIST
OBSTACLE_SIZE = _hospital_specs.OBSTACLE_SIZE
TRAIN_OBSTACLE_SPECS = _hospital_specs.TRAIN_OBSTACLE_SPECS
TRAIN_OBSTACLE_SHAPE_IDS = _hospital_specs.TRAIN_OBSTACLE_SHAPE_IDS
TRAIN_OBSTACLE_WIDTHS = _hospital_specs.TRAIN_OBSTACLE_WIDTHS
TRAIN_OBSTACLE_DEPTHS = _hospital_specs.TRAIN_OBSTACLE_DEPTHS
PLAY_OBSTACLE_SPECS = _hospital_specs.PLAY_OBSTACLE_SPECS
PLAY_OBSTACLE_SHAPE_IDS = _hospital_specs.PLAY_OBSTACLE_SHAPE_IDS
PLAY_OBSTACLE_WIDTHS = _hospital_specs.PLAY_OBSTACLE_WIDTHS
PLAY_OBSTACLE_DEPTHS = _hospital_specs.PLAY_OBSTACLE_DEPTHS
HOSPITAL_PLAY_OBSTACLE_SHAPE_IDS = _hospital_specs.HOSPITAL_PLAY_OBSTACLE_SHAPE_IDS
HOSPITAL_PLAY_OBSTACLE_WIDTHS = _hospital_specs.HOSPITAL_PLAY_OBSTACLE_WIDTHS
HOSPITAL_PLAY_OBSTACLE_DEPTHS = _hospital_specs.HOSPITAL_PLAY_OBSTACLE_DEPTHS
HOSPITAL_PLAY_OBSTACLE_HEIGHTS = _hospital_specs.HOSPITAL_PLAY_OBSTACLE_HEIGHTS
HOSPITAL_PLAY_OBSTACLE_LABELS = _hospital_specs.HOSPITAL_PLAY_OBSTACLE_LABELS
HOSPITAL_PLAY_GROUP_PAIRS = _hospital_specs.HOSPITAL_PLAY_GROUP_PAIRS
HOSPITAL_DEFAULT_COLOR = _hospital_specs.HOSPITAL_DEFAULT_COLOR
HOSPITAL_LABEL_COLORS = _hospital_specs.HOSPITAL_LABEL_COLORS
OBSTACLE_SHAPE_CUBOID = _hospital_specs.OBSTACLE_SHAPE_CUBOID
OBSTACLE_SHAPE_CYLINDER = _hospital_specs.OBSTACLE_SHAPE_CYLINDER
OBSTACLE_SHAPE_CONE = _hospital_specs.OBSTACLE_SHAPE_CONE
OBSTACLE_GROUND_CLEARANCE = _hospital_specs.OBSTACLE_GROUND_CLEARANCE
OBSTACLE_Z = _hospital_specs.OBSTACLE_Z
OBSTACLE_SPAWN_RANGE = _hospital_specs.OBSTACLE_SPAWN_RANGE
OBSTACLE_NAMES = _hospital_specs.OBSTACLE_NAMES
PLAY_OBSTACLE_NAMES = _hospital_specs.PLAY_OBSTACLE_NAMES
CURRICULUM_STEPS_PER_ITERATION = _hospital_specs.CURRICULUM_STEPS_PER_ITERATION
CURRICULUM_OBSTACLE_START_ITERATION = _hospital_specs.CURRICULUM_OBSTACLE_START_ITERATION
CURRICULUM_OBSTACLE_WARMUP_ITERATIONS = _hospital_specs.CURRICULUM_OBSTACLE_WARMUP_ITERATIONS
CURRICULUM_COLLISION_WARMUP_ITERATIONS = _hospital_specs.CURRICULUM_COLLISION_WARMUP_ITERATIONS
CURRICULUM_SPEED_START_ITERATION = _hospital_specs.CURRICULUM_SPEED_START_ITERATION
CURRICULUM_SPEED_WARMUP_ITERATIONS = _hospital_specs.CURRICULUM_SPEED_WARMUP_ITERATIONS
NAV_CURRICULUM_COLLISION_START_ITERATION = _hospital_specs.NAV_CURRICULUM_COLLISION_START_ITERATION
OBSTACLE_MIN_SPAWN_DISTANCE_INITIAL = _hospital_specs.OBSTACLE_MIN_SPAWN_DISTANCE_INITIAL
OBSTACLE_MIN_SPAWN_DISTANCE_FROM_ROBOT = _hospital_specs.OBSTACLE_MIN_SPAWN_DISTANCE_FROM_ROBOT
OBSTACLE_LIN_VEL_X = _hospital_specs.OBSTACLE_LIN_VEL_X
OBSTACLE_LIN_VEL_Y = _hospital_specs.OBSTACLE_LIN_VEL_Y
OBSTACLE_ANG_VEL_Z = _hospital_specs.OBSTACLE_ANG_VEL_Z
OBSTACLE_COLLISION_WEIGHT = _hospital_specs.OBSTACLE_COLLISION_WEIGHT
NAV_TTC_FALLBACK_OBSTACLE_RADIUS = _hospital_specs.NAV_TTC_FALLBACK_OBSTACLE_RADIUS
NAV_TTC_ROBOT_HALF_WIDTH = _hospital_specs.NAV_TTC_ROBOT_HALF_WIDTH
NAV_TTC_SAFETY_MARGIN = _hospital_specs.NAV_TTC_SAFETY_MARGIN
NAV_TTC_FRONT_MARGIN = _hospital_specs.NAV_TTC_FRONT_MARGIN
NAV_TTC_LOOKAHEAD_DISTANCE = _hospital_specs.NAV_TTC_LOOKAHEAD_DISTANCE
NAV_TTC_SUM_CLIP = _hospital_specs.NAV_TTC_SUM_CLIP
NAV_OBSTACLE_RADIUS_MARGIN = _hospital_specs.NAV_OBSTACLE_RADIUS_MARGIN
NAV_PHYSICAL_SLOT_RANDOMIZATION_START_ITERATION = _hospital_specs.NAV_PHYSICAL_SLOT_RANDOMIZATION_START_ITERATION
NAV_PHYSICAL_SLOT_RANDOMIZATION_WARMUP_ITERATIONS = _hospital_specs.NAV_PHYSICAL_SLOT_RANDOMIZATION_WARMUP_ITERATIONS
NAV_RANDOMIZE_OBSTACLE_YAW = _hospital_specs.NAV_RANDOMIZE_OBSTACLE_YAW
NAV_OBSTACLE_YAW_RANGE = _hospital_specs.NAV_OBSTACLE_YAW_RANGE
NAV_PASSABLE_GAP_ROBOT_WIDTH = _hospital_specs.NAV_PASSABLE_GAP_ROBOT_WIDTH
NAV_PASSABLE_GAP_MIN_WIDTH = _hospital_specs.NAV_PASSABLE_GAP_MIN_WIDTH
NAV_GOAL_FORWARD_RANGE = _hospital_specs.NAV_GOAL_FORWARD_RANGE
NAV_GOAL_LATERAL_RANGE = _hospital_specs.NAV_GOAL_LATERAL_RANGE
NAV_GOAL_HEADING_JITTER_RANGE = _hospital_specs.NAV_GOAL_HEADING_JITTER_RANGE
NAV_MIN_GOAL_DISTANCE = _hospital_specs.NAV_MIN_GOAL_DISTANCE
NAV_START_EXCLUSION_RADIUS = _hospital_specs.NAV_START_EXCLUSION_RADIUS
NAV_GOAL_EXCLUSION_RADIUS = _hospital_specs.NAV_GOAL_EXCLUSION_RADIUS
NAV_HEAD_ON_PROGRESS_RANGE = _hospital_specs.NAV_HEAD_ON_PROGRESS_RANGE
NAV_HEAD_ON_LATERAL_RANGE = _hospital_specs.NAV_HEAD_ON_LATERAL_RANGE
NAV_EDGE_PROGRESS_RANGE = _hospital_specs.NAV_EDGE_PROGRESS_RANGE
NAV_EDGE_LATERAL_RANGE = _hospital_specs.NAV_EDGE_LATERAL_RANGE
NAV_DIAGONAL_PROGRESS_RANGE = _hospital_specs.NAV_DIAGONAL_PROGRESS_RANGE
NAV_DIAGONAL_LATERAL_RANGE = _hospital_specs.NAV_DIAGONAL_LATERAL_RANGE
NAV_OFFPATH_PROGRESS_RANGE = _hospital_specs.NAV_OFFPATH_PROGRESS_RANGE
NAV_OFFPATH_LATERAL_RANGE = _hospital_specs.NAV_OFFPATH_LATERAL_RANGE
NAV_NARROW_GAP_PROGRESS_RANGE = _hospital_specs.NAV_NARROW_GAP_PROGRESS_RANGE
NAV_NARROW_GAP_CENTER_LATERAL_RANGE = _hospital_specs.NAV_NARROW_GAP_CENTER_LATERAL_RANGE
NAV_NARROW_GAP_HALF_WIDTH_RANGE = _hospital_specs.NAV_NARROW_GAP_HALF_WIDTH_RANGE
NAV_NARROW_GAP_WIDE_HALF_WIDTH_RANGE = _hospital_specs.NAV_NARROW_GAP_WIDE_HALF_WIDTH_RANGE
NAV_NARROW_GAP_BARELY_HALF_WIDTH_RANGE = _hospital_specs.NAV_NARROW_GAP_BARELY_HALF_WIDTH_RANGE
NAV_NARROW_GAP_PROBABILITY = _hospital_specs.NAV_NARROW_GAP_PROBABILITY
NAV_PARTIAL_BLOCKAGE_PROGRESS_RANGE = _hospital_specs.NAV_PARTIAL_BLOCKAGE_PROGRESS_RANGE
NAV_PARTIAL_BLOCKAGE_LATERAL_RANGE = _hospital_specs.NAV_PARTIAL_BLOCKAGE_LATERAL_RANGE
NAV_PARTIAL_BLOCKAGE_PROBABILITY = _hospital_specs.NAV_PARTIAL_BLOCKAGE_PROBABILITY
NAV_CLUTTERED_PROGRESS_RANGE = _hospital_specs.NAV_CLUTTERED_PROGRESS_RANGE
NAV_CLUTTERED_LATERAL_RANGE = _hospital_specs.NAV_CLUTTERED_LATERAL_RANGE
NAV_CURRICULUM_PHASE_SCHEDULE = _hospital_specs.NAV_CURRICULUM_PHASE_SCHEDULE
NAV_GOAL_HEADING_STD = _hospital_specs.NAV_GOAL_HEADING_STD
NAV_GOAL_SUCCESS_POSITION_THRESHOLD = _hospital_specs.NAV_GOAL_SUCCESS_POSITION_THRESHOLD
NAV_GOAL_SUCCESS_HEADING_THRESHOLD = _hospital_specs.NAV_GOAL_SUCCESS_HEADING_THRESHOLD
NAV_WAYPOINT_LOOKAHEAD_DISTANCE = _hospital_specs.NAV_WAYPOINT_LOOKAHEAD_DISTANCE
NAV_WAYPOINT_GOAL_SNAP_DISTANCE = _hospital_specs.NAV_WAYPOINT_GOAL_SNAP_DISTANCE
NAV_WAYPOINT_REFINEMENT_OFFSETS = _hospital_specs.NAV_WAYPOINT_REFINEMENT_OFFSETS
NAV_LOCAL_PLANNER_ACTIVATION_THRESHOLD = _hospital_specs.NAV_LOCAL_PLANNER_ACTIVATION_THRESHOLD
NAV_LOCAL_PLANNER_LATERAL_PENALTY = _hospital_specs.NAV_LOCAL_PLANNER_LATERAL_PENALTY
NAV_LOCAL_PLANNER_MIN_IMPROVEMENT = _hospital_specs.NAV_LOCAL_PLANNER_MIN_IMPROVEMENT
NAV_LOCAL_PLANNER_MAX_BLEND = _hospital_specs.NAV_LOCAL_PLANNER_MAX_BLEND
NAV_WAYPOINT_COMMAND_MIN_FORWARD = _hospital_specs.NAV_WAYPOINT_COMMAND_MIN_FORWARD
NAV_WAYPOINT_COMMAND_MAX_LATERAL = _hospital_specs.NAV_WAYPOINT_COMMAND_MAX_LATERAL
NAV_WAYPOINT_COMMAND_MAX_HEADING = _hospital_specs.NAV_WAYPOINT_COMMAND_MAX_HEADING
NAV_PASSABLE_GAP_REWARD_WEIGHT = _hospital_specs.NAV_PASSABLE_GAP_REWARD_WEIGHT
NAV_CLEARANCE_PASSABLE_GAP_RELIEF = _hospital_specs.NAV_CLEARANCE_PASSABLE_GAP_RELIEF
NAV_TTC_PASSABLE_GAP_RELIEF = _hospital_specs.NAV_TTC_PASSABLE_GAP_RELIEF
NAV_DENSE_RECOVERY_WEIGHT = _hospital_specs.NAV_DENSE_RECOVERY_WEIGHT
NAV_GRAZING_WEIGHT = _hospital_specs.NAV_GRAZING_WEIGHT
NAV_CLEARANCE_SURFACE_BUFFER = _hospital_specs.NAV_CLEARANCE_SURFACE_BUFFER
NAV_GRAZING_DISTANCE = _hospital_specs.NAV_GRAZING_DISTANCE
NAV_GRAZING_CONTACT_DISTANCE = _hospital_specs.NAV_GRAZING_CONTACT_DISTANCE
NAV_GRAZING_PASSABLE_GAP_RELIEF = _hospital_specs.NAV_GRAZING_PASSABLE_GAP_RELIEF
LIDAR_MAX_DISTANCE = _hospital_specs.LIDAR_MAX_DISTANCE
LIDAR_HORIZONTAL_FOV = _hospital_specs.LIDAR_HORIZONTAL_FOV
LIDAR_HORIZONTAL_RES = _hospital_specs.LIDAR_HORIZONTAL_RES
LIDAR_CHANNELS = _hospital_specs.LIDAR_CHANNELS
LIDAR_VERTICAL_FOV = _hospital_specs.LIDAR_VERTICAL_FOV
D456_DEPTH_MIN_DISTANCE = _hospital_specs.D456_DEPTH_MIN_DISTANCE
D456_DEPTH_MAX_DISTANCE = _hospital_specs.D456_DEPTH_MAX_DISTANCE
D456_DEPTH_HORIZONTAL_FOV_DEG = _hospital_specs.D456_DEPTH_HORIZONTAL_FOV_DEG
D456_DEPTH_VERTICAL_FOV_DEG = _hospital_specs.D456_DEPTH_VERTICAL_FOV_DEG
D456_NATIVE_DEPTH_RESOLUTION = _hospital_specs.D456_NATIVE_DEPTH_RESOLUTION
DEPTH_IMAGE_WIDTH = _hospital_specs.DEPTH_IMAGE_WIDTH
DEPTH_IMAGE_HEIGHT = _hospital_specs.DEPTH_IMAGE_HEIGHT
DEPTH_HISTORY_LENGTH = _hospital_specs.DEPTH_HISTORY_LENGTH
DEPTH_DISTILL_NUM_ENVS = _hospital_specs.DEPTH_DISTILL_NUM_ENVS
DEPTH_DISTILL_MIN_OBSTACLES = _hospital_specs.DEPTH_DISTILL_MIN_OBSTACLES
DEPTH_DISTILL_MAX_OBSTACLES = _hospital_specs.DEPTH_DISTILL_MAX_OBSTACLES
DEPTH_DISTILL_EMPTY_ENV_FRACTION = _hospital_specs.DEPTH_DISTILL_EMPTY_ENV_FRACTION
DEPTH_DISTILL_MIN_INTER_OBSTACLE_DIST = _hospital_specs.DEPTH_DISTILL_MIN_INTER_OBSTACLE_DIST
DEPTH_DISTILL_DYNAMIC_START_ITERATION = _hospital_specs.DEPTH_DISTILL_DYNAMIC_START_ITERATION
DEPTH_DISTILL_DYNAMIC_WARMUP_ITERATIONS = _hospital_specs.DEPTH_DISTILL_DYNAMIC_WARMUP_ITERATIONS
DEPTH_DISTILL_DYNAMIC_SPEED_RANGE = _hospital_specs.DEPTH_DISTILL_DYNAMIC_SPEED_RANGE
DEPTH_DISTILL_DYNAMIC_LATERAL_SPEED = _hospital_specs.DEPTH_DISTILL_DYNAMIC_LATERAL_SPEED
DEPTH_DISTILL_DYNAMIC_LONGITUDINAL_EXTENT = _hospital_specs.DEPTH_DISTILL_DYNAMIC_LONGITUDINAL_EXTENT
DEPTH_DISTILL_DYNAMIC_LATERAL_EXTENT = _hospital_specs.DEPTH_DISTILL_DYNAMIC_LATERAL_EXTENT
DEPTH_DISTILL_DYNAMIC_SPEED_CHANGE_INTERVAL = _hospital_specs.DEPTH_DISTILL_DYNAMIC_SPEED_CHANGE_INTERVAL
DEPTH_DISTILL_DYNAMIC_WANDER_FRACTION = _hospital_specs.DEPTH_DISTILL_DYNAMIC_WANDER_FRACTION
D456_CAMERA_FOCAL_LENGTH_CM = _hospital_specs.D456_CAMERA_FOCAL_LENGTH_CM
D456_CAMERA_HORIZONTAL_APERTURE_CM = _hospital_specs.D456_CAMERA_HORIZONTAL_APERTURE_CM
D456_CAMERA_VERTICAL_APERTURE_CM = _hospital_specs.D456_CAMERA_VERTICAL_APERTURE_CM
D456_CAMERA_PITCH_DOWN_DEG = _hospital_specs.D456_CAMERA_PITCH_DOWN_DEG
D456_CAMERA_PITCH_DOWN_QUAT_WXYZ = _hospital_specs.D456_CAMERA_PITCH_DOWN_QUAT_WXYZ


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


def _make_obstacle_cfg(
    name: str,
    idx: int,
    shape_kind: str = "cuboid",
    footprint_size: tuple[float, float] = (OBSTACLE_SIZE[0], OBSTACLE_SIZE[1]),
    height: float = OBSTACLE_SIZE[2],
    visual_color: tuple[float, float, float] = HOSPITAL_DEFAULT_COLOR,
) -> RigidObjectCfg:
    """Create a physical obstacle with variable footprint and fixed height."""
    width, depth = footprint_size
    center_z = height * 0.5 + OBSTACLE_GROUND_CLEARANCE
    spawn_kwargs = {
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        "collision_props": sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        "visual_material": sim_utils.PreviewSurfaceCfg(
            diffuse_color=visual_color,
        ),
        "activate_contact_sensors": True,
    }
    if shape_kind == "cuboid":
        spawn = sim_utils.CuboidCfg(
            size=(width, depth, height),
            **spawn_kwargs,
        )
    elif shape_kind == "cylinder":
        spawn = sim_utils.CylinderCfg(
            radius=max(width, depth) / 2.0,
            height=height,
            **spawn_kwargs,
        )
    elif shape_kind == "cone":
        spawn = sim_utils.ConeCfg(
            radius=max(width, depth) / 2.0,
            height=height,
            **spawn_kwargs,
        )
    else:
        raise ValueError(f"Unsupported obstacle shape: {shape_kind!r}")

    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.5 + idx * 0.5, 0.0, center_z),
        ),
    )


def make_play_obstacle_cfg(
    name: str,
    idx: int,
    shape_kind: str,
    footprint_size: tuple[float, float],
    height: float = OBSTACLE_SIZE[2],
    visual_color: tuple[float, float, float] = HOSPITAL_DEFAULT_COLOR,
) -> RigidObjectCfg:
    """Create an overridden play obstacle asset."""
    return _make_obstacle_cfg(name, idx, shape_kind, footprint_size, height, visual_color)




@configclass
class ObstacleSceneCfg(Go2wSceneCfg):
    """Scene with Go2-W robot, flat ground, physical obstacle variants, and LiDAR."""

    replicate_physics: bool = False  # each env needs independent physics

    for i, (shape_kind, footprint_size) in enumerate(TRAIN_OBSTACLE_SPECS):
        vars()[f"obstacle_{i}"] = _make_obstacle_cfg(
            f"obstacle_{i}", i, shape_kind, footprint_size
        )
    del i, shape_kind, footprint_size

    obstacle_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/obstacle_.*",
        history_length=3,
        track_air_time=False,
        # No contact filter: the sensor prim matches multiple obstacles per env, so
        # Isaac's filtered-contact reporting is unsupported here. Instead obstacles
        # are floated by OBSTACLE_GROUND_CLEARANCE so net_forces only ever reflect
        # robot↔obstacle contacts (no obstacle↔ground reaction).
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
    """Play scene with configurable obstacle capacity for visual testing."""

    hospital_ramp: RigidObjectCfg = _make_hospital_ramp_cfg()
    hospital_ramp_b: RigidObjectCfg = _make_hospital_ramp_b_cfg()

    if PLAY_MAX_OBSTACLES > NUM_OBSTACLES:
        for i in range(NUM_OBSTACLES, PLAY_MAX_OBSTACLES):
            vars()[f"obstacle_{i}"] = _make_obstacle_cfg(f"obstacle_{i}", i)
        del i


def _make_depth_camera_cfg() -> MultiMeshRayCasterCameraCfg:
    """Create a lightweight D456-like ray-cast depth camera."""
    return MultiMeshRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Head_upper",
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.095),
            rot=D456_CAMERA_PITCH_DOWN_QUAT_WXYZ,
            convention="world",
        ),
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="max",
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=D456_CAMERA_FOCAL_LENGTH_CM,
            horizontal_aperture=D456_CAMERA_HORIZONTAL_APERTURE_CM,
            vertical_aperture=D456_CAMERA_VERTICAL_APERTURE_CM,
            width=DEPTH_IMAGE_WIDTH,
            height=DEPTH_IMAGE_HEIGHT,
        ),
        max_distance=D456_DEPTH_MAX_DISTANCE,
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
class DepthObstacleSceneCfg(ObstacleSceneCfg):
    """Training scene with LiDAR compatibility plus a head-mounted depth camera."""

    depth_camera = _make_depth_camera_cfg()


@configclass
class DepthObstaclePlaySceneCfg(ObstaclePlaySceneCfg):
    """Play scene with extra obstacle slots plus a head-mounted depth camera."""

    depth_camera = _make_depth_camera_cfg()


# =============================================================================
# Events
# =============================================================================

# Shared nav task reset parameters passed to reset_navigation_goals_and_obstacles.
_NAV_RESET_PARAMS_BASE = {
    "spawn_range_x": OBSTACLE_SPAWN_RANGE["x"],
    "spawn_range_y": OBSTACLE_SPAWN_RANGE["y"],
    "obstacle_z": OBSTACLE_Z,
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
    "narrow_gap_wide_half_width_range": NAV_NARROW_GAP_WIDE_HALF_WIDTH_RANGE,
    "narrow_gap_barely_half_width_range": NAV_NARROW_GAP_BARELY_HALF_WIDTH_RANGE,
    "partial_blockage_progress_range": NAV_PARTIAL_BLOCKAGE_PROGRESS_RANGE,
    "partial_blockage_lateral_range": NAV_PARTIAL_BLOCKAGE_LATERAL_RANGE,
    "partial_blockage_probability": NAV_PARTIAL_BLOCKAGE_PROBABILITY,
    "cluttered_progress_range": NAV_CLUTTERED_PROGRESS_RANGE,
    "cluttered_lateral_range": NAV_CLUTTERED_LATERAL_RANGE,
    "phase_schedule": None,
    "steps_per_iteration": CURRICULUM_STEPS_PER_ITERATION,
    "fixed_goal_forward": None,
    "fixed_goal_lateral": None,
    "fixed_goal_heading_jitter": None,
    "fixed_scenario_template": None,
    "obstacle_radius_margin": 0.0,
    "randomize_physical_obstacle_slots": False,
    "randomize_obstacle_yaw": NAV_RANDOMIZE_OBSTACLE_YAW,
    "obstacle_yaw_range": NAV_OBSTACLE_YAW_RANGE,
    "passable_gap_min_width": NAV_PASSABLE_GAP_MIN_WIDTH,
    "passable_gap_robot_width": NAV_PASSABLE_GAP_ROBOT_WIDTH,
}


@configclass
class ObstacleEventCfg(EventCfg):
    """Base obstacle-environment events (legacy obstacle curriculum for compatibility)."""

    depth_distill_dynamic_obstacles: EventTerm | None = None
    navigation_path_update: EventTerm | None = None
    hospital_dynamic_motion: EventTerm | None = None
    hospital_velocity_resample: EventTerm | None = None
    hospital_group_update: EventTerm | None = None

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
            "obstacle_z": OBSTACLE_Z,
            "min_spawn_distance_from_robot": OBSTACLE_MIN_SPAWN_DISTANCE_FROM_ROBOT,
            "min_spawn_distance_from_robot_initial": OBSTACLE_MIN_SPAWN_DISTANCE_INITIAL,
            "min_inter_obstacle_dist": 0.8,
            "min_survival_steps": 800,
            "fixed_obstacle_shape_ids": TRAIN_OBSTACLE_SHAPE_IDS,
            "fixed_obstacle_widths": TRAIN_OBSTACLE_WIDTHS,
            "fixed_obstacle_depths": TRAIN_OBSTACLE_DEPTHS,
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


def _retarget_nav_rewards_to_play_obstacles(rewards: NavTeacherRewardsCfg) -> None:
    """Use the full play obstacle slot list for footprint-aware reward terms."""
    reward_names = (
        "nav_clearance",
        "nav_lateral_escape",
        "nav_open_path_straightness",
        "nav_open_path_goal_heading",
        "nav_impossible_gap",
        "obstacle_ttc",
        "nav_dense_recovery",
        "nav_grazing",
    )
    for reward_name in reward_names:
        getattr(rewards, reward_name).params["obstacle_names"] = PLAY_OBSTACLE_NAMES


def _configure_play_common(cfg) -> None:
    """Apply play-mode settings shared by all PLAY environment configs."""
    cfg.scene.num_envs = 16
    cfg.scene.env_spacing = 5.0
    cfg.events.push_robot = None
    cfg.events.add_base_mass = None
    _retarget_nav_rewards_to_play_obstacles(cfg.rewards)
    cfg.commands.base_velocity.debug_vis = True
    cfg.events.reset_obstacles.params = {
        **_NAV_RESET_PARAMS_BASE,
        "obstacle_names": PLAY_OBSTACLE_NAMES,
        "min_obstacles": PLAY_NUM_OBSTACLES,
        "max_obstacles": PLAY_NUM_OBSTACLES,
        "empty_env_fraction": 0.0,
        "min_inter_obstacle_dist": PLAY_MIN_INTER_OBSTACLE_DIST,
        "obstacle_radius_margin": NAV_OBSTACLE_RADIUS_MARGIN,
        "fixed_obstacle_shape_ids": PLAY_OBSTACLE_SHAPE_IDS,
        "fixed_obstacle_widths": PLAY_OBSTACLE_WIDTHS,
        "fixed_obstacle_depths": PLAY_OBSTACLE_DEPTHS,
        "randomize_physical_obstacle_slots": True,
        "physical_slot_randomization_start_iteration": 0,
        "physical_slot_randomization_warmup_iterations": 0,
    }


def _configure_play_obstacle_obs(obs_group) -> None:
    """Redirect privileged obstacle obs to the smaller PLAY obstacle set."""
    obs_group.obstacle_depth.params["obstacle_names"] = PLAY_OBSTACLE_NAMES
    obs_group.obstacle_nav_features.params["obstacle_names"] = PLAY_OBSTACLE_NAMES
    obs_group.obstacle_full_geometry.params["obstacle_names"] = PLAY_OBSTACLE_NAMES


# =============================================================================
# Observations
# =============================================================================


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


# =============================================================================
# Environment configs
# =============================================================================


@configclass
class Go2wNavTeacherEnvCfg(Go2wEnvCfg):
    """RL navigation teacher environment.

    Inherits locomotion infrastructure from Go2wEnvCfg and adds:
    - Obstacle scene (physical shape variants + LiDAR)
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
            "phase_schedule": NAV_CURRICULUM_PHASE_SCHEDULE,
            "obstacle_radius_margin": NAV_OBSTACLE_RADIUS_MARGIN,
            "fixed_obstacle_shape_ids": TRAIN_OBSTACLE_SHAPE_IDS,
            "fixed_obstacle_widths": TRAIN_OBSTACLE_WIDTHS,
            "fixed_obstacle_depths": TRAIN_OBSTACLE_DEPTHS,
            "randomize_physical_obstacle_slots": True,
            "physical_slot_randomization_start_iteration": NAV_PHYSICAL_SLOT_RANDOMIZATION_START_ITERATION,
            "physical_slot_randomization_warmup_iterations": NAV_PHYSICAL_SLOT_RANDOMIZATION_WARMUP_ITERATIONS,
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
        _configure_play_common(self)
        self.observations.policy.enable_corruption = False
        _configure_play_obstacle_obs(self.observations.policy)


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
            "phase_schedule": NAV_CURRICULUM_PHASE_SCHEDULE,
            "obstacle_radius_margin": NAV_OBSTACLE_RADIUS_MARGIN,
            "fixed_obstacle_shape_ids": TRAIN_OBSTACLE_SHAPE_IDS,
            "fixed_obstacle_widths": TRAIN_OBSTACLE_WIDTHS,
            "fixed_obstacle_depths": TRAIN_OBSTACLE_DEPTHS,
            "randomize_physical_obstacle_slots": True,
            "physical_slot_randomization_start_iteration": NAV_PHYSICAL_SLOT_RANDOMIZATION_START_ITERATION,
            "physical_slot_randomization_warmup_iterations": NAV_PHYSICAL_SLOT_RANDOMIZATION_WARMUP_ITERATIONS,
        }


@configclass
class Go2wNavRLDistillEnvCfg_PLAY(Go2wNavRLDistillEnvCfg):
    """Play/eval env for the LiDAR student distillation."""

    scene: ObstaclePlaySceneCfg = ObstaclePlaySceneCfg(num_envs=16, env_spacing=5.0)

    def __post_init__(self):
        super().__post_init__()
        _configure_play_common(self)
        self.observations.student.enable_corruption = False
        _configure_play_obstacle_obs(self.observations.teacher)


@configclass
class Go2wNavDepthRLDistillEnvCfg(Go2wNavRLDistillEnvCfg):
    """Depth-camera student distillation environment."""

    scene: DepthObstacleSceneCfg = DepthObstacleSceneCfg(num_envs=DEPTH_DISTILL_NUM_ENVS, env_spacing=8.0)
    observations: NavDepthRLDistillObsCfg = NavDepthRLDistillObsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.depth_camera.update_period = self.decimation * self.sim.dt
        self.events.reset_obstacles.params = {
            **_NAV_RESET_PARAMS_BASE,
            "obstacle_names": OBSTACLE_NAMES,
            "min_obstacles": DEPTH_DISTILL_MIN_OBSTACLES,
            "max_obstacles": DEPTH_DISTILL_MAX_OBSTACLES,
            "empty_env_fraction": DEPTH_DISTILL_EMPTY_ENV_FRACTION,
            "min_inter_obstacle_dist": DEPTH_DISTILL_MIN_INTER_OBSTACLE_DIST,
            "phase_schedule": None,
            "obstacle_radius_margin": NAV_OBSTACLE_RADIUS_MARGIN,
            "fixed_obstacle_shape_ids": TRAIN_OBSTACLE_SHAPE_IDS,
            "fixed_obstacle_widths": TRAIN_OBSTACLE_WIDTHS,
            "fixed_obstacle_depths": TRAIN_OBSTACLE_DEPTHS,
            "randomize_physical_obstacle_slots": True,
            "physical_slot_randomization_start_iteration": 0,
            "physical_slot_randomization_warmup_iterations": 0,
        }
        self.events.depth_distill_dynamic_obstacles = EventTerm(
            func=mdp.move_dynamic_play_obstacles,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={
                "obstacle_names": OBSTACLE_NAMES,
                "obstacle_z": OBSTACLE_Z,
                "longitudinal_speed_range": DEPTH_DISTILL_DYNAMIC_SPEED_RANGE,
                "lateral_speed_max": DEPTH_DISTILL_DYNAMIC_LATERAL_SPEED,
                "longitudinal_extent": DEPTH_DISTILL_DYNAMIC_LONGITUDINAL_EXTENT,
                "lateral_extent": DEPTH_DISTILL_DYNAMIC_LATERAL_EXTENT,
                "min_inter_obstacle_dist": DEPTH_DISTILL_MIN_INTER_OBSTACLE_DIST,
                "velocity_resample_interval_range": DEPTH_DISTILL_DYNAMIC_SPEED_CHANGE_INTERVAL,
                "random_trajectory_fraction": DEPTH_DISTILL_DYNAMIC_WANDER_FRACTION,
                "goal_exclusion_radius": NAV_GOAL_EXCLUSION_RADIUS,
                "start_iteration": DEPTH_DISTILL_DYNAMIC_START_ITERATION,
                "warmup_iterations": DEPTH_DISTILL_DYNAMIC_WARMUP_ITERATIONS,
                "steps_per_iteration": CURRICULUM_STEPS_PER_ITERATION,
            },
        )


@configclass
class Go2wNavDepthRLDistillEnvCfg_PLAY(Go2wNavDepthRLDistillEnvCfg):
    """Play/eval env for the depth-camera student distillation."""

    scene: DepthObstaclePlaySceneCfg = DepthObstaclePlaySceneCfg(num_envs=16, env_spacing=5.0)

    def __post_init__(self):
        super().__post_init__()
        _configure_play_common(self)
        self.events.depth_distill_dynamic_obstacles = None
        self.observations.student_state.enable_corruption = False
        self.observations.student_depth.enable_corruption = False
        _configure_play_obstacle_obs(self.observations.teacher)


# =============================================================================
# Hospital play environment (test existing Nav Teacher in hospital corridors)
# =============================================================================

# Default hospital corridor layout for play testing.
# Derived from HOSPITAL_LAYOUT_TEMPLATES["main_corridor"]:
#   corridor_kind="l_corridor", leg_length=12.0, corridor_width=2.6
HOSPITAL_PLAY_CORRIDOR_KIND = "l_corridor"
HOSPITAL_PLAY_LEG_LENGTH = 12.0
HOSPITAL_PLAY_CORRIDOR_WIDTH = 2.6
HOSPITAL_PLAY_WALL_THICKNESS = 0.20
HOSPITAL_PLAY_WALL_HEIGHT = 2.40
HOSPITAL_PLAY_EPISODE_LENGTH_S = 90.0
HOSPITAL_PLAY_ENV_SPACING = 30.0
HOSPITAL_PLAY_GOAL_DONE_RADIUS = 0.70
# Number of dynamic obstacles placed inside the corridor (excludes wall slots).
HOSPITAL_PLAY_DYNAMIC_OBSTACLE_COUNT = 12


def _shape_kind_from_id(shape_id: int) -> str:
    """Map obstacle metadata shape ids back to scene shape names."""
    if shape_id == OBSTACLE_SHAPE_CYLINDER:
        return "cylinder"
    if shape_id == OBSTACLE_SHAPE_CONE:
        return "cone"
    return "cuboid"


def _hospital_structured_slot_tables(
    corridor_kind: str,
    leg_length: float,
    corridor_width: float,
    wall_thickness: float,
    corridor_turn_length: float | None = None,
) -> dict[str, object]:
    """Return physical slot tables with walls followed by the hospital actor palette."""
    wall_specs = mdp.structured_corridor_wall_specs(
        corridor_kind,
        leg_length,
        corridor_width,
        wall_thickness,
        corridor_turn_length,
    )
    wall_count = len(wall_specs)
    slot_count = len(PLAY_OBSTACLE_NAMES)

    shape_ids = [OBSTACLE_SHAPE_CUBOID] * slot_count
    widths = [OBSTACLE_SIZE[0]] * slot_count
    depths = [OBSTACLE_SIZE[1]] * slot_count
    heights = [OBSTACLE_SIZE[2]] * slot_count
    labels = ["chair"] * slot_count
    colors = [HOSPITAL_LABEL_COLORS.get("chair", HOSPITAL_DEFAULT_COLOR)] * slot_count

    for slot_idx, (_, _, _, wall_length, wall_depth) in enumerate(wall_specs):
        shape_ids[slot_idx] = OBSTACLE_SHAPE_CUBOID
        widths[slot_idx] = wall_length
        depths[slot_idx] = wall_depth
        heights[slot_idx] = HOSPITAL_PLAY_WALL_HEIGHT
        labels[slot_idx] = "wall"
        colors[slot_idx] = HOSPITAL_LABEL_COLORS.get("wall", HOSPITAL_DEFAULT_COLOR)

    palette_slots = slot_count - wall_count
    for palette_idx in range(palette_slots):
        slot_idx = wall_count + palette_idx
        shape_ids[slot_idx] = HOSPITAL_PLAY_OBSTACLE_SHAPE_IDS[palette_idx]
        widths[slot_idx] = HOSPITAL_PLAY_OBSTACLE_WIDTHS[palette_idx]
        depths[slot_idx] = HOSPITAL_PLAY_OBSTACLE_DEPTHS[palette_idx]
        heights[slot_idx] = HOSPITAL_PLAY_OBSTACLE_HEIGHTS[palette_idx]
        labels[slot_idx] = HOSPITAL_PLAY_OBSTACLE_LABELS[palette_idx]
        colors[slot_idx] = HOSPITAL_LABEL_COLORS.get(labels[slot_idx], HOSPITAL_DEFAULT_COLOR)

    center_zs = tuple(height * 0.5 + OBSTACLE_GROUND_CLEARANCE for height in heights)
    return {
        "wall_count": wall_count,
        "shape_ids": tuple(shape_ids),
        "widths": tuple(widths),
        "depths": tuple(depths),
        "heights": tuple(heights),
        "center_zs": center_zs,
        "labels": tuple(labels),
        "colors": tuple(colors),
    }


def _apply_hospital_obstacle_asset_overrides(scene, slot_tables: dict[str, object]) -> None:
    """Make hospital physical assets match the slot metadata seen by depth sensors."""
    shape_ids = slot_tables["shape_ids"]
    widths = slot_tables["widths"]
    depths = slot_tables["depths"]
    heights = slot_tables["heights"]
    colors = slot_tables["colors"]
    for slot_idx, obstacle_name in enumerate(PLAY_OBSTACLE_NAMES):
        setattr(
            scene,
            obstacle_name,
            make_play_obstacle_cfg(
                obstacle_name,
                slot_idx,
                _shape_kind_from_id(shape_ids[slot_idx]),
                (widths[slot_idx], depths[slot_idx]),
                heights[slot_idx],
                colors[slot_idx],
            ),
        )


def _include_hospital_ramp_in_depth_camera(scene) -> None:
    """Expose both ramp halves to depth without adding them to obstacle contacts/rewards."""
    scene.depth_camera.mesh_prim_paths = [
        "/World/ground",
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/obstacle_.*",
            track_mesh_transforms=True,
            is_shared=True,
        ),
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/hospital_ramp",
            track_mesh_transforms=True,
            is_shared=True,
        ),
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/hospital_ramp_b",
            track_mesh_transforms=True,
            is_shared=True,
        ),
    ]


def _hospital_group_registry(wall_count: int) -> list[dict]:
    """Build physical group relationships from dynamic-palette relation pairs."""
    return [
        {
            "relation_type": rel,
            "leader_name": f"obstacle_{wall_count + leader}",
            "follower_name": f"obstacle_{wall_count + follower}",
        }
        for leader, follower, rel in HOSPITAL_PLAY_GROUP_PAIRS
    ]


def _hospital_motion_slot_params(
    wall_count: int,
    slot_tables: dict,
    group_registry: list[dict],
    queue_groups: list[dict] | None = None,
    seated_groups: list[dict] | None = None,
) -> dict:
    """Return moving actors plus read-only static blockers for hospital motion."""
    labels = list(slot_tables["labels"])
    center_zs = tuple(slot_tables["center_zs"])
    keep_names: set[str] = set()

    for slot_idx in range(wall_count, len(PLAY_OBSTACLE_NAMES)):
        label = labels[slot_idx]
        label_spec = _hospital_specs.HOSPITAL_LABEL_SPECS.get(label)
        if label_spec is None:
            continue
        if label_spec.motion_profile != "static" or label_spec.category in {"furniture", "misc"}:
            keep_names.add(PLAY_OBSTACLE_NAMES[slot_idx])

    for group in group_registry:
        keep_names.add(group["leader_name"])
        keep_names.add(group["follower_name"])
    for queue_group in queue_groups or ():
        keep_names.update(queue_group.get("names", ()))
    for seated_group in seated_groups or ():
        name = seated_group.get("name")
        if name is not None:
            keep_names.add(name)

    indices = [
        slot_idx
        for slot_idx, name in enumerate(PLAY_OBSTACLE_NAMES)
        if slot_idx >= wall_count and name in keep_names
    ]
    return {
        "obstacle_names": [PLAY_OBSTACLE_NAMES[slot_idx] for slot_idx in indices],
        "obstacle_labels": [labels[slot_idx] for slot_idx in indices],
        "obstacle_indices": indices,
        "obstacle_center_zs": tuple(center_zs[slot_idx] for slot_idx in indices),
    }


def _structured_path_update_event() -> EventTerm:
    """Update the rolling local waypoint every step for structured routes."""
    return EventTerm(
        func=mdp.update_navigation_path_waypoint_event,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "lookahead_distance": 1.25,
            "waypoint_reach_radius": 0.45,
            "adaptive_lookahead": True,
            "lookahead_min": 0.6,
            "curvature_scan_horizon": 2.5,
            "curvature_threshold": 0.3,
        },
    )


def _set_structured_goal_termination(cfg, position_threshold: float) -> None:
    """Terminate structured play when the robot reaches the final path goal."""
    cfg.terminations.structured_goal_reached = DoneTerm(
        func=mdp.navigation_path_final_goal_reached,
        params={"position_threshold": position_threshold},
    )


def _configure_hospital_structured_depth_play(
    cfg,
    *,
    corridor_kind: str,
    leg_length: float,
    corridor_width: float,
    wall_thickness: float,
    dynamic_obstacle_count: int,
    speed_scale: float = 1.0,
    semantic_local_poses: tuple[tuple[int, float, float, float], ...] | None = None,
    queue_groups: list[dict] | None = None,
    seated_groups: list[dict] | None = None,
    ramp_local_pose: tuple[float, ...] | None = None,
    ramp_b_local_pose: tuple[float, ...] | None = None,
    robot_start_local_xy: tuple[float, float] = (0.0, 0.0),
    min_inter_obstacle_dist: float = 0.80,
) -> None:
    """Configure a depth play env as a structured hospital scene."""
    slot_tables = _hospital_structured_slot_tables(
        corridor_kind,
        leg_length,
        corridor_width,
        wall_thickness,
    )
    _apply_hospital_obstacle_asset_overrides(cfg.scene, slot_tables)
    wall_count = int(slot_tables["wall_count"])
    cfg.events.reset_obstacles.func = mdp.reset_structured_astar_corridor
    cfg.events.reset_obstacles.params = {
        "obstacle_names": PLAY_OBSTACLE_NAMES,
        "corridor_kind": corridor_kind,
        "leg_length": leg_length,
        "corridor_width": corridor_width,
        "wall_thickness": wall_thickness,
        "dynamic_obstacle_count": dynamic_obstacle_count,
        "obstacle_z": OBSTACLE_Z,
        "min_inter_obstacle_dist": min_inter_obstacle_dist,
        "obstacle_radius_margin": NAV_OBSTACLE_RADIUS_MARGIN,
        "fixed_obstacle_shape_ids": slot_tables["shape_ids"],
        "fixed_obstacle_widths": slot_tables["widths"],
        "fixed_obstacle_depths": slot_tables["depths"],
        "fixed_obstacle_center_zs": slot_tables["center_zs"],
        "obstacle_labels": slot_tables["labels"],
        "fixed_obstacle_local_poses": semantic_local_poses,
        "robot_start_local_xy": robot_start_local_xy,
        "ramp_asset_name": HOSPITAL_RAMP_ASSET_NAME,
        "ramp_local_pose": ramp_local_pose,
        "ramp_b_asset_name": HOSPITAL_RAMP_B_ASSET_NAME,
        "ramp_b_local_pose": ramp_b_local_pose,
        "randomize_obstacle_yaw": NAV_RANDOMIZE_OBSTACLE_YAW,
        "obstacle_yaw_range": NAV_OBSTACLE_YAW_RANGE,
        "robot_inflation": 0.50,
        "clearance_cost_weight": 3.0,
        "clearance_cost_sigma": 0.60,
        "corner_rounding": True,
        "corner_radius": 0.80,
        "goal_exclusion_radius": NAV_GOAL_EXCLUSION_RADIUS,
        "dynamic_start_exclusion_radius": 1.8,
        "dynamic_robot_keepout_radius": 1.25,
        "lookahead_distance": 1.25,
        "waypoint_reach_radius": 0.45,
        "adaptive_lookahead": True,
        "lookahead_min": 0.6,
        "curvature_scan_horizon": 2.5,
        "curvature_threshold": 0.3,
    }

    cfg.events.navigation_path_update = _structured_path_update_event()
    cfg.events.hospital_velocity_resample = None
    cfg.events.hospital_group_update = None
    group_registry = _hospital_group_registry(wall_count)
    motion_slot_params = _hospital_motion_slot_params(
        wall_count,
        slot_tables,
        group_registry,
        queue_groups=queue_groups,
        seated_groups=seated_groups,
    )
    motion_params = {
        **motion_slot_params,
        "group_registry": group_registry,
        "min_inter_obstacle_dist": 0.25,
        "active_distance": 24.0,
        "goal_exclusion_radius": NAV_GOAL_EXCLUSION_RADIUS,
        "robot_keepout_radius": 1.25,
    }
    if queue_groups is not None:
        motion_params["queue_groups"] = queue_groups
    if seated_groups is not None:
        motion_params["seated_groups"] = seated_groups
    if speed_scale != 1.0:
        motion_params["speed_scale"] = speed_scale
    cfg.events.hospital_dynamic_motion = EventTerm(
        func=_hospital_events.move_hospital_dynamic_obstacles,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params=motion_params,
    )


@configclass
class Go2wHospitalPlayEnvCfg(Go2wNavTeacherEnvCfg_PLAY):
    """Play/eval env that runs the Nav Teacher policy in a hospital-style corridor.

    Uses ``reset_structured_astar_corridor`` to lay out an L-shaped corridor
    (main_corridor template: leg_length=12 m, width=2.6 m) with hospital-scale
    obstacle footprints (patients, wheelchairs, carts, beds).  No new training
    is required — load any existing Nav Teacher checkpoint directly.
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = HOSPITAL_PLAY_EPISODE_LENGTH_S
        self.scene.env_spacing = HOSPITAL_PLAY_ENV_SPACING
        slot_tables = _hospital_structured_slot_tables(
            HOSPITAL_PLAY_CORRIDOR_KIND,
            HOSPITAL_PLAY_LEG_LENGTH,
            HOSPITAL_PLAY_CORRIDOR_WIDTH,
            HOSPITAL_PLAY_WALL_THICKNESS,
        )
        _apply_hospital_obstacle_asset_overrides(self.scene, slot_tables)
        _wall_count = int(slot_tables["wall_count"])
        # _configure_play_common already called by parent; redo obstacle event
        # to use the structured corridor reset instead of the flat scatter reset.
        self.events.reset_obstacles.func = mdp.reset_structured_astar_corridor
        self.events.reset_obstacles.params = {
            "obstacle_names": PLAY_OBSTACLE_NAMES,
            "corridor_kind": HOSPITAL_PLAY_CORRIDOR_KIND,
            "leg_length": HOSPITAL_PLAY_LEG_LENGTH,
            "corridor_width": HOSPITAL_PLAY_CORRIDOR_WIDTH,
            "wall_thickness": HOSPITAL_PLAY_WALL_THICKNESS,
            "dynamic_obstacle_count": HOSPITAL_PLAY_DYNAMIC_OBSTACLE_COUNT,
            "obstacle_z": OBSTACLE_Z,
            "min_inter_obstacle_dist": 0.80,
            "obstacle_radius_margin": NAV_OBSTACLE_RADIUS_MARGIN,
            "fixed_obstacle_shape_ids": slot_tables["shape_ids"],
            "fixed_obstacle_widths": slot_tables["widths"],
            "fixed_obstacle_depths": slot_tables["depths"],
            "fixed_obstacle_center_zs": slot_tables["center_zs"],
            "obstacle_labels": slot_tables["labels"],
            "randomize_obstacle_yaw": NAV_RANDOMIZE_OBSTACLE_YAW,
            "obstacle_yaw_range": NAV_OBSTACLE_YAW_RANGE,
            "goal_exclusion_radius": NAV_GOAL_EXCLUSION_RADIUS,
            "dynamic_start_exclusion_radius": 1.8,
            "dynamic_robot_keepout_radius": 1.25,
            "lookahead_distance": 1.25,
            "waypoint_reach_radius": 0.45,
            "adaptive_lookahead": True,
            "lookahead_min": 0.6,
            "curvature_scan_horizon": 2.5,
            "curvature_threshold": 0.3,
        }
        # Redirect privileged obs to the full play obstacle list.
        _configure_play_obstacle_obs(self.observations.policy)

        self.events.navigation_path_update = _structured_path_update_event()
        _set_structured_goal_termination(self, HOSPITAL_PLAY_GOAL_DONE_RADIUS)
        self.events.hospital_velocity_resample = None
        self.events.hospital_group_update = None
        group_registry = _hospital_group_registry(_wall_count)
        motion_slot_params = _hospital_motion_slot_params(_wall_count, slot_tables, group_registry)
        self.events.hospital_dynamic_motion = EventTerm(
            func=_hospital_events.move_hospital_dynamic_obstacles,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={
                **motion_slot_params,
                "group_registry": group_registry,
                "min_inter_obstacle_dist": 0.25,
                "active_distance": 24.0,
                "goal_exclusion_radius": NAV_GOAL_EXCLUSION_RADIUS,
                "robot_keepout_radius": 1.25,
            },
        )


# =============================================================================
# Hospital play environment — Depth Camera Student
# =============================================================================


@configclass
class Go2wHospitalDepthPlayEnvCfg(Go2wNavDepthRLDistillEnvCfg_PLAY):
    """Play/eval env for the depth-camera student in a hospital-style corridor.

    Shares depth camera and student obs (state + depth stack) with the standard
    depth-distillation play env.  Replaces the flat scatter reset with
    ``reset_structured_astar_corridor`` using hospital-scale obstacle footprints.
    No new training required — load any existing depth-student checkpoint.
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = HOSPITAL_PLAY_EPISODE_LENGTH_S
        self.scene.env_spacing = HOSPITAL_PLAY_ENV_SPACING
        slot_tables = _hospital_structured_slot_tables(
            HOSPITAL_PLAY_CORRIDOR_KIND,
            HOSPITAL_PLAY_LEG_LENGTH,
            HOSPITAL_PLAY_CORRIDOR_WIDTH,
            HOSPITAL_PLAY_WALL_THICKNESS,
        )
        _apply_hospital_obstacle_asset_overrides(self.scene, slot_tables)
        _wall_count = int(slot_tables["wall_count"])
        self.events.reset_obstacles.func = mdp.reset_structured_astar_corridor
        self.events.reset_obstacles.params = {
            "obstacle_names": PLAY_OBSTACLE_NAMES,
            "corridor_kind": HOSPITAL_PLAY_CORRIDOR_KIND,
            "leg_length": HOSPITAL_PLAY_LEG_LENGTH,
            "corridor_width": HOSPITAL_PLAY_CORRIDOR_WIDTH,
            "wall_thickness": HOSPITAL_PLAY_WALL_THICKNESS,
            "dynamic_obstacle_count": HOSPITAL_PLAY_DYNAMIC_OBSTACLE_COUNT,
            "obstacle_z": OBSTACLE_Z,
            "min_inter_obstacle_dist": 0.80,
            "obstacle_radius_margin": NAV_OBSTACLE_RADIUS_MARGIN,
            "fixed_obstacle_shape_ids": slot_tables["shape_ids"],
            "fixed_obstacle_widths": slot_tables["widths"],
            "fixed_obstacle_depths": slot_tables["depths"],
            "fixed_obstacle_center_zs": slot_tables["center_zs"],
            "obstacle_labels": slot_tables["labels"],
            "randomize_obstacle_yaw": NAV_RANDOMIZE_OBSTACLE_YAW,
            "obstacle_yaw_range": NAV_OBSTACLE_YAW_RANGE,
            "goal_exclusion_radius": NAV_GOAL_EXCLUSION_RADIUS,
            "dynamic_start_exclusion_radius": 1.8,
            "dynamic_robot_keepout_radius": 1.25,
            "lookahead_distance": 1.25,
            "waypoint_reach_radius": 0.45,
            "adaptive_lookahead": True,
            "lookahead_min": 0.6,
            "curvature_scan_horizon": 2.5,
            "curvature_threshold": 0.3,
        }

        self.events.navigation_path_update = _structured_path_update_event()
        _set_structured_goal_termination(self, HOSPITAL_PLAY_GOAL_DONE_RADIUS)
        self.events.hospital_velocity_resample = None
        self.events.hospital_group_update = None
        group_registry = _hospital_group_registry(_wall_count)
        motion_slot_params = _hospital_motion_slot_params(_wall_count, slot_tables, group_registry)
        self.events.hospital_dynamic_motion = EventTerm(
            func=_hospital_events.move_hospital_dynamic_obstacles,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={
                **motion_slot_params,
                "group_registry": group_registry,
                "min_inter_obstacle_dist": 0.25,
                "active_distance": 24.0,
                "goal_exclusion_radius": NAV_GOAL_EXCLUSION_RADIUS,
                "robot_keepout_radius": 1.25,
            },
        )


# =============================================================================
# Hospital ward play environment — Depth Camera Student
# =============================================================================

# Hospital ward: long main corridor (3× leg_length) with two side branches.
# Branch 1 (at x=leg_length) is a dead-end arm populated with dynamic obstacles.
# Branch 2 (at x=2×leg_length) is the goal arm the robot navigates into.
# Wall slots are derived from the structured corridor specs; remaining slots use
# the hospital actor palette, so wall count changes do not discard key actors.
HOSPITAL_WARD_CORRIDOR_KIND = "hospital_ward"
HOSPITAL_WARD_LEG_LENGTH = 10.0    # total main = 30 m, each branch = 10 m
HOSPITAL_WARD_CORRIDOR_WIDTH = 3.0  # patient-care corridor with bed/cart passing room
HOSPITAL_WARD_WALL_THICKNESS = 0.25
HOSPITAL_WARD_DYNAMIC_OBSTACLE_COUNT = 16  # includes child/dog, pushed wheelchair/cart/gurney, self wheelchair, IV pole
_HOSPITAL_WARD_WALL_COUNT = 11             # 11 walls (open left entrance, no end cap at x=0)
HOSPITAL_WARD_EPISODE_LENGTH_S = 140.0
HOSPITAL_WARD_ENV_SPACING = 46.0
HOSPITAL_WARD_GOAL_DONE_RADIUS = 0.80


@configclass
class Go2wHospitalWardDepthPlayEnvCfg(Go2wNavDepthRLDistillEnvCfg_PLAY):
    """Depth-student play/eval env in a hospital ward floor layout.

    The ward has a 30 m main corridor with two perpendicular branches (each 10 m),
    forming two T-junctions the robot navigates through.  Branch 1 is a dead-end
    populated with dynamic obstacles that can spill into the main corridor.  Goal
    is the tip of branch 2.  Corridor width is 3.0 m, wall thickness is 0.25 m.
    Dynamic actors use the hospital label palette at 0.7× speed.
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = HOSPITAL_WARD_EPISODE_LENGTH_S
        self.scene.env_spacing = HOSPITAL_WARD_ENV_SPACING
        slot_tables = _hospital_structured_slot_tables(
            HOSPITAL_WARD_CORRIDOR_KIND,
            HOSPITAL_WARD_LEG_LENGTH,
            HOSPITAL_WARD_CORRIDOR_WIDTH,
            HOSPITAL_WARD_WALL_THICKNESS,
        )
        _apply_hospital_obstacle_asset_overrides(self.scene, slot_tables)
        _wall_count = int(slot_tables["wall_count"])
        self.events.reset_obstacles.func = mdp.reset_structured_astar_corridor
        self.events.reset_obstacles.params = {
            "obstacle_names": PLAY_OBSTACLE_NAMES,
            "corridor_kind": HOSPITAL_WARD_CORRIDOR_KIND,
            "leg_length": HOSPITAL_WARD_LEG_LENGTH,
            "corridor_width": HOSPITAL_WARD_CORRIDOR_WIDTH,
            "wall_thickness": HOSPITAL_WARD_WALL_THICKNESS,
            "dynamic_obstacle_count": HOSPITAL_WARD_DYNAMIC_OBSTACLE_COUNT,
            "obstacle_z": OBSTACLE_Z,
            "min_inter_obstacle_dist": 1.00,
            "obstacle_radius_margin": NAV_OBSTACLE_RADIUS_MARGIN,
            "fixed_obstacle_shape_ids": slot_tables["shape_ids"],
            "fixed_obstacle_widths": slot_tables["widths"],
            "fixed_obstacle_depths": slot_tables["depths"],
            "fixed_obstacle_center_zs": slot_tables["center_zs"],
            "obstacle_labels": slot_tables["labels"],
            "randomize_obstacle_yaw": NAV_RANDOMIZE_OBSTACLE_YAW,
            "obstacle_yaw_range": NAV_OBSTACLE_YAW_RANGE,
            "goal_exclusion_radius": NAV_GOAL_EXCLUSION_RADIUS,
            "dynamic_start_exclusion_radius": 1.8,
            "dynamic_robot_keepout_radius": 1.25,
            "lookahead_distance": 1.25,
            "waypoint_reach_radius": 0.45,
            "adaptive_lookahead": True,
            "lookahead_min": 0.6,
            "curvature_scan_horizon": 2.5,
            "curvature_threshold": 0.3,
        }

        self.events.navigation_path_update = _structured_path_update_event()
        _set_structured_goal_termination(self, HOSPITAL_WARD_GOAL_DONE_RADIUS)
        self.events.hospital_velocity_resample = None
        self.events.hospital_group_update = None
        group_registry = _hospital_group_registry(_wall_count)
        motion_slot_params = _hospital_motion_slot_params(_wall_count, slot_tables, group_registry)
        self.events.hospital_dynamic_motion = EventTerm(
            func=_hospital_events.move_hospital_dynamic_obstacles,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={
                **motion_slot_params,
                "group_registry": group_registry,
                "speed_scale": 0.7,
                "min_inter_obstacle_dist": 0.30,
                "active_distance": 24.0,
                "goal_exclusion_radius": NAV_GOAL_EXCLUSION_RADIUS,
                "robot_keepout_radius": 1.25,
            },
        )


# =============================================================================
# Hospital full-floor play environment — Depth Camera Student
# (constants imported from mdp/hospital/floor.py)
# =============================================================================


@configclass
class Go2wHospitalFloorDepthPlayEnvCfg(Go2wNavDepthRLDistillEnvCfg_PLAY):
    """Depth-student play/eval env for a combined hospital floor.

    Includes reception queueing, waiting bench occupancy, doorway crossing,
    ward/service flow with a pushed patient gurney, and a mild ramp connector.
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = HOSPITAL_FLOOR_EPISODE_LENGTH_S
        self.scene.env_spacing = HOSPITAL_FLOOR_ENV_SPACING
        slot_tables = _hospital_structured_slot_tables(
            HOSPITAL_FLOOR_CORRIDOR_KIND,
            HOSPITAL_FLOOR_LEG_LENGTH,
            HOSPITAL_FLOOR_CORRIDOR_WIDTH,
            HOSPITAL_FLOOR_WALL_THICKNESS,
        )
        wall_count = int(slot_tables["wall_count"])
        _configure_hospital_structured_depth_play(
            self,
            corridor_kind=HOSPITAL_FLOOR_CORRIDOR_KIND,
            leg_length=HOSPITAL_FLOOR_LEG_LENGTH,
            corridor_width=HOSPITAL_FLOOR_CORRIDOR_WIDTH,
            wall_thickness=HOSPITAL_FLOOR_WALL_THICKNESS,
            dynamic_obstacle_count=HOSPITAL_FLOOR_DYNAMIC_OBSTACLE_COUNT,
            speed_scale=0.8,
            semantic_local_poses=_hospital_floor_semantic_local_poses(wall_count, HOSPITAL_FLOOR_LEG_LENGTH),
            queue_groups=_hospital_floor_queue_groups(wall_count),
            seated_groups=_hospital_floor_seated_groups(wall_count),
            ramp_local_pose=HOSPITAL_FLOOR_RAMP_LOCAL_POSE,
            ramp_b_local_pose=HOSPITAL_FLOOR_RAMP_B_LOCAL_POSE,
            robot_start_local_xy=HOSPITAL_FLOOR_ROBOT_START_LOCAL_XY,
            min_inter_obstacle_dist=1.05,
        )
        _set_structured_goal_termination(self, HOSPITAL_FLOOR_GOAL_DONE_RADIUS)
        _include_hospital_ramp_in_depth_camera(self.scene)
