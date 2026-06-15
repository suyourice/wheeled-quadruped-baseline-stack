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
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCameraCfg, MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.utils import configclass

from . import mdp
from .mdp.hospital import events as _hospital_events
from .mdp.hospital import specs as _hospital_specs
from .mdp.hospital.specs import *  # noqa: F401, F403
from .mdp.hospital.floor import (
    make_hospital_ramp_cfg as _make_hospital_ramp_cfg,
    make_hospital_ramp_b_cfg as _make_hospital_ramp_b_cfg,
)
from .go2w_env_cfg import EventCfg, Go2wEnvCfg, Go2wSceneCfg
from .obstacle_rewards_cfg import NavTeacherRewardsCfg
from .obstacle_obs_cfg import NavDepthRLDistillObsCfg, NavRLDistillObsCfg, NavTeacherObsCfg


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
