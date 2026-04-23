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
  1. Train teacher with PPO:   train.py --task Obstacle-Teacher-Go2w-v0
  2. Distill into student:     train.py --task Obstacle-Distill-Go2w-v0
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .go2w_env_cfg import EventCfg, Go2wEnvCfg, Go2wSceneCfg

# -- Constants ----------------------------------------------------------------

NUM_OBSTACLES = 20  # Number of obstacles in the scene. Must match number of obstacle configs in ObstacleSceneCfg and obstacle_names list in observations.
OBSTACLE_SIZE = (0.3, 0.3, 0.5)  # (x, y, z) meters
OBSTACLE_SPAWN_RANGE = {"x": (-2.0, 2.0), "y": (-2.0, 2.0)}
OBSTACLE_NAMES = [f"obstacle_{i}" for i in range(NUM_OBSTACLES)]

LIDAR_MAX_DISTANCE = 20.0  # meters
LIDAR_HORIZONTAL_FOV = (0.0, 360.0)  # full 360 degrees
LIDAR_HORIZONTAL_RES = 2.0  # 2 deg resolution -> 180 rays
LIDAR_CHANNELS = 1  # single horizontal ring (2D LiDAR)
LIDAR_VERTICAL_FOV = (0.0, 0.0)  # single horizontal plane


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
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.1)),  # slightly above base to avoid ground collision
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


# =============================================================================
# Events (randomisation)
# =============================================================================


def _make_obstacle_reset_event(name: str) -> EventTerm:
    """Create a reset event that randomizes an obstacle's position."""
    return EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name),
            "pose_range": {
                "x": OBSTACLE_SPAWN_RANGE["x"],
                "y": OBSTACLE_SPAWN_RANGE["y"],
            },
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        },
    )


@configclass
class ObstacleEventCfg(EventCfg):
    """Events for the obstacle environment.

    Inherits all baseline events and adds obstacle position randomization.
    """

    # Randomize obstacle positions on every reset
    for i in range(NUM_OBSTACLES):
        vars()[f"reset_obstacle_{i}"] = _make_obstacle_reset_event(f"obstacle_{i}")
    del i


# =============================================================================
# Rewards
# =============================================================================


@configclass
class ObstacleRewardsCfg:
    """Rewards for obstacle avoidance.

    Inherits all baseline locomotion rewards and adds obstacle proximity penalty.
    """

    # -- Velocity tracking (same as baseline) ----------------------------------
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": 0.4},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    # -- Stability penalties (same as baseline) --------------------------------
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # -- Wheel usage (same as baseline) ----------------------------------------
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
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )

    # -- Wheel contact / spin (same as baseline) -------------------------------
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

    # -- Action smoothness (same as baseline) ----------------------------------
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # -- Contact penalty (same as baseline) ------------------------------------
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
            "threshold": 1.0,
        },
    )

    # -- Termination penalty (same as baseline) --------------------------------
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # -- Obstacle collision penalty (NEW) --------------------------------------
    # Fire when any obstacle body reports contact force > threshold.
    # Covers all robot body parts (wheels, legs, base) without needing to
    # distinguish ground contact from obstacle contact on the robot side.
    obstacle_collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("obstacle_contacts"),
            "threshold": 1.0, # [N]; adjust based on contact sensor sensitivity and desired penalty frequency
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
            params={"obstacle_names": OBSTACLE_NAMES},
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
            noise=Unoise(n_min=-0.02, n_max=0.02),
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
            params={"obstacle_names": OBSTACLE_NAMES},
        )

        def __post_init__(self):
            self.enable_corruption = False  # Teacher sees clean data
            self.concatenate_terms = True

    student: StudentCfg = StudentCfg()
    teacher: TeacherCfg = TeacherCfg()


# =============================================================================
# Top-level environment configs
# =============================================================================


@configclass
class Go2wObstacleTeacherEnvCfg(Go2wEnvCfg):
    """Obstacle environment for Teacher PPO training with privileged obs."""

    scene:        ObstacleSceneCfg       = ObstacleSceneCfg(num_envs=4096, env_spacing=5.0)
    observations: ObstacleTeacherObsCfg  = ObstacleTeacherObsCfg()
    rewards:      ObstacleRewardsCfg     = ObstacleRewardsCfg()
    events:       ObstacleEventCfg       = ObstacleEventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scene.lidar is not None:
            self.scene.lidar.update_period = self.decimation * self.sim.dt
        # Reduce command speed for obstacle environment (slower = more time to react)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)


@configclass
class Go2wObstacleTeacherEnvCfg_PLAY(Go2wObstacleTeacherEnvCfg):
    """Teacher evaluation environment."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs    = 16
        self.scene.env_spacing = 5.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot    = None
        self.events.add_base_mass = None


@configclass
class Go2wObstacleDistillEnvCfg(Go2wObstacleTeacherEnvCfg):
    """Obstacle distillation environment: inherits scene/rewards/events from Teacher.

    Only observations differ: two groups (student LiDAR + teacher privileged)
    instead of the single teacher policy group.
    """

    observations: ObstacleDistillObsCfg = ObstacleDistillObsCfg()


@configclass
class Go2wObstacleDistillEnvCfg_PLAY(Go2wObstacleDistillEnvCfg):
    """Distillation evaluation environment (uses student obs only)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs    = 16
        self.scene.env_spacing = 5.0
        self.observations.student.enable_corruption = False
        self.events.push_robot    = None
        self.events.add_base_mass = None
