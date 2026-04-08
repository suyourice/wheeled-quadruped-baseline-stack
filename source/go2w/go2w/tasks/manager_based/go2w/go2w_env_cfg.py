# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Go2-W flat-terrain locomotion environment.

Control strategy: WHEEL-PRIMARY
  - Primary:   4x wheel velocity  (.*_foot_joint)  -> propulsion & differential yaw
  - Secondary: 12x leg position   (hip/thigh/calf) -> stance & posture
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from go2w.assets import GO2W_CFG  # isort:skip

from . import mdp


# =============================================================================
# Scene
# =============================================================================


@configclass
class Go2wSceneCfg(InteractiveSceneCfg):
    """Flat-terrain scene with the Go2-W wheeled quadruped."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(200.0, 200.0)),
    )

    robot: ArticulationCfg = GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Contact sensor on every link.
    # Used by: base_contact termination, undesired_contacts reward.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


# =============================================================================
# Commands
# =============================================================================


@configclass
class CommandsCfg:
    """Uniform random velocity commands (lin_vel_x, lin_vel_y, ang_vel_z).

    heading_command=False: the policy receives ang_vel_z directly,
    which matches the cmd_vel interface used by ROS navigation stacks.
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 8.0),
        rel_standing_envs=0.1,       # 10% of envs get zero-velocity command (learn to stand still)
        rel_heading_envs=0.0,
        heading_command=False,       # directly command ang_vel_z, no heading-to-yaw conversion
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),  # [m/s] symmetric forward/backward
            lin_vel_y=(-1.0, 1.0),  # [m/s] symmetric left/right
            ang_vel_z=(-1.5, 1.5),  # [rad/s] ~86 deg/s max yaw rate
            heading=(0.0, 0.0),     # unused (heading_command=False)
        ),
    )


# =============================================================================
# Actions
# =============================================================================


@configclass
class ActionsCfg:
    """Hybrid action space: wheels (velocity) + legs (position).

    Total action dim = 4 (wheels) + 12 (legs) = 16.
    """

    # Wheel velocity: policy output [-1, 1] * scale -> target angular velocity
    # wheel radius ~0.086 m -> scale=10 maps action +-1 to +-0.86 m/s linear
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[".*_foot_joint"],
        scale=10.0,
    )

    # Leg position: delta on top of default standing pose
    # scale=0.5 rad keeps deviations small for stable base
    leg_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        scale=0.5,
        use_default_offset=True,
    )


# =============================================================================
# Observations
# =============================================================================


@configclass
class ObservationsCfg:
    """Observation stack for locomotion policy (~60 dims)."""

    @configclass
    class PolicyCfg(ObsGroup):

        # Base kinematics (IMU)
        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel,      noise=Unoise(n_min=-0.1,  n_max=0.1))
        base_ang_vel      = ObsTerm(func=mdp.base_ang_vel,      noise=Unoise(n_min=-0.2,  n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity,  noise=Unoise(n_min=-0.05, n_max=0.05))

        # Velocity command target
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # Joint state: 12 leg + 4 wheel = 16 DOF each
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5,  n_max=1.5))

        # Previous action (temporal feedback for smoother control)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# =============================================================================
# Events (randomisation & disturbances)
# =============================================================================


@configclass
class EventCfg:

    # -- Startup (once at training start) --------------------------------------

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range":  (0.7, 1.2),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range":      (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Mass randomisation covers future D1-T arm (~2.37 kg) + margin
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),  # Go2-W base ~6.9 kg
            "operation": "add",
        },
    )

    # -- Reset (every episode) -------------------------------------------------

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.1, 0.1),
                "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.3, 0.3),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            # Tight range: calf joints span only -2.72 ~ -0.84 rad
            "position_range": (0.8, 1.2),
            "velocity_range": (0.0, 0.0),
        },
    )

    # -- Interval (periodic disturbances) --------------------------------------

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


# =============================================================================
# Rewards
# =============================================================================


@configclass
class RewardsCfg:
    """Reward terms for wheel-primary flat locomotion.

    Design goals:
      1. Track omnidirectional velocity commands (forward/backward/lateral/diagonal + yaw)
      2. Encourage wheel usage over leg movement for propulsion
      3. Maintain stable, upright posture
    """

    # -- Velocity tracking (yaw-aligned / world frame) -------------------------
    #
    # Yaw frame: velocity is projected onto the horizontal plane using only
    # the yaw component of orientation. Immune to roll/pitch tilt distortion.
    #   r = exp( -||v_cmd_xy - v_yaw_xy||^2 / std^2 )
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": 0.4},
    )
    # World frame: yaw rate measured along the gravity-aligned z-axis.
    #   r = exp( -(w_cmd_z - w_world_z)^2 / std^2 )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    # -- Stability penalties ---------------------------------------------------
    #
    # Penalise body tilt (orientation deviation from flat)
    #   penalty = ||projected_gravity_xy||^2
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    # Penalise vertical bounce (should be ~0 on flat ground)
    #   penalty = vz^2
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    # Penalise roll/pitch angular velocity (oscillation damping)
    #   penalty = wx^2 + wy^2
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # -- Wheel usage incentives (legs only) ------------------------------------
    #
    # Penalise leg joint torques — wheels are excluded so they can spin freely.
    # Encourages the policy to rely on wheels for propulsion/yaw rather than
    # walking with legs.
    #   penalty = sum(tau_leg_i^2)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])},
    )
    # Penalise thigh/calf deviating from default standing pose.
    # Complements torque penalty: torques penalise dynamic effort,
    # deviation penalises static drift from nominal posture.
    #   penalty = sum(|theta_i - theta_default_i|)
    joint_deviation_stance = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_joint", ".*_calf_joint"])},
    )
    # Hip allowed to deviate to steer wheel orientation for diagonal/lateral movement,
    # but extreme abduction is still discouraged.
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )

    # -- Wheel contact / spin --------------------------------------------------
    #
    # Penalise wheels that lose ground contact (lifted leg gait should not occur).
    #   penalty = number of wheels not touching the ground
    wheel_contact = RewTerm(
        func=mdp.wheel_contact_penalty,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"])},
    )
    # Penalise wheel spin only when velocity command is exactly zero.
    # No penalty during locomotion so wheels can spin freely.
    #   penalty = sum(w_wheel_i^2)  if cmd == 0, else 0
    wheel_vel_zero_cmd = RewTerm(
        func=mdp.wheel_vel_zero_cmd,
        weight=-0.01,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_foot_joint"]),
        },
    )

    # -- Action smoothness -----------------------------------------------------
    #
    # Penalise rapid action changes between consecutive steps.
    #   penalty = sum((a_t - a_{t-1})^2)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # -- Contact penalty -------------------------------------------------------
    #
    # Penalise thigh/calf links touching the ground (only wheels should contact).
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
            "threshold": 1.0,
        },
    )

    # -- Termination penalty ---------------------------------------------------
    #
    # Large negative reward on non-timeout termination (fall, collapse).
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


# =============================================================================
# Terminations
# =============================================================================


@configclass
class TerminationsCfg:

    # Episode time limit
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Base body touches the ground (fallen over)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )

    # Base height too low (collapsed). Nominal standing ~0.42 m.
    root_height_below_minimum = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.25},
    )


# =============================================================================
# Top-level environment
# =============================================================================


@configclass
class Go2wEnvCfg(ManagerBasedRLEnvCfg):
    """Go2-W flat-terrain velocity-tracking environment (wheel-primary)."""

    scene:        Go2wSceneCfg    = Go2wSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions:      ActionsCfg      = ActionsCfg()
    commands:     CommandsCfg     = CommandsCfg()
    rewards:      RewardsCfg      = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events:       EventCfg        = EventCfg()
    curriculum:   None            = None

    def __post_init__(self) -> None:
        self.sim.dt           = 0.005    # 5 ms physics step
        self.decimation       = 4        # 20 ms policy step
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0

        self.viewer.eye = (6.0, 0.0, 3.0)

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class Go2wEnvCfg_PLAY(Go2wEnvCfg):
    """Evaluation environment: fewer envs, no randomisation."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs    = 16
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot    = None
        self.events.add_base_mass = None
