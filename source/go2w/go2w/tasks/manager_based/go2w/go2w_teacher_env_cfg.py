# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Go2-W teacher environment for privileged learning.

Extends the baseline flat-terrain environment with:
  - A "teacher" observation group containing privileged information
    (contact forces, base height) unavailable on the real robot.
  - A "student" observation group identical to the baseline "policy" group
    (proprioception only), used during distillation.

Training flow:
  1. Train teacher with PPO:   train.py --task Teacher-Go2w-v0
  2. Distill into student:     train.py --task Distill-Go2w-v0
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .go2w_env_cfg import Go2wEnvCfg


# =============================================================================
# Teacher environment — PPO training with privileged observations
# =============================================================================


@configclass
class TeacherObservationsCfg:
    """Observations for teacher PPO training.

    The "policy" group includes both proprioception and privileged terms.
    The teacher sees everything available in simulation.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Teacher policy obs = proprioception + privileged info."""

        # -- Proprioception (same as baseline) --
        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel,      noise=Unoise(n_min=-0.1,  n_max=0.1))
        base_ang_vel      = ObsTerm(func=mdp.base_ang_vel,      noise=Unoise(n_min=-0.2,  n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity,  noise=Unoise(n_min=-0.05, n_max=0.05))

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5,  n_max=1.5))

        actions = ObsTerm(func=mdp.last_action)

        # -- Privileged (simulation-only) --
        # Base height above ground (1 dim). Helps with posture recovery.
        base_height = ObsTerm(func=mdp.base_pos_z)
        # Net contact wrench on each foot link (4 feet x 6 = 24 dim).
        # Provides ground truth contact state for stance control.
        foot_contact_wrench = ObsTerm(
            func=mdp.body_incoming_wrench,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_foot"])},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class Go2wTeacherEnvCfg(Go2wEnvCfg):
    """Teacher environment: baseline + privileged observations for PPO training."""

    observations: TeacherObservationsCfg = TeacherObservationsCfg()


@configclass
class Go2wTeacherEnvCfg_PLAY(Go2wTeacherEnvCfg):
    """Teacher evaluation environment."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs    = 16
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot    = None
        self.events.add_base_mass = None


# =============================================================================
# Distillation environment — student learns to mimic teacher
# =============================================================================


@configclass
class DistillObservationsCfg:
    """Observations for distillation: separate student and teacher groups.

    The DistillationRunner in RSL-RL expects exactly these group names.
    Both groups are computed every step; teacher actions are the learning
    target, student actions are the prediction.
    """

    @configclass
    class StudentCfg(ObsGroup):
        """Student obs = proprioception only (same as baseline policy)."""

        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel,      noise=Unoise(n_min=-0.1,  n_max=0.1))
        base_ang_vel      = ObsTerm(func=mdp.base_ang_vel,      noise=Unoise(n_min=-0.2,  n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity,  noise=Unoise(n_min=-0.05, n_max=0.05))

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5,  n_max=1.5))

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class TeacherCfg(ObsGroup):
        """Teacher obs for distillation.

        Must match the obs dimensions of the loaded teacher checkpoint.
        For pipeline validation with baseline checkpoint: proprioception only (60 dim).
        For real privileged teacher: add base_height + foot_contact_wrench here.
        """

        # -- Proprioception (matches baseline "policy" group) --
        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel,      noise=Unoise(n_min=-0.1,  n_max=0.1))
        base_ang_vel      = ObsTerm(func=mdp.base_ang_vel,      noise=Unoise(n_min=-0.2,  n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity,  noise=Unoise(n_min=-0.05, n_max=0.05))

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5,  n_max=1.5))

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False  # Teacher sees clean data
            self.concatenate_terms = True

    student: StudentCfg = StudentCfg()
    teacher: TeacherCfg = TeacherCfg()


@configclass
class Go2wDistillEnvCfg(Go2wEnvCfg):
    """Distillation environment: student and teacher obs groups for behavior cloning."""

    observations: DistillObservationsCfg = DistillObservationsCfg()


@configclass
class Go2wDistillEnvCfg_PLAY(Go2wDistillEnvCfg):
    """Distillation evaluation environment (uses student obs only)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs    = 16
        self.scene.env_spacing = 3.0
        self.observations.student.enable_corruption = False
        self.events.push_robot    = None
        self.events.add_base_mass = None
