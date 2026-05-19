# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL configs for Go2-W RL navigation teacher and LiDAR student distillation."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class NavTeacherRunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner for the RL navigation teacher.

    The actor observes 211D privileged HLC navigation features
    (9D proprio + 180D obstacle depth + 16D geometry features + 6D action history)
    and outputs a 3D velocity command. The frozen fast-flat LLC checkpoint is
    loaded by the environment action term, not by the PPO actor.
    """

    num_steps_per_env = 128
    max_iterations = 1500
    save_interval = 100
    experiment_name = "go2w_nav_teacher_rl"
    logger = "wandb"
    wandb_project = "go2w_nav_teacher_rl"

    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.25,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class SimpleNavDistillAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    """Pure action-MSE distillation algorithm."""

    class_name: str = (
        "go2w.tasks.manager_based.go2w.distillation_algorithms:SimpleActionDistillation"
    )
    action_loss_weight: float = 1.0


@configclass
class NavRLTeacherModelCfg(RslRlMLPModelCfg):
    """Privileged teacher model (211D obs -> 3D HLC velocity, no normalization)."""

    class_name: str = "rsl_rl.models:MLPModel"
    obs_normalization: bool = False


@configclass
class NavLiDARStudentModelCfg(RslRlMLPModelCfg):
    """LiDAR student model (189D obs -> 3D HLC velocity, no normalization)."""

    class_name: str = "rsl_rl.models:MLPModel"
    obs_normalization: bool = False
    distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.35)


@configclass
class NavRLDistillRunnerCfg(RslRlDistillationRunnerCfg):
    """Distillation runner: privileged RL teacher → LiDAR student via 3D action MSE."""

    num_steps_per_env = 128
    max_iterations = 400
    save_interval = 50
    experiment_name = "go2w_nav_rl_distill"
    logger = "wandb"
    wandb_project = "go2w_nav_rl_distill"

    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    student = NavLiDARStudentModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
    )
    teacher = NavRLTeacherModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=None,
    )
    algorithm = SimpleNavDistillAlgorithmCfg(
        num_learning_epochs=4,
        learning_rate=5.0e-4,
        gradient_length=8,
        max_grad_norm=1.0,
        loss_type="mse",
        action_loss_weight=1.0,
    )
