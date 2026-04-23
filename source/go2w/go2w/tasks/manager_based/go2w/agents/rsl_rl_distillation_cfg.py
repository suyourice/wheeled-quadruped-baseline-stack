# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agent configs for Teacher PPO training and Student distillation."""

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
class TeacherRunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for training the teacher with privileged observations.

    Same hyperparameters as the baseline PPORunnerCfg, but saves to a
    separate experiment directory (go2w_teacher).
    """

    num_steps_per_env = 96
    max_iterations    = 1500
    save_interval     = 100
    experiment_name   = "go2w_teacher"
    logger            = "wandb"
    wandb_project     = "go2w_teacher_ppo"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class DistillRunnerCfg(RslRlDistillationRunnerCfg):
    """Distillation config for training a student to mimic the teacher.

    The teacher checkpoint is loaded via --load_run / --checkpoint CLI args.
    By default it searches the go2w_teacher experiment directory.
    """

    num_steps_per_env = 96
    max_iterations    = 1000
    save_interval     = 100
    experiment_name   = "go2w_distill"
    logger            = "wandb"
    wandb_project     = "go2w_distill"

    # Distillation runner auto-loads teacher from the latest checkpoint
    load_run        = ".*"
    load_checkpoint = "model_.*.pt"

    student = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.8,
        ),
    )

    teacher = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        obs_normalization=False,
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1.0e-3,
        gradient_length=15,
        max_grad_norm=1.0,
        loss_type="mse",
    )
