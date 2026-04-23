# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agent configs for obstacle avoidance Teacher and Student training."""

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
class ObstacleTeacherRunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for training the obstacle-avoidance teacher.

    Uses privileged observations (obstacle positions relative to robot).
    """

    num_steps_per_env = 96
    max_iterations    = 1500
    save_interval     = 100
    experiment_name   = "go2w_obstacle_teacher"
    logger            = "wandb"
    wandb_project     = "go2w_obstacle_teacher"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
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
class ObstacleDistillRunnerCfg(RslRlDistillationRunnerCfg):
    """Distillation config for student with LiDAR observations."""

    num_steps_per_env = 96
    max_iterations    = 1000
    save_interval     = 100
    experiment_name   = "go2w_obstacle_distill"
    logger            = "wandb"
    wandb_project     = "go2w_obstacle_distill"

    load_run        = ".*"
    load_checkpoint = "model_.*.pt"

    student = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.8,
        ),
    )

    teacher = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
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
