# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 96           # rollout buffer length per env
    max_iterations    = 1500         # total training iterations
    save_interval     = 100          # checkpoint every N iterations
    experiment_name   = "go2w_flat_wheel"
    logger            = "wandb"
    wandb_project     = "go2w_baseline_ppo"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        # Action dim = 16 (4 wheels + 12 legs); moderate net for flat terrain.
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
class FastFlatRunnerCfg(PPORunnerCfg):
    """PPO config for single-run fast-flat locomotion pre-training."""

    num_steps_per_env = 128           # matches obstacle env rollout horizon
    max_iterations    = 2000
    experiment_name   = "go2w_fast_flat"
    wandb_project     = "go2w_fast_flat"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.30,          # scale=28: 0.30*28*0.086~=0.72 m/s
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
