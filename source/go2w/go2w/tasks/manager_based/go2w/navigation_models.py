# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal-conditioned target-pose navigation student models with explicit LLC separation."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization, HiddenState, MLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable

from .observation_layout import GOAL_COMMAND_DIM, GOAL_COMMAND_START


class NavigationCommandPolicy(nn.Module):
    """Pure local-navigation policy.

    This module contains only the trainable navigation logic:

        proprio + LiDAR + local goal -> local target pose -> local `(vx, vy, yaw)` command

    It deliberately does not include the locomotion executor, so it can be swapped,
    exported, or reused independently of the LLC.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        *,
        frozen_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        frozen_activation: str = "elu",
        frozen_obs_normalization: bool = False,
        command_obs_start: int = GOAL_COMMAND_START,
        command_obs_dim: int = GOAL_COMMAND_DIM,
        representation_dim: int = 8,
        target_pose_horizon: float = 0.75,
        target_pose_yaw_horizon: float = 0.6,
        target_pose_x_clip: float = 1.5,
        target_pose_y_clip: float = 1.5,
        target_pose_yaw_clip: float = 1.2,
        target_pose_to_vx_gain: float = 1.0,
        target_pose_to_vy_gain: float = 1.0,
        target_pose_to_yaw_gain: float = 1.0,
        side_guidance_lateral_gain: float = 0.25,
        side_guidance_yaw_gain: float = 0.45,
        command_clip_xy: float = 2.0,
        command_clip_yaw: float = 2.0,
    ) -> None:
        super().__init__()
        del frozen_hidden_dims, frozen_activation, frozen_obs_normalization

        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        if len(self.obs_groups) != 1:
            raise ValueError(
                "NavigationCommandPolicy expects one 1D observation group. "
                f"Got groups: {self.obs_groups}"
            )
        self.obs_group_name = self.obs_groups[0]
        # The observation slice at [command_obs_start : command_obs_start + 3]
        # now carries the local goal in base frame: [goal_x_b, goal_y_b, goal_heading_b].
        self.command_obs_start = command_obs_start
        self.command_obs_dim = command_obs_dim
        self.representation_dim = representation_dim
        self.target_pose_horizon = target_pose_horizon
        self.target_pose_yaw_horizon = target_pose_yaw_horizon
        self.target_pose_x_clip = target_pose_x_clip
        self.target_pose_y_clip = target_pose_y_clip
        self.target_pose_yaw_clip = target_pose_yaw_clip
        self.target_pose_to_vx_gain = target_pose_to_vx_gain
        self.target_pose_to_vy_gain = target_pose_to_vy_gain
        self.target_pose_to_yaw_gain = target_pose_to_yaw_gain
        self.side_guidance_lateral_gain = side_guidance_lateral_gain
        self.side_guidance_yaw_gain = side_guidance_yaw_gain
        self.command_clip_xy = command_clip_xy
        self.command_clip_yaw = command_clip_yaw

        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = nn.Identity()

        trunk_hidden_dims = list(hidden_dims)
        if len(trunk_hidden_dims) == 0:
            raise ValueError("NavigationCommandStudent requires at least one hidden dimension.")
        latent_dim = trunk_hidden_dims[-1]
        encoder_hidden_dims = trunk_hidden_dims[:-1]
        self.encoder = MLP(self.obs_dim, latent_dim, encoder_hidden_dims, activation)

        self.rep_head = nn.Linear(latent_dim, self.representation_dim)
        self.command_decoder = MLP(
            self.representation_dim + self.command_obs_dim,
            latent_dim,
            [64],
            activation,
        )
        self.target_pose_head = nn.Linear(latent_dim, 3)
        self.side_head = nn.Linear(latent_dim, 1)

        nn.init.zeros_(self.rep_head.weight)
        nn.init.zeros_(self.rep_head.bias)
        nn.init.zeros_(self.target_pose_head.weight)
        nn.init.zeros_(self.target_pose_head.bias)
        nn.init.zeros_(self.side_head.weight)
        nn.init.zeros_(self.side_head.bias)

    def forward(
        self,
        obs,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del masks, hidden_state, stochastic_output
        navigation_command, _, _, _, _, _ = self._compute_outputs(obs)
        return navigation_command

    def get_aux_outputs(self, obs) -> dict[str, torch.Tensor]:
        navigation_command, target_pose, delta_cmd, side, base_command, obstacle_representation = self._compute_outputs(obs)
        return {
            "navigation_command": navigation_command,
            "target_pose": target_pose,
            "delta_cmd": delta_cmd,
            "side": side,
            "base_command": base_command,
            "obstacle_representation": obstacle_representation,
        }

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        del dones, hidden_state

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def as_jit(self) -> nn.Module:
        return self

    def update_normalization(self, obs) -> None:
        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            self.obs_normalizer.update(self._build_latent_input(obs))

    def _compute_outputs(
        self, obs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_input = self.obs_normalizer(self._build_latent_input(obs))
        latent = self.encoder(latent_input)
        obstacle_representation = self._build_obstacle_representation(latent)
        goal_command = self.get_base_command(obs).clone()
        nominal_command = self._goal_command_to_nominal_command(goal_command)
        decoder_input = torch.cat((goal_command, obstacle_representation), dim=-1)
        head_input = self.command_decoder(decoder_input)

        base_target_pose = self._goal_command_to_base_target_pose(goal_command)
        side = torch.tanh(self.side_head(head_input))

        # The auxiliary side prediction should directly influence the local
        # lateral/yaw correction. Otherwise we supervise left/right decisions
        # but never let that signal steer the command path in hard cases.
        target_pose_delta_logits = self.target_pose_head(head_input)
        target_pose_delta_raw = torch.cat(
            (
                torch.tanh(target_pose_delta_logits[:, 0:1]),
                torch.tanh(
                    target_pose_delta_logits[:, 1:2]
                    + self.side_guidance_lateral_gain * side
                ),
                torch.tanh(
                    target_pose_delta_logits[:, 2:3]
                    + self.side_guidance_yaw_gain * side
                ),
            ),
            dim=-1,
        )
        target_pose_delta = torch.cat(
            (
                self.target_pose_x_clip * target_pose_delta_raw[:, 0:1],
                self.target_pose_y_clip * target_pose_delta_raw[:, 1:2],
                self.target_pose_yaw_clip * target_pose_delta_raw[:, 2:3],
            ),
            dim=-1,
        )
        target_pose = torch.cat(
            (
                (base_target_pose[:, 0:1] + target_pose_delta[:, 0:1]).clamp(
                    -self.target_pose_x_clip, self.target_pose_x_clip
                ),
                (base_target_pose[:, 1:2] + target_pose_delta[:, 1:2]).clamp(
                    -self.target_pose_y_clip, self.target_pose_y_clip
                ),
                (base_target_pose[:, 2:3] + target_pose_delta[:, 2:3]).clamp(
                    -self.target_pose_yaw_clip, self.target_pose_yaw_clip
                ),
            ),
            dim=-1,
        )

        raw_navigation_command = self._target_pose_to_navigation_command(target_pose)
        navigation_command = torch.cat(
            (
                raw_navigation_command[:, 0:1].clamp(-self.command_clip_xy, self.command_clip_xy),
                raw_navigation_command[:, 1:2].clamp(-self.command_clip_xy, self.command_clip_xy),
                raw_navigation_command[:, 2:3].clamp(-self.command_clip_yaw, self.command_clip_yaw),
            ),
            dim=-1,
        )
        delta_cmd = navigation_command - nominal_command
        return navigation_command, target_pose, delta_cmd, side, nominal_command, obstacle_representation

    def _build_obstacle_representation(self, latent: torch.Tensor) -> torch.Tensor:
        raw_rep = torch.tanh(self.rep_head(latent))
        # position and blockage/openness channels live in [0, 1]
        return torch.cat(
            (
                0.5 * (raw_rep[:, 0:1] + 1.0),  # nearest forward
                raw_rep[:, 1:2],  # nearest lateral
                0.5 * (raw_rep[:, 2:7] + 1.0),
                raw_rep[:, 7:8],  # preferred side in [-1, 1]
            ),
            dim=-1,
        )

    def _target_pose_to_navigation_command(self, target_pose: torch.Tensor) -> torch.Tensor:
        """Invert the teacher's target-pose construction back to `(vx, vy, yaw)`.

        The teacher defines its short-horizon target pose linearly from command:

            x = vx * target_pose_horizon
            y = vy * target_pose_horizon
            yaw = yaw_rate * target_pose_yaw_horizon

        The student must therefore use the exact inverse here; otherwise the
        target-pose loss and command loss push in different directions.
        """
        return torch.cat(
            (
                torch.clamp(
                    self.target_pose_to_vx_gain * target_pose[:, 0:1] / max(self.target_pose_horizon, 1.0e-6),
                    min=-self.command_clip_xy,
                    max=self.command_clip_xy,
                ),
                torch.clamp(
                    self.target_pose_to_vy_gain * target_pose[:, 1:2] / max(self.target_pose_horizon, 1.0e-6),
                    min=-self.command_clip_xy,
                    max=self.command_clip_xy,
                ),
                torch.clamp(
                    self.target_pose_to_yaw_gain * target_pose[:, 2:3] / max(self.target_pose_yaw_horizon, 1.0e-6),
                    min=-self.command_clip_yaw,
                    max=self.command_clip_yaw,
                ),
            ),
            dim=-1,
        )

    def _navigation_command_to_target_pose(self, navigation_command: torch.Tensor) -> torch.Tensor:
        """Map a nominal `(vx, vy, yaw)` command to the teacher-aligned target pose."""
        return torch.cat(
            (
                (navigation_command[:, 0:1] * self.target_pose_horizon).clamp(
                    -self.target_pose_x_clip, self.target_pose_x_clip
                ),
                (navigation_command[:, 1:2] * self.target_pose_horizon).clamp(
                    -self.target_pose_y_clip, self.target_pose_y_clip
                ),
                (navigation_command[:, 2:3] * self.target_pose_yaw_horizon).clamp(
                    -self.target_pose_yaw_clip, self.target_pose_yaw_clip
                ),
            ),
            dim=-1,
        )

    def _goal_command_to_base_target_pose(self, goal_command: torch.Tensor) -> torch.Tensor:
        """Convert the remaining local goal into the nominal short-horizon target pose."""
        return torch.cat(
            (
                goal_command[:, 0:1].clamp(-self.target_pose_x_clip, self.target_pose_x_clip),
                goal_command[:, 1:2].clamp(-self.target_pose_y_clip, self.target_pose_y_clip),
                goal_command[:, 2:3].clamp(-self.target_pose_yaw_clip, self.target_pose_yaw_clip),
            ),
            dim=-1,
        )

    def _goal_command_to_nominal_command(self, goal_command: torch.Tensor) -> torch.Tensor:
        """Map the remaining local goal to the nominal LLC command before obstacle correction."""
        return self._target_pose_to_navigation_command(self._goal_command_to_base_target_pose(goal_command))

    def get_base_command(self, obs) -> torch.Tensor:
        policy_obs = obs[self.obs_group_name]
        return policy_obs[:, self.command_obs_start : self.command_obs_start + self.command_obs_dim]

    def _build_latent_input(self, obs) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        return torch.cat(obs_list, dim=-1)

    @staticmethod
    def _get_obs_dim(obs, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"NavigationCommandPolicy only supports 1D observations, got {obs[obs_group].shape}"
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim


class NavigationCommandStudent(nn.Module):
    """Training-time wrapper: separate navigation policy + separate frozen LLC.

    The important separation is explicit:

        navigation_policy(obs) -> local `(vx, vy, yaw)`
        frozen_actor(command)  -> final 16D robot action

    ``forward()`` still returns the final action for compatibility with the
    existing env rollout path, but the navigation part now lives in its own
    submodule and can be exported/replaced independently.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        *,
        frozen_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        frozen_activation: str = "elu",
        frozen_obs_normalization: bool = False,
        command_obs_start: int = GOAL_COMMAND_START,
        command_obs_dim: int = GOAL_COMMAND_DIM,
        representation_dim: int = 8,
        target_pose_horizon: float = 0.75,
        target_pose_yaw_horizon: float = 0.6,
        target_pose_x_clip: float = 1.5,
        target_pose_y_clip: float = 1.5,
        target_pose_yaw_clip: float = 1.2,
        target_pose_to_vx_gain: float = 1.0,
        target_pose_to_vy_gain: float = 1.0,
        target_pose_to_yaw_gain: float = 1.0,
        side_guidance_lateral_gain: float = 0.25,
        side_guidance_yaw_gain: float = 0.45,
        command_clip_xy: float = 2.0,
        command_clip_yaw: float = 2.0,
    ) -> None:
        super().__init__()
        self.command_obs_start = command_obs_start
        self.command_obs_dim = command_obs_dim

        self.navigation_policy = NavigationCommandPolicy(
            obs,
            obs_groups,
            obs_set,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            frozen_hidden_dims=frozen_hidden_dims,
            frozen_activation=frozen_activation,
            frozen_obs_normalization=frozen_obs_normalization,
            command_obs_start=command_obs_start,
            command_obs_dim=command_obs_dim,
            representation_dim=representation_dim,
            target_pose_horizon=target_pose_horizon,
            target_pose_yaw_horizon=target_pose_yaw_horizon,
            target_pose_x_clip=target_pose_x_clip,
            target_pose_y_clip=target_pose_y_clip,
            target_pose_yaw_clip=target_pose_yaw_clip,
            target_pose_to_vx_gain=target_pose_to_vx_gain,
            target_pose_to_vy_gain=target_pose_to_vy_gain,
            target_pose_to_yaw_gain=target_pose_to_yaw_gain,
            side_guidance_lateral_gain=side_guidance_lateral_gain,
            side_guidance_yaw_gain=side_guidance_yaw_gain,
            command_clip_xy=command_clip_xy,
            command_clip_yaw=command_clip_yaw,
        )

        self.frozen_actor = MLPModel(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=frozen_hidden_dims,
            activation=frozen_activation,
            obs_normalization=frozen_obs_normalization,
            distribution_cfg=None,
        )
        for param in self.frozen_actor.parameters():
            param.requires_grad = False

        if distribution_cfg is not None:
            dist_cfg = copy.deepcopy(distribution_cfg)
            dist_class: type[Distribution] = resolve_callable(dist_cfg.pop("class_name"))  # type: ignore[arg-type]
            self.distribution: Distribution | None = dist_class(output_dim, **dist_cfg)
        else:
            self.distribution = None

    def forward(
        self,
        obs,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del masks, hidden_state
        action_mean, _, _, _, _, _, _ = self._compute_outputs(obs)

        if self.distribution is not None:
            self.distribution.update(action_mean)
            if stochastic_output:
                return self.distribution.sample()
            return self.distribution.deterministic_output(action_mean)
        return action_mean

    def get_aux_outputs(self, obs) -> dict[str, torch.Tensor]:
        _, navigation_command, target_pose, delta_cmd, side, obstacle_representation, base_command = self._compute_outputs(obs)
        return {
            "navigation_command": navigation_command,
            "target_pose": target_pose,
            "delta_cmd": delta_cmd,
            "side": side,
            "base_command": base_command,
            "obstacle_representation": obstacle_representation,
        }

    def get_base_command(self, obs) -> torch.Tensor:
        policy_obs = obs[self.navigation_policy.obs_group_name]
        return policy_obs[:, self.command_obs_start : self.command_obs_start + self.command_obs_dim]

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        del dones, hidden_state

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    @property
    def output_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("This student has no output distribution.")
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("This student has no output distribution.")
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("This student has no output distribution.")
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        if self.distribution is None:
            return ()
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("This student has no output distribution.")
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("This student has no output distribution.")
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        return self

    def update_normalization(self, obs) -> None:
        self.navigation_policy.update_normalization(obs)

    def _compute_outputs(
        self, obs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        navigation_command, target_pose, delta_cmd, side, _base_command, obstacle_representation = self.navigation_policy._compute_outputs(obs)
        action_mean = self._run_llc_with_command(obs, navigation_command)
        return action_mean, navigation_command, target_pose, delta_cmd, side, obstacle_representation, _base_command

    def _run_llc_with_command(self, obs, navigation_command: torch.Tensor) -> torch.Tensor:
        policy_obs = obs[self.navigation_policy.obs_group_name]
        modified_policy_obs = policy_obs.clone()
        modified_policy_obs[
            :, self.command_obs_start : self.command_obs_start + self.command_obs_dim
        ] = (
            navigation_command
        )

        modified_obs = {key: obs[key] for key in obs.keys()}
        modified_obs[self.navigation_policy.obs_group_name] = modified_policy_obs
        return self.frozen_actor(modified_obs)

    def navigation_state_dict(self) -> dict[str, torch.Tensor]:
        """Export only the local navigation policy parameters."""
        return self.navigation_policy.state_dict()

    def load_navigation_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True) -> None:
        """Load only the local navigation policy parameters."""
        self.navigation_policy.load_state_dict(state_dict, strict=strict)
