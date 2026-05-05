# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom teacher models for frozen-locomotion obstacle avoidance."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization, HiddenState, MLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable


class FrozenCommandResidualActor(nn.Module):
    """Frozen fast-flat actor with a trainable obstacle-aware command residual.

    The frozen actor keeps the fast-flat locomotion manifold intact. A small residual
    head observes the current command and privileged obstacle positions, then adds a
    bounded correction to only `cmd_y` and `cmd_yaw` before the frozen actor consumes
    the modified observation. The residual head is zero-initialized so iteration 0
    exactly matches the fast-flat policy.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (128, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        *,
        frozen_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        frozen_activation: str = "elu",
        frozen_obs_normalization: bool = False,
        command_obs_start: int = 9,
        command_obs_dim: int = 3,
        obstacle_obs_start: int = 60,
        obstacle_obs_dim: int = 30,
        state_obs_dim: int = 12,
        obstacle_max_distance: float = 8.0,
        residual_vy_scale: float = 0.9,
        residual_yaw_scale: float = 1.1,
        lateral_command_clip: float = 2.0,
        yaw_command_clip: float = 2.0,
        gate_forward_distance: float = 3.5,
        gate_min_forward_distance: float = 0.2,
        gate_path_width: float = 1.2,
    ) -> None:
        super().__init__()

        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        if len(self.obs_groups) != 1:
            raise ValueError(
                "FrozenCommandResidualActor expects one 1D observation group. "
                f"Got groups: {self.obs_groups}"
            )
        self.obs_group_name = self.obs_groups[0]

        self.command_obs_start = command_obs_start
        self.command_obs_dim = command_obs_dim
        self.obstacle_obs_start = obstacle_obs_start
        self.obstacle_obs_dim = obstacle_obs_dim
        self.state_obs_dim = state_obs_dim
        self.obstacle_max_distance = obstacle_max_distance
        self.residual_vy_scale = residual_vy_scale
        self.residual_yaw_scale = residual_yaw_scale
        self.lateral_command_clip = lateral_command_clip
        self.yaw_command_clip = yaw_command_clip
        self.gate_forward_distance = gate_forward_distance
        self.gate_min_forward_distance = gate_min_forward_distance
        self.gate_path_width = gate_path_width

        residual_input_dim = self.state_obs_dim + self.obstacle_obs_dim
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(residual_input_dim)
        else:
            self.obs_normalizer = nn.Identity()

        self.residual_mlp = MLP(residual_input_dim, 2, hidden_dims, activation)
        self._zero_init_last_linear(self.residual_mlp)

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

        policy_obs = obs[self.obs_group_name]
        residual_features = self._build_residual_features(policy_obs)
        residual_features = self.obs_normalizer(residual_features)
        residual_delta = self.residual_mlp(residual_features)
        residual_gate = self._compute_residual_gate(policy_obs).unsqueeze(-1)

        scaled_delta = residual_gate * torch.tanh(residual_delta)

        modified_policy_obs = policy_obs.clone()
        commands = modified_policy_obs[
            :, self.command_obs_start : self.command_obs_start + self.command_obs_dim
        ].clone()
        commands[:, 1] = torch.clamp(
            commands[:, 1] + self.residual_vy_scale * scaled_delta[:, 0],
            min=-self.lateral_command_clip,
            max=self.lateral_command_clip,
        )
        commands[:, 2] = torch.clamp(
            commands[:, 2] + self.residual_yaw_scale * scaled_delta[:, 1],
            min=-self.yaw_command_clip,
            max=self.yaw_command_clip,
        )
        modified_policy_obs[
            :, self.command_obs_start : self.command_obs_start + self.command_obs_dim
        ] = commands

        modified_obs = {key: obs[key] for key in obs.keys()}
        modified_obs[self.obs_group_name] = modified_policy_obs

        # Keep the frozen actor parameters fixed via requires_grad=False, but still
        # allow gradients to flow from the action mean back into the residual head.
        action_mean = self.frozen_actor(modified_obs)

        if self.distribution is not None:
            self.distribution.update(action_mean)
            if stochastic_output:
                return self.distribution.sample()
            return self.distribution.deterministic_output(action_mean)

        return action_mean

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        del dones, hidden_state

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    @property
    def output_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("This actor has no output distribution.")
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("This actor has no output distribution.")
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("This actor has no output distribution.")
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        if self.distribution is None:
            return ()
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("This actor has no output distribution.")
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("This actor has no output distribution.")
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        return self

    def update_normalization(self, obs) -> None:
        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            self.obs_normalizer.update(self._build_residual_features(obs[self.obs_group_name]))

    def _build_residual_features(self, policy_obs: torch.Tensor) -> torch.Tensor:
        state_features = policy_obs[:, : self.state_obs_dim]
        obstacle_features = policy_obs[:, self.obstacle_obs_start : self.obstacle_obs_start + self.obstacle_obs_dim]
        return torch.cat((state_features, obstacle_features), dim=-1)

    def _compute_residual_gate(self, policy_obs: torch.Tensor) -> torch.Tensor:
        obstacle_positions = policy_obs[
            :, self.obstacle_obs_start : self.obstacle_obs_start + self.obstacle_obs_dim
        ].view(policy_obs.shape[0], -1, 2)
        obstacle_positions_m = obstacle_positions * self.obstacle_max_distance

        obs_x = obstacle_positions_m[..., 0]
        obs_y = obstacle_positions_m[..., 1].abs()
        valid = (obstacle_positions.abs().sum(dim=-1) > 1.0e-6).float()

        forward_span = max(self.gate_forward_distance - self.gate_min_forward_distance, 1.0e-6)
        forward_gate = ((self.gate_forward_distance - obs_x) / forward_span).clamp(0.0, 1.0)
        forward_gate = forward_gate * (obs_x > self.gate_min_forward_distance).float()
        lateral_gate = (1.0 - obs_y / self.gate_path_width).clamp(0.0, 1.0)

        return (valid * forward_gate * lateral_gate).amax(dim=1)

    @staticmethod
    def _get_obs_dim(obs, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"FrozenCommandResidualActor only supports 1D observations, got {obs[obs_group].shape}"
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim

    @staticmethod
    def _zero_init_last_linear(module: nn.Module) -> None:
        for layer in reversed(list(module.children())):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                return
        raise ValueError("Residual MLP has no final Linear layer to zero-initialize.")


class GeometricSteeringTeacher(nn.Module):
    """Rule-based steering teacher on top of a frozen fast-flat LLC.

    The steering layer is intentionally narrow in scope: it only rewrites the
    incoming velocity command using privileged obstacle positions. The frozen
    LLC remains the sole module that maps `(vx, vy, yaw)` commands to the final
    16D robot action, which keeps the layer boundary clean and reusable for
    future local-planner integration.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (128, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        *,
        frozen_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        frozen_activation: str = "elu",
        frozen_obs_normalization: bool = False,
        command_obs_start: int = 9,
        command_obs_dim: int = 3,
        obstacle_obs_start: int = 60,
        obstacle_obs_dim: int = 30,
        obstacle_max_distance: float = 8.0,
        min_command_speed: float = 0.15,
        safe_distance: float = 2.8,
        min_forward_distance: float = 0.15,
        corridor_half_width: float = 0.8,
        vy_gain: float = 0.85,
        yaw_gain: float = 1.10,
        max_delta_vy: float = 0.8,
        max_delta_yaw: float = 1.0,
        smoothing_alpha: float = 0.70,
        lateral_command_clip: float = 2.0,
        yaw_command_clip: float = 2.0,
    ) -> None:
        del hidden_dims, activation, obs_normalization, distribution_cfg
        super().__init__()

        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        if len(self.obs_groups) != 1:
            raise ValueError(
                "GeometricSteeringTeacher expects one 1D observation group. "
                f"Got groups: {self.obs_groups}"
            )
        self.obs_group_name = self.obs_groups[0]

        self.command_obs_start = command_obs_start
        self.command_obs_dim = command_obs_dim
        self.obstacle_obs_start = obstacle_obs_start
        self.obstacle_obs_dim = obstacle_obs_dim
        self.obstacle_max_distance = obstacle_max_distance
        self.min_command_speed = min_command_speed
        self.safe_distance = safe_distance
        self.min_forward_distance = min_forward_distance
        self.corridor_half_width = corridor_half_width
        self.vy_gain = vy_gain
        self.yaw_gain = yaw_gain
        self.max_delta_vy = max_delta_vy
        self.max_delta_yaw = max_delta_yaw
        self.smoothing_alpha = smoothing_alpha
        self.lateral_command_clip = lateral_command_clip
        self.yaw_command_clip = yaw_command_clip

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

        self.register_buffer("_smoothed_delta", torch.zeros(1, 2), persistent=False)
        self.register_buffer("_last_base_command", torch.zeros(1, self.command_obs_dim), persistent=False)
        self.register_buffer("_last_delta_command", torch.zeros(1, 2), persistent=False)
        self.register_buffer("_last_adjusted_command", torch.zeros(1, self.command_obs_dim), persistent=False)

    def forward(
        self,
        obs,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del masks, hidden_state, stochastic_output

        policy_obs = obs[self.obs_group_name]
        command = policy_obs[:, self.command_obs_start : self.command_obs_start + self.command_obs_dim]
        obstacle_positions = policy_obs[
            :, self.obstacle_obs_start : self.obstacle_obs_start + self.obstacle_obs_dim
        ].view(policy_obs.shape[0], -1, 2) * self.obstacle_max_distance

        delta_cmd = self._compute_steering_delta(command, obstacle_positions)
        delta_cmd = self._smooth_delta(delta_cmd)

        modified_policy_obs = policy_obs.clone()
        adjusted_command = command.clone()
        adjusted_command[:, 1] = torch.clamp(
            adjusted_command[:, 1] + delta_cmd[:, 0],
            min=-self.lateral_command_clip,
            max=self.lateral_command_clip,
        )
        adjusted_command[:, 2] = torch.clamp(
            adjusted_command[:, 2] + delta_cmd[:, 1],
            min=-self.yaw_command_clip,
            max=self.yaw_command_clip,
        )
        modified_policy_obs[:, self.command_obs_start : self.command_obs_start + self.command_obs_dim] = adjusted_command
        self._update_debug_buffers(command, delta_cmd, adjusted_command)

        modified_obs = {key: obs[key] for key in obs.keys()}
        modified_obs[self.obs_group_name] = modified_policy_obs
        return self.frozen_actor(modified_obs)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        del hidden_state
        if dones is None:
            self._smoothed_delta.zero_()
            return
        if self._smoothed_delta.shape[0] == dones.shape[0]:
            self._smoothed_delta[dones] = 0.0

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def update_normalization(self, obs) -> None:
        del obs

    @property
    def last_base_command(self) -> torch.Tensor:
        return self._last_base_command

    @property
    def last_delta_command(self) -> torch.Tensor:
        return self._last_delta_command

    @property
    def last_adjusted_command(self) -> torch.Tensor:
        return self._last_adjusted_command

    def _compute_steering_delta(self, command: torch.Tensor, obstacle_positions: torch.Tensor) -> torch.Tensor:
        batch_size = command.shape[0]
        device = command.device

        cmd_xy = command[:, :2]
        cmd_speed = cmd_xy.norm(dim=1)
        speed_gate = (cmd_speed / self.min_command_speed).clamp(0.0, 1.0)

        cmd_dir = torch.zeros_like(cmd_xy)
        moving = cmd_speed > self.min_command_speed
        cmd_dir[moving] = cmd_xy[moving] / cmd_speed[moving].unsqueeze(1)
        cmd_dir[~moving, 0] = 1.0

        obs_x = obstacle_positions[..., 0]
        obs_y = obstacle_positions[..., 1]
        valid = (obstacle_positions.abs().sum(dim=-1) > 1.0e-6)

        forward = (obstacle_positions * cmd_dir.unsqueeze(1)).sum(dim=-1)
        signed_lateral = cmd_dir[:, 0].unsqueeze(1) * obs_y - cmd_dir[:, 1].unsqueeze(1) * obs_x

        in_corridor = (
            valid
            & (forward > self.min_forward_distance)
            & (forward < self.safe_distance)
            & (signed_lateral.abs() < self.corridor_half_width)
        )
        if not torch.any(in_corridor):
            return torch.zeros(batch_size, 2, device=device)

        distance = torch.sqrt(torch.clamp(forward.square() + signed_lateral.square(), min=1.0e-6))
        distance_falloff = (1.0 - distance / self.safe_distance).clamp(0.0, 1.0)
        corridor_falloff = (1.0 - signed_lateral.abs() / self.corridor_half_width).clamp(0.0, 1.0)
        forward_urgency = (1.0 - forward / self.safe_distance).clamp(0.0, 1.0)
        influence = in_corridor.float() * distance_falloff * corridor_falloff

        push_sign = -torch.sign(signed_lateral)
        vy_delta = self.vy_gain * torch.sum(push_sign * influence, dim=1)
        yaw_delta = self.yaw_gain * torch.sum(push_sign * influence * forward_urgency, dim=1)

        delta = torch.stack((vy_delta, yaw_delta), dim=1)
        delta[:, 0] = delta[:, 0].clamp(-self.max_delta_vy, self.max_delta_vy)
        delta[:, 1] = delta[:, 1].clamp(-self.max_delta_yaw, self.max_delta_yaw)
        return speed_gate.unsqueeze(1) * delta

    def _smooth_delta(self, delta: torch.Tensor) -> torch.Tensor:
        if self._smoothed_delta.shape != delta.shape:
            self._smoothed_delta = torch.zeros_like(delta)
        self._smoothed_delta = self.smoothing_alpha * self._smoothed_delta + (1.0 - self.smoothing_alpha) * delta
        return self._smoothed_delta

    def _update_debug_buffers(
        self,
        command: torch.Tensor,
        delta_cmd: torch.Tensor,
        adjusted_command: torch.Tensor,
    ) -> None:
        if self._last_base_command.shape != command.shape:
            self._last_base_command = torch.zeros_like(command)
        if self._last_delta_command.shape != delta_cmd.shape:
            self._last_delta_command = torch.zeros_like(delta_cmd)
        if self._last_adjusted_command.shape != adjusted_command.shape:
            self._last_adjusted_command = torch.zeros_like(adjusted_command)

        self._last_base_command.copy_(command.detach())
        self._last_delta_command.copy_(delta_cmd.detach())
        self._last_adjusted_command.copy_(adjusted_command.detach())

    @staticmethod
    def _get_obs_dim(obs, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"GeometricSteeringTeacher only supports 1D observations, got {obs[obs_group].shape}"
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim
