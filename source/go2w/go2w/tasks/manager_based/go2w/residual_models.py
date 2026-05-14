# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Active goal-conditioned teacher models for frozen-locomotion local navigation."""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState

from .observation_layout import GOAL_COMMAND_DIM, GOAL_COMMAND_START, PRIVILEGED_OBSTACLE_START


class GeometricSteeringTeacher(nn.Module):
    """Rule-based steering teacher on top of a frozen fast-flat LLC.

    The steering layer first turns the current local goal pose into a nominal
    planar command, then rewrites that command using privileged obstacle
    positions. It can push the command sideways relative to the intended goal
    direction, slow progress along the current path ray, and add yaw away from
    the obstacle side. The frozen LLC remains the sole module that maps
    `(vx, vy, yaw)` commands to the final 16D robot action.
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
        command_obs_start: int = GOAL_COMMAND_START,
        command_obs_dim: int = GOAL_COMMAND_DIM,
        obstacle_obs_start: int = PRIVILEGED_OBSTACLE_START,
        obstacle_obs_dim: int = 30,
        obstacle_max_distance: float = 8.0,
        min_command_speed: float = 0.15,
        speed_reference: float = 2.0,
        safe_distance: float = 2.8,
        min_forward_distance: float = 0.15,
        corridor_half_width: float = 0.75,
        center_deadband: float = 0.05,
        side_gain: float = 1.00,
        brake_gain: float = 0.60,
        yaw_gain: float = 1.10,
        heading_align_gain: float = 1.10,
        forward_bias_gain: float = 0.85,
        local_avoid_distance: float = 1.18,
        local_avoid_gain: float = 0.78,
        local_brake_gain: float = 0.52,
        sideways_body_bias_gain: float = 0.70,
        sideways_heading_boost_gain: float = 0.75,
        diagonal_sweep_gain: float = 1.12,
        diagonal_blocked_gain: float = 0.68,
        turning_sweep_clearance_gain: float = 0.36,
        turning_speed_reduction_gain: float = 0.30,
        turn_commitment_gain: float = 0.85,
        turn_commitment_activation: float = 0.20,
        narrow_gap_speed_reduction_gain: float = 0.65,
        narrow_gap_forward_bias_gain: float = 0.45,
        narrow_gap_heading_gain: float = 1.40,
        obstacle_width: float = 0.30,
        robot_forward_clearance: float = 0.42,
        robot_side_clearance: float = 0.80,
        gap_pair_forward_tolerance: float = 0.9,
        blocked_detour_gain: float = 1.30,
        blocked_brake_gain: float = 1.00,
        blocked_yaw_gain: float = 1.70,
        guidance_blend_gain: float = 0.80,
        high_speed_safe_distance_gain: float = 0.22,
        high_speed_local_distance_gain: float = 0.12,
        high_speed_brake_gain: float = 0.22,
        high_speed_yaw_gain: float = 0.18,
        high_speed_turn_reduction_gain: float = 0.15,
        high_speed_smoothing_reduction: float = 0.12,
        max_delta_vx: float = 0.8,
        max_delta_vy: float = 0.8,
        max_delta_yaw: float = 1.5,
        smoothing_alpha: float = 0.55,
        lateral_command_clip: float = 2.0,
        yaw_command_clip: float = 2.0,
        obstacle_present_risk_floor: float = 0.30,
        target_pose_horizon: float = 0.75,
        target_pose_yaw_horizon: float = 0.6,
        target_pose_x_clip: float = 1.5,
        target_pose_y_clip: float = 1.5,
        target_pose_yaw_clip: float = 1.2,
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
        self.speed_reference = speed_reference
        self.safe_distance = safe_distance
        self.min_forward_distance = min_forward_distance
        self.corridor_half_width = corridor_half_width
        self.center_deadband = center_deadband
        self.side_gain = side_gain
        self.brake_gain = brake_gain
        self.yaw_gain = yaw_gain
        self.heading_align_gain = heading_align_gain
        self.forward_bias_gain = forward_bias_gain
        self.local_avoid_distance = local_avoid_distance
        self.local_avoid_gain = local_avoid_gain
        self.local_brake_gain = local_brake_gain
        self.sideways_body_bias_gain = sideways_body_bias_gain
        self.sideways_heading_boost_gain = sideways_heading_boost_gain
        self.diagonal_sweep_gain = diagonal_sweep_gain
        self.diagonal_blocked_gain = diagonal_blocked_gain
        self.turning_sweep_clearance_gain = turning_sweep_clearance_gain
        self.turning_speed_reduction_gain = turning_speed_reduction_gain
        self.turn_commitment_gain = turn_commitment_gain
        self.turn_commitment_activation = turn_commitment_activation
        self.narrow_gap_speed_reduction_gain = narrow_gap_speed_reduction_gain
        self.narrow_gap_forward_bias_gain = narrow_gap_forward_bias_gain
        self.narrow_gap_heading_gain = narrow_gap_heading_gain
        self.obstacle_width = obstacle_width
        self.robot_forward_clearance = robot_forward_clearance
        self.robot_side_clearance = robot_side_clearance
        self.gap_pair_forward_tolerance = gap_pair_forward_tolerance
        self.blocked_detour_gain = blocked_detour_gain
        self.blocked_brake_gain = blocked_brake_gain
        self.blocked_yaw_gain = blocked_yaw_gain
        self.guidance_blend_gain = guidance_blend_gain
        self.high_speed_safe_distance_gain = high_speed_safe_distance_gain
        self.high_speed_local_distance_gain = high_speed_local_distance_gain
        self.high_speed_brake_gain = high_speed_brake_gain
        self.high_speed_yaw_gain = high_speed_yaw_gain
        self.high_speed_turn_reduction_gain = high_speed_turn_reduction_gain
        self.high_speed_smoothing_reduction = high_speed_smoothing_reduction
        self.max_delta_vx = max_delta_vx
        self.max_delta_vy = max_delta_vy
        self.max_delta_yaw = max_delta_yaw
        self.smoothing_alpha = smoothing_alpha
        self.lateral_command_clip = lateral_command_clip
        self.yaw_command_clip = yaw_command_clip
        self.obstacle_present_risk_floor = obstacle_present_risk_floor
        self.target_pose_horizon = target_pose_horizon
        self.target_pose_yaw_horizon = target_pose_yaw_horizon
        self.target_pose_x_clip = target_pose_x_clip
        self.target_pose_y_clip = target_pose_y_clip
        self.target_pose_yaw_clip = target_pose_yaw_clip

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

        self.register_buffer("_smoothed_delta", torch.zeros(1, 3), persistent=False)
        self.register_buffer("_last_base_command", torch.zeros(1, self.command_obs_dim), persistent=False)
        self.register_buffer("_last_guidance_command", torch.zeros(1, self.command_obs_dim), persistent=False)
        self.register_buffer("_last_delta_command", torch.zeros(1, self.command_obs_dim), persistent=False)
        self.register_buffer("_last_adjusted_command", torch.zeros(1, self.command_obs_dim), persistent=False)
        self.register_buffer("_last_gap_width", torch.zeros(1), persistent=False)
        self.register_buffer("_last_gap_turn_need", torch.zeros(1), persistent=False)
        self.register_buffer("_last_gap_blocked", torch.zeros(1), persistent=False)
        self.register_buffer("_last_turn_side", torch.zeros(1), persistent=False)

    def forward(
        self,
        obs,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del masks, hidden_state, stochastic_output
        adjusted_command = self.compute_navigation_command(obs)
        return self._run_frozen_actor_with_command(obs, adjusted_command)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        del hidden_state
        if dones is None:
            self._reset_state()
            return
        if self._smoothed_delta.shape[0] == dones.shape[0]:
            self._reset_state(dones)

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def update_normalization(self, obs) -> None:
        del obs

    def compute_flat_action(self, obs) -> torch.Tensor:
        """Return the frozen-LLC action for the nominal goal-following command."""
        policy_obs = obs[self.obs_group_name]
        goal_command = policy_obs[:, self.command_obs_start : self.command_obs_start + self.command_obs_dim]
        nominal_command = self._goal_command_to_nominal_command(goal_command)
        return self._run_frozen_actor_with_command(obs, nominal_command)

    def compute_navigation_command(self, obs) -> torch.Tensor:
        """Compute the teacher's obstacle-aware `(vx, vy, yaw)` command before LLC execution."""
        policy_obs = obs[self.obs_group_name]
        goal_command = policy_obs[:, self.command_obs_start : self.command_obs_start + self.command_obs_dim]
        nominal_command = self._goal_command_to_nominal_command(goal_command)
        obstacle_positions = policy_obs[
            :, self.obstacle_obs_start : self.obstacle_obs_start + self.obstacle_obs_dim
        ].view(policy_obs.shape[0], -1, 2) * self.obstacle_max_distance

        guidance_command = self._build_guidance_command(nominal_command)
        delta_cmd, gap_width, gap_turn_need, gap_blocked = self._compute_steering_delta(
            guidance_command, obstacle_positions
        )
        delta_cmd = self._smooth_delta(delta_cmd, guidance_command[:, :2].norm(dim=1))

        adjusted_command = guidance_command.clone()
        adjusted_command[:, 0] = torch.clamp(
            adjusted_command[:, 0] + delta_cmd[:, 0],
            min=-self.lateral_command_clip,
            max=self.lateral_command_clip,
        )
        adjusted_command[:, 1] = torch.clamp(
            adjusted_command[:, 1] + delta_cmd[:, 1],
            min=-self.lateral_command_clip,
            max=self.lateral_command_clip,
        )
        adjusted_command[:, 2] = torch.clamp(
            adjusted_command[:, 2] + delta_cmd[:, 2],
            min=-self.yaw_command_clip,
            max=self.yaw_command_clip,
        )
        self._update_debug_buffers(
            nominal_command, guidance_command, delta_cmd, adjusted_command, gap_width, gap_turn_need, gap_blocked
        )
        return adjusted_command

    def compute_distillation_risk(self, obs) -> torch.Tensor:
        """Estimate when the student should prioritize teacher avoidance over flat locomotion."""
        policy_obs = obs[self.obs_group_name]
        goal_command = policy_obs[:, self.command_obs_start : self.command_obs_start + self.command_obs_dim]
        nominal_command = self._goal_command_to_nominal_command(goal_command)
        obstacle_positions = policy_obs[
            :, self.obstacle_obs_start : self.obstacle_obs_start + self.obstacle_obs_dim
        ].view(policy_obs.shape[0], -1, 2) * self.obstacle_max_distance
        return self._compute_distillation_risk_from_tensors(nominal_command, obstacle_positions)

    def compute_obstacle_representation(self, obs) -> torch.Tensor:
        """Return a compact teacher geometry representation for student alignment.

        The representation intentionally mirrors the quantities the steering logic
        actually reasons about:
          0. nearest threat forward position
          1. nearest threat lateral position
          2. center corridor blockage
          3. left corridor blockage
          4. right corridor blockage
          5. left lane openness
          6. right lane openness
          7. preferred avoidance side
        """
        policy_obs = obs[self.obs_group_name]
        goal_command = policy_obs[:, self.command_obs_start : self.command_obs_start + self.command_obs_dim]
        nominal_command = self._goal_command_to_nominal_command(goal_command)
        obstacle_positions = policy_obs[
            :, self.obstacle_obs_start : self.obstacle_obs_start + self.obstacle_obs_dim
        ].view(policy_obs.shape[0], -1, 2) * self.obstacle_max_distance
        return self._compute_obstacle_representation_from_tensors(nominal_command, obstacle_positions)

    def compute_target_pose(self, obs) -> torch.Tensor:
        """Return a short-horizon local target pose derived from the teacher command."""
        adjusted_command = self.compute_navigation_command(obs)
        return self.navigation_command_to_target_pose(adjusted_command)

    def navigation_command_to_target_pose(self, navigation_command: torch.Tensor) -> torch.Tensor:
        """Convert a local velocity command into a short-horizon local target pose."""
        target_pose = torch.zeros_like(navigation_command)
        target_pose[:, 0] = torch.clamp(
            navigation_command[:, 0] * self.target_pose_horizon,
            min=-self.target_pose_x_clip,
            max=self.target_pose_x_clip,
        )
        target_pose[:, 1] = torch.clamp(
            navigation_command[:, 1] * self.target_pose_horizon,
            min=-self.target_pose_y_clip,
            max=self.target_pose_y_clip,
        )
        target_pose[:, 2] = torch.clamp(
            navigation_command[:, 2] * self.target_pose_yaw_horizon,
            min=-self.target_pose_yaw_clip,
            max=self.target_pose_yaw_clip,
        )
        return target_pose

    def _goal_command_to_target_pose(self, goal_command: torch.Tensor) -> torch.Tensor:
        """Clip the remaining goal to the nominal short-horizon target pose."""
        target_pose = torch.zeros_like(goal_command)
        target_pose[:, 0] = torch.clamp(
            goal_command[:, 0],
            min=-self.target_pose_x_clip,
            max=self.target_pose_x_clip,
        )
        target_pose[:, 1] = torch.clamp(
            goal_command[:, 1],
            min=-self.target_pose_y_clip,
            max=self.target_pose_y_clip,
        )
        target_pose[:, 2] = torch.clamp(
            goal_command[:, 2],
            min=-self.target_pose_yaw_clip,
            max=self.target_pose_yaw_clip,
        )
        return target_pose

    def _goal_command_to_nominal_command(self, goal_command: torch.Tensor) -> torch.Tensor:
        """Map the remaining local goal to the nominal LLC command before obstacle steering."""
        return self._target_pose_to_navigation_command(self._goal_command_to_target_pose(goal_command))

    def _target_pose_to_navigation_command(self, target_pose: torch.Tensor) -> torch.Tensor:
        """Invert the teacher target-pose convention back to `(vx, vy, yaw)`."""
        command = torch.zeros_like(target_pose)
        command[:, 0] = torch.clamp(
            target_pose[:, 0] / max(self.target_pose_horizon, 1.0e-6),
            min=-self.lateral_command_clip,
            max=self.lateral_command_clip,
        )
        command[:, 1] = torch.clamp(
            target_pose[:, 1] / max(self.target_pose_horizon, 1.0e-6),
            min=-self.lateral_command_clip,
            max=self.lateral_command_clip,
        )
        command[:, 2] = torch.clamp(
            target_pose[:, 2] / max(self.target_pose_yaw_horizon, 1.0e-6),
            min=-self.yaw_command_clip,
            max=self.yaw_command_clip,
        )
        return command

    @property
    def last_base_command(self) -> torch.Tensor:
        return self._last_base_command

    @property
    def last_guidance_command(self) -> torch.Tensor:
        return self._last_guidance_command

    @property
    def last_delta_command(self) -> torch.Tensor:
        return self._last_delta_command

    @property
    def last_adjusted_command(self) -> torch.Tensor:
        return self._last_adjusted_command

    @property
    def last_navigation_command(self) -> torch.Tensor:
        """Alias for the final local navigation command sent into the LLC."""
        return self._last_adjusted_command

    @property
    def last_gap_width(self) -> torch.Tensor:
        return self._last_gap_width

    @property
    def last_gap_turn_need(self) -> torch.Tensor:
        return self._last_gap_turn_need

    @property
    def last_gap_blocked(self) -> torch.Tensor:
        return self._last_gap_blocked

    @property
    def last_turn_side(self) -> torch.Tensor:
        return self._last_turn_side

    def _build_guidance_command(self, command: torch.Tensor) -> torch.Tensor:
        if self._last_adjusted_command.shape != command.shape:
            return command

        steer_xy = self._smoothed_delta[:, :2].norm(dim=1) / max(max(self.max_delta_vx, self.max_delta_vy), 1.0e-6)
        steer_yaw = self._smoothed_delta[:, 2].abs() / max(self.max_delta_yaw, 1.0e-6)
        obstacle_state = torch.maximum(self._last_gap_turn_need, self._last_gap_blocked)
        blend = self.guidance_blend_gain * torch.maximum(torch.maximum(steer_xy, steer_yaw), obstacle_state)
        blend = blend.clamp(0.0, 1.0)

        guidance = command.clone()
        guidance[:, :2] = (1.0 - blend).unsqueeze(1) * command[:, :2] + blend.unsqueeze(1) * self._last_adjusted_command[:, :2]
        guidance[:, 2] = (1.0 - blend) * command[:, 2] + blend * self._last_adjusted_command[:, 2]
        return guidance

    def _reset_state(self, dones: torch.Tensor | None = None) -> None:
        state_buffer_names = (
            "_smoothed_delta",
            "_last_base_command",
            "_last_guidance_command",
            "_last_delta_command",
            "_last_adjusted_command",
            "_last_gap_width",
            "_last_gap_turn_need",
            "_last_gap_blocked",
            "_last_turn_side",
        )
        if dones is None:
            for name in state_buffer_names:
                buffer = getattr(self, name)
                setattr(self, name, torch.zeros_like(buffer))
            return

        dones = dones.to(dtype=torch.bool)
        for name in state_buffer_names:
            buffer = getattr(self, name)
            if buffer.shape[0] == dones.shape[0]:
                reset_mask = dones.view(-1, *([1] * (buffer.dim() - 1)))
                setattr(self, name, torch.where(reset_mask, torch.zeros_like(buffer), buffer))

    def _compute_steering_delta(
        self, command: torch.Tensor, obstacle_positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = command.shape[0]
        device = command.device

        cmd_xy = command[:, :2]
        cmd_speed = cmd_xy.norm(dim=1)
        speed_gate = (cmd_speed / self.min_command_speed).clamp(0.0, 1.0)
        speed_ratio = (cmd_speed / max(self.speed_reference, 1.0e-6)).clamp(0.0, 1.0)
        effective_safe_distance = self.safe_distance * (1.0 + self.high_speed_safe_distance_gain * speed_ratio)
        effective_local_avoid_distance = self.local_avoid_distance * (
            1.0 + self.high_speed_local_distance_gain * speed_ratio
        )
        brake_gain_scale = 1.0 + self.high_speed_brake_gain * speed_ratio
        yaw_gain_scale = 1.0 + self.high_speed_yaw_gain * speed_ratio
        turn_reduction_scale = 1.0 + self.high_speed_turn_reduction_gain * speed_ratio

        cmd_dir = torch.zeros_like(cmd_xy)
        moving = cmd_speed > self.min_command_speed
        cmd_dir[moving] = cmd_xy[moving] / cmd_speed[moving].unsqueeze(1)
        cmd_dir[~moving, 0] = 1.0

        valid = (obstacle_positions.abs().sum(dim=-1) > 1.0e-6)
        distance_body = torch.sqrt(torch.clamp(obstacle_positions.square().sum(dim=-1), min=1.0e-6))
        local_shell = valid & (distance_body < effective_local_avoid_distance.unsqueeze(1))
        local_falloff = (1.0 - distance_body / effective_local_avoid_distance.unsqueeze(1)).clamp(0.0, 1.0)
        local_weight = local_shell.float() * local_falloff.square()
        obstacle_away = -obstacle_positions / distance_body.unsqueeze(-1)
        local_repulsion_xy = torch.sum(local_weight.unsqueeze(-1) * obstacle_away, dim=1)
        local_pressure = local_repulsion_xy.norm(dim=1).clamp(0.0, 1.0)
        local_active = local_pressure > 1.0e-6

        forward = (obstacle_positions * cmd_dir.unsqueeze(1)).sum(dim=-1)
        side_dir = torch.stack((-cmd_dir[:, 1], cmd_dir[:, 0]), dim=1)
        signed_lateral = (obstacle_positions * side_dir.unsqueeze(1)).sum(dim=-1)

        in_corridor = (
            valid
            & (forward > self.min_forward_distance)
            & (forward < effective_safe_distance.unsqueeze(1))
            & (signed_lateral.abs() < self.corridor_half_width)
        )
        if not torch.any(in_corridor) and not torch.any(local_active):
            zero_delta = torch.zeros(batch_size, self.command_obs_dim, device=device)
            huge_gap = torch.full((batch_size,), 1.0e6, device=device)
            zero_metric = torch.zeros(batch_size, device=device)
            return zero_delta, huge_gap, zero_metric, zero_metric

        distance = torch.sqrt(torch.clamp(forward.square() + signed_lateral.square(), min=1.0e-6))
        distance_falloff = (1.0 - distance / effective_safe_distance.unsqueeze(1)).clamp(0.0, 1.0)
        corridor_falloff = (1.0 - signed_lateral.abs() / self.corridor_half_width).clamp(0.0, 1.0)
        forward_urgency = (1.0 - forward / effective_safe_distance.unsqueeze(1)).clamp(0.0, 1.0)
        influence = in_corridor.float() * distance_falloff * corridor_falloff

        weighted_lateral = torch.sum(signed_lateral * influence, dim=1)
        path_mass = torch.sum(influence, dim=1)
        weighted_mean_lateral = weighted_lateral / path_mass.clamp_min(1.0e-6)

        closest_idx = torch.argmax(influence, dim=1)
        batch_indices = torch.arange(batch_size, device=device)
        closest_lateral = signed_lateral[batch_indices, closest_idx]

        side_sign = -torch.sign(weighted_mean_lateral)
        ambiguous = weighted_mean_lateral.abs() <= self.center_deadband
        fallback_side = -torch.sign(closest_lateral)
        fallback_side = torch.where(
            fallback_side == 0.0,
            torch.ones_like(fallback_side),
            fallback_side,
        )
        side_sign = torch.where(ambiguous, fallback_side, side_sign)

        side_pressure = torch.sum(influence * (0.5 + 0.5 * forward_urgency), dim=1)
        brake_pressure = torch.sum(influence * corridor_falloff * forward_urgency, dim=1)
        yaw_pressure = torch.sum(influence * forward_urgency, dim=1)

        left_pressure = torch.where(
            signed_lateral > self.center_deadband, influence, torch.zeros_like(influence)
        ).amax(dim=1)
        right_pressure = torch.where(
            signed_lateral < -self.center_deadband, influence, torch.zeros_like(influence)
        ).amax(dim=1)
        gap_pressure = torch.minimum(left_pressure, right_pressure)

        pos_inf = torch.full_like(forward, float("inf"))
        neg_inf = torch.full_like(forward, float("-inf"))
        left_forward = torch.where(signed_lateral > self.center_deadband, forward, pos_inf).amin(dim=1)
        right_forward = torch.where(signed_lateral < -self.center_deadband, forward, pos_inf).amin(dim=1)
        left_lateral = torch.where(signed_lateral > self.center_deadband, signed_lateral, pos_inf).amin(dim=1)
        right_lateral = torch.where(signed_lateral < -self.center_deadband, signed_lateral, neg_inf).amax(dim=1)
        has_left = torch.isfinite(left_forward)
        has_right = torch.isfinite(right_forward)
        pair_valid = has_left & has_right & ((left_forward - right_forward).abs() <= self.gap_pair_forward_tolerance)
        gap_width = left_lateral - right_lateral - self.obstacle_width
        gap_width = torch.where(pair_valid, gap_width, torch.full_like(gap_width, 1.0e6))

        projected_sweep_clearance = self.diagonal_sweep_gain * (
            cmd_dir[:, 0].abs() * self.robot_forward_clearance
            + cmd_dir[:, 1].abs() * self.robot_side_clearance
        )
        base_turn_clearance = torch.maximum(
            torch.full_like(projected_sweep_clearance, self.robot_side_clearance),
            projected_sweep_clearance,
        )
        base_blocked_clearance = self.robot_forward_clearance + self.diagonal_blocked_gain * (
            projected_sweep_clearance - self.robot_forward_clearance
        ).clamp_min(0.0)
        base_blocked_clearance = torch.minimum(
            base_blocked_clearance,
            base_turn_clearance - 1.0e-3,
        )
        base_gap_turn_need = (
            (base_turn_clearance - gap_width)
            / (base_turn_clearance - base_blocked_clearance).clamp_min(1.0e-6)
        ).clamp(0.0, 1.0)
        turn_transition_clearance = (
            self.turning_sweep_clearance_gain * base_gap_turn_need * self.robot_side_clearance
        )
        turn_clearance = base_turn_clearance + turn_transition_clearance
        blocked_clearance = base_blocked_clearance + 0.5 * turn_transition_clearance
        blocked_clearance = torch.minimum(
            blocked_clearance,
            turn_clearance - 1.0e-3,
        )
        gap_turn_need = (
            (turn_clearance - gap_width)
            / (turn_clearance - blocked_clearance).clamp_min(1.0e-6)
        ).clamp(0.0, 1.0)
        gap_blocked = (
            (blocked_clearance - gap_width)
            / blocked_clearance.clamp_min(1.0e-6)
        ).clamp(0.0, 1.0)
        turn_commit_pressure = torch.maximum(base_gap_turn_need, torch.maximum(gap_turn_need, gap_blocked))
        if self._last_turn_side.shape[0] == batch_size:
            prev_turn_side = self._last_turn_side
        else:
            prev_turn_side = torch.zeros(batch_size, device=device)
        has_prev_turn = prev_turn_side.abs() > 0.5
        keep_turn_side = has_prev_turn & (turn_commit_pressure > self.turn_commitment_activation)
        committed_side = torch.sign(
            (1.0 - self.turn_commitment_gain) * side_sign + self.turn_commitment_gain * prev_turn_side
        )
        committed_side = torch.where(committed_side == 0.0, prev_turn_side, committed_side)
        side_sign = torch.where(keep_turn_side, committed_side, side_sign)
        next_turn_side = torch.where(
            turn_commit_pressure > self.turn_commitment_activation,
            side_sign,
            torch.zeros_like(side_sign),
        )
        if self._last_turn_side.shape[0] != batch_size:
            self._last_turn_side = torch.zeros_like(next_turn_side)
        self._last_turn_side.copy_(next_turn_side.detach())

        side_delta = self.side_gain * side_sign * side_pressure
        brake_delta = self.brake_gain * brake_gain_scale * brake_pressure
        side_delta = side_delta + self.blocked_detour_gain * gap_blocked * side_sign
        brake_delta = brake_delta + self.blocked_brake_gain * brake_gain_scale * gap_blocked
        local_brake_pressure = -(local_repulsion_xy * cmd_dir).sum(dim=1).clamp_min(0.0)
        delta_xy = side_delta.unsqueeze(1) * side_dir - brake_delta.unsqueeze(1) * cmd_dir
        delta_xy = delta_xy + self.local_avoid_gain * local_repulsion_xy
        delta_xy = delta_xy - (self.local_brake_gain * brake_gain_scale).unsqueeze(1) * local_brake_pressure.unsqueeze(1) * cmd_dir
        heading_pressure = (side_pressure + 0.5 * brake_pressure).clamp(0.0, 1.0)
        heading_pressure = torch.maximum(heading_pressure, gap_pressure)
        heading_pressure = torch.maximum(heading_pressure, gap_turn_need)
        heading_pressure = torch.maximum(heading_pressure, local_pressure)
        provisional_cmd_xy = cmd_xy + delta_xy
        narrow_gap_pressure = torch.maximum(gap_turn_need, gap_blocked)
        narrow_gap_pressure = torch.maximum(narrow_gap_pressure, local_pressure)
        target_speed = provisional_cmd_xy.norm(dim=1)
        longitudinal_ratio = cmd_xy[:, 0].abs() / cmd_speed.clamp(min=1.0e-6)
        lateral_ratio = cmd_xy[:, 1].abs() / cmd_speed.clamp(min=1.0e-6)
        narrow_gap_speed_scale = (
            1.0 - self.narrow_gap_speed_reduction_gain * narrow_gap_pressure
        ).clamp(0.35, 1.0)
        turn_phase_speed_scale = (
            1.0 - self.turning_speed_reduction_gain * turn_reduction_scale * gap_turn_need * (0.4 + 0.6 * lateral_ratio)
        ).clamp(0.45, 1.0)
        target_speed = target_speed * narrow_gap_speed_scale * turn_phase_speed_scale
        longitudinal_sign = torch.where(
            cmd_xy[:, 0] >= 0.0,
            torch.ones_like(target_speed),
            -torch.ones_like(target_speed),
        )
        turn_pressure = torch.maximum(heading_pressure, torch.maximum(gap_turn_need, gap_blocked))
        sideways_turn_pressure = lateral_ratio * turn_pressure
        forward_biased_xy = torch.stack((longitudinal_sign * target_speed, torch.zeros_like(target_speed)), dim=1)
        forward_bias_blend = (self.forward_bias_gain * heading_pressure).clamp(0.0, 1.0)
        forward_bias_blend = torch.maximum(forward_bias_blend, gap_turn_need)
        forward_bias_blend = forward_bias_blend * (1.0 - 0.75 * gap_blocked).clamp(0.0, 1.0)
        forward_axis_blend = forward_bias_blend * longitudinal_ratio
        forward_axis_blend = torch.clamp(
            forward_axis_blend
            + self.sideways_body_bias_gain * sideways_turn_pressure
            + self.narrow_gap_forward_bias_gain * narrow_gap_pressure,
            0.0,
            1.0,
        )
        aligned_cmd_xy = provisional_cmd_xy + forward_axis_blend.unsqueeze(1) * (
            forward_biased_xy - provisional_cmd_xy
        )
        delta_xy = aligned_cmd_xy - cmd_xy
        # Keep the yaw target tied to the actual motion ray before forward-axis
        # realignment, otherwise lateral commands lose their turn intent exactly
        # when we need the body to rotate to clear the feet through a tight gap.
        heading_reference_xy = longitudinal_sign.unsqueeze(1) * provisional_cmd_xy
        heading_error = torch.atan2(heading_reference_xy[:, 1], heading_reference_xy[:, 0])
        heading_gain_scale = (
            1.0
            + self.sideways_heading_boost_gain * sideways_turn_pressure
            + self.narrow_gap_heading_gain * narrow_gap_pressure
        )

        yaw_delta = self.yaw_gain * yaw_gain_scale * side_sign * yaw_pressure
        yaw_delta = yaw_delta + self.heading_align_gain * yaw_gain_scale * heading_gain_scale * heading_pressure * heading_error
        yaw_delta = yaw_delta + self.blocked_yaw_gain * yaw_gain_scale * gap_blocked * side_sign

        delta = torch.zeros(batch_size, self.command_obs_dim, device=device)
        delta[:, :2] = delta_xy
        delta[:, 2] = yaw_delta
        delta[:, 0] = delta[:, 0].clamp(-self.max_delta_vx, self.max_delta_vx)
        delta[:, 1] = delta[:, 1].clamp(-self.max_delta_vy, self.max_delta_vy)
        delta[:, 2] = delta[:, 2].clamp(-self.max_delta_yaw, self.max_delta_yaw)
        return speed_gate.unsqueeze(1) * delta, gap_width, gap_turn_need, gap_blocked

    def _compute_distillation_risk_from_tensors(
        self, command: torch.Tensor, obstacle_positions: torch.Tensor
    ) -> torch.Tensor:
        """Blend teacher imitation only where obstacle avoidance should matter."""
        cmd_xy = command[:, :2]
        cmd_speed = cmd_xy.norm(dim=1)
        speed_gate = (cmd_speed / self.min_command_speed).clamp(0.0, 1.0)
        speed_ratio = (cmd_speed / max(self.speed_reference, 1.0e-6)).clamp(0.0, 1.0)
        effective_safe_distance = self.safe_distance * (1.0 + self.high_speed_safe_distance_gain * speed_ratio)
        effective_local_avoid_distance = self.local_avoid_distance * (
            1.0 + self.high_speed_local_distance_gain * speed_ratio
        )

        cmd_dir = torch.zeros_like(cmd_xy)
        moving = cmd_speed > self.min_command_speed
        cmd_dir[moving] = cmd_xy[moving] / cmd_speed[moving].unsqueeze(1)
        cmd_dir[~moving, 0] = 1.0
        side_dir = torch.stack((-cmd_dir[:, 1], cmd_dir[:, 0]), dim=1)

        valid = obstacle_positions.abs().sum(dim=-1) > 1.0e-6
        distance_body = torch.sqrt(torch.clamp(obstacle_positions.square().sum(dim=-1), min=1.0e-6))
        local_risk = (
            valid
            * (distance_body < effective_local_avoid_distance.unsqueeze(1))
            * (1.0 - distance_body / effective_local_avoid_distance.unsqueeze(1)).clamp(0.0, 1.0)
        ).amax(dim=1)

        forward = (obstacle_positions * cmd_dir.unsqueeze(1)).sum(dim=-1)
        signed_lateral = (obstacle_positions * side_dir.unsqueeze(1)).sum(dim=-1)
        corridor_risk = (
            valid
            * (forward > self.min_forward_distance)
            * (forward < effective_safe_distance.unsqueeze(1))
            * (signed_lateral.abs() < self.corridor_half_width)
            * (1.0 - forward / effective_safe_distance.unsqueeze(1)).clamp(0.0, 1.0)
            * (1.0 - signed_lateral.abs() / self.corridor_half_width).clamp(0.0, 1.0)
        ).amax(dim=1)

        raw_risk = speed_gate * torch.maximum(local_risk, corridor_risk).clamp(0.0, 1.0)
        obstacle_present = torch.logical_or(local_risk > 1.0e-6, corridor_risk > 1.0e-6).float()
        floor_risk = speed_gate * self.obstacle_present_risk_floor * obstacle_present
        return torch.maximum(raw_risk, floor_risk).clamp(0.0, 1.0)

    def _compute_obstacle_representation_from_tensors(
        self, command: torch.Tensor, obstacle_positions: torch.Tensor
    ) -> torch.Tensor:
        """Project privileged obstacle positions into the geometry teacher actually uses."""
        batch_size = command.shape[0]
        device = command.device

        cmd_xy = command[:, :2]
        cmd_speed = cmd_xy.norm(dim=1)
        speed_ratio = (cmd_speed / max(self.speed_reference, 1.0e-6)).clamp(0.0, 1.0)
        effective_safe_distance = self.safe_distance * (1.0 + self.high_speed_safe_distance_gain * speed_ratio)

        cmd_dir = torch.zeros_like(cmd_xy)
        moving = cmd_speed > self.min_command_speed
        cmd_dir[moving] = cmd_xy[moving] / cmd_speed[moving].unsqueeze(1)
        cmd_dir[~moving, 0] = 1.0
        side_dir = torch.stack((-cmd_dir[:, 1], cmd_dir[:, 0]), dim=1)

        valid = obstacle_positions.abs().sum(dim=-1) > 1.0e-6
        forward = (obstacle_positions * cmd_dir.unsqueeze(1)).sum(dim=-1)
        signed_lateral = (obstacle_positions * side_dir.unsqueeze(1)).sum(dim=-1)

        forward_score = (
            (effective_safe_distance.unsqueeze(1) - forward)
            / effective_safe_distance.unsqueeze(1).clamp_min(1.0e-6)
        ).clamp(0.0, 1.0)

        center_score = (
            valid
            & (forward > self.min_forward_distance)
            & (forward < effective_safe_distance.unsqueeze(1))
            & (signed_lateral.abs() <= self.corridor_half_width)
        ).float() * forward_score * (
            1.0 - signed_lateral.abs() / self.corridor_half_width
        ).clamp(0.0, 1.0)

        half_lane_width = max(0.5 * self.robot_side_clearance, 1.0e-3)
        left_score = (
            valid
            & (forward > self.min_forward_distance)
            & (forward < effective_safe_distance.unsqueeze(1))
            & (signed_lateral >= 0.0)
            & (signed_lateral <= half_lane_width)
        ).float() * forward_score * (
            1.0 - signed_lateral / half_lane_width
        ).clamp(0.0, 1.0)
        right_score = (
            valid
            & (forward > self.min_forward_distance)
            & (forward < effective_safe_distance.unsqueeze(1))
            & (signed_lateral <= 0.0)
            & (-signed_lateral <= half_lane_width)
        ).float() * forward_score * (
            1.0 + signed_lateral / half_lane_width
        ).clamp(0.0, 1.0)

        center_blockage = center_score.amax(dim=1)
        left_blockage = left_score.amax(dim=1)
        right_blockage = right_score.amax(dim=1)

        side_lane_inner = self.corridor_half_width
        side_lane_outer = max(self.robot_side_clearance, self.corridor_half_width + self.obstacle_width + 0.1)

        def _masked_min(mask: torch.Tensor) -> torch.Tensor:
            masked_forward = torch.where(mask, forward, torch.full_like(forward, float("inf")))
            return masked_forward.amin(dim=1)

        left_lane_mask = (
            valid
            & (forward > self.min_forward_distance)
            & (forward < effective_safe_distance.unsqueeze(1))
            & (signed_lateral >= side_lane_inner)
            & (signed_lateral <= side_lane_outer)
        )
        right_lane_mask = (
            valid
            & (forward > self.min_forward_distance)
            & (forward < effective_safe_distance.unsqueeze(1))
            & (signed_lateral <= -side_lane_inner)
            & (signed_lateral >= -side_lane_outer)
        )

        left_open = (_masked_min(left_lane_mask) / effective_safe_distance).clamp(0.0, 1.0)
        right_open = (_masked_min(right_lane_mask) / effective_safe_distance).clamp(0.0, 1.0)
        left_open = torch.where(torch.isfinite(left_open), left_open, torch.ones_like(left_open))
        right_open = torch.where(torch.isfinite(right_open), right_open, torch.ones_like(right_open))

        threat_mask = center_score > 1.0e-6
        threat_score = torch.where(threat_mask, center_score, torch.full_like(center_score, -1.0))
        threat_idx = torch.argmax(threat_score, dim=1)
        has_threat = threat_mask.any(dim=1)
        batch_idx = torch.arange(batch_size, device=device)

        nearest_forward = torch.where(
            has_threat,
            forward[batch_idx, threat_idx] / effective_safe_distance.clamp_min(1.0e-6),
            torch.ones(batch_size, device=device),
        ).clamp(0.0, 1.0)
        nearest_lateral = torch.where(
            has_threat,
            signed_lateral[batch_idx, threat_idx] / self.corridor_half_width,
            torch.zeros(batch_size, device=device),
        ).clamp(-1.0, 1.0)

        weighted_lateral = (center_score * signed_lateral).sum(dim=1) / (center_score.sum(dim=1) + 1.0e-6)
        preferred_side = -torch.sign(weighted_lateral)
        preferred_side = torch.where(preferred_side == 0.0, torch.sign(right_open - left_open), preferred_side)
        preferred_side = torch.where(preferred_side == 0.0, torch.ones_like(preferred_side), preferred_side)

        return torch.stack(
            (
                nearest_forward,
                nearest_lateral,
                center_blockage,
                left_blockage,
                right_blockage,
                left_open,
                right_open,
                preferred_side,
            ),
            dim=-1,
        )

    def _run_frozen_actor_with_command(self, obs, adjusted_command: torch.Tensor) -> torch.Tensor:
        """Run the frozen LLC on a provided local navigation command."""
        policy_obs = obs[self.obs_group_name]
        modified_policy_obs = policy_obs.clone()
        modified_policy_obs[:, self.command_obs_start : self.command_obs_start + self.command_obs_dim] = adjusted_command

        modified_obs = {key: obs[key] for key in obs.keys()}
        modified_obs[self.obs_group_name] = modified_policy_obs
        return self.frozen_actor(modified_obs)

    def _smooth_delta(self, delta: torch.Tensor, cmd_speed: torch.Tensor | None = None) -> torch.Tensor:
        if self._smoothed_delta.shape != delta.shape:
            self._smoothed_delta = torch.zeros_like(delta)
        if cmd_speed is None:
            alpha = self.smoothing_alpha
        else:
            speed_ratio = (cmd_speed / max(self.speed_reference, 1.0e-6)).clamp(0.0, 1.0)
            alpha = self.smoothing_alpha - self.high_speed_smoothing_reduction * speed_ratio
            alpha = alpha.clamp(0.25, 0.85).unsqueeze(1)
        self._smoothed_delta = alpha * self._smoothed_delta + (1.0 - alpha) * delta
        return self._smoothed_delta

    def _update_debug_buffers(
        self,
        command: torch.Tensor,
        guidance_command: torch.Tensor,
        delta_cmd: torch.Tensor,
        adjusted_command: torch.Tensor,
        gap_width: torch.Tensor,
        gap_turn_need: torch.Tensor,
        gap_blocked: torch.Tensor,
    ) -> None:
        if self._last_base_command.shape != command.shape:
            self._last_base_command = torch.zeros_like(command)
        if self._last_guidance_command.shape != guidance_command.shape:
            self._last_guidance_command = torch.zeros_like(guidance_command)
        if self._last_delta_command.shape != delta_cmd.shape:
            self._last_delta_command = torch.zeros_like(delta_cmd)
        if self._last_adjusted_command.shape != adjusted_command.shape:
            self._last_adjusted_command = torch.zeros_like(adjusted_command)
        if self._last_gap_width.shape != gap_width.shape:
            self._last_gap_width = torch.zeros_like(gap_width)
        if self._last_gap_turn_need.shape != gap_turn_need.shape:
            self._last_gap_turn_need = torch.zeros_like(gap_turn_need)
        if self._last_gap_blocked.shape != gap_blocked.shape:
            self._last_gap_blocked = torch.zeros_like(gap_blocked)

        self._last_base_command.copy_(command.detach())
        self._last_guidance_command.copy_(guidance_command.detach())
        self._last_delta_command.copy_(delta_cmd.detach())
        self._last_adjusted_command.copy_(adjusted_command.detach())
        self._last_gap_width.copy_(gap_width.detach())
        self._last_gap_turn_need.copy_(gap_turn_need.detach())
        self._last_gap_blocked.copy_(gap_blocked.detach())

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
