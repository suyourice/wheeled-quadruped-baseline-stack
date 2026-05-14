"""Active distillation algorithm for local target-pose navigation."""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms.distillation import Distillation

from .observation_layout import DEBUG_OBS, DEBUG_OBSTACLE_START


SCENARIO_TEMPLATE_NAMES = {
    0: "empty",
    1: "head_on",
    2: "left_edge",
    3: "right_edge",
    4: "diag_left",
    5: "diag_right",
    6: "off_left",
    7: "off_right",
    8: "narrow_gap",
    9: "random_fallback",
}


class NavigationCommandDistillation(Distillation):
    """Distill teacher local target poses while preserving nominal locomotion elsewhere.

    The teacher and student both step the environment through the same frozen LLC.
    Distillation therefore focuses on the intermediate navigation target:

        privileged geometry -> teacher local target pose
        LiDAR perception    -> student local target pose

    A deterministic pose-to-command converter then maps the target pose to
    `(vx, vy, yaw)` before the LLC executes it.

    When the teacher does not perceive meaningful obstacle risk, the student is
    pulled back toward the base command rather than copying any smoothed residual.
    """

    def __init__(
        self,
        *args,
        loss_type: str = "mse",
        command_loss_weight: float = 1.0,
        target_pose_loss_weight: float = 1.0,
        representation_loss_weight: float = 0.5,
        base_anchor_weight: float = 0.1,
        yaw_loss_weight: float = 0.1,
        delta_norm_loss_weight: float = 0.05,
        delta_norm_margin: float = 0.15,
        near_waypoint_command_discount: float = 0.5,
        near_waypoint_anchor_bonus: float = 0.25,
        blocked_lateral_yaw_weight: float = 1.0,
        blocked_vx_weight: float = 1.0,
        hard_case_delta_norm_threshold: float = 0.25,
        blocked_risk_threshold: float = 0.06,
        near_waypoint_distance_threshold: float = 0.9,
        near_waypoint_heading_threshold: float = 0.7,
        near_goal_distance_threshold: float = 0.9,
        side_target_vy_scale: float = 0.8,
        side_target_yaw_scale: float = 1.5,
        side_target_deadband: float = 0.05,
        debug_obstacle_print_interval: int = 10,
        debug_obstacle_print_count: int = 6,
        debug_rollout_print_interval: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(*args, loss_type=loss_type, **kwargs)
        self.loss_type = loss_type
        self.command_loss_weight = command_loss_weight
        self.target_pose_loss_weight = target_pose_loss_weight
        self.representation_loss_weight = representation_loss_weight
        self.base_anchor_weight = base_anchor_weight
        self.yaw_loss_weight = yaw_loss_weight
        self.delta_norm_loss_weight = delta_norm_loss_weight
        self.delta_norm_margin = delta_norm_margin
        self.near_waypoint_command_discount = near_waypoint_command_discount
        self.near_waypoint_anchor_bonus = near_waypoint_anchor_bonus
        self.blocked_lateral_yaw_weight = blocked_lateral_yaw_weight
        self.blocked_vx_weight = blocked_vx_weight
        self.hard_case_delta_norm_threshold = hard_case_delta_norm_threshold
        self.blocked_risk_threshold = blocked_risk_threshold
        self.near_waypoint_distance_threshold = near_waypoint_distance_threshold
        self.near_waypoint_heading_threshold = near_waypoint_heading_threshold
        self.near_goal_distance_threshold = near_goal_distance_threshold
        self.side_target_vy_scale = side_target_vy_scale
        self.side_target_yaw_scale = side_target_yaw_scale
        self.side_target_deadband = side_target_deadband
        self.debug_obstacle_print_interval = debug_obstacle_print_interval
        self.debug_obstacle_print_count = debug_obstacle_print_count
        self.debug_rollout_print_interval = debug_rollout_print_interval
        self._debug_rollout_step = 0
        self._pending_rollout_debug: dict[str, torch.Tensor] | None = None
        self._blocked_command_axis_weights = torch.tensor(
            [self.blocked_vx_weight, self.blocked_lateral_yaw_weight, self.blocked_lateral_yaw_weight],
            device=self.device,
            dtype=torch.float32,
        )
        self._debug_root_pos_slice = DEBUG_OBS["root_position_w"].as_slice()
        self._debug_base_lin_vel_slice = DEBUG_OBS["base_lin_vel"].as_slice()
        self._debug_base_ang_vel_slice = DEBUG_OBS["base_ang_vel"].as_slice()
        self._debug_goal_command_slice = DEBUG_OBS["goal_command"].as_slice()
        self._debug_joint_pos_slice = DEBUG_OBS["joint_pos"].as_slice()
        self._debug_joint_vel_slice = DEBUG_OBS["joint_vel"].as_slice()
        self._debug_actions_slice = DEBUG_OBS["actions"].as_slice()
        self._debug_start_pos_slice = DEBUG_OBS["start_position_w"].as_slice()
        self._debug_waypoint_pos_slice = DEBUG_OBS["waypoint_position_w"].as_slice()
        self._debug_goal_pos_slice = DEBUG_OBS["goal_position_w"].as_slice()
        self._debug_scenario_code_slice = DEBUG_OBS["scenario_template_code"].as_slice()
        self._debug_obstacle_start = DEBUG_OBSTACLE_START

    def act(self, obs):
        """Sample student actions and cache teacher navigation-command supervision."""
        self.transition.actions = self.student(obs, stochastic_output=False).detach()

        teacher_model = getattr(self, "_raw_teacher", self.teacher)
        teacher_actions = teacher_model(obs).detach()

        self.transition.privileged_actions = teacher_actions
        self.transition.observations = obs

        if self._should_print_rollout_debug():
            student_model = getattr(self, "_raw_student", self.student)
            student_aux = student_model.get_aux_outputs(obs)
            teacher_command = teacher_model.compute_navigation_command(obs).detach()
            self._pending_rollout_debug = {
                "debug_obs0": obs["debug"][0].detach().clone(),
                "student_command0": student_aux["navigation_command"][0].detach().clone(),
                "student_nominal_command0": student_aux["base_command"][0].detach().clone(),
                "teacher_command0": teacher_command[0].detach().clone(),
                "student_target_pose0": student_aux["target_pose"][0].detach().clone(),
            }
        else:
            self._pending_rollout_debug = None

        return self.transition.actions  # type: ignore[return-value]

    def process_env_step(self, obs, rewards, dones, extras):
        """Record one environment step."""
        del extras
        self.student.update_normalization(self.transition.observations)
        self.transition.rewards = rewards
        self.transition.dones = dones

        self.storage.add_transition(self.transition)

        if self._pending_rollout_debug is not None:
            self._print_rollout_debug_snapshot(obs, dones)
        self._debug_rollout_step += 1

        self.transition.clear()
        self.student.reset(dones)
        self.teacher.reset(dones)

    def update(self) -> dict[str, float]:
        """Run target-pose distillation with blocked-scene emphasis."""
        self.num_updates += 1

        mean_behavior_loss = 0.0
        mean_target_pose_loss = 0.0
        mean_target_pose_position_loss = 0.0
        mean_command_loss = 0.0
        mean_delta_loss = 0.0
        mean_yaw_loss = 0.0
        mean_delta_norm_excess_loss = 0.0
        mean_representation_loss = 0.0
        mean_rep_position_loss = 0.0
        mean_side_loss = 0.0
        mean_blocked_command_loss = 0.0
        mean_blocked_delta_loss = 0.0
        mean_student_delta_norm = 0.0
        mean_teacher_delta_norm = 0.0
        mean_teacher_nearest_forward = 0.0
        mean_teacher_nearest_lateral = 0.0
        mean_student_nearest_forward = 0.0
        mean_student_nearest_lateral = 0.0
        mean_hard_case_fraction = 0.0
        mean_near_waypoint_fraction = 0.0
        yaw_sign_sum = 0.0
        yaw_sign_count = 0.0
        hard_yaw_sign_sum = 0.0
        hard_yaw_sign_count = 0.0
        close_yaw_sign_sum = 0.0
        close_yaw_sign_count = 0.0
        lateral_sign_sum = 0.0
        lateral_sign_count = 0.0
        mean_close_obstacle_fraction = 0.0
        cnt = 0
        template_stats = {
            code: {
                "count": 0.0,
                "command_sum": 0.0,
                "command_count": 0.0,
                "yaw_agree_sum": 0.0,
                "yaw_agree_count": 0.0,
            }
            for code in SCENARIO_TEMPLATE_NAMES
        }

        accumulated_loss: torch.Tensor | None = None
        accumulated_steps = 0
        printed_debug_this_update = False

        for _ in range(self.num_learning_epochs):
            self.student.reset(hidden_state=self.last_hidden_states[0])
            self.teacher.reset(hidden_state=self.last_hidden_states[1])
            self.student.detach_hidden_state()

            for step_idx, batch in enumerate(self.storage.generator()):
                del step_idx
                student_model = getattr(self, "_raw_student", self.student)
                aux_outputs = student_model.get_aux_outputs(batch.observations)

                navigation_command = aux_outputs["navigation_command"]
                target_pose = aux_outputs["target_pose"]
                delta_cmd = aux_outputs["delta_cmd"]
                side = aux_outputs["side"]
                base_command = aux_outputs["base_command"]
                obstacle_representation = aux_outputs["obstacle_representation"]

                teacher_model = getattr(self, "_raw_teacher", self.teacher)
                teacher_command = teacher_model.compute_navigation_command(batch.observations).detach()
                teacher_target_pose = teacher_model.navigation_command_to_target_pose(teacher_command).detach()
                teacher_delta_cmd = teacher_model.last_delta_command.detach().clone()
                teacher_representation = teacher_model.compute_obstacle_representation(batch.observations).detach()
                risk = teacher_model.compute_distillation_risk(batch.observations).detach().unsqueeze(-1).clamp(0.0, 1.0)
                obstacle_present = (risk > 1.0e-6).float()
                blocked = (risk > self.blocked_risk_threshold).float()
                teacher_delta_norm_per_env = teacher_delta_cmd.norm(dim=-1, keepdim=True)
                hard_case = (teacher_delta_norm_per_env > self.hard_case_delta_norm_threshold).float()
                student_goal_command = batch.observations["student"][:, self._debug_goal_command_slice]
                debug_obs = batch.observations["debug"]
                root_pos_w = debug_obs[:, self._debug_root_pos_slice]
                goal_pos_w = debug_obs[:, self._debug_goal_pos_slice]
                scenario_code = (
                    debug_obs[:, self._debug_scenario_code_slice].squeeze(-1).round().to(dtype=torch.long)
                )
                head_on_case = (scenario_code == 1).float().unsqueeze(-1)
                diagonal_case = ((scenario_code == 4) | (scenario_code == 5)).float().unsqueeze(-1)
                narrow_gap_case = (scenario_code == 8).float().unsqueeze(-1)
                goal_distance = torch.norm(goal_pos_w[:, :2] - root_pos_w[:, :2], dim=-1, keepdim=True)
                near_waypoint = (
                    (student_goal_command[:, :2].norm(dim=-1, keepdim=True) <= self.near_waypoint_distance_threshold)
                    & (student_goal_command[:, 2:3].abs() <= self.near_waypoint_heading_threshold)
                )
                near_goal = goal_distance <= self.near_goal_distance_threshold
                near_waypoint = torch.logical_or(near_waypoint, near_goal).float()
                close_obstacle = (
                    obstacle_present
                    * (teacher_representation[:, 0:1] <= 0.9).float()
                    * (teacher_representation[:, 0:1] > 0.0).float()
                )
                side_target = self._compute_side_target(teacher_delta_cmd)

                command_loss_per_env = self._per_sample_loss(
                    navigation_command, teacher_command, axis_weights=self._blocked_command_axis_weights
                )
                target_pose_loss_per_env = self._per_sample_loss(target_pose, teacher_target_pose)
                target_pose_position_loss_per_env = self._per_sample_loss(
                    target_pose[:, :2], teacher_target_pose[:, :2]
                )
                delta_loss_per_env = self._per_sample_loss(
                    delta_cmd, teacher_delta_cmd, axis_weights=self._blocked_command_axis_weights
                )
                yaw_loss_per_env = nn.functional.smooth_l1_loss(
                    delta_cmd[:, 2],
                    teacher_delta_cmd[:, 2],
                    reduction="none",
                )
                student_delta_norm_per_env = delta_cmd.norm(dim=-1, keepdim=True)
                delta_excess = (
                    student_delta_norm_per_env.squeeze(-1)
                    - teacher_delta_norm_per_env.squeeze(-1)
                    - self.delta_norm_margin
                ).clamp_min(0.0)
                delta_norm_excess_loss_per_env = delta_excess.square()
                representation_loss_per_env = self._per_sample_loss(
                    obstacle_representation,
                    teacher_representation,
                )
                rep_position_loss_per_env = self._per_sample_loss(
                    obstacle_representation[:, :2],
                    teacher_representation[:, :2],
                )
                base_anchor_loss_per_env = self._per_sample_loss(navigation_command, base_command)
                side_loss_per_env = self._per_sample_loss(side, side_target)
                yaw_sign_agreement, yaw_sign_count_batch = self._sign_agreement(
                    delta_cmd[:, 2],
                    teacher_delta_cmd[:, 2],
                    blocked.squeeze(-1),
                )
                hard_yaw_sign_agreement, hard_yaw_sign_count_batch = self._sign_agreement(
                    delta_cmd[:, 2],
                    teacher_delta_cmd[:, 2],
                    (blocked * hard_case).squeeze(-1),
                )
                close_yaw_sign_agreement, close_yaw_sign_count_batch = self._sign_agreement(
                    delta_cmd[:, 2],
                    teacher_delta_cmd[:, 2],
                    (blocked * close_obstacle).squeeze(-1),
                )
                lateral_sign_agreement, lateral_sign_count_batch = self._sign_agreement(
                    delta_cmd[:, 1],
                    teacher_delta_cmd[:, 1],
                    blocked.squeeze(-1),
                )

                if not printed_debug_this_update and self._should_print_debug():
                    self._print_obstacle_debug_snapshot(
                        teacher_model=teacher_model,
                        batch_observations=batch.observations,
                        teacher_representation=teacher_representation,
                        student_representation=obstacle_representation.detach(),
                        teacher_target_pose=teacher_target_pose,
                        student_target_pose=target_pose.detach(),
                        teacher_command=teacher_command,
                        student_command=navigation_command.detach(),
                        risk=risk,
                        blocked=blocked,
                        scenario_code=scenario_code,
                    )
                    printed_debug_this_update = True

                blocked_count = blocked.sum().clamp_min(1.0)
                blocked_command_loss = (command_loss_per_env.unsqueeze(-1) * blocked).sum() / blocked_count
                blocked_delta_loss = (delta_loss_per_env.unsqueeze(-1) * blocked).sum() / blocked_count
                student_delta_norm = delta_cmd.norm(dim=-1).mean()
                teacher_delta_norm = teacher_delta_cmd.norm(dim=-1).mean()
                # Active training baseline: keep only the four most interpretable
                # supervision terms, then add small yaw and excess-delta guards
                # for obstacle cases where the student tends to over-correct.
                risk_weight = risk.squeeze(-1)
                obstacle_weight = obstacle_present.squeeze(-1)
                near_weight = near_waypoint.squeeze(-1)
                command_weight = risk_weight * (1.0 - self.near_waypoint_command_discount * near_weight)
                command_weight = command_weight.clamp_min(0.0)
                anchor_weight = (1.0 - risk_weight) + self.near_waypoint_anchor_bonus * near_weight
                behavior_loss = (
                    self.target_pose_loss_weight
                    * obstacle_weight
                    * target_pose_loss_per_env
                    + self.command_loss_weight
                    * command_weight
                    * command_loss_per_env
                    + self.representation_loss_weight
                    * obstacle_weight
                    * representation_loss_per_env
                    + self.base_anchor_weight
                    * anchor_weight
                    * base_anchor_loss_per_env
                    + self.yaw_loss_weight
                    * command_weight
                    * yaw_loss_per_env
                    + self.delta_norm_loss_weight
                    * risk_weight
                    * delta_norm_excess_loss_per_env
                ).mean()

                accumulated_loss = behavior_loss if accumulated_loss is None else accumulated_loss + behavior_loss
                accumulated_steps += 1

                mean_behavior_loss += behavior_loss.item()
                mean_target_pose_loss += target_pose_loss_per_env.mean().item()
                mean_target_pose_position_loss += target_pose_position_loss_per_env.mean().item()
                mean_command_loss += command_loss_per_env.mean().item()
                mean_delta_loss += delta_loss_per_env.mean().item()
                mean_yaw_loss += yaw_loss_per_env.mean().item()
                mean_delta_norm_excess_loss += delta_norm_excess_loss_per_env.mean().item()
                mean_representation_loss += representation_loss_per_env.mean().item()
                mean_rep_position_loss += rep_position_loss_per_env.mean().item()
                mean_side_loss += side_loss_per_env.mean().item()
                mean_blocked_command_loss += blocked_command_loss.item()
                mean_blocked_delta_loss += blocked_delta_loss.item()
                mean_student_delta_norm += student_delta_norm.item()
                mean_teacher_delta_norm += teacher_delta_norm.item()
                mean_teacher_nearest_forward += teacher_representation[:, 0].mean().item()
                mean_teacher_nearest_lateral += teacher_representation[:, 1].mean().item()
                mean_student_nearest_forward += obstacle_representation[:, 0].mean().item()
                mean_student_nearest_lateral += obstacle_representation[:, 1].mean().item()
                mean_hard_case_fraction += hard_case.mean().item()
                mean_near_waypoint_fraction += near_waypoint.mean().item()
                yaw_sign_sum += yaw_sign_agreement * yaw_sign_count_batch
                yaw_sign_count += yaw_sign_count_batch
                hard_yaw_sign_sum += hard_yaw_sign_agreement * hard_yaw_sign_count_batch
                hard_yaw_sign_count += hard_yaw_sign_count_batch
                close_yaw_sign_sum += close_yaw_sign_agreement * close_yaw_sign_count_batch
                close_yaw_sign_count += close_yaw_sign_count_batch
                lateral_sign_sum += lateral_sign_agreement * lateral_sign_count_batch
                lateral_sign_count += lateral_sign_count_batch
                mean_close_obstacle_fraction += close_obstacle.mean().item()
                cnt += 1

                for code in SCENARIO_TEMPLATE_NAMES:
                    mask = scenario_code == code
                    if not bool(mask.any()):
                        continue
                    template_stats[code]["count"] += float(mask.float().sum().item())
                    template_stats[code]["command_sum"] += float(command_loss_per_env[mask].sum().item())
                    template_stats[code]["command_count"] += float(mask.float().sum().item())
                    tpl_yaw_agreement, tpl_yaw_count = self._sign_agreement(
                        delta_cmd[mask, 2],
                        teacher_delta_cmd[mask, 2],
                        blocked[mask].squeeze(-1),
                    )
                    if tpl_yaw_count > 0.0:
                        template_stats[code]["yaw_agree_sum"] += tpl_yaw_agreement * tpl_yaw_count
                        template_stats[code]["yaw_agree_count"] += tpl_yaw_count

                if accumulated_steps >= self.gradient_length:
                    self._step_optimizer(accumulated_loss)
                    accumulated_loss = None
                    accumulated_steps = 0

                dones = batch.dones.view(-1)
                self.student.reset(dones)
                self.teacher.reset(dones)
                self.student.detach_hidden_state(dones)

            if accumulated_steps > 0 and accumulated_loss is not None:
                self._step_optimizer(accumulated_loss)
                accumulated_loss = None
                accumulated_steps = 0

        mean_behavior_loss /= max(cnt, 1)
        mean_target_pose_loss /= max(cnt, 1)
        mean_target_pose_position_loss /= max(cnt, 1)
        mean_command_loss /= max(cnt, 1)
        mean_delta_loss /= max(cnt, 1)
        mean_yaw_loss /= max(cnt, 1)
        mean_delta_norm_excess_loss /= max(cnt, 1)
        mean_representation_loss /= max(cnt, 1)
        mean_rep_position_loss /= max(cnt, 1)
        mean_side_loss /= max(cnt, 1)
        mean_blocked_command_loss /= max(cnt, 1)
        mean_blocked_delta_loss /= max(cnt, 1)
        mean_student_delta_norm /= max(cnt, 1)
        mean_teacher_delta_norm /= max(cnt, 1)
        mean_teacher_nearest_forward /= max(cnt, 1)
        mean_teacher_nearest_lateral /= max(cnt, 1)
        mean_student_nearest_forward /= max(cnt, 1)
        mean_student_nearest_lateral /= max(cnt, 1)
        mean_hard_case_fraction /= max(cnt, 1)
        mean_near_waypoint_fraction /= max(cnt, 1)
        mean_yaw_sign_agreement = yaw_sign_sum / max(yaw_sign_count, 1.0)
        mean_hard_yaw_sign_agreement = hard_yaw_sign_sum / max(hard_yaw_sign_count, 1.0)
        mean_close_yaw_sign_agreement = close_yaw_sign_sum / max(close_yaw_sign_count, 1.0)
        mean_lateral_sign_agreement = lateral_sign_sum / max(lateral_sign_count, 1.0)
        mean_close_obstacle_fraction /= max(cnt, 1)

        self.storage.clear()
        self.last_hidden_states = (self.student.get_hidden_state(), self.teacher.get_hidden_state())
        self.student.detach_hidden_state()

        if self._should_print_debug():
            parts = []
            total_count = sum(stats["count"] for stats in template_stats.values())
            for code, name in SCENARIO_TEMPLATE_NAMES.items():
                stats = template_stats[code]
                if stats["count"] <= 0.0:
                    continue
                frac = stats["count"] / max(total_count, 1.0)
                blocked_cmd = stats["command_sum"] / max(stats["command_count"], 1.0)
                if stats["yaw_agree_count"] > 0.0:
                    yaw_agree = stats["yaw_agree_sum"] / stats["yaw_agree_count"]
                    yaw_text = f"{yaw_agree:.2f}"
                else:
                    yaw_text = "n/a"
                parts.append(f"{name}:frac={frac:.2f},cmd={blocked_cmd:.2f},yaw={yaw_text}")
            if parts:
                print(f"[nav-template-breakdown] update={self.num_updates} " + " | ".join(parts))

        return {
            "behavior": mean_behavior_loss,
            "target_pose": mean_target_pose_loss,
            "target_pose_position": mean_target_pose_position_loss,
            "command": mean_command_loss,
            "delta_cmd": mean_delta_loss,
            "yaw": mean_yaw_loss,
            "delta_norm_excess": mean_delta_norm_excess_loss,
            "representation": mean_representation_loss,
            "rep_position": mean_rep_position_loss,
            "blocked_command": mean_blocked_command_loss,
            "blocked_delta": mean_blocked_delta_loss,
            "side": mean_side_loss,
            "student_delta_norm": mean_student_delta_norm,
            "teacher_delta_norm": mean_teacher_delta_norm,
            "teacher_nearest_forward": mean_teacher_nearest_forward,
            "teacher_nearest_lateral": mean_teacher_nearest_lateral,
            "student_nearest_forward": mean_student_nearest_forward,
            "student_nearest_lateral": mean_student_nearest_lateral,
            "hard_case_fraction": mean_hard_case_fraction,
            "near_waypoint_fraction": mean_near_waypoint_fraction,
            "yaw_sign_agreement": mean_yaw_sign_agreement,
            "hard_yaw_sign_agreement": mean_hard_yaw_sign_agreement,
            "close_yaw_sign_agreement": mean_close_yaw_sign_agreement,
            "lateral_sign_agreement": mean_lateral_sign_agreement,
            "close_obstacle_fraction": mean_close_obstacle_fraction,
        }

    def _per_sample_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        axis_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.loss_type == "mse":
            per_element = nn.functional.mse_loss(prediction, target, reduction="none")
        elif self.loss_type == "huber":
            per_element = nn.functional.huber_loss(prediction, target, reduction="none")
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        if axis_weights is not None:
            per_element = per_element * axis_weights
        return per_element.mean(dim=-1)

    def _step_optimizer(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        if self.is_multi_gpu:
            self.reduce_parameters()
        if self.max_grad_norm:
            nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.student.detach_hidden_state()

    def _sign_agreement(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        magnitude_threshold: float = 0.2,
    ) -> tuple[float, float]:
        """Return sign-agreement and valid-count for non-trivial target magnitudes."""
        valid = active_mask.bool() & (target.abs() > magnitude_threshold)
        if not bool(valid.any()):
            return 0.0, 0.0
        agreement = (torch.sign(prediction[valid]) == torch.sign(target[valid])).float().mean().item()
        count = float(valid.float().sum().item())
        return agreement, count

    def _compute_side_target(self, teacher_delta_cmd: torch.Tensor) -> torch.Tensor:
        """Convert teacher steering correction into a simple left/right side target."""
        lateral_target = (teacher_delta_cmd[..., 1:2] / self.side_target_vy_scale).clamp(-1.0, 1.0)
        yaw_target = (teacher_delta_cmd[..., 2:3] / self.side_target_yaw_scale).clamp(-1.0, 1.0)
        use_yaw_fallback = lateral_target.abs() < self.side_target_deadband
        return torch.where(use_yaw_fallback, yaw_target, lateral_target)

    def _should_print_debug(self) -> bool:
        if self.debug_obstacle_print_interval <= 0:
            return False
        return self.num_updates == 1 or (self.num_updates % self.debug_obstacle_print_interval == 0)

    def _should_print_rollout_debug(self) -> bool:
        if self.debug_rollout_print_interval <= 0:
            return False
        return self._debug_rollout_step % self.debug_rollout_print_interval == 0

    def _check_debug_layout(self, debug_obs: torch.Tensor) -> None:
        expected_min_dim = self._debug_obstacle_start
        if debug_obs.shape[-1] < expected_min_dim:
            raise RuntimeError(
                "Navigation debug observation layout is inconsistent with the active "
                f"goal-conditioned task. Expected at least {expected_min_dim} dims before "
                f"obstacle positions, got {debug_obs.shape[-1]}."
            )

    def _print_obstacle_debug_snapshot(
        self,
        *,
        teacher_model,
        batch_observations,
        teacher_representation: torch.Tensor,
        student_representation: torch.Tensor,
        teacher_target_pose: torch.Tensor,
        student_target_pose: torch.Tensor,
        teacher_command: torch.Tensor,
        student_command: torch.Tensor,
        risk: torch.Tensor,
        blocked: torch.Tensor,
        scenario_code: torch.Tensor,
    ) -> None:
        teacher_obs = batch_observations[teacher_model.obs_group_name]
        debug_obs = batch_observations.get("debug", None)
        obstacle_slice = teacher_obs[
            :,
            teacher_model.obstacle_obs_start : teacher_model.obstacle_obs_start + teacher_model.obstacle_obs_dim,
        ]
        obstacle_positions = obstacle_slice.view(obstacle_slice.shape[0], -1, 2) * teacher_model.obstacle_max_distance
        sample_positions = obstacle_positions[0]
        valid = sample_positions.norm(dim=-1) > 1.0e-6
        sample_positions = sample_positions[valid]
        if self.debug_obstacle_print_count > 0:
            sample_positions = sample_positions[: self.debug_obstacle_print_count]

        pos_text = ", ".join(
            f"({xy[0].item():+.2f},{xy[1].item():+.2f})"
            for xy in sample_positions
        )
        teacher_rep = ", ".join(f"{x.item():+.3f}" for x in teacher_representation[0])
        student_rep = ", ".join(f"{x.item():+.3f}" for x in student_representation[0])
        teacher_pose = ", ".join(f"{x.item():+.3f}" for x in teacher_target_pose[0])
        student_pose = ", ".join(f"{x.item():+.3f}" for x in student_target_pose[0])
        teacher_cmd_text = ", ".join(f"{x.item():+.3f}" for x in teacher_command[0])
        student_cmd_text = ", ".join(f"{x.item():+.3f}" for x in student_command[0])

        if debug_obs is not None:
            self._check_debug_layout(debug_obs)
            root_pos = debug_obs[0, self._debug_root_pos_slice]
            base_lin_vel = debug_obs[0, self._debug_base_lin_vel_slice]
            base_ang_vel = debug_obs[0, self._debug_base_ang_vel_slice]
            goal_command = debug_obs[0, self._debug_goal_command_slice]
            joint_pos = debug_obs[0, self._debug_joint_pos_slice]
            joint_vel = debug_obs[0, self._debug_joint_vel_slice]
            start_pos = debug_obs[0, self._debug_start_pos_slice]
            waypoint_pos = debug_obs[0, self._debug_waypoint_pos_slice]
            goal_pos = debug_obs[0, self._debug_goal_pos_slice]
        else:
            root_pos = None
            base_lin_vel = teacher_obs[0, 0:3]
            base_ang_vel = teacher_obs[0, 3:6]
            goal_command = teacher_obs[0, 9:12]
            joint_pos = teacher_obs[0, 12:28]
            joint_vel = teacher_obs[0, 28:44]
            start_pos = None
            waypoint_pos = None
            goal_pos = None

        root_pos_text = "n/a" if root_pos is None else ", ".join(f"{x.item():+.3f}" for x in root_pos)
        base_lin_vel_text = ", ".join(f"{x.item():+.3f}" for x in base_lin_vel)
        base_ang_vel_text = ", ".join(f"{x.item():+.3f}" for x in base_ang_vel)
        waypoint_cmd_text = ", ".join(f"{x.item():+.3f}" for x in goal_command)
        joint_pos_text = ", ".join(f"{x.item():+.3f}" for x in joint_pos)
        joint_vel_text = ", ".join(f"{x.item():+.3f}" for x in joint_vel)
        start_pos_text = "n/a" if start_pos is None else ", ".join(f"{x.item():+.3f}" for x in start_pos)
        waypoint_pos_text = "n/a" if waypoint_pos is None else ", ".join(f"{x.item():+.3f}" for x in waypoint_pos)
        goal_pos_text = "n/a" if goal_pos is None else ", ".join(f"{x.item():+.3f}" for x in goal_pos)
        scenario_name = SCENARIO_TEMPLATE_NAMES.get(int(scenario_code[0].item()), "unknown")
        if root_pos is not None and goal_pos is not None:
            waypoint_delta = waypoint_pos - root_pos
            goal_delta = goal_pos - root_pos
            waypoint_delta_text = ", ".join(f"{x.item():+.3f}" for x in waypoint_delta)
            goal_delta_text = ", ".join(f"{x.item():+.3f}" for x in goal_delta)
            waypoint_distance = waypoint_delta[:2].norm().item()
            goal_distance = goal_delta[:2].norm().item()
        else:
            waypoint_delta_text = "n/a"
            goal_delta_text = "n/a"
            waypoint_distance = float("nan")
            goal_distance = float("nan")

        print(
            "[nav-debug] "
            f"update={self.num_updates} "
            f"risk0={risk[0, 0].item():.3f} blocked0={blocked[0, 0].item():.0f} "
            f"root_pos0=[{root_pos_text}] "
            f"start_pos0=[{start_pos_text}] "
            f"scenario0={scenario_name} "
            f"waypoint_pos0=[{waypoint_pos_text}] "
            f"goal_pos0=[{goal_pos_text}] "
            f"waypoint_delta0=[{waypoint_delta_text}] "
            f"waypoint_dist0={waypoint_distance:.3f} "
            f"goal_delta0=[{goal_delta_text}] "
            f"goal_dist0={goal_distance:.3f} "
            f"base_lin_vel0=[{base_lin_vel_text}] "
            f"base_ang_vel0=[{base_ang_vel_text}] "
            f"waypoint_cmd0=[{waypoint_cmd_text}] "
            f"obstacles0=[{pos_text}] "
            f"joint_pos0=[{joint_pos_text}] "
            f"joint_vel0=[{joint_vel_text}] "
            f"teacher_rep0=[{teacher_rep}] "
            f"student_rep0=[{student_rep}] "
            f"teacher_pose0=[{teacher_pose}] "
            f"student_pose0=[{student_pose}] "
            f"teacher_cmd0=[{teacher_cmd_text}] "
            f"student_cmd0=[{student_cmd_text}]"
        )

    def _print_rollout_debug_snapshot(self, next_obs, dones: torch.Tensor) -> None:
        if self._pending_rollout_debug is None or "debug" not in next_obs:
            return

        prev_debug = self._pending_rollout_debug["debug_obs0"]
        next_debug = next_obs["debug"][0].detach()
        self._check_debug_layout(prev_debug.unsqueeze(0))
        self._check_debug_layout(next_debug.unsqueeze(0))
        prev_root_pos = prev_debug[self._debug_root_pos_slice]
        next_root_pos = next_debug[self._debug_root_pos_slice]
        prev_start_pos = prev_debug[self._debug_start_pos_slice]
        prev_waypoint_pos = prev_debug[self._debug_waypoint_pos_slice]
        prev_goal_pos = prev_debug[self._debug_goal_pos_slice]
        delta_root = next_root_pos - prev_root_pos
        prev_base_lin_vel = prev_debug[self._debug_base_lin_vel_slice]
        next_base_lin_vel = next_debug[self._debug_base_lin_vel_slice]
        prev_goal_command = prev_debug[self._debug_goal_command_slice]
        obstacle_positions = prev_debug[self._debug_obstacle_start :].view(-1, 2)
        valid = obstacle_positions.norm(dim=-1) > 1.0e-6
        obstacle_positions = obstacle_positions[valid]
        if self.debug_obstacle_print_count > 0:
            obstacle_positions = obstacle_positions[: self.debug_obstacle_print_count]

        pos_text = ", ".join(f"({xy[0].item():+.2f},{xy[1].item():+.2f})" for xy in obstacle_positions)
        delta_root_text = ", ".join(f"{x.item():+.3f}" for x in delta_root)
        prev_vel_text = ", ".join(f"{x.item():+.3f}" for x in prev_base_lin_vel)
        next_vel_text = ", ".join(f"{x.item():+.3f}" for x in next_base_lin_vel)
        waypoint_cmd_text = ", ".join(f"{x.item():+.3f}" for x in prev_goal_command)
        start_pos_text = ", ".join(f"{x.item():+.3f}" for x in prev_start_pos)
        waypoint_pos_text = ", ".join(f"{x.item():+.3f}" for x in prev_waypoint_pos)
        goal_pos_text = ", ".join(f"{x.item():+.3f}" for x in prev_goal_pos)
        scenario_code = int(prev_debug[self._debug_scenario_code_slice].item())
        scenario_name = SCENARIO_TEMPLATE_NAMES.get(scenario_code, "unknown")
        teacher_cmd_text = ", ".join(f"{x.item():+.3f}" for x in self._pending_rollout_debug["teacher_command0"])
        student_cmd_text = ", ".join(f"{x.item():+.3f}" for x in self._pending_rollout_debug["student_command0"])
        student_nominal_cmd_text = ", ".join(
            f"{x.item():+.3f}" for x in self._pending_rollout_debug["student_nominal_command0"]
        )
        student_pose_text = ", ".join(f"{x.item():+.3f}" for x in self._pending_rollout_debug["student_target_pose0"])
        prev_waypoint_delta = prev_waypoint_pos - prev_root_pos
        prev_goal_delta = prev_goal_pos - prev_root_pos
        next_waypoint_delta = prev_waypoint_pos - next_root_pos
        next_goal_delta = prev_goal_pos - next_root_pos
        prev_waypoint_dist = prev_waypoint_delta[:2].norm().item()
        next_waypoint_dist = next_waypoint_delta[:2].norm().item()
        prev_goal_dist = prev_goal_delta[:2].norm().item()
        next_goal_dist = next_goal_delta[:2].norm().item()
        waypoint_progress = prev_waypoint_dist - next_waypoint_dist
        goal_progress = prev_goal_dist - next_goal_dist
        prev_waypoint_delta_text = ", ".join(f"{x.item():+.3f}" for x in prev_waypoint_delta)
        next_waypoint_delta_text = ", ".join(f"{x.item():+.3f}" for x in next_waypoint_delta)
        prev_goal_delta_text = ", ".join(f"{x.item():+.3f}" for x in prev_goal_delta)
        next_goal_delta_text = ", ".join(f"{x.item():+.3f}" for x in next_goal_delta)

        print(
            "[nav-rollout] "
            f"update={self.num_updates} step={self._debug_rollout_step} done0={int(dones[0].item())} "
            f"start_pos0=[{start_pos_text}] "
            f"scenario0={scenario_name} "
            f"waypoint_pos0=[{waypoint_pos_text}] "
            f"goal_pos0=[{goal_pos_text}] "
            f"waypoint_cmd0=[{waypoint_cmd_text}] "
            f"waypoint_delta_prev0=[{prev_waypoint_delta_text}] "
            f"waypoint_delta_next0=[{next_waypoint_delta_text}] "
            f"waypoint_dist_prev0={prev_waypoint_dist:.3f} "
            f"waypoint_dist_next0={next_waypoint_dist:.3f} "
            f"waypoint_progress0={waypoint_progress:+.3f} "
            f"goal_delta_prev0=[{prev_goal_delta_text}] "
            f"goal_delta_next0=[{next_goal_delta_text}] "
            f"goal_dist_prev0={prev_goal_dist:.3f} "
            f"goal_dist_next0={next_goal_dist:.3f} "
            f"goal_progress0={goal_progress:+.3f} "
            f"student_nominal_cmd0=[{student_nominal_cmd_text}] "
            f"teacher_cmd0=[{teacher_cmd_text}] "
            f"student_cmd0=[{student_cmd_text}] "
            f"student_pose0=[{student_pose_text}] "
            f"prev_vel0=[{prev_vel_text}] "
            f"next_vel0=[{next_vel_text}] "
            f"delta_root0=[{delta_root_text}] "
            f"obstacles0=[{pos_text}]"
        )
