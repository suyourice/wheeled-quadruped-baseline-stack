# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simple action-imitation distillation for Go2-W navigation."""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms.distillation import Distillation


class SimpleActionDistillation(Distillation):
    """Action-imitation distillation with configurable loss weight.

    Teacher (RL, privileged polar-depth obs) and student (LiDAR obs) both output
    3D high-level velocity commands. The shared FrozenLLCActionTerm converts
    either command into final 16D robot actions.
    """

    def __init__(
        self,
        *args,
        action_loss_weight: float = 1.0,
        safety_loss_weight: float = 0.0,
        safety_clearance_threshold: float = 0.65,
        safety_weight_clip: float = 5.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.action_loss_weight = action_loss_weight
        self.safety_loss_weight = safety_loss_weight
        self.safety_clearance_threshold = safety_clearance_threshold
        self.safety_weight_clip = safety_weight_clip

    def _safety_weights(self, observations, action_loss: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Weight risky privileged-geometry states without exposing them to the student."""
        weights = torch.ones_like(action_loss)
        risk = torch.zeros_like(action_loss)
        if self.safety_loss_weight <= 0.0 or not hasattr(observations, "keys"):
            return weights, risk
        if "teacher" not in observations.keys():
            return weights, risk

        teacher_obs = observations["teacher"]
        if teacher_obs.ndim != 2 or teacher_obs.shape[-1] < 445:
            return weights, risk

        nav_start = 9 + 180
        full_start = nav_start + 16
        full_end = full_start + 15 * 16
        full_geometry = teacher_obs[:, full_start:full_end].reshape(-1, 15, 16)

        active = full_geometry[..., 0] > 0.5
        clearance = full_geometry[..., 10] * 8.0
        safe_clearance = float(self.safety_clearance_threshold)
        clearance_fill = torch.full_like(clearance, safe_clearance)
        min_clearance = torch.where(active, clearance, clearance_fill).min(dim=1).values
        clearance_risk = ((safe_clearance - min_clearance) / max(safe_clearance, 1.0e-6)).clamp(0.0, 1.0)

        frontal_blockage = teacher_obs[:, nav_start + 3].clamp(0.0, 1.0)
        goal_path_blockage = teacher_obs[:, nav_start + 12].clamp(0.0, 1.0)
        ttc_risk = teacher_obs[:, nav_start + 13].clamp(0.0, 1.0)
        risk = torch.maximum(clearance_risk, torch.maximum(ttc_risk, goal_path_blockage))
        risk = torch.maximum(risk, 0.5 * frontal_blockage)

        weights = (1.0 + self.safety_loss_weight * risk).clamp(max=max(1.0, self.safety_weight_clip))
        return weights, risk

    def update(self) -> dict[str, float]:
        self.num_updates += 1
        total_loss = 0.0
        total_weight = 0.0
        total_risk = 0.0
        cnt = 0
        loss = 0

        for _ in range(self.num_learning_epochs):
            self.student.reset(hidden_state=self.last_hidden_states[0])
            self.teacher.reset(hidden_state=self.last_hidden_states[1])
            self.student.detach_hidden_state()
            for batch in self.storage.generator():
                student_actions = self.student(batch.observations)
                teacher_actions = batch.privileged_actions.detach()
                action_loss = nn.functional.mse_loss(
                    student_actions, teacher_actions, reduction="none"
                ).mean(dim=-1)
                safety_weights, safety_risk = self._safety_weights(batch.observations, action_loss)
                behavior_loss = self.action_loss_weight * (
                    (action_loss * safety_weights).mean()
                    / safety_weights.mean().clamp(min=1.0e-6)
                )
                loss = loss + behavior_loss
                total_loss += behavior_loss.item()
                total_weight += safety_weights.mean().item()
                total_risk += safety_risk.mean().item()
                cnt += 1

                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.student.detach_hidden_state()
                    loss = 0

                self.student.reset(batch.dones.view(-1))
                self.teacher.reset(batch.dones.view(-1))
                self.student.detach_hidden_state(batch.dones.view(-1))

        self.storage.clear()
        self.last_hidden_states = (
            self.student.get_hidden_state(),
            self.teacher.get_hidden_state(),
        )
        self.student.detach_hidden_state()
        return {
            "behavior": total_loss / max(cnt, 1),
            "safety_weight": total_weight / max(cnt, 1),
            "safety_risk": total_risk / max(cnt, 1),
        }
