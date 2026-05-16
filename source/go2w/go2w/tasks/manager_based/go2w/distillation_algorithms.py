# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simple action-imitation distillation for Go2-W navigation."""

from __future__ import annotations

import torch.nn as nn

from rsl_rl.algorithms.distillation import Distillation


class SimpleActionDistillation(Distillation):
    """Action-imitation distillation with configurable loss weight.

    Teacher (RL, privileged polar-depth obs) and student (LiDAR obs) both output
    3D high-level velocity commands. The shared FrozenLLCActionTerm converts
    either command into final 16D robot actions.
    """

    def __init__(self, *args, action_loss_weight: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.action_loss_weight = action_loss_weight

    def update(self) -> dict[str, float]:
        self.num_updates += 1
        total_loss = 0.0
        cnt = 0
        loss = 0

        for _ in range(self.num_learning_epochs):
            self.student.reset(hidden_state=self.last_hidden_states[0])
            self.teacher.reset(hidden_state=self.last_hidden_states[1])
            self.student.detach_hidden_state()
            for batch in self.storage.generator():
                student_actions = self.student(batch.observations)
                teacher_actions = batch.privileged_actions.detach()
                behavior_loss = self.action_loss_weight * nn.functional.mse_loss(
                    student_actions, teacher_actions
                )
                loss = loss + behavior_loss
                total_loss += behavior_loss.item()
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
        return {"behavior": total_loss / max(cnt, 1)}
