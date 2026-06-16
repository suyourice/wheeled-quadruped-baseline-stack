"""Curriculum helpers shared by locomotion and navigation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _curriculum_progress(
    env: ManagerBasedRLEnv,
    start_iteration: int,
    warmup_iterations: int,
    steps_per_iteration: int,
) -> float:
    """Return curriculum progress t in [0, 1] based on training step counter."""
    start_steps = start_iteration * steps_per_iteration
    warmup_steps = warmup_iterations * steps_per_iteration
    step = env.common_step_counter
    if step < start_steps:
        return 0.0
    if warmup_steps <= 0:
        return 1.0
    return max(0.0, min(1.0, (step - start_steps) / warmup_steps))


__all__ = ["_curriculum_progress"]
