# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation obstacle slotting helpers shared by random and structured resets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _separated_parked_positions(parked_world: torch.Tensor, num_slots: int) -> torch.Tensor:
    """Return distant parking poses separated so inactive assets cannot contact each other."""
    parked_positions = parked_world.unsqueeze(1).expand(-1, num_slots, -1).clone()
    slot_offsets = torch.arange(num_slots, device=parked_world.device, dtype=parked_world.dtype)
    parked_positions[:, :, 1] += slot_offsets.unsqueeze(0)
    return parked_positions


def _physical_slot_randomization_mask(
    env: ManagerBasedRLEnv,
    n: int,
    randomize_slots: bool,
    start_iteration: int,
    warmup_iterations: int,
    steps_per_iteration: int,
    device: torch.device,
) -> bool | torch.Tensor:
    """Return per-env physical-slot randomization enablement."""
    from ..common.curriculum import _curriculum_progress

    if not randomize_slots:
        return False
    progress = _curriculum_progress(env, start_iteration, warmup_iterations, steps_per_iteration)
    if progress <= 0.0:
        return False
    if progress >= 1.0:
        return True
    return torch.rand(n, device=device) < progress


def _assign_logical_positions_to_physical_slots(
    logical_positions: torch.Tensor,
    logical_active: torch.Tensor,
    parked_positions: torch.Tensor,
    randomize_slots: bool | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map sampled layout positions onto physical obstacle assets per environment."""
    n, k = logical_active.shape
    device = logical_positions.device
    logical_to_physical = torch.arange(k, device=device).unsqueeze(0).expand(n, -1)
    if isinstance(randomize_slots, torch.Tensor):
        random_perm = torch.rand((n, k), device=device).argsort(dim=1)
        logical_to_physical = torch.where(randomize_slots.unsqueeze(1), random_perm, logical_to_physical)
    elif randomize_slots:
        logical_to_physical = torch.rand((n, k), device=device).argsort(dim=1)

    physical_positions = parked_positions.clone()
    position_index = logical_to_physical.unsqueeze(-1).expand(-1, -1, logical_positions.shape[-1])
    physical_positions.scatter_(1, position_index, logical_positions)
    physical_active = torch.zeros_like(logical_active)
    physical_active.scatter_(1, logical_to_physical, logical_active)
    return physical_positions, physical_active, logical_to_physical
