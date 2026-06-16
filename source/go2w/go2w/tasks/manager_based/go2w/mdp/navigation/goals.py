"""Navigation goal-buffer helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ensure_navigation_goal_buffers(env: ManagerBasedRLEnv) -> None:
    """Create persistent start/goal buffers used by the navigation-distill task."""
    if not hasattr(env, "_go2w_goal_pos_w"):
        env._go2w_goal_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._go2w_goal_heading_w = torch.zeros(env.num_envs, device=env.device)
        env._go2w_start_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._go2w_start_heading_w = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_go2w_scenario_template_id"):
        env._go2w_scenario_template_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_go2w_initial_scenario_template_id"):
        env._go2w_initial_scenario_template_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    # Passable narrow-gap metadata: gap centerline in world frame, half width, and
    # a passable flag set only for the narrow_gap / narrow_gap_wide / narrow_gap_barely
    # scenarios. Reward helpers use these to encourage decisive gap traversal.
    if not hasattr(env, "_go2w_gap_center_w"):
        env._go2w_gap_center_w = torch.zeros(env.num_envs, 2, device=env.device)
        env._go2w_gap_dir_w = torch.zeros(env.num_envs, 2, device=env.device)
        env._go2w_gap_half_width = torch.zeros(env.num_envs, device=env.device)
        env._go2w_gap_passable = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_go2w_gap_free_half_width"):
        env._go2w_gap_free_half_width = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_go2w_gap_center_tolerance"):
        env._go2w_gap_center_tolerance = torch.zeros(env.num_envs, device=env.device)
    # Per-env stuck counter for cluttered/blocked recovery diagnostics and gating.
    if not hasattr(env, "_go2w_stuck_counter"):
        env._go2w_stuck_counter = torch.zeros(env.num_envs, device=env.device)


__all__ = ["ensure_navigation_goal_buffers"]
