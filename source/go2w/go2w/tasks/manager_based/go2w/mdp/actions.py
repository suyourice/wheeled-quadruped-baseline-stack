# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen LLC action term for Go2-W HLC navigation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
import torch.nn as nn

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class FrozenLLCActionTerm(ActionTerm):
    """HLC velocity command (3D) drives a frozen fast-flat LLC to produce 16D joint actions.

    HLC outputs [vx, vy, yaw].  This term reconstructs the 60D LLC observation from
    current robot state, feeds it through the frozen fast-flat MLP, and applies the
    resulting 16D joint targets (wheel velocity + hip/stance position).

    LLC obs layout (60D) matches POLICY_OBS in observation_layout.py:
        [0:3]  base_lin_vel_b
        [3:6]  base_ang_vel_b
        [6:9]  projected_gravity_b
        [9:12] velocity_cmd  <- HLC output
        [12:28] joint_pos_rel (articulation order, relative to default)
        [28:44] joint_vel    (articulation order)
        [44:60] last_action  (raw LLC MLP output from previous step, unscaled)

    Action scales matching FastFlatActionsCfg:
        wheel  (.*_foot_joint):                    velocity scale = 28.0
        hip    (.*_hip_joint):                     position scale = 0.35
        stance (.*_thigh_joint, .*_calf_joint):    position scale = 0.35
    """

    cfg: FrozenLLCActionTermCfg
    _asset: Articulation

    _SCALE_WHEEL: float = 28.0
    _SCALE_HIP: float = 0.35
    _SCALE_STANCE: float = 0.35

    def __init__(self, cfg: FrozenLLCActionTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self._num_envs = env.num_envs
        self._device = env.device

        # Resolve joint indices matching FastFlatActionsCfg order
        self._wheel_ids, _ = self._asset.find_joints(".*_foot_joint")
        self._hip_ids, _ = self._asset.find_joints(".*_hip_joint")
        self._stance_ids, _ = self._asset.find_joints([".*_thigh_joint", ".*_calf_joint"])

        # Buffers required by the ActionTerm contract
        self._raw_actions = torch.zeros(env.num_envs, self.action_dim, device=self._device)
        self._processed_actions = torch.zeros(env.num_envs, self.action_dim, device=self._device)

        # Raw (unscaled) 16D LLC MLP output from the previous step, used in the LLC obs
        self._llc_last_action = torch.zeros(env.num_envs, 16, device=self._device)
        self._llc_action = torch.zeros(env.num_envs, 16, device=self._device)

        self._frozen_actor = self._build_and_load_actor(cfg.llc_checkpoint_path)
        self._frozen_actor.to(self._device)
        self._frozen_actor.eval()
        for param in self._frozen_actor.parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------------
    # ActionTerm interface
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions.clamp(-2.0, 2.0)
        llc_obs = self._build_llc_obs(self._processed_actions)

        with torch.no_grad():
            self._llc_action[:] = self._frozen_actor(llc_obs)
        self._llc_last_action[:] = self._llc_action

    def apply_actions(self) -> None:
        default_pos = self._asset.data.default_joint_pos

        wheel_vel = self._SCALE_WHEEL * self._llc_action[:, 0:4]
        hip_pos = default_pos[:, self._hip_ids] + self._SCALE_HIP * self._llc_action[:, 4:8]
        stance_pos = default_pos[:, self._stance_ids] + self._SCALE_STANCE * self._llc_action[:, 8:16]

        self._asset.set_joint_velocity_target(wheel_vel, joint_ids=self._wheel_ids)
        self._asset.set_joint_position_target(hip_pos, joint_ids=self._hip_ids)
        self._asset.set_joint_position_target(stance_pos, joint_ids=self._stance_ids)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        self._llc_last_action[env_ids] = 0.0
        self._llc_action[env_ids] = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_llc_obs(self, velocity_cmd: torch.Tensor) -> torch.Tensor:
        """Reconstruct 60D LLC obs from current robot state and HLC velocity command."""
        robot = self._asset
        obs = torch.zeros(self._num_envs, 60, device=self._device)
        obs[:, 0:3] = robot.data.root_lin_vel_b
        obs[:, 3:6] = robot.data.root_ang_vel_b
        obs[:, 6:9] = robot.data.projected_gravity_b
        obs[:, 9:12] = velocity_cmd
        obs[:, 12:28] = robot.data.joint_pos - robot.data.default_joint_pos
        obs[:, 28:44] = robot.data.joint_vel
        obs[:, 44:60] = self._llc_last_action
        return obs

    def _build_and_load_actor(self, checkpoint_path: str) -> nn.Sequential:
        """Build [512, 256, 128] ELU MLP matching fast-flat actor and load checkpoint."""
        actor = nn.Sequential(
            nn.Linear(60, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 16),
        )
        if not checkpoint_path:
            raise ValueError(
                "FrozenLLCActionTerm requires llc_checkpoint_path. "
                "Pass --locomotion_checkpoint so the fast-flat LLC is loaded before env creation."
            )

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        for key in ("actor_state_dict", "model_state_dict", "policy_state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                sd = ckpt[key]
                break
        else:
            sd = ckpt

        actor_sd = {}
        for key, value in sd.items():
            if key.startswith("distribution.") or key.startswith("obs_normalizer."):
                continue
            stripped = key
            for prefix in ("actor.", "frozen_actor.", "mlp."):
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix):]
            if stripped in actor.state_dict():
                actor_sd[stripped] = value

        missing = sorted(set(actor.state_dict()) - set(actor_sd))
        if missing:
            raise RuntimeError(
                f"Could not map fast-flat actor weights from {checkpoint_path}; "
                f"missing keys: {missing}. Source keys start with: {list(sd.keys())[:10]}"
            )
        actor.load_state_dict(actor_sd)
        return actor


@configclass
class FrozenLLCActionTermCfg(ActionTermCfg):
    """Configuration for FrozenLLCActionTerm."""

    class_type: type = FrozenLLCActionTerm
    asset_name: str = "robot"
    llc_checkpoint_path: str = ""
