from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from deploy.common.go2w_deploy_config import Go2WDeployConfig, resolve_policy_path


class Go2WActor(nn.Module):
    def __init__(self, obs_dim: int = 60, action_dim: int = 16) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


class Go2WLLCPolicyWrapper:
    def __init__(self, config: Go2WDeployConfig, device: str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self.actor = Go2WActor(config.obs_dim, config.action_dim).to(self.device)
        self.default_joint_pos = np.asarray(
            config.default_joint_pos_policy_order, dtype=np.float32
        )
        self.action_default_joint_pos = np.asarray(
            config.action_default_joint_pos_policy_order, dtype=np.float32
        )
        self.command = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(config.action_dim, dtype=np.float32)

        self._load_policy()
        self.actor.eval()

    def _load_policy(self) -> None:
        checkpoint = torch.load(resolve_policy_path(self.config), map_location=self.device)
        if isinstance(checkpoint, dict) and "actor_state_dict" in checkpoint:
            state_dict = checkpoint["actor_state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise ValueError("Unsupported checkpoint format")

        # The training actor checkpoint also stores stochastic exploration
        # parameters, while deployment only evaluates the deterministic MLP.
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("distribution.")
        }
        self.actor.load_state_dict(state_dict, strict=True)

    def set_command(self, vx: float, vy: float, wz: float) -> np.ndarray:
        self.command = np.asarray(
            [
                np.clip(vx, -self.config.max_vx, self.config.max_vx),
                np.clip(vy, -self.config.max_vy, self.config.max_vy),
                np.clip(wz, -self.config.max_wz, self.config.max_wz),
            ],
            dtype=np.float32,
        )
        return self.command.copy()

    def build_observation(
        self,
        base_lin_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        projected_gravity: np.ndarray,
        joint_pos_policy_order: np.ndarray,
        joint_vel_policy_order: np.ndarray,
    ) -> np.ndarray:
        obs = np.zeros(self.config.obs_dim, dtype=np.float32)
        obs[0:3] = np.asarray(base_lin_vel, dtype=np.float32)
        obs[3:6] = np.asarray(base_ang_vel, dtype=np.float32)
        obs[6:9] = np.asarray(projected_gravity, dtype=np.float32)
        obs[9:12] = self.command

        joint_pos = np.asarray(joint_pos_policy_order, dtype=np.float32).copy()
        wheel_indices = np.asarray(self.config.wheel_joint_policy_indices, dtype=np.int64)

        wheel_pos_mode = getattr(self.config, "wheel_position_observation_mode", "wrap")
        if wheel_pos_mode == "zero":
            joint_pos[wheel_indices] = self.default_joint_pos[wheel_indices]
        elif wheel_pos_mode == "wrap":
            if self.config.wrap_wheel_position_observation:
                joint_pos[wheel_indices] = np.arctan2(
                    np.sin(joint_pos[wheel_indices]),
                    np.cos(joint_pos[wheel_indices]),
                )
        elif wheel_pos_mode == "raw":
            pass
        else:
            raise ValueError(f"Unsupported wheel_position_observation_mode: {wheel_pos_mode}")

        obs[12:28] = joint_pos - self.default_joint_pos
        obs[28:44] = np.asarray(joint_vel_policy_order, dtype=np.float32)
        obs[44:60] = self.last_action
        return obs

    def infer(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.from_numpy(obs.astype(np.float32)).to(self.device).unsqueeze(0)
            raw_action = self.actor(obs_t).squeeze(0).cpu().numpy().astype(np.float32)

        self.last_action = raw_action.copy()

        action = raw_action.copy()
        if self.config.wheel_raw_action_clip is not None:
            wheel_clip = float(self.config.wheel_raw_action_clip)
            action[0:4] = np.clip(action[0:4], -wheel_clip, wheel_clip)

        if self.config.raw_action_clip is not None:
            leg_clip = float(self.config.raw_action_clip)
            action[4:16] = np.clip(action[4:16], -leg_clip, leg_clip)

        return action.astype(np.float32)

    def split_action(self, raw_action: np.ndarray) -> dict[str, np.ndarray]:
        """Split raw LLC action in Isaac FastFlat action order.

        Isaac action order:
          raw[0:4]   wheel velocity: FL, FR, RL, RR
          raw[4:8]   hip position:   FL, FR, RL, RR
          raw[8:12]  thigh position: FL, FR, RL, RR
          raw[12:16] calf position:  FL, FR, RL, RR
        """
        raw = np.asarray(raw_action, dtype=np.float32)

        wheel_perm = np.asarray(self.config.wheel_action_permutation, dtype=np.int64)
        if wheel_perm.shape != (4,):
            raise ValueError(
                f"wheel_action_permutation must have length 4, got {wheel_perm.tolist()}"
            )

        wheel_raw = raw[0:4].copy()
        if self.config.wheel_action_bias != 0.0:
            wheel_raw = wheel_raw + float(self.config.wheel_action_bias)
        if self.config.wheel_raw_action_clip is not None:
            wheel_clip = float(self.config.wheel_raw_action_clip)
            wheel_raw = np.clip(wheel_raw, -wheel_clip, wheel_clip).astype(np.float32)

        wheel_vel = wheel_raw[wheel_perm] * self.config.wheel_action_scale

        leg_perm = np.asarray(self.config.leg_action_permutation, dtype=np.int64)
        if leg_perm.shape != (4,):
            raise ValueError(
                f"leg_action_permutation must have length 4, got {leg_perm.tolist()}"
            )

        hip_defaults = self.action_default_joint_pos[0:4]
        thigh_defaults = self.action_default_joint_pos[4:8]
        calf_defaults = self.action_default_joint_pos[8:12]

        hip_pos = hip_defaults + raw[4:8][leg_perm] * self.config.hip_action_scale
        thigh_pos = thigh_defaults + raw[8:12][leg_perm] * self.config.stance_action_scale
        calf_pos = calf_defaults + raw[12:16][leg_perm] * self.config.stance_action_scale

        stance_pos = np.concatenate([thigh_pos, calf_pos]).astype(np.float32)

        return {
            "raw": raw,
            "wheel_vel": wheel_vel.astype(np.float32),
            "hip_pos": hip_pos.astype(np.float32),
            "stance_pos": stance_pos,
        }

    def validate_targets(self, split: dict[str, np.ndarray]) -> list[str]:
        """Return target values that exceed modeled physical limits."""
        violations: list[str] = []
        wheel_vel = split["wheel_vel"]
        wheel_over = np.flatnonzero(np.abs(wheel_vel) > self.config.wheel_velocity_limit)
        if wheel_over.size:
            violations.append(
                "wheel velocity exceeds "
                f"+/-{self.config.wheel_velocity_limit:.1f} rad/s at indices "
                f"{wheel_over.tolist()}: {wheel_vel[wheel_over].round(3).tolist()}"
            )

        hip_pos = split["hip_pos"]
        hip_over = np.flatnonzero(np.abs(hip_pos) > self.config.hip_position_limit)
        if hip_over.size:
            violations.append(
                "hip position exceeds "
                f"+/-{self.config.hip_position_limit:.4f} rad at indices "
                f"{hip_over.tolist()}: {hip_pos[hip_over].round(3).tolist()}"
            )

        stance_pos = split["stance_pos"]
        stance_limits = np.asarray(self.config.stance_position_limits, dtype=np.float32)
        stance_over = np.flatnonzero(
            (stance_pos < stance_limits[:, 0]) | (stance_pos > stance_limits[:, 1])
        )
        if stance_over.size:
            violations.append(
                "stance position is outside URDF limits at indices "
                f"{stance_over.tolist()}: {stance_pos[stance_over].round(3).tolist()}"
            )
        return violations

    def clip_targets(
        self, split: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Saturate command targets to the physical constraints used in simulation."""
        clipped = {key: value.copy() for key, value in split.items()}
        reports: list[str] = []

        wheel_before = clipped["wheel_vel"].copy()
        clipped["wheel_vel"] = np.clip(
            wheel_before,
            -self.config.wheel_velocity_limit,
            self.config.wheel_velocity_limit,
        )
        wheel_changed = np.flatnonzero(clipped["wheel_vel"] != wheel_before)
        if wheel_changed.size:
            reports.append(f"wheel indices {wheel_changed.tolist()}")

        hip_before = clipped["hip_pos"].copy()
        clipped["hip_pos"] = np.clip(
            hip_before,
            -self.config.hip_position_limit,
            self.config.hip_position_limit,
        )
        hip_changed = np.flatnonzero(clipped["hip_pos"] != hip_before)
        if hip_changed.size:
            reports.append(f"hip indices {hip_changed.tolist()}")

        stance_before = clipped["stance_pos"].copy()
        stance_limits = np.asarray(self.config.stance_position_limits, dtype=np.float32)
        clipped["stance_pos"] = np.clip(
            stance_before, stance_limits[:, 0], stance_limits[:, 1]
        )
        stance_changed = np.flatnonzero(clipped["stance_pos"] != stance_before)
        if stance_changed.size:
            reports.append(
                f"stance indices {stance_changed.tolist()} "
                f"{stance_before[stance_changed].round(3).tolist()} -> "
                f"{clipped['stance_pos'][stance_changed].round(3).tolist()}"
            )
        return clipped, reports

    @staticmethod
    def validate_finite_targets(split: dict[str, np.ndarray]) -> list[str]:
        violations: list[str] = []
        for name in ("wheel_vel", "hip_pos", "stance_pos"):
            if not np.all(np.isfinite(split[name])):
                violations.append(f"{name} contains non-finite target values")
        return violations
