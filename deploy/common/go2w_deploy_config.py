from dataclasses import dataclass
from pathlib import Path


@dataclass
class Go2WDeployConfig:
    policy_path: str = "deploy/checkpoints/fast_flat.pt"

    obs_dim: int = 60
    action_dim: int = 16
    command_dim: int = 3
    control_hz: float = 50.0
    command_source: str = "wireless"
    wireless_controller_topic: str = "/wirelesscontroller"
    wireless_command_timeout_s: float = 0.5
    joystick_deadband: float = 0.05
    base_velocity_source: str = "wheel"
    base_velocity_topic: str = "/lf/sportmodestate"
    base_velocity_timeout_s: float = 0.3
    wheel_radius_m: float = 0.086
    wrap_wheel_position_observation: bool = True
    wheel_position_observation_mode: str = "wrap"

    # MuJoCo qpos/qvel order is per-leg:
    # [FL hip/thigh/calf/wheel, FR ..., RL ..., RR ...].
    # Isaac policy order is grouped:
    # [hips FL/FR/RL/RR, thighs FL/FR/RL/RR, calves FL/FR/RL/RR, wheels FL/FR/RL/RR].
    mujoco_to_policy_joint_order: tuple[int, ...] = (
        0, 4, 8, 12,
        1, 5, 9, 13,
        2, 6, 10, 14,
        3, 7, 11, 15,
    )

    max_vx: float = 0.50
    max_vy: float = 0.00
    max_wz: float = 0.20

    wheel_action_scale: float = 28.0
    hip_action_scale: float = 0.35
    stance_action_scale: float = 0.35
    raw_action_clip: float | None = None
    wheel_raw_action_clip: float | None = None
    wheel_action_bias: float = 0.0
    wheel_action_permutation: tuple[int, ...] = (0, 1, 2, 3)
    wheel_observation_permutation: tuple[int, ...] = (0, 1, 2, 3)
    leg_action_permutation: tuple[int, ...] = (0, 1, 2, 3)
    use_grouped_stance_actions: bool = True
    wheel_velocity_limit: float = 30.1
    hip_position_limit: float = 1.0472
    stance_position_limits: tuple[tuple[float, float], ...] = (
        (-1.5708, 3.4907),  # FL thigh
        (-1.5708, 3.4907),  # FR thigh
        (-0.5236, 4.5379),  # RL thigh
        (-0.5236, 4.5379),  # RR thigh
        (-2.7227, -0.83776),  # FL calf
        (-2.7227, -0.83776),  # FR calf
        (-2.7227, -0.83776),  # RL calf
        (-2.7227, -0.83776),  # RR calf
    )

    default_joint_pos_policy_order: tuple[float, ...] = (
        0.0, 0.0, 0.0, 0.0,       # hips: FL, FR, RL, RR
        0.4, 0.4, 0.4, 0.4,       # thighs: FL, FR, RL, RR
        -0.84, -0.84, -0.84, -0.84,  # calves: FL, FR, RL, RR
        0.0, 0.0, 0.0, 0.0,       # wheels: FL, FR, RL, RR
    )
    action_default_joint_pos_policy_order: tuple[float, ...] = (
        0.0, 0.0, 0.0, 0.0,       # hips: FL, FR, RL, RR
        0.4, 0.4, 0.4, 0.4,       # thighs: FL, FR, RL, RR
        -0.84, -0.84, -0.84, -0.84,  # calves: FL, FR, RL, RR
        0.0, 0.0, 0.0, 0.0,       # wheels: FL, FR, RL, RR
    )

    # Unitree motor order observed on the real Go2-W:
    #   FR: hip=0, thigh=1, calf=2, wheel=12
    #   FL: hip=3, thigh=4, calf=5, wheel=13
    #   RR: hip=6, thigh=7, calf=8, wheel=14
    #   RL: hip=9, thigh=10, calf=11, wheel=15
    #
    # Policy joint order follows the Isaac fast-flat observation order:
    #   hips   = [FL, FR, RL, RR]
    #   thighs = [FL, FR, RL, RR]
    #   calves = [FL, FR, RL, RR]
    #   wheels = [FL, FR, RL, RR]
    unitree_to_policy_indices: tuple[int, ...] = (
        3, 0, 9, 6,
        4, 1, 10, 7,
        5, 2, 11, 8,
        13, 12, 15, 14,
    )
    wheel_joint_policy_indices: tuple[int, ...] = (12, 13, 14, 15)

    leg_kp: float = 20.0
    leg_kd: float = 1.0
    wheel_kp: float = 0.0
    wheel_kd: float = 0.5
    motor_mode: int = 1


def resolve_policy_path(config: Go2WDeployConfig) -> Path:
    path = Path(config.policy_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Policy file does not exist: {path}")
    return path
