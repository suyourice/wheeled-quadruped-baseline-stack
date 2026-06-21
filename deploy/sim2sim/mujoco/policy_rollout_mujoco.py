from __future__ import annotations

from pathlib import Path

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from deploy.common.go2w_deploy_config import Go2WDeployConfig
from deploy.common.go2w_llc_policy_wrapper import Go2WLLCPolicyWrapper
from deploy.sim2sim.mujoco.generate_mujoco_model import main as generate_model


DEFAULT_JOINT_POS_MUJOCO_ORDER = np.array(
    [
        0.0, 0.4, -0.84, 0.0,  # FL
        0.0, 0.4, -0.84, 0.0,  # FR
        0.0, 0.4, -0.84, 0.0,  # RL
        0.0, 0.4, -0.84, 0.0,  # RR
    ],
    dtype=np.float64,
)

CALF_ACTION_DEFAULT_INDICES = np.array([8, 9, 10, 11], dtype=np.int64)


def mujoco_joint_state_to_policy_order(
    config: Go2WDeployConfig,
    joint_pos_mujoco_order: np.ndarray,
    joint_vel_mujoco_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mujoco_to_policy = np.asarray(config.mujoco_to_policy_joint_order, dtype=np.int64)
    return (
        joint_pos_mujoco_order[mujoco_to_policy],
        joint_vel_mujoco_order[mujoco_to_policy],
    )


def quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_to_pitch_wxyz(q: np.ndarray) -> float:
    w, x, y, z = q
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    return float(np.arcsin(sinp))


def quat_to_yaw_wxyz(q: np.ndarray) -> float:
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def compute_projected_gravity(qpos_quat_wxyz: np.ndarray) -> np.ndarray:
    rot_body_to_world = quat_to_rotmat_wxyz(qpos_quat_wxyz)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    gravity_body = rot_body_to_world.T @ gravity_world
    return gravity_body.astype(np.float32)


def world_vector_to_body(qpos_quat_wxyz: np.ndarray, vector_world: np.ndarray) -> np.ndarray:
    rot_body_to_world = quat_to_rotmat_wxyz(qpos_quat_wxyz)
    return rot_body_to_world.T @ np.asarray(vector_world, dtype=np.float64)


def set_initial_state(data: mujoco.MjData, base_height: float) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = np.array([0.0, 0.0, base_height], dtype=np.float64)
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qpos[7:23] = DEFAULT_JOINT_POS_MUJOCO_ORDER


def targets_to_mujoco_ctrl(
    split: dict[str, np.ndarray],
    wheel_sign: float,
) -> np.ndarray:
    wheel = split["wheel_vel"]
    hip = split["hip_pos"]
    stance = split["stance_pos"]

    ctrl = np.zeros(16, dtype=np.float64)

    # stance order from wrapper is grouped:
    # [FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf]
    ctrl[0:3] = [hip[0], stance[0], stance[4]]
    ctrl[3:6] = [hip[1], stance[1], stance[5]]
    ctrl[6:9] = [hip[2], stance[2], stance[6]]
    ctrl[9:12] = [hip[3], stance[3], stance[7]]

    ctrl[12:16] = wheel_sign * np.array([wheel[0], wheel[1], wheel[2], wheel[3]])
    return ctrl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--sim-time", type=float, default=5.0)
    parser.add_argument("--max-vx", type=float, default=None)
    parser.add_argument("--max-vy", type=float, default=None)
    parser.add_argument("--max-wz", type=float, default=None)
    parser.add_argument("--wheel-sign", type=float, default=1.0)
    parser.add_argument(
        "--disable-target-clip",
        action="store_true",
        help="Disable deploy-side joint target clipping for parity testing.",
    )
    parser.add_argument(
        "--print-target-violations",
        action="store_true",
        help="Print the first few joint target limit violations.",
    )
    parser.add_argument("--swap-lr-obs", action="store_true")
    parser.add_argument("--wheel-target-delta-limit", type=float, default=None)
    parser.add_argument("--debug-window-start", type=float, default=None)
    parser.add_argument("--debug-window-end", type=float, default=None)
    parser.add_argument("--debug-window-stride", type=int, default=10)
    parser.add_argument("--action-calf-default", type=float, default=None)
    parser.add_argument("--use-training-action-config", action="store_true")
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Launch the MuJoCo passive viewer during rollout.",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Throttle the rollout to approximately real time when the viewer is enabled.",
    )
    parser.add_argument(
        "--viewer-sync-stride",
        type=int,
        default=10,
        help="Synchronize the MuJoCo viewer every N simulation steps.",
    )
    args = parser.parse_args()

    generate_model()

    model_path = Path(__file__).resolve().parent / "generated" / "go2w_mujoco_actuated.xml"
    print(f"Loading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    set_initial_state(data, base_height=0.50)
    mujoco.mj_forward(model, data)

    config = Go2WDeployConfig()

    if args.max_vx is not None:
        config.max_vx = float(args.max_vx)
    if args.max_vy is not None:
        config.max_vy = float(args.max_vy)
    if args.max_wz is not None:
        config.max_wz = float(args.max_wz)

    if args.use_training_action_config:
        config.wheel_action_scale = 28.0
        config.hip_action_scale = 0.35
        config.stance_action_scale = 0.35
        config.action_default_joint_pos_policy_order = config.default_joint_pos_policy_order
        print(
            "Using FastFlat training action config: "
            "wheel_scale=28.0, hip_scale=0.35, stance_scale=0.35, "
            "action_default=default_joint_pos"
        )

    wrapper = Go2WLLCPolicyWrapper(config=config, device="cpu")
    if args.action_calf_default is not None:
        wrapper.action_default_joint_pos[CALF_ACTION_DEFAULT_INDICES] = float(
            args.action_calf_default
        )
        print(f"action_calf_default_override={float(args.action_calf_default):.4f}")

    command = wrapper.set_command(vx=args.vx, vy=args.vy, wz=args.wz)
    print(f"command={command.round(4).tolist()}")
    print(f"wheel_sign={args.wheel_sign:.1f}")

    sim_time = args.sim_time
    control_dt = 1.0 / config.control_hz
    sim_dt = model.opt.timestep
    steps_per_control = max(1, int(round(control_dt / sim_dt)))
    total_steps = int(round(sim_time / sim_dt))

    print(f"sim_dt={sim_dt:.6f}")
    print(f"control_dt={control_dt:.6f}")
    print(f"steps_per_control={steps_per_control}")
    print(f"total_steps={total_steps}")

    viewer = None
    wall_start_time = time.time()
    sim_start_time = float(data.time)

    if args.viewer:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.sync()
        print("MuJoCo passive viewer launched.")

    current_ctrl = np.zeros(16, dtype=np.float64)
    initial_base_x = float(data.qpos[0])
    initial_base_y = float(data.qpos[1])
    previous_xy = data.qpos[0:2].copy()
    trajectory_length_xy = 0.0
    max_abs_pitch = 0.0
    initial_yaw = quat_to_yaw_wxyz(data.qpos[3:7])
    previous_yaw = initial_yaw
    unwrapped_yaw = 0.0
    min_base_z = float(data.qpos[2])
    max_abs_wheel_target = 0.0
    max_abs_raw_action = 0.0
    max_leg_tracking_error = 0.0
    sum_leg_tracking_error = 0.0
    leg_tracking_samples = 0
    final_leg_actual = np.zeros(12, dtype=np.float64)
    final_leg_target = np.zeros(12, dtype=np.float64)
    violation_count = 0
    clip_count = 0

    try:
        for step in range(total_steps):
            if viewer is not None and not viewer.is_running():
                print("Viewer closed; stopping rollout.")
                break

            if step % steps_per_control == 0:
                base_quat = data.qpos[3:7].copy()
                base_lin_vel = world_vector_to_body(base_quat, data.qvel[0:3]).astype(np.float32)
                base_ang_vel = world_vector_to_body(base_quat, data.qvel[3:6]).astype(np.float32)
                projected_gravity = compute_projected_gravity(base_quat)

                joint_pos_mujoco_order = data.qpos[7:23].copy().astype(np.float32)
                joint_vel_mujoco_order = data.qvel[6:22].copy().astype(np.float32)
                joint_pos_policy_order, joint_vel_policy_order = mujoco_joint_state_to_policy_order(
                    config,
                    joint_pos_mujoco_order,
                    joint_vel_mujoco_order,
                )

                if args.swap_lr_obs:
                    lr_obs_perm = np.asarray(
                        [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11],
                        dtype=np.int64,
                    )
                    joint_pos_policy_order = joint_pos_policy_order[lr_obs_perm]
                    joint_vel_policy_order = joint_vel_policy_order[lr_obs_perm]

                obs = wrapper.build_observation(
                    base_lin_vel=base_lin_vel,
                    base_ang_vel=base_ang_vel,
                    projected_gravity=projected_gravity,
                    joint_pos_policy_order=joint_pos_policy_order,
                    joint_vel_policy_order=joint_vel_policy_order,
                )

                raw_action = wrapper.infer(obs)
                split = wrapper.split_action(raw_action)
                violations = wrapper.validate_targets(split)

                if args.disable_target_clip:
                    clipped = split
                    clip_reports = []
                else:
                    clipped, clip_reports = wrapper.clip_targets(split)

                if violations:
                    violation_count += 1
                    if args.print_target_violations and violation_count <= 5:
                        print(f"[target_violation {violation_count}]")
                        for msg in violations:
                            print(f"  - {msg}")

                if clip_reports:
                    clip_count += 1

                next_ctrl = targets_to_mujoco_ctrl(clipped, wheel_sign=args.wheel_sign)

                if args.wheel_target_delta_limit is not None:
                    delta_limit = float(args.wheel_target_delta_limit)
                    wheel_delta = next_ctrl[12:16] - current_ctrl[12:16]
                    wheel_delta = np.clip(wheel_delta, -delta_limit, delta_limit)
                    next_ctrl[12:16] = current_ctrl[12:16] + wheel_delta

                current_ctrl = next_ctrl

                if (
                    args.debug_window_start is not None
                    and args.debug_window_end is not None
                    and args.debug_window_start <= data.time <= args.debug_window_end
                    and (step // steps_per_control) % max(1, args.debug_window_stride) == 0
                ):
                    body_vx = float(base_lin_vel[0])
                    print(
                        "debug "
                        f"time={data.time:.3f} "
                        f"base_x={float(data.qpos[0]):.4f} "
                        f"body_vx={body_vx:.4f} "
                        f"raw_wheels={[round(float(x), 3) for x in raw_action[0:4]]} "
                        f"ctrl_wheels={[round(float(x), 3) for x in current_ctrl[12:16]]}"
                    )

                max_abs_raw_action = max(max_abs_raw_action, float(np.max(np.abs(raw_action))))
                max_abs_wheel_target = max(
                    max_abs_wheel_target,
                    float(np.max(np.abs(clipped["wheel_vel"]))),
                )

            data.ctrl[:] = current_ctrl
            mujoco.mj_step(model, data)

            base_x = float(data.qpos[0])
            base_y = float(data.qpos[1])
            base_z = float(data.qpos[2])
            current_xy = data.qpos[0:2].copy()
            trajectory_length_xy += float(np.linalg.norm(current_xy - previous_xy))
            previous_xy = current_xy
            pitch = quat_to_pitch_wxyz(data.qpos[3:7])
            min_base_z = min(min_base_z, base_z)

            current_yaw = quat_to_yaw_wxyz(data.qpos[3:7])
            yaw_delta = (current_yaw - previous_yaw + np.pi) % (2.0 * np.pi) - np.pi
            unwrapped_yaw += float(yaw_delta)
            previous_yaw = current_yaw

            max_abs_pitch = max(max_abs_pitch, abs(pitch))

            leg_qpos_indices = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14])
            final_leg_actual = data.qpos[7:23][leg_qpos_indices].copy()
            final_leg_target = current_ctrl[0:12].copy()
            leg_error = np.abs(final_leg_actual - final_leg_target)
            max_leg_tracking_error = max(max_leg_tracking_error, float(np.max(leg_error)))
            sum_leg_tracking_error += float(np.mean(leg_error))
            leg_tracking_samples += 1

            if viewer is not None and step % max(1, args.viewer_sync_stride) == 0:
                viewer.sync()

                if args.realtime:
                    sim_elapsed = float(data.time) - sim_start_time
                    wall_elapsed = time.time() - wall_start_time
                    sleep_time = sim_elapsed - wall_elapsed
                    if sleep_time > 0.0:
                        time.sleep(min(sleep_time, 0.05))

            if step % int(round(0.25 / sim_dt)) == 0:
                print(
                    f"step={step:05d} "
                    f"time={data.time:.3f} "
                    f"base_x={base_x:.4f} "
                    f"base_z={base_z:.4f} "
                    f"pitch={pitch:.4f} "
                    f"max_abs_pitch={max_abs_pitch:.4f} "
                    f"ctrl_wheels={current_ctrl[12:16].round(3).tolist()}"
                )

            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                print("failure: nonfinite state")
                break

            if base_z < 0.15 or abs(pitch) > 1.2:
                print("failure: likely fallen")
                break

    except KeyboardInterrupt:
        print("KeyboardInterrupt received; stopping rollout.")

    finally:
        if viewer is not None:
            try:
                viewer.close()
            except Exception as exc:
                print(f"Viewer close warning: {exc}")

    print("\nSummary")
    print(f"final_time={data.time:.4f}")
    final_base_x = float(data.qpos[0])
    final_base_y = float(data.qpos[1])
    world_dx = final_base_x - initial_base_x
    world_dy = final_base_y - initial_base_y
    mean_world_vx = world_dx / data.time if data.time > 0.0 else 0.0
    mean_world_vy = world_dy / data.time if data.time > 0.0 else 0.0
    print(f"final_base_x={final_base_x:.4f}")
    print(f"final_base_y={final_base_y:.4f}")
    print(f"world_dx={world_dx:.4f}")
    print(f"world_dy={world_dy:.4f}")
    print(f"mean_world_vx={mean_world_vx:.4f}")
    print(f"mean_world_vy={mean_world_vy:.4f}")
    print(f"trajectory_length_xy={trajectory_length_xy:.4f}")
    print(f"final_base_z={float(data.qpos[2]):.4f}")
    print(f"min_base_z={min_base_z:.4f}")
    print(f"final_pitch_rad={quat_to_pitch_wxyz(data.qpos[3:7]):.4f}")
    print(f"max_abs_pitch_rad={max_abs_pitch:.4f}")
    final_yaw = quat_to_yaw_wxyz(data.qpos[3:7])
    mean_yaw_rate = unwrapped_yaw / data.time if data.time > 0.0 else 0.0
    print(f"final_yaw_rad={final_yaw:.4f}")
    print(f"unwrapped_yaw_rad={unwrapped_yaw:.4f}")
    print(f"mean_yaw_rate={mean_yaw_rate:.4f}")
    mean_leg_tracking_error = (
        sum_leg_tracking_error / leg_tracking_samples
        if leg_tracking_samples > 0
        else 0.0
    )
    print(f"max_abs_raw_action={max_abs_raw_action:.4f}")
    print(f"max_abs_wheel_target={max_abs_wheel_target:.4f}")
    print(f"max_leg_tracking_error={max_leg_tracking_error:.4f}")
    print(f"mean_leg_tracking_error={mean_leg_tracking_error:.4f}")
    print(f"final_leg_actual={final_leg_actual.round(4).tolist()}")
    print(f"final_leg_target={final_leg_target.round(4).tolist()}")
    print(f"violation_count={violation_count}")
    print(f"clip_count={clip_count}")


if __name__ == "__main__":
    main()
