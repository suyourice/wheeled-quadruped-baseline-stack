# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation debug / visualization helpers for play.py.

This module is imported at runtime (after AppLauncher.launch()), inside play.py's
deferred-import block, so all Isaac/IsaacLab symbols are available at import time.
"""

import argparse
import math

import torch
from isaaclab.utils.math import quat_from_angle_axis

from go2w.tasks.manager_based.go2w.mdp.nav_scenarios import NAV_SCENARIO_NAMES as _NAV_SCENARIO_ID_TO_NAME
from go2w.tasks.manager_based.go2w.mdp.obstacle_geometry import footprint_clearance
from go2w.tasks.manager_based.go2w.observation_layout import POLICY_OBS


NAV_LIVE_LABEL_INTERVAL = 3
NAV_LIVE_LABEL_SCALE = 0.040
NAV_LIVE_LABEL_MAX = 40
NAV_LIVE_LABEL_MIN_Z = 3.10


def _format_vector(values: torch.Tensor, precision: int = 3) -> str:
    return ", ".join(f"{value.item():+.{precision}f}" for value in values)


def _format_eval_metrics(metrics: dict[str, float], completed_episodes: int, avg_episode_length: float) -> str:
    preferred_keys = [
        "goal_reached_rate",
        "spl",
        "time_out_rate",
        "base_contact_rate",
        "root_height_below_minimum_rate",
        "multi_term_fraction",
    ]
    parts = [f"episodes={completed_episodes}", f"avg_episode_len={avg_episode_length:.2f}"]
    for key in preferred_keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    return " ".join(parts)


_NAV_OBSTACLE_LABEL_SHORT_NAMES: dict[str, str] = {
    "patient_ambulatory": "PT",
    "patient_with_iv": "PT_IV",
    "wheelchair_patient": "WC",
    "gurney_patient": "BED",
    "reception_desk": "DESK",
    "queue_patient": "Q_PT",
    "queue_visitor": "Q_VIS",
    "seated_patient": "S_PT",
    "seated_visitor": "S_VIS",
    "doorway_patient": "D_PT",
    "doorway_staff": "D_STAFF",
    "cleaning_machine": "CLEAN",
    "fallen_object": "FALL",
    "elderly": "ELDER",
    "adult": "ADULT",
    "child": "CHILD",
    "staff": "STAFF",
    "visitor": "VIS",
    "cart": "CART",
    "wheelchair": "WC_S",
    "gurney": "BED",
    "iv_pole": "IV",
    "dog": "DOG",
    "chair": "CHAIR",
    "bench": "BENCH",
    "trash_bin": "TRASH",
    "table": "TABLE",
}


def _short_obstacle_label(label: str) -> str:
    """Return a compact viewport/debug-map label."""
    return _NAV_OBSTACLE_LABEL_SHORT_NAMES.get(label, label).upper()


_BITMAP_FONT_5X7: dict[str, tuple[str, ...]] = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "01010", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "01010", "00100", "00100", "00100", "01010", "10001"),
    "Y": ("10001", "01010", "00100", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
}


class _LiveObstacleLabelDrawer:
    """Draw viewport-only bitmap labels over obstacle tops using Isaac debug draw."""

    def __init__(self, scale: float, max_labels: int):
        self.scale = max(scale, 0.018)
        self.max_labels = max(0, max_labels)
        self.enabled = False
        self._draw = None
        try:
            from isaacsim.util.debug_draw import _debug_draw

            self._draw = _debug_draw.acquire_debug_draw_interface()
            self.enabled = True
        except Exception as exc:
            print(f"[WARN] Live obstacle labels disabled: debug draw unavailable ({exc}).")

    def clear(self) -> None:
        if self._draw is None:
            return
        try:
            self._draw.clear_points()
        except Exception:
            pass

    def update(self, base_env, env_index: int) -> None:
        if not self.enabled or self._draw is None or self.max_labels <= 0:
            return
        records = _get_nav_obstacle_records(base_env, env_index, include_walls=False)[: self.max_labels]
        shadow_points: list[tuple[float, float, float]] = []
        obstacle_points: list[tuple[float, float, float]] = []
        zone_points: list[tuple[float, float, float]] = []
        situation_points: list[tuple[float, float, float]] = []

        # XY positions of already-placed labels — used to stagger z when labels are close.
        _placed_xy: list[tuple[float, float]] = []
        _stagger_step = self.scale * 9.5   # ~one font row + gap per stagger level
        _min_sep = self.scale * 22.0       # ~4-char label width + margin

        def _stagger_z(x: float, y: float, base_z: float) -> float:
            count = sum(1 for px, py in _placed_xy if math.hypot(x - px, y - py) < _min_sep)
            return base_z + count * _stagger_step

        def add_label(
            label: str,
            x: float,
            y: float,
            center_z: float,
            scale: float,
            target: list[tuple[float, float, float]],
        ) -> None:
            base_z = max(NAV_LIVE_LABEL_MIN_Z, center_z + 0.28, 2.0 * center_z + 0.03)
            z = _stagger_z(x, y, base_z)
            _placed_xy.append((x, y))
            points = self._label_points(label, x, y, center_z, scale, z_override=z)
            shadow_points.extend((px + scale * 0.16, py - scale * 0.16, pz - 0.004) for px, py, pz in points)
            target.extend(points)

        for rec in records:
            add_label(rec["short_label"], rec["x"], rec["y"], rec["z"], self.scale, obstacle_points)
        for ann in _get_hospital_live_annotations(base_env, env_index):
            target = zone_points if ann["kind"] == "zone" else situation_points
            add_label(
                ann["label"],
                ann["x"],
                ann["y"],
                ann.get("z", 0.95),
                self.scale * ann.get("scale", 1.0),
                target,
            )
        try:
            self._draw.clear_points()
            if shadow_points:
                self._draw.draw_points(shadow_points, [(0.0, 0.0, 0.0, 1.0)] * len(shadow_points), [6] * len(shadow_points))
            if obstacle_points:
                self._draw.draw_points(obstacle_points, [(1.0, 1.0, 1.0, 1.0)] * len(obstacle_points), [4] * len(obstacle_points))
            if zone_points:
                self._draw.draw_points(zone_points, [(0.20, 0.92, 1.0, 1.0)] * len(zone_points), [4] * len(zone_points))
            if situation_points:
                self._draw.draw_points(situation_points, [(1.0, 0.82, 0.18, 1.0)] * len(situation_points), [4] * len(situation_points))
        except Exception as exc:
            self.enabled = False
            print(f"[WARN] Live obstacle labels disabled after draw failure: {exc}")

    def _label_points(self, label: str, x: float, y: float, center_z: float, scale: float, z_override: float | None = None) -> list[tuple[float, float, float]]:
        chars = [ch if ch in _BITMAP_FONT_5X7 else "_" for ch in label.upper()]
        if not chars:
            return []
        char_stride = 6
        total_cols = len(chars) * char_stride - 1
        z = z_override if z_override is not None else max(NAV_LIVE_LABEL_MIN_Z, center_z + 0.28, 2.0 * center_z + 0.03)
        points: list[tuple[float, float, float]] = []
        for char_idx, ch in enumerate(chars):
            bitmap = _BITMAP_FONT_5X7.get(ch)
            if bitmap is None:
                continue
            col_base = char_idx * char_stride
            for row_idx, row in enumerate(bitmap):
                for col_idx, bit in enumerate(row):
                    if bit != "1":
                        continue
                    px = x + (col_base + col_idx - total_cols * 0.5) * scale
                    py = y + (3.0 - row_idx) * scale
                    points.append((px, py, z))
        return points


def _obstacle_label_map(base_env, obstacle_names: list[str]) -> dict[str, str]:
    """Resolve obstacle_name -> semantic label from reset or hospital motion config."""
    label_by_name: dict[str, str] = {}
    try:
        reset_params = base_env.cfg.events.reset_obstacles.params
        reset_labels = list(reset_params.get("obstacle_labels", []))
        if len(reset_labels) == len(obstacle_names):
            label_by_name.update(dict(zip(obstacle_names, reset_labels)))
    except Exception:
        pass

    try:
        hospital_motion = getattr(base_env.cfg.events, "hospital_dynamic_motion", None)
        motion_params = getattr(hospital_motion, "params", None) if hospital_motion is not None else None
        if motion_params is not None:
            motion_names = list(motion_params.get("obstacle_names", []))
            motion_labels = list(motion_params.get("obstacle_labels", []))
            if len(motion_names) == len(motion_labels):
                label_by_name.update(dict(zip(motion_names, motion_labels)))
    except Exception:
        pass
    return label_by_name


def _nav_debug_corridor_plot_args(base_env, args_cli: argparse.Namespace) -> list[str]:
    """Build corridor overlay args for plot_nav_debug from CLI or env cfg."""
    corridor_kind = None
    leg_length = args_cli.corridor_leg_length
    corridor_width = args_cli.corridor_width
    clearance_cost_weight = args_cli.astar_clearance_cost_weight
    clearance_cost_sigma = args_cli.astar_clearance_cost_sigma
    corner_rounding = args_cli.corner_rounding
    corner_radius = args_cli.corner_radius

    if args_cli.structured_env != "none":
        corridor_kind = args_cli.structured_env
    else:
        try:
            reset_params = base_env.cfg.events.reset_obstacles.params
            corridor_kind = reset_params.get("corridor_kind", None)
            leg_length = reset_params.get("leg_length", leg_length)
            corridor_width = reset_params.get("corridor_width", corridor_width)
            clearance_cost_weight = reset_params.get("clearance_cost_weight", clearance_cost_weight)
            clearance_cost_sigma = reset_params.get("clearance_cost_sigma", clearance_cost_sigma)
            corner_rounding = reset_params.get("corner_rounding", corner_rounding)
            corner_radius = reset_params.get("corner_radius", corner_radius)
        except Exception:
            corridor_kind = None

    if corridor_kind is None or str(corridor_kind).lower() == "none":
        return []
    args = [
        "--corridor", str(corridor_kind),
        "--leg_length", str(leg_length),
        "--corridor_width", str(corridor_width),
        "--alpha", str(clearance_cost_weight),
        "--sigma", str(clearance_cost_sigma),
    ]
    if corner_rounding:
        args.extend(["--corner_rounding", "--corner_radius", str(corner_radius)])
    return args


def _get_nav_obstacle_records(base_env, env_index: int, include_walls: bool = False) -> list[dict]:
    """Collect active obstacle positions and semantic labels for one env."""
    if base_env is None:
        return []
    try:
        reset_params = base_env.cfg.events.reset_obstacles.params
        obstacle_names = list(reset_params.get("obstacle_names", []))
    except Exception:
        return []
    if not obstacle_names:
        return []

    ei = max(0, min(env_index, base_env.num_envs - 1))
    label_by_name = _obstacle_label_map(base_env, obstacle_names)
    active_mask = getattr(base_env, "_go2w_obstacle_active_mask", None)
    widths = getattr(base_env, "_go2w_obstacle_width", None)
    depths = getattr(base_env, "_go2w_obstacle_depth", None)
    shape_ids = getattr(base_env, "_go2w_obstacle_shape_id", None)

    records: list[dict] = []
    for idx, name in enumerate(obstacle_names):
        label = label_by_name.get(name, "obstacle")
        if label == "wall" and not include_walls:
            continue
        try:
            pos = base_env.scene[name].data.root_pos_w[ei]
        except Exception:
            continue

        x, y, z = pos[0].item(), pos[1].item(), pos[2].item()
        if abs(x) > 500.0 or abs(y) > 500.0:
            continue
        if active_mask is not None and active_mask.shape[1] > idx and not bool(active_mask[ei, idx].item()):
            continue

        width = float(widths[ei, idx].item()) if widths is not None and widths.shape[1] > idx else 0.0
        depth = float(depths[ei, idx].item()) if depths is not None and depths.shape[1] > idx else 0.0
        shape_id = int(shape_ids[ei, idx].item()) if shape_ids is not None and shape_ids.shape[1] > idx else -1
        records.append({
            "name": name,
            "label": label,
            "short_label": _short_obstacle_label(label),
            "x": x,
            "y": y,
            "z": z,
            "width": width,
            "depth": depth,
            "shape_id": shape_id,
        })
    return records


def _get_hospital_live_annotations(base_env, env_index: int) -> list[dict]:
    """Return viewport-only semantic floor/situation labels for the hospital floor."""
    if base_env is None:
        return []
    try:
        reset_params = base_env.cfg.events.reset_obstacles.params
        corridor_kind = str(reset_params.get("corridor_kind", "")).lower()
    except Exception:
        return []
    if corridor_kind != "hospital_floor":
        return []
    required = (
        "_go2w_structured_corridor_start_xy",
        "_go2w_structured_corridor_yaw",
        "_go2w_structured_corridor_leg_length",
    )
    if any(not hasattr(base_env, attr) for attr in required):
        return []

    ei = max(0, min(env_index, base_env.num_envs - 1))
    try:
        origin = base_env._go2w_structured_corridor_start_xy[ei]
        yaw = float(base_env._go2w_structured_corridor_yaw[ei].item())
        leg_length = float(base_env._go2w_structured_corridor_leg_length[ei].item())
    except Exception:
        return []
    if leg_length <= 0.0:
        return []

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    def local(label: str, kind: str, x: float, y: float, scale: float = 0.90) -> dict:
        world_x = float(origin[0].item()) + x * cos_yaw - y * sin_yaw
        world_y = float(origin[1].item()) + x * sin_yaw + y * cos_yaw
        return {"label": label, "kind": kind, "x": world_x, "y": world_y, "z": 0.95, "scale": scale}

    L = leg_length
    zones = (
        local("RECEPTION", "zone", 0.72 * L, -0.55 * L, 0.95),
        local("WAITING", "zone", 1.05 * L, -0.20 * L, 0.95),
        local("MAIN", "zone", 1.40 * L, 0.00, 0.95),
        local("SERVICE", "zone", 2.05 * L, -0.50 * L, 0.95),
        local("WARD", "zone", 2.00 * L, 0.52 * L, 0.95),
        local("ROOMS", "zone", 2.75 * L, 1.24 * L, 0.95),
        local("IMAGING", "zone", 2.60 * L, 0.72 * L, 0.95),
        local("RAMP", "zone", 3.20 * L, L, 0.95),
        local("PHARM", "zone", 3.84 * L, 0.72 * L, 0.95),
        local("ELEV", "zone", 3.95 * L, 1.22 * L, 0.95),
    )
    situations = (
        local("QUEUE", "situation", 0.82 * L, -0.24 * L, 0.80),
        local("SEATED", "situation", 1.02 * L, -0.08 * L, 0.80),
        local("LEASH", "situation", 1.34 * L, 0.95, 0.80),
        local("WC_PASS", "situation", 1.58 * L, -0.95, 0.80),
        local("IV_WALK", "situation", 1.84 * L, 1.24, 0.80),
        local("CART_PUSH", "situation", 2.08 * L, -0.33 * L, 0.80),
        local("BED_PUSH", "situation", 2.58 * L, L - 0.06, 0.80),
        local("DOOR_XING", "situation", 2.78 * L, 1.36 * L, 0.80),
        local("CLEAN", "situation", 3.28 * L, 0.70 * L, 0.80),
    )
    return list(zones + situations)


def _print_nav_obstacle_label_log(base_env, step_count: int, env_index: int) -> None:
    """Emit a parseable labeled obstacle snapshot for top-down plots."""
    records = _get_nav_obstacle_records(base_env, env_index, include_walls=False)
    parts = [
        (
            f"{rec['name']}:{rec['short_label']}:{rec['x']:+.3f}:{rec['y']:+.3f}:"
            f"{rec['width']:.3f}:{rec['depth']:.3f}"
        )
        for rec in records
    ]
    print(
        "[GO2W_NAV_OBSTACLES] "
        f"step={step_count} env={env_index} active_count={len(records)} "
        f"obstacles={';'.join(parts)}"
    )


def _get_nav_env_info(base_env, env_index: int, last_hlc_cmd: torch.Tensor | None) -> dict:
    """Collect nav task state for the watched env. Returns empty dict if not a nav task."""
    info: dict = {}
    if not hasattr(base_env, "_go2w_goal_pos_w"):
        return info

    ei = max(0, min(env_index, base_env.num_envs - 1))

    scenario_id = int(base_env._go2w_scenario_template_id[ei].item()) if hasattr(base_env, "_go2w_scenario_template_id") else -1
    info["scenario"] = _NAV_SCENARIO_ID_TO_NAME.get(scenario_id, f"id{scenario_id}")

    goal = base_env._go2w_goal_pos_w[ei]
    info["goal"] = (goal[0].item(), goal[1].item(), goal[2].item())

    goals_reached = float(base_env._go2w_goals_reached_episode[ei].item()) if hasattr(base_env, "_go2w_goals_reached_episode") else 0.0
    info["goals_reached"] = goals_reached

    # HLC command (last policy output, 3D: vx vy yaw)
    if last_hlc_cmd is not None and last_hlc_cmd.ndim == 2 and last_hlc_cmd.shape[0] > ei:
        cmd = last_hlc_cmd[ei]
        info["hlc_cmd"] = (cmd[0].item(), cmd[1].item(), cmd[2].item())

    obstacle_records = _get_nav_obstacle_records(base_env, ei, include_walls=False)
    active_obs = [(rec["x"], rec["y"]) for rec in obstacle_records]
    info["obstacles"] = active_obs
    info["obstacle_records"] = obstacle_records

    return info


def _print_navigation_play_log(obs, dones: torch.Tensor, step_count: int, env_index: int,
                                base_env=None, last_hlc_cmd=None) -> None:
    """Print navigation task diagnostics from the policy obs group."""
    # Works for both PPO (policy group) and distillation (student/teacher groups).
    group_key = "policy" if isinstance(obs, dict) and "policy" in obs else None
    if group_key is None and isinstance(obs, dict) and "student" in obs:
        group_key = "student"
    if group_key is None and isinstance(obs, dict) and "student_state" in obs:
        group_key = "student_state"
    if group_key is None:
        return
    data = obs[group_key]
    if data.ndim != 2 or data.shape[0] == 0:
        return
    env_index = max(0, min(env_index, data.shape[0] - 1))
    row = data[env_index]

    if row.numel() in (15, 189):
        base_lin_vel = row[0:3]
        projected_gravity = row[3:6]
        goal_command = row[6:9]
        state_text = f"projected_gravity=[{_format_vector(projected_gravity)}]"
    else:
        base_lin_vel = row[POLICY_OBS["base_lin_vel"].as_slice()]
        base_ang_vel = row[POLICY_OBS["base_ang_vel"].as_slice()]
        goal_command = row[POLICY_OBS["goal_command"].as_slice()]
        state_text = f"base_ang_vel=[{_format_vector(base_ang_vel)}]"
    done = int(dones[env_index].item()) if dones.numel() > env_index else 0

    nav = _get_nav_env_info(base_env, env_index, last_hlc_cmd) if base_env is not None else {}
    scenario_str = f" scenario={nav['scenario']}" if "scenario" in nav else ""
    goals_str = f" goals={nav['goals_reached']:.0f}" if "goals_reached" in nav else ""
    hlc_str = ""
    if "hlc_cmd" in nav:
        vx, vy, yaw = nav["hlc_cmd"]
        hlc_str = f" hlc=[{vx:+.2f},{vy:+.2f},{yaw:+.2f}]"

    print(
        "[nav-play] "
        f"step={step_count} env={env_index} done={done}"
        f"{scenario_str}{goals_str}{hlc_str} "
        f"goal_cmd=[{_format_vector(goal_command)}] "
        f"base_lin_vel=[{_format_vector(base_lin_vel)}] "
        f"{state_text}"
    )


def _print_nav_contact_debug(
    base_env, step_count: int, env_index: int, last_hlc_cmd: torch.Tensor | None
) -> None:
    """Print geometry and raw contact diagnostics for one navigation environment."""
    if base_env is None or not hasattr(base_env, "_go2w_scenario_template_id"):
        return
    reset_params = base_env.cfg.events.reset_obstacles.params
    obstacle_names = list(reset_params.get("obstacle_names", []))
    if not obstacle_names:
        return

    ei = max(0, min(env_index, base_env.num_envs - 1))
    robot_xy_all = base_env.scene["robot"].data.root_pos_w[:, :2]
    positions_all = torch.stack([base_env.scene[name].data.root_pos_w[:, :2] for name in obstacle_names], dim=1)
    center_distances_all = (positions_all - robot_xy_all.unsqueeze(1)).norm(dim=-1)
    positions = positions_all[ei]
    center_distances = center_distances_all[ei]
    active = base_env._go2w_obstacle_active_mask[ei]
    robot_safety_radius = float(base_env.cfg.rewards.nav_clearance.params.get("robot_safety_radius", 0.0))
    clearances = footprint_clearance(
        base_env,
        obstacle_names,
        center_distances_all,
        robot_safety_radius,
    )[ei]
    masked_center = torch.where(active, center_distances, torch.full_like(center_distances, 8.0))
    masked_clearance = torch.where(active, clearances, torch.full_like(clearances, 8.0))
    inactive_min_center = torch.where(
        ~active, center_distances, torch.full_like(center_distances, float("inf"))
    ).min()

    contact_sensor = base_env.scene.sensors["obstacle_contacts"]
    contact_force_max = float(
        contact_sensor.data.net_forces_w_history[ei].norm(dim=-1).max().item()
    )
    threshold = float(base_env.cfg.rewards.obstacle_collision.params.get("threshold", 1.0))
    scenario_id = int(base_env._go2w_scenario_template_id[ei].item())
    scenario = _NAV_SCENARIO_ID_TO_NAME.get(scenario_id, f"id{scenario_id}")
    start_pos = base_env._go2w_start_pos_w[ei]
    start_yaw = base_env._go2w_start_heading_w[ei]
    goal_pos = base_env._go2w_goal_pos_w[ei]
    robot_xy = robot_xy_all[ei]
    raw_action = (
        last_hlc_cmd[ei] if last_hlc_cmd is not None else torch.zeros(3, device=base_env.device)
    )
    executed_action = base_env.action_manager.get_term("llc_cmd").processed_actions[ei]
    positions_text = (
        ", ".join(
            f"{name}:{'A' if bool(active[idx].item()) else 'I'}"
            f"({positions[idx, 0].item():+.2f},{positions[idx, 1].item():+.2f})"
            f"/shape={int(base_env._go2w_obstacle_shape_id[ei, idx].item())}"
            f"/size=({base_env._go2w_obstacle_width[ei, idx].item():.2f},"
            f"{base_env._go2w_obstacle_depth[ei, idx].item():.2f})"
            f"/radius={base_env._go2w_obstacle_effective_radius[ei, idx].item():.3f}"
            for idx, name in enumerate(obstacle_names)
            if bool(active[idx].item())
        )
        if step_count == 0
        else "see step=0"
    )
    print(
        "[nav-contact-debug] "
        f"step={step_count} scenario={scenario} active_count={int(active.sum().item())} "
        f"start=({start_pos[0].item():+.3f},{start_pos[1].item():+.3f},{start_yaw.item():+.3f}) "
        f"robot=({robot_xy[0].item():+.3f},{robot_xy[1].item():+.3f}) "
        f"goal=({goal_pos[0].item():+.3f},{goal_pos[1].item():+.3f}) "
        f"raw_action=({raw_action[0].item():+.3f},{raw_action[1].item():+.3f},{raw_action[2].item():+.3f}) "
        f"executed_action=({executed_action[0].item():+.3f},{executed_action[1].item():+.3f},{executed_action[2].item():+.3f}) "
        f"min_center={masked_center.min().item():.4f} "
        f"min_footprint_clearance={masked_clearance.min().item():.4f} "
        f"inactive_min_center={inactive_min_center.item():.4f} "
        f"contact_force_max={contact_force_max:.4f} collision={int(contact_force_max > threshold)} "
        f"active_positions=[{positions_text}]"
    )


def _print_nav_episode_log(
    base_env,
    done_ids: torch.Tensor,
    env_index: int,
    last_hlc_cmd: torch.Tensor | None,
    episode_collision_counts: dict[int, int],
) -> None:
    """Print a summary when an episode ends for the watched env."""
    if base_env is None or not hasattr(base_env, "_go2w_goal_pos_w"):
        return
    ei = max(0, min(env_index, base_env.num_envs - 1))
    if ei not in done_ids.tolist():
        return

    nav = _get_nav_env_info(base_env, ei, last_hlc_cmd)
    if not nav:
        return

    goal_x, goal_y, goal_z = nav.get("goal", (0, 0, 0))
    hlc_str = ""
    if "hlc_cmd" in nav:
        vx, vy, yaw = nav["hlc_cmd"]
        hlc_str = f"  hlc_cmd: vx={vx:+.3f} vy={vy:+.3f} yaw={yaw:+.3f}\n"

    obs_lines = ""
    obstacle_records = nav.get("obstacle_records", [])
    if obstacle_records:
        obs_parts = [
            f"    {rec['short_label']:<12} [{rec['x']:+.2f}, {rec['y']:+.2f}]"
            for rec in obstacle_records[:48]
        ]
        if len(obstacle_records) > len(obs_parts):
            obs_parts.append(f"    ... {len(obstacle_records) - len(obs_parts)} more")
        obs_lines = "  obstacles (" + str(len(obstacle_records)) + " active):\n" + "\n".join(obs_parts) + "\n"
    else:
        obs_lines = "  obstacles: none active\n"

    collisions = episode_collision_counts.get(ei, 0)

    print(
        f"[nav-episode] env={ei}\n"
        f"  scenario:      {nav.get('scenario', '?')}\n"
        f"  goal_world:    [{goal_x:+.3f}, {goal_y:+.3f}, {goal_z:+.3f}]\n"
        f"  goals_reached: {nav.get('goals_reached', 0):.0f}\n"
        f"  collisions:    {collisions}\n"
        f"{hlc_str}"
        f"{obs_lines}"
        ,
        end="",
    )


def _get_nav_path_line_markers(
    base_env,
    env_index: int,
    z_offset: float = 0.16,
    max_segments: int = 160,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return cylinder marker transforms for the watched env's remaining navigation path."""
    if (
        base_env is None
        or not hasattr(base_env, "_go2w_navigation_path_w")
        or not hasattr(base_env, "_go2w_navigation_path_count")
        or not hasattr(base_env, "_go2w_goal_pos_w")
    ):
        return None

    ei = max(0, min(env_index, base_env.num_envs - 1))
    path_count = int(base_env._go2w_navigation_path_count[ei].item())
    if path_count < 2:
        return None

    final_idx = path_count - 1
    if hasattr(base_env, "_go2w_navigation_path_target_index"):
        target_idx = int(base_env._go2w_navigation_path_target_index[ei].item())
    elif hasattr(base_env, "_go2w_navigation_path_nearest_index"):
        target_idx = int(base_env._go2w_navigation_path_nearest_index[ei].item())
    else:
        target_idx = 0
    target_idx = max(0, min(target_idx, final_idx))

    robot_pos = base_env.scene["robot"].data.root_pos_w[ei, :3].clone()
    current_goal = base_env._go2w_goal_pos_w[ei, :3].clone()
    tail = base_env._go2w_navigation_path_w[ei, target_idx + 1 : final_idx + 1, :3]
    points = torch.cat((robot_pos.view(1, 3), current_goal.view(1, 3), tail), dim=0)
    points[:, 2] = robot_pos[2] + z_offset

    if points.shape[0] > max_segments + 1:
        sample_idx = torch.linspace(
            0,
            points.shape[0] - 1,
            max_segments + 1,
            device=points.device,
        ).round().long().unique()
        points = points[sample_idx]

    start = points[:-1]
    end = points[1:]
    direction = end - start
    lengths = direction.norm(dim=-1)
    valid = lengths > 0.03
    if not valid.any():
        return None

    start = start[valid]
    end = end[valid]
    direction = direction[valid]
    lengths = lengths[valid]
    positions = (start + end) * 0.5

    direction_norm = direction / lengths.unsqueeze(-1).clamp(min=1.0e-6)
    default_axis = torch.zeros_like(direction_norm)
    default_axis[:, 2] = 1.0
    rotation_axis = torch.linalg.cross(default_axis, direction_norm, dim=-1)
    rotation_axis_norm = rotation_axis.norm(dim=-1)
    fallback_axis = torch.zeros_like(rotation_axis)
    fallback_axis[:, 0] = 1.0
    rotation_axis = torch.where(
        (rotation_axis_norm > 1.0e-6).unsqueeze(-1),
        rotation_axis / rotation_axis_norm.unsqueeze(-1).clamp(min=1.0e-6),
        fallback_axis,
    )
    cos_angle = (default_axis * direction_norm).sum(dim=-1).clamp(-1.0, 1.0)
    orientations = quat_from_angle_axis(torch.acos(cos_angle), rotation_axis)
    scales = torch.ones(positions.shape[0], 3, device=positions.device, dtype=positions.dtype)
    scales[:, 2] = lengths
    return positions, orientations, scales


def _ablation_apply_direct_goal(base_env) -> None:
    """Ablation helper: bypass A* and point the policy directly at the final path endpoint.

    Sets _go2w_goal_pos_w to the last A* waypoint and advances target_index to the
    final index so the policy sees the corridor end-goal at all times, with no rolling
    lookahead. This is the no-global-planning baseline for the SR/SPL ablation study.
    """
    if not hasattr(base_env, "_go2w_navigation_path_w"):
        return
    path_count = base_env._go2w_navigation_path_count.clamp(min=1)
    final_idx = (path_count - 1).clamp(min=0)
    row_idx = torch.arange(base_env.num_envs, device=base_env.device)
    final_pos = base_env._go2w_navigation_path_w[row_idx, final_idx]          # (N, 3)
    robot_xy = base_env.scene["robot"].data.root_pos_w[:, :2]
    base_env._go2w_goal_pos_w[:] = final_pos
    base_env._go2w_navigation_path_final_distance[:] = (robot_xy - final_pos[:, :2]).norm(dim=-1)
    base_env._go2w_navigation_path_target_index[:] = final_idx
