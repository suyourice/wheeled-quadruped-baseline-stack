# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Import-safe helpers shared by play and training entrypoints.

Import-safe before AppLauncher: module-level imports are stdlib only; anything
that needs Isaac Sim or rsl_rl is imported inside the function that uses it.
"""

from __future__ import annotations

import copy
import random

DEFAULT_COMMAND_PATH_FORWARD_RANGE = (1.6, 2.4)
DEFAULT_COMMAND_PATH_LATERAL_RANGE = (-0.35, 0.35)
DEFAULT_COMMAND_PATH_MIN_SPEED = 0.2

# Sentinel distinguishing "argument not given" from an explicit None.
_UNSET = object()


class TeeStream:
    """Mirrors writes to multiple streams (stdout + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def __getattr__(self, name):
        return getattr(self._streams[0], name)


def format_eval_metrics(metrics: dict[str, float], completed_episodes: int, avg_episode_length: float) -> str:
    """Format the common play and teacher-evaluation summary metrics."""
    preferred_keys = (
        "goal_reached_rate",
        "spl",
        "time_out_rate",
        "base_contact_rate",
        "root_height_below_minimum_rate",
        "multi_term_fraction",
    )
    parts = [f"episodes={completed_episodes}", f"avg_episode_len={avg_episode_length:.2f}"]
    for key in preferred_keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    return " ".join(parts)


def resolve_play_seed(args_cli, default_seed: int | None) -> int:
    """Return --seed when given, otherwise a fresh random seed per run."""
    if args_cli.seed is not None:
        return args_cli.seed
    base_seed = default_seed if default_seed is not None else 0
    return (base_seed + random.SystemRandom().randrange(1, 2_147_483_647)) % 2_147_483_647


def build_teacher_policy(env, obs, agent_cfg, device: str, locomotion_checkpoint: str | None):
    """Instantiate the rule-based steering teacher for direct play/evaluation."""
    from rsl_rl.utils import resolve_callable

    from checkpoint_utils import load_teacher_locomotion_checkpoint

    teacher_cfg = getattr(agent_cfg, "teacher", None)
    if teacher_cfg is None:
        raise ValueError(
            "--teacher_steering requires a distillation runner config with a 'teacher' model config."
        )
    if "teacher" not in obs:
        raise ValueError(
            "--teacher_steering requires an environment exposing a 'teacher' observation group. "
            "Use a distillation play task such as Nav-ObstacleFlat-Distill-Lidar-Go2w-Play-v0."
        )
    if locomotion_checkpoint is None:
        raise ValueError("--teacher_steering requires --locomotion_checkpoint for the frozen LLC.")

    teacher_cfg_dict = copy.deepcopy(teacher_cfg.to_dict())
    teacher_class = resolve_callable(teacher_cfg_dict.pop("class_name"))
    teacher = teacher_class(obs, {"teacher": ["teacher"]}, "teacher", env.num_actions, **teacher_cfg_dict)
    teacher = teacher.to(device)
    teacher.eval()
    load_teacher_locomotion_checkpoint(teacher, locomotion_checkpoint, device)
    return teacher


def _iter_obstacle_name_params(cfg_obj, prefix: str):
    """Yield every manager term whose params explicitly name obstacle slots."""
    for name, value in vars(cfg_obj).items():
        if value is None or name.startswith("_"):
            continue
        params = getattr(value, "params", None)
        if isinstance(params, dict):
            if "obstacle_names" in params:
                yield f"{prefix}.{name}", params["obstacle_names"]
            continue
        if hasattr(value, "__dict__"):  # observation group — one level deeper
            yield from _iter_obstacle_name_params(value, f"{prefix}.{name}")


def preflight_check_cfg_obstacle_slots(cfg, scene_obstacle_names=None) -> int:
    """Assert exact obstacle-slot coverage for a task configuration.

    This is intentionally usable before an Isaac environment is created.  It
    catches both a short list and a same-length list with a missing/duplicated
    slot, which was the teacher-oracle 16-versus-20 failure mode.
    """
    import re

    if scene_obstacle_names is None:
        scene = getattr(cfg, "scene", None)
        scene_obstacle_names = [
            name for name in vars(scene).keys()
            if re.fullmatch(r"obstacle_\d+", name)
        ]
    expected = tuple(sorted(scene_obstacle_names, key=lambda name: int(name.rsplit("_", 1)[1])))
    if not expected:
        return 0

    mismatches = []
    expected_set = set(expected)
    for manager_name in ("observations", "rewards", "events", "terminations"):
        manager_cfg = getattr(cfg, manager_name, None)
        if manager_cfg is None:
            continue
        for term_path, names in _iter_obstacle_name_params(manager_cfg, manager_name):
            if not isinstance(names, (list, tuple)):
                mismatches.append(f"{term_path}: obstacle_names must be a list/tuple, got {type(names).__name__}")
                continue
            actual = tuple(names)
            if len(actual) != len(expected) or set(actual) != expected_set or len(set(actual)) != len(actual):
                missing = sorted(expected_set - set(actual))
                extra = sorted(set(actual) - expected_set)
                duplicate_count = len(actual) - len(set(actual))
                details = [f"{len(actual)} names vs {len(expected)} scene slots"]
                if missing:
                    details.append(f"missing={missing}")
                if extra:
                    details.append(f"extra={extra}")
                if duplicate_count:
                    details.append(f"duplicates={duplicate_count}")
                mismatches.append(f"{term_path}: {', '.join(details)}")

    if mismatches:
        raise RuntimeError(
            "Obstacle-slot mismatch between scene and manager terms — these terms "
            "silently ignore part of the scene:\n  " + "\n  ".join(mismatches)
        )
    return len(expected)


def preflight_check_obstacle_slots(env) -> None:
    """Run the obstacle-slot preflight against the instantiated scene."""
    import os
    import re

    if os.environ.get("GO2W_SKIP_PREFLIGHT") == "1":
        print("[WARN] Preflight obstacle-slot check skipped (GO2W_SKIP_PREFLIGHT=1).")
        return

    scene = getattr(env, "scene", None)
    rigid_objects = getattr(scene, "rigid_objects", None)
    if not rigid_objects:
        return
    scene_names = [name for name in rigid_objects.keys() if re.fullmatch(r"obstacle_\d+", name)]
    scene_slots = preflight_check_cfg_obstacle_slots(env.cfg, scene_names)
    if scene_slots:
        print(f"[INFO] Preflight OK: all obstacle_names terms match {scene_slots} scene slots.")


def override_play_obstacle_count(env_cfg, num_obstacles: int | None):
    """Override active obstacle count for obstacle play configs."""
    if num_obstacles is None:
        return
    if num_obstacles < 0:
        raise ValueError("--num_obstacles must be >= 0.")

    events_cfg = getattr(env_cfg, "events", None)
    reset_obstacles = getattr(events_cfg, "reset_obstacles", None) if events_cfg is not None else None
    if reset_obstacles is None:
        raise ValueError("--num_obstacles requires an obstacle play task with a reset_obstacles event.")

    params = reset_obstacles.params
    obstacle_names = params.get("obstacle_names", [])
    max_available = len(obstacle_names)
    if num_obstacles > max_available:
        raise ValueError(
            f"--num_obstacles={num_obstacles} exceeds the play scene capacity ({max_available}). "
            "Increase PLAY_MAX_OBSTACLES in mdp/navigation/hospital/specs.py if you need more."
        )

    if "start_iteration" in params:
        params["start_iteration"] = 0
    if "warmup_iterations" in params:
        params["warmup_iterations"] = 0
    params["min_obstacles"] = num_obstacles
    params["max_obstacles"] = num_obstacles
    print(f"[INFO] Active play obstacles: {num_obstacles}/{max_available}")


def override_play_command_path_spawn(
    env_cfg,
    command_path_obstacles: int | None,
    command_path_forward_range,
    command_path_lateral_range,
    command_path_min_speed: float | None,
    command_path_reference_xy=_UNSET,
):
    """Override command-direction obstacle spawn for obstacle play configs.

    ``command_path_reference_xy`` is only written into the reset params when the
    caller passes it explicitly (play_cmd.py); play.py leaves it untouched.
    """
    if (
        command_path_obstacles is None
        and command_path_forward_range is None
        and command_path_lateral_range is None
        and command_path_min_speed is None
    ):
        return

    events_cfg = getattr(env_cfg, "events", None)
    reset_obstacles = getattr(events_cfg, "reset_obstacles", None) if events_cfg is not None else None
    if reset_obstacles is None:
        raise ValueError("--command_path_obstacles requires an obstacle play task with a reset_obstacles event.")

    params = reset_obstacles.params
    obstacle_names = params.get("obstacle_names", [])
    max_available = len(obstacle_names)
    active_obstacles = int(params.get("max_obstacles", max_available))

    count = params.get("command_path_obstacles", 0) if command_path_obstacles is None else command_path_obstacles
    if count < 0:
        raise ValueError("--command_path_obstacles must be >= 0.")
    if count > active_obstacles:
        raise ValueError(
            f"--command_path_obstacles={count} exceeds active obstacle count ({active_obstacles}). "
            "Increase --num_obstacles or lower the command-path count."
        )

    forward_range = tuple(command_path_forward_range) if command_path_forward_range is not None else params.get(
        "command_path_forward_range", DEFAULT_COMMAND_PATH_FORWARD_RANGE
    )
    lateral_range = tuple(command_path_lateral_range) if command_path_lateral_range is not None else params.get(
        "command_path_lateral_range", DEFAULT_COMMAND_PATH_LATERAL_RANGE
    )
    min_speed = command_path_min_speed
    if min_speed is None:
        min_speed = params.get("command_path_min_speed", DEFAULT_COMMAND_PATH_MIN_SPEED)

    params["command_path_obstacles"] = count
    params["command_name"] = "base_velocity"
    if command_path_reference_xy is not _UNSET:
        params["command_path_reference_xy"] = command_path_reference_xy
    params["command_path_forward_range"] = forward_range
    params["command_path_lateral_range"] = lateral_range
    params["command_path_min_speed"] = min_speed
    reference_note = (
        "" if command_path_reference_xy is _UNSET else f", reference={command_path_reference_xy}"
    )
    print(
        "[INFO] Command-path obstacle spawn: "
        f"count={count}, forward={forward_range}, lateral={lateral_range}, min_speed={min_speed:.2f}"
        f"{reference_note}"
    )
