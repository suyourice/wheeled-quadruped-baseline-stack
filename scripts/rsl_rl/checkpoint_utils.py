# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared checkpoint utilities used by train.py, play.py, and play_cmd.py."""

from __future__ import annotations


def find_state_dict(ckpt: dict, candidates: tuple[str, ...], label: str) -> tuple[str, dict]:
    """Return the first matching state dict from a checkpoint."""
    for key in candidates:
        if key in ckpt and isinstance(ckpt[key], dict):
            return key, ckpt[key]
    raise ValueError(f"No {label} state dict found. Keys found: {list(ckpt.keys())}")


def load_padded_state_dict(model, src_sd: dict, device: str, label: str, strip_distribution: bool = False) -> None:
    """Load a state dict, zero-padding first-layer inputs when obs dims grow."""
    import torch

    src_sd = {k: v.to(device) for k, v in src_sd.items()}
    if strip_distribution:
        src_sd = {k: v for k, v in src_sd.items() if not k.startswith("distribution.")}

    current_sd = model.state_dict()
    new_sd = {}
    for key, tgt in current_sd.items():
        if key not in src_sd:
            # Keep current init for keys that do not exist in source, such as distribution params.
            new_sd[key] = tgt
            continue

        src = src_sd[key]
        if src.shape == tgt.shape:
            new_sd[key] = src
        elif len(src.shape) == 2 and src.shape[0] == tgt.shape[0] and src.shape[1] < tgt.shape[1]:
            # First-layer input expansion: append zero columns for new obstacle obs dims.
            n_pad = tgt.shape[1] - src.shape[1]
            pad = torch.zeros(src.shape[0], n_pad, dtype=src.dtype, device=device)
            new_sd[key] = torch.cat([src, pad], dim=1)
            print(f"[INFO] Zero-padded {label} '{key}': {tuple(src.shape)} -> {tuple(new_sd[key].shape)}")
        else:
            print(
                f"[WARN] Shape mismatch in {label} '{key}': "
                f"src={tuple(src.shape)}, tgt={tuple(tgt.shape)}; keeping current init"
            )
            new_sd[key] = tgt

    model.load_state_dict(new_sd)


def load_teacher_locomotion_checkpoint(teacher, ckpt_path: str, device: str) -> None:
    """Initialize the teacher frozen LLC from a flat locomotion checkpoint."""
    import torch

    teacher_target = getattr(teacher, "frozen_actor", None)
    if teacher_target is None:
        raise ValueError("Requires a teacher with a frozen_actor attribute.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor_key, actor_sd = find_state_dict(
        ckpt,
        ("actor_state_dict", "model_state_dict", "policy_state_dict"),
        "actor",
    )
    load_padded_state_dict(teacher_target, actor_sd, device, "teacher frozen actor", strip_distribution=True)
    print(f"[INFO] Loaded teacher frozen LLC from '{actor_key}' in: {ckpt_path}")


def apply_hospital_curriculum_offset(env_cfg, iteration_offset: int, *, strict: bool = True) -> None:
    """Inject a hospital teacher curriculum offset without changing the schedule itself.

    With ``strict=True`` (train), tasks whose reset params lack
    ``curriculum_iteration_offset`` raise; with ``strict=False`` (play), they
    are skipped with an informational message.
    """
    if iteration_offset < 0:
        raise ValueError("--hospital_curriculum_iteration_offset must be non-negative.")

    reset_obstacles = getattr(getattr(env_cfg, "events", None), "reset_obstacles", None)
    params = getattr(reset_obstacles, "params", None)
    if not isinstance(params, dict) or "curriculum_iteration_offset" not in params:
        if strict:
            raise ValueError(
                "--hospital_curriculum_iteration_offset is only supported for hospital teacher tasks "
                "whose reset_obstacles params include 'curriculum_iteration_offset'."
            )
        print(
            "[INFO] reset_obstacles has no 'curriculum_iteration_offset' — skipping curriculum offset "
            "(play env uses a fixed layout, no curriculum needed)."
        )
        return
    params["curriculum_iteration_offset"] = int(iteration_offset)
    print(f"[INFO] Hospital curriculum iteration offset: {iteration_offset}")


def configure_frozen_llc_action(env_cfg, ckpt_path: str | None, task_name: str = "") -> bool:
    """Inject the fast-flat LLC checkpoint into HLC action configs before env creation."""
    import os

    actions_cfg = getattr(env_cfg, "actions", None)
    llc_cmd_cfg = getattr(actions_cfg, "llc_cmd", None)
    if llc_cmd_cfg is None:
        return False
    if not ckpt_path:
        raise ValueError(
            f"Task '{task_name}' uses FrozenLLCActionTerm and requires --locomotion_checkpoint "
            "before gym.make() so the frozen fast-flat LLC can be loaded."
        )
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Task '{task_name}' uses FrozenLLCActionTerm, but --locomotion_checkpoint does not exist: "
            f"{ckpt_path}"
        )
    llc_cmd_cfg.llc_checkpoint_path = ckpt_path
    print(f"[INFO] Frozen LLC checkpoint injected into action term: {ckpt_path}")
    return True
