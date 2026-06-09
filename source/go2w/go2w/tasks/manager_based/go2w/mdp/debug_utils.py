# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared navigation debug helpers controlled by GO2W_NAV_DEBUG env vars."""

from __future__ import annotations

import os

import torch


def nav_debug_enabled() -> bool:
    return os.environ.get("GO2W_NAV_DEBUG", "").lower() in ("1", "true", "yes", "on")


def nav_debug_interval() -> int:
    try:
        return max(1, int(os.environ.get("GO2W_NAV_DEBUG_INTERVAL", "20")))
    except ValueError:
        return 20


def nav_debug_env_id() -> int:
    try:
        return int(os.environ.get("GO2W_NAV_DEBUG_ENV", "0"))
    except ValueError:
        return 0


def fmt_xy(xy: torch.Tensor) -> str:
    vals = xy.detach().cpu().tolist()
    return f"({float(vals[0]):+.2f},{float(vals[1]):+.2f})"
