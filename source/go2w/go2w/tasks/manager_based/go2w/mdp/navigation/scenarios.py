# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation scenario identifiers shared by reset, rewards, and play debug."""

from __future__ import annotations


NAV_SCENARIO_CODES: dict[str, int] = {
    "empty": 0,
    "head_on": 1,
    "left_edge": 2,
    "right_edge": 3,
    "diag_left": 4,
    "diag_right": 5,
    "off_left": 6,
    "off_right": 7,
    "narrow_gap": 8,
    "random_fallback": 9,
    "partial_blockage_left_open": 10,
    "partial_blockage_right_open": 11,
    "cluttered": 12,
    "narrow_gap_wide": 13,
    "narrow_gap_barely": 14,
}

NAV_SCENARIO_NAMES: dict[int, str] = {code: name for name, code in NAV_SCENARIO_CODES.items()}
NAV_RANDOM_FALLBACK_SCENARIO_ID = NAV_SCENARIO_CODES["random_fallback"]
