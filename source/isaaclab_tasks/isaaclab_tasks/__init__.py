# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal isaaclab_tasks stub for go2w.

Only the utils sub-package is included. The full task suite from the original
isaaclab_tasks is intentionally omitted — go2w registers its own environments.
"""

import os
import toml

ISAACLAB_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
ISAACLAB_TASKS_METADATA = toml.load(os.path.join(ISAACLAB_TASKS_EXT_DIR, "config", "extension.toml"))
__version__ = ISAACLAB_TASKS_METADATA["package"]["version"]

from .utils import import_packages  # noqa: F401, E402
