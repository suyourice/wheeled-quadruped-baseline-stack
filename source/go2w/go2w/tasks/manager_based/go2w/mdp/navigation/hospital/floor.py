# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hospital full-floor scene configuration: constants, ramp geometry, and semantic placements.

Kept separate from cfg/hospital/env.py so the environment classes stay focused on
the cfg object hierarchy.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

# ---------------------------------------------------------------------------
# Ramp asset names
# ---------------------------------------------------------------------------

HOSPITAL_RAMP_ASSET_NAME = "hospital_ramp"
HOSPITAL_RAMP_B_ASSET_NAME = "hospital_ramp_b"

# ---------------------------------------------------------------------------
# Ramp geometry
# Two thin inclined cuboids form a ∧ shape:
#   hospital_ramp  = ascending half  (entry bottom edge at z=0, rises to ridge)
#   hospital_ramp_b = descending half (ridge to exit bottom edge at z=0)
# ---------------------------------------------------------------------------

HOSPITAL_RAMP_HALF_LENGTH = 1.2       # each slab length along corridor (m)
HOSPITAL_RAMP_PEAK_HEIGHT = 0.07      # ridge height above ground (m)
HOSPITAL_RAMP_PITCH = math.atan2(HOSPITAL_RAMP_PEAK_HEIGHT, HOSPITAL_RAMP_HALF_LENGTH)
HOSPITAL_RAMP_BOX_THICKNESS = 0.01   # slab thickness (m) — entry step ≈ 1 cm
HOSPITAL_RAMP_BOX_WIDTH = 3.4        # spans the 3.6 m corridor
# Center height so the entry (-X) bottom edge of each slab sits at z = 0.
# Derivation: for Ry(-pitch) the back (-X) bottom corner is at
#   z = -(half_len)*sin(pitch) - (half_thick)*cos(pitch) + z_center = 0
HOSPITAL_RAMP_BOX_Z = (
    (HOSPITAL_RAMP_HALF_LENGTH / 2.0) * math.sin(HOSPITAL_RAMP_PITCH)
    + (HOSPITAL_RAMP_BOX_THICKNESS / 2.0) * math.cos(HOSPITAL_RAMP_PITCH)
)


def _make_hospital_ramp_half_cfg(prim_name: str) -> RigidObjectCfg:
    """Return a RigidObjectCfg for one inclined half-slab of the ∧-ramp."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + prim_name,
        spawn=sim_utils.CuboidCfg(
            size=(HOSPITAL_RAMP_HALF_LENGTH, HOSPITAL_RAMP_BOX_WIDTH, HOSPITAL_RAMP_BOX_THICKNESS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.42)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1000.0, 1000.0, 0.20)),
    )


def make_hospital_ramp_cfg() -> RigidObjectCfg:
    return _make_hospital_ramp_half_cfg(HOSPITAL_RAMP_ASSET_NAME)


def make_hospital_ramp_b_cfg() -> RigidObjectCfg:
    return _make_hospital_ramp_half_cfg(HOSPITAL_RAMP_B_ASSET_NAME)


# ---------------------------------------------------------------------------
# Hospital full-floor layout constants
# ---------------------------------------------------------------------------

HOSPITAL_FLOOR_CORRIDOR_KIND = "hospital_floor"
HOSPITAL_FLOOR_LEG_LENGTH = 14.0
HOSPITAL_FLOOR_CORRIDOR_WIDTH = 3.6
HOSPITAL_FLOOR_WALL_THICKNESS = 0.25
HOSPITAL_FLOOR_DYNAMIC_OBSTACLE_COUNT = 32
HOSPITAL_FLOOR_EPISODE_LENGTH_S = 220.0
HOSPITAL_FLOOR_ENV_SPACING = 96.0
HOSPITAL_FLOOR_ROBOT_START_LOCAL_XY = (2.0, 0.0)
HOSPITAL_FLOOR_GOAL_DONE_RADIUS = 0.85

# Ramp placement in local corridor frame (local_x, local_y, yaw, pitch, z, roll).
#
# Pitch sign convention with Ry(pitch):
#   Ry(-θ) → -X face (robot entry) stays low, +X face rises   → ascending half
#   Ry(+θ) → +X face (robot exit)  stays low, -X face is high → descending half
_RAMP_CENTER_X = 3.20 * HOSPITAL_FLOOR_LEG_LENGTH
HOSPITAL_FLOOR_RAMP_LOCAL_POSE = (
    _RAMP_CENTER_X - HOSPITAL_RAMP_HALF_LENGTH / 2.0,
    HOSPITAL_FLOOR_LEG_LENGTH,
    0.0,
    -HOSPITAL_RAMP_PITCH,   # ascending: entry (-X) bottom at z=0
    HOSPITAL_RAMP_BOX_Z,
    0.0,
)
HOSPITAL_FLOOR_RAMP_B_LOCAL_POSE = (
    _RAMP_CENTER_X + HOSPITAL_RAMP_HALF_LENGTH / 2.0,
    HOSPITAL_FLOOR_LEG_LENGTH,
    0.0,
    +HOSPITAL_RAMP_PITCH,   # descending: exit (+X) bottom at z=0
    HOSPITAL_RAMP_BOX_Z,
    0.0,
)


# ---------------------------------------------------------------------------
# Semantic obstacle placements for the hospital floor
# ---------------------------------------------------------------------------

def hospital_floor_semantic_local_poses(
    wall_count: int, leg_length: float
) -> tuple[tuple[int, float, float, float], ...]:
    """Fixed world-frame placements for named hospital actors and furniture."""
    L = leg_length

    def slot(palette_idx: int, x: float, y: float, yaw: float) -> tuple[int, float, float, float]:
        return (wall_count + palette_idx, x, y, yaw)

    return (
        slot(21, 0.64 * L, -0.50 * L, 0.0),            # reception desk
        slot(22, 0.80 * L, -0.30 * L, math.pi * 0.5),  # queue front
        slot(23, 0.80 * L, -0.24 * L, math.pi * 0.5),
        slot(24, 0.80 * L, -0.18 * L, math.pi * 0.5),
        slot(25, 0.88 * L, -0.10 * L, 0.0),            # waiting bench
        slot(26, 0.95 * L, -0.08 * L, 0.0),            # seated patient
        slot(27, 1.08 * L, -0.08 * L, 0.0),            # seated visitor
        slot(28, 2.74 * L, 1.18 * L, -math.pi * 0.5),  # doorway patient
        slot(29, 2.80 * L, 1.24 * L, -math.pi * 0.5),  # doorway staff
        slot(30, 3.72 * L, 1.16 * L, math.pi),         # elevator/service queue
        slot(31, 2.16 * L, -0.42 * L, 0.0),            # supply cart in service bay
        slot(12, 2.52 * L, L - 0.42, 0.0),             # staff pushing gurney
        slot(13, 2.62 * L, L - 0.42, 0.0),             # gurney with patient
        slot(8,  1.30 * L,  0.72, 0.0),                # dog handler
        slot(9,  1.36 * L,  0.72, 0.0),                # leashed dog
        slot(14, 1.78 * L,  0.78, 0.0),                # patient with IV
        slot(15, 1.84 * L,  0.98, 0.0),                # IV pole
        slot(16, 2.54 * L,  L - 2.05, math.pi * 0.5),  # chair in imaging/check bay
        slot(17, 3.72 * L,  L - 1.60, 0.0),            # side bench in pharmacy/service bay
        slot(4,  1.52 * L, -0.58, 0.0),                # staff pushing wheelchair
        slot(5,  1.62 * L, -0.58, 0.0),                # wheelchair patient
        slot(6,  2.03 * L, -0.35 * L, 0.0),            # staff pushing cart
        slot(7,  2.11 * L, -0.35 * L, 0.0),            # cart
        slot(10, 3.02 * L,  L + 0.72, math.pi),        # self-propelled wheelchair (elevator side)
        slot(11, 3.28 * L,  0.72 * L, math.pi * 0.5),  # cleaning machine (ramp landing)
        slot(20, 1.38 * L,  1.52, 0.0),                # fallen object (near north wall)
        slot(0,  3.90 * L,  L + 0.52, math.pi),        # ambulatory patient (far connector)
        slot(1,  2.58 * L,  0.72 * L, math.pi * 0.5),  # elderly patient (imaging bay)
        slot(2,  1.44 * L, -1.35, 0.0),                # guardian/adult (near south wall)
        slot(3,  1.49 * L, -1.35, 0.0),                # child with guardian
        slot(18, 1.60 * L,  1.45, 0.0),                # trash bin (corridor wall)
        slot(19, 2.60 * L,  L - 2.5, 0.0),             # table (imaging/check bay)
    )


def hospital_floor_queue_groups(wall_count: int) -> list[dict]:
    """Reception queue behavior group."""
    return [
        {
            "names": [
                f"obstacle_{wall_count + 22}",
                f"obstacle_{wall_count + 23}",
                f"obstacle_{wall_count + 24}",
            ],
            "spacing": 0.70,
            "advance_speed": 0.28,
            "shuffle_amplitude": 0.035,
            "idle_interval_range": (1.2, 2.8),
            "wave_delay_per_person": 0.18,
        },
        {
            "names": [f"obstacle_{wall_count + 30}"],
            "spacing": 0.55,
            "advance_speed": 0.18,
            "shuffle_amplitude": 0.015,
            "idle_interval_range": (2.0, 4.0),
            "advance_direction_local_xy": (0.0, 1.0),
        },
    ]


def hospital_floor_seated_groups(wall_count: int) -> list[dict]:
    """Waiting-area seated-person stand-up behaviors."""
    return [
        {
            "name": f"obstacle_{wall_count + 26}",
            "merge_offset_xy": (0.0, 0.90),
            "stand_delay_range": (5.0, 10.0),
            "stand_duration": 2.6,
        },
        {
            "name": f"obstacle_{wall_count + 27}",
            "merge_offset_xy": (0.0, 0.78),
            "stand_delay_range": (9.0, 15.0),
            "stand_duration": 3.0,
        },
    ]


__all__ = [
    "HOSPITAL_RAMP_ASSET_NAME",
    "HOSPITAL_RAMP_B_ASSET_NAME",
    "HOSPITAL_RAMP_HALF_LENGTH",
    "HOSPITAL_RAMP_PEAK_HEIGHT",
    "HOSPITAL_RAMP_PITCH",
    "HOSPITAL_RAMP_BOX_THICKNESS",
    "HOSPITAL_RAMP_BOX_WIDTH",
    "HOSPITAL_RAMP_BOX_Z",
    "make_hospital_ramp_cfg",
    "make_hospital_ramp_b_cfg",
    "HOSPITAL_FLOOR_CORRIDOR_KIND",
    "HOSPITAL_FLOOR_LEG_LENGTH",
    "HOSPITAL_FLOOR_CORRIDOR_WIDTH",
    "HOSPITAL_FLOOR_WALL_THICKNESS",
    "HOSPITAL_FLOOR_DYNAMIC_OBSTACLE_COUNT",
    "HOSPITAL_FLOOR_EPISODE_LENGTH_S",
    "HOSPITAL_FLOOR_ENV_SPACING",
    "HOSPITAL_FLOOR_ROBOT_START_LOCAL_XY",
    "HOSPITAL_FLOOR_GOAL_DONE_RADIUS",
    "HOSPITAL_FLOOR_RAMP_LOCAL_POSE",
    "HOSPITAL_FLOOR_RAMP_B_LOCAL_POSE",
    "hospital_floor_semantic_local_poses",
    "hospital_floor_queue_groups",
    "hospital_floor_seated_groups",
]
