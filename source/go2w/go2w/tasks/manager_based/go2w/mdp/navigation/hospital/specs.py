# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hospital-specific specification tables for modular navigation scenarios."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..local_planning.obstacle_geometry import (
    OBSTACLE_SHAPE_CONE,
    OBSTACLE_SHAPE_CUBOID,
    OBSTACLE_SHAPE_CYLINDER,
)

TRAIN_PHYSICAL_OBSTACLE_SLOTS = 15
PLAY_PHYSICAL_OBSTACLE_SLOTS = 64
PLAY_DEFAULT_ACTIVE_OBSTACLES = 5

NUM_OBSTACLES = TRAIN_PHYSICAL_OBSTACLE_SLOTS
PRIVILEGED_OBSTACLE_SLOTS = 15
PLAY_MAX_OBSTACLES = PLAY_PHYSICAL_OBSTACLE_SLOTS
PLAY_NUM_OBSTACLES = PLAY_DEFAULT_ACTIVE_OBSTACLES
PLAY_MIN_INTER_OBSTACLE_DIST = 0.7

OBSTACLE_SIZE = (0.3, 0.3, 0.5)
TRAIN_OBSTACLE_SPECS = (
    ("cuboid", (0.30, 0.30)),
    ("cuboid", (0.30, 0.30)),
    ("cuboid", (0.30, 0.30)),
    ("cuboid", (0.30, 0.30)),
    ("cuboid", (0.30, 0.30)),
    ("cuboid", (0.30, 0.30)),
    ("cuboid", (0.30, 0.30)),
    ("cuboid", (0.30, 0.30)),
    ("cuboid", (0.30, 0.30)),
    ("cuboid", (0.30, 0.30)),
    ("cuboid", (0.18, 0.24)),
    ("cuboid", (0.46, 0.30)),
    ("cuboid", (0.46, 0.46)),
    ("cylinder", (0.44, 0.44)),
    ("cone", (0.54, 0.54)),
)
_SHAPE_ID_BY_KIND = {
    "cuboid": OBSTACLE_SHAPE_CUBOID,
    "cylinder": OBSTACLE_SHAPE_CYLINDER,
    "cone": OBSTACLE_SHAPE_CONE,
}
TRAIN_OBSTACLE_SHAPE_IDS = tuple(_SHAPE_ID_BY_KIND[kind] for kind, _ in TRAIN_OBSTACLE_SPECS)
TRAIN_OBSTACLE_WIDTHS = tuple(size[0] for _, size in TRAIN_OBSTACLE_SPECS)
TRAIN_OBSTACLE_DEPTHS = tuple(size[1] for _, size in TRAIN_OBSTACLE_SPECS)
PLAY_OBSTACLE_SPECS = TRAIN_OBSTACLE_SPECS + (
    ("cuboid", (OBSTACLE_SIZE[0], OBSTACLE_SIZE[1])),
) * (PLAY_PHYSICAL_OBSTACLE_SLOTS - TRAIN_PHYSICAL_OBSTACLE_SLOTS)
PLAY_OBSTACLE_SHAPE_IDS = tuple(_SHAPE_ID_BY_KIND[kind] for kind, _ in PLAY_OBSTACLE_SPECS)
PLAY_OBSTACLE_WIDTHS = tuple(size[0] for _, size in PLAY_OBSTACLE_SPECS)
PLAY_OBSTACLE_DEPTHS = tuple(size[1] for _, size in PLAY_OBSTACLE_SPECS)

# Hospital play scene: 48 slots with realistic human-scale footprints. The
# first slots form a dynamic hospital actor palette; structured layouts remap
# this palette after their wall slots so wall count does not discard patients.
HOSPITAL_PLAY_OBSTACLE_SPECS = (
    ("cylinder", (0.50, 0.50)),  # ambulatory patient
    ("cylinder", (0.46, 0.46)),  # elderly person
    ("cylinder", (0.50, 0.50)),  # adult guardian
    ("cylinder", (0.34, 0.34)),  # child
    ("cylinder", (0.48, 0.48)),  # staff pushing wheelchair
    ("cuboid", (0.70, 1.00)),    # wheelchair patient
    ("cylinder", (0.48, 0.48)),  # staff pushing cart
    ("cuboid", (0.60, 0.90)),    # cart
    ("cylinder", (0.48, 0.48)),  # visitor / dog handler
    ("cylinder", (0.35, 0.35)),  # leashed dog
    ("cuboid", (0.68, 0.98)),    # self-propelled wheelchair patient
    ("cuboid", (0.55, 0.80)),    # cleaning machine
    ("cylinder", (0.48, 0.48)),  # staff pushing gurney
    ("cuboid", (1.95, 0.95)),    # gurney with a lying patient
    ("cylinder", (0.52, 0.52)),  # patient with IV
    ("cylinder", (0.22, 0.22)),  # IV pole
    ("cuboid", (0.35, 0.55)),    # chair
    ("cuboid", (1.20, 0.55)),    # bench
    ("cylinder", (0.32, 0.32)),  # trash bin
    ("cuboid", (0.80, 0.80)),    # small table
    ("cuboid", (0.25, 0.35)),    # fallen object
    ("cuboid", (2.60, 0.75)),    # reception desk
    ("cylinder", (0.50, 0.50)),  # queueing patient
    ("cylinder", (0.48, 0.48)),  # queueing visitor
    ("cylinder", (0.48, 0.48)),  # queueing visitor
    ("cuboid", (1.80, 0.55)),    # waiting-area bench
    ("cylinder", (0.45, 0.45)),  # seated patient
    ("cylinder", (0.45, 0.45)),  # seated visitor
    ("cylinder", (0.50, 0.50)),  # doorway patient
    ("cylinder", (0.48, 0.48)),  # doorway staff
    ("cylinder", (0.50, 0.50)),  # elevator/waiting patient
    ("cuboid", (0.60, 0.90)),    # supply cart
) + (("cuboid", (OBSTACLE_SIZE[0], OBSTACLE_SIZE[1])),) * (PLAY_PHYSICAL_OBSTACLE_SLOTS - 32)
assert len(HOSPITAL_PLAY_OBSTACLE_SPECS) == PLAY_PHYSICAL_OBSTACLE_SLOTS
HOSPITAL_PLAY_OBSTACLE_SHAPE_IDS = tuple(_SHAPE_ID_BY_KIND[kind] for kind, _ in HOSPITAL_PLAY_OBSTACLE_SPECS)
HOSPITAL_PLAY_OBSTACLE_WIDTHS = tuple(size[0] for _, size in HOSPITAL_PLAY_OBSTACLE_SPECS)
HOSPITAL_PLAY_OBSTACLE_DEPTHS = tuple(size[1] for _, size in HOSPITAL_PLAY_OBSTACLE_SPECS)
HOSPITAL_PLAY_OBSTACLE_HEIGHTS = (
    1.70, 1.60, 1.75, 1.25, 1.75,
    1.30, 1.75, 1.05, 1.70, 0.55,
    1.30, 0.75, 1.75, 1.20, 1.70,
    1.45, 0.85, 0.50, 0.75, 0.75,
    0.20, 1.10, 1.70, 1.70, 1.70,
    0.50, 1.25, 1.25, 1.70, 1.75,
    1.70, 1.05,
) + (OBSTACLE_SIZE[2],) * (PLAY_PHYSICAL_OBSTACLE_SLOTS - 32)
assert len(HOSPITAL_PLAY_OBSTACLE_HEIGHTS) == PLAY_PHYSICAL_OBSTACLE_SLOTS

# Label assigned to each play obstacle slot (used for velocity resampling).
# Order mirrors HOSPITAL_PLAY_OBSTACLE_SPECS.
HOSPITAL_PLAY_OBSTACLE_LABELS: tuple[str, ...] = (
    "patient_ambulatory", "elderly", "adult", "child", "staff",
    "wheelchair_patient", "staff", "cart", "visitor", "dog",
    "wheelchair_patient", "cleaning_machine", "staff", "gurney_patient", "patient_with_iv",
    "iv_pole", "chair", "bench", "trash_bin", "table", "fallen_object",
    "reception_desk", "queue_patient", "queue_visitor", "queue_visitor",
    "bench", "seated_patient", "seated_visitor", "doorway_patient", "doorway_staff",
    "queue_patient", "cart",
) + ("chair",) * (PLAY_PHYSICAL_OBSTACLE_SLOTS - 32)
assert len(HOSPITAL_PLAY_OBSTACLE_LABELS) == PLAY_PHYSICAL_OBSTACLE_SLOTS

# Dynamic-palette leader→follower group pairs:
# (leader_dynamic_idx, follower_dynamic_idx, relation_type).
# Structured layouts add their wall_count offset when building physical names.
HOSPITAL_PLAY_GROUP_PAIRS: tuple[tuple[int, int, str], ...] = (
    (2, 3, "guardian_child"),
    (4, 5, "wheelchair_assisted"),
    (6, 7, "pusher_payload"),
    (8, 9, "handler_dog"),
    (12, 13, "pusher_gurney"),
    (14, 15, "patient_iv"),
)

HOSPITAL_DEFAULT_COLOR = (0.80, 0.20, 0.20)
HOSPITAL_LABEL_COLORS: dict[str, tuple[float, float, float]] = {
    "wall": (0.72, 0.36, 0.38),
    "patient_ambulatory": (0.22, 0.44, 0.88),
    "elderly": (0.42, 0.50, 0.68),
    "adult": (0.20, 0.62, 0.46),
    "child": (0.95, 0.78, 0.22),
    "staff": (0.10, 0.58, 0.70),
    "visitor": (0.55, 0.42, 0.82),
    "dog": (0.55, 0.34, 0.18),
    "wheelchair_patient": (0.28, 0.30, 0.38),
    "cart": (0.84, 0.58, 0.18),
    "cleaning_machine": (0.18, 0.58, 0.36),
    "gurney_patient": (0.92, 0.92, 0.86),
    "patient_with_iv": (0.25, 0.52, 0.88),
    "iv_pole": (0.82, 0.84, 0.86),
    "chair": (0.50, 0.42, 0.34),
    "bench": (0.42, 0.34, 0.26),
    "trash_bin": (0.24, 0.34, 0.30),
    "table": (0.62, 0.48, 0.34),
    "fallen_object": (0.70, 0.38, 0.22),
    "reception_desk": (0.30, 0.46, 0.62),
    "queue_patient": (0.95, 0.55, 0.22),
    "queue_visitor": (0.72, 0.45, 0.76),
    "seated_patient": (0.35, 0.50, 0.82),
    "seated_visitor": (0.65, 0.48, 0.78),
    "doorway_patient": (0.24, 0.60, 0.86),
    "doorway_staff": (0.08, 0.62, 0.70),
}

OBSTACLE_GROUND_CLEARANCE = 0.05
OBSTACLE_Z = OBSTACLE_SIZE[2] / 2 + OBSTACLE_GROUND_CLEARANCE
OBSTACLE_SPAWN_RANGE = {"x": (-3.5, 3.5), "y": (-2.5, 2.5)}
OBSTACLE_NAMES = [f"obstacle_{i}" for i in range(TRAIN_PHYSICAL_OBSTACLE_SLOTS)]
PLAY_OBSTACLE_NAMES = [f"obstacle_{i}" for i in range(PLAY_PHYSICAL_OBSTACLE_SLOTS)]

CURRICULUM_STEPS_PER_ITERATION = 128
CURRICULUM_OBSTACLE_START_ITERATION = 1700
CURRICULUM_OBSTACLE_WARMUP_ITERATIONS = 1000
CURRICULUM_COLLISION_WARMUP_ITERATIONS = 300
CURRICULUM_SPEED_START_ITERATION = 0
CURRICULUM_SPEED_WARMUP_ITERATIONS = 800
NAV_CURRICULUM_COLLISION_START_ITERATION = 0
OBSTACLE_MIN_SPAWN_DISTANCE_INITIAL = 2.2
OBSTACLE_MIN_SPAWN_DISTANCE_FROM_ROBOT = 1.2
OBSTACLE_LIN_VEL_X = (-2.0, 2.0)
OBSTACLE_LIN_VEL_Y = (-2.0, 2.0)
OBSTACLE_ANG_VEL_Z = (-2.0, 2.0)
OBSTACLE_COLLISION_WEIGHT = -40.0
NAV_TTC_FALLBACK_OBSTACLE_RADIUS = 0.22
NAV_TTC_ROBOT_HALF_WIDTH = 0.30
NAV_TTC_SAFETY_MARGIN = 0.05
NAV_TTC_FRONT_MARGIN = 0.20
NAV_TTC_LOOKAHEAD_DISTANCE = 2.2
NAV_TTC_SUM_CLIP = 1.5
NAV_OBSTACLE_RADIUS_MARGIN = 0.03
NAV_PHYSICAL_SLOT_RANDOMIZATION_START_ITERATION = 500
NAV_PHYSICAL_SLOT_RANDOMIZATION_WARMUP_ITERATIONS = 500
NAV_RANDOMIZE_OBSTACLE_YAW = True
NAV_OBSTACLE_YAW_RANGE = (-math.pi, math.pi)
NAV_PASSABLE_GAP_ROBOT_WIDTH = 0.44
NAV_PASSABLE_GAP_MIN_WIDTH = 0.50

NAV_GOAL_FORWARD_RANGE = (2.5, 4.5)
NAV_GOAL_LATERAL_RANGE = (-1.5, 1.5)
NAV_GOAL_HEADING_JITTER_RANGE = (-0.35, 0.35)
NAV_MIN_GOAL_DISTANCE = 2.0
NAV_START_EXCLUSION_RADIUS = 1.0
NAV_GOAL_EXCLUSION_RADIUS = 0.9
NAV_HEAD_ON_PROGRESS_RANGE = (0.2, 0.85)
NAV_HEAD_ON_LATERAL_RANGE = (-0.25, 0.25)
NAV_EDGE_PROGRESS_RANGE = (0.25, 0.8)
NAV_EDGE_LATERAL_RANGE = (0.55, 1.1)
NAV_DIAGONAL_PROGRESS_RANGE = (0.15, 0.7)
NAV_DIAGONAL_LATERAL_RANGE = (0.8, 1.6)
NAV_OFFPATH_PROGRESS_RANGE = (0.3, 0.9)
NAV_OFFPATH_LATERAL_RANGE = (1.3, 2.2)
NAV_NARROW_GAP_PROGRESS_RANGE = (0.35, 0.75)
NAV_NARROW_GAP_CENTER_LATERAL_RANGE = (-0.15, 0.15)
NAV_NARROW_GAP_HALF_WIDTH_RANGE = (0.45, 0.65)
NAV_NARROW_GAP_WIDE_HALF_WIDTH_RANGE = (0.60, 0.80)
NAV_NARROW_GAP_BARELY_HALF_WIDTH_RANGE = (0.40, 0.52)
NAV_NARROW_GAP_PROBABILITY = 0.25
NAV_PARTIAL_BLOCKAGE_PROGRESS_RANGE = (0.2, 0.75)
NAV_PARTIAL_BLOCKAGE_LATERAL_RANGE = (0.5, 1.15)
NAV_PARTIAL_BLOCKAGE_PROBABILITY = 0.20
NAV_CLUTTERED_PROGRESS_RANGE = (0.15, 0.85)
NAV_CLUTTERED_LATERAL_RANGE = (-1.2, 1.2)

NAV_CURRICULUM_PHASE_SCHEDULE = {
    "0": (
        "head_on", "left_edge", "right_edge",
        "diag_left", "diag_right", "off_left", "off_right",
        "narrow_gap",
    ),
    "500": (
        "head_on", "left_edge", "right_edge",
        "diag_left", "diag_right", "off_left", "off_right",
        "narrow_gap",
        "partial_blockage_left_open", "partial_blockage_right_open",
    ),
    "1000": (
        "head_on", "left_edge", "right_edge",
        "diag_left", "diag_right", "off_left", "off_right",
        "narrow_gap", "narrow_gap_wide", "narrow_gap_barely",
        "partial_blockage_left_open", "partial_blockage_right_open",
        "cluttered",
    ),
}

NAV_GOAL_HEADING_STD = 0.8
NAV_GOAL_SUCCESS_POSITION_THRESHOLD = 0.35
NAV_GOAL_SUCCESS_HEADING_THRESHOLD = 0.6

NAV_WAYPOINT_LOOKAHEAD_DISTANCE = 1.25
NAV_WAYPOINT_GOAL_SNAP_DISTANCE = 1.0
NAV_WAYPOINT_REFINEMENT_OFFSETS = (0.0, 0.45, -0.45, 0.70, -0.70)
NAV_LOCAL_PLANNER_ACTIVATION_THRESHOLD = 0.22
NAV_LOCAL_PLANNER_LATERAL_PENALTY = 0.16
NAV_LOCAL_PLANNER_MIN_IMPROVEMENT = 0.07
NAV_LOCAL_PLANNER_MAX_BLEND = 0.65
#                                        train   play
NAV_WAYPOINT_COMMAND_MIN_FORWARD       = -0.3;  NAV_WAYPOINT_COMMAND_MIN_FORWARD_PLAY  = 0.0
NAV_WAYPOINT_COMMAND_MAX_LATERAL       =  1.5;  NAV_WAYPOINT_COMMAND_MAX_LATERAL_PLAY  = 0.25
NAV_WAYPOINT_COMMAND_MAX_HEADING       =  0.90; NAV_WAYPOINT_COMMAND_MAX_HEADING_PLAY  = 0.90

NAV_PASSABLE_GAP_REWARD_WEIGHT = 1.5
NAV_CLEARANCE_PASSABLE_GAP_RELIEF = 0.5
NAV_TTC_PASSABLE_GAP_RELIEF = 0.5
NAV_DENSE_RECOVERY_WEIGHT = 1.0
NAV_GRAZING_WEIGHT = -0.5
NAV_CLEARANCE_SURFACE_BUFFER = 0.20
NAV_GRAZING_DISTANCE = 0.05
NAV_GRAZING_CONTACT_DISTANCE = -0.10
NAV_GRAZING_PASSABLE_GAP_RELIEF = 0.4

LIDAR_MAX_DISTANCE = 20.0
LIDAR_HORIZONTAL_FOV = (0.0, 360.0)
LIDAR_HORIZONTAL_RES = 2.0
LIDAR_CHANNELS = 1
LIDAR_VERTICAL_FOV = (0.0, 0.0)

D456_DEPTH_MIN_DISTANCE = 0.60
D456_DEPTH_MAX_DISTANCE = 6.0
D456_DEPTH_HORIZONTAL_FOV_DEG = 86.0
D456_DEPTH_VERTICAL_FOV_DEG = 57.0
D456_NATIVE_DEPTH_RESOLUTION = (1280, 720)
DEPTH_IMAGE_WIDTH = 128
DEPTH_IMAGE_HEIGHT = 72
DEPTH_HISTORY_LENGTH = 3
DEPTH_HISTORY_LENGTH_LONG = 8
DEPTH_CAMERA_SPARSE_STRIDE = 5
DEPTH_DISTILL_NUM_ENVS = 512
DEPTH_DISTILL_MIN_OBSTACLES = 6
DEPTH_DISTILL_MAX_OBSTACLES = 10
DEPTH_DISTILL_EMPTY_ENV_FRACTION = 0.05
DEPTH_DISTILL_MIN_INTER_OBSTACLE_DIST = 0.9
DEPTH_DISTILL_DYNAMIC_START_ITERATION = 250
DEPTH_DISTILL_DYNAMIC_WARMUP_ITERATIONS = 250
DEPTH_DISTILL_DYNAMIC_SPEED_RANGE = (0.03, 0.40)
DEPTH_DISTILL_DYNAMIC_LATERAL_SPEED = 0.08
DEPTH_DISTILL_DYNAMIC_LONGITUDINAL_EXTENT = 1.4
DEPTH_DISTILL_DYNAMIC_LATERAL_EXTENT = 0.35
DEPTH_DISTILL_DYNAMIC_SPEED_CHANGE_INTERVAL = (1.2, 2.8)
DEPTH_DISTILL_DYNAMIC_WANDER_FRACTION = 0.15
D456_CAMERA_FOCAL_LENGTH_CM = 24.0
D456_CAMERA_HORIZONTAL_APERTURE_CM = 2.0 * D456_CAMERA_FOCAL_LENGTH_CM * math.tan(
    math.radians(D456_DEPTH_HORIZONTAL_FOV_DEG) * 0.5
)
D456_CAMERA_VERTICAL_APERTURE_CM = 2.0 * D456_CAMERA_FOCAL_LENGTH_CM * math.tan(
    math.radians(D456_DEPTH_VERTICAL_FOV_DEG) * 0.5
)
D456_CAMERA_PITCH_DOWN_DEG = 5.0
_D456_CAMERA_PITCH_HALF_RAD = math.radians(D456_CAMERA_PITCH_DOWN_DEG) * 0.5
D456_CAMERA_PITCH_DOWN_QUAT_WXYZ = (
    math.cos(_D456_CAMERA_PITCH_HALF_RAD),
    0.0,
    math.sin(_D456_CAMERA_PITCH_HALF_RAD),
    0.0,
)


def _quat_multiply_wxyz(q1, q2):
    """Quaternion multiplication (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )

_D456_YAW_LEFT_QUAT  = (math.cos(math.pi / 4), 0.0, 0.0,  math.sin(math.pi / 4))
_D456_YAW_RIGHT_QUAT = (math.cos(math.pi / 4), 0.0, 0.0, -math.sin(math.pi / 4))
_D456_YAW_180_QUAT   = (0.0, 0.0, 0.0, 1.0)

D456_CAMERA_LEFT_QUAT_WXYZ  = _quat_multiply_wxyz(_D456_YAW_LEFT_QUAT,  D456_CAMERA_PITCH_DOWN_QUAT_WXYZ)
D456_CAMERA_RIGHT_QUAT_WXYZ = _quat_multiply_wxyz(_D456_YAW_RIGHT_QUAT, D456_CAMERA_PITCH_DOWN_QUAT_WXYZ)
D456_CAMERA_REAR_QUAT_WXYZ  = _quat_multiply_wxyz(_D456_YAW_180_QUAT,   D456_CAMERA_PITCH_DOWN_QUAT_WXYZ)


@dataclass(frozen=True)
class HospitalLabelSpec:
    label: str
    category: str
    primitive_kind: str
    footprint_range: tuple[float, float]
    height_range: tuple[float, float]
    speed_range: tuple[float, float]
    yaw_rate_range: tuple[float, float]
    priority: float
    motion_profile: str
    relation_type: str = "independent"
    notes: str = ""


HOSPITAL_LABEL_SPECS: dict[str, HospitalLabelSpec] = {
    "patient_ambulatory": HospitalLabelSpec("patient_ambulatory", "patient", "cylinder", (0.40, 0.60), (1.45, 1.80), (0.20, 0.70), (-0.8, 0.8), 3.0, "careful_walk", notes="Walking patient, highest avoidance priority."),
    "patient_with_iv": HospitalLabelSpec("patient_with_iv", "patient", "cylinder", (0.42, 0.62), (1.45, 1.80), (0.18, 0.60), (-0.8, 0.8), 3.0, "careful_walk", relation_type="patient_iv", notes="Walking patient paired with an IV pole."),
    "wheelchair_patient": HospitalLabelSpec("wheelchair_patient", "patient", "cuboid", (0.60, 0.95), (1.05, 1.45), (0.20, 0.80), (-0.9, 0.9), 3.0, "careful_roll", relation_type="wheelchair_assisted", notes="Wheelchair user, always top-priority avoidance."),
    "elderly": HospitalLabelSpec("elderly", "person", "cylinder", (0.40, 0.60), (1.45, 1.75), (0.15, 0.55), (-0.6, 0.6), 3.0, "slow_walk", notes="Older person, highest avoidance priority group."),
    "child": HospitalLabelSpec("child", "person", "cylinder", (0.28, 0.40), (1.05, 1.45), (0.60, 1.80), (-2.0, 2.0), 3.0, "burst_runner", relation_type="guardian_child", notes="Child can run or wander even when escorted."),
    "adult": HospitalLabelSpec("adult", "person", "cylinder", (0.38, 0.55), (1.55, 1.90), (0.60, 1.40), (-1.2, 1.2), 1.7, "walk", notes="Generic adult visitor or helper."),
    "staff": HospitalLabelSpec("staff", "person", "cylinder", (0.38, 0.55), (1.55, 1.90), (0.80, 1.50), (-1.2, 1.2), 1.7, "purposeful_walk", notes="Doctor or nurse; same priority band as adults."),
    "visitor": HospitalLabelSpec("visitor", "person", "cylinder", (0.38, 0.55), (1.55, 1.90), (0.50, 1.20), (-1.2, 1.2), 1.7, "wander", notes="Generic hospital visitor."),
    "cart": HospitalLabelSpec("cart", "equipment", "cuboid", (0.55, 0.90), (0.80, 1.20), (0.30, 1.00), (-0.5, 0.5), 1.2, "pushed_payload", relation_type="pusher_payload", notes="Cart always moves with a human pusher."),
    "wheelchair": HospitalLabelSpec("wheelchair", "equipment", "cuboid", (0.55, 0.90), (0.80, 1.20), (0.30, 1.00), (-0.5, 0.5), 1.2, "pushed_payload", relation_type="pusher_payload", notes="Wheelchair equipment when modeled as separate payload."),
    "gurney": HospitalLabelSpec("gurney", "equipment", "cuboid", (0.95, 1.30), (0.90, 1.20), (0.10, 0.55), (-0.2, 0.2), 1.2, "pushed_payload", relation_type="pusher_payload", notes="Bed or gurney pushed by staff."),
    "gurney_patient": HospitalLabelSpec("gurney_patient", "patient_equipment", "cuboid", (0.90, 1.10), (1.05, 1.30), (0.10, 0.55), (-0.2, 0.2), 3.0, "pushed_payload", relation_type="pusher_payload", notes="Patient lying on a pushed gurney/bed trolley."),
    "iv_pole": HospitalLabelSpec("iv_pole", "equipment", "cylinder", (0.16, 0.28), (1.20, 1.70), (0.10, 0.60), (-0.8, 0.8), 1.2, "pushed_payload", relation_type="patient_iv", notes="IV pole coupled to a walking patient."),
    "cleaning_machine": HospitalLabelSpec("cleaning_machine", "equipment", "cuboid", (0.35, 0.60), (0.45, 0.90), (0.20, 0.80), (-0.8, 0.8), 1.2, "cleaning_pass", notes="Small ride-on or push-cleaner unit."),
    "dog": HospitalLabelSpec("dog", "animal", "cylinder", (0.25, 0.40), (0.35, 0.60), (0.40, 1.50), (-2.0, 2.0), 1.2, "leashed_pet", relation_type="handler_dog", notes="Always leashed in hospital scenes."),
    "queue_patient": HospitalLabelSpec("queue_patient", "patient", "cylinder", (0.40, 0.60), (1.45, 1.80), (0.03, 0.16), (-0.25, 0.25), 3.0, "queue_wait", notes="Patient standing in a reception/elevator queue with small shuffling motion."),
    "queue_visitor": HospitalLabelSpec("queue_visitor", "person", "cylinder", (0.38, 0.55), (1.55, 1.90), (0.03, 0.14), (-0.25, 0.25), 1.7, "queue_wait", notes="Visitor standing in a queue with small shuffling motion."),
    "seated_patient": HospitalLabelSpec("seated_patient", "patient", "cylinder", (0.38, 0.55), (1.05, 1.35), (0.0, 0.0), (0.0, 0.0), 3.0, "static", notes="Patient seated on a bench or chair in the waiting area."),
    "seated_visitor": HospitalLabelSpec("seated_visitor", "person", "cylinder", (0.38, 0.55), (1.05, 1.35), (0.0, 0.0), (0.0, 0.0), 1.7, "static", notes="Visitor seated on a bench or chair in the waiting area."),
    "doorway_patient": HospitalLabelSpec("doorway_patient", "patient", "cylinder", (0.40, 0.60), (1.45, 1.80), (0.25, 0.90), (-1.0, 1.0), 3.0, "door_crossing", notes="Patient entering/exiting a room into the corridor."),
    "doorway_staff": HospitalLabelSpec("doorway_staff", "person", "cylinder", (0.38, 0.55), (1.55, 1.90), (0.40, 1.20), (-1.2, 1.2), 1.7, "door_crossing", notes="Staff member crossing from a room or service area into the corridor."),
    "reception_desk": HospitalLabelSpec("reception_desk", "furniture", "cuboid", (2.50, 3.50), (1.00, 1.30), (0.0, 0.0), (0.0, 0.0), 0.5, "static", notes="Large fixed desk in reception area."),
    "chair": HospitalLabelSpec("chair", "furniture", "cuboid", (0.35, 0.60), (0.40, 0.90), (0.0, 0.0), (0.0, 0.0), 0.5, "static"),
    "bench": HospitalLabelSpec("bench", "furniture", "cuboid", (0.60, 1.90), (0.40, 0.90), (0.0, 0.0), (0.0, 0.0), 0.5, "static"),
    "table": HospitalLabelSpec("table", "furniture", "cuboid", (0.80, 1.60), (0.70, 1.10), (0.0, 0.0), (0.0, 0.0), 0.5, "static"),
    "trash_bin": HospitalLabelSpec("trash_bin", "furniture", "cylinder", (0.25, 0.40), (0.60, 0.90), (0.0, 0.0), (0.0, 0.0), 0.5, "static"),
    "fallen_object": HospitalLabelSpec("fallen_object", "misc", "cuboid", (0.15, 0.30), (0.10, 0.25), (0.0, 0.0), (0.0, 0.0), 0.5, "static"),
}


__all__ = [name for name in globals() if name.isupper() or name in {"HospitalLabelSpec", "HOSPITAL_LABEL_SPECS"}]
