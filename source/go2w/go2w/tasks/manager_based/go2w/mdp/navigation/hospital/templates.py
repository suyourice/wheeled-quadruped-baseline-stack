# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Composable hospital layout templates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HospitalLayoutTemplate:
    """Declarative hospital floorplan template."""

    name: str
    corridor_kind: str
    leg_length: float
    corridor_width: float
    room_count: int
    room_sizes: tuple[tuple[float, float], ...]
    has_reception: bool = False
    has_ramp: bool = False
    has_hall: bool = False
    notes: str = ""


@dataclass(frozen=True)
class HospitalSemanticZone:
    """Semantic zone within a hospital floorplan."""

    name: str
    zone_type: str
    center_xy: tuple[float, float]
    size_xy: tuple[float, float]
    notes: str = ""


HOSPITAL_LAYOUT_TEMPLATES: dict[str, HospitalLayoutTemplate] = {
    "main_corridor": HospitalLayoutTemplate("main_corridor", "l_corridor", 12.0, 2.6, 6, ((3.0, 3.0),) * 6, True, False, False, "Hospital main corridor with rooms on both sides."),
    "ward_l": HospitalLayoutTemplate("ward_l", "l_corridor", 10.0, 2.6, 4, ((3.0, 3.0), (3.0, 3.5), (2.5, 3.0), (3.5, 3.0)), True, False, False, "L-shaped ward with rooms and a reception corner."),
    "cross_junction": HospitalLayoutTemplate("cross_junction", "serpentine_corridor", 8.0, 2.8, 8, ((2.8, 3.0), (3.0, 3.0), (3.2, 3.0), (3.0, 3.5), (2.8, 2.8), (3.0, 3.0), (3.2, 3.2), (3.0, 3.0)), False, False, True, "Crossing corridors with a central open hall."),
    "emergency_bay": HospitalLayoutTemplate("emergency_bay", "serpentine_corridor", 9.0, 3.0, 5, ((4.0, 3.0), (3.0, 3.0), (3.5, 3.0), (4.0, 3.5), (3.0, 3.0)), True, False, True, "Busy emergency area with wide hall and side rooms."),
    "ramp_connector": HospitalLayoutTemplate("ramp_connector", "l_corridor", 9.0, 2.4, 3, ((3.0, 3.0), (3.0, 3.0), (3.5, 3.0)), False, True, False, "Connector corridor with a gentle ramp section."),
    "hospital_floor": HospitalLayoutTemplate("hospital_floor", "hospital_floor", 14.0, 3.6, 28, ((3.4, 3.6),) * 28, True, True, True, "Larger combined hospital floor with reception, waiting, ward branch, emergency/service bay, room-door spur, imaging/check bay, ramp/rehab landing, pharmacy/service bay, elevator/exit alcove, and a mild ramp connector."),
}


HOSPITAL_FLOOR_ZONES: tuple[HospitalSemanticZone, ...] = (
    HospitalSemanticZone("reception", "queue", (10.5, -4.6), (8.4, 9.1), "Enclosed reception and queue bay off the main corridor."),
    HospitalSemanticZone("waiting", "seating", (14.7, -2.1), (7.5, 3.6), "Waiting benches with seated patients and visitors near the corridor edge."),
    HospitalSemanticZone("main_corridor", "corridor", (14.0, 0.0), (28.0, 3.6), "Entrance and main corridor flow between reception and wards."),
    HospitalSemanticZone("ward_branch", "ward", (28.0, 7.0), (3.6, 14.0), "Ward branch with slower patients and pushed equipment."),
    HospitalSemanticZone("emergency_service", "service", (28.0, -4.9), (9.8, 9.8), "Emergency/service bay with carts and staff flow."),
    HospitalSemanticZone("ramp_connector", "ramp", (43.4, 14.0), (30.8, 3.6), "Longer ramp/service connector toward the far side of the floor."),
    HospitalSemanticZone("doorway", "door", (38.5, 17.1), (5.6, 6.3), "Room entry/exit crossing point on the connector."),
    HospitalSemanticZone("imaging_check", "clinical", (36.5, 11.1), (5.9, 5.9), "Imaging/check bay attached to the connector."),
    HospitalSemanticZone("ramp_landing", "ramp", (45.9, 11.3), (5.0, 5.3), "Ramp or rehab landing area beside the connector."),
    HospitalSemanticZone("pharmacy_service", "service", (53.8, 11.1), (6.7, 5.9), "Pharmacy/meds service bay with staff and equipment flow."),
    HospitalSemanticZone("elevator_exit", "elevator", (55.0, 16.7), (6.3, 5.3), "Elevator/exit alcove near the final goal side."),
)


__all__ = [
    "HospitalLayoutTemplate",
    "HospitalSemanticZone",
    "HOSPITAL_LAYOUT_TEMPLATES",
    "HOSPITAL_FLOOR_ZONES",
]
