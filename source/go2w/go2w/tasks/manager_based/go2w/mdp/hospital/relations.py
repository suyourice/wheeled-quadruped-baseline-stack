# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Group relationship rules for hospital scenarios."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HospitalRelationSpec:
    """Declarative description of a coupled group behavior."""

    relation_type: str
    leader_label: str
    follower_label: str
    desired_offset_xy: tuple[float, float]
    max_separation: float
    rejoin_distance: float
    priority_boost: float = 0.0
    notes: str = ""


HOSPITAL_RELATION_SPECS: dict[str, HospitalRelationSpec] = {
    "guardian_child": HospitalRelationSpec("guardian_child", "adult", "child", (-0.6, 0.0), 1.5, 1.0, 0.6, "Child can break away briefly, then rejoin the adult guardian."),
    "handler_dog": HospitalRelationSpec("handler_dog", "visitor", "dog", (0.70, 0.0), 0.95, 0.65, 0.2, "Dog is always leashed and stays just ahead of the handler inside the corridor."),
    "pusher_payload": HospitalRelationSpec("pusher_payload", "staff", "cart", (0.8, 0.0), 1.0, 0.8, 0.3, "Staff stays behind the short side while the cart/gurney/wheelchair payload remains ahead."),
    "pusher_gurney": HospitalRelationSpec("pusher_gurney", "staff", "gurney_patient", (1.3, 0.0), 1.5, 1.3, 0.3, "Staff behind a long gurney; larger offset keeps the staff clear of the 1.95 m gurney body."),
    "patient_iv": HospitalRelationSpec("patient_iv", "patient_with_iv", "iv_pole", (-0.45, 0.25), 0.9, 0.7, 0.4, "Patient walking with an IV pole or tethered equipment."),
    "wheelchair_assisted": HospitalRelationSpec("wheelchair_assisted", "staff", "wheelchair_patient", (0.8, 0.0), 1.0, 0.8, 0.6, "Wheelchair user in front of a pushing assistant."),
}


__all__ = ["HospitalRelationSpec", "HOSPITAL_RELATION_SPECS"]
