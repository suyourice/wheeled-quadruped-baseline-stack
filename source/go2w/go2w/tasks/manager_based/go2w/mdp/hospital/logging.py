# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Logging helpers for reproducible hospital scenario runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class HospitalEpisodeManifest:
    """Minimal replayable description of a sampled hospital episode."""

    episode_seed: int
    template_name: str
    layout_name: str
    scenario_name: str
    dynamic_density: float
    label_summary: dict[str, int]
    relation_summary: dict[str, int]
    notes: str = ""


@dataclass(frozen=True)
class HospitalEventRecord:
    """Single event entry for debug/event logs."""

    step: int
    env_id: int
    event_type: str
    label: str
    detail: str


def write_jsonl(path: str | Path, records: list[dict | HospitalEpisodeManifest | HospitalEventRecord]) -> None:
    """Write dictionaries or dataclass records as a JSONL file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            if hasattr(record, "__dataclass_fields__"):
                payload = asdict(record)
            else:
                payload = record
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


__all__ = ["HospitalEpisodeManifest", "HospitalEventRecord", "write_jsonl"]
