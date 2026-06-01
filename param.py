from __future__ import annotations

from typing import Any

MODELS = [
    "PDT",
    "PDDT",
    "SDT",
    "SDDT",
    "OODDT",
    "OOPDT",
    "OOPDDT",
    "MSEQCT",
]

SCARCITIES = [round(0.1 * s, 1) for s in range(2, 11)]

DEFAULT_PARAMS: dict[str, Any] = {
    "number_of_processes": 2,
    "arrival_rate": 0.3,
    "batch_size": 2.0,
    "max_phases": 3,
    "min_base_duration": 2.0,
    "max_base_duration": 5.0,
    "min_resource_ratio": 0.6,
    "resource_ratio_center": 1.5,
    "resource_ratio_spread": 1.0,
    "timeout": 600,
    "seed": None,
}

ARG_TYPES: dict[str, type] = {
    "batch_size": float,
    "seed": int,
}
