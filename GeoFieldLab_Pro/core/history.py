"""Processing history helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .grid_tools import GridData, grid_statistics


SOFTWARE_VERSION = "GEONARBA V1"


def make_history_entry(
    method_name: str,
    input_layer: str,
    output_layer: str,
    parameters: dict[str, Any] | None = None,
    result: GridData | None = None,
    warnings: list[str] | None = None,
    computation_time_s: float | None = None,
    formula: str | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method_name": method_name,
        "input_layer": input_layer,
        "output_layer": output_layer,
        "parameters": parameters or {},
        "statistics": grid_statistics(result) if result is not None else {},
        "warnings": warnings or [],
        "computation_time_s": computation_time_s,
        "formula": formula or (result.metadata.get("formula") if result is not None else None),
        "software_version": SOFTWARE_VERSION,
    }


def add_history(history: list[dict[str, Any]], entry: dict[str, Any]) -> None:
    history.append(entry)
