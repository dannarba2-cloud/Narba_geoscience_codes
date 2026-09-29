"""Desktop project/session model for GeoFieldLab Pro."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.grid_operations import grid_from_serializable, grid_to_serializable
from core.grid_tools import GridData
from core.profile import ProfileData


def _profile_to_serializable(profile: ProfileData) -> dict[str, Any]:
    return {
        "distance": profile.distance.tolist(),
        "values": profile.values.tolist(),
        "name": profile.name,
        "units": profile.units,
        "metadata": profile.metadata,
    }


def _profile_from_serializable(payload: dict[str, Any]) -> ProfileData:
    return ProfileData(
        distance=np.asarray(payload["distance"], dtype=float),
        values=np.asarray(payload["values"], dtype=float),
        name=str(payload.get("name") or "Profile"),
        units=str(payload.get("units") or "unknown"),
        metadata=dict(payload.get("metadata") or {}),
    )


@dataclass
class ProjectSession:
    """Serializable desktop project state."""

    project_name: str = "Untitled GeoFieldLab Project"
    raw_data: pd.DataFrame | None = None
    input_file_name: str | None = None
    layers: "OrderedDict[str, GridData]" = field(default_factory=OrderedDict)
    profiles: "OrderedDict[str, ProfileData]" = field(default_factory=OrderedDict)
    history: list[dict[str, Any]] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    raps_summary: dict[str, Any] | None = None
    recent_files: list[str] = field(default_factory=list)
    dirty: bool = False

    def _unique_name(self, requested: str, existing: set[str]) -> str:
        base = requested.strip() or "Layer"
        if base not in existing:
            return base
        index = 2
        while f"{base} ({index})" in existing:
            index += 1
        return f"{base} ({index})"

    def add_layer(self, grid: GridData, replace_existing: bool = False) -> GridData:
        name = grid.name if replace_existing else self._unique_name(grid.name, set(self.layers))
        stored = replace(grid, name=name) if name != grid.name else grid
        self.layers[name] = stored
        self.dirty = True
        return stored

    def add_profile(self, profile: ProfileData, replace_existing: bool = False) -> ProfileData:
        name = profile.name if replace_existing else self._unique_name(profile.name, set(self.profiles))
        stored = replace(profile, name=name) if name != profile.name else profile
        self.profiles[name] = stored
        self.dirty = True
        return stored

    def get_layer(self, name: str | None = None) -> GridData:
        if not self.layers:
            raise ValueError("No grid layers are available.")
        if name is None:
            return next(reversed(self.layers.values()))
        if name not in self.layers:
            raise KeyError(f"Unknown layer: {name}")
        return self.layers[name]

    def get_profile(self, name: str | None = None) -> ProfileData:
        if not self.profiles:
            raise ValueError("No profiles are available.")
        if name is None:
            return next(reversed(self.profiles.values()))
        if name not in self.profiles:
            raise KeyError(f"Unknown profile: {name}")
        return self.profiles[name]

    def add_history(self, entry: dict[str, Any]) -> None:
        self.history.append(entry)
        self.dirty = True

    def add_recent_file(self, path: str | Path) -> None:
        value = str(path)
        self.recent_files = [item for item in self.recent_files if item != value]
        self.recent_files.insert(0, value)
        self.recent_files = self.recent_files[:10]

    def to_dict(self) -> dict[str, Any]:
        raw_payload = None
        if self.raw_data is not None:
            raw_payload = self.raw_data.to_dict(orient="split")
        return {
            "project_name": self.project_name,
            "raw_data": raw_payload,
            "input_file_name": self.input_file_name,
            "layers": [grid_to_serializable(layer) for layer in self.layers.values()],
            "profiles": [_profile_to_serializable(profile) for profile in self.profiles.values()],
            "history": self.history,
            "settings": self.settings,
            "raps_summary": self.raps_summary,
            "recent_files": self.recent_files,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProjectSession":
        raw_payload = payload.get("raw_data")
        raw_data = None
        if raw_payload:
            raw_data = pd.DataFrame(data=raw_payload["data"], columns=raw_payload["columns"])
        session = cls(
            project_name=str(payload.get("project_name") or "Untitled GeoFieldLab Project"),
            raw_data=raw_data,
            input_file_name=payload.get("input_file_name"),
            history=list(payload.get("history") or []),
            settings=dict(payload.get("settings") or {}),
            raps_summary=payload.get("raps_summary"),
            recent_files=list(payload.get("recent_files") or []),
            dirty=False,
        )
        for layer_payload in payload.get("layers") or []:
            layer = grid_from_serializable(layer_payload)
            session.layers[layer.name] = layer
        for profile_payload in payload.get("profiles") or []:
            profile = _profile_from_serializable(profile_payload)
            session.profiles[profile.name] = profile
        return session

    def save(self, path: str | Path) -> None:
        self.add_recent_file(path)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        self.dirty = False

    @classmethod
    def load(cls, path: str | Path) -> "ProjectSession":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        session = cls.from_dict(payload)
        session.add_recent_file(path)
        session.dirty = False
        return session
