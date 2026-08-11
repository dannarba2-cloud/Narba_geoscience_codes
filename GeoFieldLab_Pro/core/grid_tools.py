"""Grid handling utilities for GEONARBA.

The scientific routines in the app exchange data through the GridData
dataclass. Coordinates are stored as x/y vectors or x/y mesh arrays, and all
methods preserve spacing, units, CRS, and metadata where possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import griddata


@dataclass
class GridData:
    """Regular anomaly grid with metadata preserved through processing."""

    values: np.ndarray
    x: np.ndarray
    y: np.ndarray
    dx: float
    dy: float
    name: str = "Layer"
    units: str = "unknown"
    crs: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        self.dx = float(self.dx)
        self.dy = float(self.dy)
        if self.values.ndim != 2:
            raise ValueError("GridData.values must be a 2D array.")
        if not np.isfinite(self.dx) or self.dx <= 0:
            raise ValueError("Grid spacing dx must be a positive finite value.")
        if not np.isfinite(self.dy) or self.dy <= 0:
            raise ValueError("Grid spacing dy must be a positive finite value.")

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    @property
    def x_vector(self) -> np.ndarray:
        if self.x.ndim == 1:
            return self.x
        return self.x[0, :]

    @property
    def y_vector(self) -> np.ndarray:
        if self.y.ndim == 1:
            return self.y
        return self.y[:, 0]

    @property
    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        if self.x.ndim == 2 and self.y.ndim == 2:
            return self.x, self.y
        return np.meshgrid(self.x_vector, self.y_vector)

    def with_values(
        self,
        values: np.ndarray,
        name: str,
        units: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "GridData":
        merged_metadata = dict(self.metadata)
        if metadata:
            merged_metadata.update(metadata)
        return replace(
            self,
            values=np.asarray(values, dtype=float),
            name=name,
            units=self.units if units is None else units,
            metadata=merged_metadata,
        )

    def statistics(self) -> dict[str, float]:
        return grid_statistics(self)

    def to_dataframe(self, value_column: str = "value") -> pd.DataFrame:
        xx, yy = self.mesh
        return pd.DataFrame(
            {
                "x": xx.ravel(),
                "y": yy.ravel(),
                value_column: self.values.ravel(),
                "layer_name": self.name,
                "units": self.units,
            }
        )


def _finite_unique(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return np.unique(values)


def estimate_grid_spacing(values: np.ndarray) -> float:
    """Estimate representative spacing from sorted unique coordinates."""

    unique = _finite_unique(values)
    if unique.size < 2:
        raise ValueError("At least two unique coordinates are required.")
    diffs = np.diff(np.sort(unique))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        raise ValueError("Coordinate spacing could not be estimated.")
    return float(np.median(diffs))


def _is_regular_spacing(unique: np.ndarray, tolerance: float = 1e-6) -> bool:
    if unique.size < 2:
        return False
    diffs = np.diff(np.sort(unique))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return False
    scale = max(float(np.nanmedian(np.abs(diffs))), 1.0)
    return bool(np.allclose(diffs, np.median(diffs), rtol=tolerance, atol=tolerance * scale))


def detect_regular_grid(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Detect whether point data form a complete regular grid."""

    required = data[[x_col, y_col, value_col]].dropna()
    duplicate_count = int(required.duplicated(subset=[x_col, y_col]).sum())
    unique_points = required.drop_duplicates(subset=[x_col, y_col])
    xs = np.sort(_finite_unique(unique_points[x_col].to_numpy()))
    ys = np.sort(_finite_unique(unique_points[y_col].to_numpy()))
    expected = int(xs.size * ys.size)
    actual = int(len(unique_points))
    regular_spacing = _is_regular_spacing(xs, tolerance) and _is_regular_spacing(ys, tolerance)
    complete = expected == actual and expected > 0
    return {
        "is_regular": bool(regular_spacing and complete and duplicate_count == 0),
        "regular_spacing": bool(regular_spacing),
        "complete_grid": bool(complete),
        "duplicate_count": duplicate_count,
        "missing_cell_count": max(expected - actual, 0),
        "nx": int(xs.size),
        "ny": int(ys.size),
        "point_count": int(len(required)),
        "unique_point_count": actual,
        "dx": estimate_grid_spacing(xs) if xs.size > 1 else np.nan,
        "dy": estimate_grid_spacing(ys) if ys.size > 1 else np.nan,
    }


def reshape_regular_grid(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    name: str = "Original anomaly",
    units: str = "unknown",
    crs: str | None = None,
) -> GridData:
    """Reshape complete regular x/y/value points into GridData."""

    clean = data[[x_col, y_col, value_col]].dropna().copy()
    if clean.duplicated(subset=[x_col, y_col]).any():
        raise ValueError("Regular grid contains duplicate x/y coordinates.")
    pivot = clean.pivot(index=y_col, columns=x_col, values=value_col).sort_index().sort_index(axis=1)
    if pivot.isna().any().any():
        raise ValueError("Regular grid is incomplete; interpolation is required.")
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    dx = estimate_grid_spacing(x)
    dy = estimate_grid_spacing(y)
    return GridData(
        values=pivot.to_numpy(dtype=float),
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        name=name,
        units=units,
        crs=crs,
        metadata={"source": "regular_grid", "x_col": x_col, "y_col": y_col, "value_col": value_col},
    )


def interpolate_scattered_to_grid(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    spacing: float,
    method: str = "linear",
    fill_method: str = "nearest",
    name: str = "Interpolated anomaly",
    units: str = "unknown",
    crs: str | None = None,
) -> GridData:
    """Interpolate scattered points to a regular grid with scipy.griddata."""

    if spacing <= 0 or not np.isfinite(spacing):
        raise ValueError("Grid spacing must be a positive finite number.")
    clean = data[[x_col, y_col, value_col]].dropna().copy()
    clean = clean.drop_duplicates(subset=[x_col, y_col], keep="last")
    if len(clean) < 3:
        raise ValueError("At least three valid points are required for interpolation.")

    points = clean[[x_col, y_col]].to_numpy(dtype=float)
    values = clean[value_col].to_numpy(dtype=float)
    x_min, x_max = float(np.min(points[:, 0])), float(np.max(points[:, 0]))
    y_min, y_max = float(np.min(points[:, 1])), float(np.max(points[:, 1]))
    x = np.arange(x_min, x_max + spacing * 0.5, spacing, dtype=float)
    y = np.arange(y_min, y_max + spacing * 0.5, spacing, dtype=float)
    xx, yy = np.meshgrid(x, y)

    chosen_method = method if method in {"linear", "cubic", "nearest"} else "linear"
    grid = griddata(points, values, (xx, yy), method=chosen_method)
    if np.isnan(grid).any():
        grid = fill_nan_grid(grid, x, y, method=fill_method)

    return GridData(
        values=grid,
        x=x,
        y=y,
        dx=float(spacing),
        dy=float(spacing),
        name=name,
        units=units,
        crs=crs,
        metadata={
            "source": "scattered_interpolation",
            "interpolation_method": chosen_method,
            "fill_method": fill_method,
            "x_col": x_col,
            "y_col": y_col,
            "value_col": value_col,
        },
    )


def fill_nan_grid(
    values: np.ndarray,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    method: str = "nearest",
    constant: float = 0.0,
) -> np.ndarray:
    """Fill NaNs in a regular grid for FFT-safe computations."""

    arr = np.asarray(values, dtype=float).copy()
    if not np.isnan(arr).any():
        return arr
    if method == "reject":
        raise ValueError("Grid contains NaN values. Fill or mask NaNs before computation.")
    if method == "constant":
        arr[np.isnan(arr)] = constant
        return arr
    if method != "nearest":
        raise ValueError(f"Unsupported NaN fill method: {method}")

    ny, nx = arr.shape
    if x is None:
        x = np.arange(nx, dtype=float)
    if y is None:
        y = np.arange(ny, dtype=float)
    xx, yy = np.meshgrid(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    valid = np.isfinite(arr)
    if not valid.any():
        raise ValueError("Grid contains no finite values.")
    arr[~valid] = griddata(
        (xx[valid], yy[valid]),
        arr[valid],
        (xx[~valid], yy[~valid]),
        method="nearest",
    )
    return arr


def coordinate_warning(x: np.ndarray, y: np.ndarray) -> str | None:
    """Return a warning when coordinates appear to be longitude/latitude."""

    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.ndim == 2:
        x_values = x_values.ravel()
    if y_values.ndim == 2:
        y_values = y_values.ravel()
    x_values = x_values[np.isfinite(x_values)]
    y_values = y_values[np.isfinite(y_values)]
    if x_values.size == 0 or y_values.size == 0:
        return None
    if (
        np.nanmin(x_values) >= -180
        and np.nanmax(x_values) <= 180
        and np.nanmin(y_values) >= -90
        and np.nanmax(y_values) <= 90
    ):
        return (
            "Coordinates appear to be longitude/latitude. FFT derivatives, RAPS, "
            "upward continuation, and wavelength filters require projected "
            "coordinates in meters. Reproject to UTM before scientific interpretation."
        )
    return None


def grid_extent(grid: GridData) -> dict[str, float]:
    x = grid.x_vector
    y = grid.y_vector
    return {
        "xmin": float(np.nanmin(x)),
        "xmax": float(np.nanmax(x)),
        "ymin": float(np.nanmin(y)),
        "ymax": float(np.nanmax(y)),
        "width": float(np.nanmax(x) - np.nanmin(x)),
        "height": float(np.nanmax(y) - np.nanmin(y)),
    }


def grid_statistics(grid: GridData | np.ndarray) -> dict[str, float]:
    values = grid.values if isinstance(grid, GridData) else np.asarray(grid, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}
    return {
        "min": float(np.nanmin(finite)),
        "max": float(np.nanmax(finite)),
        "mean": float(np.nanmean(finite)),
        "std": float(np.nanstd(finite)),
    }


def validate_fft_ready(grid: GridData, fill_method: str = "nearest") -> GridData:
    """Return a GridData copy with finite values suitable for FFT methods."""

    if np.isfinite(grid.values).all():
        return grid
    filled = fill_nan_grid(grid.values, grid.x_vector, grid.y_vector, method=fill_method)
    metadata = dict(grid.metadata)
    metadata["nan_fill_method"] = fill_method
    return grid.with_values(filled, name=grid.name, units=grid.units, metadata=metadata)
