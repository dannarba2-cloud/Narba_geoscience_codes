"""Core grid transformations inspired by GMinterp V1 workflows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .grid_tools import GridData, estimate_grid_spacing, grid_statistics, reshape_regular_grid


def extended_grid_statistics(grid: GridData | np.ndarray) -> dict[str, float | int]:
    """Return professional QA statistics for a grid or raw matrix."""

    values = grid.values if isinstance(grid, GridData) else np.asarray(grid, dtype=float)
    finite = values[np.isfinite(values)]
    base = grid_statistics(values)
    if finite.size == 0:
        return {
            **base,
            "median": np.nan,
            "rms": np.nan,
            "finite_count": 0,
            "nan_count": int(np.isnan(values).sum()),
        }
    return {
        **base,
        "median": float(np.nanmedian(finite)),
        "rms": float(np.sqrt(np.nanmean(finite**2))),
        "finite_count": int(finite.size),
        "nan_count": int(np.isnan(values).sum()),
    }


def matrix_to_xyz(
    matrix: np.ndarray,
    dx: float,
    dy: float | None = None,
    x0: float = 0.0,
    y0: float = 0.0,
    value_column: str = "value",
) -> pd.DataFrame:
    """Convert a regular matrix to x/y/value point rows."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be a 2D array.")
    if dx <= 0 or not np.isfinite(dx):
        raise ValueError("dx must be a positive finite number.")
    dy = dx if dy is None else float(dy)
    if dy <= 0 or not np.isfinite(dy):
        raise ValueError("dy must be a positive finite number.")

    ny, nx = values.shape
    x = x0 + np.arange(nx, dtype=float) * dx
    y = y0 + np.arange(ny, dtype=float) * dy
    xx, yy = np.meshgrid(x, y)
    return pd.DataFrame({"x": xx.ravel(), "y": yy.ravel(), value_column: values.ravel()})


def xyz_to_grid(
    data: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    value_col: str = "value",
    name: str = "XYZ grid",
    units: str = "unknown",
    crs: str | None = None,
) -> GridData:
    """Convert complete regular x/y/value rows into a GridData object."""

    return reshape_regular_grid(data, x_col, y_col, value_col, name=name, units=units, crs=crs)


def grid_to_matrix(grid: GridData) -> np.ndarray:
    """Return a defensive matrix copy for external export or editing."""

    return np.asarray(grid.values, dtype=float).copy()


def crop_grid(
    grid: GridData,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    name: str | None = None,
) -> GridData:
    """Crop a grid by coordinate bounds."""

    x_low, x_high = sorted([float(xmin), float(xmax)])
    y_low, y_high = sorted([float(ymin), float(ymax)])
    x = grid.x_vector
    y = grid.y_vector
    x_mask = (x >= x_low) & (x <= x_high)
    y_mask = (y >= y_low) & (y <= y_high)
    if not x_mask.any() or not y_mask.any():
        raise ValueError("Crop bounds do not overlap the grid.")
    cropped_values = grid.values[np.ix_(y_mask, x_mask)]
    cropped_x = x[x_mask].copy()
    cropped_y = y[y_mask].copy()
    metadata = {
        **grid.metadata,
        "method": "crop_grid",
        "bounds": {"xmin": x_low, "xmax": x_high, "ymin": y_low, "ymax": y_high},
    }
    return GridData(
        values=cropped_values,
        x=cropped_x,
        y=cropped_y,
        dx=estimate_grid_spacing(cropped_x) if cropped_x.size > 1 else grid.dx,
        dy=estimate_grid_spacing(cropped_y) if cropped_y.size > 1 else grid.dy,
        name=name or f"{grid.name}_crop",
        units=grid.units,
        crs=grid.crs,
        metadata=metadata,
    )


def flip_grid(grid: GridData, axis: str, name: str | None = None) -> GridData:
    """Flip grid values without changing the coordinate frame."""

    axis_key = axis.lower().replace("-", "_").replace(" ", "_")
    if axis_key in {"x", "left_right", "horizontal"}:
        values = np.fliplr(grid.values)
    elif axis_key in {"y", "up_down", "vertical"}:
        values = np.flipud(grid.values)
    elif axis_key in {"both", "xy"}:
        values = np.flipud(np.fliplr(grid.values))
    else:
        raise ValueError("axis must be x, y, or both.")
    return grid.with_values(
        values,
        name=name or f"{grid.name}_flip_{axis_key}",
        metadata={"method": "flip_grid", "axis": axis_key},
    )


def rotate_grid(grid: GridData, degrees_clockwise: int, name: str | None = None) -> GridData:
    """Rotate grid values by 90-degree increments clockwise."""

    if degrees_clockwise % 90 != 0:
        raise ValueError("Only 90-degree increments are supported.")
    normalized = degrees_clockwise % 360
    turns_clockwise = normalized // 90
    values = np.rot90(grid.values, k=-turns_clockwise)
    if normalized in {90, 270}:
        x = np.nanmin(grid.y_vector) + np.arange(values.shape[1], dtype=float) * grid.dy
        y = np.nanmin(grid.x_vector) + np.arange(values.shape[0], dtype=float) * grid.dx
        dx = grid.dy
        dy = grid.dx
    else:
        x = grid.x_vector.copy()
        y = grid.y_vector.copy()
        dx = grid.dx
        dy = grid.dy
    return GridData(
        values=values,
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        name=name or f"{grid.name}_rot{normalized}",
        units=grid.units,
        crs=grid.crs,
        metadata={**grid.metadata, "method": "rotate_grid", "degrees_clockwise": normalized},
    )


def grid_to_serializable(grid: GridData) -> dict[str, Any]:
    """Convert a grid to JSON-serializable project state."""

    return {
        "values": grid.values.tolist(),
        "x": grid.x_vector.tolist(),
        "y": grid.y_vector.tolist(),
        "dx": grid.dx,
        "dy": grid.dy,
        "name": grid.name,
        "units": grid.units,
        "crs": grid.crs,
        "metadata": grid.metadata,
    }


def grid_from_serializable(payload: dict[str, Any]) -> GridData:
    """Restore a grid from JSON project state."""

    return GridData(
        values=np.asarray(payload["values"], dtype=float),
        x=np.asarray(payload["x"], dtype=float),
        y=np.asarray(payload["y"], dtype=float),
        dx=float(payload["dx"]),
        dy=float(payload["dy"]),
        name=str(payload.get("name") or "Layer"),
        units=str(payload.get("units") or "unknown"),
        crs=payload.get("crs"),
        metadata=dict(payload.get("metadata") or {}),
    )
