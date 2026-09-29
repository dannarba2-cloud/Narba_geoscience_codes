"""Profile operations for desktop GEONARBA workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from .grid_tools import GridData


@dataclass
class ProfileData:
    """A 1D anomaly profile with distance coordinates and metadata."""

    distance: np.ndarray
    values: np.ndarray
    name: str = "Profile"
    units: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.distance = np.asarray(self.distance, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        if self.distance.ndim != 1 or self.values.ndim != 1:
            raise ValueError("Profile distance and values must be 1D arrays.")
        if self.distance.size != self.values.size:
            raise ValueError("Profile distance and values must have the same length.")
        if self.distance.size < 2:
            raise ValueError("A profile requires at least two samples.")

    @property
    def dx(self) -> float:
        diffs = np.diff(np.sort(np.unique(self.distance[np.isfinite(self.distance)])))
        diffs = diffs[diffs > 0]
        return float(np.nanmedian(diffs)) if diffs.size else np.nan

    def with_values(
        self,
        distance: np.ndarray,
        values: np.ndarray,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> "ProfileData":
        merged = dict(self.metadata)
        if metadata:
            merged.update(metadata)
        return ProfileData(distance=distance, values=values, name=name, units=self.units, metadata=merged)

    def to_dataframe(self, value_column: str = "value") -> pd.DataFrame:
        return pd.DataFrame(
            {
                "distance": self.distance,
                value_column: self.values,
                "profile_name": self.name,
                "units": self.units,
            }
        )


def profile_statistics(profile: ProfileData | np.ndarray) -> dict[str, float | int]:
    """Return finite-value statistics for a profile."""

    values = profile.values if isinstance(profile, ProfileData) else np.asarray(profile, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "rms": np.nan,
            "finite_count": 0,
            "nan_count": int(np.isnan(values).sum()),
        }
    return {
        "min": float(np.nanmin(finite)),
        "max": float(np.nanmax(finite)),
        "mean": float(np.nanmean(finite)),
        "std": float(np.nanstd(finite)),
        "median": float(np.nanmedian(finite)),
        "rms": float(np.sqrt(np.nanmean(finite**2))),
        "finite_count": int(finite.size),
        "nan_count": int(np.isnan(values).sum()),
    }


def extract_cross_section(
    grid: GridData,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    spacing: float | None = None,
    name: str | None = None,
) -> ProfileData:
    """Sample a grid along a straight x/y line."""

    start = np.asarray(start_xy, dtype=float)
    end = np.asarray(end_xy, dtype=float)
    if start.shape != (2,) or end.shape != (2,):
        raise ValueError("start_xy and end_xy must be (x, y) pairs.")
    length = float(np.linalg.norm(end - start))
    if length <= 0 or not np.isfinite(length):
        raise ValueError("Cross-section length must be positive.")
    spacing = min(grid.dx, grid.dy) if spacing is None else float(spacing)
    if spacing <= 0 or not np.isfinite(spacing):
        raise ValueError("Cross-section spacing must be a positive finite number.")

    sample_count = max(int(np.ceil(length / spacing)) + 1, 2)
    distance = np.linspace(0.0, length, sample_count)
    fraction = distance / length
    xs = start[0] + (end[0] - start[0]) * fraction
    ys = start[1] + (end[1] - start[1]) * fraction

    x_order = np.argsort(grid.x_vector)
    y_order = np.argsort(grid.y_vector)
    values = grid.values[np.ix_(y_order, x_order)]
    interpolator = RegularGridInterpolator(
        (grid.y_vector[y_order], grid.x_vector[x_order]),
        values,
        bounds_error=False,
        fill_value=np.nan,
    )
    sampled = interpolator(np.column_stack([ys, xs]))
    return ProfileData(
        distance=distance,
        values=sampled,
        name=name or f"{grid.name}_cross_section",
        units=grid.units,
        metadata={
            "method": "extract_cross_section",
            "source_layer": grid.name,
            "start_xy": tuple(float(v) for v in start),
            "end_xy": tuple(float(v) for v in end),
            "length": length,
            "spacing": spacing,
            "sample_count": sample_count,
        },
    )


def crop_profile(profile: ProfileData, distance_min: float, distance_max: float, name: str | None = None) -> ProfileData:
    """Crop a profile by distance bounds."""

    low, high = sorted([float(distance_min), float(distance_max)])
    mask = (profile.distance >= low) & (profile.distance <= high)
    if not mask.any():
        raise ValueError("Profile crop bounds do not overlap the profile.")
    return profile.with_values(
        profile.distance[mask].copy(),
        profile.values[mask].copy(),
        name or f"{profile.name}_crop",
        {"method": "crop_profile", "bounds": {"min": low, "max": high}},
    )


def flip_profile(profile: ProfileData, name: str | None = None) -> ProfileData:
    """Reverse profile sample order while keeping distance increasing from zero."""

    distance = profile.distance - float(np.nanmin(profile.distance))
    distance = float(np.nanmax(distance)) - distance[::-1]
    return profile.with_values(
        distance,
        profile.values[::-1].copy(),
        name or f"{profile.name}_flip",
        {"method": "flip_profile"},
    )


def interpolate_profile(profile: ProfileData, spacing: float, name: str | None = None) -> ProfileData:
    """Resample a profile to regular distance spacing."""

    if spacing <= 0 or not np.isfinite(spacing):
        raise ValueError("Profile spacing must be a positive finite number.")
    order = np.argsort(profile.distance)
    distance = profile.distance[order]
    values = profile.values[order]
    valid = np.isfinite(distance) & np.isfinite(values)
    if valid.sum() < 2:
        raise ValueError("At least two finite profile samples are required for interpolation.")
    start = float(np.nanmin(distance[valid]))
    stop = float(np.nanmax(distance[valid]))
    new_distance = np.arange(start, stop + spacing * 0.5, spacing, dtype=float)
    if new_distance[-1] < stop:
        new_distance = np.append(new_distance, stop)
    new_values = np.interp(new_distance, distance[valid], values[valid])
    return profile.with_values(
        new_distance,
        new_values,
        name or f"{profile.name}_interp_{spacing:g}",
        {"method": "interpolate_profile", "spacing": float(spacing)},
    )


def running_average_profile(profile: ProfileData, window_size: int, name: str | None = None) -> ProfileData:
    """Smooth a profile with a centered running average."""

    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    if window_size == 1:
        return profile.with_values(profile.distance.copy(), profile.values.copy(), name or f"{profile.name}_avg1")
    values = profile.values.astype(float)
    valid = np.isfinite(values)
    if not valid.any():
        raise ValueError("Profile contains no finite values.")
    filled = values.copy()
    if not valid.all():
        filled[~valid] = np.interp(profile.distance[~valid], profile.distance[valid], values[valid])
    left = window_size // 2
    right = window_size - left - 1
    padded = np.pad(filled, (left, right), mode="edge")
    kernel = np.ones(window_size, dtype=float) / float(window_size)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return profile.with_values(
        profile.distance.copy(),
        smoothed,
        name or f"{profile.name}_avg{window_size}",
        {"method": "running_average_profile", "window_size": int(window_size)},
    )


def remove_linear_trend_profile(profile: ProfileData, name: str | None = None) -> ProfileData:
    """Remove the best-fit linear trend from profile values."""

    valid = np.isfinite(profile.distance) & np.isfinite(profile.values)
    if valid.sum() < 2:
        raise ValueError("At least two finite samples are required to remove a trend.")
    slope, intercept = np.polyfit(profile.distance[valid], profile.values[valid], deg=1)
    trend = slope * profile.distance + intercept
    return profile.with_values(
        profile.distance.copy(),
        profile.values - trend,
        name or f"{profile.name}_detrended",
        {"method": "remove_linear_trend_profile", "slope": float(slope), "intercept": float(intercept)},
    )


def polynomial_fit_profile(profile: ProfileData, degree: int, name: str | None = None) -> tuple[ProfileData, ProfileData]:
    """Return fitted polynomial trend and residual profiles."""

    if degree < 0:
        raise ValueError("degree must be non-negative.")
    valid = np.isfinite(profile.distance) & np.isfinite(profile.values)
    if valid.sum() <= degree:
        raise ValueError("Not enough finite samples for the requested polynomial degree.")
    coeffs = np.polyfit(profile.distance[valid], profile.values[valid], deg=degree)
    fitted_values = np.polyval(coeffs, profile.distance)
    fit = profile.with_values(
        profile.distance.copy(),
        fitted_values,
        name or f"{profile.name}_poly{degree}",
        {"method": "polynomial_fit_profile", "degree": int(degree), "coefficients": coeffs.tolist()},
    )
    residual = profile.with_values(
        profile.distance.copy(),
        profile.values - fitted_values,
        f"{profile.name}_poly{degree}_residual",
        {"method": "polynomial_residual_profile", "degree": int(degree), "trend_layer": fit.name},
    )
    return fit, residual
