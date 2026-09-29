"""Radially averaged power spectrum (RAPS) analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .grid_tools import GridData, grid_extent, validate_fft_ready


@dataclass
class SegmentFit:
    k_min: float
    k_max: float
    slope: float
    intercept: float
    r_squared: float
    depth_m: float
    depth_km: float
    formula: str = "depth = -slope / 2 for ln(power) vs angular wavenumber k (rad/m)"


@dataclass
class RAPSResult:
    k: np.ndarray
    power: np.ndarray
    log_power: np.ndarray
    bin_counts: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    fit: SegmentFit | None = None


def _detrend_plane(values: np.ndarray) -> np.ndarray:
    ny, nx = values.shape
    yy, xx = np.indices(values.shape)
    valid = np.isfinite(values)
    design = np.column_stack([xx[valid], yy[valid], np.ones(valid.sum())])
    coeffs, *_ = np.linalg.lstsq(design, values[valid], rcond=None)
    trend = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
    return values - trend


def _cosine_taper(n: int, fraction: float = 0.1) -> np.ndarray:
    if n <= 1:
        return np.ones(n)
    edge = max(int(round(n * fraction)), 1)
    weights = np.ones(n)
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, edge)))
    weights[:edge] = ramp
    weights[-edge:] = ramp[::-1]
    return weights


def _window_2d(shape: tuple[int, int], window: str) -> np.ndarray:
    ny, nx = shape
    if window == "none":
        return np.ones(shape)
    if window == "hanning":
        return np.outer(np.hanning(ny), np.hanning(nx))
    if window == "cosine":
        return np.outer(_cosine_taper(ny), _cosine_taper(nx))
    raise ValueError("window must be none, hanning, or cosine.")


def _zero_pad(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return values
    if mode != "zero":
        raise ValueError("padding must be none or zero for RAPS.")
    return np.pad(values, ((0, values.shape[0]), (0, values.shape[1])), mode="constant")


def _wavenumber_components(shape: tuple[int, int], dx: float, dy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k = np.sqrt(kx_grid**2 + ky_grid**2)
    return kx_grid, ky_grid, k


def compute_raps(
    grid: GridData,
    remove_mean: bool = True,
    detrend: bool = False,
    window: str = "hanning",
    padding: str = "zero",
    radial_bin_size: float | None = None,
    log_power: bool = True,
) -> RAPSResult:
    """Compute radial average power spectrum using angular wavenumber k in rad/m."""

    ready = validate_fft_ready(grid)
    values = ready.values.astype(float).copy()
    if remove_mean:
        values -= float(np.nanmean(values))
    if detrend:
        values = _detrend_plane(values)
    values *= _window_2d(values.shape, window)
    values = _zero_pad(values, padding)

    _, _, k = _wavenumber_components(values.shape, ready.dx, ready.dy)
    fft_values = np.fft.fft2(values)
    power = np.abs(fft_values) ** 2

    k_flat = k.ravel()
    power_flat = power.ravel()
    valid = np.isfinite(k_flat) & np.isfinite(power_flat) & (k_flat > 0)
    k_flat = k_flat[valid]
    power_flat = power_flat[valid]
    if k_flat.size < 4:
        raise ValueError("Grid is too small for RAPS analysis.")

    if radial_bin_size is None or radial_bin_size <= 0:
        nonzero = np.unique(np.round(k_flat[k_flat > 0], decimals=15))
        radial_bin_size = float(np.nanmedian(np.diff(nonzero))) if nonzero.size > 2 else float(np.nanmax(k_flat) / 40)
    edges = np.arange(0.0, float(np.nanmax(k_flat)) + radial_bin_size, radial_bin_size)
    if edges.size < 3:
        edges = np.linspace(0.0, float(np.nanmax(k_flat)), 32)

    bin_index = np.digitize(k_flat, edges) - 1
    centers: list[float] = []
    avg_power: list[float] = []
    counts: list[int] = []
    for idx in range(len(edges) - 1):
        mask = bin_index == idx
        if not mask.any():
            continue
        centers.append(float(0.5 * (edges[idx] + edges[idx + 1])))
        avg_power.append(float(np.nanmean(power_flat[mask])))
        counts.append(int(mask.sum()))

    k_radial = np.asarray(centers, dtype=float)
    power_radial = np.asarray(avg_power, dtype=float)
    counts_arr = np.asarray(counts, dtype=int)
    if log_power:
        log_radial = np.log(np.maximum(power_radial, np.finfo(float).tiny))
    else:
        log_radial = power_radial.copy()

    extent = grid_extent(grid)
    metadata = {
        "method": "raps",
        "remove_mean": remove_mean,
        "detrend": detrend,
        "window": window,
        "padding": padding,
        "radial_bin_size": float(radial_bin_size),
        "wavenumber_unit": "rad/m",
        "spectrum_convention": "natural log of radially averaged power vs angular radial wavenumber",
        "depth_formula": "depth = -slope / 2",
        "study_width_m": min(extent["width"], extent["height"]),
        "warning": (
            "RAPS is sensitive to windowing, padding, grid size, regional trends, and selected slope segment. "
            "Small areas such as 50 km x 50 km cannot reliably resolve very deep sources."
        ),
    }
    return RAPSResult(k=k_radial, power=power_radial, log_power=log_radial, bin_counts=counts_arr, metadata=metadata)


def fit_raps_segment(
    result: RAPSResult,
    k_min: float | None = None,
    k_max: float | None = None,
    automatic: bool = True,
) -> SegmentFit:
    """Fit a linear segment and estimate depth from slope."""

    k = np.asarray(result.k, dtype=float)
    y = np.asarray(result.log_power, dtype=float)
    valid = np.isfinite(k) & np.isfinite(y) & (k > 0)
    k_valid = k[valid]
    y_valid = y[valid]
    if k_valid.size < 4:
        raise ValueError("Not enough valid RAPS bins for segment fitting.")

    if automatic:
        start = max(int(k_valid.size * 0.15), 0)
        stop = min(max(int(k_valid.size * 0.65), start + 4), k_valid.size)
        mask = np.zeros_like(k_valid, dtype=bool)
        mask[start:stop] = True
    else:
        if k_min is None or k_max is None:
            raise ValueError("Manual segment fitting requires k_min and k_max.")
        low, high = sorted([float(k_min), float(k_max)])
        mask = (k_valid >= low) & (k_valid <= high)
    if mask.sum() < 3:
        raise ValueError("Selected RAPS segment must contain at least three bins.")

    slope, intercept = np.polyfit(k_valid[mask], y_valid[mask], deg=1)
    fitted = slope * k_valid[mask] + intercept
    residual = y_valid[mask] - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_valid[mask] - np.mean(y_valid[mask])) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    depth = -float(slope) / 2.0
    fit = SegmentFit(
        k_min=float(np.min(k_valid[mask])),
        k_max=float(np.max(k_valid[mask])),
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        depth_m=float(depth),
        depth_km=float(depth / 1000.0),
    )
    result.fit = fit
    return fit
