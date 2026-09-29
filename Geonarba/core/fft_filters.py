"""Frequency-domain filters for potential-field grids."""

from __future__ import annotations

import numpy as np

from .fourier import pad_values as _pad_values, wavenumber_grid as _wavenumber_grid
from .grid_tools import GridData, validate_fft_ready


def compute_upward_continuation(
    grid: GridData,
    height: float,
    padding: str = "reflect",
    remove_mean: bool = False,
) -> GridData:
    """Compute upward continuation: IFFT(FFT(T) * exp(-|k| * height))."""

    if height < 0 or not np.isfinite(height):
        raise ValueError("Upward continuation height must be a non-negative finite number.")
    if padding not in {"none", "zero", "reflect"}:
        raise ValueError("padding must be one of: none, zero, reflect.")

    ready = validate_fft_ready(grid)
    values = ready.values.copy()
    mean_value = float(np.nanmean(values)) if remove_mean else 0.0
    values = values - mean_value
    padded, crop = _pad_values(values, padding)
    k = _wavenumber_grid(padded.shape, ready.dx, ready.dy)
    response = np.exp(-k * height)
    continued = np.real(np.fft.ifft2(np.fft.fft2(padded) * response))[crop] + mean_value
    return ready.with_values(
        continued,
        name=f"{grid.name}_UC_{height:g}m",
        units=grid.units,
        metadata={
            "method": "upward_continuation",
            "height_m": float(height),
            "padding": padding,
            "remove_mean": remove_mean,
            "formula": "UC = IFFT(FFT(T) * exp(-|k| * height))",
            "warning": "Upward continuation smooths anomalies and is not a unique geological separation.",
        },
    )


def compute_residual_from_upward(
    grid: GridData,
    height: float,
    padding: str = "reflect",
    remove_mean: bool = False,
) -> tuple[GridData, GridData]:
    """Return upward-continued regional and residual grids."""

    regional = compute_upward_continuation(grid, height=height, padding=padding, remove_mean=remove_mean)
    residual_values = grid.values - regional.values
    residual = grid.with_values(
        residual_values,
        name=f"{grid.name}_Residual_UC_{height:g}m",
        units=grid.units,
        metadata={
            "method": "residual_from_upward_continuation",
            "height_m": float(height),
            "padding": padding,
            "formula": "Residual = Original - UpwardContinued",
            "regional_layer": regional.name,
        },
    )
    return regional, residual


def _wavelength_to_k(wavelength: float) -> float:
    if wavelength <= 0 or not np.isfinite(wavelength):
        raise ValueError("Wavelength cutoff must be a positive finite number.")
    return 2.0 * np.pi / wavelength


def _lowpass_response(k: np.ndarray, cutoff_k: float, order: int) -> np.ndarray:
    return 1.0 / (1.0 + (k / max(cutoff_k, np.finfo(float).eps)) ** (2 * order))


def _highpass_response(k: np.ndarray, cutoff_k: float, order: int) -> np.ndarray:
    return 1.0 - _lowpass_response(k, cutoff_k, order)


def compute_butterworth_filter(
    grid: GridData,
    cutoff: float | tuple[float, float],
    order: int = 4,
    filter_type: str = "lowpass",
    cutoff_unit: str = "wavelength",
    padding: str = "reflect",
) -> GridData:
    """Compute Butterworth low-pass, high-pass, or band-pass filter."""

    if order < 1:
        raise ValueError("Butterworth order must be at least 1.")
    if filter_type not in {"lowpass", "highpass", "bandpass"}:
        raise ValueError("filter_type must be lowpass, highpass, or bandpass.")
    if cutoff_unit != "wavelength":
        raise ValueError("Only wavelength cutoffs in meters are supported in V1.")

    ready = validate_fft_ready(grid)
    values, crop = _pad_values(ready.values, padding)
    k = _wavenumber_grid(values.shape, ready.dx, ready.dy)

    if filter_type == "bandpass":
        if not isinstance(cutoff, tuple) or len(cutoff) != 2:
            raise ValueError("Band-pass cutoff must be a (long_wavelength, short_wavelength) tuple.")
        long_wavelength, short_wavelength = sorted([float(cutoff[0]), float(cutoff[1])], reverse=True)
        k_low = _wavelength_to_k(long_wavelength)
        k_high = _wavelength_to_k(short_wavelength)
        response = _highpass_response(k, k_low, order) * _lowpass_response(k, k_high, order)
        label = f"BP_{long_wavelength:g}-{short_wavelength:g}m"
        cutoff_metadata: float | tuple[float, float] = (long_wavelength, short_wavelength)
    else:
        cutoff_value = float(cutoff)
        cutoff_k = _wavelength_to_k(cutoff_value)
        response = (
            _lowpass_response(k, cutoff_k, order)
            if filter_type == "lowpass"
            else _highpass_response(k, cutoff_k, order)
        )
        label = f"{'LP' if filter_type == 'lowpass' else 'HP'}_{cutoff_value:g}m"
        cutoff_metadata = cutoff_value

    filtered = np.real(np.fft.ifft2(np.fft.fft2(values) * response))[crop]
    return ready.with_values(
        filtered,
        name=f"{grid.name}_Butterworth_{label}",
        units=grid.units,
        metadata={
            "method": "butterworth_filter",
            "filter_type": filter_type,
            "cutoff_wavelength_m": cutoff_metadata,
            "order": order,
            "padding": padding,
            "formula": "Frequency-domain Butterworth radial filter using angular wavenumber.",
            "warning": "Regional/residual separation depends on cutoff wavelength and filter order.",
        },
    )


def compute_butterworth_regional_residual(
    grid: GridData,
    cutoff_wavelength: float,
    order: int = 4,
    padding: str = "reflect",
) -> tuple[GridData, GridData]:
    """Return low-pass regional and residual anomaly grids."""

    regional = compute_butterworth_filter(
        grid,
        cutoff=cutoff_wavelength,
        order=order,
        filter_type="lowpass",
        padding=padding,
    )
    residual = grid.with_values(
        grid.values - regional.values,
        name=f"{grid.name}_Residual_BW_{cutoff_wavelength:g}m",
        units=grid.units,
        metadata={
            "method": "butterworth_regional_residual",
            "cutoff_wavelength_m": float(cutoff_wavelength),
            "order": order,
            "padding": padding,
            "formula": "Residual = Original - ButterworthLowPassRegional",
            "regional_layer": regional.name,
        },
    )
    return regional, residual
