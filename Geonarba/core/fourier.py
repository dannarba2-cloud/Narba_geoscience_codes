"""Shared wavenumber-domain helpers for regular grids.

Convention: angular wavenumbers k = 2*pi/wavelength (rad/m); grid rows are y
(north), columns are x (east). All filters pad, apply a response H(kx, ky, k),
and crop back so that every FFT method in GEONARBA shares one code path.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .grid_tools import GridData, validate_fft_ready

PADDING_MODES = ("reflect", "zero", "none")


def pad_values(values: np.ndarray, padding: str) -> tuple[np.ndarray, tuple[slice, slice]]:
    """Pad by half the grid size on each side (reflect or zero) and return the crop slices."""

    if padding not in PADDING_MODES:
        raise ValueError(f"padding must be one of: {', '.join(PADDING_MODES)}.")
    if padding == "none":
        return values, (slice(None), slice(None))
    pad_y = max(values.shape[0] // 2, 1)
    pad_x = max(values.shape[1] // 2, 1)
    if padding == "reflect":
        padded = np.pad(values, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    else:
        mean = float(np.mean(values))
        padded = np.pad(values - mean, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant") + mean
    return padded, (slice(pad_y, pad_y + values.shape[0]), slice(pad_x, pad_x + values.shape[1]))


def wavenumbers(shape: tuple[int, int], dx: float, dy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (kx, ky, |k|) angular wavenumber meshes in rad/m for an FFT of `shape`."""

    ny, nx = shape
    kx, ky = np.meshgrid(2.0 * np.pi * np.fft.fftfreq(nx, d=dx), 2.0 * np.pi * np.fft.fftfreq(ny, d=dy))
    return kx, ky, np.hypot(kx, ky)


def wavenumber_grid(shape: tuple[int, int], dx: float, dy: float) -> np.ndarray:
    return wavenumbers(shape, dx, dy)[2]


Response = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def apply_response(
    grid: GridData,
    response: Response,
    padding: str = "reflect",
    remove_mean: bool = False,
    complex_output: bool = False,
) -> np.ndarray:
    """Filter a grid with a wavenumber response H(kx, ky, k); returns cropped values."""

    ready = validate_fft_ready(grid)
    values = ready.values.astype(float)
    mean = float(np.mean(values)) if remove_mean else 0.0
    padded, crop = pad_values(values - mean, padding)
    kx, ky, k = wavenumbers(padded.shape, ready.dx, ready.dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.nan_to_num(response(kx, ky, k), nan=0.0, posinf=0.0, neginf=0.0)
    out = np.fft.ifft2(np.fft.fft2(padded) * h)[crop]
    return out if complex_output else np.real(out) + (mean if remove_mean else 0.0)


def direction_cosines(inclination_deg: float, declination_deg: float) -> tuple[float, float, float]:
    """Unit vector (east, north, down) for a field of given inclination/declination."""

    inc, dec = np.radians(inclination_deg), np.radians(declination_deg)
    return float(np.cos(inc) * np.sin(dec)), float(np.cos(inc) * np.cos(dec)), float(np.sin(inc))


def theta_operator(kx: np.ndarray, ky: np.ndarray, k: np.ndarray, inc: float, dec: float) -> np.ndarray:
    """Blakely (1995) directional operator Theta = i*(a*kx + b*ky)/|k| + c for a unit vector (a, b, c).

    With FFT sign convention exp(-i k x), d/dx -> i*kx and d/dz (down) -> |k|, so the
    field component along the unit vector is (i*a*kx + i*b*ky + c*|k|)/|k|.
    """

    a, b, c = direction_cosines(inc, dec)
    safe_k = np.where(k == 0, 1.0, k)
    theta = (1j * (a * kx + b * ky) + c * k) / safe_k
    return np.where(k == 0, c, theta)
