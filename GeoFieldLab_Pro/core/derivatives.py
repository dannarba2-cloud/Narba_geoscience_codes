"""Derivative filters for regular potential-field grids."""

from __future__ import annotations

import numpy as np

from .grid_tools import GridData, validate_fft_ready


def _edge_order(grid: GridData) -> int:
    return 2 if min(grid.values.shape) >= 3 else 1


def _pad_values(values: np.ndarray, padding: str) -> tuple[np.ndarray, tuple[slice, slice]]:
    if padding == "none":
        return values, (slice(None), slice(None))
    pad_y = max(values.shape[0] // 2, 1)
    pad_x = max(values.shape[1] // 2, 1)
    mode = "reflect" if padding == "reflect" else "constant"
    padded = np.pad(values, ((pad_y, pad_y), (pad_x, pad_x)), mode=mode)
    return padded, (slice(pad_y, pad_y + values.shape[0]), slice(pad_x, pad_x + values.shape[1]))


def _wavenumber_grid(shape: tuple[int, int], dx: float, dy: float) -> np.ndarray:
    ny, nx = shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    return np.sqrt(kx_grid**2 + ky_grid**2)


def compute_dx(grid: GridData) -> GridData:
    """Compute horizontal derivative in the x direction."""

    _, dtdx = np.gradient(grid.values, grid.dy, grid.dx, edge_order=_edge_order(grid))
    return grid.with_values(
        dtdx,
        name=f"{grid.name}_dX",
        units=f"{grid.units}/m",
        metadata={"method": "x_derivative", "formula": "dT/dx"},
    )


def compute_dy(grid: GridData) -> GridData:
    """Compute horizontal derivative in the y direction."""

    dtdy, _ = np.gradient(grid.values, grid.dy, grid.dx, edge_order=_edge_order(grid))
    return grid.with_values(
        dtdy,
        name=f"{grid.name}_dY",
        units=f"{grid.units}/m",
        metadata={"method": "y_derivative", "formula": "dT/dy"},
    )


def compute_thg(grid: GridData) -> GridData:
    """Compute total horizontal gradient: sqrt(dTdx^2 + dTdy^2)."""

    dtdy, dtdx = np.gradient(grid.values, grid.dy, grid.dx, edge_order=_edge_order(grid))
    thg = np.sqrt(dtdx**2 + dtdy**2)
    return grid.with_values(
        thg,
        name=f"{grid.name}_THG",
        units=f"{grid.units}/m",
        metadata={
            "method": "total_horizontal_gradient",
            "formula": "THG = sqrt((dT/dx)^2 + (dT/dy)^2)",
        },
    )


def compute_vertical_derivative_fft(
    grid: GridData,
    order: int = 1,
    padding: str = "none",
    fill_method: str = "nearest",
) -> GridData:
    """Compute vertical derivative by multiplying FFT(T) by |k|^order."""

    if order < 1:
        raise ValueError("Vertical derivative order must be at least 1.")
    if padding not in {"none", "zero", "reflect"}:
        raise ValueError("padding must be one of: none, zero, reflect.")

    ready = validate_fft_ready(grid, fill_method=fill_method)
    values, crop = _pad_values(ready.values, padding)
    k = _wavenumber_grid(values.shape, ready.dx, ready.dy)
    result = np.real(np.fft.ifft2((k**order) * np.fft.fft2(values)))[crop]
    method_name = "FVD" if order == 1 else "SVD" if order == 2 else f"VDR{order}"
    warning = (
        "First vertical derivative enhances shallow sources and noise."
        if order == 1
        else "Second and higher vertical derivatives strongly amplify noise; interpret with caution."
    )
    return ready.with_values(
        result,
        name=f"{grid.name}_{method_name}",
        units=f"{grid.units}/m^{order}",
        metadata={
            "method": "vertical_derivative_fft",
            "order": order,
            "padding": padding,
            "formula": "IFFT(|k|^order * FFT(T))",
            "warning": warning,
        },
    )


def compute_fvd(grid: GridData, padding: str = "none") -> GridData:
    return compute_vertical_derivative_fft(grid, order=1, padding=padding)


def compute_svd(grid: GridData, padding: str = "none") -> GridData:
    return compute_vertical_derivative_fft(grid, order=2, padding=padding)
