"""Derivative filters for regular potential-field grids."""

from __future__ import annotations

import numpy as np

from .fourier import pad_values as _pad_values, wavenumber_grid as _wavenumber_grid
from .grid_tools import GridData, validate_fft_ready


def _edge_order(grid: GridData) -> int:
    return 2 if min(grid.values.shape) >= 3 else 1


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
    padding: str = "reflect",
    fill_method: str = "nearest",
) -> GridData:
    """Compute vertical derivative by multiplying FFT(T) by |k|^order."""

    if order < 1:
        raise ValueError("Vertical derivative order must be at least 1.")
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


def compute_fvd(grid: GridData, padding: str = "reflect") -> GridData:
    return compute_vertical_derivative_fft(grid, order=1, padding=padding)


def compute_svd(grid: GridData, padding: str = "reflect") -> GridData:
    return compute_vertical_derivative_fft(grid, order=2, padding=padding)
