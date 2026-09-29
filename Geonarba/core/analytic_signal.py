"""Analytic signal amplitude methods."""

from __future__ import annotations

import numpy as np

from .derivatives import compute_fvd
from .grid_tools import GridData


def compute_analytic_signal(grid: GridData, padding: str = "none") -> GridData:
    """Compute analytic signal amplitude from x, y, and vertical derivatives."""

    dtdy, dtdx = np.gradient(grid.values, grid.dy, grid.dx, edge_order=2 if min(grid.shape) >= 3 else 1)
    dtdz = compute_fvd(grid, padding=padding).values
    asa = np.sqrt(dtdx**2 + dtdy**2 + dtdz**2)
    return grid.with_values(
        asa,
        name=f"{grid.name}_ASA",
        units=f"{grid.units}/m",
        metadata={
            "method": "analytic_signal_amplitude",
            "padding": padding,
            "formula": "ASA = sqrt((dT/dx)^2 + (dT/dy)^2 + (dT/dz)^2)",
        },
    )
