"""Tilt derivative filters."""

from __future__ import annotations

import numpy as np

from .derivatives import compute_fvd, compute_thg
from .grid_tools import GridData


def compute_tilt_derivative(grid: GridData, epsilon: float = 1e-10, padding: str = "none") -> GridData:
    """Compute TILT/TDR = arctan(FVD / (THG + epsilon))."""

    thg = compute_thg(grid).values
    fvd = compute_fvd(grid, padding=padding).values
    tilt = np.arctan(fvd / (thg + epsilon))
    return grid.with_values(
        tilt,
        name=f"{grid.name}_TILT",
        units="radians",
        metadata={
            "method": "tilt_derivative",
            "epsilon": epsilon,
            "padding": padding,
            "formula": "TILT = arctan(FVD / (THG + epsilon))",
        },
    )
