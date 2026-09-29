"""Edge-detection filters."""

from __future__ import annotations

import numpy as np

from .analytic_signal import compute_analytic_signal
from .derivatives import compute_fvd, compute_thg
from .grid_tools import GridData
from .tilt import compute_tilt_derivative


def _normalize(values: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot normalize a grid with no finite values.")
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    return (values - vmin) / (vmax - vmin + epsilon)


def compute_fsed(
    grid: GridData,
    base: str = "thg",
    gain: float = 5.0,
    method: str = "logistic",
    epsilon: float = 1e-12,
) -> GridData:
    """Compute Fast Sigmoid Edge Detection from a selected base field."""

    base_key = base.lower().strip()
    if base_key in {"thg ratio", "thg_ratio", "ratio"}:
        # Ratio form (Pham et al., 2020-2021 style): R = dTHG/dz / |grad_h THG| -> bounded sigmoid, peaks over edges.
        thg = compute_thg(grid)
        vdr = compute_fvd(thg).values
        gy, gx = np.gradient(thg.values, grid.dy, grid.dx)
        ratio = vdr / (np.hypot(gx, gy) + epsilon * np.nanmax(np.abs(vdr)) + np.finfo(float).tiny)
        values = 1.0 / (1.0 + np.exp(-gain * ratio)) if method == "logistic" else 0.5 * (1 + ratio / (1.0 + np.abs(ratio)))
        return grid.with_values(
            values,
            name=f"{grid.name}_FSED_ratio_{method}",
            units="dimensionless",
            metadata={
                "method": "fsed",
                "base": "thg_ratio",
                "variant": method,
                "formula": "R = VDR(THG)/|grad_h THG|; logistic 1/(1+e^{-gR}) or fast sigmoid 0.5(1 + R/(1+|R|))",
                "warning": "Edge filters highlight mathematical gradients, not automatically faults.",
            },
        )
    if base_key == "thg":
        base_grid = compute_thg(grid)
    elif base_key == "asa":
        base_grid = compute_analytic_signal(grid)
    elif base_key in {"absolute tilt", "abs_tilt", "tilt"}:
        tilt_grid = compute_tilt_derivative(grid)
        base_grid = tilt_grid.with_values(
            np.abs(tilt_grid.values),
            name=f"{grid.name}_AbsTILT",
            units="radians",
            metadata={"method": "absolute_tilt_base"},
        )
    elif base_key in {"direct", "input"}:
        base_grid = grid
    else:
        raise ValueError("base must be THG ratio, THG, ASA, absolute TILT, or direct input.")

    normalized = _normalize(base_grid.values, epsilon=epsilon)
    if method == "logistic":
        fsed = 1.0 / (1.0 + np.exp(-float(gain) * (normalized - 0.5)))
    elif method == "fast_sigmoid":
        x = 2.0 * normalized - 1.0
        fsed = x / (1.0 + np.abs(x))
        fsed = _normalize(fsed, epsilon=epsilon)
    else:
        raise ValueError("method must be logistic or fast_sigmoid.")

    return grid.with_values(
        fsed,
        name=f"{grid.name}_FSED_{base_key}_{method}",
        units="dimensionless",
        metadata={
            "method": "fsed",
            "base": base_key,
            "gain": float(gain),
            "variant": method,
            "formula": "Logistic: 1/(1+exp(-gain*(normalized-0.5))); fast sigmoid: x/(1+abs(x)).",
            "warning": "Edge filters highlight mathematical gradients, not automatically faults.",
        },
    )


def threshold_edges(grid: GridData, threshold: float = 0.5) -> GridData:
    if threshold < 0 or threshold > 1:
        raise ValueError("FSED threshold should be between 0 and 1.")
    binary = (grid.values >= threshold).astype(float)
    return grid.with_values(
        binary,
        name=f"{grid.name}_Binary_{threshold:g}",
        units="binary",
        metadata={"method": "edge_threshold", "threshold": float(threshold)},
    )
