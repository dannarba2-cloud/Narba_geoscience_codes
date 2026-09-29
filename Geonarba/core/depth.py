"""Source depth estimation: Euler, Werner, SPI, tilt-depth, Curie point depth and heat flow."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .derivatives import compute_fvd
from .grid_tools import GridData, validate_fft_ready
from .profile import ProfileData
from .spectral import compute_raps
from .tilt import compute_tilt_derivative


def _gradients(grid: GridData, padding: str = "reflect") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ready = validate_fft_ready(grid)
    ty, tx = np.gradient(ready.values, ready.dy, ready.dx)
    tz = compute_fvd(ready, padding=padding).values
    return tx, ty, tz


def euler_deconvolution(
    grid: GridData,
    structural_index: float = 1.0,
    window: int = 10,
    step: int | None = None,
    max_depth_error: float = 0.15,
    padding: str = "reflect",
) -> pd.DataFrame:
    """3D Euler deconvolution (Reid et al., 1990).

    Solves x0 Tx + y0 Ty + z0 Tz + N B = x Tx + y Ty + N T in moving windows (z down, observation z=0).
    Solutions are kept when z0 > 0, sigma_z/z0 < max_depth_error and (x0, y0) lie within 1.5 windows.
    """

    if window < 3:
        raise ValueError("Euler window must be at least 3 cells.")
    step = step or max(window // 2, 1)
    ready = validate_fft_ready(grid)
    tx, ty, tz = _gradients(ready, padding)
    xx, yy = ready.mesh
    t = ready.values
    n = float(structural_index)
    rows = []
    ny, nx = t.shape
    for i in range(0, ny - window + 1, step):
        for j in range(0, nx - window + 1, step):
            sl = (slice(i, i + window), slice(j, j + window))
            a = np.column_stack([tx[sl].ravel(), ty[sl].ravel(), tz[sl].ravel(), np.full(window * window, n)])
            b = xx[sl].ravel() * tx[sl].ravel() + yy[sl].ravel() * ty[sl].ravel() + n * t[sl].ravel()
            sol, res, rank, _ = np.linalg.lstsq(a, b, rcond=None)
            if rank < 4:
                continue
            dof = max(len(b) - 4, 1)
            sigma2 = float(np.sum((a @ sol - b) ** 2)) / dof
            cov = sigma2 * np.linalg.pinv(a.T @ a)
            x0, y0, z0, base = sol
            sz = float(np.sqrt(max(cov[2, 2], 0.0)))
            cx, cy = float(xx[sl].mean()), float(yy[sl].mean())
            radius = 1.5 * window * max(ready.dx, ready.dy)
            if z0 <= 0 or sz / z0 > max_depth_error or np.hypot(x0 - cx, y0 - cy) > radius:
                continue
            rows.append({"x": x0, "y": y0, "depth": z0, "depth_std": sz, "base_level": base,
                         "structural_index": n, "window_x": cx, "window_y": cy})
    return pd.DataFrame(rows, columns=["x", "y", "depth", "depth_std", "base_level", "structural_index", "window_x", "window_y"])


def werner_deconvolution(
    profile: ProfileData,
    window: int = 7,
    model: str = "dike",
    min_depth: float = 0.0,
) -> pd.DataFrame:
    """Werner deconvolution (Werner, 1953; Hartman et al., 1971) on a regularly sampled profile.

    Thin-dike total field T = [A(x-x0) + B z] / ((x-x0)^2 + z^2) is linear in
    T x^2 = a0 + a1 x + b0 T + b1 x T, giving x0 = b1/2 and z = sqrt(-b0 - x0^2).
    Contacts use the horizontal derivative of the profile (model='contact').
    """

    x = profile.distance
    v = profile.values if model == "dike" else np.gradient(profile.values, x)
    rows = []
    for s in range(0, len(x) - window + 1):
        xs, ts = x[s:s + window], v[s:s + window]
        if not np.isfinite(ts).all():
            continue
        xc = xs.mean()
        xr = xs - xc
        a = np.column_stack([np.ones_like(xr), xr, ts, xr * ts])
        sol, *_ = np.linalg.lstsq(a, ts * xr**2, rcond=None)
        a0, a1, b0, b1 = sol
        x0 = b1 / 2.0
        z2 = -b0 - x0**2
        if z2 <= 0:
            continue
        z = float(np.sqrt(z2))
        if z < min_depth or abs(x0) > np.ptp(xs):
            continue
        susceptibility_term = float(np.hypot(a1, (a0 + a1 * x0) / z))
        rows.append({"x": x0 + xc, "depth": z, "amplitude": susceptibility_term, "window_center": xc, "model": model})
    return pd.DataFrame(rows, columns=["x", "depth", "amplitude", "window_center", "model"])


def source_parameter_imaging(grid: GridData, max_depth: float | None = None, padding: str = "reflect") -> GridData:
    """SPI (Thurston & Smith, 1997): depth = 1 / K, K = |grad(tilt)| (local wavenumber). Valid over contacts."""

    tilt = compute_tilt_derivative(grid, padding=padding).values
    ky, kx = np.gradient(tilt, grid.dy, grid.dx)
    k = np.hypot(kx, ky)
    depth = 1.0 / np.maximum(k, np.finfo(float).tiny)
    limit = max_depth or 0.5 * min(np.ptp(grid.x_vector), np.ptp(grid.y_vector))
    depth = np.where(depth > limit, np.nan, depth)
    return grid.with_values(depth, name=f"{grid.name}_SPI_depth", units="m",
                            metadata={"method": "source_parameter_imaging", "formula": "depth = 1/sqrt(tilt_x^2 + tilt_y^2)",
                                      "warning": "SPI depths are valid at contact edges (use with THG/ASA peaks)."})


def tilt_depth(grid: GridData, tilt_tolerance_deg: float = 2.0, padding: str = "reflect") -> tuple[GridData, pd.DataFrame]:
    """Tilt-depth (Salem et al., 2007): on the tilt = 0 contour, depth = 1 / |grad tilt| for vertical contacts."""

    tilt = compute_tilt_derivative(grid, padding=padding).values
    gy, gx = np.gradient(tilt, grid.dy, grid.dx)
    grad = np.hypot(gx, gy)
    mask = np.abs(tilt) < np.radians(tilt_tolerance_deg)
    depth = np.where(mask, 1.0 / np.maximum(grad, np.finfo(float).tiny), np.nan)
    xx, yy = grid.mesh
    table = pd.DataFrame({"x": xx[mask], "y": yy[mask], "depth": depth[mask]})
    out = grid.with_values(depth, name=f"{grid.name}_TiltDepth", units="m",
                           metadata={"method": "tilt_depth", "formula": "z = 1/|grad(theta)| at theta = 0"})
    return out, table


def curie_point_depth(
    grid: GridData,
    top_k_range: tuple[float, float] | None = None,
    centroid_k_range: tuple[float, float] | None = None,
    window: str = "hanning",
) -> dict:
    """Centroid method (Okubo et al., 1985; Tanaka et al., 1999).

    ln(P^1/2) = A - |k| Zt (higher wavenumbers);  ln(P^1/2/|k|) = B - |k| Z0 (lowest wavenumbers);
    Zb = 2 Z0 - Zt. k in rad/m, depths in m.
    """

    raps = compute_raps(grid, window=window, padding="zero", detrend=True)
    k = raps.k
    amp = np.sqrt(raps.power)
    ok = (k > 0) & (amp > 0)
    k, amp = k[ok], amp[ok]
    kmax = k.max()
    t_lo, t_hi = top_k_range or (0.05 * kmax, 0.25 * kmax)
    c_lo, c_hi = centroid_k_range or (k[0], max(0.04 * kmax, k[min(5, len(k) - 1)]))
    mt = (k >= t_lo) & (k <= t_hi)
    mc = (k >= c_lo) & (k <= c_hi)
    if mt.sum() < 3 or mc.sum() < 3:
        raise ValueError("Wavenumber ranges must each contain at least 3 spectral bins; widen them or use a larger window.")
    slope_t, _ = np.polyfit(k[mt], np.log(amp[mt]), 1)
    slope_c, _ = np.polyfit(k[mc], np.log(amp[mc] / k[mc]), 1)
    zt, z0 = -slope_t, -slope_c
    zb = 2 * z0 - zt
    width = min(np.ptp(grid.x_vector), np.ptp(grid.y_vector))
    return {
        "top_depth_m": float(zt),
        "centroid_depth_m": float(z0),
        "curie_depth_m": float(zb),
        "top_k_range": (float(t_lo), float(t_hi)),
        "centroid_k_range": (float(c_lo), float(c_hi)),
        "window_width_m": float(width),
        "warning": (
            "Window should be several times (>= ~3-5x) the expected Curie depth; results below "
            f"~{width / 3:.0f} m resolvable depth for this window are unreliable."
            if zb > width / 3 else "Centroid-method depths depend on the chosen wavenumber ranges."
        ),
    }


def heat_flow_from_curie(
    curie_depth_m,
    thermal_conductivity: float = 2.5,
    curie_temperature: float = 580.0,
    surface_temperature: float = 15.0,
    heat_production: float = 0.0,
) -> np.ndarray:
    """Surface heat flow (mW/m^2) from Curie depth for a 1D steady-state conductive geotherm.

    q0 = k (Tc - T0) / Zb + A Zb / 2   (uniform heat production A in uW/m^3; A = 0 gives the linear model).
    """

    zb = np.asarray(curie_depth_m, float)
    return (thermal_conductivity * (curie_temperature - surface_temperature) / zb + heat_production * 1e-6 * zb / 2) * 1e3


def curie_depth_map(
    grid: GridData,
    window_size: float = 100000.0,
    step: float = 25000.0,
    thermal_conductivity: float = 2.5,
    curie_temperature: float = 580.0,
    surface_temperature: float = 15.0,
    heat_production: float = 0.0,
) -> tuple[GridData, GridData, pd.DataFrame]:
    """Windowed centroid-method CPD map and derived heat flow map."""

    nwx = int(round(window_size / grid.dx))
    nwy = int(round(window_size / grid.dy))
    sx, sy = max(int(round(step / grid.dx)), 1), max(int(round(step / grid.dy)), 1)
    ny, nx = grid.values.shape
    if nwx > nx or nwy > ny:
        raise ValueError("Window is larger than the grid; reduce window_size.")
    xs, ys, zb_rows = [], [], []
    yi = list(range(0, ny - nwy + 1, sy))
    xi = list(range(0, nx - nwx + 1, sx))
    zb = np.full((len(yi), len(xi)), np.nan)
    for a, i in enumerate(yi):
        for b, j in enumerate(xi):
            sub = GridData(grid.values[i:i + nwy, j:j + nwx], grid.x_vector[j:j + nwx], grid.y_vector[i:i + nwy], grid.dx, grid.dy)
            try:
                res = curie_point_depth(sub)
                zb[a, b] = res["curie_depth_m"] if res["curie_depth_m"] > 0 else np.nan
                zb_rows.append({"x": sub.x_vector.mean(), "y": sub.y_vector.mean(), **{k: v for k, v in res.items() if "range" not in k and k != "warning"}})
            except ValueError:
                continue
    cx = np.array([grid.x_vector[j:j + nwx].mean() for j in xi])
    cy = np.array([grid.y_vector[i:i + nwy].mean() for i in yi])
    dxw = step if len(cx) < 2 else float(np.diff(cx).mean())
    dyw = step if len(cy) < 2 else float(np.diff(cy).mean())
    zgrid = GridData(zb, cx, cy, dxw, dyw, name=f"{grid.name}_CurieDepth", units="m", crs=grid.crs,
                     metadata={"method": "curie_depth_map", "window_m": window_size})
    q = heat_flow_from_curie(zb, thermal_conductivity, curie_temperature, surface_temperature, heat_production)
    qgrid = zgrid.with_values(q, name=f"{grid.name}_HeatFlow", units="mW/m^2",
                              metadata={"method": "heat_flow", "formula": "q = k (Tc - T0)/Zb + A Zb/2",
                                        "thermal_conductivity": thermal_conductivity})
    return zgrid, qgrid, pd.DataFrame(zb_rows)
