"""Gridding of scattered data: minimum curvature, kriging, cokriging, IDW, splines and TIN."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
from scipy.optimize import curve_fit
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay, cKDTree

from .grid_tools import GridData


def _points(table: pd.DataFrame, x_col: str, y_col: str, value_col: str) -> tuple[np.ndarray, np.ndarray]:
    sub = table[[x_col, y_col, value_col]].apply(pd.to_numeric, errors="coerce").dropna()
    sub = sub.groupby([x_col, y_col], as_index=False)[value_col].mean()
    if len(sub) < 3:
        raise ValueError("At least three valid points are required.")
    return sub[[x_col, y_col]].to_numpy(float), sub[value_col].to_numpy(float)


def _target(pts: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    if spacing <= 0:
        raise ValueError("Grid spacing must be positive.")
    return (np.arange(pts[:, 0].min(), pts[:, 0].max() + spacing / 2, spacing),
            np.arange(pts[:, 1].min(), pts[:, 1].max() + spacing / 2, spacing))


def _grid(values, gx, gy, spacing, name, meta) -> GridData:
    return GridData(values, gx, gy, spacing, spacing, name=name, metadata=meta)


def minimum_curvature(
    table: pd.DataFrame, x_col: str = "x", y_col: str = "y", value_col: str = "value", spacing: float = 1000.0, tension: float = 0.0,
) -> GridData:
    """Minimum-curvature surface (Briggs, 1974; Smith & Wessel, 1990):
    minimize (1-T) ||Laplacian u||^2 + T ||grad u||^2 subject (weighted) to bilinear data constraints."""

    pts, vals = _points(table, x_col, y_col, value_col)
    gx, gy = _target(pts, spacing)
    nx, ny = len(gx), len(gy)
    n = nx * ny
    fx = (pts[:, 0] - gx[0]) / spacing
    fy = (pts[:, 1] - gy[0]) / spacing
    i0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
    j0 = np.clip(np.floor(fx).astype(int), 0, nx - 2)
    ty, tx = fy - i0, fx - j0
    rows = np.repeat(np.arange(len(vals)), 4)
    cols = np.stack([i0 * nx + j0, i0 * nx + j0 + 1, (i0 + 1) * nx + j0, (i0 + 1) * nx + j0 + 1], 1).ravel()
    w = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty], 1).ravel()
    a = sparse.csr_matrix((w, (rows, cols)), shape=(len(vals), n))
    d2x = sparse.diags([1, -2, 1], [0, 1, 2], shape=(nx - 2, nx))
    d2y = sparse.diags([1, -2, 1], [0, 1, 2], shape=(ny - 2, ny))
    d1x = sparse.diags([-1, 1], [0, 1], shape=(nx - 1, nx))
    d1y = sparse.diags([-1, 1], [0, 1], shape=(ny - 1, ny))
    lap = sparse.vstack([sparse.kron(sparse.identity(ny), d2x), sparse.kron(d2y, sparse.identity(nx))])
    grad = sparse.vstack([sparse.kron(sparse.identity(ny), d1x), sparse.kron(d1y, sparse.identity(nx))])
    beta = 1e3
    mean = float(vals.mean())
    lhs = (1 - tension) * (lap.T @ lap) + tension * (grad.T @ grad) + beta * (a.T @ a) + 1e-9 * sparse.identity(n)
    u = spsolve(lhs.tocsc(), beta * (a.T @ (vals - mean))) + mean
    return _grid(u.reshape(ny, nx), gx, gy, spacing, f"{value_col}_MinCurv",
                 {"method": "minimum_curvature", "tension": tension, "fit_rms": float(np.sqrt(np.mean((a @ u - vals) ** 2)))})


VARIOGRAMS = {
    "spherical": lambda h, n, s, r: n + s * np.where(h < r, 1.5 * h / r - 0.5 * (h / r) ** 3, 1.0),
    "exponential": lambda h, n, s, r: n + s * (1 - np.exp(-3 * h / r)),
    "gaussian": lambda h, n, s, r: n + s * (1 - np.exp(-3 * (h / r) ** 2)),
}


def experimental_variogram(pts: np.ndarray, vals: np.ndarray, n_lags: int = 15, max_lag: float | None = None, max_pairs: int = 400000):
    rng = np.random.default_rng(0)
    n = len(vals)
    i, j = np.triu_indices(n, 1)
    if len(i) > max_pairs:
        sel = rng.choice(len(i), max_pairs, replace=False)
        i, j = i[sel], j[sel]
    h = np.hypot(*(pts[i] - pts[j]).T)
    g = 0.5 * (vals[i] - vals[j]) ** 2
    max_lag = max_lag or h.max() / 2
    edges = np.linspace(0, max_lag, n_lags + 1)
    idx = np.digitize(h, edges) - 1
    lags, gam, cnt = [], [], []
    for k in range(n_lags):
        m = idx == k
        if m.sum() > 5:
            lags.append(h[m].mean())
            gam.append(g[m].mean())
            cnt.append(int(m.sum()))
    return np.array(lags), np.array(gam), np.array(cnt)


def fit_variogram(lags, gam, model: str = "spherical") -> tuple[float, float, float]:
    f = VARIOGRAMS[model]
    p0 = [gam.min() * 0.1, gam.max(), lags.max() / 2]
    popt, _ = curve_fit(f, lags, gam, p0=p0, bounds=([0, 1e-12, lags.min() * 0.5], [gam.max(), gam.max() * 5, lags.max() * 5]), maxfev=20000)
    return tuple(float(v) for v in popt)


def ordinary_kriging(
    table: pd.DataFrame, x_col: str = "x", y_col: str = "y", value_col: str = "value", spacing: float = 1000.0,
    model: str = "spherical", neighbors: int = 16,
) -> tuple[GridData, GridData, dict]:
    """Ordinary kriging with a fitted variogram and k-nearest neighbourhoods; returns estimate and kriging variance."""

    pts, vals = _points(table, x_col, y_col, value_col)
    lags, gam, _ = experimental_variogram(pts, vals)
    nug, sill, rng_ = fit_variogram(lags, gam, model)
    gfun = lambda h: VARIOGRAMS[model](h, nug, sill, rng_) * (h > 0)  # noqa: E731
    gx, gy = _target(pts, spacing)
    xx, yy = np.meshgrid(gx, gy)
    targets = np.column_stack([xx.ravel(), yy.ravel()])
    k = min(neighbors, len(vals))
    _, nb = cKDTree(pts).query(targets, k=k)
    est = np.empty(len(targets))
    var = np.empty(len(targets))
    for t, idx in enumerate(nb):
        p = pts[idx]
        a = np.ones((k + 1, k + 1))
        a[:k, :k] = gfun(np.hypot(*(p[:, None, :] - p[None, :, :]).transpose(2, 0, 1)))
        a[k, k] = 0
        b = np.ones(k + 1)
        b[:k] = gfun(np.hypot(*(p - targets[t]).T))
        w = np.linalg.lstsq(a, b, rcond=None)[0]
        est[t] = w[:k] @ vals[idx]
        var[t] = w @ b
    meta = {"method": "ordinary_kriging", "variogram": model, "nugget": nug, "sill": sill, "range": rng_}
    return (_grid(est.reshape(xx.shape), gx, gy, spacing, f"{value_col}_OK", meta),
            _grid(var.reshape(xx.shape), gx, gy, spacing, f"{value_col}_OK_variance", meta),
            {"nugget": nug, "partial_sill": sill, "range": rng_, "lags": lags.tolist(), "semivariance": gam.tolist()})


def collocated_cokriging(
    table: pd.DataFrame, secondary: GridData, x_col: str = "x", y_col: str = "y", value_col: str = "value",
    model: str = "spherical", neighbors: int = 16,
) -> tuple[GridData, dict]:
    """Collocated ordinary cokriging with the Markov model 1 (Xu et al., 1992): C12(h) = rho12 sqrt(C11(0) C22(0)) rho11(h).

    Output is on the secondary grid; the secondary is linearly calibrated to primary units at the sample locations.
    """

    pts, vals = _points(table, x_col, y_col, value_col)
    from scipy.interpolate import RegularGridInterpolator

    sec_at = RegularGridInterpolator((secondary.y_vector, secondary.x_vector), secondary.values, bounds_error=False, fill_value=None)
    s_pts = sec_at(pts[:, ::-1])
    s_all = secondary.values
    slope, intercept = np.polyfit(s_pts, vals, 1)  # calibrate secondary into primary units at the samples
    s_std = intercept + slope * s_all
    rho12 = float(np.corrcoef(vals, s_pts)[0, 1])
    lags, gam, _ = experimental_variogram(pts, vals)
    nug, sill, rng_ = fit_variogram(lags, gam, model)
    c0 = nug + sill
    cov = lambda h: np.where(h > 0, c0 - VARIOGRAMS[model](h, nug, sill, rng_), c0)  # noqa: E731
    xx, yy = secondary.mesh
    targets = np.column_stack([xx.ravel(), yy.ravel()])
    k = min(neighbors, len(vals))
    _, nb = cKDTree(pts).query(targets, k=k)
    est = np.empty(len(targets))
    for t, idx in enumerate(nb):
        p = pts[idx]
        d = np.hypot(*(p[:, None, :] - p[None, :, :]).transpose(2, 0, 1))
        dt = np.hypot(*(p - targets[t]).T)
        a = np.zeros((k + 2, k + 2))
        a[:k, :k] = cov(d)
        a[:k, k] = a[k, :k] = rho12 * cov(dt)
        a[k, k] = c0
        a[:k, k + 1] = a[k + 1, :k] = 1
        a[k, k + 1] = a[k + 1, k] = 1  # single unbiasedness: sum(w_primary) + w_secondary = 1 (standardized secondary)
        b = np.zeros(k + 2)
        b[:k] = cov(dt)
        b[k] = rho12 * c0
        b[k + 1] = 1
        w = np.linalg.lstsq(a, b, rcond=None)[0]
        est[t] = w[:k] @ vals[idx] + w[k] * s_std.ravel()[t]
    return (secondary.with_values(est.reshape(xx.shape), name=f"{value_col}_CoKriged", units="",
                                  metadata={"method": "collocated_cokriging", "rho12": rho12}),
            {"correlation_primary_secondary": rho12, "range": rng_, "sill": c0})


def idw(
    table: pd.DataFrame, x_col: str = "x", y_col: str = "y", value_col: str = "value", spacing: float = 1000.0,
    power: float = 2.0, neighbors: int = 12,
) -> GridData:
    """Inverse distance weighting (Shepard, 1968) over k nearest neighbours."""

    pts, vals = _points(table, x_col, y_col, value_col)
    gx, gy = _target(pts, spacing)
    xx, yy = np.meshgrid(gx, gy)
    d, i = cKDTree(pts).query(np.column_stack([xx.ravel(), yy.ravel()]), k=min(neighbors, len(vals)))
    d = np.atleast_2d(d.T).T if d.ndim == 1 else d
    w = 1 / np.maximum(d, 1e-12) ** power
    est = (w * vals[i]).sum(1) / w.sum(1)
    return _grid(est.reshape(xx.shape), gx, gy, spacing, f"{value_col}_IDW", {"method": "idw", "power": power})


def spline_interpolation(
    table: pd.DataFrame, x_col: str = "x", y_col: str = "y", value_col: str = "value", spacing: float = 1000.0,
    smoothing: float = 0.0, kernel: str = "thin_plate_spline",
) -> GridData:
    """Radial-basis spline (thin-plate spline = minimum bending energy surface)."""

    pts, vals = _points(table, x_col, y_col, value_col)
    gx, gy = _target(pts, spacing)
    xx, yy = np.meshgrid(gx, gy)
    scale = np.ptp(pts, axis=0).max()
    rbf = RBFInterpolator(pts / scale, vals, kernel=kernel, smoothing=smoothing, neighbors=min(len(vals), 64))
    est = rbf(np.column_stack([xx.ravel(), yy.ravel()]) / scale)
    return _grid(est.reshape(xx.shape), gx, gy, spacing, f"{value_col}_Spline", {"method": "rbf_spline", "kernel": kernel})


def tin_interpolation(
    table: pd.DataFrame, x_col: str = "x", y_col: str = "y", value_col: str = "value", spacing: float = 1000.0,
) -> tuple[GridData, pd.DataFrame]:
    """Delaunay TIN: exact at samples, planar facets between; returns the rasterized TIN and its triangle table."""

    pts, vals = _points(table, x_col, y_col, value_col)
    tri = Delaunay(pts)
    gx, gy = _target(pts, spacing)
    xx, yy = np.meshgrid(gx, gy)
    est = LinearNDInterpolator(tri, vals)(xx, yy)
    faces = pd.DataFrame(tri.simplices, columns=["v1", "v2", "v3"])
    return _grid(est, gx, gy, spacing, f"{value_col}_TIN", {"method": "delaunay_tin", "triangles": int(len(faces))}), faces
