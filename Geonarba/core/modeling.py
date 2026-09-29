"""Forward modeling and linear inversion built on analytic prism kernels."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

from .grid_tools import GridData
from .prisms import SI_TO_MGAL, gz_kernel, magnetic_kernel
from .profile import ProfileData


def parse_polygon(text: str) -> np.ndarray:
    """Parse 'x,z; x,z; ...' (m, z down) into an (n, 2) array."""

    pts = [tuple(float(v) for v in item.split(",")) for item in text.replace("\n", ";").split(";") if item.strip()]
    arr = np.asarray(pts, float)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 3:
        raise ValueError("Polygon needs at least three 'x,z' vertices separated by ';'.")
    return arr


def polygon_to_columns(vertices: np.ndarray, column_width: float) -> np.ndarray:
    """Discretize a simple polygon (x, z-down) into vertical columns [x1, x2, z_top, z_bottom]."""

    xs = np.arange(vertices[:, 0].min() + column_width / 2, vertices[:, 0].max(), column_width)
    closed = np.vstack([vertices, vertices[:1]])
    cols = []
    for xc in xs:
        hits = []
        for (x1, z1), (x2, z2) in zip(closed[:-1], closed[1:]):
            if (x1 <= xc < x2) or (x2 <= xc < x1):
                hits.append(z1 + (xc - x1) * (z2 - z1) / (x2 - x1))
        hits.sort()
        for top, bottom in zip(hits[0::2], hits[1::2]):
            cols.append([xc - column_width / 2, xc + column_width / 2, top, bottom])
    return np.asarray(cols, float).reshape(-1, 4)


def forward_2_5d(
    x_obs: np.ndarray,
    bodies: list[dict],
    strike_half_length: float = 10000.0,
    observation_elevation: float = 0.0,
    column_width: float | None = None,
    field: str = "gravity",
    inclination: float = 60.0,
    declination: float = 0.0,
    profile_azimuth: float = 90.0,
) -> np.ndarray:
    """2.5D forward response of polygonal bodies with finite strike +-L (Nagy prism columns).

    bodies: [{'vertices': (n,2) x,z-down array, 'property': density contrast kg/m^3 or magnetization A/m}]
    Gravity returns mGal; magnetics returns total-field nT (profile azimuth rotates the field declination).
    """

    x_obs = np.asarray(x_obs, float)
    obs = np.column_stack([x_obs, np.zeros_like(x_obs), np.full_like(x_obs, -observation_elevation)])
    total = np.zeros_like(x_obs)
    for body in bodies:
        verts = np.asarray(body["vertices"], float)
        width = column_width or max(np.ptp(verts[:, 0]) / 200.0, 1.0)
        cols = polygon_to_columns(verts, width)
        if cols.size == 0:
            continue
        prisms = np.column_stack([cols[:, 0], cols[:, 1], np.full(len(cols), -strike_half_length),
                                  np.full(len(cols), strike_half_length), cols[:, 2], cols[:, 3]])
        if field == "gravity":
            total += gz_kernel(obs, prisms).sum(axis=1) * body["property"] * SI_TO_MGAL
        else:
            # Profile x-axis points along `profile_azimuth`; express declination relative to it.
            rel_dec = declination - (profile_azimuth - 90.0)
            total += magnetic_kernel(obs, prisms, inclination, rel_dec).sum(axis=1) * body["property"]
    return total


def forward_2_5d_profile(
    x_min: float,
    x_max: float,
    spacing: float,
    polygon: str,
    property_value: float,
    strike_half_length: float = 10000.0,
    field: str = "gravity",
    inclination: float = 60.0,
    declination: float = 0.0,
    observed: ProfileData | None = None,
) -> tuple[ProfileData, dict]:
    x = observed.distance if observed is not None else np.arange(x_min, x_max + spacing / 2, spacing)
    calc = forward_2_5d(x, [{"vertices": parse_polygon(polygon), "property": property_value}],
                        strike_half_length, field=field, inclination=inclination, declination=declination)
    units = "mGal" if field == "gravity" else "nT"
    prof = ProfileData(x, calc, name=f"2.5D_{field}_model", units=units,
                       metadata={"method": "forward_2_5d", "strike_half_length_m": strike_half_length})
    report = {"max_response": float(np.max(np.abs(calc)))}
    if observed is not None:
        report["rms_misfit"] = float(np.sqrt(np.nanmean((observed.values - calc) ** 2)))
    return prof, report


def build_mesh(grid: GridData, n_z: int, max_depth: float, top: float = 0.0, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Prism mesh below a grid: returns prisms (n,6) and cell centres (n,3)."""

    xv, yv = grid.x_vector[::stride], grid.y_vector[::stride]
    dx, dy = grid.dx * stride, grid.dy * stride
    z_edges = top + np.linspace(0, 1, n_z + 1) ** 1.5 * (max_depth - top)  # finer near surface
    zz, yy, xx = np.meshgrid(0.5 * (z_edges[:-1] + z_edges[1:]), yv, xv, indexing="ij")
    iz = np.repeat(np.arange(n_z), len(yv) * len(xv))
    prisms = np.column_stack([
        xx.ravel() - dx / 2, xx.ravel() + dx / 2, yy.ravel() - dy / 2, yy.ravel() + dy / 2,
        z_edges[iz], z_edges[iz + 1],
    ])
    return prisms, np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def _gradient_ops(nz: int, ny: int, nx: int) -> list[sparse.csr_matrix]:
    def d1(n):
        return sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n)) if n > 1 else sparse.csr_matrix((0, n))

    iz, iy, ix = sparse.identity(nz), sparse.identity(ny), sparse.identity(nx)
    return [
        sparse.kron(sparse.kron(iz, iy), d1(nx)).tocsr(),
        sparse.kron(sparse.kron(iz, d1(ny)), ix).tocsr(),
        sparse.kron(sparse.kron(d1(nz), iy), ix).tocsr(),
    ]


def _centered_gradients(nz: int, ny: int, nx: int) -> list[sparse.csr_matrix]:
    """Same-size central-difference operators (for cross-gradients)."""

    def c1(n):
        if n < 2:
            return sparse.csr_matrix((n, n))
        m = sparse.diags([-0.5 * np.ones(n - 1), 0.5 * np.ones(n - 1)], [-1, 1], shape=(n, n)).tolil()
        m[0, 0], m[0, 1], m[n - 1, n - 2], m[n - 1, n - 1] = -1, 1, -1, 1
        return m.tocsr()

    iz, iy, ix = sparse.identity(nz), sparse.identity(ny), sparse.identity(nx)
    return [
        sparse.kron(sparse.kron(iz, iy), c1(nx)).tocsr(),
        sparse.kron(sparse.kron(iz, c1(ny)), ix).tocsr(),
        sparse.kron(sparse.kron(c1(nz), iy), ix).tocsr(),
    ]


def _slices_from_model(model: np.ndarray, grid: GridData, centres: np.ndarray, shape, stride: int, prefix: str, units: str) -> list[GridData]:
    nz, ny, nx = shape
    vol = model.reshape(nz, ny, nx)
    xv, yv = grid.x_vector[::stride], grid.y_vector[::stride]
    zc = centres.reshape(nz, ny, nx, 3)[:, 0, 0, 2]
    return [
        GridData(vol[k], xv, yv, grid.dx * stride, grid.dy * stride, name=f"{prefix}_z{zc[k]:.0f}m", units=units,
                 crs=grid.crs, metadata={"method": prefix, "depth_m": float(zc[k])})
        for k in range(nz)
    ]


def voxel_inversion(
    grid: GridData,
    field: str = "gravity",
    n_z: int = 8,
    max_depth: float = 10000.0,
    stride: int = 2,
    regularization: float = 1e-2,
    depth_weight_beta: float = 2.0,
    inclination: float = 60.0,
    declination: float = 0.0,
    max_cells: int = 12000,
    operator: str = "first-derivative",
    alpha_s: float = 1e-4,
    alpha_x: float = 1.0,
    alpha_y: float = 1.0,
    alpha_z: float = 1.0,
    reference_value: float = 0.0,
    error_percent: float = 2.0,
    error_floor: float = 0.0,
    lambda_mode: str = "discrepancy",
    solver: str = "cg",
) -> tuple[list[GridData], pd.DataFrame, GridData, GridData, "ProfileData", dict]:
    """3D linear Tikhonov inversion (Li & Oldenburg, 1996/1998) for density contrast (kg/m^3) or magnetization (A/m).

    Phi = ||W_d (G m - d)||^2 + lambda ||W_m W_z (m - m_ref)||^2 with depth weighting w(z) = (z + z0)^(-beta/2),
    W_d = 1/sigma, anisotropic 1st/2nd-derivative or smallness operator, lambda by discrepancy / L-curve / GCV / fixed.
    Returns depth slices, model table, predicted and residual grids, the L-curve and a QC report.
    """

    from .inversion import data_errors, misfit_report, regularization_operator, tikhonov_invert

    sub = grid.values[::stride, ::stride]
    ny, nx = sub.shape
    if nx * ny * n_z > max_cells:
        raise ValueError(f"Mesh too large ({nx * ny * n_z} cells); increase stride or reduce n_z (limit {max_cells}).")
    prisms, centres = build_mesh(grid, n_z, max_depth, stride=stride)
    xx, yy = np.meshgrid(grid.x_vector[::stride], grid.y_vector[::stride])
    obs = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, -1.0)])
    base = float(np.nanmean(sub))
    d = sub.ravel() - base
    if field == "gravity":
        gmat = gz_kernel(obs, prisms) * SI_TO_MGAL
        units = "kg/m^3"
    else:
        gmat = magnetic_kernel(obs, prisms, inclination, declination)
        units = "A/m"
    z0 = 0.5 * (prisms[0, 5] - prisms[0, 4])
    w = (centres[:, 2] + z0) ** (-depth_weight_beta / 2.0)
    w /= w.max()
    sigma = data_errors(d, error_percent, error_floor or None)
    wm = regularization_operator((n_z, ny, nx), operator, alpha_s, alpha_x, alpha_y, alpha_z, cell_weights=w)
    res = tikhonov_invert(gmat, d, sigma, wm, reference_value, lambda_mode, regularization, solver)
    model = res.model
    slices = _slices_from_model(model, grid, centres, (n_z, ny, nx), stride, f"{field}_voxel", units)
    table = pd.DataFrame({"x": centres[:, 0], "y": centres[:, 1], "z": centres[:, 2], units: model})
    sub_grid = GridData(sub, grid.x_vector[::stride], grid.y_vector[::stride], grid.dx * stride, grid.dy * stride, crs=grid.crs)
    pred = sub_grid.with_values(res.predicted.reshape(ny, nx) + base, f"{grid.name}_voxel_predicted", grid.units,
                                {"method": "voxel_prediction"})
    resid = sub_grid.with_values(sub - pred.values, f"{grid.name}_voxel_residual", grid.units, {"method": "voxel_residual"})
    report = {**res.report, **misfit_report(d, res.predicted, sigma, (ny, nx), (grid.dx * stride, grid.dy * stride)),
              "cells": int(len(model)), "operator": operator, "alphas": [alpha_s, alpha_x, alpha_y, alpha_z],
              "depth_weight_beta": depth_weight_beta, "error_model": f"{error_percent}% + floor"}
    return slices, table, pred, resid, lcurve_profile(res.lcurve, f"{grid.name}_L_curve"), report


def lcurve_profile(curve: dict, name: str = "L_curve") -> "ProfileData | None":
    """L-curve as a log-log profile (phi_d on x, phi_m on y); the chosen lambda is stored in metadata."""

    if not curve:
        return None
    phid, phim = np.asarray(curve["phi_d"]), np.asarray(curve["phi_m"])
    order = np.argsort(phid)
    k = curve["corner_index"]
    return ProfileData(phid[order], np.maximum(phim[order], 1e-300), name=name, units="phi_m (roughness)",
                       metadata={"axis": "phi_d (data misfit)", "log_axes": True, "method": "l_curve",
                                 "selected": [float(phid[k]), float(phim[k])], "lambda": curve["lambda"]})


def equivalent_sources(
    table: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    value_col: str = "value",
    spacing: float = 1000.0,
    output_elevation: float = 0.0,
    source_depth_factor: float = 3.0,
    damping: float = 1e-3,
) -> GridData:
    """Equivalent-source gridding/continuation (Dampney, 1969): point masses below each station.

    Stations may lie on an uneven surface (z_col = elevation, m); predictions are made on a
    regular grid at a constant `output_elevation`.
    """

    x = table[x_col].to_numpy(float)
    y = table[y_col].to_numpy(float)
    h = table[z_col].to_numpy(float) if z_col and z_col in table else np.zeros(len(table))
    v = table[value_col].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(h) & np.isfinite(v)
    x, y, h, v = x[ok], y[ok], h[ok], v[ok]
    from scipy.spatial import cKDTree

    nn = cKDTree(np.column_stack([x, y])).query(np.column_stack([x, y]), k=2)[0][:, 1]
    depth = source_depth_factor * float(np.median(nn))
    src = np.column_stack([x, y, -h + depth])  # z down

    def kernel(px, py, pz):
        dz = src[None, :, 2] - pz[:, None]
        r = np.sqrt((src[None, :, 0] - px[:, None]) ** 2 + (src[None, :, 1] - py[:, None]) ** 2 + dz**2)
        return dz / r**3

    a = kernel(x, y, -h)
    lam = damping * np.trace(a.T @ a) / len(v)
    mean = float(np.mean(v))
    coef = np.linalg.solve(a.T @ a + lam * np.eye(len(v)), a.T @ (v - mean))
    gx = np.arange(x.min(), x.max() + spacing / 2, spacing)
    gy = np.arange(y.min(), y.max() + spacing / 2, spacing)
    xx, yy = np.meshgrid(gx, gy)
    pred = kernel(xx.ravel(), yy.ravel(), np.full(xx.size, -output_elevation)) @ coef + mean
    fit_rms = float(np.sqrt(np.mean((a @ coef + mean - v) ** 2)))
    return GridData(pred.reshape(xx.shape), gx, gy, spacing, spacing, name=f"{value_col}_EqSrc_{output_elevation:g}m",
                    metadata={"method": "equivalent_sources", "source_depth_m": depth, "damping": damping,
                              "fit_rms": fit_rms, "output_elevation_m": output_elevation,
                              "formula": "d = sum_j c_j dz_j / r_j^3 (point sources), least squares with damping"})


def joint_inversion_cross_gradient(
    gravity: GridData,
    magnetic: GridData,
    n_z: int = 6,
    max_depth: float = 8000.0,
    stride: int = 3,
    regularization: float = 1e-2,
    cross_gradient_weight: float = 1.0,
    iterations: int = 5,
    inclination: float = 60.0,
    declination: float = 0.0,
    max_cells: int = 5000,
) -> tuple[list[GridData], dict]:
    """Joint gravity-magnetic inversion with cross-gradient coupling (Gallardo & Meju, 2003).

    Gauss-Newton on [G1 m1 - d1; G2 m2 - d2; alpha R m; beta t(m1, m2)], t = grad m1 x grad m2 linearized.
    """

    if gravity.values.shape != magnetic.values.shape:
        raise ValueError("Gravity and magnetic grids must share the same shape.")
    ny, nx = gravity.values[::stride, ::stride].shape
    n = nx * ny * n_z
    if n > max_cells:
        raise ValueError(f"Mesh too large ({n} cells); increase stride (limit {max_cells}).")
    prisms, centres = build_mesh(gravity, n_z, max_depth, stride=stride)
    xx, yy = np.meshgrid(gravity.x_vector[::stride], gravity.y_vector[::stride])
    obs = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, -1.0)])
    d1 = gravity.values[::stride, ::stride].ravel()
    d2 = magnetic.values[::stride, ::stride].ravel()
    d1, d2 = d1 - d1.mean(), d2 - d2.mean()
    g1 = gz_kernel(obs, prisms) * SI_TO_MGAL
    g2 = magnetic_kernel(obs, prisms, inclination, declination)
    w = (centres[:, 2] + 0.5 * (prisms[0, 5] - prisms[0, 4])) ** -1.0
    w /= w.max()
    s1, s2 = np.linalg.norm(g1 / w, 2), np.linalg.norm(g2 / w, 2)
    a1, a2 = sparse.csr_matrix(g1 / w / s1), sparse.csr_matrix(g2 / w / s2)
    reg = sparse.vstack([sparse.identity(n) * 1e-3] + _gradient_ops(n_z, ny, nx)) * np.sqrt(regularization)
    dx_, dy_, dz_ = _centered_gradients(n_z, ny, nx)
    m1, m2 = np.zeros(n), np.zeros(n)
    history = []
    zero = sparse.csr_matrix((reg.shape[0], n))
    for it in range(iterations):
        gx1, gy1, gz1 = dx_ @ m1, dy_ @ m1, dz_ @ m1
        gx2, gy2, gz2 = dx_ @ m2, dy_ @ m2, dz_ @ m2
        t = np.concatenate([gy1 * gz2 - gz1 * gy2, gz1 * gx2 - gx1 * gz2, gx1 * gy2 - gy1 * gx2])
        dg = sparse.diags
        b1 = sparse.vstack([dg(gz2) @ dy_ - dg(gy2) @ dz_, dg(gx2) @ dz_ - dg(gz2) @ dx_, dg(gy2) @ dx_ - dg(gx2) @ dy_])
        b2 = sparse.vstack([dg(gy1) @ dz_ - dg(gz1) @ dy_, dg(gz1) @ dx_ - dg(gx1) @ dz_, dg(gx1) @ dy_ - dg(gy1) @ dx_])
        beta = cross_gradient_weight if it > 0 else 0.0
        lhs = sparse.vstack([
            sparse.hstack([a1, sparse.csr_matrix(a1.shape)]),
            sparse.hstack([sparse.csr_matrix(a2.shape), a2]),
            sparse.hstack([reg, zero]),
            sparse.hstack([zero, reg]),
            beta * sparse.hstack([b1, b2]),
        ]).tocsr()
        rhs = np.concatenate([
            d1 / s1 - a1 @ m1, d2 / s2 - a2 @ m2, -(reg @ m1), -(reg @ m2), -beta * t,
        ])
        step = lsqr(lhs, rhs, iter_lim=1500, atol=1e-8, btol=1e-8)[0]
        m1, m2 = m1 + step[:n], m2 + step[n:]
        history.append(float(np.linalg.norm(np.concatenate([gy1 * gz2 - gz1 * gy2]))))
    rho, mag = m1 / w, m2 / w
    shape = (n_z, ny, nx)
    slices = _slices_from_model(rho, gravity, centres, shape, stride, "joint_density", "kg/m^3") + \
        _slices_from_model(mag, gravity, centres, shape, stride, "joint_magnetization", "A/m")
    return slices, {
        "cells": n,
        "gravity_rms_mGal": float(np.sqrt(np.mean((g1 @ rho - d1) ** 2))),
        "magnetic_rms_nT": float(np.sqrt(np.mean((g2 @ mag - d2) ** 2))),
        "cross_gradient_norm_history": history,
    }
