"""GIS and spatial analysis on regular grids (DEMs and geophysical rasters)."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra

from .derivatives import compute_thg
from .grid_tools import GridData
from .profile import ProfileData


@dataclass
class RasterImage:
    """RGB image (ny, nx, 3) in [0, 1] georeferenced by an extent (xmin, xmax, ymin, ymax)."""

    rgb: np.ndarray
    extent: tuple[float, float, float, float]
    name: str = "Image"
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- map algebra
_ALLOWED_FUNCS = {name: getattr(np, name) for name in (
    "sqrt", "abs", "log", "log10", "exp", "sin", "cos", "tan", "arctan", "arctan2", "hypot", "minimum", "maximum", "where", "clip")}
_ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.Call, ast.Name, ast.Load, ast.Constant,
                  ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.Gt, ast.Lt, ast.GtE, ast.LtE,
                  ast.Eq, ast.NotEq, ast.And, ast.Or, ast.BitAnd, ast.BitOr, ast.Invert, ast.keyword)


def map_algebra(expression: str, grids: dict[str, GridData], name: str = "MapAlgebra") -> GridData:
    """Evaluate a raster expression such as 'where((A > 5) & (B < 0), A - B, 0)' on same-shape grids (A, B, C, ...).

    Only arithmetic, comparisons and whitelisted numpy functions are allowed (no attribute access, no imports).
    """

    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Disallowed syntax in expression: {type(node).__name__}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS):
            raise ValueError("Only whitelisted numpy functions may be called.")
        if isinstance(node, ast.Name) and node.id not in grids and node.id not in _ALLOWED_FUNCS:
            raise ValueError(f"Unknown name '{node.id}'. Available grids: {', '.join(grids)}")
    shapes = {g.values.shape for g in grids.values()}
    if len(shapes) > 1:
        raise ValueError("All grids in an expression must share one shape.")
    env = {**_ALLOWED_FUNCS, **{k: g.values for k, g in grids.items()}}
    values = eval(compile(tree, "<map_algebra>", "eval"), {"__builtins__": {}}, env)  # noqa: S307 - AST whitelisted above
    ref = next(iter(grids.values()))
    return ref.with_values(np.asarray(values, float) * np.ones(ref.values.shape), name=name, units="",
                           metadata={"method": "map_algebra", "expression": expression})


# ---------------------------------------------------------------- terrain
def terrain_derivatives(dem: GridData) -> dict[str, GridData]:
    """Slope (deg), aspect (deg from north, downslope), profile & plan curvature (Zevenbergen & Thorne, 1987)."""

    z = dem.values
    zy, zx = np.gradient(z, dem.dy, dem.dx)
    zyy, zyx = np.gradient(zy, dem.dy, dem.dx)
    _, zxx = np.gradient(zx, dem.dy, dem.dx)
    p = zx**2 + zy**2
    slope = np.degrees(np.arctan(np.sqrt(p)))
    aspect = (np.degrees(np.arctan2(-zx, -zy)) + 360) % 360
    with np.errstate(divide="ignore", invalid="ignore"):
        prof = -(zxx * zx**2 + 2 * zyx * zx * zy + zyy * zy**2) / (p * (1 + p) ** 1.5)
        plan = -(zxx * zy**2 - 2 * zyx * zx * zy + zyy * zx**2) / (p ** 1.5)
    prof, plan = np.nan_to_num(prof), np.nan_to_num(plan)
    return {
        "slope": dem.with_values(slope, f"{dem.name}_slope", "deg", {"method": "slope"}),
        "aspect": dem.with_values(aspect, f"{dem.name}_aspect", "deg", {"method": "aspect"}),
        "profile_curvature": dem.with_values(prof, f"{dem.name}_profile_curv", "1/m", {"method": "profile_curvature"}),
        "plan_curvature": dem.with_values(plan, f"{dem.name}_plan_curv", "1/m", {"method": "plan_curvature"}),
    }


def hillshade(grid: GridData, azimuth: float = 315.0, altitude: float = 45.0, z_factor: float = 1.0, multidirectional: bool = False) -> GridData:
    """Lambertian hillshade; multidirectional averages azimuths 225/270/315/360 weighted by aspect (Mark, 1992)."""

    zy, zx = np.gradient(grid.values * z_factor, grid.dy, grid.dx)
    slope = np.arctan(np.hypot(zx, zy))
    aspect = np.arctan2(-zx, -zy)
    alt = np.radians(altitude)

    def shade(az):
        return np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(np.radians(az) - aspect), 0, 1)

    if multidirectional:
        azs = np.array([225.0, 270.0, 315.0, 360.0])
        w = np.array([np.sin(aspect - np.radians(a)) ** 2 for a in azs])
        hs = (w * np.array([shade(a) for a in azs])).sum(0) / np.maximum(w.sum(0), 1e-9)
    else:
        hs = shade(azimuth)
    return grid.with_values(hs, f"{grid.name}_hillshade{'_MD' if multidirectional else ''}", "",
                            {"method": "hillshade", "azimuth": azimuth, "altitude": altitude, "multidirectional": multidirectional})


# ---------------------------------------------------------------- lineaments
def extract_lineaments(grid: GridData, threshold_percentile: float = 85.0, min_pixels: int = 8, use_thg: bool = True) -> tuple[GridData, pd.DataFrame]:
    """Ridge (non-maximum suppressed) extraction of gradient maxima, connected-component lineaments with PCA orientation."""

    g = compute_thg(grid).values if use_thg else grid.values
    gy, gx = np.gradient(g)
    ridge = np.zeros_like(g, bool)
    # non-maximum suppression across the local gradient-of-g direction (Canny-like)
    ang = (np.round(np.degrees(np.arctan2(gy, gx)) / 45) * 45) % 180
    offsets = {0: (0, 1), 45: (1, 1), 90: (1, 0), 135: (1, -1)}
    pad = np.pad(g, 1, mode="edge")
    for a, (di, dj) in offsets.items():
        m = ang == a
        fwd = pad[1 + di: 1 + di + g.shape[0], 1 + dj: 1 + dj + g.shape[1]]
        bwd = pad[1 - di: 1 - di + g.shape[0], 1 - dj: 1 - dj + g.shape[1]]
        ridge |= m & (g >= fwd) & (g >= bwd)
    ridge &= g >= np.nanpercentile(g, threshold_percentile)
    labels, n = ndimage.label(ridge, structure=np.ones((3, 3)))
    xx, yy = grid.mesh
    rows = []
    keep = np.zeros_like(ridge)
    for lab in range(1, n + 1):
        m = labels == lab
        if m.sum() < min_pixels:
            continue
        keep |= m
        pts = np.column_stack([xx[m], yy[m]])
        c = pts.mean(0)
        _, s, vt = np.linalg.svd(pts - c, full_matrices=False)
        d = vt[0]
        proj = (pts - c) @ d
        length = float(np.ptp(proj))
        rows.append({"id": lab, "x_center": c[0], "y_center": c[1], "azimuth_deg": float(np.degrees(np.arctan2(d[0], d[1])) % 180),
                     "length_m": length, "x1": c[0] + d[0] * proj.min(), "y1": c[1] + d[1] * proj.min(),
                     "x2": c[0] + d[0] * proj.max(), "y2": c[1] + d[1] * proj.max(),
                     "linearity": float(s[0] / max(s.sum(), 1e-12))})
    return (grid.with_values(keep.astype(float), f"{grid.name}_lineament_pixels", "binary", {"method": "lineament_extraction"}),
            pd.DataFrame(rows))


def lineament_density(lineaments: pd.DataFrame, reference: GridData, radius: float = 5000.0) -> GridData:
    """Length of lineaments per unit area (km/km^2) within a circular search radius."""

    xx, yy = reference.mesh
    dens = np.zeros_like(xx)
    for _, r in lineaments.iterrows():
        n = max(int(r.length_m / min(reference.dx, reference.dy)) + 2, 2)
        seg = r.length_m / (n - 1) / 1000
        for x, y in zip(np.linspace(r.x1, r.x2, n), np.linspace(r.y1, r.y2, n)):
            dens += seg * ((xx - x) ** 2 + (yy - y) ** 2 <= radius**2)
    area = np.pi * (radius / 1000) ** 2
    return reference.with_values(dens / area, f"{reference.name}_lineament_density", "km/km^2",
                                 {"method": "lineament_density", "radius_m": radius})


def rose_diagram(lineaments: pd.DataFrame, bin_deg: float = 10.0, weight_by_length: bool = True) -> tuple[ProfileData, dict]:
    """Axial (0-180 deg) azimuth frequency histogram of lineaments."""

    az = lineaments["azimuth_deg"].to_numpy(float) % 180
    w = lineaments["length_m"].to_numpy(float) if weight_by_length else np.ones_like(az)
    edges = np.arange(0, 180 + bin_deg, bin_deg)
    hist, _ = np.histogram(az, edges, weights=w)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # circular mean for axial data (doubling angles)
    ang2 = np.radians(2 * az)
    mean_dir = (np.degrees(np.arctan2((w * np.sin(ang2)).sum(), (w * np.cos(ang2)).sum())) / 2) % 180
    return (ProfileData(centers, hist, name="rose_diagram", units="length" if weight_by_length else "count",
                        metadata={"method": "rose_diagram", "plot": "rose", "bin_deg": bin_deg}),
            {"dominant_azimuth_deg": float(centers[np.argmax(hist)]), "circular_mean_azimuth_deg": float(mean_dir), "n": int(len(az))})


# ---------------------------------------------------------------- prospectivity
def weights_of_evidence(evidence: GridData, deposits: pd.DataFrame, x_col: str = "x", y_col: str = "y", threshold: float = 0.5,
                        unit_cell_km2: float = 1.0) -> tuple[GridData, dict]:
    """Binary weights of evidence (Bonham-Carter et al., 1989): W+, W-, contrast C and posterior probability."""

    b = evidence.values >= threshold
    ix = np.clip(np.round((deposits[x_col].to_numpy(float) - evidence.x_vector[0]) / evidence.dx).astype(int), 0, b.shape[1] - 1)
    iy = np.clip(np.round((deposits[y_col].to_numpy(float) - evidence.y_vector[0]) / evidence.dy).astype(int), 0, b.shape[0] - 1)
    cell_area = evidence.dx * evidence.dy / 1e6 / unit_cell_km2
    n_t = b.size * cell_area
    n_b = b.sum() * cell_area
    d = np.zeros_like(b)
    np.add.at(d, (iy, ix), 1)
    n_d = float(d.sum())
    n_bd = float(d[b].sum())
    eps = 0.5
    p_b_d = (n_bd + eps) / (n_d + 2 * eps)
    p_b_nd = (n_b - n_bd + eps) / (n_t - n_d + 2 * eps)
    w_plus = np.log(p_b_d / p_b_nd)
    w_minus = np.log((1 - p_b_d) / (1 - p_b_nd))
    prior = n_d / n_t
    logit = np.log(prior / (1 - prior)) + np.where(b, w_plus, w_minus)
    post = 1 / (1 + np.exp(-logit))
    s_c = np.sqrt(1 / (n_bd + eps) + 1 / (n_b - n_bd + eps) + 1 / (n_d - n_bd + eps) + 1 / (n_t - n_b - n_d + n_bd + eps))
    return (evidence.with_values(post, f"{evidence.name}_WofE_posterior", "probability", {"method": "weights_of_evidence"}),
            {"W_plus": float(w_plus), "W_minus": float(w_minus), "contrast_C": float(w_plus - w_minus),
             "studentized_contrast": float((w_plus - w_minus) / s_c), "prior_probability": float(prior), "deposits": int(n_d)})


def _normalize(v: np.ndarray, invert: bool = False) -> np.ndarray:
    lo, hi = np.nanmin(v), np.nanmax(v)
    out = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    return 1 - out if invert else out


def fuzzy_membership(grid: GridData, kind: str = "linear", midpoint: float | None = None, spread: float = 5.0, invert: bool = False) -> np.ndarray:
    if kind == "linear":
        return _normalize(grid.values, invert)
    mid = np.nanmedian(grid.values) if midpoint is None else midpoint
    scale = np.nanstd(grid.values) or 1.0
    mu = 1 / (1 + np.exp(-spread * (grid.values - mid) / scale))
    return 1 - mu if invert else mu


def fuzzy_overlay(grids: list[GridData], operator: str = "gamma", gamma: float = 0.9, membership: str = "linear") -> GridData:
    """Fuzzy logic overlay (Bonham-Carter, 1994): AND=min, OR=max, product, algebraic sum, gamma."""

    mu = np.array([fuzzy_membership(g, membership) for g in grids])
    if operator == "and":
        out = mu.min(0)
    elif operator == "or":
        out = mu.max(0)
    elif operator == "product":
        out = mu.prod(0)
    elif operator == "sum":
        out = 1 - (1 - mu).prod(0)
    else:
        out = (1 - (1 - mu).prod(0)) ** gamma * mu.prod(0) ** (1 - gamma)
    return grids[0].with_values(out, f"fuzzy_{operator}", "membership", {"method": "fuzzy_overlay", "operator": operator, "gamma": gamma})


RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def ahp_weights(pairwise: str) -> dict:
    """Saaty AHP: principal eigenvector weights and consistency ratio from 'a,b,c; d,e,f; ...' (rows)."""

    mat = np.array([[float(eval(v, {"__builtins__": {}})) for v in row.split(",")] for row in pairwise.split(";") if row.strip()])  # noqa: S307 - numbers/fractions only
    n = mat.shape[0]
    if mat.shape != (n, n):
        raise ValueError("Pairwise matrix must be square.")
    vals, vecs = np.linalg.eig(mat)
    i = int(np.argmax(vals.real))
    w = np.abs(vecs[:, i].real)
    w /= w.sum()
    ci = (vals[i].real - n) / (n - 1) if n > 1 else 0.0
    cr = ci / RANDOM_INDEX.get(n, 1.49) if RANDOM_INDEX.get(n, 1.49) > 0 else 0.0
    return {"weights": w.tolist(), "lambda_max": float(vals[i].real), "consistency_ratio": float(cr), "consistent": bool(cr < 0.1)}


def ahp_overlay(grids: list[GridData], pairwise: str, invert: str = "") -> tuple[GridData, dict]:
    """MCDA weighted linear combination with AHP weights on min-max normalized layers ('invert' = comma list of layer indices)."""

    info = ahp_weights(pairwise)
    if len(info["weights"]) != len(grids):
        raise ValueError(f"Pairwise matrix is {len(info['weights'])}x{len(info['weights'])} but {len(grids)} grids were given.")
    inv = {int(v) for v in invert.split(",") if v.strip()}
    score = sum(w * _normalize(g.values, i in inv) for i, (w, g) in enumerate(zip(info["weights"], grids)))
    return grids[0].with_values(score, "AHP_suitability", "score", {"method": "ahp_mcda", **info}), info


# ---------------------------------------------------------------- georeferencing
def fit_georeference(gcps: pd.DataFrame, col: str = "col", row: str = "row", x: str = "x", y: str = "y", order: int = 1) -> dict:
    """Least-squares affine (order 1) or 2nd-order polynomial transform from pixel (col,row) to world (x,y)."""

    c, r = gcps[col].to_numpy(float), gcps[row].to_numpy(float)
    terms = [np.ones_like(c), c, r] + ([c * r, c**2, r**2] if order == 2 else [])
    a = np.column_stack(terms)
    if len(c) < a.shape[1]:
        raise ValueError(f"Need at least {a.shape[1]} GCPs for order {order}.")
    cx = np.linalg.lstsq(a, gcps[x].to_numpy(float), rcond=None)[0]
    cy = np.linalg.lstsq(a, gcps[y].to_numpy(float), rcond=None)[0]
    res = np.hypot(a @ cx - gcps[x], a @ cy - gcps[y])
    return {"order": order, "coef_x": cx.tolist(), "coef_y": cy.tolist(), "rms_m": float(np.sqrt(np.mean(res**2))), "residuals": res.tolist()}


def rectify_image(image: GridData, gcps: pd.DataFrame, spacing: float, order: int = 1) -> tuple[GridData, dict]:
    """Georeference a raster given in pixel coordinates (x = column, y = row) and resample onto a world grid."""

    fwd = fit_georeference(gcps, order=order)
    inv = fit_georeference(gcps.rename(columns={"col": "x", "row": "y", "x": "col", "y": "row"}), order=order)
    ny, nx = image.values.shape
    cc, rr = np.meshgrid(np.arange(nx), np.arange(ny))
    t = lambda coef, c, r: sum(k * v for k, v in zip(coef, [np.ones_like(c), c, r] + ([c * r, c**2, r**2] if order == 2 else [])))  # noqa: E731
    wx, wy = t(fwd["coef_x"], cc, rr), t(fwd["coef_y"], cc, rr)
    gx = np.arange(wx.min(), wx.max() + spacing / 2, spacing)
    gy = np.arange(wy.min(), wy.max() + spacing / 2, spacing)
    xx, yy = np.meshgrid(gx, gy)
    col, row = t(inv["coef_x"], xx, yy), t(inv["coef_y"], xx, yy)
    vals = ndimage.map_coordinates(image.values, [row, col], order=1, cval=np.nan)
    return GridData(vals, gx, gy, spacing, spacing, name=f"{image.name}_georef", units=image.units, metadata={"method": "georeference", **fwd}), fwd


# ---------------------------------------------------------------- spatial statistics & proximity
def zonal_statistics(values: GridData, zones: GridData) -> pd.DataFrame:
    z = np.round(zones.values).astype(int)
    v = values.values
    rows = []
    for zone in np.unique(z):
        m = (z == zone) & np.isfinite(v)
        if m.any():
            rows.append({"zone": int(zone), "count": int(m.sum()), "mean": float(v[m].mean()), "std": float(v[m].std()),
                         "min": float(v[m].min()), "max": float(v[m].max()), "median": float(np.median(v[m])),
                         "area_km2": float(m.sum() * values.dx * values.dy / 1e6)})
    return pd.DataFrame(rows)


def euclidean_distance(features: GridData, threshold: float = 0.5) -> GridData:
    """Distance (m) from each cell to the nearest feature cell (value >= threshold)."""

    mask = features.values >= threshold
    if not mask.any():
        raise ValueError("No feature cells above the threshold.")
    d = ndimage.distance_transform_edt(~mask, sampling=(features.dy, features.dx))
    return features.with_values(d, f"{features.name}_distance", "m", {"method": "euclidean_distance"})


def buffer_zone(features: GridData, distance: float, threshold: float = 0.5) -> GridData:
    d = euclidean_distance(features, threshold).values
    return features.with_values((d <= distance).astype(float), f"{features.name}_buffer_{distance:g}m", "binary",
                                {"method": "buffer", "distance_m": distance})


def buffer_points(points: pd.DataFrame, reference: GridData, distance: float, x_col: str = "x", y_col: str = "y") -> GridData:
    xx, yy = reference.mesh
    inside = np.zeros_like(xx, bool)
    for x, y in zip(points[x_col], points[y_col]):
        inside |= (xx - x) ** 2 + (yy - y) ** 2 <= distance**2
    return reference.with_values(inside.astype(float), f"points_buffer_{distance:g}m", "binary", {"method": "buffer_points"})


def _weights_kernel(grid: GridData, distance: float) -> np.ndarray:
    ri, rj = int(np.ceil(distance / grid.dy)), int(np.ceil(distance / grid.dx))
    ii, jj = np.mgrid[-ri:ri + 1, -rj:rj + 1]
    return ((ii * grid.dy) ** 2 + (jj * grid.dx) ** 2 <= distance**2).astype(float)


def morans_i(grid: GridData, distance: float | None = None) -> dict:
    """Global Moran's I with binary distance-band weights and normality z-score."""

    distance = distance or 1.5 * max(grid.dx, grid.dy)
    v = grid.values
    m = np.isfinite(v)
    z = np.where(m, v - v[m].mean(), 0.0)
    k = _weights_kernel(grid, distance)
    k[k.shape[0] // 2, k.shape[1] // 2] = 0
    lag = ndimage.convolve(z, k, mode="constant")
    wsum_cell = ndimage.convolve(m.astype(float), k, mode="constant") * m
    s0 = wsum_cell.sum()
    n = m.sum()
    i_val = n / s0 * (z * lag)[m].sum() / (z[m] ** 2).sum()
    e_i = -1 / (n - 1)
    s1, s2 = 2 * s0, 4 * (wsum_cell**2).sum()  # symmetric binary weights
    var = (n**2 * s1 - n * s2 + 3 * s0**2) / ((n**2 - 1) * s0**2) - e_i**2  # normality assumption (Cliff & Ord, 1981)
    return {"morans_I": float(i_val), "expected_I": e_i, "z_score": float((i_val - e_i) / np.sqrt(max(var, 1e-30))), "distance_m": distance}


def getis_ord_gi_star(grid: GridData, distance: float | None = None) -> GridData:
    """Getis-Ord Gi* z-scores with binary distance-band weights (self included)."""

    distance = distance or 2.5 * max(grid.dx, grid.dy)
    v = grid.values
    m = np.isfinite(v)
    x = np.where(m, v, 0.0)
    n = m.sum()
    xbar = x[m].mean()
    s = np.sqrt((x[m] ** 2).mean() - xbar**2)
    k = _weights_kernel(grid, distance)
    wx = ndimage.convolve(x, k, mode="constant")
    w = ndimage.convolve(m.astype(float), k, mode="constant")
    w2 = w  # binary weights: sum w^2 = sum w
    z = (wx - xbar * w) / (s * np.sqrt(np.maximum((n * w2 - w**2) / (n - 1), 1e-30)))
    return grid.with_values(np.where(m, z, np.nan), f"{grid.name}_GiStar", "z-score", {"method": "getis_ord_gi_star", "distance_m": distance})


def trend_surface(grid: GridData, order: int = 2) -> tuple[GridData, GridData]:
    """Least-squares polynomial trend surface (regional) and residual."""

    xx, yy = grid.mesh
    xs = (xx - xx.mean()) / max(np.ptp(xx), 1e-12)
    ys = (yy - yy.mean()) / max(np.ptp(yy), 1e-12)
    terms = [xs**i * ys**j for i in range(order + 1) for j in range(order + 1 - i)]
    a = np.column_stack([t.ravel() for t in terms])
    m = np.isfinite(grid.values.ravel())
    coef = np.linalg.lstsq(a[m], grid.values.ravel()[m], rcond=None)[0]
    trend = (a @ coef).reshape(grid.values.shape)
    return (grid.with_values(trend, f"{grid.name}_trend{order}", metadata={"method": "trend_surface", "order": order}),
            grid.with_values(grid.values - trend, f"{grid.name}_trend{order}_residual", metadata={"method": "trend_residual", "order": order}))


def ternary_rgb(red: GridData, green: GridData, blue: GridData, clip_percent: float = 2.0) -> RasterImage:
    """RGB ternary composite (e.g. K-Th-U) with per-channel percentile stretch."""

    def stretch(v):
        lo, hi = np.nanpercentile(v, [clip_percent, 100 - clip_percent])
        return np.clip((v - lo) / max(hi - lo, 1e-12), 0, 1)

    rgb = np.dstack([stretch(red.values), stretch(green.values), stretch(blue.values)])
    x, y = red.x_vector, red.y_vector
    return RasterImage(np.nan_to_num(rgb), (float(x.min()), float(x.max()), float(y.min()), float(y.max())),
                       name=f"RGB_{red.name}_{green.name}_{blue.name}", metadata={"method": "ternary_rgb"})


def contour_lines(grid: GridData, levels: int | str = 10) -> pd.DataFrame:
    """Isolines as vertex table (level, line_id, x, y) via matplotlib's marching squares."""

    from matplotlib.figure import Figure

    lv = [float(v) for v in levels.split(",")] if isinstance(levels, str) and "," in levels else int(levels)
    fig = Figure()
    cs = fig.add_subplot(111).contour(grid.x_vector, grid.y_vector, grid.values, lv)
    rows, line_id = [], 0
    for level, segs in zip(cs.levels, cs.allsegs):
        for seg in segs:
            line_id += 1
            rows.extend({"level": float(level), "line_id": line_id, "x": float(px), "y": float(py)} for px, py in seg)
    return pd.DataFrame(rows, columns=["level", "line_id", "x", "y"])


def viewshed(dem: GridData, observer_x: float, observer_y: float, observer_height: float = 2.0, target_height: float = 0.0,
             samples: int = 200) -> GridData:
    """Line-of-sight visibility: a cell is visible if no intermediate terrain rises above the sight line."""

    from scipy.interpolate import RegularGridInterpolator

    f = RegularGridInterpolator((dem.y_vector, dem.x_vector), dem.values, bounds_error=False, fill_value=-np.inf)
    z0 = float(f([[observer_y, observer_x]])[0]) + observer_height
    xx, yy = dem.mesh
    tz = dem.values + target_height
    t = np.linspace(0, 1, samples)[1:-1]
    px = observer_x + t[:, None, None] * (xx - observer_x)
    py = observer_y + t[:, None, None] * (yy - observer_y)
    terrain = f(np.column_stack([py.ravel(), px.ravel()])).reshape(px.shape)
    sight = z0 + t[:, None, None] * (tz - z0)
    visible = ~(terrain > sight + 1e-6).any(axis=0)
    return dem.with_values(visible.astype(float), f"{dem.name}_viewshed", "binary",
                           {"method": "viewshed", "observer": (observer_x, observer_y, observer_height)})


def least_cost_path(cost: GridData, start_x: float, start_y: float, end_x: float, end_y: float) -> tuple[pd.DataFrame, dict]:
    """8-connected Dijkstra least-cost path; edge cost = mean cell cost x step length."""

    ny, nx = cost.values.shape
    c = np.where(np.isfinite(cost.values), cost.values, 1e12)
    idx = np.arange(nx * ny).reshape(ny, nx)
    rows, cols, w = [], [], []
    for di, dj in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        jlo, jhi = max(0, -dj), nx - max(0, dj)
        a = idx[: ny - di, jlo:jhi]
        b = idx[di:, jlo + dj: jhi + dj]
        step = np.hypot(di * cost.dy, dj * cost.dx)
        rows.append(a.ravel())
        cols.append(b.ravel())
        w.append(0.5 * (c.ravel()[a.ravel()] + c.ravel()[b.ravel()]) * step)
    graph = sparse.csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny,) * 2)
    node = lambda x, y: int(np.argmin(np.abs(cost.y_vector - y))) * nx + int(np.argmin(np.abs(cost.x_vector - x)))  # noqa: E731
    s, e = node(start_x, start_y), node(end_x, end_y)
    dist, pred = dijkstra(graph, directed=False, indices=s, return_predecessors=True)
    path = [e]
    while path[-1] != s and pred[path[-1]] >= 0:
        path.append(pred[path[-1]])
    path = path[::-1]
    xx, yy = cost.mesh
    return (pd.DataFrame({"x": xx.ravel()[path], "y": yy.ravel()[path]}),
            {"total_cost": float(dist[e]), "cells": len(path)})


def cut_fill(surface_top: GridData, surface_base: GridData) -> tuple[GridData, dict]:
    """Volumes between two surfaces: positive = top above base (fill/volume), negative = cut."""

    diff = surface_top.values - surface_base.values
    area = surface_top.dx * surface_top.dy
    return (surface_top.with_values(diff, "surface_difference", "m", {"method": "cut_fill"}),
            {"volume_above_m3": float(np.nansum(np.clip(diff, 0, None)) * area), "volume_below_m3": float(-np.nansum(np.clip(diff, None, 0)) * area),
             "net_volume_m3": float(np.nansum(diff) * area)})


_D8 = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]


def fill_depressions(dem: GridData) -> np.ndarray:
    """Priority-flood depression filling (Barnes et al., 2014)."""

    import heapq

    z = dem.values.copy()
    ny, nx = z.shape
    done = np.zeros_like(z, bool)
    heap = []
    for i in range(ny):
        for j in (0, nx - 1):
            heapq.heappush(heap, (z[i, j], i, j))
            done[i, j] = True
    for j in range(nx):
        for i in (0, ny - 1):
            if not done[i, j]:
                heapq.heappush(heap, (z[i, j], i, j))
                done[i, j] = True
    while heap:
        h, i, j = heapq.heappop(heap)
        for di, dj in _D8:
            a, b = i + di, j + dj
            if 0 <= a < ny and 0 <= b < nx and not done[a, b]:
                done[a, b] = True
                z[a, b] = max(z[a, b], h + 1e-6)
                heapq.heappush(heap, (z[a, b], a, b))
    return z


def flow_accumulation(dem: GridData, pour_x: float | None = None, pour_y: float | None = None) -> dict[str, GridData]:
    """D8 flow direction and accumulation (O'Callaghan & Mark, 1984) on a depression-filled DEM; optional watershed."""

    z = fill_depressions(dem)
    ny, nx = z.shape
    pad = np.pad(z, 1, constant_values=np.inf)
    drops = []
    for di, dj in _D8:
        dist = np.hypot(di * dem.dy, dj * dem.dx)
        nb = pad[1 + di: 1 + di + ny, 1 + dj: 1 + dj + nx]
        drops.append((z - nb) / dist)
    drops = np.array(drops)
    direction = np.where(drops.max(0) > 0, drops.argmax(0), -1)
    acc = np.ones_like(z)
    order = np.argsort(z, axis=None)[::-1]
    for flat in order:
        i, j = divmod(int(flat), nx)
        d = direction[i, j]
        if d >= 0:
            a, b = i + _D8[d][0], j + _D8[d][1]
            if 0 <= a < ny and 0 <= b < nx:
                acc[a, b] += acc[i, j]
    out = {"filled": dem.with_values(z, f"{dem.name}_filled", metadata={"method": "priority_flood"}),
           "flow_direction": dem.with_values(direction.astype(float), f"{dem.name}_D8_dir", "code", {"method": "d8"}),
           "flow_accumulation": dem.with_values(np.log10(acc), f"{dem.name}_flow_acc_log10", "log10(cells)", {"method": "d8_accumulation"})}
    if pour_x is not None and pour_y is not None:
        pi = int(np.argmin(np.abs(dem.y_vector - pour_y)))
        pj = int(np.argmin(np.abs(dem.x_vector - pour_x)))
        basin = np.zeros_like(z, bool)
        basin[pi, pj] = True
        stack = [(pi, pj)]
        while stack:
            i, j = stack.pop()
            for k, (di, dj) in enumerate(_D8):
                a, b = i - di, j - dj
                if 0 <= a < ny and 0 <= b < nx and not basin[a, b] and direction[a, b] == k:
                    basin[a, b] = True
                    stack.append((a, b))
        out["watershed"] = dem.with_values(basin.astype(float), f"{dem.name}_watershed", "binary", {"method": "watershed"})
    return out


def vector_to_raster(polygons: pd.DataFrame, reference: GridData, id_col: str = "id", x_col: str = "x", y_col: str = "y",
                     value_col: str = "") -> GridData:
    """Rasterize polygons given as ordered vertex rows (id, x, y[, value]); later polygons overwrite earlier ones."""

    xx, yy = reference.mesh
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    out = np.full(xx.size, np.nan)
    for pid, grp in polygons.groupby(id_col, sort=False):
        inside = MplPath(grp[[x_col, y_col]].to_numpy(float)).contains_points(pts)
        out[inside] = float(grp[value_col].iloc[0]) if value_col else float(pid)
    return reference.with_values(out.reshape(xx.shape), "rasterized_polygons", "class", {"method": "vector_to_raster"})


def raster_to_vector(classes: GridData) -> pd.DataFrame:
    """Polygon boundaries of each class value as vertex rows (class, ring_id, x, y)."""

    rows, ring = [], 0
    from matplotlib.figure import Figure

    for value in np.unique(classes.values[np.isfinite(classes.values)]):
        mask = (classes.values == value).astype(float)
        cs = Figure().add_subplot(111).contour(classes.x_vector, classes.y_vector, mask, [0.5])
        for seg in cs.allsegs[0]:
            ring += 1
            rows.extend({"class": float(value), "ring_id": ring, "x": float(x), "y": float(y)} for x, y in seg)
    return pd.DataFrame(rows, columns=["class", "ring_id", "x", "y"])
