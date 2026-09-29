"""Gravity reductions: normal gravity, free-air, Bouguer, terrain, isostatic, drift and Earth tide."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .fourier import apply_response
from .grid_tools import GridData
from .prisms import SI_TO_MGAL, gz_kernel

G = 6.674e-11

ELLIPSOIDS = {
    # gamma_e (m/s^2), k, e^2  -- Somigliana closed form (Moritz, 1980; NIMA 2000)
    "GRS80": (9.7803267715, 0.001931851353, 0.00669438002290),
    "WGS84": (9.7803253359, 0.00193185265241, 0.00669437999013),
}


def normal_gravity(latitude_deg, ellipsoid: str = "GRS80") -> np.ndarray:
    """Somigliana normal gravity on the ellipsoid (mGal)."""

    ge, k, e2 = ELLIPSOIDS[ellipsoid]
    s2 = np.sin(np.radians(np.asarray(latitude_deg, float))) ** 2
    return ge * (1 + k * s2) / np.sqrt(1 - e2 * s2) * SI_TO_MGAL


def free_air_correction(height_m, latitude_deg=45.0, second_order: bool = True) -> np.ndarray:
    """Free-air correction (mGal, add to observed). Second order: Hinze et al. (2005)."""

    h = np.asarray(height_m, float)
    if not second_order:
        return 0.3086 * h
    s2 = np.sin(np.radians(np.asarray(latitude_deg, float))) ** 2
    return (0.3087691 - 0.0004398 * s2) * h - 7.2125e-8 * h**2


def bouguer_slab(height_m, density: float = 2670.0) -> np.ndarray:
    """Infinite-slab Bouguer correction 2*pi*G*rho*h (mGal)."""

    return 2 * np.pi * G * density * np.asarray(height_m, float) * SI_TO_MGAL


def curvature_correction(height_m, density: float = 2670.0) -> np.ndarray:
    """Bullard B spherical-cap curvature correction to 166.7 km (LaFehr, 1991), scaled by density (mGal)."""

    h = np.asarray(height_m, float)
    return (1.464e-3 * h - 3.533e-7 * h**2 + 4.5e-14 * h**3) * density / 2670.0


def gravity_anomalies(
    table: pd.DataFrame,
    lat_col: str = "lat",
    height_col: str = "elevation",
    gravity_col: str = "gravity",
    density: float = 2670.0,
    ellipsoid: str = "GRS80",
    second_order_free_air: bool = True,
    curvature: bool = False,
    terrain_col: str = "",
) -> pd.DataFrame:
    """Free-air, simple and complete Bouguer anomalies (mGal) for station rows.

    FA = g_obs - gamma + FAC(h);  SBA = FA - 2 pi G rho h [- BB];  CBA = SBA + TC.
    """

    lat = table[lat_col].to_numpy(float)
    h = table[height_col].to_numpy(float)
    g = table[gravity_col].to_numpy(float)
    out = table.copy()
    out["normal_gravity_mGal"] = normal_gravity(lat, ellipsoid)
    out["free_air_correction_mGal"] = free_air_correction(h, lat, second_order_free_air)
    out["free_air_anomaly_mGal"] = g - out["normal_gravity_mGal"] + out["free_air_correction_mGal"]
    out["bouguer_correction_mGal"] = bouguer_slab(h, density) + (curvature_correction(h, density) if curvature else 0.0)
    out["simple_bouguer_anomaly_mGal"] = out["free_air_anomaly_mGal"] - out["bouguer_correction_mGal"]
    if terrain_col:
        out["complete_bouguer_anomaly_mGal"] = out["simple_bouguer_anomaly_mGal"] + out[terrain_col].to_numpy(float)
    return out


def terrain_correction_points(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, dem: GridData, density: float = 2670.0, radius: float = 20000.0
) -> np.ndarray:
    """Terrain correction (mGal, always >= 0) from DEM prisms between station height and cell height.

    Each DEM cell inside `radius` is a prism from the station elevation to the cell elevation.
    Hills above and valleys below both reduce observed gravity relative to the Bouguer slab,
    so the correction is sum |gz| (Hammer, 1939 logic, exact prism kernels; the station's own
    cell is skipped - inner-zone correction should be added from field notes if significant).
    """

    xx, yy = dem.mesh
    cx, cy, ch = xx.ravel(), yy.ravel(), dem.values.ravel()
    tc = np.zeros(len(x))
    for i, (xs, ys, zs) in enumerate(zip(x, y, z)):
        d = np.hypot(cx - xs, cy - ys)
        sel = (d <= radius) & (d > 0.5 * min(dem.dx, dem.dy)) & np.isfinite(ch)
        if not sel.any():
            continue
        hi = np.maximum(ch[sel], zs)
        lo = np.minimum(ch[sel], zs)
        prisms = np.column_stack([cx[sel] - dem.dx / 2, cx[sel] + dem.dx / 2, cy[sel] - dem.dy / 2, cy[sel] + dem.dy / 2,
                                  zs - hi, zs - lo])  # z down relative to station
        thick = prisms[:, 5] - prisms[:, 4] > 1e-6
        if thick.any():
            tc[i] = np.abs(gz_kernel(np.zeros((1, 3)), prisms[thick])[0]).sum() * density * SI_TO_MGAL
    return tc


def terrain_correction_table(
    table: pd.DataFrame, dem: GridData, x_col: str = "x", y_col: str = "y", height_col: str = "elevation",
    density: float = 2670.0, radius: float = 20000.0,
) -> pd.DataFrame:
    out = table.copy()
    out["terrain_correction_mGal"] = terrain_correction_points(
        table[x_col].to_numpy(float), table[y_col].to_numpy(float), table[height_col].to_numpy(float), dem, density, radius)
    return out


def terrain_correction_grid(dem: GridData, density: float = 2670.0, radius: float = 10000.0, stride: int = 1) -> GridData:
    """Terrain correction evaluated at DEM nodes (station on the topographic surface)."""

    xv, yv = dem.x_vector[::stride], dem.y_vector[::stride]
    xx, yy = np.meshgrid(xv, yv)
    zz = dem.values[::stride, ::stride]
    tc = terrain_correction_points(xx.ravel(), yy.ravel(), zz.ravel(), dem, density, radius)
    return GridData(tc.reshape(xx.shape), xv, yv, dem.dx * stride, dem.dy * stride, name=f"{dem.name}_TerrainCorr",
                    units="mGal", crs=dem.crs,
                    metadata={"method": "terrain_correction", "density": density, "radius_m": radius,
                              "formula": "TC = sum |gz(prism between station and DEM cell)|"})


def isostatic_residual(
    bouguer: GridData,
    topography: GridData,
    crust_density: float = 2670.0,
    mantle_density: float = 3270.0,
    water_density: float = 1030.0,
    compensation_depth: float = 30000.0,
    padding: str = "reflect",
) -> tuple[GridData, GridData, GridData]:
    """Airy-Heiskanen isostatic residual: IR = BA - g_root.

    Root r = h rho_c/(rho_m - rho_c) on land, anti-root r = h (rho_c - rho_w)/(rho_m - rho_c) at sea (h<0).
    g_root (Parker first term): F[g] = -2 pi G (rho_m - rho_c) e^{-|k| Tc} F[r].
    """

    if bouguer.values.shape != topography.values.shape:
        raise ValueError("Bouguer and topography grids must have the same shape.")
    h = topography.values
    contrast = mantle_density - crust_density
    root = np.where(h >= 0, h * crust_density / contrast, h * (crust_density - water_density) / contrast)
    root_grid = topography.with_values(root, name=f"{topography.name}_AiryRoot", units="m",
                                       metadata={"method": "airy_root"})
    g_root = -2 * np.pi * G * contrast * SI_TO_MGAL * apply_response(
        root_grid, lambda kx, ky, k: np.exp(-k * compensation_depth), padding=padding)
    regional = bouguer.with_values(g_root, name=f"{bouguer.name}_IsostaticRegional", units="mGal",
                                   metadata={"method": "airy_isostatic_regional", "compensation_depth_m": compensation_depth})
    residual = bouguer.with_values(bouguer.values - g_root, name=f"{bouguer.name}_IsostaticResidual", units="mGal",
                                   metadata={"method": "isostatic_residual", "formula": "IR = BA - g_root(Airy)",
                                             "crust_density": crust_density, "mantle_density": mantle_density})
    return residual, regional, root_grid


def longman_tide(times_utc, latitude_deg: float, longitude_deg: float, elevation_m: float = 0.0) -> np.ndarray:
    """Luni-solar Earth-tide gravity correction (mGal) after Longman (1959), Love factor 1.16.

    Returns the correction to ADD to observed readings.
    """

    mu, mass_moon, mass_sun = 6.674e-8, 7.3537e25, 1.993e33
    e, m, c, c1 = 0.05490, 0.074804, 3.84402e10, 1.495e13
    a_earth, incl_moon, omega = 6.378270e8, 0.08979719, np.radians(23.452)
    love = 1.16
    times = pd.to_datetime(pd.Series(times_utc), utc=True)
    epoch = pd.Timestamp("1899-12-31 12:00:00", tz="UTC")
    tt = ((times - epoch).dt.total_seconds() / 86400.0 / 36525.0).to_numpy()
    hours = (times.dt.hour + times.dt.minute / 60 + times.dt.second / 3600).to_numpy()
    phi = np.radians(latitude_deg)
    lam = np.radians(longitude_deg)

    s = 4.72000889397 + 8399.70927456 * tt + 3.45575191895e-5 * tt**2 + 3.49065850637e-8 * tt**3
    p = 5.83515162814 + 71.0180412089 * tt + 1.80108282532e-4 * tt**2 + 1.74532925199e-7 * tt**3
    h = 4.88162798259 + 628.331950894 * tt + 5.23598775598e-6 * tt**2
    n = 4.52360161181 - 33.757146295 * tt + 3.6264063347e-5 * tt**2 + 3.39369576777e-8 * tt**3
    p1 = 4.90822941839 + 3.0005264e-2 * tt + 7.9024630e-6 * tt**2 + 5.81776417e-9 * tt**3
    e1 = 0.01675104 - 4.180e-5 * tt - 1.26e-7 * tt**2

    big_i = np.arccos(np.cos(omega) * np.cos(incl_moon) - np.sin(omega) * np.sin(incl_moon) * np.cos(n))
    nu = np.arcsin(np.sin(incl_moon) * np.sin(n) / np.sin(big_i))
    t = np.radians(15.0 * (hours - 12.0)) + lam
    chi = t + h - nu
    cos_alpha = np.cos(n) * np.cos(nu) + np.sin(n) * np.sin(nu) * np.cos(omega)
    sin_alpha = np.sin(omega) * np.sin(n) / np.sin(big_i)
    alpha = 2 * np.arctan(sin_alpha / (1 + cos_alpha))
    xi = n - alpha
    sigma = s - xi
    ll = (sigma + 2 * e * np.sin(s - p) + 1.25 * e**2 * np.sin(2 * (s - p)) + 3.75 * m * e * np.sin(s - 2 * h + p)
          + 11 / 8 * m**2 * np.sin(2 * (s - h)))
    chi1 = t + h
    l1 = h + 2 * e1 * np.sin(h - p1)
    cos_theta = (np.sin(phi) * np.sin(big_i) * np.sin(ll)
                 + np.cos(phi) * (np.cos(big_i / 2) ** 2 * np.cos(ll - chi) + np.sin(big_i / 2) ** 2 * np.cos(ll + chi)))
    cos_phi_sun = (np.sin(phi) * np.sin(omega) * np.sin(l1)
                   + np.cos(phi) * (np.cos(omega / 2) ** 2 * np.cos(l1 - chi1) + np.sin(omega / 2) ** 2 * np.cos(l1 + chi1)))
    cgeo = np.sqrt(1.0 / (1 + 0.006738 * np.sin(phi) ** 2))
    r = cgeo * a_earth + elevation_m * 100.0
    ap = 1 / (c * (1 - e**2))
    ap1 = 1 / (c1 * (1 - e1**2))
    inv_d = 1 / c + ap * e * np.cos(s - p) + ap * e**2 * np.cos(2 * (s - p)) + 15 / 8 * ap * m * e * np.cos(s - 2 * h + p) \
        + ap * m**2 * np.cos(2 * (s - h))
    inv_dd = 1 / c1 + ap1 * e1 * np.cos(h - p1)
    gm = mu * mass_moon * r * inv_d**3 * (3 * cos_theta**2 - 1) + 1.5 * mu * mass_moon * r**2 * inv_d**4 * (
        5 * cos_theta**3 - 3 * cos_theta)
    gs = mu * mass_sun * r * inv_dd**3 * (3 * cos_phi_sun**2 - 1)
    return (gm + gs) * 1e3 * love  # gal -> mGal


def drift_tide_correction(
    table: pd.DataFrame,
    time_col: str = "time",
    station_col: str = "station",
    reading_col: str = "reading",
    base_station: str = "BASE",
    latitude: float = 0.0,
    longitude: float = 0.0,
    apply_tide: bool = True,
) -> pd.DataFrame:
    """Microgravity reduction: add Longman tide, then remove instrument drift interpolated between base readings.

    Drift is modelled piecewise-linearly from repeated base-station occupations:
    g_corr = reading + tide - (base(t) - base(t0)).
    """

    out = table.copy()
    times = pd.to_datetime(out[time_col], utc=True)
    out["tide_mGal"] = longman_tide(times, latitude, longitude) if apply_tide else 0.0
    corrected = out[reading_col].to_numpy(float) + out["tide_mGal"].to_numpy(float)
    is_base = out[station_col].astype(str) == str(base_station)
    if is_base.sum() < 2:
        raise ValueError("At least two base-station readings are required for drift correction.")
    tsec = (times - times.min()).dt.total_seconds().to_numpy()
    base_t, base_g = tsec[is_base.to_numpy()], corrected[is_base.to_numpy()]
    drift = np.interp(tsec, base_t, base_g - base_g[0])
    out["drift_mGal"] = drift
    out["corrected_mGal"] = corrected - drift
    out["relative_to_base_mGal"] = out["corrected_mGal"] - base_g[0]
    return out
