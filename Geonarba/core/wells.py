"""Well logging petrophysics and gamma-ray spectrometry / radiogenic heat."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.optimize import nnls

from .grid_tools import GridData
from .profile import ProfileData


def read_las(path: str | Path) -> pd.DataFrame:
    """Minimal LAS 2.0 reader (~C curve mnemonics, ~A ASCII data; NULL -> NaN)."""

    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    section, curves, null, data = "", [], -999.25, []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("~"):
            section = line[1].upper()
            continue
        if section == "W" and line.upper().startswith("NULL"):
            null = float(line.split(":")[0].split()[-1])
        elif section == "C":
            curves.append(line.split(".")[0].strip())
        elif section == "A":
            data.extend(float(v) for v in line.split())
    arr = np.asarray(data, float).reshape(-1, len(curves))
    df = pd.DataFrame(arr, columns=curves)
    return df.replace(null, np.nan)


def shale_volume_gr(table: pd.DataFrame, gr_col: str = "GR", gr_clean: float | None = None, gr_shale: float | None = None,
                    method: str = "linear") -> pd.DataFrame:
    """Vsh from gamma ray: IGR = (GR - GRclean)/(GRshale - GRclean); Larionov tertiary/older corrections."""

    gr = table[gr_col].to_numpy(float)
    lo = gr_clean if gr_clean is not None else np.nanpercentile(gr, 5)
    hi = gr_shale if gr_shale is not None else np.nanpercentile(gr, 95)
    igr = np.clip((gr - lo) / (hi - lo), 0, 1)
    if method == "larionov_tertiary":
        vsh = 0.083 * (2 ** (3.7 * igr) - 1)
    elif method == "larionov_older":
        vsh = 0.33 * (2 ** (2 * igr) - 1)
    else:
        vsh = igr
    out = table.copy()
    out["IGR"], out["VSH_GR"] = igr, np.clip(vsh, 0, 1)
    return out


def sp_baseline_shift(
    table: pd.DataFrame, depth_col: str = "DEPTH", sp_col: str = "SP", window: int = 101, permeable_threshold_mv: float = 15.0,
    rmf_rw_ratio: float | None = None, temperature_c: float = 75.0,
) -> pd.DataFrame:
    """Shale-baseline tracking (rolling high percentile), baseline shift correction, Vsh_SP and permeable flag.

    SSP = most negative deflection; Vsh_SP = 1 - (SP - baseline)/SSP; Rw from SSP = -K log(Rmf/Rw), K = 61 + 0.133 T(F).
    """

    sp = table[sp_col].to_numpy(float)
    baseline = pd.Series(sp).rolling(window, center=True, min_periods=1).quantile(0.9).to_numpy()
    baseline = median_filter(baseline, size=max(window // 2, 1), mode="nearest")
    corrected = sp - baseline
    ssp = float(np.nanmin(corrected))
    out = table.copy()
    out["SP_BASELINE"] = baseline
    out["SP_CORR"] = corrected
    out["VSH_SP"] = np.clip(1 - corrected / ssp, 0, 1) if ssp < 0 else np.nan
    out["PERMEABLE"] = corrected < -abs(permeable_threshold_mv)
    if rmf_rw_ratio is None:
        k = 61 + 0.133 * (temperature_c * 9 / 5 + 32)
        out.attrs["Rmf_over_Rw_from_SSP"] = float(10 ** (-ssp / k))
    return out


def archie_saturation(
    table: pd.DataFrame, rt_col: str = "RT", phi_col: str = "PHI", rw: float = 0.05, a: float = 1.0, m: float = 2.0, n: float = 2.0,
) -> pd.DataFrame:
    """Archie (1942): Sw = (a Rw / (phi^m Rt))^(1/n); Sh = 1 - Sw; BVW = phi Sw."""

    rt = table[rt_col].to_numpy(float)
    phi = np.clip(table[phi_col].to_numpy(float), 1e-4, 1)
    sw = np.clip((a * rw / (phi**m * rt)) ** (1 / n), 0, 1)
    out = table.copy()
    out["SW"], out["SH"], out["BVW"] = sw, 1 - sw, phi * sw
    out["RO"] = a * rw / phi**m
    return out


MATRIX_DENSITY = {"sandstone": 2.65, "limestone": 2.71, "dolomite": 2.87, "anhydrite": 2.98, "halite": 2.03}


def density_neutron(
    table: pd.DataFrame, rhob_col: str = "RHOB", nphi_col: str = "NPHI", rho_matrix: float = 2.65, rho_fluid: float = 1.0,
) -> pd.DataFrame:
    """Density porosity, density-neutron crossplot porosity (RMS for gas, mean otherwise), apparent matrix density & lithology."""

    rhob = table[rhob_col].to_numpy(float)
    nphi = table[nphi_col].to_numpy(float)
    nphi = nphi / 100 if np.nanmax(nphi) > 1.5 else nphi
    phid = (rho_matrix - rhob) / (rho_matrix - rho_fluid)
    gas = nphi < phid - 0.02
    phi = np.where(gas, np.sqrt((phid**2 + nphi**2) / 2), (phid + nphi) / 2)
    rhomaa = (rhob - phi * rho_fluid) / np.maximum(1 - phi, 1e-3)
    names = list(MATRIX_DENSITY)
    vals = np.array([MATRIX_DENSITY[k] for k in names])
    lith = np.array(names)[np.argmin(np.abs(rhomaa[:, None] - vals[None, :]), axis=1)]
    out = table.copy()
    out["PHID"], out["PHIDN"], out["RHOMAA"], out["LITHOLOGY"], out["GAS_FLAG"] = phid, np.clip(phi, 0, 0.5), rhomaa, lith, gas
    return out


def sonic_integration(
    table: pd.DataFrame, depth_col: str = "DEPTH", dt_col: str = "DT", dt_unit: str = "us/ft", start_twt: float = 0.0,
    start_depth: float | None = None,
) -> pd.DataFrame:
    """Integrate sonic transit time into one-way/two-way time; interval and RMS velocities (m, s)."""

    z = table[depth_col].to_numpy(float)
    dt = table[dt_col].to_numpy(float) * (1e-6 / 0.3048 if dt_unit == "us/ft" else 1e-6)
    dz = np.diff(z, prepend=start_depth if start_depth is not None else z[0])
    owt = np.nancumsum(dt * dz)
    twt = start_twt + 2 * owt
    vint = 1 / dt
    vrms = np.sqrt(np.nancumsum(vint**2 * dt * dz) / np.maximum(owt, 1e-12))
    out = table.copy()
    out["OWT_s"], out["TWT_s"], out["VINT_m_s"], out["VRMS_m_s"] = owt, twt, vint, vrms
    return out


def dipmeter_sinusoid_fit(
    table: pd.DataFrame, feature_col: str = "feature", depth_col: str = "depth", azimuth_col: str = "azimuth",
    borehole_diameter: float = 0.216,
) -> pd.DataFrame:
    """Fit z(theta) = z0 + A cos(theta - theta0) per image-log feature: dip = atan(2A / D), dip azimuth = theta0 (deepest point)."""

    rows = []
    for fid, grp in table.groupby(feature_col):
        th = np.radians(grp[azimuth_col].to_numpy(float))
        z = grp[depth_col].to_numpy(float)
        if len(z) < 3:
            continue
        a = np.column_stack([np.ones_like(th), np.cos(th), np.sin(th)])
        (z0, c, s), *_ = np.linalg.lstsq(a, z, rcond=None)
        amp = np.hypot(c, s)
        rows.append({"feature": fid, "depth": z0, "dip_deg": float(np.degrees(np.arctan(2 * amp / borehole_diameter))),
                     "dip_azimuth_deg": float(np.degrees(np.arctan2(s, c)) % 360),
                     "fit_rms_m": float(np.sqrt(np.mean((a @ [z0, c, s] - z) ** 2)))})
    return pd.DataFrame(rows)


def nmr_t2_inversion(
    table: pd.DataFrame, time_col: str = "time_ms", amp_col: str = "amplitude", n_bins: int = 64, t2_min: float = 0.3,
    t2_max: float = 3000.0, regularization: float = 0.5, t2_cutoff: float = 33.0, coates_c: float = 10.0,
) -> tuple[ProfileData, dict]:
    """Multi-exponential T2 inversion (NNLS with Tikhonov): A(t) = sum_i f_i exp(-t / T2_i).

    BVI = sum f(T2 < cutoff); FFI = rest; Coates k = (phi/C)^4 (FFI/BVI)^2 mD (phi in %).
    """

    t = table[time_col].to_numpy(float)
    y = table[amp_col].to_numpy(float)
    t2 = np.geomspace(t2_min, t2_max, n_bins)
    kernel = np.exp(-t[:, None] / t2[None, :])
    lam = regularization * 1e-2 * np.linalg.norm(kernel, 2)  # mild smoothing; too large biases porosity low
    a = np.vstack([kernel, lam * np.eye(n_bins)])
    f, _ = nnls(a, np.concatenate([y, np.zeros(n_bins)]))
    phi = float(f.sum())
    bvi = float(f[t2 < t2_cutoff].sum())
    ffi = phi - bvi
    perm = (phi * (100 if phi <= 1 else 1) / coates_c) ** 4 * (ffi / max(bvi, 1e-9)) ** 2
    return (ProfileData(t2, f, name="NMR_T2_distribution", units="porosity units", metadata={"axis": "T2_ms", "log_x": True}),
            {"total_porosity": phi, "BVI": bvi, "FFI": ffi, "T2_log_mean_ms": float(np.exp(np.sum(f * np.log(t2)) / max(phi, 1e-12))),
             "coates_permeability_mD": float(perm), "t2_cutoff_ms": t2_cutoff})


def gamma_spectrometry(
    table: pd.DataFrame, k_col: str = "K", u_col: str = "eU", th_col: str = "eTh",
) -> pd.DataFrame:
    """Ratios, F-parameter (K eU / eTh) and air dose rate D = 13.078 K + 5.675 eU + 2.494 eTh nGy/h (IAEA, 2003)."""

    k, u, th = (table[c].to_numpy(float) for c in (k_col, u_col, th_col))
    out = table.copy()
    out["eTh_K"] = th / np.maximum(k, 1e-6)
    out["eU_eTh"] = u / np.maximum(th, 1e-6)
    out["eU_K"] = u / np.maximum(k, 1e-6)
    out["F_parameter"] = k * u / np.maximum(th, 1e-6)
    out["dose_rate_nGy_h"] = 13.078 * k + 5.675 * u + 2.494 * th
    return out


def radiogenic_heat_production(k_pct, u_ppm, th_ppm, density: float = 2670.0) -> np.ndarray:
    """Rybach (1988): A (uW/m^3) = 1e-5 rho (9.52 U + 2.56 Th + 3.48 K)."""

    return 1e-5 * density * (9.52 * np.asarray(u_ppm, float) + 2.56 * np.asarray(th_ppm, float) + 3.48 * np.asarray(k_pct, float))


def heat_production_grid(k: GridData, u: GridData, th: GridData, density: float = 2670.0) -> GridData:
    return k.with_values(radiogenic_heat_production(k.values, u.values, th.values, density), name="Radiogenic_heat_production",
                         units="uW/m^3", metadata={"method": "rybach_1988", "density": density,
                                                   "formula": "A = 1e-5 rho (9.52 U + 2.56 Th + 3.48 K)"})


def heat_production_table(table: pd.DataFrame, k_col: str = "K", u_col: str = "eU", th_col: str = "eTh", density: float = 2670.0) -> pd.DataFrame:
    out = table.copy()
    out["heat_production_uW_m3"] = radiogenic_heat_production(table[k_col], table[u_col], table[th_col], density)
    return out


def log_profile(table: pd.DataFrame, depth_col: str, value_col: str) -> ProfileData:
    return ProfileData(table[depth_col].to_numpy(float), table[value_col].to_numpy(float), name=value_col,
                       metadata={"axis": "depth", "log_track": True})
