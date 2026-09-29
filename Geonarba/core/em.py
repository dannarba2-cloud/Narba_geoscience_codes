"""Electromagnetic methods: MT, CSAMT, TEM, airborne EM CDT, FDEM (LIN) and GPR processing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from .grid_tools import GridData
from .profile import ProfileData
from .seismic import Section, stolt_migration

MU0 = 4e-7 * np.pi


# ---------------------------------------------------------------- MT
def mt_impedance_tensor(
    table: pd.DataFrame, ex: str = "ex", ey: str = "ey", hx: str = "hx", hy: str = "hy", hz: str = "",
    sampling_rate: float = 100.0, window_length: float = 64.0, bands_per_decade: int = 6,
) -> tuple[pd.DataFrame, dict]:
    """Least-squares impedance Z = <E H*><H H*>^-1 (and tipper) from band-averaged cross-spectra.

    Units: E in mV/km, B in nT -> rho_a = 0.2 |Z|^2 / f (ohm.m), phase = arg(Z) (deg).
    """

    cols = {k: table[v].to_numpy(float) for k, v in dict(ex=ex, ey=ey, hx=hx, hy=hy).items()}
    use_hz = bool(hz)
    if use_hz:
        cols["hz"] = table[hz].to_numpy(float)
    nw = int(window_length * sampling_rate)
    freqs = np.fft.rfftfreq(nw, 1 / sampling_rate)
    taper = np.hanning(nw)
    specs = {k: [] for k in cols}
    for s in range(0, len(cols["ex"]) - nw + 1, nw // 2):
        for k, v in cols.items():
            seg = v[s:s + nw]
            specs[k].append(np.fft.rfft((seg - seg.mean()) * taper))
    if not specs["ex"]:
        raise ValueError("Time series shorter than one window.")
    specs = {k: np.array(v) for k, v in specs.items()}
    fmin = 2 * sampling_rate / nw
    edges = 10 ** np.arange(np.log10(fmin), np.log10(sampling_rate / 4), 1 / bands_per_decade)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (freqs >= lo) & (freqs < hi)
        if sel.sum() < 2:
            continue
        hmat = np.stack([specs["hx"][:, sel].ravel(), specs["hy"][:, sel].ravel()], axis=1)
        hh = hmat.conj().T @ hmat
        if np.linalg.cond(hh) > 1e12:
            continue
        inv = np.linalg.inv(hh)
        zx = (specs["ex"][:, sel].ravel() @ hmat.conj()) @ inv
        zy = (specs["ey"][:, sel].ravel() @ hmat.conj()) @ inv
        f = float(np.sqrt(lo * hi))
        row = {"frequency": f, "period": 1 / f}
        for name, val in (("zxx", zx[0]), ("zxy", zx[1]), ("zyx", zy[0]), ("zyy", zy[1])):
            row[f"{name}_re"], row[f"{name}_im"] = val.real, val.imag
        for name, val in (("xy", zx[1]), ("yx", zy[0])):
            row[f"rho_{name}"] = 0.2 * abs(val) ** 2 / f
            row[f"phase_{name}"] = float(np.degrees(np.angle(val)))
        row["phase_yx"] = row["phase_yx"] + 180 if row["phase_yx"] < -90 else row["phase_yx"]
        det = zx[0] * zy[1] - zx[1] * zy[0]
        row["rho_det"] = 0.2 * abs(np.sqrt(det)) ** 2 / f
        row["phase_det"] = float(np.degrees(np.angle(np.sqrt(det))))
        if use_hz:
            t = (specs["hz"][:, sel].ravel() @ hmat.conj()) @ inv
            row.update({"tx_re": t[0].real, "tx_im": t[0].imag, "ty_re": t[1].real, "ty_im": t[1].imag})
            row["induction_arrow_real_mag"] = float(np.hypot(t[0].real, t[1].real))
            row["induction_arrow_real_azimuth_parkinson"] = float(np.degrees(np.arctan2(-t[1].real, -t[0].real)))
        rows.append(row)
    return pd.DataFrame(rows), {"windows": int(specs["ex"].shape[0]), "bands": len(rows)}


def mt_tipper_arrows(tipper_table: pd.DataFrame, x_col: str = "x", y_col: str = "y", period: float | None = None) -> pd.DataFrame:
    """Parkinson (reversed real) induction arrows per site from a multi-site tipper table."""

    t = tipper_table if period is None else tipper_table.iloc[(tipper_table["period"] - period).abs().argsort()]
    out = t.copy()
    out["arrow_dx"] = -out["ty_re"]  # east component (y = east in MT convention)
    out["arrow_dy"] = -out["tx_re"]  # north component
    out["arrow_length"] = np.hypot(out["arrow_dx"], out["arrow_dy"])
    out["points_to"] = "conductor (Parkinson convention)"
    return out


def mt1d_forward(freqs: np.ndarray, rho: np.ndarray, thick: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wait (1954) impedance recursion for a layered half-space: returns rho_a (ohm.m), phase (deg)."""

    w = 2 * np.pi * np.asarray(freqs, float)
    k = np.sqrt(1j * w[:, None] * MU0 / rho[None, :])  # e^{-i w t}: phase of Z is 45 deg over a half-space
    z = 1j * w * MU0 / k[:, -1]
    for i in range(len(rho) - 2, -1, -1):
        zi = 1j * w * MU0 / k[:, i]
        arg = k[:, i] * thick[i]
        th = np.where(arg.real > 20, 1.0 + 0j, np.tanh(np.where(arg.real > 20, 0, arg)))
        z = zi * (z + zi * th) / (zi + z * th)
    return np.abs(z) ** 2 / (w * MU0), np.degrees(np.angle(z))


def mt_occam_1d(
    table: pd.DataFrame, freq_col: str = "frequency", rho_col: str = "rho_det", phase_col: str = "phase_det",
    n_layers: int = 40, error_percent: float = 5.0, target_rms: float = 1.0, iterations: int = 12,
    operator: str = "first-derivative",
) -> tuple[ProfileData, ProfileData, ProfileData, dict]:
    """Occam 1D inversion (Constable et al., 1987): smoothest log-resistivity model reaching the target misfit.

    Roughness uses 1st (default) or 2nd derivatives of log10(rho) with depth; data weighted by 1/sigma.
    Returns the model, the model response, normalized residuals (should look like white noise) and a QC report.
    """

    from .inversion import misfit_report

    f = table[freq_col].to_numpy(float)
    rho_obs = table[rho_col].to_numpy(float)
    ph_obs = table[phase_col].to_numpy(float)
    skin_min = 503 * np.sqrt(np.min(rho_obs) / f.max())
    skin_max = 503 * np.sqrt(np.max(rho_obs) / f.min())
    tops = np.concatenate([[0], np.geomspace(skin_min / 4, skin_max * 1.5, n_layers - 1)])
    thick = np.diff(tops)
    d = np.concatenate([np.log10(rho_obs), ph_obs])
    err = np.concatenate([np.full(len(f), error_percent / 100 / np.log(10) * 2), np.full(len(f), error_percent / 100 * 28.65)])
    wd = 1 / err

    def fwd(m):
        r, p = mt1d_forward(f, 10**m, thick)
        return np.concatenate([np.log10(r), p])

    rough = np.diff(np.eye(n_layers), n=2 if operator == "second-derivative" else 1, axis=0)
    rtr = rough.T @ rough
    m = np.full(n_layers, np.log10(np.median(rho_obs)))
    history = []
    for _ in range(iterations):
        pred = fwd(m)
        jac = np.zeros((len(d), n_layers))
        for j in range(n_layers):
            dm = np.zeros(n_layers)
            dm[j] = 0.01
            jac[:, j] = (fwd(m + dm) - pred) / 0.01
        wj = jac * wd[:, None]
        dhat = (d - pred + jac @ m) * wd
        best = None
        for mu in np.geomspace(1e-3, 1e4, 30):
            mnew = np.linalg.solve(mu * rtr + wj.T @ wj, wj.T @ dhat)
            rms = float(np.sqrt(np.mean(((fwd(mnew) - d) * wd) ** 2)))
            key = (rms > target_rms * 1.01, -mu if rms <= target_rms * 1.01 else rms)
            if best is None or key < best[0]:
                best = (key, mnew, rms, mu)
        _, m, rms, mu = best
        history.append(rms)
        if rms <= target_rms * 1.01 and len(history) > 2 and abs(history[-1] - history[-2]) < 1e-3:
            break
    r, p = mt1d_forward(f, 10**m, thick)
    depth_mid = tops + np.append(thick, thick[-1]) / 2
    pred = fwd(m)
    qc = misfit_report(d, pred, err)
    nres = (d - pred) / err
    return (ProfileData(depth_mid, 10**m, name="MT_Occam_resistivity_vs_depth", units="ohm.m", metadata={"axis": "depth_m", "log_axes": True}),
            ProfileData(1 / f, r, name="MT_Occam_response_rho_a", units="ohm.m", metadata={"axis": "period_s", "log_axes": True}),
            ProfileData(np.arange(len(nres), dtype=float), nres, name="MT_Occam_normalized_residuals", units="residual / sigma",
                        metadata={"axis": "datum index (rho_a then phase)"}),
            {**qc, "rms_history": history, "final_rms": history[-1], "layers": n_layers, "operator": operator,
             "roughness": float(np.sum((rough @ m) ** 2))})


def csamt_cagniard(
    table: pd.DataFrame, freq_col: str = "frequency", ex_col: str = "ex", hy_col: str = "hy", phase_col: str = "",
    tx_rx_distance: float = 5000.0,
) -> pd.DataFrame:
    """Scalar CSAMT: rho_a = 0.2 |Ex/Hy|^2 / f (mV/km, nT); near-field flag when r < 3 skin depths; Bostick depth."""

    f = table[freq_col].to_numpy(float)
    rho = 0.2 * (table[ex_col].to_numpy(float) / table[hy_col].to_numpy(float)) ** 2 / f
    out = table.copy()
    out["rho_a"] = rho
    out["skin_depth_m"] = 503 * np.sqrt(rho / f)
    out["far_field"] = tx_rx_distance > 3 * out["skin_depth_m"]
    if phase_col:
        phi = np.radians(table[phase_col].to_numpy(float))
        out["bostick_rho"] = rho * (np.pi / (2 * phi) - 1)
    else:
        order = np.argsort(1 / f)
        slope = np.gradient(np.log(rho[order]), np.log(1 / f[order]))
        m = np.empty_like(slope)
        m[order] = slope
        out["bostick_rho"] = rho * (1 + m) / np.maximum(1 - m, 1e-3)
    out["bostick_depth_m"] = np.sqrt(rho / (2 * np.pi * f * MU0))
    return out


# ---------------------------------------------------------------- TEM / AEM
def tem_late_time_rho(
    table: pd.DataFrame, time_col: str = "time", response_col: str = "dbdt", tx_area: float = 10000.0, tx_turns: int = 1,
    current: float = 1.0, rx_area: float = 1.0, fit_last_gates: int = 6,
) -> tuple[pd.DataFrame, dict]:
    """Central-loop late-time apparent resistivity (Spies & Frischknecht, 1991) and exponential decay constant.

    rho_a(t) = (mu0 / (4 pi t)) * (2 mu0 M / (5 t v))^(2/3),  v = response / rx_area (V/m^2), M = I A n.
    tau from ln V = a - t / tau over the last gates (confined conductor indicator).
    """

    t = table[time_col].to_numpy(float)
    v = np.abs(table[response_col].to_numpy(float)) / rx_area
    moment = current * tx_area * tx_turns
    rho = MU0 / (4 * np.pi * t) * (2 * MU0 * moment / (5 * t * v)) ** (2 / 3)
    out = table.copy()
    out["rho_a_late_time"] = rho
    out["diffusion_depth_m"] = np.sqrt(2 * t * rho / MU0)
    n = min(fit_last_gates, len(t))
    slope, _ = np.polyfit(t[-n:], np.log(v[-n:]), 1)
    tau = -1 / slope if slope < 0 else np.inf
    power, _ = np.polyfit(np.log(t[-n:]), np.log(v[-n:]), 1)
    return out, {"decay_time_constant_s": float(tau), "late_time_power_law_exponent": float(power),
                 "interpretation": "exponential late decay (large tau) suggests a confined conductor; t^-2.5 suggests a half-space"}


def aem_cdt(
    table: pd.DataFrame, x_col: str = "x", time_col: str = "time", response_col: str = "dbdt", moment: float = 5e5,
    depth_step: float = 5.0, max_depth: float = 300.0, depth_factor: float = 0.707,
) -> GridData:
    """Airborne TEM conductivity-depth transform: halfspace late-time sigma_a per gate at diffusion depth
    d = depth_factor * sqrt(2 t / (mu0 sigma_a)); interpolated on a regular distance-depth section."""

    x_all = table[x_col].to_numpy(float)
    xs = np.unique(x_all)
    z = np.arange(0, max_depth + depth_step / 2, depth_step)
    img = np.full((len(z), len(xs)), np.nan)
    for j, x in enumerate(xs):
        sel = x_all == x
        t = table[time_col].to_numpy(float)[sel]
        v = np.abs(table[response_col].to_numpy(float)[sel])
        rho = MU0 / (4 * np.pi * t) * (2 * MU0 * moment / (5 * t * v)) ** (2 / 3)
        sig = 1 / rho
        d = depth_factor * np.sqrt(2 * t / (MU0 * sig))
        order = np.argsort(d)
        img[:, j] = np.interp(z, d[order], np.log10(sig[order] * 1000), left=np.nan, right=np.nan)
    dx = float(np.median(np.diff(xs))) if len(xs) > 1 else 1.0
    return GridData(img, xs, z, dx, depth_step, name="AEM_CDT_log10_mS_per_m", units="log10(mS/m)",
                    metadata={"method": "aem_cdt", "vertical_axis": "depth"})


def fdem_lin_conductivity(
    table: pd.DataFrame, quadrature_col: str = "quadrature_ppt", frequency: float = 9800.0, coil_separation: float = 3.66,
) -> pd.DataFrame:
    """McNeill (1980) low-induction-number apparent conductivity: sigma_a = 4 Q / (omega mu0 s^2)."""

    q = table[quadrature_col].to_numpy(float) / 1000.0
    out = table.copy()
    out["sigma_a_mS_per_m"] = 4 * q / (2 * np.pi * frequency * MU0 * coil_separation**2) * 1000
    b = coil_separation * np.sqrt(2 * np.pi * frequency * MU0 * np.maximum(out["sigma_a_mS_per_m"], 1e-6) / 1000 / 2)
    out["induction_number"] = b
    out["lin_valid"] = b < 0.3
    return out


# ---------------------------------------------------------------- GPR
def gpr_dewow(section: Section, window: float = 5e-9) -> Section:
    """Remove low-frequency 'wow' by subtracting a running mean along each trace."""

    n = max(int(round(window / section.dt)), 3)
    return section.derive(section.data - uniform_filter1d(section.data, n, axis=0, mode="nearest"), f"{section.name}_dewow",
                          metadata={"method": "gpr_dewow", "window_s": window})


def gpr_background_removal(section: Section, window_traces: int = 0) -> Section:
    """Subtract the mean trace (global) or a moving average over `window_traces` traces."""

    if window_traces and window_traces < section.ntr:
        bg = uniform_filter1d(section.data, window_traces, axis=1, mode="nearest")
    else:
        bg = section.data.mean(axis=1, keepdims=True)
    return section.derive(section.data - bg, f"{section.name}_bgr", metadata={"method": "gpr_background_removal"})


def gpr_migration(section: Section, velocity: float = 1.0e8) -> Section:
    """Constant-velocity Stolt f-k migration (exploding reflector, v/2). Velocity in m/s (0.1 m/ns = 1e8)."""

    return stolt_migration(section, velocity, exploding_reflector=True)


def gpr_hyperbola_velocity(depth_m: float, apex_time_s: float) -> float:
    return 2 * depth_m / apex_time_s


def mt_apparent_profile(table: pd.DataFrame, rho_col: str = "rho_det", freq_col: str = "frequency") -> ProfileData:
    return ProfileData(1 / table[freq_col].to_numpy(float), table[rho_col].to_numpy(float), name=f"MT_{rho_col}", units="ohm.m",
                       metadata={"axis": "period_s", "log_axes": True})
