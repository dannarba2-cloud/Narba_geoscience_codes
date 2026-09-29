"""Passive and surface-wave seismology: HVSR, SPAC, MASW, receiver functions, focal mechanisms, interferometry."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import j0

from .grid_tools import GridData
from .profile import ProfileData
from .seismic import Section


def konno_ohmachi(freqs: np.ndarray, spectrum: np.ndarray, b: float = 40.0) -> np.ndarray:
    """Konno & Ohmachi (1998) log-window smoothing."""

    out = np.zeros_like(spectrum)
    f = np.where(freqs > 0, freqs, np.nan)
    for i, fc in enumerate(freqs):
        if fc <= 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            x = b * np.log10(f / fc)
            w = np.where(np.abs(x) < 1e-6, 1.0, (np.sin(x) / x) ** 4)
        w = np.nan_to_num(w)
        out[i] = np.sum(w * spectrum) / np.sum(w)
    return out


def hvsr(
    table: pd.DataFrame, north_col: str = "north", east_col: str = "east", vertical_col: str = "vertical",
    sampling_rate: float = 100.0, window_length: float = 30.0, f_min: float = 0.2, f_max: float = 20.0,
    smoothing_b: float = 40.0, shear_velocity: float = 300.0,
) -> tuple[ProfileData, dict]:
    """Nakamura HVSR: H = sqrt((N^2 + E^2)/2), windowed, Konno-Ohmachi smoothed, geometric-mean average.

    Bedrock depth from f0: h = Vs / (4 f0) and power law h = 96 f0^-1.388 (Ibs-von Seht & Wohlenberg, 1999).
    """

    n, e, z = (table[c].to_numpy(float) for c in (north_col, east_col, vertical_col))
    nw = int(window_length * sampling_rate)
    if nw < 16 or len(z) < nw:
        raise ValueError("Record shorter than one window.")
    freqs = np.fft.rfftfreq(nw, 1 / sampling_rate)
    band = (freqs >= f_min) & (freqs <= f_max)
    taper = np.hanning(nw)
    ratios = []
    for s in range(0, len(z) - nw + 1, nw // 2):
        spec = [np.abs(np.fft.rfft((c[s:s + nw] - c[s:s + nw].mean()) * taper)) for c in (n, e, z)]
        sm = [konno_ohmachi(freqs[band], sp[band], smoothing_b) for sp in spec]
        h = np.sqrt((sm[0] ** 2 + sm[1] ** 2) / 2)
        ratios.append(np.log(h / np.maximum(sm[2], 1e-30)))
    ratios = np.array(ratios)
    mean = np.exp(ratios.mean(axis=0))
    std = ratios.std(axis=0)
    i0 = int(np.argmax(mean))
    f0 = float(freqs[band][i0])
    return (ProfileData(freqs[band], mean, name="HVSR", units="H/V", metadata={"method": "hvsr", "axis": "frequency_Hz",
                                                                                "log_std": std.tolist()}),
            {"f0_Hz": f0, "A0": float(mean[i0]), "windows": int(len(ratios)),
             "depth_quarter_wavelength_m": shear_velocity / (4 * f0), "depth_power_law_m": 96.0 * f0 ** -1.388,
             "sesame_clear_peak": bool(mean[i0] > 2.0)})


def spac(
    table: pd.DataFrame, center_col: str = "center", ring_prefix: str = "ring", radius: float = 20.0,
    sampling_rate: float = 100.0, window_length: float = 20.0, f_min: float = 1.0, f_max: float = 30.0,
) -> tuple[ProfileData, ProfileData, dict]:
    """Aki (1957) SPAC: rho(f) = <Re coherency(center, ring_i)> = J0(2 pi f r / c(f)); solve on the first J0 branch."""

    center = table[center_col].to_numpy(float)
    rings = [table[c].to_numpy(float) for c in table.columns if str(c).startswith(ring_prefix)]
    if not rings:
        raise ValueError(f"No ring station columns starting with '{ring_prefix}'.")
    nw = int(window_length * sampling_rate)
    freqs = np.fft.rfftfreq(nw, 1 / sampling_rate)
    taper = np.hanning(nw)
    sxy = np.zeros((len(rings), len(freqs)), complex)
    sxx = np.zeros(len(freqs))
    syy = np.zeros((len(rings), len(freqs)))
    for s in range(0, len(center) - nw + 1, nw // 2):
        c = np.fft.rfft(center[s:s + nw] * taper)
        sxx += np.abs(c) ** 2
        for i, r in enumerate(rings):
            rf = np.fft.rfft(r[s:s + nw] * taper)
            sxy[i] += c * np.conj(rf)
            syy[i] += np.abs(rf) ** 2
    coh = np.real(sxy / np.sqrt(sxx * syy + 1e-30)).mean(axis=0)
    band = (freqs >= f_min) & (freqs <= f_max)
    fb, rho = freqs[band], coh[band]
    vel = np.full(len(fb), np.nan)
    for i, (f, r) in enumerate(zip(fb, rho)):
        if -0.40 < r < 0.999:
            try:
                arg = brentq(lambda a: j0(a) - r, 1e-6, 3.8317)
                vel[i] = 2 * np.pi * f * radius / arg
            except ValueError:
                pass
    ok = np.isfinite(vel)
    return (ProfileData(fb, rho, name="SPAC_coefficient", units="rho", metadata={"axis": "frequency_Hz"}),
            ProfileData(fb[ok], vel[ok], name="SPAC_phase_velocity", units="m/s", metadata={"axis": "frequency_Hz"}),
            {"radius_m": radius, "ring_stations": len(rings), "valid_frequencies": int(ok.sum())})


def masw_dispersion(
    gather: Section, c_min: float = 100.0, c_max: float = 1000.0, n_c: int = 181, f_min: float = 5.0, f_max: float = 60.0,
    vs_factor: float = 1.09, depth_factor: float = 1 / 3,
) -> tuple[GridData, ProfileData, pd.DataFrame]:
    """MASW phase-shift dispersion image (Park et al., 1998) with a fundamental-mode pick.

    Approximate Vs profile from the pick: Vs ~= 1.09 c(f) at depth ~= lambda/3 (wavelength-depth approximation).
    """

    offsets = gather.header("offset")
    spec = np.fft.rfft(gather.data, axis=0)
    freqs = np.fft.rfftfreq(gather.nt, gather.dt)
    band = (freqs >= f_min) & (freqs <= f_max)
    norm = spec[band] / np.maximum(np.abs(spec[band]), 1e-30)
    cs = np.linspace(c_min, c_max, n_c)
    img = np.zeros((n_c, band.sum()))
    for i, c in enumerate(cs):
        phase = np.exp(1j * 2 * np.pi * freqs[band][:, None] * offsets[None, :] / c)
        img[i] = np.abs((norm * phase).sum(axis=1)) / len(offsets)
    fb = freqs[band]
    pick = cs[np.argmax(img, axis=0)]
    df = float(np.median(np.diff(fb))) if len(fb) > 1 else 1.0
    image = GridData(img, fb, cs, df, float(cs[1] - cs[0]), name=f"{gather.name}_dispersion", units="normalized",
                     metadata={"method": "masw_phase_shift", "x_axis": "frequency_Hz", "y_axis": "phase_velocity_m_s"})
    curve = ProfileData(fb, pick, name=f"{gather.name}_dispersion_curve", units="m/s", metadata={"axis": "frequency_Hz"})
    wavelength = pick / fb
    vs = pd.DataFrame({"depth_m": wavelength * depth_factor, "vs_m_per_s": pick * vs_factor,
                       "frequency_Hz": fb}).sort_values("depth_m")
    return image, curve, vs


def receiver_function(
    table: pd.DataFrame, vertical_col: str = "vertical", radial_col: str = "radial", sampling_rate: float = 20.0,
    water_level: float = 0.01, gaussian: float = 2.5, pre_signal: float = 5.0,
) -> ProfileData:
    """Water-level frequency-domain deconvolution R/Z with Gaussian low-pass (Langston, 1979)."""

    z = table[vertical_col].to_numpy(float)
    r = table[radial_col].to_numpy(float)
    n = int(2 ** np.ceil(np.log2(2 * len(z))))
    fz, fr = np.fft.rfft(z, n), np.fft.rfft(r, n)
    w = 2 * np.pi * np.fft.rfftfreq(n, 1 / sampling_rate)
    denom = np.maximum(np.abs(fz) ** 2, water_level * np.max(np.abs(fz) ** 2))
    gauss = np.exp(-(w**2) / (4 * gaussian**2))
    rf = np.fft.irfft(fr * np.conj(fz) / denom * gauss * np.exp(-1j * w * pre_signal), n)[: len(z)]
    t = np.arange(len(rf)) / sampling_rate - pre_signal
    return ProfileData(t, rf, name="receiver_function", units="", metadata={"method": "water_level_deconvolution", "axis": "time_s",
                                                                           "gaussian": gaussian, "water_level": water_level})


def h_kappa_stack(
    rf: ProfileData, vp: float = 6.3, ray_parameter: float = 0.06, h_min: float = 20.0, h_max: float = 60.0,
    k_min: float = 1.6, k_max: float = 2.0, weights: tuple[float, float, float] = (0.7, 0.2, 0.1),
) -> tuple[GridData, dict]:
    """Zhu & Kanamori (2000) H-kappa stacking (vp km/s, p s/km, H km)."""

    hs = np.linspace(h_min, h_max, 161)
    ks = np.linspace(k_min, k_max, 81)
    hh, kk = np.meshgrid(hs, ks)
    vs = vp / kk
    qa = np.sqrt(1 / vp**2 - ray_parameter**2)
    qb = np.sqrt(1 / vs**2 - ray_parameter**2)
    t1, t2, t3 = hh * (qb - qa), hh * (qb + qa), 2 * hh * qb
    amp = lambda tt: np.interp(tt, rf.distance, rf.values)  # noqa: E731
    s = weights[0] * amp(t1) + weights[1] * amp(t2) - weights[2] * amp(t3)
    i = np.unravel_index(np.argmax(s), s.shape)
    grid = GridData(s, hs, ks, hs[1] - hs[0], ks[1] - ks[0], name="H_kappa_stack", units="stack",
                    metadata={"x_axis": "H_km", "y_axis": "Vp/Vs"})
    return grid, {"H_km": float(hh[i]), "Vp_Vs": float(kk[i]), "Ps_time_s": float(t1[i])}


def _radiation_sign(strike, dip, rake, az, takeoff):
    phi, d, lam = np.radians(strike), np.radians(dip), np.radians(rake)
    n = np.array([-np.sin(d) * np.sin(phi), np.sin(d) * np.cos(phi), -np.cos(d)])
    s = np.array([np.cos(lam) * np.cos(phi) + np.cos(d) * np.sin(lam) * np.sin(phi),
                  np.cos(lam) * np.sin(phi) - np.cos(d) * np.sin(lam) * np.cos(phi),
                  -np.sin(lam) * np.sin(d)])
    a, i = np.radians(az), np.radians(takeoff)
    g = np.array([np.sin(i) * np.cos(a), np.sin(i) * np.sin(a), np.cos(i)])
    return 2 * (g.T @ n) * (g.T @ s)


def focal_mechanism(
    table: pd.DataFrame, azimuth_col: str = "azimuth", takeoff_col: str = "takeoff", polarity_col: str = "polarity",
    strike_step: float = 5.0, dip_step: float = 5.0, rake_step: float = 10.0,
) -> tuple[pd.DataFrame, dict]:
    """Grid-search double-couple fit to P first-motion polarities (Aki & Richards radiation pattern, NED frame)."""

    az = table[azimuth_col].to_numpy(float)
    to = table[takeoff_col].to_numpy(float)
    pol = np.sign(table[polarity_col].to_numpy(float))
    best = []
    for strike in np.arange(0, 360, strike_step):
        for dip in np.arange(dip_step, 90 + 1e-9, dip_step):
            for rake in np.arange(-180, 180, rake_step):
                pred = np.sign(_radiation_sign(strike, dip, rake, az, to))
                misfit = float(np.mean(pred != pol))
                best.append((misfit, strike, dip, rake))
    best.sort(key=lambda b: b[0])
    top = pd.DataFrame(best[:20], columns=["misfit", "strike", "dip", "rake"])
    m, strike, dip, rake = best[0]
    kind = ("strike-slip" if abs(rake) < 30 or abs(rake) > 150 else "reverse" if rake > 0 else "normal")
    return top, {"strike": strike, "dip": dip, "rake": rake, "polarity_misfit": m, "fault_type": kind,
                 "n_polarities": int(len(pol))}


def ambient_noise_interferometry(
    table: pd.DataFrame, station_a: str = "a", station_b: str = "b", sampling_rate: float = 50.0,
    window_length: float = 60.0, max_lag: float = 10.0, one_bit: bool = True, whiten: bool = True,
) -> tuple[ProfileData, ProfileData]:
    """Cross-correlation stacking (Bensen et al., 2007): EGF ~ -d/dt of the symmetric stacked correlation."""

    a = table[station_a].to_numpy(float)
    b = table[station_b].to_numpy(float)
    nw = int(window_length * sampling_rate)
    nlag = int(max_lag * sampling_rate)
    stack = np.zeros(2 * nlag + 1)
    count = 0
    for s in range(0, len(a) - nw + 1, nw):
        wa, wb = a[s:s + nw] - a[s:s + nw].mean(), b[s:s + nw] - b[s:s + nw].mean()
        if one_bit:
            wa, wb = np.sign(wa), np.sign(wb)
        fa, fb = np.fft.rfft(wa, 2 * nw), np.fft.rfft(wb, 2 * nw)
        if whiten:
            fa, fb = fa / np.maximum(np.abs(fa), 1e-30), fb / np.maximum(np.abs(fb), 1e-30)
        cc = np.fft.irfft(fb * np.conj(fa), 2 * nw)
        cc = np.concatenate([cc[-nlag:], cc[: nlag + 1]])
        stack += cc
        count += 1
    if count == 0:
        raise ValueError("Records shorter than one correlation window.")
    stack /= count
    lags = np.arange(-nlag, nlag + 1) / sampling_rate
    sym = 0.5 * (stack[nlag:] + stack[nlag::-1])
    egf = -np.gradient(sym, 1 / sampling_rate)
    return (ProfileData(lags, stack, name="noise_cross_correlation", units="", metadata={"axis": "lag_s", "windows": count}),
            ProfileData(lags[nlag:], egf, name="empirical_greens_function", units="", metadata={"axis": "time_s"}))
