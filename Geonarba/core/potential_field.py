"""Potential-field transforms: RTP/RTE, continuation, pseudogravity, edge maps, spectral filters.

Sign conventions (Blakely, 1995): x east, y north, z down; FFT derivative d/dx -> i*kx,
d/dz(down) -> |k|; upward continuation by h -> exp(-|k| h).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .analytic_signal import compute_analytic_signal
from .derivatives import compute_thg
from .fourier import apply_response, theta_operator
from .grid_tools import GridData
from .spectral import compute_raps

CM = 1e-7  # mu0 / 4pi, SI
G = 6.674e-11


def _low_latitude_warning(inc: float) -> str | None:
    if abs(inc) < 20:
        return (
            f"Field inclination {inc:g} deg is low: RTP amplifies N-S wavenumbers. "
            "Use stabilization > 0, RTE, or analytic-signal methods."
        )
    return None


def reduce_to_pole(
    grid: GridData,
    inclination: float,
    declination: float,
    mag_inclination: float | None = None,
    mag_declination: float | None = None,
    stabilization: float = 0.0,
    padding: str = "reflect",
) -> GridData:
    """RTP = F^-1[ F(T) / (Theta_m * Theta_f) ], Wiener-stabilized: conj(D)/(|D|^2 + eps^2)."""

    mi = inclination if mag_inclination is None else mag_inclination
    md = declination if mag_declination is None else mag_declination

    def response(kx, ky, k):
        d = theta_operator(kx, ky, k, mi, md) * theta_operator(kx, ky, k, inclination, declination)
        d = np.where(k == 0, 1.0, d)
        return np.conj(d) / (np.abs(d) ** 2 + stabilization**2)

    values = apply_response(grid, response, padding=padding, remove_mean=True)
    return grid.with_values(
        values,
        name=f"{grid.name}_RTP",
        metadata={
            "method": "reduction_to_pole",
            "inclination": inclination,
            "declination": declination,
            "mag_inclination": mi,
            "mag_declination": md,
            "stabilization": stabilization,
            "formula": "RTP = IFFT( FFT(T) * conj(D) / (|D|^2 + eps^2) ), D = Theta_m * Theta_f",
            "warning": _low_latitude_warning(inclination) or "RTP assumes uniform magnetization direction (induced unless remanence given).",
        },
    )


def reduce_to_equator(
    grid: GridData,
    inclination: float,
    declination: float,
    mag_inclination: float | None = None,
    mag_declination: float | None = None,
    stabilization: float = 0.0,
    padding: str = "reflect",
) -> GridData:
    """RTE = F^-1[ F(T) * Theta_eq^2 / (Theta_m Theta_f) ] with equatorial field (I=0, D unchanged)."""

    mi = inclination if mag_inclination is None else mag_inclination
    md = declination if mag_declination is None else mag_declination

    def response(kx, ky, k):
        d = theta_operator(kx, ky, k, mi, md) * theta_operator(kx, ky, k, inclination, declination)
        num = theta_operator(kx, ky, k, 0.0, md) * theta_operator(kx, ky, k, 0.0, declination)
        d = np.where(k == 0, 1.0, d)
        num = np.where(k == 0, 1.0, num)
        return num * np.conj(d) / (np.abs(d) ** 2 + stabilization**2)

    values = apply_response(grid, response, padding=padding, remove_mean=True)
    return grid.with_values(
        values,
        name=f"{grid.name}_RTE",
        metadata={
            "method": "reduction_to_equator",
            "inclination": inclination,
            "declination": declination,
            "stabilization": stabilization,
            "formula": "RTE = IFFT( FFT(T) * Theta_eq_m Theta_eq_f / (Theta_m Theta_f) )",
            "warning": "RTE anomalies are negative (lows) over sources and elongated E-W; interpret accordingly.",
        },
    )


def downward_continuation(
    grid: GridData,
    depth: float,
    regularization: float = 1e-3,
    padding: str = "reflect",
) -> GridData:
    """Tikhonov/Wiener-regularized downward continuation: H = e^{kh} / (1 + alpha e^{2kh})."""

    if depth <= 0:
        raise ValueError("Continuation depth must be positive (meters below the observation plane).")
    if regularization < 0:
        raise ValueError("Regularization must be non-negative.")

    def response(kx, ky, k):
        e = np.exp(np.minimum(k * depth, 700.0))
        return e / (1.0 + regularization * e**2)

    values = apply_response(grid, response, padding=padding)
    max_gain = 1.0 / (2.0 * np.sqrt(regularization)) if regularization > 0 else np.inf
    return grid.with_values(
        values,
        name=f"{grid.name}_DC_{depth:g}m",
        metadata={
            "method": "downward_continuation",
            "depth_m": depth,
            "regularization": regularization,
            "max_gain": float(max_gain),
            "formula": "DC = IFFT( FFT(T) * e^{|k|h} / (1 + alpha e^{2|k|h}) )",
            "warning": "Downward continuation is unstable; never continue below the shallowest source. Increase alpha if the result is noisy.",
        },
    )


def theta_map(grid: GridData, padding: str = "reflect") -> GridData:
    """Theta map (Wijns et al., 2005): cos(theta) = THG / |A| = cos(tilt angle)."""

    thg = compute_thg(grid).values
    asa = compute_analytic_signal(grid, padding=padding).values
    values = thg / np.maximum(asa, np.finfo(float).tiny)
    return grid.with_values(
        np.clip(values, 0.0, 1.0),
        name=f"{grid.name}_ThetaMap",
        units="dimensionless",
        metadata={"method": "theta_map", "formula": "cos(theta) = THG / ASA"},
    )


def pseudogravity(
    grid: GridData,
    inclination: float,
    declination: float,
    density_contrast: float = 100.0,
    magnetization: float = 1.0,
    mag_inclination: float | None = None,
    mag_declination: float | None = None,
    stabilization: float = 0.01,
    padding: str = "reflect",
) -> GridData:
    """Poisson/Baranov pseudogravity: g = F^-1[ (G rho / (Cm M)) F(T) / (Theta_m Theta_f |k|) ]  (mGal from nT)."""

    mi = inclination if mag_inclination is None else mag_inclination
    md = declination if mag_declination is None else mag_declination
    scale = G * density_contrast / (CM * magnetization) * 1e-9 * 1e5  # nT -> T, m/s^2 -> mGal

    def response(kx, ky, k):
        d = theta_operator(kx, ky, k, mi, md) * theta_operator(kx, ky, k, inclination, declination) * k
        h = scale * np.conj(d) / (np.abs(d) ** 2 + (stabilization * np.max(np.abs(d))) ** 2)
        return np.where(k == 0, 0.0, h)

    values = apply_response(grid, response, padding=padding, remove_mean=True)
    return grid.with_values(
        values,
        name=f"{grid.name}_PseudoGravity",
        units="mGal",
        metadata={
            "method": "pseudogravity",
            "density_contrast": density_contrast,
            "magnetization_A_per_m": magnetization,
            "formula": "PG = IFFT( G*rho/(Cm*M) * FFT(T) / (Theta_m Theta_f |k|) )",
            "warning": "Pseudogravity assumes a fixed density/magnetization ratio and uniform magnetization direction.",
        },
    )


def directional_cosine_filter(
    grid: GridData,
    strike_azimuth: float,
    power: float = 2.0,
    mode: str = "pass",
    padding: str = "reflect",
) -> GridData:
    """Directional cosine filter: H = |cos(az_k - (strike + 90))|^n passes features striking `strike_azimuth`."""

    if mode not in {"pass", "reject"}:
        raise ValueError("mode must be pass or reject.")
    normal = np.radians(strike_azimuth + 90.0)

    def response(kx, ky, k):
        az = np.arctan2(kx, ky)
        h = np.abs(np.cos(az - normal)) ** power
        h = np.where(k == 0, 1.0, h)
        return h if mode == "pass" else np.where(k == 0, 1.0, 1.0 - h)

    values = apply_response(grid, response, padding=padding)
    return grid.with_values(
        values,
        name=f"{grid.name}_DirCos_{mode}_{strike_azimuth:g}",
        metadata={
            "method": "directional_cosine_filter",
            "strike_azimuth_deg": strike_azimuth,
            "power": power,
            "mode": mode,
            "formula": "H = |cos(theta_k - (strike+90))|^n",
        },
    )


def matched_filter(grid: GridData, n_layers: int = 2, padding: str = "reflect") -> tuple[list[GridData], dict]:
    """Matched filtering (Syberg, 1972; Phillips, 2001).

    Fits the radial amplitude spectrum with sum_i A_i exp(-k h_i) (equivalent layers),
    then extracts layer j with H_j = A_j e^{-k h_j} / sum_i A_i e^{-k h_i}.
    """

    if n_layers < 2 or n_layers > 4:
        raise ValueError("n_layers must be between 2 and 4.")
    raps = compute_raps(grid, padding="zero")
    k = raps.k
    amp = np.sqrt(raps.power)
    valid = (k > 0) & np.isfinite(amp) & (amp > 0)
    k, amp = k[valid], amp[valid]
    log_amp = np.log(amp)
    kmax = float(k.max())
    width = min(np.ptp(grid.x_vector), np.ptp(grid.y_vector))
    depth_guess = np.geomspace(max(grid.dx, grid.dy), width / 4.0, n_layers)[::-1]

    def model(p):
        a = np.exp(p[:n_layers])
        h = np.exp(p[n_layers:])
        return np.log(np.sum(a[:, None] * np.exp(-np.outer(h, k)), axis=0) + np.finfo(float).tiny)

    p0 = np.concatenate([np.full(n_layers, log_amp.max()) - np.arange(n_layers), np.log(depth_guess)])
    fit = least_squares(lambda p: model(p) - log_amp, p0)
    a = np.exp(fit.x[:n_layers])
    h = np.exp(fit.x[n_layers:])
    order = np.argsort(h)[::-1]
    a, h = a[order], h[order]

    outputs = []
    for j in range(n_layers):
        def response(kx, ky, kk, j=j):
            terms = a[:, None, None] * np.exp(-h[:, None, None] * kk[None])
            return terms[j] / np.maximum(terms.sum(axis=0), np.finfo(float).tiny)

        values = apply_response(grid, response, padding=padding, remove_mean=j != 0)
        outputs.append(
            grid.with_values(
                values,
                name=f"{grid.name}_MF_layer{j + 1}_{h[j]:.0f}m",
                metadata={
                    "method": "matched_filter",
                    "layer": j + 1,
                    "equivalent_depth_m": float(h[j]),
                    "formula": "H_j = A_j e^{-k h_j} / sum_i A_i e^{-k h_i}",
                    "warning": "Equivalent-layer depths are ensemble averages, not individual source depths.",
                },
            )
        )
    summary = {
        "equivalent_depths_m": [float(v) for v in h],
        "amplitudes": [float(v) for v in a],
        "k_max_rad_per_m": kmax,
        "fit_cost": float(fit.cost),
    }
    return outputs, summary


def igrf_total_field(lon, lat, height_m, date: datetime) -> np.ndarray:
    """IGRF-14 total field intensity (nT) using ppigrf."""

    try:
        import ppigrf
    except ImportError as exc:  # pragma: no cover - dependency listed in requirements
        raise ImportError("IGRF removal requires the 'ppigrf' package: pip install ppigrf") from exc
    be, bn, bu = ppigrf.igrf(np.asarray(lon, float), np.asarray(lat, float), np.asarray(height_m, float) / 1000.0, date)
    return np.sqrt(np.ravel(be) ** 2 + np.ravel(bn) ** 2 + np.ravel(bu) ** 2)


def igrf_field_direction(lon: float, lat: float, height_m: float, date: datetime) -> dict:
    """Return IGRF inclination, declination and intensity at a point (useful for RTP inputs)."""

    import ppigrf

    be, bn, bu = (float(np.ravel(v)[0]) for v in ppigrf.igrf(lon, lat, height_m / 1000.0, date))
    horizontal = np.hypot(be, bn)
    return {
        "inclination_deg": float(np.degrees(np.arctan2(-bu, horizontal))),
        "declination_deg": float(np.degrees(np.arctan2(be, bn))),
        "total_field_nT": float(np.sqrt(be**2 + bn**2 + bu**2)),
    }


def igrf_removal_table(
    table: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    field_col: str = "total_field",
    height_col: str = "",
    date: str = "2025-01-01",
    height_m: float = 0.0,
) -> pd.DataFrame:
    """Subtract IGRF-14 total field at each station: dT = T_obs - |B_IGRF|."""

    when = datetime.fromisoformat(date)
    heights = table[height_col].to_numpy(float) if height_col else np.full(len(table), height_m)
    igrf = igrf_total_field(table[lon_col].to_numpy(float), table[lat_col].to_numpy(float), heights, when)
    out = table.copy()
    out["igrf_nT"] = igrf
    out["magnetic_anomaly_nT"] = out[field_col].to_numpy(float) - igrf
    return out
