"""Seismic reflection processing on 2D sections (samples x traces)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import solve_toeplitz
from scipy.ndimage import map_coordinates, median_filter
from scipy.signal import hilbert, stft

from .grid_tools import GridData
from .profile import ProfileData


@dataclass
class Section:
    """Seismic/radar section: data[nt, ntraces], sample interval dt (s or m), per-trace headers."""

    data: np.ndarray
    dt: float
    x: np.ndarray | None = None
    name: str = "Section"
    units: str = "amplitude"
    sample_unit: str = "s"
    t0: float = 0.0
    headers: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, float)
        if self.data.ndim == 1:
            self.data = self.data[:, None]
        if self.data.ndim != 2:
            raise ValueError("Section data must be 2D (samples x traces).")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Sample interval must be positive.")
        self.x = np.arange(self.ntr, dtype=float) if self.x is None else np.asarray(self.x, float)
        self.headers = {k: np.asarray(v) for k, v in self.headers.items()}

    @property
    def nt(self) -> int:
        return self.data.shape[0]

    @property
    def ntr(self) -> int:
        return self.data.shape[1]

    @property
    def t(self) -> np.ndarray:
        return self.t0 + np.arange(self.nt) * self.dt

    def header(self, key: str) -> np.ndarray:
        if key in self.headers:
            return self.headers[key].astype(float)
        if key in {"offset", "x"}:
            return self.x
        raise KeyError(f"Section has no '{key}' header. Available: {', '.join(self.headers) or 'none'}")

    def derive(self, data: np.ndarray, name: str, **changes: Any) -> "Section":
        meta = {**self.metadata, **changes.pop("metadata", {})}
        return replace(self, data=np.asarray(data, float), name=name, metadata=meta, **changes)

    def select(self, mask: np.ndarray, name: str | None = None) -> "Section":
        return replace(self, data=self.data[:, mask], x=self.x[mask], name=name or self.name,
                       headers={k: v[mask] for k, v in self.headers.items()})


def ricker(freq: float, dt: float, length: float | None = None) -> np.ndarray:
    length = length or 2.5 / freq
    t = np.arange(-length / 2, length / 2 + dt / 2, dt)
    a = (np.pi * freq * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def _velocity_function(t0: np.ndarray, velocity: float | str) -> np.ndarray:
    """Constant velocity, or 't0:v, t0:v, ...' pairs interpolated in time (RMS velocity)."""

    if isinstance(velocity, str):
        pairs = [tuple(float(v) for v in item.split(":")) for item in velocity.replace(";", ",").split(",") if item.strip()]
        tv = np.asarray(pairs, float)
        return np.interp(t0, tv[:, 0], tv[:, 1])
    return np.full_like(t0, float(velocity))


def nmo_correction(gather: Section, velocity: float | str = 2000.0, stretch_mute: float = 0.5) -> Section:
    """NMO: sample the trace at t = sqrt(t0^2 + x^2 / v_rms(t0)^2); mute where stretch (t/t0 - 1) > limit."""

    t0 = gather.t
    v = _velocity_function(t0, velocity)
    offsets = gather.header("offset")
    out = np.zeros_like(gather.data)
    for j, x in enumerate(offsets):
        tx = np.sqrt(t0**2 + (x / v) ** 2)
        tr = np.interp(tx, gather.t, gather.data[:, j], left=0.0, right=0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            stretch = tx / np.maximum(t0, 1e-12) - 1.0
        tr[stretch > stretch_mute] = 0.0
        out[:, j] = tr
    return gather.derive(out, f"{gather.name}_NMO", metadata={"method": "nmo", "velocity": velocity, "stretch_mute": stretch_mute})


def cmp_stack(gathers: Section, velocity: float | str | None = None, stretch_mute: float = 0.5) -> Section:
    """CMP stack: NMO (if velocity given) then fold-normalized sum per CMP header (single gather -> 1 trace)."""

    data = nmo_correction(gathers, velocity, stretch_mute) if velocity is not None else gathers
    cmps = gathers.headers.get("cmp", np.zeros(gathers.ntr))
    unique = np.unique(cmps)
    out = np.zeros((gathers.nt, len(unique)))
    fold = np.zeros((gathers.nt, len(unique)))
    for i, c in enumerate(unique):
        sel = cmps == c
        out[:, i] = data.data[:, sel].sum(axis=1)
        fold[:, i] = (data.data[:, sel] != 0).sum(axis=1)
    stacked = out / np.maximum(fold, 1)
    return Section(stacked, gathers.dt, x=unique.astype(float), name=f"{gathers.name}_stack", units=gathers.units,
                   t0=gathers.t0, headers={"cmp": unique}, metadata={"method": "cmp_stack", "max_fold": int(fold.max())})


def dmo_correction(nmo_gathers: Section, cmp_spacing: float | None = None, n_b: int = 21) -> Section:
    """Kirchhoff DMO on NMO-corrected prestack data (Deregowski & Rocca, 1981).

    Each common-offset sample at (x, tn) is spread along b in [-h, h]: t0 = tn * sqrt(1 - b^2/h^2).
    """

    cmps = nmo_gathers.header("cmp")
    offsets = nmo_gathers.header("offset")
    ucmp = np.unique(cmps)
    dcmp = cmp_spacing or (float(np.median(np.diff(ucmp))) if len(ucmp) > 1 else 1.0)
    out = np.zeros_like(nmo_gathers.data)
    t = nmo_gathers.t
    for off in np.unique(np.abs(offsets)):
        sel = np.where(np.abs(offsets) == off)[0]
        h = off / 2.0
        if h == 0 or len(sel) < 2:
            out[:, sel] = nmo_gathers.data[:, sel]
            continue
        pos = cmps[sel]
        order = np.argsort(pos)
        sel, pos = sel[order], pos[order]
        panel = nmo_gathers.data[:, sel]
        result = np.zeros_like(panel)
        for b in np.linspace(-h, h, n_b)[1:-1]:
            factor = np.sqrt(1 - (b / h) ** 2)
            for j in range(len(sel)):
                k = np.interp(pos[j] + b, pos, np.arange(len(pos)), left=np.nan, right=np.nan)
                if np.isnan(k) or abs(pos[int(round(k))] - (pos[j] + b)) > dcmp / 2:
                    continue
                # output sample t0 receives input from tn = t0 / factor
                result[:, int(round(k))] += np.interp(t / factor, t, panel[:, j], left=0.0, right=0.0)
        out[:, sel] = result / (n_b - 2)
    return nmo_gathers.derive(out, f"{nmo_gathers.name}_DMO", metadata={"method": "dmo_kirchhoff"})


def kirchhoff_prestack_time_migration(
    gathers: Section,
    velocity: float | str = 2000.0,
    aperture: float = 2000.0,
    output_x: np.ndarray | None = None,
) -> Section:
    """Kirchhoff PreSTM with double-square-root traveltime and RMS velocity v(t0).

    t = sqrt((t0/2)^2 + (x - s)^2 / v^2) + sqrt((t0/2)^2 + (x - r)^2 / v^2).
    """

    s = gathers.header("source_x")
    r = gathers.header("receiver_x")
    xo = np.unique((s + r) / 2) if output_x is None else np.asarray(output_x, float)
    t0 = gathers.t
    v = _velocity_function(t0, velocity)
    image = np.zeros((len(t0), len(xo)))
    for j in range(gathers.ntr):
        mid = 0.5 * (s[j] + r[j])
        near = np.abs(xo - mid) <= aperture
        if not near.any():
            continue
        xs = xo[near][None, :]
        tt = np.sqrt((t0[:, None] / 2) ** 2 + ((xs - s[j]) / v[:, None]) ** 2) + \
            np.sqrt((t0[:, None] / 2) ** 2 + ((xs - r[j]) / v[:, None]) ** 2)
        idx = (tt - gathers.t0) / gathers.dt
        vals = map_coordinates(gathers.data[:, j], [idx.ravel()], order=1, mode="constant", cval=0.0).reshape(idx.shape)
        obliquity = (t0[:, None] / np.maximum(tt, 1e-9))
        image[:, near] += vals * obliquity
    return Section(image, gathers.dt, x=xo, name=f"{gathers.name}_PreSTM", units=gathers.units, t0=gathers.t0,
                   metadata={"method": "kirchhoff_prestm", "velocity": velocity, "aperture_m": aperture})


def kirchhoff_prestack_depth_migration(
    gathers: Section,
    velocity_model: GridData,
    aperture: float = 3000.0,
) -> GridData:
    """Kirchhoff PreSDM: image(x,z) = sum_traces d(t_s(x,z) + t_r(x,z)); traveltimes by shortest-path ray tracing.

    velocity_model: GridData with x = distance, y = depth (m, increasing down), values = interval velocity (m/s).
    """

    from .refraction import traveltime_field

    s = gathers.header("source_x")
    r = gathers.header("receiver_x")
    tables: dict[float, np.ndarray] = {}

    def table(pos: float) -> np.ndarray:
        key = round(float(pos), 3)
        if key not in tables:
            tables[key] = traveltime_field(velocity_model, pos, float(velocity_model.y_vector[0]))
        return tables[key]

    xx, _ = velocity_model.mesh
    image = np.zeros_like(velocity_model.values)
    for j in range(gathers.ntr):
        tt = table(s[j]) + table(r[j])
        mid = 0.5 * (s[j] + r[j])
        mask = np.abs(xx - mid) <= aperture
        idx = (tt[mask] - gathers.t0) / gathers.dt
        image[mask] += map_coordinates(gathers.data[:, j], [idx], order=1, mode="constant", cval=0.0)
    return velocity_model.with_values(image, name=f"{gathers.name}_PreSDM", units=gathers.units,
                                      metadata={"method": "kirchhoff_presdm", "vertical_axis": "depth",
                                                "aperture_m": aperture, "traveltimes": "shortest-path (Dijkstra)"})


def deconvolution(
    section: Section,
    mode: str = "spiking",
    operator_length: float = 0.1,
    prediction_lag: float = 0.02,
    prewhitening: float = 0.01,
) -> Section:
    """Wiener spiking or predictive (gap) deconvolution (Robinson & Treitel), Toeplitz solve per trace."""

    n = max(int(round(operator_length / section.dt)), 2)
    lag = max(int(round(prediction_lag / section.dt)), 1)
    out = np.zeros_like(section.data)
    for j in range(section.ntr):
        tr = section.data[:, j]
        full = np.correlate(tr, tr, "full")[len(tr) - 1:]
        if full[0] == 0:
            continue
        r = full[: n + lag + 1].copy()
        r = np.pad(r, (0, max(0, n + lag + 1 - len(r))))
        col = r[:n].copy()
        col[0] *= 1 + prewhitening
        if mode == "spiking":
            rhs = np.zeros(n)
            rhs[0] = 1.0
            f = solve_toeplitz(col, rhs)
            out[:, j] = np.convolve(tr, f)[: len(tr)]
        else:
            a = solve_toeplitz(col, r[lag:lag + n])
            pred = np.convolve(tr, a)[: len(tr)]
            shifted = np.zeros_like(tr)
            shifted[lag:] = pred[: len(tr) - lag]
            out[:, j] = tr - shifted
    return section.derive(out, f"{section.name}_decon_{mode}",
                          metadata={"method": f"{mode}_deconvolution", "operator_length_s": operator_length,
                                    "prediction_lag_s": prediction_lag, "prewhitening": prewhitening})


def fk_filter(section: Section, min_velocity: float = 1500.0, taper: float = 0.2, mode: str = "reject_slow") -> Section:
    """f-k fan filter: reject (or pass) energy with apparent velocity |f/k| below `min_velocity`."""

    dx = float(np.median(np.abs(np.diff(section.x)))) if section.ntr > 1 else 1.0
    spec = np.fft.fft2(section.data)
    f = np.fft.fftfreq(section.nt, section.dt)[:, None]
    k = np.fft.fftfreq(section.ntr, dx)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        vapp = np.abs(f) / np.abs(k)
    vapp = np.where(k == 0, np.inf, vapp)
    lo = min_velocity * (1 - taper)
    w = np.clip((vapp - lo) / max(min_velocity - lo, 1e-9), 0, 1)
    w = 0.5 - 0.5 * np.cos(np.pi * w)
    if mode == "pass_slow":
        w = 1 - w
    out = np.real(np.fft.ifft2(spec * w))
    return section.derive(out, f"{section.name}_fk", metadata={"method": "fk_filter", "min_velocity": min_velocity, "mode": mode})


def zoeppritz_aki_richards(vp1, vs1, rho1, vp2, vs2, rho2, angles_deg) -> np.ndarray:
    """Aki-Richards linearized P-P reflection coefficient vs incidence angle."""

    th = np.radians(np.asarray(angles_deg, float))
    vp, vs, rho = (vp1 + vp2) / 2, (vs1 + vs2) / 2, (rho1 + rho2) / 2
    dvp, dvs, drho = vp2 - vp1, vs2 - vs1, rho2 - rho1
    k = (vs / vp) ** 2
    s2 = np.sin(th) ** 2
    return 0.5 * (1 - 4 * k * s2) * drho / rho + dvp / (2 * np.cos(th) ** 2 * vp) - 4 * k * s2 * dvs / vs


def avo_modeling(vp1=2400.0, vs1=1200.0, rho1=2300.0, vp2=2200.0, vs2=1400.0, rho2=2100.0, max_angle=40.0) -> tuple[ProfileData, dict]:
    angles = np.linspace(0, max_angle, 41)
    rc = zoeppritz_aki_richards(vp1, vs1, rho1, vp2, vs2, rho2, angles)
    a, b = np.polyfit(np.sin(np.radians(angles)) ** 2, rc, 1)[::-1]
    avo_class = "I" if a > 0.02 else "II" if abs(a) <= 0.02 else ("III" if b < 0 else "IV")
    return (ProfileData(angles, rc, name="AVO_Rpp_vs_angle", units="Rpp", metadata={"method": "aki_richards"}),
            {"intercept_A": float(a), "gradient_B": float(b), "rutherford_williams_class": avo_class})


def avo_intercept_gradient(angle_gather: Section, angle_header: str = "angle") -> tuple[Section, Section, Section]:
    """Shuey two-term fit per time sample: R(theta) = A + B sin^2(theta); returns A, B and A*B sections."""

    s2 = np.sin(np.radians(angle_gather.header(angle_header))) ** 2
    design = np.column_stack([np.ones_like(s2), s2])
    coef, *_ = np.linalg.lstsq(design, angle_gather.data.T, rcond=None)
    a, b = coef[0], coef[1]
    make = lambda v, n: Section(v[:, None], angle_gather.dt, x=np.array([0.0]), name=f"{angle_gather.name}_{n}",  # noqa: E731
                                 t0=angle_gather.t0, metadata={"method": "avo_shuey"})
    return make(a, "Intercept"), make(b, "Gradient"), make(a * b, "AxB")


def acoustic_impedance_inversion(
    section: Section,
    wavelet_freq: float = 30.0,
    initial_impedance: float = 5e6,
    background_trend: float = 0.0,
    regularization: float = 0.1,
    reflectivity_scale: float | None = None,
) -> Section:
    """Model-based impedance inversion: d = W (1/2) D ln Z; min ||W D m/2 - d||^2 + lambda ||m - m0||^2."""

    n = section.nt
    w = ricker(wavelet_freq, section.dt)
    w = w if len(w) % 2 else w[:-1]  # odd length keeps the wavelet centred on each sample
    half = len(w) // 2
    wmat = np.zeros((n, n))
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        wmat[lo:hi, i] = w[half - (i - lo): half + (hi - i)]
    diff = np.eye(n, k=1) - np.eye(n)
    diff[-1] = 0
    op = wmat @ (0.5 * diff)
    scale = reflectivity_scale or (np.sqrt(np.mean(section.data**2)) / 0.05 if np.any(section.data) else 1.0)
    m0 = np.log(initial_impedance) + background_trend * section.t
    lhs = op.T @ op + regularization * np.eye(n)
    out = np.zeros_like(section.data)
    for j in range(section.ntr):
        d = section.data[:, j] / scale
        m = m0 + np.linalg.solve(lhs, op.T @ (d - op @ m0))
        out[:, j] = np.exp(m)
    return section.derive(out, f"{section.name}_AI", units="kg/m^2/s * m/s",
                          metadata={"method": "model_based_impedance_inversion", "reflectivity_scale": float(scale),
                                    "warning": "Absolute impedance needs a well-calibrated low-frequency model and wavelet scale."})


def complex_trace_attributes(section: Section) -> dict[str, Section]:
    """Taner et al. (1979): envelope, instantaneous phase/frequency, cosine of phase, sweetness."""

    analytic = hilbert(section.data, axis=0)
    env = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic), axis=0)
    freq = np.gradient(phase, section.dt, axis=0) / (2 * np.pi)
    sweet = env / np.sqrt(np.maximum(np.abs(freq), 1e-3))
    base = section.name
    return {
        "envelope": section.derive(env, f"{base}_envelope", metadata={"method": "envelope"}),
        "phase": section.derive(np.angle(analytic), f"{base}_inst_phase", units="rad", metadata={"method": "instantaneous_phase"}),
        "frequency": section.derive(freq, f"{base}_inst_freq", units="Hz", metadata={"method": "instantaneous_frequency"}),
        "cos_phase": section.derive(np.cos(np.angle(analytic)), f"{base}_cos_phase", units="", metadata={"method": "cosine_phase"}),
        "sweetness": section.derive(sweet, f"{base}_sweetness", metadata={"method": "sweetness"}),
    }


def coherence(section: Section, window_traces: int = 3, window_time: float = 0.02) -> Section:
    """Semblance-based coherence (Marfurt et al., 1998) in a sliding traces x time window (1 = continuous)."""

    nw = max(int(round(window_time / section.dt)), 1)
    kt = np.ones(2 * nw + 1)
    half = window_traces // 2
    d = section.data
    padded = np.pad(d, ((0, 0), (half, half)), mode="edge")
    sums = np.zeros_like(d)
    energy = np.zeros_like(d)
    for off in range(window_traces):
        sums += padded[:, off:off + d.shape[1]]
        energy += padded[:, off:off + d.shape[1]] ** 2
    num = np.apply_along_axis(lambda c: np.convolve(c, kt, "same"), 0, sums**2)
    den = window_traces * np.apply_along_axis(lambda c: np.convolve(c, kt, "same"), 0, energy)
    coh = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)
    return section.derive(coh, f"{section.name}_coherence", units="", metadata={"method": "semblance_coherence"})


def coherence_volume(volume: np.ndarray, window: tuple[int, int, int] = (5, 3, 3)) -> np.ndarray:
    """3D semblance coherence cube for volume[t, inline, xline]."""

    from scipy.ndimage import uniform_filter

    nt, ni, nx = window
    tr_sum = uniform_filter(volume, size=(1, ni, nx)) * ni * nx
    en_sum = uniform_filter(volume**2, size=(1, ni, nx)) * ni * nx
    num = uniform_filter(tr_sum**2, size=(nt, 1, 1))
    den = ni * nx * uniform_filter(en_sum, size=(nt, 1, 1))
    return np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)


def spectral_decomposition(section: Section, frequencies: str = "10,20,30,40", window: float = 0.064) -> list[Section]:
    """STFT spectral decomposition: amplitude sections at the requested frequencies (Partyka et al., 1999)."""

    fs = 1.0 / section.dt
    nper = max(int(round(window * fs)), 8)
    f, tt, z = stft(section.data.T, fs=fs, nperseg=nper, noverlap=nper - 1, boundary="even", padded=True)
    amp = np.abs(z)  # (ntr, nf, nt')
    outs = []
    for target in [float(v) for v in str(frequencies).split(",") if v.strip()]:
        i = int(np.argmin(np.abs(f - target)))
        vals = np.array([np.interp(section.t - section.t0, tt, amp[j, i]) for j in range(section.ntr)]).T
        outs.append(section.derive(vals, f"{section.name}_SD_{f[i]:.0f}Hz", metadata={"method": "stft_spectral_decomposition", "frequency_Hz": float(f[i])}))
    return outs


def q_spectral_ratio(
    section: Section, trace_1: int = 0, trace_2: int = -1, time_1: float | None = None, time_2: float | None = None,
    window: float = 0.1, f_min: float = 10.0, f_max: float = 60.0,
) -> tuple[ProfileData, dict]:
    """Spectral-ratio Q (Bath, 1974): ln(A2/A1) = -pi f dt / Q + c between two arrivals."""

    n = max(int(round(window / section.dt)), 8)
    nfft = 4 * n

    def spec(tr, t_center):
        i0 = int(round((t_center - section.t0) / section.dt)) - n // 2
        seg = np.zeros(n)
        lo, hi = max(i0, 0), min(i0 + n, section.nt)
        seg[lo - i0: hi - i0] = section.data[lo:hi, tr]
        return np.fft.rfftfreq(nfft, section.dt), np.abs(np.fft.rfft(seg * np.hanning(n), nfft))

    tr2 = trace_2 % section.ntr
    t1 = time_1 if time_1 is not None else section.t[int(np.argmax(np.abs(section.data[:, trace_1])))]
    t2 = time_2 if time_2 is not None else section.t[int(np.argmax(np.abs(section.data[:, tr2])))]
    f, a1 = spec(trace_1, t1)
    _, a2 = spec(tr2, t2)
    band = (f >= f_min) & (f <= f_max) & (a1 > 0) & (a2 > 0)
    ratio = np.log(a2[band] / a1[band])
    slope, intercept = np.polyfit(f[band], ratio, 1)
    dt_travel = t2 - t1
    q = -np.pi * dt_travel / slope if slope < 0 else np.inf
    return (ProfileData(f[band], ratio, name="Q_spectral_ratio", units="ln(A2/A1)", metadata={"method": "spectral_ratio"}),
            {"Q": float(q), "slope": float(slope), "travel_time_difference_s": float(dt_travel)})


def vsp_corridor_stack(
    vsp: Section, first_break_velocity: float = 2500.0, corridor_length: float = 0.1, median_traces: int = 7,
) -> tuple[Section, Section, ProfileData]:
    """Zero-offset VSP: flatten on first breaks, median-filter downgoing, extract upgoing, shift to TWT, corridor stack.

    Trace x = receiver depth (m); first breaks t_fb = depth / v (or 'first_break' header if present).
    """

    depth = vsp.x
    tfb = vsp.headers.get("first_break", depth / first_break_velocity).astype(float)
    t = vsp.t
    flat = np.array([np.interp(t + tfb[j], t, vsp.data[:, j], left=0, right=0) for j in range(vsp.ntr)]).T
    down_flat = median_filter(flat, size=(1, median_traces), mode="nearest")
    down = np.array([np.interp(t - tfb[j], t, down_flat[:, j], left=0, right=0) for j in range(vsp.ntr)]).T
    up = vsp.data - down
    up_twt = np.array([np.interp(t - tfb[j], t, up[:, j], left=0, right=0) for j in range(vsp.ntr)]).T
    mask = np.zeros_like(up_twt)
    for j in range(vsp.ntr):
        mask[(t >= 2 * tfb[j]) & (t <= 2 * tfb[j] + corridor_length), j] = 1.0
    stack = (up_twt * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
    return (vsp.derive(up_twt, f"{vsp.name}_upgoing_TWT", metadata={"method": "vsp_upgoing"}),
            vsp.derive(down, f"{vsp.name}_downgoing", metadata={"method": "vsp_downgoing_median"}),
            ProfileData(t, stack, name=f"{vsp.name}_corridor_stack", units=vsp.units, metadata={"method": "corridor_stack", "axis": "time_s"}))


def stolt_migration(section: Section, velocity: float = 2000.0, exploding_reflector: bool = True) -> Section:
    """Stolt f-k migration (constant velocity), for zero-offset seismic (v/2) or GPR (set exploding_reflector)."""

    v = velocity / 2.0 if exploding_reflector else velocity
    nt, nx = section.data.shape
    dx = float(np.median(np.abs(np.diff(section.x)))) if nx > 1 else 1.0
    ntp = 2 * nt
    spec = np.fft.fft2(section.data, s=(ntp, nx))
    f = np.fft.fftfreq(ntp, section.dt)
    kx = np.fft.fftfreq(nx, dx)
    out = np.zeros_like(spec)
    for j, k in enumerate(kx):
        fmig = np.sign(f) * np.sqrt(f**2 + (v * k) ** 2)
        col = spec[:, j]
        order = np.argsort(f)
        re = np.interp(fmig, f[order], col.real[order], left=0, right=0)
        im = np.interp(fmig, f[order], col.imag[order], left=0, right=0)
        scale = np.where(np.abs(fmig) > 0, np.abs(f) / np.maximum(np.abs(fmig), 1e-12), 0.0)
        out[:, j] = (re + 1j * im) * scale
    mig = np.real(np.fft.ifft2(out))[:nt]
    return section.derive(mig, f"{section.name}_Stolt", metadata={"method": "stolt_fk_migration", "velocity": velocity})


def section_to_table(section: Section) -> pd.DataFrame:
    return pd.DataFrame(section.data, columns=[f"tr{j}" for j in range(section.ntr)]).assign(time=section.t)
