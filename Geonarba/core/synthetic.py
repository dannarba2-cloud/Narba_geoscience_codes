"""Physically consistent synthetic demo datasets for every GEONARBA workflow (used by the Demo Library and tests)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .electrical import cole_cole, make_array, make_crosshole_array, sp_forward, ves_forward
from .em import MU0, mt1d_forward
from .gis import RasterImage
from .grid_tools import GridData
from .gravity import bouguer_slab, free_air_correction, longman_tide, normal_gravity
from .prisms import gz_kernel, magnetic_kernel, susceptibility_to_magnetization
from .profile import ProfileData
from .seismic import Section, ricker, zoeppritz_aki_richards

RNG = np.random.default_rng(42)
INC, DEC = 30.0, 5.0  # demo geomagnetic field direction


def _survey_grid(n: int = 64, spacing: float = 500.0):
    x = np.arange(n) * spacing
    xx, yy = np.meshgrid(x, x)
    return x, xx, yy


def magnetic_grid(n: int = 64, spacing: float = 500.0) -> GridData:
    """TMI anomaly (nT) of three induced prisms at I=30, D=5 plus noise."""

    x, xx, yy = _survey_grid(n, spacing)
    obs = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, -100.0)])
    prisms = np.array([[8000, 12000, 9000, 14000, 800, 3000], [18000, 26000, 20000, 22000, 1500, 6000], [22000, 24000, 6000, 12000, 400, 1200]])
    chi = np.array([0.03, 0.02, 0.05])
    m = susceptibility_to_magnetization(chi, 45000.0)
    tmi = (magnetic_kernel(obs, prisms, INC, DEC) @ m).reshape(xx.shape)
    return GridData(tmi + RNG.normal(0, 0.5, xx.shape), x, x, spacing, spacing, name="Demo_TMI", units="nT",
                    metadata={"inclination": INC, "declination": DEC, "source": "synthetic prisms"})


def gravity_grid(n: int = 64, spacing: float = 500.0) -> GridData:
    """Bouguer anomaly (mGal): sedimentary basin (-), dense intrusion (+), regional trend."""

    x, xx, yy = _survey_grid(n, spacing)
    obs = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    prisms = np.array([[6000, 16000, 6000, 20000, 0, 2500], [20000, 26000, 18000, 24000, 1500, 5000], [10000, 12000, 22000, 30000, 300, 1500]])
    rho = np.array([-250.0, 300.0, 200.0])
    g = (gz_kernel(obs, prisms) @ rho).reshape(xx.shape) * 1e5
    return GridData(g + 0.0002 * xx + RNG.normal(0, 0.1, xx.shape), x, x, spacing, spacing, name="Demo_Bouguer", units="mGal",
                    metadata={"source": "synthetic prisms + regional"})


def dem_grid(n: int = 64, spacing: float = 500.0) -> GridData:
    x, xx, yy = _survey_grid(n, spacing)
    z = 400 + 600 * np.exp(-((xx - 20000) ** 2 + (yy - 18000) ** 2) / (2 * 6000**2)) + 150 * np.sin(xx / 3000) * np.cos(yy / 4000)
    z += 80 * np.exp(-((xx - 8000) ** 2) / (2 * 1500**2))
    return GridData(z, x, x, spacing, spacing, name="Demo_DEM", units="m")


def radiometric_grids(n: int = 64, spacing: float = 500.0) -> tuple[GridData, GridData, GridData]:
    x, xx, yy = _survey_grid(n, spacing)
    granite = np.exp(-((xx - 20000) ** 2 + (yy - 18000) ** 2) / (2 * 5000**2))
    alteration = np.exp(-((xx - 9000) ** 2 + (yy - 25000) ** 2) / (2 * 2500**2))
    k = 1.0 + 2.5 * granite + 2.0 * alteration + RNG.normal(0, 0.1, xx.shape)
    u = 1.5 + 4.0 * granite + RNG.normal(0, 0.2, xx.shape)
    th = 6.0 + 18.0 * granite - 3 * alteration + RNG.normal(0, 0.5, xx.shape)
    mk = lambda v, name, units: GridData(np.clip(v, 0.01, None), x, x, spacing, spacing, name=name, units=units)  # noqa: E731
    return mk(k, "Demo_K", "%"), mk(u, "Demo_eU", "ppm"), mk(th, "Demo_eTh", "ppm")


def gravity_stations(dem: GridData, n: int = 150) -> pd.DataFrame:
    """Observed absolute gravity at DEM-surface stations consistent with normal gravity, FA and Bouguer reductions."""

    xs = RNG.uniform(dem.x_vector.min(), dem.x_vector.max(), n)
    ys = RNG.uniform(dem.y_vector.min(), dem.y_vector.max(), n)
    from scipy.interpolate import RegularGridInterpolator

    h = RegularGridInterpolator((dem.y_vector, dem.x_vector), dem.values)(np.column_stack([ys, xs]))
    lat = 35.0 + ys / 111000.0
    lon = 10.0 + xs / (111000.0 * np.cos(np.radians(35)))
    anomaly = 5 * np.exp(-((xs - 15000) ** 2 + (ys - 15000) ** 2) / (2 * 5000**2))
    g = normal_gravity(lat) - free_air_correction(h, lat) + bouguer_slab(h) + anomaly + RNG.normal(0, 0.02, n)
    return pd.DataFrame({"station": np.arange(n), "x": xs, "y": ys, "lon": lon, "lat": lat, "elevation": h, "gravity": g})


def drift_survey() -> pd.DataFrame:
    t0 = pd.Timestamp("2025-03-10 07:00:00", tz="UTC")
    times, stations = [], []
    for loop in range(4):
        times.append(t0 + pd.Timedelta(minutes=150 * loop))
        stations.append("BASE")
        for s in range(6):
            times.append(t0 + pd.Timedelta(minutes=150 * loop + 20 * (s + 1)))
            stations.append(f"S{loop * 6 + s + 1}")
    times.append(t0 + pd.Timedelta(minutes=600))
    stations.append("BASE")
    times = pd.to_datetime(times)
    hours = (times - t0).total_seconds() / 3600
    true = np.array([0.0 if s == "BASE" else 0.05 * int(s[1:]) for s in stations])
    reading = 2500.0 + true + 0.012 * hours - longman_tide(times, 35.0, 10.0) + RNG.normal(0, 0.003, len(times))
    return pd.DataFrame({"time": times.astype(str), "station": stations, "reading": reading})


def magnetic_stations(n: int = 80) -> pd.DataFrame:
    from .potential_field import igrf_total_field
    from datetime import datetime

    lon = RNG.uniform(10.0, 10.3, n)
    lat = RNG.uniform(35.0, 35.3, n)
    try:
        igrf = igrf_total_field(lon, lat, np.zeros(n), datetime(2025, 1, 1))
    except ImportError:
        igrf = np.full(n, 44000.0)
    anomaly = 150 * np.exp(-((lon - 10.15) ** 2 + (lat - 35.15) ** 2) / (2 * 0.03**2))
    return pd.DataFrame({"lon": lon, "lat": lat, "total_field": igrf + anomaly + RNG.normal(0, 1, n)})


# ---------------------------------------------------------------- seismic
REFLECTORS = [(0.35, 1800.0, 0.3), (0.60, 2200.0, -0.2), (0.85, 2600.0, 0.25), (1.10, 3000.0, 0.2)]  # t0, v_rms, rc


def cmp_gathers(n_cmp: int = 24, n_offsets: int = 24, dt: float = 0.004, nt: int = 400) -> Section:
    """Multi-CMP prestack data: hyperbolic reflections, a dipping reflector, linear ground roll and noise."""

    t = np.arange(nt) * dt
    w = ricker(25, dt)
    offsets = np.arange(1, n_offsets + 1) * 50.0
    cmps = np.arange(n_cmp) * 25.0
    data, off_h, cmp_h, sx, rx = [], [], [], [], []
    for c in cmps:
        for x in offsets:
            tr = np.zeros(nt)
            for t0, v, rc in REFLECTORS:
                t0c = t0 + 0.0004 * c / 25 * (t0 > 0.8)  # gently dipping deep reflectors
                tt = np.sqrt(t0c**2 + (x / v) ** 2)
                k = int(round(tt / dt))
                if k < nt:
                    tr[k] += rc
            tr = np.convolve(tr, w, "same")
            tg = x / 400.0
            tr += 0.6 * np.exp(-((t - tg) / 0.03) ** 2) * np.sin(2 * np.pi * 8 * (t - tg)) * (t > tg - 0.1)
            data.append(tr + RNG.normal(0, 0.02, nt))
            off_h.append(x)
            cmp_h.append(c)
            sx.append(c - x / 2)
            rx.append(c + x / 2)
    return Section(np.array(data).T, dt, x=np.array(off_h), name="Demo_CMP_gathers",
                   headers={"offset": np.array(off_h), "cmp": np.array(cmp_h), "source_x": np.array(sx), "receiver_x": np.array(rx)},
                   metadata={"velocity_function": ",".join(f"{t0}:{v}" for t0, v, _ in REFLECTORS)})


def zero_offset_section(ntr: int = 120, nt: int = 300, dt: float = 0.004) -> tuple[Section, np.ndarray]:
    """Stacked-like section with a normal fault, a channel and a diffractor; also returns true reflectivity."""

    refl = np.zeros((nt, ntr))
    for base, amp in [(60, 0.2), (100, -0.15), (140, 0.25), (190, 0.2), (230, -0.2)]:
        for j in range(ntr):
            k = base + int(0.15 * j) + (12 if j > ntr // 2 else 0)
            if k < nt:
                refl[k, j] += amp
    refl[118:121, 30:45] -= 0.3  # channel
    refl[80, 90] += 1.0
    w = ricker(30, dt)
    data = np.apply_along_axis(lambda c: np.convolve(c, w, "same"), 0, refl)
    data += RNG.normal(0, 0.01, data.shape)
    return Section(data, dt, x=np.arange(ntr) * 12.5, name="Demo_stack_section"), refl


def angle_gather(nt: int = 250, dt: float = 0.004) -> Section:
    angles = np.arange(0, 41, 2.0)
    w = ricker(30, dt)
    data = np.zeros((nt, len(angles)))
    layers = [(80, (2400, 1200, 2300, 2200, 1400, 2100)), (150, (2500, 1250, 2350, 2900, 1500, 2450))]
    for k, props in layers:
        rc = zoeppritz_aki_richards(*props, angles)
        data[k] += rc
    data = np.apply_along_axis(lambda c: np.convolve(c, w, "same"), 0, data)
    return Section(data, dt, x=angles, name="Demo_angle_gather", headers={"angle": angles})


def velocity_model(nx: int = 120, nz: int = 60, h: float = 10.0) -> GridData:
    x, z = np.arange(nx) * h, np.arange(nz) * h
    zz, xx = np.meshgrid(z, x, indexing="ij")
    v = 1800 + 1.2 * zz
    v[zz > 350 + 0.2 * (xx - 600)] = 3200
    v[(zz - 250) ** 2 + (xx - 600) ** 2 < 70**2] = 2600
    return GridData(v, x, z, h, h, name="Demo_velocity_model", units="m/s", metadata={"vertical_axis": "depth"})


def refraction_shot(v1: float = 500.0, v2: float = 2500.0, h: float = 12.0, n: int = 48, spacing: float = 3.0, dt: float = 0.0005, nt: int = 400) -> Section:
    x = np.arange(1, n + 1) * spacing
    ic = np.arcsin(v1 / v2)
    t_first = np.minimum(x / v1, x / v2 + 2 * h * np.cos(ic) / v1)
    t = np.arange(nt) * dt
    tw = np.arange(int(0.04 / dt)) * dt
    w = np.sin(2 * np.pi * 60 * tw) * np.exp(-tw / 0.008)  # causal wavelet: energy starts exactly at the first break
    data = np.zeros((nt, n))
    for j, tf in enumerate(t_first):
        k = int(round(tf / dt))
        if k < nt:
            data[k, j] = 1.0
    data = np.apply_along_axis(lambda c: np.convolve(c, w)[:nt], 0, data)
    data += RNG.normal(0, 0.03, data.shape)
    return Section(data, dt, x=x, name="Demo_refraction_shot", headers={"offset": x, "true_first_break": t_first})


def refraction_reversed(v1: float = 500.0, v2: float = 2500.0, length: float = 120.0) -> pd.DataFrame:
    """Forward/reverse head-wave times over a gently dipping refractor (for plus-minus and GRM)."""

    xs = np.arange(15.0, length - 14, 3.0)
    ic = np.arcsin(v1 / v2)
    depth = lambda x: 10 + 0.05 * x  # noqa: E731
    tf = xs / v2 + (depth(0) + depth(xs)) * np.cos(ic) / v1
    tr = (length - xs) / v2 + (depth(length) + depth(xs)) * np.cos(ic) / v1
    t_ab = length / v2 + (depth(0) + depth(length)) * np.cos(ic) / v1
    return pd.DataFrame({"x": xs, "t_forward": tf, "t_reverse": tr, "true_depth": depth(xs) * np.cos(np.arctan(0.05)),
                         "reciprocal_time": t_ab})


def refraction_picks() -> pd.DataFrame:
    from .refraction import traveltime_field

    x = np.arange(0, 121, 2.0)
    z = np.arange(0, 41, 2.0)
    zz, xx = np.meshgrid(z, x, indexing="ij")
    v = np.where(zz < 8 + 0.08 * xx, 600.0, 2400.0)
    model = GridData(v, x, z, 2, 2)
    rows = []
    for s in np.arange(0, 121, 20.0):
        tt = traveltime_field(model, s, 0.0)
        for r in np.arange(4, 121, 4.0):
            if abs(r - s) > 1:
                rows.append({"source_x": s, "receiver_x": r, "pick_time": float(tt[0, int(r / 2)]) + RNG.normal(0, 0.0003)})
    return pd.DataFrame(rows)


def masw_shot(n: int = 48, spacing: float = 1.0, dt: float = 0.001, nt: int = 1000) -> Section:
    x = 5.0 + np.arange(n) * spacing
    f = np.fft.rfftfreq(nt, dt)
    c = 150 + 250 / (1 + (f / 15) ** 2)  # normally dispersive Rayleigh phase velocity
    src = np.fft.rfft(ricker(20, dt, 0.2), nt)
    spec = src[:, None] * np.exp(-1j * 2 * np.pi * f[:, None] * x[None, :] / np.maximum(c[:, None], 1)) / np.sqrt(x)[None, :]
    data = np.fft.irfft(spec, nt, axis=0)
    return Section(data + RNG.normal(0, 0.002, data.shape), dt, x=x, name="Demo_MASW_shot", headers={"offset": x})


def vsp_section(n: int = 40, dz: float = 25.0, dt: float = 0.002, nt: int = 600, v: float = 2500.0) -> Section:
    depth = 100 + np.arange(n) * dz
    t = np.arange(nt) * dt
    w = ricker(30, dt)
    data = np.zeros((nt, n))
    reflectors = [(900.0, 0.3), (1300.0, -0.25)]
    for j, zr in enumerate(depth):
        for zref, rc in reflectors:
            if zr < zref:
                k = int(round((2 * zref - zr) / v / dt))
                if k < nt:
                    data[k, j] += rc
    data = np.apply_along_axis(lambda c: np.convolve(c, w, "same"), 0, data)
    # direct arrival with constant-Q attenuation (Q = 60): A(f) = W(f) exp(-pi f t / Q)
    f = np.fft.rfftfreq(nt, dt)
    wf = np.fft.rfft(w, nt)
    for j, zr in enumerate(depth):
        tfb = zr / v
        spec = wf * np.exp(-np.pi * f * tfb / 60.0) * np.exp(-2j * np.pi * f * (tfb - (len(w) // 2) * dt))
        data[:, j] += np.fft.irfft(spec, nt)
    return Section(data + RNG.normal(0, 0.005, data.shape), dt, x=depth, name="Demo_VSP", headers={"first_break": depth / v})


def three_component_noise(duration: float = 600.0, fs: float = 100.0, f0: float = 2.5) -> pd.DataFrame:
    n = int(duration * fs)
    f = np.fft.rfftfreq(n, 1 / fs)
    res = 1 + 3.0 / np.sqrt(1 + ((f - f0) / (0.15 * f0)) ** 2)
    comp = lambda gain: np.fft.irfft((RNG.normal(size=len(f)) + 1j * RNG.normal(size=len(f))) * gain, n)  # noqa: E731
    return pd.DataFrame({"time": np.arange(n) / fs, "north": comp(res), "east": comp(res), "vertical": comp(np.ones_like(f))})


def spac_array(duration: float = 300.0, fs: float = 100.0, radius: float = 20.0, n_ring: int = 6) -> pd.DataFrame:
    n = int(duration * fs)
    f = np.fft.rfftfreq(n, 1 / fs)
    c = 200 + 400 / (1 + (f / 5) ** 2)
    pos = [(0.0, 0.0)] + [(radius * np.cos(a), radius * np.sin(a)) for a in np.linspace(0, 2 * np.pi, n_ring, endpoint=False)]
    out = {f"ring{i}": np.zeros(n) for i in range(1, n_ring + 1)}
    out["center"] = np.zeros(n)
    for _ in range(40):  # plane waves from random azimuths
        az = RNG.uniform(0, 2 * np.pi)
        amp = RNG.normal(size=len(f)) + 1j * RNG.normal(size=len(f))
        for i, (px, py) in enumerate(pos):
            delay = (px * np.cos(az) + py * np.sin(az)) / np.maximum(c, 1)
            sig = np.fft.irfft(amp * np.exp(-1j * 2 * np.pi * f * delay), n)
            out["center" if i == 0 else f"ring{i}"] += sig
    return pd.DataFrame(out)


def receiver_function_data(fs: float = 20.0, h: float = 35.0, vp: float = 6.3, k: float = 1.75, p: float = 0.06) -> pd.DataFrame:
    n = int(60 * fs)
    t = np.arange(n) / fs
    src = np.exp(-((t - 10) / 0.8) ** 2) * np.sin(2 * np.pi * 0.4 * (t - 10))
    vs = vp / k
    qa, qb = np.sqrt(1 / vp**2 - p**2), np.sqrt(1 / vs**2 - p**2)
    ps, ppps, ppss = h * (qb - qa), h * (qb + qa), 2 * h * qb
    shift = lambda s, dt: np.interp(t - dt, t, s, left=0, right=0)  # noqa: E731
    radial = 0.2 * src + 0.35 * shift(src, ps) + 0.15 * shift(src, ppps) - 0.1 * shift(src, ppss)
    return pd.DataFrame({"time": t, "vertical": src + RNG.normal(0, 0.005, n), "radial": radial + RNG.normal(0, 0.005, n)})


def focal_polarities(strike: float = 30.0, dip: float = 60.0, rake: float = -90.0, n: int = 40) -> pd.DataFrame:
    from .passive import _radiation_sign

    az = RNG.uniform(0, 360, n)
    to = RNG.uniform(20, 80, n)
    pol = np.sign(_radiation_sign(strike, dip, rake, az, to))
    return pd.DataFrame({"azimuth": az, "takeoff": to, "polarity": pol})


def noise_pair(duration: float = 1800.0, fs: float = 50.0, distance: float = 2000.0, velocity: float = 1000.0) -> pd.DataFrame:
    n = int(duration * fs)
    lag = int(distance / velocity * fs)
    src = RNG.normal(size=n + lag)
    return pd.DataFrame({"a": src[lag:] + 0.3 * RNG.normal(size=n), "b": src[:n] + 0.3 * RNG.normal(size=n)})


# ---------------------------------------------------------------- electrical / EM
def ves_sounding() -> pd.DataFrame:
    ab2 = np.geomspace(1.5, 300, 25)
    rho = ves_forward(ab2, np.array([80.0, 15.0, 400.0]), np.array([4.0, 18.0]))
    return pd.DataFrame({"ab2": ab2, "rhoa": rho * np.exp(RNG.normal(0, 0.02, len(ab2)))})


def ert_survey() -> pd.DataFrame:
    from .electrical import ert_forward

    tab = make_array(24, 5.0, "dipole-dipole", 6)
    x = np.arange(-60, 180, 1.0)
    z = np.arange(0, 60, 1.0)
    zz, xx = np.meshgrid(z, x, indexing="ij")
    rho = np.full(zz.shape, 120.0)
    rho[(xx > 40) & (xx < 65) & (zz > 4) & (zz < 14)] = 12.0
    rho[zz > 25] = 400.0
    out = ert_forward(tab, GridData(rho, x, z, 1, 1))
    out["rhoa"] *= np.exp(RNG.normal(0, 0.01, len(out)))
    charge = np.where((xx > 40) & (xx < 65) & (zz > 4) & (zz < 14), 0.08, 0.005)
    ip = ert_forward(tab, GridData(rho / (1 - charge), x, z, 1, 1))
    out["chargeability_mV_V"] = 1000 * (1 - out["rhoa"] / ip["rhoa"]).clip(lower=0)
    return out


def crosshole_survey() -> pd.DataFrame:
    from .electrical import ert_forward

    tab = make_crosshole_array(0.0, 16.0, 24.0, 2.0)
    x = np.arange(-10, 27, 0.5)
    z = np.arange(0, 40, 0.5)
    zz, xx = np.meshgrid(z, x, indexing="ij")
    rho = np.full(zz.shape, 200.0)
    rho[(xx - 8) ** 2 + (zz - 12) ** 2 < 3**2] = 20.0
    return ert_forward(tab, GridData(rho, x, z, 0.5, 0.5))


def sip_spectrum() -> pd.DataFrame:
    f = np.geomspace(0.01, 10000, 30)
    z = cole_cole(f, 150.0, 0.25, 0.05, 0.5)
    return pd.DataFrame({"frequency": f, "amplitude": np.abs(z), "phase_mrad": np.angle(z) * 1000 + RNG.normal(0, 0.2, len(f))})


def sp_profile() -> ProfileData:
    x = np.linspace(-200, 200, 81)
    return ProfileData(x, sp_forward(x, -8000.0, 15.0, 35.0, 30.0, 1.5) + RNG.normal(0, 0.3, len(x)), name="Demo_SP", units="mV")


def vlf_profile() -> ProfileData:
    x = np.arange(0, 400, 10.0)
    tilt = 25 * np.arctan((x - 200) / 25) / (np.pi / 2) * np.exp(-((x - 200) / 150) ** 2)
    return ProfileData(x, -tilt + RNG.normal(0, 0.5, len(x)), name="Demo_VLF_tilt", units="%")


def mt_time_series(fs: float = 100.0, duration: float = 600.0) -> pd.DataFrame:
    n = int(fs * duration)
    f = np.fft.rfftfreq(n, 1 / fs)
    rho, thick = np.array([100.0, 10.0, 1000.0]), np.array([1000.0, 2000.0])
    fz = np.where(f > 0, f, f[1])
    z_si = np.sqrt(mt1d_forward(fz, rho, thick)[0] * 2 * np.pi * fz * MU0) * np.exp(1j * np.radians(mt1d_forward(fz, rho, thick)[1]))
    z_field = z_si / (MU0 * 1e3)  # numpy FFT is e^{+iwt}: Zxy phase in the first quadrant; units (mV/km)/nT
    hx = RNG.normal(size=len(f)) + 1j * RNG.normal(size=len(f))
    hy = RNG.normal(size=len(f)) + 1j * RNG.normal(size=len(f))
    ex, ey = z_field * hy, -z_field * hx
    hz = 0.2 * hx - 0.1 * hy
    to_t = lambda s: np.fft.irfft(s, n)  # noqa: E731
    return pd.DataFrame({"ex": to_t(ex), "ey": to_t(ey), "hx": to_t(hx), "hy": to_t(hy), "hz": to_t(hz)})


def mt_response() -> pd.DataFrame:
    f = np.geomspace(1e-3, 1e3, 30)
    r, p = mt1d_forward(f, np.array([100.0, 10.0, 1000.0]), np.array([1000.0, 2000.0]))
    return pd.DataFrame({"frequency": f, "rho_det": r * np.exp(RNG.normal(0, 0.03, len(f))), "phase_det": p + RNG.normal(0, 0.8, len(f))})


def csamt_data() -> pd.DataFrame:
    f = np.geomspace(1, 8192, 14)
    r, p = mt1d_forward(f, np.array([300.0, 30.0]), np.array([150.0]))
    hy = np.ones_like(f)
    ex = np.sqrt(r * f / 0.2) * hy
    return pd.DataFrame({"frequency": f, "ex": ex, "hy": hy, "phase": p})


def tem_decay(rho: float = 50.0, tau: float = 2e-3) -> pd.DataFrame:
    t = np.geomspace(1e-5, 1e-2, 25)
    moment = 1e4
    halfspace = (2 * MU0 * moment / 5) / t * (MU0 / (4 * np.pi * t * rho)) ** 1.5  # consistent with late-time rho_a
    conductor = 2e-6 * np.exp(-t / tau)
    return pd.DataFrame({"time": t, "dbdt": halfspace + conductor})


def aem_line() -> pd.DataFrame:
    rows = []
    t = np.geomspace(2e-5, 8e-3, 18)
    moment = 5e5
    for x in np.arange(0, 3001, 50.0):
        rho = 100 - 85 * np.exp(-((x - 1500) / 300) ** 2)
        v = (2 * MU0 * moment / 5) / t * (MU0 / (4 * np.pi * t * rho)) ** 1.5
        rows += [{"x": x, "time": ti, "dbdt": vi} for ti, vi in zip(t, v)]
    return pd.DataFrame(rows)


def fdem_survey(n: int = 200) -> pd.DataFrame:
    x = RNG.uniform(0, 200, n)
    y = RNG.uniform(0, 200, n)
    sigma = 20 + 60 * np.exp(-((x - 120) ** 2 + (y - 80) ** 2) / (2 * 25**2))  # mS/m
    q = sigma / 1000 * 2 * np.pi * 9800 * MU0 * 3.66**2 / 4 * 1000
    return pd.DataFrame({"x": x, "y": y, "quadrature_ppt": q + RNG.normal(0, 0.05, n)})


def gpr_section(ntr: int = 100, nt: int = 256, dt: float = 0.2e-9, v: float = 1e8) -> Section:
    x = np.arange(ntr) * 0.05
    t = np.arange(nt) * dt
    w = ricker(400e6, dt)
    data = np.zeros((nt, ntr))
    for x0, z0 in [(1.5, 0.6), (3.5, 1.0)]:
        tt = 2 * np.sqrt((x - x0) ** 2 + z0**2) / v
        k = np.round(tt / dt).astype(int)
        ok = k < nt
        data[k[ok], np.arange(ntr)[ok]] = 1.0
    data[int(0.8 / v * 2 / dt), :] += 0.5
    data = np.apply_along_axis(lambda c: np.convolve(c, w, "same"), 0, data)
    data += 0.8 * np.exp(-t / 8e-9)[:, None]  # wow
    data += 0.6 * np.exp(-((t - 3e-9) / 1e-9) ** 2)[:, None]  # direct/air wave banding
    return Section(data + RNG.normal(0, 0.02, data.shape), dt, x=x, name="Demo_GPR", units="mV")


# ---------------------------------------------------------------- wells
def well_logs(n: int = 400) -> pd.DataFrame:
    depth = 1500 + np.arange(n) * 0.5
    lith = np.where((depth % 60) < 20, "shale", np.where((depth % 60) < 45, "sandstone", "limestone"))
    vsh = np.where(lith == "shale", 0.85, 0.1) + RNG.normal(0, 0.03, n)
    phi = np.where(lith == "sandstone", 0.24, np.where(lith == "limestone", 0.12, 0.06)) + RNG.normal(0, 0.01, n)
    hc = (lith == "sandstone") & (depth < 1600)
    sw = np.where(hc, 0.3, 1.0)
    rt = 0.05 / (phi**2 * sw**2)
    rt = np.where(lith == "shale", 2.0, rt)
    rma = np.where(lith == "limestone", 2.71, 2.65)
    rhob = rma * (1 - phi) + phi * 1.0
    nphi = phi + np.where(lith == "shale", 0.2, 0) - np.where(hc, 0.05, 0)
    gr = 25 + 110 * np.clip(vsh, 0, 1) + RNG.normal(0, 3, n)
    sp = -60 * (1 - np.clip(vsh, 0, 1)) + 0.02 * (depth - depth[0]) + RNG.normal(0, 1, n)
    dt = np.where(lith == "shale", 100, np.where(lith == "sandstone", 85, 60)) - 0 * phi
    return pd.DataFrame({"DEPTH": depth, "GR": gr, "SP": sp, "RT": rt, "RHOB": rhob, "NPHI": nphi, "DT": dt, "PHI": phi,
                         "lithology": lith, "porosity": phi})


def dipmeter_picks(diameter: float = 0.216) -> pd.DataFrame:
    rows = []
    for fid, (z0, dip, azi) in enumerate([(1510.0, 20.0, 120.0), (1532.0, 35.0, 200.0), (1555.0, 10.0, 80.0)]):
        amp = diameter / 2 * np.tan(np.radians(dip))
        for th in np.arange(0, 360, 30.0):
            rows.append({"feature": fid, "azimuth": th, "depth": z0 + amp * np.cos(np.radians(th - azi)) + RNG.normal(0, 0.001)})
    return pd.DataFrame(rows)


def nmr_echo_train() -> pd.DataFrame:
    t = np.arange(1, 2001) * 0.6
    amp = 0.06 * np.exp(-t / 8) + 0.18 * np.exp(-t / 150) + 0.04 * np.exp(-t / 900)
    return pd.DataFrame({"time_ms": t, "amplitude": amp + RNG.normal(0, 0.002, len(t))})


# ---------------------------------------------------------------- GIS helpers
def deposits(n: int = 25) -> pd.DataFrame:
    return pd.DataFrame({"x": RNG.normal(20000, 2500, n), "y": RNG.normal(18000, 2500, n)})


def geology_zones(ref: GridData) -> GridData:
    xx, yy = ref.mesh
    zones = 1 + (xx > 12000).astype(int) + (yy > 20000).astype(int) * 2
    return ref.with_values(zones.astype(float), "Demo_geology_zones", "class")


def polygons_table() -> pd.DataFrame:
    rows = []
    for pid, (cx, cy, r) in enumerate([(10000, 10000, 4000), (22000, 22000, 5000)], start=1):
        for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            rows.append({"id": pid, "x": cx + r * np.cos(a), "y": cy + r * np.sin(a), "value": pid * 10})
    return pd.DataFrame(rows)


def scanned_map() -> tuple[GridData, pd.DataFrame]:
    """A 'scanned' raster in pixel coordinates (x = column, y = row) plus ground-control points."""

    img = np.add.outer(np.arange(80), np.arange(100)).astype(float) % 20
    grid = GridData(img, np.arange(100.0), np.arange(80.0), 1, 1, name="Demo_scanned_map", units="DN")
    gcps = pd.DataFrame({"col": [0, 99, 0, 99, 50], "row": [0, 0, 79, 79, 40]})
    gcps["x"] = 500000 + 25 * gcps["col"] + 2 * gcps["row"]
    gcps["y"] = 4000000 + 25 * gcps["row"] - 1.5 * gcps["col"]
    return grid, gcps


def temperature_observations() -> pd.DataFrame:
    z = np.array([1000, 3000, 6000, 10000, 15000.0])
    return pd.DataFrame({"depth": z, "temperature": 15 + 0.07 * z / 2.5 - 1.5e-6 * z**2 / 5})


def demo_library() -> dict[str, object]:
    """All demo datasets keyed by display name."""

    dem = dem_grid()
    k, u, th = radiometric_grids()
    stack, _ = zero_offset_section()
    img, gcps = scanned_map()
    grav = gravity_grid()
    lib: dict[str, object] = {
        "Demo_TMI": magnetic_grid(),
        "Demo_Bouguer": grav,
        "Demo_DEM": dem,
        "Demo_K": k, "Demo_eU": u, "Demo_eTh": th,
        "Demo_geology_zones": geology_zones(grav),
        "Demo_scanned_map": img,
        "Demo_velocity_model": velocity_model(),
        "Demo_gravity_stations": gravity_stations(dem),
        "Demo_gravity_drift_survey": drift_survey(),
        "Demo_magnetic_stations": magnetic_stations(),
        "Demo_CMP_gathers": cmp_gathers(),
        "Demo_stack_section": stack,
        "Demo_angle_gather": angle_gather(),
        "Demo_refraction_shot": refraction_shot(),
        "Demo_refraction_forward_reverse": refraction_reversed(),
        "Demo_refraction_picks": refraction_picks(),
        "Demo_MASW_shot": masw_shot(),
        "Demo_VSP": vsp_section(),
        "Demo_3C_ambient_noise": three_component_noise(),
        "Demo_SPAC_array": spac_array(),
        "Demo_receiver_function_ZR": receiver_function_data(),
        "Demo_first_motion_polarities": focal_polarities(),
        "Demo_noise_station_pair": noise_pair(),
        "Demo_VES": ves_sounding(),
        "Demo_ERT_dipole_dipole": ert_survey(),
        "Demo_crosshole_ERT": crosshole_survey(),
        "Demo_SIP_spectrum": sip_spectrum(),
        "Demo_SP": sp_profile(),
        "Demo_VLF_tilt": vlf_profile(),
        "Demo_MT_time_series": mt_time_series(),
        "Demo_MT_response": mt_response(),
        "Demo_CSAMT": csamt_data(),
        "Demo_TEM_decay": tem_decay(),
        "Demo_AEM_line": aem_line(),
        "Demo_FDEM_survey": fdem_survey(),
        "Demo_GPR": gpr_section(),
        "Demo_well_logs": well_logs(),
        "Demo_dipmeter_picks": dipmeter_picks(),
        "Demo_NMR_echo_train": nmr_echo_train(),
        "Demo_deposits": deposits(),
        "Demo_polygons": polygons_table(),
        "Demo_GCPs": gcps,
        "Demo_temperature_observations": temperature_observations(),
        "Demo_RGB_KThU": None,
    }
    from scipy.ndimage import gaussian_filter

    from .gis import extract_lineaments, ternary_rgb, terrain_derivatives
    from .passive import receiver_function
    from .profile import extract_cross_section
    from .seismic_wave import acoustic_modeling

    tmi = lib["Demo_TMI"]
    lib["Demo_RGB_KThU"] = ternary_rgb(k, th, u)
    lib["Demo_TMI_profile"] = extract_cross_section(tmi, (0.0, 11500.0), (31500.0, 11500.0), 250.0, name="Demo_TMI_profile")
    residual = grav.with_values(grav.values - 0.0002 * grav.mesh[0], "Demo_Bouguer_residual")
    lib["Demo_Bouguer_profile"] = extract_cross_section(residual, (0.0, 21000.0), (31500.0, 21000.0), 250.0, name="Demo_Bouguer_profile")
    lib["Demo_receiver_function"] = receiver_function(lib["Demo_receiver_function_ZR"], pre_signal=10.0)
    vm = lib["Demo_velocity_model"]
    lib["Demo_FWI_shot_1"] = acoustic_modeling(vm, 300.0, 10.0, 0.8)
    lib["Demo_FWI_shot_2"] = acoustic_modeling(vm, 900.0, 10.0, 0.8)
    pixels, lineaments = extract_lineaments(tmi, 85.0, 6)
    lib["Demo_lineament_pixels"] = pixels
    lib["Demo_lineaments"] = lineaments
    lib["Demo_DEM_smoothed"] = dem.with_values(gaussian_filter(dem.values, 4) - 20.0, "Demo_DEM_smoothed")
    lib["Demo_DEM_slope"] = terrain_derivatives(dem)["slope"]
    from .gis import trend_surface

    lib["Demo_Bouguer_trend"] = trend_surface(grav, 2)[0].with_values(trend_surface(grav, 2)[0].values, "Demo_Bouguer_trend")
    xx, yy = k.mesh
    sel = RNG.choice(xx.size, 300, replace=False)
    lib["Demo_well_logs_radiometric"] = pd.DataFrame({"x": xx.ravel()[sel], "y": yy.ravel()[sel], "K": k.values.ravel()[sel],
                                                      "eU": u.values.ravel()[sel], "eTh": th.values.ravel()[sel]})
    return lib


__all__ = ["demo_library", "RasterImage"]
