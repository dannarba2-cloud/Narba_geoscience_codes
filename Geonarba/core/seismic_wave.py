"""2D acoustic finite-difference modeling and time-domain full-waveform inversion (adjoint state)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .grid_tools import GridData
from .seismic import Section, ricker


def _laplacian(u: np.ndarray, dx: float) -> np.ndarray:
    """4th-order Laplacian (zero outside)."""

    lap = np.zeros_like(u)
    c0, c1, c2 = -5 / 2, 4 / 3, -1 / 12
    lap[2:-2, 2:-2] = (
        2 * c0 * u[2:-2, 2:-2]
        + c1 * (u[3:-1, 2:-2] + u[1:-3, 2:-2] + u[2:-2, 3:-1] + u[2:-2, 1:-3])
        + c2 * (u[4:, 2:-2] + u[:-4, 2:-2] + u[2:-2, 4:] + u[2:-2, :-4])
    ) / dx**2
    return lap


def _sponge(shape: tuple[int, int], width: int = 20, strength: float = 0.015) -> np.ndarray:
    nz, nx = shape
    w = np.ones(shape)
    ramp = np.exp(-((strength * (width - np.arange(width))) ** 2))
    for i in range(width):
        w[:, i] *= ramp[i]
        w[:, nx - 1 - i] *= ramp[i]
        w[nz - 1 - i, :] *= ramp[i]
    return w


def _propagate(vel: np.ndarray, dx: float, dt: float, nt: int, sources: list[tuple[int, int, np.ndarray]],
               rec_z: int, store: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    nz, nx = vel.shape
    damp = _sponge((nz, nx))
    c2 = (vel * dt) ** 2
    u_prev = np.zeros((nz, nx))
    u = np.zeros((nz, nx))
    traces = np.zeros((nt, nx))
    field = np.zeros((nt, nz, nx), np.float32) if store else None
    for it in range(nt):
        u_next = 2 * u - u_prev + c2 * _laplacian(u, dx)
        for iz, ix, wav in sources:
            if np.ndim(wav) == 0:
                continue
            if wav.ndim == 1:
                u_next[iz, ix] += c2[iz, ix] * wav[it] / dx**2
            else:  # adjoint: wavelet per receiver column along a row
                u_next[iz, :] += c2[iz, :] * wav[it] / dx**2
        u_next *= damp
        u *= damp
        u_prev, u = u, u_next
        traces[it] = u[rec_z]
        if store:
            field[it] = u
    return traces, field


def acoustic_modeling(
    velocity: GridData, source_x: float, frequency: float = 10.0, duration: float = 1.0, receiver_depth_index: int = 2,
) -> Section:
    """Single-shot 2D acoustic synthetic (receivers along the surface row). velocity: x = distance, y = depth."""

    dx = velocity.dx
    vmax = float(np.max(velocity.values))
    dt = 0.4 * dx / vmax
    nt = int(duration / dt)
    wav = np.zeros(nt)
    w = ricker(frequency, dt)
    wav[: len(w)] = w
    ix = int(np.argmin(np.abs(velocity.x_vector - source_x)))
    traces, _ = _propagate(velocity.values, dx, dt, nt, [(receiver_depth_index, ix, wav)], receiver_depth_index)
    xr = velocity.x_vector
    return Section(traces, dt, x=xr - xr[ix], name=f"shot_{source_x:g}m", headers={
        "offset": xr - xr[ix], "source_x": np.full(len(xr), xr[ix]), "receiver_x": xr},
        metadata={"method": "acoustic_fd_2d", "frequency_Hz": frequency})


def full_waveform_inversion(
    true_or_observed: list[Section],
    initial_velocity: GridData,
    frequency: float = 8.0,
    iterations: int = 10,
    v_min: float = 1400.0,
    v_max: float = 5000.0,
    mute_rows: int = 4,
) -> tuple[GridData, dict]:
    """Acoustic FWI: minimize 0.5 ||d_calc - d_obs||^2 over m = 1/v^2 with L-BFGS-B; gradient by adjoint state.

    grad_m J = - sum_t lambda(x, t) d2u/dt2(x, t), lambda = back-propagated residuals.
    Observed shots must be recorded on the same grid/time axis as `acoustic_modeling` produces.
    """

    dx = initial_velocity.dx
    shape = initial_velocity.values.shape
    shots = true_or_observed
    dt, nt = shots[0].dt, shots[0].nt
    rec_z = 2
    wav = np.zeros(nt)
    w = ricker(frequency, dt)
    wav[: len(w)] = w
    src_ix = [int(np.argmin(np.abs(initial_velocity.x_vector - s.header("source_x")[0]))) for s in shots]
    history: list[float] = []

    def objective(mvec):
        vel = 1.0 / np.sqrt(mvec.reshape(shape))
        grad = np.zeros(shape)
        misfit = 0.0
        for shot, ix in zip(shots, src_ix):
            calc, field = _propagate(vel, dx, dt, nt, [(rec_z, ix, wav)], rec_z, store=True)
            res = calc - shot.data
            misfit += 0.5 * float(np.sum(res**2))
            adj, adj_field = _propagate(vel, dx, dt, nt, [(rec_z, 0, res[::-1].copy())], rec_z, store=True)
            lam = adj_field[::-1]
            utt = np.gradient(np.gradient(field, dt, axis=0), dt, axis=0)
            grad += -np.sum(lam * utt, axis=0) * dx**2  # discrete adjoint: lambda = dx^2 * injected field, no dt
        grad[:mute_rows] = 0
        history.append(misfit)
        return misfit, grad.ravel()

    m0 = (1.0 / initial_velocity.values**2).ravel()
    bounds = [(1 / v_max**2, 1 / v_min**2)] * m0.size
    scale = 1.0 / np.abs(m0).max()

    norm: list[float] = []

    def scaled(x):
        f, g = objective(x / scale)
        if not norm:
            norm.append(f if f > 0 else 1.0)
        return f / norm[0], g / scale / norm[0]

    res = minimize(scaled, m0 * scale, jac=True, method="L-BFGS-B",
                   bounds=[(lo * scale, hi * scale) for lo, hi in bounds], options={"maxiter": iterations, "maxfun": iterations * 3})
    vel = 1.0 / np.sqrt((res.x / scale).reshape(shape))
    out = initial_velocity.with_values(vel, name="FWI_velocity", units="m/s", metadata={"method": "acoustic_fwi", "vertical_axis": "depth",
                                                                                        "frequency_Hz": frequency})
    return out, {"misfit_history": history, "iterations": int(res.nit), "converged": bool(res.success)}
