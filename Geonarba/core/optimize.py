"""Global optimizers, Bayesian sampling, TSVD and a small physics-informed network, with geophysical applications."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .electrical import SP_SHAPES, sp_forward, ves_forward
from .em import mt1d_forward
from .grid_tools import GridData
from .profile import ProfileData

Objective = Callable[[np.ndarray], float]


# ---------------------------------------------------------------- generic optimizers
def particle_swarm(f: Objective, lower, upper, n_particles: int = 40, iterations: int = 200, seed: int = 0,
                   inertia: float = 0.72, c1: float = 1.49, c2: float = 1.49) -> tuple[np.ndarray, float, list[float]]:
    """Kennedy & Eberhart (1995) PSO with constriction-style coefficients (Clerc & Kennedy, 2002)."""

    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    x = rng.uniform(lo, hi, (n_particles, lo.size))
    v = np.zeros_like(x)
    pbest, pval = x.copy(), np.array([f(p) for p in x])
    g = pbest[np.argmin(pval)].copy()
    hist = []
    for _ in range(iterations):
        r1, r2 = rng.random(x.shape), rng.random(x.shape)
        v = inertia * v + c1 * r1 * (pbest - x) + c2 * r2 * (g - x)
        x = np.clip(x + v, lo, hi)
        val = np.array([f(p) for p in x])
        better = val < pval
        pbest[better], pval[better] = x[better], val[better]
        g = pbest[np.argmin(pval)].copy()
        hist.append(float(pval.min()))
    return g, float(pval.min()), hist


def simulated_annealing(f: Objective, lower, upper, iterations: int = 4000, t0: float = 1.0, cooling: float = 0.998,
                        step: float = 0.1, seed: int = 0) -> tuple[np.ndarray, float, list[float]]:
    """Metropolis simulated annealing (Kirkpatrick et al., 1983) with geometric cooling in a bounded box."""

    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    x = rng.uniform(lo, hi)
    fx = f(x)
    best, fbest = x.copy(), fx
    temp = t0 * max(abs(fx), 1e-12)
    hist = []
    for _ in range(iterations):
        cand = np.clip(x + rng.normal(0, step, x.size) * (hi - lo), lo, hi)
        fc = f(cand)
        if fc < fx or rng.random() < np.exp(-(fc - fx) / max(temp, 1e-300)):
            x, fx = cand, fc
            if fx < fbest:
                best, fbest = x.copy(), fx
        temp *= cooling
        hist.append(fbest)
    return best, float(fbest), hist


def genetic_algorithm(f: Objective, lower, upper, population: int = 60, generations: int = 120, crossover: float = 0.9,
                      mutation: float = 0.1, elite: int = 2, seed: int = 0) -> tuple[np.ndarray, float, list[float]]:
    """Real-coded GA: tournament selection, blend (BLX-0.5) crossover, Gaussian mutation, elitism."""

    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    pop = rng.uniform(lo, hi, (population, lo.size))
    fit = np.array([f(p) for p in pop])
    hist = []
    for _ in range(generations):
        order = np.argsort(fit)
        children = [pop[i].copy() for i in order[:elite]]
        while len(children) < population:
            a, b = (pop[min(rng.integers(0, population, 3), key=lambda i: fit[i])] for _ in range(2))
            if rng.random() < crossover:
                d = np.abs(a - b)
                child = rng.uniform(np.minimum(a, b) - 0.5 * d, np.maximum(a, b) + 0.5 * d)
            else:
                child = a.copy()
            mask = rng.random(lo.size) < mutation
            child[mask] += rng.normal(0, 0.1, mask.sum()) * (hi - lo)[mask]
            children.append(np.clip(child, lo, hi))
        pop = np.array(children)
        fit = np.array([f(p) for p in pop])
        hist.append(float(fit.min()))
    i = int(np.argmin(fit))
    return pop[i], float(fit[i]), hist


def metropolis_hastings(log_post: Callable[[np.ndarray], float], x0, step, n_samples: int = 20000, burn_in: int = 5000,
                        seed: int = 0) -> tuple[np.ndarray, float]:
    """Random-walk Metropolis-Hastings sampler; returns post-burn-in samples and acceptance rate."""

    rng = np.random.default_rng(seed)
    x = np.asarray(x0, float)
    lp = log_post(x)
    samples, accepted = [], 0
    for i in range(n_samples):
        cand = x + rng.normal(0, 1, x.size) * step
        lc = log_post(cand)
        if np.log(rng.random()) < lc - lp:
            x, lp = cand, lc
            accepted += 1
        if i >= burn_in:
            samples.append(x.copy())
    return np.array(samples), accepted / n_samples


def tsvd_solve(g: np.ndarray, d: np.ndarray, k: int | None = None, rel_tol: float = 1e-3) -> tuple[np.ndarray, dict]:
    """Truncated SVD: m = sum_{i<=k} (u_i . d / s_i) v_i (Hansen, 1987)."""

    u, s, vt = np.linalg.svd(g, full_matrices=False)
    k = k or int(np.sum(s > rel_tol * s[0]))
    m = vt[:k].T @ ((u[:, :k].T @ d) / s[:k])
    return m, {"k": k, "singular_values": s.tolist(), "condition_number": float(s[0] / s[-1])}


# ---------------------------------------------------------------- applications
def pso_sp_inversion(profile: ProfileData, shape: str = "sphere", depth_max: float = 200.0, particles: int = 40,
                     iterations: int = 200) -> tuple[ProfileData, dict]:
    """Invert a self-potential profile for a polarized sphere/cylinder (K, x0, h, alpha) with PSO."""

    q = SP_SHAPES[shape]
    x, v = profile.distance, profile.values

    def solve(p):  # K enters linearly: variable projection, PSO searches (x0, h, alpha) only
        basis = sp_forward(x, 1.0, p[0], p[1], p[2], q)
        k = float(basis @ v / max(basis @ basis, 1e-300))
        return k, k * basis

    f = lambda p: float(np.sqrt(np.mean((solve(p)[1] - v) ** 2)))  # noqa: E731
    best, err, hist = particle_swarm(f, [x.min(), 1.0, -90.0], [x.max(), depth_max, 90.0], particles, iterations)
    k, calc = solve(best)
    return (profile.with_values(x, calc, f"{profile.name}_PSO_{shape}", {"method": "pso_sp_inversion"}),
            {"K": k, "x0_m": best[0], "depth_m": best[1], "polarization_angle_deg": best[2], "shape_factor_q": q,
             "rms_mV": err, "convergence": hist[::10]})


def gravity_body_forward(x, amp, x0, depth, q):
    """Simple-shape residual gravity: g = A z / ((x - x0)^2 + z^2)^q  (q = 1.5 sphere, 1 horizontal cylinder, 0.5 vertical cylinder)."""

    return amp * depth / ((np.asarray(x) - x0) ** 2 + depth**2) ** q


def sa_residual_gravity_inversion(profile: ProfileData, depth_max: float = 5000.0, iterations: int = 5000) -> tuple[ProfileData, dict]:
    """Simulated annealing for amplitude, x0, depth and shape factor q of a residual gravity anomaly."""

    x, v = profile.distance, profile.values

    def solve(p):  # amplitude is linear: projected out
        basis = gravity_body_forward(x, 1.0, *p)
        amp = float(basis @ v / max(basis @ basis, 1e-300))
        return amp, amp * basis

    f = lambda p: float(np.sqrt(np.mean((solve(p)[1] - v) ** 2)))  # noqa: E731
    lo, hi = np.array([x.min(), 10.0, 0.5]), np.array([x.max(), depth_max, 1.5])
    best, err, hist = simulated_annealing(f, lo, hi, iterations=iterations)
    best = np.clip(minimize(f, best, method="Nelder-Mead").x, lo, hi)  # polish
    amp, calc = solve(best)
    shape = min(SP_SHAPES.items(), key=lambda kv: abs(kv[1] - best[2]))[0]
    return (profile.with_values(x, calc, f"{profile.name}_SA_model", {"method": "simulated_annealing"}),
            {"amplitude": amp, "x0_m": best[0], "depth_m": best[1], "shape_factor_q": best[2], "closest_shape": shape,
             "rms": f(best), "convergence": hist[::100]})


def ga_mt_inversion(table: pd.DataFrame, freq_col: str = "frequency", rho_col: str = "rho_det", phase_col: str = "phase_det",
                    n_layers: int = 3, generations: int = 120) -> tuple[ProfileData, pd.DataFrame, dict]:
    """Genetic-algorithm layered MT inversion in log10(rho), log10(h)."""

    f = table[freq_col].to_numpy(float)
    ro, ph = table[rho_col].to_numpy(float), table[phase_col].to_numpy(float)

    def misfit(p):
        r, pp = mt1d_forward(f, 10 ** p[:n_layers], 10 ** p[n_layers:])
        return float(np.sqrt(np.mean((np.log10(r) - np.log10(ro)) ** 2) + np.mean(((pp - ph) / 45) ** 2)))

    lo = [0.0] * n_layers + [1.0] * (n_layers - 1)
    hi = [4.0] * n_layers + [4.5] * (n_layers - 1)
    best, err, hist = genetic_algorithm(misfit, lo, hi, generations=generations)
    rho, thick = 10 ** best[:n_layers], 10 ** best[n_layers:]
    r, _ = mt1d_forward(f, rho, thick)
    layers = pd.DataFrame({"layer": np.arange(1, n_layers + 1), "resistivity_ohm_m": rho, "thickness_m": np.append(thick, np.inf)})
    return (ProfileData(1 / f, r, name="GA_MT_response", units="ohm.m", metadata={"axis": "period_s", "log_axes": True}),
            layers, {"misfit": err, "convergence": hist[::10]})


def mcmc_ves_inversion(table: pd.DataFrame, spacing_col: str = "ab2", rhoa_col: str = "rhoa", n_layers: int = 3,
                       noise_percent: float = 3.0, n_samples: int = 15000, burn_in: int = 5000) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Bayesian VES inversion: uniform log-priors, Gaussian log-data likelihood, Metropolis-Hastings sampling."""

    s = table[spacing_col].to_numpy(float)
    d = np.log(table[rhoa_col].to_numpy(float))
    sigma = noise_percent / 100
    lo = np.array([np.log(0.1)] * n_layers + [np.log(0.1)] * (n_layers - 1))
    hi = np.array([np.log(1e5)] * n_layers + [np.log(s.max())] * (n_layers - 1))

    def log_post(p):
        if np.any(p < lo) or np.any(p > hi):
            return -np.inf
        pred = np.log(ves_forward(s, np.exp(p[:n_layers]), np.exp(p[n_layers:])))
        return -0.5 * float(np.sum((pred - d) ** 2)) / sigma**2

    from .electrical import ves_inversion

    _, _, layers, _ = ves_inversion(table, spacing_col, rhoa_col, n_layers)
    x0 = np.log(np.concatenate([layers["resistivity_ohm_m"].to_numpy(), layers["thickness_m"].to_numpy()[:-1]]))
    samples, acc = metropolis_hastings(log_post, x0, 0.03, n_samples, burn_in)
    names = [f"rho{i + 1}" for i in range(n_layers)] + [f"h{i + 1}" for i in range(n_layers - 1)]
    vals = np.exp(samples)
    summary = pd.DataFrame({"parameter": names, "p05": np.percentile(vals, 5, 0), "p50": np.percentile(vals, 50, 0),
                            "p95": np.percentile(vals, 95, 0), "mean": vals.mean(0)})
    return summary, pd.DataFrame(vals[::10], columns=names), {"acceptance_rate": acc, "samples": int(len(vals))}


def tsvd_gravity_inversion(profile: ProfileData, depth_top: float = 500.0, depth_bottom: float = 5000.0, n_z: int = 10,
                           truncation: int = 0) -> tuple[GridData, ProfileData, dict]:
    """Linear 2D density inversion of a gravity profile (infinite horizontal line masses per cell) by truncated SVD."""

    x = profile.distance
    xc = np.linspace(x.min(), x.max(), max(len(x) // 2, 4))
    zc = np.linspace(depth_top, depth_bottom, n_z)
    xx, zz = np.meshgrid(xc, zc)
    area = (xc[1] - xc[0]) * (zc[1] - zc[0]) if n_z > 1 else (xc[1] - xc[0]) * depth_top
    g = 2 * 6.674e-11 * area * zz.ravel()[None, :] / ((x[:, None] - xx.ravel()[None, :]) ** 2 + zz.ravel()[None, :] ** 2) * 1e5
    d = profile.values - np.mean(profile.values)
    m, info = tsvd_solve(g, d, truncation or None)
    model = GridData(m.reshape(zz.shape), xc, zc, xc[1] - xc[0], zc[1] - zc[0] if n_z > 1 else 1.0, name="TSVD_density",
                     units="kg/m^3", metadata={"vertical_axis": "depth", "method": "tsvd"})
    info["rms_mGal"] = float(np.sqrt(np.mean((g @ m - d) ** 2)))
    info.pop("singular_values")
    return model, profile.with_values(x, g @ m + np.mean(profile.values), f"{profile.name}_TSVD_fit"), info


def tikhonov_gravity_profile(
    profile: ProfileData, depth_top: float = 500.0, depth_bottom: float = 5000.0, n_z: int = 10, operator: str = "first-derivative",
    alpha_s: float = 1e-4, alpha_x: float = 1.0, alpha_z: float = 1.0, reference_density: float = 0.0, error_percent: float = 0.0,
    error_floor: float = 0.05, lambda_mode: str = "discrepancy", lam: float = 1.0, solver: str = "direct",
) -> tuple[GridData, ProfileData, ProfileData, ProfileData, dict]:
    """2D Tikhonov/Occam density inversion of a gravity profile (horizontal line-mass cells).

    Exposes every trade-off control: operator (1st/2nd derivative, smallness), anisotropy (alpha_x vs alpha_z),
    reference model, data errors (percent + floor -> W_d = 1/sigma), lambda by discrepancy / L-curve / GCV / fixed,
    and direct / CG / subspace solvers. Returns the section, data fit, normalized residuals, L-curve and QC.
    """

    from .inversion import data_errors, regularization_operator, tikhonov_invert
    from .modeling import lcurve_profile

    x = profile.distance
    nx = max(len(x) // 2, 4)
    xc = np.linspace(x.min(), x.max(), nx)
    zc = np.linspace(depth_top, depth_bottom, n_z)
    xx, zz = np.meshgrid(xc, zc)
    area = (xc[1] - xc[0]) * ((zc[1] - zc[0]) if n_z > 1 else depth_top)
    g = 2 * 6.674e-11 * area * zz.ravel()[None, :] / ((x[:, None] - xx.ravel()[None, :]) ** 2 + zz.ravel()[None, :] ** 2) * 1e5
    base = float(np.mean(profile.values))
    d = profile.values - base
    sigma = data_errors(d, error_percent, error_floor)
    wm = regularization_operator((n_z, nx), operator, alpha_s, alpha_x, 1.0, alpha_z)
    res = tikhonov_invert(g, d, sigma, wm, reference_density, lambda_mode, lam, solver)
    model = GridData(res.model.reshape(zz.shape), xc, zc, xc[1] - xc[0], zc[1] - zc[0] if n_z > 1 else 1.0,
                     name="Tikhonov_density", units="kg/m^3", metadata={"vertical_axis": "depth", "method": "tikhonov"})
    fit = profile.with_values(x, res.predicted + base, f"{profile.name}_Tikhonov_fit")
    nres = profile.with_values(x, (d - res.predicted) / sigma, f"{profile.name}_normalized_residuals")
    return model, fit, nres, lcurve_profile(res.lcurve, "Tikhonov_L_curve"), res.report


def pinn_geotherm(
    depth_max: float = 30000.0, surface_temperature: float = 15.0, surface_heat_flow: float = 0.065, conductivity: float = 2.5,
    heat_production: float = 1e-6, observations: pd.DataFrame | None = None, depth_col: str = "depth", temp_col: str = "temperature",
    hidden: int = 16, seed: int = 0,
) -> tuple[ProfileData, dict]:
    """Physics-informed neural network (Raissi et al., 2019) for the 1D steady geotherm k T'' = -A.

    One hidden tanh layer with analytic derivatives; loss = PDE residual + boundary conditions (+ data).
    With temperature observations, heat production A is learned as an extra parameter (inverse mode).
    Analytic check: T = T0 + q0 z / k - A z^2 / (2k).
    """

    rng = np.random.default_rng(seed)
    zs = depth_max
    zc = np.linspace(0, 1, 64)
    tscale = surface_temperature + surface_heat_flow * zs / conductivity
    inverse = observations is not None and len(observations) > 0
    zo = observations[depth_col].to_numpy(float) / zs if inverse else None
    to = observations[temp_col].to_numpy(float) / tscale if inverse else None

    def unpack(p):
        w, b, a = p[:hidden], p[hidden:2 * hidden], p[2 * hidden:3 * hidden]
        c = p[3 * hidden]
        log_a = p[3 * hidden + 1] if inverse else np.log(heat_production)
        return w, b, a, c, log_a

    def net(p, z):
        w, b, a, c, _ = unpack(p)
        h = np.tanh(np.outer(z, w) + b)
        u = h @ a + c
        du = (1 - h**2) @ (a * w)
        d2u = (-2 * h * (1 - h**2)) @ (a * w**2)
        return u, du, d2u

    def loss(p):
        _, _, _, _, log_a = unpack(p)
        a_val = np.exp(log_a)
        u, du, d2u = net(p, zc)
        # nondimensional: k T''(z) = -A  ->  (k tscale / zs^2) u'' = -A
        pde = conductivity * tscale / zs**2 * d2u + a_val
        l_pde = np.mean((pde / max(a_val, 1e-9)) ** 2) if a_val > 0 else np.mean(d2u**2)
        u0, du0, _ = net(p, np.array([0.0]))
        l_bc = (u0[0] - surface_temperature / tscale) ** 2 + (conductivity * tscale / zs * du0[0] - surface_heat_flow) ** 2 / surface_heat_flow**2
        l_data = np.mean((net(p, zo)[0] - to) ** 2) * 100 if inverse else 0.0
        return l_pde * 1e-2 + l_bc + l_data

    p0 = np.concatenate([rng.normal(0, 1, hidden), rng.normal(0, 1, hidden), rng.normal(0, 0.1, hidden), [0.0]])
    if inverse:
        p0 = np.append(p0, np.log(heat_production))
    res = minimize(loss, p0, method="L-BFGS-B", options={"maxiter": 3000})
    z = np.linspace(0, 1, 200)
    temp = net(res.x, z)[0] * tscale
    a_fit = float(np.exp(unpack(res.x)[4]))
    analytic = surface_temperature + surface_heat_flow * z * zs / conductivity - a_fit * (z * zs) ** 2 / (2 * conductivity)
    return (ProfileData(z * zs, temp, name="PINN_geotherm", units="degC", metadata={"axis": "depth_m", "analytic": analytic.tolist()}),
            {"heat_production_uW_m3": a_fit * 1e6, "max_abs_error_vs_analytic_C": float(np.max(np.abs(temp - analytic))),
             "loss": float(res.fun), "mode": "inverse" if inverse else "forward"})
