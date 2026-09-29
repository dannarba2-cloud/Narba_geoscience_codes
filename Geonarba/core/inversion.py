"""Shared Tikhonov / Occam inversion framework: operators, data weights, lambda selection, solvers and QC.

Objective (Tikhonov, 1963; Constable et al., 1987; Li & Oldenburg, 1996):
    Phi = Phi_d + lambda * Phi_m
    Phi_d = || W_d (G m - d) ||^2,           W_d = diag(1/sigma)
    Phi_m = alpha_s ||W_s (m - m_ref)||^2 + sum_i alpha_i ||D_i (m - m_ref)||^2   (i = x, y, z; 1st or 2nd derivative)
Lambda selection: fixed, L-curve corner (Hansen, 1992), GCV (Wahba, 1977) or discrepancy principle (chi^2 = N, Occam).
Solvers: direct normal equations, CG (LSQR on the stacked system) or subspace (Kennett & Williamson, 1988; Oldenburg et al., 1993).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse, stats
from scipy.sparse.linalg import lsqr

LAMBDA_MODES = ("discrepancy", "l-curve", "gcv", "fixed")
OPERATORS = ("first-derivative", "second-derivative", "smallness")
SOLVERS = ("direct", "cg", "subspace")


# ---------------------------------------------------------------- operators & weights
def _d1(n: int) -> sparse.csr_matrix:
    return sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n)).tocsr() if n > 1 else sparse.csr_matrix((0, n))


def _d2(n: int) -> sparse.csr_matrix:
    return sparse.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n)).tocsr() if n > 2 else sparse.csr_matrix((0, n))


def regularization_operator(
    shape: tuple[int, ...],
    operator: str = "first-derivative",
    alpha_s: float = 1e-4,
    alpha_x: float = 1.0,
    alpha_y: float = 1.0,
    alpha_z: float = 1.0,
    cell_weights: np.ndarray | None = None,
) -> sparse.csr_matrix:
    """Stacked W_m for a model of `shape` ((n,), (nz, nx) or (nz, ny, nx), last axis = x).

    Anisotropy: alpha_x/alpha_y/alpha_z weight smoothing per axis (e.g. alpha_x = alpha_y > alpha_z for layered geology).
    cell_weights (e.g. depth weighting) multiply every model cell before differencing.
    """

    if operator not in OPERATORS:
        raise ValueError(f"operator must be one of {OPERATORS}")
    n = int(np.prod(shape))
    blocks = [np.sqrt(alpha_s) * sparse.identity(n, format="csr")]
    if operator != "smallness":
        diff = _d1 if operator == "first-derivative" else _d2
        axes = {1: ("x",), 2: ("z", "x"), 3: ("z", "y", "x")}[len(shape)]
        alphas = {"x": alpha_x, "y": alpha_y, "z": alpha_z}
        for ax, name in enumerate(axes):
            ops = [sparse.identity(s, format="csr") for s in shape]
            ops[ax] = diff(shape[ax])
            full = ops[0]
            for o in ops[1:]:
                full = sparse.kron(full, o, format="csr")
            if full.shape[0]:
                blocks.append(np.sqrt(alphas[name]) * full)
    wm = sparse.vstack(blocks).tocsr()
    if cell_weights is not None:
        wm = wm @ sparse.diags(np.asarray(cell_weights, float).ravel())
    return wm


def data_errors(d: np.ndarray, percent: float = 5.0, floor: float | None = None) -> np.ndarray:
    """sigma_i = percent/100 * |d_i| + floor  (floor defaults to 1% of the data RMS so small values stay finite)."""

    d = np.asarray(d, float)
    floor = floor if floor is not None and floor > 0 else 0.01 * float(np.sqrt(np.mean(d**2)) or 1.0)
    return percent / 100.0 * np.abs(d) + floor


# ---------------------------------------------------------------- solvers
def _solve(g, dw, wm, lam, m_ref, solver: str, m0=None, iters: int = 200):
    """Minimize ||G m - d||^2 + lam ||Wm (m - m_ref)||^2 with data already weighted (G, dw = W_d G, W_d d)."""

    n = g.shape[1]
    rhs_shift = dw - g @ m_ref
    if solver == "direct":
        gd = g.toarray() if sparse.issparse(g) else g
        lhs = gd.T @ gd + lam * (wm.T @ wm).toarray()
        dm = np.linalg.solve(lhs + 1e-12 * np.trace(lhs) / n * np.eye(n), gd.T @ rhs_shift)
    elif solver == "cg":
        a = sparse.vstack([sparse.csr_matrix(g), np.sqrt(lam) * wm]).tocsr()
        dm = lsqr(a, np.concatenate([rhs_shift, np.zeros(wm.shape[0])]), atol=1e-10, btol=1e-10, iter_lim=iters,
                  x0=None if m0 is None else m0 - m_ref)[0]
    elif solver == "subspace":
        dm = np.zeros(n) if m0 is None else m0 - m_ref
        wtw = wm.T @ wm
        prev = None
        for _ in range(iters):
            r = rhs_shift - g @ dm
            gd_ = g.T @ r                    # steepest descent of Phi_d
            gm_ = -lam * (wtw @ dm)          # steepest descent of lambda*Phi_m
            basis = [v for v in (gd_, gm_, prev) if v is not None and np.linalg.norm(v) > 0]
            if not basis:
                break
            b = np.column_stack([v / np.linalg.norm(v) for v in basis])
            gb = g @ b
            h = gb.T @ gb + lam * b.T @ (wtw @ b)
            step_coef = np.linalg.lstsq(h, gb.T @ r - lam * b.T @ (wtw @ dm), rcond=None)[0]
            step = b @ step_coef
            dm = dm + step
            prev = step
            if np.linalg.norm(step) < 1e-8 * (np.linalg.norm(dm) + 1e-30):
                break
    else:
        raise ValueError(f"solver must be one of {SOLVERS}")
    return m_ref + dm


def _gcv(g, wm, lam, residual_norm2: float, n_data: int, probes: int = 12) -> float:
    """GCV(lambda) = N ||r||^2 / (N - trace(H))^2, H = G (G'G + lam Wm'Wm)^-1 G' (dense or Hutchinson estimate)."""

    gd = g.toarray() if sparse.issparse(g) else g
    n = gd.shape[1]
    if n <= 2000:
        lhs = gd.T @ gd + lam * (wm.T @ wm).toarray()
        lhs += 1e-12 * np.trace(lhs) / n * np.eye(n)
        tr = float(np.trace(gd @ np.linalg.solve(lhs, gd.T)))
    else:  # Hutchinson: trace(H) ~ mean z' H z, with H z = G argmin ||G x - z||^2 + lam ||Wm x||^2 (LSQR)
        rng = np.random.default_rng(0)
        a = sparse.vstack([sparse.csr_matrix(gd), np.sqrt(lam) * wm]).tocsr()
        zeros = np.zeros(wm.shape[0])
        est = []
        for _ in range(probes):
            z = rng.choice([-1.0, 1.0], size=n_data)
            est.append(z @ (gd @ lsqr(a, np.concatenate([z, zeros]), atol=1e-8, btol=1e-8, iter_lim=300)[0]))
        tr = float(np.mean(est))
    return n_data * residual_norm2 / max((n_data - tr) ** 2, 1e-30)


def _corner(log_phid: np.ndarray, log_phim: np.ndarray) -> int:
    """Index of maximum curvature of the parametric L-curve (log Phi_d, log Phi_m)."""

    t = np.arange(len(log_phid), dtype=float)
    x1, y1 = np.gradient(log_phid, t), np.gradient(log_phim, t)
    x2, y2 = np.gradient(x1, t), np.gradient(y1, t)
    kappa = (x1 * y2 - y1 * x2) / np.maximum((x1**2 + y1**2) ** 1.5, 1e-30)
    kappa[[0, -1]] = -np.inf
    return int(np.argmax(kappa))


@dataclass
class InversionResult:
    model: np.ndarray
    predicted: np.ndarray
    lam: float
    report: dict
    lcurve: dict = field(default_factory=dict)


def tikhonov_invert(
    g,
    d: np.ndarray,
    sigma: np.ndarray,
    wm: sparse.csr_matrix,
    m_ref: np.ndarray | float = 0.0,
    lambda_mode: str = "discrepancy",
    lam: float = 1.0,
    solver: str = "cg",
    n_lambdas: int = 24,
    target_misfit: float = 1.0,
) -> InversionResult:
    """Linear Tikhonov inversion with automatic trade-off selection; returns model, prediction, lambda, QC and L-curve."""

    if lambda_mode not in LAMBDA_MODES:
        raise ValueError(f"lambda_mode must be one of {LAMBDA_MODES}")
    d = np.asarray(d, float)
    wd = 1.0 / np.asarray(sigma, float)
    gw = (sparse.diags(wd) @ g) if sparse.issparse(g) else g * wd[:, None]
    dw = d * wd
    n = gw.shape[1]
    m_ref = np.full(n, float(m_ref)) if np.isscalar(m_ref) else np.asarray(m_ref, float)
    nd = len(d)

    def run(l, m0=None):
        m = _solve(gw, dw, wm, l, m_ref, solver, m0)
        r = dw - gw @ m
        return m, float(r @ r), float(np.sum((wm @ (m - m_ref)) ** 2))

    curve = {}
    if lambda_mode == "fixed":
        chosen = float(lam)
        m, phid, phim = run(chosen)
    else:
        gd = gw.toarray() if sparse.issparse(gw) else gw
        scale = np.trace(gd.T @ gd) / max(np.sum(wm.multiply(wm)), 1e-30)
        lams = scale * np.logspace(-6, 3, n_lambdas)
        sols, phids, phims, gcvs = [], [], [], []
        m0 = None
        for l in lams[::-1]:  # large -> small, warm start
            m0, phid_, phim_ = run(l, m0)
            sols.append(m0)
            phids.append(phid_)
            phims.append(phim_)
            if lambda_mode == "gcv":
                gcvs.append(_gcv(gw, wm, l, phid_, nd))
        # results were computed large -> small lambda; reverse them to match the ascending `lams`
        sols, gcvs = sols[::-1], gcvs[::-1]
        phids, phims = np.array(phids[::-1]), np.array(phims[::-1])
        # L-curve/GCV degenerate when misfit -> 0 (underdetermined problems): search only 0.1 <= normalized RMS <= 10
        band = np.where((phids >= 0.01 * nd) & (phids <= 100 * nd))[0]
        band = band if band.size >= 4 else np.arange(len(lams))
        if lambda_mode == "l-curve":
            k = int(band[_corner(np.log10(phids[band] + 1e-300), np.log10(phims[band] + 1e-300))])
        elif lambda_mode == "gcv":
            k = int(band[np.argmin(np.asarray(gcvs)[band])])
        else:  # discrepancy: largest lambda (smoothest) whose chi^2 <= target^2 * N
            ok = np.where(phids <= (target_misfit**2) * nd)[0]
            k = int(ok[-1]) if ok.size else int(np.argmin(phids))  # lambdas ascending
        chosen = float(lams[k])
        edge_warning = k <= 1 or k >= len(lams) - 2
        m, phid, phim = sols[k], float(phids[k]), float(phims[k])
        target = (target_misfit**2) * nd
        if lambda_mode == "discrepancy" and phid <= target and k + 1 < len(lams) and phids[k + 1] > target:
            lo, hi = np.log(lams[k]), np.log(lams[k + 1])  # bisect in log(lambda) so chi^2 -> N
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                m_mid, pd_mid, pm_mid = run(np.exp(mid), m)
                if pd_mid <= target:
                    lo, m, phid, phim, chosen = mid, m_mid, pd_mid, pm_mid, float(np.exp(mid))
                else:
                    hi = mid
        curve = {"lambda": lams.tolist(), "phi_d": phids.tolist(), "phi_m": phims.tolist(), "corner_index": k}
        if gcvs:
            curve["gcv"] = gcvs
    pred = (g @ m) if not sparse.issparse(g) else g @ m
    report = misfit_report(d, pred, sigma)
    report.update({"lambda": chosen, "lambda_mode": lambda_mode, "solver": solver, "phi_d": phid, "phi_m": phim})
    if lambda_mode != "fixed" and edge_warning:
        report["lambda_warning"] = ("Selected lambda is at the edge of the search range (no interior L-corner / GCV minimum): "
                                    "errors may be overestimated or data nearly noise-free; prefer the discrepancy principle.")
    return InversionResult(m, np.asarray(pred), chosen, report, curve)


# ---------------------------------------------------------------- QC / verification
def residual_analysis(observed, predicted, error_percent: float = 0.0, error_floor: float = 0.0):
    """Observed - calculated residual map, normalized residuals and randomness diagnostics for any two same-shape grids."""

    if observed.values.shape != predicted.values.shape:
        raise ValueError("Observed and predicted grids must have the same shape.")
    res = observed.values - predicted.values
    sigma = data_errors(observed.values.ravel(), error_percent, error_floor or None) if (error_percent or error_floor) else None
    report = misfit_report(observed.values, predicted.values, sigma, observed.values.shape, (observed.dx, observed.dy))
    out = [observed.with_values(res, f"{observed.name}_residual", metadata={"method": "residual"})]
    if sigma is not None:
        out.append(observed.with_values(res / sigma.reshape(res.shape), f"{observed.name}_normalized_residual", "residual/sigma",
                                        {"method": "normalized_residual"}))
    return out, report


# ---------------------------------------------------------------- QC / verification
def misfit_report(observed: np.ndarray, predicted: np.ndarray, sigma: np.ndarray | None = None, grid_shape: tuple | None = None,
                  spacing: tuple[float, float] | None = None) -> dict:
    """RMS, chi^2, normalized RMS (target 1) and residual randomness diagnostics."""

    obs, pred = np.asarray(observed, float).ravel(), np.asarray(predicted, float).ravel()
    res = obs - pred
    ok = np.isfinite(res)
    res = res[ok]
    out: dict = {"n_data": int(res.size), "rms": float(np.sqrt(np.mean(res**2))), "mean_residual": float(np.mean(res))}
    if sigma is not None:
        nres = res / np.asarray(sigma, float).ravel()[ok]
        chi2 = float(np.sum(nres**2))
        nrms = float(np.sqrt(chi2 / res.size))
        out.update({"chi2": chi2, "normalized_rms": nrms,
                    "fit_assessment": "fits to noise level (target 1)" if 0.8 <= nrms <= 1.2 else
                    ("overfitting: fitting noise, increase lambda" if nrms < 0.8 else "underfitting: missing structure or lambda too high")})
    else:
        nres = res / (np.std(res) or 1.0)
    if res.size >= 3:
        dw = float(np.sum(np.diff(res) ** 2) / max(np.sum(res**2), 1e-30))
        out["durbin_watson"] = dw  # ~2 = uncorrelated along the data order
        out["lag1_autocorrelation"] = float(np.corrcoef(res[:-1], res[1:])[0, 1]) if np.std(res) > 0 else 0.0
    if res.size >= 20:
        out["normality_p_value"] = float(stats.normaltest(nres).pvalue)
    if grid_shape is not None and ok.all():
        from .gis import morans_i
        from .grid_tools import GridData

        dx, dy = spacing or (1.0, 1.0)
        rg = GridData(res.reshape(grid_shape), np.arange(grid_shape[1]) * dx, np.arange(grid_shape[0]) * dy, dx, dy)
        mi = morans_i(rg)
        out["residual_morans_I"] = mi["morans_I"]
        out["residual_morans_z"] = mi["z_score"]
    z = out.get("residual_morans_z")
    coherent = (abs(z) > 3) if z is not None else (abs(out.get("lag1_autocorrelation", 0.0)) > 0.5)
    out["residuals"] = "coherent pattern: model misses structure or smoothing too strong" if coherent else "random (white-noise-like)"
    return out
