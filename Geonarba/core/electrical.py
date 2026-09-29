"""DC resistivity, IP, SIP and self-potential: 1D VES, 2.5D finite-volume ERT (surface and cross-hole)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.fft import fht, fhtoffset
from scipy.optimize import least_squares
from scipy.sparse.linalg import splu

from .grid_tools import GridData
from .profile import ProfileData


# ---------------------------------------------------------------- 1D VES
def resistivity_transform(lam: np.ndarray, rho: np.ndarray, thick: np.ndarray) -> np.ndarray:
    """Pekeris recursion: T_N = rho_N; T_i = (T_{i+1} + rho_i tanh(lam h_i)) / (1 + T_{i+1} tanh(lam h_i)/rho_i)."""

    t = np.full_like(lam, rho[-1], dtype=float)
    for r, h in zip(rho[-2::-1], thick[::-1]):
        th = np.tanh(lam * h)
        t = (t + r * th) / (1 + t * th / r)
    return t


def ves_forward(spacings: np.ndarray, rho: np.ndarray, thick: np.ndarray, array: str = "schlumberger") -> np.ndarray:
    """Apparent resistivity of a layered earth via FFTLog Hankel transforms (Hamilton, 2000; scipy.fft.fht).

    Schlumberger (AB/2 = s): rho_a = rho1 + s^2 int (T - rho1) J1(lam s) lam dlam.
    Wenner (a):            rho_a = rho1 + 2a int (T - rho1) [J0(lam a) - J0(2 lam a)] dlam.
    """

    rho = np.asarray(rho, float)
    thick = np.asarray(thick, float)
    s = np.asarray(spacings, float)
    n = 1024
    dln = np.log(1e8) / n * 2
    lc = 1.0 / np.sqrt(s.min() * s.max())

    def hankel(mu: float, weight_pow: float) -> tuple[np.ndarray, np.ndarray]:
        offset = fhtoffset(dln, mu=mu)
        lam = lc * np.exp((np.arange(n) - n / 2) * dln)
        a = (resistivity_transform(lam, rho, thick) - rho[0]) * lam**weight_pow
        k = np.exp(offset) / lam[::-1]
        return k, fht(a, dln, mu=mu, offset=offset)

    if array == "schlumberger":
        k, a = hankel(1.0, 1.0)  # A(k) = k int (T-rho1) lam J1(k lam) dlam
        val = np.interp(np.log(s), np.log(k), a)
        return rho[0] + s * val
    if array == "wenner":
        k, a = hankel(0.0, 0.0)  # A(k) = k int (T-rho1) J0(k lam) dlam
        f = lambda x: np.interp(np.log(x), np.log(k), a) / x  # noqa: E731
        return rho[0] + 2 * s * (f(s) - f(2 * s))
    raise ValueError("array must be schlumberger or wenner.")


def ves_inversion(
    table: pd.DataFrame, spacing_col: str = "ab2", rhoa_col: str = "rhoa", n_layers: int = 3, array: str = "schlumberger",
) -> tuple[ProfileData, ProfileData, pd.DataFrame, dict]:
    """Layered VES inversion: damped Gauss-Newton (Levenberg-Marquardt) in log parameters."""

    s = table[spacing_col].to_numpy(float)
    d = table[rhoa_col].to_numpy(float)
    order = np.argsort(s)
    s, d = s[order], d[order]
    rho0 = np.full(n_layers, np.exp(np.mean(np.log(d))))
    rho0[0], rho0[-1] = d[0], d[-1]
    thick0 = np.geomspace(s.min(), s.max() / 3, n_layers - 1)

    def resid(p):
        return np.log(ves_forward(s, np.exp(p[:n_layers]), np.exp(p[n_layers:]), array)) - np.log(d)

    fit = least_squares(resid, np.log(np.concatenate([rho0, thick0])), method="lm")
    rho, thick = np.exp(fit.x[:n_layers]), np.exp(fit.x[n_layers:])
    rms = float(np.sqrt(np.mean(fit.fun**2)) * 100)
    ss = np.geomspace(s.min(), s.max(), 60)
    depth_tops = np.concatenate([[0], np.cumsum(thick)])
    layers = pd.DataFrame({"layer": np.arange(1, n_layers + 1), "resistivity_ohm_m": rho,
                           "thickness_m": np.append(thick, np.inf), "top_depth_m": depth_tops})
    return (ProfileData(s, d, name="VES_observed", units="ohm.m", metadata={"axis": "AB/2_m", "log_axes": True}),
            ProfileData(ss, ves_forward(ss, rho, thick, array), name="VES_model_response", units="ohm.m",
                        metadata={"axis": "AB/2_m", "log_axes": True}),
            layers, {"rms_percent": rms, "array": array})


# ---------------------------------------------------------------- 2.5D finite-volume engine
class ResistivityMesh:
    """Node-based 2.5D finite-volume mesh (x right, z down) with padded boundaries."""

    def __init__(self, electrodes: np.ndarray, h: float | None = None, depth: float | None = None, n_pad: int = 8, growth: float = 1.5):
        ex, ez = electrodes[:, 0], electrodes[:, 1]
        span = np.ptp(ex) if np.ptp(ex) > 0 else np.ptp(ez)
        spacing = np.min(np.diff(np.unique(np.round(np.concatenate([ex, ez]), 6)))) if len(np.unique(ex)) > 1 else span / 10
        self.h = h or spacing / 2
        depth = depth or max(span / 4, ez.max() * 1.2 + 2 * self.h)
        xi = np.arange(ex.min(), ex.max() + self.h / 2, self.h)
        zi = np.arange(0, depth + self.h / 2, self.h)
        pad = self.h * np.cumprod(np.full(n_pad, growth))
        self.xn = np.concatenate([xi[0] - pad[::-1].cumsum()[::-1], xi, xi[-1] + pad.cumsum()])
        self.zn = np.concatenate([zi, zi[-1] + pad.cumsum()])
        self.n_pad = n_pad
        self.nx, self.nz = len(self.xn), len(self.zn)
        self.hx, self.hz = np.diff(self.xn), np.diff(self.zn)
        self.ncell = (self.nx - 1) * (self.nz - 1)
        self.electrode_nodes = np.array([self.node(x, z) for x, z in electrodes])
        self._assemble_topology()

    def node(self, x: float, z: float) -> int:
        i = int(np.argmin(np.abs(self.zn - z)))
        j = int(np.argmin(np.abs(self.xn - x)))
        return i * self.nx + j

    def _assemble_topology(self) -> None:
        nxc, nzc = self.nx - 1, self.nz - 1
        ci, cj = np.meshgrid(np.arange(nzc), np.arange(nxc), indexing="ij")
        ci, cj = ci.ravel(), cj.ravel()
        n00 = ci * self.nx + cj
        n01, n10, n11 = n00 + 1, n00 + self.nx, n00 + self.nx + 1
        hx, hz = self.hx[cj], self.hz[ci]
        gh, gv = hz / (2 * hx), hx / (2 * hz)
        # edges: (a, b, conductance-per-sigma, cell)
        self.edges = [(n00, n01, gh), (n10, n11, gh), (n00, n10, gv), (n01, n11, gv)]
        self.mass = [(n, hx * hz / 4) for n in (n00, n01, n10, n11)]
        self.cells = np.arange(len(ci))
        self.n_nodes = self.nx * self.nz
        bottom = np.arange((self.nz - 1) * self.nx, self.nz * self.nx)
        sides = np.concatenate([np.arange(self.nz) * self.nx, np.arange(self.nz) * self.nx + self.nx - 1])
        self.dirichlet = np.unique(np.concatenate([bottom, sides]))

    def stiffness(self, sigma: np.ndarray, k: float) -> sparse.csc_matrix:
        rows, cols, vals = [], [], []
        for a, b, g in self.edges:
            w = sigma * g
            rows += [a, b, a, b]
            cols += [a, b, b, a]
            vals += [w, w, -w, -w]
        for n, area in self.mass:
            rows.append(n)
            cols.append(n)
            vals.append(sigma * k**2 * area)
        mat = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.n_nodes,) * 2)
        keep = np.ones(self.n_nodes, bool)
        keep[self.dirichlet] = False
        mat = mat.tolil()
        for n in self.dirichlet:
            mat.rows[n], mat.data[n] = [n], [1.0]
        mat = mat.tocsr()
        mask = sparse.diags(keep.astype(float))
        return (mask @ mat @ mask + sparse.diags((~keep).astype(float))).tocsc()

    def cell_grid(self, values: np.ndarray, name: str, units: str) -> GridData:
        """Crop padding and return the inner uniform cells as a GridData (y = depth)."""

        vol = values.reshape(self.nz - 1, self.nx - 1)
        p = self.n_pad
        inner = vol[:-p, p:-p]
        xc = 0.5 * (self.xn[p:-p - 1] + self.xn[p + 1:-p])
        zc = 0.5 * (self.zn[:-p - 1] + self.zn[1:-p])
        return GridData(inner, xc, zc, self.h, self.h, name=name, units=units, metadata={"vertical_axis": "depth"})


WAVENUMBERS = np.geomspace(1e-4, 3.0, 16)


def _potentials(mesh: ResistivityMesh, sigma: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """u[k, electrode, node]: transformed potentials for unit current (I/2 source) at each electrode."""

    ne = len(mesh.electrode_nodes)
    rhs = np.zeros((mesh.n_nodes, ne))
    rhs[mesh.electrode_nodes, np.arange(ne)] = 0.5
    out = np.zeros((len(ks), ne, mesh.n_nodes))
    for i, k in enumerate(ks):
        out[i] = splu(mesh.stiffness(sigma, k)).solve(rhs).T
    return out


def _quadrature(ks: np.ndarray) -> np.ndarray:
    lnk = np.log(ks)
    w = np.gradient(lnk) * ks
    return 2.0 / np.pi * w


def _data_index(mesh_nodes_idx: dict, rows: np.ndarray) -> np.ndarray:
    return np.array([[mesh_nodes_idx[tuple(r[i])] for i in range(4)] for r in rows])


def _electrodes_from_table(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cols = ["a_x", "a_z", "b_x", "b_z", "m_x", "m_z", "n_x", "n_z"]
    missing = [c for c in cols if c not in table]
    if missing:
        raise ValueError(f"Resistivity table needs columns {cols}; missing {missing}.")
    pts = table[cols].to_numpy(float).reshape(-1, 4, 2)
    uniq, inv = np.unique(np.round(pts.reshape(-1, 2), 6), axis=0, return_inverse=True)
    return uniq, inv.reshape(-1, 4)


def _forward(mesh: ResistivityMesh, sigma: np.ndarray, quad: np.ndarray, abmn: np.ndarray, want_jacobian: bool = False):
    u = _potentials(mesh, sigma, WAVENUMBERS)
    en = mesh.electrode_nodes
    ua = np.einsum("k,ken->en", quad, u)  # real-space potentials per source electrode
    a, b, m, n = abmn.T
    v = ua[a, en[m]] - ua[a, en[n]] - ua[b, en[m]] + ua[b, en[n]]
    if not want_jacobian:
        return v, None
    jac = np.zeros((len(abmn), mesh.ncell))
    for ki, k in enumerate(WAVENUMBERS):
        src = u[ki, a] - u[ki, b]       # (nd, nodes)
        rec = u[ki, m] - u[ki, n]
        contrib = np.zeros((len(abmn), mesh.ncell))
        for e1, e2, g in mesh.edges:
            contrib += g[None, :] * (src[:, e1] - src[:, e2]) * (rec[:, e1] - rec[:, e2])
        for node, area in mesh.mass:
            contrib += k**2 * area[None, :] * src[:, node] * rec[:, node]
        jac -= quad[ki] * contrib * 2.0  # reciprocity: potentials were for I/2 sources
    return v, jac


def ert_forward(table: pd.DataFrame, model: GridData | None = None, background_resistivity: float = 100.0) -> pd.DataFrame:
    """Apparent resistivity for ABMN rows; geometric factors computed numerically from a homogeneous run."""

    elec, abmn = _electrodes_from_table(table)
    mesh = ResistivityMesh(elec)
    quad = _quadrature(WAVENUMBERS)
    rho = np.full(mesh.ncell, background_resistivity)
    if model is not None:
        xc = 0.5 * (mesh.xn[:-1] + mesh.xn[1:])
        zc = 0.5 * (mesh.zn[:-1] + mesh.zn[1:])
        zz, xx = np.meshgrid(zc, xc, indexing="ij")
        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator((model.y_vector, model.x_vector), model.values, bounds_error=False, fill_value=None)
        rho = np.clip(interp(np.column_stack([zz.ravel(), xx.ravel()])), 0.1, 1e6)
    v_hom, _ = _forward(mesh, np.full(mesh.ncell, 1.0 / background_resistivity), quad, abmn)
    v, _ = _forward(mesh, 1.0 / rho, quad, abmn)
    out = table.copy()
    out["rhoa"] = background_resistivity * v / v_hom
    return out


def ert_inversion(
    table: pd.DataFrame,
    rhoa_col: str = "rhoa",
    iterations: int = 6,
    smoothness: float = 0.3,
    chargeability_col: str = "",
    error_percent: float = 3.0,
    lambda_mode: str = "discrepancy",
    operator: str = "first-derivative",
    alpha_s: float = 1e-3,
    alpha_x: float = 1.0,
    alpha_z: float = 1.0,
    reference_resistivity: float = 0.0,
) -> tuple[list[GridData], pd.DataFrame, dict]:
    """Occam-style Gauss-Newton ERT inversion (deGroot-Hedlin & Constable, 1990) in log resistivity.

    Each iteration solves the linearized Tikhonov problem with data weights 1/sigma (sigma = error_percent in ln rho_a),
    an anisotropic 1st/2nd-derivative operator (alpha_x horizontal, alpha_z vertical), a reference half-space and
    lambda chosen by the discrepancy principle (Occam: smoothest model reaching chi^2 = N), L-curve, GCV or fixed cooling.
    Works for surface and cross-hole geometries. With `chargeability_col`, a linearized IP inversion follows.
    Returns sections, a data-residual table and a QC report.
    """

    from .inversion import misfit_report, regularization_operator, tikhonov_invert

    elec, abmn = _electrodes_from_table(table)
    mesh = ResistivityMesh(elec)
    quad = _quadrature(WAVENUMBERS)
    d = np.log(table[rhoa_col].to_numpy(float))
    rho_bg = float(np.exp(np.median(d)))
    v_hom, _ = _forward(mesh, np.full(mesh.ncell, 1.0 / rho_bg), quad, abmn)
    k_num = rho_bg * 1.0 / v_hom  # numerical geometric factor per datum (rho_a = k V)
    m_ref = np.log(reference_resistivity if reference_resistivity > 0 else rho_bg)
    m = np.full(mesh.ncell, m_ref)
    nzc, nxc = mesh.nz - 1, mesh.nx - 1
    wm = regularization_operator((nzc, nxc), operator, alpha_s, alpha_x, 1.0, alpha_z)
    sigma_d = np.full(len(d), error_percent / 100.0)  # relative error ~ absolute error in ln(rho_a)
    history, lambdas = [], []
    jac_ln = None

    def predict(model, jacobian=False):
        v, jac = _forward(mesh, np.exp(-model), quad, abmn, want_jacobian=jacobian)
        return np.log(np.maximum(k_num * v, 1e-12)), v, jac

    for it in range(iterations):
        pred, v, jac = predict(m, True)
        # d ln(rho_a)/d ln(rho_c) = -(sigma_c / V) dV/dsigma_c
        jac_ln = -(jac * np.exp(-m)[None, :]) / v[:, None]
        nrms = float(np.sqrt(np.mean(((d - pred) / sigma_d) ** 2)))
        history.append(nrms)
        if nrms <= 1.0 and it > 0:
            break
        # 'fixed' mode: scale-free cooling lambda = smoothness * 0.6^it * tr(J'Wd'WdJ) / tr(Wm'Wm)
        fixed_lam = smoothness * (0.6**it) * np.sum((jac_ln / sigma_d[:, None]) ** 2) / max(wm.multiply(wm).sum(), 1e-30)
        res = tikhonov_invert(jac_ln, d - pred + jac_ln @ m, sigma_d, wm, m_ref, lambda_mode, fixed_lam, "cg", n_lambdas=16)
        m = np.clip(res.model, np.log(0.1), np.log(1e5))
        lambdas.append(res.lam)
    pred, _, _ = predict(m)
    qc = misfit_report(d, pred, sigma_d)
    outputs = [mesh.cell_grid(np.exp(m), "ERT_resistivity", "ohm.m")]
    cov = np.abs(jac_ln).sum(axis=0) if jac_ln is not None else np.zeros(mesh.ncell)
    outputs.append(mesh.cell_grid(np.log10(cov + 1e-12), "ERT_sensitivity_log10", "log10"))
    residuals = table.copy()
    residuals["rhoa_calc"] = np.exp(pred)
    residuals["misfit_percent"] = 100 * (np.exp(d - pred) - 1)
    residuals["normalized_residual"] = (d - pred) / sigma_d
    report = {**qc, "normalized_rms_history": history, "lambda_history": lambdas, "lambda_mode": lambda_mode,
              "operator": operator, "alpha_x": alpha_x, "alpha_z": alpha_z, "reference_resistivity": float(np.exp(m_ref)),
              "error_percent": error_percent, "cells": mesh.ncell, "data": int(len(d))}
    if chargeability_col and jac_ln is not None:
        ma = table[chargeability_col].to_numpy(float)
        ma = ma / 1000.0 if np.nanmax(ma) > 1 else ma  # mV/V -> fraction
        sig_ip = np.full(len(ma), max(0.05 * float(np.max(np.abs(ma))), 1e-4))
        ip = tikhonov_invert(jac_ln, ma, sig_ip, wm, 0.0, lambda_mode, 1.0, "cg", n_lambdas=16)  # Seigel: m_a ~= J_ln eta
        eta = np.clip(ip.model, 0, 1)
        outputs.append(mesh.cell_grid(eta * 1000, "IP_chargeability", "mV/V"))
        report["ip_normalized_rms"] = ip.report["normalized_rms"]
    return outputs, residuals, report


def make_array(n_electrodes: int = 24, spacing: float = 5.0, array: str = "dipole-dipole", max_n: int = 6) -> pd.DataFrame:
    """Surface electrode configurations (Wenner, Schlumberger, dipole-dipole)."""

    x = np.arange(n_electrodes) * spacing
    rows = []
    for i in range(n_electrodes):
        for n in range(1, max_n + 1):
            if array == "dipole-dipole":
                q = (i, i + 1, i + 1 + n, i + 2 + n)
            elif array == "wenner":
                q = (i, i + 3 * n, i + n, i + 2 * n)
            else:  # schlumberger: A M N B with MN = a, AM = n a
                q = (i, i + 2 * n + 1, i + n, i + n + 1)
            if max(q) < n_electrodes:
                a, b, m, nn = q
                rows.append({"a_x": x[a], "a_z": 0, "b_x": x[b], "b_z": 0, "m_x": x[m], "m_z": 0, "n_x": x[nn], "n_z": 0})
    return pd.DataFrame(rows)


def make_crosshole_array(x1: float = 0.0, x2: float = 20.0, depth: float = 30.0, spacing: float = 2.0) -> pd.DataFrame:
    """Bipole-bipole cross-hole geometry: A,M in hole 1; B,N in hole 2 (AM-BN)."""

    z = np.arange(spacing, depth + spacing / 2, spacing)
    rows = []
    for i in range(len(z) - 1):
        for j in range(len(z) - 1):
            rows.append({"a_x": x1, "a_z": z[i], "b_x": x2, "b_z": z[i], "m_x": x1, "m_z": z[j + 1] if j + 1 != i else z[j],
                         "n_x": x2, "n_z": z[j + 1] if j + 1 != i else z[j]})
    return pd.DataFrame(rows).drop_duplicates()


# ---------------------------------------------------------------- SIP / SP
def cole_cole(freq: np.ndarray, rho0: float, m: float, tau: float, c: float) -> np.ndarray:
    """Pelton et al. (1978): rho(w) = rho0 [1 - m (1 - 1/(1 + (i w tau)^c))]."""

    iwt = (1j * 2 * np.pi * np.asarray(freq, float) * tau) ** c
    return rho0 * (1 - m * (1 - 1 / (1 + iwt)))


def sip_cole_cole_fit(
    table: pd.DataFrame, freq_col: str = "frequency", amp_col: str = "amplitude", phase_col: str = "phase_mrad",
) -> tuple[ProfileData, ProfileData, dict]:
    """Fit a Cole-Cole model to complex resistivity spectra (amplitude ohm.m, phase mrad, negative for IP)."""

    f = table[freq_col].to_numpy(float)
    z = table[amp_col].to_numpy(float) * np.exp(1j * table[phase_col].to_numpy(float) / 1000.0)

    def resid(p):
        model = cole_cole(f, np.exp(p[0]), 1 / (1 + np.exp(-p[1])), np.exp(p[2]), 1 / (1 + np.exp(-p[3])))
        return np.concatenate([np.log(np.abs(model)) - np.log(np.abs(z)), (np.angle(model) - np.angle(z)) * 5])

    p0 = [np.log(np.abs(z).max()), 0.0, np.log(1 / (2 * np.pi * f[np.argmin(np.angle(z))])), 0.0]
    fit = least_squares(resid, p0)
    rho0, m, tau, c = np.exp(fit.x[0]), 1 / (1 + np.exp(-fit.x[1])), np.exp(fit.x[2]), 1 / (1 + np.exp(-fit.x[3]))
    ff = np.geomspace(f.min(), f.max(), 100)
    model = cole_cole(ff, rho0, m, tau, c)
    return (ProfileData(f, -np.angle(z) * 1000, name="SIP_observed_-phase", units="mrad", metadata={"axis": "frequency_Hz", "log_x": True}),
            ProfileData(ff, -np.angle(model) * 1000, name="SIP_ColeCole_-phase", units="mrad", metadata={"axis": "frequency_Hz", "log_x": True}),
            {"rho0_ohm_m": float(rho0), "chargeability": float(m), "tau_s": float(tau), "c": float(c),
             "peak_frequency_Hz": float(1 / (2 * np.pi * tau)),
             "note": "Larger tau ~ larger grains / pore throats (Pelton et al., 1978; Revil et al.)."})


SP_SHAPES = {"sphere": 1.5, "horizontal_cylinder": 1.0, "vertical_cylinder": 0.5}


def sp_forward(x: np.ndarray, k: float, x0: float, depth: float, alpha_deg: float, q: float) -> np.ndarray:
    """Yungul (1950) polarized body: V = K [(x-x0) cos a + h sin a] / ((x-x0)^2 + h^2)^q."""

    a = np.radians(alpha_deg)
    dx = np.asarray(x, float) - x0
    return k * (dx * np.cos(a) + depth * np.sin(a)) / (dx**2 + depth**2) ** q


def vlf_fraser_filter(profile: ProfileData) -> ProfileData:
    """Fraser (1969): F_{i+1.5} = (M_{i+2} + M_{i+3}) - (M_i + M_{i+1}); peaks over conductors."""

    m = profile.values
    f = (m[2:-1] + m[3:]) - (m[:-3] + m[1:-2])
    x = 0.5 * (profile.distance[1:-2] + profile.distance[2:-1])
    return profile.with_values(x, f, f"{profile.name}_Fraser", {"method": "fraser_filter"})
