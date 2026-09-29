"""Analytic checks that pin the geophysics (signs, units, conventions)."""

import numpy as np
import pandas as pd

from core.depth import euler_deconvolution, werner_deconvolution
from core.electrical import ves_forward
from core.em import mt1d_forward
from core.gravity import bouguer_slab, normal_gravity
from core.grid_tools import GridData
from core.interpolation import minimum_curvature
from core.potential_field import reduce_to_pole
from core.prisms import G, gz_kernel, magnetic_kernel, tensor_kernels
from core.profile import ProfileData
from core.refraction import plus_minus
from core.seismic import Section, nmo_correction, ricker


def test_prism_matches_point_mass_and_dipole():
    cube = np.array([[-50, 50, -50, 50, 950, 1050.0]])
    volume = 100.0**3
    obs = np.array([[700.0, -300.0, -50.0]])
    r = np.array([0, 0, 1000.0]) - obs[0]
    assert np.isclose(gz_kernel(obs, cube)[0, 0], G * volume * r[2] / np.linalg.norm(r) ** 3, rtol=1e-4)
    t = tensor_kernels(obs, cube)
    rv = -r
    R = np.linalg.norm(rv)
    assert np.isclose(t["xz"][0, 0], volume * 3 * rv[0] * rv[2] / R**5, rtol=1e-4)
    # vertical dipole under a vertical field: dT = Cm * 2 m / d^3
    assert np.isclose(magnetic_kernel([0, 0, 0], cube, 90, 0)[0, 0], 1e-7 * volume * 2 / 1000.0**3 * 1e9, rtol=1e-4)


def test_normal_gravity_and_slab():
    assert np.allclose(normal_gravity([0, 90]), [978032.67715, 983218.6368], atol=1e-3)
    assert np.isclose(bouguer_slab(100.0, 2670.0), 11.19, atol=0.01)


def test_rtp_is_identity_at_the_pole_and_centres_anomaly():
    x = np.arange(0, 20001, 250.0)
    xx, yy = np.meshgrid(x, x)
    obs = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    body = np.array([[9500, 10500, 9500, 10500, 1000, 2000.0]])
    pole = magnetic_kernel(obs, body, 90, 0).reshape(xx.shape)
    grid = GridData(pole, x, x, 250, 250)
    assert np.allclose(reduce_to_pole(grid, 90, 0).values, pole, atol=1e-6 * np.abs(pole).max())
    tilted = GridData(magnetic_kernel(obs, body, 45, 20).reshape(xx.shape), x, x, 250, 250)
    rtp = reduce_to_pole(tilted, 45, 20).values
    i, j = np.unravel_index(np.argmax(rtp), rtp.shape)
    assert abs(x[j] - 10000) <= 500 and abs(x[i] - 10000) <= 500


def test_euler_and_werner_recover_depth():
    x = np.arange(-20000, 20001, 500.0)
    xx, yy = np.meshgrid(x, x)
    obs = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    g = (gz_kernel(obs, np.array([[-250, 250, -250, 250, 2750, 3250.0]]))[:, 0] * 500 * 1e5).reshape(xx.shape)
    sol = euler_deconvolution(GridData(g, x, x, 500, 500), structural_index=2, window=10)
    assert abs(sol.depth.median() - 3000) < 300
    xs = np.arange(-10000, 10001, 100.0)
    dike = magnetic_kernel(np.column_stack([xs, 0 * xs, 0 * xs]), np.array([[-10, 10, -1e6, 1e6, 1000, 1e5]]), 90, 0)[:, 0]
    w = werner_deconvolution(ProfileData(xs, dike), window=9)
    assert abs(w.depth.median() - 1000) < 20 and abs(w.x.median()) < 20


def test_ves_matches_two_layer_image_series():
    s = np.geomspace(1, 1000, 20)
    k = (10.0 - 100.0) / (110.0)
    n = np.arange(1, 400)
    exact = 100 * (1 + 2 * np.sum(k ** n[None] * s[:, None] ** 3 / (s[:, None] ** 2 + (20.0 * n[None]) ** 2) ** 1.5, axis=1))
    assert np.allclose(ves_forward(s, np.array([100.0, 10.0]), np.array([10.0])), exact, rtol=5e-3)


def test_mt_half_space():
    rho, phase = mt1d_forward(np.geomspace(1e-3, 1e3, 7), np.array([100.0]), np.array([]))
    assert np.allclose(rho, 100.0) and np.allclose(phase, 45.0)


def test_nmo_flattens_hyperbola():
    dt, t0, v = 0.004, 0.5, 2000.0
    offsets = np.arange(0, 1001, 100.0)
    data = np.zeros((300, len(offsets)))
    for j, x in enumerate(offsets):
        data[int(round(np.sqrt(t0**2 + (x / v) ** 2) / dt)), j] = 1.0
    w = ricker(25, dt)
    data = np.apply_along_axis(lambda c: np.convolve(c, w, "same"), 0, data)
    out = nmo_correction(Section(data, dt, x=offsets, headers={"offset": offsets}), v)
    peaks = np.argmax(out.data, axis=0) * dt
    assert np.allclose(peaks, t0, atol=2 * dt)


def test_plus_minus_flat_refractor():
    v1, v2, h, length = 500.0, 2500.0, 20.0, 200.0
    ic = np.arcsin(v1 / v2)
    xs = np.arange(20, 181, 5.0)
    delay = 2 * h * np.cos(ic) / v1
    table = pd.DataFrame({"x": xs, "t_forward": xs / v2 + delay, "t_reverse": (length - xs) / v2 + delay})
    out, info = plus_minus(table, v1=v1, reciprocal_time=length / v2 + delay)
    assert np.isclose(info["v2_m_per_s"], v2) and np.allclose(out.depth, h)


def test_tikhonov_discrepancy_solvers_and_residual_qc():
    from core.inversion import data_errors, misfit_report, regularization_operator, tikhonov_invert

    rng = np.random.default_rng(0)
    x, xc = np.linspace(0, 1e4, 80), np.linspace(0, 1e4, 100)
    g = 2 * 6.674e-11 * 1500.0 / ((x[:, None] - xc[None, :]) ** 2 + 1500.0**2) * 1e10
    true = 300 * np.exp(-((xc - 4000) / 800) ** 2) - 150 * np.exp(-((xc - 7500) / 500) ** 2)
    sigma = data_errors(g @ true, 2.0)
    d = g @ true + rng.normal(0, 1, len(x)) * sigma
    wm = regularization_operator((100,), "first-derivative")
    results = [tikhonov_invert(g, d, sigma, wm, 0.0, "discrepancy", solver=s) for s in ("direct", "cg", "subspace")]
    for r in results:
        assert abs(r.report["normalized_rms"] - 1.0) < 0.02  # Occam target reached
        assert r.report["residuals"].startswith("random")
    assert np.allclose(results[0].model, results[1].model, atol=1.0)
    assert np.sqrt(np.mean((results[0].model - true) ** 2)) < 0.2 * np.abs(true).max()
    for mode in ("l-curve", "gcv"):
        assert 0.1 < tikhonov_invert(g, d, sigma, wm, 0.0, mode, solver="direct").report["normalized_rms"] < 2
    trend = np.linspace(0, 5, 80)  # a missed coherent signal must be flagged
    assert misfit_report(d + trend, d, sigma)["residuals"].startswith("coherent")


def test_minimum_curvature_honours_data():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 1e4, 200), rng.uniform(0, 1e4, 200)
    f = np.sin(x / 2000) + np.cos(y / 1500)
    g = minimum_curvature(pd.DataFrame({"x": x, "y": y, "value": f}), spacing=250.0)
    xx, yy = g.mesh
    assert np.sqrt(np.mean((g.values - (np.sin(xx / 2000) + np.cos(yy / 1500))) ** 2)) < 0.05
