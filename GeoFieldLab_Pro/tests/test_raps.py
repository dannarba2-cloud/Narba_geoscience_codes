import numpy as np

from core.grid_tools import GridData
from core.spectral import compute_raps, fit_raps_segment


def make_grid():
    x = np.linspace(0, 50000, 51)
    y = np.linspace(0, 50000, 51)
    xx, yy = np.meshgrid(x, y)
    values = 30 * np.exp(-(((xx - 23000) ** 2 + (yy - 25000) ** 2) / (2 * 8000**2)))
    values += 10 * np.exp(-(((xx - 36000) ** 2 + (yy - 16000) ** 2) / (2 * 3500**2)))
    return GridData(values=values, x=x, y=y, dx=1000, dy=1000, name="test", units="mGal")


def test_raps_bins_power_and_metadata():
    result = compute_raps(make_grid())
    assert np.all(np.diff(result.k) > 0)
    assert np.isfinite(result.power).all()
    assert np.isfinite(result.log_power).all()
    assert result.metadata["wavenumber_unit"] == "rad/m"
    assert result.metadata["depth_formula"] == "depth = -slope / 2"


def test_automatic_segment_fit_returns_depth():
    result = compute_raps(make_grid())
    fit = fit_raps_segment(result, automatic=True)
    assert np.isfinite(fit.slope)
    assert fit.formula.startswith("depth = -slope / 2")
    if fit.slope < 0:
        assert fit.depth_m > 0


def test_manual_segment_fit_returns_depth():
    result = compute_raps(make_grid())
    k_min = float(np.quantile(result.k, 0.2))
    k_max = float(np.quantile(result.k, 0.6))
    fit = fit_raps_segment(result, k_min=k_min, k_max=k_max, automatic=False)
    assert fit.k_min >= k_min
    assert fit.k_max <= k_max
    assert np.isfinite(fit.depth_m)
