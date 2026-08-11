import numpy as np

from core.fft_filters import (
    compute_butterworth_filter,
    compute_butterworth_regional_residual,
    compute_residual_from_upward,
    compute_upward_continuation,
)
from core.grid_tools import GridData


def make_grid():
    x = np.linspace(0, 50000, 51)
    y = np.linspace(0, 50000, 51)
    xx, yy = np.meshgrid(x, y)
    shallow = 18 * np.exp(-(((xx - 18000) ** 2 + (yy - 26000) ** 2) / (2 * 3500**2)))
    broad = 40 * np.exp(-(((xx - 30000) ** 2 + (yy - 24000) ** 2) / (2 * 13000**2)))
    return GridData(values=shallow + broad, x=x, y=y, dx=1000, dy=1000, name="test", units="mGal")


def test_upward_continuation_shape_and_smoothing():
    grid = make_grid()
    upward = compute_upward_continuation(grid, height=5000)
    assert upward.values.shape == grid.values.shape
    assert np.isfinite(upward.values).all()
    assert np.nanstd(upward.values) < np.nanstd(grid.values)


def test_upward_residual_reconstructs_original():
    grid = make_grid()
    regional, residual = compute_residual_from_upward(grid, height=5000)
    assert np.allclose(regional.values + residual.values, grid.values)


def test_butterworth_outputs_are_finite():
    grid = make_grid()
    low = compute_butterworth_filter(grid, cutoff=10000, filter_type="lowpass")
    high = compute_butterworth_filter(grid, cutoff=10000, filter_type="highpass")
    band = compute_butterworth_filter(grid, cutoff=(25000, 5000), filter_type="bandpass")
    for result in [low, high, band]:
        assert result.values.shape == grid.values.shape
        assert np.isfinite(result.values).all()


def test_butterworth_regional_residual_reconstructs_original():
    grid = make_grid()
    regional, residual = compute_butterworth_regional_residual(grid, cutoff_wavelength=10000)
    assert np.allclose(regional.values + residual.values, grid.values)
