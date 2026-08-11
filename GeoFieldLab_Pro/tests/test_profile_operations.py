import numpy as np

from core.grid_tools import GridData
from core.profile import (
    ProfileData,
    extract_cross_section,
    interpolate_profile,
    polynomial_fit_profile,
    profile_statistics,
    remove_linear_trend_profile,
    running_average_profile,
)


def make_grid():
    x = np.arange(0, 6, dtype=float)
    y = np.arange(0, 6, dtype=float)
    xx, yy = np.meshgrid(x, y)
    values = xx + 2 * yy
    return GridData(values=values, x=x, y=y, dx=1, dy=1, name="plane", units="mGal")


def test_extract_cross_section_samples_regular_grid():
    grid = make_grid()
    profile = extract_cross_section(grid, (0, 0), (5, 0), spacing=1)
    assert profile.name == "plane_cross_section"
    assert np.array_equal(profile.distance, np.arange(0, 6, dtype=float))
    assert np.allclose(profile.values, np.arange(0, 6, dtype=float))
    assert profile.metadata["sample_count"] == 6


def test_profile_processing_outputs_are_finite_and_sized():
    profile = ProfileData(
        distance=np.array([0, 1, 2, 3, 4], dtype=float),
        values=np.array([1, 3, 5, 7, 9], dtype=float),
        units="nT",
    )
    interpolated = interpolate_profile(profile, spacing=0.5)
    assert interpolated.distance.size == 9

    averaged = running_average_profile(profile, window_size=3)
    assert averaged.values.shape == profile.values.shape
    assert np.isfinite(averaged.values).all()

    detrended = remove_linear_trend_profile(profile)
    assert np.allclose(detrended.values, 0, atol=1e-12)

    fit, residual = polynomial_fit_profile(profile, degree=1)
    assert np.allclose(fit.values, profile.values)
    assert np.allclose(residual.values, 0, atol=1e-12)

    stats = profile_statistics(profile)
    assert stats["finite_count"] == 5
    assert stats["rms"] > 0
