import numpy as np

from core.derivatives import compute_fvd, compute_svd, compute_thg
from core.grid_tools import GridData


def make_grid():
    x = np.linspace(0, 50000, 51)
    y = np.linspace(0, 50000, 51)
    xx, yy = np.meshgrid(x, y)
    values = 20 * np.exp(-(((xx - 20000) ** 2 + (yy - 25000) ** 2) / (2 * 5000**2)))
    values += 35 * np.exp(-(((xx - 34000) ** 2 + (yy - 18000) ** 2) / (2 * 11000**2)))
    return GridData(values=values, x=x, y=y, dx=1000, dy=1000, name="test", units="mGal")


def test_derivative_shapes_and_finite_values():
    grid = make_grid()
    for result in [compute_thg(grid), compute_fvd(grid), compute_svd(grid)]:
        assert result.values.shape == grid.values.shape
        assert np.isfinite(result.values).all()


def test_metadata_preserved_and_input_not_mutated():
    grid = make_grid()
    original = grid.values.copy()
    result = compute_thg(grid)
    assert result.dx == grid.dx
    assert result.dy == grid.dy
    assert result.units == "mGal/m"
    assert np.array_equal(grid.values, original)
