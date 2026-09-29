import numpy as np

from core.grid_operations import (
    crop_grid,
    extended_grid_statistics,
    flip_grid,
    matrix_to_xyz,
    rotate_grid,
    xyz_to_grid,
)


def test_matrix_xyz_roundtrip_and_statistics():
    matrix = np.arange(12, dtype=float).reshape(3, 4)
    xyz = matrix_to_xyz(matrix, dx=10, dy=20, x0=100, y0=200)
    assert list(xyz.columns) == ["x", "y", "value"]
    assert xyz.shape == (12, 3)

    grid = xyz_to_grid(xyz, name="roundtrip", units="mGal")
    assert grid.values.shape == matrix.shape
    assert np.array_equal(grid.values, matrix)
    assert np.array_equal(grid.x_vector, np.array([100, 110, 120, 130], dtype=float))
    assert np.array_equal(grid.y_vector, np.array([200, 220, 240], dtype=float))

    stats = extended_grid_statistics(grid)
    assert stats["finite_count"] == 12
    assert stats["nan_count"] == 0
    assert stats["median"] == 5.5


def test_crop_flip_and_rotate_grid_do_not_mutate_input():
    matrix = np.arange(20, dtype=float).reshape(4, 5)
    grid = xyz_to_grid(matrix_to_xyz(matrix, dx=1, dy=1), name="base")
    original = grid.values.copy()

    cropped = crop_grid(grid, xmin=1, xmax=3, ymin=1, ymax=2)
    assert cropped.values.shape == (2, 3)
    assert np.array_equal(cropped.values, matrix[1:3, 1:4])

    flipped = flip_grid(grid, "x")
    assert np.array_equal(flipped.values, np.fliplr(matrix))

    rotated = rotate_grid(grid, 90)
    assert rotated.values.shape == (5, 4)
    assert np.array_equal(rotated.values, np.rot90(matrix, k=-1))
    assert rotated.dx == grid.dy
    assert rotated.dy == grid.dx

    assert np.array_equal(grid.values, original)
