from pathlib import Path

import numpy as np

from core.analytic_signal import compute_analytic_signal
from core.data_io import read_table
from core.derivatives import compute_fvd, compute_svd, compute_thg
from core.edge_detection import compute_fsed
from core.exports import export_grid_csv
from core.fft_filters import compute_butterworth_regional_residual, compute_residual_from_upward
from core.grid_tools import detect_regular_grid, reshape_regular_grid
from core.spectral import compute_raps, fit_raps_segment
from core.tilt import compute_tilt_derivative


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "synthetic_anomaly.csv"


def test_complete_example_workflow():
    data = read_table(EXAMPLE)
    detection = detect_regular_grid(data, "x", "y", "anomaly")
    assert detection["is_regular"]
    grid = reshape_regular_grid(data, "x", "y", "anomaly", name="Original anomaly", units="mGal")

    products = [
        compute_thg(grid),
        compute_fvd(grid),
        compute_svd(grid),
        compute_tilt_derivative(grid),
        compute_analytic_signal(grid),
        compute_fsed(grid, base="THG"),
        compute_fsed(grid, base="absolute TILT"),
    ]
    regional_uc, residual_uc = compute_residual_from_upward(grid, height=5000)
    regional_bw, residual_bw = compute_butterworth_regional_residual(grid, cutoff_wavelength=10000)
    products.extend([regional_uc, residual_uc, regional_bw, residual_bw])

    raps = compute_raps(grid)
    fit = fit_raps_segment(raps, automatic=True)
    assert np.isfinite(fit.depth_m)

    for product in products:
        assert product.values.shape == grid.values.shape
        assert np.isfinite(product.values).all()
        csv_text = export_grid_csv(product)
        assert "x,y,value,layer_name,units" in csv_text.splitlines()[0]
