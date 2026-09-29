"""Desktop workflow controller that connects UI actions to core methods."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.analytic_signal import compute_analytic_signal
from core.data_io import data_quality_report, read_table, validate_numeric_columns
from core.derivatives import compute_fvd, compute_svd, compute_thg
from core.edge_detection import compute_fsed, threshold_edges
from core.exports import export_grid_csv, export_history_json, export_markdown_report, export_png
from core.fft_filters import (
    compute_butterworth_filter,
    compute_butterworth_regional_residual,
    compute_residual_from_upward,
)
from core.grid_operations import crop_grid, extended_grid_statistics, flip_grid, rotate_grid
from core.grid_tools import (
    GridData,
    coordinate_warning,
    detect_regular_grid,
    grid_statistics,
    interpolate_scattered_to_grid,
    reshape_regular_grid,
)
from core.history import make_history_entry
from core.profile import (
    ProfileData,
    crop_profile,
    extract_cross_section,
    flip_profile,
    interpolate_profile,
    polynomial_fit_profile,
    profile_statistics,
    remove_linear_trend_profile,
    running_average_profile,
)
from core.spectral import compute_raps, fit_raps_segment
from core.tilt import compute_tilt_derivative

from .project import ProjectSession


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DATASET = PROJECT_ROOT / "examples" / "synthetic_anomaly.csv"


class ProjectController:
    """High-level operations for the desktop application."""

    def __init__(self, session: ProjectSession | None = None) -> None:
        self.session = session or ProjectSession()

    def new_project(self, name: str = "Untitled GeoFieldLab Project") -> ProjectSession:
        self.session = ProjectSession(project_name=name)
        return self.session

    def load_project(self, path: str | Path) -> ProjectSession:
        self.session = ProjectSession.load(path)
        return self.session

    def save_project(self, path: str | Path) -> None:
        self.session.save(path)

    def load_table(self, path: str | Path) -> pd.DataFrame:
        data = read_table(Path(path))
        self.session.raw_data = data
        self.session.input_file_name = str(path)
        self.session.add_recent_file(path)
        self.session.dirty = True
        return data

    def load_example_dataset(self) -> pd.DataFrame:
        return self.load_table(EXAMPLE_DATASET)

    def data_quality(self, x_col: str, y_col: str, value_col: str) -> dict[str, Any]:
        if self.session.raw_data is None:
            raise ValueError("Load a table before running data quality checks.")
        converted = validate_numeric_columns(self.session.raw_data, [x_col, y_col, value_col])
        return data_quality_report(converted, x_col, y_col, value_col)

    def detect_grid(self, x_col: str, y_col: str, value_col: str) -> dict[str, Any]:
        if self.session.raw_data is None:
            raise ValueError("Load a table before detecting a grid.")
        converted = validate_numeric_columns(self.session.raw_data, [x_col, y_col, value_col])
        return detect_regular_grid(converted, x_col, y_col, value_col)

    def create_grid(
        self,
        x_col: str,
        y_col: str,
        value_col: str,
        units: str = "unknown",
        crs: str | None = None,
        mode: str = "auto-detect",
        spacing: float | None = None,
        interpolation_method: str = "linear",
        fill_method: str = "nearest",
        name: str = "Original anomaly",
    ) -> GridData:
        if self.session.raw_data is None:
            raise ValueError("Load a table before creating a grid.")
        converted = validate_numeric_columns(self.session.raw_data, [x_col, y_col, value_col])
        detection = detect_regular_grid(converted, x_col, y_col, value_col)
        use_regular = detection["is_regular"] if mode == "auto-detect" else mode == "regular grid"
        start = time.perf_counter()
        if use_regular:
            grid = reshape_regular_grid(converted, x_col, y_col, value_col, name=name, units=units, crs=crs)
        else:
            grid_spacing = spacing or float(detection.get("dx") or detection.get("dy") or 1.0)
            grid = interpolate_scattered_to_grid(
                converted,
                x_col,
                y_col,
                value_col,
                spacing=grid_spacing,
                method=interpolation_method,
                fill_method=fill_method,
                name=name,
                units=units,
                crs=crs,
            )
        elapsed = time.perf_counter() - start
        stored = self.session.add_layer(grid, replace_existing=name in self.session.layers)
        warnings = [warning for warning in [coordinate_warning(stored.x, stored.y)] if warning]
        self.session.settings.update(
            {
                "x_col": x_col,
                "y_col": y_col,
                "value_col": value_col,
                "units": units,
                "crs": crs,
                "grid_mode": mode,
            }
        )
        self.session.add_history(
            make_history_entry(
                method_name="Create grid",
                input_layer=self.session.input_file_name or "session data",
                output_layer=stored.name,
                parameters={
                    "mode": mode,
                    "interpolation_method": None if use_regular else interpolation_method,
                    "grid_spacing": stored.dx,
                    "fill_method": None if use_regular else fill_method,
                },
                result=stored,
                warnings=warnings,
                computation_time_s=elapsed,
                formula="Regular reshape or scipy.interpolate.griddata",
            )
        )
        return stored

    def _record_grid_result(
        self,
        method_name: str,
        input_grid: GridData,
        result: GridData,
        parameters: dict[str, Any] | None = None,
        computation_time_s: float | None = None,
        extra_warnings: list[str] | None = None,
    ) -> GridData:
        stored = self.session.add_layer(result)
        warnings = [warning for warning in [coordinate_warning(stored.x, stored.y), stored.metadata.get("warning")] if warning]
        if extra_warnings:
            warnings.extend(extra_warnings)
        self.session.add_history(
            make_history_entry(
                method_name=method_name,
                input_layer=input_grid.name,
                output_layer=stored.name,
                parameters=parameters or {},
                result=stored,
                warnings=warnings,
                computation_time_s=computation_time_s,
                formula=stored.metadata.get("formula"),
            )
        )
        return stored

    def run_grid_method(self, method_name: str, layer_name: str | None, func, parameters: dict[str, Any]) -> GridData:
        grid = self.session.get_layer(layer_name)
        original_values = grid.values.copy()
        start = time.perf_counter()
        result = func(grid)
        elapsed = time.perf_counter() - start
        if not np.array_equal(grid.values, original_values, equal_nan=True):
            raise RuntimeError(f"{method_name} mutated input layer '{grid.name}'.")
        return self._record_grid_result(method_name, grid, result, parameters, elapsed)

    def compute_thg(self, layer_name: str | None = None) -> GridData:
        return self.run_grid_method("THG", layer_name, compute_thg, {})

    def compute_fvd(self, layer_name: str | None = None, padding: str = "none") -> GridData:
        return self.run_grid_method("FVD", layer_name, lambda grid: compute_fvd(grid, padding=padding), {"padding": padding})

    def compute_svd(self, layer_name: str | None = None, padding: str = "none") -> GridData:
        return self.run_grid_method("SVD", layer_name, lambda grid: compute_svd(grid, padding=padding), {"padding": padding})

    def compute_tilt(self, layer_name: str | None = None, epsilon: float = 1e-10, padding: str = "none") -> GridData:
        return self.run_grid_method(
            "TILT",
            layer_name,
            lambda grid: compute_tilt_derivative(grid, epsilon=epsilon, padding=padding),
            {"epsilon": epsilon, "padding": padding},
        )

    def compute_analytic_signal(self, layer_name: str | None = None, padding: str = "none") -> GridData:
        return self.run_grid_method(
            "ASA",
            layer_name,
            lambda grid: compute_analytic_signal(grid, padding=padding),
            {"padding": padding},
        )

    def compute_upward_residual(
        self,
        layer_name: str | None = None,
        height: float = 5000.0,
        padding: str = "reflect",
        remove_mean: bool = False,
    ) -> tuple[GridData, GridData]:
        grid = self.session.get_layer(layer_name)
        start = time.perf_counter()
        regional, residual = compute_residual_from_upward(grid, height=height, padding=padding, remove_mean=remove_mean)
        elapsed = time.perf_counter() - start
        params = {"height": height, "padding": padding, "remove_mean": remove_mean}
        return (
            self._record_grid_result("Upward continuation", grid, regional, params, elapsed),
            self._record_grid_result("Residual from upward continuation", grid, residual, params, elapsed),
        )

    def compute_butterworth(
        self,
        layer_name: str | None = None,
        cutoff: float | tuple[float, float] = 10000.0,
        order: int = 4,
        filter_type: str = "lowpass",
        padding: str = "reflect",
    ) -> GridData:
        return self.run_grid_method(
            "Butterworth filter",
            layer_name,
            lambda grid: compute_butterworth_filter(
                grid,
                cutoff=cutoff,
                order=order,
                filter_type=filter_type,
                padding=padding,
            ),
            {"cutoff": cutoff, "order": order, "filter_type": filter_type, "padding": padding},
        )

    def compute_butterworth_regional_residual(
        self,
        layer_name: str | None = None,
        cutoff_wavelength: float = 10000.0,
        order: int = 4,
        padding: str = "reflect",
    ) -> tuple[GridData, GridData]:
        grid = self.session.get_layer(layer_name)
        start = time.perf_counter()
        regional, residual = compute_butterworth_regional_residual(
            grid,
            cutoff_wavelength=cutoff_wavelength,
            order=order,
            padding=padding,
        )
        elapsed = time.perf_counter() - start
        params = {"cutoff_wavelength": cutoff_wavelength, "order": order, "padding": padding}
        return (
            self._record_grid_result("Butterworth regional", grid, regional, params, elapsed),
            self._record_grid_result("Butterworth residual", grid, residual, params, elapsed),
        )

    def compute_fsed(
        self,
        layer_name: str | None = None,
        base: str = "THG",
        gain: float = 5.0,
        method: str = "logistic",
        threshold: float | None = None,
    ) -> GridData | tuple[GridData, GridData]:
        result = self.run_grid_method(
            "FSED",
            layer_name,
            lambda grid: compute_fsed(grid, base=base, gain=gain, method=method),
            {"base": base, "gain": gain, "method": method},
        )
        if threshold is None:
            return result
        binary = threshold_edges(result, threshold=threshold)
        stored_binary = self._record_grid_result("FSED threshold", result, binary, {"threshold": threshold}, 0.0)
        return result, stored_binary

    def compute_raps(
        self,
        layer_name: str | None = None,
        remove_mean: bool = True,
        detrend: bool = False,
        window: str = "hanning",
        padding: str = "zero",
        radial_bin_size: float | None = None,
        automatic_fit: bool = True,
        k_min: float | None = None,
        k_max: float | None = None,
    ) -> dict[str, Any]:
        grid = self.session.get_layer(layer_name)
        start = time.perf_counter()
        result = compute_raps(
            grid,
            remove_mean=remove_mean,
            detrend=detrend,
            window=window,
            padding=padding,
            radial_bin_size=radial_bin_size,
        )
        fit = fit_raps_segment(result, k_min=k_min, k_max=k_max, automatic=automatic_fit)
        elapsed = time.perf_counter() - start
        summary = {
            "input_layer": grid.name,
            "spectrum_convention": result.metadata["spectrum_convention"],
            "wavenumber_unit": result.metadata["wavenumber_unit"],
            "depth_formula": result.metadata["depth_formula"],
            "slope": fit.slope,
            "r_squared": fit.r_squared,
            "estimated_depth_m": fit.depth_m,
            "estimated_depth_km": fit.depth_km,
        }
        self.session.raps_summary = summary
        self.session.add_history(
            make_history_entry(
                method_name="RAPS",
                input_layer=grid.name,
                output_layer="RAPS spectrum",
                parameters={
                    "remove_mean": remove_mean,
                    "detrend": detrend,
                    "window": window,
                    "padding": padding,
                    "radial_bin_size": radial_bin_size or "auto",
                    "automatic_fit": automatic_fit,
                    "k_min": k_min,
                    "k_max": k_max,
                },
                result=grid,
                warnings=[result.metadata["warning"]],
                computation_time_s=elapsed,
                formula=result.metadata["depth_formula"],
            )
        )
        return summary

    def crop_grid(self, layer_name: str | None, xmin: float, xmax: float, ymin: float, ymax: float) -> GridData:
        grid = self.session.get_layer(layer_name)
        start = time.perf_counter()
        result = crop_grid(grid, xmin, xmax, ymin, ymax)
        return self._record_grid_result(
            "Crop grid",
            grid,
            result,
            {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax},
            time.perf_counter() - start,
        )

    def flip_grid(self, layer_name: str | None, axis: str) -> GridData:
        grid = self.session.get_layer(layer_name)
        start = time.perf_counter()
        result = flip_grid(grid, axis=axis)
        return self._record_grid_result("Flip grid", grid, result, {"axis": axis}, time.perf_counter() - start)

    def rotate_grid(self, layer_name: str | None, degrees_clockwise: int) -> GridData:
        grid = self.session.get_layer(layer_name)
        start = time.perf_counter()
        result = rotate_grid(grid, degrees_clockwise=degrees_clockwise)
        return self._record_grid_result(
            "Rotate grid",
            grid,
            result,
            {"degrees_clockwise": degrees_clockwise},
            time.perf_counter() - start,
        )

    def extract_cross_section(
        self,
        layer_name: str | None,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        spacing: float | None = None,
    ) -> ProfileData:
        grid = self.session.get_layer(layer_name)
        start = time.perf_counter()
        profile = extract_cross_section(grid, start_xy=start_xy, end_xy=end_xy, spacing=spacing)
        stored = self.session.add_profile(profile)
        self.session.add_history(
            make_history_entry(
                method_name="Cross-section",
                input_layer=grid.name,
                output_layer=stored.name,
                parameters={"start_xy": start_xy, "end_xy": end_xy, "spacing": spacing},
                result=None,
                warnings=[],
                computation_time_s=time.perf_counter() - start,
                formula="RegularGridInterpolator sampled along a straight line",
            )
        )
        return stored

    def transform_profile(self, profile_name: str | None, operation: str, **parameters: Any) -> ProfileData | tuple[ProfileData, ProfileData]:
        profile = self.session.get_profile(profile_name)
        start = time.perf_counter()
        if operation == "crop":
            result = crop_profile(profile, parameters["distance_min"], parameters["distance_max"])
        elif operation == "flip":
            result = flip_profile(profile)
        elif operation == "interpolate":
            result = interpolate_profile(profile, parameters["spacing"])
        elif operation == "running_average":
            result = running_average_profile(profile, parameters["window_size"])
        elif operation == "remove_linear_trend":
            result = remove_linear_trend_profile(profile)
        elif operation == "polynomial_fit":
            fit, residual = polynomial_fit_profile(profile, parameters["degree"])
            stored_fit = self.session.add_profile(fit)
            stored_residual = self.session.add_profile(residual)
            self.session.add_history(
                make_history_entry(
                    method_name="Profile polynomial fit",
                    input_layer=profile.name,
                    output_layer=f"{stored_fit.name}; {stored_residual.name}",
                    parameters=parameters,
                    result=None,
                    warnings=[],
                    computation_time_s=time.perf_counter() - start,
                )
            )
            return stored_fit, stored_residual
        else:
            raise ValueError(f"Unsupported profile operation: {operation}")
        stored = self.session.add_profile(result)
        self.session.add_history(
            make_history_entry(
                method_name=f"Profile {operation}",
                input_layer=profile.name,
                output_layer=stored.name,
                parameters=parameters,
                result=None,
                warnings=[],
                computation_time_s=time.perf_counter() - start,
            )
        )
        return stored

    def layer_statistics(self, layer_name: str | None = None) -> dict[str, float | int]:
        return extended_grid_statistics(self.session.get_layer(layer_name))

    def profile_statistics(self, profile_name: str | None = None) -> dict[str, float | int]:
        return profile_statistics(self.session.get_profile(profile_name))

    def export_layer_csv(self, layer_name: str | None, path: str | Path | None = None) -> str:
        return export_grid_csv(self.session.get_layer(layer_name), path=path)

    def export_layer_png(self, layer_name: str | None, path: str | Path | None = None) -> bytes:
        return export_png(self.session.get_layer(layer_name), path=path)

    def export_profile_csv(self, profile_name: str | None, path: str | Path | None = None) -> str:
        text = self.session.get_profile(profile_name).to_dataframe().to_csv(index=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def export_history_json(self, path: str | Path | None = None) -> str:
        return export_history_json(self.session.history, path=path)

    def export_markdown_report(self, path: str | Path | None = None) -> str:
        return export_markdown_report(
            project_name=self.session.project_name,
            layers=self.session.layers,
            history=self.session.history,
            input_file=self.session.input_file_name,
            raps_summary=self.session.raps_summary,
            path=path,
        )
