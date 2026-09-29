from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st

from core.analytic_signal import compute_analytic_signal
from core.data_io import data_quality_report, read_table
from core.derivatives import compute_fvd, compute_svd, compute_thg
from core.edge_detection import compute_fsed, threshold_edges
from core.exports import export_history_json, export_markdown_report
from core.fft_filters import (
    compute_butterworth_filter,
    compute_butterworth_regional_residual,
    compute_residual_from_upward,
)
from core.grid_tools import (
    GridData,
    coordinate_warning,
    detect_regular_grid,
    grid_statistics,
    interpolate_scattered_to_grid,
    reshape_regular_grid,
)
from core.history import add_history, make_history_entry
from core.spectral import compute_raps, fit_raps_segment
from core.tilt import compute_tilt_derivative
from core.visualization import plot_raps
from ui.method_registry import FUTURE_MODULES, SCIENTIFIC_WARNINGS
from ui.panels import add_layer, initialize_session_state, layer_selector, show_grid_summary
from ui.theme import app_header, apply_theme, badge_row, section_intro


APP_NAME = "GEONARBA"
PROJECT_ROOT = Path(__file__).resolve().parent
EXAMPLE_DATASET = PROJECT_ROOT / "examples" / "synthetic_anomaly.csv"


st.set_page_config(page_title=f"{APP_NAME} V1", page_icon="G", layout="wide")
apply_theme()
initialize_session_state()


def warn_science() -> None:
    with st.expander("Scientific warnings", expanded=False):
        for warning in SCIENTIFIC_WARNINGS:
            st.warning(warning)


def store_result(
    method_name: str,
    input_grid: GridData,
    result: GridData,
    parameters: dict,
    computation_time_s: float,
    extra_warnings: list[str] | None = None,
) -> None:
    warnings = []
    coord = coordinate_warning(result.x, result.y)
    if coord:
        warnings.append(coord)
    if result.metadata.get("warning"):
        warnings.append(str(result.metadata["warning"]))
    if extra_warnings:
        warnings.extend(extra_warnings)
    add_layer(result)
    add_history(
        st.session_state.history,
        make_history_entry(
            method_name=method_name,
            input_layer=input_grid.name,
            output_layer=result.name,
            parameters=parameters,
            result=result,
            warnings=warnings,
            computation_time_s=computation_time_s,
            formula=result.metadata.get("formula"),
        ),
    )
    st.success(f"Computed {result.name}")
    show_grid_summary(result, title=f"Result: {result.name}", export_key=method_name)


def run_grid_method(
    method_name: str,
    input_grid: GridData,
    func: Callable[[], GridData],
    parameters: dict,
    extra_warnings: list[str] | None = None,
) -> None:
    try:
        start = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - start
        store_result(method_name, input_grid, result, parameters, elapsed, extra_warnings)
    except Exception as exc:
        st.error(f"{method_name} failed: {exc}")


def sidebar() -> str:
    st.sidebar.title(APP_NAME)
    st.sidebar.caption("Potential-field processing workspace")
    st.sidebar.metric("Layers", len(st.session_state.layers))
    st.sidebar.metric("History steps", len(st.session_state.history))
    sections = [
        "Project / Data Import",
        "Grid Settings",
        "Original Map",
        "Regional-Residual",
        "Derivatives",
        "Tilt Filters",
        "Analytic Signal",
        "Edge Detection / FSED",
        "Spectral Analysis / RAPS",
        "Export and Report",
        "Future Modules",
    ]
    selected = st.sidebar.radio("Workspace", sections)
    if st.session_state.layers:
        st.sidebar.divider()
        st.sidebar.subheader("Layer inventory")
        for layer_name in list(st.session_state.layers.keys())[-8:]:
            st.sidebar.caption(layer_name)
    st.sidebar.divider()
    st.sidebar.subheader("Planned modules")
    for category, items in FUTURE_MODULES.items():
        with st.sidebar.expander(category):
            for item in items:
                st.caption(f"{item} - planned for V2/V3")
    return selected


def panel_project_import() -> None:
    st.header("Project / Data Import")
    section_intro(
        "Data import",
        "Load CSV, TXT, XYZ, or DAT anomaly data, inspect the table, and map the coordinate/value columns before gridding.",
    )
    badge_row(["CSV/TXT/XYZ/DAT", "Column mapping", "Data quality", "Coordinate warning"])
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Load example dataset", use_container_width=True):
            try:
                st.session_state.raw_data = read_table(EXAMPLE_DATASET)
                st.session_state.input_file_name = str(EXAMPLE_DATASET)
                st.success("Example dataset loaded.")
            except Exception as exc:
                st.error(f"Could not load example dataset: {exc}")
        uploaded = st.file_uploader("Upload CSV/TXT/XYZ/DAT", type=["csv", "txt", "xyz", "dat"])
        if uploaded is not None:
            try:
                st.session_state.raw_data = read_table(uploaded, filename=uploaded.name)
                st.session_state.input_file_name = uploaded.name
                st.success(f"Loaded {uploaded.name}")
            except Exception as exc:
                st.error(str(exc))
    with col2:
        if st.session_state.raw_data is None:
            st.info("Load the example dataset or upload your own file.")
        else:
            st.subheader("Data preview")
            st.dataframe(st.session_state.raw_data.head(100), use_container_width=True)

    data = st.session_state.raw_data
    if data is not None:
        numeric_candidates = list(data.columns)
        settings = st.session_state.settings
        c1, c2, c3 = st.columns(3)
        with c1:
            settings["x_col"] = st.selectbox("X column", numeric_candidates, index=0, key="x_col")
        with c2:
            settings["y_col"] = st.selectbox(
                "Y column",
                numeric_candidates,
                index=min(1, len(numeric_candidates) - 1),
                key="y_col",
            )
        with c3:
            settings["value_col"] = st.selectbox(
                "Anomaly column",
                numeric_candidates,
                index=min(2, len(numeric_candidates) - 1),
                key="value_col",
            )
        c4, c5, c6 = st.columns(3)
        with c4:
            settings["units"] = st.selectbox("Anomaly unit", ["mGal", "nT", "Eotvos", "custom/unknown"], key="units")
        with c5:
            settings["grid_mode"] = st.selectbox("Data type", ["auto-detect", "regular grid", "scattered points"], key="grid_mode")
        with c6:
            settings["crs"] = st.text_input("CRS text (optional)", value=settings.get("crs", ""), key="crs_text")

        try:
            converted = data.copy()
            for col in [settings["x_col"], settings["y_col"], settings["value_col"]]:
                converted[col] = pd.to_numeric(converted[col], errors="coerce")
            report = data_quality_report(converted, settings["x_col"], settings["y_col"], settings["value_col"])
            st.subheader("Data quality")
            st.dataframe(pd.DataFrame([report]), hide_index=True, use_container_width=True)
            coord = coordinate_warning(converted[settings["x_col"]], converted[settings["y_col"]])
            if coord:
                st.warning(coord)
        except Exception as exc:
            st.error(f"Data quality check failed: {exc}")
    warn_science()


def panel_grid_settings() -> None:
    st.header("Grid Settings")
    section_intro(
        "Grid preparation",
        "Detect regular grids automatically, or interpolate scattered observations to an FFT-ready regular grid.",
    )
    data = st.session_state.raw_data
    if data is None:
        st.info("Load data first.")
        return
    settings = st.session_state.settings
    required = ["x_col", "y_col", "value_col"]
    if not all(key in settings for key in required):
        st.info("Choose columns in Project / Data Import first.")
        return

    converted = data.copy()
    for col in [settings["x_col"], settings["y_col"], settings["value_col"]]:
        converted[col] = pd.to_numeric(converted[col], errors="coerce")
    detection = detect_regular_grid(converted, settings["x_col"], settings["y_col"], settings["value_col"])
    st.subheader("Grid detection")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Regular grid", "Yes" if detection["is_regular"] else "No")
    d2.metric("Unique points", f"{detection['unique_point_count']:,}")
    d3.metric("Grid cells", f"{detection['nx']} x {detection['ny']}")
    d4.metric("Duplicate XY", f"{detection['duplicate_count']:,}")
    with st.expander("Detection details"):
        st.dataframe(pd.DataFrame([detection]), hide_index=True, use_container_width=True)

    spacing_default = float(np.nanmedian([detection.get("dx", np.nan), detection.get("dy", np.nan)]))
    if not np.isfinite(spacing_default) or spacing_default <= 0:
        spacing_default = 1000.0
    c1, c2, c3 = st.columns(3)
    with c1:
        spacing = st.number_input("Interpolation grid spacing (m)", min_value=0.0001, value=float(spacing_default), step=float(spacing_default))
    with c2:
        interp_method = st.selectbox("Interpolation method", ["linear", "cubic", "nearest"])
    with c3:
        fill_method = st.selectbox("NaN handling after interpolation", ["nearest", "constant", "reject"])

    if st.button("Create / update original grid", type="primary"):
        try:
            mode = settings.get("grid_mode", "auto-detect")
            use_regular = detection["is_regular"] if mode == "auto-detect" else mode == "regular grid"
            if use_regular:
                grid = reshape_regular_grid(
                    converted,
                    settings["x_col"],
                    settings["y_col"],
                    settings["value_col"],
                    name="Original anomaly",
                    units=settings.get("units", "unknown"),
                    crs=settings.get("crs") or None,
                )
            else:
                grid = interpolate_scattered_to_grid(
                    converted,
                    settings["x_col"],
                    settings["y_col"],
                    settings["value_col"],
                    spacing=spacing,
                    method=interp_method,
                    fill_method=fill_method,
                    name="Original anomaly",
                    units=settings.get("units", "unknown"),
                    crs=settings.get("crs") or None,
                )
            add_layer(grid)
            st.session_state.original_grid = grid
            add_history(
                st.session_state.history,
                make_history_entry(
                    method_name="Create grid",
                    input_layer=st.session_state.input_file_name or "session data",
                    output_layer=grid.name,
                    parameters={
                        "mode": mode,
                        "interpolation_method": interp_method if not use_regular else None,
                        "grid_spacing_m": spacing,
                        "fill_method": fill_method,
                    },
                    result=grid,
                    warnings=[coordinate_warning(grid.x, grid.y)] if coordinate_warning(grid.x, grid.y) else [],
                    formula="Regular reshape or scipy.interpolate.griddata",
                ),
            )
            st.success("Original grid is ready.")
        except Exception as exc:
            st.error(f"Could not create grid: {exc}")

    if "Original anomaly" in st.session_state.layers:
        show_grid_summary(st.session_state.layers["Original anomaly"], title="Original grid", export_key="original_grid")


def panel_original_map() -> None:
    st.header("Original Map")
    section_intro(
        "Original anomaly",
        "Review the gridded anomaly before applying filters. Confirm that coordinates are projected in meters.",
    )
    original = st.session_state.layers.get("Original anomaly")
    if original is None:
        st.info("Create the original grid first.")
        return
    show_grid_summary(original, title="Original anomaly map", export_key="original_map")


def panel_derivatives() -> None:
    st.header("Derivatives")
    section_intro(
        "Gradient products",
        "Compute horizontal and vertical derivatives. Vertical derivatives are FFT-based and can amplify high-frequency noise.",
    )
    grid = layer_selector("Input layer", key="derivatives_input")
    if grid is None:
        return
    padding = st.selectbox("FFT padding for vertical derivatives", ["none", "reflect", "zero"], index=0)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Compute THG", use_container_width=True):
            run_grid_method("THG", grid, lambda: compute_thg(grid), {"padding": None})
    with c2:
        if st.button("Compute FVD", use_container_width=True):
            run_grid_method("FVD", grid, lambda: compute_fvd(grid, padding=padding), {"padding": padding})
    with c3:
        if st.button("Compute SVD", use_container_width=True):
            run_grid_method("SVD", grid, lambda: compute_svd(grid, padding=padding), {"padding": padding})


def panel_tilt() -> None:
    st.header("Tilt Filters")
    section_intro(
        "Tilt derivative",
        "Balance shallow and deep anomaly edges using TILT = arctan(FVD / (THG + epsilon)).",
    )
    grid = layer_selector("Input layer", key="tilt_input")
    if grid is None:
        return
    c1, c2 = st.columns(2)
    with c1:
        epsilon = st.number_input("Epsilon", min_value=1e-15, value=1e-10, format="%.1e")
    with c2:
        padding = st.selectbox("FFT padding", ["none", "reflect", "zero"], key="tilt_padding")
    if st.button("Compute TILT / TDR", type="primary"):
        run_grid_method(
            "TILT",
            grid,
            lambda: compute_tilt_derivative(grid, epsilon=epsilon, padding=padding),
            {"epsilon": epsilon, "padding": padding},
        )
    result = st.session_state.current_result
    if isinstance(result, GridData) and result.name.endswith("_TILT"):
        st.caption("TILT is stored in radians. For visual inspection, degrees = radians * 180/pi.")


def panel_analytic_signal() -> None:
    st.header("Analytic Signal")
    section_intro(
        "Analytic Signal Amplitude",
        "Combine x, y, and vertical derivatives into a positive source-boundary enhancement map.",
    )
    grid = layer_selector("Input layer", key="asa_input")
    if grid is None:
        return
    padding = st.selectbox("FFT padding", ["none", "reflect", "zero"], key="asa_padding")
    if st.button("Compute Analytic Signal Amplitude", type="primary"):
        run_grid_method("ASA", grid, lambda: compute_analytic_signal(grid, padding=padding), {"padding": padding})


def panel_regional_residual() -> None:
    st.header("Regional-Residual")
    section_intro(
        "Regional-residual separation",
        "Separate long- and short-wavelength anomaly components using upward continuation or Butterworth filters.",
    )
    grid = layer_selector("Input layer", key="rr_input")
    if grid is None:
        return
    tabs = st.tabs(["Upward continuation", "Butterworth"])
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        with c1:
            height = st.number_input("Continuation height (m)", min_value=0.0, value=max(grid.dx, grid.dy) * 5.0, step=max(grid.dx, grid.dy))
        with c2:
            padding = st.selectbox("Padding", ["reflect", "none", "zero"], key="uc_padding")
        with c3:
            remove_mean = st.checkbox("Remove mean before FFT", value=False)
        if st.button("Compute upward regional and residual", type="primary"):
            try:
                start = time.perf_counter()
                regional, residual = compute_residual_from_upward(grid, height=height, padding=padding, remove_mean=remove_mean)
                elapsed = time.perf_counter() - start
                for result, name in [(regional, "Upward continuation"), (residual, "Residual from upward continuation")]:
                    store_result(
                        name,
                        grid,
                        result,
                        {"height_m": height, "padding": padding, "remove_mean": remove_mean},
                        elapsed,
                    )
            except Exception as exc:
                st.error(f"Upward continuation failed: {exc}")
    with tabs[1]:
        c1, c2, c3 = st.columns(3)
        with c1:
            filter_type = st.selectbox("Filter type", ["lowpass", "highpass", "bandpass", "regional/residual"])
        with c2:
            order = st.number_input("Butterworth order", min_value=1, max_value=12, value=4, step=1)
        with c3:
            padding = st.selectbox("Padding", ["reflect", "none", "zero"], key="bw_padding")
        if filter_type == "bandpass":
            c4, c5 = st.columns(2)
            with c4:
                long_wavelength = st.number_input("Long wavelength bound (m)", min_value=1.0, value=20000.0, step=1000.0)
            with c5:
                short_wavelength = st.number_input("Short wavelength bound (m)", min_value=1.0, value=5000.0, step=500.0)
        else:
            cutoff = st.number_input("Cutoff wavelength (m)", min_value=1.0, value=10000.0, step=1000.0)
        if st.button("Compute Butterworth", type="primary"):
            try:
                start = time.perf_counter()
                if filter_type == "regional/residual":
                    regional, residual = compute_butterworth_regional_residual(grid, cutoff_wavelength=cutoff, order=int(order), padding=padding)
                    elapsed = time.perf_counter() - start
                    for result, name in [(regional, "Butterworth regional"), (residual, "Butterworth residual")]:
                        store_result(name, grid, result, {"cutoff_wavelength_m": cutoff, "order": int(order), "padding": padding}, elapsed)
                else:
                    cutoff_arg = (long_wavelength, short_wavelength) if filter_type == "bandpass" else cutoff
                    result = compute_butterworth_filter(grid, cutoff=cutoff_arg, order=int(order), filter_type=filter_type, padding=padding)
                    elapsed = time.perf_counter() - start
                    store_result("Butterworth filter", grid, result, {"cutoff": cutoff_arg, "order": int(order), "filter_type": filter_type, "padding": padding}, elapsed)
            except Exception as exc:
                st.error(f"Butterworth filtering failed: {exc}")


def panel_fsed() -> None:
    st.header("Edge Detection / FSED")
    section_intro(
        "Fast Sigmoid Edge Detection",
        "Normalize a gradient-style base field and enhance edges with logistic or fast-sigmoid transforms.",
    )
    grid = layer_selector("Input layer", key="fsed_input")
    if grid is None:
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        base = st.selectbox("Base field", ["THG", "ASA", "absolute TILT", "direct input"])
    with c2:
        method = st.selectbox("FSED method", ["logistic", "fast_sigmoid"])
    with c3:
        gain = st.slider("Logistic gain", min_value=0.5, max_value=20.0, value=5.0, step=0.5)
    threshold = st.slider("Binary threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    if st.button("Compute FSED", type="primary"):
        try:
            start = time.perf_counter()
            result = compute_fsed(grid, base=base, gain=gain, method=method)
            elapsed = time.perf_counter() - start
            store_result("FSED", grid, result, {"base": base, "gain": gain, "method": method}, elapsed)
            binary = threshold_edges(result, threshold=threshold)
            store_result("FSED threshold", result, binary, {"threshold": threshold}, 0.0)
        except Exception as exc:
            st.error(f"FSED failed: {exc}")


def panel_raps() -> None:
    st.header("Spectral Analysis / RAPS")
    section_intro(
        "Radially averaged power spectrum",
        "Estimate source-depth trends from ln(power) versus angular radial wavenumber. Segment choice controls the result.",
    )
    grid = layer_selector("Input layer", key="raps_input")
    if grid is None:
        return
    st.warning("Small study areas such as 50 km x 50 km cannot reliably resolve very deep sources.")
    c1, c2, c3 = st.columns(3)
    with c1:
        remove_mean = st.checkbox("Remove mean", value=True)
        detrend = st.checkbox("Linear detrend", value=False)
    with c2:
        window = st.selectbox("Window", ["hanning", "cosine", "none"])
        padding = st.selectbox("Zero padding", ["zero", "none"], key="raps_padding")
    with c3:
        bin_size = st.number_input("Radial bin size (rad/m, 0=auto)", min_value=0.0, value=0.0, format="%.6g")
        fit_mode = st.selectbox("Segment fit", ["automatic", "manual"])

    manual_k_min = None
    manual_k_max = None
    if fit_mode == "manual":
        if st.session_state.raps_result is not None:
            k_values = st.session_state.raps_result.k
            finite_k = k_values[np.isfinite(k_values) & (k_values > 0)]
            default_min = float(np.quantile(finite_k, 0.2)) if finite_k.size else float(np.pi / max(grid.dx, grid.dy) * 0.1)
            default_max = float(np.quantile(finite_k, 0.6)) if finite_k.size else float(np.pi / min(grid.dx, grid.dy) * 0.6)
        else:
            default_min = float(np.pi / max(grid.dx, grid.dy) * 0.1)
            default_max = float(np.pi / min(grid.dx, grid.dy) * 0.6)
        c4, c5 = st.columns(2)
        with c4:
            manual_k_min = st.number_input("Manual k min (rad/m)", value=default_min, min_value=0.0, format="%.8g")
        with c5:
            manual_k_max = st.number_input("Manual k max (rad/m)", value=default_max, min_value=0.0, format="%.8g")

    if st.button("Compute RAPS", type="primary"):
        try:
            start = time.perf_counter()
            result = compute_raps(
                grid,
                remove_mean=remove_mean,
                detrend=detrend,
                window=window,
                padding=padding,
                radial_bin_size=bin_size or None,
            )
            if fit_mode == "automatic":
                fit = fit_raps_segment(result, automatic=True)
            elif manual_k_min is not None and manual_k_max is not None:
                fit = fit_raps_segment(result, k_min=manual_k_min, k_max=manual_k_max, automatic=False)
            else:
                fit = None
            elapsed = time.perf_counter() - start
            st.session_state.raps_result = result
            st.session_state.raps_summary = {
                "input_layer": grid.name,
                "spectrum_convention": result.metadata["spectrum_convention"],
                "wavenumber_unit": result.metadata["wavenumber_unit"],
                "depth_formula": result.metadata["depth_formula"],
                "slope": fit.slope if fit else None,
                "r_squared": fit.r_squared if fit else None,
                "estimated_depth_m": fit.depth_m if fit else None,
                "estimated_depth_km": fit.depth_km if fit else None,
            }
            add_history(
                st.session_state.history,
                make_history_entry(
                    method_name="RAPS",
                    input_layer=grid.name,
                    output_layer="RAPS spectrum",
                    parameters={"remove_mean": remove_mean, "detrend": detrend, "window": window, "padding": padding, "radial_bin_size": bin_size or "auto", "fit_mode": fit_mode},
                    result=grid,
                    warnings=[result.metadata["warning"]],
                    computation_time_s=elapsed,
                    formula=result.metadata["depth_formula"],
                ),
            )
            st.success("RAPS computed.")
        except Exception as exc:
            st.error(f"RAPS failed: {exc}")

    if st.session_state.raps_result is not None:
        result = st.session_state.raps_result
        st.plotly_chart(plot_raps(result), use_container_width=True)
        if result.fit is not None:
            st.dataframe(pd.DataFrame([st.session_state.raps_summary]), hide_index=True, use_container_width=True)
        st.caption("Convention: ln(radially averaged power) vs angular radial wavenumber k (rad/m). Depth = -slope / 2.")


def panel_export_report() -> None:
    st.header("Export and Report")
    section_intro(
        "Deliverables",
        "Export selected grids, figures, processing history, and a reproducible Markdown processing report.",
    )
    if not st.session_state.layers:
        st.info("No layers are available yet.")
        return
    grid = layer_selector("Layer to export", key="export_input")
    if grid is not None:
        show_grid_summary(grid, title=f"Export layer: {grid.name}", export_key="export_panel")

    history_json = export_history_json(st.session_state.history)
    st.download_button("Download processing history JSON", history_json, "geonarb_history.json", "application/json")
    report = export_markdown_report(
        project_name=f"{APP_NAME} V1",
        layers=st.session_state.layers,
        history=st.session_state.history,
        input_file=st.session_state.input_file_name,
        raps_summary=st.session_state.raps_summary,
    )
    st.download_button("Download processing report Markdown", report, "geonarb_processing_report.md", "text/markdown")
    st.subheader("Processing history")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)


def panel_future_modules() -> None:
    st.header("Future Modules")
    section_intro(
        "Roadmap",
        "These modules are planned for V2/V3 and are intentionally disabled in V1.",
    )
    for category, items in FUTURE_MODULES.items():
        with st.expander(category, expanded=True):
            for item in items:
                st.checkbox(item, value=False, disabled=True)


section = sidebar()
app_header(APP_NAME, section)

if section == "Project / Data Import":
    panel_project_import()
elif section == "Grid Settings":
    panel_grid_settings()
elif section == "Original Map":
    panel_original_map()
elif section == "Regional-Residual":
    panel_regional_residual()
elif section == "Derivatives":
    panel_derivatives()
elif section == "Tilt Filters":
    panel_tilt()
elif section == "Analytic Signal":
    panel_analytic_signal()
elif section == "Edge Detection / FSED":
    panel_fsed()
elif section == "Spectral Analysis / RAPS":
    panel_raps()
elif section == "Export and Report":
    panel_export_report()
elif section == "Future Modules":
    panel_future_modules()
