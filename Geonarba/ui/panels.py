"""Reusable Streamlit panels and state helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from core.exports import export_grid_csv, export_png
from core.grid_tools import GridData, coordinate_warning, grid_extent, grid_statistics
from core.visualization import plot_grid_plotly, plot_histogram
from ui.theme import badge_row


def initialize_session_state() -> None:
    defaults: dict[str, Any] = {
        "raw_data": None,
        "input_file_name": None,
        "layers": {},
        "current_result": None,
        "history": [],
        "settings": {},
        "raps_result": None,
        "raps_summary": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def layer_names() -> list[str]:
    return list(st.session_state.layers.keys())


def layer_selector(label: str, key: str) -> GridData | None:
    names = layer_names()
    if not names:
        st.info("Create or load a grid first.")
        return None
    selected = st.selectbox(label, names, key=key)
    return st.session_state.layers[selected]


def add_layer(grid: GridData) -> None:
    st.session_state.layers[grid.name] = grid
    st.session_state.current_result = grid


def stats_table(grid: GridData) -> pd.DataFrame:
    stats = grid_statistics(grid)
    return pd.DataFrame(
        [
            {"Metric": "Minimum", "Value": stats["min"]},
            {"Metric": "Maximum", "Value": stats["max"]},
            {"Metric": "Mean", "Value": stats["mean"]},
            {"Metric": "Standard deviation", "Value": stats["std"]},
        ]
    )


def show_grid_summary(grid: GridData, title: str | None = None, export_key: str = "export") -> None:
    st.subheader(title or grid.name)
    warning = coordinate_warning(grid.x, grid.y)
    if warning:
        st.warning(warning)
    extent = grid_extent(grid)
    badge_row(
        [
            f"{grid.shape[1]} x {grid.shape[0]} cells",
            f"dx {grid.dx:g} m",
            f"dy {grid.dy:g} m",
            f"units {grid.units}",
            f"width {extent['width'] / 1000:g} km",
            f"height {extent['height'] / 1000:g} km",
        ]
    )
    stats = grid_statistics(grid)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Minimum", f"{stats['min']:.6g}")
    m2.metric("Maximum", f"{stats['max']:.6g}")
    m3.metric("Mean", f"{stats['mean']:.6g}")
    m4.metric("Std. deviation", f"{stats['std']:.6g}")
    map_col, hist_col = st.columns([2, 1])
    with map_col:
        st.plotly_chart(plot_grid_plotly(grid), use_container_width=True)
    with hist_col:
        st.plotly_chart(plot_histogram(grid), use_container_width=True)
    export_grid_controls(grid, key_prefix=export_key)


def export_grid_controls(grid: GridData, key_prefix: str) -> None:
    csv_text = export_grid_csv(grid)
    png_bytes = export_png(grid)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download grid CSV",
            data=csv_text,
            file_name=f"{safe_file_name(grid.name)}.csv",
            mime="text/csv",
            key=f"{key_prefix}_csv_{grid.name}",
        )
    with col2:
        st.download_button(
            "Download figure PNG",
            data=png_bytes,
            file_name=f"{safe_file_name(grid.name)}.png",
            mime="image/png",
            key=f"{key_prefix}_png_{grid.name}",
        )


def safe_file_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name).strip("_")
