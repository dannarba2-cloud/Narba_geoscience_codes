"""Plotting helpers for GEONARBA."""

from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from .grid_tools import GridData
from .spectral import RAPSResult


def plot_grid_plotly(grid: GridData, title: str | None = None, contour: bool = False) -> go.Figure:
    x = grid.x_vector
    y = grid.y_vector
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=x,
            y=y,
            z=grid.values,
            colorscale="Viridis",
            colorbar={"title": grid.units},
            hovertemplate="x=%{x}<br>y=%{y}<br>value=%{z:.4g}<extra></extra>",
        )
    )
    if contour:
        fig.add_trace(
            go.Contour(
                x=x,
                y=y,
                z=grid.values,
                contours={"coloring": "lines"},
                line={"width": 1, "color": "black"},
                showscale=False,
            )
        )
    fig.update_layout(
        title=title or grid.name,
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        margin={"l": 30, "r": 20, "t": 50, "b": 35},
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def plot_histogram(grid: GridData) -> go.Figure:
    finite = grid.values[np.isfinite(grid.values)]
    fig = go.Figure(
        go.Histogram(
            x=finite,
            nbinsx=40,
            marker_color="#3b82f6",
            hovertemplate="value=%{x:.4g}<br>count=%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Histogram: {grid.name}",
        xaxis_title=grid.units,
        yaxis_title="Count",
        margin={"l": 30, "r": 20, "t": 50, "b": 35},
    )
    return fig


def plot_raps(result: RAPSResult) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.k,
            y=result.log_power,
            mode="lines+markers",
            name="RAPS",
            hovertemplate="k=%{x:.4g} rad/m<br>ln(power)=%{y:.4g}<extra></extra>",
        )
    )
    if result.fit is not None:
        mask = (result.k >= result.fit.k_min) & (result.k <= result.fit.k_max)
        fitted = result.fit.slope * result.k[mask] + result.fit.intercept
        fig.add_trace(
            go.Scatter(
                x=result.k[mask],
                y=fitted,
                mode="lines",
                name=f"Fit depth={result.fit.depth_km:.2f} km",
                line={"dash": "dash", "color": "#ef4444"},
            )
        )
    fig.update_layout(
        title="RAPS: ln(power) vs angular radial wavenumber",
        xaxis_title="Radial angular wavenumber k (rad/m)",
        yaxis_title="ln(radial average power)",
        margin={"l": 30, "r": 20, "t": 50, "b": 35},
    )
    return fig


def create_png_figure(grid: GridData, title: str | None = None, dpi: int = 200) -> bytes:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = ax.imshow(
        grid.values,
        extent=[
            float(np.min(grid.x_vector)),
            float(np.max(grid.x_vector)),
            float(np.min(grid.y_vector)),
            float(np.max(grid.y_vector)),
        ],
        origin="lower",
        cmap="viridis",
        aspect="equal",
    )
    ax.set_title(title or grid.name)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    fig.colorbar(image, ax=ax, label=grid.units)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    plt.close(fig)
    return buffer.getvalue()
