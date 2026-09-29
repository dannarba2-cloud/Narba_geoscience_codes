"""Export helpers for grids, figures, history, and reports."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .grid_tools import GridData, coordinate_warning, grid_extent, grid_statistics
from .visualization import create_png_figure


def export_grid_csv(grid: GridData, path: str | Path | None = None) -> str:
    csv_text = grid.to_dataframe(value_column="value").to_csv(index=False)
    if path is not None:
        Path(path).write_text(csv_text, encoding="utf-8")
    return csv_text


def export_png(grid: GridData, path: str | Path | None = None, dpi: int = 200) -> bytes:
    png = create_png_figure(grid, dpi=dpi)
    if path is not None:
        Path(path).write_bytes(png)
    return png


def export_history_json(history: list[dict[str, Any]], path: str | Path | None = None) -> str:
    text = json.dumps(history, indent=2, default=str)
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def export_markdown_report(
    project_name: str,
    layers: dict[str, GridData],
    history: list[dict[str, Any]],
    input_file: str | None = None,
    raps_summary: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> str:
    original = next(iter(layers.values())) if layers else None
    coord_warning = coordinate_warning(original.x, original.y) if original is not None else None
    lines = [
        f"# {project_name} Processing Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Input file: {input_file or 'Example/session data'}",
        "",
    ]
    if original is not None:
        extent = grid_extent(original)
        lines.extend(
            [
                "## Grid Summary",
                "",
                f"- Grid spacing: dx={original.dx:g} m, dy={original.dy:g} m",
                f"- Extent: {extent['xmin']:g} to {extent['xmax']:g} m X, {extent['ymin']:g} to {extent['ymax']:g} m Y",
                f"- CRS: {original.crs or 'not specified'}",
                "",
            ]
        )
    if coord_warning:
        lines.extend(["## Coordinate Warning", "", coord_warning, ""])
    lines.extend(["## Layers", ""])
    for name, layer in layers.items():
        stats = grid_statistics(layer)
        lines.append(
            f"- {name}: min={stats['min']:.6g}, max={stats['max']:.6g}, "
            f"mean={stats['mean']:.6g}, std={stats['std']:.6g}, units={layer.units}"
        )
    lines.extend(["", "## Processing History", ""])
    for idx, entry in enumerate(history, start=1):
        lines.extend(
            [
                f"### Step {idx}: {entry.get('method_name', 'unknown')}",
                "",
                f"- Input: {entry.get('input_layer')}",
                f"- Output: {entry.get('output_layer')}",
                f"- Parameters: `{json.dumps(entry.get('parameters', {}), default=str)}`",
                f"- Formula: {entry.get('formula') or 'not recorded'}",
                f"- Warnings: {'; '.join(entry.get('warnings') or []) or 'none'}",
                "",
            ]
        )
    if raps_summary:
        lines.extend(["## RAPS Depth Estimate", ""])
        for key, value in raps_summary.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.extend(
        [
            "## Scientific Warnings",
            "",
            "- FFT derivatives, upward continuation, RAPS, and wavelength filters require regular projected coordinates in meters.",
            "- FVD and especially SVD amplify shallow noise.",
            "- Regional/residual separation is parameter-dependent and non-unique.",
            "- Small areas such as 50 km x 50 km cannot reliably resolve very deep sources.",
            "- Edge filters show mathematical gradients; geological validation is required.",
            "",
        ]
    )
    report = "\n".join(lines)
    if path is not None:
        Path(path).write_text(report, encoding="utf-8")
    return report
