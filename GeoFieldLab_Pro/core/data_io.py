"""Input helpers for CSV/TXT/XYZ anomaly data."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".txt", ".xyz", ".dat"}


def read_table(file: str | Path | BinaryIO, filename: str | None = None) -> pd.DataFrame:
    """Read CSV/TXT/XYZ data with automatic delimiter detection."""

    suffix = Path(filename or getattr(file, "name", "")).suffix.lower()
    if suffix and suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension '{suffix}'. Use CSV, TXT, XYZ, or DAT.")
    try:
        data = pd.read_csv(file, sep=None, engine="python", comment="#")
    except Exception as exc:
        raise ValueError(f"Could not read table: {exc}") from exc
    if data.empty:
        raise ValueError("Input file is empty.")
    data.columns = [str(col).replace("\ufeff", "").strip() for col in data.columns]
    return data


def validate_numeric_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric values and report invalid rows."""

    if not columns:
        raise ValueError("No columns were selected.")
    missing = [col for col in columns if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    converted = data.copy()
    for col in columns:
        converted[col] = pd.to_numeric(converted[col], errors="coerce")
    return converted


def data_quality_report(data: pd.DataFrame, x_col: str, y_col: str, value_col: str) -> dict[str, int]:
    selected = data[[x_col, y_col, value_col]]
    return {
        "rows": int(len(data)),
        "missing_selected_values": int(selected.isna().sum().sum()),
        "duplicate_coordinates": int(selected.duplicated(subset=[x_col, y_col]).sum()),
        "valid_rows": int(selected.dropna().shape[0]),
    }


def load_example_dataset(path: str | Path) -> pd.DataFrame:
    return read_table(Path(path))
