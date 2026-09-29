"""Safe launcher for the GeoFieldLab Pro PySide6 desktop app."""

from __future__ import annotations

import importlib.util


QT_AVAILABLE = importlib.util.find_spec("PySide6") is not None


def main() -> int:
    """Start the desktop GUI, loading Qt only when the app is launched."""

    try:
        from .qt_app import main as qt_main
    except Exception as exc:
        print("GeoFieldLab Pro Desktop could not initialize the Qt runtime.")
        print(f"Reason: {exc}")
        print("Try reinstalling the desktop dependencies with: pip install -r requirements.txt")
        return 1
    return qt_main()
