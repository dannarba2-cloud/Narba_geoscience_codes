import json
import os
from pathlib import Path

import numpy as np
import pytest

from desktop.controller import ProjectController
from desktop.main import QT_AVAILABLE


def test_desktop_main_module_is_importable_without_launching_gui():
    assert isinstance(QT_AVAILABLE, bool)


def test_qt_main_window_constructs_offscreen():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from desktop import qt_app

    if not qt_app.QT_AVAILABLE:
        pytest.skip(f"Qt runtime unavailable: {qt_app.QT_IMPORT_ERROR}")
    app = qt_app.QApplication.instance() or qt_app.QApplication([])
    window = qt_app.MainWindow(ProjectController())
    assert window.windowTitle() == "GEONARBA - GeoFieldLab Pro Desktop"
    menus = [a.text().replace("&&", "\0").replace("&", "").replace("\0", "&") for a in window.menuBar().actions()]
    assert {"File", "Data", "Potential Fields", "Seismic", "GIS & Spatial Analysis", "Help"} <= set(menus)
    window.close()
    app.processEvents()


def test_controller_acceptance_workflow_and_project_roundtrip(tmp_path: Path):
    controller = ProjectController()
    data = controller.load_example_dataset()
    assert not data.empty

    grid = controller.create_grid("x", "y", "anomaly", units="mGal", mode="auto-detect")
    original_values = grid.values.copy()

    thg = controller.compute_thg(grid.name)
    fvd = controller.compute_fvd(grid.name)
    svd = controller.compute_svd(grid.name)
    tilt = controller.compute_tilt(grid.name)
    asa = controller.compute_analytic_signal(grid.name)
    regional, residual = controller.compute_upward_residual(grid.name, height=5000)
    fsed, threshold = controller.compute_fsed(grid.name, threshold=0.5)
    raps = controller.compute_raps(grid.name)
    profile = controller.extract_cross_section(grid.name, (0, 0), (50000, 50000), spacing=1000)

    for layer in [thg, fvd, svd, tilt, asa, regional, residual, fsed, threshold]:
        assert layer.values.shape == grid.values.shape
        assert np.isfinite(layer.values).all()
    assert np.array_equal(controller.session.layers[grid.name].values, original_values)
    assert np.isfinite(raps["estimated_depth_m"])
    assert profile.values.size > 2

    csv_path = tmp_path / "layer.csv"
    png_path = tmp_path / "layer.png"
    history_path = tmp_path / "history.json"
    report_path = tmp_path / "report.md"
    profile_path = tmp_path / "profile.csv"
    project_path = tmp_path / "project.gflp"

    controller.export_layer_csv(thg.name, csv_path)
    controller.export_layer_png(thg.name, png_path)
    controller.export_history_json(history_path)
    controller.export_markdown_report(report_path)
    controller.export_profile_csv(profile.name, profile_path)
    controller.save_project(project_path)

    assert csv_path.read_text(encoding="utf-8").splitlines()[0] == "x,y,value,layer_name,units"
    assert png_path.read_bytes().startswith(b"\x89PNG")
    assert json.loads(history_path.read_text(encoding="utf-8"))
    assert "# Untitled GeoFieldLab Project Processing Report" in report_path.read_text(encoding="utf-8")
    assert profile_path.read_text(encoding="utf-8").splitlines()[0] == "distance,value,profile_name,units"

    restored = ProjectController()
    restored.load_project(project_path)
    assert list(restored.session.layers) == list(controller.session.layers)
    assert list(restored.session.profiles) == list(controller.session.profiles)
    assert len(restored.session.history) == len(controller.session.history)
