"""PySide6 desktop shell for GeoFieldLab Pro."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

QT_IMPORT_ERROR: Exception | None = None

try:  # pragma: no cover - exercised by desktop smoke tests when Qt exists.
    import PySide6
    import shiboken6

    pyside_dir = Path(PySide6.__file__).resolve().parent
    shiboken_dir = Path(shiboken6.__file__).resolve().parent
    os.environ["PATH"] = f"{pyside_dir}{os.pathsep}{shiboken_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(pyside_dir))
        os.add_dll_directory(str(shiboken_dir))

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QSplitter,
        QStackedWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    QT_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on local GUI deps.
    QT_IMPORT_ERROR = exc
    QT_AVAILABLE = False

from core.grid_tools import grid_statistics
from core.profile import profile_statistics
from desktop.controller import ProjectController


if QT_AVAILABLE:

    class MapCanvas(FigureCanvas):
        """Matplotlib canvas for grid and profile previews."""

        def __init__(self) -> None:
            self.figure = Figure(figsize=(7, 5), constrained_layout=True)
            super().__init__(self.figure)

        def plot_grid(self, grid, mode: str = "image") -> None:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            x = grid.x_vector
            y = grid.y_vector
            if mode == "contour":
                artist = ax.contour(x, y, grid.values, 20)
                ax.clabel(artist, inline=True, fontsize=7)
            elif mode == "filled contour":
                artist = ax.contourf(x, y, grid.values, 20, cmap="viridis")
                self.figure.colorbar(artist, ax=ax, label=grid.units)
            elif mode == "pseudocolor":
                artist = ax.pcolormesh(x, y, grid.values, shading="auto", cmap="viridis")
                self.figure.colorbar(artist, ax=ax, label=grid.units)
            else:
                artist = ax.imshow(
                    grid.values,
                    extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
                    origin="lower",
                    cmap="viridis",
                    aspect="equal",
                )
                self.figure.colorbar(artist, ax=ax, label=grid.units)
            ax.set_title(grid.name)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_aspect("equal", adjustable="box")
            self.draw_idle()

        def plot_profile(self, profile) -> None:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(profile.distance, profile.values, "-o", linewidth=1.2, markersize=3)
            ax.set_title(profile.name)
            ax.set_xlabel("Distance (m)")
            ax.set_ylabel(profile.units)
            ax.grid(True, alpha=0.25)
            self.draw_idle()


    class MainWindow(QMainWindow):
        """Main desktop window."""

        NAV_ITEMS = [
            "Project",
            "Data",
            "Map",
            "Profile",
            "Regional/Residual",
            "Derivatives",
            "Filters",
            "Edges",
            "Spectrum",
            "Export",
        ]

        def __init__(self, controller: ProjectController | None = None) -> None:
            super().__init__()
            self.controller = controller or ProjectController()
            self.setWindowTitle("GeoFieldLab Pro Desktop V1")
            self.resize(1320, 820)
            self.statusBar().showMessage("Ready")
            self._build_menu()
            self._build_layout()
            self._refresh_all()

        def _build_menu(self) -> None:
            file_menu = self.menuBar().addMenu("File")
            file_menu.addAction("New Project", self._new_project)
            file_menu.addAction("Open Project", self._open_project)
            file_menu.addAction("Save Project", self._save_project)
            file_menu.addSeparator()
            file_menu.addAction("Load Example Dataset", self._load_example)
            file_menu.addAction("Import Table", self._import_table)
            file_menu.addSeparator()
            file_menu.addAction("Export Report", self._export_report)
            file_menu.addAction("Exit", self.close)

        def _build_layout(self) -> None:
            splitter = QSplitter(Qt.Horizontal)
            splitter.addWidget(self._build_sidebar())
            self.stack = QStackedWidget()
            self.pages = {
                "Project": self._build_project_page(),
                "Data": self._build_data_page(),
                "Map": self._build_map_page(),
                "Profile": self._build_profile_page(),
                "Regional/Residual": self._build_regional_page(),
                "Derivatives": self._build_derivatives_page(),
                "Filters": self._build_filters_page(),
                "Edges": self._build_edges_page(),
                "Spectrum": self._build_spectrum_page(),
                "Export": self._build_export_page(),
            }
            for name in self.NAV_ITEMS:
                self.stack.addWidget(self.pages[name])
            splitter.addWidget(self.stack)
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            self.setCentralWidget(splitter)

        def _build_sidebar(self) -> QWidget:
            frame = QFrame()
            frame.setMinimumWidth(260)
            layout = QVBoxLayout(frame)
            title = QLabel("GEONARBA")
            title.setStyleSheet("font-size: 22px; font-weight: 700;")
            subtitle = QLabel("Professional potential-field workspace")
            subtitle.setWordWrap(True)
            layout.addWidget(title)
            layout.addWidget(subtitle)

            self.nav = QListWidget()
            self.nav.addItems(self.NAV_ITEMS)
            self.nav.setCurrentRow(0)
            self.nav.currentRowChanged.connect(self.stack_set_index)
            layout.addWidget(self.nav)

            layout.addWidget(QLabel("Layers"))
            self.layer_list = QListWidget()
            self.layer_list.currentRowChanged.connect(self._plot_selected_layer)
            layout.addWidget(self.layer_list, stretch=1)

            layout.addWidget(QLabel("Profiles"))
            self.profile_list = QListWidget()
            self.profile_list.currentRowChanged.connect(self._plot_selected_profile)
            layout.addWidget(self.profile_list, stretch=1)

            self.session_label = QLabel()
            self.session_label.setWordWrap(True)
            layout.addWidget(self.session_label)
            return frame

        def stack_set_index(self, index: int) -> None:
            self.stack.setCurrentIndex(index)

        def _page(self, title: str) -> tuple[QWidget, QVBoxLayout]:
            widget = QWidget()
            layout = QVBoxLayout(widget)
            heading = QLabel(title)
            heading.setStyleSheet("font-size: 20px; font-weight: 700;")
            layout.addWidget(heading)
            return widget, layout

        def _build_project_page(self) -> QWidget:
            page, layout = self._page("Project")
            form = QFormLayout()
            self.project_name = QLineEdit(self.controller.session.project_name)
            form.addRow("Project name", self.project_name)
            self.input_file_label = QLabel("No input loaded")
            self.input_file_label.setWordWrap(True)
            form.addRow("Input file", self.input_file_label)
            layout.addLayout(form)
            buttons = QHBoxLayout()
            buttons.addWidget(self._button("Load Example", self._load_example))
            buttons.addWidget(self._button("Import Table", self._import_table))
            buttons.addWidget(self._button("Open Project", self._open_project))
            buttons.addWidget(self._button("Save Project", self._save_project))
            layout.addLayout(buttons)
            self.project_summary = QTextEdit()
            self.project_summary.setReadOnly(True)
            layout.addWidget(self.project_summary, stretch=1)
            return page

        def _build_data_page(self) -> QWidget:
            page, layout = self._page("Data")
            form = QFormLayout()
            self.x_col = QComboBox()
            self.y_col = QComboBox()
            self.value_col = QComboBox()
            self.units = QComboBox()
            self.units.addItems(["mGal", "nT", "Eotvos", "custom/unknown"])
            self.crs = QLineEdit()
            self.grid_mode = QComboBox()
            self.grid_mode.addItems(["auto-detect", "regular grid", "scattered points"])
            self.spacing = QDoubleSpinBox()
            self.spacing.setRange(0.0001, 1_000_000_000.0)
            self.spacing.setValue(1000.0)
            self.spacing.setDecimals(4)
            self.interp_method = QComboBox()
            self.interp_method.addItems(["linear", "cubic", "nearest"])
            self.fill_method = QComboBox()
            self.fill_method.addItems(["nearest", "constant", "reject"])
            for label, widget in [
                ("X column", self.x_col),
                ("Y column", self.y_col),
                ("Anomaly column", self.value_col),
                ("Units", self.units),
                ("CRS text", self.crs),
                ("Grid mode", self.grid_mode),
                ("Interpolation spacing", self.spacing),
                ("Interpolation method", self.interp_method),
                ("NaN handling", self.fill_method),
            ]:
                form.addRow(label, widget)
            layout.addLayout(form)
            actions = QHBoxLayout()
            actions.addWidget(self._button("Detect Grid", self._detect_grid))
            actions.addWidget(self._button("Create / Update Original Grid", self._create_grid))
            layout.addLayout(actions)
            self.data_output = QTextEdit()
            self.data_output.setReadOnly(True)
            layout.addWidget(self.data_output, stretch=1)
            return page

        def _build_map_page(self) -> QWidget:
            page, layout = self._page("Map")
            toolbar = QHBoxLayout()
            self.map_mode = QComboBox()
            self.map_mode.addItems(["image", "pseudocolor", "contour", "filled contour"])
            toolbar.addWidget(QLabel("Display mode"))
            toolbar.addWidget(self.map_mode)
            toolbar.addWidget(self._button("Refresh Map", self._plot_selected_layer))
            layout.addLayout(toolbar)
            self.map_canvas = MapCanvas()
            layout.addWidget(self.map_canvas, stretch=1)
            self.map_stats = QTextEdit()
            self.map_stats.setReadOnly(True)
            self.map_stats.setMaximumHeight(120)
            layout.addWidget(self.map_stats)
            return page

        def _build_profile_page(self) -> QWidget:
            page, layout = self._page("Profile")
            grid = QGridLayout()
            self.start_x = self._double_box(0.0)
            self.start_y = self._double_box(0.0)
            self.end_x = self._double_box(50000.0)
            self.end_y = self._double_box(50000.0)
            self.section_spacing = self._double_box(1000.0)
            labels = [
                ("Start X", self.start_x),
                ("Start Y", self.start_y),
                ("End X", self.end_x),
                ("End Y", self.end_y),
                ("Spacing", self.section_spacing),
            ]
            for idx, (label, widget) in enumerate(labels):
                grid.addWidget(QLabel(label), idx // 3, (idx % 3) * 2)
                grid.addWidget(widget, idx // 3, (idx % 3) * 2 + 1)
            layout.addLayout(grid)
            actions = QHBoxLayout()
            actions.addWidget(self._button("Extract Cross-section", self._extract_cross_section))
            self.profile_window = QSpinBox()
            self.profile_window.setRange(1, 999)
            self.profile_window.setValue(5)
            actions.addWidget(QLabel("Average window"))
            actions.addWidget(self.profile_window)
            actions.addWidget(self._button("Running Average", self._profile_running_average))
            actions.addWidget(self._button("Remove Linear Trend", self._profile_remove_trend))
            layout.addLayout(actions)
            self.profile_canvas = MapCanvas()
            layout.addWidget(self.profile_canvas, stretch=1)
            self.profile_stats = QTextEdit()
            self.profile_stats.setReadOnly(True)
            self.profile_stats.setMaximumHeight(110)
            layout.addWidget(self.profile_stats)
            return page

        def _build_regional_page(self) -> QWidget:
            page, layout = self._page("Regional / Residual")
            form = QFormLayout()
            self.uc_height = self._double_box(5000.0)
            self.uc_padding = QComboBox()
            self.uc_padding.addItems(["reflect", "none", "zero"])
            self.uc_remove_mean = QCheckBox()
            self.bw_cutoff = self._double_box(10000.0)
            self.bw_order = QSpinBox()
            self.bw_order.setRange(1, 12)
            self.bw_order.setValue(4)
            self.bw_padding = QComboBox()
            self.bw_padding.addItems(["reflect", "none", "zero"])
            form.addRow("Continuation height", self.uc_height)
            form.addRow("Continuation padding", self.uc_padding)
            form.addRow("Remove mean", self.uc_remove_mean)
            form.addRow("Butterworth cutoff wavelength", self.bw_cutoff)
            form.addRow("Butterworth order", self.bw_order)
            form.addRow("Butterworth padding", self.bw_padding)
            layout.addLayout(form)
            actions = QHBoxLayout()
            actions.addWidget(self._button("Upward Regional + Residual", self._compute_upward))
            actions.addWidget(self._button("Butterworth Regional + Residual", self._compute_butterworth_rr))
            layout.addLayout(actions)
            layout.addStretch()
            return page

        def _build_derivatives_page(self) -> QWidget:
            page, layout = self._page("Derivatives")
            self.derivative_padding = QComboBox()
            self.derivative_padding.addItems(["none", "reflect", "zero"])
            layout.addWidget(QLabel("Vertical derivative FFT padding"))
            layout.addWidget(self.derivative_padding)
            actions = QHBoxLayout()
            actions.addWidget(self._button("THG", lambda: self._compute_grid_method(self.controller.compute_thg)))
            actions.addWidget(self._button("FVD", self._compute_fvd))
            actions.addWidget(self._button("SVD", self._compute_svd))
            actions.addWidget(self._button("TILT", self._compute_tilt))
            actions.addWidget(self._button("ASA", self._compute_asa))
            layout.addLayout(actions)
            layout.addStretch()
            return page

        def _build_filters_page(self) -> QWidget:
            page, layout = self._page("Filters and Grid Tools")
            form = QFormLayout()
            self.filter_type = QComboBox()
            self.filter_type.addItems(["lowpass", "highpass", "bandpass"])
            self.filter_cutoff = self._double_box(10000.0)
            self.filter_cutoff2 = self._double_box(5000.0)
            self.filter_order = QSpinBox()
            self.filter_order.setRange(1, 12)
            self.filter_order.setValue(4)
            form.addRow("Butterworth type", self.filter_type)
            form.addRow("Cutoff / long wavelength", self.filter_cutoff)
            form.addRow("Short wavelength for band-pass", self.filter_cutoff2)
            form.addRow("Order", self.filter_order)
            layout.addLayout(form)
            actions = QHBoxLayout()
            actions.addWidget(self._button("Run Butterworth", self._compute_butterworth))
            actions.addWidget(self._button("Flip X", lambda: self._grid_transform("flip_x")))
            actions.addWidget(self._button("Flip Y", lambda: self._grid_transform("flip_y")))
            actions.addWidget(self._button("Rotate 90", lambda: self._grid_transform("rotate_90")))
            layout.addLayout(actions)
            layout.addStretch()
            return page

        def _build_edges_page(self) -> QWidget:
            page, layout = self._page("Edges")
            form = QFormLayout()
            self.fsed_base = QComboBox()
            self.fsed_base.addItems(["THG", "ASA", "absolute TILT", "direct input"])
            self.fsed_method = QComboBox()
            self.fsed_method.addItems(["logistic", "fast_sigmoid"])
            self.fsed_gain = self._double_box(5.0)
            self.fsed_threshold = self._double_box(0.5, maximum=1.0)
            form.addRow("Base field", self.fsed_base)
            form.addRow("Method", self.fsed_method)
            form.addRow("Gain", self.fsed_gain)
            form.addRow("Threshold", self.fsed_threshold)
            layout.addLayout(form)
            layout.addWidget(self._button("Compute FSED + Threshold", self._compute_fsed))
            layout.addStretch()
            return page

        def _build_spectrum_page(self) -> QWidget:
            page, layout = self._page("Spectrum")
            form = QFormLayout()
            self.raps_remove_mean = QCheckBox()
            self.raps_remove_mean.setChecked(True)
            self.raps_detrend = QCheckBox()
            self.raps_window = QComboBox()
            self.raps_window.addItems(["hanning", "cosine", "none"])
            self.raps_padding = QComboBox()
            self.raps_padding.addItems(["zero", "none"])
            form.addRow("Remove mean", self.raps_remove_mean)
            form.addRow("Linear detrend", self.raps_detrend)
            form.addRow("Window", self.raps_window)
            form.addRow("Padding", self.raps_padding)
            layout.addLayout(form)
            layout.addWidget(self._button("Compute RAPS Depth Estimate", self._compute_raps))
            self.raps_output = QTextEdit()
            self.raps_output.setReadOnly(True)
            layout.addWidget(self.raps_output, stretch=1)
            return page

        def _build_export_page(self) -> QWidget:
            page, layout = self._page("Export")
            actions = QGridLayout()
            buttons = [
                ("Export Layer CSV", self._export_layer_csv),
                ("Export Layer PNG", self._export_layer_png),
                ("Export Profile CSV", self._export_profile_csv),
                ("Export History JSON", self._export_history),
                ("Export Markdown Report", self._export_report),
                ("Save Project State", self._save_project),
            ]
            for idx, (label, callback) in enumerate(buttons):
                actions.addWidget(self._button(label, callback), idx // 2, idx % 2)
            layout.addLayout(actions)
            self.export_output = QTextEdit()
            self.export_output.setReadOnly(True)
            layout.addWidget(self.export_output, stretch=1)
            return page

        def _button(self, text: str, callback: Callable[[], object]) -> QPushButton:
            button = QPushButton(text)
            button.clicked.connect(callback)
            return button

        def _double_box(self, value: float, maximum: float = 1_000_000_000.0) -> QDoubleSpinBox:
            widget = QDoubleSpinBox()
            widget.setRange(-maximum, maximum)
            widget.setDecimals(6)
            widget.setValue(value)
            widget.setSingleStep(max(abs(value) * 0.1, 1.0))
            return widget

        def _selected_layer_name(self) -> str | None:
            item = self.layer_list.currentItem()
            return item.text() if item is not None else None

        def _selected_profile_name(self) -> str | None:
            item = self.profile_list.currentItem()
            return item.text() if item is not None else None

        def _run(self, label: str, callback: Callable[[], object]) -> object | None:
            try:
                result = callback()
                self.statusBar().showMessage(f"{label} complete")
                self._refresh_all()
                return result
            except Exception as exc:
                QMessageBox.critical(self, label, str(exc))
                self.statusBar().showMessage(f"{label} failed")
                return None

        def _refresh_all(self) -> None:
            self.project_name.setText(self.controller.session.project_name)
            self.input_file_label.setText(self.controller.session.input_file_name or "No input loaded")
            self.session_label.setText(
                f"Layers: {len(self.controller.session.layers)} | "
                f"Profiles: {len(self.controller.session.profiles)} | "
                f"History: {len(self.controller.session.history)}"
            )
            self._refresh_columns()
            self._refresh_layers()
            self._refresh_profiles()
            self._refresh_project_summary()

        def _refresh_columns(self) -> None:
            columns = []
            if self.controller.session.raw_data is not None:
                columns = [str(column) for column in self.controller.session.raw_data.columns]
            for combo in [self.x_col, self.y_col, self.value_col]:
                current = combo.currentText()
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(columns)
                if current in columns:
                    combo.setCurrentText(current)
                combo.blockSignals(False)
            if len(columns) >= 3:
                self.x_col.setCurrentIndex(0)
                self.y_col.setCurrentIndex(1)
                self.value_col.setCurrentIndex(2)

        def _refresh_layers(self) -> None:
            selected = self._selected_layer_name()
            self.layer_list.blockSignals(True)
            self.layer_list.clear()
            self.layer_list.addItems(list(self.controller.session.layers))
            if selected in self.controller.session.layers:
                self.layer_list.setCurrentRow(list(self.controller.session.layers).index(selected))
            elif self.controller.session.layers:
                self.layer_list.setCurrentRow(len(self.controller.session.layers) - 1)
            self.layer_list.blockSignals(False)
            self._plot_selected_layer()

        def _refresh_profiles(self) -> None:
            selected = self._selected_profile_name()
            self.profile_list.blockSignals(True)
            self.profile_list.clear()
            self.profile_list.addItems(list(self.controller.session.profiles))
            if selected in self.controller.session.profiles:
                self.profile_list.setCurrentRow(list(self.controller.session.profiles).index(selected))
            elif self.controller.session.profiles:
                self.profile_list.setCurrentRow(len(self.controller.session.profiles) - 1)
            self.profile_list.blockSignals(False)
            self._plot_selected_profile()

        def _refresh_project_summary(self) -> None:
            session = self.controller.session
            lines = [
                f"Project: {session.project_name}",
                f"Input: {session.input_file_name or 'none'}",
                f"Raw rows: {0 if session.raw_data is None else len(session.raw_data)}",
                f"Layers: {', '.join(session.layers) or 'none'}",
                f"Profiles: {', '.join(session.profiles) or 'none'}",
                f"Dirty: {session.dirty}",
            ]
            self.project_summary.setPlainText("\n".join(lines))

        def _new_project(self) -> None:
            self.controller.new_project()
            self._refresh_all()

        def _load_example(self) -> None:
            self._run("Load example", self.controller.load_example_dataset)

        def _import_table(self) -> None:
            path, _ = QFileDialog.getOpenFileName(self, "Import data table", "", "Data (*.csv *.txt *.xyz *.dat);;All files (*.*)")
            if path:
                self._run("Import table", lambda: self.controller.load_table(path))

        def _open_project(self) -> None:
            path, _ = QFileDialog.getOpenFileName(self, "Open GeoFieldLab project", "", "GeoFieldLab Project (*.gflp *.json);;All files (*.*)")
            if path:
                self._run("Open project", lambda: self.controller.load_project(path))

        def _save_project(self) -> None:
            self.controller.session.project_name = self.project_name.text().strip() or self.controller.session.project_name
            path, _ = QFileDialog.getSaveFileName(self, "Save GeoFieldLab project", "", "GeoFieldLab Project (*.gflp);;JSON (*.json)")
            if path:
                self._run("Save project", lambda: self.controller.save_project(path))

        def _detect_grid(self) -> None:
            def action() -> dict:
                detection = self.controller.detect_grid(self.x_col.currentText(), self.y_col.currentText(), self.value_col.currentText())
                quality = self.controller.data_quality(self.x_col.currentText(), self.y_col.currentText(), self.value_col.currentText())
                self.data_output.setPlainText(f"Detection:\n{detection}\n\nQuality:\n{quality}")
                if detection.get("dx"):
                    self.spacing.setValue(float(detection["dx"]))
                return detection

            self._run("Detect grid", action)

        def _create_grid(self) -> None:
            self._run(
                "Create grid",
                lambda: self.controller.create_grid(
                    self.x_col.currentText(),
                    self.y_col.currentText(),
                    self.value_col.currentText(),
                    units=self.units.currentText(),
                    crs=self.crs.text().strip() or None,
                    mode=self.grid_mode.currentText(),
                    spacing=self.spacing.value(),
                    interpolation_method=self.interp_method.currentText(),
                    fill_method=self.fill_method.currentText(),
                ),
            )

        def _plot_selected_layer(self) -> None:
            try:
                name = self._selected_layer_name()
                if name is None:
                    self.map_stats.setPlainText("No layer selected.")
                    return
                grid = self.controller.session.get_layer(name)
                self.map_canvas.plot_grid(grid, self.map_mode.currentText())
                stats = grid_statistics(grid)
                self.map_stats.setPlainText("\n".join(f"{key}: {value:.6g}" for key, value in stats.items()))
            except Exception as exc:
                self.map_stats.setPlainText(str(exc))

        def _plot_selected_profile(self) -> None:
            try:
                name = self._selected_profile_name()
                if name is None:
                    self.profile_stats.setPlainText("No profile selected.")
                    return
                profile = self.controller.session.get_profile(name)
                self.profile_canvas.plot_profile(profile)
                stats = profile_statistics(profile)
                self.profile_stats.setPlainText("\n".join(f"{key}: {value}" for key, value in stats.items()))
            except Exception as exc:
                self.profile_stats.setPlainText(str(exc))

        def _compute_grid_method(self, method: Callable[[str | None], object]) -> None:
            self._run("Compute method", lambda: method(self._selected_layer_name()))

        def _compute_fvd(self) -> None:
            self._run("Compute FVD", lambda: self.controller.compute_fvd(self._selected_layer_name(), self.derivative_padding.currentText()))

        def _compute_svd(self) -> None:
            self._run("Compute SVD", lambda: self.controller.compute_svd(self._selected_layer_name(), self.derivative_padding.currentText()))

        def _compute_tilt(self) -> None:
            self._run("Compute TILT", lambda: self.controller.compute_tilt(self._selected_layer_name(), padding=self.derivative_padding.currentText()))

        def _compute_asa(self) -> None:
            self._run("Compute ASA", lambda: self.controller.compute_analytic_signal(self._selected_layer_name(), self.derivative_padding.currentText()))

        def _compute_upward(self) -> None:
            self._run(
                "Compute upward continuation",
                lambda: self.controller.compute_upward_residual(
                    self._selected_layer_name(),
                    height=self.uc_height.value(),
                    padding=self.uc_padding.currentText(),
                    remove_mean=self.uc_remove_mean.isChecked(),
                ),
            )

        def _compute_butterworth_rr(self) -> None:
            self._run(
                "Compute Butterworth regional/residual",
                lambda: self.controller.compute_butterworth_regional_residual(
                    self._selected_layer_name(),
                    cutoff_wavelength=self.bw_cutoff.value(),
                    order=self.bw_order.value(),
                    padding=self.bw_padding.currentText(),
                ),
            )

        def _compute_butterworth(self) -> None:
            cutoff = (
                (self.filter_cutoff.value(), self.filter_cutoff2.value())
                if self.filter_type.currentText() == "bandpass"
                else self.filter_cutoff.value()
            )
            self._run(
                "Compute Butterworth",
                lambda: self.controller.compute_butterworth(
                    self._selected_layer_name(),
                    cutoff=cutoff,
                    order=self.filter_order.value(),
                    filter_type=self.filter_type.currentText(),
                ),
            )

        def _grid_transform(self, operation: str) -> None:
            if operation == "flip_x":
                self._run("Flip X", lambda: self.controller.flip_grid(self._selected_layer_name(), "x"))
            elif operation == "flip_y":
                self._run("Flip Y", lambda: self.controller.flip_grid(self._selected_layer_name(), "y"))
            elif operation == "rotate_90":
                self._run("Rotate 90", lambda: self.controller.rotate_grid(self._selected_layer_name(), 90))

        def _compute_fsed(self) -> None:
            self._run(
                "Compute FSED",
                lambda: self.controller.compute_fsed(
                    self._selected_layer_name(),
                    base=self.fsed_base.currentText(),
                    gain=self.fsed_gain.value(),
                    method=self.fsed_method.currentText(),
                    threshold=self.fsed_threshold.value(),
                ),
            )

        def _compute_raps(self) -> None:
            def action() -> dict:
                summary = self.controller.compute_raps(
                    self._selected_layer_name(),
                    remove_mean=self.raps_remove_mean.isChecked(),
                    detrend=self.raps_detrend.isChecked(),
                    window=self.raps_window.currentText(),
                    padding=self.raps_padding.currentText(),
                )
                self.raps_output.setPlainText("\n".join(f"{key}: {value}" for key, value in summary.items()))
                return summary

            self._run("Compute RAPS", action)

        def _extract_cross_section(self) -> None:
            self._run(
                "Extract cross-section",
                lambda: self.controller.extract_cross_section(
                    self._selected_layer_name(),
                    (self.start_x.value(), self.start_y.value()),
                    (self.end_x.value(), self.end_y.value()),
                    spacing=self.section_spacing.value(),
                ),
            )

        def _profile_running_average(self) -> None:
            self._run(
                "Profile running average",
                lambda: self.controller.transform_profile(
                    self._selected_profile_name(),
                    "running_average",
                    window_size=self.profile_window.value(),
                ),
            )

        def _profile_remove_trend(self) -> None:
            self._run(
                "Profile remove trend",
                lambda: self.controller.transform_profile(self._selected_profile_name(), "remove_linear_trend"),
            )

        def _export_layer_csv(self) -> None:
            path, _ = QFileDialog.getSaveFileName(self, "Export layer CSV", "", "CSV (*.csv)")
            if path:
                self._run("Export layer CSV", lambda: self.controller.export_layer_csv(self._selected_layer_name(), path))
                self.export_output.append(f"Layer CSV: {path}")

        def _export_layer_png(self) -> None:
            path, _ = QFileDialog.getSaveFileName(self, "Export layer PNG", "", "PNG (*.png)")
            if path:
                self._run("Export layer PNG", lambda: self.controller.export_layer_png(self._selected_layer_name(), path))
                self.export_output.append(f"Layer PNG: {path}")

        def _export_profile_csv(self) -> None:
            path, _ = QFileDialog.getSaveFileName(self, "Export profile CSV", "", "CSV (*.csv)")
            if path:
                self._run("Export profile CSV", lambda: self.controller.export_profile_csv(self._selected_profile_name(), path))
                self.export_output.append(f"Profile CSV: {path}")

        def _export_history(self) -> None:
            path, _ = QFileDialog.getSaveFileName(self, "Export history JSON", "", "JSON (*.json)")
            if path:
                self._run("Export history", lambda: self.controller.export_history_json(path))
                self.export_output.append(f"History JSON: {path}")

        def _export_report(self) -> None:
            path, _ = QFileDialog.getSaveFileName(self, "Export Markdown report", "", "Markdown (*.md)")
            if path:
                self._run("Export report", lambda: self.controller.export_markdown_report(path))
                self.export_output.append(f"Report: {path}")


def main() -> int:
    if not QT_AVAILABLE:
        print("PySide6 desktop dependencies are not installed.")
        print(f"Import error: {QT_IMPORT_ERROR}")
        print("Install with: pip install -r requirements.txt")
        return 1
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return int(app.exec())
