"""PySide6 desktop shell for GEONARBA (GeoFieldLab Pro).

Layout (standard GIS/processing-suite structure):
  menu bar  : File | Data | one menu per scientific domain (category sub-menus -> methods) | View | Help
  left dock : Project Explorer (Grids, Profiles, Sections, Tables, Images)
  right dock: Processing Toolbox (searchable domain > category > method tree) + auto-generated parameter panel
  centre    : tabs Viewer | Table | Report | Data Import | History
  bottom    : Log
The toolbox, menus and parameter forms are generated from core.registry, so every method has the same UX.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

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

    from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, QThread, Signal
    from PySide6.QtGui import QAction, QKeySequence
    from PySide6.QtWidgets import (
        QAbstractItemView, QApplication, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout,
        QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QMainWindow, QMenu, QMessageBox,
        QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QSplitter, QTableView, QTabWidget, QTextEdit, QTreeWidget,
        QTreeWidgetItem, QVBoxLayout, QWidget,
    )
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
    from matplotlib.figure import Figure

    QT_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on local GUI deps.
    QT_IMPORT_ERROR = exc
    QT_AVAILABLE = False

import numpy as np
import pandas as pd

from core.grid_operations import extended_grid_statistics
from core.registry import BY_KEY, REGISTRY, Method, domains, numbered_count
from desktop.controller import ProjectController

APP_TITLE = "GEONARBA - GeoFieldLab Pro Desktop"
KIND_LABELS = {"grid": "Grids", "profile": "Profiles", "section": "Sections", "table": "Tables", "image": "Images"}
KIND_STORES = {"grid": "layers", "profile": "profiles", "section": "sections", "table": "tables", "image": "images"}
COLORMAPS = ("viridis", "turbo", "RdBu_r", "seismic", "gray", "terrain", "magma", "coolwarm")


if QT_AVAILABLE:

    class DataFrameModel(QAbstractTableModel):
        def __init__(self, frame: pd.DataFrame | None = None, max_rows: int = 5000) -> None:
            super().__init__()
            self.frame = pd.DataFrame() if frame is None else frame.head(max_rows)

        def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
            return len(self.frame)

        def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
            return len(self.frame.columns)

        def data(self, index, role=Qt.DisplayRole):
            if role == Qt.DisplayRole and index.isValid():
                v = self.frame.iat[index.row(), index.column()]
                return f"{v:.6g}" if isinstance(v, (float, np.floating)) else str(v)
            return None

        def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
            if role != Qt.DisplayRole:
                return None
            return str(self.frame.columns[section]) if orientation == Qt.Horizontal else str(self.frame.index[section])

    class Canvas(FigureCanvas):
        """Matplotlib canvas that knows how to draw every GEONARBA data type."""

        def __init__(self) -> None:
            self.figure = Figure(figsize=(7, 5), constrained_layout=True)
            super().__init__(self.figure)

        def show_object(self, obj: Any, mode: str = "image", cmap: str = "viridis") -> None:
            self.figure.clear()
            kind = type(obj).__name__
            if kind == "GridData":
                self._grid(obj, mode, cmap)
            elif kind == "ProfileData":
                self._profile(obj)
            elif kind == "Section":
                self._section(obj, cmap if cmap in ("seismic", "gray", "RdBu_r") else "seismic")
            elif kind == "RasterImage":
                ax = self.figure.add_subplot(111)
                ax.imshow(obj.rgb, extent=obj.extent, origin="lower")
                ax.set_title(obj.name)
            elif kind == "DataFrame":
                self._table(obj)
            self.draw_idle()

        def _grid(self, grid, mode: str, cmap: str) -> None:
            ax = self.figure.add_subplot(111)
            x, y, v = grid.x_vector, grid.y_vector, grid.values
            depth_axis = grid.metadata.get("vertical_axis") == "depth"
            if mode == "contour":
                art = ax.contour(x, y, v, 20, cmap=cmap)
                ax.clabel(art, inline=True, fontsize=7)
            elif mode == "filled contour":
                art = ax.contourf(x, y, v, 24, cmap=cmap)
                self.figure.colorbar(art, ax=ax, label=grid.units)
            elif mode == "shaded relief":
                from matplotlib.colors import LightSource

                finite = np.nan_to_num(v, nan=float(np.nanmean(v)) if np.isfinite(v).any() else 0.0)
                rgb = LightSource(315, 45).shade(finite, cmap=__import__("matplotlib").colormaps[cmap], blend_mode="soft",
                                                 vert_exag=1.0, dx=grid.dx, dy=grid.dy)
                ax.imshow(rgb, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", aspect="auto" if depth_axis else "equal")
            else:
                lo, hi = (np.nanpercentile(v, [1, 99]) if np.isfinite(v).any() else (0, 1))
                art = ax.imshow(v, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", cmap=cmap, vmin=lo, vmax=hi,
                                aspect="auto" if depth_axis or grid.metadata.get("x_axis") else "equal", interpolation="nearest")
                self.figure.colorbar(art, ax=ax, label=grid.units)
            if depth_axis:
                ax.invert_yaxis()
            ax.set_xlabel(grid.metadata.get("x_axis", "X (m)"))
            ax.set_ylabel(grid.metadata.get("y_axis", "Depth (m)" if depth_axis else "Y (m)"))
            ax.set_title(grid.name)

        def _profile(self, prof) -> None:
            meta = prof.metadata
            if meta.get("plot") == "rose":
                ax = self.figure.add_subplot(111, projection="polar")
                width = np.radians(meta.get("bin_deg", 10.0))
                theta = np.radians(prof.distance)
                ax.bar(np.concatenate([theta, theta + np.pi]), np.concatenate([prof.values, prof.values]), width=width, alpha=0.8)
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.set_title(prof.name)
                return
            ax = self.figure.add_subplot(111)
            if meta.get("log_track"):
                ax.plot(prof.values, prof.distance, lw=1)
                ax.invert_yaxis()
                ax.set_ylabel("Depth")
                ax.set_xlabel(prof.units)
            else:
                ax.plot(prof.distance, prof.values, "-", lw=1.3, marker="o" if len(prof.values) < 80 else None, ms=3)
                if "analytic" in meta:
                    ax.plot(prof.distance, meta["analytic"], "--", lw=1, label="analytic")
                    ax.legend()
                if meta.get("method") == "l_curve" and meta.get("selected"):
                    ax.plot(*meta["selected"], "r*", ms=14, label="selected lambda")
                    ax.legend()
                ax.set_xlabel(meta.get("axis", "Distance (m)"))
                ax.set_ylabel(prof.units)
                if meta.get("log_axes"):
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                elif meta.get("log_x"):
                    ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            ax.set_title(prof.name)

        def _section(self, sec, cmap: str) -> None:
            ax = self.figure.add_subplot(111)
            d = sec.data
            clip = float(np.nanpercentile(np.abs(d), 98)) or 1.0
            symmetric = np.nanmin(d) < 0
            ax.imshow(d, extent=[sec.x.min(), sec.x.max(), sec.t[-1], sec.t[0]], aspect="auto", cmap=cmap if symmetric else "viridis",
                      vmin=-clip if symmetric else np.nanmin(d), vmax=clip, interpolation="bilinear")
            if sec.ntr <= 60 and symmetric:
                scale = (np.ptp(sec.x) / max(sec.ntr, 1)) * 0.8 / clip
                for j in range(sec.ntr):
                    ax.plot(sec.x[j] + d[:, j] * scale, sec.t, "k", lw=0.4)
            ax.set_xlabel("Offset / trace position")
            ax.set_ylabel("Time (s)" if sec.sample_unit == "s" else "Depth (m)")
            ax.set_title(sec.name)

        def _table(self, table: pd.DataFrame) -> None:
            ax = self.figure.add_subplot(111)
            cols = set(table.columns)
            if {"x1", "y1", "x2", "y2"} <= cols:
                for _, r in table.iterrows():
                    ax.plot([r.x1, r.x2], [r.y1, r.y2], "-", lw=1.5)
                ax.set_aspect("equal", adjustable="datalim")
            elif {"x", "y"} <= cols:
                num = [c for c in table.select_dtypes("number").columns if c not in ("x", "y")]
                color = "depth" if "depth" in cols else (num[0] if num else None)
                if "line_id" in cols or ("ring_id" in cols):
                    key = "line_id" if "line_id" in cols else "ring_id"
                    for _, grp in table.groupby(key):
                        ax.plot(grp.x, grp.y, lw=0.8)
                else:
                    sc = ax.scatter(table.x, table.y, c=table[color] if color else None, s=14, cmap="viridis")
                    if color:
                        self.figure.colorbar(sc, ax=ax, label=color)
                ax.set_aspect("equal", adjustable="datalim")
            else:
                num = table.select_dtypes("number")
                if num.shape[1] >= 2:
                    ax.plot(num.iloc[:, 0], num.iloc[:, 1:min(num.shape[1], 5)], lw=1)
                    ax.legend(num.columns[1:min(num.shape[1], 5)], fontsize=7)
                    ax.set_xlabel(num.columns[0])
            ax.grid(True, alpha=0.3)

    class Worker(QObject):
        finished = Signal(object, object)

        def __init__(self, fn: Callable[[], Any]) -> None:
            super().__init__()
            self.fn = fn

        def run(self) -> None:
            try:
                self.finished.emit(self.fn(), None)
            except Exception as exc:  # surfaced in the UI
                self.finished.emit(None, exc)

    class ParameterPanel(QWidget):
        """Inputs + parameters form generated from a registry Method."""

        def __init__(self, window: "MainWindow") -> None:
            super().__init__()
            self.window = window
            self.method: Method | None = None
            self.input_widgets: dict[str, Any] = {}
            self.param_widgets: dict[str, Any] = {}
            layout = QVBoxLayout(self)
            self.title = QLabel("Select a method in the toolbox")
            self.title.setWordWrap(True)
            self.title.setStyleSheet("font-size: 15px; font-weight: 700;")
            self.path = QLabel("")
            self.path.setStyleSheet("color: #6b7280;")
            self.desc = QLabel("")
            self.desc.setWordWrap(True)
            self.ref = QLabel("")
            self.ref.setWordWrap(True)
            self.ref.setStyleSheet("color: #6b7280; font-style: italic;")
            for w in (self.title, self.path, self.desc, self.ref):
                layout.addWidget(w)
            self.inputs_box = QGroupBox("Inputs")
            self.inputs_form = QFormLayout(self.inputs_box)
            self.params_box = QGroupBox("Parameters")
            self.params_form = QFormLayout(self.params_box)
            layout.addWidget(self.inputs_box)
            layout.addWidget(self.params_box)
            row = QHBoxLayout()
            self.run_button = QPushButton("Run")
            self.run_button.setDefault(True)
            self.run_button.clicked.connect(self.window.run_current)
            self.demo_button = QPushButton("Use demo data")
            self.demo_button.clicked.connect(self.use_demo)
            self.reset_button = QPushButton("Defaults")
            self.reset_button.clicked.connect(lambda: self.set_method(self.method) if self.method else None)
            for b in (self.run_button, self.demo_button, self.reset_button):
                row.addWidget(b)
            layout.addLayout(row)
            layout.addStretch()
            self.run_button.setEnabled(False)

        def _clear(self, form: QFormLayout) -> None:
            while form.rowCount():
                form.removeRow(0)

        def set_method(self, method: Method | None) -> None:
            self.method = method
            self._clear(self.inputs_form)
            self._clear(self.params_form)
            self.input_widgets.clear()
            self.param_widgets.clear()
            if method is None:
                return
            self.title.setText(method.label)
            self.path.setText(f"{method.domain}  >  {method.category}")
            self.desc.setText(method.description + ("  (computationally heavy - runs in background)" if method.slow else ""))
            self.ref.setText(f"Reference: {method.reference}" if method.reference else "")
            session = self.window.controller.session
            for spec in method.inputs:
                if spec.kind in {"grids", "sections"}:
                    w = QListWidget()
                    w.setSelectionMode(QAbstractItemView.MultiSelection)
                    w.addItems(session.names(spec.kind))
                    w.setMaximumHeight(110)
                else:
                    w = QComboBox()
                    if spec.optional:
                        w.addItem("(none)")
                    w.addItems(session.names(spec.kind))
                    w.currentTextChanged.connect(self.refresh_columns)
                self.input_widgets[spec.name] = w
                self.inputs_form.addRow(spec.label + (" (optional)" if spec.optional else ""), w)
            self.inputs_box.setVisible(bool(method.inputs))
            for p in method.params:
                if p.kind == "bool":
                    w = QCheckBox()
                    w.setChecked(bool(p.default))
                elif p.kind in {"choice", "column"}:
                    w = QComboBox()
                    w.setEditable(p.kind == "column")
                    if p.kind == "choice":
                        w.addItems([str(c) for c in p.choices])
                    w.setCurrentText(str(p.default))
                elif p.kind == "int":
                    w = QSpinBox()
                    w.setRange(-1_000_000_000, 1_000_000_000)
                    w.setValue(int(p.default))
                else:
                    w = QLineEdit(f"{p.default:g}" if p.kind == "float" else str(p.default))
                if p.help:
                    w.setToolTip(p.help)
                self.param_widgets[p.name] = w
                self.params_form.addRow(p.label, w)
            self.params_box.setVisible(bool(method.params))
            self.run_button.setEnabled(True)
            self.select_demo_defaults()
            self.refresh_columns()

        def select_demo_defaults(self) -> None:
            """Pre-select inputs matching the method's demo datasets when they are loaded."""

            if not self.method:
                return
            for spec in self.method.inputs:
                w = self.input_widgets[spec.name]
                demos = spec.demo if isinstance(spec.demo, tuple) else (spec.demo,)
                if isinstance(w, QListWidget):
                    for i in range(w.count()):
                        w.item(i).setSelected(w.item(i).text() in demos)
                elif not spec.optional and demos[0] and w.findText(demos[0]) >= 0:
                    w.setCurrentText(demos[0])

        def use_demo(self) -> None:
            if not self.method:
                return
            names = []
            for spec in self.method.inputs:
                names += list(spec.demo) if isinstance(spec.demo, tuple) else ([spec.demo] if spec.demo else [])
            added = self.window.controller.load_demo_library(names)
            self.window.refresh_explorer()
            self.set_method(self.method)
            for spec in self.method.inputs:
                w = self.input_widgets[spec.name]
                demos = spec.demo if isinstance(spec.demo, tuple) else (spec.demo,)
                if isinstance(w, QComboBox) and demos[0] and w.findText(demos[0]) >= 0:
                    w.setCurrentText(demos[0])
            self.refresh_columns()
            self.window.log(f"Demo data ready for {self.method.label}: {', '.join(added) or 'already loaded'}")

        def refresh_columns(self) -> None:
            if not self.method:
                return
            session = self.window.controller.session
            for p in self.method.params:
                if p.kind != "column":
                    continue
                w = self.param_widgets[p.name]
                src = self.input_widgets.get(p.source)
                name = src.currentText() if isinstance(src, QComboBox) else ""
                cols = [str(c) for c in session.tables[name].columns] if name in session.tables else []
                current = w.currentText() or str(p.default)
                w.blockSignals(True)
                w.clear()
                w.addItems(cols)
                w.setCurrentText(current if (current in cols or not cols) else (str(p.default) if str(p.default) in cols else cols[0]))
                w.blockSignals(False)

        def values(self) -> tuple[dict[str, Any], dict[str, Any]]:
            inputs: dict[str, Any] = {}
            for spec in self.method.inputs:
                w = self.input_widgets[spec.name]
                if isinstance(w, QListWidget):
                    inputs[spec.name] = [i.text() for i in w.selectedItems()]
                else:
                    t = w.currentText()
                    inputs[spec.name] = None if t in ("", "(none)") else t
            params: dict[str, Any] = {}
            for p in self.method.params:
                w = self.param_widgets[p.name]
                if p.kind == "bool":
                    params[p.name] = w.isChecked()
                elif p.kind == "int":
                    params[p.name] = w.value()
                elif p.kind in {"choice", "column"}:
                    txt = w.currentText()
                    params[p.name] = type(p.default)(txt) if p.kind == "choice" and not isinstance(p.default, str) else txt
                elif p.kind == "float":
                    try:
                        params[p.name] = float(w.text())
                    except ValueError as exc:
                        raise ValueError(f"'{p.label}' must be a number.") from exc
                else:
                    params[p.name] = w.text()
            return inputs, params

    class MainWindow(QMainWindow):
        def __init__(self, controller: ProjectController | None = None) -> None:
            super().__init__()
            self.controller = controller or ProjectController()
            self.current_object: str | None = None
            self._thread: QThread | None = None
            self.setWindowTitle(APP_TITLE)
            self.resize(1500, 900)
            self._build_central()
            self._build_docks()
            self._build_menus()
            self.statusBar().showMessage(f"Ready - {numbered_count()} numbered methods, {len(REGISTRY)} tools")
            self.refresh_explorer()

        # ---------------------------------------------------------------- layout
        def _build_central(self) -> None:
            self.tabs = QTabWidget()
            viewer = QWidget()
            v = QVBoxLayout(viewer)
            bar = QHBoxLayout()
            self.map_mode = QComboBox()
            self.map_mode.addItems(["image", "shaded relief", "filled contour", "contour"])
            self.cmap = QComboBox()
            self.cmap.addItems(COLORMAPS)
            self.map_mode.currentTextChanged.connect(lambda _: self.show_current())
            self.cmap.currentTextChanged.connect(lambda _: self.show_current())
            bar.addWidget(QLabel("Display"))
            bar.addWidget(self.map_mode)
            bar.addWidget(QLabel("Colour map"))
            bar.addWidget(self.cmap)
            bar.addStretch()
            v.addLayout(bar)
            self.canvas = Canvas()
            v.addWidget(NavigationToolbar2QT(self.canvas, viewer))
            v.addWidget(self.canvas, stretch=1)
            self.info = QLabel("")
            self.info.setWordWrap(True)
            v.addWidget(self.info)
            self.tabs.addTab(viewer, "Viewer")
            self.table_view = QTableView()
            self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
            self.tabs.addTab(self.table_view, "Table")
            self.report_view = QTextEdit()
            self.report_view.setReadOnly(True)
            self.tabs.addTab(self.report_view, "Report")
            self.tabs.addTab(self._build_import_page(), "Data Import")
            self.history_view = QTextEdit()
            self.history_view.setReadOnly(True)
            self.tabs.addTab(self.history_view, "History")
            self.setCentralWidget(self.tabs)

        def _build_import_page(self) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.addWidget(QLabel("Create a regular grid from an imported X / Y / value table (regular reshape or interpolation)."))
            form = QFormLayout()
            self.x_col, self.y_col, self.value_col = QComboBox(), QComboBox(), QComboBox()
            self.units = QComboBox()
            self.units.addItems(["mGal", "nT", "Eotvos", "m", "custom/unknown"])
            self.units.setEditable(True)
            self.crs = QLineEdit()
            self.grid_mode = QComboBox()
            self.grid_mode.addItems(["auto-detect", "regular grid", "scattered points"])
            self.spacing = QDoubleSpinBox()
            self.spacing.setRange(0.0001, 1e9)
            self.spacing.setDecimals(4)
            self.spacing.setValue(1000.0)
            self.interp_method = QComboBox()
            self.interp_method.addItems(["linear", "cubic", "nearest"])
            self.fill_method = QComboBox()
            self.fill_method.addItems(["nearest", "constant", "reject"])
            self.grid_name = QLineEdit("Original anomaly")
            for label, w in [("X column", self.x_col), ("Y column", self.y_col), ("Value column", self.value_col), ("Units", self.units),
                             ("CRS", self.crs), ("Grid mode", self.grid_mode), ("Spacing (m)", self.spacing),
                             ("Interpolation", self.interp_method), ("NaN handling", self.fill_method), ("Grid name", self.grid_name)]:
                form.addRow(label, w)
            layout.addLayout(form)
            row = QHBoxLayout()
            for text, cb in (("Import Table / LAS...", self.import_file), ("Load Example XYZ", self.load_example),
                             ("Detect Grid", self.detect_grid), ("Create Grid", self.create_grid)):
                b = QPushButton(text)
                b.clicked.connect(cb)
                row.addWidget(b)
            layout.addLayout(row)
            self.import_output = QPlainTextEdit()
            self.import_output.setReadOnly(True)
            layout.addWidget(self.import_output, stretch=1)
            return page

        def _build_docks(self) -> None:
            self.explorer = QTreeWidget()
            self.explorer.setMinimumWidth(250)
            self.explorer.setHeaderLabels(["Project data", "Info"])
            self.explorer.itemSelectionChanged.connect(self._explorer_selected)
            self.explorer.setContextMenuPolicy(Qt.CustomContextMenu)
            self.explorer.customContextMenuRequested.connect(self._explorer_menu)
            self.explorer_dock = QDockWidget("Project Explorer", self)
            self.explorer_dock.setObjectName("explorer")
            self.explorer_dock.setWidget(self.explorer)
            self.addDockWidget(Qt.LeftDockWidgetArea, self.explorer_dock)

            toolbox = QWidget()
            toolbox.setMinimumWidth(380)
            tl = QVBoxLayout(toolbox)
            self.search = QLineEdit()
            self.search.setPlaceholderText("Search methods (name, number, keyword)...")
            self.search.textChanged.connect(self._filter_toolbox)
            tl.addWidget(self.search)
            self.toolbox = QTreeWidget()
            self.toolbox.setHeaderHidden(True)
            self.toolbox.itemSelectionChanged.connect(self._toolbox_selected)
            splitter = QSplitter(Qt.Vertical)
            splitter.addWidget(self.toolbox)
            self.panel = ParameterPanel(self)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(self.panel)
            splitter.addWidget(scroll)
            splitter.setSizes([380, 520])
            tl.addWidget(splitter)
            self._populate_toolbox()
            self.toolbox_dock = QDockWidget("Processing Toolbox", self)
            self.toolbox_dock.setObjectName("toolbox")
            self.toolbox_dock.setWidget(toolbox)
            self.addDockWidget(Qt.RightDockWidgetArea, self.toolbox_dock)

            self.log_view = QPlainTextEdit()
            self.log_view.setReadOnly(True)
            self.log_dock = QDockWidget("Log", self)
            self.log_dock.setObjectName("log")
            self.log_dock.setWidget(self.log_view)
            self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
            self.resizeDocks([self.explorer_dock, self.toolbox_dock], [290, 430], Qt.Horizontal)
            self.resizeDocks([self.log_dock], [120], Qt.Vertical)

        def _populate_toolbox(self) -> None:
            self.toolbox.clear()
            for domain, cats in domains().items():
                d_item = QTreeWidgetItem([domain])
                font = d_item.font(0)
                font.setBold(True)
                d_item.setFont(0, font)
                self.toolbox.addTopLevelItem(d_item)
                for cat, methods in cats.items():
                    c_item = QTreeWidgetItem([cat])
                    d_item.addChild(c_item)
                    for m in methods:
                        it = QTreeWidgetItem([m.label])
                        it.setData(0, Qt.UserRole, m.key)
                        it.setToolTip(0, m.description)
                        c_item.addChild(it)

        def _build_menus(self) -> None:
            mb = self.menuBar()
            f = mb.addMenu("&File")
            self._action(f, "New Project", self.new_project, QKeySequence.New)
            self._action(f, "Open Project...", self.open_project, QKeySequence.Open)
            self._action(f, "Save Project...", self.save_project, QKeySequence.Save)
            f.addSeparator()
            self._action(f, "Import Table / LAS...", self.import_file, "Ctrl+I")
            self._action(f, "Import Section (matrix)...", self.import_section)
            self._action(f, "Load Demo Library", self.load_demo_library, "Ctrl+D")
            f.addSeparator()
            exp = f.addMenu("Export")
            self._action(exp, "Selected Object (CSV)...", self.export_selected_csv)
            self._action(exp, "Current Figure (PNG)...", self.export_figure)
            self._action(exp, "Processing History (JSON)...", self.export_history)
            self._action(exp, "Processing Report (Markdown)...", self.export_report)
            f.addSeparator()
            self._action(f, "Exit", self.close, QKeySequence.Quit)

            data = mb.addMenu("&Data")
            self._action(data, "Grid Creation (Data Import tab)", lambda: self.tabs.setCurrentIndex(3))
            for cat, methods in domains().get("Data & Grid Tools", {}).items():
                sub = data.addMenu(_menu_text(cat))
                for m in methods:
                    self._action(sub, m.label, lambda key=m.key: self.open_method(key))

            for domain, cats in domains().items():
                if domain == "Data & Grid Tools":
                    continue
                menu = mb.addMenu(_menu_text(domain.split(" (")[0]))
                for cat, methods in cats.items():
                    sub = menu.addMenu(_menu_text(cat))
                    for m in methods:
                        self._action(sub, m.label, lambda key=m.key: self.open_method(key))

            view = mb.addMenu("&View")
            for dock in (self.explorer_dock, self.toolbox_dock, self.log_dock):
                view.addAction(dock.toggleViewAction())
            help_menu = mb.addMenu("&Help")
            self._action(help_menu, "Method Catalogue", self.show_catalogue)
            self._action(help_menu, "Scientific Warnings", self.show_warnings)
            self._action(help_menu, "About GEONARBA", self.show_about)

        def _action(self, menu: QMenu, text: str, cb: Callable, shortcut=None) -> QAction:
            act = menu.addAction(_menu_text(text))
            act.triggered.connect(cb)
            if shortcut:
                act.setShortcut(QKeySequence(shortcut))
            return act

        # ---------------------------------------------------------------- explorer
        def refresh_explorer(self) -> None:
            session = self.controller.session
            self.explorer.blockSignals(True)
            self.explorer.clear()
            for kind, label in KIND_LABELS.items():
                store = getattr(session, KIND_STORES[kind])
                top = QTreeWidgetItem([f"{label} ({len(store)})", ""])
                font = top.font(0)
                font.setBold(True)
                top.setFont(0, font)
                self.explorer.addTopLevelItem(top)
                for name, obj in store.items():
                    it = QTreeWidgetItem([name, self._describe(obj)])
                    it.setData(0, Qt.UserRole, name)
                    top.addChild(it)
                    if name == self.current_object:
                        it.setSelected(True)
                top.setExpanded(bool(store))
            self.explorer.resizeColumnToContents(0)
            self.explorer.blockSignals(False)
            self.history_view.setPlainText(json.dumps(session.history[-50:], indent=2, default=str))
            if self.panel.method:
                inputs = {k: (w.currentText() if isinstance(w, QComboBox) else None) for k, w in self.panel.input_widgets.items()}
                self.panel.set_method(self.panel.method)
                for k, v in inputs.items():
                    w = self.panel.input_widgets.get(k)
                    if isinstance(w, QComboBox) and v and w.findText(v) >= 0:
                        w.setCurrentText(v)
            self._refresh_import_columns()

        @staticmethod
        def _describe(obj: Any) -> str:
            kind = type(obj).__name__
            if kind == "GridData":
                return f"{obj.values.shape[1]}x{obj.values.shape[0]} {obj.units}"
            if kind == "ProfileData":
                return f"{obj.values.size} pts {obj.units}"
            if kind == "Section":
                return f"{obj.nt}x{obj.ntr}"
            if kind == "DataFrame":
                return f"{len(obj)} rows"
            return ""

        def _explorer_selected(self) -> None:
            items = self.explorer.selectedItems()
            name = items[0].data(0, Qt.UserRole) if items else None
            if name:
                self.current_object = name
                self.show_current()

        def _explorer_menu(self, pos) -> None:
            item = self.explorer.itemAt(pos)
            name = item.data(0, Qt.UserRole) if item else None
            if not name:
                return
            menu = QMenu(self)
            menu.addAction("View", self.show_current)
            menu.addAction("Export CSV...", self.export_selected_csv)
            menu.addAction("Delete", lambda: (self.controller.remove(name), setattr(self, "current_object", None), self.refresh_explorer()))
            menu.exec(self.explorer.viewport().mapToGlobal(pos))

        def show_current(self) -> None:
            name = self.current_object
            if not name:
                return
            try:
                obj = self.controller.session.get_object(name)
            except KeyError:
                return
            self.canvas.show_object(obj, self.map_mode.currentText(), self.cmap.currentText())
            kind = type(obj).__name__
            if kind == "DataFrame":
                self.table_view.setModel(DataFrameModel(obj))
                self.info.setText(f"{name}: {len(obj)} rows x {len(obj.columns)} columns")
            elif kind == "GridData":
                stats = extended_grid_statistics(obj)
                self.info.setText(f"{name} | dx={obj.dx:g} dy={obj.dy:g} | " + "  ".join(f"{k}={v:.5g}" for k, v in stats.items()))
                self.table_view.setModel(DataFrameModel(obj.to_dataframe()))
            elif kind == "ProfileData":
                self.info.setText(f"{name} | {obj.values.size} samples | min={np.nanmin(obj.values):.5g} max={np.nanmax(obj.values):.5g}")
                self.table_view.setModel(DataFrameModel(obj.to_dataframe()))
            elif kind == "Section":
                self.info.setText(f"{name} | {obj.nt} samples x {obj.ntr} traces | dt={obj.dt:g} {obj.sample_unit} | headers: {', '.join(obj.headers) or 'none'}")
            else:
                self.info.setText(name)

        # ---------------------------------------------------------------- toolbox
        def _filter_toolbox(self, text: str) -> None:
            text = text.lower().strip()
            for i in range(self.toolbox.topLevelItemCount()):
                d = self.toolbox.topLevelItem(i)
                d_visible = False
                for j in range(d.childCount()):
                    c = d.child(j)
                    c_visible = False
                    for k in range(c.childCount()):
                        m = BY_KEY[c.child(k).data(0, Qt.UserRole)]
                        hit = not text or text in f"{m.label} {m.description} {m.category} {m.domain}".lower()
                        c.child(k).setHidden(not hit)
                        c_visible |= hit
                    c.setHidden(not c_visible)
                    c.setExpanded(bool(text) and c_visible)
                    d_visible |= c_visible
                d.setHidden(not d_visible)
                d.setExpanded(bool(text) and d_visible)

        def _toolbox_selected(self) -> None:
            items = self.toolbox.selectedItems()
            key = items[0].data(0, Qt.UserRole) if items else None
            if key:
                self.panel.set_method(BY_KEY[key])

        def open_method(self, key: str) -> None:
            self.toolbox_dock.show()
            self.panel.set_method(BY_KEY[key])
            self.log(f"Opened {BY_KEY[key].label}")

        def run_current(self) -> None:
            method = self.panel.method
            if method is None or self._thread is not None:
                return
            try:
                inputs, params = self.panel.values()
            except ValueError as exc:
                QMessageBox.warning(self, method.name, str(exc))
                return
            self.panel.run_button.setEnabled(False)
            self.statusBar().showMessage(f"Running {method.label}...")
            self.log(f"Running {method.label} with {params}")
            self._running = method
            self._thread = QThread(self)
            self._worker = Worker(lambda: self.controller.run_method(method.key, inputs, params))
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            # bound slot on a main-thread QObject -> queued connection, UI updates stay on the GUI thread
            self._worker.finished.connect(self._run_finished)
            self._thread.start()

        def _run_finished(self, result: Any, error: Exception | None) -> None:
            method = self._running
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self.panel.run_button.setEnabled(True)
            if error is not None:
                self.statusBar().showMessage(f"{method.label} failed")
                self.log(f"ERROR in {method.label}: {error}")
                QMessageBox.critical(self, method.name, str(error))
                return
            outputs = result["outputs"]
            self.log(f"{method.label} finished in {result['seconds']:.2f} s -> {', '.join(n for _, n in outputs) or 'report'}")
            for w in result.get("warnings", []):
                self.log(f"  warning: {w}")
            if outputs:
                self.current_object = outputs[0][1]
            self.refresh_explorer()
            if result["report"] is not None:
                self.report_view.setPlainText(f"{method.label}\n\n" + json.dumps(result["report"], indent=2, default=_jsonable))
            if outputs:
                self.show_current()
                self.tabs.setCurrentIndex(1 if outputs[0][0] == "table" else 0)
            elif result["report"] is not None:
                self.tabs.setCurrentIndex(2)
            self.statusBar().showMessage(f"{method.label} complete")

        def log(self, text: str) -> None:
            self.log_view.appendPlainText(text)

        # ---------------------------------------------------------------- file actions
        def _guard(self, label: str, fn: Callable[[], Any]) -> Any:
            try:
                out = fn()
                self.statusBar().showMessage(f"{label} complete")
                return out
            except Exception as exc:
                QMessageBox.critical(self, label, str(exc))
                self.log(f"ERROR {label}: {exc}")
                return None

        def new_project(self) -> None:
            self.controller.new_project()
            self.current_object = None
            self.refresh_explorer()
            self.log("New project")

        def open_project(self) -> None:
            path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "GEONARBA Project (*.gflp *.json);;All files (*.*)")
            if path and self._guard("Open project", lambda: self.controller.load_project(path)):
                self.refresh_explorer()

        def save_project(self) -> None:
            path, _ = QFileDialog.getSaveFileName(self, "Save project", "", "GEONARBA Project (*.gflp)")
            if path:
                self._guard("Save project", lambda: self.controller.save_project(path))

        def import_file(self) -> None:
            path, _ = QFileDialog.getOpenFileName(self, "Import table", "", "Data (*.csv *.txt *.xyz *.dat *.las);;All files (*.*)")
            if path:
                name = self._guard("Import", lambda: self.controller.import_file(path))
                if name:
                    self.current_object = name
                    self.refresh_explorer()
                    self.show_current()
                    self.log(f"Imported {path} as '{name}'")

        def import_section(self) -> None:
            path, _ = QFileDialog.getOpenFileName(self, "Import section matrix (samples x traces)", "", "Matrix (*.csv *.txt *.npy)")
            if not path:
                return
            dt, ok = QInputDialog.getDouble(self, "Sample interval", "Sample interval (s, or m for depth):", 0.004, 1e-12, 1e6, 6)
            if ok:
                dx, ok2 = QInputDialog.getDouble(self, "Trace spacing", "Trace spacing (m):", 1.0, 1e-9, 1e9, 4)
                if ok2 and self._guard("Import section", lambda: self.controller.import_section(path, dt, dx)):
                    self.refresh_explorer()

        def load_demo_library(self) -> None:
            added = self._guard("Load demo library", self.controller.load_demo_library)
            if added is not None:
                self.refresh_explorer()
                self.log(f"Loaded {len(added)} demo datasets")

        def export_selected_csv(self) -> None:
            name = self.current_object
            if not name:
                return
            path, _ = QFileDialog.getSaveFileName(self, "Export CSV", f"{name}.csv", "CSV (*.csv)")
            if not path:
                return
            obj = self.controller.session.get_object(name)
            kind = type(obj).__name__
            frame = obj if kind == "DataFrame" else obj.to_dataframe() if hasattr(obj, "to_dataframe") else \
                pd.DataFrame(obj.data, columns=[f"tr{j}" for j in range(obj.ntr)]).assign(time=obj.t) if kind == "Section" else None
            if frame is None:
                QMessageBox.information(self, "Export", "Use 'Current Figure (PNG)' for images.")
                return
            self._guard("Export CSV", lambda: frame.to_csv(path, index=False))

        def export_figure(self) -> None:
            path, _ = QFileDialog.getSaveFileName(self, "Export figure", "figure.png", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
            if path:
                self._guard("Export figure", lambda: self.canvas.figure.savefig(path, dpi=200))

        def export_history(self) -> None:
            path, _ = QFileDialog.getSaveFileName(self, "Export history", "history.json", "JSON (*.json)")
            if path:
                self._guard("Export history", lambda: self.controller.export_history_json(path))

        def export_report(self) -> None:
            path, _ = QFileDialog.getSaveFileName(self, "Export report", "report.md", "Markdown (*.md)")
            if path:
                self._guard("Export report", lambda: self.controller.export_markdown_report(path))

        # ---------------------------------------------------------------- grid creation
        def _refresh_import_columns(self) -> None:
            raw = self.controller.session.raw_data
            cols = [str(c) for c in raw.columns] if raw is not None else []
            for i, combo in enumerate((self.x_col, self.y_col, self.value_col)):
                current = combo.currentText()
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(cols)
                if current in cols:
                    combo.setCurrentText(current)
                elif len(cols) > i:
                    combo.setCurrentIndex(i)
                combo.blockSignals(False)

        def load_example(self) -> None:
            if self._guard("Load example", self.controller.load_example_dataset) is not None:
                self.refresh_explorer()
                self.import_output.setPlainText("Example XYZ loaded: select columns and press 'Create Grid'.")

        def detect_grid(self) -> None:
            def action():
                det = self.controller.detect_grid(self.x_col.currentText(), self.y_col.currentText(), self.value_col.currentText())
                q = self.controller.data_quality(self.x_col.currentText(), self.y_col.currentText(), self.value_col.currentText())
                self.import_output.setPlainText(f"Detection:\n{json.dumps(det, indent=2, default=str)}\n\nQuality:\n{json.dumps(q, indent=2)}")
                if det.get("dx") and np.isfinite(det["dx"]):
                    self.spacing.setValue(float(det["dx"]))
                return det

            self._guard("Detect grid", action)

        def create_grid(self) -> None:
            grid = self._guard("Create grid", lambda: self.controller.create_grid(
                self.x_col.currentText(), self.y_col.currentText(), self.value_col.currentText(), units=self.units.currentText(),
                crs=self.crs.text().strip() or None, mode=self.grid_mode.currentText(), spacing=self.spacing.value(),
                interpolation_method=self.interp_method.currentText(), fill_method=self.fill_method.currentText(),
                name=self.grid_name.text().strip() or "Original anomaly"))
            if grid is not None:
                self.current_object = grid.name
                self.refresh_explorer()
                self.show_current()
                self.tabs.setCurrentIndex(0)

        # ---------------------------------------------------------------- help
        def show_catalogue(self) -> None:
            lines = [f"GEONARBA method catalogue - {numbered_count()} numbered methods, {len(REGISTRY)} tools\n"]
            for domain, cats in domains().items():
                lines.append(f"\n{domain}")
                for cat, methods in cats.items():
                    lines.append(f"  {cat}")
                    lines += [f"    {m.label}" for m in methods]
            self.report_view.setPlainText("\n".join(lines))
            self.tabs.setCurrentIndex(2)

        def show_warnings(self) -> None:
            from ui.method_registry import SCIENTIFIC_WARNINGS

            self.report_view.setPlainText("Scientific warnings\n\n" + "\n".join(f"- {w}" for w in SCIENTIFIC_WARNINGS))
            self.tabs.setCurrentIndex(2)

        def show_about(self) -> None:
            QMessageBox.about(self, "About GEONARBA", f"{APP_TITLE}\n\nGravity, magnetic, seismic, electrical/EM, well-log, "
                              f"machine-learning and GIS processing workspace.\n{numbered_count()} numbered methods, {len(REGISTRY)} tools.")


def _menu_text(text: str) -> str:
    """Qt treats a single '&' as a mnemonic marker; escape literal ampersands in generated labels."""

    return text.replace("&", "&&")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return str(value)


def main() -> int:
    if not QT_AVAILABLE:
        print("PySide6 desktop dependencies are not installed.")
        print(f"Import error: {QT_IMPORT_ERROR}")
        print("Install with: pip install -r requirements.txt")
        return 1
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("GEONARBA")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return int(app.exec())
