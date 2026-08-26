"""Lightweight interactive 1x1 through 4x4 cluster comparison board."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QObject, QRectF, Qt, QThread, Signal
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..comparison import (
    CCF_PLANE_CORONAL,
    CCF_PLANE_HORIZONTAL,
    CCF_PLANE_SAGITTAL,
    REDUCTION_PROJECTION,
    REDUCTION_SLICE,
    SOURCE_CCF_HEATMAP,
    SOURCE_CCF_SOMAS,
    SOURCE_FLATMAP_ARBOR_HEATMAP,
    SOURCE_FLATMAP_SOMAS,
    ComparisonBoardState,
    ComparisonCellSpec,
    compatible_camera_groups,
    comparison_membership_provenance,
    comparison_provenance,
    compose_tinted_heatmaps,
    shared_intensity_maxima,
)
from ..comparison_data import (
    ComparisonDataProvider,
    ComparisonHeatmapGroup,
    ComparisonRenderData,
)


logger = logging.getLogger(__name__)

_SOURCE_LABELS = {
    SOURCE_FLATMAP_SOMAS: "Flatmap somas",
    SOURCE_FLATMAP_ARBOR_HEATMAP: "Flatmap arbor heatmap",
    SOURCE_CCF_SOMAS: "CCFv3 somas",
    SOURCE_CCF_HEATMAP: "Existing CCFv3 heatmap(s)",
}

_PLANE_LABELS = {
    CCF_PLANE_CORONAL: "Coronal",
    CCF_PLANE_HORIZONTAL: "Horizontal",
    CCF_PLANE_SAGITTAL: "Sagittal",
}

_REDUCTION_LABELS = {
    REDUCTION_PROJECTION: "Full projection",
    REDUCTION_SLICE: "Slice / slab",
}


def _cluster_legend_entries(
    render: ComparisonRenderData,
    *,
    include_overlap: bool,
) -> list[str]:
    """Return colored, human-readable cluster mappings for one render."""
    entries: list[str] = []
    match_by_label = {match.candidate_label: match for match in render.matches}
    assignment_name = str(render.provenance.get("assignment_name") or "Assignment")
    for label in sorted(render.colors):
        color = np.clip(np.asarray(render.colors[label])[:3], 0.0, 1.0)
        hex_color = "#" + "".join(f"{int(value * 255):02x}" for value in color)
        match = match_by_label.get(label)
        if match is not None and match.reference_label is not None:
            relation = f"Reference {match.reference_label} ← {assignment_name} {label}"
            if include_overlap and match.shared_file_ids:
                relation += f" ({match.shared_file_ids:,} shared file IDs)"
        elif match is not None:
            relation = f"{assignment_name} {label} · unmatched"
        else:
            relation = f"Cluster {label}"
        entries.append(f'<span style="color:{hex_color};">■</span> {relation}')
    return entries


class _ComparisonRenderWorker(QObject):
    finished = Signal(object)

    def __init__(
        self,
        provider: ComparisonDataProvider,
        cells: list[ComparisonCellSpec],
        reference_assignment_id: str | None,
        request_token: int,
    ) -> None:
        super().__init__()
        self._provider = provider
        self._cells = cells
        self._reference_assignment_id = reference_assignment_id
        self._request_token = int(request_token)

    def run(self) -> None:
        output: list[
            tuple[ComparisonCellSpec, ComparisonRenderData | None, str | None]
        ] = []
        for cell in self._cells:
            try:
                prepared, rendered = self._provider.render_cell(
                    cell,
                    reference_assignment_id=self._reference_assignment_id,
                )
                output.append((prepared, rendered, None))
            except Exception as exc:
                logger.exception("Comparison cell render failed: %s", cell.cell_id)
                output.append((cell, None, str(exc)))
        self.finished.emit((self._request_token, output))


class _ComparisonPlotCell(QFrame):
    selected = Signal(str)
    camera_changed = Signal(str, object)

    def __init__(self, cell_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.cell_id = str(cell_id)
        self._render: ComparisonRenderData | None = None
        self._suppress_camera_signal = False
        self.setFrameShape(QFrame.StyledPanel)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        self._title = QLabel("Comparison")
        self._title.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._title)
        self._subtitle = QLabel("")
        self._subtitle.setWordWrap(True)
        layout.addWidget(self._subtitle)

        self.plot = pg.PlotWidget(background=(18, 18, 18))
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        self.plot.showGrid(x=False, y=False)
        self.plot.getPlotItem().hideButtons()
        layout.addWidget(self.plot, stretch=1)

        self._hover = QLabel("Move over the panel to inspect coordinates.")
        self._hover.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(self._hover)
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.plot.scene().sigMouseClicked.connect(
            lambda _event: self.selected.emit(self.cell_id)
        )
        self.plot.getViewBox().sigRangeChanged.connect(self._on_range_changed)
        self._legend = QLabel("")
        self._legend.setWordWrap(True)
        layout.addWidget(self._legend)
        self.set_selected(False)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.selected.emit(self.cell_id)
        super().mousePressEvent(event)

    def set_selected(self, selected: bool) -> None:
        color = "#3daee9" if selected else "#555555"
        self.setStyleSheet(f"QFrame {{ border: 2px solid {color}; }}")

    def show_error(self, title: str, message: str) -> None:
        self._render = None
        self._title.setText(title)
        self._subtitle.setText(message)
        self._legend.clear()
        self.plot.clear()
        text = pg.TextItem(
            "Source unavailable", color=(255, 170, 80), anchor=(0.5, 0.5)
        )
        text.setPos(0.5, 0.5)
        self.plot.addItem(text)
        self.plot.setXRange(0.0, 1.0, padding=0.0)
        self.plot.setYRange(0.0, 1.0, padding=0.0)

    def set_render(
        self,
        render: ComparisonRenderData,
        *,
        point_size: float,
        opacity: float,
        intensity_max: float | None,
        camera_rect: tuple[float, float, float, float] | None,
        intensity_override: bool,
    ) -> None:
        self._render = render
        self._title.setText(render.title)
        override_text = " · intensity override" if intensity_override else ""
        self._subtitle.setText(
            f"{render.subtitle} · {render.assigned_count:,} assigned · "
            f"{render.omitted_count:,} omitted{override_text}"
        )
        legend_entries = _cluster_legend_entries(render, include_overlap=False)
        visible_entries = legend_entries[:8]
        if len(legend_entries) > len(visible_entries):
            visible_entries.append(
                f"+ {len(legend_entries) - len(visible_entries)} more clusters"
            )
        self._legend.setText(" · ".join(visible_entries))
        self.plot.clear()
        self._set_axis_labels(render)

        if render.heatmaps:
            rgba, resolved_max = compose_tinted_heatmaps(
                render.heatmaps,
                render.colors,
                intensity_max=intensity_max,
                opacity=opacity,
            )
            render.intensity_max = resolved_max
            image = pg.ImageItem(
                image=rgba,
                axisOrder="row-major",
            )
            x_min, x_max = render.x_bounds
            y_min, y_max = render.y_bounds
            image.setRect(QRectF(x_min, y_min, x_max - x_min, y_max - y_min))
            self.plot.addItem(image)
        elif render.points is not None and len(render.points):
            colors = render.point_colors
            if colors is not None:
                display_colors = np.clip(np.asarray(colors, dtype=float), 0.0, 1.0)
                if display_colors.ndim == 2 and display_colors.shape[1] >= 4:
                    display_colors = display_colors.copy()
                    display_colors[:, 3] *= float(opacity)
                brushes = [
                    pg.mkBrush(*(color[:4] * 255).astype(int))
                    for color in display_colors
                ]
            else:
                brushes = pg.mkBrush(
                    220,
                    220,
                    220,
                    int(round(255 * float(opacity))),
                )
            scatter = pg.ScatterPlotItem(
                x=render.points[:, 0],
                y=render.points[:, 1],
                size=float(point_size),
                pen=None,
                brush=brushes,
                pxMode=True,
            )
            self.plot.addItem(scatter)

        self._suppress_camera_signal = True
        try:
            if camera_rect is not None:
                x_min, x_max, y_min, y_max = camera_rect
            else:
                x_min, x_max = render.x_bounds
                y_min, y_max = render.y_bounds
            self.plot.setXRange(x_min, x_max, padding=0.0)
            self.plot.setYRange(y_min, y_max, padding=0.0)
        finally:
            self._suppress_camera_signal = False

    def _set_axis_labels(self, render: ComparisonRenderData) -> None:
        if render.source_kind.startswith("flatmap"):
            bottom, left = "Flatmap X", "Flatmap Y"
        else:
            plane = str(render.provenance.get("ccf_plane") or "coronal")
            if plane == CCF_PLANE_CORONAL:
                bottom, left = "Left–right (µm)", "Dorsal–ventral (µm)"
            elif plane == CCF_PLANE_HORIZONTAL:
                bottom, left = "Left–right (µm)", "Rostral–caudal (µm)"
            else:
                bottom, left = "Rostral–caudal (µm)", "Dorsal–ventral (µm)"
        self.plot.setLabel("bottom", bottom)
        self.plot.setLabel("left", left)

    def _on_mouse_moved(self, scene_position) -> None:
        if not self.plot.sceneBoundingRect().contains(scene_position):
            return
        point = self.plot.getPlotItem().vb.mapSceneToView(scene_position)
        value_text = ""
        render = self._render
        if render is not None and render.heatmaps:
            x_min, x_max = render.x_bounds
            y_min, y_max = render.y_bounds
            first = next(iter(render.heatmaps.values()))
            if x_max > x_min and y_max > y_min and first.size:
                column = int((point.x() - x_min) / (x_max - x_min) * first.shape[1])
                row = int((point.y() - y_min) / (y_max - y_min) * first.shape[0])
                if 0 <= row < first.shape[0] and 0 <= column < first.shape[1]:
                    total = sum(
                        float(volume[row, column])
                        for volume in render.heatmaps.values()
                    )
                    value_text = f" · count {total:g}"
        self._hover.setText(f"x {point.x():.3f} · y {point.y():.3f}{value_text}")

    def _on_range_changed(self, _view_box, ranges) -> None:
        if self._suppress_camera_signal:
            return
        try:
            x_range, y_range = ranges
            rect = (
                float(x_range[0]),
                float(x_range[1]),
                float(y_range[0]),
                float(y_range[1]),
            )
        except (TypeError, ValueError, IndexError):
            return
        self.camera_changed.emit(self.cell_id, rect)

    def clear_links(self) -> None:
        view_box = self.plot.getViewBox()
        view_box.setXLink(None)
        view_box.setYLink(None)

    def link_to(self, other: "_ComparisonPlotCell") -> None:
        view_box = self.plot.getViewBox()
        view_box.setXLink(other.plot.getViewBox())
        view_box.setYLink(other.plot.getViewBox())


class ComparisonBoardWindow(QMainWindow):
    """A lightweight, project-backed grid of interactive comparison plots."""

    def __init__(
        self,
        *,
        provider: ComparisonDataProvider,
        state: ComparisonBoardState | None = None,
        state_changed_callback: Callable[[ComparisonBoardState], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("SWC Viewer Comparison Board")
        self.resize(1500, 900)
        self._provider = provider
        self._state = state or ComparisonBoardState()
        self._state_changed_callback = state_changed_callback
        self._selected_cell_id: str | None = (
            self._state.cells[0].cell_id if self._state.cells else None
        )
        self._plots: dict[str, _ComparisonPlotCell] = {}
        self._renders: dict[str, ComparisonRenderData] = {}
        self._errors: dict[str, str] = {}
        self._render_thread: QThread | None = None
        self._render_worker: _ComparisonRenderWorker | None = None
        self._pending_refresh = False
        self._render_request_token = 0
        self._heatmap_groups: tuple[ComparisonHeatmapGroup, ...] = ()
        self._setup_ui()
        self._refresh_source_controls()
        self._rebuild_grid()
        if self._state.cells:
            self.refresh_board()

    def state(self) -> ComparisonBoardState:
        return ComparisonBoardState.from_state(self._state.to_state())

    def set_state(self, state: ComparisonBoardState) -> None:
        self._state = ComparisonBoardState.from_state(state.to_state())
        self._selected_cell_id = (
            self._state.cells[0].cell_id if self._state.cells else None
        )
        self._sync_layout_controls()
        self._refresh_source_controls()
        self._rebuild_grid()
        self.refresh_board()

    def set_reference_assignment_id(self, assignment_id: str | None) -> None:
        """Synchronize the board reference with the main Compare tab."""
        normalized = str(assignment_id) if assignment_id not in (None, "") else None
        if self._state.reference_assignment_id == normalized:
            return
        self._state.reference_assignment_id = normalized
        index = self._reference_combo.findData(normalized)
        if index < 0 and normalized is not None:
            self._reference_combo.addItem(
                f"Missing reference ({normalized})", normalized
            )
            index = self._reference_combo.count() - 1
        blocked = self._reference_combo.blockSignals(True)
        self._reference_combo.setCurrentIndex(max(0, index))
        self._reference_combo.blockSignals(blocked)
        self._notify_state_changed()
        self.refresh_board()

    def refresh_sources(self) -> None:
        self._refresh_source_controls()
        self.refresh_board()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Rows:"))
        self._rows_spin = QSpinBox()
        self._rows_spin.setRange(1, 4)
        self._rows_spin.setValue(self._state.rows)
        self._rows_spin.valueChanged.connect(self._on_layout_changed)
        toolbar.addWidget(self._rows_spin)
        toolbar.addWidget(QLabel("Columns:"))
        self._columns_spin = QSpinBox()
        self._columns_spin.setRange(1, 4)
        self._columns_spin.setValue(self._state.columns)
        self._columns_spin.valueChanged.connect(self._on_layout_changed)
        toolbar.addWidget(self._columns_spin)

        self._add_btn = QPushButton("Add Cell")
        self._add_btn.clicked.connect(self._add_cell)
        toolbar.addWidget(self._add_btn)
        self._duplicate_btn = QPushButton("Duplicate")
        self._duplicate_btn.clicked.connect(self._duplicate_cell)
        toolbar.addWidget(self._duplicate_btn)
        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._remove_cell)
        toolbar.addWidget(self._remove_btn)
        self._earlier_btn = QPushButton("Move Earlier")
        self._earlier_btn.clicked.connect(lambda: self._move_cell(-1))
        toolbar.addWidget(self._earlier_btn)
        self._later_btn = QPushButton("Move Later")
        self._later_btn.clicked.connect(lambda: self._move_cell(1))
        toolbar.addWidget(self._later_btn)

        toolbar.addWidget(QLabel("Reference:"))
        self._reference_combo = QComboBox()
        self._reference_combo.currentIndexChanged.connect(self._on_reference_changed)
        toolbar.addWidget(self._reference_combo)
        self._shared_intensity_cb = QCheckBox("Share comparable intensity")
        self._shared_intensity_cb.setChecked(self._state.shared_intensity)
        self._shared_intensity_cb.toggled.connect(self._on_shared_intensity_changed)
        toolbar.addWidget(self._shared_intensity_cb)
        toolbar.addStretch()
        self._refresh_btn = QPushButton("Refresh Sources")
        self._refresh_btn.clicked.connect(self.refresh_sources)
        toolbar.addWidget(self._refresh_btn)
        self._export_btn = QPushButton("Export Board...")
        self._export_btn.clicked.connect(self._export_board)
        toolbar.addWidget(self._export_btn)
        root.addLayout(toolbar)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, stretch=1)

        self._grid_host = QWidget()
        self._grid_layout = QGridLayout(self._grid_host)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(self._grid_host)

        inspector_scroll = QScrollArea()
        inspector_scroll.setWidgetResizable(True)
        inspector_scroll.setMinimumWidth(330)
        inspector = QWidget()
        inspector_scroll.setWidget(inspector)
        inspector_layout = QVBoxLayout(inspector)
        editor = QGroupBox("Selected Cell")
        editor_form = QFormLayout(editor)

        self._title_edit = QLineEdit()
        editor_form.addRow("Title:", self._title_edit)
        self._source_combo = QComboBox()
        for source_kind, label in _SOURCE_LABELS.items():
            self._source_combo.addItem(label, source_kind)
        self._source_combo.currentIndexChanged.connect(self._update_editor_visibility)
        editor_form.addRow("Map:", self._source_combo)
        self._assignment_combo = QComboBox()
        editor_form.addRow("Assignment:", self._assignment_combo)
        self._heatmap_combo = QComboBox()
        editor_form.addRow("Heatmap source:", self._heatmap_combo)

        self._flatmap_style_combo = QComboBox()
        self._flatmap_style_combo.addItem("Both hemispheres, shaped", "both_shaped")
        self._flatmap_style_combo.addItem("Both hemispheres, square", "both_square")
        editor_form.addRow("Flatmap style:", self._flatmap_style_combo)
        self._y_bins_spin = QSpinBox()
        self._y_bins_spin.setRange(1, 4096)
        self._y_bins_spin.setValue(256)
        editor_form.addRow("Y bins:", self._y_bins_spin)
        self._x_bins_label = QLabel("Derived when the cell is rendered")
        editor_form.addRow("X bins:", self._x_bins_label)

        self._plane_combo = QComboBox()
        for plane, label in _PLANE_LABELS.items():
            self._plane_combo.addItem(label, plane)
        editor_form.addRow("CCF plane:", self._plane_combo)
        self._reduction_combo = QComboBox()
        for reduction, label in _REDUCTION_LABELS.items():
            self._reduction_combo.addItem(label, reduction)
        self._reduction_combo.currentIndexChanged.connect(
            self._update_editor_visibility
        )
        editor_form.addRow("CCF view:", self._reduction_combo)
        self._slice_position_spin = QDoubleSpinBox()
        self._slice_position_spin.setRange(0.0, 100_000.0)
        self._slice_position_spin.setDecimals(1)
        self._slice_position_spin.setSuffix(" µm")
        editor_form.addRow("Slice position:", self._slice_position_spin)
        self._slab_thickness_spin = QDoubleSpinBox()
        self._slab_thickness_spin.setRange(0.1, 100_000.0)
        self._slab_thickness_spin.setDecimals(1)
        self._slab_thickness_spin.setValue(25.0)
        self._slab_thickness_spin.setSuffix(" µm")
        editor_form.addRow("Slab thickness:", self._slab_thickness_spin)

        self._point_size_spin = QDoubleSpinBox()
        self._point_size_spin.setRange(1.0, 50.0)
        self._point_size_spin.setValue(6.0)
        editor_form.addRow("Point size:", self._point_size_spin)
        self._opacity_spin = QDoubleSpinBox()
        self._opacity_spin.setRange(0.0, 1.0)
        self._opacity_spin.setSingleStep(0.05)
        self._opacity_spin.setValue(0.85)
        editor_form.addRow("Opacity:", self._opacity_spin)
        self._intensity_override_cb = QCheckBox("Use per-cell maximum")
        self._intensity_override_cb.toggled.connect(self._update_editor_visibility)
        editor_form.addRow("Intensity:", self._intensity_override_cb)
        self._intensity_max_spin = QDoubleSpinBox()
        self._intensity_max_spin.setRange(0.001, 1_000_000_000.0)
        self._intensity_max_spin.setDecimals(3)
        self._intensity_max_spin.setValue(1.0)
        editor_form.addRow("Maximum:", self._intensity_max_spin)
        self._camera_linked_cb = QCheckBox("Link when compatible")
        self._camera_linked_cb.setChecked(True)
        editor_form.addRow("Navigation:", self._camera_linked_cb)

        self._apply_btn = QPushButton("Apply and Render")
        self._apply_btn.clicked.connect(self._apply_editor)
        editor_form.addRow(self._apply_btn)
        inspector_layout.addWidget(editor)

        self._legend_label = QLabel("Select a rendered cell to inspect its legend.")
        self._legend_label.setWordWrap(True)
        inspector_layout.addWidget(self._legend_label)
        inspector_layout.addStretch()
        splitter.addWidget(inspector_scroll)
        splitter.setStretchFactor(0, 1)

        self._status = QLabel("Add a cell to begin comparing cluster maps.")
        self._status.setWordWrap(True)
        root.addWidget(self._status)
        self._update_editor_visibility()
        self._update_action_states()

    def _sync_layout_controls(self) -> None:
        for control, value in (
            (self._rows_spin, self._state.rows),
            (self._columns_spin, self._state.columns),
        ):
            blocked = control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(blocked)
        blocked = self._shared_intensity_cb.blockSignals(True)
        self._shared_intensity_cb.setChecked(self._state.shared_intensity)
        self._shared_intensity_cb.blockSignals(blocked)

    def _refresh_source_controls(self) -> None:
        assignments = self._provider.assignments()
        current_reference = self._state.reference_assignment_id
        blocked = self._reference_combo.blockSignals(True)
        self._reference_combo.clear()
        self._reference_combo.addItem("No color matching", None)
        for assignment in assignments:
            self._reference_combo.addItem(assignment.name, assignment.assignment_id)
        reference_index = self._reference_combo.findData(current_reference)
        if current_reference is not None and reference_index < 0:
            self._reference_combo.addItem(
                f"Missing reference ({current_reference})",
                current_reference,
            )
            reference_index = self._reference_combo.count() - 1
        self._reference_combo.setCurrentIndex(max(0, reference_index))
        self._reference_combo.blockSignals(blocked)

        current_assignment = self._assignment_combo.currentData()
        self._assignment_combo.clear()
        for assignment in assignments:
            self._assignment_combo.addItem(assignment.name, assignment.assignment_id)
        if current_assignment is not None:
            index = self._assignment_combo.findData(current_assignment)
            if index >= 0:
                self._assignment_combo.setCurrentIndex(index)

        self._heatmap_groups = self._provider.heatmap_groups()
        current_sources = self._heatmap_combo.currentData()
        self._heatmap_combo.clear()
        for group in self._heatmap_groups:
            self._heatmap_combo.addItem(group.label, group.source_ids)
        if current_sources is not None:
            index = self._heatmap_combo.findData(current_sources)
            if index >= 0:
                self._heatmap_combo.setCurrentIndex(index)

        self._load_selected_into_editor()

    def _selected_cell(self) -> ComparisonCellSpec | None:
        return next(
            (
                cell
                for cell in self._state.cells
                if cell.cell_id == self._selected_cell_id
            ),
            None,
        )

    def _select_cell(self, cell_id: str) -> None:
        self._selected_cell_id = str(cell_id)
        for current_id, plot in self._plots.items():
            plot.set_selected(current_id == self._selected_cell_id)
        self._load_selected_into_editor()
        self._update_action_states()
        self._update_legend()

    def _load_selected_into_editor(self) -> None:
        cell = self._selected_cell()
        if cell is None:
            self._title_edit.clear()
            self._update_editor_visibility()
            return
        self._title_edit.setText(cell.title)
        self._set_combo_data(self._source_combo, cell.source_kind)
        self._set_combo_data(
            self._assignment_combo,
            cell.assignment_id,
            missing_label="Missing assignment",
        )
        self._set_combo_data(
            self._heatmap_combo,
            cell.comparison_source_ids,
            missing_label="Missing heatmap source",
        )
        self._set_combo_data(self._flatmap_style_combo, cell.flatmap_style)
        self._y_bins_spin.setValue(cell.y_bins)
        self._x_bins_label.setText(
            str(cell.x_bins)
            if cell.x_bins is not None
            else "Derived when the cell is rendered"
        )
        self._set_combo_data(self._plane_combo, cell.ccf_plane)
        self._set_combo_data(self._reduction_combo, cell.reduction)
        self._slice_position_spin.setValue(cell.slice_position_um or 0.0)
        self._slab_thickness_spin.setValue(cell.slab_thickness_um or 25.0)
        self._point_size_spin.setValue(cell.point_size)
        self._opacity_spin.setValue(cell.opacity)
        self._intensity_override_cb.setChecked(cell.intensity_max_override is not None)
        if cell.intensity_max_override is not None:
            self._intensity_max_spin.setValue(cell.intensity_max_override)
        self._camera_linked_cb.setChecked(cell.camera_linked)
        self._update_editor_visibility()

    @staticmethod
    def _set_combo_data(
        combo: QComboBox,
        value: object,
        *,
        missing_label: str | None = None,
    ) -> None:
        index = combo.findData(value)
        if index < 0 and missing_label is not None and value not in (None, (), ""):
            combo.addItem(f"{missing_label} ({value})", value)
            index = combo.count() - 1
        if index >= 0:
            combo.setCurrentIndex(index)

    def _update_editor_visibility(self, *_args) -> None:
        source_kind = self._source_combo.currentData()
        has_cell = self._selected_cell() is not None
        flatmap = source_kind in {
            SOURCE_FLATMAP_SOMAS,
            SOURCE_FLATMAP_ARBOR_HEATMAP,
        }
        ccf = source_kind in {SOURCE_CCF_SOMAS, SOURCE_CCF_HEATMAP}
        heatmap = source_kind in {
            SOURCE_FLATMAP_ARBOR_HEATMAP,
            SOURCE_CCF_HEATMAP,
        }
        existing_heatmap = source_kind == SOURCE_CCF_HEATMAP
        for widget in (
            self._title_edit,
            self._source_combo,
            self._assignment_combo,
            self._camera_linked_cb,
            self._opacity_spin,
        ):
            widget.setEnabled(has_cell)
        self._assignment_combo.setEnabled(has_cell and not existing_heatmap)
        self._heatmap_combo.setEnabled(has_cell and existing_heatmap)
        self._flatmap_style_combo.setEnabled(has_cell and flatmap)
        self._y_bins_spin.setEnabled(has_cell and flatmap)
        self._plane_combo.setEnabled(has_cell and ccf)
        self._reduction_combo.setEnabled(has_cell and ccf)
        slab = ccf and self._reduction_combo.currentData() == REDUCTION_SLICE
        self._slice_position_spin.setEnabled(has_cell and slab)
        self._slab_thickness_spin.setEnabled(has_cell and slab)
        self._point_size_spin.setEnabled(
            has_cell and source_kind in {SOURCE_FLATMAP_SOMAS, SOURCE_CCF_SOMAS}
        )
        self._intensity_override_cb.setEnabled(has_cell and heatmap)
        self._intensity_max_spin.setEnabled(
            has_cell and heatmap and self._intensity_override_cb.isChecked()
        )

    def _apply_editor(self) -> None:
        cell = self._selected_cell()
        if cell is None:
            return
        source_kind = str(self._source_combo.currentData())
        assignment_id = self._assignment_combo.currentData()
        source_ids: tuple[str, ...] = ()
        if source_kind == SOURCE_CCF_HEATMAP:
            source_ids = tuple(self._heatmap_combo.currentData() or ())
            selected_group = next(
                (
                    group
                    for group in self._heatmap_groups
                    if group.source_ids == source_ids
                ),
                None,
            )
            assignment_id = (
                selected_group.assignment_id
                if selected_group is not None
                else cell.assignment_id
            )
        old_flatmap_identity = (cell.flatmap_style, cell.y_bins)
        old_coordinate_identity = (
            cell.source_kind,
            cell.flatmap_style,
            cell.ccf_plane,
            cell.reduction,
            cell.comparison_source_ids,
        )
        cell.title = self._title_edit.text().strip() or "Comparison"
        cell.source_kind = source_kind
        cell.assignment_id = None if assignment_id in (None, "") else str(assignment_id)
        cell.comparison_source_ids = source_ids
        cell.flatmap_style = str(self._flatmap_style_combo.currentData())
        cell.y_bins = int(self._y_bins_spin.value())
        if old_flatmap_identity != (cell.flatmap_style, cell.y_bins):
            # A new policy choice derives x exactly once on the next render.
            cell.x_bins = None
            cell.x_bounds = None
            cell.y_bounds = None
        cell.ccf_plane = str(self._plane_combo.currentData())
        cell.reduction = str(self._reduction_combo.currentData())
        cell.slice_position_um = float(self._slice_position_spin.value())
        cell.slab_thickness_um = float(self._slab_thickness_spin.value())
        cell.point_size = float(self._point_size_spin.value())
        cell.opacity = float(self._opacity_spin.value())
        cell.intensity_max_override = (
            float(self._intensity_max_spin.value())
            if self._intensity_override_cb.isChecked()
            else None
        )
        cell.camera_linked = self._camera_linked_cb.isChecked()
        new_coordinate_identity = (
            cell.source_kind,
            cell.flatmap_style,
            cell.ccf_plane,
            cell.reduction,
            cell.comparison_source_ids,
        )
        if old_coordinate_identity != new_coordinate_identity:
            cell.coordinate_provenance = {}
            cell.camera_rect = None
        else:
            self._propagate_linked_slice_position(cell)
        self._notify_state_changed()
        self.refresh_board()

    def _propagate_linked_slice_position(self, source: ComparisonCellSpec) -> None:
        """Synchronize physical slice position across already-compatible cells."""
        if (
            not source.camera_linked
            or source.source_kind not in {SOURCE_CCF_SOMAS, SOURCE_CCF_HEATMAP}
            or source.reduction != REDUCTION_SLICE
        ):
            return
        source_render = self._renders.get(source.cell_id)
        if source_render is None:
            return
        for candidate in self._state.cells:
            if candidate.cell_id == source.cell_id or not candidate.camera_linked:
                continue
            candidate_render = self._renders.get(candidate.cell_id)
            if (
                candidate_render is not None
                and candidate_render.compatibility_key
                == source_render.compatibility_key
                and candidate.ccf_plane == source.ccf_plane
                and candidate.reduction == REDUCTION_SLICE
            ):
                candidate.slice_position_um = source.slice_position_um

    def _on_layout_changed(self, _value: int) -> None:
        rows = int(self._rows_spin.value())
        columns = int(self._columns_spin.value())
        capacity = rows * columns
        if len(self._state.cells) > capacity:
            self._status.setText(
                f"Remove {len(self._state.cells) - capacity} cell(s) before "
                f"shrinking the board to {rows}×{columns}."
            )
            self._sync_layout_controls()
            return
        self._state.resize(rows, columns)
        self._rebuild_grid()
        self._notify_state_changed()

    def _default_assignment_id(self) -> str | None:
        assignments = self._provider.assignments()
        return assignments[0].assignment_id if assignments else None

    def _add_cell(self) -> None:
        if len(self._state.cells) >= self._state.capacity:
            self._status.setText("The current grid is full; enlarge it first.")
            return
        assignment_id = self._default_assignment_id()
        assignment = self._provider.assignment(assignment_id)
        number = len(self._state.cells) + 1
        cell = ComparisonCellSpec(
            title=(
                assignment.name if assignment is not None else f"Comparison {number}"
            ),
            assignment_id=assignment_id,
        )
        self._state.add_cell(cell)
        self._selected_cell_id = cell.cell_id
        self._rebuild_grid()
        self._notify_state_changed()
        self.refresh_board()

    def _duplicate_cell(self) -> None:
        cell = self._selected_cell()
        if cell is None or len(self._state.cells) >= self._state.capacity:
            return
        duplicate = self._state.duplicate_cell(cell.cell_id)
        self._selected_cell_id = duplicate.cell_id
        self._rebuild_grid()
        self._notify_state_changed()
        self.refresh_board()

    def _remove_cell(self) -> None:
        cell = self._selected_cell()
        if cell is None:
            return
        index = self._state.cells.index(cell)
        self._state.remove_cell(cell.cell_id)
        self._renders.pop(cell.cell_id, None)
        self._errors.pop(cell.cell_id, None)
        self._selected_cell_id = (
            self._state.cells[min(index, len(self._state.cells) - 1)].cell_id
            if self._state.cells
            else None
        )
        self._rebuild_grid()
        self._notify_state_changed()

    def _move_cell(self, offset: int) -> None:
        cell = self._selected_cell()
        if cell is None:
            return
        index = self._state.cells.index(cell)
        target = index + int(offset)
        if target < 0 or target >= len(self._state.cells):
            return
        self._state.move_cell(cell.cell_id, offset)
        self._rebuild_grid()
        self._notify_state_changed()

    def _on_reference_changed(self, index: int) -> None:
        self._state.reference_assignment_id = (
            self._reference_combo.itemData(index) if index >= 0 else None
        )
        self._notify_state_changed()
        self.refresh_board()

    def _on_shared_intensity_changed(self, checked: bool) -> None:
        self._state.shared_intensity = bool(checked)
        self._notify_state_changed()
        self._redraw_rendered_cells()

    def _clear_grid_layout(self) -> None:
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _rebuild_grid(self) -> None:
        self._clear_grid_layout()
        self._plots = {}
        for index, cell in enumerate(self._state.cells):
            plot = _ComparisonPlotCell(cell.cell_id)
            plot.selected.connect(self._select_cell)
            plot.camera_changed.connect(self._on_cell_camera_changed)
            plot.set_selected(cell.cell_id == self._selected_cell_id)
            self._plots[cell.cell_id] = plot
            self._grid_layout.addWidget(
                plot,
                index // self._state.columns,
                index % self._state.columns,
            )
            render = self._renders.get(cell.cell_id)
            if render is not None:
                self._draw_one(cell, render)
            elif cell.cell_id in self._errors:
                plot.show_error(cell.title, self._errors[cell.cell_id])
        for row in range(self._state.rows):
            self._grid_layout.setRowStretch(row, 1)
        for column in range(self._state.columns):
            self._grid_layout.setColumnStretch(column, 1)
        self._load_selected_into_editor()
        self._update_action_states()
        self._apply_camera_links()

    def refresh_board(self) -> None:
        self._render_request_token += 1
        if self._render_thread is not None and self._render_thread.isRunning():
            self._pending_refresh = True
            return
        if not self._state.cells:
            self._pending_refresh = False
            self._status.setText("Add a cell to begin comparing cluster maps.")
            return
        self._status.setText("Rendering comparison cells...")
        self._pending_refresh = False
        thread = QThread(self)
        worker = _ComparisonRenderWorker(
            self._provider,
            [
                ComparisonCellSpec.from_state(cell.to_state())
                for cell in self._state.cells
            ],
            self._state.reference_assignment_id,
            self._render_request_token,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_render_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_render_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._render_thread = thread
        self._render_worker = worker
        self._update_action_states()
        thread.start()

    def _on_render_finished(self, payload: object) -> None:
        if not isinstance(payload, tuple) or len(payload) != 2:
            return
        request_token, output = payload
        if request_token != self._render_request_token:
            return
        if not isinstance(output, list):
            return
        by_id = {cell.cell_id: cell for cell in self._state.cells}
        self._errors.clear()
        for prepared, render, error in output:
            if not isinstance(prepared, ComparisonCellSpec):
                continue
            current = by_id.get(prepared.cell_id)
            if current is None:
                continue
            # Carry authoritative derived grid values back into persistent state.
            current.x_bins = prepared.x_bins
            current.x_bounds = prepared.x_bounds
            current.y_bounds = prepared.y_bounds
            if render is not None:
                current.coordinate_provenance = dict(render.coordinate_provenance)
                self._renders[current.cell_id] = render
            elif error is not None:
                self._renders.pop(current.cell_id, None)
                self._errors[current.cell_id] = str(error)
        self._notify_state_changed()
        self._redraw_rendered_cells()
        for cell_id, error in self._errors.items():
            plot = self._plots.get(cell_id)
            cell = by_id.get(cell_id)
            if plot is not None and cell is not None:
                plot.show_error(cell.title, error)
        completed = len(self._renders)
        failed = len(self._errors)
        self._status.setText(
            f"Rendered {completed} comparison cell(s)"
            + (f"; {failed} source(s) unavailable." if failed else ".")
        )

    def _on_render_thread_finished(self) -> None:
        self._render_thread = None
        self._render_worker = None
        self._update_action_states()
        if self._pending_refresh:
            self.refresh_board()

    def _shared_intensity_maxima(self) -> dict[tuple[object, ...], float]:
        return shared_intensity_maxima(
            [
                (
                    render.intensity_key,
                    render.observed_intensity_max,
                    cell.intensity_max_override,
                )
                for cell in self._state.cells
                if (render := self._renders.get(cell.cell_id)) is not None
            ]
        )

    def _redraw_rendered_cells(self) -> None:
        for cell in self._state.cells:
            render = self._renders.get(cell.cell_id)
            if render is not None:
                self._draw_one(cell, render)
        self._apply_camera_links()
        self._update_legend()

    def _draw_one(self, cell: ComparisonCellSpec, render: ComparisonRenderData) -> None:
        plot = self._plots.get(cell.cell_id)
        if plot is None:
            return
        if cell.intensity_max_override is not None:
            intensity_max = cell.intensity_max_override
        elif self._state.shared_intensity and render.intensity_key is not None:
            intensity_max = self._shared_intensity_maxima().get(
                render.intensity_key, render.observed_intensity_max
            )
        else:
            intensity_max = render.observed_intensity_max
        plot.set_render(
            render,
            point_size=cell.point_size,
            opacity=cell.opacity,
            intensity_max=intensity_max,
            camera_rect=cell.camera_rect,
            intensity_override=cell.intensity_max_override is not None,
        )

    def _apply_camera_links(self) -> None:
        for plot in self._plots.values():
            plot.clear_links()
        compatibility = {
            cell_id: render.compatibility_key
            for cell_id, render in self._renders.items()
            if cell_id in self._plots
        }
        for cell_ids in compatible_camera_groups(
            self._state.cells,
            compatibility,
        ):
            plots = [self._plots[cell_id] for cell_id in cell_ids]
            if len(plots) < 2:
                continue
            anchor = plots[0]
            for plot in plots[1:]:
                plot.link_to(anchor)

    def _on_cell_camera_changed(self, cell_id: str, rect: object) -> None:
        if not isinstance(rect, tuple) or len(rect) != 4:
            return
        cell = next(
            (
                candidate
                for candidate in self._state.cells
                if candidate.cell_id == cell_id
            ),
            None,
        )
        if cell is None:
            return
        cell.camera_rect = tuple(float(value) for value in rect)
        self._notify_state_changed()

    def _update_action_states(self) -> None:
        has_cell = self._selected_cell() is not None
        busy = self._render_thread is not None and self._render_thread.isRunning()
        self._add_btn.setEnabled(
            not busy and len(self._state.cells) < self._state.capacity
        )
        self._duplicate_btn.setEnabled(
            not busy and has_cell and len(self._state.cells) < self._state.capacity
        )
        self._remove_btn.setEnabled(not busy and has_cell)
        self._earlier_btn.setEnabled(not busy and has_cell)
        self._later_btn.setEnabled(not busy and has_cell)
        self._apply_btn.setEnabled(not busy and has_cell)
        self._export_btn.setEnabled(not busy and bool(self._state.cells))

    def _update_legend(self) -> None:
        render = self._renders.get(str(self._selected_cell_id))
        if render is None:
            self._legend_label.setText("Select a rendered cell to inspect its legend.")
            return
        lines = ["<b>Display legend</b>"]
        lines.extend(_cluster_legend_entries(render, include_overlap=True))
        if render.intensity_max is not None:
            lines.append(f"Shared/display maximum: {render.intensity_max:g}")
        self._legend_label.setText("<br>".join(lines))

    def _notify_state_changed(self) -> None:
        callback = self._state_changed_callback
        if callable(callback):
            callback(self.state())

    @staticmethod
    def _json_default(value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return list(value)
        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable"
        )

    def _export_board(self) -> None:
        output, _ = QFileDialog.getSaveFileName(
            self,
            "Export Comparison Board",
            "comparison_board.png",
            "PNG Images (*.png);;All Files (*)",
        )
        if not output:
            return
        path = Path(output)
        if path.suffix.lower() != ".png":
            path = path.with_suffix(".png")
        pixmap = self._grid_host.grab()
        if not pixmap.save(str(path), "PNG"):
            self._status.setText(f"Could not write comparison image: {path}")
            return
        reference = self._provider.assignment(self._state.reference_assignment_id)
        reference_provenance = None
        if reference is not None:
            reference_labels = sorted(set(reference.assignments.values()))
            reference_cohort = set(reference.input_file_ids)
            reference_cohort.update(reference.unassigned_neuron_ids)
            reference_provenance = {
                "assignment_id": reference.assignment_id,
                "assignment_name": reference.name,
                "cluster_ids": reference_labels,
                "saved_palette": {
                    str(label): list(reference.label_colors.get(label, ()))
                    for label in reference_labels
                },
                "assigned_neurons": len(reference.assignments),
                "omitted_or_unassigned_neurons": len(
                    reference_cohort.difference(reference.assignments)
                ),
            }
        provenance = comparison_provenance(
            self._state,
            cells=[
                self._renders[cell.cell_id].to_provenance()
                for cell in self._state.cells
                if cell.cell_id in self._renders
            ],
            source_parquet=(
                str(self._provider.parquet_path())
                if self._provider.parquet_path() is not None
                else None
            ),
            source_signature=self._provider.source_signature(),
            reference_assignment=reference_provenance,
            membership_comparisons=comparison_membership_provenance(
                self._state,
                assignments=self._provider.assignments(),
                assignment_id_by_cell={
                    cell_id: (
                        str(assignment_id)
                        if (assignment_id := render.provenance.get("assignment_id"))
                        not in (None, "")
                        else None
                    )
                    for cell_id, render in self._renders.items()
                },
            ),
        )
        sidecar = path.with_suffix(".comparison.json")
        sidecar.write_text(
            json.dumps(
                provenance,
                indent=2,
                sort_keys=True,
                default=self._json_default,
            )
            + "\n"
        )
        self._status.setText(f"Exported {path.name} and {sidecar.name}.")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        # Keep the regular Qt window and its board model alive so reopening is
        # instant and an in-flight DuckDB worker is never destroyed with QThread.
        self._notify_state_changed()
        self.hide()
        event.ignore()
