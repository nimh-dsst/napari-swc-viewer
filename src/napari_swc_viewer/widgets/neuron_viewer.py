"""Main neuron viewer widget combining all components.

This widget provides a unified interface for:
1. Loading neuron data from Parquet files
2. Selecting brain regions to filter neurons
3. Visualizing neurons as points or lines
4. Displaying Allen CCF reference layers
"""

from __future__ import annotations

import colorsys
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas
from napari.utils.notifications import show_info, show_warning
from qtpy.QtCore import QEvent, Qt, QThread, QTimer, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..analysis.mask import (
    build_binary_mask_from_threshold_range,
    isolate_heatmap_volume_to_region_ids,
    merge_heatmap_volumes,
    otsu_threshold_positive,
    smooth_heatmap_volume,
)
from ..analysis.histogram import _build_histogram_plot_series
from ..auto_center import (
    center_to_depth_world,
    compute_center_of_rendered_neurons,
    depth_axis_from_not_displayed,
)
from ..db import NeuronDatabase
from ..logging_utils import configure_debug_logging, startup_timing
from ..neuron_table_ops import ClusterFilterSelection
from ..point_import import (
    POINT_PARQUET_ORIGIN_NOT_RECORDED,
    PointImportError,
    build_grouped_point_heatmap_volumes,
    format_atlas_validation_summary,
    load_and_standardize_point_csv,
    load_standard_point_parquet_selection,
    summarize_standard_point_parquet_groups,
    validate_point_metadata_against_atlas,
)
from .reference_layers import (
    add_allen_template,
    add_brain_outline,
    add_region_mesh,
    add_region_segmentation,
    remove_region_layers,
    remove_region_segmentation,
)
from .analysis_tab import AnalysisTabWidget
from .collapsible_section import CollapsibleSection
from .mask_layer_selector import MaskLayerSelectorWidget
from .neuron_table import NeuronTableWidget
from .region_selector import RegionSelectorWidget
from .slice_projection import NeuronSliceProjector, SomaSliceProjector

if TYPE_CHECKING:
    import napari

logger = logging.getLogger(__name__)

_POINT_HEATMAP_BASE_COLORS = [
    (1.0, 0.0, 0.0, 1.0),    # red
    (0.0, 0.8, 0.0, 1.0),    # green
    (1.0, 0.9, 0.0, 1.0),    # yellow
    (0.1, 0.4, 1.0, 1.0),    # blue
    (1.0, 0.4, 0.0, 1.0),    # orange
    (0.8, 0.0, 0.8, 1.0),    # magenta
    (0.0, 0.8, 0.8, 1.0),    # cyan
    (0.6, 0.3, 0.0, 1.0),    # brown
]

_SOMA_SLICE_PROJECTION_POINT_SIZE = 100
_HISTOGRAM_BIN_COUNT = 256
_POINT_PREVIEW_LABEL_COLUMN = 0
_POINT_PREVIEW_ORIGIN_COLUMN = 1
_POINT_PREVIEW_COUNT_COLUMN = 2
_SCENE_RENDER_MODE_FULL = "full"
_SCENE_RENDER_MODE_SOMA = "soma"
_REGION_QUERY_SCOPE_WHOLE = "whole"
_REGION_QUERY_SCOPE_CURRENT = "current"
_DEFAULT_NEURON_RGBA = (0.5, 0.5, 0.5, 1.0)
_CLUSTER_FILTER_ALL = "all"
_CLUSTER_FILTER_UNCLUSTERED = "unclustered"
_CLUSTER_FILTER_CLUSTER = "cluster"
_MANUAL_HEATMAP_ALL_LABEL = "All Manual Heatmaps"
_GREEK_HEATMAP_IDENTIFIERS = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
)


def _point_heatmap_color(index: int) -> tuple[float, float, float, float]:
    """Return a distinct RGBA color for a heatmap layer."""
    if index < len(_POINT_HEATMAP_BASE_COLORS):
        return _POINT_HEATMAP_BASE_COLORS[index]

    # Spread additional labels across the hue wheel to avoid reusing colors.
    hue = (index * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (red, green, blue, 1.0)


def _point_heatmap_layer_name(
    label: str,
    origin_csv: str,
    *,
    include_origin: bool,
) -> str:
    """Return the viewer layer name for an imported point heatmap."""

    if include_origin:
        return f"Points Heatmap: {label} [{origin_csv}]"
    return f"Points Heatmap: {label}"


def _layer_metadata(layer) -> dict:
    """Return layer metadata as a mutable dict-like object."""
    metadata = getattr(layer, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _mask_layer_color(source_layers: list) -> tuple[float, float, float, float] | None:
    """Derive a labels-layer color from source heatmap metadata."""
    colors = []
    for layer in source_layers:
        color = _layer_metadata(layer).get("color")
        if color is None:
            continue
        rgba = np.asarray(color, dtype=float).reshape(-1)
        if rgba.size == 3:
            rgba = np.append(rgba, 1.0)
        if rgba.size >= 4:
            colors.append(rgba[:4])

    if not colors:
        return None

    if len(colors) == 1:
        rgba = colors[0]
    else:
        rgba = np.mean(np.vstack(colors), axis=0)
        rgba[3] = 1.0
    rgba = np.clip(rgba, 0.0, 1.0)
    return tuple(float(value) for value in rgba[:4])


def _shared_blur_sigma(source_layers: list) -> float | None:
    """Return a common blur sigma if all source layers share one."""
    values: list[float] = []
    for layer in source_layers:
        value = _layer_metadata(layer).get("blur_sigma")
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue

    if not values:
        return None
    first = values[0]
    if all(np.isclose(first, value) for value in values[1:]):
        return first
    return None


def _heatmap_contrast_limits(volume: np.ndarray) -> tuple[float, float]:
    """Return stable full-volume contrast limits for a heatmap layer."""
    if volume.size == 0:
        return (0.0, 1.0)

    upper = float(np.nanmax(volume))
    if not np.isfinite(upper) or upper <= 0.0:
        return (0.0, 1.0)
    return (0.0, upper)


class _ClusterFilterComboBox(QComboBox):
    """Compact checkable cluster filter for the Data tab."""

    selection_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._updating = False
        self._available_cluster_ids: tuple[int, ...] = ()
        self._has_unclustered_option = False
        self.setEditable(True)
        line_edit = self.lineEdit()
        if line_edit is not None:
            line_edit.setReadOnly(True)
            line_edit.setText("All")
        view = self.view()
        viewport = view.viewport()
        if viewport is not None:
            viewport.installEventFilter(self)
        else:
            view.pressed.connect(self._on_item_pressed)
        self.activated.connect(self._restore_display_text)

    def set_cluster_options(
        self,
        cluster_ids: list[int] | tuple[int, ...],
        *,
        include_unclustered: bool,
        selection: ClusterFilterSelection | None = None,
    ) -> None:
        """Populate the dropdown while preserving a valid selection."""
        available_ids = tuple(sorted(int(cluster_id) for cluster_id in cluster_ids))
        available_set = set(available_ids)
        self._available_cluster_ids = available_ids
        self._has_unclustered_option = bool(include_unclustered)

        selected = selection or ClusterFilterSelection()
        if not selected.is_all:
            selected = ClusterFilterSelection(
                selected.cluster_ids & available_set,
                selected.include_unclustered and include_unclustered,
            )

        signals_blocked = self.blockSignals(True)
        self._updating = True
        try:
            self.clear()
            all_checked = selected.is_all
            self._add_check_item(
                "All",
                _CLUSTER_FILTER_ALL,
                checked=all_checked,
            )
            if include_unclustered:
                self._add_check_item(
                    "Unclustered",
                    _CLUSTER_FILTER_UNCLUSTERED,
                    checked=selected.include_unclustered,
                )
            for cluster_id in available_ids:
                self._add_check_item(
                    f"Cluster {cluster_id}",
                    (_CLUSTER_FILTER_CLUSTER, cluster_id),
                    checked=cluster_id in selected.cluster_ids,
                )
            self.setCurrentIndex(0)
        finally:
            self._updating = False
            self.blockSignals(signals_blocked)

        self._update_display_text()

    def cluster_filter_selection(self) -> ClusterFilterSelection:
        """Return the currently checked cluster filter."""
        cluster_ids: set[int] = set()
        include_unclustered = False
        for index in range(self.count()):
            if not self._item_checked(index):
                continue
            data = self.itemData(index)
            if data == _CLUSTER_FILTER_UNCLUSTERED:
                include_unclustered = True
            elif (
                isinstance(data, tuple)
                and len(data) == 2
                and data[0] == _CLUSTER_FILTER_CLUSTER
            ):
                cluster_ids.add(int(data[1]))
        return ClusterFilterSelection(frozenset(cluster_ids), include_unclustered)

    def _add_check_item(
        self,
        text: str,
        data: object,
        *,
        checked: bool,
    ) -> None:
        self.addItem(text, data)
        index = self.count() - 1
        self._set_item_checked(index, checked)
        item_getter = getattr(self.model(), "item", None)
        item = (
            item_getter(index, self.modelColumn()) if callable(item_getter) else None
        )
        if item is not None:
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)

    def _item_checked(self, index: int) -> bool:
        return self.itemData(index, Qt.CheckStateRole) == Qt.Checked

    def _set_item_checked(self, index: int, checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked
        self.setItemData(index, state, Qt.CheckStateRole)

    def eventFilter(self, watched, event) -> bool:
        """Toggle popup checkboxes directly and keep the popup open."""
        viewport = self.view().viewport()
        if watched is viewport:
            row = self._event_row(event)
            event_type = event.type()
            if row is not None and event_type in (
                QEvent.MouseButtonPress,
                QEvent.MouseButtonDblClick,
            ):
                return True
            if row is not None and event_type == QEvent.MouseButtonRelease:
                self._toggle_item_at_row(row)
                return True
        return super().eventFilter(watched, event)

    def _event_row(self, event) -> int | None:
        pos_getter = getattr(event, "pos", None)
        if not callable(pos_getter):
            return None
        model_index = self.view().indexAt(pos_getter())
        is_valid = getattr(model_index, "isValid", None)
        if callable(is_valid) and not is_valid():
            return None
        row_getter = getattr(model_index, "row", None)
        if not callable(row_getter):
            return None
        row = int(row_getter())
        if row < 0 or row >= self.count():
            return None
        return row

    def _on_item_pressed(self, model_index) -> None:
        """Toggle the pressed row without relying on current-index changes."""
        row_getter = getattr(model_index, "row", None)
        if not callable(row_getter):
            return
        self._toggle_item_at_row(int(row_getter()))

    def _toggle_item_at_row(self, index: int) -> None:
        """Toggle one popup item and emit the updated selection."""
        if self._updating:
            return
        if index < 0 or index >= self.count():
            return

        data = self.itemData(index)
        if data == _CLUSTER_FILTER_ALL:
            self._check_all_only()
        else:
            self._set_item_checked(index, not self._item_checked(index))
            self._set_item_checked(0, False)
            if not self._has_specific_selection():
                self._set_item_checked(0, True)

        self.setCurrentIndex(0)
        self._update_display_text()
        self.selection_changed.emit(self.cluster_filter_selection())

    def _restore_display_text(self, _index: int) -> None:
        """Keep the editable combo display on the compact selection summary."""
        if self._updating:
            return
        self.setCurrentIndex(0)
        self._update_display_text()

    def _check_all_only(self) -> None:
        for index in range(self.count()):
            self._set_item_checked(index, index == 0)

    def _has_specific_selection(self) -> bool:
        return any(self._item_checked(index) for index in range(1, self.count()))

    def _update_display_text(self) -> None:
        selection = self.cluster_filter_selection()
        text = self._selection_text(selection)
        line_edit = self.lineEdit()
        if line_edit is not None:
            line_edit.setText(text)
        self.setToolTip(text)

    @staticmethod
    def _selection_text(selection: ClusterFilterSelection) -> str:
        if selection.is_all:
            return "All"

        cluster_ids = sorted(selection.cluster_ids)
        if not cluster_ids:
            return "Unclustered"
        if len(cluster_ids) == 1:
            cluster_text = f"Cluster {cluster_ids[0]}"
        else:
            cluster_text = f"{len(cluster_ids)} clusters"
        if selection.include_unclustered:
            return f"{cluster_text} + Unclustered"
        return cluster_text


class NeuronViewerWidget(QWidget):
    """Main widget for viewing neurons with brain region filtering.

    This widget integrates:
    - Parquet file loading and database querying
    - Hierarchical region selection
    - Neuron visualization (points or lines)
    - Allen CCF reference layers

    Parameters
    ----------
    napari_viewer : napari.Viewer
        The napari viewer instance.
    """

    def __init__(self, napari_viewer: napari.Viewer):
        log_path = configure_debug_logging()
        with startup_timing(logger, "neuron_viewer_init") as timing:
            super().__init__()
            if log_path is not None:
                timing.set(log_path=log_path)
                logger.debug(
                    "Debug logging enabled for NeuronViewerWidget: %s",
                    log_path,
                )
            self.viewer = napari_viewer
            self._db: NeuronDatabase | None = None
            self._atlas: BrainGlobeAtlas | None = None
            self._current_neuron_layers: list = []
            self._current_region_layers: list = []
            self._highlighted_file_ids: set[str] | None = None
            self._last_soma_selection: set = set()  # track no-op highlights
            self._auto_center_applied_once = False
            self._region_query_source = "Atlas Regions"
            self._region_query_scope = _REGION_QUERY_SCOPE_WHOLE
            self._mask_bounds_source = "manual"
            self._histogram_line_sync_active = False
            self._point_parquet_path: str | None = None
            self._point_parquet_has_origin_csv = False
            self._point_preview_counts: dict[tuple[str, str], int] = {}
            self._scene_render_modes: dict[object, str] = {}
            self._scene_display_state: dict[object, dict[str, object]] = {}

            with startup_timing(
                logger,
                "neuron_viewer_init_phase",
                phase="slice_projectors",
            ):
                self._slice_projector = NeuronSliceProjector(
                    napari_viewer,
                    tolerance=100.0,
                )
                self._soma_slice_projector = SomaSliceProjector(
                    napari_viewer,
                    tolerance=100.0,
                    point_size=_SOMA_SLICE_PROJECTION_POINT_SIZE,
                    highlight_callback=self._on_soma_selected,
                )

            # Conversion worker state
            self._convert_thread: QThread | None = None
            self._convert_worker = None
            self._point_convert_thread: QThread | None = None
            self._point_convert_worker = None
            self._point_append_thread: QThread | None = None
            self._point_append_worker = None
            self._selected_heatmap_thread: QThread | None = None
            self._selected_heatmap_worker = None
            self._selected_heatmap_request_file_ids: tuple[str, ...] = ()
            self._cached_atlas_thread: QThread | None = None
            self._cached_atlas_worker = None
            self._show_template_after_cached_atlas_load = False

            with startup_timing(
                logger,
                "neuron_viewer_init_phase",
                phase="setup_ui",
            ):
                self._setup_ui()
            with startup_timing(
                logger,
                "neuron_viewer_init_phase",
                phase="connect_layer_events",
            ):
                self._connect_layer_events()
            with startup_timing(
                logger,
                "neuron_viewer_init_phase",
                phase="refresh_heatmap_layer_list",
            ):
                self._refresh_heatmap_layer_list()
            with startup_timing(
                logger,
                "neuron_viewer_init_phase",
                phase="refresh_histogram_layer_list",
            ):
                self._refresh_histogram_layer_list()
            with startup_timing(
                logger,
                "neuron_viewer_init_phase",
                phase="refresh_mask_layer_options",
            ):
                self._refresh_mask_layer_options()

            with startup_timing(
                logger,
                "neuron_viewer_init_phase",
                phase="connect_ndisplay_event",
            ):
                self.viewer.dims.events.ndisplay.connect(
                    self._on_ndisplay_changed
                )

            with startup_timing(
                logger,
                "neuron_viewer_init_phase",
                phase="schedule_cached_template_autoload",
            ):
                QTimer.singleShot(0, self._start_cached_template_autoload)

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)

        # Tabs for organization
        tabs = QTabWidget()
        layout.addWidget(tabs)

        with startup_timing(logger, "neuron_viewer_setup_tab", tab="Data"):
            data_tab = QWidget()
            tabs.addTab(data_tab, "Data")
            self._setup_data_tab(data_tab)

        with startup_timing(logger, "neuron_viewer_setup_tab", tab="Regions"):
            regions_tab = QWidget()
            tabs.addTab(regions_tab, "Regions")
            self._setup_regions_tab(regions_tab)

        with startup_timing(
            logger,
            "neuron_viewer_setup_tab",
            tab="Visualization",
        ):
            viz_tab = QWidget()
            tabs.addTab(viz_tab, "Visualization")
            self._setup_viz_tab(viz_tab)

        with startup_timing(logger, "neuron_viewer_setup_tab", tab="Reference"):
            ref_tab = QWidget()
            tabs.addTab(ref_tab, "Reference")
            self._setup_reference_tab(ref_tab)

        with startup_timing(logger, "neuron_viewer_setup_tab", tab="Analysis"):
            self._analysis_tab = AnalysisTabWidget(self.viewer)
            self._analysis_tab.set_slice_projector(self._slice_projector)
            self._analysis_tab.set_current_table_file_ids_provider(
                self._current_table_file_ids
            )
            self._analysis_tab.cluster_colors_updated.connect(
                self._on_cluster_colors_updated
            )
            tabs.addTab(self._analysis_tab, "Analysis")

        with startup_timing(logger, "neuron_viewer_setup_tab", tab="Tools"):
            tools_tab = QWidget()
            tabs.addTab(tools_tab, "Tools")
            self._setup_tools_tab(tools_tab)

        with startup_timing(logger, "neuron_viewer_setup_tab", tab="Histogram"):
            histogram_tab = QWidget()
            tabs.addTab(histogram_tab, "Histogram")
            self._setup_histogram_tab(histogram_tab)

    def _setup_data_tab(self, parent: QWidget) -> None:
        """Set up the data loading tab."""
        parent_layout = QVBoxLayout(parent)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        parent_layout.addWidget(scroll_area)

        content = QWidget()
        scroll_area.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)

        # SWC to Parquet conversion
        convert_section = CollapsibleSection(
            "Convert SWC to Parquet",
            expanded=False,
        )
        convert_layout = convert_section.content_layout()

        convert_btn_row = QHBoxLayout()
        convert_dir_btn = QPushButton("From Directory...")
        convert_dir_btn.clicked.connect(self._convert_from_directory)
        convert_btn_row.addWidget(convert_dir_btn)

        convert_files_btn = QPushButton("From Files...")
        convert_files_btn.clicked.connect(self._convert_from_files)
        convert_btn_row.addWidget(convert_files_btn)
        convert_layout.addLayout(convert_btn_row)

        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Resolution (μm):"))
        self._convert_resolution_spin = QSpinBox()
        self._convert_resolution_spin.setRange(10, 100)
        self._convert_resolution_spin.setValue(25)
        res_row.addWidget(self._convert_resolution_spin)
        convert_layout.addLayout(res_row)

        hemisphere_row = QHBoxLayout()
        hemisphere_row.addWidget(QLabel("Hemisphere alignment:"))
        self._convert_hemisphere_combo = QComboBox()
        self._convert_hemisphere_combo.addItem("None", None)
        self._convert_hemisphere_combo.addItem("Left", "left")
        self._convert_hemisphere_combo.addItem("Right", "right")
        hemisphere_row.addWidget(self._convert_hemisphere_combo)
        convert_layout.addLayout(hemisphere_row)

        self._convert_progress = QProgressBar()
        self._convert_progress.setVisible(False)
        convert_layout.addWidget(self._convert_progress)

        self._convert_status_label = QLabel("")
        convert_layout.addWidget(self._convert_status_label)

        layout.addWidget(convert_section)

        # File selection
        file_section = CollapsibleSection("SWC Parquet Data")
        file_layout = file_section.content_layout()

        file_row = QHBoxLayout()
        self._file_label = QLabel("No file loaded")
        self._file_label.setWordWrap(True)
        file_row.addWidget(self._file_label)

        load_btn = QPushButton("Load...")
        load_btn.clicked.connect(self._load_parquet)
        file_row.addWidget(load_btn)
        file_layout.addLayout(file_row)

        # Stats
        self._stats_label = QLabel("")
        file_layout.addWidget(self._stats_label)

        layout.addWidget(file_section)

        # Atlas selection
        atlas_section = CollapsibleSection("Atlas")
        atlas_layout = atlas_section.content_layout()

        atlas_row = QHBoxLayout()

        atlas_row.addWidget(QLabel("Atlas:"))
        self._atlas_combo = QComboBox()
        self._atlas_combo.addItems(
            [
                "allen_mouse_10um",
                "allen_mouse_25um",
                "allen_mouse_50um",
            ]
        )
        self._atlas_combo.setCurrentText("allen_mouse_25um")
        atlas_row.addWidget(self._atlas_combo)

        load_atlas_btn = QPushButton("Load Atlas")
        load_atlas_btn.clicked.connect(self._load_atlas)
        atlas_row.addWidget(load_atlas_btn)

        atlas_layout.addLayout(atlas_row)

        # Atlas status label
        self._atlas_status_label = QLabel("Atlas: Not loaded")
        atlas_layout.addWidget(self._atlas_status_label)
        layout.addWidget(atlas_section)

        # Standardized point Parquet import
        point_section = CollapsibleSection(
            "Point Parquet Import",
            expanded=False,
        )
        point_layout = point_section.content_layout()

        point_create_row = QHBoxLayout()
        self._create_point_from_directory_btn = QPushButton("Create From Directory...")
        self._create_point_from_directory_btn.clicked.connect(
            self._convert_point_csv_from_directory
        )
        point_create_row.addWidget(self._create_point_from_directory_btn)

        self._create_point_from_files_btn = QPushButton("Create From File(s)...")
        self._create_point_from_files_btn.clicked.connect(
            self._convert_point_csv_from_files
        )
        point_create_row.addWidget(self._create_point_from_files_btn)
        point_layout.addLayout(point_create_row)

        point_row = QHBoxLayout()
        self._point_file_label = QLabel("No point parquet imported")
        self._point_file_label.setWordWrap(True)
        point_row.addWidget(self._point_file_label)

        self._open_point_parquet_btn = QPushButton("Open Point Parquet...")
        self._open_point_parquet_btn.clicked.connect(self._open_point_parquet)
        point_row.addWidget(self._open_point_parquet_btn)

        self._append_point_file_btn = QPushButton("Append Point file")
        self._append_point_file_btn.clicked.connect(self._append_point_file)
        point_row.addWidget(self._append_point_file_btn)
        point_layout.addLayout(point_row)

        self._point_preview_table = QTableWidget(0, 3)
        self._point_preview_table.setHorizontalHeaderLabels(
            ["Label", "Origin CSV", "Points"]
        )
        self._point_preview_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._point_preview_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._point_preview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._point_preview_table.verticalHeader().setVisible(False)
        point_header = self._point_preview_table.horizontalHeader()
        point_header.setSectionResizeMode(
            _POINT_PREVIEW_LABEL_COLUMN,
            QHeaderView.Stretch,
        )
        point_header.setSectionResizeMode(
            _POINT_PREVIEW_ORIGIN_COLUMN,
            QHeaderView.Stretch,
        )
        point_header.setSectionResizeMode(
            _POINT_PREVIEW_COUNT_COLUMN,
            QHeaderView.ResizeToContents,
        )
        self._point_preview_table.itemSelectionChanged.connect(
            self._update_point_import_controls
        )
        point_layout.addWidget(self._point_preview_table)

        self._import_selected_point_heatmaps_btn = QPushButton(
            "Import Selected Heatmaps"
        )
        self._import_selected_point_heatmaps_btn.setEnabled(False)
        self._import_selected_point_heatmaps_btn.clicked.connect(
            self._import_selected_point_heatmaps
        )
        point_layout.addWidget(self._import_selected_point_heatmaps_btn)

        self._point_append_progress = QProgressBar()
        self._point_append_progress.setVisible(False)
        point_layout.addWidget(self._point_append_progress)

        self._point_import_status_label = QLabel("")
        self._point_import_status_label.setWordWrap(True)
        point_layout.addWidget(self._point_import_status_label)

        layout.addWidget(point_section)

        # Selected neurons table
        neurons_section = CollapsibleSection("Selected Neurons")
        neurons_layout = neurons_section.content_layout()

        self._neuron_table = NeuronTableWidget()
        self._neuron_table.colors_changed.connect(self._apply_neuron_colors)
        self._neuron_table.visibility_changed.connect(self._apply_neuron_visibility)
        self._neuron_table.selection_changed.connect(self._highlight_selected_neurons)
        state_changed = getattr(self._neuron_table, "state_changed", None)
        if state_changed is not None:
            state_changed.connect(self._refresh_neuron_table_summary)
        neurons_layout.addWidget(self._neuron_table)

        manual_heatmap_row = QHBoxLayout()
        manual_heatmap_row.addWidget(QLabel("Manual Heatmap:"))
        self._manual_heatmap_combo = QComboBox()
        self._manual_heatmap_combo.currentIndexChanged.connect(
            self._on_manual_heatmap_selection_changed
        )
        manual_heatmap_row.addWidget(self._manual_heatmap_combo, 1)
        neurons_layout.addLayout(manual_heatmap_row)
        self._refresh_manual_heatmap_combo()

        cluster_filter_row = QHBoxLayout()
        cluster_filter_row.addWidget(QLabel("Cluster:"))

        self._cluster_filter_combo = _ClusterFilterComboBox()
        self._cluster_filter_combo.set_cluster_options(
            [],
            include_unclustered=False,
            selection=ClusterFilterSelection(),
        )
        self._cluster_filter_combo.selection_changed.connect(
            self._on_cluster_filter_changed
        )
        cluster_filter_row.addWidget(self._cluster_filter_combo, 1)
        neurons_layout.addLayout(cluster_filter_row)

        cluster_action_row = QHBoxLayout()

        self._hide_others_btn = QPushButton("Hide Others")
        self._hide_others_btn.setEnabled(False)
        self._hide_others_btn.clicked.connect(self._hide_not_in_selected_cluster)
        cluster_action_row.addWidget(self._hide_others_btn)

        self._show_all_btn = QPushButton("Show All")
        self._show_all_btn.setEnabled(False)
        self._show_all_btn.clicked.connect(self._show_all_neurons)
        cluster_action_row.addWidget(self._show_all_btn)

        self._recolor_cluster_btn = QPushButton("Recolor Selection")
        self._recolor_cluster_btn.setEnabled(False)
        self._recolor_cluster_btn.clicked.connect(self._recolor_selected_cluster)
        cluster_action_row.addWidget(self._recolor_cluster_btn)

        self._apply_existing_clusters_btn = QPushButton("Apply Existing Clusters")
        self._apply_existing_clusters_btn.setEnabled(False)
        self._apply_existing_clusters_btn.setVisible(False)
        self._apply_existing_clusters_btn.clicked.connect(
            self._apply_existing_clusters_from_analysis
        )
        cluster_action_row.addWidget(self._apply_existing_clusters_btn)

        neurons_layout.addLayout(cluster_action_row)

        self._selected_neurons_hint_label = QLabel(
            "Only Neurons highlighted in the above table will be added to "
            "scene. Use cmd+A or ctrl+A to select all."
        )
        self._selected_neurons_hint_label.setWordWrap(True)
        neurons_layout.addWidget(self._selected_neurons_hint_label)

        neuron_btn_row = QHBoxLayout()
        self._render_btn = QPushButton("Add Full Neurons")
        self._render_btn.clicked.connect(self._render_selected_neurons)
        neuron_btn_row.addWidget(self._render_btn)

        self._render_soma_only_btn = QPushButton("Add Soma Only")
        self._render_soma_only_btn.clicked.connect(self._render_selected_soma_only)
        neuron_btn_row.addWidget(self._render_soma_only_btn)

        self._remove_selected_btn = QPushButton("Remove Selected")
        self._remove_selected_btn.clicked.connect(self._remove_selected_neurons)
        neuron_btn_row.addWidget(self._remove_selected_btn)

        self._clear_neurons_btn = QPushButton("Clear All")
        self._clear_neurons_btn.clicked.connect(self._clear_neuron_layers)
        neuron_btn_row.addWidget(self._clear_neurons_btn)

        neurons_layout.addLayout(neuron_btn_row)

        heatmap_btn_row = QHBoxLayout()
        self._add_selected_heatmap_btn = QPushButton("Add Heatmap")
        self._add_selected_heatmap_btn.clicked.connect(
            self._add_selected_neurons_heatmap
        )
        heatmap_btn_row.addWidget(self._add_selected_heatmap_btn)

        self._remove_selected_from_table_btn = QPushButton("Remove Selected From Table")
        self._remove_selected_from_table_btn.clicked.connect(
            self._remove_selected_from_table
        )
        heatmap_btn_row.addWidget(self._remove_selected_from_table_btn)

        self._clear_table_btn = QPushButton("Clear Table")
        self._clear_table_btn.clicked.connect(self._clear_neuron_table)
        heatmap_btn_row.addWidget(self._clear_table_btn)
        heatmap_btn_row.addStretch()
        neurons_layout.addLayout(heatmap_btn_row)

        self._neuron_table_summary_label = QLabel("")
        self._neuron_table_summary_label.setWordWrap(True)
        neurons_layout.addWidget(self._neuron_table_summary_label)

        centering_row = QHBoxLayout()
        centering_row.addWidget(QLabel("Centering:"))
        self._centering_mode_combo = QComboBox()
        self._centering_mode_combo.addItem("Same", "same")
        self._centering_mode_combo.addItem("Auto", "auto")
        self._centering_mode_combo.setCurrentIndex(0)
        centering_row.addWidget(self._centering_mode_combo)
        centering_row.addStretch()
        neurons_layout.addLayout(centering_row)

        self._render_progress = QProgressBar()
        self._render_progress.setVisible(False)
        neurons_layout.addWidget(self._render_progress)

        self._render_status_label = QLabel("")
        neurons_layout.addWidget(self._render_status_label)
        self._refresh_neuron_table_summary()
        self._refresh_apply_existing_clusters_button()
        self._update_selected_neuron_heatmap_controls()

        layout.addWidget(neurons_section)
        layout.addStretch()

    def _setup_regions_tab(self, parent: QWidget) -> None:
        """Set up the region selection tab."""
        parent_layout = QVBoxLayout(parent)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        parent_layout.addWidget(scroll_area)

        content = QWidget()
        scroll_area.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Query source:"))
        self._region_query_source_combo = QComboBox()
        self._region_query_source_combo.addItems(["Atlas Regions", "Mask Layer"])
        self._region_query_source_combo.currentTextChanged.connect(
            self._on_region_query_source_changed
        )
        source_row.addWidget(self._region_query_source_combo)
        layout.addLayout(source_row)

        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Search scope:"))
        self._region_query_scope_combo = QComboBox()
        self._region_query_scope_combo.addItem("Whole Parquet", _REGION_QUERY_SCOPE_WHOLE)
        self._region_query_scope_combo.addItem("Current Table", _REGION_QUERY_SCOPE_CURRENT)
        self._region_query_scope_combo.currentTextChanged.connect(
            self._on_region_query_scope_changed
        )
        scope_row.addWidget(self._region_query_scope_combo)
        layout.addLayout(scope_row)

        self._region_query_stack = QStackedWidget()

        atlas_page = QWidget()
        atlas_layout = QVBoxLayout(atlas_page)
        atlas_layout.setContentsMargins(0, 0, 0, 0)

        self._atlas_region_scope_stack = QStackedWidget()

        whole_page = QWidget()
        whole_layout = QVBoxLayout(whole_page)
        whole_layout.setContentsMargins(0, 0, 0, 0)
        self._whole_parquet_region_selector = RegionSelectorWidget()
        self._whole_parquet_region_selector.selection_changed.connect(
            self._on_regions_selected
        )
        whole_layout.addWidget(self._whole_parquet_region_selector)
        self._atlas_region_scope_stack.addWidget(whole_page)

        current_page = QWidget()
        current_layout = QVBoxLayout(current_page)
        current_layout.setContentsMargins(0, 0, 0, 0)
        self._current_table_region_selector = RegionSelectorWidget()
        self._current_table_region_selector.selection_changed.connect(
            self._on_regions_selected
        )
        current_layout.addWidget(self._current_table_region_selector)
        self._atlas_region_scope_stack.addWidget(current_page)

        atlas_layout.addWidget(self._atlas_region_scope_stack)
        self._region_query_stack.addWidget(atlas_page)

        mask_page = QWidget()
        mask_layout = QVBoxLayout(mask_page)
        mask_layout.setContentsMargins(0, 0, 0, 0)

        self._mask_layer_selector = MaskLayerSelectorWidget()
        self._mask_layer_selector.selection_changed.connect(
            self._on_mask_layer_selection_changed
        )
        mask_layout.addWidget(self._mask_layer_selector)

        self._mask_query_hint_label = QLabel("")
        self._mask_query_hint_label.setWordWrap(True)
        mask_layout.addWidget(self._mask_query_hint_label)
        mask_layout.addStretch()
        self._region_query_stack.addWidget(mask_page)

        layout.addWidget(self._region_query_stack)

        atlas_btn_row = QHBoxLayout()
        self._atlas_query_any_node_btn = QPushButton(
            "Find Neurons with Any Node in Selected Regions"
        )
        self._atlas_query_any_node_btn.clicked.connect(
            self._query_atlas_neurons_any_node
        )
        self._atlas_query_any_node_btn.setEnabled(False)
        atlas_btn_row.addWidget(self._atlas_query_any_node_btn)

        self._atlas_query_soma_btn = QPushButton(
            "Find Neurons with Soma in Selected Regions"
        )
        self._atlas_query_soma_btn.clicked.connect(self._query_atlas_neurons_soma)
        self._atlas_query_soma_btn.setEnabled(False)
        atlas_btn_row.addWidget(self._atlas_query_soma_btn)
        layout.addLayout(atlas_btn_row)

        mask_btn_row = QHBoxLayout()
        self._mask_query_any_node_btn = QPushButton(
            "Find Neurons with Any Node in Selected Mask Layers"
        )
        self._mask_query_any_node_btn.clicked.connect(
            self._query_mask_neurons_any_node
        )
        self._mask_query_any_node_btn.setEnabled(False)
        mask_btn_row.addWidget(self._mask_query_any_node_btn)

        self._mask_query_soma_btn = QPushButton(
            "Find Neurons with Soma in Selected Mask Layers"
        )
        self._mask_query_soma_btn.clicked.connect(self._query_mask_neurons_soma)
        self._mask_query_soma_btn.setEnabled(False)
        mask_btn_row.addWidget(self._mask_query_soma_btn)
        layout.addLayout(mask_btn_row)

        clear_table_row = QHBoxLayout()
        self._regions_clear_table_btn = QPushButton("Clear Table")
        self._regions_clear_table_btn.clicked.connect(self._clear_neuron_table)
        clear_table_row.addWidget(self._regions_clear_table_btn)
        clear_table_row.addStretch()
        layout.addLayout(clear_table_row)

        self._regions_status_label = QLabel("")
        self._regions_status_label.setWordWrap(True)
        layout.addWidget(self._regions_status_label)
        layout.addStretch()
        self._on_region_query_source_changed(self._region_query_source_combo.currentText())

    def _setup_tools_tab(self, parent: QWidget) -> None:
        """Set up the blur-generation tools tab."""
        parent_layout = QVBoxLayout(parent)

        self._tools_scroll_area = QScrollArea()
        self._tools_scroll_area.setWidgetResizable(True)
        parent_layout.addWidget(self._tools_scroll_area)

        self._tools_scroll_content = QWidget()
        self._tools_scroll_area.setWidget(self._tools_scroll_content)

        layout = QVBoxLayout(self._tools_scroll_content)
        layout.setContentsMargins(0, 0, 0, 0)

        sources_group = QGroupBox("Heatmap Sources")
        sources_layout = QVBoxLayout(sources_group)
        self._heatmap_layer_list = QListWidget()
        self._heatmap_layer_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        self._heatmap_layer_list.itemSelectionChanged.connect(
            self._update_tools_controls
        )
        sources_layout.addWidget(self._heatmap_layer_list)
        self._tools_hint_label = QLabel("")
        self._tools_hint_label.setWordWrap(True)
        sources_layout.addWidget(self._tools_hint_label)
        layout.addWidget(sources_group)

        blur_group = QGroupBox("Blur Generation")
        blur_layout = QVBoxLayout(blur_group)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Create mode:"))
        self._tools_create_mode_combo = QComboBox()
        self._tools_create_mode_combo.addItem("Separate layers", "separate")
        self._tools_create_mode_combo.addItem("Merged layer", "merged")
        self._tools_create_mode_combo.currentIndexChanged.connect(
            self._update_tools_controls
        )
        mode_row.addWidget(self._tools_create_mode_combo)
        blur_layout.addLayout(mode_row)

        sigma_row = QHBoxLayout()
        self._mask_sigma_label = QLabel("Gaussian sigma (voxels):")
        sigma_row.addWidget(self._mask_sigma_label)
        self._mask_sigma_spin = QDoubleSpinBox()
        self._mask_sigma_spin.setRange(0.0, 20.0)
        self._mask_sigma_spin.setDecimals(2)
        self._mask_sigma_spin.setSingleStep(0.25)
        self._mask_sigma_spin.setValue(1.0)
        sigma_row.addWidget(self._mask_sigma_spin)
        blur_layout.addLayout(sigma_row)
        self._mask_sigma_units_label = QLabel(
            "1 voxel = atlas voxel size; load an atlas to see the micron equivalent."
        )
        self._mask_sigma_units_label.setWordWrap(True)
        blur_layout.addWidget(self._mask_sigma_units_label)

        self._create_blur_btn = QPushButton("Create Blurred Layer")
        self._create_blur_btn.clicked.connect(self._create_blurred_layers_from_heatmaps)
        blur_layout.addWidget(self._create_blur_btn)

        layout.addWidget(blur_group)

        isolation_group = QGroupBox("Region Isolation")
        isolation_layout = QVBoxLayout(isolation_group)

        isolation_mode_row = QHBoxLayout()
        isolation_mode_row.addWidget(QLabel("Create mode:"))
        self._region_isolation_create_mode_combo = QComboBox()
        self._region_isolation_create_mode_combo.addItem("Separate layers", "separate")
        self._region_isolation_create_mode_combo.addItem("Merged layer", "merged")
        self._region_isolation_create_mode_combo.currentIndexChanged.connect(
            self._update_tools_controls
        )
        isolation_mode_row.addWidget(self._region_isolation_create_mode_combo)
        isolation_layout.addLayout(isolation_mode_row)

        with startup_timing(
            logger,
            "neuron_viewer_tools_region_selector",
            log_start=False,
        ):
            self._tools_region_selector = RegionSelectorWidget()
            self._tools_region_selector.selection_changed.connect(
                lambda _acronyms: self._update_tools_controls()
            )
            isolation_layout.addWidget(self._tools_region_selector)

        self._create_region_isolated_heatmap_btn = QPushButton(
            "Create Isolated Heatmap"
        )
        self._create_region_isolated_heatmap_btn.clicked.connect(
            self._create_region_isolated_heatmaps
        )
        isolation_layout.addWidget(self._create_region_isolated_heatmap_btn)

        layout.addWidget(isolation_group)

        self._tools_status_label = QLabel("")
        self._tools_status_label.setWordWrap(True)
        layout.addWidget(self._tools_status_label)

        layout.addStretch()
        self._update_mask_sigma_units_label()
        self._update_tools_controls()

    def _setup_histogram_tab(self, parent: QWidget) -> None:
        """Set up the histogram and thresholding tab."""
        import pyqtgraph as pg

        layout = QVBoxLayout(parent)
        self._histogram_pg = pg

        self._histogram_sources_section = CollapsibleSection(
            "Histogram Sources",
            expanded=True,
        )
        sources_layout = self._histogram_sources_section.content_layout()
        self._histogram_layer_list = QListWidget()
        self._histogram_layer_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        self._histogram_layer_list.itemSelectionChanged.connect(
            self._on_histogram_layer_selection_changed
        )
        sources_layout.addWidget(self._histogram_layer_list)
        self._histogram_hint_label = QLabel("")
        self._histogram_hint_label.setWordWrap(True)
        sources_layout.addWidget(self._histogram_hint_label)
        layout.addWidget(self._histogram_sources_section)

        self._histogram_plot_section = CollapsibleSection(
            "Intensity Histogram",
            expanded=True,
        )
        plot_layout = self._histogram_plot_section.content_layout()

        self._histogram_include_zero_cb = QCheckBox("Include zero-valued background")
        self._histogram_include_zero_cb.setChecked(False)
        self._histogram_include_zero_cb.toggled.connect(self._update_histogram_plot)
        plot_layout.addWidget(self._histogram_include_zero_cb)

        self._histogram_plot_widget = pg.PlotWidget()
        self._histogram_plot_widget.setMinimumHeight(260)
        self._histogram_plot_item = self._histogram_plot_widget.getPlotItem()
        self._histogram_plot_item.setTitle("Intensity Histogram")
        self._histogram_plot_item.setLabel("bottom", "Intensity")
        self._histogram_plot_item.setLabel("left", "Voxel count")
        self._histogram_plot_item.showGrid(x=True, y=True, alpha=0.15)
        self._histogram_plot_item.getViewBox().setMouseEnabled(x=True, y=True)
        self._histogram_curve_items: list = []
        self._histogram_message_item = None
        self._histogram_plot_legend = None
        self._histogram_lower_line = pg.InfiniteLine(
            angle=90,
            movable=True,
            pen=pg.mkPen(color="#c43c39", width=2),
        )
        self._histogram_lower_line.setBounds((-1_000_000.0, 1_000_000.0))
        self._histogram_lower_line.sigPositionChanged.connect(
            lambda _line: self._on_histogram_bound_line_moved("lower")
        )
        self._histogram_lower_line.sigPositionChangeFinished.connect(
            lambda _line: self._on_histogram_bound_line_move_finished("lower")
        )
        self._histogram_upper_line = pg.InfiniteLine(
            angle=90,
            movable=True,
            pen=pg.mkPen(color="#2f6db2", width=2),
        )
        self._histogram_upper_line.setBounds((-1_000_000.0, 1_000_000.0))
        self._histogram_upper_line.sigPositionChanged.connect(
            lambda _line: self._on_histogram_bound_line_moved("upper")
        )
        self._histogram_upper_line.sigPositionChangeFinished.connect(
            lambda _line: self._on_histogram_bound_line_move_finished("upper")
        )
        plot_layout.addWidget(self._histogram_plot_widget)

        layout.addWidget(self._histogram_plot_section)

        self._histogram_mask_section = CollapsibleSection(
            "Mask Creation",
            expanded=False,
        )
        mask_layout = self._histogram_mask_section.content_layout()

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Create mode:"))
        self._histogram_create_mode_combo = QComboBox()
        self._histogram_create_mode_combo.addItem("Separate layers", "separate")
        self._histogram_create_mode_combo.addItem("Merged layer", "merged")
        self._histogram_create_mode_combo.currentIndexChanged.connect(
            self._update_histogram_controls
        )
        mask_layout.addLayout(mode_row)
        mode_row.addWidget(self._histogram_create_mode_combo)

        lower_row = QHBoxLayout()
        lower_row.addWidget(QLabel("Lower bound:"))
        self._mask_lower_threshold_spin = QDoubleSpinBox()
        self._mask_lower_threshold_spin.setRange(-1_000_000.0, 1_000_000.0)
        self._mask_lower_threshold_spin.setDecimals(4)
        self._mask_lower_threshold_spin.setSingleStep(0.01)
        self._mask_lower_threshold_spin.setValue(1.0)
        self._mask_lower_threshold_spin.valueChanged.connect(
            self._mark_mask_bounds_manual
        )
        lower_row.addWidget(self._mask_lower_threshold_spin)
        mask_layout.addLayout(lower_row)

        bounds_btn_row = QHBoxLayout()
        self._mask_use_otsu_btn = QPushButton("Use Otsu Lower")
        self._mask_use_otsu_btn.clicked.connect(self._fill_mask_lower_bound_from_otsu)
        bounds_btn_row.addWidget(self._mask_use_otsu_btn)

        self._mask_use_contrast_btn = QPushButton("Use Layer Contrast")
        self._mask_use_contrast_btn.clicked.connect(
            self._copy_selected_layer_contrast_to_bounds
        )
        bounds_btn_row.addWidget(self._mask_use_contrast_btn)
        mask_layout.addLayout(bounds_btn_row)

        upper_row = QHBoxLayout()
        self._mask_use_upper_bound_cb = QCheckBox("Use upper bound")
        self._mask_use_upper_bound_cb.toggled.connect(self._on_mask_upper_bound_toggled)
        upper_row.addWidget(self._mask_use_upper_bound_cb)
        self._mask_upper_threshold_spin = QDoubleSpinBox()
        self._mask_upper_threshold_spin.setRange(-1_000_000.0, 1_000_000.0)
        self._mask_upper_threshold_spin.setDecimals(4)
        self._mask_upper_threshold_spin.setSingleStep(0.01)
        self._mask_upper_threshold_spin.setValue(0.0)
        self._mask_upper_threshold_spin.valueChanged.connect(
            self._mark_mask_bounds_manual
        )
        upper_row.addWidget(self._mask_upper_threshold_spin)
        mask_layout.addLayout(upper_row)

        self._create_mask_btn = QPushButton("Create Mask Layer")
        self._create_mask_btn.clicked.connect(self._create_masks_from_heatmaps)
        mask_layout.addWidget(self._create_mask_btn)

        layout.addWidget(self._histogram_mask_section)

        self._histogram_status_label = QLabel("")
        self._histogram_status_label.setWordWrap(True)
        layout.addWidget(self._histogram_status_label)

        layout.addStretch()
        self._on_mask_upper_bound_toggled(False)
        self._update_histogram_controls()
        self._update_histogram_plot()

    def _setup_viz_tab(self, parent: QWidget) -> None:
        """Set up the visualization settings tab."""
        layout = QVBoxLayout(parent)

        # Render mode
        mode_group = QGroupBox("Render Mode")
        mode_layout = QVBoxLayout(mode_group)

        self._render_mode_combo = QComboBox()
        self._render_mode_combo.addItems(["Lines", "Points", "Both"])
        self._render_mode_combo.setCurrentText("Lines")
        mode_layout.addWidget(self._render_mode_combo)

        layout.addWidget(mode_group)

        # Point settings
        point_group = QGroupBox("Point Settings")
        point_layout = QVBoxLayout(point_group)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Size:"))
        self._point_size_spin = QSpinBox()
        self._point_size_spin.setRange(1, 50)
        self._point_size_spin.setValue(5)
        size_row.addWidget(self._point_size_spin)
        point_layout.addLayout(size_row)

        self._color_by_type_cb = QCheckBox("Color by node type")
        self._color_by_type_cb.setChecked(True)
        point_layout.addWidget(self._color_by_type_cb)

        layout.addWidget(point_group)

        # Line settings
        line_group = QGroupBox("Line Settings")
        line_layout = QVBoxLayout(line_group)

        width_row = QHBoxLayout()
        width_row.addWidget(QLabel("Width:"))
        self._line_width_spin = QSpinBox()
        self._line_width_spin.setRange(1, 20)
        self._line_width_spin.setValue(4)
        self._line_width_spin.valueChanged.connect(self._update_line_width)
        width_row.addWidget(self._line_width_spin)
        line_layout.addLayout(width_row)

        layout.addWidget(line_group)

        # 2D Slice Projection settings
        slice_group = QGroupBox("2D Slice Projection")
        slice_layout = QVBoxLayout(slice_group)

        self._show_slice_projection_cb = QCheckBox("Show in 2D slices")
        self._show_slice_projection_cb.setChecked(False)
        self._show_slice_projection_cb.stateChanged.connect(self._toggle_slice_projection)
        slice_layout.addWidget(self._show_slice_projection_cb)

        self._slice_warning_label = QLabel(
            "Warning: Slice navigation is slower when projection is on."
        )
        self._slice_warning_label.setStyleSheet("color: #cc7700; font-style: italic;")
        self._slice_warning_label.setWordWrap(True)
        self._slice_warning_label.setVisible(False)
        slice_layout.addWidget(self._slice_warning_label)

        thickness_row = QHBoxLayout()
        thickness_row.addWidget(QLabel("Slice thickness (μm):"))
        self._slice_thickness_spin = QSpinBox()
        self._slice_thickness_spin.setRange(10, 2500)
        self._slice_thickness_spin.setValue(100)
        self._slice_thickness_spin.valueChanged.connect(self._update_slice_thickness)
        thickness_row.addWidget(self._slice_thickness_spin)
        slice_layout.addLayout(thickness_row)

        layout.addWidget(slice_group)

        # Opacity
        opacity_group = QGroupBox("Opacity")
        opacity_layout = QHBoxLayout(opacity_group)

        self._opacity_slider = QSlider(Qt.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(80)
        opacity_layout.addWidget(self._opacity_slider)

        self._opacity_label = QLabel("80%")
        self._opacity_slider.valueChanged.connect(
            lambda v: self._opacity_label.setText(f"{v}%")
        )
        opacity_layout.addWidget(self._opacity_label)

        layout.addWidget(opacity_group)

        layout.addStretch()

    def _setup_reference_tab(self, parent: QWidget) -> None:
        """Set up the reference layers tab."""
        layout = QVBoxLayout(parent)

        # Template
        template_group = QGroupBox("Reference Template")
        template_layout = QVBoxLayout(template_group)

        self._show_template_cb = QCheckBox("Show template")
        self._show_template_cb.setChecked(False)
        self._show_template_cb.setToolTip(
            "Load and show the Allen reference template on demand."
        )
        self._show_template_cb.stateChanged.connect(self._toggle_template)
        template_layout.addWidget(self._show_template_cb)

        template_opacity_row = QHBoxLayout()
        template_opacity_row.addWidget(QLabel("Opacity:"))
        self._template_opacity_slider = QSlider(Qt.Horizontal)
        self._template_opacity_slider.setRange(0, 100)
        self._template_opacity_slider.setValue(30)
        self._template_opacity_slider.valueChanged.connect(self._update_template_opacity)
        template_opacity_row.addWidget(self._template_opacity_slider)
        template_layout.addLayout(template_opacity_row)

        layout.addWidget(template_group)

        # Brain outline
        outline_group = QGroupBox("Brain Outline")
        outline_layout = QVBoxLayout(outline_group)

        self._show_outline_cb = QCheckBox("Show brain outline")
        self._show_outline_cb.setChecked(False)
        self._show_outline_cb.stateChanged.connect(self._toggle_outline)
        outline_layout.addWidget(self._show_outline_cb)

        layout.addWidget(outline_group)

        # Region meshes
        mesh_group = QGroupBox("Region Meshes")
        mesh_layout = QVBoxLayout(mesh_group)

        self._show_region_meshes_cb = QCheckBox("Show selected region meshes")
        self._show_region_meshes_cb.setChecked(False)
        self._show_region_meshes_cb.stateChanged.connect(self._toggle_region_meshes)
        mesh_layout.addWidget(self._show_region_meshes_cb)

        mesh_opacity_row = QHBoxLayout()
        mesh_opacity_row.addWidget(QLabel("Opacity:"))
        self._mesh_opacity_slider = QSlider(Qt.Horizontal)
        self._mesh_opacity_slider.setRange(0, 100)
        self._mesh_opacity_slider.setValue(30)
        mesh_opacity_row.addWidget(self._mesh_opacity_slider)
        mesh_layout.addLayout(mesh_opacity_row)

        layout.addWidget(mesh_group)

        # Region Segmentation (2D)
        seg_group = QGroupBox("Region Segmentation (2D)")
        seg_layout = QVBoxLayout(seg_group)

        self._show_region_seg_cb = QCheckBox("Show selected region segmentation")
        self._show_region_seg_cb.setChecked(False)
        self._show_region_seg_cb.stateChanged.connect(self._toggle_region_segmentation)
        seg_layout.addWidget(self._show_region_seg_cb)

        seg_opacity_row = QHBoxLayout()
        seg_opacity_row.addWidget(QLabel("Opacity:"))
        self._seg_opacity_slider = QSlider(Qt.Horizontal)
        self._seg_opacity_slider.setRange(0, 100)
        self._seg_opacity_slider.setValue(30)
        self._seg_opacity_slider.valueChanged.connect(self._update_seg_opacity)
        seg_opacity_row.addWidget(self._seg_opacity_slider)
        seg_layout.addLayout(seg_opacity_row)

        layout.addWidget(seg_group)

        layout.addStretch()

    def _load_parquet(self) -> None:
        """Open file dialog and load Parquet file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Parquet File",
            "",
            "Parquet Files (*.parquet);;All Files (*)",
        )

        if not filepath:
            return

        try:
            self._db = NeuronDatabase(filepath)
            self._file_label.setText(Path(filepath).name)

            # Update stats
            stats = self._db.get_statistics()
            self._stats_label.setText(
                f"Nodes: {stats['n_nodes']:,} | "
                f"Files: {stats['n_files']:,} | "
                f"Subjects: {stats['n_subjects']:,} | "
                f"Regions: {stats['n_regions']:,}"
            )

            self._set_region_query_buttons_enabled(True)
            self._analysis_tab.set_database(self._db)
            self._regions_status_label.setText("")
            logger.info(f"Loaded Parquet file: {filepath}")

        except Exception as e:
            logger.error(f"Failed to load Parquet file: {e}")
            self._file_label.setText(f"Error: {e}")

    def _load_atlas(self) -> None:
        """Load the selected BrainGlobe atlas."""
        atlas_name = self._atlas_combo.currentText()

        if self._cached_atlas_autoload_running():
            self._atlas_status_label.setText(
                f"Atlas: Loading cached {atlas_name} in background..."
            )
            return

        self._atlas_status_label.setText(f"Atlas: Loading {atlas_name}...")
        # Force UI update
        self._atlas_status_label.repaint()

        try:
            with startup_timing(logger, "load_atlas", atlas=atlas_name) as timing:
                with startup_timing(
                    logger,
                    "load_atlas_phase",
                    phase="BrainGlobeAtlas",
                    atlas=atlas_name,
                ) as atlas_timing:
                    self._atlas = BrainGlobeAtlas(
                        atlas_name,
                        check_latest=False,
                    )
                    structure_count = len(self._atlas.structures)
                    atlas_timing.set(structures=structure_count)
                    timing.set(structures=structure_count)

                self._apply_loaded_atlas(self._atlas, atlas_name)
                logger.info(f"Loaded atlas: {atlas_name}")

        except Exception as e:
            logger.error(f"Failed to load atlas: {e}")
            self._atlas_status_label.setText(f"Atlas: Error - {e}")
            self._update_mask_sigma_units_label()
            self._update_point_import_controls()

    def _apply_loaded_atlas(self, atlas, atlas_name: str) -> None:
        """Store a loaded atlas and refresh atlas-dependent UI."""
        self._atlas = atlas

        selectors = []
        for attr_name in (
            "_whole_parquet_region_selector",
            "_current_table_region_selector",
            "_region_selector",
            "_tools_region_selector",
        ):
            selector = getattr(self, attr_name, None)
            if selector is None or any(
                existing is selector for _name, existing in selectors
            ):
                continue
            selectors.append((attr_name, selector))

        for attr_name, selector in selectors:
            set_atlas = getattr(selector, "set_atlas", None)
            if not callable(set_atlas):
                continue
            with startup_timing(
                logger,
                "load_atlas_selector",
                selector=attr_name,
                atlas=atlas_name,
            ) as selector_timing:
                set_atlas(atlas)
                selector_timing.set(
                    items=len(getattr(selector, "_items_by_id", {}))
                )
        self._atlas_status_label.setText(
            f"Atlas: {atlas_name} ({len(atlas.structures)} structures)"
        )
        with startup_timing(
            logger,
            "load_atlas_phase",
            phase="analysis_tab_set_atlas",
            atlas=atlas_name,
        ):
            self._analysis_tab.set_atlas(atlas)
        with startup_timing(
            logger,
            "load_atlas_phase",
            phase="update_mask_sigma_units_label",
            atlas=atlas_name,
        ):
            self._update_mask_sigma_units_label()
        with startup_timing(
            logger,
            "load_atlas_phase",
            phase="refresh_heatmap_layer_list",
            atlas=atlas_name,
        ):
            self._refresh_heatmap_layer_list()
        with startup_timing(
            logger,
            "load_atlas_phase",
            phase="refresh_histogram_layer_list",
            atlas=atlas_name,
        ):
            self._refresh_histogram_layer_list()
        with startup_timing(
            logger,
            "load_atlas_phase",
            phase="refresh_mask_layer_options",
            atlas=atlas_name,
        ):
            self._refresh_mask_layer_options()
        with startup_timing(
            logger,
            "load_atlas_phase",
            phase="update_point_import_controls",
            atlas=atlas_name,
        ):
            self._update_point_import_controls()

    def _cached_atlas_autoload_running(self) -> bool:
        """Return whether a cached atlas auto-load worker is active."""
        thread = getattr(self, "_cached_atlas_thread", None)
        is_running = getattr(thread, "isRunning", None)
        return bool(thread is not None and callable(is_running) and is_running())

    def _set_template_checkbox_checked(
        self,
        checked: bool,
        *,
        emit_signal: bool,
    ) -> None:
        """Set the template checkbox without optionally emitting stateChanged."""
        checkbox = getattr(self, "_show_template_cb", None)
        if checkbox is None:
            return

        if emit_signal or not hasattr(checkbox, "blockSignals"):
            checkbox.setChecked(checked)
            return

        previous = checkbox.blockSignals(True)
        try:
            checkbox.setChecked(checked)
        finally:
            checkbox.blockSignals(previous)

    def _set_template_checkbox_enabled(self, enabled: bool) -> None:
        """Enable or disable the template checkbox when available."""
        checkbox = getattr(self, "_show_template_cb", None)
        if checkbox is not None and hasattr(checkbox, "setEnabled"):
            checkbox.setEnabled(enabled)

    def _start_cached_template_autoload(self) -> None:
        """Auto-load and show the reference template only when the atlas is cached."""
        if self._atlas is not None or self._cached_atlas_autoload_running():
            return

        from ..workers import CachedAtlasLoadWorker, cached_brainglobe_atlas_dir

        atlas_name = self._atlas_combo.currentText()
        with startup_timing(
            logger,
            "cached_template_autoload",
            atlas=atlas_name,
        ) as timing:
            atlas_dir = cached_brainglobe_atlas_dir(atlas_name)
            timing.set(cached=atlas_dir is not None, atlas_dir=atlas_dir)
            if atlas_dir is None:
                self._set_template_checkbox_checked(False, emit_signal=False)
                return

            self._show_template_after_cached_atlas_load = True
            self._set_template_checkbox_checked(True, emit_signal=False)
            self._set_template_checkbox_enabled(False)
            self._atlas_status_label.setText(
                f"Atlas: Loading cached {atlas_name} in background..."
            )

            thread = QThread()
            worker = CachedAtlasLoadWorker(atlas_name, atlas_dir)
            self._cached_atlas_thread = thread
            self._cached_atlas_worker = worker
            worker.moveToThread(thread)

            thread.started.connect(worker.run)
            worker.finished.connect(self._on_cached_template_atlas_loaded)
            worker.error.connect(self._on_cached_template_atlas_error)
            worker.finished.connect(thread.quit)
            worker.error.connect(thread.quit)
            thread.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(
                lambda: self._cleanup_cached_atlas_thread(thread, worker)
            )
            thread.start()

    def _on_cached_template_atlas_loaded(self, atlas) -> None:
        """Apply a cached atlas loaded in the background and show the template."""
        atlas_name = str(
            getattr(atlas, "atlas_name", None)
            or self._atlas_combo.currentText()
        )
        try:
            with startup_timing(
                logger,
                "cached_template_autoload_apply",
                atlas=atlas_name,
            ):
                if self._atlas is None:
                    self._apply_loaded_atlas(atlas, atlas_name)
                self._set_template_checkbox_enabled(True)
                if (
                    self._show_template_after_cached_atlas_load
                    or self._show_template_cb.isChecked()
                ):
                    self._toggle_template(True)
                    self._set_template_checkbox_checked(
                        True,
                        emit_signal=False,
                    )
        except Exception as e:
            logger.error("Failed to show cached atlas template: %s", e)
            self._atlas_status_label.setText(f"Atlas: Template error - {e}")
            self._set_template_checkbox_checked(False, emit_signal=False)
            self._set_template_checkbox_enabled(True)
        finally:
            self._show_template_after_cached_atlas_load = False

    def _on_cached_template_atlas_error(self, error_msg: str) -> None:
        """Handle cached atlas auto-load failure."""
        self._show_template_after_cached_atlas_load = False
        self._set_template_checkbox_checked(False, emit_signal=False)
        self._set_template_checkbox_enabled(True)
        self._atlas_status_label.setText(
            f"Atlas: Cached auto-load failed - {error_msg}"
        )
        logger.error("Cached atlas auto-load failed: %s", error_msg)

    def _cleanup_cached_atlas_thread(
        self,
        thread: QThread,
        worker: object,
    ) -> None:
        """Release cached atlas worker objects after the thread stops."""
        if self._cached_atlas_thread is thread:
            self._cached_atlas_thread = None
        if self._cached_atlas_worker is worker:
            self._cached_atlas_worker = None

    def _open_point_parquet(self) -> None:
        """Open file dialog and preview a standardized point Parquet."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Point Parquet File",
            "",
            "Parquet Files (*.parquet);;All Files (*)",
        )
        if not filepath:
            return

        self._load_point_parquet_file(filepath)

    def _append_point_file(self) -> None:
        """Prompt for a point CSV or point Parquet and save a combined Parquet."""

        input_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Point File to Append",
            "",
            "Point Files (*.csv *.parquet);;CSV Files (*.csv);;Parquet Files (*.parquet);;All Files (*)",
        )
        if not input_path:
            return

        input_suffix = Path(input_path).suffix.lower()
        mapping_path: str | None = None
        if input_suffix == ".csv":
            try:
                _standardized, source_description = load_and_standardize_point_csv(input_path)
            except PointImportError:
                mapping_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select Mapping JSON",
                    "",
                    "JSON Files (*.json);;All Files (*)",
                )
                if not mapping_path:
                    return
            else:
                self._point_import_status_label.setText(
                    f"Detected {source_description} in {Path(input_path).name}; "
                    "no mapping JSON needed."
                )
        elif input_suffix == ".parquet":
            self._point_import_status_label.setText(
                f"Selected point Parquet {Path(input_path).name}; "
                "it must exactly match the target schema."
            )
        else:
            self._point_import_status_label.setText(
                "Select a point CSV or point Parquet file to append."
            )
            return

        parquet_path = self._select_point_parquet_source_for_append()
        if parquet_path is None:
            return

        if (
            input_suffix == ".parquet"
            and Path(input_path).resolve() == Path(parquet_path).resolve()
        ):
            self._point_import_status_label.setText(
                "Choose a different point Parquet to append than the target file."
            )
            return

        default_output_name = (
            f"{Path(parquet_path).stem}_with_{Path(input_path).stem}.parquet"
        )
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save New Point Parquet File",
            str(Path(parquet_path).with_name(default_output_name)),
            "Parquet Files (*.parquet);;All Files (*)",
        )
        if not output_path:
            return

        if Path(output_path).resolve() == Path(parquet_path).resolve():
            self._point_import_status_label.setText(
                "Choose a new output Parquet path instead of overwriting the source file."
            )
            return

        self._start_point_file_append(
            input_path,
            mapping_path,
            parquet_path,
            output_path,
        )

    def _select_point_parquet_source_for_append(self) -> str | None:
        """Return the loaded point Parquet when available, otherwise prompt for one."""

        if self._point_parquet_path:
            loaded_path = Path(self._point_parquet_path)
            if loaded_path.exists():
                self._point_import_status_label.setText(
                    f"Using loaded point Parquet {loaded_path.name} as the append source. "
                    "Choose where to save the combined file."
                )
                return str(loaded_path)

            self._point_import_status_label.setText(
                f"Loaded point Parquet {loaded_path.name} is no longer available. "
                "Select a source Parquet file."
            )

        parquet_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Existing Point Parquet File",
            "",
            "Parquet Files (*.parquet);;All Files (*)",
        )
        return parquet_path or None

    def _convert_point_csv_from_directory(self) -> None:
        """Create a point Parquet from all top-level CSV files in a directory."""

        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory of Point CSV Files",
        )
        if not directory:
            return

        csv_paths = sorted(
            str(path)
            for path in Path(directory).glob("*.csv")
            if path.is_file()
        )
        if not csv_paths:
            self._point_import_status_label.setText("No CSV files found in directory.")
            return

        default_name = f"{Path(directory).name}_points.parquet"
        self._prompt_point_csv_output_and_convert(csv_paths, default_name)

    def _convert_point_csv_from_files(self) -> None:
        """Create a point Parquet from one or more selected CSV files."""

        csv_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Point CSV Files",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not csv_paths:
            return

        default_name = (
            f"{Path(csv_paths[0]).stem}.parquet"
            if len(csv_paths) == 1
            else "points.parquet"
        )
        self._prompt_point_csv_output_and_convert(csv_paths, default_name)

    def _prompt_point_csv_output_and_convert(
        self,
        csv_paths: list[str],
        default_name: str,
    ) -> None:
        """Ask for mapping if needed, then ask for output path and start conversion."""

        mapping_path = self._select_point_csv_mapping_if_needed(csv_paths)
        if mapping_path is False:
            return

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Point Parquet File",
            default_name,
            "Parquet Files (*.parquet)",
        )
        if not output_path:
            return

        self._start_point_csv_conversion(
            csv_paths,
            output_path,
            mapping_path if isinstance(mapping_path, str) else None,
        )

    def _select_point_csv_mapping_if_needed(
        self,
        csv_paths: list[str],
    ) -> str | bool | None:
        """Return a mapping path if autodetect fails for any selected CSV."""

        for csv_path in csv_paths:
            try:
                _standardized, _source = load_and_standardize_point_csv(csv_path)
            except PointImportError:
                mapping_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select Mapping JSON",
                    "",
                    "JSON Files (*.json);;All Files (*)",
                )
                return mapping_path if mapping_path else False

        if len(csv_paths) == 1:
            self._point_import_status_label.setText(
                f"Detected point CSV headers in {Path(csv_paths[0]).name}; "
                "no mapping JSON needed."
            )
        else:
            self._point_import_status_label.setText(
                "Detected point CSV headers automatically; no mapping JSON needed."
            )
        return None

    def _point_preview_key_from_row(self, row: int) -> tuple[str, str] | None:
        """Return the selected label/origin key for a preview table row."""

        item = self._point_preview_table.item(row, _POINT_PREVIEW_LABEL_COLUMN)
        if item is None:
            return None

        key = item.data(Qt.UserRole)
        if isinstance(key, tuple) and len(key) == 2:
            return (str(key[0]), str(key[1]))

        label = item.text().strip()
        origin_item = self._point_preview_table.item(row, _POINT_PREVIEW_ORIGIN_COLUMN)
        origin_csv = origin_item.text().strip() if origin_item is not None else ""
        if not label:
            return None
        return (label, origin_csv or POINT_PARQUET_ORIGIN_NOT_RECORDED)

    def _selected_point_preview_keys(self) -> list[tuple[str, str]]:
        """Return the selected label/origin pairs from the preview table."""

        rows = sorted({index.row() for index in self._point_preview_table.selectedIndexes()})
        selected: list[tuple[str, str]] = []
        for row in rows:
            key = self._point_preview_key_from_row(row)
            if key is not None:
                selected.append(key)
        return selected

    def _update_point_import_controls(self) -> None:
        """Enable or disable point import actions based on current selection."""

        operation_running = self._point_operation_running()
        ready = (
            bool(self._point_parquet_path)
            and bool(self._selected_point_preview_keys())
            and self._atlas is not None
            and not operation_running
        )
        self._import_selected_point_heatmaps_btn.setEnabled(ready)
        self._open_point_parquet_btn.setEnabled(not operation_running)
        self._append_point_file_btn.setEnabled(not operation_running)
        self._create_point_from_directory_btn.setEnabled(not operation_running)
        self._create_point_from_files_btn.setEnabled(not operation_running)

    def _point_operation_running(self) -> bool:
        """Return whether a point conversion or append worker is currently active."""

        return bool(
            (self._point_append_thread is not None and self._point_append_thread.isRunning())
            or (
                self._point_convert_thread is not None
                and self._point_convert_thread.isRunning()
            )
        )

    def _populate_point_parquet_preview(self, summary_df) -> None:
        """Populate the point parquet preview table from a grouped summary."""

        self._point_preview_counts.clear()
        was_blocked = self._point_preview_table.blockSignals(True)
        try:
            self._point_preview_table.clearContents()
            self._point_preview_table.setRowCount(len(summary_df))
            for row_index, row in enumerate(summary_df.itertuples(index=False)):
                label = str(row.label)
                origin_csv = str(row.origin_csv)
                point_count = int(row.point_count)
                key = (label, origin_csv)
                self._point_preview_counts[key] = point_count

                label_item = QTableWidgetItem(label)
                label_item.setData(Qt.UserRole, key)
                origin_item = QTableWidgetItem(origin_csv)
                count_item = QTableWidgetItem(f"{point_count:,}")
                count_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                count_item.setData(Qt.UserRole, point_count)

                self._point_preview_table.setItem(
                    row_index,
                    _POINT_PREVIEW_LABEL_COLUMN,
                    label_item,
                )
                self._point_preview_table.setItem(
                    row_index,
                    _POINT_PREVIEW_ORIGIN_COLUMN,
                    origin_item,
                )
                self._point_preview_table.setItem(
                    row_index,
                    _POINT_PREVIEW_COUNT_COLUMN,
                    count_item,
                )
        finally:
            self._point_preview_table.blockSignals(was_blocked)

        self._point_preview_table.clearSelection()
        self._update_point_import_controls()

    def _load_point_parquet_file(
        self,
        filepath: str,
        *,
        success_message: str | None = None,
    ) -> None:
        """Load standardized point Parquet metadata into the preview table."""

        try:
            summary_df = summarize_standard_point_parquet_groups(filepath)
        except PointImportError as e:
            logger.error(f"Failed to load point Parquet: {e}")
            self._point_import_status_label.setText(f"Error: {e}")
            return

        self._point_parquet_path = filepath
        self._point_parquet_has_origin_csv = bool(
            summary_df.attrs.get("has_origin_csv", False)
        )
        self._point_file_label.setText(Path(filepath).name)

        if summary_df.empty:
            self._point_preview_counts.clear()
            self._point_preview_table.clearContents()
            self._point_preview_table.setRowCount(0)
            self._update_point_import_controls()
            self._point_import_status_label.setText("No points found in file.")
            return

        self._populate_point_parquet_preview(summary_df)
        total_points = int(summary_df["point_count"].sum())
        if success_message is None:
            action_message = "Select rows and click Import Selected Heatmaps."
            if self._atlas is None:
                action_message = "Load an atlas to enable Import Selected Heatmaps."
            success_message = (
                f"Loaded {total_points:,} point(s) across {len(summary_df)} "
                f"label/origin row(s). {action_message}"
            )
        self._point_import_status_label.setText(success_message)
        logger.info(
            "Loaded point Parquet preview %s with %d selectable row(s) and %d points",
            filepath,
            len(summary_df),
            total_points,
        )

    def _start_point_file_append(
        self,
        input_path: str,
        mapping_path: str | None,
        parquet_path: str,
        output_path: str,
    ) -> None:
        """Launch the background point-file append worker."""

        from ..workers import AppendPointFileWorker

        self._point_append_progress.setVisible(True)
        self._point_append_progress.setRange(0, 0)
        self._point_import_status_label.setText(
            f"Saving {Path(output_path).name} from {Path(parquet_path).name} + {Path(input_path).name}..."
        )

        thread = QThread()
        worker = AppendPointFileWorker(
            input_path,
            mapping_path,
            parquet_path,
            output_path,
        )
        self._point_append_thread = thread
        self._point_append_worker = worker
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_point_append_progress)
        worker.finished.connect(self._on_point_append_finished)
        worker.error.connect(self._on_point_append_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda: self._cleanup_point_append_thread(thread, worker)
        )

        self._update_point_import_controls()
        thread.start()

    def _start_point_csv_conversion(
        self,
        csv_paths: list[str],
        output_path: str,
        mapping_path: str | None,
    ) -> None:
        """Launch the background point-CSV conversion worker."""

        from ..workers import ConvertPointCSVWorker

        self._point_append_progress.setVisible(True)
        self._point_append_progress.setRange(0, max(1, len(csv_paths)))
        self._point_append_progress.setValue(0)
        self._point_import_status_label.setText(
            f"Creating {Path(output_path).name} from {len(csv_paths)} point CSV file(s)..."
        )

        thread = QThread()
        worker = ConvertPointCSVWorker(
            csv_paths,
            output_path,
            mapping_path,
        )
        self._point_convert_thread = thread
        self._point_convert_worker = worker
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_point_convert_progress)
        worker.finished.connect(self._on_point_convert_finished)
        worker.error.connect(self._on_point_convert_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda: self._cleanup_point_convert_thread(thread, worker)
        )

        self._update_point_import_controls()
        thread.start()

    def _on_point_convert_progress(self, message: str, current: int, total: int) -> None:
        """Handle point-CSV conversion progress updates."""

        self._point_append_progress.setRange(0, max(1, total))
        self._point_append_progress.setValue(current)
        self._point_import_status_label.setText(message)

    def _on_point_convert_finished(self, parquet_path: str, summary: object) -> None:
        """Handle point-CSV conversion completion."""

        self._point_append_progress.setVisible(False)
        self._point_append_progress.setRange(0, 1)
        self._point_append_progress.setValue(0)
        message = (
            f"Created {Path(parquet_path).name} from {summary.processed_files} CSV "
            f"file(s) with {summary.rows_written:,} point(s)."
        )
        self._load_point_parquet_file(parquet_path, success_message=message)
        self._update_point_import_controls()
        logger.info(
            "Created point parquet %s from %d CSV file(s) with %d point(s)",
            parquet_path,
            summary.processed_files,
            summary.rows_written,
        )

    def _on_point_convert_error(self, error_msg: str) -> None:
        """Handle point-CSV conversion failure."""

        self._point_append_progress.setVisible(False)
        self._point_append_progress.setRange(0, 1)
        self._point_append_progress.setValue(0)
        self._point_import_status_label.setText(f"Error: {error_msg}")
        self._update_point_import_controls()
        logger.error(f"Point CSV conversion failed: {error_msg}")

    def _on_point_append_progress(self, message: str, _current: int, _total: int) -> None:
        """Handle point-file append progress updates."""

        self._point_import_status_label.setText(message)

    def _on_point_append_finished(self, parquet_path: str, summary: object) -> None:
        """Handle point-file append completion."""

        self._point_append_progress.setVisible(False)
        self._point_append_progress.setRange(0, 1)
        self._point_append_progress.setValue(0)
        message = (
            f"Saved {Path(parquet_path).name} with {summary.appended_rows:,} "
            f"added point(s) ({summary.total_rows:,} total)."
        )
        self._load_point_parquet_file(parquet_path, success_message=message)
        self._update_point_import_controls()
        logger.info(
            "Saved point parquet %s with %d added point(s) (%d total)",
            parquet_path,
            summary.appended_rows,
            summary.total_rows,
        )

    def _on_point_append_error(self, error_msg: str) -> None:
        """Handle point-file append failure."""

        self._point_append_progress.setVisible(False)
        self._point_append_progress.setRange(0, 1)
        self._point_append_progress.setValue(0)
        self._point_import_status_label.setText(f"Error: {error_msg}")
        self._update_point_import_controls()
        logger.error(f"Point CSV append failed: {error_msg}")

    def _cleanup_point_convert_thread(self, thread: QThread, worker: object) -> None:
        """Release point conversion worker objects after the thread stops."""

        if self._point_convert_thread is thread:
            self._point_convert_thread = None
        if self._point_convert_worker is worker:
            self._point_convert_worker = None
        self._update_point_import_controls()

    def _cleanup_point_append_thread(self, thread: QThread, worker: object) -> None:
        """Release point append worker objects after the thread stops."""

        if self._point_append_thread is thread:
            self._point_append_thread = None
        if self._point_append_worker is worker:
            self._point_append_worker = None
        self._update_point_import_controls()

    def _import_selected_point_heatmaps(self) -> None:
        """Create heatmap layers for the selected preview rows."""
        from napari.utils.colormaps import Colormap

        if self._atlas is None:
            message = "Load an atlas before importing point Parquet."
            self._point_import_status_label.setText(message)
            show_warning(message)
            return

        if not self._point_parquet_path:
            self._point_import_status_label.setText(
                "Load a point Parquet before importing heatmaps."
            )
            return

        selected_keys = self._selected_point_preview_keys()
        if not selected_keys:
            self._point_import_status_label.setText(
                "Select at least one label/origin row."
            )
            return

        try:
            points_df = load_standard_point_parquet_selection(
                self._point_parquet_path,
                selected_keys,
            )
        except PointImportError as e:
            logger.error(f"Failed to load selected point Parquet rows: {e}")
            self._point_import_status_label.setText(f"Error: {e}")
            return

        if points_df.empty:
            self._point_import_status_label.setText(
                "No points matched the selected rows."
            )
            return

        validation_summary = validate_point_metadata_against_atlas(
            points_df,
            self._atlas,
        )
        if validation_summary.has_mismatches:
            show_warning(format_atlas_validation_summary(validation_summary))

        opacity = self._opacity_slider.value() / 100.0
        group_columns = (
            ("label", "origin_csv")
            if self._point_parquet_has_origin_csv
            else ("label",)
        )
        grouped_heatmaps = build_grouped_point_heatmap_volumes(
            points_df,
            self._atlas,
            group_columns,
        )
        columns = list(points_df.columns)
        created_layers = 0

        for color_idx, (label, origin_csv) in enumerate(selected_keys):
            group_key = (
                (label, origin_csv)
                if self._point_parquet_has_origin_csv
                else (label,)
            )
            volume = grouped_heatmaps.get(group_key)
            if volume is None:
                continue

            layer_name = _point_heatmap_layer_name(
                label,
                origin_csv,
                include_origin=self._point_parquet_has_origin_csv,
            )
            for layer in list(self.viewer.layers):
                if layer.name == layer_name:
                    self.viewer.layers.remove(layer)

            nonzero_voxels = int((volume > 0).sum())
            rgba = _point_heatmap_color(color_idx)
            colormap = Colormap(
                colors=[[0.0, 0.0, 0.0, 0.0], list(rgba)],
                name=f"point_heatmap_{color_idx}",
            )
            self.viewer.add_image(
                volume,
                name=layer_name,
                blending="additive",
                rendering="mip",
                colormap=colormap,
                opacity=opacity,
                metadata={
                    "source_path": self._point_parquet_path,
                    "label": label,
                    "origin_csv": origin_csv,
                    "point_count": self._point_preview_counts.get((label, origin_csv), 0),
                    "nonzero_voxels": nonzero_voxels,
                    "columns": columns,
                    "color": rgba,
                    "heatmap_source": True,
                    "heatmap_native_grid": True,
                    "atlas_name": getattr(self._atlas, "atlas_name", None),
                    "heatmap_kind": "point_import",
                },
            )
            created_layers += 1

        message = (
            f"Imported {len(points_df):,} selected point(s) into {created_layers} "
            "heatmap layer(s)."
        )
        if validation_summary.has_mismatches:
            message += (
                f" Atlas validation found "
                f"{validation_summary.total_mismatched_rows} mismatched row(s)."
            )
        self._point_import_status_label.setText(message)
        logger.info(
            "Imported %d selected point heatmaps from %s covering %d points",
            created_layers,
            self._point_parquet_path,
            len(points_df),
        )
        self._refresh_heatmap_layer_list()
        self._refresh_histogram_layer_list()
        self._refresh_mask_layer_options()

    def _connect_layer_events(self) -> None:
        """Refresh tool and mask selectors when viewer layers change."""
        layer_events = getattr(getattr(self.viewer, "layers", None), "events", None)
        if layer_events is None:
            return
        for event_name in ("inserted", "removed", "reordered"):
            signal = getattr(layer_events, event_name, None)
            if signal is not None:
                signal.connect(self._on_viewer_layers_changed)

    def _on_viewer_layers_changed(self, _event=None) -> None:
        """Refresh UI that depends on viewer layers."""
        self._refresh_heatmap_layer_list()
        self._refresh_histogram_layer_list()
        self._refresh_mask_layer_options()
        self._update_tools_controls()
        self._update_histogram_controls()
        sync_heatmaps = getattr(self, "_sync_neuron_table_heatmap_membership", None)
        if callable(sync_heatmaps):
            sync_heatmaps()
        refresh_manual_heatmaps = getattr(self, "_refresh_manual_heatmap_combo", None)
        if callable(refresh_manual_heatmaps):
            refresh_manual_heatmaps()

    def _iter_viewer_layers(self) -> list:
        """Return current viewer layers as a list."""
        try:
            return list(self.viewer.layers)
        except Exception:
            return []

    def _manual_heatmap_layers(self) -> list:
        """Return selected-neuron heatmap layers created from the Data tab."""
        return [
            layer for layer in self._iter_viewer_layers()
            if _layer_metadata(layer).get("heatmap_kind") == "selected_neurons"
        ]

    @staticmethod
    def _normalise_layer_file_ids(file_ids: object) -> tuple[object, ...]:
        """Return layer file IDs as a tuple without splitting string IDs."""
        if file_ids is None:
            return ()
        if isinstance(file_ids, (str, bytes)):
            return (file_ids,)
        try:
            return tuple(file_ids)
        except TypeError:
            return (file_ids,)

    def _current_selected_neuron_heatmap_layers_by_file_id(
        self,
    ) -> dict[object, tuple[str, ...]]:
        """Return Data-tab selected-neuron heatmap layer names by file ID."""
        layer_names_by_file_id: dict[object, list[str]] = {}
        for layer in self._manual_heatmap_layers():
            layer_name = str(getattr(layer, "name", ""))
            if not layer_name:
                continue

            metadata = _layer_metadata(layer)
            for file_id in self._normalise_layer_file_ids(metadata.get("file_ids", [])):
                names = layer_names_by_file_id.setdefault(file_id, [])
                if layer_name not in names:
                    names.append(layer_name)

        return {
            file_id: tuple(layer_names)
            for file_id, layer_names in layer_names_by_file_id.items()
        }

    def _sync_neuron_table_heatmap_membership(self) -> None:
        """Sync the neuron table Heatmap column from Data-tab heatmap layers."""
        table = getattr(self, "_neuron_table", None)
        setter = getattr(table, "set_heatmap_layers_by_file_id", None)
        if not callable(setter):
            return
        setter(self._current_selected_neuron_heatmap_layers_by_file_id())

    def _manual_heatmap_combo_options(self) -> list[tuple[str, tuple[object, ...]]]:
        """Return manual heatmap dropdown options as layer-name/file-ID tuples."""
        options: list[tuple[str, tuple[object, ...]]] = []
        for layer in self._manual_heatmap_layers():
            layer_name = str(getattr(layer, "name", ""))
            if not layer_name:
                continue
            file_ids = self._normalise_layer_file_ids(
                _layer_metadata(layer).get("file_ids", [])
            )
            options.append((layer_name, file_ids))
        return options

    def _manual_heatmap_combo_data(self) -> tuple[str, tuple[object, ...]] | None:
        """Return the currently selected manual heatmap dropdown payload."""
        combo = getattr(self, "_manual_heatmap_combo", None)
        if combo is None:
            return None
        current_data = getattr(combo, "currentData", None)
        data = current_data() if callable(current_data) else None
        if (
            isinstance(data, tuple)
            and len(data) == 2
            and isinstance(data[0], str)
        ):
            return data[0], self._normalise_layer_file_ids(data[1])
        return None

    def _refresh_manual_heatmap_combo(self) -> None:
        """Refresh the Data-tab manual heatmap selector."""
        combo = getattr(self, "_manual_heatmap_combo", None)
        if combo is None:
            return

        previous = self._manual_heatmap_combo_data()
        previous_name = previous[0] if previous is not None else None
        options = self._manual_heatmap_combo_options()

        signals_blocked = combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem(_MANUAL_HEATMAP_ALL_LABEL, None)
            selected_index = 0
            for layer_name, file_ids in options:
                combo.addItem(layer_name, (layer_name, tuple(file_ids)))
                if layer_name == previous_name:
                    selected_index = combo.count() - 1
            combo.setCurrentIndex(selected_index)
        finally:
            combo.blockSignals(signals_blocked)

        set_enabled = getattr(combo, "setEnabled", None)
        if callable(set_enabled):
            set_enabled(bool(options))
        if previous_name is not None and selected_index == 0:
            self._on_manual_heatmap_selection_changed()

    def _on_manual_heatmap_selection_changed(self, _index: int = 0) -> None:
        """Filter table rows for the chosen manual heatmap layer."""
        self._apply_neuron_table_filters()

    def _selected_manual_heatmap_file_ids(self) -> tuple[object, ...] | None:
        """Return the currently selected manual heatmap file IDs, if any."""
        data = self._manual_heatmap_combo_data()
        return None if data is None else data[1]

    def _apply_neuron_table_filters(self) -> None:
        """Apply current Data-tab row filters to the neuron table."""
        table = getattr(self, "_neuron_table", None)
        if table is None:
            return

        selection = self._selected_cluster_filter()
        heatmap_file_ids = self._selected_manual_heatmap_file_ids()
        applier = getattr(table, "apply_filters", None)
        if callable(applier):
            applier(selection, heatmap_file_ids)
            return

        cluster_applier = getattr(table, "apply_cluster_filter", None)
        if callable(cluster_applier):
            cluster_applier(selection)

    def _current_atlas_name(self) -> str | None:
        """Return the currently loaded atlas name, if any."""
        if self._atlas is None:
            return None
        return str(getattr(self._atlas, "atlas_name", "")) or None

    def _selected_heatmap_running(self) -> bool:
        """Return whether a selected-neuron heatmap worker is active."""
        thread = getattr(self, "_selected_heatmap_thread", None)
        return bool(
            thread is not None
            and thread.isRunning()
        )

    def _update_selected_neuron_heatmap_controls(self) -> None:
        """Enable or disable the selected-neuron heatmap action."""
        if hasattr(self, "_add_selected_heatmap_btn"):
            self._add_selected_heatmap_btn.setEnabled(
                not self._selected_heatmap_running()
            )

    def _refresh_neuron_table_summary(self) -> None:
        """Refresh the summary text shown below the neuron table actions."""
        if not hasattr(self, "_neuron_table_summary_label"):
            return
        if not hasattr(self, "_neuron_table"):
            self._neuron_table_summary_label.setText("")
            return

        summary = self._neuron_table.summary()
        counts_line = (
            f"In table: {summary.table_count:,} | "
            f"Added to scene: {summary.added_count:,} | "
            f"Visible: {summary.visible_count:,}"
        )
        if not summary.cluster_counts:
            clusters_line = "Clusters: none"
        else:
            parts = []
            for cluster_id, count in summary.cluster_counts:
                if cluster_id is None:
                    parts.append(f"Unclustered: {count:,}")
                else:
                    parts.append(f"Cluster {cluster_id}: {count:,}")
            clusters_line = "Clusters: " + ", ".join(parts)

        self._neuron_table_summary_label.setText(
            f"{counts_line}\n{clusters_line}"
        )

    def _selected_neuron_heatmap_base_name(
        self,
        file_ids: list[object] | tuple[object, ...],
    ) -> str:
        """Return the base layer name for a selected-neuron heatmap."""
        return f"{self._next_manual_heatmap_identifier()} Heatmap"

    def _unique_layer_name(self, base_name: str) -> str:
        """Return a viewer layer name that does not collide with existing names."""
        existing_names = {
            str(getattr(layer, "name", ""))
            for layer in self._iter_viewer_layers()
        }
        if base_name not in existing_names:
            return base_name

        suffix = 2
        while f"{base_name} ({suffix})" in existing_names:
            suffix += 1
        return f"{base_name} ({suffix})"

    @staticmethod
    def _greek_heatmap_identifier(index: int) -> str:
        """Return a deterministic Greek-word identifier for a zero-based index."""
        if index < 0:
            raise ValueError("index must be non-negative")

        alphabet = _GREEK_HEATMAP_IDENTIFIERS
        base = len(alphabet)
        length = 1
        remaining = int(index)
        block_size = base
        while remaining >= block_size:
            remaining -= block_size
            length += 1
            block_size *= base

        indices = [0] * length
        for offset in range(length - 1, -1, -1):
            indices[offset] = remaining % base
            remaining //= base
        return " ".join(alphabet[i] for i in indices)

    def _next_manual_heatmap_identifier(self) -> str:
        """Return the first Greek identifier whose layer name is unused."""
        existing_names = {
            str(getattr(layer, "name", ""))
            for layer in self._iter_viewer_layers()
        }
        index = 0
        while True:
            identifier = NeuronViewerWidget._greek_heatmap_identifier(index)
            if f"{identifier} Heatmap" not in existing_names:
                return identifier
            index += 1

    def _selected_neuron_heatmap_layer_name(
        self,
        file_ids: list[object] | tuple[object, ...],
    ) -> str:
        """Return the next unique layer name for a selected-neuron heatmap."""
        return f"{self._next_manual_heatmap_identifier()} Heatmap"

    def _update_mask_sigma_units_label(self) -> None:
        """Show Gaussian sigma units in voxels with atlas micron equivalence."""
        if not hasattr(self, "_mask_sigma_units_label"):
            return

        if self._atlas is None:
            self._mask_sigma_units_label.setText(
                "Sigma is measured in atlas voxels. Load an atlas to see the micron equivalent."
            )
            return

        resolution = np.asarray(self._atlas.resolution, dtype=float)
        if np.allclose(resolution, resolution[0]):
            self._mask_sigma_units_label.setText(
                f"Sigma is measured in atlas voxels. "
                f"With the current atlas, 1 voxel = {resolution[0]:g} µm."
            )
        else:
            self._mask_sigma_units_label.setText(
                "Sigma is measured in atlas voxels. "
                f"With the current atlas, 1 voxel = "
                f"{resolution[2]:g} µm (X), {resolution[1]:g} µm (Y), "
                f"{resolution[0]:g} µm (Z)."
            )

    def _heatmap_layer_eligibility(self, layer) -> tuple[bool, str]:
        """Return whether a layer is eligible as a Tools heatmap source."""
        if self._atlas is None:
            return False, "Load an atlas to select heatmaps."

        metadata = _layer_metadata(layer)
        if not metadata.get("heatmap_source"):
            return False, "Not an app heatmap layer."
        if not metadata.get("heatmap_native_grid", False):
            return False, "Heatmap is not on the native atlas voxel grid."
        if metadata.get("atlas_name") != self._current_atlas_name():
            return False, "Heatmap atlas does not match the loaded atlas."

        data = np.asarray(getattr(layer, "data", np.array([])))
        atlas_shape = tuple(np.asarray(self._atlas.annotation).shape)
        if data.ndim != 3 or tuple(data.shape) != atlas_shape:
            return False, "Heatmap shape does not match the loaded atlas."
        return True, ""

    def _generated_mask_layers(self) -> list:
        """Return generated mask layers eligible for Regions queries."""
        masks = []
        atlas_name = self._current_atlas_name()
        for layer in self._iter_viewer_layers():
            metadata = _layer_metadata(layer)
            if not metadata.get("mask_query_source"):
                continue
            if atlas_name is not None and metadata.get("atlas_name") != atlas_name:
                continue
            masks.append(layer)
        return masks

    def _eligible_heatmap_layers_with_exclusions(self) -> tuple[list, list[str]]:
        """Return eligible heatmap layers and exclusion messages."""
        eligible_layers = []
        excluded_messages: list[str] = []
        for layer in self._iter_viewer_layers():
            eligible, reason = self._heatmap_layer_eligibility(layer)
            if eligible:
                eligible_layers.append(layer)
            elif _layer_metadata(layer).get("heatmap_source"):
                excluded_messages.append(f"{layer.name}: {reason}")
        return eligible_layers, excluded_messages

    def _refresh_heatmap_layer_list(self) -> None:
        """Refresh the Tools heatmap selector list."""
        if not hasattr(self, "_heatmap_layer_list"):
            return

        previous = {
            item.text()
            for item in self._heatmap_layer_list.selectedItems()
        }
        self._heatmap_layer_list.clear()

        eligible_layers, excluded_messages = self._eligible_heatmap_layers_with_exclusions()
        eligible_names = [layer.name for layer in eligible_layers]
        for layer in eligible_layers:
            self._heatmap_layer_list.addItem(layer.name)

        for index in range(self._heatmap_layer_list.count()):
            item = self._heatmap_layer_list.item(index)
            if item.text() in previous:
                item.setSelected(True)

        if not eligible_names:
            if excluded_messages:
                self._tools_hint_label.setText(
                    "No eligible native-grid heatmaps. "
                    + " ".join(excluded_messages[:3])
                )
            else:
                self._tools_hint_label.setText(
                    "No eligible heatmap layers are available."
                )
        elif excluded_messages:
            self._tools_hint_label.setText(
                f"{len(eligible_names)} eligible heatmap layer(s). "
                + "Excluded: "
                + "; ".join(excluded_messages[:3])
            )
        else:
            self._tools_hint_label.setText(
                f"{len(eligible_names)} eligible heatmap layer(s)."
            )
        self._update_tools_controls()

    def _refresh_histogram_layer_list(self) -> None:
        """Refresh the Histogram-tab source list."""
        if not hasattr(self, "_histogram_layer_list"):
            return

        previous = {
            item.text()
            for item in self._histogram_layer_list.selectedItems()
        }
        self._histogram_layer_list.clear()

        eligible_layers, excluded_messages = self._eligible_heatmap_layers_with_exclusions()
        eligible_names = [layer.name for layer in eligible_layers]
        for layer in eligible_layers:
            self._histogram_layer_list.addItem(layer.name)

        for index in range(self._histogram_layer_list.count()):
            item = self._histogram_layer_list.item(index)
            if item.text() in previous:
                item.setSelected(True)

        if not eligible_names:
            if excluded_messages:
                self._histogram_hint_label.setText(
                    "No eligible native-grid heatmaps. "
                    + " ".join(excluded_messages[:3])
                )
            else:
                self._histogram_hint_label.setText(
                    "No eligible heatmap layers are available."
                )
        elif excluded_messages:
            self._histogram_hint_label.setText(
                f"{len(eligible_names)} eligible heatmap layer(s). "
                + "Excluded: "
                + "; ".join(excluded_messages[:3])
            )
        else:
            self._histogram_hint_label.setText(
                f"{len(eligible_names)} eligible heatmap layer(s)."
            )
        self._update_histogram_controls()
        self._update_histogram_plot()

    def _refresh_mask_layer_options(self) -> None:
        """Refresh Regions-tab mask layer options."""
        if not hasattr(self, "_mask_layer_selector"):
            return

        masks = self._generated_mask_layers()
        mask_entries = []
        for layer in masks:
            metadata = _layer_metadata(layer)
            mask_entries.append(
                {
                    "name": layer.name,
                    "sources": metadata.get("source_heatmap_layers", []),
                }
            )
        self._mask_layer_selector.set_mask_layers(mask_entries)
        hint = (
            f"{len(masks)} generated mask layer(s) available."
            if masks
            else "No generated mask layers are available."
        )
        self._mask_query_hint_label.setText(hint)

    def _current_tools_create_mode(self) -> str:
        """Return the active Tools create mode."""
        combo = getattr(self, "_tools_create_mode_combo", None)
        mode = combo.currentData() if combo is not None else None
        return "merged" if mode == "merged" else "separate"

    def _current_region_isolation_create_mode(self) -> str:
        """Return the active Tools region-isolation create mode."""
        combo = getattr(self, "_region_isolation_create_mode_combo", None)
        mode = combo.currentData() if combo is not None else None
        return "merged" if mode == "merged" else "separate"

    def _current_histogram_create_mode(self) -> str:
        """Return the active Histogram create mode."""
        combo = getattr(self, "_histogram_create_mode_combo", None)
        mode = combo.currentData() if combo is not None else None
        return "merged" if mode == "merged" else "separate"

    def _selected_region_isolation_entries(self) -> list[tuple[int, str]]:
        """Return directly selected Tools isolation regions as IDs and acronyms."""
        selector = getattr(self, "_tools_region_selector", None)
        if selector is None:
            return []

        entries: list[tuple[int, str]] = []
        if hasattr(selector, "get_selected_ids"):
            selected_ids = selector.get_selected_ids(include_children=False)
            structure_map = getattr(selector, "_structure_map", {})
            for struct_id in selected_ids:
                struct = structure_map.get(int(struct_id), {})
                acronym = str(struct.get("acronym", "")).strip()
                if acronym:
                    entries.append((int(struct_id), acronym))
            if entries:
                return entries

        if hasattr(selector, "get_single_selected_region"):
            selected = selector.get_single_selected_region()
            if selected is not None:
                struct_id, acronym = selected
                return [(int(struct_id), str(acronym))]
        return []

    def _selected_region_isolation_region_ids(self) -> list[int]:
        """Return effective atlas annotation IDs for Tools region isolation."""
        selector = getattr(self, "_tools_region_selector", None)
        if selector is None or not hasattr(selector, "get_selected_ids"):
            return []

        include_children = True
        if hasattr(selector, "include_children_enabled"):
            include_children = bool(selector.include_children_enabled())
        return [
            int(region_id)
            for region_id in selector.get_selected_ids(
                include_children=include_children
            )
        ]

    def _region_isolation_label(self, acronyms: list[str]) -> str:
        """Return a compact region label for isolated heatmap layer names."""
        if not acronyms:
            return "selected regions"
        if len(acronyms) <= 2:
            return ", ".join(acronyms)
        return f"{', '.join(acronyms[:2])} +{len(acronyms) - 2} more"

    def _selected_layer_names_from_widget(self, widget: QListWidget | None) -> set[str]:
        """Return selected layer names from a list widget."""
        if widget is None:
            return set()
        return {item.text() for item in widget.selectedItems()}

    def _selected_layers_from_widget(self, widget: QListWidget | None) -> list:
        """Return selected eligible heatmap layers for a list widget."""
        selected_names = self._selected_layer_names_from_widget(widget)
        if not selected_names:
            return []
        layers = []
        for layer in self._iter_viewer_layers():
            if layer.name in selected_names and self._heatmap_layer_eligibility(layer)[0]:
                layers.append(layer)
        return layers

    def _selected_heatmap_layers(self) -> list:
        """Return selected eligible heatmap layers from the Tools tab."""
        return self._selected_layers_from_widget(
            getattr(self, "_heatmap_layer_list", None)
        )

    def _selected_histogram_layers(self) -> list:
        """Return selected eligible heatmap layers from the Histogram tab."""
        return self._selected_layers_from_widget(
            getattr(self, "_histogram_layer_list", None)
        )

    def _layer_contrast_limits(self, layer) -> tuple[float, float] | None:
        """Return a layer's contrast limits if available."""
        contrast_limits = getattr(layer, "contrast_limits", None)
        if contrast_limits is None:
            return None
        try:
            limits_array = np.asarray(contrast_limits, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if limits_array.size < 2:
            return None
        lower, upper = sorted((float(limits_array[0]), float(limits_array[1])))
        return lower, upper

    def _histogram_plot_color(self, layer) -> tuple[float, float, float, float] | None:
        """Return a plotting color derived from layer metadata."""
        color = _layer_metadata(layer).get("color")
        if color is None:
            return None
        rgba = np.asarray(color, dtype=float).reshape(-1)
        if rgba.size == 3:
            rgba = np.append(rgba, 1.0)
        if rgba.size < 4:
            return None
        rgba = np.clip(rgba[:4], 0.0, 1.0)
        return tuple(float(value) for value in rgba)

    def _histogram_plot_pen(self, layer, series_index: int):
        """Return a pyqtgraph pen for one histogram series."""
        pg = getattr(self, "_histogram_pg", None)
        if pg is None:
            return None

        rgba = self._histogram_plot_color(layer)
        if rgba is None:
            return pg.mkPen(color=pg.intColor(series_index), width=2)

        rgba255 = tuple(int(round(value * 255.0)) for value in rgba)
        return pg.mkPen(color=rgba255, width=2)

    def _reset_histogram_plot_items(self) -> None:
        """Clear histogram curves and annotations while keeping the widget."""
        plot_item = getattr(self, "_histogram_plot_item", None)
        if plot_item is None:
            return

        legend = getattr(self, "_histogram_plot_legend", None)
        if legend is not None:
            scene = legend.scene()
            if scene is not None:
                scene.removeItem(legend)
            if getattr(plot_item, "legend", None) is legend:
                plot_item.legend = None
            self._histogram_plot_legend = None

        plot_item.clear()
        plot_item.setTitle("Intensity Histogram")
        plot_item.setLabel("bottom", "Intensity")
        plot_item.setLabel("left", "Voxel count")
        plot_item.showGrid(x=True, y=True, alpha=0.15)

        self._histogram_curve_items = []
        self._histogram_message_item = None
        self._histogram_plot_legend = plot_item.addLegend(offset=(12, 12))
        plot_item.addItem(self._histogram_lower_line, ignoreBounds=True)
        plot_item.addItem(self._histogram_upper_line, ignoreBounds=True)

    def _set_histogram_message(
        self,
        message: str,
        *,
        x_center: float = 0.5,
        y_center: float = 0.5,
    ) -> None:
        """Display a centered placeholder message in the histogram plot."""
        plot_item = getattr(self, "_histogram_plot_item", None)
        pg = getattr(self, "_histogram_pg", None)
        if plot_item is None or pg is None:
            return

        message_item = pg.TextItem(text=message, anchor=(0.5, 0.5))
        message_item.setPos(float(x_center), float(y_center))
        plot_item.addItem(message_item)
        self._histogram_message_item = message_item

    def _refresh_histogram_threshold_lines(self) -> None:
        """Update threshold guide positions without rebuilding the chart."""
        lower_line = getattr(self, "_histogram_lower_line", None)
        upper_line = getattr(self, "_histogram_upper_line", None)
        if lower_line is None or upper_line is None:
            return

        has_selection = bool(self._selected_histogram_layers())
        lower_value = (
            float(self._mask_lower_threshold_spin.value())
            if hasattr(self, "_mask_lower_threshold_spin")
            else 0.0
        )
        show_upper = bool(
            has_selection
            and hasattr(self, "_mask_use_upper_bound_cb")
            and self._mask_use_upper_bound_cb.isChecked()
        )

        self._histogram_line_sync_active = True
        try:
            lower_line.setPos(lower_value)
            lower_line.setVisible(has_selection)
            lower_line.setMovable(has_selection)

            if show_upper and hasattr(self, "_mask_upper_threshold_spin"):
                upper_line.setPos(float(self._mask_upper_threshold_spin.value()))
            upper_line.setVisible(show_upper)
            upper_line.setMovable(show_upper)
        finally:
            self._histogram_line_sync_active = False

    def _sync_mask_bounds_from_lines(
        self,
        *,
        lower: float,
        upper: float | None,
    ) -> None:
        """Update mask bound widgets from dragged histogram lines."""
        self._mask_lower_threshold_spin.blockSignals(True)
        self._mask_upper_threshold_spin.blockSignals(True)
        try:
            self._mask_lower_threshold_spin.setValue(float(lower))
            if upper is not None:
                self._mask_upper_threshold_spin.setValue(float(upper))
        finally:
            self._mask_upper_threshold_spin.blockSignals(False)
            self._mask_lower_threshold_spin.blockSignals(False)

    def _on_histogram_bound_line_moved(self, which: str) -> None:
        """Sync dragged histogram guide lines back into the bound controls."""
        if self._histogram_line_sync_active:
            return

        selected_layers = self._selected_histogram_layers()
        if not selected_layers:
            return

        lower_line = getattr(self, "_histogram_lower_line", None)
        upper_line = getattr(self, "_histogram_upper_line", None)
        if lower_line is None or upper_line is None:
            return

        lower = float(self._mask_lower_threshold_spin.value())
        use_upper = bool(self._mask_use_upper_bound_cb.isChecked())
        upper = (
            float(self._mask_upper_threshold_spin.value())
            if use_upper
            else None
        )

        if which == "lower":
            lower = float(lower_line.value())
            if upper is not None and lower > upper:
                upper = lower
        else:
            if upper is None:
                return
            upper = float(upper_line.value())
            if upper < lower:
                lower = upper

        self._sync_mask_bounds_from_lines(lower=lower, upper=upper)
        self._refresh_histogram_threshold_lines()

    def _on_histogram_bound_line_move_finished(self, which: str) -> None:
        """Mark dragged histogram bounds as manual after the user releases them."""
        if self._histogram_line_sync_active:
            return
        if which == "upper" and not self._mask_use_upper_bound_cb.isChecked():
            return
        self._mask_bounds_source = "manual"

    def _update_tools_controls(self, *_args) -> None:
        """Enable or disable Tools actions based on current selection."""
        if not hasattr(self, "_heatmap_layer_list"):
            return

        selected_layers = self._selected_heatmap_layers()
        ready = self._atlas is not None and bool(selected_layers)
        if hasattr(self, "_create_blur_btn"):
            self._create_blur_btn.setEnabled(ready)
        if hasattr(self, "_create_region_isolated_heatmap_btn"):
            has_regions = bool(self._selected_region_isolation_region_ids())
            self._create_region_isolated_heatmap_btn.setEnabled(ready and has_regions)

    def _update_histogram_controls(self) -> None:
        """Enable or disable Histogram actions based on current selection."""
        if not hasattr(self, "_histogram_layer_list"):
            return

        selected_layers = self._selected_histogram_layers()
        has_selection = bool(selected_layers)
        ready = self._atlas is not None and has_selection
        can_use_otsu = ready and (
            len(selected_layers) == 1
            or self._current_histogram_create_mode() == "merged"
        )

        contrast_reason = None
        if self._atlas is None:
            contrast_reason = "Load an atlas before syncing layer contrast."
        elif not selected_layers:
            contrast_reason = "Select one eligible heatmap layer to sync contrast limits."
        elif len(selected_layers) != 1:
            contrast_reason = "Select exactly one eligible heatmap layer to sync contrast limits."
        elif self._layer_contrast_limits(selected_layers[0]) is None:
            contrast_reason = "The selected layer does not expose contrast limits."

        if hasattr(self, "_create_mask_btn"):
            self._create_mask_btn.setEnabled(ready)
        if hasattr(self, "_mask_use_otsu_btn"):
            self._mask_use_otsu_btn.setEnabled(can_use_otsu)
            self._mask_use_otsu_btn.setToolTip(
                ""
                if can_use_otsu
                else "Select one heatmap layer, or switch Create mode to Merged layer."
            )
        if hasattr(self, "_mask_use_contrast_btn"):
            can_use_contrast = contrast_reason is None
            self._mask_use_contrast_btn.setEnabled(can_use_contrast)
            self._mask_use_contrast_btn.setToolTip(
                "" if can_use_contrast else contrast_reason
            )
        self._refresh_histogram_threshold_lines()

    def _mark_mask_bounds_manual(self, *_args) -> None:
        """Record that mask bounds were edited manually."""
        self._mask_bounds_source = "manual"
        self._refresh_histogram_threshold_lines()

    def _on_mask_upper_bound_toggled(self, checked: bool) -> None:
        """Enable or disable the upper threshold input."""
        if hasattr(self, "_mask_upper_threshold_spin"):
            self._mask_upper_threshold_spin.setEnabled(bool(checked))
        self._mask_bounds_source = "manual"
        self._refresh_histogram_threshold_lines()

    def _set_mask_bounds(
        self,
        lower: float,
        *,
        upper: float | None = None,
        enable_upper: bool,
        bounds_source: str,
    ) -> None:
        """Update Histogram mask bounds without marking them as manual edits."""
        self._mask_lower_threshold_spin.blockSignals(True)
        self._mask_upper_threshold_spin.blockSignals(True)
        self._mask_use_upper_bound_cb.blockSignals(True)

        self._mask_lower_threshold_spin.setValue(float(lower))
        if upper is not None:
            self._mask_upper_threshold_spin.setValue(float(upper))
        self._mask_use_upper_bound_cb.setChecked(bool(enable_upper))

        self._mask_use_upper_bound_cb.blockSignals(False)
        self._mask_upper_threshold_spin.blockSignals(False)
        self._mask_lower_threshold_spin.blockSignals(False)

        self._mask_upper_threshold_spin.setEnabled(bool(enable_upper))
        self._mask_bounds_source = bounds_source
        self._refresh_histogram_threshold_lines()

    def _on_histogram_layer_selection_changed(self) -> None:
        """Refresh histogram controls and plot after source selection changes."""
        self._update_histogram_controls()
        self._update_histogram_plot()

    def _update_histogram_plot(self, *_args) -> None:
        """Redraw the histogram plot for the selected layers."""
        plot_item = getattr(self, "_histogram_plot_item", None)
        if plot_item is None:
            return

        self._reset_histogram_plot_items()
        selected_layers = self._selected_histogram_layers()

        if not selected_layers:
            plot_item.setXRange(0.0, 1.0, padding=0.0)
            plot_item.setYRange(0.0, 1.0, padding=0.0)
            self._set_histogram_message("Select one or more eligible heatmap layers.")
            self._refresh_histogram_threshold_lines()
            return

        include_zero = bool(self._histogram_include_zero_cb.isChecked())
        _, series = _build_histogram_plot_series(
            [
                (layer.name, np.asarray(layer.data, dtype=np.float32))
                for layer in selected_layers
            ],
            bins=_HISTOGRAM_BIN_COUNT,
            include_zero=include_zero,
        )

        has_values = False
        for index, (layer, entry) in enumerate(zip(selected_layers, series)):
            x = np.asarray(entry["x"], dtype=np.float32)
            y = np.asarray(entry["y"], dtype=np.float32)
            if x.size == 0 or y.size == 0:
                continue
            has_values = True
            curve = plot_item.plot(
                x=x,
                y=y,
                pen=self._histogram_plot_pen(layer, index),
            )
            self._histogram_curve_items.append(curve)
            if self._histogram_plot_legend is not None:
                self._histogram_plot_legend.addItem(curve, str(entry["name"]))

        if not has_values:
            message = (
                "No finite values remain after excluding zero-valued voxels."
                if not include_zero
                else "No finite values are available for the selected layers."
            )
            plot_item.setXRange(0.0, 1.0, padding=0.0)
            plot_item.setYRange(0.0, 1.0, padding=0.0)
            self._set_histogram_message(message)
        else:
            plot_item.autoRange()

        self._refresh_histogram_threshold_lines()

    def _fill_mask_lower_bound_from_otsu(self) -> None:
        """Populate the lower bound from an Otsu threshold."""
        selected_layers = self._selected_histogram_layers()
        if not selected_layers:
            message = "Select at least one eligible heatmap layer."
            self._histogram_status_label.setText(message)
            return

        if len(selected_layers) > 1 and self._current_histogram_create_mode() != "merged":
            message = (
                "Use Otsu Lower with one selected layer, or switch Create mode to Merged layer."
            )
            self._histogram_status_label.setText(message)
            return

        if len(selected_layers) == 1:
            volume = np.asarray(selected_layers[0].data, dtype=np.float32)
        else:
            volume = merge_heatmap_volumes(
                [np.asarray(layer.data, dtype=np.float32) for layer in selected_layers]
            )

        threshold = otsu_threshold_positive(volume)
        self._set_mask_bounds(
            threshold,
            upper=float(self._mask_upper_threshold_spin.value())
            if self._mask_use_upper_bound_cb.isChecked()
            else None,
            enable_upper=self._mask_use_upper_bound_cb.isChecked(),
            bounds_source="otsu_lower",
        )
        self._histogram_status_label.setText(
            f"Lower bound set to Otsu threshold {threshold:.4f}."
        )

    def _copy_selected_layer_contrast_to_bounds(self) -> None:
        """Copy the selected layer contrast limits into the mask bounds."""
        selected_layers = self._selected_histogram_layers()
        if len(selected_layers) != 1:
            message = "Select exactly one eligible heatmap layer to sync contrast limits."
            self._histogram_status_label.setText(message)
            return

        limits = self._layer_contrast_limits(selected_layers[0])
        if limits is None:
            message = "The selected layer does not expose contrast limits."
            self._histogram_status_label.setText(message)
            return

        lower, upper = limits
        self._set_mask_bounds(
            lower,
            upper=upper,
            enable_upper=True,
            bounds_source="contrast_limits",
        )
        self._histogram_status_label.setText(
            f"Copied contrast limits to mask bounds: {lower:.4f} to {upper:.4f}."
        )

    def _selected_mask_threshold_bounds(self) -> tuple[float, float | None] | None:
        """Return the configured lower and optional upper bounds."""
        lower = float(self._mask_lower_threshold_spin.value())
        upper = None
        if self._mask_use_upper_bound_cb.isChecked():
            upper = float(self._mask_upper_threshold_spin.value())
            if upper < lower:
                self._histogram_status_label.setText(
                    "Upper bound must be greater than or equal to lower bound."
                )
                return None
        return lower, upper

    def _select_heatmap_layer_names(self, names: list[str]) -> None:
        """Select Tools heatmap list items by name."""
        if not hasattr(self, "_heatmap_layer_list"):
            return

        selected_names = set(names)
        self._heatmap_layer_list.blockSignals(True)
        self._heatmap_layer_list.clearSelection()
        for index in range(self._heatmap_layer_list.count()):
            item = self._heatmap_layer_list.item(index)
            item.setSelected(item.text() in selected_names)
        self._heatmap_layer_list.blockSignals(False)
        self._update_tools_controls()

    def _select_histogram_layer_names(self, names: list[str]) -> None:
        """Select Histogram heatmap list items by name."""
        if not hasattr(self, "_histogram_layer_list"):
            return

        selected_names = set(names)
        self._histogram_layer_list.blockSignals(True)
        self._histogram_layer_list.clearSelection()
        for index in range(self._histogram_layer_list.count()):
            item = self._histogram_layer_list.item(index)
            item.setSelected(item.text() in selected_names)
        self._histogram_layer_list.blockSignals(False)
        self._update_histogram_controls()
        self._update_histogram_plot()

    def _selected_mask_query_layers(self) -> list:
        """Return the currently selected generated mask layers."""
        if not hasattr(self, "_mask_layer_selector"):
            return []
        names = set(self._mask_layer_selector.get_selected_layer_names())
        if not names:
            return []
        return [
            layer for layer in self._generated_mask_layers()
            if layer.name in names
        ]

    def _on_mask_layer_selection_changed(self, selected_names: list[str]) -> None:
        """Update Regions status text when mask selection changes."""
        count = len(selected_names)
        if count == 0:
            self._regions_status_label.setText("")
        elif count == 1:
            self._regions_status_label.setText("1 mask layer selected for querying.")
        else:
            self._regions_status_label.setText(
                f"{count} mask layers selected for querying."
            )

    def _create_blurred_layers_from_heatmaps(self) -> None:
        """Create blurred image layers from the selected heatmaps."""
        if self._atlas is None:
            message = "Load an atlas before creating blurred heatmap layers."
            self._tools_status_label.setText(message)
            show_warning(message)
            return

        selected_layers = self._selected_heatmap_layers()
        if not selected_layers:
            message = "Select at least one eligible heatmap layer."
            self._tools_status_label.setText(message)
            return

        sigma = float(self._mask_sigma_spin.value())
        create_mode = self._current_tools_create_mode()
        created_layers = []

        try:
            if create_mode == "merged":
                merged_volume = merge_heatmap_volumes(
                    [np.asarray(layer.data, dtype=np.float32) for layer in selected_layers]
                )
                created_layers.append(
                    self._add_blurred_heatmap_layer(
                        layer_name=f"Blurred: merged {len(selected_layers)} heatmaps",
                        volume=merged_volume,
                        source_layers=selected_layers,
                        sigma=sigma,
                        merge_mode="merged_sum",
                    )
                )
            else:
                for layer in selected_layers:
                    created_layers.append(
                        self._add_blurred_heatmap_layer(
                            layer_name=f"Blurred: {layer.name}",
                            volume=np.asarray(layer.data, dtype=np.float32),
                            source_layers=[layer],
                            sigma=sigma,
                            merge_mode="separate",
                        )
                    )
        except Exception as e:
            logger.error("Failed to create blurred heatmap layers: %s", e)
            self._tools_status_label.setText(f"Failed to create blurred heatmap layers: {e}")
            return

        self._refresh_heatmap_layer_list()
        self._refresh_histogram_layer_list()
        self._select_heatmap_layer_names([layer.name for layer in created_layers])
        self._select_histogram_layer_names([layer.name for layer in created_layers])
        nonempty = sum(int(np.any(np.asarray(layer.data) > 0)) for layer in created_layers)
        self._tools_status_label.setText(
            f"Created {len(created_layers)} blurred layer(s); {nonempty} contain nonzero voxels."
        )

    def _create_region_isolated_heatmaps(self) -> None:
        """Create heatmap image layers limited to selected atlas regions."""
        if self._atlas is None:
            message = "Load an atlas before creating isolated heatmap layers."
            self._tools_status_label.setText(message)
            show_warning(message)
            return

        selected_layers = self._selected_heatmap_layers()
        if not selected_layers:
            message = "Select at least one eligible heatmap layer."
            self._tools_status_label.setText(message)
            return

        region_ids = self._selected_region_isolation_region_ids()
        if not region_ids:
            message = "Select at least one atlas region for isolation."
            self._tools_status_label.setText(message)
            return

        selected_regions = self._selected_region_isolation_entries()
        selected_region_ids = [
            region_id for region_id, _acronym in selected_regions
        ]
        selected_region_acronyms = [
            acronym for _region_id, acronym in selected_regions
        ]
        region_label = self._region_isolation_label(selected_region_acronyms)
        selector = getattr(self, "_tools_region_selector", None)
        include_children = (
            bool(selector.include_children_enabled())
            if selector is not None and hasattr(selector, "include_children_enabled")
            else True
        )
        create_mode = self._current_region_isolation_create_mode()
        created_layers = []

        try:
            if create_mode == "merged":
                merged_volume = merge_heatmap_volumes(
                    [np.asarray(layer.data, dtype=np.float32) for layer in selected_layers]
                )
                isolated = isolate_heatmap_volume_to_region_ids(
                    merged_volume,
                    self._atlas,
                    region_ids,
                )
                created_layers.append(
                    self._add_region_isolated_heatmap_layer(
                        layer_name=(
                            f"Region Isolated ({region_label}): "
                            f"merged {len(selected_layers)} heatmaps"
                        ),
                        volume=isolated,
                        source_layers=selected_layers,
                        selected_region_ids=selected_region_ids,
                        selected_region_acronyms=selected_region_acronyms,
                        region_ids=region_ids,
                        include_children=include_children,
                        merge_mode="merged_sum",
                    )
                )
            else:
                for layer in selected_layers:
                    isolated = isolate_heatmap_volume_to_region_ids(
                        np.asarray(layer.data, dtype=np.float32),
                        self._atlas,
                        region_ids,
                    )
                    created_layers.append(
                        self._add_region_isolated_heatmap_layer(
                            layer_name=f"Region Isolated ({region_label}): {layer.name}",
                            volume=isolated,
                            source_layers=[layer],
                            selected_region_ids=selected_region_ids,
                            selected_region_acronyms=selected_region_acronyms,
                            region_ids=region_ids,
                            include_children=include_children,
                            merge_mode="separate",
                        )
                    )
        except Exception as e:
            logger.error("Failed to create isolated heatmap layers: %s", e)
            self._tools_status_label.setText(f"Failed to create isolated heatmap layers: {e}")
            return

        self._refresh_heatmap_layer_list()
        self._refresh_histogram_layer_list()
        self._select_heatmap_layer_names([layer.name for layer in created_layers])
        self._select_histogram_layer_names([layer.name for layer in created_layers])
        nonempty = sum(int(np.any(np.asarray(layer.data) > 0)) for layer in created_layers)
        self._tools_status_label.setText(
            f"Created {len(created_layers)} isolated heatmap layer(s); "
            f"{nonempty} contain nonzero voxels."
        )

    def _create_masks_from_heatmaps(self) -> None:
        """Create binary mask label layers from selected heatmap image layers."""
        if self._atlas is None:
            message = "Load an atlas before creating mask layers."
            self._histogram_status_label.setText(message)
            show_warning(message)
            return

        selected_layers = self._selected_histogram_layers()
        if not selected_layers:
            message = "Select at least one eligible heatmap layer."
            self._histogram_status_label.setText(message)
            return

        bounds = self._selected_mask_threshold_bounds()
        if bounds is None:
            return
        lower_threshold, upper_threshold = bounds

        create_mode = self._current_histogram_create_mode()
        created_layers = []

        try:
            if create_mode == "merged":
                merged_volume = merge_heatmap_volumes(
                    [np.asarray(layer.data, dtype=np.float32) for layer in selected_layers]
                )
                mask = build_binary_mask_from_threshold_range(
                    merged_volume,
                    lower_threshold=lower_threshold,
                    upper_threshold=upper_threshold,
                )
                layer_name = f"Mask: merged {len(selected_layers)} heatmaps"
                created_layers.append(
                    self._add_mask_layer(
                        layer_name=layer_name,
                        mask=mask,
                        source_layers=selected_layers,
                        lower_threshold=lower_threshold,
                        upper_threshold=upper_threshold,
                        bounds_source=self._mask_bounds_source,
                        merge_mode="merged_sum",
                    )
                )
            else:
                for layer in selected_layers:
                    mask = build_binary_mask_from_threshold_range(
                        np.asarray(layer.data, dtype=np.float32),
                        lower_threshold=lower_threshold,
                        upper_threshold=upper_threshold,
                    )
                    created_layers.append(
                        self._add_mask_layer(
                            layer_name=f"Mask: {layer.name}",
                            mask=mask,
                            source_layers=[layer],
                            lower_threshold=lower_threshold,
                            upper_threshold=upper_threshold,
                            bounds_source=self._mask_bounds_source,
                            merge_mode="separate",
                        )
                    )
        except Exception as e:
            logger.error("Failed to create mask layers: %s", e)
            self._histogram_status_label.setText(f"Failed to create mask layers: {e}")
            return

        nonempty = sum(int(np.asarray(layer.data).sum() > 0) for layer in created_layers)
        bounds_text = f"lower {lower_threshold:.4f}"
        if upper_threshold is not None:
            bounds_text += f", upper {upper_threshold:.4f}"
        self._histogram_status_label.setText(
            f"Created {len(created_layers)} mask layer(s); {nonempty} contain nonzero voxels ({bounds_text})."
        )
        self._refresh_mask_layer_options()

    def _add_blurred_heatmap_layer(
        self,
        layer_name: str,
        volume: np.ndarray,
        source_layers: list,
        sigma: float,
        merge_mode: str,
    ):
        """Add or replace a blurred heatmap image layer."""
        from napari.utils.colormaps import Colormap

        for layer in list(self._iter_viewer_layers()):
            if layer.name == layer_name:
                self.viewer.layers.remove(layer)

        blurred = smooth_heatmap_volume(volume, sigma=sigma)
        first_layer = source_layers[0]
        rgba = _mask_layer_color(source_layers)

        colormap = getattr(first_layer, "colormap", None) if len(source_layers) == 1 else None
        if colormap is None and rgba is not None:
            colormap = Colormap(
                colors=[[0.0, 0.0, 0.0, 0.0], list(rgba)],
                name=f"blurred_{layer_name.lower().replace(' ', '_')}",
            )
        elif colormap is None:
            colormap = "hot"

        metadata = dict(_layer_metadata(first_layer)) if len(source_layers) == 1 else {}
        metadata.update(
            {
                "heatmap_source": True,
                "heatmap_native_grid": True,
                "heatmap_kind": "blurred",
                "blur_sigma": float(sigma),
                "source_heatmap_layers": [layer.name for layer in source_layers],
                "atlas_name": self._current_atlas_name(),
                "color": rgba,
                "merge_mode": merge_mode,
            }
        )

        add_kwargs = {
            "name": layer_name,
            "colormap": colormap,
            "blending": getattr(first_layer, "blending", "additive"),
            "rendering": getattr(first_layer, "rendering", "mip"),
            "opacity": getattr(first_layer, "opacity", self._opacity_slider.value() / 100.0),
            "visible": True,
            "metadata": metadata,
        }
        for attr_name in ("scale", "translate"):
            value = getattr(first_layer, attr_name, None)
            if value is not None:
                add_kwargs[attr_name] = value

        layer = self.viewer.add_image(
            blurred,
            **add_kwargs,
        )
        return layer

    def _add_region_isolated_heatmap_layer(
        self,
        layer_name: str,
        volume: np.ndarray,
        source_layers: list,
        selected_region_ids: list[int],
        selected_region_acronyms: list[str],
        region_ids: list[int],
        include_children: bool,
        merge_mode: str,
    ):
        """Add one post-hoc region-isolated heatmap image layer."""
        from napari.utils.colormaps import Colormap

        first_layer = source_layers[0]
        rgba = _mask_layer_color(source_layers)

        colormap = getattr(first_layer, "colormap", None) if len(source_layers) == 1 else None
        if colormap is None and rgba is not None:
            colormap = Colormap(
                colors=[[0.0, 0.0, 0.0, 0.0], list(rgba)],
                name=f"region_isolated_{layer_name.lower().replace(' ', '_')}",
            )
        elif colormap is None:
            colormap = "hot"

        contrast_limits = _heatmap_contrast_limits(volume)
        metadata = dict(_layer_metadata(first_layer)) if len(source_layers) == 1 else {}
        for stale_key in (
            "heatmap_region",
            "heatmap_selected_region_id",
            "heatmap_selected_region_acronym",
        ):
            metadata.pop(stale_key, None)
        metadata.update(
            {
                "heatmap_source": True,
                "heatmap_native_grid": True,
                "heatmap_kind": "region_isolated",
                "source_heatmap_layers": [layer.name for layer in source_layers],
                "heatmap_selected_region_ids": [
                    int(region_id) for region_id in selected_region_ids
                ],
                "heatmap_selected_region_acronyms": [
                    str(acronym) for acronym in selected_region_acronyms
                ],
                "heatmap_region_ids": [int(region_id) for region_id in region_ids],
                "heatmap_include_child_regions": bool(include_children),
                "atlas_name": self._current_atlas_name(),
                "color": rgba,
                "merge_mode": merge_mode,
                "heatmap_contrast_limits": contrast_limits,
                "heatmap_autocontrast_policy": "stable_full_volume",
            }
        )

        add_kwargs = {
            "name": self._unique_layer_name(layer_name),
            "colormap": colormap,
            "blending": getattr(first_layer, "blending", "additive"),
            "rendering": getattr(first_layer, "rendering", "mip"),
            "opacity": getattr(first_layer, "opacity", self._opacity_slider.value() / 100.0),
            "visible": True,
            "contrast_limits": contrast_limits,
            "metadata": metadata,
        }
        for attr_name in ("scale", "translate"):
            value = getattr(first_layer, attr_name, None)
            if value is not None:
                add_kwargs[attr_name] = value

        return self.viewer.add_image(
            np.asarray(volume, dtype=np.float32),
            **add_kwargs,
        )

    def _add_mask_layer(
        self,
        layer_name: str,
        mask: np.ndarray,
        source_layers: list,
        lower_threshold: float,
        upper_threshold: float | None,
        bounds_source: str,
        merge_mode: str,
    ):
        """Add or replace a generated binary mask layer."""
        from napari.utils import DirectLabelColormap

        for layer in list(self._iter_viewer_layers()):
            if layer.name == layer_name:
                self.viewer.layers.remove(layer)

        labels = np.asarray(mask, dtype=np.uint8)
        rgba = _mask_layer_color(source_layers)
        color_dict = {
            None: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            0: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
        color_dict[1] = np.array(
            rgba if rgba is not None else (0.8, 0.8, 0.8, 1.0),
            dtype=np.float32,
        )

        layer = self.viewer.add_labels(
            labels,
            name=layer_name,
            opacity=self._opacity_slider.value() / 100.0,
            visible=True,
            colormap=DirectLabelColormap(color_dict=color_dict),
            metadata={
                "mask_query_source": True,
                "source_heatmap_layers": [layer.name for layer in source_layers],
                "sigma": _shared_blur_sigma(source_layers),
                "threshold_mode": "range",
                "threshold_value": float(lower_threshold),
                "lower_threshold": float(lower_threshold),
                "upper_threshold": (
                    None if upper_threshold is None else float(upper_threshold)
                ),
                "bounds_source": bounds_source,
                "merge_mode": merge_mode,
                "atlas_name": self._current_atlas_name(),
                "color": rgba,
            },
        )
        return layer

    def _on_region_query_source_changed(self, text: str) -> None:
        """Switch Regions tab between atlas and mask query modes."""
        self._region_query_source = text
        if not hasattr(self, "_region_query_stack"):
            return

        show_mask_buttons = text == "Mask Layer"
        self._region_query_stack.setCurrentIndex(1 if show_mask_buttons else 0)
        self._sync_region_query_scope_selector()
        for button in self._atlas_region_query_buttons():
            button.setVisible(not show_mask_buttons)
        for button in self._mask_layer_query_buttons():
            button.setVisible(show_mask_buttons)
        self._regions_status_label.setText("")
        if not show_mask_buttons:
            self._sync_active_region_reference_layers()

    def _on_region_query_scope_changed(self, _text: str) -> None:
        """Track the current region-query scope selection."""
        self._region_query_scope = self._selected_region_query_scope()
        self._sync_region_query_scope_selector()
        if hasattr(self, "_regions_status_label"):
            self._regions_status_label.setText("")
        if getattr(self, "_region_query_source", "Atlas Regions") == "Atlas Regions":
            self._sync_active_region_reference_layers()

    def _selected_region_query_scope(self) -> str:
        """Return the active search scope for region and mask queries."""
        combo = getattr(self, "_region_query_scope_combo", None)
        if combo is None:
            scope = getattr(self, "_region_query_scope", _REGION_QUERY_SCOPE_WHOLE)
            if scope in {_REGION_QUERY_SCOPE_WHOLE, _REGION_QUERY_SCOPE_CURRENT}:
                return scope
            return _REGION_QUERY_SCOPE_WHOLE

        current_data = getattr(combo, "currentData", None)
        if callable(current_data):
            data = current_data()
            if data in {_REGION_QUERY_SCOPE_WHOLE, _REGION_QUERY_SCOPE_CURRENT}:
                return str(data)

        current_text = getattr(combo, "currentText", None)
        text = current_text() if callable(current_text) else ""
        if text == "Current Table":
            return _REGION_QUERY_SCOPE_CURRENT
        return _REGION_QUERY_SCOPE_WHOLE

    def _current_table_file_ids(self) -> list[object]:
        """Return the file IDs currently present in the neuron table."""
        table = getattr(self, "_neuron_table", None)
        if table is None:
            return []

        file_ids_getter = getattr(table, "file_ids", None)
        if callable(file_ids_getter):
            return list(file_ids_getter())

        entries = getattr(table, "_entries", {})
        return list(entries.keys())

    def _resolve_region_query_file_scope(
        self,
    ) -> tuple[bool, list[object] | None, str, int | None]:
        """Resolve the optional base file-ID restriction for region queries."""
        scope = self._selected_region_query_scope()
        if scope != _REGION_QUERY_SCOPE_CURRENT:
            return True, None, "whole parquet", None

        file_ids = self._current_table_file_ids()
        if file_ids:
            return True, file_ids, "current table", len(file_ids)

        self._regions_status_label.setText(
            "Current table is empty; switch search scope to Whole Parquet or populate the table first."
        )
        return False, None, "current table", 0

    @staticmethod
    def _query_scope_status_suffix(scope_label: str, input_count: int | None) -> str:
        """Format a scope-aware suffix for query status messages."""
        suffix = f" within {scope_label}"
        if input_count is not None:
            suffix += f" (from {input_count} input neurons)"
        return suffix

    def _region_selector_for_scope(
        self,
        scope: str | None = None,
    ) -> RegionSelectorWidget | None:
        """Return the atlas-region selector widget for the requested scope."""
        selected_scope = scope
        if selected_scope not in {
            _REGION_QUERY_SCOPE_WHOLE,
            _REGION_QUERY_SCOPE_CURRENT,
        }:
            selected_scope = self._selected_region_query_scope()

        if selected_scope == _REGION_QUERY_SCOPE_CURRENT:
            selector = getattr(self, "_current_table_region_selector", None)
            if selector is not None:
                return selector
        else:
            selector = getattr(self, "_whole_parquet_region_selector", None)
            if selector is not None:
                return selector

        return getattr(self, "_region_selector", None)

    def _active_region_selector(self) -> RegionSelectorWidget | None:
        """Return the atlas-region selector for the current search scope."""
        return self._region_selector_for_scope()

    def _active_region_preview_acronyms(self) -> list[str]:
        """Return directly selected atlas acronyms for preview layers."""
        selector = self._active_region_selector()
        if selector is None:
            return []

        get_selected = getattr(selector, "get_selected_acronyms", None)
        if not callable(get_selected):
            return []
        return list(get_selected(include_children=False))

    def _sync_region_query_scope_selector(self) -> None:
        """Show the atlas selector that matches the active query scope."""
        stack = getattr(self, "_atlas_region_scope_stack", None)
        if stack is None:
            return

        index = (
            1
            if self._selected_region_query_scope() == _REGION_QUERY_SCOPE_CURRENT
            else 0
        )
        stack.setCurrentIndex(index)

    def _sync_active_region_reference_layers(self) -> None:
        """Refresh region preview layers from the active atlas selector."""
        acronyms = self._active_region_preview_acronyms()

        show_meshes = getattr(self, "_show_region_meshes_cb", None)
        if show_meshes is not None and show_meshes.isChecked():
            self._update_region_meshes(acronyms)

        show_segmentation = getattr(self, "_show_region_seg_cb", None)
        if show_segmentation is not None and show_segmentation.isChecked():
            self._update_region_segmentation(acronyms)

    @staticmethod
    def _atlas_query_details(
        acronyms: list[str],
        *,
        include_children: bool,
    ) -> str:
        """Format the exact atlas-region query that was executed."""
        if len(acronyms) == 1:
            selection = acronyms[0]
        else:
            selection = f"union of {', '.join(acronyms)}"
        descendants = "on" if include_children else "off"
        return f"Query: {selection}; descendants: {descendants}."

    def _atlas_region_query_buttons(self) -> tuple[QPushButton, QPushButton]:
        """Return the atlas-region query buttons."""
        return (
            self._atlas_query_any_node_btn,
            self._atlas_query_soma_btn,
        )

    def _mask_layer_query_buttons(self) -> tuple[QPushButton, QPushButton]:
        """Return the mask-layer query buttons."""
        return (
            self._mask_query_any_node_btn,
            self._mask_query_soma_btn,
        )

    def _set_region_query_buttons_enabled(self, enabled: bool) -> None:
        """Enable or disable all Regions-tab query buttons."""
        for button in (
            *self._atlas_region_query_buttons(),
            *self._mask_layer_query_buttons(),
        ):
            button.setEnabled(enabled)

    def _query_atlas_neurons_any_node(self) -> None:
        """Query neurons with any node in the selected atlas regions."""
        self._regions_status_label.setText(
            "Searching for neurons with any node in selected atlas regions. Please wait..."
        )
        QApplication.processEvents()
        self._query_neurons_by_region(soma_only=False)

    def _query_atlas_neurons_soma(self) -> None:
        """Query neurons with soma in the selected atlas regions."""
        self._regions_status_label.setText(
            "Searching for neurons with soma in selected atlas regions. Please wait..."
        )
        QApplication.processEvents()
        self._query_neurons_by_region(soma_only=True)

    def _query_mask_neurons_any_node(self) -> None:
        """Query neurons with any node in the selected mask layers."""
        self._regions_status_label.setText(
            "Searching for neurons with any node in selected mask layers. Please wait..."
        )
        QApplication.processEvents()
        self._query_neurons_by_mask(soma_only=False)

    def _query_mask_neurons_soma(self) -> None:
        """Query neurons with soma in the selected mask layers."""
        self._regions_status_label.setText(
            "Searching for neurons with soma in selected mask layers. Please wait..."
        )
        QApplication.processEvents()
        self._query_neurons_by_mask(soma_only=True)

    def _on_regions_selected(self, acronyms: list[str]) -> None:
        """Handle region selection changes."""
        _ = acronyms
        preview_acronyms = self._active_region_preview_acronyms()

        # Update region meshes if enabled
        if self._show_region_meshes_cb.isChecked():
            self._update_region_meshes(preview_acronyms)

        # Update region segmentation if enabled
        if self._show_region_seg_cb.isChecked():
            self._update_region_segmentation(preview_acronyms)

    def _query_neurons_by_region(self, soma_only: bool = False) -> None:
        """Query neurons in selected regions."""
        if self._db is None:
            return

        selector = self._active_region_selector()
        if selector is None:
            self._regions_status_label.setText("Select at least one atlas region.")
            return

        get_query_acronyms = getattr(selector, "get_query_acronyms", None)
        acronyms = list(get_query_acronyms()) if callable(get_query_acronyms) else []
        if not acronyms:
            self._regions_status_label.setText("Select at least one atlas region.")
            return

        try:
            proceed, base_file_ids, scope_label, input_count = (
                self._resolve_region_query_file_scope()
            )
            if not proceed:
                return

            result = self._db.get_neurons_by_region(
                acronyms,
                soma_only=soma_only,
                file_ids=base_file_ids,
            )
            self._populate_neuron_table(
                result,
                preserve_existing=base_file_ids is not None,
            )
            membership = "soma" if soma_only else "any node"
            include_children = False
            include_children_enabled = getattr(
                selector,
                "include_children_enabled",
                None,
            )
            if callable(include_children_enabled):
                include_children = bool(include_children_enabled())
            query_details = NeuronViewerWidget._atlas_query_details(
                acronyms,
                include_children=include_children,
            )
            self._regions_status_label.setText(
                "Found "
                f"{len(result)} neuron(s) with {membership} in selected atlas regions"
                f"{self._query_scope_status_suffix(scope_label, input_count)}. "
                f"{query_details}"
            )
            logger.info(
                "Found %d neurons with %s in selected atlas regions within %s",
                len(result),
                membership,
                scope_label,
            )

        except Exception as e:
            logger.error(f"Query failed: {e}")
            self._regions_status_label.setText(f"Region query failed: {e}")

    def _query_neurons_by_mask(self, soma_only: bool = False) -> None:
        """Query neurons using a generated mask layer."""
        if self._db is None or self._atlas is None:
            return

        layers = self._selected_mask_query_layers()
        if not layers:
            self._regions_status_label.setText("Select at least one generated mask layer.")
            return

        mask = np.logical_or.reduce([np.asarray(layer.data) > 0 for layer in layers])
        if not mask.any():
            message = "Selected mask layer selection is empty and cannot be queried."
            self._regions_status_label.setText(message)
            show_warning(message)
            return

        try:
            proceed, base_file_ids, scope_label, input_count = (
                self._resolve_region_query_file_scope()
            )
            if not proceed:
                return

            result = self._db.get_neurons_by_mask(
                mask,
                self._atlas,
                soma_only=soma_only,
                file_ids=base_file_ids,
            )
            self._populate_neuron_table(
                result,
                preserve_existing=base_file_ids is not None,
            )
            membership = "soma" if soma_only else "any node"
            selected_names = ", ".join(layer.name for layer in layers[:3])
            if len(layers) > 3:
                selected_names += ", ..."
            self._regions_status_label.setText(
                "Found "
                f"{len(result)} neuron(s) with {membership} in "
                f"{len(layers)} selected mask layer(s)"
                f"{self._query_scope_status_suffix(scope_label, input_count)}: "
                f"{selected_names}"
            )
            logger.info(
                "Found %d neurons with %s in %d selected mask layers within %s",
                len(result),
                membership,
                len(layers),
                scope_label,
            )
        except Exception as e:
            logger.error(f"Mask query failed: {e}")
            self._regions_status_label.setText(f"Mask query failed: {e}")

    def _current_table_file_ids_in_scene(self) -> list[object]:
        """Return the subset of current table IDs that are rendered in the scene."""
        rendered = self._current_scene_file_ids()
        return [file_id for file_id in self._current_table_file_ids() if file_id in rendered]

    def _cache_scene_display_state(self, file_ids: list[object] | tuple[object, ...]) -> None:
        """Preserve base color and visibility for rendered neurons leaving the table."""
        scene_state = getattr(self, "_scene_display_state", None)
        if scene_state is None:
            self._scene_display_state = {}
            scene_state = self._scene_display_state

        for file_id in file_ids:
            color, visible = self._base_display_state_for_file_id(file_id)
            scene_state[file_id] = {
                "color": list(color),
                "visible": bool(visible),
            }

    def _discard_scene_display_state(
        self,
        file_ids: list[object] | tuple[object, ...],
    ) -> None:
        """Drop scene-only display state for neurons that are back in the table."""
        scene_state = getattr(self, "_scene_display_state", None)
        if not scene_state:
            return
        for file_id in file_ids:
            scene_state.pop(file_id, None)

    def _sync_after_neuron_table_membership_change(self) -> None:
        """Refresh derived UI and rendered colors after table membership changes."""
        self._last_soma_selection = set()
        self._refresh_cluster_filter_controls()
        refresh_clusters = getattr(
            self, "_refresh_apply_existing_clusters_button", None
        )
        if callable(refresh_clusters):
            refresh_clusters()
        self._refresh_neuron_table_summary()

        selected_getter = getattr(self._neuron_table, "get_selected_file_ids", None)
        selected_file_ids = selected_getter() if callable(selected_getter) else []
        current_layers = getattr(self, "_current_neuron_layers", [])
        if current_layers:
            rendered_ids = set(getattr(self, "_scene_render_modes", {}).keys())
            selected = set(selected_file_ids) & rendered_ids
            if not selected or selected == rendered_ids:
                self._highlighted_file_ids = None
            else:
                self._highlighted_file_ids = selected
            self._update_layer_colors(self._build_effective_color_map())
        else:
            self._highlighted_file_ids = None

    def _populate_neuron_table(self, result, *, preserve_existing: bool = False) -> None:
        """Populate or subset the neuron table from a query result."""
        self._cache_scene_display_state(self._current_table_file_ids_in_scene())

        if preserve_existing:
            matched_file_ids = result["file_id"].tolist()
            self._neuron_table.retain_file_ids(matched_file_ids)
        else:
            neurons = [
                (row["file_id"], row["subject"])
                for _, row in result.iterrows()
            ]
            self._neuron_table.populate(neurons)

        self._discard_scene_display_state(self._current_table_file_ids())
        self._neuron_table.set_added_file_ids(self._current_scene_file_ids())
        sync_heatmaps = getattr(self, "_sync_neuron_table_heatmap_membership", None)
        if callable(sync_heatmaps):
            sync_heatmaps()
        refresh_manual_heatmaps = getattr(self, "_refresh_manual_heatmap_combo", None)
        if callable(refresh_manual_heatmaps):
            refresh_manual_heatmaps()
        manual_heatmap_data = getattr(self, "_manual_heatmap_combo_data", None)
        manual_heatmap_handler = getattr(
            self,
            "_on_manual_heatmap_selection_changed",
            None,
        )
        if (
            callable(manual_heatmap_data)
            and manual_heatmap_data() is not None
            and callable(manual_heatmap_handler)
        ):
            manual_heatmap_handler()
        self._sync_after_neuron_table_membership_change()

    def _selected_cluster_filter(self) -> ClusterFilterSelection:
        """Return selected cluster groups from the Data tab dropdown."""
        getter = getattr(self._cluster_filter_combo, "cluster_filter_selection", None)
        if callable(getter):
            return getter()

        idx = self._cluster_filter_combo.currentIndex()
        if idx < 0:
            return ClusterFilterSelection()
        data = self._cluster_filter_combo.itemData(idx)
        if data is None:
            return ClusterFilterSelection()
        try:
            return ClusterFilterSelection(frozenset({int(data)}))
        except (TypeError, ValueError):
            return ClusterFilterSelection()

    def _selected_cluster_from_filter(self) -> int | None:
        """Return the single selected cluster, if the filter is single-cluster."""
        selection = self._selected_cluster_filter()
        if selection.include_unclustered or len(selection.cluster_ids) != 1:
            return None
        return next(iter(selection.cluster_ids))

    def _refresh_cluster_filter_controls(self) -> None:
        """Refresh cluster filter dropdown options from table cluster assignments."""
        previous = self._selected_cluster_filter()
        cluster_ids = self._neuron_table.available_cluster_ids()
        has_unclustered = False
        unclustered_getter = getattr(self._neuron_table, "has_unclustered_entries", None)
        if callable(unclustered_getter):
            has_unclustered = bool(unclustered_getter())

        setter = getattr(self._cluster_filter_combo, "set_cluster_options", None)
        if callable(setter):
            setter(
                cluster_ids,
                include_unclustered=has_unclustered,
                selection=previous,
            )
        else:
            self._cluster_filter_combo.blockSignals(True)
            try:
                self._cluster_filter_combo.clear()
                self._cluster_filter_combo.addItem("All")
                self._cluster_filter_combo.setItemData(0, None)
                for cluster_id in cluster_ids:
                    self._cluster_filter_combo.addItem(f"Cluster {cluster_id}")
                    self._cluster_filter_combo.setItemData(
                        self._cluster_filter_combo.count() - 1,
                        int(cluster_id),
                    )
                if previous.include_unclustered or len(previous.cluster_ids) != 1:
                    selected_cluster = None
                else:
                    selected_cluster = next(iter(previous.cluster_ids), None)
                if selected_cluster is not None:
                    idx = self._cluster_filter_combo.findData(selected_cluster)
                    self._cluster_filter_combo.setCurrentIndex(idx if idx >= 0 else 0)
                else:
                    self._cluster_filter_combo.setCurrentIndex(0)
            finally:
                self._cluster_filter_combo.blockSignals(False)

        self._on_cluster_filter_changed(self._selected_cluster_filter())

    def _on_cluster_filter_changed(self, _selection: object = None) -> None:
        """Filter table rows by selected cluster groups and update action buttons."""
        selection = self._selected_cluster_filter()
        self._apply_neuron_table_filters()

        has_filter = not selection.is_all
        has_entries = bool(self._neuron_table.get_visibility_map())
        self._hide_others_btn.setEnabled(has_filter)
        self._recolor_cluster_btn.setEnabled(has_filter)
        self._show_all_btn.setEnabled(has_entries)

    def _refresh_apply_existing_clusters_button(self) -> None:
        """Show the cached-cluster reapply button only when it can do useful work."""
        button = getattr(self, "_apply_existing_clusters_btn", None)
        analysis_tab = getattr(self, "_analysis_tab", None)
        if button is None:
            return

        has_overlap = False
        checker = getattr(analysis_tab, "has_cached_clusters_for_current_table", None)
        if callable(checker):
            try:
                has_overlap = bool(checker())
            except Exception:
                has_overlap = False

        button.setVisible(has_overlap)
        button.setEnabled(has_overlap)

    def _apply_existing_clusters_from_analysis(self) -> None:
        """Reapply cached cluster state to the current table and rendered neurons."""
        analysis_tab = getattr(self, "_analysis_tab", None)
        applier = getattr(analysis_tab, "apply_cluster_colors", None)
        if not callable(applier):
            return

        summary = applier()
        matched_table_count = int(getattr(summary, "matched_table_count", 0))
        if matched_table_count <= 0:
            self._refresh_apply_existing_clusters_button()
            return

        message = f"Applied cached cluster data to {matched_table_count} table neuron(s)."
        rendered_count = int(getattr(summary, "rendered_count", 0))
        if rendered_count > 0:
            colored_count = int(getattr(summary, "colored_count", 0))
            message += (
                f" Recolored {colored_count}/{rendered_count} rendered neuron(s)."
            )
            gray_count = int(getattr(summary, "gray_count", 0))
            if gray_count > 0:
                message += f" {gray_count} shown in gray."

        self._render_status_label.setText(message)

    def _hide_not_in_selected_cluster(self) -> None:
        """Set visibility off for neurons outside the selected cluster groups."""
        selection = self._selected_cluster_filter()
        if selection.is_all:
            return
        self._neuron_table.hide_all_not_in_cluster(selection)

    def _show_all_neurons(self) -> None:
        """Restore visibility on for all neurons in the table."""
        self._neuron_table.set_all_visible()

    def _recolor_selected_cluster(self) -> None:
        """Recolor selected cluster groups with turbo and gray non-selected neurons."""
        selection = self._selected_cluster_filter()
        if selection.is_all:
            return
        self._neuron_table.recolor_cluster_turbo(
            selection,
            gray_others=True,
        )

    def _remove_selected_from_table(self) -> None:
        """Remove selected neurons from the table while keeping the scene intact."""
        selected_file_ids = self._neuron_table.get_selected_file_ids()
        if not selected_file_ids:
            self._render_status_label.setText(
                "Select at least one neuron row to remove from the table."
            )
            return

        rendered_selected = [
            file_id for file_id in selected_file_ids
            if file_id in self._current_scene_file_ids()
        ]
        self._cache_scene_display_state(rendered_selected)
        self._neuron_table.remove_file_ids(selected_file_ids)
        self._discard_scene_display_state(self._current_table_file_ids())
        self._neuron_table.set_added_file_ids(self._current_scene_file_ids())
        self._sync_after_neuron_table_membership_change()

        message = f"Removed {len(selected_file_ids)} neuron(s) from the table."
        self._render_status_label.setText(message)
        self._regions_status_label.setText(message)

    def _clear_neuron_table(self) -> None:
        """Clear the neuron table while leaving rendered scene layers in place."""
        self._cache_scene_display_state(self._current_table_file_ids_in_scene())
        self._neuron_table.clear()
        self._discard_scene_display_state(self._current_table_file_ids())
        self._neuron_table.set_added_file_ids(self._current_scene_file_ids())
        self._sync_after_neuron_table_membership_change()

        if hasattr(self, "_render_status_label"):
            self._render_status_label.setText("Cleared neuron table.")
        if hasattr(self, "_regions_status_label"):
            self._regions_status_label.setText("Cleared neuron table.")

    def _add_selected_neurons_heatmap(self) -> None:
        """Build a node-count heatmap for the currently selected neurons."""
        if self._selected_heatmap_running():
            return
        if self._db is None:
            self._render_status_label.setText(
                "Load a neuron Parquet before creating a heatmap."
            )
            return
        if self._atlas is None:
            self._render_status_label.setText(
                "Load an atlas before creating a neuron heatmap."
            )
            return

        selected_file_ids = [
            str(file_id)
            for file_id in self._neuron_table.get_selected_file_ids()
        ]
        if not selected_file_ids:
            self._render_status_label.setText(
                "Select at least one neuron row to create a heatmap."
            )
            return

        self._start_selected_neuron_heatmap(selected_file_ids)

    def _start_selected_neuron_heatmap(self, file_ids: list[str]) -> None:
        """Start the background worker that builds a selected-neuron heatmap."""
        from ..workers import HeatmapWorker

        self._selected_heatmap_request_file_ids = tuple(str(file_id) for file_id in file_ids)

        self._render_progress.setVisible(True)
        self._render_progress.setRange(0, 0)
        self._render_status_label.setText(
            f"Building heatmap for {len(self._selected_heatmap_request_file_ids)} selected neuron(s)..."
        )

        thread = QThread()
        worker = HeatmapWorker(
            parquet_path=str(self._db.parquet_path),
            atlas=self._atlas,
            file_ids=list(self._selected_heatmap_request_file_ids),
        )
        self._selected_heatmap_thread = thread
        self._selected_heatmap_worker = worker
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_selected_neuron_heatmap_progress)
        worker.finished.connect(self._on_selected_neuron_heatmap_finished)
        worker.error.connect(self._on_selected_neuron_heatmap_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda: self._cleanup_selected_heatmap_thread(thread, worker)
        )

        self._update_selected_neuron_heatmap_controls()
        thread.start()

    def _on_selected_neuron_heatmap_progress(
        self,
        message: str,
        current: int,
        total: int,
    ) -> None:
        """Update the Data-tab progress widgets for selected heatmap creation."""
        self._render_progress.setVisible(True)
        self._render_progress.setRange(0, max(1, total))
        self._render_progress.setValue(current)
        self._render_status_label.setText(message)

    def _add_selected_neuron_heatmap_layer(
        self,
        volume: np.ndarray,
        file_ids: list[str] | tuple[str, ...],
    ):
        """Add one selected-neuron heatmap layer to the viewer."""
        layer_name = self._selected_neuron_heatmap_layer_name(file_ids)
        manual_heatmap_id = layer_name.removesuffix(" Heatmap")
        contrast_limits = _heatmap_contrast_limits(volume)
        metadata = {
            "heatmap_source": True,
            "heatmap_native_grid": True,
            "heatmap_kind": "selected_neurons",
            "manual_heatmap_id": manual_heatmap_id,
            "atlas_name": self._current_atlas_name(),
            "source_path": (
                str(self._db.parquet_path)
                if self._db is not None
                else None
            ),
            "file_ids": [str(file_id) for file_id in file_ids],
            "selection_count": len(file_ids),
            "heatmap_contrast_limits": contrast_limits,
            "heatmap_autocontrast_policy": "stable_full_volume",
        }
        return self.viewer.add_image(
            volume,
            name=layer_name,
            colormap="hot",
            blending="additive",
            rendering="mip",
            opacity=self._opacity_slider.value() / 100.0,
            contrast_limits=contrast_limits,
            metadata=metadata,
        )

    def _on_selected_neuron_heatmap_finished(self, volume: np.ndarray) -> None:
        """Handle successful selected-neuron heatmap creation."""
        file_ids = self._selected_heatmap_request_file_ids
        if not file_ids:
            return

        layer = self._add_selected_neuron_heatmap_layer(volume, file_ids)
        self._render_progress.setVisible(False)
        self._render_progress.setRange(0, 1)
        self._render_progress.setValue(0)
        self._render_status_label.setText(
            f"Added {layer.name} with {(volume > 0).sum():,} non-zero voxels."
        )
        logger.info(
            "Created selected-neuron heatmap %s for %d neuron(s)",
            layer.name,
            len(file_ids),
        )
        sync_heatmaps = getattr(self, "_sync_neuron_table_heatmap_membership", None)
        if callable(sync_heatmaps):
            sync_heatmaps()
        refresh_manual_heatmaps = getattr(self, "_refresh_manual_heatmap_combo", None)
        if callable(refresh_manual_heatmaps):
            refresh_manual_heatmaps()
        self._refresh_heatmap_layer_list()
        self._refresh_histogram_layer_list()
        self._refresh_mask_layer_options()

    def _on_selected_neuron_heatmap_error(self, error_msg: str) -> None:
        """Handle a selected-neuron heatmap worker failure."""
        self._render_progress.setVisible(False)
        self._render_progress.setRange(0, 1)
        self._render_progress.setValue(0)
        self._render_status_label.setText(f"Error: {error_msg}")
        logger.error("Selected-neuron heatmap failed: %s", error_msg)

    def _cleanup_selected_heatmap_thread(
        self,
        thread: QThread,
        worker: object,
    ) -> None:
        """Release selected-neuron heatmap worker objects after completion."""
        if self._selected_heatmap_thread is thread:
            self._selected_heatmap_thread = None
        if self._selected_heatmap_worker is worker:
            self._selected_heatmap_worker = None
        self._selected_heatmap_request_file_ids = ()
        self._update_selected_neuron_heatmap_controls()

    def _render_selected_neurons(self) -> None:
        """Render selected neurons with full trace geometry."""
        self._render_selected_with_mode(_SCENE_RENDER_MODE_FULL)

    def _render_selected_soma_only(self) -> None:
        """Render selected neurons using the shared soma-only layers."""
        self._render_selected_with_mode(_SCENE_RENDER_MODE_SOMA)

    def _render_selected_with_mode(self, render_mode: str) -> None:
        """Apply a scene render mode to the selected neurons and rebuild."""
        selected_file_ids = self._neuron_table.get_selected_file_ids()
        if not selected_file_ids:
            return

        changed = False
        for file_id in selected_file_ids:
            if self._scene_render_modes.get(file_id) == render_mode:
                continue
            self._scene_render_modes[file_id] = render_mode
            changed = True

        if not changed:
            mode_label = (
                "full traces"
                if render_mode == _SCENE_RENDER_MODE_FULL
                else "soma only"
            )
            self._render_status_label.setText(
                f"All selected neurons are already in the scene as {mode_label}."
            )
            return

        self._render_scene()

    def _remove_selected_neurons(self) -> None:
        """Remove selected neurons from the scene while leaving others in place."""
        selected_file_ids = set(self._neuron_table.get_selected_file_ids())
        if not selected_file_ids:
            return

        current_file_ids = self._current_scene_file_ids()
        if not current_file_ids:
            return

        removed_file_ids = current_file_ids & selected_file_ids
        if not removed_file_ids:
            self._render_status_label.setText(
                "No selected neurons are currently in the scene."
            )
            return

        for file_id in removed_file_ids:
            self._scene_render_modes.pop(file_id, None)
            scene_state = getattr(self, "_scene_display_state", None)
            if scene_state is not None:
                scene_state.pop(file_id, None)

        if not self._scene_render_modes:
            depth_state = self._capture_depth_state()
            self._clear_neuron_layers()
            self._restore_depth_state(depth_state)
            self._render_status_label.setText("Cleared all neurons from the scene.")
            return

        self._render_scene()

    def _render_scene(self) -> None:
        """Rebuild the neuron scene from the current per-neuron render modes."""
        if not self._scene_render_modes or self._db is None:
            return

        scene_render_modes = dict(self._scene_render_modes)
        file_ids = sorted(scene_render_modes, key=str)
        full_file_ids = [
            fid for fid in file_ids
            if scene_render_modes.get(fid) == _SCENE_RENDER_MODE_FULL
        ]
        n = len(file_ids)
        depth_state = self._capture_depth_state()
        use_auto_centering = self._use_auto_centering()

        # Show progress UI
        self._render_btn.setEnabled(False)
        self._render_soma_only_btn.setEnabled(False)
        self._remove_selected_btn.setEnabled(False)
        self._render_progress.setRange(0, n)
        self._render_progress.setValue(0)
        self._render_progress.setVisible(True)
        self._render_status_label.setText(f"Querying {n} neurons...")
        QApplication.processEvents()

        # Clear existing neuron layers
        self._clear_neuron_layers(reset_render_state=False)

        render_mode = self._render_mode_combo.currentText()
        opacity = self._opacity_slider.value() / 100.0

        color_map = self._build_effective_color_map()
        neuron_colors = []
        render_color_map: dict[object, list[float]] = {}
        for file_id in file_ids:
            fallback_color, _visible = self._base_display_state_for_file_id(file_id)
            color = list(color_map.get(file_id, fallback_color))
            neuron_colors.append(color)
            render_color_map[file_id] = color

        # Scale to match atlas mesh (coordinates are in microns)
        scale = None
        if self._atlas is not None:
            scale = [1.0 / res for res in self._atlas.resolution]

        line_data: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
        points_df = None

        # --- Lines ---
        if full_file_ids and render_mode in ("Lines", "Both"):
            # Single batch query for all neurons
            line_data = self._db.get_neuron_lines_batch(full_file_ids)

            self._render_status_label.setText(f"Building line segments for {n} neurons...")
            QApplication.processEvents()

            all_lines = []
            all_edge_colors = []
            projector_batch = {}
            rendered_file_ids = []
            segments_per_neuron = []

            for i, file_id in enumerate(full_file_ids):
                if file_id not in line_data:
                    continue
                color = render_color_map[file_id]
                coords, edges = line_data[file_id]
                if len(edges) == 0:
                    continue

                # Vectorized line segment building
                segments = np.stack(
                    [coords[edges[:, 0]], coords[edges[:, 1]]], axis=1
                )
                all_lines.append(segments)

                color_arr = np.empty((len(edges), 4))
                color_arr[:] = color[:4]
                all_edge_colors.append(color_arr)

                projector_batch[file_id] = (coords, edges, tuple(color))
                rendered_file_ids.append(file_id)
                segments_per_neuron.append(len(edges))

                self._render_progress.setValue(i + 1)
                if (i + 1) % 10 == 0:
                    QApplication.processEvents()

            if all_lines:
                merged_lines = np.concatenate(all_lines)
                merged_colors = np.concatenate(all_edge_colors)

                total_segs = len(merged_lines)
                self._render_status_label.setText(
                    f"Adding {total_segs:,} line segments to viewer..."
                )
                self._render_progress.setRange(0, 0)  # indeterminate
                QApplication.processEvents()

                layer = self.viewer.add_shapes(
                    merged_lines,
                    shape_type="line",
                    edge_width=self._line_width_spin.value(),
                    edge_color=merged_colors,
                    name="Neuron Lines",
                    opacity=opacity,
                    scale=scale,
                    metadata={
                        "file_ids": rendered_file_ids,
                        "segments_per_neuron": segments_per_neuron,
                    },
                )
                self._current_neuron_layers.append(layer)

            # Batch update slice projector (single rebuild)
            self._slice_projector.set_scale(scale)
            self._slice_projector.add_neuron_data_batch(projector_batch)

        # --- Points ---
        if full_file_ids and render_mode in ("Points", "Both"):
            self._render_status_label.setText("Querying point data...")
            self._render_progress.setRange(0, 0)  # indeterminate
            QApplication.processEvents()

            # Single batch query for all neurons
            points_df = self._db.get_neurons_for_rendering(full_file_ids)

            if not points_df.empty:
                self._render_status_label.setText(
                    f"Adding {len(points_df):,} points to viewer..."
                )
                QApplication.processEvents()

                coords = points_df[["x", "y", "z"]].values

                if self._color_by_type_cb.isChecked():
                    type_colors = {
                        1: [1, 0, 0, 1],  # Soma - red
                        2: [0, 0, 1, 1],  # Axon - blue
                        3: [0, 1, 0, 1],  # Basal dendrite - green
                        4: [1, 1, 0, 1],  # Apical dendrite - yellow
                    }
                    colors = np.array(
                        [
                            type_colors.get(t, [0.5, 0.5, 0.5, 1])
                            for t in points_df["type"].values
                        ]
                    )
                else:
                    # Per-point color based on which neuron each point belongs to
                    colors = np.array(
                        [
                            render_color_map.get(fid, list(_DEFAULT_NEURON_RGBA))[:4]
                            for fid in points_df["file_id"].values
                        ]
                    )

                layer = self.viewer.add_points(
                    coords,
                    size=self._point_size_spin.value(),
                    face_color=colors,
                    name="Neuron Points",
                    opacity=opacity,
                    scale=scale,
                    metadata={
                        "file_ids_per_point": points_df["file_id"].values.tolist(),
                        "point_types": points_df["type"].values.tolist(),
                        "base_face_colors": colors.copy(),
                    },
                )
                self._current_neuron_layers.append(layer)

        # --- Soma Labels ---
        soma_df = self._db.get_soma_locations(file_ids)
        if not soma_df.empty:
            soma_coords = soma_df[["x", "y", "z"]].values
            soma_fids = soma_df["file_id"].values.tolist()
            soma_colors = np.array(
                [
                    render_color_map.get(fid, list(_DEFAULT_NEURON_RGBA))[:4]
                    for fid in soma_fids
                ]
            )
            # Use neuron_id for the label text (shorter than file_id)
            labels = soma_df["neuron_id"].astype(str).values.tolist()

            soma_layer = self.viewer.add_points(
                soma_coords,
                size=50,
                face_color=soma_colors,
                border_color="white",
                border_width=0.05,
                text={"string": labels, "size": 10, "color": "white"},
                name="Soma Labels",
                opacity=0.7,
                scale=scale,
                metadata={"file_ids": soma_fids},
            )
            soma_layer.mode = "select"
            # Disable point movement — select mode allows clicking to
            # select but dragging would move points; override _move to
            # prevent that.
            soma_layer._move = lambda indices, position: None
            soma_layer.events.highlight.connect(self._on_soma_selected)
            self._current_neuron_layers.append(soma_layer)

        soma_projection_batch = self._build_soma_projection_batch(
            file_ids=file_ids,
            neuron_colors=neuron_colors,
            points_df=points_df,
        )
        self._soma_slice_projector.set_scale(scale)
        self._soma_slice_projector.point_size = _SOMA_SLICE_PROJECTION_POINT_SIZE
        self._soma_slice_projector.add_soma_data_batch(soma_projection_batch)

        # Hide neuron layers if currently in 2D mode (the ndisplay event
        # only fires on *changes*, so layers added while already in 2D
        # would otherwise stay visible).
        if self.viewer.dims.ndisplay == 2:
            self._apply_layer_visibility(False)

        # Default to showing the 2D slice projection once neurons are present.
        # This keeps "Show in 2D slices" on across subsequent additions.
        if self._current_neuron_layers:
            self._show_slice_projection_cb.setChecked(True)

        if use_auto_centering:
            centered = self._maybe_auto_center_slice(
                line_data=line_data,
                points_df=points_df,
                soma_df=soma_df,
                scale=scale,
            )
            if not centered:
                self._restore_depth_state(depth_state)
        else:
            self._restore_depth_state(depth_state)
        self._neuron_table.set_added_file_ids(self._current_scene_file_ids())

        # Hide progress UI
        self._render_progress.setVisible(False)
        self._render_status_label.setText(f"Rendered {n} neurons.")
        self._render_btn.setEnabled(True)
        self._render_soma_only_btn.setEnabled(True)
        self._remove_selected_btn.setEnabled(True)

    def _use_auto_centering(self) -> bool:
        """Return whether the Data tab centering mode is set to Auto."""
        return self._centering_mode_combo.currentData() == "auto"

    def _capture_depth_state(self) -> tuple[int, float] | None:
        """Capture current depth axis and point so Add/Remove can restore it."""
        depth_axis = depth_axis_from_not_displayed(
            getattr(self.viewer.dims, "not_displayed", None)
        )
        try:
            depth_value = float(self.viewer.dims.point[depth_axis])
        except Exception:
            return None
        return depth_axis, depth_value

    def _restore_depth_state(self, depth_state: tuple[int, float] | None) -> None:
        """Restore a previously captured depth state."""
        if depth_state is None:
            return

        depth_axis, depth_value = depth_state
        try:
            self.viewer.dims.set_point(depth_axis, depth_value)
            return
        except Exception:
            pass

        fallback_axis = depth_axis_from_not_displayed(
            getattr(self.viewer.dims, "not_displayed", None)
        )
        try:
            self.viewer.dims.set_point(fallback_axis, depth_value)
        except Exception:
            logger.debug("Failed to restore depth slice position.", exc_info=True)

    def _current_scene_file_ids(self) -> set[object]:
        """Collect neuron file IDs currently tracked in the scene state."""
        return set(self._scene_render_modes.keys())

    def _build_soma_projection_batch(
        self,
        file_ids: list[object],
        neuron_colors: list[list[float]],
        points_df,
    ) -> dict[str, tuple[np.ndarray, tuple]]:
        """Build a per-neuron soma-point batch for the 2D projection layer."""
        soma_points_df = None
        if points_df is not None and not points_df.empty and "type" in points_df:
            soma_points_df = points_df[points_df["type"] == 1]

        color_map = {
            fid: tuple(color[:4])
            for fid, color in zip(file_ids, neuron_colors)
        }
        default_color = (0.5, 0.5, 0.5, 1.0)
        batch = {}

        if soma_points_df is not None and not soma_points_df.empty:
            for file_id, group in soma_points_df.groupby("file_id", sort=True):
                coords = group[["x", "y", "z"]].values.astype(np.float64)
                batch[file_id] = (coords, color_map.get(file_id, default_color))

        missing_file_ids = [fid for fid in file_ids if fid not in batch]
        if missing_file_ids and self._db is not None:
            missing_soma_points_df = self._db.get_soma_points(missing_file_ids)
            if not missing_soma_points_df.empty:
                for file_id, group in missing_soma_points_df.groupby("file_id", sort=True):
                    coords = group[["x", "y", "z"]].values.astype(np.float64)
                    batch[file_id] = (coords, color_map.get(file_id, default_color))

        return batch

    def _soma_projection_active_in_2d(self) -> bool:
        """Return whether the shared 2D projection is active in 2D mode."""
        return bool(
            self.viewer.dims.ndisplay == 2
            and self._show_slice_projection_cb.isChecked()
        )

    def _set_neuron_points_soma_visibility(
        self,
        layer,
        hide_soma_points: bool,
    ) -> None:
        """Hide or restore soma-node entries within the Neuron Points layer."""
        meta = _layer_metadata(layer)
        point_types = np.asarray(meta.get("point_types", []))
        if point_types.size == 0:
            return

        base_colors = meta.get("base_face_colors")
        if base_colors is None:
            face_color = getattr(layer, "face_color", None)
            if face_color is None:
                return
            base_colors = np.asarray(face_color, dtype=float)
        else:
            base_colors = np.asarray(base_colors, dtype=float)

        if base_colors.ndim != 2 or base_colors.shape[0] != point_types.shape[0]:
            return

        meta["base_face_colors"] = base_colors.copy()
        layer.metadata = meta

        colors = base_colors.copy()
        if hide_soma_points:
            colors[point_types == 1, 3] = 0.0
        layer.face_color = colors

    def _sync_soma_projection_overlay_state(self) -> None:
        """Keep soma labels and raw soma nodes in sync with 2D projection state."""
        projection_active = self._soma_projection_active_in_2d()
        hide_soma_points = projection_active and (
            self._render_mode_combo.currentText() == "Points"
        )
        for layer in self._current_neuron_layers:
            if layer.name == "Soma Labels" and self.viewer.dims.ndisplay == 2:
                layer.visible = not projection_active
            elif layer.name == "Neuron Points":
                self._set_neuron_points_soma_visibility(layer, hide_soma_points)

    def _compute_center_of_rendered_neurons(
        self,
        line_data: dict[str, tuple[np.ndarray, np.ndarray]] | None,
        points_df,
        soma_df,
    ) -> np.ndarray | None:
        """Compute a center point with line > points > soma fallback priority."""
        return compute_center_of_rendered_neurons(
            line_data=line_data,
            points_df=points_df,
            soma_df=soma_df,
        )

    def _set_slice_depth_from_center(
        self,
        center_xyz: np.ndarray,
        scale: list[float] | None,
    ) -> bool:
        """Move the active depth slice to the center point."""
        depth_axis = depth_axis_from_not_displayed(
            getattr(self.viewer.dims, "not_displayed", None)
        )
        target_world = center_to_depth_world(center_xyz, depth_axis, scale)

        try:
            self.viewer.dims.set_point(depth_axis, target_world)
            return True
        except Exception:
            logger.debug("Failed to auto-center depth slice.", exc_info=True)
            return False

    def _maybe_auto_center_slice(
        self,
        line_data: dict[str, tuple[np.ndarray, np.ndarray]] | None,
        points_df,
        soma_df,
        scale: list[float] | None,
    ) -> bool:
        """Auto-center the depth slice once per widget session."""
        if self._auto_center_applied_once:
            return False

        center_xyz = self._compute_center_of_rendered_neurons(
            line_data=line_data,
            points_df=points_df,
            soma_df=soma_df,
        )
        if center_xyz is None:
            return False

        if self._set_slice_depth_from_center(center_xyz, scale):
            self._auto_center_applied_once = True
            return True
        return False

    def _base_display_state_for_file_id(
        self,
        file_id: object,
    ) -> tuple[list[float], bool]:
        """Return the stored base color and visibility for a neuron."""
        entries = getattr(self._neuron_table, "_entries", {})
        entry = entries.get(file_id)
        if entry is not None:
            return list(entry.color), bool(entry.visible)

        color_getter = getattr(self._neuron_table, "get_color", None)
        visibility_getter = getattr(self._neuron_table, "get_visibility_map", None)
        if callable(color_getter):
            color = list(color_getter(file_id))
            visible_map = visibility_getter() if callable(visibility_getter) else {}
            visible = bool(visible_map.get(file_id, True))
            return color, visible

        scene_state = getattr(self, "_scene_display_state", {})
        state = scene_state.get(file_id)
        if isinstance(state, dict):
            color = state.get("color", _DEFAULT_NEURON_RGBA)
            visible = bool(state.get("visible", True))
            return list(color), visible

        return list(_DEFAULT_NEURON_RGBA), True

    def _build_effective_color_map(self) -> dict[str, list[float]]:
        """Build a color map accounting for visibility and highlight state.

        - Hidden neurons get alpha=0.
        - When a highlight is active, non-highlighted neurons are dimmed to
          alpha=0.1 so the highlighted ones stand out.
        """
        highlight = getattr(self, "_highlighted_file_ids", None)
        result = {}
        entries = getattr(self._neuron_table, "_entries", {})
        file_ids = list(entries.keys())
        scene_state = getattr(self, "_scene_display_state", {})
        for fid in scene_state:
            if fid not in entries:
                file_ids.append(fid)

        for fid in file_ids:
            color, visible = self._base_display_state_for_file_id(fid)
            if not visible:
                color[3] = 0.0
            elif highlight is not None and fid not in highlight:
                color[3] = 0.1
            result[fid] = color
        return result

    def _update_layer_colors(self, color_map: dict[str, list[float]]) -> None:
        """Apply a color map to all neuron layers."""
        default_color = list(_DEFAULT_NEURON_RGBA)

        for layer in self._current_neuron_layers:
            if layer.name == "Neuron Lines":
                meta = layer.metadata or {}
                file_ids = meta.get("file_ids", [])
                seg_counts = meta.get("segments_per_neuron", [])
                if file_ids and seg_counts:
                    parts = []
                    for fid, count in zip(file_ids, seg_counts):
                        c = color_map.get(fid, default_color)
                        arr = np.empty((count, 4))
                        arr[:] = c[:4]
                        parts.append(arr)
                    layer.edge_color = np.concatenate(parts)

            elif layer.name == "Neuron Points":
                meta = layer.metadata or {}
                fids = meta.get("file_ids_per_point", [])
                if fids:
                    colors = np.array(
                        [color_map.get(fid, default_color)[:4] for fid in fids]
                    )
                    meta["base_face_colors"] = colors.copy()
                    layer.metadata = meta
                    layer.face_color = colors

            elif layer.name == "Soma Labels":
                meta = layer.metadata or {}
                fids = meta.get("file_ids", [])
                if fids:
                    colors = np.array(
                        [color_map.get(fid, default_color)[:4] for fid in fids]
                    )
                    layer.face_color = colors

        # Update slice projector
        self._slice_projector.update_neuron_colors(color_map)
        self._soma_slice_projector.update_neuron_colors(color_map)
        self._sync_soma_projection_overlay_state()

    def _apply_neuron_colors(self, changed: dict[str, list[float]]) -> None:
        """Handle color changes from the neuron table."""
        if not self._current_neuron_layers:
            return
        color_map = self._build_effective_color_map()
        self._update_layer_colors(color_map)

    def _apply_neuron_visibility(self, visibility_map: dict[str, bool]) -> None:
        """Handle visibility changes from the neuron table."""
        if not self._current_neuron_layers:
            return
        color_map = self._build_effective_color_map()
        self._update_layer_colors(color_map)

    def _highlight_selected_neurons(self, selected_file_ids: list[str]) -> None:
        """Highlight selected neurons by dimming all others.

        When some (but not all) neurons are selected in the table, the
        non-selected rendered neurons are dimmed to alpha=0.1. When nothing
        is selected or all are selected, highlighting is cleared.
        """
        if not self._current_neuron_layers:
            self._highlighted_file_ids = None
            return

        rendered_ids = set(getattr(self, "_scene_render_modes", {}).keys())
        selected = set(selected_file_ids) & rendered_ids
        if not selected or selected == rendered_ids:
            # Nothing selected or everything selected → clear highlight
            self._highlighted_file_ids = None
        else:
            self._highlighted_file_ids = selected

        color_map = self._build_effective_color_map()
        self._update_layer_colors(color_map)

    def _on_soma_selected(self, event) -> None:
        """Handle point selection on the Soma Labels layer.

        The highlight event fires on every mouse hover, not just clicks.
        We track the previous selection and only process when it actually
        changes to avoid expensive color updates on every mouse move.
        """
        layer = event.source
        current = set(layer.selected_data)
        if current == self._last_soma_selection:
            return
        self._last_soma_selection = current

        if not current:
            return

        file_ids = layer.metadata.get("file_ids", [])
        selected_fids = [
            file_ids[i] for i in current if i < len(file_ids)
        ]
        if selected_fids:
            self._neuron_table.select_file_ids(selected_fids)

    def _on_cluster_colors_updated(self, result, color_map: dict) -> None:
        """Handle cluster color updates from the analysis tab."""
        self._neuron_table.update_cluster_assignments(result)
        self._neuron_table.update_colors(color_map, emit_signal=False)
        self._neuron_table.sort_by_cluster()
        self._refresh_cluster_filter_controls()
        self._refresh_apply_existing_clusters_button()

    def _clear_neuron_layers(self, reset_render_state: bool = True) -> None:
        """Remove all current neuron layers and optionally reset scene state."""
        for layer in self._current_neuron_layers:
            try:
                self.viewer.layers.remove(layer)
            except ValueError:
                pass  # Layer already removed

        self._current_neuron_layers.clear()
        if reset_render_state:
            self._scene_render_modes.clear()
            scene_state = getattr(self, "_scene_display_state", None)
            if scene_state is not None:
                scene_state.clear()

        # Clear slice projector data
        self._slice_projector.clear()
        self._soma_slice_projector.clear()
        self._neuron_table.set_added_file_ids(self._current_scene_file_ids())

    def _toggle_template(self, state: int) -> None:
        """Toggle the template layer visibility."""
        requested = bool(state)
        with startup_timing(
            logger,
            "toggle_template",
            requested_state=requested,
            atlas_loaded=self._atlas is not None,
        ) as timing:
            if self._atlas is None:
                if not requested:
                    self._show_template_after_cached_atlas_load = False
                    timing.set(result="no_atlas_hide")
                    return
                if self._cached_atlas_autoload_running():
                    self._show_template_after_cached_atlas_load = True
                    timing.set(result="cached_atlas_loading")
                    return
                with startup_timing(
                    logger,
                    "toggle_template_phase",
                    phase="load_atlas",
                ):
                    self._load_atlas()
                if self._atlas is None:
                    self._show_template_cb.setChecked(False)
                    timing.set(result="atlas_unavailable")
                    return

            layer_name = "Allen Template"

            if requested:
                existing = [
                    layer for layer in self.viewer.layers
                    if layer.name == layer_name
                ]
                timing.set(existing_template_layers=len(existing))
                if not existing:
                    opacity = self._template_opacity_slider.value() / 100.0
                    with startup_timing(
                        logger,
                        "toggle_template_phase",
                        phase="add_allen_template",
                        opacity=opacity,
                    ):
                        add_allen_template(
                            self.viewer,
                            self._atlas,
                            opacity=opacity,
                        )
            else:
                removed = False
                for layer in self.viewer.layers:
                    if layer.name == layer_name:
                        self.viewer.layers.remove(layer)
                        removed = True
                        break
                timing.set(removed=removed)

    def _update_template_opacity(self, value: int) -> None:
        """Update the template layer opacity."""
        opacity = value / 100.0
        for layer in self.viewer.layers:
            if layer.name == "Allen Template":
                layer.opacity = opacity
                break

    def _toggle_outline(self, state: int) -> None:
        """Toggle the brain outline visibility."""
        if self._atlas is None:
            self._load_atlas()
            if self._atlas is None:
                self._show_outline_cb.setChecked(False)
                return

        layer_name = "Brain Outline"

        if bool(state):
            existing = [
                layer for layer in self.viewer.layers if layer.name == layer_name
            ]
            if not existing:
                # Switch to 3D mode for mesh viewing
                if self.viewer.dims.ndisplay == 2:
                    self.viewer.dims.ndisplay = 3
                    show_info("Switched to 3D view for mesh display")
                add_brain_outline(self.viewer, self._atlas)
        else:
            for layer in self.viewer.layers:
                if layer.name == layer_name:
                    self.viewer.layers.remove(layer)
                    break

    def _toggle_region_meshes(self, state: int) -> None:
        """Toggle region mesh visibility."""
        if bool(state):
            self._update_region_meshes(self._active_region_preview_acronyms())
        else:
            remove_region_layers(self.viewer)

    def _update_region_meshes(self, acronyms: list[str]) -> None:
        """Update displayed region meshes."""
        if self._atlas is None:
            self._load_atlas()
            if self._atlas is None:
                return

        # Remove existing region meshes
        remove_region_layers(self.viewer)

        if not self._show_region_meshes_cb.isChecked():
            return

        if not acronyms:
            return

        # Switch to 3D mode for mesh viewing
        if self.viewer.dims.ndisplay == 2:
            self.viewer.dims.ndisplay = 3
            show_info("Switched to 3D view for mesh display")

        # Add new meshes
        opacity = self._mesh_opacity_slider.value() / 100.0
        for acronym in acronyms:
            add_region_mesh(self.viewer, self._atlas, acronym, opacity=opacity)

    def _toggle_region_segmentation(self, state: int) -> None:
        """Toggle region segmentation visibility."""
        if bool(state):
            self._update_region_segmentation(self._active_region_preview_acronyms())
        else:
            remove_region_segmentation(self.viewer)

    def _update_region_segmentation(self, acronyms: list[str]) -> None:
        """Update the region segmentation layer for selected regions."""
        if self._atlas is None:
            self._load_atlas()
            if self._atlas is None:
                return

        remove_region_segmentation(self.viewer)

        if not self._show_region_seg_cb.isChecked():
            return

        if not acronyms:
            return

        opacity = self._seg_opacity_slider.value() / 100.0
        add_region_segmentation(
            self.viewer, self._atlas, acronyms, opacity=opacity
        )

    def _update_seg_opacity(self, value: int) -> None:
        """Update the region segmentation layer opacity."""
        opacity = value / 100.0
        for layer in self.viewer.layers:
            if layer.name == "Region Segmentation":
                layer.opacity = opacity
                break

    def _on_ndisplay_changed(self, event) -> None:
        """Auto-hide neuron line/point layers in 2D to keep slice scrubbing fast."""
        if not self._current_neuron_layers:
            return

        is_2d = self.viewer.dims.ndisplay == 2
        if is_2d:
            self.viewer.status = "Switching to 2D view..."
        else:
            self.viewer.status = "Rendering 3D neuron layers..."
        # Defer the heavy work so the status bar paints first
        QTimer.singleShot(0, lambda: self._apply_layer_visibility(not is_2d))

    def _apply_layer_visibility(self, visible: bool) -> None:
        """Set visibility on all neuron layers and clear the status message.

        In 2D mode, the raw points layer can stay visible for `Points` render
        mode while line layers are hidden for responsiveness.
        """
        show_points_in_2d = (
            not visible and self._render_mode_combo.currentText() == "Points"
        )
        for layer in self._current_neuron_layers:
            if not visible and layer.name == "Soma Labels":
                layer.visible = True
                continue
            if show_points_in_2d and layer.name == "Neuron Points":
                layer.visible = True
                continue
            layer.visible = visible
        self._sync_soma_projection_overlay_state()
        self.viewer.status = "Ready"

    def _toggle_slice_projection(self, state: int) -> None:
        """Toggle the 2D slice projection visibility."""
        enabled = bool(state)
        self._slice_projector.enabled = enabled
        self._soma_slice_projector.enabled = enabled
        self._slice_warning_label.setVisible(enabled)
        if self._current_neuron_layers and self.viewer.dims.ndisplay == 2:
            self._apply_layer_visibility(False)

    def _update_slice_thickness(self, value: int) -> None:
        """Update the slice projection thickness/tolerance."""
        self._slice_projector.tolerance = float(value)
        self._soma_slice_projector.tolerance = float(value)

    def _update_line_width(self, value: int) -> None:
        """Update line width for both neuron layers and projection."""
        for layer in self._current_neuron_layers:
            if hasattr(layer, "edge_width"):
                layer.edge_width = value
        self._slice_projector.edge_width = value

    # --- SWC-to-Parquet conversion ---

    def _convert_from_directory(self) -> None:
        """Pick a directory of SWC files and convert to Parquet."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory of SWC Files"
        )
        if not directory:
            return

        swc_files = sorted(Path(directory).rglob("*.swc"))
        if not swc_files:
            self._convert_status_label.setText("No SWC files found in directory.")
            return

        self._prompt_output_and_convert([str(f) for f in swc_files])

    def _convert_from_files(self) -> None:
        """Pick individual SWC files and convert to Parquet."""
        filepaths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select SWC Files",
            "",
            "SWC Files (*.swc);;All Files (*)",
        )
        if not filepaths:
            return

        self._prompt_output_and_convert(filepaths)

    def _prompt_output_and_convert(self, swc_paths: list[str]) -> None:
        """Ask for output path and start conversion."""
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Parquet File",
            "neurons.parquet",
            "Parquet Files (*.parquet)",
        )
        if not output_path:
            return

        self._start_conversion(swc_paths, output_path)

    def _start_conversion(self, swc_paths: list[str], output_path: str) -> None:
        """Launch the background conversion worker."""
        from ..workers import ConvertWorker

        resolution = self._convert_resolution_spin.value()
        hemisphere = self._convert_hemisphere_combo.currentData()
        atlas_name = self._atlas_combo.currentText()

        status = f"Converting {len(swc_paths)} SWC files..."
        if hemisphere is not None:
            status = (
                f"Converting {len(swc_paths)} SWC files "
                f"(aligning to {str(hemisphere).title()})..."
            )

        self._convert_progress.setVisible(True)
        self._convert_progress.setRange(0, len(swc_paths))
        self._convert_progress.setValue(0)
        self._convert_status_label.setText(status)

        self._convert_thread = QThread()
        self._convert_worker = ConvertWorker(
            swc_paths,
            output_path,
            resolution,
            hemisphere=hemisphere,
            atlas_name=atlas_name,
        )
        self._convert_worker.moveToThread(self._convert_thread)

        self._convert_thread.started.connect(self._convert_worker.run)
        self._convert_worker.progress.connect(self._on_convert_progress)
        self._convert_worker.finished.connect(self._on_convert_finished)
        self._convert_worker.error.connect(self._on_convert_error)
        self._convert_worker.finished.connect(self._convert_thread.quit)
        self._convert_worker.error.connect(self._convert_thread.quit)

        self._convert_thread.start()

    def _on_convert_progress(self, message: str, current: int, total: int) -> None:
        """Handle conversion progress updates."""
        self._convert_progress.setValue(current)
        self._convert_status_label.setText(message)

    def _on_convert_finished(self, output_path: str, summary: object) -> None:
        """Handle conversion completion."""
        self._convert_progress.setVisible(False)
        summary_parts = [f"Converted {summary.processed_files} file(s)"]
        if summary.failed_files:
            summary_parts.append(f"skipped {summary.failed_files}")
        if summary.flipped_files:
            summary_parts.append(f"flipped {summary.flipped_files}")
        self._convert_status_label.setText(
            f"Done! {', '.join(summary_parts)} -> {Path(output_path).name}"
        )
        logger.info(f"SWC-to-Parquet conversion complete: {output_path}")

    def _on_convert_error(self, error_msg: str) -> None:
        """Handle conversion error."""
        self._convert_progress.setVisible(False)
        self._convert_status_label.setText(f"Error: {error_msg}")
        logger.error(f"SWC-to-Parquet conversion failed: {error_msg}")
