"""Qt widget for lookup-based isocortex flatmap projection."""

from __future__ import annotations

import logging
from pathlib import Path
import re
from types import MethodType
from typing import Callable

import numpy as np
import pandas as pd
from napari.utils.notifications import show_info, show_warning
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..flatmap_export import export_projected_nodes_csv
from ..analysis.flatmap_correlation import FlatmapVoxelCorrelationSource
from ..flatmap_heatmap import (
    DEFAULT_FLATMAP_DEPTH_BIN_UM,
    DEFAULT_FLATMAP_XY_BINS,
    FlatmapGroupedVolume,
    FlatmapLookupStats,
    FlatmapRenderResult,
    FlatmapRenderSummary,
    build_flatmap_cluster_volumes,
    build_flatmap_file_id_volumes,
    build_flatmap_render_data,
    build_flatmap_render_data_from_projected_nodes,
    compute_flatmap_lookup_stats,
)
from ..flatmap_labels import (
    FlatmapRegionLabelsResult,
    build_flatmap_region_label_volume,
)
from ..flatmap_loader import FLATMAP_STYLE_FILENAMES, load_flatmap_volume_set
from ..flatmap_parquet import (
    FLATMAP_V3_AUGMENTED_COLUMNS,
    FLATMAP_V3_STYLE_COLUMN_MAPPING,
    augment_neuron_parquet_with_flatmap,
    flatmap_invalid_code_to_reason,
    read_flatmap_parquet_transform_info,
)
from ..flatmap_projection import (
    COORDINATE_MODE_MICRONS,
    COORDINATE_MODE_VOXELS,
    DEFAULT_CCF_RESOLUTION_UM,
    FLATMAP_LOOKUP_DIRECT,
    FLATMAP_LOOKUP_MIRRORED,
    FLATMAP_LOOKUP_MIRRORED_DEPTH,
    FLATMAP_LOOKUP_UNMAPPED,
    FlatmapProjectionResult,
    ProjectionSummary,
    build_projected_segments,
    project_and_build_segments,
    summarize_projection,
)

logger = logging.getLogger(__name__)

_SOURCE_SELECTED = "selected"
_SOURCE_ALL = "all"
_RENDER_HEATMAP = "heatmap"
_RENDER_POINTS = "points"
_HEATMAP_COLOR_SINGLE = "single"
_HEATMAP_COLOR_INDIVIDUAL = "individual"
_HEATMAP_COLOR_CLUSTER = "cluster"
_PROJECTION_SOURCE_PRECOMPUTED = "precomputed"
_PROJECTION_SOURCE_RECOMPUTE = "recompute"
_OLD_SHAPES_LAYER_NAME = "Isocortex Flatmap Traces"
_HEATMAP_LAYER_NAME = "Isocortex Flatmap Heatmap"
_GROUPED_HEATMAP_LAYER_PREFIX = f"{_HEATMAP_LAYER_NAME}: "
_POINTS_LAYER_NAME = "Isocortex Flatmap Points"
_REGION_LABELS_LAYER_NAME = "Flatmap Region Labels"
_REGION_SURFACES_LAYER_NAME = "Flatmap Region Surfaces"
_REGION_OUTLINES_LAYER_NAME = "Flatmap Region Outlines"
_REGION_LABEL_ATLAS_DEFAULT = "allen_mouse_10um"
_REGION_LABEL_ATLAS_OPTIONS = (
    "allen_mouse_10um",
    "allen_mouse_25um",
    "allen_mouse_50um",
)
_FLATMAP_RENDER_LAYER_NAMES = {
    _OLD_SHAPES_LAYER_NAME,
    _HEATMAP_LAYER_NAME,
    _POINTS_LAYER_NAME,
}
_DEFAULT_TRACE_COLOR = np.asarray([0.5, 0.5, 0.5, 1.0], dtype=float)


class FlatmapProjectionWidget(QWidget):
    """Project loaded neuron rows into precomputed isocortex flatmap space."""

    def __init__(
        self,
        viewer,
        *,
        database_provider: Callable[[], object | None],
        selected_file_ids_provider: Callable[[], list[object]],
        table_file_ids_provider: Callable[[], list[object]],
        color_map_provider: Callable[[], dict[object, list[float]]],
        cluster_map_provider: Callable[[], dict[object, int | None]] | None = None,
        atlas_provider: Callable[[], object | None] | None = None,
        selected_region_ids_provider: Callable[[], list[int]] | None = None,
        selected_parent_region_ids_provider: Callable[[], list[int]] | None = None,
        selected_region_acronyms_provider: Callable[[], list[str]] | None = None,
        display_viewer_provider: Callable[..., object | None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._display_viewer_provider = display_viewer_provider
        self._last_display_viewer = None
        self._database_provider = database_provider
        self._selected_file_ids_provider = selected_file_ids_provider
        self._table_file_ids_provider = table_file_ids_provider
        self._color_map_provider = color_map_provider
        self._cluster_map_provider = cluster_map_provider or (lambda: {})
        self._atlas_provider = atlas_provider or (lambda: None)
        self._selected_region_ids_provider = selected_region_ids_provider or (
            lambda: []
        )
        self._selected_parent_region_ids_provider = (
            selected_parent_region_ids_provider or self._selected_region_ids_provider
        )
        self._selected_region_acronyms_provider = selected_region_acronyms_provider or (
            lambda: []
        )

        self._flatmap_path: Path | None = None
        self._depth_path: Path | None = None
        self._preprocess_lookup_dir: Path | None = None
        self._region_cache_dir: Path | None = None
        self._region_cache = None
        self._active_cache_profile = None
        self._pending_cache_profile_id: str | None = None
        self._projection_layer = None
        self._region_labels_layer = None
        self._region_surfaces_layers: list[object] = []
        self._region_outlines_layers: list[object] = []
        self._region_label_atlas_cache: dict[str, object] = {}
        self._region_label_atlas_load_thread = None
        self._region_label_atlas_load_worker = None
        self._pending_region_label_request = False
        self._last_projected_nodes: pd.DataFrame | None = None
        self._last_summary: ProjectionSummary | None = None
        self._last_render_summary: FlatmapRenderSummary | None = None
        self._last_render_mode: str | None = None
        self._last_flatmap_style: str | None = None
        self._last_coordinate_mode: str | None = None
        self._last_volume_shape: tuple[int, int, int] | None = None
        self._last_lookup_stats: FlatmapLookupStats | None = None
        self._last_input_file_ids: tuple[str, ...] = ()
        self._last_flatmap_path: str | None = None
        self._last_depth_path: str | None = None
        self._last_projection_source: str | None = None
        self._last_cache_dir: str | None = None
        self._last_cache_profile_id: str | None = None
        self._flatmap_correlation_source_changed_callback = None
        self._lookup_stats_cache_key: tuple[object, ...] | None = None
        self._lookup_stats_cache: FlatmapLookupStats | None = None

        self._setup_ui()

    def _resolve_display_viewer(self, *, create: bool):
        """Return the viewer used for flatmap display layers."""
        provider = getattr(self, "_display_viewer_provider", None)
        if callable(provider):
            try:
                viewer = provider(create=create)
            except TypeError:
                if not create:
                    return getattr(self, "_last_display_viewer", None)
                viewer = provider()
            if viewer is not None:
                self._last_display_viewer = viewer
            return viewer
        return getattr(self, "_viewer", None)

    def _display_viewer(self):
        return self._resolve_display_viewer(create=True)

    def _current_display_viewer(self):
        return self._resolve_display_viewer(create=False)

    def _display_layers(self, *, create: bool = True):
        viewer = self._resolve_display_viewer(create=create)
        if viewer is None:
            return None
        return getattr(viewer, "layers", None)

    def _setup_ui(self) -> None:
        """Build the tab UI."""
        parent_layout = QVBoxLayout(self)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        parent_layout.addWidget(scroll_area)

        content = QWidget()
        scroll_area.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)

        files_group = QGroupBox("Flatmap Lookup Files")
        files_layout = QVBoxLayout(files_group)

        projection_source_row = QHBoxLayout()
        projection_source_row.addWidget(QLabel("Source:"))
        self._projection_source_combo = QComboBox()
        self._projection_source_combo.addItem(
            "Precomputed Parquet + Cache",
            _PROJECTION_SOURCE_PRECOMPUTED,
        )
        self._projection_source_combo.addItem(
            "Recompute from NRRDs",
            _PROJECTION_SOURCE_RECOMPUTE,
        )
        self._projection_source_combo.currentIndexChanged.connect(
            self._on_projection_source_changed
        )
        projection_source_row.addWidget(self._projection_source_combo)
        files_layout.addLayout(projection_source_row)

        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Style:"))
        self._style_combo = QComboBox()
        self._style_combo.addItem("Both hemispheres, shaped", "both_shaped")
        self._style_combo.addItem("Both hemispheres, square", "both_square")
        self._style_combo.addItem("Single hemisphere, shaped", "single_shaped")
        self._style_combo.addItem("Single hemisphere, square", "single_square")
        self._style_combo.currentIndexChanged.connect(self._on_flatmap_style_changed)
        style_row.addWidget(self._style_combo)
        files_layout.addLayout(style_row)

        self._expected_filename_label = QLabel("")
        self._expected_filename_label.setWordWrap(True)
        files_layout.addWidget(self._expected_filename_label)

        flatmap_row = QHBoxLayout()
        self._flatmap_path_label = QLabel("No flatmap selected")
        self._flatmap_path_label.setWordWrap(True)
        flatmap_row.addWidget(self._flatmap_path_label, stretch=1)
        flatmap_btn = QPushButton("Choose Flatmap...")
        flatmap_btn.clicked.connect(self._choose_flatmap_path)
        flatmap_row.addWidget(flatmap_btn)
        files_layout.addLayout(flatmap_row)

        depth_row = QHBoxLayout()
        self._depth_path_label = QLabel("No depth selected")
        self._depth_path_label.setWordWrap(True)
        depth_row.addWidget(self._depth_path_label, stretch=1)
        depth_btn = QPushButton("Choose Depth...")
        depth_btn.clicked.connect(self._choose_depth_path)
        depth_row.addWidget(depth_btn)
        files_layout.addLayout(depth_row)

        lookup_dir_row = QHBoxLayout()
        self._lookup_dir_label = QLabel("No preprocessing lookup directory selected")
        self._lookup_dir_label.setWordWrap(True)
        lookup_dir_row.addWidget(self._lookup_dir_label, stretch=1)
        lookup_dir_btn = QPushButton("Lookup directory...")
        lookup_dir_btn.clicked.connect(self._choose_preprocess_lookup_dir)
        lookup_dir_row.addWidget(lookup_dir_btn)
        files_layout.addLayout(lookup_dir_row)
        lookup_resolution_row = QHBoxLayout()
        lookup_resolution_row.addWidget(QLabel("Lookup resolution:"))
        self._lookup_resolution_spin = QSpinBox()
        self._lookup_resolution_spin.setRange(0, 100)
        self._lookup_resolution_spin.setSpecialValueText("From NRRD header")
        self._lookup_resolution_spin.setSuffix(" um")
        lookup_resolution_row.addWidget(self._lookup_resolution_spin)
        files_layout.addLayout(lookup_resolution_row)
        layout.addWidget(files_group)

        cache_group = QGroupBox("Flatmap Region Cache")
        cache_layout = QVBoxLayout(cache_group)
        cache_dir_row = QHBoxLayout()
        self._cache_dir_label = QLabel("No cache directory selected")
        self._cache_dir_label.setWordWrap(True)
        cache_dir_row.addWidget(self._cache_dir_label, stretch=1)
        cache_dir_btn = QPushButton("Choose Cache Directory...")
        cache_dir_btn.clicked.connect(self._choose_cache_directory)
        cache_dir_row.addWidget(cache_dir_btn)
        cache_layout.addLayout(cache_dir_row)

        cache_profile_row = QHBoxLayout()
        cache_profile_row.addWidget(QLabel("Profile:"))
        self._cache_profile_combo = QComboBox()
        self._cache_profile_combo.currentIndexChanged.connect(
            self._on_cache_profile_changed
        )
        cache_profile_row.addWidget(self._cache_profile_combo, stretch=1)
        cache_layout.addLayout(cache_profile_row)

        cache_grid_row = QHBoxLayout()
        cache_grid_row.addWidget(QLabel("New profile XY bins:"))
        self._cache_build_xy_bins_spin = QSpinBox()
        self._cache_build_xy_bins_spin.setRange(1, 4096)
        self._cache_build_xy_bins_spin.setSingleStep(16)
        self._cache_build_xy_bins_spin.setValue(DEFAULT_FLATMAP_XY_BINS)
        cache_grid_row.addWidget(self._cache_build_xy_bins_spin)
        cache_grid_row.addWidget(QLabel("Depth bin:"))
        self._cache_build_depth_bin_spin = QDoubleSpinBox()
        self._cache_build_depth_bin_spin.setRange(0.001, 1000.0)
        self._cache_build_depth_bin_spin.setDecimals(3)
        self._cache_build_depth_bin_spin.setSingleStep(5.0)
        self._cache_build_depth_bin_spin.setSuffix(" um")
        self._cache_build_depth_bin_spin.setValue(float(DEFAULT_FLATMAP_DEPTH_BIN_UM))
        cache_grid_row.addWidget(self._cache_build_depth_bin_spin)
        cache_layout.addLayout(cache_grid_row)

        cache_build_row = QHBoxLayout()
        self._build_cache_btn = QPushButton("Build Cache Profile...")
        self._build_cache_btn.clicked.connect(self._build_cache_profile)
        cache_build_row.addWidget(self._build_cache_btn)
        self._cancel_cache_btn = QPushButton("Cancel")
        self._cancel_cache_btn.setEnabled(False)
        self._cancel_cache_btn.clicked.connect(self._cancel_cache_build)
        cache_build_row.addWidget(self._cancel_cache_btn)
        self._cache_status_label = QLabel("No cache profile active.")
        self._cache_status_label.setWordWrap(True)
        cache_build_row.addWidget(self._cache_status_label, stretch=1)
        cache_layout.addLayout(cache_build_row)
        layout.addWidget(cache_group)

        options_group = QGroupBox("Projection Options")
        options_layout = QVBoxLayout(options_group)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Input:"))
        self._source_combo = QComboBox()
        self._source_combo.addItem(
            "Selected table rows, otherwise all", _SOURCE_SELECTED
        )
        self._source_combo.addItem("All table rows", _SOURCE_ALL)
        source_row.addWidget(self._source_combo)
        options_layout.addLayout(source_row)

        coordinate_row = QHBoxLayout()
        coordinate_row.addWidget(QLabel("Coordinates:"))
        self._coordinate_mode_combo = QComboBox()
        self._coordinate_mode_combo.addItem(
            "CCF microns from NRRD header",
            COORDINATE_MODE_MICRONS,
        )
        self._coordinate_mode_combo.addItem(
            "10 um voxel indices", COORDINATE_MODE_VOXELS
        )
        coordinate_row.addWidget(self._coordinate_mode_combo)
        options_layout.addLayout(coordinate_row)

        render_row = QHBoxLayout()
        render_row.addWidget(QLabel("Render:"))
        self._render_mode_combo = QComboBox()
        self._render_mode_combo.addItem("3D Heatmap", _RENDER_HEATMAP)
        self._render_mode_combo.addItem("3D Points", _RENDER_POINTS)
        render_row.addWidget(self._render_mode_combo)
        render_row.addWidget(QLabel("Heatmap colors:"))
        self._heatmap_color_mode_combo = QComboBox()
        self._heatmap_color_mode_combo.addItem("Single color", _HEATMAP_COLOR_SINGLE)
        self._heatmap_color_mode_combo.addItem(
            "Individual neurons",
            _HEATMAP_COLOR_INDIVIDUAL,
        )
        self._heatmap_color_mode_combo.addItem("Cluster", _HEATMAP_COLOR_CLUSTER)
        render_row.addWidget(self._heatmap_color_mode_combo)
        options_layout.addLayout(render_row)

        xy_bins_row = QHBoxLayout()
        xy_bins_row.addWidget(QLabel("XY bins:"))
        self._xy_bins_spin = QSpinBox()
        self._xy_bins_spin.setRange(1, 4096)
        self._xy_bins_spin.setSingleStep(16)
        self._xy_bins_spin.setValue(DEFAULT_FLATMAP_XY_BINS)
        xy_bins_row.addWidget(self._xy_bins_spin)
        options_layout.addLayout(xy_bins_row)

        depth_bin_row = QHBoxLayout()
        depth_bin_row.addWidget(QLabel("Depth bin:"))
        self._depth_bin_spin = QDoubleSpinBox()
        self._depth_bin_spin.setRange(0.001, 1000.0)
        self._depth_bin_spin.setDecimals(3)
        self._depth_bin_spin.setSingleStep(5.0)
        self._depth_bin_spin.setSuffix(" um")
        self._depth_bin_spin.setValue(float(DEFAULT_FLATMAP_DEPTH_BIN_UM))
        depth_bin_row.addWidget(self._depth_bin_spin)
        options_layout.addLayout(depth_bin_row)

        self._negative_one_sentinel_cb = QCheckBox("Treat flatmap (-1, -1) as invalid")
        self._negative_one_sentinel_cb.setChecked(True)
        options_layout.addWidget(self._negative_one_sentinel_cb)
        self._zero_sentinel_cb = QCheckBox("Treat flatmap (0, 0) as invalid")
        options_layout.addWidget(self._zero_sentinel_cb)
        self._exclude_depth_minus_one_cb = QCheckBox("Exclude depth -1 nodes")
        self._exclude_depth_minus_one_cb.setChecked(True)
        options_layout.addWidget(self._exclude_depth_minus_one_cb)
        layout.addWidget(options_group)

        actions_row = QHBoxLayout()
        self._project_btn = QPushButton("Project to Flatmap")
        self._project_btn.clicked.connect(self._project)
        actions_row.addWidget(self._project_btn)
        self._export_btn = QPushButton("Export CSV...")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_csv)
        actions_row.addWidget(self._export_btn)
        self._augment_parquet_btn = QPushButton("Prepare Whole Parquet...")
        self._augment_parquet_btn.clicked.connect(self._augment_parquet)
        actions_row.addWidget(self._augment_parquet_btn)
        self._cancel_augment_btn = QPushButton("Cancel Preparation")
        self._cancel_augment_btn.setEnabled(False)
        self._cancel_augment_btn.clicked.connect(self._cancel_parquet_preparation)
        actions_row.addWidget(self._cancel_augment_btn)
        layout.addLayout(actions_row)

        self._projection_progress_bar = QProgressBar()
        self._projection_progress_bar.setRange(0, 1)
        self._projection_progress_bar.setValue(0)
        self._projection_progress_bar.setVisible(False)
        layout.addWidget(self._projection_progress_bar)

        labels_group = QGroupBox("Cached Regions")
        labels_layout = QVBoxLayout(labels_group)
        atlas_row = QHBoxLayout()
        atlas_row.addWidget(QLabel("Atlas:"))
        self._region_label_atlas_combo = QComboBox()
        for atlas_name in _REGION_LABEL_ATLAS_OPTIONS:
            self._region_label_atlas_combo.addItem(atlas_name, atlas_name)
        self._region_label_atlas_combo.setCurrentText(_REGION_LABEL_ATLAS_DEFAULT)
        atlas_row.addWidget(self._region_label_atlas_combo)
        labels_layout.addLayout(atlas_row)
        labels_actions_row = QHBoxLayout()
        self._region_labels_btn = QPushButton("Show Region Labels")
        self._region_labels_btn.clicked.connect(self._create_region_labels)
        labels_actions_row.addWidget(self._region_labels_btn)
        self._clear_region_labels_btn = QPushButton("Clear Region Labels")
        self._clear_region_labels_btn.clicked.connect(self._clear_region_labels)
        labels_actions_row.addWidget(self._clear_region_labels_btn)
        labels_layout.addLayout(labels_actions_row)

        geometry_actions_row = QHBoxLayout()
        self._region_surfaces_btn = QPushButton("Show Region Surfaces")
        self._region_surfaces_btn.clicked.connect(self._create_region_surfaces)
        geometry_actions_row.addWidget(self._region_surfaces_btn)
        self._region_outlines_btn = QPushButton("Show Region Outlines")
        self._region_outlines_btn.clicked.connect(self._create_region_outlines)
        geometry_actions_row.addWidget(self._region_outlines_btn)
        self._clear_region_geometry_btn = QPushButton("Clear Geometry")
        self._clear_region_geometry_btn.clicked.connect(self._clear_region_geometry)
        geometry_actions_row.addWidget(self._clear_region_geometry_btn)
        labels_layout.addLayout(geometry_actions_row)
        self._region_labels_status_label = QLabel("No flatmap region labels created.")
        self._region_labels_status_label.setWordWrap(True)
        labels_layout.addWidget(self._region_labels_status_label)
        layout.addWidget(labels_group)

        summary_group = QGroupBox("Projection Summary")
        summary_layout = QVBoxLayout(summary_group)
        self._summary_label = QLabel("No projection run yet.")
        self._summary_label.setWordWrap(True)
        summary_layout.addWidget(self._summary_label)
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        summary_layout.addWidget(self._status_label)
        layout.addWidget(summary_group)

        warning = QLabel(
            "Flatmap coordinates are for visualization and indexing only. "
            "Keep neurite length, branch geometry, and 3D distance calculations "
            "in the original CCF coordinate space."
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)
        layout.addStretch()

        self._update_expected_filename_label()
        self._update_cached_region_controls()

    def set_flatmap_path(self, path: str | Path | None) -> None:
        """Set the flatmap path, primarily for tests and scripted use."""
        self._flatmap_path = Path(path) if path else None
        text = str(self._flatmap_path) if self._flatmap_path else "No flatmap selected"
        self._flatmap_path_label.setText(text)
        self._notify_flatmap_correlation_source_changed()

    def set_depth_path(self, path: str | Path | None) -> None:
        """Set the depth path, primarily for tests and scripted use."""
        self._depth_path = Path(path) if path else None
        text = str(self._depth_path) if self._depth_path else "No depth selected"
        self._depth_path_label.setText(text)
        self._notify_flatmap_correlation_source_changed()

    def set_flatmap_correlation_source_changed_callback(self, callback) -> None:
        """Set a callback invoked when the latest flatmap clustering source changes."""
        self._flatmap_correlation_source_changed_callback = callback

    def _notify_flatmap_correlation_source_changed(self) -> None:
        callback = getattr(
            self,
            "_flatmap_correlation_source_changed_callback",
            None,
        )
        if callable(callback):
            callback()

    def _current_projection_source(self) -> str:
        """Return the explicit precomputed/recompute selection.

        Widgets constructed directly by legacy tests do not have the selector;
        those retain the historical auto-detection behavior.
        """
        combo = getattr(self, "_projection_source_combo", None)
        current_data = getattr(combo, "currentData", None)
        if not callable(current_data):
            return "legacy_auto"
        value = current_data()
        if value == _PROJECTION_SOURCE_RECOMPUTE:
            return _PROJECTION_SOURCE_RECOMPUTE
        return _PROJECTION_SOURCE_PRECOMPUTED

    def _on_projection_source_changed(self) -> None:
        source = self._current_projection_source()
        self._invalidate_flatmap_grid_layers()
        self._update_style_choices_for_source(source)
        if source == _PROJECTION_SOURCE_RECOMPUTE:
            message = "NRRD recomputation is selected explicitly."
            self._set_cache_grid_locked(False)
        else:
            message = "Viewing will use precomputed Parquet/cache data only."
            if getattr(self, "_active_cache_profile", None) is not None:
                self._on_cache_profile_changed()
        self._update_cached_region_controls()
        status = getattr(self, "_status_label", None)
        if status is not None:
            status.setText(message)
        self._notify_flatmap_correlation_source_changed()

    def _update_style_choices_for_source(self, source: str) -> None:
        combo = getattr(self, "_style_combo", None)
        count = getattr(combo, "count", None)
        item_data = getattr(combo, "itemData", None)
        model_getter = getattr(combo, "model", None)
        if not all(callable(value) for value in (count, item_data, model_getter)):
            return
        bilateral = {"both_shaped", "both_square"}
        model = model_getter()
        for index in range(count()):
            item = model.item(index)
            if item is not None:
                item.setEnabled(
                    source == _PROJECTION_SOURCE_RECOMPUTE
                    or str(item_data(index)) in bilateral
                )
        if (
            source == _PROJECTION_SOURCE_PRECOMPUTED
            and self._current_style_key() not in bilateral
        ):
            for index in range(count()):
                if str(item_data(index)) == "both_shaped":
                    combo.setCurrentIndex(index)
                    break

    def _on_flatmap_style_changed(self) -> None:
        self._invalidate_flatmap_grid_layers()
        self._update_expected_filename_label()
        self._refresh_cache_profiles()
        self._notify_flatmap_correlation_source_changed()

    def _invalidate_flatmap_grid_layers(self) -> None:
        """Remove render state that belongs to a previous style/cache grid."""
        self._remove_projection_layer(create=False)
        self._clear_named_region_layers(_REGION_LABELS_LAYER_NAME)
        self._clear_region_surface_layers()
        self._clear_region_outline_layers()
        self._region_labels_layer = None
        self._last_projected_nodes = None
        self._last_summary = None
        self._last_render_summary = None
        self._last_render_mode = None
        self._last_flatmap_style = None
        self._last_coordinate_mode = None
        self._last_volume_shape = None
        self._last_lookup_stats = None
        self._last_input_file_ids = ()
        self._last_flatmap_path = None
        self._last_depth_path = None
        self._last_projection_source = None
        self._last_cache_dir = None
        self._last_cache_profile_id = None
        export_button = getattr(self, "_export_btn", None)
        if export_button is not None:
            export_button.setEnabled(False)

    def _choose_preprocess_lookup_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Choose Bilateral Flatmap Lookup Directory",
        )
        if not path:
            return
        self._preprocess_lookup_dir = Path(path)
        self._lookup_dir_label.setText(str(self._preprocess_lookup_dir))

    def set_cache_directory(
        self,
        path: str | Path | None,
        *,
        profile_id: str | None = None,
    ) -> None:
        """Open a region cache directory and refresh compatible profiles."""
        previous_directory = getattr(self, "_region_cache_dir", None)
        previous_cache = getattr(self, "_region_cache", None)
        next_directory = Path(path) if path else None
        if next_directory is None:
            self._deactivate_cache_profile()
            self._region_cache_dir = None
            self._region_cache = None
            self._pending_cache_profile_id = None
            self._close_region_cache(previous_cache)
            self._cache_dir_label.setText("No cache directory selected")
            self._refresh_cache_profiles()
            return

        from ..flatmap_region_cache import open_region_cache

        # Open and validate first so a bad new path cannot leave an old active
        # profile attached to the wrong directory.
        next_cache = open_region_cache(next_directory)
        if previous_directory != next_directory:
            self._invalidate_flatmap_grid_layers()
            self._active_cache_profile = None
        self._region_cache_dir = next_directory
        self._region_cache = next_cache
        self._pending_cache_profile_id = str(profile_id) if profile_id else None
        self._set_cache_grid_locked(False)
        if previous_cache is not next_cache:
            self._close_region_cache(previous_cache)
        self._cache_dir_label.setText(str(next_directory))
        self._refresh_cache_profiles()

    @staticmethod
    def _close_region_cache(cache) -> None:
        """Release a superseded cache without assuming a concrete cache type."""
        close = getattr(cache, "close", None)
        if callable(close):
            close()

    def active_cache_reference(self) -> dict[str, str] | None:
        """Return the external cache reference stored in project bundles."""
        cache_dir = getattr(self, "_region_cache_dir", None)
        profile = getattr(self, "_active_cache_profile", None)
        profile_id = self._cache_profile_id(profile)
        if cache_dir is None or not profile_id:
            return None
        return {"path": str(cache_dir), "profile_id": profile_id}

    def restore_cache_reference(self, reference: object) -> None:
        """Restore a project bundle's external cache path/profile selection."""
        if not isinstance(reference, dict):
            return
        path = reference.get("path")
        profile_id = str(reference.get("profile_id") or "")
        if not path:
            return
        try:
            self.set_cache_directory(str(path), profile_id=profile_id or None)
        except Exception:
            # A project reference supersedes session state: never leave an old
            # cache active when the project's external cache is unavailable.
            self.set_cache_directory(None)
            raise

    def refresh_cache_profiles(self) -> None:
        """Re-evaluate cache compatibility after atlas or Parquet changes."""
        self._refresh_cache_profiles()

    def invalidate_loaded_parquet_projection(self) -> None:
        """Clear flatmap state before associating the tab with a new Parquet."""
        self._invalidate_flatmap_grid_layers()

    def _deactivate_cache_profile(self) -> None:
        if getattr(self, "_active_cache_profile", None) is not None:
            self._invalidate_flatmap_grid_layers()
        self._active_cache_profile = None
        self._set_cache_grid_locked(False)
        self._update_cached_region_controls()
        self._notify_flatmap_correlation_source_changed()

    def _choose_cache_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Choose Flatmap Region Cache Directory",
        )
        if not path:
            return
        try:
            self.set_cache_directory(path, profile_id=None)
        except Exception as exc:
            logger.exception("Failed to open flatmap region cache")
            message = f"Flatmap region cache is incompatible or corrupt: {exc}"
            self._cache_status_label.setText(message)
            show_warning(message)

    @staticmethod
    def _cache_profile_id(profile) -> str:
        return str(getattr(profile, "profile_id", "") or "")

    @staticmethod
    def _atlas_family_name(name: object) -> str:
        return re.sub(r"_(?:10|25|50)um$", "", str(name or ""))

    def _refresh_cache_profiles(self) -> None:
        """Show only profiles compatible with the v3 Parquet/style/catalog."""
        combo = getattr(self, "_cache_profile_combo", None)
        if combo is None:
            return
        combo.blockSignals(True)
        combo.clear()
        cache = getattr(self, "_region_cache", None)
        if cache is None:
            combo.blockSignals(False)
            self._deactivate_cache_profile()
            return

        try:
            info = read_flatmap_parquet_transform_info(
                self._current_source_parquet_path()
            )
        except Exception as exc:
            combo.blockSignals(False)
            self._deactivate_cache_profile()
            self._cache_status_label.setText(
                f"Load a version-3 neuron Parquet to select a cache profile: {exc}"
            )
            return
        if info.format_version < 3 or not info.lookup_set_id:
            combo.blockSignals(False)
            self._deactivate_cache_profile()
            self._cache_status_label.setText(
                "Legacy Parquets can render neurons, but exact cache overlays "
                "require version-3 preprocessing."
            )
            return

        atlas = self._atlas_provider()
        if atlas is None:
            combo.blockSignals(False)
            self._deactivate_cache_profile()
            self._cache_status_label.setText(
                "Load a matching BrainGlobe atlas structure catalog to use the cache."
            )
            return

        style = self._current_style_key()
        current_atlas_name = str(getattr(atlas, "atlas_name", "") or "")
        from ..flatmap_region_cache import structure_catalog_id

        structures = getattr(atlas, "structures", None)
        try:
            current_catalog_id = structure_catalog_id(structures)
        except (AttributeError, TypeError, ValueError):
            combo.blockSignals(False)
            self._deactivate_cache_profile()
            self._cache_status_label.setText(
                "The loaded atlas does not expose a usable structure catalog."
            )
            return
        current_atlas_version = (
            getattr(atlas, "local_version", None)
            or getattr(atlas, "atlas_version", None)
            or getattr(atlas, "version", None)
        )
        current_atlas_version = self._atlas_version_text(current_atlas_version)
        shared_depth = (info.metadata or {}).get("shared_depth_definition")
        if not isinstance(shared_depth, dict):
            combo.blockSignals(False)
            self._deactivate_cache_profile()
            self._cache_status_label.setText(
                "Version-3 Parquet metadata is missing its shared depth definition."
            )
            return
        try:
            parquet_mirror_axis = int(shared_depth["mirror_coord_axis"])
        except (KeyError, TypeError, ValueError):
            combo.blockSignals(False)
            self._deactivate_cache_profile()
            self._cache_status_label.setText(
                "Version-3 Parquet metadata has no valid depth mirror axis."
            )
            return
        mismatch_messages: list[str] = []
        for profile in cache.profiles.values():
            mismatches = list(
                profile.compatibility_mismatches(
                    lookup_set_id=info.lookup_set_id,
                    atlas_name=current_atlas_name,
                    atlas_version=current_atlas_version,
                    structure_catalog_id=current_catalog_id,
                    style=style,
                    mirror_depth_fallback=True,
                    mirror_coord_axis=parquet_mirror_axis,
                )
            )
            cached_atlas_name = str(profile.atlas.get("name", "") or "")
            if mismatches:
                mismatch_messages.append(
                    f"{profile.profile_id[:12]}: " + "; ".join(mismatches)
                )
                continue
            style_cache = profile.style(style)
            grid = style_cache.grid_spec
            label = (
                f"{profile.profile_id[:12]} — {cached_atlas_name}, "
                f"{grid.get('xy_bins')} XY / {grid.get('depth_bin_um')} um"
            )
            combo.addItem(label, profile)

        combo.blockSignals(False)
        if combo.count():
            target_index = 0
            pending_profile_id = str(
                getattr(self, "_pending_cache_profile_id", "") or ""
            )
            if pending_profile_id:
                for index in range(combo.count()):
                    if (
                        self._cache_profile_id(combo.itemData(index))
                        == pending_profile_id
                    ):
                        target_index = index
                        break
            combo.setCurrentIndex(target_index)
            self._on_cache_profile_changed()
        else:
            self._deactivate_cache_profile()
            detail = mismatch_messages[0] if mismatch_messages else "no profiles"
            self._cache_status_label.setText(
                f"No compatible cache profile: {detail}. "
                "Recomputation will not start automatically."
            )

    def _on_cache_profile_changed(self) -> None:
        combo = getattr(self, "_cache_profile_combo", None)
        profile = combo.currentData() if combo is not None else None
        previous_profile_id = self._cache_profile_id(
            getattr(self, "_active_cache_profile", None)
        )
        next_profile_id = self._cache_profile_id(profile)
        if previous_profile_id and previous_profile_id != next_profile_id:
            self._invalidate_flatmap_grid_layers()
        self._active_cache_profile = profile
        if profile is None:
            self._set_cache_grid_locked(False)
            self._update_cached_region_controls()
            self._notify_flatmap_correlation_source_changed()
            return
        self._pending_cache_profile_id = self._cache_profile_id(profile)
        style_cache = profile.style(self._current_style_key())
        grid = style_cache.grid_spec
        self._xy_bins_spin.setValue(int(grid["xy_bins"]))
        self._depth_bin_spin.setValue(float(grid["depth_bin_um"]))
        self._exclude_depth_minus_one_cb.setChecked(True)
        locked = self._current_projection_source() != _PROJECTION_SOURCE_RECOMPUTE
        self._set_cache_grid_locked(locked)
        suffix = "grid controls are locked" if locked else "NRRD fallback is active"
        self._cache_status_label.setText(
            f"Active cache profile {profile.profile_id}; {suffix}."
        )
        self._update_cached_region_controls()
        self._notify_flatmap_correlation_source_changed()

    def _update_cached_region_controls(self) -> None:
        cache_enabled = (
            self._current_projection_source() == _PROJECTION_SOURCE_PRECOMPUTED
            and getattr(self, "_active_cache_profile", None) is not None
        )
        for name in (
            "_region_surfaces_btn",
            "_region_outlines_btn",
            "_clear_region_geometry_btn",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(cache_enabled)
        labels_button = getattr(self, "_region_labels_btn", None)
        if labels_button is not None:
            labels_button.setEnabled(
                cache_enabled
                or self._current_projection_source() != _PROJECTION_SOURCE_PRECOMPUTED
            )
        atlas_combo = getattr(self, "_region_label_atlas_combo", None)
        if atlas_combo is not None:
            atlas_combo.setEnabled(
                self._current_projection_source() != _PROJECTION_SOURCE_PRECOMPUTED
            )

    def _set_cache_grid_locked(self, locked: bool) -> None:
        for name in (
            "_xy_bins_spin",
            "_depth_bin_spin",
            "_exclude_depth_minus_one_cb",
            "_negative_one_sentinel_cb",
            "_zero_sentinel_cb",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(not bool(locked))

    @staticmethod
    def _cache_profile_bounds(
        profile,
        style: str,
    ) -> dict[str, tuple[float, float]] | None:
        if profile is None:
            return None
        try:
            grid = profile.style(style).grid_spec
            return {
                "x_bounds": tuple(float(value) for value in grid["x_bounds"]),
                "y_bounds": tuple(float(value) for value in grid["y_bounds"]),
                "depth_range_um": tuple(
                    float(value) for value in grid["depth_bounds_um"]
                ),
            }
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _atlas_annotation_tiff_path(atlas) -> Path:
        """Resolve BrainGlobe's on-disk annotation without reading the volume."""
        direct = getattr(atlas, "annotation_path", None)
        if direct and Path(direct).is_file():
            return Path(direct)
        for attribute in ("root_dir", "atlas_dir", "brainglobe_dir"):
            root = getattr(atlas, attribute, None)
            if root:
                candidate = Path(root) / "annotation.tiff"
                if candidate.is_file():
                    return candidate
        from ..workers import cached_brainglobe_atlas_dir

        atlas_name = str(getattr(atlas, "atlas_name", "") or "")
        atlas_dir = cached_brainglobe_atlas_dir(atlas_name)
        if atlas_dir is not None:
            candidate = atlas_dir / "annotation.tiff"
            if candidate.is_file():
                return candidate
        raise RuntimeError(
            "The matching BrainGlobe annotation.tiff could not be located. "
            "Cache generation requires the exact on-disk atlas grid."
        )

    @staticmethod
    def _atlas_version_text(value: object | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, (tuple, list)):
            text = ".".join(str(part).strip() for part in value)
        else:
            text = str(value).strip()
        return text or None

    @staticmethod
    def _atlas_version(atlas, annotation_path: Path) -> str | None:
        raw = (
            getattr(atlas, "local_version", None)
            or getattr(atlas, "atlas_version", None)
            or getattr(atlas, "version", None)
        )
        normalized = FlatmapProjectionWidget._atlas_version_text(raw)
        if normalized is not None:
            return normalized
        match = re.search(r"_v([^/]+)$", annotation_path.parent.name)
        return match.group(1) if match else None

    def _build_cache_profile(self) -> None:
        """Validate inputs and launch one atomic cache-profile build."""
        lookup_dir = getattr(self, "_preprocess_lookup_dir", None)
        if lookup_dir is None:
            show_warning("Choose the bilateral lookup directory first.")
            return
        atlas = self._atlas_provider()
        if atlas is None:
            show_warning("Load the exact BrainGlobe atlas before building a cache.")
            return
        cache_dir = getattr(self, "_region_cache_dir", None)
        if cache_dir is None:
            selected = QFileDialog.getExistingDirectory(
                self,
                "Choose or Create Flatmap Region Cache Directory",
            )
            if not selected:
                return
            cache_dir = Path(selected)
            self._region_cache_dir = cache_dir
            self._cache_dir_label.setText(str(cache_dir))
        try:
            annotation_path = self._atlas_annotation_tiff_path(atlas)
        except Exception as exc:
            show_warning(str(exc))
            return

        from qtpy.QtCore import QThread

        from ..workers import RegionCacheBuildWorker

        resolution_control = getattr(self, "_lookup_resolution_spin", None)
        raw_lookup_resolution = (
            int(resolution_control.value()) if resolution_control is not None else 0
        )
        atlas_resolution = tuple(
            float(value) for value in np.asarray(atlas.resolution).reshape(-1)
        )
        worker = RegionCacheBuildWorker(
            cache_dir=cache_dir,
            lookup_dir=lookup_dir,
            annotation_path=annotation_path,
            atlas_name=str(getattr(atlas, "atlas_name", "") or ""),
            atlas_version=self._atlas_version(atlas, annotation_path),
            atlas_resolution_um=atlas_resolution,
            atlas_structures=getattr(atlas, "structures", None),
            xy_bins=self._current_cache_build_xy_bins(),
            depth_bin_um=self._current_cache_build_depth_bin_um(),
            lookup_resolution_um=(
                float(raw_lookup_resolution) if raw_lookup_resolution > 0 else None
            ),
        )
        thread = QThread()
        self._cache_build_thread = thread
        self._cache_build_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_cache_build_progress)
        worker.finished.connect(self._on_cache_build_finished)
        worker.error.connect(self._on_cache_build_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._cleanup_cache_build(thread, worker))
        self._build_cache_btn.setEnabled(False)
        self._cancel_cache_btn.setEnabled(True)
        self._cache_status_label.setText("Starting region-cache build...")
        thread.start()

    def _on_cache_build_progress(
        self,
        message: str,
        current: int,
        total: int,
    ) -> None:
        suffix = f" ({current}/{total})" if total > 0 else ""
        self._cache_status_label.setText(f"{message}{suffix}")

    def _cancel_cache_build(self) -> None:
        worker = getattr(self, "_cache_build_worker", None)
        cancel = getattr(worker, "cancel", None)
        if callable(cancel):
            cancel()
            self._cache_status_label.setText("Cancelling cache build...")
            self._cancel_cache_btn.setEnabled(False)

    def _on_cache_build_finished(self, profile) -> None:
        profile_id = self._cache_profile_id(profile)
        try:
            self.set_cache_directory(self._region_cache_dir)
            combo = self._cache_profile_combo
            opened = False
            for index in range(combo.count()):
                if self._cache_profile_id(combo.itemData(index)) == profile_id:
                    combo.setCurrentIndex(index)
                    self._on_cache_profile_changed()
                    opened = True
                    break
            if opened:
                self._cache_status_label.setText(
                    f"Built and opened cache profile {profile_id}."
                )
                show_info(f"Built flatmap region-cache profile {profile_id}")
                return
            current_text = getattr(self._cache_status_label, "text", None)
            detail = current_text() if callable(current_text) else ""
            message = (
                f"Built cache profile {profile_id}, but it is not compatible with "
                f"the loaded Parquet/atlas/style. {detail}"
            ).strip()
            self._cache_status_label.setText(message)
            show_warning(message)
        finally:
            close = getattr(profile, "close", None)
            if callable(close):
                close()

    def _on_cache_build_error(self, message: str) -> None:
        self._cache_status_label.setText(f"Region-cache build failed: {message}")
        show_warning(f"Region-cache build failed: {message}")

    def _cleanup_cache_build(self, thread, worker) -> None:
        if getattr(self, "_cache_build_thread", None) is thread:
            self._cache_build_thread = None
        if getattr(self, "_cache_build_worker", None) is worker:
            self._cache_build_worker = None
        self._build_cache_btn.setEnabled(True)
        self._cancel_cache_btn.setEnabled(False)

    def _current_style_key(self) -> str:
        key = self._style_combo.currentData()
        return str(key or "both_shaped")

    def _current_style_filename(self) -> str:
        return FLATMAP_STYLE_FILENAMES.get(
            self._current_style_key(),
            FLATMAP_STYLE_FILENAMES["both_shaped"],
        )

    def _current_coordinate_mode(self) -> str:
        mode = self._coordinate_mode_combo.currentData()
        return str(mode or COORDINATE_MODE_MICRONS)

    def _current_render_mode(self) -> str:
        mode = self._render_mode_combo.currentData()
        return str(mode or _RENDER_HEATMAP)

    def _current_heatmap_color_mode(self) -> str:
        combo = getattr(self, "_heatmap_color_mode_combo", None)
        current_data = getattr(combo, "currentData", None)
        mode = current_data() if callable(current_data) else None
        if mode in {
            _HEATMAP_COLOR_SINGLE,
            _HEATMAP_COLOR_INDIVIDUAL,
            _HEATMAP_COLOR_CLUSTER,
        }:
            return str(mode)
        return _HEATMAP_COLOR_SINGLE

    def _current_xy_bins(self) -> int:
        profile = getattr(self, "_active_cache_profile", None)
        if (
            profile is not None
            and self._current_projection_source() == _PROJECTION_SOURCE_PRECOMPUTED
        ):
            return int(profile.style(self._current_style_key()).grid_spec["xy_bins"])
        return int(self._xy_bins_spin.value())

    def _current_depth_bin_um(self) -> float:
        profile = getattr(self, "_active_cache_profile", None)
        if (
            profile is not None
            and self._current_projection_source() == _PROJECTION_SOURCE_PRECOMPUTED
        ):
            return float(
                profile.style(self._current_style_key()).grid_spec["depth_bin_um"]
            )
        return float(self._depth_bin_spin.value())

    def _current_cache_build_xy_bins(self) -> int:
        control = getattr(self, "_cache_build_xy_bins_spin", None)
        if control is None:
            return self._current_xy_bins()
        return int(control.value())

    def _current_cache_build_depth_bin_um(self) -> float:
        control = getattr(self, "_cache_build_depth_bin_spin", None)
        if control is None:
            return self._current_depth_bin_um()
        return float(control.value())

    def _current_source_mode(self) -> str:
        mode = self._source_combo.currentData()
        return str(mode or _SOURCE_SELECTED)

    def _current_region_label_atlas_name(self) -> str:
        combo = getattr(self, "_region_label_atlas_combo", None)
        current_text = getattr(combo, "currentText", None)
        if callable(current_text):
            atlas_name = str(current_text() or "").strip()
            if atlas_name:
                return atlas_name
        return _REGION_LABEL_ATLAS_DEFAULT

    def _update_expected_filename_label(self) -> None:
        filename = self._current_style_filename()
        self._expected_filename_label.setText(
            f"Expected Zenodo v4.1 flatmap file for this style: {filename}"
        )

    def _choose_flatmap_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Flatmap NRRD",
            "",
            "NRRD Files (*.nrrd);;All Files (*)",
        )
        if path:
            self.set_flatmap_path(path)

    def _choose_depth_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Depth NRRD",
            "",
            "NRRD Files (*.nrrd);;All Files (*)",
        )
        if path:
            self.set_depth_path(path)

    @staticmethod
    def _deduplicate_file_ids(file_ids: list[object]) -> list[object]:
        out: list[object] = []
        seen: set[object] = set()
        for file_id in file_ids:
            if file_id in seen:
                continue
            seen.add(file_id)
            out.append(file_id)
        return out

    def _file_ids_for_source(self, source_mode: str | None = None) -> list[object]:
        mode = source_mode or self._current_source_mode()
        table_ids = self._deduplicate_file_ids(
            list(self._table_file_ids_provider() or [])
        )
        if mode == _SOURCE_ALL:
            return table_ids

        selected_ids = self._deduplicate_file_ids(
            list(self._selected_file_ids_provider() or [])
        )
        return selected_ids if selected_ids else table_ids

    def _query_nodes(self, file_ids: list[object]) -> pd.DataFrame:
        db = self._database_provider()
        if db is None:
            raise RuntimeError("Load a neuron Parquet before projecting to flatmap.")
        if not file_ids:
            raise RuntimeError("No neurons are available to project.")

        getter = getattr(db, "get_neurons_for_rendering", None)
        if not callable(getter):
            raise RuntimeError(
                "Loaded neuron database does not support rendering queries."
            )
        nodes = getter(file_ids)
        if nodes is None or nodes.empty:
            raise RuntimeError("No neuron rows matched the requested file IDs.")
        return nodes

    def _lookup_files_ready(self) -> bool:
        return self._flatmap_path is not None and self._depth_path is not None

    def _projection_request_ready(self) -> None:
        if self._flatmap_path is None:
            raise RuntimeError("Choose a flatmap NRRD file before this action.")
        if self._depth_path is None:
            raise RuntimeError("Choose depth.nrrd before this action.")

    @staticmethod
    def _has_parquet_flatmap_depth_columns(nodes: pd.DataFrame) -> bool:
        names = set(nodes.columns)
        return bool(
            {"x_flat", "y_flat", "depth_um"}.issubset(names)
            or {
                "x_flat_shaped",
                "y_flat_shaped",
                "x_flat_square",
                "y_flat_square",
                "depth_um",
            }.issubset(names)
        )

    def _project(self) -> None:
        """Run projection from the current UI state and render the layer."""
        projection_source = self._current_projection_source()
        if projection_source == _PROJECTION_SOURCE_RECOMPUTE:
            self._projection_request_ready()
            use_lookup_files = True
        elif projection_source == _PROJECTION_SOURCE_PRECOMPUTED:
            use_lookup_files = False
        else:
            use_lookup_files = self._lookup_files_ready()
        total_steps = 6 if use_lookup_files else 4
        self._set_projection_controls_enabled(False)
        self._set_projection_progress(
            "Querying neuron rows...",
            0,
            total_steps,
        )
        try:
            file_ids = self._file_ids_for_source()
            nodes = self._query_nodes(file_ids)

            if use_lookup_files:
                result, render_result, lookup_stats = self._project_from_lookup_files(
                    nodes,
                    progress_callback=self._set_projection_progress,
                    progress_total=total_steps,
                )
                flatmap_style = self._current_style_filename()
                coordinate_mode = self._current_coordinate_mode()
                source_note = "lookup NRRDs"
            else:
                if not self._has_parquet_flatmap_depth_columns(nodes):
                    if projection_source == "legacy_auto":
                        raise RuntimeError(
                            "Choose both flatmap and depth NRRD files, or load an "
                            "augmented Parquet with x_flat, y_flat, and depth_um "
                            "columns."
                        )
                    raise RuntimeError(
                        "Precomputed viewing requires a version-3 Parquet with "
                        "bilateral shaped/square flatmap and depth columns. "
                        "Choose Recompute from NRRDs explicitly to use lookup files."
                    )
                if projection_source == _PROJECTION_SOURCE_PRECOMPUTED:
                    self._validate_precomputed_parquet_contract(nodes)
                result, render_result, lookup_stats = (
                    self._project_from_parquet_columns(
                        nodes,
                        progress_callback=self._set_projection_progress,
                        progress_total=total_steps,
                    )
                )
                flatmap_style = (
                    "precomputed_parquet"
                    if projection_source == "legacy_auto"
                    else self._current_style_key()
                )
                coordinate_mode = "parquet_columns"
                source_note = "Parquet flatmap/depth columns"

            self._set_projection_progress(
                "Updating flatmap layer...",
                total_steps - 1,
                total_steps,
            )
            self._apply_projection_result(
                result,
                render_result,
                flatmap_style=flatmap_style,
                coordinate_mode=coordinate_mode,
                lookup_stats=lookup_stats,
                input_file_ids=tuple(str(file_id) for file_id in file_ids),
            )
            self._set_projection_progress("Done", total_steps, total_steps)
            self._status_label.setText(
                f"Rendered {render_result.summary.rendered_nodes:,} of "
                f"{render_result.summary.total_nodes:,} projected node(s) using "
                f"{source_note}."
            )
            show_info("Flatmap projection complete.")
        except Exception as exc:
            logger.exception("Flatmap projection failed")
            self._status_label.setText(f"Flatmap projection failed: {exc}")
            show_warning(f"Flatmap projection failed: {exc}")
        finally:
            self._hide_projection_progress()
            self._set_projection_controls_enabled(True)

    def _validate_precomputed_parquet_contract(self, nodes: pd.DataFrame) -> None:
        """Reject partial/corrupt v3 data before fixed-grid rendering."""
        names = set(nodes.columns)
        v3_markers = {
            column
            for mapping in FLATMAP_V3_STYLE_COLUMN_MAPPING.values()
            for column in mapping.values()
        }
        if not names.intersection(v3_markers):
            # Complete legacy x_flat/y_flat/depth_um files remain usable for
            # neuron-only rendering with their historical subset bounds.
            return
        missing = sorted(set(FLATMAP_V3_AUGMENTED_COLUMNS).difference(names))
        if missing:
            raise RuntimeError(
                "Version-3 Parquet is missing required flatmap column(s): "
                f"{missing}. Regenerate it with Prepare Whole Parquet."
            )
        style = self._current_style_key()
        if style not in FLATMAP_V3_STYLE_COLUMN_MAPPING:
            raise RuntimeError(
                "Version-3 precomputed viewing supports only bilateral shaped "
                "and bilateral square styles. Choose Recompute from NRRDs for "
                "a unilateral style."
            )
        info = read_flatmap_parquet_transform_info(self._current_source_parquet_path())
        if info.format_version < 3 or not info.lookup_set_id:
            raise RuntimeError(
                "Bilateral flatmap columns require complete version-3 metadata "
                "with a lookup-set ID. Regenerate the Parquet."
            )
        metadata = info.metadata
        bounds = (
            self._bounds_from_projection_metadata(metadata, style)
            if isinstance(metadata, dict)
            else None
        )
        if bounds is None:
            raise RuntimeError(
                f"Version-3 Parquet has no valid canonical bounds for {style}. "
                "Regenerate the Parquet instead of deriving bounds from this query."
            )

    def _project_from_lookup_files(
        self,
        nodes: pd.DataFrame,
        *,
        progress_callback: Callable[[str, int, int], None] | None = None,
        progress_total: int = 6,
    ) -> tuple[FlatmapProjectionResult, FlatmapRenderResult, FlatmapLookupStats]:
        self._emit_projection_progress(
            progress_callback,
            "Loading flatmap lookup files...",
            1,
            progress_total,
        )
        volume_set = load_flatmap_volume_set(self._flatmap_path, self._depth_path)
        self._emit_projection_progress(
            progress_callback,
            "Computing flatmap lookup statistics...",
            2,
            progress_total,
        )
        lookup_stats = self._lookup_stats_for_volume_set(
            volume_set,
            invalid_zero_sentinel=self._zero_sentinel_cb.isChecked(),
            invalid_negative_one_sentinel=(self._negative_one_sentinel_cb.isChecked()),
        )
        self._emit_projection_progress(
            progress_callback,
            "Projecting nodes into flatmap space...",
            3,
            progress_total,
        )
        result = project_and_build_segments(
            nodes,
            volume_set.flatmap,
            volume_set.depth,
            flatmap_style=self._current_style_filename(),
            coordinate_mode=self._current_coordinate_mode(),
            invalid_zero_sentinel=self._zero_sentinel_cb.isChecked(),
            invalid_negative_one_sentinel=(self._negative_one_sentinel_cb.isChecked()),
            resolution_um=DEFAULT_CCF_RESOLUTION_UM,
            space_directions=volume_set.space_directions,
            space_origin=volume_set.space_origin,
            mirror_fallback=True,
        )
        self._emit_projection_progress(
            progress_callback,
            "Building flatmap render data...",
            4,
            progress_total,
        )
        render_result = build_flatmap_render_data(
            result.projected_nodes,
            volume_set.flatmap,
            volume_set.depth,
            xy_bins=self._current_xy_bins(),
            depth_bin_um=self._current_depth_bin_um(),
            include_depth_minus_one=(not self._exclude_depth_minus_one_cb.isChecked()),
            invalid_zero_sentinel=self._zero_sentinel_cb.isChecked(),
            invalid_negative_one_sentinel=(self._negative_one_sentinel_cb.isChecked()),
            lookup_stats=lookup_stats,
        )
        return result, render_result, lookup_stats

    def _project_from_parquet_columns(
        self,
        nodes: pd.DataFrame,
        *,
        progress_callback: Callable[[str, int, int], None] | None = None,
        progress_total: int = 4,
    ) -> tuple[FlatmapProjectionResult, FlatmapRenderResult, None]:
        self._emit_projection_progress(
            progress_callback,
            "Reading precomputed flatmap columns...",
            1,
            progress_total,
        )
        result = self._projection_result_from_parquet_columns(nodes)
        self._emit_projection_progress(
            progress_callback,
            "Building flatmap render data...",
            2,
            progress_total,
        )
        canonical_bounds = self._canonical_render_bounds()
        render_result = build_flatmap_render_data_from_projected_nodes(
            result.projected_nodes,
            xy_bins=self._current_xy_bins(),
            depth_bin_um=self._current_depth_bin_um(),
            include_depth_minus_one=(not self._exclude_depth_minus_one_cb.isChecked()),
            **canonical_bounds,
        )
        return result, render_result, None

    def _canonical_render_bounds(self) -> dict[str, tuple[float, float]]:
        """Return canonical style/depth bounds from the cache or v3 Parquet."""
        profile = getattr(self, "_active_cache_profile", None)
        if profile is not None:
            bounds = self._cache_profile_bounds(profile, self._current_style_key())
            if bounds is not None:
                return bounds

        try:
            info = read_flatmap_parquet_transform_info(
                self._current_source_parquet_path()
            )
        except Exception:
            logger.debug("Could not inspect flatmap Parquet bounds", exc_info=True)
            return {}
        metadata = info.metadata
        if not isinstance(metadata, dict):
            return {}
        return (
            self._bounds_from_projection_metadata(
                metadata,
                self._current_style_key(),
            )
            or {}
        )

    @staticmethod
    def _bounds_from_projection_metadata(
        metadata: dict[str, object],
        style: str,
    ) -> dict[str, tuple[float, float]] | None:
        canonical = metadata.get("canonical_bounds")
        if not isinstance(canonical, dict):
            return None
        style_bounds = canonical.get(style)
        if not isinstance(style_bounds, dict):
            return None
        try:
            x_values = tuple(float(value) for value in style_bounds["x"])
            y_values = tuple(float(value) for value in style_bounds["y"])
            depth_values = tuple(float(value) for value in style_bounds["depth_um"])
        except (KeyError, TypeError, ValueError):
            return None
        if not all(len(values) == 2 for values in (x_values, y_values, depth_values)):
            return None
        return {
            "x_bounds": (x_values[0], x_values[1]),
            "y_bounds": (y_values[0], y_values[1]),
            "depth_range_um": (depth_values[0], depth_values[1]),
        }

    @staticmethod
    def _emit_projection_progress(
        progress_callback: Callable[[str, int, int], None] | None,
        message: str,
        current: int,
        total: int,
    ) -> None:
        if progress_callback is not None:
            progress_callback(message, current, total)

    def _set_projection_controls_enabled(self, enabled: bool) -> None:
        button = getattr(self, "_project_btn", None)
        set_enabled = getattr(button, "setEnabled", None)
        if callable(set_enabled):
            set_enabled(bool(enabled))

    def _set_projection_progress(
        self,
        message: str,
        current: int,
        total: int,
    ) -> None:
        status_label = getattr(self, "_status_label", None)
        if status_label is not None:
            status_label.setText(str(message))

        progress_bar = getattr(self, "_projection_progress_bar", None)
        if progress_bar is not None:
            set_visible = getattr(progress_bar, "setVisible", None)
            if callable(set_visible):
                set_visible(True)
            if int(total) > 0:
                maximum = int(total)
                value = max(0, min(int(current), maximum))
                set_range = getattr(progress_bar, "setRange", None)
                if callable(set_range):
                    set_range(0, maximum)
                set_value = getattr(progress_bar, "setValue", None)
                if callable(set_value):
                    set_value(value)
            else:
                set_range = getattr(progress_bar, "setRange", None)
                if callable(set_range):
                    set_range(0, 0)
        self._flush_projection_progress_updates()

    def _hide_projection_progress(self) -> None:
        progress_bar = getattr(self, "_projection_progress_bar", None)
        if progress_bar is None:
            return
        set_range = getattr(progress_bar, "setRange", None)
        if callable(set_range):
            set_range(0, 1)
        set_value = getattr(progress_bar, "setValue", None)
        if callable(set_value):
            set_value(0)
        set_visible = getattr(progress_bar, "setVisible", None)
        if callable(set_visible):
            set_visible(False)

    @staticmethod
    def _flush_projection_progress_updates() -> None:
        try:
            from qtpy.QtWidgets import QApplication
        except ImportError:
            return

        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def _projection_result_from_parquet_columns(
        self,
        nodes: pd.DataFrame,
    ) -> FlatmapProjectionResult:
        source = self._normalise_precomputed_style_columns(nodes.reset_index(drop=True))
        missing = [
            column
            for column in ("x_flat", "y_flat", "depth_um")
            if column not in source.columns
        ]
        if missing:
            raise RuntimeError(
                f"Loaded Parquet is missing reusable flatmap/depth column(s): {missing}"
            )

        x_flat = pd.to_numeric(source["x_flat"], errors="coerce").to_numpy(dtype=float)
        y_flat = pd.to_numeric(source["y_flat"], errors="coerce").to_numpy(dtype=float)
        depth_um = pd.to_numeric(source["depth_um"], errors="coerce").to_numpy(
            dtype=float
        )
        flatmap_valid = self._bool_column_or_default(
            source,
            "flatmap_valid",
            np.isfinite(x_flat) & np.isfinite(y_flat),
        )
        depth_valid = self._bool_column_or_default(
            source,
            "depth_valid",
            np.isfinite(depth_um) & (depth_um >= 0.0),
        )
        if "flatmap_projection_valid" in source.columns:
            valid = (
                source["flatmap_projection_valid"].fillna(False).astype(bool).to_numpy()
            )
        elif "valid" in source.columns:
            valid = source["valid"].fillna(False).astype(bool).to_numpy()
        else:
            valid = flatmap_valid & depth_valid

        invalid_reason = self._parquet_invalid_reasons(
            source,
            flatmap_valid=flatmap_valid,
            depth_valid=depth_valid,
            valid=valid,
        )
        if "flatmap_lookup_mode" in source.columns:
            lookup_mode = source["flatmap_lookup_mode"].fillna("").astype(str)
            lookup_mode = lookup_mode.reset_index(drop=True)
            lookup_mode = lookup_mode.where(
                lookup_mode.isin(
                    [
                        FLATMAP_LOOKUP_DIRECT,
                        FLATMAP_LOOKUP_MIRRORED_DEPTH,
                        FLATMAP_LOOKUP_MIRRORED,
                        FLATMAP_LOOKUP_UNMAPPED,
                    ]
                ),
                np.where(
                    valid,
                    FLATMAP_LOOKUP_DIRECT,
                    FLATMAP_LOOKUP_UNMAPPED,
                ),
            )
        else:
            lookup_mode = pd.Series(
                np.where(
                    valid,
                    FLATMAP_LOOKUP_DIRECT,
                    FLATMAP_LOOKUP_UNMAPPED,
                ),
                index=range(len(source)),
            )

        projected = pd.DataFrame(
            {
                "file_id": source["file_id"].reset_index(drop=True),
                "neuron_id": self._column_or_default(source, "neuron_id", ""),
                "subject": self._column_or_default(source, "subject", ""),
                "node_id": source["node_id"].reset_index(drop=True),
                "parent_id": source["parent_id"].reset_index(drop=True),
                "type": source["type"].reset_index(drop=True),
                "x_um": pd.to_numeric(source["x"], errors="coerce"),
                "y_um": pd.to_numeric(source["y"], errors="coerce"),
                "z_um": pd.to_numeric(source["z"], errors="coerce"),
                "voxel_i": self._column_or_default(source, "voxel_i", pd.NA),
                "voxel_j": self._column_or_default(source, "voxel_j", pd.NA),
                "voxel_k": self._column_or_default(source, "voxel_k", pd.NA),
                "x_flat": x_flat,
                "y_flat": y_flat,
                "depth_um": depth_um,
                "flatmap_valid": flatmap_valid,
                "depth_valid": depth_valid,
                "valid": valid,
                "invalid_reason": invalid_reason,
                "region_id": self._column_or_default(source, "region_id", pd.NA),
                "region_acronym": self._column_or_default(
                    source,
                    "region_acronym",
                    "",
                ),
                "flatmap_style": "precomputed_parquet",
                "coordinate_mode": "parquet_columns",
                "flatmap_lookup_mode": lookup_mode,
            }
        )

        segments = build_projected_segments(projected)
        summary = summarize_projection(projected, segments)
        return FlatmapProjectionResult(projected, segments, summary)

    def _normalise_precomputed_style_columns(
        self,
        source: pd.DataFrame,
    ) -> pd.DataFrame:
        """Map the selected v3 style columns onto the renderer's generic names."""
        style_key = self._current_style_key()
        if style_key == "both_shaped":
            suffix = "shaped"
        elif style_key == "both_square":
            suffix = "square"
        else:
            raise RuntimeError(
                "Version-3 precomputed coordinates support bilateral shaped "
                "and bilateral square styles only."
            )

        x_column = f"x_flat_{suffix}"
        y_column = f"y_flat_{suffix}"
        if not {x_column, y_column}.issubset(source.columns):
            if {"x_flat", "y_flat"}.issubset(source.columns):
                return source
        missing = [
            column
            for column in (x_column, y_column, "depth_um")
            if column not in source.columns
        ]
        if missing:
            raise RuntimeError(
                f"Loaded Parquet is missing version-3 precomputed column(s): {missing}"
            )

        normalised = source.copy()
        normalised["x_flat"] = normalised[x_column]
        normalised["y_flat"] = normalised[y_column]
        aliases = {
            f"flatmap_{suffix}_valid": "flatmap_valid",
            f"flatmap_{suffix}_projection_valid": "flatmap_projection_valid",
            f"flatmap_{suffix}_invalid_code": "flatmap_invalid_code",
            f"flatmap_{suffix}_lookup_mode": "flatmap_lookup_mode",
        }
        for style_column, generic_column in aliases.items():
            if style_column in normalised.columns:
                normalised[generic_column] = normalised[style_column]
        if "depth_lookup_mode" in normalised.columns:
            depth_modes = normalised["depth_lookup_mode"].fillna("").astype(str)
            combined_modes = (
                normalised.get("flatmap_lookup_mode", "")
                if "flatmap_lookup_mode" in normalised.columns
                else pd.Series([""] * len(normalised), index=normalised.index)
            )
            combined_modes = combined_modes.fillna("").astype(str).copy()
            # A recovered depth only makes a valid mirrored-depth projection
            # when the selected style's original-voxel XY lookup succeeded.
            # Keep an independently unmapped XY lookup unmapped.
            combined_modes.loc[
                (depth_modes == FLATMAP_LOOKUP_MIRRORED_DEPTH)
                & (combined_modes == FLATMAP_LOOKUP_DIRECT)
            ] = FLATMAP_LOOKUP_MIRRORED_DEPTH
            combined_modes.loc[depth_modes == FLATMAP_LOOKUP_UNMAPPED] = (
                FLATMAP_LOOKUP_UNMAPPED
            )
            normalised["flatmap_lookup_mode"] = combined_modes
        return normalised

    @staticmethod
    def _column_or_default(
        table: pd.DataFrame,
        column: str,
        default: object,
    ) -> pd.Series:
        if column in table.columns:
            return table[column].reset_index(drop=True)
        return pd.Series([default] * len(table), index=range(len(table)))

    @staticmethod
    def _bool_column_or_default(
        table: pd.DataFrame,
        column: str,
        default: np.ndarray,
    ) -> np.ndarray:
        if column in table.columns:
            return table[column].fillna(False).astype(bool).to_numpy()
        return np.asarray(default, dtype=bool)

    @staticmethod
    def _parquet_invalid_reasons(
        table: pd.DataFrame,
        *,
        flatmap_valid: np.ndarray,
        depth_valid: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
        if "flatmap_invalid_code" in table.columns:
            reasons = [
                flatmap_invalid_code_to_reason(code)
                for code in table["flatmap_invalid_code"].tolist()
            ]
            return np.asarray(reasons, dtype=object)
        if "invalid_reason" in table.columns:
            return table["invalid_reason"].fillna("").astype(str).to_numpy()

        reasons = np.full(len(table), "", dtype=object)
        reasons[~flatmap_valid] = "invalid_flatmap"
        reasons[flatmap_valid & ~depth_valid] = "invalid_depth"
        reasons[valid] = ""
        return reasons

    @staticmethod
    def _path_signature(path: Path) -> tuple[str, int | None, int | None]:
        try:
            stat = path.stat()
        except OSError:
            return (str(path), None, None)
        return (str(path), int(stat.st_size), int(stat.st_mtime_ns))

    def _lookup_stats_cache_key_for(
        self,
        volume_set,
        *,
        invalid_zero_sentinel: bool,
        invalid_negative_one_sentinel: bool,
    ) -> tuple[object, ...]:
        return (
            self._path_signature(Path(volume_set.flatmap_path)),
            self._path_signature(Path(volume_set.depth_path)),
            bool(invalid_zero_sentinel),
            bool(invalid_negative_one_sentinel),
            tuple(volume_set.flatmap.shape),
            str(volume_set.flatmap.dtype),
            tuple(volume_set.depth.shape),
            str(volume_set.depth.dtype),
        )

    def _lookup_stats_for_volume_set(
        self,
        volume_set,
        *,
        invalid_zero_sentinel: bool,
        invalid_negative_one_sentinel: bool,
    ) -> FlatmapLookupStats:
        key = self._lookup_stats_cache_key_for(
            volume_set,
            invalid_zero_sentinel=invalid_zero_sentinel,
            invalid_negative_one_sentinel=invalid_negative_one_sentinel,
        )
        cached_key = getattr(self, "_lookup_stats_cache_key", None)
        cached_stats = getattr(self, "_lookup_stats_cache", None)
        if cached_key == key and cached_stats is not None:
            return cached_stats

        stats = compute_flatmap_lookup_stats(
            volume_set.flatmap,
            volume_set.depth,
            invalid_zero_sentinel=invalid_zero_sentinel,
            invalid_negative_one_sentinel=invalid_negative_one_sentinel,
        )
        self._lookup_stats_cache_key = key
        self._lookup_stats_cache = stats
        return stats

    def _apply_projection_result(
        self,
        result: FlatmapProjectionResult,
        render_result: FlatmapRenderResult,
        *,
        flatmap_style: str | None = None,
        coordinate_mode: str | None = None,
        lookup_stats: FlatmapLookupStats | None = None,
        input_file_ids: tuple[str, ...] = (),
    ) -> None:
        self._last_projected_nodes = render_result.projected_nodes
        self._last_summary = result.summary
        self._last_render_summary = render_result.summary
        self._last_render_mode = self._current_render_mode()
        self._last_flatmap_style = flatmap_style or self._current_style_filename()
        self._last_coordinate_mode = coordinate_mode or self._current_coordinate_mode()
        self._last_volume_shape = tuple(
            int(size) for size in render_result.volume.shape
        )
        self._last_lookup_stats = lookup_stats
        self._last_input_file_ids = tuple(input_file_ids)
        self._last_flatmap_path = (
            str(self._flatmap_path) if self._flatmap_path else None
        )
        self._last_depth_path = str(self._depth_path) if self._depth_path else None
        self._last_projection_source = self._current_projection_source()
        active_profile = getattr(self, "_active_cache_profile", None)
        if self._last_projection_source == _PROJECTION_SOURCE_PRECOMPUTED:
            self._last_cache_dir = (
                str(self._region_cache_dir) if self._region_cache_dir else None
            )
            self._last_cache_profile_id = self._cache_profile_id(active_profile)
        else:
            self._last_cache_dir = None
            self._last_cache_profile_id = None
        self._summary_label.setText(
            self._format_render_summary(result.summary, render_result.summary)
        )
        self._create_or_update_render_layer(
            render_result,
            result.summary,
            flatmap_style=flatmap_style or self._current_style_filename(),
            coordinate_mode=coordinate_mode or self._current_coordinate_mode(),
            render_mode=self._current_render_mode(),
        )
        self._export_btn.setEnabled(not render_result.projected_nodes.empty)
        self._notify_flatmap_correlation_source_changed()

    def latest_flatmap_correlation_source(
        self,
    ) -> FlatmapVoxelCorrelationSource | None:
        """Return the latest heatmap render as a flatmap-clustering source."""
        if self._last_render_mode != _RENDER_HEATMAP:
            return None
        if not self._latest_heatmap_layer_is_rendered():
            return None
        projected_nodes = getattr(self, "_last_projected_nodes", None)
        render_summary = getattr(self, "_last_render_summary", None)
        volume_shape = getattr(self, "_last_volume_shape", None)
        if projected_nodes is None or render_summary is None or volume_shape is None:
            return None
        if projected_nodes.empty or int(render_summary.traces_represented) < 2:
            return None
        last_source = getattr(self, "_last_projection_source", None)
        if last_source == _PROJECTION_SOURCE_PRECOMPUTED:
            if self._current_style_key() != getattr(self, "_last_flatmap_style", None):
                return None
            current_profile_id = self._cache_profile_id(
                getattr(self, "_active_cache_profile", None)
            )
            if current_profile_id != str(
                getattr(self, "_last_cache_profile_id", "") or ""
            ):
                return None
        else:
            if (
                getattr(self, "_last_lookup_stats", None) is None
                or not self._last_flatmap_path
                or not self._last_depth_path
            ):
                return None
            if (
                self._flatmap_path is None
                or self._depth_path is None
                or str(self._flatmap_path) != self._last_flatmap_path
                or str(self._depth_path) != self._last_depth_path
            ):
                return None

        input_file_ids = tuple(getattr(self, "_last_input_file_ids", ()) or ())
        if not input_file_ids and "file_id" in projected_nodes.columns:
            input_file_ids = tuple(
                str(value)
                for value in self._deduplicate_file_ids(
                    projected_nodes["file_id"].tolist()
                )
            )

        return FlatmapVoxelCorrelationSource(
            projected_nodes=projected_nodes,
            volume_shape=tuple(int(size) for size in volume_shape),
            input_file_ids=input_file_ids,
            xy_bins=int(render_summary.xy_bins),
            depth_bin_um=float(render_summary.depth_bin_um),
            include_depth_minus_one=bool(render_summary.includes_depth_minus_one_plane),
            flatmap_style=getattr(self, "_last_flatmap_style", None),
            coordinate_mode=getattr(self, "_last_coordinate_mode", None),
            flatmap_path=self._last_flatmap_path,
            depth_path=self._last_depth_path,
            invalid_zero_sentinel=self._zero_sentinel_cb.isChecked(),
            invalid_negative_one_sentinel=self._negative_one_sentinel_cb.isChecked(),
            mirror_depth_fallback=True,
            mirror_coord_axis=2,
            lookup_stats=getattr(self, "_last_lookup_stats", None),
            cache_dir=getattr(self, "_last_cache_dir", None),
            cache_profile_id=getattr(self, "_last_cache_profile_id", None),
            cache_style=(
                self._current_style_key()
                if hasattr(self, "_style_combo")
                else getattr(self, "_last_flatmap_style", None)
            ),
        )

    def _create_region_labels(self) -> None:
        """Create or update the selected-region flatmap labels layer."""
        try:
            result = self._create_region_labels_from_current_state()
            if result is not None:
                show_info("Flatmap region labels complete.")
        except Exception as exc:
            logger.exception("Flatmap region label creation failed")
            message = f"Flatmap region labels failed: {exc}"
            self._set_region_labels_status(message)
            show_warning(message)

    def _create_region_labels_from_current_state(self):
        """Build a flatmap region-label volume and show it as a Labels layer."""
        if self._current_projection_source() == _PROJECTION_SOURCE_PRECOMPUTED:
            return self._create_cached_region_labels()
        self._projection_request_ready()
        selected_region_ids = self._selected_region_ids_for_labels()
        atlas_name = self._current_region_label_atlas_name()
        atlas = self._region_label_atlas_for_name(atlas_name)
        if atlas is None:
            self._start_region_label_atlas_load(atlas_name)
            return None
        return self._create_region_labels_for_atlas(
            atlas,
            selected_region_ids,
        )

    def _require_active_cache_profile(self):
        profile = getattr(self, "_active_cache_profile", None)
        if profile is None:
            raise RuntimeError(
                "No compatible cache profile is active. Choose a cache directory "
                "containing the loaded Parquet lookup-set ID and atlas catalog."
            )
        return profile

    def _selected_parent_region_ids(self) -> list[int]:
        values = self._selected_parent_region_ids_provider() or []
        return sorted({int(value) for value in values if int(value) > 0})

    def _create_cached_region_labels(self):
        from ..flatmap_region_cache import materialize_region_selection

        profile = self._require_active_cache_profile()
        selected_region_ids = self._selected_region_ids_for_labels()
        result = materialize_region_selection(
            profile,
            selected_region_ids,
            style=self._current_style_key(),
            direct_region_ids=self._selected_parent_region_ids(),
            include_surfaces=False,
            include_outlines=False,
        )
        atlas = self._atlas_provider()
        metadata = {
            "projection_kind": "flatmap_region_labels",
            "source": "precomputed_cache",
            "cache_path": str(self._region_cache_dir),
            "cache_profile_id": result.profile_id,
            "flatmap_style": self._current_style_key(),
            "selected_region_ids": [int(value) for value in result.selected_region_ids],
            "represented_region_ids": [
                int(value) for value in result.represented_region_ids
            ],
            "summary": result.summary.to_dict(),
        }
        layer = self._create_or_update_region_labels_layer(
            result,
            metadata,
            atlas=atlas,
        )
        self._region_labels_layer = layer
        self._focus_projection_view(layer, result.labels)
        message = (
            f"Loaded {result.summary.labeled_bins:,} cached region bin(s) "
            f"from profile {result.profile_id}."
        )
        self._set_region_labels_status(message)
        return result

    def _selected_region_ids_for_labels(self) -> list[int]:
        selected_region_ids = sorted(
            {
                int(region_id)
                for region_id in (self._selected_region_ids_provider() or [])
                if int(region_id) > 0
            }
        )
        if not selected_region_ids:
            raise RuntimeError(
                "Select at least one atlas region before creating labels."
            )
        return selected_region_ids

    def _create_region_labels_for_atlas(
        self,
        atlas,
        selected_region_ids: list[int],
    ):
        volume_set = load_flatmap_volume_set(self._flatmap_path, self._depth_path)
        lookup_stats = self._lookup_stats_for_volume_set(
            volume_set,
            invalid_zero_sentinel=self._zero_sentinel_cb.isChecked(),
            invalid_negative_one_sentinel=self._negative_one_sentinel_cb.isChecked(),
        )
        result = build_flatmap_region_label_volume(
            np.asarray(atlas.annotation),
            volume_set.flatmap,
            volume_set.depth,
            selected_region_ids=selected_region_ids,
            xy_bins=self._current_xy_bins(),
            depth_bin_um=self._current_depth_bin_um(),
            invalid_zero_sentinel=self._zero_sentinel_cb.isChecked(),
            invalid_negative_one_sentinel=self._negative_one_sentinel_cb.isChecked(),
            lookup_stats=lookup_stats,
            mirror_depth_fallback=True,
            mirror_coord_axis=2,
        )
        metadata = self._region_labels_metadata(result, atlas)
        layer = self._create_or_update_region_labels_layer(
            result,
            metadata,
            atlas=atlas,
        )
        self._region_labels_layer = layer
        self._focus_projection_view(layer, result.labels)

        message = (
            "Created flatmap region labels: "
            f"{result.summary.labeled_voxels:,} labeled voxel(s) from "
            f"{len(result.selected_region_ids):,} selected region ID(s)."
        )
        self._status_label.setText(message)
        label = getattr(self, "_region_labels_status_label", None)
        if label is not None:
            label.setText(message)
        return result

    def _region_label_atlas_for_name(self, atlas_name: str):
        cached = self._region_label_atlas_cache.get(atlas_name)
        if cached is not None:
            return cached

        provider_atlas = self._atlas_provider()
        provider_name = str(getattr(provider_atlas, "atlas_name", "") or "")
        if provider_atlas is not None and provider_name == atlas_name:
            self._region_label_atlas_cache[atlas_name] = provider_atlas
            return provider_atlas
        return None

    def _start_region_label_atlas_load(self, atlas_name: str) -> None:
        self._pending_region_label_request = True
        if self._region_label_atlas_load_running():
            self._set_region_labels_status(
                f"Loading region-label atlas {atlas_name}..."
            )
            return

        from qtpy.QtCore import QThread

        from ..workers import AtlasLoadWorker

        self._set_region_label_controls_enabled(False)
        self._set_region_labels_status(f"Loading region-label atlas {atlas_name}...")

        thread = QThread()
        worker = AtlasLoadWorker(atlas_name)
        self._region_label_atlas_load_thread = thread
        self._region_label_atlas_load_worker = worker
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.status.connect(self._on_region_label_atlas_load_status)
        worker.finished.connect(
            lambda atlas, expected=atlas_name: (
                self._on_region_label_atlas_load_finished(
                    atlas,
                    expected,
                )
            )
        )
        worker.error.connect(self._on_region_label_atlas_load_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda: self._cleanup_region_label_atlas_load_thread(thread, worker)
        )
        thread.start()

    def _region_label_atlas_load_running(self) -> bool:
        thread = getattr(self, "_region_label_atlas_load_thread", None)
        is_running = getattr(thread, "isRunning", None)
        return bool(thread is not None and callable(is_running) and is_running())

    def _on_region_label_atlas_load_status(self, message: str) -> None:
        self._set_region_labels_status(str(message))

    def _on_region_label_atlas_load_finished(self, atlas, atlas_name: str) -> None:
        resolved_name = str(getattr(atlas, "atlas_name", "") or atlas_name)
        self._region_label_atlas_cache[atlas_name] = atlas
        self._region_label_atlas_cache[resolved_name] = atlas
        self._set_region_label_controls_enabled(True)
        self._set_region_labels_status(f"Loaded region-label atlas {resolved_name}.")

        if self._pending_region_label_request:
            self._pending_region_label_request = False
            self._create_region_labels()

    def _on_region_label_atlas_load_error(self, error_msg: str) -> None:
        self._pending_region_label_request = False
        self._set_region_label_controls_enabled(True)
        message = f"Region-label atlas load failed: {error_msg}"
        logger.error(message)
        self._set_region_labels_status(message)
        show_warning(message)

    def _cleanup_region_label_atlas_load_thread(self, thread, worker) -> None:
        if getattr(self, "_region_label_atlas_load_thread", None) is thread:
            self._region_label_atlas_load_thread = None
        if getattr(self, "_region_label_atlas_load_worker", None) is worker:
            self._region_label_atlas_load_worker = None

    def _set_region_label_controls_enabled(self, enabled: bool) -> None:
        for widget_name in (
            "_region_label_atlas_combo",
            "_region_labels_btn",
            "_clear_region_labels_btn",
            "_region_surfaces_btn",
            "_region_outlines_btn",
            "_clear_region_geometry_btn",
        ):
            widget = getattr(self, widget_name, None)
            set_enabled = getattr(widget, "setEnabled", None)
            if callable(set_enabled):
                effective = bool(enabled)
                if widget_name in {
                    "_region_surfaces_btn",
                    "_region_outlines_btn",
                    "_clear_region_geometry_btn",
                }:
                    effective = effective and (
                        self._current_projection_source()
                        == _PROJECTION_SOURCE_PRECOMPUTED
                        and getattr(self, "_active_cache_profile", None) is not None
                    )
                elif widget_name == "_region_labels_btn":
                    effective = effective and (
                        self._current_projection_source()
                        != _PROJECTION_SOURCE_PRECOMPUTED
                        or getattr(self, "_active_cache_profile", None) is not None
                    )
                set_enabled(effective)

    def _set_region_labels_status(self, message: str) -> None:
        status_label = getattr(self, "_status_label", None)
        if status_label is not None:
            status_label.setText(message)
        region_status_label = getattr(self, "_region_labels_status_label", None)
        if region_status_label is not None:
            region_status_label.setText(message)

    def _region_labels_metadata(
        self,
        result: FlatmapRegionLabelsResult,
        atlas,
    ) -> dict[str, object]:
        acronyms = [
            str(acronym)
            for acronym in (self._selected_region_acronyms_provider() or [])
        ]
        return {
            "projection_kind": "flatmap_region_labels",
            "flatmap_style": self._current_style_filename(),
            "flatmap_path": str(self._flatmap_path) if self._flatmap_path else "",
            "depth_path": str(self._depth_path) if self._depth_path else "",
            "atlas_name": str(getattr(atlas, "atlas_name", "")),
            "selected_region_ids": [
                int(region_id) for region_id in result.selected_region_ids
            ],
            "selected_region_acronyms": acronyms,
            "represented_region_ids": [
                int(region_id) for region_id in result.represented_region_ids
            ],
            "summary": result.summary.to_dict(),
        }

    def _create_or_update_region_labels_layer(
        self,
        result: FlatmapRegionLabelsResult,
        metadata: dict[str, object],
        *,
        atlas=None,
    ):
        viewer = self._display_viewer()
        layer = self._region_labels_layer
        if not self._layer_is_in_viewer(layer, viewer=viewer):
            self._region_labels_layer = None
            layer = None
        layer = layer or self._find_layer_by_name(
            _REGION_LABELS_LAYER_NAME,
            viewer=viewer,
        )
        colormap = self._region_label_colormap(atlas, result.represented_region_ids)
        kwargs: dict[str, object] = {
            "name": _REGION_LABELS_LAYER_NAME,
            "opacity": 0.35,
            "visible": True,
            "metadata": metadata,
        }
        if colormap is not None:
            kwargs["colormap"] = colormap

        if layer is None:
            layer = viewer.add_labels(result.labels, **kwargs)
        else:
            blocker = getattr(getattr(layer, "events", None), "blocker_all", None)
            if callable(blocker):
                with blocker():
                    self._set_region_labels_layer_data(
                        layer, result, metadata, colormap
                    )
            else:
                self._set_region_labels_layer_data(layer, result, metadata, colormap)
            refresh = getattr(layer, "refresh", None)
            if callable(refresh):
                refresh()

        setattr(layer, "_napari_swc_flatmap_region_labels_result", result)
        return layer

    @staticmethod
    def _set_region_labels_layer_data(
        layer,
        result: FlatmapRegionLabelsResult,
        metadata: dict[str, object],
        colormap,
    ) -> None:
        layer.data = result.labels
        layer.metadata = metadata
        layer.opacity = 0.35
        layer.visible = True
        if colormap is not None:
            layer.colormap = colormap

    @staticmethod
    def _atlas_structure_for_region_id(atlas, region_id: int):
        structures = getattr(atlas, "structures", None)
        if structures is None:
            return None
        try:
            return structures[int(region_id)]
        except (KeyError, TypeError):
            return None

    @classmethod
    def _region_label_colormap(cls, atlas, region_ids: list[int]):
        if atlas is None:
            return None
        try:
            from napari.utils import DirectLabelColormap
        except Exception:
            return None

        color_dict: dict[int | None, np.ndarray] = {
            None: np.array([0, 0, 0, 0], dtype=np.float32),
            0: np.array([0, 0, 0, 0], dtype=np.float32),
        }
        for region_id in region_ids:
            structure = cls._atlas_structure_for_region_id(atlas, int(region_id))
            if structure is None:
                rgb = [128, 128, 128]
            else:
                rgb = structure.get("rgb_triplet", [128, 128, 128])
            rgba = np.asarray(
                [
                    float(rgb[0]) / 255.0,
                    float(rgb[1]) / 255.0,
                    float(rgb[2]) / 255.0,
                    1.0,
                ],
                dtype=np.float32,
            )
            color_dict[int(region_id)] = rgba
        return DirectLabelColormap(color_dict=color_dict)

    def _clear_region_labels(self) -> None:
        """Remove the flatmap region labels layer if present."""
        layers = self._display_layers(create=False)
        layer = self._region_labels_layer
        if not self._layer_is_in_viewer(layer):
            layer = self._find_layer_by_name(
                _REGION_LABELS_LAYER_NAME,
                create=False,
            )
        if layer is not None and layers is not None:
            try:
                layers.remove(layer)
            except ValueError:
                pass
        self._region_labels_layer = None
        message = "Cleared flatmap region labels."
        self._status_label.setText(message)
        label = getattr(self, "_region_labels_status_label", None)
        if label is not None:
            label.setText("No flatmap region labels created.")

    @classmethod
    def _atlas_region_rgba(cls, atlas, region_id: int) -> np.ndarray:
        structure = cls._atlas_structure_for_region_id(atlas, region_id)
        rgb = (
            structure.get("rgb_triplet", [128, 128, 128])
            if structure is not None
            else [128, 128, 128]
        )
        return np.asarray(
            [float(rgb[0]) / 255, float(rgb[1]) / 255, float(rgb[2]) / 255, 1],
            dtype=np.float32,
        )

    def _cached_geometry_inputs(self):
        profile = self._require_active_cache_profile()
        direct_ids = self._selected_parent_region_ids()
        if not direct_ids:
            raise RuntimeError(
                "Select at least one parent atlas region before showing cached geometry."
            )
        atlas = self._atlas_provider()
        if atlas is None:
            raise RuntimeError(
                "Load a matching BrainGlobe atlas structure catalog for region colors."
            )
        return profile, direct_ids, atlas

    def _create_region_surfaces(self) -> None:
        """Show cached descendant-union exposed-face shells for selected parents."""
        try:
            from napari.utils.colormaps import Colormap

            from ..flatmap_region_cache import materialize_region_surface

            profile, direct_ids, atlas = self._cached_geometry_inputs()
            self._clear_region_surface_layers()
            viewer = self._display_viewer()
            created = []
            for region_id in direct_ids:
                surface = materialize_region_surface(
                    profile,
                    region_id,
                    style=self._current_style_key(),
                )
                if surface is None or not len(surface.faces):
                    continue
                rgba = self._atlas_region_rgba(atlas, region_id)
                name = (
                    _REGION_SURFACES_LAYER_NAME
                    if len(direct_ids) == 1
                    else f"{_REGION_SURFACES_LAYER_NAME}: {region_id}"
                )
                layer = viewer.add_surface(
                    (
                        np.array(surface.vertices, dtype=np.float32, copy=True),
                        np.array(surface.faces, dtype=np.int32, copy=True),
                        np.ones(len(surface.vertices), dtype=np.float32),
                    ),
                    name=name,
                    colormap=Colormap(np.vstack([rgba, rgba])),
                    contrast_limits=(0.0, 1.0),
                    opacity=0.45,
                    metadata={
                        "projection_kind": "flatmap_region_surface",
                        "source": "precomputed_cache",
                        "cache_path": str(self._region_cache_dir),
                        "cache_profile_id": profile.profile_id,
                        "flatmap_style": self._current_style_key(),
                        "region_id": int(region_id),
                        "component_count": int(surface.component_count),
                    },
                )
                created.append(layer)
            self._region_surfaces_layers = created
            message = f"Loaded {len(created)} cached region surface layer(s)."
            self._set_region_labels_status(message)
            if not created:
                show_warning(
                    "The selected cache profile has no surface for this selection."
                )
        except Exception as exc:
            logger.exception("Cached flatmap region surfaces failed")
            show_warning(f"Cached flatmap region surfaces failed: {exc}")

    def _create_region_outlines(self) -> None:
        """Show cached per-depth XY perimeter vectors for selected parents."""
        try:
            from ..flatmap_region_cache import materialize_region_outlines

            profile, direct_ids, atlas = self._cached_geometry_inputs()
            self._clear_region_outline_layers()
            viewer = self._display_viewer()
            created = []
            for region_id in direct_ids:
                outlines = materialize_region_outlines(
                    profile,
                    region_id,
                    style=self._current_style_key(),
                )
                if outlines is None or not len(outlines.vectors):
                    continue
                rgba = self._atlas_region_rgba(atlas, region_id)
                name = (
                    _REGION_OUTLINES_LAYER_NAME
                    if len(direct_ids) == 1
                    else f"{_REGION_OUTLINES_LAYER_NAME}: {region_id}"
                )
                layer = viewer.add_vectors(
                    np.array(outlines.vectors, dtype=np.float32, copy=True),
                    name=name,
                    edge_color=rgba,
                    edge_width=1.5,
                    opacity=0.9,
                    metadata={
                        "projection_kind": "flatmap_region_outlines",
                        "source": "precomputed_cache",
                        "cache_path": str(self._region_cache_dir),
                        "cache_profile_id": profile.profile_id,
                        "flatmap_style": self._current_style_key(),
                        "region_id": int(region_id),
                    },
                )
                created.append(layer)
            self._region_outlines_layers = created
            message = f"Loaded {len(created)} cached region outline layer(s)."
            self._set_region_labels_status(message)
            if not created:
                show_warning(
                    "The selected cache profile has no outlines for this selection."
                )
        except Exception as exc:
            logger.exception("Cached flatmap region outlines failed")
            show_warning(f"Cached flatmap region outlines failed: {exc}")

    def _clear_named_region_layers(self, prefix: str) -> None:
        layers = self._display_layers(create=False)
        if layers is None:
            return
        for layer in list(layers):
            if str(getattr(layer, "name", "")).startswith(prefix):
                try:
                    layers.remove(layer)
                except ValueError:
                    pass

    def _clear_region_surface_layers(self) -> None:
        self._clear_named_region_layers(_REGION_SURFACES_LAYER_NAME)
        self._region_surfaces_layers = []

    def _clear_region_outline_layers(self) -> None:
        self._clear_named_region_layers(_REGION_OUTLINES_LAYER_NAME)
        self._region_outlines_layers = []

    def _clear_region_geometry(self) -> None:
        self._clear_region_surface_layers()
        self._clear_region_outline_layers()
        self._set_region_labels_status("Cleared cached flatmap region geometry.")

    def _color_for_file_id(
        self,
        file_id: object,
        color_map: dict[object, list[float]],
    ) -> np.ndarray:
        raw = color_map.get(file_id)
        if raw is None:
            raw = color_map.get(str(file_id))
        if raw is None:
            return _DEFAULT_TRACE_COLOR.copy()

        color = np.asarray(raw, dtype=float).reshape(-1)
        if color.size < 4:
            color = np.pad(color, (0, 4 - color.size), constant_values=1.0)
        return np.clip(color[:4], 0.0, 1.0)

    def _colors_for_file_ids(self, file_ids: list[object]) -> np.ndarray:
        color_map = self._color_map_provider() or {}
        if len(file_ids) == 0:
            return np.empty((0, 4), dtype=float)
        return np.vstack(
            [self._color_for_file_id(file_id, color_map) for file_id in file_ids]
        )

    @staticmethod
    def _heatmap_contrast_limits(volume: np.ndarray) -> tuple[float, float]:
        upper = float(np.nanmax(volume)) if volume.size else 0.0
        if not np.isfinite(upper) or upper <= 0.0:
            return (0.0, 1.0)
        return (0.0, upper)

    @staticmethod
    def _render_layer_name(render_mode: str) -> str:
        return (
            _POINTS_LAYER_NAME if render_mode == _RENDER_POINTS else _HEATMAP_LAYER_NAME
        )

    @staticmethod
    def _is_flatmap_render_layer_name(name: object) -> bool:
        return name in _FLATMAP_RENDER_LAYER_NAMES or (
            isinstance(name, str) and name.startswith(_GROUPED_HEATMAP_LAYER_PREFIX)
        )

    def _find_layer_by_name(self, name: str, *, viewer=None, create: bool = True):
        if viewer is None:
            layers = self._display_layers(create=create)
        else:
            layers = getattr(viewer, "layers", None)
        if layers is None:
            return None
        for layer in layers:
            if getattr(layer, "name", None) == name:
                return layer
        return None

    def _layer_is_in_viewer(self, layer, *, viewer=None) -> bool:
        layers = (
            getattr(viewer, "layers", None)
            if viewer is not None
            else self._display_layers(create=False)
        )
        if layer is None or layers is None:
            return False
        return any(existing is layer for existing in layers)

    def _latest_heatmap_layer_is_rendered(self) -> bool:
        """Return whether the latest flatmap render still has a heatmap layer."""
        layer = getattr(self, "_projection_layer", None)
        if self._layer_is_in_viewer(layer):
            metadata = getattr(layer, "metadata", {}) or {}
            if metadata.get("flatmap_render_mode") == _RENDER_HEATMAP:
                return True

        layers = self._display_layers(create=False) or ()
        for candidate in layers:
            name = getattr(candidate, "name", None)
            metadata = getattr(candidate, "metadata", {}) or {}
            if (
                self._is_flatmap_render_layer_name(name)
                and metadata.get("flatmap_render_mode") == _RENDER_HEATMAP
            ):
                return True
        return False

    def _cached_projection_layer_for_name(self, name: str):
        layer = getattr(self, "_projection_layer", None)
        if layer is None:
            return None
        if getattr(layer, "name", None) != name or not self._layer_is_in_viewer(layer):
            self._projection_layer = None
            return None
        return layer

    def _remove_projection_layer(
        self,
        *,
        except_name: str | None = None,
        create: bool = True,
    ) -> None:
        layers = self._display_layers(create=create)
        if layers is None:
            self._projection_layer = None
            return
        for layer in list(layers):
            name = getattr(layer, "name", None)
            if not self._is_flatmap_render_layer_name(name) or name == except_name:
                continue
            try:
                layers.remove(layer)
            except ValueError:
                pass
            if layer is self._projection_layer:
                self._projection_layer = None
        if self._projection_layer is not None and (
            getattr(self._projection_layer, "name", None) != except_name
            or not self._layer_is_in_viewer(self._projection_layer)
        ):
            self._projection_layer = None

    def _render_metadata(
        self,
        projection_summary: ProjectionSummary,
        render_summary: FlatmapRenderSummary,
        *,
        flatmap_style: str,
        coordinate_mode: str,
        render_mode: str,
        heatmap_color_mode: str | None = None,
    ) -> dict[str, object]:
        metadata = {
            "projection_kind": "isocortex_flatmap",
            "flatmap_render_mode": render_mode,
            "flatmap_style": flatmap_style,
            "coordinate_mode": coordinate_mode,
            "projection_summary": projection_summary.to_dict(),
            "render_summary": render_summary.to_dict(),
            "flatmap_path": str(self._flatmap_path) if self._flatmap_path else "",
            "depth_path": str(self._depth_path) if self._depth_path else "",
        }
        if render_mode == _RENDER_HEATMAP:
            metadata["flatmap_heatmap_color_mode"] = (
                heatmap_color_mode or _HEATMAP_COLOR_SINGLE
            )
        return metadata

    def _set_layer_state(
        self,
        layer,
        projected_nodes: pd.DataFrame,
        summary: ProjectionSummary,
        render_summary: FlatmapRenderSummary,
    ) -> None:
        setattr(layer, "_napari_swc_flatmap_projected_nodes", projected_nodes)
        setattr(layer, "_napari_swc_flatmap_summary", summary)
        setattr(layer, "_napari_swc_flatmap_render_summary", render_summary)

    @staticmethod
    def _format_render_summary(
        projection_summary: ProjectionSummary,
        render_summary: FlatmapRenderSummary,
    ) -> str:
        depth_minus_one_action = (
            "rendered" if render_summary.includes_depth_minus_one_plane else "excluded"
        )
        return (
            f"Input nodes: {projection_summary.total_nodes:,}\n"
            f"Flatmap-valid nodes: {render_summary.flatmap_valid_nodes:,}\n"
            f"Depth-valid nodes: {render_summary.depth_valid_nodes:,}\n"
            f"Depth -1 nodes {depth_minus_one_action}: "
            f"{render_summary.depth_minus_one_nodes:,}\n"
            f"Rendered nodes: {render_summary.rendered_nodes:,}\n"
            f"Nonzero heatmap voxels: {render_summary.nonzero_voxels:,}\n"
            f"Represented traces: {render_summary.traces_represented:,} "
            f"of {projection_summary.total_traces:,}\n"
            f"Invalid flatmap/depth: "
            f"{projection_summary.invalid_flatmap_nodes:,}/"
            f"{projection_summary.invalid_depth_nodes:,}\n"
            f"Lookup direct/mirrored-depth/mirrored/unmapped: "
            f"{projection_summary.direct_lookup_nodes:,}/"
            f"{projection_summary.mirrored_depth_lookup_nodes:,}/"
            f"{projection_summary.mirrored_lookup_nodes:,}/"
            f"{projection_summary.unmapped_lookup_nodes:,}"
        )

    def _create_or_update_render_layer(
        self,
        render_result: FlatmapRenderResult,
        projection_summary: ProjectionSummary,
        *,
        flatmap_style: str,
        coordinate_mode: str,
        render_mode: str,
    ):
        """Create or update the napari depth-aware flatmap render layer."""
        if render_result.summary.rendered_nodes == 0:
            self._remove_projection_layer(create=False)
            return None

        heatmap_color_mode = (
            self._current_heatmap_color_mode()
            if render_mode == _RENDER_HEATMAP
            else _HEATMAP_COLOR_SINGLE
        )
        layer_name = self._render_layer_name(render_mode)
        if (
            render_mode == _RENDER_HEATMAP
            and heatmap_color_mode != _HEATMAP_COLOR_SINGLE
        ):
            self._remove_projection_layer()
        else:
            self._remove_projection_layer(except_name=layer_name)
        metadata = self._render_metadata(
            projection_summary,
            render_result.summary,
            flatmap_style=flatmap_style,
            coordinate_mode=coordinate_mode,
            render_mode=render_mode,
            heatmap_color_mode=heatmap_color_mode,
        )
        layer = self._cached_projection_layer_for_name(
            layer_name
        ) or self._find_layer_by_name(layer_name)

        if render_mode == _RENDER_POINTS:
            layer = self._create_or_update_points_layer(layer, render_result, metadata)
        elif heatmap_color_mode == _HEATMAP_COLOR_SINGLE:
            layer = self._create_or_update_heatmap_layer(layer, render_result, metadata)
        else:
            layers = self._create_grouped_heatmap_layers(
                render_result,
                projection_summary,
                metadata,
                heatmap_color_mode=heatmap_color_mode,
            )
            layer = layers[0] if layers else None

        self._projection_layer = layer
        if layer is None:
            return None
        self._set_layer_state(
            layer,
            render_result.projected_nodes,
            projection_summary,
            render_result.summary,
        )
        data = (
            render_result.points
            if render_mode == _RENDER_POINTS
            else render_result.volume
        )
        self._focus_projection_view(layer, data)
        return layer

    def _create_or_update_heatmap_layer(
        self,
        layer,
        render_result: FlatmapRenderResult,
        metadata: dict[str, object],
    ):
        volume = render_result.volume
        contrast_limits = self._heatmap_contrast_limits(volume)
        metadata = dict(metadata)
        metadata["flatmap_heatmap_contrast_limits"] = contrast_limits
        if layer is None:
            layer = self._display_viewer().add_image(
                volume,
                name=_HEATMAP_LAYER_NAME,
                colormap="hot",
                blending="additive",
                rendering="mip",
                opacity=0.8,
                contrast_limits=contrast_limits,
                metadata=metadata,
            )
            self._install_heatmap_layer_workarounds(layer)
            self._store_heatmap_contrast_limits(layer, contrast_limits)
            return layer

        self._install_heatmap_layer_workarounds(layer)
        blocker = getattr(getattr(layer, "events", None), "blocker_all", None)
        if callable(blocker):
            with blocker():
                layer.data = volume
                layer.metadata = metadata
                self._apply_heatmap_contrast_limits(layer, contrast_limits)
                layer.visible = True
        else:
            layer.data = volume
            layer.metadata = metadata
            self._apply_heatmap_contrast_limits(layer, contrast_limits)
            layer.visible = True
        refresh = getattr(layer, "refresh", None)
        if callable(refresh):
            refresh()
        self._store_heatmap_contrast_limits(layer, contrast_limits)
        return layer

    def _create_grouped_heatmap_layers(
        self,
        render_result: FlatmapRenderResult,
        projection_summary: ProjectionSummary,
        metadata: dict[str, object],
        *,
        heatmap_color_mode: str,
    ) -> list[object]:
        groups = self._grouped_heatmap_volumes(
            render_result,
            heatmap_color_mode=heatmap_color_mode,
        )
        layers = []
        for group in groups:
            color = self._color_for_heatmap_group(
                group,
                heatmap_color_mode=heatmap_color_mode,
            )
            layer = self._add_grouped_heatmap_layer(
                group,
                metadata,
                color,
                heatmap_color_mode=heatmap_color_mode,
            )
            self._set_layer_state(
                layer,
                render_result.projected_nodes,
                projection_summary,
                render_result.summary,
            )
            layers.append(layer)
        return layers

    def _grouped_heatmap_volumes(
        self,
        render_result: FlatmapRenderResult,
        *,
        heatmap_color_mode: str,
    ) -> list[FlatmapGroupedVolume]:
        if heatmap_color_mode == _HEATMAP_COLOR_INDIVIDUAL:
            return build_flatmap_file_id_volumes(
                render_result.projected_nodes,
                tuple(render_result.volume.shape),
            )
        if heatmap_color_mode == _HEATMAP_COLOR_CLUSTER:
            return build_flatmap_cluster_volumes(
                render_result.projected_nodes,
                tuple(render_result.volume.shape),
                self._cluster_map_provider() or {},
            )
        return []

    @staticmethod
    def _grouped_heatmap_layer_name(
        group: FlatmapGroupedVolume,
        *,
        heatmap_color_mode: str,
    ) -> str:
        if heatmap_color_mode == _HEATMAP_COLOR_CLUSTER:
            return f"{_GROUPED_HEATMAP_LAYER_PREFIX}{group.label}"
        return f"{_GROUPED_HEATMAP_LAYER_PREFIX}{group.label}"

    def _color_for_heatmap_group(
        self,
        group: FlatmapGroupedVolume,
        *,
        heatmap_color_mode: str,
    ) -> np.ndarray:
        if heatmap_color_mode == _HEATMAP_COLOR_CLUSTER and group.group_key is None:
            return _DEFAULT_TRACE_COLOR.copy()
        color_map = self._color_map_provider() or {}
        for file_id in group.source_file_ids:
            if file_id in color_map or str(file_id) in color_map:
                return self._color_for_file_id(file_id, color_map)
        return _DEFAULT_TRACE_COLOR.copy()

    @staticmethod
    def _solid_tint_colormap(color: np.ndarray, name: str):
        rgba = np.asarray(color, dtype=float).reshape(-1)
        if rgba.size < 4:
            rgba = np.pad(rgba, (0, 4 - rgba.size), constant_values=1.0)
        rgba = np.clip(rgba[:4], 0.0, 1.0)
        try:
            from napari.utils.colormaps import Colormap
        except Exception:
            return "hot"
        return Colormap(
            colors=[
                [0.0, 0.0, 0.0, 0.0],
                [float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])],
            ],
            name=name,
        )

    def _add_grouped_heatmap_layer(
        self,
        group: FlatmapGroupedVolume,
        metadata: dict[str, object],
        color: np.ndarray,
        *,
        heatmap_color_mode: str,
    ):
        volume = group.volume
        contrast_limits = self._heatmap_contrast_limits(volume)
        group_metadata = dict(metadata)
        color_values = [float(value) for value in np.asarray(color)[:4]]
        group_metadata.update(
            {
                "flatmap_heatmap_color_mode": heatmap_color_mode,
                "flatmap_heatmap_group_key": group.group_key,
                "flatmap_heatmap_group_label": group.label,
                "flatmap_heatmap_group_color": color_values,
                "flatmap_heatmap_group_rendered_nodes": group.rendered_nodes,
                "flatmap_heatmap_group_nonzero_voxels": group.nonzero_voxels,
                "source_file_ids": list(group.source_file_ids),
                "file_ids": list(group.source_file_ids),
                "color": color_values,
                "flatmap_heatmap_contrast_limits": contrast_limits,
            }
        )
        layer_name = self._grouped_heatmap_layer_name(
            group,
            heatmap_color_mode=heatmap_color_mode,
        )
        layer = self._display_viewer().add_image(
            volume,
            name=layer_name,
            colormap=self._solid_tint_colormap(color, layer_name),
            blending="additive",
            rendering="mip",
            opacity=0.8,
            contrast_limits=contrast_limits,
            metadata=group_metadata,
        )
        self._install_heatmap_layer_workarounds(layer)
        self._store_heatmap_contrast_limits(layer, contrast_limits)
        return layer

    def _install_heatmap_layer_workarounds(self, layer) -> None:
        self._install_heatmap_status_guard(layer)
        self._install_heatmap_thumbnail_workarounds(layer)

    def _install_heatmap_status_guard(self, layer) -> None:
        """Avoid napari status errors while a heatmap slice catches up to 3D."""
        if getattr(layer, "_napari_swc_flatmap_status_guard_installed", False):
            return

        original_get_status = getattr(layer, "get_status", None)
        if not callable(original_get_status):
            return

        def guarded_get_status(
            position=None,
            *,
            view_direction=None,
            dims_displayed=None,
            world=False,
            value=None,
        ):
            try:
                return original_get_status(
                    position,
                    view_direction=view_direction,
                    dims_displayed=dims_displayed,
                    world=world,
                    value=value,
                )
            except IndexError as exc:
                if not self._is_stale_3d_status_slice(layer, dims_displayed, exc):
                    raise
                return self._status_without_sampled_value(layer, position)

        setattr(layer, "_napari_swc_flatmap_original_get_status", original_get_status)
        setattr(layer, "get_status", guarded_get_status)
        setattr(layer, "_napari_swc_flatmap_status_guard_installed", True)

    def _install_heatmap_thumbnail_workarounds(self, layer) -> None:
        """Keep generated heatmap thumbnails stable across 2D/3D axis changes."""
        if getattr(layer, "_napari_swc_flatmap_thumbnail_workarounds_installed", False):
            return

        widget = self
        original_update_thumbnail = getattr(layer, "_update_thumbnail", None)
        if callable(original_update_thumbnail):

            def safe_update_thumbnail(bound_layer) -> None:
                try:
                    original_update_thumbnail()
                except RuntimeError as error:
                    if not widget._is_thumbnail_rank_mismatch_error(error):
                        raise
                    if not getattr(
                        bound_layer,
                        "_napari_swc_flatmap_thumbnail_warning_logged",
                        False,
                    ):
                        logger.warning(
                            "Suppressed napari thumbnail update failure for "
                            "flatmap heatmap '%s': %s",
                            getattr(bound_layer, "name", "<unnamed>"),
                            error,
                        )
                        bound_layer._napari_swc_flatmap_thumbnail_warning_logged = True

            layer._update_thumbnail = MethodType(safe_update_thumbnail, layer)

        original_reset_contrast_limits = getattr(layer, "reset_contrast_limits", None)
        if callable(original_reset_contrast_limits):

            def stable_reset_contrast_limits(bound_layer, mode=None) -> None:
                if not widget._heatmap_requires_stable_limits(bound_layer):
                    original_reset_contrast_limits(mode)
                    return
                limits = widget._heatmap_stored_contrast_limits(bound_layer)
                if limits is None:
                    original_reset_contrast_limits(mode)
                    return
                widget._apply_heatmap_contrast_limits(bound_layer, limits)

            layer.reset_contrast_limits = MethodType(
                stable_reset_contrast_limits, layer
            )

        original_reset_contrast_limits_range = getattr(
            layer,
            "reset_contrast_limits_range",
            None,
        )
        if callable(original_reset_contrast_limits_range):

            def stable_reset_contrast_limits_range(bound_layer, mode=None) -> None:
                if not widget._heatmap_requires_stable_limits(bound_layer):
                    original_reset_contrast_limits_range(mode)
                    return
                limits = widget._heatmap_stored_contrast_limits(bound_layer)
                if limits is None:
                    original_reset_contrast_limits_range(mode)
                    return
                bound_layer.contrast_limits_range = limits

            layer.reset_contrast_limits_range = MethodType(
                stable_reset_contrast_limits_range,
                layer,
            )

        original_update_slice_response = getattr(layer, "_update_slice_response", None)
        if callable(original_update_slice_response):

            def stable_update_slice_response(bound_layer, response):
                keep_auto = bool(getattr(bound_layer, "_keep_auto_contrast", False))
                if not keep_auto or not widget._heatmap_requires_stable_limits(
                    bound_layer,
                    response,
                ):
                    return original_update_slice_response(response)

                bound_layer._keep_auto_contrast = False
                try:
                    result = original_update_slice_response(response)
                finally:
                    bound_layer._keep_auto_contrast = True

                limits = widget._heatmap_stored_contrast_limits(bound_layer)
                if limits is not None:
                    widget._apply_heatmap_contrast_limits(bound_layer, limits)
                return result

            layer._update_slice_response = MethodType(
                stable_update_slice_response,
                layer,
            )

        setattr(layer, "_napari_swc_flatmap_thumbnail_workarounds_installed", True)

    @staticmethod
    def _is_thumbnail_rank_mismatch_error(error: RuntimeError) -> bool:
        return "sequence argument must have length equal to input rank" in str(error)

    @staticmethod
    def _heatmap_ndisplay(layer, response=None) -> int | None:
        slice_input = getattr(response, "slice_input", None)
        ndisplay = getattr(slice_input, "ndisplay", None)
        if isinstance(ndisplay, (int, np.integer)):
            return int(ndisplay)

        slice_input = getattr(layer, "_slice_input", None)
        ndisplay = getattr(slice_input, "ndisplay", None)
        if isinstance(ndisplay, (int, np.integer)):
            return int(ndisplay)
        return None

    def _heatmap_requires_stable_limits(self, layer, response=None) -> bool:
        return self._heatmap_ndisplay(layer, response) == 3

    @staticmethod
    def _heatmap_stored_contrast_limits(layer) -> tuple[float, float] | None:
        raw_limits = getattr(
            layer,
            "_napari_swc_flatmap_heatmap_contrast_limits",
            None,
        )
        if raw_limits is None:
            metadata = getattr(layer, "metadata", None)
            if isinstance(metadata, dict):
                raw_limits = metadata.get("flatmap_heatmap_contrast_limits")
        if raw_limits is None:
            return None

        try:
            values = np.asarray(raw_limits, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if values.size < 2:
            return None

        lower = float(values[0])
        upper = float(values[1])
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            return None
        return (lower, upper)

    @staticmethod
    def _store_heatmap_contrast_limits(layer, limits: tuple[float, float]) -> None:
        setattr(layer, "_napari_swc_flatmap_heatmap_contrast_limits", limits)
        metadata = getattr(layer, "metadata", None)
        if isinstance(metadata, dict):
            metadata["flatmap_heatmap_contrast_limits"] = limits

    @staticmethod
    def _apply_heatmap_contrast_limits(layer, limits: tuple[float, float]) -> None:
        keep_auto = bool(getattr(layer, "_keep_auto_contrast", False))
        if keep_auto:
            layer._keep_auto_contrast = False
        try:
            layer.contrast_limits_range = limits
            layer.contrast_limits = limits
        finally:
            if hasattr(layer, "_keep_auto_contrast"):
                layer._keep_auto_contrast = keep_auto

    @staticmethod
    def _is_stale_3d_status_slice(layer, dims_displayed, exc: IndexError) -> bool:
        if dims_displayed is None or len(dims_displayed) != 3:
            return False

        raw = getattr(
            getattr(getattr(layer, "_slice", None), "image", None),
            "raw",
            None,
        )
        if raw is not None and np.asarray(raw).ndim < len(dims_displayed):
            return True

        return "too many indices for array" in str(exc)

    @staticmethod
    def _status_without_sampled_value(layer, position) -> dict[str, str]:
        source_info = getattr(layer, "_get_source_info", None)
        if callable(source_info):
            status = source_info().copy()
        else:
            name = str(getattr(layer, "name", ""))
            status = {
                "layer_name": name,
                "layer_base": name,
                "source_type": "",
                "plugin": "",
            }

        coords_str = ""
        if position is not None:
            ndim = int(getattr(layer, "ndim", 0) or 0)
            coords = np.asarray(position)
            if ndim > 0:
                coords = coords[-ndim:]
            rounded = np.round(coords).astype(int)
            coords_str = f" [{' '.join(map(str, rounded))}]"

        status["coordinates"] = ": ".join((coords_str, ""))
        status["coords"] = coords_str
        status["value"] = ""
        return status

    def _create_or_update_points_layer(
        self,
        layer,
        render_result: FlatmapRenderResult,
        metadata: dict[str, object],
    ):
        points = render_result.points
        colors = self._colors_for_file_ids(render_result.point_file_ids)
        if layer is None:
            return self._display_viewer().add_points(
                points,
                name=_POINTS_LAYER_NAME,
                size=2.0,
                face_color=colors,
                border_width=0.0,
                blending="translucent",
                metadata=metadata,
            )

        blocker = getattr(getattr(layer, "events", None), "blocker_all", None)
        if callable(blocker):
            with blocker():
                layer.data = points
                layer.face_color = colors
                layer.metadata = metadata
                layer.visible = True
        else:
            layer.data = points
            layer.face_color = colors
            layer.metadata = metadata
            layer.visible = True
        refresh = getattr(layer, "refresh", None)
        if callable(refresh):
            refresh()
        return layer

    def _focus_projection_view(self, layer, data: np.ndarray) -> None:
        """Switch to 3D and center the camera on the flatmap render bounds."""
        try:
            layer.visible = True
        except Exception:
            pass

        viewer = self._display_viewer()
        dims = getattr(viewer, "dims", None)
        if dims is not None and getattr(dims, "ndisplay", None) != 3:
            try:
                dims.ndisplay = 3
            except Exception:
                logger.debug("Failed to switch viewer to 3D display.", exc_info=True)
        self._reslice_layer_for_current_dims(layer)

        layers = getattr(viewer, "layers", None)
        selection = getattr(layers, "selection", None)
        if selection is not None:
            try:
                selection.active = layer
            except Exception:
                logger.debug("Failed to activate flatmap layer.", exc_info=True)

        array = np.asarray(data, dtype=float)
        if array.ndim == 3:
            coords = np.argwhere(array > 0)
            if len(coords) == 0:
                lower = np.zeros(3, dtype=float)
                upper = np.asarray(array.shape, dtype=float) - 1.0
            else:
                lower = np.min(coords, axis=0).astype(float)
                upper = np.max(coords, axis=0).astype(float)
        else:
            coords = array.reshape(-1, 3)
            finite_mask = np.all(np.isfinite(coords), axis=1)
            if not finite_mask.any():
                return
            finite = coords[finite_mask]
            lower = np.min(finite, axis=0)
            upper = np.max(finite, axis=0)
        center = tuple(((lower + upper) / 2.0).tolist())
        span = float(np.max(upper - lower))

        camera = getattr(viewer, "camera", None)
        if camera is None:
            reset_view = getattr(viewer, "reset_view", None)
            if callable(reset_view):
                reset_view()
            return

        try:
            camera.center = center
        except Exception:
            logger.debug("Failed to center camera on flatmap layer.", exc_info=True)

        if span > 0.0 and np.isfinite(span):
            try:
                camera.zoom = float(np.clip(600.0 / span, 0.01, 10_000.0))
            except Exception:
                logger.debug("Failed to zoom camera to flatmap layer.", exc_info=True)

    def _reslice_layer_for_current_dims(self, layer) -> None:
        viewer = self._current_display_viewer()
        dims = getattr(viewer, "dims", None)
        if dims is None:
            return
        slice_dims = getattr(layer, "_slice_dims", None)
        if not callable(slice_dims):
            return
        try:
            slice_dims(dims, force=True)
        except Exception:
            logger.debug("Failed to refresh flatmap layer slice.", exc_info=True)

    def _export_csv(self) -> None:
        if self._last_projected_nodes is None or self._last_projected_nodes.empty:
            show_warning("Run a flatmap projection before exporting CSV.")
            return

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Flatmap Projection CSV",
            "flatmap_projection.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if output_path:
            self._export_current_projection_to_path(output_path)

    def _export_current_projection_to_path(self, output_path: str | Path) -> Path:
        """Export the current projected node table to a specific CSV path."""
        if self._last_projected_nodes is None or self._last_projected_nodes.empty:
            raise RuntimeError("Run a flatmap projection before exporting CSV.")
        saved = export_projected_nodes_csv(self._last_projected_nodes, output_path)
        self._status_label.setText(f"Exported flatmap projection to {saved}.")
        show_info(f"Exported flatmap projection to {saved}")
        return saved

    def _current_source_parquet_path(self) -> Path:
        db = self._database_provider()
        if db is None:
            raise RuntimeError("Load a neuron Parquet before saving augmented Parquet.")
        parquet_path = getattr(db, "parquet_path", None)
        if parquet_path is None:
            raise RuntimeError("Loaded neuron database does not expose a Parquet path.")
        return Path(parquet_path)

    def _augment_parquet(self) -> None:
        try:
            source_path = self._current_source_parquet_path()
            lookup_dir = getattr(self, "_preprocess_lookup_dir", None)
            if lookup_dir is None:
                raise RuntimeError(
                    "Choose a lookup directory containing bilateral shaped, "
                    "bilateral square, and depth NRRDs first."
                )
        except Exception as exc:
            show_warning(str(exc))
            return

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Prepare Whole Flatmap Parquet",
            str(source_path.with_name(f"{source_path.stem}_flatmap.parquet")),
            "Parquet Files (*.parquet);;All Files (*)",
        )
        if not output_path:
            return

        output = Path(output_path)
        if output.resolve() == source_path.resolve():
            from qtpy.QtWidgets import QMessageBox

            answer = QMessageBox.question(
                self,
                "Replace Source Parquet?",
                "This will atomically replace the loaded source Parquet after "
                "all rows are prepared. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self._start_parquet_preparation(source_path, output, lookup_dir)

    def _start_parquet_preparation(
        self,
        source_path: Path,
        output_path: Path,
        lookup_dir: Path,
    ) -> None:
        """Run whole-file bilateral preprocessing in a cancellable QThread."""
        from qtpy.QtCore import QThread

        from ..workers import FlatmapParquetPreparationWorker

        resolution_control = getattr(self, "_lookup_resolution_spin", None)
        raw_resolution = (
            int(resolution_control.value()) if resolution_control is not None else 0
        )
        lookup_resolution_um = float(raw_resolution) if raw_resolution > 0 else None
        thread = QThread()
        worker = FlatmapParquetPreparationWorker(
            source_path,
            output_path,
            lookup_dir,
            lookup_resolution_um=lookup_resolution_um,
        )
        self._augment_thread = thread
        self._augment_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._set_projection_progress)
        worker.finished.connect(self._on_parquet_preparation_finished)
        worker.error.connect(self._on_parquet_preparation_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda: self._cleanup_parquet_preparation(thread, worker)
        )
        self._augment_parquet_btn.setEnabled(False)
        self._cancel_augment_btn.setEnabled(True)
        self._set_projection_progress("Preparing whole Parquet...", 0, 0)
        thread.start()

    def _cancel_parquet_preparation(self) -> None:
        worker = getattr(self, "_augment_worker", None)
        cancel = getattr(worker, "cancel", None)
        if callable(cancel):
            cancel()
            self._status_label.setText("Cancelling Parquet preparation...")
            self._cancel_augment_btn.setEnabled(False)

    def _on_parquet_preparation_finished(self, summary) -> None:
        self._hide_projection_progress()
        message = (
            f"Prepared {getattr(summary, 'rows', 0):,} row(s) in "
            f"{getattr(summary, 'output_parquet', '')}."
        )
        self._status_label.setText(message)
        show_info(message)

    def _on_parquet_preparation_error(self, message: str) -> None:
        self._hide_projection_progress()
        self._status_label.setText(f"Flatmap Parquet preparation failed: {message}")
        show_warning(f"Flatmap Parquet preparation failed: {message}")

    def _cleanup_parquet_preparation(self, thread, worker) -> None:
        if getattr(self, "_augment_thread", None) is thread:
            self._augment_thread = None
        if getattr(self, "_augment_worker", None) is worker:
            self._augment_worker = None
        self._augment_parquet_btn.setEnabled(True)
        self._cancel_augment_btn.setEnabled(False)

    def _augment_current_parquet_to_path(self, output_path: str | Path):
        """Save a Parquet file augmented with NRRD-derived flatmap columns."""
        self._projection_request_ready()
        source_path = self._current_source_parquet_path()
        file_ids = self._file_ids_for_source()
        if not file_ids:
            raise RuntimeError("No neurons are available to save.")
        summary = augment_neuron_parquet_with_flatmap(
            source_path,
            output_path,
            self._flatmap_path,
            self._depth_path,
            file_ids=file_ids,
            flatmap_style=self._current_style_filename(),
            coordinate_mode=self._current_coordinate_mode(),
            invalid_zero_sentinel=self._zero_sentinel_cb.isChecked(),
            invalid_negative_one_sentinel=self._negative_one_sentinel_cb.isChecked(),
        )
        self._status_label.setText(
            "Saved augmented Parquet to "
            f"{summary.output_parquet} "
            f"({summary.rows:,} rows from {len(file_ids):,} file ID(s); "
            f"{summary.direct_rows:,} direct, "
            f"{getattr(summary, 'mirrored_depth_rows', 0):,} mirrored-depth, "
            f"{summary.mirrored_rows:,} mirrored, "
            f"{summary.unmapped_rows:,} unmapped)."
        )
        show_info(f"Saved augmented Parquet to {summary.output_parquet}")
        return summary
