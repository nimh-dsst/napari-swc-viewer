"""Analysis tab widget for the clustering pipeline.

Provides UI for:
1. Region mask selection and dilation parameters
2. Correlation matrix computation with hierarchical clustering
3. Clustermap visualization (embedded matplotlib canvas)
4. Node count heatmap generation
5. Applying neuron colors by cluster assignment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from types import MethodType
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import QThread, Signal
from qtpy.QtGui import QColor, QIcon, QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .collapsible_section import CollapsibleSection
from .node_type_selector import NodeTypeSelectorComboBox
from .region_selector import RegionSelectorWidget

if TYPE_CHECKING:
    import napari
    from brainglobe_atlasapi import BrainGlobeAtlas

    from ..analysis.clustering import ClusterRegionSelection, ClusterResult
    from ..db import NeuronDatabase

logger = logging.getLogger(__name__)

_ANALYSIS_SCOPE_WHOLE = "whole"
_ANALYSIS_SCOPE_CURRENT = "current"


@dataclass(frozen=True)
class _HeatmapRequest:
    """Immutable specification for one analysis heatmap build."""

    selected_region_id: int | None
    selected_region_acronym: str | None
    region_ids: tuple[int, ...] | None
    cluster_label: int | None
    file_ids: tuple[str, ...] | None
    node_types: tuple[int, ...] | None
    soma_radius_um: float | None
    depth_bin_factor: int
    depth_axis: int


@dataclass(frozen=True)
class _ClusterColorApplicationSummary:
    """Summary of cached cluster application to the table and rendered layers."""

    matched_table_count: int
    updated_layer_count: int
    rendered_count: int
    colored_count: int
    gray_count: int


def _analysis_heatmap_contrast_limits(
    volume: np.ndarray,
) -> tuple[float, float]:
    """Return stable full-volume contrast limits for an analysis heatmap."""
    if volume.size == 0:
        return (0.0, 1.0)

    upper = float(np.nanmax(volume))
    if not np.isfinite(upper) or upper <= 0.0:
        return (0.0, 1.0)
    return (0.0, upper)


def _populate_embedded_clustermap_figure(
    figure: Figure,
    result: ClusterResult,
    cluster_color_map: dict[str, list[float]] | None = None,
    *,
    figsize: tuple[float, float] = (6.0, 6.0),
    dpi: int | None = None,
) -> Figure:
    """Populate the persistent dendrogram preview figure used by the tab."""
    from ..analysis.export import populate_clustermap_figure

    return populate_clustermap_figure(
        figure,
        result,
        cluster_color_map,
        figsize=figsize,
        dpi=dpi,
    )


def _analysis_heatmap_stored_limits(
    layer: Any,
) -> tuple[float, float] | None:
    """Return stored contrast limits from analysis heatmap metadata."""
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        return None

    raw_limits = metadata.get("heatmap_contrast_limits")
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


def _apply_analysis_heatmap_contrast_limits(
    layer: Any,
    limits: tuple[float, float],
) -> None:
    """Apply stored full-volume contrast limits to an analysis heatmap layer."""
    keep_auto = bool(getattr(layer, "_keep_auto_contrast", False))
    if keep_auto:
        layer._keep_auto_contrast = False
    try:
        layer.contrast_limits_range = limits
        layer.contrast_limits = limits
    finally:
        layer._keep_auto_contrast = keep_auto


def _is_thumbnail_rank_mismatch_error(error: RuntimeError) -> bool:
    """Return whether napari thumbnail generation hit the known rank bug."""
    return "sequence argument must have length equal to input rank" in str(
        error
    )


def _analysis_heatmap_ndisplay(
    layer: Any, response: Any | None = None
) -> int | None:
    """Return the current display dimensionality for a heatmap layer."""
    slice_input = getattr(response, "slice_input", None)
    ndisplay = getattr(slice_input, "ndisplay", None)
    if isinstance(ndisplay, int):
        return ndisplay

    slice_input = getattr(layer, "_slice_input", None)
    ndisplay = getattr(slice_input, "ndisplay", None)
    if isinstance(ndisplay, int):
        return ndisplay
    return None


def _analysis_heatmap_requires_stable_limits(
    layer: Any, response: Any | None = None
) -> bool:
    """Return whether the 3D napari workaround should be active."""
    return _analysis_heatmap_ndisplay(layer, response) == 3


def _install_analysis_heatmap_layer_workarounds(layer: Any) -> None:
    """Install stable contrast behavior on one analysis heatmap layer."""
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        return
    if metadata.get("heatmap_kind") != "analysis":
        return
    if getattr(layer, "_analysis_heatmap_workarounds_installed", False):
        return

    original_update_thumbnail = getattr(layer, "_update_thumbnail", None)
    if callable(original_update_thumbnail):

        def _safe_update_thumbnail(self) -> None:
            try:
                original_update_thumbnail()
            except RuntimeError as error:
                if not _is_thumbnail_rank_mismatch_error(error):
                    raise
                if not getattr(
                    self, "_analysis_heatmap_thumbnail_warning_logged", False
                ):
                    logger.warning(
                        "Suppressed napari thumbnail update failure for "
                        "analysis heatmap '%s': %s",
                        getattr(self, "name", "<unnamed>"),
                        error,
                    )
                    self._analysis_heatmap_thumbnail_warning_logged = True

        layer._update_thumbnail = MethodType(_safe_update_thumbnail, layer)

    original_reset_contrast_limits = getattr(
        layer, "reset_contrast_limits", None
    )
    if callable(original_reset_contrast_limits):

        def _stable_reset_contrast_limits(self, mode=None) -> None:
            if not _analysis_heatmap_requires_stable_limits(self):
                original_reset_contrast_limits(mode)
                return
            limits = _analysis_heatmap_stored_limits(self)
            if limits is None:
                original_reset_contrast_limits(mode)
                return
            _apply_analysis_heatmap_contrast_limits(self, limits)

        layer.reset_contrast_limits = MethodType(
            _stable_reset_contrast_limits, layer
        )

    original_reset_contrast_limits_range = getattr(
        layer, "reset_contrast_limits_range", None
    )
    if callable(original_reset_contrast_limits_range):

        def _stable_reset_contrast_limits_range(self, mode=None) -> None:
            if not _analysis_heatmap_requires_stable_limits(self):
                original_reset_contrast_limits_range(mode)
                return
            limits = _analysis_heatmap_stored_limits(self)
            if limits is None:
                original_reset_contrast_limits_range(mode)
                return
            self.contrast_limits_range = limits

        layer.reset_contrast_limits_range = MethodType(
            _stable_reset_contrast_limits_range, layer
        )

    original_update_slice_response = getattr(
        layer, "_update_slice_response", None
    )
    if callable(original_update_slice_response):

        def _stable_update_slice_response(self, response) -> Any:
            keep_auto = bool(getattr(self, "_keep_auto_contrast", False))
            if not keep_auto or not _analysis_heatmap_requires_stable_limits(
                self, response
            ):
                return original_update_slice_response(response)

            self._keep_auto_contrast = False
            try:
                result = original_update_slice_response(response)
            finally:
                self._keep_auto_contrast = True

            limits = _analysis_heatmap_stored_limits(self)
            if limits is not None:
                _apply_analysis_heatmap_contrast_limits(self, limits)
            return result

        layer._update_slice_response = MethodType(
            _stable_update_slice_response, layer
        )

    layer._analysis_heatmap_workarounds_installed = True


class AnalysisTabWidget(QWidget):
    """Widget providing the Analysis tab UI.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    parent : QWidget, optional
        Parent widget.
    """

    cluster_colors_updated = Signal(object, dict)

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None):
        super().__init__(parent)
        self._viewer = viewer
        self._db: NeuronDatabase | None = None
        self._atlas: BrainGlobeAtlas | None = None
        self._parquet_path: str | None = None
        self._worker_thread: QThread | None = None
        self._current_worker = None
        self._last_cluster_result: ClusterResult | None = None
        self._cluster_color_map: dict[str, list[float]] | None = None
        self._actual_n_clusters: int = 0
        self._heatmap_layer = None
        self._current_heatmap_request: _HeatmapRequest | None = None
        self._pending_heatmap_requests: list[_HeatmapRequest] = []
        self._completed_heatmap_requests: list[_HeatmapRequest] = []
        self._last_heatmap_requests: list[_HeatmapRequest] = []
        self._active_heatmap_total: int = 0
        self._active_heatmap_index: int = 0
        self._heatmap_batch_mode: bool = False
        self._slice_projector = None
        self._dataset_region_ids: set[int] = set()
        self._clustermap_rendered = False
        self._cluster_region_query_scope = _ANALYSIS_SCOPE_WHOLE
        self._current_table_file_ids_provider = None
        self._setup_ui()

        # Rebuild heatmap when the user reorders axes in napari
        self._viewer.dims.events.order.connect(self._on_dims_order_changed)

    def set_database(self, db: NeuronDatabase) -> None:
        """Set the database connection."""
        self._db = db
        self._parquet_path = str(db.parquet_path)
        self.refresh_available_regions_from_database()
        self._update_button_states()

    def refresh_available_regions_from_database(self) -> None:
        """Cache dataset region IDs and refresh analysis selectors."""
        if self._db is None:
            self._dataset_region_ids = set()
            self._refresh_analysis_region_selectors()
            return

        regions_df = self._db.get_unique_regions()
        normalized_ids: set[int] = set()
        region_values = (
            regions_df["region_id"] if "region_id" in regions_df else []
        )
        for region_id in list(region_values):
            if region_id is None:
                continue
            if isinstance(region_id, (float, np.floating)) and np.isnan(
                region_id
            ):
                continue

            try:
                value = int(region_id)
            except (TypeError, ValueError):
                continue
            if value > 0:
                normalized_ids.add(value)

        self._dataset_region_ids = normalized_ids
        self._refresh_analysis_region_selectors()

    def set_atlas(self, atlas: BrainGlobeAtlas) -> None:
        """Set the atlas instance."""
        self._atlas = atlas
        self._refresh_analysis_region_selectors()
        self._update_button_states()
        self._update_voxel_depth_label()

    def set_slice_projector(self, projector) -> None:
        """Set the slice projector for updating 2D projection colors."""
        self._slice_projector = projector

    def set_current_table_file_ids_provider(self, provider) -> None:
        """Set a callback returning the current neuron-table file IDs."""
        self._current_table_file_ids_provider = provider

    def _raw_current_table_file_ids(self) -> list[object]:
        """Return current neuron-table file IDs from the configured provider."""
        provider = self.__dict__.get("_current_table_file_ids_provider")
        if not callable(provider):
            return []
        try:
            file_ids = provider()
        except Exception:
            return []
        if file_ids is None:
            return []
        return list(file_ids)

    def _matched_current_table_file_ids(self) -> list[object]:
        """Return current table file IDs that exist in the cached cluster result."""
        result = getattr(self, "_last_cluster_result", None)
        if result is None:
            return []

        current_file_ids = self._raw_current_table_file_ids()
        if not current_file_ids:
            return []

        result_ids = list(getattr(result, "neuron_ids", []))
        if not result_ids:
            return []

        result_id_set = set(result_ids)
        result_id_strs = {str(file_id) for file_id in result_ids}
        return [
            file_id
            for file_id in current_file_ids
            if file_id in result_id_set or str(file_id) in result_id_strs
        ]

    def has_cached_clusters_for_current_table(self) -> bool:
        """Return whether cached clustering overlaps the current table."""
        if self._cluster_color_map is None or self._last_cluster_result is None:
            return False
        return bool(self._matched_current_table_file_ids())

    def _update_button_states(self) -> None:
        """Enable/disable buttons based on loaded data."""
        ready = self._db is not None and self._atlas is not None
        busy = (
            self._worker_thread is not None and self._worker_thread.isRunning()
        )
        self._run_corr_btn.setEnabled(ready and not busy)
        self._run_heat_btn.setEnabled(ready and not busy)
        has_cluster_heatmap_options = (
            ready
            and not busy
            and self._last_cluster_result is not None
            and getattr(self, "_heat_cluster_combo", None) is not None
            and self._heat_cluster_combo.count() > 1
        )
        if hasattr(self, "_add_all_cluster_heatmaps_btn"):
            self._add_all_cluster_heatmaps_btn.setEnabled(
                has_cluster_heatmap_options
            )
        analysis_ready = self._last_cluster_result is not None and not busy
        if hasattr(self, "_render_clustermap_btn"):
            self._render_clustermap_btn.setEnabled(analysis_ready)
        if hasattr(self, "_build_clustermap_btn"):
            self._build_clustermap_btn.setEnabled(analysis_ready)
        if hasattr(self, "_export_title_edit"):
            self._export_title_edit.setEnabled(analysis_ready)
            self._export_xlabel_edit.setEnabled(analysis_ready)
            self._export_dpi_combo.setEnabled(analysis_ready)
            self._save_cluster_workbook_btn.setEnabled(analysis_ready)
            self._save_distance_workbook_btn.setEnabled(analysis_ready)
            self._save_extended_parquet_btn.setEnabled(analysis_ready)
            self._save_dendrogram_btn.setEnabled(analysis_ready)

    def _on_thread_finished(self) -> None:
        """Clear worker references after the thread has stopped."""
        self._worker_thread = None
        self._current_worker = None

    def _setup_ui(self) -> None:
        outer_layout = QVBoxLayout(self)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        outer_layout.addWidget(self._scroll_area)

        self._scroll_content = QWidget()
        self._scroll_area.setWidget(self._scroll_content)

        layout = QVBoxLayout(self._scroll_content)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- Clustering group ---
        self._clustering_section = CollapsibleSection(
            "Clustering",
            expanded=True,
        )
        corr_layout = self._clustering_section.content_layout()

        # Clustering method
        method_type_row = QHBoxLayout()
        method_type_row.addWidget(QLabel("Method:"))
        self._clustering_method_combo = QComboBox()
        self._clustering_method_combo.addItems(
            ["Voxel Correlation", "Soma Location"]
        )
        self._clustering_method_combo.currentTextChanged.connect(
            self._on_clustering_method_changed
        )
        method_type_row.addWidget(self._clustering_method_combo)
        corr_layout.addLayout(method_type_row)

        # Algorithm (only for Soma Location)
        self._algorithm_row = QHBoxLayout()
        self._algorithm_label = QLabel("Algorithm:")
        self._algorithm_row.addWidget(self._algorithm_label)
        self._algorithm_combo = QComboBox()
        self._algorithm_combo.addItems(["Hierarchical", "K-Means", "DBSCAN"])
        self._algorithm_combo.currentTextChanged.connect(
            self._on_algorithm_changed
        )
        self._algorithm_row.addWidget(self._algorithm_combo)
        corr_layout.addLayout(self._algorithm_row)

        # Search scope
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Search scope:"))
        self._cluster_region_scope_combo = QComboBox()
        self._cluster_region_scope_combo.addItem(
            "Whole Parquet",
            _ANALYSIS_SCOPE_WHOLE,
        )
        self._cluster_region_scope_combo.addItem(
            "Current Table",
            _ANALYSIS_SCOPE_CURRENT,
        )
        self._cluster_region_scope_combo.currentTextChanged.connect(
            self._on_cluster_region_scope_changed
        )
        scope_row.addWidget(self._cluster_region_scope_combo)
        corr_layout.addLayout(scope_row)

        # Target region
        region_row = QHBoxLayout()
        region_row.addWidget(QLabel("Target region:"))
        self._cluster_region_summary_label = QLabel("None selected")
        region_row.addWidget(self._cluster_region_summary_label)
        corr_layout.addLayout(region_row)

        self._cluster_region_section = CollapsibleSection(
            "Select Target Region",
            expanded=False,
        )
        cluster_region_layout = self._cluster_region_section.content_layout()

        self._cluster_region_scope_stack = QStackedWidget()

        whole_page = QWidget()
        whole_layout = QVBoxLayout(whole_page)
        whole_layout.setContentsMargins(0, 0, 0, 0)
        self._whole_parquet_cluster_region_selector = RegionSelectorWidget(
            single_select=False,
            show_include_children=False,
            force_include_children=True,
        )
        self._whole_parquet_cluster_region_selector.selection_changed.connect(
            self._on_cluster_region_selection_changed
        )
        whole_layout.addWidget(self._whole_parquet_cluster_region_selector)
        self._cluster_region_scope_stack.addWidget(whole_page)

        current_page = QWidget()
        current_layout = QVBoxLayout(current_page)
        current_layout.setContentsMargins(0, 0, 0, 0)
        self._current_table_cluster_region_selector = RegionSelectorWidget(
            single_select=False,
            show_include_children=False,
            force_include_children=True,
        )
        self._current_table_cluster_region_selector.selection_changed.connect(
            self._on_cluster_region_selection_changed
        )
        current_layout.addWidget(self._current_table_cluster_region_selector)
        self._cluster_region_scope_stack.addWidget(current_page)

        cluster_region_layout.addWidget(self._cluster_region_scope_stack)
        corr_layout.addWidget(self._cluster_region_section)

        # Dilation fraction
        dilation_row = QHBoxLayout()
        dilation_row.addWidget(QLabel("Dilation %:"))
        self._dilation_spin = QSpinBox()
        self._dilation_spin.setRange(0, 100)
        self._dilation_spin.setValue(0)
        self._dilation_spin.setSuffix("%")
        dilation_row.addWidget(self._dilation_spin)
        corr_layout.addLayout(dilation_row)

        # Linkage method
        self._linkage_row = QHBoxLayout()
        self._linkage_label = QLabel("Linkage:")
        self._linkage_row.addWidget(self._linkage_label)
        self._method_combo = QComboBox()
        self._method_combo.addItems(["average", "ward", "complete", "single"])
        self._method_combo.setCurrentText("ward")
        self._linkage_row.addWidget(self._method_combo)
        corr_layout.addLayout(self._linkage_row)

        # Number of clusters
        self._clusters_row = QHBoxLayout()
        self._clusters_label = QLabel("Clusters:")
        self._clusters_row.addWidget(self._clusters_label)
        self._n_clusters_spin = QSpinBox()
        self._n_clusters_spin.setRange(2, 50)
        self._n_clusters_spin.setValue(5)
        self._clusters_row.addWidget(self._n_clusters_spin)
        corr_layout.addLayout(self._clusters_row)

        # DBSCAN eps
        self._eps_row = QHBoxLayout()
        self._eps_label = QLabel("Eps (μm):")
        self._eps_row.addWidget(self._eps_label)
        self._eps_spin = QDoubleSpinBox()
        self._eps_spin.setRange(1.0, 10000.0)
        self._eps_spin.setValue(100.0)
        self._eps_spin.setSuffix(" μm")
        self._eps_spin.setDecimals(1)
        self._eps_row.addWidget(self._eps_spin)
        corr_layout.addLayout(self._eps_row)

        # DBSCAN min_samples
        self._min_samples_row = QHBoxLayout()
        self._min_samples_label = QLabel("Min samples:")
        self._min_samples_row.addWidget(self._min_samples_label)
        self._min_samples_spin = QSpinBox()
        self._min_samples_spin.setRange(1, 100)
        self._min_samples_spin.setValue(5)
        self._min_samples_row.addWidget(self._min_samples_spin)
        corr_layout.addLayout(self._min_samples_row)

        # Run button
        self._run_corr_btn = QPushButton("Run Clustering")
        self._run_corr_btn.setEnabled(False)
        self._run_corr_btn.clicked.connect(self._run_clustering_pipeline)
        corr_layout.addWidget(self._run_corr_btn)

        layout.addWidget(self._clustering_section)

        # Set initial visibility
        self._on_clustering_method_changed(
            self._clustering_method_combo.currentText()
        )

        # --- Node Count Heatmap group ---
        self._heatmap_section = CollapsibleSection(
            "Node Count Heatmap",
            expanded=True,
        )
        heat_layout = self._heatmap_section.content_layout()

        heat_region_row = QHBoxLayout()
        heat_region_row.addWidget(QLabel("Region filter:"))
        self._heat_region_summary_label = QLabel("All regions")
        heat_region_row.addWidget(self._heat_region_summary_label)
        heat_layout.addLayout(heat_region_row)

        self._heat_region_section = CollapsibleSection(
            "Select Heatmap Region",
            expanded=False,
        )
        heat_region_selector_layout = (
            self._heat_region_section.content_layout()
        )
        self._heat_region_selector = RegionSelectorWidget(
            single_select=True,
            show_include_children=False,
            force_include_children=True,
        )
        self._heat_region_selector.selection_changed.connect(
            self._on_heat_region_selection_changed
        )
        heat_region_selector_layout.addWidget(self._heat_region_selector)
        heat_layout.addWidget(self._heat_region_section)

        # Cluster filter
        cluster_filter_row = QHBoxLayout()
        cluster_filter_row.addWidget(QLabel("Cluster filter:"))
        self._heat_cluster_combo = QComboBox()
        self._heat_cluster_combo.addItem("All neurons")
        self._heat_cluster_combo.setEnabled(False)
        cluster_filter_row.addWidget(self._heat_cluster_combo)
        heat_layout.addLayout(cluster_filter_row)

        node_type_row = QHBoxLayout()
        node_type_row.addWidget(QLabel("Node types:"))
        self._heat_node_type_combo = NodeTypeSelectorComboBox()
        node_type_row.addWidget(self._heat_node_type_combo)
        heat_layout.addLayout(node_type_row)

        soma_radius_row = QHBoxLayout()
        self._heat_soma_radius_enabled_cb = QCheckBox(
            "Filter by soma distance"
        )
        self._heat_soma_radius_enabled_cb.setChecked(False)
        self._heat_soma_radius_enabled_cb.setToolTip(
            "Restrict heatmap nodes to a radius around each neuron's soma"
        )
        soma_radius_row.addWidget(self._heat_soma_radius_enabled_cb)
        soma_radius_row.addWidget(QLabel("Soma radius:"))
        self._heat_soma_radius_spin = QDoubleSpinBox()
        self._heat_soma_radius_spin.setRange(0.0, 100000.0)
        self._heat_soma_radius_spin.setValue(0.0)
        self._heat_soma_radius_spin.setSuffix(" μm")
        self._heat_soma_radius_spin.setDecimals(1)
        self._heat_soma_radius_spin.setToolTip(
            "Maximum node distance from each neuron's soma when enabled"
        )
        self._heat_soma_radius_spin.setEnabled(False)
        self._heat_soma_radius_enabled_cb.toggled.connect(
            self._heat_soma_radius_spin.setEnabled
        )
        soma_radius_row.addWidget(self._heat_soma_radius_spin)
        heat_layout.addLayout(soma_radius_row)

        # Depth bin factor
        depth_bin_row = QHBoxLayout()
        depth_bin_row.addWidget(QLabel("Depth bin factor:"))
        self._depth_bin_spin = QSpinBox()
        self._depth_bin_spin.setRange(1, 20)
        self._depth_bin_spin.setValue(1)
        self._depth_bin_spin.setToolTip(
            "Merge N depth-planes into one voxel along the slicing axis"
        )
        self._depth_bin_spin.valueChanged.connect(
            self._update_voxel_depth_label
        )
        depth_bin_row.addWidget(self._depth_bin_spin)
        heat_layout.addLayout(depth_bin_row)

        self._voxel_depth_label = QLabel("Voxel depth: — μm")
        heat_layout.addWidget(self._voxel_depth_label)

        heat_action_row = QHBoxLayout()
        self._run_heat_btn = QPushButton("Build Heatmap Volume")
        self._run_heat_btn.setEnabled(False)
        self._run_heat_btn.clicked.connect(self._run_heatmap_pipeline)
        heat_action_row.addWidget(self._run_heat_btn)

        self._add_all_cluster_heatmaps_btn = QPushButton(
            "Add All Cluster Heatmaps"
        )
        self._add_all_cluster_heatmaps_btn.setEnabled(False)
        self._add_all_cluster_heatmaps_btn.clicked.connect(
            self._run_all_cluster_heatmaps
        )
        heat_action_row.addWidget(self._add_all_cluster_heatmaps_btn)
        heat_layout.addLayout(heat_action_row)

        layout.addWidget(self._heatmap_section)

        # --- Progress bar ---
        self._progress_section = CollapsibleSection(
            "Progress",
            expanded=True,
        )
        progress_layout = self._progress_section.content_layout()
        self._progress_label = QLabel("")
        progress_layout.addWidget(self._progress_label)
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        progress_layout.addWidget(self._progress_bar)
        layout.addWidget(self._progress_section)

        # --- Matplotlib canvas for clustermap ---
        self._clustermap_section = CollapsibleSection(
            "Clustermap",
            expanded=False,
        )
        clustermap_layout = self._clustermap_section.content_layout()
        self._clustermap_status_label = QLabel(
            "Run clustering, then click 'Build Dendrogram' to render the cluster map."
        )
        clustermap_layout.addWidget(self._clustermap_status_label)
        self._build_clustermap_btn = QPushButton("Build Dendrogram")
        self._build_clustermap_btn.clicked.connect(
            self._render_clustermap_requested
        )
        clustermap_layout.addWidget(self._build_clustermap_btn)
        self._render_clustermap_btn = self._build_clustermap_btn
        self._figure = Figure(figsize=(6, 6))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setMinimumHeight(400)
        clustermap_layout.addWidget(self._canvas)
        layout.addWidget(self._clustermap_section)

        self._export_section = CollapsibleSection(
            "Export Results",
            expanded=False,
        )
        export_layout = self._export_section.content_layout()

        export_title_row = QHBoxLayout()
        export_title_row.addWidget(QLabel("Title:"))
        self._export_title_edit = QLineEdit()
        export_title_row.addWidget(self._export_title_edit)
        export_layout.addLayout(export_title_row)

        export_xlabel_row = QHBoxLayout()
        export_xlabel_row.addWidget(QLabel("X label:"))
        self._export_xlabel_edit = QLineEdit()
        export_xlabel_row.addWidget(self._export_xlabel_edit)
        export_layout.addLayout(export_xlabel_row)

        export_dpi_row = QHBoxLayout()
        export_dpi_row.addWidget(QLabel("DPI:"))
        self._export_dpi_combo = QComboBox()
        self._export_dpi_combo.addItem("150 dpi")
        self._export_dpi_combo.setItemData(0, 150)
        self._export_dpi_combo.addItem("300 dpi")
        self._export_dpi_combo.setItemData(1, 300)
        self._export_dpi_combo.addItem("600 dpi")
        self._export_dpi_combo.setItemData(2, 600)
        self._export_dpi_combo.setCurrentIndex(1)
        export_dpi_row.addWidget(self._export_dpi_combo)
        export_layout.addLayout(export_dpi_row)

        self._save_cluster_workbook_btn = QPushButton("Save Cluster Workbook")
        self._save_cluster_workbook_btn.clicked.connect(
            self._save_cluster_workbook
        )
        export_layout.addWidget(self._save_cluster_workbook_btn)

        self._save_distance_workbook_btn = QPushButton(
            "Save Distance Workbook"
        )
        self._save_distance_workbook_btn.clicked.connect(
            self._save_distance_workbook
        )
        export_layout.addWidget(self._save_distance_workbook_btn)

        self._save_extended_parquet_btn = QPushButton("Save Extended Parquet")
        self._save_extended_parquet_btn.clicked.connect(
            self._save_extended_parquet
        )
        export_layout.addWidget(self._save_extended_parquet_btn)

        self._save_dendrogram_btn = QPushButton("Save Dendrogram")
        self._save_dendrogram_btn.clicked.connect(self._save_dendrogram)
        export_layout.addWidget(self._save_dendrogram_btn)

        layout.addWidget(self._export_section)

        layout.addStretch()
        self._show_clustermap_message(
            "Run clustering, then click Render Dendrogram."
        )
        self._sync_cluster_region_scope_selector()
        self._refresh_analysis_region_selectors()
        self._update_button_states()

    def _format_selected_region_text(
        self,
        selector: RegionSelectorWidget | None,
        *,
        empty_text: str,
    ) -> str:
        """Return a compact summary string for a selector's current region(s)."""
        regions = self._selected_regions(selector)
        if not regions:
            return empty_text

        labels: list[str] = []
        for struct_id, acronym in regions[:2]:
            struct = selector._structure_map.get(struct_id, {})
            name = str(struct.get("name", "")).strip()
            if name and name != acronym:
                labels.append(f"{acronym} ({name})")
            else:
                labels.append(acronym)

        if len(regions) <= 2:
            return ", ".join(labels)
        return f"{', '.join(labels)} +{len(regions) - 2} more"

    def _update_region_summary_labels(self) -> None:
        """Refresh the visible region summary labels for Analysis workflows."""
        self._cluster_region_summary_label.setText(
            self._format_selected_region_text(
                self._active_cluster_region_selector(),
                empty_text="None selected",
            )
        )
        self._heat_region_summary_label.setText(
            self._format_selected_region_text(
                getattr(self, "_heat_region_selector", None),
                empty_text="All regions",
            )
        )

    def _analysis_allowed_structure_ids(self) -> set[int]:
        """Return dataset-backed visible structure IDs for Analysis selectors."""
        if self._atlas is None or not self._dataset_region_ids:
            return set()

        allowed_ids: set[int] = set()
        for region_id in self._dataset_region_ids:
            struct = self._atlas.structures.get(int(region_id))
            if struct is None:
                continue
            allowed_ids.add(int(region_id))
            for path_id in struct.get("structure_id_path", []) or []:
                try:
                    allowed_ids.add(int(path_id))
                except (TypeError, ValueError):
                    continue
        return allowed_ids

    def _refresh_analysis_region_selectors(self) -> None:
        """Rebuild Analysis region selectors from atlas hierarchy and dataset IDs."""
        heat_selector = getattr(self, "_heat_region_selector", None)
        cluster_selectors = self._cluster_region_selectors()
        if not cluster_selectors or heat_selector is None:
            return

        selectors = (*cluster_selectors, heat_selector)
        previous_regions = [
            self._selected_regions(selector) for selector in selectors
        ]

        if self._atlas is None or not self._dataset_region_ids:
            for selector in selectors:
                selector.clear()
            self._update_region_summary_labels()
            return

        allowed_ids = self._analysis_allowed_structure_ids()
        for selector, prior_regions in zip(selectors, previous_regions):
            selector.set_allowed_structure_ids(allowed_ids)
            selector.set_atlas(self._atlas)
            retained = [
                (region_id, acronym)
                for region_id, acronym in prior_regions
                if region_id in allowed_ids
            ]
            if len(retained) > 1 and hasattr(selector, "select_regions"):
                selector.select_regions(
                    [acronym for _region_id, acronym in retained]
                )
            else:
                previous_id = retained[0][0] if retained else None
                selector.select_region_by_id(previous_id)

        self._update_region_summary_labels()

    def _selected_cluster_region_scope(self) -> str:
        """Return the active clustering scope."""
        combo = getattr(self, "_cluster_region_scope_combo", None)
        if combo is None:
            scope = getattr(
                self,
                "_cluster_region_query_scope",
                _ANALYSIS_SCOPE_WHOLE,
            )
            if scope in {_ANALYSIS_SCOPE_WHOLE, _ANALYSIS_SCOPE_CURRENT}:
                return scope
            return _ANALYSIS_SCOPE_WHOLE

        current_data = getattr(combo, "currentData", None)
        if callable(current_data):
            data = current_data()
            if data in {_ANALYSIS_SCOPE_WHOLE, _ANALYSIS_SCOPE_CURRENT}:
                return str(data)

        current_text = getattr(combo, "currentText", None)
        text = current_text() if callable(current_text) else ""
        if text == "Current Table":
            return _ANALYSIS_SCOPE_CURRENT
        return _ANALYSIS_SCOPE_WHOLE

    def _cluster_region_selectors(self) -> tuple[RegionSelectorWidget, ...]:
        """Return all clustering region selectors, de-duplicated."""
        selectors = []
        for attr_name in (
            "_whole_parquet_cluster_region_selector",
            "_current_table_cluster_region_selector",
            "_cluster_region_selector",
        ):
            selector = getattr(self, attr_name, None)
            if selector is None or selector in selectors:
                continue
            selectors.append(selector)
        return tuple(selectors)

    def _cluster_region_selector_for_scope(
        self,
        scope: str | None = None,
    ) -> RegionSelectorWidget | None:
        """Return the clustering region selector for one scope."""
        selected_scope = scope
        if selected_scope not in {
            _ANALYSIS_SCOPE_WHOLE,
            _ANALYSIS_SCOPE_CURRENT,
        }:
            selected_scope = self._selected_cluster_region_scope()

        if selected_scope == _ANALYSIS_SCOPE_CURRENT:
            selector = getattr(self, "_current_table_cluster_region_selector", None)
            if selector is not None:
                return selector
        else:
            selector = getattr(self, "_whole_parquet_cluster_region_selector", None)
            if selector is not None:
                return selector

        return getattr(self, "_cluster_region_selector", None)

    def _active_cluster_region_selector(self) -> RegionSelectorWidget | None:
        """Return the currently visible clustering selector."""
        return self._cluster_region_selector_for_scope()

    def _sync_cluster_region_scope_selector(self) -> None:
        """Show the clustering selector matching the active scope."""
        stack = getattr(self, "_cluster_region_scope_stack", None)
        if stack is None:
            return

        index = (
            1
            if self._selected_cluster_region_scope() == _ANALYSIS_SCOPE_CURRENT
            else 0
        )
        stack.setCurrentIndex(index)

    def _current_table_file_ids(self) -> list[str]:
        """Return file IDs currently present in the main neuron table."""
        return [str(file_id) for file_id in self._raw_current_table_file_ids()]

    def _resolve_cluster_query_file_scope(
        self,
    ) -> tuple[bool, list[str] | None, str, int | None]:
        """Resolve the optional current-table restriction for clustering."""
        scope = self._selected_cluster_region_scope()
        if scope != _ANALYSIS_SCOPE_CURRENT:
            return True, None, "whole parquet", None

        file_ids = self._current_table_file_ids()
        if file_ids:
            return True, file_ids, "current table", len(file_ids)

        self._progress_label.setText(
            "Current table is empty; switch clustering scope to Whole Parquet or populate the table first."
        )
        return False, None, "current table", 0

    def _selected_regions(
        self,
        selector: RegionSelectorWidget | None,
    ) -> list[tuple[int, str]]:
        """Return the directly selected regions for a selector."""
        if selector is None:
            return []

        if hasattr(selector, "get_selected_ids"):
            selected_ids = selector.get_selected_ids(include_children=False)
            regions: list[tuple[int, str]] = []
            for struct_id in selected_ids:
                struct = selector._structure_map.get(int(struct_id), {})
                acronym = str(struct.get("acronym", "")).strip()
                if acronym:
                    regions.append((int(struct_id), acronym))
            if regions:
                return regions

        selected = selector.get_single_selected_region()
        return [selected] if selected is not None else []

    def _selected_region(
        self, selector: RegionSelectorWidget | None
    ) -> tuple[int, str] | None:
        """Return the first directly selected region for a selector, if any."""
        regions = self._selected_regions(selector)
        return regions[0] if regions else None

    def _selected_cluster_region(self) -> tuple[int, str] | None:
        """Return the first currently selected clustering region."""
        return self._selected_region(self._active_cluster_region_selector())

    def _selected_cluster_regions(self) -> list[tuple[int, str]]:
        """Return all currently selected clustering regions."""
        return self._selected_regions(self._active_cluster_region_selector())

    def _selected_heat_region(self) -> tuple[int, str] | None:
        """Return the currently selected heatmap region."""
        return self._selected_region(
            getattr(self, "_heat_region_selector", None)
        )

    def _represented_region_ids_for_selection(
        self, region_id: int
    ) -> list[int]:
        """Return represented dataset region IDs inside a selected atlas region."""
        if self._atlas is None or not self._dataset_region_ids:
            return []

        selected_region_id = int(region_id)
        represented_ids: list[int] = []
        for candidate_id in sorted(self._dataset_region_ids):
            struct = self._atlas.structures.get(int(candidate_id))
            if struct is None:
                continue
            try:
                path_ids = [
                    int(path_id)
                    for path_id in struct.get("structure_id_path", []) or []
                ]
            except (TypeError, ValueError):
                continue
            if selected_region_id in path_ids:
                represented_ids.append(int(candidate_id))
        return represented_ids

    def _represented_region_entries_for_selections(
        self,
        region_ids: list[int],
    ) -> list[tuple[int, str]]:
        """Return represented dataset region IDs/acronyms for selected regions."""
        represented: list[tuple[int, str]] = []
        seen_ids: set[int] = set()
        if self._atlas is None:
            return represented

        for region_id in region_ids:
            for represented_id in self._represented_region_ids_for_selection(
                region_id
            ):
                if represented_id in seen_ids:
                    continue
                struct = self._atlas.structures.get(int(represented_id), {})
                acronym = str(struct.get("acronym", "")).strip()
                if not acronym:
                    continue
                seen_ids.add(int(represented_id))
                represented.append((int(represented_id), acronym))
        return represented

    def _selected_cluster_region_selection(
        self,
    ) -> ClusterRegionSelection | None:
        """Return the full clustering region-selection payload."""
        from ..analysis.clustering import ClusterRegionSelection

        selected_regions = self._selected_cluster_regions()
        if not selected_regions:
            return None

        represented_regions = self._represented_region_entries_for_selections(
            [region_id for region_id, _acronym in selected_regions]
        )
        return ClusterRegionSelection(
            selected_region_ids=[
                region_id for region_id, _acronym in selected_regions
            ],
            selected_region_acronyms=[
                acronym for _region_id, acronym in selected_regions
            ],
            represented_region_ids=[
                region_id for region_id, _acronym in represented_regions
            ],
            represented_region_acronyms=[
                acronym for _region_id, acronym in represented_regions
            ],
        )

    def _on_cluster_region_selection_changed(
        self, _acronyms: list[str]
    ) -> None:
        """Keep the clustering region summary in sync with tree selection."""
        self._update_region_summary_labels()

    def _on_cluster_region_scope_changed(self, _text: str) -> None:
        """Switch the visible clustering selector to the active scope."""
        self._cluster_region_query_scope = self._selected_cluster_region_scope()
        self._sync_cluster_region_scope_selector()
        self._update_region_summary_labels()

    def _on_heat_region_selection_changed(self, _acronyms: list[str]) -> None:
        """Keep the heatmap region summary in sync with tree selection."""
        self._update_region_summary_labels()

    def _on_clustering_method_changed(self, text: str) -> None:
        """Show/hide UI rows based on the selected clustering method."""
        is_soma = text == "Soma Location"

        # Algorithm row: only for soma
        self._algorithm_label.setVisible(is_soma)
        self._algorithm_combo.setVisible(is_soma)

        if is_soma:
            self._on_algorithm_changed(self._algorithm_combo.currentText())
        else:
            # Voxel Correlation: show linkage + clusters, hide DBSCAN params
            self._linkage_label.setVisible(True)
            self._method_combo.setVisible(True)
            self._clusters_label.setVisible(True)
            self._n_clusters_spin.setVisible(True)
            self._eps_label.setVisible(False)
            self._eps_spin.setVisible(False)
            self._min_samples_label.setVisible(False)
            self._min_samples_spin.setVisible(False)

    def _on_algorithm_changed(self, text: str) -> None:
        """Show/hide UI rows based on the selected soma algorithm."""
        is_dbscan = text == "DBSCAN"
        is_hierarchical = text == "Hierarchical"

        self._linkage_label.setVisible(is_hierarchical)
        self._method_combo.setVisible(is_hierarchical)
        self._clusters_label.setVisible(not is_dbscan)
        self._n_clusters_spin.setVisible(not is_dbscan)
        self._eps_label.setVisible(is_dbscan)
        self._eps_spin.setVisible(is_dbscan)
        self._min_samples_label.setVisible(is_dbscan)
        self._min_samples_spin.setVisible(is_dbscan)

    def _run_clustering_pipeline(self) -> None:
        """Start the appropriate clustering pipeline in a background thread."""
        if self._db is None or self._atlas is None:
            return

        if self._worker_thread is not None and self._worker_thread.isRunning():
            return

        proceed, base_file_ids, _scope_label, _input_count = (
            self._resolve_cluster_query_file_scope()
        )
        if not proceed:
            return

        region_selection = self._selected_cluster_region_selection()
        if region_selection is None:
            self._progress_label.setText("Select at least one target region.")
            return
        if not region_selection.represented_region_ids:
            self._progress_label.setText(
                "Selected region(s) have no represented dataset regions."
            )
            return

        dilation = self._dilation_spin.value() / 100.0
        clustering_method = self._clustering_method_combo.currentText()

        if clustering_method == "Soma Location":
            self._run_soma_clustering(
                region_selection,
                dilation,
                file_ids=base_file_ids,
            )
        else:
            self._run_correlation_clustering(
                region_selection,
                dilation,
                file_ids=base_file_ids,
            )

    def _run_correlation_clustering(
        self,
        region_selection: ClusterRegionSelection,
        dilation: float,
        *,
        file_ids: list[str] | None = None,
    ) -> None:
        """Start the voxel correlation + clustering pipeline."""
        from ..workers import CorrelationWorker

        method = self._method_combo.currentText()
        n_clusters = self._n_clusters_spin.value()

        worker = CorrelationWorker(
            parquet_path=self._parquet_path,
            atlas=self._atlas,
            region_selection=region_selection,
            dilation_fraction=dilation,
            linkage_method=method,
            n_clusters=n_clusters,
            file_ids=file_ids,
        )

        self._start_background_worker(worker, self._on_correlation_finished)

    def _run_soma_clustering(
        self,
        region_selection: ClusterRegionSelection,
        dilation: float,
        *,
        file_ids: list[str] | None = None,
    ) -> None:
        """Start the soma-location clustering pipeline."""
        from ..workers import SomaClusterWorker

        algorithm_text = self._algorithm_combo.currentText()
        algorithm_map = {
            "Hierarchical": "hierarchical",
            "K-Means": "kmeans",
            "DBSCAN": "dbscan",
        }
        algorithm = algorithm_map[algorithm_text]

        worker = SomaClusterWorker(
            parquet_path=self._parquet_path,
            atlas=self._atlas,
            region_selection=region_selection,
            dilation_fraction=dilation,
            algorithm=algorithm,
            linkage_method=self._method_combo.currentText(),
            n_clusters=self._n_clusters_spin.value(),
            eps=self._eps_spin.value(),
            min_samples=self._min_samples_spin.value(),
            file_ids=file_ids,
        )

        self._start_background_worker(worker, self._on_correlation_finished)

    def _start_background_worker(self, worker, finished_slot) -> None:
        """Wire up and start a background worker in a QThread."""
        thread = QThread()
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(finished_slot)
        worker.finished.connect(thread.quit)
        worker.error.connect(self._on_error)
        worker.error.connect(thread.quit)
        thread.finished.connect(self._on_thread_finished)
        thread.finished.connect(self._update_button_states)
        thread.finished.connect(thread.deleteLater)

        # Keep references to prevent garbage collection
        self._worker_thread = thread
        self._current_worker = worker

        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)  # indeterminate
        self._update_button_states()

        thread.start()

    def _run_heatmap_pipeline(self) -> None:
        """Start the heatmap pipeline in a background thread."""
        if self._db is None or self._atlas is None:
            return

        if self._worker_thread is not None and self._worker_thread.isRunning():
            return

        request = self._selected_heatmap_request()
        if request is None:
            return

        self._start_heatmap_requests([request], batch_mode=False)

    def _run_all_cluster_heatmaps(self) -> None:
        """Queue heatmaps for every cluster shown in the dropdown."""
        if self._db is None or self._atlas is None:
            return

        if self._worker_thread is not None and self._worker_thread.isRunning():
            return

        requests = self._all_cluster_heatmap_requests()
        if not requests:
            self._progress_label.setText(
                "Run clustering before adding cluster heatmaps."
            )
            return

        self._start_heatmap_requests(requests, batch_mode=True)

    def _current_heatmap_region_filter(
        self,
    ) -> tuple[int | None, str | None, tuple[int, ...] | None] | None:
        """Return the current region filter for heatmap creation."""
        selected_region = self._selected_heat_region()
        if selected_region is None:
            return (None, None, None)

        region_id, acronym = selected_region
        represented_ids = tuple(
            self._represented_region_ids_for_selection(region_id)
        )
        if not represented_ids:
            self._progress_label.setText(
                "Selected region has no represented dataset regions."
            )
            return None
        return (int(region_id), acronym, represented_ids)

    def _cluster_file_ids(self, cluster_label: int) -> tuple[str, ...]:
        """Return neuron IDs assigned to one cluster label."""
        result = self._last_cluster_result
        if result is None:
            return ()

        mask = np.asarray(result.labels) == int(cluster_label)
        return tuple(
            str(neuron_id)
            for neuron_id, matched in zip(result.neuron_ids, mask)
            if matched
        )

    def _selected_heatmap_node_types(self) -> tuple[int, ...] | None:
        """Return the Analysis heatmap node-type filter."""
        combo = getattr(self, "_heat_node_type_combo", None)
        getter = getattr(combo, "selected_node_types", None)
        if callable(getter):
            return getter()
        return None

    def _selected_heatmap_soma_radius_um(self) -> float | None:
        """Return the Analysis heatmap soma-radius filter."""
        checkbox = getattr(self, "_heat_soma_radius_enabled_cb", None)
        is_checked = getattr(checkbox, "isChecked", None)
        if not callable(is_checked) or not is_checked():
            return None

        spin = getattr(self, "_heat_soma_radius_spin", None)
        value_getter = getattr(spin, "value", None)
        if not callable(value_getter):
            return None
        radius = float(value_getter())
        if radius <= 0.0:
            return None
        return radius

    def _build_heatmap_request(
        self,
        cluster_label: int | None,
        *,
        depth_axis: int | None = None,
        depth_bin_factor: int | None = None,
    ) -> _HeatmapRequest | None:
        """Build one heatmap request from current UI state."""
        region_filter = self._current_heatmap_region_filter()
        if region_filter is None:
            return None

        selected_region_id, selected_region_acronym, region_ids = region_filter
        request_cluster = (
            None if cluster_label is None else int(cluster_label)
        )
        file_ids = None
        if request_cluster is not None:
            file_ids = self._cluster_file_ids(request_cluster)
            if not file_ids:
                self._progress_label.setText(
                    f"No neurons found for cluster {request_cluster}."
                )
                return None

        resolved_depth_axis = (
            self._current_depth_axis()
            if depth_axis is None
            else int(depth_axis)
        )
        resolved_depth_bin = (
            self._depth_bin_spin.value()
            if depth_bin_factor is None
            else int(depth_bin_factor)
        )

        return _HeatmapRequest(
            selected_region_id=selected_region_id,
            selected_region_acronym=selected_region_acronym,
            region_ids=region_ids,
            cluster_label=request_cluster,
            file_ids=file_ids,
            node_types=self._selected_heatmap_node_types(),
            soma_radius_um=self._selected_heatmap_soma_radius_um(),
            depth_bin_factor=resolved_depth_bin,
            depth_axis=resolved_depth_axis,
        )

    def _selected_heatmap_request(self) -> _HeatmapRequest | None:
        """Return the heatmap request for the current dropdown selection."""
        cluster_label = None
        cluster_idx = self._heat_cluster_combo.currentIndex()
        if cluster_idx > 0:
            cluster_label = self._heat_cluster_combo.itemData(cluster_idx)
        return self._build_heatmap_request(cluster_label)

    def _all_cluster_heatmap_labels(self) -> list[int]:
        """Return all concrete cluster labels shown in the heatmap dropdown."""
        labels: list[int] = []
        seen: set[int] = set()
        combo = getattr(self, "_heat_cluster_combo", None)
        if combo is not None:
            for index in range(1, combo.count()):
                label = combo.itemData(index)
                if label is None:
                    continue
                value = int(label)
                if value in seen:
                    continue
                seen.add(value)
                labels.append(value)
        if labels:
            return labels

        result = self._last_cluster_result
        if result is None:
            return []
        return [int(label) for label in sorted(np.unique(result.labels).tolist())]

    def _all_cluster_heatmap_requests(self) -> list[_HeatmapRequest]:
        """Return heatmap requests for every cluster-specific option."""
        depth_axis = self._current_depth_axis()
        depth_bin_factor = self._depth_bin_spin.value()
        requests: list[_HeatmapRequest] = []
        for cluster_label in self._all_cluster_heatmap_labels():
            request = self._build_heatmap_request(
                cluster_label,
                depth_axis=depth_axis,
                depth_bin_factor=depth_bin_factor,
            )
            if request is not None:
                requests.append(request)
        return requests

    def _start_heatmap_requests(
        self,
        requests: list[_HeatmapRequest],
        *,
        batch_mode: bool,
    ) -> None:
        """Start one or more heatmap requests in sequence."""
        if not requests:
            return

        self._pending_heatmap_requests = list(requests)
        self._completed_heatmap_requests = []
        self._current_heatmap_request = None
        self._active_heatmap_total = len(requests)
        self._active_heatmap_index = 0
        self._heatmap_batch_mode = bool(batch_mode)
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)
        self._update_button_states()
        self._start_next_heatmap_request()

    def _start_next_heatmap_request(self) -> None:
        """Start the next queued heatmap request."""
        if not self._pending_heatmap_requests:
            return

        from ..workers import HeatmapWorker

        request = self._pending_heatmap_requests.pop(0)
        self._current_heatmap_request = request
        self._active_heatmap_index = len(self._completed_heatmap_requests) + 1

        worker = HeatmapWorker(
            parquet_path=self._parquet_path,
            atlas=self._atlas,
            region_ids=(
                list(request.region_ids)
                if request.region_ids is not None
                else None
            ),
            file_ids=(
                list(request.file_ids) if request.file_ids is not None else None
            ),
            node_types=(
                list(request.node_types)
                if request.node_types is not None
                else None
            ),
            soma_radius_um=request.soma_radius_um,
            depth_bin_factor=request.depth_bin_factor,
            depth_axis=request.depth_axis,
        )

        thread = QThread()
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_heatmap_progress)
        worker.finished.connect(self._on_heatmap_finished)
        worker.finished.connect(thread.quit)
        worker.error.connect(self._on_heatmap_error)
        worker.error.connect(thread.quit)
        thread.finished.connect(self._on_heatmap_thread_finished)
        thread.finished.connect(thread.deleteLater)

        self._worker_thread = thread
        self._current_worker = worker

        if self._heatmap_batch_mode:
            self._progress_label.setText(
                f"Building heatmap {self._active_heatmap_index}/"
                f"{self._active_heatmap_total}: "
                f"{self._heatmap_layer_name(request)}"
            )
        else:
            self._progress_label.setText("Building heatmap volume...")
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)
        self._update_button_states()

        thread.start()

    def _on_progress(self, step_name: str, current: int, total: int) -> None:
        """Handle progress updates from workers."""
        self._progress_label.setText(f"Step {current}/{total}: {step_name}")
        self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(current)

    def _on_heatmap_progress(
        self,
        step_name: str,
        current: int,
        total: int,
    ) -> None:
        """Handle progress updates for queued heatmap builds."""
        request = self._current_heatmap_request
        if not self._heatmap_batch_mode or request is None:
            self._on_progress(step_name, current, total)
            return

        overall_total = max(int(total) * self._active_heatmap_total, 1)
        overall_current = (
            (self._active_heatmap_index - 1) * int(total)
        ) + int(current)
        self._progress_label.setText(
            f"Heatmap {self._active_heatmap_index}/"
            f"{self._active_heatmap_total}: "
            f"{self._heatmap_layer_name(request)} - {step_name}"
        )
        self._progress_bar.setRange(0, overall_total)
        self._progress_bar.setValue(overall_current)

    def _on_correlation_finished(self, result: ClusterResult) -> None:
        """Handle completed correlation pipeline."""
        finish_start = perf_counter()
        distance_matrix = getattr(result, "distance_matrix", None)
        logger.debug(
            "_on_correlation_finished start: neurons=%d distance_shape=%s distance_dtype=%s distance_nbytes=%s",
            len(result.neuron_ids),
            getattr(distance_matrix, "shape", None),
            getattr(distance_matrix, "dtype", None),
            getattr(distance_matrix, "nbytes", None),
        )
        self._last_cluster_result = result
        color_map_start = perf_counter()
        self._build_cluster_color_map()
        color_summary = self._color_neurons_by_cluster()
        logger.debug(
            "_on_correlation_finished color map built: elapsed=%.3fs actual_clusters=%d",
            perf_counter() - color_map_start,
            self._actual_n_clusters,
        )
        self._progress_bar.setVisible(False)

        requested_k = self._n_clusters_spin.value()
        actual_k = self._actual_n_clusters
        if actual_k < requested_k:
            cluster_msg = (
                f"{actual_k} of {requested_k} requested clusters found"
            )
        else:
            cluster_msg = f"{actual_k} clusters"
        progress_message = (
            f"Clustering complete: {len(result.neuron_ids)} neurons, "
            f"{cluster_msg}. Click Render Dendrogram to view. "
            "Table updated and sorted by cluster."
        )
        if color_summary.colored_count > 0 and color_summary.rendered_count > 0:
            neuron_word = (
                "neuron" if color_summary.rendered_count == 1 else "neurons"
            )
            progress_message += (
                f" Auto-colored {color_summary.colored_count}/"
                f"{color_summary.rendered_count} rendered {neuron_word} "
                "by cluster."
            )
            if color_summary.gray_count > 0:
                progress_message += (
                    f" {color_summary.gray_count} shown in gray."
                )
        self._progress_label.setText(progress_message)
        clustermap_message = (
            "Clustering complete. Click 'Build Dendrogram' to render the cluster map."
        )
        clustermap_status = getattr(self, "_clustermap_status_label", None)
        if clustermap_status is not None:
            clustermap_status.setText(clustermap_message)
        self._update_cluster_filter_combo()
        self._update_button_states()
        show_placeholder = "_show_clustermap_message" in self.__dict__ or (
            hasattr(self, "_figure") and hasattr(self, "_canvas")
        )
        if show_placeholder:
            self._show_clustermap_message(
                "Clustering complete. Click Render Dendrogram to view."
            )
        logger.debug(
            "_on_correlation_finished worker result ready; clustermap render deferred until button click"
        )
        if self._cluster_color_map is not None:
            try:
                self.cluster_colors_updated.emit(result, self._cluster_color_map)
            except RuntimeError:
                pass
        logger.debug(
            "_on_correlation_finished complete: total_elapsed=%.3fs",
            perf_counter() - finish_start,
        )

    def _on_heatmap_finished(self, volume: np.ndarray) -> None:
        """Handle completed heatmap pipeline."""
        request = self._current_heatmap_request
        if request is None:
            return

        layer = self._add_analysis_heatmap_layer(volume, request)
        self._completed_heatmap_requests.append(request)
        self._heatmap_layer = layer

        if self._heatmap_batch_mode:
            self._progress_label.setText(
                f"Added {self._heatmap_layer_name(request)} "
                f"({self._active_heatmap_index}/"
                f"{self._active_heatmap_total})"
            )
            return

        self._progress_label.setText(
            f"{layer.name}: {(volume > 0).sum():,} non-zero voxels"
        )

    def _heatmap_layer_name(self, request: _HeatmapRequest) -> str:
        """Return the napari layer name for one heatmap request."""
        region_part = (
            f" {request.selected_region_acronym}"
            if request.selected_region_acronym
            else ""
        )
        filter_parts: list[str] = []
        if request.node_types is not None:
            filter_parts.append(
                NodeTypeSelectorComboBox.selection_text(request.node_types)
            )
        if request.soma_radius_um is not None:
            filter_parts.append(f"{request.soma_radius_um:g} μm soma radius")
        filter_part = f" ({', '.join(filter_parts)})" if filter_parts else ""
        if request.cluster_label is None:
            return f"Node Count{region_part}{filter_part} Heatmap"
        return f"Cluster {request.cluster_label}{region_part}{filter_part} Heatmap"

    def _add_analysis_heatmap_layer(
        self,
        volume: np.ndarray,
        request: _HeatmapRequest,
    ):
        """Add or replace one analysis heatmap layer."""
        from napari.utils.colormaps import Colormap

        layer_name = self._heatmap_layer_name(request)
        contrast_limits = _analysis_heatmap_contrast_limits(volume)

        if request.cluster_label is not None:
            rgba = getattr(self, "_cluster_label_colors", {}).get(
                request.cluster_label, [0.5, 0.5, 0.5, 1.0]
            )
            colormap = Colormap(
                colors=[[0, 0, 0, 0], [rgba[0], rgba[1], rgba[2], 1.0]],
                name=f"cluster_{request.cluster_label}",
            )
        else:
            colormap = "hot"

        for layer in list(self._viewer.layers):
            if layer.name == layer_name:
                self._viewer.layers.remove(layer)

        scale = [1.0, 1.0, 1.0]
        scale[request.depth_axis] = float(request.depth_bin_factor)
        source_file_ids = (
            list(request.file_ids) if request.file_ids is not None else None
        )
        metadata = {
            "heatmap_source": True,
            "heatmap_native_grid": request.depth_bin_factor == 1,
            "atlas_name": getattr(self._atlas, "atlas_name", None),
            "heatmap_kind": "analysis",
            "heatmap_region": request.selected_region_acronym,
            "heatmap_selected_region_id": request.selected_region_id,
            "heatmap_selected_region_acronym": request.selected_region_acronym,
            "heatmap_region_ids": (
                list(request.region_ids)
                if request.region_ids is not None
                else None
            ),
            "heatmap_node_types": (
                list(request.node_types)
                if request.node_types is not None
                else None
            ),
            "heatmap_node_type_labels": (
                NodeTypeSelectorComboBox.metadata_labels(request.node_types)
            ),
            "heatmap_soma_radius_um": request.soma_radius_um,
            "heatmap_cluster": request.cluster_label,
            "file_ids": source_file_ids,
            "source_file_ids": source_file_ids,
            "depth_bin_factor": request.depth_bin_factor,
            "depth_axis": request.depth_axis,
            "heatmap_contrast_limits": contrast_limits,
            "heatmap_autocontrast_policy": "stable_full_volume",
        }

        layer = self._viewer.add_image(
            volume,
            name=layer_name,
            colormap=colormap,
            blending="additive",
            rendering="mip",
            opacity=0.7,
            visible=True,
            scale=scale,
            contrast_limits=contrast_limits,
            metadata=metadata,
        )
        _install_analysis_heatmap_layer_workarounds(layer)
        return layer

    def _on_heatmap_thread_finished(self) -> None:
        """Advance the heatmap queue or finalize the completed run."""
        self._on_thread_finished()

        if self._pending_heatmap_requests:
            self._start_next_heatmap_request()
            return

        if self._completed_heatmap_requests:
            self._last_heatmap_requests = list(
                self._completed_heatmap_requests
            )

        completed_count = len(self._completed_heatmap_requests)
        batch_mode = self._heatmap_batch_mode
        self._current_heatmap_request = None
        self._completed_heatmap_requests = []
        self._active_heatmap_total = 0
        self._active_heatmap_index = 0
        self._heatmap_batch_mode = False
        self._progress_bar.setVisible(False)
        if batch_mode and completed_count > 0:
            suffix = "" if completed_count == 1 else "s"
            self._progress_label.setText(
                f"Added {completed_count} cluster heatmap{suffix} to scene"
            )
        self._update_button_states()

    def _on_heatmap_error(self, message: str) -> None:
        """Handle a heatmap worker failure and stop any remaining queue."""
        if self._completed_heatmap_requests:
            self._last_heatmap_requests = list(
                self._completed_heatmap_requests
            )
        self._pending_heatmap_requests = []
        self._current_heatmap_request = None
        self._completed_heatmap_requests = []
        self._active_heatmap_total = 0
        self._active_heatmap_index = 0
        self._heatmap_batch_mode = False
        self._on_error(message)

    def _on_error(self, message: str) -> None:
        """Handle worker errors."""
        self._progress_bar.setVisible(False)
        self._progress_label.setText(f"Error: {message}")
        self._update_button_states()
        logger.error(f"Analysis pipeline error: {message}")

    def _render_clustermap_requested(self) -> None:
        """Render the latest clustermap only when explicitly requested."""
        if self._last_cluster_result is None:
            return
        if (
            getattr(self, "_build_clustermap_btn", None) is not None
            and getattr(self, "_clustermap_status_label", None) is not None
        ):
            self._build_clustermap_on_demand()
            return
        self._draw_clustermap(self._last_cluster_result)

    def _show_clustermap_message(self, message: str) -> None:
        """Reset the embedded clustermap canvas to a placeholder message."""
        if not hasattr(self, "_figure") or not hasattr(self, "_canvas"):
            return
        self._clustermap_rendered = False
        clear = getattr(self._figure, "clear", None)
        if callable(clear):
            clear()
            add_subplot = getattr(self._figure, "add_subplot", None)
            if callable(add_subplot):
                ax = self._figure.add_subplot(111)
                text_kwargs = {
                    "ha": "center",
                    "va": "center",
                }
                transform = getattr(ax, "transAxes", None)
                if transform is not None:
                    text_kwargs["transform"] = transform
                ax.text(
                    0.5,
                    0.5,
                    message,
                    **text_kwargs,
                )
                axis_off = getattr(ax, "set_axis_off", None)
                if callable(axis_off):
                    axis_off()
        self._canvas.draw()

    def _clustermap_canvas_logical_pixel_size(self) -> tuple[int, int] | None:
        """Return the current canvas size in logical Qt pixels, if available."""
        width = height = None

        width_method = getattr(self._canvas, "width", None)
        if callable(width_method):
            try:
                width = int(width_method())
            except (TypeError, ValueError):
                width = None

        height_method = getattr(self._canvas, "height", None)
        if callable(height_method):
            try:
                height = int(height_method())
            except (TypeError, ValueError):
                height = None

        if width is None:
            try:
                width = int(getattr(self._canvas, "_width", 0))
            except (TypeError, ValueError):
                width = None
        if height is None:
            try:
                height = int(getattr(self._canvas, "_height", 0))
            except (TypeError, ValueError):
                height = None

        if width is None or height is None:
            size_attr = getattr(self._canvas, "size", None)
            size_obj = size_attr() if callable(size_attr) else size_attr
            if size_obj is not None:
                if width is None:
                    width_getter = getattr(size_obj, "width", None)
                    if callable(width_getter):
                        width = int(width_getter())
                if height is None:
                    height_getter = getattr(size_obj, "height", None)
                    if callable(height_getter):
                        height = int(height_getter())

        if width is None or height is None or width <= 0 or height <= 0:
            return None
        return (width, height)

    def _clustermap_canvas_physical_pixel_size(
        self,
    ) -> tuple[int, int] | None:
        """Return the current canvas size in physical pixels, if available."""
        get_width_height = getattr(self._canvas, "get_width_height", None)
        if callable(get_width_height):
            try:
                width, height = get_width_height(physical=True)
            except TypeError:
                width = height = None
            else:
                try:
                    width = int(width)
                    height = int(height)
                except (TypeError, ValueError):
                    width = height = None
                if width > 0 and height > 0:
                    return (width, height)

        logical_size = self._clustermap_canvas_logical_pixel_size()
        if logical_size is None:
            return None

        device_pixel_ratio = getattr(self._canvas, "device_pixel_ratio", 1.0)
        if callable(device_pixel_ratio):
            try:
                ratio = float(device_pixel_ratio())
            except (TypeError, ValueError):
                ratio = 1.0
        else:
            try:
                ratio = float(device_pixel_ratio)
            except (TypeError, ValueError):
                ratio = 1.0

        if not np.isfinite(ratio) or ratio <= 0.0:
            ratio = 1.0

        width, height = logical_size
        physical_width = int(round(width * ratio))
        physical_height = int(round(height * ratio))
        if physical_width <= 0 or physical_height <= 0:
            return None
        return (physical_width, physical_height)

    def _clustermap_preview_figsize(
        self, dpi: float
    ) -> tuple[float, float] | None:
        """Return the preview figure size using backend-owned geometry."""
        get_size_inches = getattr(self._figure, "get_size_inches", None)
        if callable(get_size_inches):
            try:
                size_values = np.asarray(get_size_inches(), dtype=float).reshape(
                    -1
                )
            except (TypeError, ValueError):
                size_values = np.array([], dtype=float)
            if size_values.size >= 2:
                width_in = float(size_values[0])
                height_in = float(size_values[1])
                if (
                    np.isfinite(width_in)
                    and np.isfinite(height_in)
                    and width_in > 0.0
                    and height_in > 0.0
                ):
                    return (width_in, height_in)

        size = self._clustermap_canvas_physical_pixel_size()
        if size is None:
            return None

        width_px, height_px = size
        if not np.isfinite(dpi) or dpi <= 0.0:
            return None
        return (max(width_px, 1) / dpi, max(height_px, 1) / dpi)

    def _draw_clustermap(self, result: ClusterResult) -> None:
        """Draw the clustermap preview into the embedded canvas."""
        draw_start = perf_counter()
        self._figure.clear()

        try:
            dpi_getter = getattr(self._figure, "get_dpi", None)
            dpi = float(dpi_getter()) if callable(dpi_getter) else float(
                getattr(self._figure, "dpi", 100.0) or 100.0
            )
            figsize = self._clustermap_preview_figsize(dpi)
            physical_size = self._clustermap_canvas_physical_pixel_size()
            if figsize is None:
                figsize = (6.0, 6.0)
            logger.debug(
                "_draw_clustermap start: distance_shape=%s distance_dtype=%s distance_nbytes=%s linkage_shape=%s physical_canvas_size=%s figsize=%s dpi=%.3f",
                getattr(result.distance_matrix, "shape", None),
                getattr(result.distance_matrix, "dtype", None),
                getattr(result.distance_matrix, "nbytes", None),
                getattr(result.linkage_matrix, "shape", None),
                physical_size,
                figsize,
                dpi,
            )
            figure_start = perf_counter()
            _populate_embedded_clustermap_figure(
                self._figure,
                result,
                self._cluster_color_map,
                figsize=figsize,
                dpi=int(round(dpi)),
            )
            logger.debug(
                "_draw_clustermap populate_clustermap_figure complete: elapsed=%.3fs",
                perf_counter() - figure_start,
            )

            self._clustermap_rendered = True
            canvas_start = perf_counter()
            self._canvas.draw()
            logger.debug(
                "_draw_clustermap canvas draw complete: elapsed=%.3fs",
                perf_counter() - canvas_start,
            )
            logger.debug(
                "_draw_clustermap complete: total_elapsed=%.3fs",
                perf_counter() - draw_start,
            )

        except Exception as e:
            logger.exception("Failed to draw clustermap")
            ax = self._figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                f"Error drawing clustermap:\n{e}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            self._clustermap_rendered = True
            self._canvas.draw()

    def _build_clustermap_on_demand(self) -> None:
        """Render the cached clustering result into the clustermap canvas."""
        result = self._last_cluster_result
        if result is None:
            self._clustermap_status_label.setText(
                "Run clustering before building the dendrogram."
            )
            return

        logger.debug(
            "_build_clustermap_on_demand start: neurons=%d distance_shape=%s",
            len(result.neuron_ids),
            getattr(result.distance_matrix, "shape", None),
        )
        self._build_clustermap_btn.setEnabled(False)
        self._clustermap_status_label.setText(
            f"Rendering dendrogram for {len(result.neuron_ids)} neurons..."
        )
        draw_start = perf_counter()
        try:
            self._draw_clustermap(result)
            self._clustermap_status_label.setText(
                f"Dendrogram ready for {len(result.neuron_ids)} neurons."
            )
        finally:
            self._update_button_states()
        logger.debug(
            "_build_clustermap_on_demand complete: elapsed=%.3fs",
            perf_counter() - draw_start,
        )

    def _build_cluster_color_map(self) -> None:
        """Build and cache the neuron_id -> RGBA color mapping from cluster results.

        Called once when clustering completes. The cached map is reused
        when rendered neurons are auto-colored after clustering and when
        cached colors are reapplied manually.
        """
        if self._last_cluster_result is None:
            self._cluster_color_map = None
            return

        start = perf_counter()
        result = self._last_cluster_result
        unique_labels = np.unique(result.labels)
        n_clusters = int(len(unique_labels))

        logger.info(
            f"Building cluster color map: {len(result.neuron_ids)} neurons, "
            f"{n_clusters} unique clusters (labels: {unique_labels.tolist()})"
        )

        # Use explicit colors for small cluster counts to guarantee
        # visually distinct colors; fall back to tab10/tab20 otherwise.
        _CUSTOM_COLORS: dict[int, list[list[float]]] = {
            1: [
                [0.12, 0.47, 0.71, 1.0],  # blue (all same cluster)
            ],
            2: [
                [0.12, 0.47, 0.71, 1.0],  # blue
                [0.84, 0.15, 0.16, 1.0],  # red
            ],
            3: [
                [0.12, 0.47, 0.71, 1.0],  # blue
                [0.84, 0.15, 0.16, 1.0],  # red
                [0.17, 0.63, 0.17, 1.0],  # green
            ],
        }

        # Map each unique label to a color index (0, 1, 2, ...) so colors
        # are assigned correctly even when labels are non-contiguous.
        label_to_idx = {int(lab): i for i, lab in enumerate(unique_labels)}

        color_map: dict[str, list[float]] = {}
        if n_clusters in _CUSTOM_COLORS:
            palette = _CUSTOM_COLORS[n_clusters]
            for neuron_id, label in zip(result.neuron_ids, result.labels):
                color_map[neuron_id] = palette[label_to_idx[int(label)]]
        else:
            cmap = plt.get_cmap("tab10" if n_clusters <= 10 else "tab20")
            for neuron_id, label in zip(result.neuron_ids, result.labels):
                idx = label_to_idx[int(label)]
                color_map[neuron_id] = list(cmap(idx / n_clusters))

        self._cluster_color_map = color_map
        self._actual_n_clusters = n_clusters
        # Build reverse map: cluster_label -> RGBA color (first neuron's color)
        self._cluster_label_colors: dict[int, list[float]] = {}
        for neuron_id, label in zip(result.neuron_ids, result.labels):
            lab = int(label)
            if lab not in self._cluster_label_colors:
                self._cluster_label_colors[lab] = color_map[neuron_id]
        logger.info(
            f"Built cluster color map: {len(color_map)} neurons, "
            f"{n_clusters} clusters"
        )
        logger.debug(
            "_build_cluster_color_map complete: elapsed=%.3fs",
            perf_counter() - start,
        )

    def _update_cluster_filter_combo(self) -> None:
        """Populate the heatmap cluster filter dropdown with cluster options.

        Each item shows a color swatch icon, the cluster number, and the
        neuron count for that cluster.
        """
        self._heat_cluster_combo.clear()
        self._heat_cluster_combo.addItem("All neurons")

        result = self._last_cluster_result
        if result is None or not hasattr(self, "_cluster_label_colors"):
            self._heat_cluster_combo.setEnabled(False)
            return

        unique_labels = sorted(np.unique(result.labels).tolist())
        for label in unique_labels:
            rgba = self._cluster_label_colors.get(label, [0.5, 0.5, 0.5, 1.0])
            count = int(np.sum(result.labels == label))

            # Create a small color swatch icon
            pixmap = QPixmap(16, 16)
            pixmap.fill(QColor.fromRgbF(rgba[0], rgba[1], rgba[2], rgba[3]))
            icon = QIcon(pixmap)

            self._heat_cluster_combo.addItem(
                icon, f"Cluster {label}  ({count} neurons)"
            )
            # Store the label as item data for retrieval
            self._heat_cluster_combo.setItemData(
                self._heat_cluster_combo.count() - 1, label
            )

        self._heat_cluster_combo.setEnabled(True)

    def _analysis_export_basename(self) -> str:
        """Return the basename used for analysis export defaults."""
        if self._parquet_path:
            stem = Path(self._parquet_path).stem
            if stem:
                return stem
        return "analysis"

    def _export_title(self) -> str:
        """Return the current export figure title."""
        return self._export_title_edit.text().strip()

    def _export_x_label(self) -> str:
        """Return the current export x-axis label."""
        return self._export_xlabel_edit.text().strip()

    def _selected_export_dpi(self) -> int:
        """Return the selected raster DPI preset."""
        index = self._export_dpi_combo.currentIndex()
        value = self._export_dpi_combo.itemData(index) if index >= 0 else None
        return int(value) if value is not None else 300

    def _prompt_analysis_export_path(
        self,
        dialog_title: str,
        default_name: str,
        file_filter: str,
    ) -> str | None:
        """Prompt for a save path for one analysis export."""
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            dialog_title,
            default_name,
            file_filter,
        )
        return output_path or None

    def _start_analysis_export(
        self, export_kind: str, output_path: str
    ) -> None:
        """Start one workbook/parquet export in a background thread."""
        from ..workers import AnalysisExportWorker

        if self._last_cluster_result is None:
            return

        worker = AnalysisExportWorker(
            export_kind,
            output_path,
            self._last_cluster_result,
            self._cluster_color_map,
            figure_title=self._export_title(),
            x_label=self._export_x_label(),
            y_label="",
        )
        self._progress_label.setText(f"Saving {Path(output_path).name}...")
        self._start_background_worker(worker, self._on_export_finished)

    def _on_export_finished(self, output_path: str) -> None:
        """Handle a completed workbook/parquet export."""
        self._progress_bar.setVisible(False)
        self._progress_label.setText(f"Saved {Path(output_path).name}")
        self._update_button_states()

    def _save_cluster_workbook(self) -> None:
        """Prompt for and save the cluster workbook export."""
        output_path = self._prompt_analysis_export_path(
            "Save Cluster Workbook",
            f"{self._analysis_export_basename()}_clusters.xlsx",
            "Excel Files (*.xlsx);;All Files (*)",
        )
        if output_path:
            self._start_analysis_export("cluster_workbook", output_path)

    def _save_distance_workbook(self) -> None:
        """Prompt for and save the distance workbook export."""
        output_path = self._prompt_analysis_export_path(
            "Save Distance Workbook",
            f"{self._analysis_export_basename()}_distances.xlsx",
            "Excel Files (*.xlsx);;All Files (*)",
        )
        if output_path:
            self._start_analysis_export("distance_workbook", output_path)

    def _save_extended_parquet(self) -> None:
        """Prompt for and save the extended parquet export."""
        output_path = self._prompt_analysis_export_path(
            "Save Extended Parquet",
            f"{self._analysis_export_basename()}_cluster_analysis.parquet",
            "Parquet Files (*.parquet);;All Files (*)",
        )
        if output_path:
            self._start_analysis_export("extended_parquet", output_path)

    def _save_dendrogram(self) -> None:
        """Prompt for and save the current clustermap-style dendrogram."""
        from ..analysis.export import save_dendrogram_figure

        if self._last_cluster_result is None:
            return

        dpi = self._selected_export_dpi()
        output_path = self._prompt_analysis_export_path(
            "Save Dendrogram",
            f"{self._analysis_export_basename()}_clustermap_{dpi}dpi.png",
            "PNG Files (*.png);;All Files (*)",
        )
        if not output_path:
            return

        self._progress_label.setText(f"Saving {Path(output_path).name}...")
        try:
            save_dendrogram_figure(
                output_path,
                self._last_cluster_result,
                self._cluster_color_map,
                title=self._export_title(),
                x_label=self._export_x_label(),
                y_label="",
                dpi=dpi,
            )
        except Exception as error:
            self._on_error(str(error))
            return

        self._progress_label.setText(f"Saved {Path(output_path).name}")

    def _current_depth_axis(self) -> int:
        """Return the current depth (slicing) axis from napari dims.

        The depth axis is the first non-displayed dimension. Falls back
        to axis 0 if all dimensions are displayed.
        """
        try:
            not_displayed = list(self._viewer.dims.not_displayed)
            if not_displayed:
                return int(not_displayed[0])
        except Exception:
            pass
        return 0

    def _update_voxel_depth_label(self, _value: int | None = None) -> None:
        """Update the voxel depth label based on atlas resolution and bin factor."""
        if self._atlas is None:
            self._voxel_depth_label.setText("Voxel depth: — μm")
            return
        resolution = float(self._atlas.resolution[0])
        depth = resolution * self._depth_bin_spin.value()
        self._voxel_depth_label.setText(f"Voxel depth: {depth:g} μm")

    def _on_dims_order_changed(self, event=None) -> None:
        """Rebuild heatmap when the user reorders axes in napari."""
        if not self._last_heatmap_requests:
            return
        if self._worker_thread is not None and self._worker_thread.isRunning():
            return
        if self._db is None or self._atlas is None:
            return
        updated_requests = [
            replace(request, depth_axis=self._current_depth_axis())
            for request in self._last_heatmap_requests
        ]
        self._start_heatmap_requests(
            updated_requests,
            batch_mode=len(updated_requests) > 1,
        )

    def apply_cluster_colors(self) -> _ClusterColorApplicationSummary:
        """Reapply cached cluster colors and assignments to the current table."""
        if self._cluster_color_map is None or self._last_cluster_result is None:
            return _ClusterColorApplicationSummary(
                matched_table_count=0,
                updated_layer_count=0,
                rendered_count=0,
                colored_count=0,
                gray_count=0,
            )

        summary = self._color_neurons_by_cluster()
        try:
            self.cluster_colors_updated.emit(
                self._last_cluster_result,
                self._cluster_color_map,
            )
        except RuntimeError:
            pass
        return summary

    def _color_neurons_by_cluster(self) -> _ClusterColorApplicationSummary:
        """Color existing neuron layers by their cluster assignment.

        Works with the batched single-layer rendering where all neurons
        are merged into one ``Neuron Lines`` and/or ``Neuron Points``
        layer.  Layer metadata (``file_ids``, ``segments_per_neuron``,
        ``file_ids_per_point``) is used to map cluster labels back to
        individual segments/points. Returns a small summary for status
        text composed by the caller.
        """
        if (
            self._cluster_color_map is None
            or self._last_cluster_result is None
        ):
            return _ClusterColorApplicationSummary(
                matched_table_count=0,
                updated_layer_count=0,
                rendered_count=0,
                colored_count=0,
                gray_count=0,
            )

        color_map = self._cluster_color_map
        matched_table_count = len(self._matched_current_table_file_ids())
        default_color = [0.5, 0.5, 0.5, 1.0]
        updated = 0
        n_rendered = 0
        n_colored = 0

        for layer in self._viewer.layers:
            if layer.name == "Neuron Lines":
                meta = layer.metadata or {}
                file_ids = meta.get("file_ids", [])
                seg_counts = meta.get("segments_per_neuron", [])
                if file_ids and seg_counts:
                    n_rendered += len(file_ids)
                    n_colored += sum(1 for fid in file_ids if fid in color_map)
                    parts = []
                    for fid, count in zip(file_ids, seg_counts):
                        c = color_map.get(fid, default_color)
                        arr = np.empty((count, 4))
                        arr[:] = c[:4]
                        parts.append(arr)
                    layer.edge_color = np.concatenate(parts)
                    updated += 1

            elif layer.name == "Neuron Points":
                meta = layer.metadata or {}
                fids = meta.get("file_ids_per_point", [])
                if fids:
                    unique_fids = set(fids)
                    if n_rendered == 0:
                        n_rendered = len(unique_fids)
                        n_colored = sum(
                            1 for fid in unique_fids if fid in color_map
                        )
                    colors = np.array(
                        [color_map.get(fid, default_color)[:4] for fid in fids]
                    )
                    layer.face_color = colors
                    updated += 1

        # Also update the 2D slice projector colors
        if self._slice_projector is not None:
            self._slice_projector.update_neuron_colors(color_map)

        n_gray = max(n_rendered - n_colored, 0)
        return _ClusterColorApplicationSummary(
            matched_table_count=matched_table_count,
            updated_layer_count=updated,
            rendered_count=n_rendered,
            colored_count=n_colored,
            gray_count=n_gray,
        )

    def _flush_progress_updates(self) -> None:
        """Give Qt a chance to repaint synchronous progress text."""
        try:
            from qtpy.QtWidgets import QApplication
        except ImportError:
            return

        app = QApplication.instance()
        if app is not None:
            app.processEvents()
