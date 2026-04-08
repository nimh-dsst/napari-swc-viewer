"""Analysis tab widget for the clustering pipeline.

Provides UI for:
1. Region mask selection and dilation parameters
2. Correlation matrix computation with hierarchical clustering
3. Clustermap visualization (embedded matplotlib canvas)
4. Node count heatmap generation
5. Coloring neurons by cluster assignment
"""

from __future__ import annotations

import logging
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
    QVBoxLayout,
    QWidget,
)

from .collapsible_section import CollapsibleSection
from .region_selector import RegionSelectorWidget

if TYPE_CHECKING:
    import napari
    from brainglobe_atlasapi import BrainGlobeAtlas

    from ..analysis.clustering import ClusterRegionSelection, ClusterResult
    from ..db import NeuronDatabase

logger = logging.getLogger(__name__)


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
        self._pending_heatmap_cluster: int | None = (
            None  # cluster label for in-flight heatmap
        )
        self._pending_heatmap_region: str | None = (
            None  # region acronym for in-flight heatmap
        )
        self._pending_heatmap_depth_bin: int = 1
        self._pending_heatmap_depth_axis: int = 0
        self._last_heatmap_file_ids: list[str] | None = None
        self._slice_projector = None
        self._dataset_region_ids: set[int] = set()
        self._clustermap_rendered = False
        self._clustermap_refresh_pending = False
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

    def _update_button_states(self) -> None:
        """Enable/disable buttons based on loaded data."""
        ready = self._db is not None and self._atlas is not None
        busy = (
            self._worker_thread is not None and self._worker_thread.isRunning()
        )
        self._run_corr_btn.setEnabled(ready and not busy)
        self._run_heat_btn.setEnabled(ready and not busy)
        analysis_ready = self._last_cluster_result is not None and not busy
        self._color_by_cluster_btn.setEnabled(analysis_ready)
        if hasattr(self, "_export_title_edit"):
            self._render_clustermap_btn.setEnabled(analysis_ready)
            self._export_title_edit.setEnabled(analysis_ready)
            self._export_xlabel_edit.setEnabled(analysis_ready)
            self._export_ylabel_edit.setEnabled(analysis_ready)
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
        self._cluster_region_selector = RegionSelectorWidget(
            single_select=False,
            show_include_children=False,
            force_include_children=True,
        )
        self._cluster_region_selector.selection_changed.connect(
            self._on_cluster_region_selection_changed
        )
        cluster_region_layout.addWidget(self._cluster_region_selector)
        corr_layout.addWidget(self._cluster_region_section)

        # Dilation fraction
        dilation_row = QHBoxLayout()
        dilation_row.addWidget(QLabel("Dilation %:"))
        self._dilation_spin = QSpinBox()
        self._dilation_spin.setRange(0, 100)
        self._dilation_spin.setValue(20)
        self._dilation_spin.setSuffix("%")
        dilation_row.addWidget(self._dilation_spin)
        corr_layout.addLayout(dilation_row)

        # Linkage method
        self._linkage_row = QHBoxLayout()
        self._linkage_label = QLabel("Linkage:")
        self._linkage_row.addWidget(self._linkage_label)
        self._method_combo = QComboBox()
        self._method_combo.addItems(["average", "ward", "complete", "single"])
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

        # Color neurons by cluster
        self._color_by_cluster_btn = QPushButton("Color Neurons by Cluster")
        self._color_by_cluster_btn.setEnabled(False)
        self._color_by_cluster_btn.clicked.connect(
            self._color_neurons_by_cluster
        )
        corr_layout.addWidget(self._color_by_cluster_btn)

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

        self._run_heat_btn = QPushButton("Build Heatmap Volume")
        self._run_heat_btn.setEnabled(False)
        self._run_heat_btn.clicked.connect(self._run_heatmap_pipeline)
        heat_layout.addWidget(self._run_heat_btn)

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
        self._clustermap_section.expanded_changed.connect(
            self._on_clustermap_section_expanded_changed
        )
        clustermap_layout = self._clustermap_section.content_layout()
        self._render_clustermap_btn = QPushButton("Render Dendrogram")
        self._render_clustermap_btn.clicked.connect(
            self._render_clustermap_requested
        )
        clustermap_layout.addWidget(self._render_clustermap_btn)
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

        export_ylabel_row = QHBoxLayout()
        export_ylabel_row.addWidget(QLabel("Y label:"))
        self._export_ylabel_edit = QLineEdit()
        export_ylabel_row.addWidget(self._export_ylabel_edit)
        export_layout.addLayout(export_ylabel_row)

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
                getattr(self, "_cluster_region_selector", None),
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
        cluster_selector = getattr(self, "_cluster_region_selector", None)
        heat_selector = getattr(self, "_heat_region_selector", None)
        if cluster_selector is None or heat_selector is None:
            return

        selectors = (cluster_selector, heat_selector)
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
        return self._selected_region(
            getattr(self, "_cluster_region_selector", None)
        )

    def _selected_cluster_regions(self) -> list[tuple[int, str]]:
        """Return all currently selected clustering regions."""
        return self._selected_regions(
            getattr(self, "_cluster_region_selector", None)
        )

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
            self._run_soma_clustering(region_selection, dilation)
        else:
            self._run_correlation_clustering(region_selection, dilation)

    def _run_correlation_clustering(
        self,
        region_selection: ClusterRegionSelection,
        dilation: float,
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
        )

        self._start_background_worker(worker, self._on_correlation_finished)

    def _run_soma_clustering(
        self,
        region_selection: ClusterRegionSelection,
        dilation: float,
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

        from ..workers import HeatmapWorker

        selected_region = self._selected_heat_region()
        region = selected_region[1] if selected_region is not None else None
        region_ids = (
            self._represented_region_ids_for_selection(selected_region[0])
            if selected_region is not None
            else None
        )
        if selected_region is not None and not region_ids:
            self._progress_label.setText(
                "Selected region has no represented dataset regions."
            )
            return
        self._pending_heatmap_region = region

        # Determine cluster filter
        file_ids = None
        cluster_idx = self._heat_cluster_combo.currentIndex()
        if cluster_idx > 0:  # 0 = "All neurons"
            cluster_label = self._heat_cluster_combo.itemData(cluster_idx)
            self._pending_heatmap_cluster = cluster_label
            result = self._last_cluster_result
            if result is not None:
                mask = result.labels == cluster_label
                file_ids = [
                    nid for nid, m in zip(result.neuron_ids, mask) if m
                ]
        else:
            self._pending_heatmap_cluster = None

        # Determine depth axis from napari dims
        depth_axis = self._current_depth_axis()
        depth_bin_factor = self._depth_bin_spin.value()

        self._pending_heatmap_depth_bin = depth_bin_factor
        self._pending_heatmap_depth_axis = depth_axis
        self._last_heatmap_file_ids = file_ids

        worker = HeatmapWorker(
            parquet_path=self._parquet_path,
            atlas=self._atlas,
            region_ids=region_ids,
            file_ids=file_ids,
            depth_bin_factor=depth_bin_factor,
            depth_axis=depth_axis,
        )

        thread = QThread()
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_heatmap_finished)
        worker.finished.connect(thread.quit)
        worker.error.connect(self._on_error)
        worker.error.connect(thread.quit)
        thread.finished.connect(self._on_thread_finished)
        thread.finished.connect(self._update_button_states)
        thread.finished.connect(thread.deleteLater)

        self._worker_thread = thread
        self._current_worker = worker

        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)
        self._update_button_states()

        thread.start()

    def _on_progress(self, step_name: str, current: int, total: int) -> None:
        """Handle progress updates from workers."""
        self._progress_label.setText(f"Step {current}/{total}: {step_name}")
        self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(current)

    def _on_correlation_finished(self, result: ClusterResult) -> None:
        """Handle completed correlation pipeline."""
        finish_start = perf_counter()
        logger.debug(
            "_on_correlation_finished start: neurons=%d distance_shape=%s distance_dtype=%s distance_nbytes=%s",
            len(result.neuron_ids),
            getattr(result.distance_matrix, "shape", None),
            getattr(result.distance_matrix, "dtype", None),
            getattr(result.distance_matrix, "nbytes", None),
        )
        self._last_cluster_result = result
        color_map_start = perf_counter()
        self._build_cluster_color_map()
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
        self._progress_label.setText(
            f"Clustering complete: {len(result.neuron_ids)} neurons, "
            f"{cluster_msg}. Click Render Dendrogram to view."
        )
        self._update_button_states()
        self._update_cluster_filter_combo()
        self._show_clustermap_message(
            "Clustering complete. Click Render Dendrogram to view."
        )

    def _on_heatmap_finished(self, volume: np.ndarray) -> None:
        """Handle completed heatmap pipeline."""
        from napari.utils.colormaps import Colormap

        self._progress_bar.setVisible(False)
        self._update_button_states()

        cluster_label = self._pending_heatmap_cluster
        region = self._pending_heatmap_region
        self._pending_heatmap_cluster = None
        self._pending_heatmap_region = None

        region_part = f" {region}" if region else ""

        if cluster_label is not None:
            # Cluster-specific heatmap with derived colormap
            rgba = self._cluster_label_colors.get(
                cluster_label, [0.5, 0.5, 0.5, 1.0]
            )
            layer_name = f"Cluster {cluster_label}{region_part} Heatmap"
            colormap = Colormap(
                colors=[[0, 0, 0, 0], [rgba[0], rgba[1], rgba[2], 1.0]],
                name=f"cluster_{cluster_label}",
            )
        else:
            layer_name = f"Node Count{region_part} Heatmap"
            colormap = "hot"

        self._progress_label.setText(
            f"{layer_name}: {(volume > 0).sum():,} non-zero voxels"
        )
        contrast_limits = _analysis_heatmap_contrast_limits(volume)

        # Remove existing layer with the same name
        for layer in list(self._viewer.layers):
            if layer.name == layer_name:
                self._viewer.layers.remove(layer)

        # Set scale so the binned depth axis stays spatially aligned
        scale = [1.0, 1.0, 1.0]
        scale[self._pending_heatmap_depth_axis] = float(
            self._pending_heatmap_depth_bin
        )
        metadata = {
            "heatmap_source": True,
            "heatmap_native_grid": self._pending_heatmap_depth_bin == 1,
            "atlas_name": getattr(self._atlas, "atlas_name", None),
            "heatmap_kind": "analysis",
            "heatmap_region": region,
            "heatmap_cluster": cluster_label,
            "depth_bin_factor": self._pending_heatmap_depth_bin,
            "depth_axis": self._pending_heatmap_depth_axis,
            "heatmap_contrast_limits": contrast_limits,
            "heatmap_autocontrast_policy": "stable_full_volume",
        }

        self._heatmap_layer = self._viewer.add_image(
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
        _install_analysis_heatmap_layer_workarounds(self._heatmap_layer)

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
        self._draw_clustermap(self._last_cluster_result)

    def _show_clustermap_message(self, message: str) -> None:
        """Reset the embedded clustermap canvas to a placeholder message."""
        clear = getattr(self._figure, "clear", None)
        if callable(clear):
            clear()
            add_subplot = getattr(self._figure, "add_subplot", None)
            if callable(add_subplot):
                ax = self._figure.add_subplot(111)
                ax.text(
                    0.5,
                    0.5,
                    message,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
        self._canvas.draw()

    def _draw_clustermap(self, result: ClusterResult) -> None:
        """Draw a seaborn clustermap into the embedded canvas."""
        from ..analysis.export import build_clustermap_figure

        self._figure.clear()

        try:
            new_figure = build_clustermap_figure(
                result,
                self._cluster_color_map,
            )

            old_fig = self._figure
            self._figure = new_figure
            self._canvas.figure = self._figure
            self._canvas.draw()
            plt.close(old_fig)
            logger.debug(
                "_draw_clustermap closed previous figure; total_elapsed=%.3fs",
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
            self._schedule_clustermap_layout_refresh()

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

        Called once when clustering completes.  The cached map is reused
        on every subsequent ``apply_cluster_colors`` / button-click so
        that colors are deterministic regardless of which neurons are
        currently rendered.
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

    def _export_y_label(self) -> str:
        """Return the current export y-axis label."""
        return self._export_ylabel_edit.text().strip()

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
            y_label=self._export_y_label(),
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
                y_label=self._export_y_label(),
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
        if self._heatmap_layer is None:
            return
        if self._worker_thread is not None and self._worker_thread.isRunning():
            return
        if self._db is None or self._atlas is None:
            return
        self._run_heatmap_pipeline()

    def apply_cluster_colors(self) -> None:
        """Apply cached cluster colors to currently rendered neuron layers.

        Safe to call at any time. Does nothing if no cluster result exists.
        This is intentionally explicit and should be triggered by the
        'Color Neurons by Cluster' button, not by clustering completion.
        """
        self._color_neurons_by_cluster()

    def _color_neurons_by_cluster(self) -> None:
        """Color existing neuron layers by their cluster assignment.

        Works with the batched single-layer rendering where all neurons
        are merged into one ``Neuron Lines`` and/or ``Neuron Points``
        layer.  Layer metadata (``file_ids``, ``segments_per_neuron``,
        ``file_ids_per_point``) is used to map cluster labels back to
        individual segments/points.
        """
        if (
            self._cluster_color_map is None
            or self._last_cluster_result is None
        ):
            return

        progress_bar = self.__dict__.get("_progress_bar")
        progress_label = self.__dict__.get("_progress_label")
        if progress_bar is not None:
            progress_bar.setVisible(True)
            progress_bar.setRange(0, 0)
        if progress_label is not None:
            progress_label.setText("Applying cluster colors...")
        self._flush_progress_updates()

        color_map = self._cluster_color_map
        clustered_total = len(self._last_cluster_result.neuron_ids)
        n_clusters = self._actual_n_clusters
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

        try:
            self.cluster_colors_updated.emit(
                self._last_cluster_result, color_map
            )
        except RuntimeError:
            pass

        n_gray = n_rendered - n_colored
        msg = (
            f"Applied cluster colors: table {clustered_total} clustered neurons "
            f"({n_clusters} clusters)"
        )
        if n_rendered > 0:
            msg += f"; rendered {n_colored}/{n_rendered} neurons"
            if n_gray > 0:
                msg += f" — {n_gray} neuron(s) not in region shown in gray"
        if progress_bar is not None:
            progress_bar.setVisible(False)
        if progress_label is not None:
            progress_label.setText(msg)

    def _flush_progress_updates(self) -> None:
        """Give Qt a chance to repaint synchronous progress text."""
        try:
            from qtpy.QtWidgets import QApplication
        except ImportError:
            return

        app = QApplication.instance()
        if app is not None:
            app.processEvents()
