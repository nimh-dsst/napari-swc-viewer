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
from types import MethodType
from time import perf_counter
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import QThread, QTimer, Signal
from qtpy.QtGui import QColor, QIcon, QPixmap
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
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

    from ..analysis.clustering import ClusterResult
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
    return "sequence argument must have length equal to input rank" in str(error)


def _analysis_heatmap_ndisplay(layer: Any, response: Any | None = None) -> int | None:
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

    original_reset_contrast_limits = getattr(layer, "reset_contrast_limits", None)
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

    original_update_slice_response = getattr(layer, "_update_slice_response", None)
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
        self._last_cluster_result: ClusterResult | None = None
        self._cluster_color_map: dict[str, list[float]] | None = None
        self._actual_n_clusters: int = 0
        self._heatmap_layer = None
        self._pending_heatmap_cluster: int | None = None  # cluster label for in-flight heatmap
        self._pending_heatmap_region: str | None = None  # region acronym for in-flight heatmap
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
        region_values = regions_df["region_id"] if "region_id" in regions_df else []
        for region_id in list(region_values):
            if region_id is None:
                continue
            if isinstance(region_id, (float, np.floating)) and np.isnan(region_id):
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
        busy = self._worker_thread is not None and self._worker_thread.isRunning()
        self._run_corr_btn.setEnabled(ready and not busy)
        self._run_heat_btn.setEnabled(ready and not busy)
        self._color_by_cluster_btn.setEnabled(self._last_cluster_result is not None)
        self._build_clustermap_btn.setEnabled(
            self._last_cluster_result is not None and not busy
        )

    def _on_thread_finished(self) -> None:
        """Clear worker references after the thread has stopped."""
        self._worker_thread = None
        self._current_worker = None

    def _setup_ui(self) -> None:
        parent_layout = QVBoxLayout(self)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        parent_layout.addWidget(scroll_area)

        content = QWidget()
        scroll_area.setWidget(content)

        layout = QVBoxLayout(content)
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
        self._clustering_method_combo.addItems(["Voxel Correlation", "Soma Location"])
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
            single_select=True,
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
        self._color_by_cluster_btn.clicked.connect(self._color_neurons_by_cluster)
        corr_layout.addWidget(self._color_by_cluster_btn)

        layout.addWidget(self._clustering_section)

        # Set initial visibility
        self._on_clustering_method_changed(self._clustering_method_combo.currentText())

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
        heat_region_selector_layout = self._heat_region_section.content_layout()
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
        self._depth_bin_spin.valueChanged.connect(self._update_voxel_depth_label)
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
        self._clustermap_status_label = QLabel(
            "Run clustering, then click 'Build Dendrogram' to render the cluster map."
        )
        clustermap_layout.addWidget(self._clustermap_status_label)
        self._build_clustermap_btn = QPushButton("Build Dendrogram")
        self._build_clustermap_btn.setEnabled(False)
        self._build_clustermap_btn.clicked.connect(self._build_clustermap_on_demand)
        clustermap_layout.addWidget(self._build_clustermap_btn)
        self._figure = Figure(figsize=(6, 6))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setMinimumHeight(400)
        clustermap_layout.addWidget(self._canvas)
        layout.addWidget(self._clustermap_section)

        layout.addStretch()
        self._refresh_analysis_region_selectors()

    def _format_selected_region_text(
        self,
        selector: RegionSelectorWidget | None,
        *,
        empty_text: str,
    ) -> str:
        """Return a compact summary string for a selector's current region."""
        if selector is None:
            return empty_text

        selected = selector.get_single_selected_region()
        if selected is None:
            return empty_text

        struct_id, acronym = selected
        struct = selector._structure_map.get(struct_id, {})
        name = str(struct.get("name", "")).strip()
        if name and name != acronym:
            return f"{acronym} ({name})"
        return acronym

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
        previous_ids = []
        for selector in selectors:
            selected = selector.get_single_selected_region()
            previous_ids.append(selected[0] if selected is not None else None)

        if self._atlas is None or not self._dataset_region_ids:
            for selector in selectors:
                selector.clear()
            self._update_region_summary_labels()
            return

        allowed_ids = self._analysis_allowed_structure_ids()
        for selector, previous_id in zip(selectors, previous_ids):
            selector.set_allowed_structure_ids(allowed_ids)
            selector.set_atlas(self._atlas)
            selector.select_region_by_id(
                previous_id if previous_id in allowed_ids else None
            )

        self._update_region_summary_labels()

    def _selected_region(self, selector: RegionSelectorWidget | None) -> tuple[int, str] | None:
        """Return the single selected region for a selector, if any."""
        if selector is None:
            return None
        return selector.get_single_selected_region()

    def _selected_cluster_region(self) -> tuple[int, str] | None:
        """Return the currently selected clustering region."""
        return self._selected_region(getattr(self, "_cluster_region_selector", None))

    def _selected_heat_region(self) -> tuple[int, str] | None:
        """Return the currently selected heatmap region."""
        return self._selected_region(getattr(self, "_heat_region_selector", None))

    def _represented_region_ids_for_selection(self, region_id: int) -> list[int]:
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
                path_ids = [int(path_id) for path_id in struct.get("structure_id_path", []) or []]
            except (TypeError, ValueError):
                continue
            if selected_region_id in path_ids:
                represented_ids.append(int(candidate_id))
        return represented_ids

    def _on_cluster_region_selection_changed(self, _acronyms: list[str]) -> None:
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

        selected = self._selected_cluster_region()
        if selected is None:
            self._progress_label.setText("Select a target region.")
            return
        _region_id, region = selected

        dilation = self._dilation_spin.value() / 100.0
        clustering_method = self._clustering_method_combo.currentText()

        if clustering_method == "Soma Location":
            self._run_soma_clustering(region, dilation)
        else:
            self._run_correlation_clustering(region, dilation)

    def _run_correlation_clustering(self, region: str, dilation: float) -> None:
        """Start the voxel correlation + clustering pipeline."""
        from ..workers import CorrelationWorker

        method = self._method_combo.currentText()
        n_clusters = self._n_clusters_spin.value()

        worker = CorrelationWorker(
            parquet_path=self._parquet_path,
            atlas=self._atlas,
            region_acronym=region,
            dilation_fraction=dilation,
            linkage_method=method,
            n_clusters=n_clusters,
        )

        self._start_worker(worker)

    def _run_soma_clustering(self, region: str, dilation: float) -> None:
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
            region_acronym=region,
            dilation_fraction=dilation,
            algorithm=algorithm,
            linkage_method=self._method_combo.currentText(),
            n_clusters=self._n_clusters_spin.value(),
            eps=self._eps_spin.value(),
            min_samples=self._min_samples_spin.value(),
        )

        self._start_worker(worker)

    def _start_worker(self, worker) -> None:
        """Wire up and start a clustering worker in a background thread."""
        thread = QThread()
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_correlation_finished)
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
        self._run_corr_btn.setEnabled(False)
        self._run_heat_btn.setEnabled(False)
        self._build_clustermap_btn.setEnabled(False)
        self._clustermap_status_label.setText(
            "Clustering in progress. Build the dendrogram after results are ready."
        )

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
        self._run_corr_btn.setEnabled(False)
        self._run_heat_btn.setEnabled(False)

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
            f"{cluster_msg}"
        )
        self._clustermap_status_label.setText(
            "Clustering complete. Click 'Build Dendrogram' to render the cluster map."
        )
        self._update_button_states()
        self._update_cluster_filter_combo()
        logger.debug(
            "_on_correlation_finished worker result ready; clustermap render deferred until button click"
        )

        # Notify the neuron table of cluster assignments and colors
        if self._cluster_color_map is not None:
            self.cluster_colors_updated.emit(result, self._cluster_color_map)
        logger.debug(
            "_on_correlation_finished complete: total_elapsed=%.3fs",
            perf_counter() - finish_start,
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
        scale[self._pending_heatmap_depth_axis] = float(self._pending_heatmap_depth_bin)
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

    def _attach_clustermap_figure(self, figure: Figure) -> None:
        """Attach a matplotlib figure to the persistent embedded canvas."""
        logger.debug(
            "_attach_clustermap_figure start: figure_id=%s canvas_id=%s",
            id(figure),
            id(self._canvas),
        )
        figure.set_canvas(self._canvas)
        self._figure = figure
        self._canvas.figure = figure
        update_geometry = getattr(self._canvas, "updateGeometry", None)
        if callable(update_geometry):
            update_geometry()
        logger.debug("_attach_clustermap_figure complete")

    def _clustermap_canvas_pixel_size(self) -> tuple[int, int] | None:
        """Return the current canvas size in pixels, if available."""
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

    def _refresh_clustermap_layout(self) -> None:
        """Resize and redraw the clustermap using the canvas's current geometry."""
        if not getattr(self, "_clustermap_rendered", False):
            logger.debug("_refresh_clustermap_layout skipped: no rendered clustermap")
            return

        size = self._clustermap_canvas_pixel_size()
        if size is None:
            logger.debug(
                "_refresh_clustermap_layout skipped: canvas size unavailable"
            )
            return

        width_px, height_px = size
        dpi_getter = getattr(self._figure, "get_dpi", None)
        dpi = float(dpi_getter()) if callable(dpi_getter) else float(
            getattr(self._figure, "dpi", 100.0) or 100.0
        )
        width_in = max(width_px, 1) / dpi
        height_in = max(height_px, 1) / dpi
        self._figure.set_size_inches(width_in, height_in, forward=False)

        for widget in (
            getattr(self, "_clustermap_section", None),
            getattr(self, "_canvas", None),
        ):
            if widget is None:
                continue
            update_geometry = getattr(widget, "updateGeometry", None)
            if callable(update_geometry):
                update_geometry()

        logger.debug(
            "_refresh_clustermap_layout resized figure: width_px=%d height_px=%d dpi=%.3f",
            width_px,
            height_px,
            dpi,
        )
        draw_idle = getattr(self._canvas, "draw_idle", None)
        if callable(draw_idle):
            draw_idle()

        canvas_start = perf_counter()
        self._canvas.draw()
        logger.debug(
            "_refresh_clustermap_layout canvas draw complete: elapsed=%.3fs",
            perf_counter() - canvas_start,
        )

    def _run_scheduled_clustermap_layout_refresh(self) -> None:
        """Clear the pending flag and perform the deferred layout refresh."""
        self._clustermap_refresh_pending = False
        self._refresh_clustermap_layout()

    def _schedule_clustermap_layout_refresh(self) -> None:
        """Queue a layout-aware clustermap redraw on the next Qt event cycle."""
        if getattr(self, "_clustermap_refresh_pending", False):
            logger.debug(
                "_schedule_clustermap_layout_refresh skipped: refresh already pending"
            )
            return
        self._clustermap_refresh_pending = True
        logger.debug("_schedule_clustermap_layout_refresh queued")
        QTimer.singleShot(0, self._run_scheduled_clustermap_layout_refresh)

    def _on_clustermap_section_expanded_changed(self, expanded: bool) -> None:
        """Refresh clustermap layout when the section becomes visible again."""
        logger.debug(
            "_on_clustermap_section_expanded_changed: expanded=%s rendered=%s",
            expanded,
            getattr(self, "_clustermap_rendered", False),
        )
        if expanded and getattr(self, "_clustermap_rendered", False):
            self._schedule_clustermap_layout_refresh()

    def _draw_clustermap(self, result: ClusterResult) -> None:
        """Draw a seaborn clustermap into the embedded canvas."""
        draw_start = perf_counter()
        self._figure.clear()

        # Build per-neuron cluster color strip for row_colors / col_colors
        cluster_colors = None
        if self._cluster_color_map is not None:
            cluster_colors = [
                self._cluster_color_map.get(nid, [0.5, 0.5, 0.5, 1.0])[:3]
                for nid in result.neuron_ids
            ]

        # Use seaborn clustermap with precomputed linkage
        try:
            logger.debug(
                "_draw_clustermap start: distance_shape=%s distance_dtype=%s distance_nbytes=%s linkage_shape=%s",
                getattr(result.distance_matrix, "shape", None),
                getattr(result.distance_matrix, "dtype", None),
                getattr(result.distance_matrix, "nbytes", None),
                getattr(result.linkage_matrix, "shape", None),
            )
            seaborn_start = perf_counter()
            g = sns.clustermap(
                result.distance_matrix,
                row_linkage=result.linkage_matrix,
                col_linkage=result.linkage_matrix,
                row_colors=cluster_colors,
                col_colors=cluster_colors,
                cmap="coolwarm",
                center=0,
                figsize=(6, 6),
                xticklabels=False,
                yticklabels=False,
            )
            logger.debug(
                "_draw_clustermap seaborn.clustermap complete: elapsed=%.3fs",
                perf_counter() - seaborn_start,
            )

            # Copy the clustermap figure content to our embedded canvas figure
            # seaborn.clustermap creates its own figure, so we replace ours
            old_fig = self._figure
            self._attach_clustermap_figure(g.fig)
            self._clustermap_rendered = True
            logger.debug("_draw_clustermap canvas figure swapped")
            self._schedule_clustermap_layout_refresh()
            logger.debug(
                "_draw_clustermap layout refresh scheduled after figure swap"
            )

            # Close the reference to the old figure
            plt.close(old_fig)
            logger.debug(
                "_draw_clustermap closed previous figure; total_elapsed=%.3fs",
                perf_counter() - draw_start,
            )

        except Exception as e:
            logger.exception("Failed to draw clustermap")
            ax = self._figure.add_subplot(111)
            ax.text(
                0.5, 0.5,
                f"Error drawing clustermap:\n{e}",
                ha="center", va="center",
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

        Safe to call at any time.  Does nothing if no cluster result exists.
        Called automatically after neuron rendering and by the
        'Color Neurons by Cluster' button.
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
        if self._cluster_color_map is None:
            return

        color_map = self._cluster_color_map
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
                        n_colored = sum(1 for fid in unique_fids if fid in color_map)
                    colors = np.array(
                        [color_map.get(fid, default_color)[:4] for fid in fids]
                    )
                    layer.face_color = colors
                    updated += 1

        # Also update the 2D slice projector colors
        if self._slice_projector is not None:
            self._slice_projector.update_neuron_colors(color_map)

        n_gray = n_rendered - n_colored
        msg = f"Colored {n_colored}/{n_rendered} neurons ({n_clusters} clusters)"
        if n_gray > 0:
            msg += f" — {n_gray} neuron(s) not in region shown in gray"
        self._progress_label.setText(msg)
