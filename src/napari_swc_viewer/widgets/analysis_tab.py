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
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import QThread, Qt
from qtpy.QtGui import QColor, QIcon, QPixmap
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import napari
    from brainglobe_atlasapi import BrainGlobeAtlas

    from ..analysis.clustering import ClusterResult
    from ..db import NeuronDatabase

logger = logging.getLogger(__name__)


class AnalysisTabWidget(QWidget):
    """Widget providing the Analysis tab UI.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    parent : QWidget, optional
        Parent widget.
    """

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
        self._slice_projector = None
        self._setup_ui()

    def set_database(self, db: NeuronDatabase) -> None:
        """Set the database connection."""
        self._db = db
        self._parquet_path = str(db.parquet_path)
        self._update_button_states()

    def set_atlas(self, atlas: BrainGlobeAtlas) -> None:
        """Set the atlas instance."""
        self._atlas = atlas
        self._update_button_states()

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

    def _on_thread_finished(self) -> None:
        """Clear worker references after the thread has stopped."""
        self._worker_thread = None
        self._current_worker = None

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- Clustering group ---
        corr_group = QGroupBox("Clustering")
        corr_layout = QVBoxLayout(corr_group)

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
        self._region_combo = QComboBox()
        self._region_combo.setEditable(True)
        self._region_combo.addItems(["GPe", "CP", "VISp", "MOp", "SSp"])
        region_row.addWidget(self._region_combo)
        corr_layout.addLayout(region_row)

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

        layout.addWidget(corr_group)

        # Set initial visibility
        self._on_clustering_method_changed(self._clustering_method_combo.currentText())

        # --- Node Count Heatmap group ---
        heat_group = QGroupBox("Node Count Heatmap")
        heat_layout = QVBoxLayout(heat_group)

        heat_region_row = QHBoxLayout()
        heat_region_row.addWidget(QLabel("Region filter:"))
        self._heat_region_combo = QComboBox()
        self._heat_region_combo.setEditable(True)
        self._heat_region_combo.addItems(["", "CP", "GPe", "VISp"])
        heat_region_row.addWidget(self._heat_region_combo)
        heat_layout.addLayout(heat_region_row)

        # Cluster filter
        cluster_filter_row = QHBoxLayout()
        cluster_filter_row.addWidget(QLabel("Cluster filter:"))
        self._heat_cluster_combo = QComboBox()
        self._heat_cluster_combo.addItem("All neurons")
        self._heat_cluster_combo.setEnabled(False)
        cluster_filter_row.addWidget(self._heat_cluster_combo)
        heat_layout.addLayout(cluster_filter_row)

        self._run_heat_btn = QPushButton("Build Heatmap Volume")
        self._run_heat_btn.setEnabled(False)
        self._run_heat_btn.clicked.connect(self._run_heatmap_pipeline)
        heat_layout.addWidget(self._run_heat_btn)

        layout.addWidget(heat_group)

        # --- Progress bar ---
        self._progress_label = QLabel("")
        layout.addWidget(self._progress_label)
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        # --- Matplotlib canvas for clustermap ---
        self._figure = Figure(figsize=(6, 6))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setMinimumHeight(400)
        layout.addWidget(self._canvas)

        layout.addStretch()

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

        region = self._region_combo.currentText().strip()
        if not region:
            self._progress_label.setText("Please enter a target region acronym")
            return

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

        thread.start()

    def _run_heatmap_pipeline(self) -> None:
        """Start the heatmap pipeline in a background thread."""
        if self._db is None or self._atlas is None:
            return

        if self._worker_thread is not None and self._worker_thread.isRunning():
            return

        from ..workers import HeatmapWorker

        region = self._heat_region_combo.currentText().strip() or None
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

        worker = HeatmapWorker(
            parquet_path=self._parquet_path,
            atlas=self._atlas,
            region_acronym=region,
            file_ids=file_ids,
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
        self._last_cluster_result = result
        self._build_cluster_color_map()
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
        self._update_button_states()
        self._update_cluster_filter_combo()

        # Draw clustermap
        self._draw_clustermap(result)

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

        # Remove existing layer with the same name
        for layer in list(self._viewer.layers):
            if layer.name == layer_name:
                self._viewer.layers.remove(layer)

        self._heatmap_layer = self._viewer.add_image(
            volume,
            name=layer_name,
            colormap=colormap,
            blending="additive",
            rendering="mip",
            opacity=0.7,
            visible=True,
        )

    def _on_error(self, message: str) -> None:
        """Handle worker errors."""
        self._progress_bar.setVisible(False)
        self._progress_label.setText(f"Error: {message}")
        self._update_button_states()
        logger.error(f"Analysis pipeline error: {message}")

    def _draw_clustermap(self, result: ClusterResult) -> None:
        """Draw a seaborn clustermap into the embedded canvas."""
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

            # Copy the clustermap figure content to our embedded canvas figure
            # seaborn.clustermap creates its own figure, so we replace ours
            old_fig = self._figure
            self._figure = g.fig
            self._canvas.figure = self._figure
            self._canvas.draw()

            # Close the reference to the old figure
            plt.close(old_fig)

        except Exception as e:
            logger.exception("Failed to draw clustermap")
            ax = self._figure.add_subplot(111)
            ax.text(
                0.5, 0.5,
                f"Error drawing clustermap:\n{e}",
                ha="center", va="center",
                transform=ax.transAxes,
            )
            self._canvas.draw()

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
