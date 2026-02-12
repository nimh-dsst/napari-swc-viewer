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
from qtpy.QtWidgets import (
    QComboBox,
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
        self._heatmap_layer = None
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

        # --- Correlation Clustering group ---
        corr_group = QGroupBox("Correlation Clustering")
        corr_layout = QVBoxLayout(corr_group)

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
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Linkage:"))
        self._method_combo = QComboBox()
        self._method_combo.addItems(["average", "ward", "complete", "single"])
        method_row.addWidget(self._method_combo)
        corr_layout.addLayout(method_row)

        # Number of clusters
        clusters_row = QHBoxLayout()
        clusters_row.addWidget(QLabel("Clusters:"))
        self._n_clusters_spin = QSpinBox()
        self._n_clusters_spin.setRange(2, 50)
        self._n_clusters_spin.setValue(5)
        clusters_row.addWidget(self._n_clusters_spin)
        corr_layout.addLayout(clusters_row)

        # Run button
        self._run_corr_btn = QPushButton("Compute Correlation + Cluster")
        self._run_corr_btn.setEnabled(False)
        self._run_corr_btn.clicked.connect(self._run_correlation_pipeline)
        corr_layout.addWidget(self._run_corr_btn)

        # Color neurons by cluster
        self._color_by_cluster_btn = QPushButton("Color Neurons by Cluster")
        self._color_by_cluster_btn.setEnabled(False)
        self._color_by_cluster_btn.clicked.connect(self._color_neurons_by_cluster)
        corr_layout.addWidget(self._color_by_cluster_btn)

        layout.addWidget(corr_group)

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

    def _run_correlation_pipeline(self) -> None:
        """Start the correlation + clustering pipeline in a background thread."""
        if self._db is None or self._atlas is None:
            return

        if self._worker_thread is not None and self._worker_thread.isRunning():
            return

        from ..workers import CorrelationWorker

        region = self._region_combo.currentText().strip()
        if not region:
            self._progress_label.setText("Please enter a target region acronym")
            return

        dilation = self._dilation_spin.value() / 100.0
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

        worker = HeatmapWorker(
            parquet_path=self._parquet_path,
            atlas=self._atlas,
            region_acronym=region,
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
        self._progress_bar.setVisible(False)
        self._progress_label.setText(
            f"Clustering complete: {len(result.neuron_ids)} neurons, "
            f"{int(result.labels.max())} clusters"
        )
        self._update_button_states()

        # Draw clustermap
        self._draw_clustermap(result)

    def _on_heatmap_finished(self, volume: np.ndarray) -> None:
        """Handle completed heatmap pipeline."""
        self._progress_bar.setVisible(False)
        self._progress_label.setText(
            f"Heatmap complete: {(volume > 0).sum():,} non-zero voxels"
        )
        self._update_button_states()

        # Add as napari image layer
        layer_name = "Node Count Heatmap"
        # Remove existing heatmap layer
        for layer in list(self._viewer.layers):
            if layer.name == layer_name:
                self._viewer.layers.remove(layer)

        self._heatmap_layer = self._viewer.add_image(
            volume,
            name=layer_name,
            colormap="hot",
            blending="additive",
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

        # Use seaborn clustermap with precomputed linkage
        try:
            g = sns.clustermap(
                result.distance_matrix,
                row_linkage=result.linkage_matrix,
                col_linkage=result.linkage_matrix,
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

    def _color_neurons_by_cluster(self) -> None:
        """Color existing neuron layers by their cluster assignment.

        Works with the batched single-layer rendering where all neurons
        are merged into one ``Neuron Lines`` and/or ``Neuron Points``
        layer.  Layer metadata (``file_ids``, ``segments_per_neuron``,
        ``file_ids_per_point``) is used to map cluster labels back to
        individual segments/points.
        """
        if self._last_cluster_result is None:
            return

        result = self._last_cluster_result
        n_clusters = int(result.labels.max())
        cmap = plt.get_cmap("tab10" if n_clusters <= 10 else "tab20")

        # Build neuron_id -> RGBA color mapping
        color_map: dict[str, list[float]] = {}
        for neuron_id, label in zip(result.neuron_ids, result.labels):
            color_map[neuron_id] = list(cmap((label - 1) / n_clusters))

        default_color = [0.5, 0.5, 0.5, 1.0]
        updated = 0

        for layer in self._viewer.layers:
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
                    updated += 1

            elif layer.name == "Neuron Points":
                meta = layer.metadata or {}
                fids = meta.get("file_ids_per_point", [])
                if fids:
                    colors = np.array(
                        [color_map.get(fid, default_color)[:4] for fid in fids]
                    )
                    layer.face_color = colors
                    updated += 1

        # Also update the 2D slice projector colors
        if self._slice_projector is not None:
            self._slice_projector.update_neuron_colors(color_map)

        self._progress_label.setText(
            f"Colored {updated} layer(s) by cluster ({n_clusters} clusters)"
        )
