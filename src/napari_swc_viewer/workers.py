"""Background workers for long-running analysis tasks.

Each worker runs in a QThread and emits progress/finished/error signals
so the napari UI stays responsive. Workers create their own DuckDB
connections since DuckDB connections are not thread-safe.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import QObject, Signal

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas

    from .analysis.clustering import ClusterResult

logger = logging.getLogger(__name__)


class CorrelationWorker(QObject):
    """Compute the full correlation + clustering pipeline in background.

    Steps:
    1. Extract and dilate region mask
    2. Build voxel ID map
    3. Compute pairwise Pearson correlations via DuckDB
    4. Build correlation matrix
    5. Hierarchical clustering

    Signals
    -------
    progress(str, int, int)
        (step_name, current_step, total_steps)
    finished(ClusterResult)
        Emitted with the clustering result on success.
    error(str)
        Emitted with error message on failure.
    """

    progress = Signal(str, int, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        parquet_path: str,
        atlas: BrainGlobeAtlas,
        region_acronym: str,
        dilation_fraction: float = 0.2,
        linkage_method: str = "average",
        n_clusters: int = 5,
    ):
        super().__init__()
        self._parquet_path = parquet_path
        self._atlas = atlas
        self._region_acronym = region_acronym
        self._dilation_fraction = dilation_fraction
        self._linkage_method = linkage_method
        self._n_clusters = n_clusters

    def run(self) -> None:
        """Execute the full pipeline."""
        try:
            import duckdb

            from .analysis.clustering import compute_clustermap_data
            from .analysis.correlation import (
                compute_pearson_correlation_matrix,
                correlation_long_to_matrix,
            )
            from .analysis.mask import get_expanded_region_voxel_ids

            total = 5
            self.progress.emit("Extracting and dilating region mask...", 1, total)
            voxel_id_map = get_expanded_region_voxel_ids(
                self._atlas,
                self._region_acronym,
                self._dilation_fraction,
            )

            self.progress.emit("Computing pairwise correlations...", 2, total)
            conn = duckdb.connect()
            try:
                resolution = float(self._atlas.resolution[0])
                corr_df = compute_pearson_correlation_matrix(
                    conn,
                    self._parquet_path,
                    voxel_id_map,
                    resolution=resolution,
                )
            finally:
                conn.close()

            self.progress.emit("Building correlation matrix...", 3, total)
            mat_df, mat = correlation_long_to_matrix(corr_df)

            self.progress.emit("Clustering...", 4, total)
            result = compute_clustermap_data(
                mat,
                list(mat_df.columns),
                method=self._linkage_method,
                n_clusters=self._n_clusters,
            )

            self.progress.emit("Done", 5, total)
            self.finished.emit(result)

        except Exception as e:
            logger.exception("Correlation pipeline failed")
            self.error.emit(str(e))


class HeatmapWorker(QObject):
    """Build a node-count heatmap volume in the background.

    Signals
    -------
    progress(str, int, int)
        (step_name, current_step, total_steps)
    finished(NDArray)
        Emitted with the 3D volume on success.
    error(str)
        Emitted with error message on failure.
    """

    progress = Signal(str, int, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        parquet_path: str,
        atlas: BrainGlobeAtlas,
        region_acronym: str | None = None,
        file_ids: list[str] | None = None,
    ):
        super().__init__()
        self._parquet_path = parquet_path
        self._atlas = atlas
        self._region_acronym = region_acronym
        self._file_ids = file_ids

    def run(self) -> None:
        """Execute the heatmap pipeline."""
        try:
            import duckdb

            from .analysis.heatmap import build_node_counts_volume

            self.progress.emit("Building heatmap volume...", 1, 2)
            conn = duckdb.connect()
            try:
                volume = build_node_counts_volume(
                    conn,
                    self._parquet_path,
                    self._atlas,
                    region_acronym=self._region_acronym,
                    file_ids=self._file_ids,
                )
            finally:
                conn.close()

            self.progress.emit("Done", 2, 2)
            self.finished.emit(volume)

        except Exception as e:
            logger.exception("Heatmap pipeline failed")
            self.error.emit(str(e))
