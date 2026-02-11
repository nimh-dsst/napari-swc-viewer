"""Background workers for long-running analysis tasks.

Each worker runs in a QThread and emits progress/finished/error signals
so the napari UI stays responsive. Workers create their own DuckDB
connections since DuckDB connections are not thread-safe.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import QObject, Signal

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas

    from .analysis.clustering import ClusterResult

logger = logging.getLogger(__name__)


class ConvertWorker(QObject):
    """Convert SWC files to annotated Parquet in the background.

    Signals
    -------
    progress(str, int, int)
        (message, files_processed, total_files)
    finished(str, int)
        (output_path, n_files_processed)
    error(str)
        Emitted with error message on failure.
    """

    progress = Signal(str, int, int)
    finished = Signal(str, int)
    error = Signal(str)

    def __init__(
        self,
        swc_paths: list[str],
        output_path: str,
        resolution: int = 25,
    ):
        super().__init__()
        self._swc_paths = [Path(p) for p in swc_paths]
        self._output_path = Path(output_path)
        self._resolution = resolution

    def run(self) -> None:
        """Execute the conversion pipeline."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            from .parquet import NEURON_SCHEMA, swc_to_annotated_rows
            from .region import build_region_lookup, setup_allen_sdk

            total = len(self._swc_paths)
            self.progress.emit("Setting up Allen SDK...", 0, total)

            _, annotation_volume, structure_tree = setup_allen_sdk(
                self._resolution
            )
            region_lookup = build_region_lookup(structure_tree)

            all_rows: list[dict] = []
            processed = 0

            for swc_path in self._swc_paths:
                try:
                    self.progress.emit(
                        f"Processing {swc_path.name}...", processed, total
                    )
                    rows = swc_to_annotated_rows(
                        swc_path,
                        annotation_volume,
                        structure_tree,
                        region_lookup,
                        self._resolution,
                    )
                    all_rows.extend(rows)
                    processed += 1
                except Exception as e:
                    logger.error(f"Error processing {swc_path}: {e}")

            if all_rows:
                table = pa.Table.from_pylist(all_rows, schema=NEURON_SCHEMA)
                pq.write_table(table, self._output_path, compression="snappy")

            self.finished.emit(str(self._output_path), processed)

        except Exception as e:
            logger.exception("SWC-to-Parquet conversion failed")
            self.error.emit(str(e))


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
