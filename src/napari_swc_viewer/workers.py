"""Background workers for long-running analysis tasks.

Each worker runs in a QThread and emits progress/finished/error signals
so the napari UI stays responsive. Workers create their own DuckDB
connections since DuckDB connections are not thread-safe.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import QObject, Signal

from .logging_utils import startup_timing

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas

    from .analysis.clustering import ClusterRegionSelection, ClusterResult

logger = logging.getLogger(__name__)


def cached_brainglobe_atlas_dir(
    atlas_name: str,
    *,
    brainglobe_dir: str | Path | None = None,
    config_dir: str | Path | None = None,
) -> Path | None:
    """Return the single local cache directory for an atlas, if available."""
    if brainglobe_dir is None:
        try:
            from brainglobe_atlasapi import config

            conf = config.read_config(config_dir)
            brainglobe_dir = conf["default_dirs"]["brainglobe_dir"]
        except Exception:
            logger.debug(
                "Failed to resolve BrainGlobe cache directory.",
                exc_info=True,
            )
            return None

    root = Path(brainglobe_dir).expanduser()
    candidates = sorted(
        path for path in root.glob(f"{atlas_name}_v*") if path.is_dir()
    )
    if len(candidates) != 1:
        return None
    return candidates[0]


def load_cached_brainglobe_atlas(atlas_name: str, atlas_dir: str | Path):
    """Load an already-cached BrainGlobe atlas without remote checks/downloads."""
    from brainglobe_atlasapi.core import Atlas

    atlas = Atlas(Path(atlas_dir))
    atlas.atlas_name = atlas_name
    return atlas


class CachedAtlasLoadWorker(QObject):
    """Load a locally cached BrainGlobe atlas in the background."""

    finished = Signal(object)
    error = Signal(str)

    def __init__(self, atlas_name: str, atlas_dir: str | Path):
        super().__init__()
        self._atlas_name = str(atlas_name)
        self._atlas_dir = Path(atlas_dir)

    def run(self) -> None:
        """Load the cached atlas and emit it."""
        try:
            with startup_timing(
                logger,
                "cached_atlas_load_worker",
                atlas=self._atlas_name,
                atlas_dir=self._atlas_dir,
            ):
                atlas = load_cached_brainglobe_atlas(
                    self._atlas_name,
                    self._atlas_dir,
                )
            self.finished.emit(atlas)
        except Exception as e:
            logger.exception("Cached atlas load failed")
            self.error.emit(str(e))


def _attach_cluster_run_metadata(
    result,
    *,
    atlas,
    parquet_path: str,
    region_selection,
    analysis_method: str,
    clustering_algorithm: str,
    distance_metric: str,
    clustering_linkage: str | None,
    dendrogram_linkage: str | None,
    dilation_fraction: float,
    requested_cluster_count: int | None,
    dbscan_eps: float | None = None,
    dbscan_min_samples: int | None = None,
):
    """Populate the cluster result with reproducibility metadata."""
    from .analysis.clustering import ClusterRunMetadata

    result.metadata = ClusterRunMetadata.from_region_selection(
        region_selection=region_selection,
        analysis_method=analysis_method,
        clustering_algorithm=clustering_algorithm,
        distance_metric=distance_metric,
        clustering_linkage=clustering_linkage,
        dendrogram_linkage=dendrogram_linkage,
        dilation_fraction=dilation_fraction,
        requested_cluster_count=requested_cluster_count,
        actual_cluster_count=len(np.unique(result.labels)),
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        atlas_name=getattr(atlas, "atlas_name", None),
        atlas_resolution_um=tuple(float(value) for value in getattr(atlas, "resolution", ()) or ()),
        source_parquet_path=str(Path(parquet_path)),
        dendrogram_leaf_order=[int(value) for value in result.reorder_indices.tolist()],
    )
    return result


class ConvertWorker(QObject):
    """Convert SWC files to annotated Parquet in the background.

    Signals
    -------
    progress(str, int, int)
        (message, files_processed, total_files)
    finished(str, object)
        (output_path, BatchParquetConversionSummary)
    error(str)
        Emitted with error message on failure.
    """

    progress = Signal(str, int, int)
    finished = Signal(str, object)
    error = Signal(str)

    def __init__(
        self,
        swc_paths: str | Path | list[str | Path],
        output_path: str,
        resolution: int = 25,
        hemisphere: str | None = None,
        atlas_name: str = "allen_mouse_25um",
        coord_axis: int = 2,
        recursive: bool = False,
        n_workers: int = 1,
        source_mode: str | None = None,
        cached_atlas: BrainGlobeAtlas | None = None,
        use_cached_annotation: bool = False,
    ):
        super().__init__()
        self._swc_source = (
            Path(swc_paths)
            if isinstance(swc_paths, (str, Path))
            else [Path(p) for p in swc_paths]
        )
        self._output_path = Path(output_path)
        self._resolution = resolution
        self._hemisphere = hemisphere
        self._atlas_name = atlas_name
        self._coord_axis = coord_axis
        self._recursive = recursive
        self._n_workers = n_workers
        self._source_mode = source_mode or (
            "files"
            if isinstance(self._swc_source, list)
            else "directory"
            if recursive
            else "path"
        )
        self._cached_atlas = cached_atlas
        self._use_cached_annotation = use_cached_annotation

    def _source_log_value(self) -> str:
        """Return a compact source description for conversion timing logs."""
        if isinstance(self._swc_source, list):
            count = len(self._swc_source)
            if count <= 50:
                return str([str(path) for path in self._swc_source])
            sample = self._swc_source[:10] + self._swc_source[-10:]
            return f"{count} files sample={[str(path) for path in sample]}"
        return str(self._swc_source)

    def _cached_atlas_conversion_inputs(self):
        """Return optional cached atlas-derived conversion inputs."""
        if self._cached_atlas is None:
            return None, None, None

        from .hemisphere import get_atlas_midline
        from .region import BrainGlobeStructureTree, build_region_lookup

        cached_start = perf_counter()
        cached_midline = None
        if self._hemisphere is not None:
            cached_midline = get_atlas_midline(
                self._cached_atlas,
                self._coord_axis,
            )

        cached_annotation_volume = None
        cached_region_lookup = None
        if self._use_cached_annotation:
            cached_annotation_volume = self._cached_atlas.annotation
            cached_region_lookup = build_region_lookup(
                BrainGlobeStructureTree(self._cached_atlas)
            )

        logger.debug(
            (
                "swc_conversion_worker_cached_atlas_ready source_mode=%s "
                "atlas=%s midline=%s annotation=%s region_lookup=%s "
                "elapsed_s=%.6f"
            ),
            self._source_mode,
            getattr(self._cached_atlas, "atlas_name", None),
            cached_midline,
            cached_annotation_volume is not None,
            len(cached_region_lookup) if cached_region_lookup is not None else 0,
            perf_counter() - cached_start,
        )
        return cached_midline, cached_annotation_volume, cached_region_lookup

    def run(self) -> None:
        """Execute the conversion pipeline."""
        run_start = perf_counter()
        try:
            from .parquet import batch_convert_swc_to_parquet

            total = len(self._swc_source) if isinstance(self._swc_source, list) else 0
            logger.debug(
                (
                    "swc_conversion_worker_start source_mode=%s source=%s "
                    "file_count=%s recursive=%s output=%s resolution=%s "
                    "hemisphere=%s atlas=%s coord_axis=%d n_workers=%d "
                    "cached_atlas=%s cached_annotation=%s"
                ),
                self._source_mode,
                self._source_log_value(),
                total if isinstance(self._swc_source, list) else "unknown",
                self._recursive,
                self._output_path,
                self._resolution,
                self._hemisphere,
                self._atlas_name,
                self._coord_axis,
                self._n_workers,
                self._cached_atlas is not None,
                self._use_cached_annotation,
            )
            initial_message = (
                "Searching for SWC files..."
                if not isinstance(self._swc_source, list)
                else "Preparing SWC-to-Parquet conversion..."
            )
            self.progress.emit(
                initial_message,
                0,
                total,
            )
            batch_start = perf_counter()
            cached_midline, cached_annotation_volume, cached_region_lookup = (
                self._cached_atlas_conversion_inputs()
            )
            summary = batch_convert_swc_to_parquet(
                self._swc_source,
                self._output_path,
                recursive=self._recursive,
                hemisphere=self._hemisphere,
                atlas_name=self._atlas_name,
                coord_axis=self._coord_axis,
                midline=cached_midline,
                annotate_regions=True,
                resolution=self._resolution,
                n_workers=self._n_workers,
                annotation_volume=cached_annotation_volume,
                region_lookup=cached_region_lookup,
                source_mode=self._source_mode,
                progress_callback=self.progress.emit,
            )
            batch_elapsed_s = perf_counter() - batch_start
            logger.debug(
                (
                    "swc_conversion_worker_batch_ok source_mode=%s "
                    "elapsed_s=%.6f discovered=%d processed=%d failed=%d "
                    "rows=%d flipped=%d"
                ),
                self._source_mode,
                batch_elapsed_s,
                summary.discovered_files,
                summary.processed_files,
                summary.failed_files,
                summary.rows_written,
                summary.flipped_files,
            )
            for failed_path, failure_message in summary.failures:
                logger.debug(
                    (
                        "swc_conversion_worker_file_failure source_mode=%s "
                        "file=%s error=%s"
                    ),
                    self._source_mode,
                    failed_path,
                    failure_message,
                )
            if summary.discovered_files == 0:
                raise ValueError("No SWC files found")
            if summary.processed_files == 0:
                raise ValueError("No SWC files were successfully processed")

            total = summary.discovered_files
            self.progress.emit("Finalizing Parquet...", total, total)
            logger.debug(
                "swc_conversion_worker_finished source_mode=%s elapsed_s=%.6f",
                self._source_mode,
                perf_counter() - run_start,
            )
            self.finished.emit(str(self._output_path), summary)

        except Exception as e:
            logger.exception(
                "SWC-to-Parquet conversion failed source_mode=%s elapsed_s=%.6f",
                self._source_mode,
                perf_counter() - run_start,
            )
            self.error.emit(str(e))


class AppendPointFileWorker(QObject):
    """Append a point CSV or point Parquet into an existing point Parquet."""

    progress = Signal(str, int, int)
    finished = Signal(str, object)
    error = Signal(str)

    def __init__(
        self,
        input_path: str,
        mapping_path: str | None,
        parquet_path: str,
        output_path: str | None = None,
    ):
        super().__init__()
        self._input_path = Path(input_path)
        self._mapping_path = Path(mapping_path) if mapping_path is not None else None
        self._parquet_path = Path(parquet_path)
        self._output_path = (
            Path(output_path) if output_path is not None else self._parquet_path
        )

    def run(self) -> None:
        """Execute the append pipeline."""
        try:
            from .point_import import (
                append_point_csv_to_parquet,
                append_point_parquet_to_parquet,
            )

            if self._input_path.suffix.lower() == ".parquet":
                self.progress.emit("Validating point Parquet schemas...", 1, 3)
                summary = append_point_parquet_to_parquet(
                    self._input_path,
                    self._parquet_path,
                    self._output_path,
                )
            else:
                self.progress.emit("Validating point CSV and target Parquet...", 1, 3)
                summary = append_point_csv_to_parquet(
                    self._input_path,
                    self._mapping_path,
                    self._parquet_path,
                    self._output_path,
                )
            self.progress.emit("Refreshing saved point Parquet...", 2, 3)
            self.progress.emit("Done", 3, 3)
            self.finished.emit(str(self._output_path), summary)

        except Exception as e:
            logger.exception("Point file append failed")
            self.error.emit(str(e))


AppendPointCSVWorker = AppendPointFileWorker


class ConvertPointCSVWorker(QObject):
    """Convert one or more point CSV files into a point Parquet in the background."""

    progress = Signal(str, int, int)
    finished = Signal(str, object)
    error = Signal(str)

    def __init__(
        self,
        csv_paths: list[str],
        output_path: str,
        mapping_path: str | None = None,
    ):
        super().__init__()
        self._csv_paths = [Path(path) for path in csv_paths]
        self._output_path = Path(output_path)
        self._mapping_path = Path(mapping_path) if mapping_path is not None else None

    def run(self) -> None:
        """Execute the point-CSV conversion pipeline."""
        try:
            from .point_import import convert_point_csv_files_to_parquet

            summary = convert_point_csv_files_to_parquet(
                self._csv_paths,
                self._output_path,
                self._mapping_path,
                progress_callback=self.progress.emit,
            )
            self.finished.emit(str(self._output_path), summary)

        except Exception as e:
            logger.exception("Point CSV conversion failed")
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
        region_selection: ClusterRegionSelection,
        dilation_fraction: float = 0.2,
        linkage_method: str = "average",
        n_clusters: int = 5,
        file_ids: list[str] | None = None,
    ):
        super().__init__()
        self._parquet_path = parquet_path
        self._atlas = atlas
        self._region_selection = region_selection
        self._dilation_fraction = dilation_fraction
        self._linkage_method = linkage_method
        self._n_clusters = n_clusters
        self._file_ids = None if file_ids is None else [str(file_id) for file_id in file_ids]

    def run(self) -> None:
        """Execute the full pipeline."""
        try:
            import duckdb

            from .analysis.clustering import compute_clustermap_data
            from .analysis.correlation import (
                compute_pearson_correlation_matrix,
                correlation_long_to_matrix,
            )
            from .analysis.mask import get_expanded_region_voxel_ids_for_regions

            total = 5
            self.progress.emit("Extracting and dilating region mask...", 1, total)
            voxel_id_map = get_expanded_region_voxel_ids_for_regions(
                self._atlas,
                self._region_selection.selected_region_acronyms,
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
                    file_ids=self._file_ids,
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
            _attach_cluster_run_metadata(
                result,
                atlas=self._atlas,
                parquet_path=self._parquet_path,
                region_selection=self._region_selection,
                analysis_method="voxel_correlation",
                clustering_algorithm="hierarchical",
                distance_metric="one_minus_pearson_r",
                clustering_linkage=self._linkage_method,
                dendrogram_linkage=self._linkage_method,
                dilation_fraction=self._dilation_fraction,
                requested_cluster_count=self._n_clusters,
            )

            self.progress.emit("Done", 5, total)
            self.finished.emit(result)

        except Exception as e:
            logger.exception("Correlation pipeline failed")
            self.error.emit(str(e))


class SomaClusterWorker(QObject):
    """Cluster neurons by soma location in a background thread.

    Steps:
    1. Build expanded region mask
    2. Query soma locations from parquet
    3. Filter somas to those inside the expanded region
    4. Cluster using the chosen algorithm

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
        region_selection: ClusterRegionSelection,
        dilation_fraction: float = 0.2,
        algorithm: str = "hierarchical",
        linkage_method: str = "ward",
        n_clusters: int = 5,
        eps: float = 100.0,
        min_samples: int = 5,
        file_ids: list[str] | None = None,
    ):
        super().__init__()
        self._parquet_path = parquet_path
        self._atlas = atlas
        self._region_selection = region_selection
        self._dilation_fraction = dilation_fraction
        self._algorithm = algorithm
        self._linkage_method = linkage_method
        self._n_clusters = n_clusters
        self._eps = eps
        self._min_samples = min_samples
        self._file_ids = None if file_ids is None else [str(file_id) for file_id in file_ids]

    def run(self) -> None:
        """Execute the soma clustering pipeline."""
        try:
            import duckdb

            from .analysis.clustering import (
                cluster_somas_dbscan,
                cluster_somas_hierarchical,
                cluster_somas_kmeans,
            )
            from .analysis.mask import get_expanded_region_voxel_ids_for_regions

            total = 4
            run_start = perf_counter()
            resolution = float(self._atlas.resolution[0])
            logger.debug(
                "SomaClusterWorker start: algorithm=%s linkage=%s n_clusters=%d eps=%s min_samples=%d region=%s dilation_fraction=%.3f resolution=%.3f parquet=%s",
                self._algorithm,
                self._linkage_method,
                self._n_clusters,
                self._eps,
                self._min_samples,
                ",".join(self._region_selection.selected_region_acronyms),
                self._dilation_fraction,
                resolution,
                self._parquet_path,
            )
            self.progress.emit("Extracting and dilating region mask...", 1, total)
            mask_start = perf_counter()
            voxel_id_map = get_expanded_region_voxel_ids_for_regions(
                self._atlas,
                self._region_selection.selected_region_acronyms,
                self._dilation_fraction,
            )
            logger.debug(
                "SomaClusterWorker mask ready: shape=%s dtype=%s elapsed=%.3fs",
                voxel_id_map.shape,
                voxel_id_map.dtype,
                perf_counter() - mask_start,
            )

            self.progress.emit("Querying soma locations...", 2, total)
            parquet_escaped = str(self._parquet_path).replace("\\", "/")
            Z, Y, X = voxel_id_map.shape

            query_start = perf_counter()
            conn = duckdb.connect()
            try:
                # Query soma locations (type=1) grouped by file
                soma_df = conn.execute(f"""
                    SELECT
                        file_id,
                        AVG(x) AS x, AVG(y) AS y, AVG(z) AS z
                    FROM read_parquet('{parquet_escaped}')
                    WHERE type = 1
                    GROUP BY file_id
                    ORDER BY file_id
                """).fetchdf()
            finally:
                conn.close()
            logger.debug(
                "SomaClusterWorker soma query complete: rows=%d columns=%s elapsed=%.3fs",
                len(soma_df),
                list(soma_df.columns),
                perf_counter() - query_start,
            )

            if self._file_ids is not None:
                scope_ids = set(self._file_ids)
                soma_df = soma_df[
                    soma_df["file_id"].astype(str).isin(scope_ids)
                ].reset_index(drop=True)
                logger.debug(
                    "SomaClusterWorker current-table scope applied: rows=%d",
                    len(soma_df),
                )

            if soma_df.empty:
                self.error.emit("No soma nodes found in the dataset.")
                return

            # Convert soma coordinates to voxel indices and filter to region
            filter_start = perf_counter()
            coords = soma_df[["x", "y", "z"]].values
            # Axis mapping matches correlation.py: x->zi, y->yi, z->xi
            zi = np.floor(coords[:, 0] / resolution).astype(int)
            yi = np.floor(coords[:, 1] / resolution).astype(int)
            xi = np.floor(coords[:, 2] / resolution).astype(int)

            in_bounds = (
                (xi >= 0) & (xi < X)
                & (yi >= 0) & (yi < Y)
                & (zi >= 0) & (zi < Z)
            )
            voxel_ids = np.full(len(coords), -1, dtype=np.int32)
            voxel_ids[in_bounds] = voxel_id_map[
                zi[in_bounds], yi[in_bounds], xi[in_bounds]
            ]
            in_region = voxel_ids >= 0

            filtered_coords = coords[in_region]
            filtered_ids = soma_df["file_id"].values[in_region].tolist()

            logger.info(
                f"Soma filtering: {len(coords)} total somas, "
                f"{len(filtered_ids)} in region(s) "
                f"'{', '.join(self._region_selection.selected_region_acronyms)}'"
            )
            logger.debug(
                "SomaClusterWorker filtering complete: total=%d in_bounds=%d in_region=%d filtered_shape=%s filtered_dtype=%s elapsed=%.3fs",
                len(coords),
                int(in_bounds.sum()),
                int(in_region.sum()),
                filtered_coords.shape,
                filtered_coords.dtype,
                perf_counter() - filter_start,
            )

            if len(filtered_ids) < 2:
                self.error.emit(
                    f"Only {len(filtered_ids)} soma(s) found in "
                    f"'{', '.join(self._region_selection.selected_region_acronyms)}' "
                    "— need at least 2 for clustering."
                )
                return

            self.progress.emit(f"Clustering {len(filtered_ids)} somas ({self._algorithm})...", 3, total)

            if self._algorithm == "hierarchical":
                result = cluster_somas_hierarchical(
                    filtered_coords, filtered_ids,
                    method=self._linkage_method,
                    n_clusters=self._n_clusters,
                )
                clustering_linkage = self._linkage_method
                dendrogram_linkage = self._linkage_method
                requested_cluster_count = self._n_clusters
            elif self._algorithm == "kmeans":
                result = cluster_somas_kmeans(
                    filtered_coords, filtered_ids,
                    n_clusters=self._n_clusters,
                )
                clustering_linkage = None
                dendrogram_linkage = "average"
                requested_cluster_count = self._n_clusters
            elif self._algorithm == "dbscan":
                result = cluster_somas_dbscan(
                    filtered_coords, filtered_ids,
                    eps=self._eps,
                    min_samples=self._min_samples,
                )
                clustering_linkage = None
                dendrogram_linkage = "average"
                requested_cluster_count = None
            else:
                self.error.emit(f"Unknown algorithm: {self._algorithm}")
                return

            _attach_cluster_run_metadata(
                result,
                atlas=self._atlas,
                parquet_path=self._parquet_path,
                region_selection=self._region_selection,
                analysis_method="soma_location",
                clustering_algorithm=self._algorithm,
                distance_metric="euclidean_um",
                clustering_linkage=clustering_linkage,
                dendrogram_linkage=dendrogram_linkage,
                dilation_fraction=self._dilation_fraction,
                requested_cluster_count=requested_cluster_count,
                dbscan_eps=self._eps if self._algorithm == "dbscan" else None,
                dbscan_min_samples=self._min_samples if self._algorithm == "dbscan" else None,
            )

            self.progress.emit("Done", 4, total)
            logger.debug(
                "SomaClusterWorker finished: total_elapsed=%.3fs neuron_count=%d",
                perf_counter() - run_start,
                len(result.neuron_ids),
            )
            self.finished.emit(result)

        except Exception as e:
            logger.exception("Soma clustering pipeline failed")
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
        region_ids: list[int] | None = None,
        file_ids: list[str] | None = None,
        depth_bin_factor: int = 1,
        depth_axis: int = 0,
    ):
        super().__init__()
        self._parquet_path = parquet_path
        self._atlas = atlas
        self._region_ids = region_ids
        self._file_ids = file_ids
        self._depth_bin_factor = depth_bin_factor
        self._depth_axis = depth_axis

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
                    region_ids=self._region_ids,
                    file_ids=self._file_ids,
                    depth_bin_factor=self._depth_bin_factor,
                    depth_axis=self._depth_axis,
                )
            finally:
                conn.close()

            self.progress.emit("Done", 2, 2)
            self.finished.emit(volume)

        except Exception as e:
            logger.exception("Heatmap pipeline failed")
            self.error.emit(str(e))


class AnalysisExportWorker(QObject):
    """Export analysis outputs that benefit from background execution."""

    progress = Signal(str, int, int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        export_kind: str,
        output_path: str,
        cluster_result: ClusterResult,
        cluster_color_map: dict[str, list[float]] | None = None,
        *,
        figure_title: str = "",
        x_label: str = "",
        y_label: str = "",
    ):
        super().__init__()
        self._export_kind = str(export_kind)
        self._output_path = Path(output_path)
        self._cluster_result = cluster_result
        self._cluster_color_map = cluster_color_map or {}
        self._figure_title = figure_title
        self._x_label = x_label
        self._y_label = y_label

    def run(self) -> None:
        """Execute one analysis export."""
        try:
            from .analysis.export import (
                export_cluster_workbook,
                export_distance_workbook,
                export_extended_parquet,
            )

            if self._export_kind == "cluster_workbook":
                export_cluster_workbook(
                    self._output_path,
                    self._cluster_result,
                    self._cluster_color_map,
                    figure_title=self._figure_title,
                    x_label=self._x_label,
                    y_label=self._y_label,
                    progress_callback=self.progress.emit,
                )
            elif self._export_kind == "distance_workbook":
                export_distance_workbook(
                    self._output_path,
                    self._cluster_result,
                    self._cluster_color_map,
                    figure_title=self._figure_title,
                    x_label=self._x_label,
                    y_label=self._y_label,
                    progress_callback=self.progress.emit,
                )
            elif self._export_kind == "extended_parquet":
                export_extended_parquet(
                    self._output_path,
                    self._cluster_result,
                    progress_callback=self.progress.emit,
                )
            else:
                raise ValueError(f"Unknown analysis export kind: {self._export_kind}")

            self.finished.emit(str(self._output_path))

        except Exception as e:
            logger.exception("Analysis export failed")
            self.error.emit(str(e))
