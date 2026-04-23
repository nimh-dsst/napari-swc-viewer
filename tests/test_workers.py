"""Tests for background workers."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

from napari_swc_viewer.analysis.clustering import ClusterRegionSelection, ClusterResult
from napari_swc_viewer.point_import import PointParquetAppendSummary
from napari_swc_viewer.parquet import BatchParquetConversionSummary


class _BoundSignal:
    """Minimal Qt-like signal implementation for worker tests."""

    def __init__(self) -> None:
        self._callbacks: list = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, *args) -> None:
        for callback in list(self._callbacks):
            callback(*args)


class _Signal:
    """Descriptor that mimics ``qtpy.QtCore.Signal`` enough for tests."""

    def __init__(self, *_args, **_kwargs) -> None:
        self._storage_name = ""

    def __set_name__(self, _owner, name: str) -> None:
        self._storage_name = f"__signal_{name}"

    def __get__(self, instance, _owner):
        if instance is None:
            return self
        if self._storage_name not in instance.__dict__:
            instance.__dict__[self._storage_name] = _BoundSignal()
        return instance.__dict__[self._storage_name]


class _QObject:
    """Minimal QObject stand-in."""

    def __init__(self, *_args, **_kwargs) -> None:
        return None


def _import_workers_module():
    """Import the workers module with a stubbed QtCore dependency."""
    qtpy_module = types.ModuleType("qtpy")
    qtcore_module = types.ModuleType("qtpy.QtCore")
    qtcore_module.QObject = _QObject
    qtcore_module.Signal = _Signal
    qtpy_module.QtCore = qtcore_module
    previous_qtpy = sys.modules.get("qtpy")
    previous_qtcore = sys.modules.get("qtpy.QtCore")
    previous_workers = sys.modules.get("napari_swc_viewer.workers")

    try:
        sys.modules["qtpy"] = qtpy_module
        sys.modules["qtpy.QtCore"] = qtcore_module
        sys.modules.pop("napari_swc_viewer.workers", None)
        return importlib.import_module("napari_swc_viewer.workers")
    finally:
        if previous_workers is None:
            sys.modules.pop("napari_swc_viewer.workers", None)
        else:
            sys.modules["napari_swc_viewer.workers"] = previous_workers

        if previous_qtpy is None:
            sys.modules.pop("qtpy", None)
        else:
            sys.modules["qtpy"] = previous_qtpy

        if previous_qtcore is None:
            sys.modules.pop("qtpy.QtCore", None)
        else:
            sys.modules["qtpy.QtCore"] = previous_qtcore


def _make_cluster_result(neuron_ids: list[str], labels: list[int]) -> ClusterResult:
    """Build a small cluster result fixture."""
    n_neurons = len(neuron_ids)
    return ClusterResult(
        correlation_matrix=np.eye(n_neurons, dtype=np.float32),
        distance_matrix=np.eye(n_neurons, dtype=np.float32),
        linkage_matrix=np.zeros((max(n_neurons - 1, 1), 4), dtype=np.float64),
        neuron_ids=list(neuron_ids),
        reorder_indices=np.arange(n_neurons - 1, -1, -1, dtype=np.intp),
        labels=np.asarray(labels, dtype=np.int32),
    )


class _FakeDuckConnection:
    """Very small DuckDB connection stub used for worker tests."""

    def __init__(self, dataframe: pd.DataFrame | None = None):
        self._dataframe = dataframe if dataframe is not None else pd.DataFrame()

    def execute(self, _query):
        return self

    def fetchdf(self):
        return self._dataframe

    def close(self) -> None:
        return None


def test_convert_worker_uses_batch_conversion_with_alignment(monkeypatch, tmp_path):
    """ConvertWorker should delegate to the batch helper with UI-selected options."""
    workers = _import_workers_module()
    ConvertWorker = workers.ConvertWorker
    calls: dict[str, object] = {}

    def fake_batch_convert_swc_to_parquet(
        input_path,
        output_path,
        *,
        recursive,
        hemisphere,
        atlas_name,
        coord_axis,
        annotate_regions,
        resolution,
        progress_callback,
        **_kwargs,
    ):
        calls["input_path"] = input_path
        calls["output_path"] = output_path
        calls["recursive"] = recursive
        calls["hemisphere"] = hemisphere
        calls["atlas_name"] = atlas_name
        calls["coord_axis"] = coord_axis
        calls["annotate_regions"] = annotate_regions
        calls["resolution"] = resolution
        progress_callback("Processing a.swc...", 0, 2)
        return BatchParquetConversionSummary(
            discovered_files=2,
            processed_files=2,
            flipped_files=1,
            rows_written=4,
        )

    monkeypatch.setattr(
        "napari_swc_viewer.parquet.batch_convert_swc_to_parquet",
        fake_batch_convert_swc_to_parquet,
    )

    output_path = tmp_path / "neurons.parquet"
    worker = ConvertWorker(
        ["a.swc", "b.swc"],
        str(output_path),
        resolution=25,
        hemisphere="right",
        atlas_name="allen_mouse_10um",
    )

    progress_events: list[tuple[str, int, int]] = []
    finished: list[tuple[str, BatchParquetConversionSummary]] = []
    errors: list[str] = []
    worker.progress.connect(
        lambda message, current, total: progress_events.append((message, current, total))
    )
    worker.finished.connect(lambda path, summary: finished.append((path, summary)))
    worker.error.connect(lambda message: errors.append(message))

    worker.run()

    assert [Path(p).name for p in calls["input_path"]] == ["a.swc", "b.swc"]
    assert Path(calls["output_path"]) == output_path
    assert calls["recursive"] is False
    assert calls["hemisphere"] == "right"
    assert calls["atlas_name"] == "allen_mouse_10um"
    assert calls["coord_axis"] == 2
    assert calls["annotate_regions"] is True
    assert calls["resolution"] == 25
    assert progress_events[0] == ("Preparing SWC-to-Parquet conversion...", 0, 2)
    assert progress_events[-1] == ("Finalizing Parquet...", 2, 2)
    assert not errors
    assert finished[0][0] == str(output_path)
    assert finished[0][1].flipped_files == 1


def test_correlation_worker_uses_multi_region_mask_and_attaches_metadata(monkeypatch):
    """CorrelationWorker should union selected regions and persist run metadata."""
    workers = _import_workers_module()
    CorrelationWorker = workers.CorrelationWorker
    calls: dict[str, object] = {}

    def fake_get_expanded_region_voxel_ids_for_regions(atlas, acronyms, increase_fraction):
        calls["atlas"] = atlas
        calls["acronyms"] = list(acronyms)
        calls["increase_fraction"] = float(increase_fraction)
        return np.zeros((2, 2, 2), dtype=np.int32)

    def fake_compute_pearson_correlation_matrix(
        conn,
        parquet_path,
        voxel_id_map,
        resolution,
        file_ids=None,
    ):
        calls["parquet_path"] = parquet_path
        calls["voxel_id_map_shape"] = voxel_id_map.shape
        calls["resolution"] = resolution
        calls["file_ids"] = file_ids
        return pd.DataFrame({"swc_id_1": [], "swc_id_2": [], "r": []})

    def fake_correlation_long_to_matrix(_corr_df):
        return pd.DataFrame([[1.0, 0.4], [0.4, 1.0]], columns=["n1", "n2"]), np.array(
            [[1.0, 0.4], [0.4, 1.0]],
            dtype=np.float32,
        )

    def fake_compute_clustermap_data(mat, neuron_ids, method, n_clusters):
        calls["linkage_method"] = method
        calls["n_clusters"] = n_clusters
        assert list(neuron_ids) == ["n1", "n2"]
        return _make_cluster_result(["n1", "n2"], [1, 2])

    monkeypatch.setattr(
        "napari_swc_viewer.analysis.mask.get_expanded_region_voxel_ids_for_regions",
        fake_get_expanded_region_voxel_ids_for_regions,
    )
    monkeypatch.setattr(
        "napari_swc_viewer.analysis.correlation.compute_pearson_correlation_matrix",
        fake_compute_pearson_correlation_matrix,
    )
    monkeypatch.setattr(
        "napari_swc_viewer.analysis.correlation.correlation_long_to_matrix",
        fake_correlation_long_to_matrix,
    )
    monkeypatch.setattr(
        "napari_swc_viewer.analysis.clustering.compute_clustermap_data",
        fake_compute_clustermap_data,
    )
    monkeypatch.setattr("duckdb.connect", lambda: _FakeDuckConnection())

    atlas = types.SimpleNamespace(
        resolution=(25.0, 25.0, 25.0),
        atlas_name="fake_atlas",
    )
    region_selection = ClusterRegionSelection(
        selected_region_ids=[184, 500],
        selected_region_acronyms=["FRP", "CP"],
        represented_region_ids=[68, 500],
        represented_region_acronyms=["FRP1", "CP"],
    )
    worker = CorrelationWorker(
        parquet_path="neurons.parquet",
        atlas=atlas,
        region_selection=region_selection,
        dilation_fraction=0.3,
        linkage_method="average",
        n_clusters=4,
        file_ids=["n1", "n2"],
    )

    finished: list[ClusterResult] = []
    errors: list[str] = []
    worker.finished.connect(lambda result: finished.append(result))
    worker.error.connect(lambda message: errors.append(message))

    worker.run()

    assert not errors
    assert calls["acronyms"] == ["FRP", "CP"]
    assert calls["increase_fraction"] == 0.3
    assert calls["voxel_id_map_shape"] == (2, 2, 2)
    assert calls["resolution"] == 25.0
    assert calls["file_ids"] == ["n1", "n2"]
    metadata = finished[0].metadata
    assert metadata is not None
    assert metadata.analysis_method == "voxel_correlation"
    assert metadata.clustering_algorithm == "hierarchical"
    assert metadata.distance_metric == "one_minus_pearson_r"
    assert metadata.clustering_linkage == "average"
    assert metadata.dendrogram_linkage == "average"
    assert metadata.selected_region_ids == [184, 500]
    assert metadata.represented_region_acronyms == ["FRP1", "CP"]
    assert metadata.requested_cluster_count == 4
    assert metadata.actual_cluster_count == 2
    assert metadata.atlas_name == "fake_atlas"
    assert metadata.dendrogram_leaf_order == [1, 0]


def test_soma_cluster_worker_hierarchical_attaches_true_linkage(monkeypatch):
    """Hierarchical soma clustering should record the chosen linkage for clustering and dendrograms."""
    workers = _import_workers_module()
    SomaClusterWorker = workers.SomaClusterWorker

    monkeypatch.setattr(
        "napari_swc_viewer.analysis.mask.get_expanded_region_voxel_ids_for_regions",
        lambda atlas, acronyms, increase_fraction: np.zeros((4, 4, 4), dtype=np.int32),
    )
    monkeypatch.setattr(
        "napari_swc_viewer.analysis.clustering.cluster_somas_hierarchical",
        lambda coords, neuron_ids, method, n_clusters: _make_cluster_result(neuron_ids, [1, 2]),
    )
    monkeypatch.setattr("duckdb.connect", lambda: _FakeDuckConnection(
        pd.DataFrame(
            {
                "file_id": ["n1", "n2"],
                "x": [0.0, 25.0],
                "y": [0.0, 25.0],
                "z": [0.0, 25.0],
            }
        )
    ))

    worker = SomaClusterWorker(
        parquet_path="neurons.parquet",
        atlas=types.SimpleNamespace(resolution=(25.0, 25.0, 25.0), atlas_name="fake_atlas"),
        region_selection=ClusterRegionSelection(
            selected_region_ids=[184],
            selected_region_acronyms=["FRP"],
            represented_region_ids=[68],
            represented_region_acronyms=["FRP1"],
        ),
        dilation_fraction=0.2,
        algorithm="hierarchical",
        linkage_method="ward",
        n_clusters=3,
    )
    finished: list[ClusterResult] = []
    worker.finished.connect(lambda result: finished.append(result))

    worker.run()

    metadata = finished[0].metadata
    assert metadata is not None
    assert metadata.clustering_algorithm == "hierarchical"
    assert metadata.clustering_linkage == "ward"
    assert metadata.dendrogram_linkage == "ward"
    assert metadata.requested_cluster_count == 3
    assert metadata.distance_metric == "euclidean_um"


def test_soma_cluster_worker_filters_to_current_table_file_ids(monkeypatch):
    """Soma clustering should respect an optional current-table file-id subset."""
    workers = _import_workers_module()
    SomaClusterWorker = workers.SomaClusterWorker
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        "napari_swc_viewer.analysis.mask.get_expanded_region_voxel_ids_for_regions",
        lambda atlas, acronyms, increase_fraction: np.zeros((4, 4, 4), dtype=np.int32),
    )

    def fake_cluster_somas_hierarchical(coords, neuron_ids, method, n_clusters):
        observed["neuron_ids"] = list(neuron_ids)
        observed["coords_shape"] = tuple(coords.shape)
        return _make_cluster_result(list(neuron_ids), [1, 2])

    monkeypatch.setattr(
        "napari_swc_viewer.analysis.clustering.cluster_somas_hierarchical",
        fake_cluster_somas_hierarchical,
    )
    monkeypatch.setattr(
        "duckdb.connect",
        lambda: _FakeDuckConnection(
            pd.DataFrame(
                {
                    "file_id": ["n1", "n2", "n3"],
                    "x": [0.0, 25.0, 50.0],
                    "y": [0.0, 25.0, 50.0],
                    "z": [0.0, 25.0, 50.0],
                }
            )
        ),
    )

    worker = SomaClusterWorker(
        parquet_path="neurons.parquet",
        atlas=types.SimpleNamespace(
            resolution=(25.0, 25.0, 25.0),
            atlas_name="fake_atlas",
        ),
        region_selection=ClusterRegionSelection(
            selected_region_ids=[184],
            selected_region_acronyms=["FRP"],
            represented_region_ids=[68],
            represented_region_acronyms=["FRP1"],
        ),
        algorithm="hierarchical",
        linkage_method="ward",
        n_clusters=2,
        file_ids=["n1", "n2"],
    )

    finished: list[ClusterResult] = []
    worker.finished.connect(lambda result: finished.append(result))

    worker.run()

    assert observed["neuron_ids"] == ["n1", "n2"]
    assert observed["coords_shape"] == (2, 3)
    assert finished[0].neuron_ids == ["n1", "n2"]


def test_soma_cluster_worker_kmeans_uses_synthesized_dendrogram_linkage(monkeypatch):
    """K-means soma clustering should keep clustering linkage empty and dendrogram linkage synthetic."""
    workers = _import_workers_module()
    SomaClusterWorker = workers.SomaClusterWorker

    monkeypatch.setattr(
        "napari_swc_viewer.analysis.mask.get_expanded_region_voxel_ids_for_regions",
        lambda atlas, acronyms, increase_fraction: np.zeros((4, 4, 4), dtype=np.int32),
    )
    monkeypatch.setattr(
        "napari_swc_viewer.analysis.clustering.cluster_somas_kmeans",
        lambda coords, neuron_ids, n_clusters: _make_cluster_result(neuron_ids, [1, 1]),
    )
    monkeypatch.setattr("duckdb.connect", lambda: _FakeDuckConnection(
        pd.DataFrame(
            {
                "file_id": ["n1", "n2"],
                "x": [0.0, 25.0],
                "y": [0.0, 25.0],
                "z": [0.0, 25.0],
            }
        )
    ))

    worker = SomaClusterWorker(
        parquet_path="neurons.parquet",
        atlas=types.SimpleNamespace(resolution=(25.0, 25.0, 25.0), atlas_name="fake_atlas"),
        region_selection=ClusterRegionSelection(
            selected_region_ids=[184, 500],
            selected_region_acronyms=["FRP", "CP"],
            represented_region_ids=[68, 500],
            represented_region_acronyms=["FRP1", "CP"],
        ),
        algorithm="kmeans",
        n_clusters=2,
    )
    finished: list[ClusterResult] = []
    worker.finished.connect(lambda result: finished.append(result))

    worker.run()

    metadata = finished[0].metadata
    assert metadata is not None
    assert metadata.clustering_algorithm == "kmeans"
    assert metadata.clustering_linkage is None
    assert metadata.dendrogram_linkage == "average"
    assert metadata.requested_cluster_count == 2


def test_soma_cluster_worker_dbscan_records_dbscan_parameters(monkeypatch):
    """DBSCAN soma clustering should keep clustering linkage empty and persist DBSCAN params."""
    workers = _import_workers_module()
    SomaClusterWorker = workers.SomaClusterWorker

    monkeypatch.setattr(
        "napari_swc_viewer.analysis.mask.get_expanded_region_voxel_ids_for_regions",
        lambda atlas, acronyms, increase_fraction: np.zeros((4, 4, 4), dtype=np.int32),
    )
    monkeypatch.setattr(
        "napari_swc_viewer.analysis.clustering.cluster_somas_dbscan",
        lambda coords, neuron_ids, eps, min_samples: _make_cluster_result(neuron_ids, [1, 2]),
    )
    monkeypatch.setattr("duckdb.connect", lambda: _FakeDuckConnection(
        pd.DataFrame(
            {
                "file_id": ["n1", "n2"],
                "x": [0.0, 25.0],
                "y": [0.0, 25.0],
                "z": [0.0, 25.0],
            }
        )
    ))

    worker = SomaClusterWorker(
        parquet_path="neurons.parquet",
        atlas=types.SimpleNamespace(resolution=(25.0, 25.0, 25.0), atlas_name="fake_atlas"),
        region_selection=ClusterRegionSelection(
            selected_region_ids=[184],
            selected_region_acronyms=["FRP"],
            represented_region_ids=[68],
            represented_region_acronyms=["FRP1"],
        ),
        algorithm="dbscan",
        eps=150.0,
        min_samples=7,
    )
    finished: list[ClusterResult] = []
    worker.finished.connect(lambda result: finished.append(result))

    worker.run()

    metadata = finished[0].metadata
    assert metadata is not None
    assert metadata.clustering_algorithm == "dbscan"
    assert metadata.clustering_linkage is None
    assert metadata.dendrogram_linkage == "average"
    assert metadata.requested_cluster_count is None
    assert metadata.dbscan_eps == 150.0
    assert metadata.dbscan_min_samples == 7


def test_append_point_file_worker_routes_csv_to_append_helper(monkeypatch, tmp_path):
    """AppendPointFileWorker should route CSV input to the CSV append helper."""
    workers = _import_workers_module()
    AppendPointFileWorker = workers.AppendPointFileWorker
    calls: dict[str, object] = {}

    def fake_append_point_csv_to_parquet(
        csv_path,
        mapping_path,
        parquet_path,
        output_path,
    ):
        calls["csv_path"] = csv_path
        calls["mapping_path"] = mapping_path
        calls["parquet_path"] = parquet_path
        calls["output_path"] = output_path
        return PointParquetAppendSummary(appended_rows=2, total_rows=5)

    monkeypatch.setattr(
        "napari_swc_viewer.point_import.append_point_csv_to_parquet",
        fake_append_point_csv_to_parquet,
    )

    worker = AppendPointFileWorker(
        str(tmp_path / "points.csv"),
        str(tmp_path / "mapping.json"),
        str(tmp_path / "points.parquet"),
        str(tmp_path / "points_out.parquet"),
    )
    progress_events: list[tuple[str, int, int]] = []
    finished: list[tuple[str, PointParquetAppendSummary]] = []
    errors: list[str] = []
    worker.progress.connect(
        lambda message, current, total: progress_events.append((message, current, total))
    )
    worker.finished.connect(lambda path, summary: finished.append((path, summary)))
    worker.error.connect(lambda message: errors.append(message))

    worker.run()

    assert Path(calls["csv_path"]).name == "points.csv"
    assert Path(calls["mapping_path"]).name == "mapping.json"
    assert Path(calls["parquet_path"]).name == "points.parquet"
    assert Path(calls["output_path"]).name == "points_out.parquet"
    assert progress_events[0] == ("Validating point CSV and target Parquet...", 1, 3)
    assert progress_events[-1] == ("Done", 3, 3)
    assert not errors
    assert finished[0][0].endswith("points_out.parquet")
    assert finished[0][1].appended_rows == 2
    assert finished[0][1].total_rows == 5


def test_append_point_file_worker_routes_parquet_to_append_helper(monkeypatch, tmp_path):
    """AppendPointFileWorker should route Parquet input to the Parquet append helper."""
    workers = _import_workers_module()
    AppendPointFileWorker = workers.AppendPointFileWorker
    calls: dict[str, object] = {}

    def fake_append_point_parquet_to_parquet(
        input_parquet_path,
        parquet_path,
        output_path,
    ):
        calls["input_parquet_path"] = input_parquet_path
        calls["parquet_path"] = parquet_path
        calls["output_path"] = output_path
        return PointParquetAppendSummary(appended_rows=3, total_rows=8)

    monkeypatch.setattr(
        "napari_swc_viewer.point_import.append_point_parquet_to_parquet",
        fake_append_point_parquet_to_parquet,
    )

    worker = AppendPointFileWorker(
        str(tmp_path / "points_in.parquet"),
        None,
        str(tmp_path / "points.parquet"),
        str(tmp_path / "points_out.parquet"),
    )
    progress_events: list[tuple[str, int, int]] = []
    finished: list[tuple[str, PointParquetAppendSummary]] = []
    errors: list[str] = []
    worker.progress.connect(
        lambda message, current, total: progress_events.append((message, current, total))
    )
    worker.finished.connect(lambda path, summary: finished.append((path, summary)))
    worker.error.connect(lambda message: errors.append(message))

    worker.run()

    assert Path(calls["input_parquet_path"]).name == "points_in.parquet"
    assert Path(calls["parquet_path"]).name == "points.parquet"
    assert Path(calls["output_path"]).name == "points_out.parquet"
    assert progress_events[0] == ("Validating point Parquet schemas...", 1, 3)
    assert not errors
    assert finished[0][1].appended_rows == 3
    assert finished[0][1].total_rows == 8


def test_append_point_file_worker_emits_error(monkeypatch, tmp_path):
    """AppendPointFileWorker should surface append failures through error."""
    workers = _import_workers_module()
    AppendPointFileWorker = workers.AppendPointFileWorker

    def fake_append_point_csv_to_parquet(
        _csv_path,
        _mapping_path,
        _parquet_path,
        _output_path,
    ):
        raise ValueError("schema mismatch")

    monkeypatch.setattr(
        "napari_swc_viewer.point_import.append_point_csv_to_parquet",
        fake_append_point_csv_to_parquet,
    )

    worker = AppendPointFileWorker(
        str(tmp_path / "points.csv"),
        str(tmp_path / "mapping.json"),
        str(tmp_path / "points.parquet"),
        str(tmp_path / "points_out.parquet"),
    )
    finished: list[tuple[str, PointParquetAppendSummary]] = []
    errors: list[str] = []
    worker.finished.connect(lambda path, summary: finished.append((path, summary)))
    worker.error.connect(lambda message: errors.append(message))

    worker.run()

    assert not finished
    assert errors == ["schema mismatch"]


def test_convert_point_csv_worker_delegates_to_batch_helper(monkeypatch, tmp_path):
    """ConvertPointCSVWorker should call the batch point CSV helper."""
    workers = _import_workers_module()
    ConvertPointCSVWorker = workers.ConvertPointCSVWorker
    calls: dict[str, object] = {}

    def fake_convert_point_csv_files_to_parquet(
        csv_paths,
        output_path,
        mapping_path,
        progress_callback,
    ):
        calls["csv_paths"] = csv_paths
        calls["output_path"] = output_path
        calls["mapping_path"] = mapping_path
        progress_callback("Processing point CSV 1/2: one.csv", 0, 2)
        return types.SimpleNamespace(
            discovered_files=2,
            processed_files=2,
            rows_written=4,
        )

    monkeypatch.setattr(
        "napari_swc_viewer.point_import.convert_point_csv_files_to_parquet",
        fake_convert_point_csv_files_to_parquet,
    )

    worker = ConvertPointCSVWorker(
        [str(tmp_path / "one.csv"), str(tmp_path / "two.csv")],
        str(tmp_path / "points.parquet"),
        str(tmp_path / "mapping.json"),
    )
    progress_events: list[tuple[str, int, int]] = []
    finished: list[tuple[str, object]] = []
    errors: list[str] = []
    worker.progress.connect(
        lambda message, current, total: progress_events.append((message, current, total))
    )
    worker.finished.connect(lambda path, summary: finished.append((path, summary)))
    worker.error.connect(lambda message: errors.append(message))

    worker.run()

    assert [Path(path).name for path in calls["csv_paths"]] == ["one.csv", "two.csv"]
    assert Path(calls["output_path"]).name == "points.parquet"
    assert Path(calls["mapping_path"]).name == "mapping.json"
    assert progress_events[0] == ("Processing point CSV 1/2: one.csv", 0, 2)
    assert not errors
    assert finished[0][0].endswith("points.parquet")
    assert finished[0][1].processed_files == 2
    assert finished[0][1].rows_written == 4


def test_convert_point_csv_worker_emits_error(monkeypatch, tmp_path):
    """ConvertPointCSVWorker should surface point conversion failures."""
    workers = _import_workers_module()
    ConvertPointCSVWorker = workers.ConvertPointCSVWorker

    def fake_convert_point_csv_files_to_parquet(
        _csv_paths,
        _output_path,
        _mapping_path,
        progress_callback=None,
    ):
        raise ValueError("bad headers")

    monkeypatch.setattr(
        "napari_swc_viewer.point_import.convert_point_csv_files_to_parquet",
        fake_convert_point_csv_files_to_parquet,
    )

    worker = ConvertPointCSVWorker(
        [str(tmp_path / "one.csv")],
        str(tmp_path / "points.parquet"),
        None,
    )
    finished: list[tuple[str, object]] = []
    errors: list[str] = []
    worker.finished.connect(lambda path, summary: finished.append((path, summary)))
    worker.error.connect(lambda message: errors.append(message))

    worker.run()

    assert not finished
    assert errors == ["bad headers"]
