"""End-to-end tests for parquet-driven flatmap-space clustering.

These exercise the real DuckDB path (no rendered heatmap and no mocking of the
analysis functions), covering both voxel correlation and soma-distance
clustering in flat map + depth space.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from napari_swc_viewer.analysis.flatmap_correlation import (
    compute_flatmap_voxel_correlation_from_parquet,
    query_flatmap_soma_coordinates,
)
from napari_swc_viewer.flatmap_parquet import read_flatmap_parquet_transform_info


def _v3_augmented_frame(
    *,
    n_neurons: int = 4,
    nodes_per_neuron: int = 25,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a version-3-style neuron frame with the full augmented columns."""
    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {name: [] for name in (
        "file_id",
        "node_id",
        "parent_id",
        "type",
        "x",
        "y",
        "z",
        "x_flat_shaped",
        "y_flat_shaped",
        "flatmap_shaped_valid",
        "flatmap_shaped_projection_valid",
        "flatmap_shaped_invalid_code",
        "flatmap_shaped_lookup_mode",
        "x_flat_square",
        "y_flat_square",
        "flatmap_square_valid",
        "flatmap_square_projection_valid",
        "flatmap_square_invalid_code",
        "flatmap_square_lookup_mode",
        "depth_um",
        "depth_valid",
        "depth_invalid_code",
        "depth_lookup_mode",
    )}
    node_id = 0
    for neuron in range(n_neurons):
        # Cluster each neuron's nodes around a distinct flatmap centroid so
        # correlation and soma distance are both well defined.
        cx = 10.0 + neuron * 20.0
        cy = 10.0 + neuron * 15.0
        cd = 100.0 + neuron * 120.0
        for local in range(nodes_per_neuron):
            xf = float(cx + rng.uniform(-5.0, 5.0))
            yf = float(cy + rng.uniform(-5.0, 5.0))
            depth = float(cd + rng.uniform(-20.0, 20.0))
            node_type = 1 if local == 0 else 3
            rows["file_id"].append(f"neuron_{neuron}")
            rows["node_id"].append(node_id)
            rows["parent_id"].append(-1 if local == 0 else node_id - 1)
            rows["type"].append(node_type)
            rows["x"].append(xf * 10.0)
            rows["y"].append(yf * 10.0)
            rows["z"].append(depth)
            rows["x_flat_shaped"].append(xf)
            rows["y_flat_shaped"].append(yf)
            rows["flatmap_shaped_valid"].append(True)
            rows["flatmap_shaped_projection_valid"].append(True)
            rows["flatmap_shaped_invalid_code"].append(0)
            rows["flatmap_shaped_lookup_mode"].append("direct")
            rows["x_flat_square"].append(xf + 1.0)
            rows["y_flat_square"].append(yf + 1.0)
            rows["flatmap_square_valid"].append(True)
            rows["flatmap_square_projection_valid"].append(True)
            rows["flatmap_square_invalid_code"].append(0)
            rows["flatmap_square_lookup_mode"].append("direct")
            rows["depth_um"].append(depth)
            rows["depth_valid"].append(True)
            rows["depth_invalid_code"].append(0)
            rows["depth_lookup_mode"].append("direct")
            node_id += 1
    frame = pd.DataFrame(rows)
    frame["node_id"] = frame["node_id"].astype(np.int32)
    frame["parent_id"] = frame["parent_id"].astype(np.int32)
    frame["type"] = frame["type"].astype(np.int32)
    for name in (
        "x", "y", "z",
        "x_flat_shaped", "y_flat_shaped",
        "x_flat_square", "y_flat_square",
        "depth_um",
    ):
        frame[name] = frame[name].astype(np.float32)
    return frame


@pytest.fixture()
def flatmap_parquet(tmp_path):
    frame = _v3_augmented_frame()
    path = tmp_path / "neurons_v3.parquet"
    frame.to_parquet(path, index=False)
    return frame, str(path)


def _fake_atlas():
    return types.SimpleNamespace(
        atlas_name="test_atlas",
        resolution=(25.0, 25.0, 25.0),
    )


def test_transform_info_detects_v3_styles(flatmap_parquet) -> None:
    _frame, path = flatmap_parquet
    info = read_flatmap_parquet_transform_info(path)
    assert set(info.available_styles) == {"both_shaped", "both_square"}
    assert info.has_v3_depth is True


def test_compute_flatmap_voxel_correlation_from_parquet(flatmap_parquet) -> None:
    _frame, path = flatmap_parquet
    result, count_data, provenance = (
        compute_flatmap_voxel_correlation_from_parquet(
            path,
            style="both_shaped",
            xy_bins=32,
            depth_bin_um=50.0,
            n_clusters=2,
        )
    )
    assert len(result.neuron_ids) == 4
    assert len(result.labels) == 4
    assert count_data.count_matrix.shape[0] == 4
    assert count_data.count_matrix.shape[1] == len(count_data.voxel_ids)
    assert count_data.rendered_node_count > 0
    assert provenance.style == "both_shaped"
    assert provenance.xy_bins == 32
    assert provenance.volume_shape[1] == 32
    assert result.metadata is None  # metadata attached by the worker, not here


def test_compute_flatmap_correlation_respects_file_id_subset(
    flatmap_parquet,
) -> None:
    _frame, path = flatmap_parquet
    result, _count, _prov = compute_flatmap_voxel_correlation_from_parquet(
        path,
        style="both_square",
        xy_bins=16,
        depth_bin_um=75.0,
        n_clusters=2,
        file_ids=["neuron_0", "neuron_1", "neuron_2"],
    )
    assert set(result.neuron_ids) == {"neuron_0", "neuron_1", "neuron_2"}


def test_query_flatmap_soma_coordinates(flatmap_parquet) -> None:
    _frame, path = flatmap_parquet
    ids, coords = query_flatmap_soma_coordinates(path, style="both_shaped")
    assert ids == ["neuron_0", "neuron_1", "neuron_2", "neuron_3"]
    assert coords.shape == (4, 3)
    # Soma centroids should be ordered and well separated along each axis.
    assert np.all(np.diff(coords[:, 0]) > 0)
    assert np.all(np.diff(coords[:, 2]) > 0)


def test_flatmap_parquet_correlation_worker(flatmap_parquet) -> None:
    from napari_swc_viewer import workers

    _frame, path = flatmap_parquet
    finished: list = []
    errors: list = []
    worker = workers.FlatmapParquetCorrelationWorker(
        parquet_path=path,
        atlas=_fake_atlas(),
        style="both_shaped",
        xy_bins=32,
        depth_bin_um=50.0,
        n_clusters=2,
    )
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)
    worker.run()

    assert errors == []
    assert len(finished) == 1
    result = finished[0]
    assert len(result.neuron_ids) == 4
    assert result.metadata is not None
    assert result.metadata.analysis_method == "flatmap_voxel_correlation"


def test_flatmap_soma_cluster_worker(flatmap_parquet) -> None:
    from napari_swc_viewer import workers

    _frame, path = flatmap_parquet
    finished: list = []
    errors: list = []
    worker = workers.FlatmapSomaClusterWorker(
        parquet_path=path,
        atlas=_fake_atlas(),
        style="both_shaped",
        algorithm="hierarchical",
        linkage_method="ward",
        n_clusters=2,
    )
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)
    worker.run()

    assert errors == []
    assert len(finished) == 1
    result = finished[0]
    assert len(result.neuron_ids) == 4
    assert result.metadata is not None
    assert result.metadata.analysis_method == "flatmap_soma_location"
