from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from napari_swc_viewer.flatmap_heatmap import (
    FLATMAP_HEATMAP_COLOR_CLUSTER,
    FLATMAP_HEATMAP_COLOR_INDIVIDUAL,
    FLATMAP_HEATMAP_COLOR_SINGLE,
    MAX_FLATMAP_HEATMAP_VOXELS,
    build_allen_layer_cluster_volumes,
    build_allen_layer_file_id_volumes,
    build_allen_layer_heatmap_volume_result,
    build_allen_layer_stack_from_projected_nodes,
    build_flatmap_cluster_volumes,
    build_flatmap_file_id_volumes,
    build_flatmap_heatmap_volume_result,
    build_flatmap_render_data,
    build_flatmap_render_data_from_projected_nodes,
    compute_depth_range,
    compute_flatmap_bounds_from_parquet,
    compute_flatmap_lookup_stats,
    compute_flatmap_xy_bounds,
    depth_plane_labels,
)
from napari_swc_viewer.isocortex_layers import AllenIsocortexLayerMap


def _lookup_volumes() -> tuple[np.ndarray, np.ndarray]:
    flatmap = np.zeros((2, 2, 2, 2), dtype=float)
    flatmap[..., 0] = np.asarray([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, -1.0]]])
    flatmap[..., 1] = np.asarray([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, -1.0]]])
    depth = np.asarray([[[0.0, 25.0], [50.0, 75.0]], [[100.0, 125.0], [150.0, -1.0]]])
    return flatmap, depth


def _projected_nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file_id": ["a.swc", "a.swc", "b.swc", "c.swc", "d.swc"],
            "x_flat": [0.0, 0.1, 6.0, 2.0, -1.0],
            "y_flat": [0.0, 0.1, 6.0, 2.0, -1.0],
            "depth_um": [0.0, 10.0, -1.0, 50.0, 10.0],
            "flatmap_valid": [True, True, True, True, False],
            "depth_valid": [True, True, False, True, True],
        }
    )


def _legacy_flatmap_xy_bounds(
    flatmap: np.ndarray,
    *,
    invalid_zero_sentinel: bool = False,
    invalid_negative_one_sentinel: bool = True,
) -> tuple[tuple[float, float], tuple[float, float]]:
    flat_xy = np.asarray(flatmap, dtype=float).reshape(-1, 2)
    valid = np.all(np.isfinite(flat_xy), axis=1)
    if invalid_negative_one_sentinel:
        valid &= ~((flat_xy[:, 0] == -1.0) & (flat_xy[:, 1] == -1.0))
    if invalid_zero_sentinel:
        valid &= ~((flat_xy[:, 0] == 0.0) & (flat_xy[:, 1] == 0.0))
    valid_xy = flat_xy[valid]
    return (
        (float(np.min(valid_xy[:, 0])), float(np.max(valid_xy[:, 0]))),
        (float(np.min(valid_xy[:, 1])), float(np.max(valid_xy[:, 1]))),
    )


def _legacy_depth_range(depth: np.ndarray) -> tuple[float, float]:
    values = np.asarray(depth, dtype=float)
    valid_values = values[np.isfinite(values) & (values >= 0.0)]
    lower = float(np.min(valid_values))
    upper = float(np.max(valid_values))
    return lower, upper if upper > lower else lower + 1.0


def test_compute_bounds_ignore_negative_one_sentinel() -> None:
    flatmap, depth = _lookup_volumes()

    assert compute_flatmap_xy_bounds(flatmap) == ((0.0, 6.0), (0.0, 6.0))
    assert compute_depth_range(depth) == (0.0, 150.0)


def test_chunked_lookup_stats_match_full_array_reference() -> None:
    flatmap, depth = _lookup_volumes()
    flatmap[0, 0, 1] = (np.nan, 99.0)
    flatmap[0, 1, 0] = (np.inf, 4.0)
    flatmap[1, 0, 1] = (0.0, 0.0)
    depth[0, 0, 0] = np.nan
    depth[0, 0, 1] = np.inf

    stats = compute_flatmap_lookup_stats(
        flatmap,
        depth,
        invalid_zero_sentinel=True,
        chunk_voxels=2,
    )

    expected_x_bounds, expected_y_bounds = _legacy_flatmap_xy_bounds(
        flatmap,
        invalid_zero_sentinel=True,
    )
    assert stats.x_bounds == expected_x_bounds
    assert stats.y_bounds == expected_y_bounds
    assert stats.depth_range_um == _legacy_depth_range(depth)
    assert stats.flatmap_valid_voxels == 3
    assert stats.depth_valid_voxels == 5


def test_chunked_lookup_stats_reject_all_invalid_flatmap() -> None:
    flatmap = np.full((2, 2, 2, 2), -1.0)
    depth = np.ones((2, 2, 2), dtype=float)

    with pytest.raises(ValueError, match="valid x/y"):
        compute_flatmap_lookup_stats(flatmap, depth, chunk_voxels=2)


def test_chunked_lookup_stats_reject_all_invalid_depth() -> None:
    flatmap = np.ones((2, 2, 2, 2), dtype=float)
    depth = np.full((2, 2, 2), -1.0)

    with pytest.raises(ValueError, match="valid non-negative depths"):
        compute_flatmap_lookup_stats(flatmap, depth, chunk_voxels=2)


def test_flatmap_heatmap_includes_depth_minus_one_in_sentinel_plane() -> None:
    flatmap, depth = _lookup_volumes()

    result = build_flatmap_render_data(
        _projected_nodes(),
        flatmap,
        depth,
        xy_bins=4,
        depth_bin_um=25.0,
        include_depth_minus_one=True,
    )

    assert result.volume.shape == (8, 4, 4)
    assert result.summary.rendered_nodes == 4
    assert result.summary.depth_minus_one_nodes == 1
    assert result.summary.nonzero_voxels == 3
    assert result.projected_nodes["render_valid"].tolist() == [
        True,
        True,
        True,
        True,
        False,
    ]
    assert result.projected_nodes["depth_bin"].tolist() == [1, 1, 0, 3, -1]
    assert result.projected_nodes["depth_bin_label"].tolist()[2] == "depth -1"
    assert result.volume[1, 0, 0] == 2.0
    assert result.volume[0, 3, 3] == 1.0
    assert result.volume[3, 1, 1] == 1.0
    np.testing.assert_array_equal(
        result.points,
        np.asarray(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 3.0, 3.0], [3.0, 1.0, 1.0]]
        ),
    )
    assert result.point_file_ids == ["a.swc", "a.swc", "b.swc", "c.swc"]


def test_flatmap_heatmap_excludes_depth_minus_one_when_requested() -> None:
    flatmap, depth = _lookup_volumes()

    result = build_flatmap_render_data(
        _projected_nodes(),
        flatmap,
        depth,
        xy_bins=4,
        depth_bin_um=25.0,
        include_depth_minus_one=False,
    )

    assert result.volume.shape == (7, 4, 4)
    assert result.summary.rendered_nodes == 3
    assert result.summary.excluded_depth_minus_one_nodes == 1
    assert result.projected_nodes["render_valid"].tolist() == [
        True,
        True,
        False,
        True,
        False,
    ]
    assert result.projected_nodes["depth_bin"].tolist() == [0, 0, -1, 2, -1]


def test_flatmap_heatmap_reuses_precomputed_lookup_stats() -> None:
    flatmap, depth = _lookup_volumes()
    stats = compute_flatmap_lookup_stats(flatmap, depth, chunk_voxels=2)

    expected = build_flatmap_render_data(
        _projected_nodes(),
        flatmap,
        depth,
        xy_bins=4,
        depth_bin_um=25.0,
        include_depth_minus_one=True,
    )
    actual = build_flatmap_render_data(
        _projected_nodes(),
        flatmap,
        depth,
        xy_bins=4,
        depth_bin_um=25.0,
        include_depth_minus_one=True,
        lookup_stats=stats,
    )

    np.testing.assert_array_equal(actual.volume, expected.volume)
    np.testing.assert_array_equal(actual.points, expected.points)
    assert actual.point_file_ids == expected.point_file_ids
    assert actual.summary.to_dict() == expected.summary.to_dict()
    pd.testing.assert_frame_equal(actual.projected_nodes, expected.projected_nodes)


def test_flatmap_heatmap_from_projected_nodes_uses_projected_bounds() -> None:
    result = build_flatmap_render_data_from_projected_nodes(
        _projected_nodes(),
        xy_bins=4,
        depth_bin_um=25.0,
        include_depth_minus_one=True,
    )

    assert result.volume.shape == (4, 4, 4)
    assert result.summary.x_flat_min == 0.0
    assert result.summary.x_flat_max == 6.0
    assert result.summary.depth_min_um == 0.0
    assert result.summary.depth_max_um == 50.0
    assert result.summary.rendered_nodes == 4
    assert result.summary.depth_minus_one_nodes == 1
    assert result.projected_nodes["render_valid"].tolist() == [
        True,
        True,
        True,
        True,
        False,
    ]
    assert result.projected_nodes["depth_bin"].tolist() == [1, 1, 0, 3, -1]
    assert result.volume[1, 0, 0] == 2.0
    assert result.volume[0, 3, 3] == 1.0
    assert result.volume[3, 1, 1] == 1.0


def test_flatmap_heatmap_from_projected_nodes_infers_validity_flags() -> None:
    projected = pd.DataFrame(
        {
            "file_id": ["a.swc", "b.swc", "c.swc"],
            "x_flat": [0.0, np.nan, 2.0],
            "y_flat": [0.0, 1.0, 2.0],
            "depth_um": [0.0, 10.0, -1.0],
        }
    )

    result = build_flatmap_render_data_from_projected_nodes(
        projected,
        xy_bins=2,
        depth_bin_um=25.0,
        include_depth_minus_one=True,
    )

    assert result.summary.rendered_nodes == 2
    assert result.summary.flatmap_valid_nodes == 2
    assert result.summary.depth_valid_nodes == 1
    assert result.projected_nodes["flatmap_valid"].tolist() == [True, False, True]
    assert result.projected_nodes["depth_valid"].tolist() == [True, True, False]
    assert result.projected_nodes["render_valid"].tolist() == [True, False, True]


def test_precomputed_subset_uses_canonical_bounds_for_cache_alignment() -> None:
    full = pd.DataFrame(
        {
            "file_id": ["left.swc", "right.swc"],
            "x_flat": [10.0, 90.0],
            "y_flat": [10.0, 90.0],
            "depth_um": [10.0, 90.0],
            "flatmap_valid": [True, True],
            "depth_valid": [True, True],
        }
    )
    canonical = {
        "x_bounds": (0.0, 100.0),
        "y_bounds": (0.0, 100.0),
        "depth_range_um": (0.0, 100.0),
    }
    complete = build_flatmap_render_data_from_projected_nodes(
        full,
        xy_bins=10,
        depth_bin_um=10.0,
        include_depth_minus_one=False,
        **canonical,
    )
    subset = build_flatmap_render_data_from_projected_nodes(
        full.iloc[[1]],
        xy_bins=10,
        depth_bin_um=10.0,
        include_depth_minus_one=False,
        **canonical,
    )

    assert subset.volume.shape == complete.volume.shape == (11, 10, 10)
    row = subset.projected_nodes.iloc[0]
    assert (int(row.depth_bin), int(row.y_flat_bin), int(row.x_flat_bin)) == (
        9,
        9,
        9,
    )


def _binned_projected_nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file_id": ["a.swc", "a.swc", "b.swc", "c.swc", "skip.swc"],
            "render_valid": [True, True, True, True, False],
            "depth_bin": [0, 0, 1, 1, 0],
            "y_flat_bin": [1, 1, 2, 0, 0],
            "x_flat_bin": [2, 2, 0, 1, 0],
        }
    )


def test_build_flatmap_file_id_volumes_splits_rendered_node_counts() -> None:
    groups = build_flatmap_file_id_volumes(
        _binned_projected_nodes(),
        (2, 3, 3),
    )

    assert [group.label for group in groups] == ["a.swc", "b.swc", "c.swc"]
    assert [group.source_file_ids for group in groups] == [
        ("a.swc",),
        ("b.swc",),
        ("c.swc",),
    ]
    assert [group.rendered_nodes for group in groups] == [2, 1, 1]
    assert groups[0].volume[0, 1, 2] == 2.0
    assert groups[1].volume[1, 2, 0] == 1.0
    assert groups[2].volume[1, 0, 1] == 1.0
    assert sum(float(group.volume.sum()) for group in groups) == 4.0


def test_build_flatmap_cluster_volumes_groups_by_cluster_with_unclustered_last() -> (
    None
):
    groups = build_flatmap_cluster_volumes(
        _binned_projected_nodes(),
        (2, 3, 3),
        {"a.swc": 2, "b.swc": 1},
    )

    assert [group.group_key for group in groups] == [1, 2, None]
    assert [group.label for group in groups] == [
        "Cluster 1",
        "Cluster 2",
        "Unclustered",
    ]
    assert [group.source_file_ids for group in groups] == [
        ("b.swc",),
        ("a.swc",),
        ("c.swc",),
    ]
    assert groups[0].volume[1, 2, 0] == 1.0
    assert groups[1].volume[0, 1, 2] == 2.0
    assert groups[2].volume[1, 0, 1] == 1.0


def _allen_layer_map() -> AllenIsocortexLayerMap:
    region_ids = (101, 102, 103, 104, 105, 106)
    return AllenIsocortexLayerMap(
        atlas_name="allen_mouse_25um",
        isocortex_region_id=315,
        region_to_layer_index={
            region_id: index for index, region_id in enumerate(region_ids)
        },
        region_ids_by_layer=tuple((region_id,) for region_id in region_ids),
    )


def _allen_layer_nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file_id": [
                "a.swc",
                "a.swc",
                "b.swc",
                "b.swc",
                "c.swc",
                "d.swc",
                "e.swc",
                "f.swc",
            ],
            "x_flat": [0.0, 0.0, 9.9, 5.0, 2.0, 3.0, 4.0, np.nan],
            "y_flat": [0.0, 0.0, 9.9, 5.0, 2.0, 3.0, 4.0, 1.0],
            "flatmap_valid": [True, True, True, True, True, True, True, True],
            "region_id": [101, 101, 102, 104, 105, 106, 999, 103],
        }
    )


def test_build_allen_layer_stack_counts_and_excludes_nodes() -> None:
    result = build_allen_layer_stack_from_projected_nodes(
        _allen_layer_nodes(),
        _allen_layer_map(),
        xy_bins=10,
        x_bounds=(0.0, 10.0),
        y_bounds=(0.0, 10.0),
    )

    assert result.volume.shape == (6, 10, 10)
    assert tuple(result.volume.sum(axis=(1, 2)).astype(int)) == (
        2,
        1,
        0,
        1,
        1,
        1,
    )
    assert result.volume[0, 0, 0] == 2.0
    assert result.volume[1, 9, 9] == 1.0
    assert result.summary.rendered_nodes == 6
    assert result.summary.excluded_nodes == 2
    assert result.summary.invalid_flatmap_nodes == 1
    assert result.summary.excluded_non_layer_nodes == 1
    assert result.summary.layer_node_counts == (2, 1, 0, 1, 1, 1)
    assert result.summary.to_dict()["excluded_nodes"] == 2
    assert result.projected_nodes["allen_layer_label"].tolist()[:4] == [
        "L1",
        "L1",
        "L2/3",
        "L5",
    ]
    assert result.projected_nodes["render_valid"].sum() == 6


def test_allen_layer_grouped_volumes_preserve_counts() -> None:
    result = build_allen_layer_stack_from_projected_nodes(
        _allen_layer_nodes(),
        _allen_layer_map(),
        xy_bins=10,
        x_bounds=(0.0, 10.0),
        y_bounds=(0.0, 10.0),
    )

    individual = build_allen_layer_file_id_volumes(
        result.projected_nodes,
        result.volume.shape,
    )
    clusters = build_allen_layer_cluster_volumes(
        result.projected_nodes,
        result.volume.shape,
        {"a.swc": 1, "b.swc": 2},
    )

    assert sum(float(group.volume.sum()) for group in individual) == 6.0
    assert [group.label for group in clusters] == [
        "Cluster 1",
        "Cluster 2",
        "Unclustered",
    ]
    assert sum(float(group.volume.sum()) for group in clusters) == 6.0


def test_allen_layer_stack_clips_xy_bins_to_canonical_bounds() -> None:
    nodes = pd.DataFrame(
        {
            "file_id": ["low.swc", "high.swc"],
            "x_flat": [-100.0, 100.0],
            "y_flat": [-100.0, 100.0],
            "flatmap_valid": [True, True],
            "region_id": [101, 101],
        }
    )

    result = build_allen_layer_stack_from_projected_nodes(
        nodes,
        _allen_layer_map(),
        xy_bins=4,
        x_bounds=(0.0, 10.0),
        y_bounds=(0.0, 10.0),
    )

    assert result.volume[0, 0, 0] == 1.0
    assert result.volume[0, 3, 3] == 1.0


# ---------------------------------------------------------------------------
# DuckDB-backed precomputed heatmap rendering.
# ---------------------------------------------------------------------------


def _v3_flatmap_frame(seed: int = 0, n: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5.0, 120.0, n).astype(np.float32)
    y = rng.uniform(-5.0, 90.0, n).astype(np.float32)
    depth = rng.uniform(-30.0, 900.0, n).astype(np.float32)
    depth[rng.integers(0, n, 150)] = -1.0
    flatmap_valid = rng.random(n) > 0.1
    depth_valid = (depth >= 0.0) & (rng.random(n) > 0.05)
    file_index = rng.integers(0, 6, n)
    return pd.DataFrame(
        {
            "file_id": [f"neuron_{i}" for i in file_index],
            "node_id": np.arange(n, dtype=np.int32),
            "parent_id": np.full(n, -1, dtype=np.int32),
            "type": np.full(n, 3, dtype=np.int32),
            "x_flat_shaped": x,
            "y_flat_shaped": y,
            "x_flat_square": x + 1.0,
            "y_flat_square": y + 1.0,
            "depth_um": depth,
            "flatmap_shaped_valid": flatmap_valid,
            "flatmap_square_valid": flatmap_valid,
            "depth_valid": depth_valid,
        }
    )


def _pandas_reference_volume(
    frame: pd.DataFrame,
    *,
    suffix: str,
    xy_bins: int,
    depth_bin_um: float,
    include: bool,
    x_bounds,
    y_bounds,
    depth_range_um,
):
    projected = pd.DataFrame(
        {
            "file_id": frame["file_id"].to_numpy(),
            "x_flat": pd.to_numeric(frame[f"x_flat_{suffix}"]),
            "y_flat": pd.to_numeric(frame[f"y_flat_{suffix}"]),
            "depth_um": pd.to_numeric(frame["depth_um"]),
            "flatmap_valid": frame[f"flatmap_{suffix}_valid"].astype(bool),
            "depth_valid": frame["depth_valid"].astype(bool),
        }
    )
    return build_flatmap_render_data_from_projected_nodes(
        projected,
        xy_bins=xy_bins,
        depth_bin_um=depth_bin_um,
        include_depth_minus_one=include,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        depth_range_um=depth_range_um,
    )


@pytest.fixture()
def _flatmap_parquet(tmp_path):
    frame = _v3_flatmap_frame()
    path = tmp_path / "flatmap_v3.parquet"
    frame.to_parquet(path, index=False)
    return frame, str(path)


@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize(
    "style_key,suffix", [("both_shaped", "shaped"), ("both_square", "square")]
)
def test_duckdb_single_volume_matches_pandas(
    _flatmap_parquet, include, style_key, suffix
):
    _frame, path = _flatmap_parquet
    x_bounds, y_bounds, depth_range = (0.0, 118.0), (0.0, 88.0), (0.0, 890.0)
    reference = _pandas_reference_volume(
        _frame,
        suffix=suffix,
        xy_bins=32,
        depth_bin_um=50.0,
        include=include,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        depth_range_um=depth_range,
    )
    conn = duckdb.connect()
    try:
        result = build_flatmap_heatmap_volume_result(
            conn,
            path,
            style_key=style_key,
            color_mode=FLATMAP_HEATMAP_COLOR_SINGLE,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            depth_range_um=depth_range,
            xy_bins=32,
            depth_bin_um=50.0,
            include_depth_minus_one=include,
        )
    finally:
        conn.close()

    assert result.volume.shape == reference.volume.shape
    np.testing.assert_array_equal(result.volume, reference.volume)
    assert result.render_summary.rendered_nodes == reference.summary.rendered_nodes
    assert result.render_summary.nonzero_voxels == reference.summary.nonzero_voxels
    assert (
        result.render_summary.flatmap_valid_nodes
        == reference.summary.flatmap_valid_nodes
    )
    assert (
        result.render_summary.depth_valid_nodes == reference.summary.depth_valid_nodes
    )


@pytest.mark.parametrize("include", [False, True])
def test_duckdb_grouped_volumes_sum_to_single(_flatmap_parquet, include):
    _frame, path = _flatmap_parquet
    x_bounds, y_bounds, depth_range = (0.0, 118.0), (0.0, 88.0), (0.0, 890.0)
    reference = _pandas_reference_volume(
        _frame,
        suffix="shaped",
        xy_bins=24,
        depth_bin_um=75.0,
        include=include,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        depth_range_um=depth_range,
    )
    conn = duckdb.connect()
    try:
        individual = build_flatmap_heatmap_volume_result(
            conn,
            path,
            style_key="both_shaped",
            color_mode=FLATMAP_HEATMAP_COLOR_INDIVIDUAL,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            depth_range_um=depth_range,
            xy_bins=24,
            depth_bin_um=75.0,
            include_depth_minus_one=include,
        )
        cluster_map = {f"neuron_{i}": i % 3 for i in range(6)}
        cluster = build_flatmap_heatmap_volume_result(
            conn,
            path,
            style_key="both_shaped",
            color_mode=FLATMAP_HEATMAP_COLOR_CLUSTER,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            depth_range_um=depth_range,
            xy_bins=24,
            depth_bin_um=75.0,
            include_depth_minus_one=include,
            cluster_map=cluster_map,
        )
    finally:
        conn.close()

    assert individual.volume is None
    combined_individual = np.zeros(individual.volume_shape, dtype=np.float32)
    for group in individual.grouped_volumes:
        combined_individual += group.volume
    np.testing.assert_array_equal(combined_individual, reference.volume)

    combined_cluster = np.zeros(cluster.volume_shape, dtype=np.float32)
    for group in cluster.grouped_volumes:
        combined_cluster += group.volume
    np.testing.assert_array_equal(combined_cluster, reference.volume)
    assert [group.label for group in cluster.grouped_volumes][:3] == [
        "Cluster 0",
        "Cluster 1",
        "Cluster 2",
    ]


def test_duckdb_file_id_subset_and_empty_selection(_flatmap_parquet):
    _frame, path = _flatmap_parquet
    bounds = dict(
        x_bounds=(0.0, 118.0), y_bounds=(0.0, 88.0), depth_range_um=(0.0, 890.0)
    )
    conn = duckdb.connect()
    try:
        subset = build_flatmap_heatmap_volume_result(
            conn,
            path,
            style_key="both_shaped",
            color_mode=FLATMAP_HEATMAP_COLOR_SINGLE,
            xy_bins=16,
            depth_bin_um=100.0,
            include_depth_minus_one=False,
            file_ids=["neuron_0", "neuron_1"],
            **bounds,
        )
        empty = build_flatmap_heatmap_volume_result(
            conn,
            path,
            style_key="both_shaped",
            color_mode=FLATMAP_HEATMAP_COLOR_SINGLE,
            xy_bins=16,
            depth_bin_um=100.0,
            include_depth_minus_one=False,
            file_ids=[],
            **bounds,
        )
    finally:
        conn.close()

    assert subset.render_summary.traces_represented <= 2
    assert subset.render_summary.rendered_nodes > 0
    assert empty.render_summary.rendered_nodes == 0
    assert float(empty.volume.sum()) == 0.0


def test_duckdb_bounds_fallback_matches_projected_nodes_bounds(_flatmap_parquet):
    _frame, path = _flatmap_parquet
    conn = duckdb.connect()
    try:
        bounds = compute_flatmap_bounds_from_parquet(
            conn, path, style_key="both_shaped", file_ids=None
        )
    finally:
        conn.close()
    assert set(bounds) == {"x_bounds", "y_bounds", "depth_range_um"}
    assert bounds["x_bounds"][0] < bounds["x_bounds"][1]
    assert bounds["depth_range_um"][0] >= 0.0


def test_duckdb_voxel_guard_rejects_oversized_grid(_flatmap_parquet):
    _frame, path = _flatmap_parquet
    conn = duckdb.connect()
    try:
        with pytest.raises(ValueError, match="too large"):
            build_flatmap_heatmap_volume_result(
                conn,
                path,
                style_key="both_shaped",
                color_mode=FLATMAP_HEATMAP_COLOR_SINGLE,
                x_bounds=(0.0, 118.0),
                y_bounds=(0.0, 88.0),
                depth_range_um=(0.0, 890.0),
                xy_bins=int(MAX_FLATMAP_HEATMAP_VOXELS**0.5) + 1000,
                depth_bin_um=1.0,
                include_depth_minus_one=False,
            )
    finally:
        conn.close()


def test_duckdb_allen_layer_stack_matches_pandas(tmp_path) -> None:
    frame = _v3_flatmap_frame(n=600)
    layer_regions = np.asarray([101, 102, 103, 104, 105, 106, 999])
    frame["region_id"] = np.resize(layer_regions, len(frame))
    path = tmp_path / "allen_layers.parquet"
    frame.to_parquet(path, index=False)
    layer_map = _allen_layer_map()
    projected = pd.DataFrame(
        {
            "file_id": frame["file_id"],
            "x_flat": frame["x_flat_shaped"],
            "y_flat": frame["y_flat_shaped"],
            "flatmap_valid": frame["flatmap_shaped_valid"],
            "region_id": frame["region_id"],
        }
    )
    reference = build_allen_layer_stack_from_projected_nodes(
        projected,
        layer_map,
        xy_bins=20,
        x_bounds=(0.0, 118.0),
        y_bounds=(0.0, 88.0),
    )

    conn = duckdb.connect()
    try:
        single = build_allen_layer_heatmap_volume_result(
            conn,
            str(path),
            style_key="both_shaped",
            color_mode=FLATMAP_HEATMAP_COLOR_SINGLE,
            layer_map=layer_map,
            x_bounds=(0.0, 118.0),
            y_bounds=(0.0, 88.0),
            xy_bins=20,
        )
        individual = build_allen_layer_heatmap_volume_result(
            conn,
            str(path),
            style_key="both_shaped",
            color_mode=FLATMAP_HEATMAP_COLOR_INDIVIDUAL,
            layer_map=layer_map,
            x_bounds=(0.0, 118.0),
            y_bounds=(0.0, 88.0),
            xy_bins=20,
        )
        cluster = build_allen_layer_heatmap_volume_result(
            conn,
            str(path),
            style_key="both_shaped",
            color_mode=FLATMAP_HEATMAP_COLOR_CLUSTER,
            layer_map=layer_map,
            x_bounds=(0.0, 118.0),
            y_bounds=(0.0, 88.0),
            xy_bins=20,
            cluster_map={f"neuron_{index}": index % 2 for index in range(6)},
        )
    finally:
        conn.close()

    np.testing.assert_array_equal(single.volume, reference.volume)
    assert single.summary.layer_node_counts == reference.summary.layer_node_counts
    assert single.summary.excluded_non_layer_nodes == (
        reference.summary.excluded_non_layer_nodes
    )
    for grouped in (individual, cluster):
        combined = np.zeros(grouped.volume_shape, dtype=np.float32)
        for group in grouped.grouped_volumes:
            combined += group.volume
        np.testing.assert_array_equal(combined, reference.volume)


def test_duckdb_allen_layer_stack_requires_region_id(
    _flatmap_parquet,
) -> None:
    _frame, path = _flatmap_parquet
    conn = duckdb.connect()
    try:
        with pytest.raises(ValueError, match="region_id"):
            build_allen_layer_heatmap_volume_result(
                conn,
                path,
                style_key="both_shaped",
                color_mode=FLATMAP_HEATMAP_COLOR_SINGLE,
                layer_map=_allen_layer_map(),
                x_bounds=(0.0, 118.0),
                y_bounds=(0.0, 88.0),
                xy_bins=20,
            )
    finally:
        conn.close()


def test_duckdb_allen_layer_stack_selected_files_and_empty_selection(
    tmp_path,
) -> None:
    frame = _v3_flatmap_frame(n=300)
    frame["region_id"] = np.resize(
        np.asarray([101, 102, 103, 104, 105, 106, 999]),
        len(frame),
    )
    path = tmp_path / "allen_layer_selection.parquet"
    frame.to_parquet(path, index=False)
    selected_ids = ["neuron_0", "neuron_1"]
    reference_nodes = pd.DataFrame(
        {
            "file_id": frame["file_id"],
            "x_flat": frame["x_flat_shaped"],
            "y_flat": frame["y_flat_shaped"],
            "flatmap_valid": frame["flatmap_shaped_valid"],
            "region_id": frame["region_id"],
        }
    )
    reference_nodes = reference_nodes[
        reference_nodes["file_id"].isin(selected_ids)
    ].reset_index(drop=True)
    reference = build_allen_layer_stack_from_projected_nodes(
        reference_nodes,
        _allen_layer_map(),
        xy_bins=12,
        x_bounds=(0.0, 118.0),
        y_bounds=(0.0, 88.0),
    )

    conn = duckdb.connect()
    try:
        selected = build_allen_layer_heatmap_volume_result(
            conn,
            str(path),
            style_key="both_shaped",
            color_mode=FLATMAP_HEATMAP_COLOR_SINGLE,
            layer_map=_allen_layer_map(),
            x_bounds=(0.0, 118.0),
            y_bounds=(0.0, 88.0),
            xy_bins=12,
            file_ids=selected_ids,
        )
        empty = build_allen_layer_heatmap_volume_result(
            conn,
            str(path),
            style_key="both_shaped",
            color_mode=FLATMAP_HEATMAP_COLOR_SINGLE,
            layer_map=_allen_layer_map(),
            x_bounds=(0.0, 118.0),
            y_bounds=(0.0, 88.0),
            xy_bins=12,
            file_ids=[],
        )
    finally:
        conn.close()

    np.testing.assert_array_equal(selected.volume, reference.volume)
    assert selected.summary.traces_represented <= 2
    assert empty.volume.shape == (6, 12, 12)
    assert float(empty.volume.sum()) == 0.0
    assert empty.summary.rendered_nodes == 0


def test_depth_plane_labels_name_each_micron_interval() -> None:
    labels = depth_plane_labels(
        {
            "depth_bins": 3,
            "depth_bin_um": 25.0,
            "depth_min_um": 0.0,
            "includes_depth_minus_one_plane": False,
        }
    )

    assert labels == ("0-25 um", "25-50 um", "50-75 um")


def test_depth_plane_labels_name_the_sentinel_plane_first() -> None:
    labels = depth_plane_labels(
        {
            "depth_bins": 3,
            "depth_bin_um": 25.0,
            "depth_min_um": 0.0,
            "includes_depth_minus_one_plane": True,
        }
    )

    assert labels == ("depth -1", "0-25 um", "25-50 um")


def test_depth_plane_labels_offset_by_the_lower_depth_bound() -> None:
    labels = depth_plane_labels(
        {
            "depth_bins": 2,
            "depth_bin_um": 50.0,
            "depth_min_um": 100.0,
            "includes_depth_minus_one_plane": False,
        }
    )

    assert labels == ("100-150 um", "150-200 um")


@pytest.mark.parametrize(
    "summary",
    [
        {},
        {"depth_bins": 0, "depth_bin_um": 25.0},
        {"depth_bins": 4, "depth_bin_um": 0.0},
        {"depth_bins": "many", "depth_bin_um": 25.0},
    ],
)
def test_depth_plane_labels_return_empty_without_depth_binning(summary) -> None:
    assert depth_plane_labels(summary) == ()
