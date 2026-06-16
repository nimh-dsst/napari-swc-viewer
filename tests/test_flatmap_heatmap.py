from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from napari_swc_viewer.flatmap_heatmap import (
    build_flatmap_render_data,
    compute_flatmap_lookup_stats,
    compute_depth_range,
    compute_flatmap_xy_bounds,
)


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
        np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 3.0, 3.0], [3.0, 1.0, 1.0]]),
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
