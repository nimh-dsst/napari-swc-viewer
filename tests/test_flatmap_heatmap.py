from __future__ import annotations

import numpy as np
import pandas as pd

from napari_swc_viewer.flatmap_heatmap import (
    build_flatmap_render_data,
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


def test_compute_bounds_ignore_negative_one_sentinel() -> None:
    flatmap, depth = _lookup_volumes()

    assert compute_flatmap_xy_bounds(flatmap) == ((0.0, 6.0), (0.0, 6.0))
    assert compute_depth_range(depth) == (0.0, 150.0)


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
