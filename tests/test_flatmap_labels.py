from __future__ import annotations

import numpy as np
import pytest

from napari_swc_viewer.flatmap_labels import build_flatmap_region_label_volume


def _grid_volumes(
    shape: tuple[int, int, int] = (2, 2, 2),
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.indices(shape, dtype=float)
    flatmap = np.stack((grid[0] * 10.0, grid[1] * 10.0), axis=-1)
    depth = grid[2] * 25.0
    return flatmap.astype(np.float32), depth.astype(np.float32)


def test_build_flatmap_region_label_volume_remaps_selected_voxels_to_bins() -> None:
    annotation = np.zeros((2, 2, 2), dtype=np.int32)
    annotation[1, 1, 1] = 101
    flatmap, depth = _grid_volumes()

    result = build_flatmap_region_label_volume(
        annotation,
        flatmap,
        depth,
        selected_region_ids=[101],
        xy_bins=2,
        depth_bin_um=25.0,
    )

    assert result.labels.shape == (2, 2, 2)
    assert result.labels[1, 1, 1] == 101
    assert np.count_nonzero(result.labels) == 1
    assert result.selected_region_ids == [101]
    assert result.represented_region_ids == [101]
    assert result.summary.selected_source_voxels == 1
    assert result.summary.valid_source_voxels == 1
    assert result.summary.labeled_voxels == 1


def test_build_flatmap_region_label_volume_filters_selected_region_ids() -> None:
    annotation = np.array([[[1, 2], [3, 4]]], dtype=np.int32)
    flatmap, depth = _grid_volumes(annotation.shape)

    result = build_flatmap_region_label_volume(
        annotation,
        flatmap,
        depth,
        selected_region_ids=[2, 4],
        xy_bins=2,
        depth_bin_um=25.0,
    )

    assert sorted(np.unique(result.labels).tolist()) == [0, 2, 4]
    assert result.summary.selected_source_voxels == 2
    assert result.summary.labeled_voxels == 2


def test_build_flatmap_region_label_volume_uses_majority_region_for_collisions() -> None:
    annotation = np.asarray([[[7]], [[7]], [[9]]], dtype=np.int32)
    flatmap = np.zeros((3, 1, 1, 2), dtype=np.float32)
    depth = np.zeros((3, 1, 1), dtype=np.float32)

    result = build_flatmap_region_label_volume(
        annotation,
        flatmap,
        depth,
        selected_region_ids=[7, 9],
        xy_bins=1,
        depth_bin_um=25.0,
    )

    assert result.labels.shape == (1, 1, 1)
    assert result.labels[0, 0, 0] == 7
    assert result.summary.collision_voxels == 1
    assert result.summary.valid_source_voxels == 3
    assert result.summary.labeled_voxels == 1


def test_build_flatmap_region_label_volume_ties_choose_smaller_region_id() -> None:
    annotation = np.asarray([[[9]], [[7]]], dtype=np.int32)
    flatmap = np.zeros((2, 1, 1, 2), dtype=np.float32)
    depth = np.zeros((2, 1, 1), dtype=np.float32)

    result = build_flatmap_region_label_volume(
        annotation,
        flatmap,
        depth,
        selected_region_ids=[7, 9],
        xy_bins=1,
        depth_bin_um=25.0,
    )

    assert result.labels[0, 0, 0] == 7
    assert result.summary.collision_voxels == 1


def test_build_flatmap_region_label_volume_leaves_invalid_lookup_voxels_unlabeled() -> None:
    annotation = np.asarray([[[1]], [[2]], [[3]]], dtype=np.int32)
    flatmap = np.zeros((3, 1, 1, 2), dtype=np.float32)
    depth = np.zeros((3, 1, 1), dtype=np.float32)
    flatmap[1, 0, 0] = (-1.0, -1.0)
    depth[2, 0, 0] = -1.0

    result = build_flatmap_region_label_volume(
        annotation,
        flatmap,
        depth,
        selected_region_ids=[1, 2, 3],
        xy_bins=1,
        depth_bin_um=25.0,
    )

    assert result.labels[0, 0, 0] == 1
    assert result.summary.selected_source_voxels == 3
    assert result.summary.valid_source_voxels == 1
    assert result.represented_region_ids == [1]


def test_build_flatmap_region_label_volume_rejects_shape_mismatch() -> None:
    flatmap, depth = _grid_volumes()

    with pytest.raises(ValueError, match="annotation shape"):
        build_flatmap_region_label_volume(
            np.zeros((1, 2, 2), dtype=np.int32),
            flatmap,
            depth,
            selected_region_ids=[1],
        )
