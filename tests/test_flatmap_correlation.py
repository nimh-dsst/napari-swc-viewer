from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from napari_swc_viewer.analysis.flatmap_correlation import (
    FlatmapVoxelCorrelationSource,
    build_flatmap_count_matrix,
    compute_flatmap_voxel_correlation_result,
    pearson_correlation_from_counts,
)


def _source(projected_nodes: pd.DataFrame) -> FlatmapVoxelCorrelationSource:
    return FlatmapVoxelCorrelationSource(
        projected_nodes=projected_nodes,
        volume_shape=(2, 2, 2),
        input_file_ids=("a.swc", "b.swc", "c.swc"),
        xy_bins=2,
        depth_bin_um=25.0,
        include_depth_minus_one=False,
    )


def test_build_flatmap_count_matrix_uses_rendered_binned_nodes() -> None:
    projected = pd.DataFrame(
        {
            "file_id": ["a.swc", "a.swc", "b.swc", "c.swc"],
            "render_valid": [True, True, True, False],
            "depth_bin": [0, 0, 1, 0],
            "y_flat_bin": [0, 0, 1, 1],
            "x_flat_bin": [0, 1, 1, 1],
        }
    )

    result = build_flatmap_count_matrix(_source(projected))

    assert result.neuron_ids == ["a.swc", "b.swc"]
    assert result.unassigned_neuron_ids == ["c.swc"]
    assert result.rendered_node_count == 3
    assert result.voxel_ids.tolist() == [0, 1, 7]
    np.testing.assert_array_equal(
        result.count_matrix,
        np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )


def test_build_flatmap_count_matrix_includes_mirrored_rendered_nodes() -> None:
    projected = pd.DataFrame(
        {
            "file_id": ["a.swc", "b.swc", "c.swc"],
            "flatmap_lookup_mode": ["mirrored", "direct", "unmapped"],
            "render_valid": [True, True, False],
            "depth_bin": [0, 1, 1],
            "y_flat_bin": [0, 1, 1],
            "x_flat_bin": [0, 1, 0],
        }
    )

    result = build_flatmap_count_matrix(_source(projected))

    assert result.neuron_ids == ["a.swc", "b.swc"]
    assert result.unassigned_neuron_ids == ["c.swc"]
    assert result.rendered_node_count == 2


def test_build_flatmap_count_matrix_applies_region_mask() -> None:
    projected = pd.DataFrame(
        {
            "file_id": ["a.swc", "b.swc", "c.swc"],
            "render_valid": [True, True, True],
            "depth_bin": [0, 0, 1],
            "y_flat_bin": [0, 1, 1],
            "x_flat_bin": [0, 0, 1],
        }
    )
    mask = np.zeros((2, 2, 2), dtype=bool)
    mask[0, 0, 0] = True

    result = build_flatmap_count_matrix(_source(projected), region_mask=mask)

    assert result.neuron_ids == ["a.swc"]
    assert result.unassigned_neuron_ids == ["b.swc", "c.swc"]
    assert result.voxel_ids.tolist() == [0]


def test_pearson_correlation_from_counts_handles_zero_variance_rows() -> None:
    corr = pearson_correlation_from_counts(
        np.array(
            [
                [1.0, 2.0, 0.0],
                [1.0, 2.0, 0.0],
                [3.0, 3.0, 3.0],
            ],
            dtype=np.float32,
        )
    )

    np.testing.assert_allclose(np.diag(corr), 1.0)
    assert corr[0, 1] == pytest.approx(1.0)
    assert corr[0, 2] == pytest.approx(0.0)
    assert corr[2, 0] == pytest.approx(0.0)


def test_compute_flatmap_voxel_correlation_result_records_unassigned() -> None:
    projected = pd.DataFrame(
        {
            "file_id": ["a.swc", "b.swc", "c.swc"],
            "render_valid": [True, True, False],
            "depth_bin": [0, 0, 1],
            "y_flat_bin": [0, 1, 1],
            "x_flat_bin": [0, 1, 1],
        }
    )

    result, count_data = compute_flatmap_voxel_correlation_result(
        _source(projected),
        method="average",
        n_clusters=2,
    )

    assert result.neuron_ids == ["a.swc", "b.swc"]
    assert result.unassigned_neuron_ids == ["c.swc"]
    assert count_data.rendered_node_count == 2
    assert result.correlation_matrix.shape == (2, 2)


def test_compute_flatmap_voxel_correlation_requires_two_rendered_neurons() -> None:
    projected = pd.DataFrame(
        {
            "file_id": ["a.swc", "b.swc"],
            "render_valid": [True, False],
            "depth_bin": [0, 0],
            "y_flat_bin": [0, 1],
            "x_flat_bin": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="at least 2 rendered neurons"):
        compute_flatmap_voxel_correlation_result(_source(projected))
