from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest

from napari_swc_viewer.flatmap_heatmap import build_flatmap_render_data
from napari_swc_viewer.flatmap_projection import (
    COORDINATE_MODE_MICRONS,
    COORDINATE_MODE_VOXELS,
    FLATMAP_LOOKUP_DIRECT,
    FLATMAP_LOOKUP_MIRRORED,
    FLATMAP_LOOKUP_MIRRORED_DEPTH,
    FLATMAP_LOOKUP_UNMAPPED,
    build_projected_segments,
    coordinates_to_voxel_indices,
    project_and_build_segments,
    project_neuron_nodes_to_flatmap,
    summarize_projection,
)


def _volumes(shape: tuple[int, int, int] = (4, 4, 4)) -> tuple[np.ndarray, np.ndarray]:
    grid = np.indices(shape, dtype=float)
    flatmap = np.stack((grid[0] + 0.25, grid[1] + 0.5), axis=-1)
    depth = grid[2] + 100.0
    return flatmap.astype(float), depth.astype(float)


def _nodes(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults = {
        "file_id": "cell.swc",
        "neuron_id": "cell",
        "subject": "subject",
        "type": 3,
        "region_id": 1,
        "region_acronym": "VISp",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_coordinates_to_voxel_indices_uses_nearest_10_um_lookup() -> None:
    voxels, finite = coordinates_to_voxel_indices(
        np.asarray([[4.9, 5.0, 14.9], [15.0, 25.1, 0.0]]),
        coordinate_mode=COORDINATE_MODE_MICRONS,
    )

    np.testing.assert_array_equal(voxels, [[0, 1, 1], [2, 3, 0]])
    np.testing.assert_array_equal(finite, [True, True])


def test_coordinates_to_voxel_indices_accepts_voxel_mode() -> None:
    voxels, finite = coordinates_to_voxel_indices(
        np.asarray([[0.49, 0.5, 1.49], [1.5, 2.51, 0.0]]),
        coordinate_mode=COORDINATE_MODE_VOXELS,
    )

    np.testing.assert_array_equal(voxels, [[0, 1, 1], [2, 3, 0]])
    np.testing.assert_array_equal(finite, [True, True])


def test_coordinates_to_voxel_indices_uses_nrrd_spatial_transform() -> None:
    directions = np.asarray([[0.0, 0.0, 10.0], [0.0, 10.0, 0.0], [10.0, 0.0, 0.0]])
    voxels, finite = coordinates_to_voxel_indices(
        np.asarray([[30.0, 20.0, 10.0], [10.0, 20.0, 30.0]]),
        coordinate_mode=COORDINATE_MODE_MICRONS,
        space_directions=directions,
        space_origin=np.zeros(3),
    )

    np.testing.assert_array_equal(voxels, [[1, 2, 3], [3, 2, 1]])
    np.testing.assert_array_equal(finite, [True, True])


def test_project_neuron_nodes_to_flatmap_preserves_metadata_for_valid_points() -> None:
    flatmap, depth = _volumes()
    projected = project_neuron_nodes_to_flatmap(
        _nodes(
            [
                {"node_id": 1, "parent_id": -1, "x": 10.0, "y": 20.0, "z": 30.0},
            ]
        ),
        flatmap,
        depth,
        flatmap_style="flatmap_both_shaped.nrrd",
    )

    row = projected.iloc[0]
    assert row["valid"] is np.True_
    assert row["flatmap_valid"] is np.True_
    assert row["depth_valid"] is np.True_
    assert row["invalid_reason"] == ""
    assert row["voxel_i"] == 1
    assert row["voxel_j"] == 2
    assert row["voxel_k"] == 3
    assert row["x_flat"] == pytest.approx(1.25)
    assert row["y_flat"] == pytest.approx(2.5)
    assert row["depth_um"] == pytest.approx(103.0)
    assert row["region_acronym"] == "VISp"
    assert row["flatmap_style"] == "flatmap_both_shaped.nrrd"
    assert row["flatmap_lookup_mode"] == FLATMAP_LOOKUP_DIRECT


def test_project_neuron_nodes_to_flatmap_uses_nrrd_spatial_transform() -> None:
    flatmap, depth = _volumes()
    directions = np.asarray([[0.0, 0.0, 10.0], [0.0, 10.0, 0.0], [10.0, 0.0, 0.0]])

    projected = project_neuron_nodes_to_flatmap(
        _nodes(
            [
                {"node_id": 1, "parent_id": -1, "x": 30.0, "y": 20.0, "z": 10.0},
            ]
        ),
        flatmap,
        depth,
        space_directions=directions,
        space_origin=np.zeros(3),
    )

    row = projected.iloc[0]
    assert row["valid"] is np.True_
    assert row["voxel_i"] == 1
    assert row["voxel_j"] == 2
    assert row["voxel_k"] == 3
    assert row["x_flat"] == pytest.approx(1.25)
    assert row["y_flat"] == pytest.approx(2.5)


def test_project_neuron_nodes_to_flatmap_rejects_negative_depth_sentinel() -> None:
    flatmap, depth = _volumes()
    depth[1, 0, 0] = -1.0

    projected = project_neuron_nodes_to_flatmap(
        _nodes(
            [
                {"node_id": 1, "parent_id": -1, "x": 10.0, "y": 0.0, "z": 0.0},
            ]
        ),
        flatmap,
        depth,
    )

    row = projected.iloc[0]
    assert row["valid"] is np.False_
    assert row["flatmap_valid"] is np.True_
    assert row["depth_valid"] is np.False_
    assert row["invalid_reason"] == "invalid_depth"
    assert row["flatmap_lookup_mode"] == FLATMAP_LOOKUP_UNMAPPED


def test_project_neuron_nodes_to_flatmap_keeps_zero_zero_valid_by_default() -> None:
    flatmap, depth = _volumes()
    flatmap[1, 0, 0] = (0.0, 0.0)

    default_projected = project_neuron_nodes_to_flatmap(
        _nodes(
            [
                {"node_id": 1, "parent_id": -1, "x": 10.0, "y": 0.0, "z": 0.0},
            ]
        ),
        flatmap,
        depth,
    )
    explicit_projected = project_neuron_nodes_to_flatmap(
        _nodes(
            [
                {"node_id": 1, "parent_id": -1, "x": 10.0, "y": 0.0, "z": 0.0},
            ]
        ),
        flatmap,
        depth,
        invalid_zero_sentinel=True,
    )

    assert default_projected.iloc[0]["valid"] is np.True_
    assert default_projected.iloc[0]["flatmap_lookup_mode"] == FLATMAP_LOOKUP_DIRECT
    assert explicit_projected.iloc[0]["valid"] is np.False_
    assert explicit_projected.iloc[0]["invalid_reason"] == "invalid_flatmap"


def test_project_neuron_nodes_to_flatmap_mirrors_only_invalid_direct_depth() -> None:
    flatmap, depth = _volumes()
    depth[1, 0, 0] = -1.0

    projected = project_neuron_nodes_to_flatmap(
        _nodes(
            [
                {"node_id": 1, "parent_id": -1, "x": 10.0, "y": 0.0, "z": 0.0},
            ]
        ),
        flatmap,
        depth,
        mirror_fallback=True,
        mirror_midline=15.0,
    )

    row = projected.iloc[0]
    assert row["valid"] is np.True_
    assert row["flatmap_lookup_mode"] == FLATMAP_LOOKUP_MIRRORED_DEPTH
    assert row["z_um"] == pytest.approx(0.0)
    assert row["voxel_k"] == 0
    assert row["x_flat"] == pytest.approx(1.25)
    assert row["y_flat"] == pytest.approx(0.5)
    assert row["depth_um"] == pytest.approx(103.0)
    assert row["invalid_reason"] == ""


def test_project_neuron_nodes_to_flatmap_keeps_direct_reason_when_mirror_fails() -> (
    None
):
    flatmap, depth = _volumes()
    depth[1, 0, 0] = -1.0
    depth[1, 0, 3] = -1.0

    projected = project_neuron_nodes_to_flatmap(
        _nodes(
            [
                {"node_id": 1, "parent_id": -1, "x": 10.0, "y": 0.0, "z": 0.0},
            ]
        ),
        flatmap,
        depth,
        mirror_fallback=True,
        mirror_midline=15.0,
    )

    row = projected.iloc[0]
    assert row["valid"] is np.False_
    assert row["flatmap_lookup_mode"] == FLATMAP_LOOKUP_UNMAPPED
    assert row["voxel_k"] == 0
    assert row["depth_um"] == pytest.approx(-1.0)
    assert row["invalid_reason"] == "invalid_depth"


def test_project_neuron_nodes_to_flatmap_voxel_mode_mirrors_at_grid_center() -> None:
    flatmap, depth = _volumes()
    depth[1, 0, 0] = -1.0

    projected = project_neuron_nodes_to_flatmap(
        _nodes(
            [
                {"node_id": 1, "parent_id": -1, "x": 1.0, "y": 0.0, "z": 0.0},
            ]
        ),
        flatmap,
        depth,
        coordinate_mode=COORDINATE_MODE_VOXELS,
        mirror_fallback=True,
    )

    row = projected.iloc[0]
    assert row["valid"] is np.True_
    assert row["flatmap_lookup_mode"] == FLATMAP_LOOKUP_MIRRORED_DEPTH
    assert row["voxel_k"] == 0
    assert row["depth_um"] == pytest.approx(103.0)


def test_project_mirrored_depth_honors_nrrd_spatial_transform() -> None:
    flatmap, depth = _volumes()
    depth[1, 0, 0] = -1.0

    projected = project_neuron_nodes_to_flatmap(
        _nodes(
            [
                {"node_id": 1, "parent_id": -1, "x": 10.0, "y": 0.0, "z": 0.0},
            ]
        ),
        flatmap,
        depth,
        space_directions=np.diag([10.0, 10.0, 10.0]),
        space_origin=np.zeros(3),
        mirror_fallback=True,
        mirror_midline=15.0,
    )

    row = projected.iloc[0]
    assert row["flatmap_lookup_mode"] == FLATMAP_LOOKUP_MIRRORED_DEPTH
    assert row["voxel_k"] == 0
    assert row["x_flat"] == pytest.approx(1.25)
    assert row["depth_um"] == pytest.approx(103.0)


def test_project_mirrored_depth_honors_custom_mirror_axis() -> None:
    flatmap, depth = _volumes()
    depth[0, 1, 0] = -1.0
    depth[3, 1, 0] = 333.0

    projected = project_neuron_nodes_to_flatmap(
        _nodes(
            [
                {"node_id": 1, "parent_id": -1, "x": 0.0, "y": 10.0, "z": 0.0},
            ]
        ),
        flatmap,
        depth,
        mirror_fallback=True,
        mirror_coord_axis=0,
        mirror_midline=15.0,
    )

    row = projected.iloc[0]
    assert row["flatmap_lookup_mode"] == FLATMAP_LOOKUP_MIRRORED_DEPTH
    assert (row["voxel_i"], row["voxel_j"], row["voxel_k"]) == (0, 1, 0)
    assert row["x_flat"] == pytest.approx(0.25)
    assert row["depth_um"] == pytest.approx(333.0)


def test_project_neuron_nodes_to_flatmap_full_mirrors_invalid_flatmap() -> None:
    flatmap, depth = _volumes()
    flatmap[1, 0, 0] = (-1.0, -1.0)

    projected = project_neuron_nodes_to_flatmap(
        _nodes(
            [
                {"node_id": 1, "parent_id": -1, "x": 1.0, "y": 0.0, "z": 0.0},
            ]
        ),
        flatmap,
        depth,
        coordinate_mode=COORDINATE_MODE_VOXELS,
        mirror_fallback=True,
    )

    row = projected.iloc[0]
    assert row["valid"] is np.True_
    assert row["flatmap_lookup_mode"] == FLATMAP_LOOKUP_MIRRORED
    assert row["voxel_k"] == 3
    assert row["depth_um"] == pytest.approx(103.0)


@pytest.mark.parametrize(
    ("invalid_lookup", "expected_mode", "expected_voxel_k"),
    [
        ("depth", FLATMAP_LOOKUP_MIRRORED_DEPTH, 0),
        ("flatmap", FLATMAP_LOOKUP_MIRRORED, 3),
    ],
)
def test_mirror_fallback_supports_copy_on_write_without_mutating_inputs(
    invalid_lookup: str,
    expected_mode: str,
    expected_voxel_k: int,
) -> None:
    flatmap, depth = _volumes()
    if invalid_lookup == "depth":
        depth[1, 0, 0] = -1.0
    else:
        flatmap[1, 0, 0] = (-1.0, -1.0)
    nodes = _nodes(
        [
            {
                "node_id": 1,
                "parent_id": -1,
                "x": 1.0,
                "y": 0.0,
                "z": 0.0,
            },
        ]
    )
    original_nodes = nodes.copy(deep=True)
    original_flatmap = flatmap.copy()
    original_depth = depth.copy()

    copy_on_write = (
        nullcontext()
        if int(pd.__version__.partition(".")[0]) >= 3
        else pd.option_context("mode.copy_on_write", True)
    )
    with copy_on_write:
        projected = project_neuron_nodes_to_flatmap(
            nodes,
            flatmap,
            depth,
            coordinate_mode=COORDINATE_MODE_VOXELS,
            mirror_fallback=True,
        )

    row = projected.iloc[0]
    assert row["valid"] is np.True_
    assert row["flatmap_lookup_mode"] == expected_mode
    assert row["voxel_k"] == expected_voxel_k
    assert row["depth_um"] == pytest.approx(103.0)
    pd.testing.assert_frame_equal(nodes, original_nodes)
    np.testing.assert_array_equal(flatmap, original_flatmap)
    np.testing.assert_array_equal(depth, original_depth)


@pytest.mark.parametrize("invalid_depth", [-1.0, np.nan])
def test_project_bilateral_nodes_preserve_distinct_flatmap_panels(
    invalid_depth: float,
) -> None:
    flatmap = np.full((1, 1, 4, 2), -1.0, dtype=np.float32)
    flatmap[0, 0, 0] = (0.1, 0.5)
    flatmap[0, 0, 3] = (1.9, 0.5)
    depth = np.full((1, 1, 4), invalid_depth, dtype=np.float32)
    depth[0, 0, 3] = 100.0
    nodes = _nodes(
        [
            {
                "file_id": "left.swc",
                "node_id": 1,
                "parent_id": -1,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
            },
            {
                "file_id": "right.swc",
                "node_id": 1,
                "parent_id": -1,
                "x": 0.0,
                "y": 0.0,
                "z": 3.0,
            },
        ]
    )

    projected = project_neuron_nodes_to_flatmap(
        nodes,
        flatmap,
        depth,
        coordinate_mode=COORDINATE_MODE_VOXELS,
        mirror_fallback=True,
    )

    assert len(projected) == 2
    assert projected["x_flat"].tolist() == pytest.approx([0.1, 1.9])
    assert projected["voxel_k"].tolist() == [0, 3]
    assert projected["depth_um"].tolist() == pytest.approx([100.0, 100.0])
    assert projected["flatmap_lookup_mode"].tolist() == [
        FLATMAP_LOOKUP_MIRRORED_DEPTH,
        FLATMAP_LOOKUP_DIRECT,
    ]
    summary = summarize_projection(projected, build_projected_segments(projected))
    assert summary.direct_lookup_nodes == 1
    assert summary.mirrored_depth_lookup_nodes == 1
    assert summary.mirrored_lookup_nodes == 0
    assert summary.unmapped_lookup_nodes == 0
    assert summary.to_dict()["mirrored_depth_lookup_nodes"] == 1
    render = build_flatmap_render_data(
        projected,
        flatmap,
        depth,
        xy_bins=20,
        depth_bin_um=25.0,
        include_depth_minus_one=False,
    )
    assert render.projected_nodes["x_flat_bin"].tolist() == [0, 19]
    assert render.summary.rendered_nodes == 2
    assert render.summary.nonzero_voxels == 2

    unilateral = project_neuron_nodes_to_flatmap(
        nodes.iloc[[0]].reset_index(drop=True),
        flatmap,
        depth,
        coordinate_mode=COORDINATE_MODE_VOXELS,
        mirror_fallback=True,
    )
    assert len(unilateral) == 1
    assert unilateral["x_flat"].tolist() == pytest.approx([0.1])


def test_projection_classifies_out_of_bounds_missing_and_invalid_values() -> None:
    flatmap, depth = _volumes()
    flatmap[1, 0, 0] = np.nan
    flatmap[2, 0, 0] = (0.0, 0.0)
    flatmap[3, 0, 0] = (-1.0, -1.0)
    depth[3, 1, 0] = np.nan
    projected = project_neuron_nodes_to_flatmap(
        _nodes(
            [
                {"node_id": 1, "parent_id": -1, "x": 1000.0, "y": 0.0, "z": 0.0},
                {"node_id": 2, "parent_id": -1, "x": np.nan, "y": 0.0, "z": 0.0},
                {"node_id": 3, "parent_id": -1, "x": 10.0, "y": 0.0, "z": 0.0},
                {"node_id": 4, "parent_id": -1, "x": 20.0, "y": 0.0, "z": 0.0},
                {"node_id": 5, "parent_id": -1, "x": 30.0, "y": 0.0, "z": 0.0},
                {"node_id": 6, "parent_id": -1, "x": 30.0, "y": 10.0, "z": 0.0},
            ]
        ),
        flatmap,
        depth,
        invalid_zero_sentinel=True,
    )

    assert projected["invalid_reason"].tolist() == [
        "out_of_bounds",
        "missing_input",
        "invalid_flatmap",
        "invalid_flatmap",
        "invalid_flatmap",
        "invalid_depth",
    ]
    assert not projected["valid"].any()


def test_build_projected_segments_skips_only_invalid_edges() -> None:
    flatmap, depth = _volumes()
    flatmap[1, 0, 0] = np.nan
    result = project_and_build_segments(
        _nodes(
            [
                {"node_id": 1, "parent_id": -1, "x": 0.0, "y": 0.0, "z": 0.0},
                {"node_id": 2, "parent_id": 1, "x": 10.0, "y": 0.0, "z": 0.0},
                {"node_id": 3, "parent_id": 2, "x": 20.0, "y": 0.0, "z": 0.0},
                {"node_id": 4, "parent_id": 1, "x": 30.0, "y": 0.0, "z": 0.0},
            ]
        ),
        flatmap,
        depth,
    )

    assert result.segments.data.shape == (1, 2, 2)
    assert result.segments.source_node_ids == [1]
    assert result.segments.target_node_ids == [4]
    assert result.summary.rendered_segments == 1
    assert result.summary.traces_with_partial_projection == 1
    assert result.summary.direct_lookup_nodes == 3
    assert result.summary.unmapped_lookup_nodes == 1


def test_build_projected_segments_honors_a_chosen_validity_column() -> None:
    # A depth-collapsed render marks nodes valid on flatmap XY alone, so the
    # middle node here is rendered even though its depth is invalid.  Selecting
    # that render's own column has to keep the edges the render included.
    nodes = pd.DataFrame(
        {
            "file_id": ["a.swc"] * 3,
            "node_id": [1, 2, 3],
            "parent_id": [-1, 1, 2],
            "x_flat": [0.0, 1.0, 2.0],
            "y_flat": [0.0, 1.0, 2.0],
            "valid": [True, False, True],
            "render_valid": [True, True, True],
        }
    )

    depth_gated = build_projected_segments(nodes)
    collapsed = build_projected_segments(nodes, validity_column="render_valid")

    assert depth_gated.data.shape == (0, 2, 2)
    assert collapsed.data.shape == (2, 2, 2)
    assert collapsed.source_node_ids == [1, 2]
    assert collapsed.target_node_ids == [2, 3]


def test_build_projected_segments_keeps_neurons_separate_by_file_id() -> None:
    # node_id is only unique within a file_id, so two neurons reusing the same
    # ids must not be joined into a cross-neuron edge.
    nodes = pd.DataFrame(
        {
            "file_id": ["a.swc", "a.swc", "b.swc", "b.swc"],
            "node_id": [1, 2, 1, 2],
            "parent_id": [-1, 1, -1, 1],
            "x_flat": [0.0, 1.0, 10.0, 11.0],
            "y_flat": [0.0, 1.0, 10.0, 11.0],
            "valid": [True] * 4,
        }
    )

    segments = build_projected_segments(nodes)

    assert segments.data.shape == (2, 2, 2)
    assert segments.file_ids == ["a.swc", "b.swc"]
    np.testing.assert_allclose(segments.data[0, 0], [0.0, 0.0])
    np.testing.assert_allclose(segments.data[1, 0], [10.0, 10.0])


def test_summarize_projection_counts_multiple_invalid_reasons() -> None:
    projected = pd.DataFrame(
        {
            "file_id": ["a", "a", "b"],
            "valid": [True, False, False],
            "invalid_reason": ["", "out_of_bounds", "invalid_depth"],
        }
    )
    segments = build_projected_segments(
        pd.DataFrame(
            {
                "file_id": [],
                "node_id": [],
                "parent_id": [],
                "x_flat": [],
                "y_flat": [],
                "valid": [],
            }
        )
    )

    summary = summarize_projection(projected, segments)

    assert summary.total_nodes == 3
    assert summary.valid_nodes == 1
    assert summary.out_of_bounds_nodes == 1
    assert summary.invalid_depth_nodes == 1
    assert summary.total_traces == 2
    assert summary.traces_with_partial_projection == 1
