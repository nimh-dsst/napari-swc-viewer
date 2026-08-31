from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import napari_neuron_navigator.comparison_data as comparison_data_module
from napari_neuron_navigator.cluster_assignments import ClusterAssignmentStore
from napari_neuron_navigator.comparison import (
    CCF_PLANE_CORONAL,
    REDUCTION_PROJECTION,
    SOURCE_CCF_HEATMAP,
    SOURCE_CCF_SOMAS,
    SOURCE_FLATMAP_SOMAS,
    ComparisonCellSpec,
)
from napari_neuron_navigator.comparison_data import ComparisonDataProvider


def _provider(
    path: Path,
    store: ClusterAssignmentStore,
    *,
    layers=(),
) -> ComparisonDataProvider:
    return ComparisonDataProvider(
        database_provider=lambda: SimpleNamespace(parquet_path=path),
        assignment_store_provider=lambda: store,
        viewer_layers_provider=lambda: list(layers),
        atlas_provider=lambda: SimpleNamespace(
            atlas_name="allen_mouse_25um",
            resolution=(25.0, 25.0, 25.0),
            annotation=SimpleNamespace(shape=(4, 5, 6)),
        ),
    )


def test_ccf_soma_render_is_scoped_and_colored_exclusively_by_file_id(
    tmp_path: Path,
):
    path = tmp_path / "neurons.parquet"
    pd.DataFrame(
        {
            "file_id": ["file-a", "file-a", "file-b", "file-b", "file-c"],
            "subject": ["A", "A", "B", "B", "C"],
            # The repeated display ID must not merge the first two neurons.
            "neuron_id": ["duplicate", "duplicate", "duplicate", "duplicate", "other"],
            "node_id": [1, 2, 1, 2, 1],
            "parent_id": [-1, 1, -1, 1, -1],
            "type": [1, 3, 1, 3, 1],
            "x": [10.0, 11.0, 20.0, 21.0, 30.0],
            "y": [100.0, 101.0, 200.0, 201.0, 300.0],
            "z": [1000.0, 1001.0, 2000.0, 2001.0, 3000.0],
        }
    ).to_parquet(path, index=False)
    store = ClusterAssignmentStore()
    assignment = store.add(
        name="Run A",
        assignments={"file-a": 1, "file-b": 2},
        input_file_ids=["file-a", "file-b", "file-c"],
        unassigned_neuron_ids=["file-c"],
        label_colors={
            1: [1.0, 0.0, 0.0, 1.0],
            2: [0.0, 1.0, 0.0, 1.0],
        },
    )
    provider = _provider(path, store)
    cell = ComparisonCellSpec(
        source_kind=SOURCE_CCF_SOMAS,
        assignment_id=assignment.assignment_id,
        ccf_plane=CCF_PLANE_CORONAL,
        reduction=REDUCTION_PROJECTION,
    )

    _prepared, render = provider.render_cell(
        cell,
        reference_assignment_id=assignment.assignment_id,
    )

    np.testing.assert_array_equal(
        render.points,
        np.asarray([[1000.0, 100.0], [2000.0, 200.0]]),
    )
    assert render.assigned_count == 2
    assert render.omitted_count == 1
    assert render.x_bounds == (-12.5, 137.5)
    assert render.y_bounds == (-12.5, 112.5)
    assert render.provenance["rendered_somas"] == 2
    assert render.provenance["original_cluster_ids"] == [1, 2]


def test_heatmap_groups_use_stable_source_ids_and_assignment_metadata(tmp_path: Path):
    path = tmp_path / "empty.parquet"
    path.write_bytes(b"signature-only")
    store = ClusterAssignmentStore()
    assignment = store.add(
        name="Run A",
        assignments={"file-a": 1, "file-b": 2},
        input_file_ids=["file-a", "file-b"],
    )
    signature = {
        "atlas_name": "allen_mouse_25um",
        "region_ids": [10],
        "node_types": [2],
        "soma_radius_um": None,
        "depth_axis": 0,
        "depth_bin_factor": 2,
    }
    layers = [
        SimpleNamespace(
            name=f"Cluster {label} Heatmap",
            data=np.full((2, 3, 4), label, dtype=np.float32),
            scale=(2.0, 1.0, 1.0),
            translate=(0.0, 0.0, 0.0),
            metadata={
                "heatmap_kind": "analysis",
                "atlas_name": "allen_mouse_25um",
                "heatmap_cluster": label,
                "heatmap_region_ids": [10],
                "heatmap_node_types": [2],
                "heatmap_soma_radius_um": None,
                "depth_axis": 0,
                "depth_bin_factor": 2,
                "comparison_source_id": f"source-{label}",
                "comparison_assignment_id": assignment.assignment_id,
                "comparison_assignment_name": assignment.name,
                "comparison_filter_signature": signature,
            },
        )
        for label in (1, 2)
    ]
    provider = _provider(path, store, layers=layers)

    groups = provider.heatmap_groups()

    assert len(groups) == 1
    assert groups[0].source_ids == ("source-1", "source-2")
    assert groups[0].assignment_id == assignment.assignment_id
    assert groups[0].label == "Run A — 2 cluster heatmaps"


def test_heatmap_composition_rejects_incompatible_source_geometry(tmp_path: Path):
    path = tmp_path / "empty.parquet"
    path.write_bytes(b"signature-only")
    store = ClusterAssignmentStore()
    assignment = store.add(
        name="Run A",
        assignments={"file-a": 1, "file-b": 2},
        input_file_ids=["file-a", "file-b"],
    )

    def layer(label: int, shape: tuple[int, int, int]):
        return SimpleNamespace(
            name=f"Cluster {label}",
            data=np.ones(shape, dtype=np.float32),
            scale=(1.0, 1.0, 1.0),
            translate=(0.0, 0.0, 0.0),
            metadata={
                "heatmap_kind": "analysis",
                "atlas_name": "allen_mouse_25um",
                "heatmap_cluster": label,
                "comparison_source_id": f"source-{label}",
                "comparison_assignment_id": assignment.assignment_id,
                "heatmap_region_ids": None,
                "heatmap_node_types": None,
                "heatmap_soma_radius_um": None,
                "depth_axis": 0,
                "depth_bin_factor": 1,
            },
        )

    provider = _provider(
        path,
        store,
        layers=[layer(1, (2, 3, 4)), layer(2, (3, 3, 4))],
    )
    cell = ComparisonCellSpec(
        source_kind=SOURCE_CCF_HEATMAP,
        assignment_id=assignment.assignment_id,
        comparison_source_ids=("source-1", "source-2"),
    )

    with pytest.raises(ValueError, match="same shape"):
        provider.render_cell(cell, reference_assignment_id=assignment.assignment_id)


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("scale", "same scale"),
        ("assignment", "provenance"),
        ("filter", "provenance"),
        ("atlas", "provenance"),
    ],
)
def test_heatmap_composition_validates_scale_assignment_filter_and_atlas(
    tmp_path: Path,
    mismatch: str,
    message: str,
):
    path = tmp_path / "empty.parquet"
    path.write_bytes(b"signature-only")
    store = ClusterAssignmentStore()
    assignment = store.add(
        name="Run A",
        assignments={"file-a": 1, "file-b": 2},
        input_file_ids=["file-a", "file-b"],
    )

    def layer(label: int):
        signature = {
            "atlas_name": "allen_mouse_25um",
            "region_ids": [10],
            "node_types": [2],
            "depth_axis": 0,
            "depth_bin_factor": 1,
        }
        metadata = {
            "heatmap_kind": "analysis",
            "atlas_name": "allen_mouse_25um",
            "heatmap_cluster": label,
            "comparison_source_id": f"source-{label}",
            "comparison_assignment_id": assignment.assignment_id,
            "comparison_filter_signature": signature,
        }
        scale = (1.0, 1.0, 1.0)
        if label == 2 and mismatch == "scale":
            scale = (2.0, 1.0, 1.0)
        if label == 2 and mismatch == "assignment":
            metadata["comparison_assignment_id"] = "different-assignment"
        if label == 2 and mismatch == "filter":
            metadata["comparison_filter_signature"] = {
                **signature,
                "node_types": [3],
            }
        if label == 2 and mismatch == "atlas":
            metadata["comparison_filter_signature"] = {
                **signature,
                "atlas_name": "different_atlas",
            }
        return SimpleNamespace(
            name=f"Cluster {label}",
            data=np.ones((2, 3, 4), dtype=np.float32),
            scale=scale,
            translate=(0.0, 0.0, 0.0),
            metadata=metadata,
        )

    provider = _provider(path, store, layers=[layer(1), layer(2)])
    cell = ComparisonCellSpec(
        source_kind=SOURCE_CCF_HEATMAP,
        assignment_id=assignment.assignment_id,
        comparison_source_ids=("source-1", "source-2"),
    )

    with pytest.raises(ValueError, match=message):
        provider.render_cell(cell, reference_assignment_id=assignment.assignment_id)


def test_missing_assignment_and_missing_heatmap_are_not_substituted_by_name(
    tmp_path: Path,
):
    path = tmp_path / "empty.parquet"
    path.write_bytes(b"signature-only")
    store = ClusterAssignmentStore()
    provider = _provider(path, store)

    with pytest.raises(ValueError, match="assignment.*missing"):
        provider.render_cell(
            ComparisonCellSpec(
                source_kind=SOURCE_CCF_SOMAS,
                assignment_id="deleted-id",
            ),
            reference_assignment_id=None,
        )
    with pytest.raises(ValueError, match="source is missing"):
        provider.render_cell(
            ComparisonCellSpec(
                source_kind=SOURCE_CCF_HEATMAP,
                comparison_source_ids=("deleted-source-id",),
            ),
            reference_assignment_id=None,
        )


def test_prepare_flatmap_recipe_never_rederives_a_stored_x_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "metadata.parquet"
    path.write_bytes(b"signature-only")
    provider = _provider(path, ClusterAssignmentStore())
    grid = SimpleNamespace(
        x_bounds=(-0.90425, 0.90425),
        y_bounds=(-0.4717, 0.4717),
    )
    monkeypatch.setattr(
        comparison_data_module,
        "read_flatmap_parquet_transform_info",
        lambda _path: SimpleNamespace(grid_spec=lambda _style: grid),
    )

    def forbidden_derivation(**_kwargs):
        raise AssertionError("stored x_bins must be authoritative")

    monkeypatch.setattr(
        comparison_data_module,
        "resolve_flatmap_bin_counts",
        forbidden_derivation,
    )
    prepared = provider.prepare_spec(
        ComparisonCellSpec(
            source_kind=SOURCE_FLATMAP_SOMAS,
            flatmap_style="both_shaped",
            y_bins=256,
            x_bins=491,
            x_bounds=(-0.9, 0.9),
            y_bounds=(-0.47, 0.47),
        )
    )

    assert prepared.x_bins == 491
    assert prepared.x_bounds == (-0.9, 0.9)
    assert prepared.y_bounds == (-0.47, 0.47)
