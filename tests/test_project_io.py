from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from napari_swc_viewer.cluster_assignments import ClusterAssignmentStore
from napari_swc_viewer.project_io import (
    PROJECT_BUNDLE_FORMAT,
    ENHANCED_NEURON_COLUMNS,
    export_enhanced_neuron_parquet,
    export_filtered_project_neuron_parquet,
    load_project_bundle,
    read_enhanced_parquet_metadata,
    save_project_bundle,
)


def _write_source_parquet(path: Path) -> None:
    pd.DataFrame(
        {
            "file_id": ["n1", "n1", "n2"],
            "node_id": [1, 2, 1],
            "type": [1, 3, 1],
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
            "z": [7.0, 8.0, 9.0],
            "radius": [1.0, 1.0, 1.0],
            "parent_id": [-1, 1, -1],
            "region_id": [1, 1, 2],
            "region_name": ["A", "A", "B"],
            "region_acronym": ["A", "A", "B"],
            "subject": ["s1", "s1", "s2"],
            "neuron_id": ["neuron1", "neuron1", "neuron2"],
        }
    ).to_parquet(path, index=False)


def _write_three_neuron_source_parquet(path: Path) -> None:
    pd.DataFrame(
        {
            "file_id": ["n1", "n1", "n2", "n3", "n3"],
            "node_id": [1, 2, 1, 1, 2],
            "type": [1, 3, 1, 1, 3],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [6.0, 7.0, 8.0, 9.0, 10.0],
            "z": [11.0, 12.0, 13.0, 14.0, 15.0],
            "radius": [1.0, 1.0, 1.0, 2.0, 2.0],
            "parent_id": [-1, 1, -1, -1, 1],
            "region_id": [1, 1, 2, 3, 3],
            "region_name": ["A", "A", "B", "C", "C"],
            "region_acronym": ["A", "A", "B", "C", "C"],
            "subject": ["s1", "s1", "s2", "s3", "s3"],
            "neuron_id": ["neuron1", "neuron1", "neuron2", "neuron3", "neuron3"],
        }
    ).to_parquet(path, index=False)


def _add_flatmap_v3_schema_metadata(path: Path) -> dict[bytes, bytes]:
    """Attach representative v3 and custom Arrow metadata to a fixture."""
    table = pq.read_table(path)
    metadata = {
        b"napari_swc_viewer.flatmap_projection_json": (
            b'{"version":3,"lookup_set_id":"lookup-test"}'
        ),
        b"custom.dataset_metadata": b"must-survive",
    }
    fields = [
        field.with_metadata({b"units": b"micrometer"}) if field.name == "x" else field
        for field in table.schema
    ]
    schema = pa.schema(fields, metadata=metadata)
    pq.write_table(pa.Table.from_arrays(table.columns, schema=schema), path)
    return metadata


def _table_state() -> dict[str, object]:
    return {
        "version": 1,
        "entries": [
            {
                "file_id": "n1",
                "subject": "s1",
                "label": "projection",
                "group": "A",
                "tags": ["axon", "reviewed"],
                "notes": "keep",
                "cluster_id": 3,
                "color": [0.1, 0.2, 0.3, 1.0],
                "visible": False,
            }
        ],
    }


def _subset_table_state() -> dict[str, object]:
    return {
        "version": 1,
        "entries": [
            {
                "file_id": "n3",
                "subject": "s3",
                "label": "target",
                "group": "GPe",
                "tags": ["filtered"],
                "notes": "save",
                "cluster_id": 7,
            },
            {
                "file_id": "n1",
                "subject": "s1",
                "label": "source",
                "group": "CTX",
                "tags": ["reviewed"],
                "notes": "keep",
                "cluster_id": 2,
            },
        ],
    }


def _multi_assignment_table_state() -> dict[str, object]:
    store = ClusterAssignmentStore()
    soma = store.add(
        name="Soma Location 1",
        assignments={"n1": 1, "n2": 1, "n3": 2},
        input_file_ids=["n1", "n2", "n3"],
        activate=True,
    )
    store.add(
        name="Voxel Correlation 1",
        assignments={"n1": 2, "n2": 1},
        input_file_ids=["n1", "n2"],
        parent_assignment_id=soma.assignment_id,
        parent_cluster_ids=[1],
        activate=True,
    )
    return {
        "version": 2,
        "entries": [
            {"file_id": "n1", "subject": "s1", "cluster_id": 2},
            {"file_id": "n2", "subject": "s2", "cluster_id": 1},
            {"file_id": "n3", "subject": "s3", "cluster_id": None},
        ],
        "cluster_assignments": store.to_state(),
    }


@dataclass
class _DummyLayer:
    name: str
    data: np.ndarray
    metadata: dict[str, object]
    opacity: float = 0.6
    visible: bool = True
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0)
    blending: str = "additive"
    rendering: str = "mip"
    contrast_limits: tuple[float, float] = (0.0, 1.0)
    colormap: object = field(default_factory=lambda: SimpleNamespace(name="hot"))


class _DummyColormap:
    def __init__(self, name: str, colors: list[list[float]]) -> None:
        self.name = name
        self.colors = np.asarray(colors, dtype=np.float32)
        self.controls = np.linspace(0.0, 1.0, len(colors), dtype=np.float32)
        self.interpolation = "linear"
        self.low_color = None
        self.high_color = None
        self.nan_color = [0.0, 0.0, 0.0, 0.0]


def test_export_enhanced_neuron_parquet_round_trips_labels_and_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "enhanced.parquet"
    _write_source_parquet(source)

    export_enhanced_neuron_parquet(
        source,
        output,
        table_state=_table_state(),
        metadata={"atlas_name": "fake_atlas"},
    )

    loaded = (
        pd.read_parquet(output)
        .sort_values(["file_id", "node_id"])
        .reset_index(drop=True)
    )
    assert len(loaded) == 3
    assert all(column in loaded.columns for column in ENHANCED_NEURON_COLUMNS)
    assert loaded.loc[loaded["file_id"] == "n1", "neuron_label"].unique().tolist() == [
        "projection"
    ]
    cluster_values = (
        loaded.loc[loaded["file_id"] == "n1", "cluster_assignment"]
        .dropna()
        .unique()
        .tolist()
    )
    assert cluster_values == [3]
    assert loaded.loc[loaded["file_id"] == "n2", "cluster_assignment"].isna().all()

    payload = read_enhanced_parquet_metadata(output)
    assert payload["metadata"]["atlas_name"] == "fake_atlas"
    assert payload["table_state"]["entries"][0]["label"] == "projection"
    assert payload["enhanced_columns"] == [
        *ENHANCED_NEURON_COLUMNS,
        "cluster_imported_cluster_assignment",
    ]


def test_enhanced_parquet_writes_all_named_assignment_columns_and_active_mirror(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "enhanced.parquet"
    _write_three_neuron_source_parquet(source)

    export_enhanced_neuron_parquet(
        source,
        output,
        table_state=_multi_assignment_table_state(),
    )

    loaded = pd.read_parquet(output)
    assert set(loaded.loc[loaded["file_id"] == "n1", "cluster_soma_location_1"]) == {1}
    assert set(loaded.loc[loaded["file_id"] == "n3", "cluster_soma_location_1"]) == {2}
    assert set(
        loaded.loc[loaded["file_id"] == "n1", "cluster_voxel_correlation_1"]
    ) == {2}
    assert (
        loaded.loc[loaded["file_id"] == "n3", "cluster_voxel_correlation_1"]
        .isna()
        .all()
    )
    assert set(loaded.loc[loaded["file_id"] == "n2", "cluster_assignment"]) == {1}
    assert loaded.loc[loaded["file_id"] == "n3", "cluster_assignment"].isna().all()

    payload = read_enhanced_parquet_metadata(output)
    registry = payload["table_state"]["cluster_assignments"]
    assert registry["active_assignment_id"] == registry["sets"][1]["assignment_id"]
    assert [item["name"] for item in registry["sets"]] == [
        "Soma Location 1",
        "Voxel Correlation 1",
    ]


def test_enhanced_parquet_uses_collision_safe_assignment_column(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "enhanced.parquet"
    _write_three_neuron_source_parquet(source)
    source_df = pd.read_parquet(source)
    source_df["cluster_soma_location_1"] = 99
    source_df.to_parquet(source, index=False)

    export_enhanced_neuron_parquet(
        source,
        output,
        table_state=_multi_assignment_table_state(),
    )

    loaded = pd.read_parquet(output)
    assert set(loaded["cluster_soma_location_1"]) == {99}
    assert "cluster_soma_location_1_2" in loaded.columns
    payload = read_enhanced_parquet_metadata(output)
    assert (
        payload["table_state"]["cluster_assignments"]["sets"][0]["column_name"]
        == "cluster_soma_location_1_2"
    )


def test_enhanced_parquet_assignment_column_cannot_replace_active_mirror(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "enhanced.parquet"
    _write_source_parquet(source)
    store = ClusterAssignmentStore()
    store.add(
        name="Assignment",
        assignments={"n1": 7},
        input_file_ids=["n1", "n2"],
    )
    state = {
        "version": 2,
        "entries": [{"file_id": "n1"}, {"file_id": "n2"}],
        "cluster_assignments": store.to_state(),
    }

    export_enhanced_neuron_parquet(source, output, table_state=state)

    loaded = pd.read_parquet(output)
    assert "cluster_assignment" in loaded.columns
    assert "cluster_assignment_2" in loaded.columns
    assert set(loaded.loc[loaded["file_id"] == "n1", "cluster_assignment"]) == {7}


def test_enhanced_and_filtered_exports_preserve_flatmap_v3_schema_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    enhanced = tmp_path / "enhanced.parquet"
    filtered = tmp_path / "filtered.parquet"
    _write_three_neuron_source_parquet(source)
    original_metadata = _add_flatmap_v3_schema_metadata(source)

    export_enhanced_neuron_parquet(
        source,
        enhanced,
        table_state=_subset_table_state(),
    )
    export_filtered_project_neuron_parquet(
        source,
        filtered,
        table_state=_subset_table_state(),
    )

    enhanced_schema = pq.read_schema(enhanced)
    filtered_schema = pq.read_schema(filtered)
    for key, value in original_metadata.items():
        assert enhanced_schema.metadata[key] == value
        assert filtered_schema.metadata[key] == value
    assert enhanced_schema.field("x").metadata == {b"units": b"micrometer"}
    assert filtered_schema.field("x").metadata == {b"units": b"micrometer"}
    assert pd.read_parquet(filtered)["file_id"].tolist() == ["n3", "n3", "n1", "n1"]


def test_read_enhanced_parquet_metadata_accepts_canonical_parquet(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    _write_source_parquet(source)

    payload = read_enhanced_parquet_metadata(source)

    assert payload["has_project_metadata"] is False
    assert payload["enhanced_columns"] == []
    assert payload["table_state"]["entries"] == []


def test_read_enhanced_parquet_metadata_imports_legacy_cluster_id(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy-clusters.parquet"
    _write_source_parquet(source)
    frame = pd.read_parquet(source)
    frame["cluster_id"] = [4, 4, 9]
    frame.to_parquet(source, index=False)

    payload = read_enhanced_parquet_metadata(source)

    entries = {
        entry["file_id"]: entry["cluster_id"]
        for entry in payload["table_state"]["entries"]
    }
    assert entries == {"n1": 4, "n2": 9}


def test_save_project_bundle_writes_only_current_table_neurons(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "saved.swcv"
    _write_three_neuron_source_parquet(source)

    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=_subset_table_state(),
        layers=[],
        atlas_name="fake_atlas",
    )

    bundle = load_project_bundle(bundle_path)
    saved = pd.read_parquet(bundle.source_parquet_path)
    assert saved["file_id"].tolist() == ["n3", "n3", "n1", "n1"]
    assert "n2" not in set(saved["file_id"])
    assert saved.loc[saved["file_id"] == "n3", "neuron_label"].unique().tolist() == [
        "target"
    ]
    assert saved.loc[
        saved["file_id"] == "n1", "cluster_assignment"
    ].unique().tolist() == [2]


def test_load_project_bundle_accepts_version_one_manifest_and_table_state(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "legacy.swcv"
    _write_source_parquet(source)
    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=_table_state(),
        layers=[],
    )
    manifest_path = bundle_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["version"] = "1"
    manifest_path.write_text(json.dumps(manifest))

    bundle = load_project_bundle(bundle_path)

    assert bundle.manifest["version"] == "1"
    assert bundle.table_state["version"] == "1"
    assert bundle.table_state["entries"][0]["cluster_id"] == 3


def test_project_bundle_filters_v2_assignments_to_current_membership(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "saved.swcv"
    _write_three_neuron_source_parquet(source)
    state = _multi_assignment_table_state()
    state["entries"] = state["entries"][:2]

    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=state,
        layers=[],
    )

    bundle = load_project_bundle(bundle_path)
    registry = bundle.table_state["cluster_assignments"]
    assert bundle.manifest["version"] == "2"
    assert [item["name"] for item in registry["sets"]] == [
        "Soma Location 1",
        "Voxel Correlation 1",
    ]
    for assignment in registry["sets"]:
        assert set(assignment["assignments"]) <= {"n1", "n2"}
        assert set(assignment["input_file_ids"]) <= {"n1", "n2"}


def test_project_bundle_keeps_collision_resolved_assignment_column_stable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "saved.swcv"
    reexported = tmp_path / "reexported.parquet"
    _write_three_neuron_source_parquet(source)
    frame = pd.read_parquet(source)
    frame["cluster_soma_location_1"] = 99
    frame.to_parquet(source, index=False)

    state = _multi_assignment_table_state()
    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=state,
        layers=[],
    )
    bundle = load_project_bundle(bundle_path)
    first_set = bundle.table_state["cluster_assignments"]["sets"][0]
    assert first_set["column_name"] == "cluster_soma_location_1_2"

    export_enhanced_neuron_parquet(
        bundle.source_parquet_path,
        reexported,
        table_state=bundle.table_state,
    )
    loaded = pd.read_parquet(reexported)
    assert set(loaded["cluster_soma_location_1"]) == {99}
    assert "cluster_soma_location_1_2" in loaded.columns
    assert "cluster_soma_location_1_2_2" not in loaded.columns


def test_save_project_bundle_replaces_existing_enhanced_columns(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "saved.swcv"
    _write_three_neuron_source_parquet(source)
    source_df = pd.read_parquet(source)
    source_df["neuron_label"] = "old"
    source_df["neuron_group"] = "old_group"
    source_df["neuron_tags_json"] = '["old"]'
    source_df["neuron_notes"] = "old notes"
    source_df["cluster_assignment"] = 99
    source_df.to_parquet(source, index=False)

    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=_subset_table_state(),
        layers=[],
    )

    schema = pq.read_schema(bundle_path / "data" / "source_neurons.parquet")
    for column in ENHANCED_NEURON_COLUMNS:
        assert schema.names.count(column) == 1

    saved = pd.read_parquet(bundle_path / "data" / "source_neurons.parquet")
    assert set(saved["neuron_label"]) == {"source", "target"}
    assert set(saved["neuron_group"]) == {"CTX", "GPe"}
    assert set(saved["cluster_assignment"]) == {2, 7}


def test_save_project_bundle_manifest_records_filtered_source_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "saved.swcv"
    _write_three_neuron_source_parquet(source)

    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=_subset_table_state(),
        layers=[],
    )

    bundle = load_project_bundle(bundle_path)
    source_info = bundle.manifest["source_parquet"]
    assert source_info["path"] == "data/source_neurons.parquet"
    assert source_info["is_filtered_subset"] is True
    assert source_info["filter"] == {
        "type": "current_table_file_ids",
        "file_id_count": 2,
    }
    assert source_info["original_path"] == str(source)
    assert source_info["original_size_bytes"] == source.stat().st_size
    assert source_info["original_mtime_ns"] == source.stat().st_mtime_ns
    assert "sha256" in source_info


def test_project_bundle_references_external_flatmap_cache_without_copying_it(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "saved.swcv"
    cache_path = tmp_path / "shared-flatmap-cache"
    cache_path.mkdir()
    (cache_path / "flatmap-region-cache.json").write_text("{}\n")
    _write_three_neuron_source_parquet(source)

    reference = {
        "path": str(cache_path),
        "profile_id": "lookup-test__atlas-test__256x25",
    }
    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=_subset_table_state(),
        layers=[],
        flatmap_cache_reference=reference,
    )

    bundle = load_project_bundle(bundle_path)
    assert bundle.manifest["flatmap_cache"] == reference
    assert bundle.flatmap_cache_reference == reference
    assert not (bundle_path / "flatmap-region-cache.json").exists()


def test_save_project_bundle_rejects_empty_table(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "empty.swcv"
    _write_source_parquet(source)

    with pytest.raises(ValueError, match="at least one neuron"):
        save_project_bundle(
            bundle_path,
            source_parquet_path=source,
            table_state={"version": 1, "entries": []},
            layers=[],
        )

    assert not (bundle_path / "data" / "source_neurons.parquet").exists()


def test_save_and_load_project_bundle_preserves_mask_array_and_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "saved.swcv"
    _write_source_parquet(source)
    progress_events: list[tuple[str, int, int]] = []

    heatmap = _DummyLayer(
        name="alpha Heatmap",
        data=np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        metadata={
            "heatmap_source": True,
            "heatmap_kind": "selected_neurons",
            "file_ids": ["n1"],
            "source_path": str(source),
            "color": [1.0, 0.0, 0.0, 1.0],
            "atlas_name": "fake_atlas",
        },
        colormap=_DummyColormap(
            "alpha_red",
            [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]],
        ),
    )
    mask = _DummyLayer(
        name="Mask: alpha Heatmap",
        data=np.array([[[0, 1], [1, 0]]], dtype=np.uint8),
        metadata={
            "mask_query_source": True,
            "source_heatmap_layers": ["alpha Heatmap"],
            "query_excluded_file_ids": ["n1", "n2", "n1"],
            "lower_threshold": 0.5,
            "upper_threshold": 2.0,
            "merge_mode": "separate",
            "atlas_name": "fake_atlas",
        },
    )
    ignored = _DummyLayer(
        name="Neuron Lines",
        data=np.zeros((1, 2, 3), dtype=float),
        metadata={"file_ids": ["n1"]},
    )

    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=_table_state(),
        layers=[heatmap, mask, ignored],
        atlas_name="fake_atlas",
        progress_callback=lambda message, current, total: progress_events.append(
            (message, current, total)
        ),
    )

    assert progress_events[0] == ("Preparing project bundle...", 0, 6)
    assert progress_events[-1] == ("Done", 6, 6)
    assert any(
        "Saving layer 1/2: alpha Heatmap" in event[0] for event in progress_events
    )
    assert any(
        "Saving layer 2/2: Mask: alpha Heatmap" in event[0] for event in progress_events
    )

    bundle = load_project_bundle(bundle_path)
    assert bundle.manifest["format"] == PROJECT_BUNDLE_FORMAT
    assert bundle.source_parquet_path.exists()
    assert bundle.table_state["entries"][0]["file_id"] == "n1"
    assert len(bundle.layers) == 2
    restored_heatmap = next(
        layer for layer in bundle.layers if layer.metadata["layer_type"] == "image"
    )
    colormap = restored_heatmap.metadata["display"]["colormap"]
    assert colormap["name"] == "alpha_red"
    assert colormap["colors"] == [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]]
    restored_mask = next(
        layer for layer in bundle.layers if layer.metadata["layer_type"] == "labels"
    )
    np.testing.assert_array_equal(restored_mask.data, mask.data)
    source_sets = restored_mask.metadata["source_neuron_sets"]
    assert source_sets[0]["count"] == 2
    assert source_sets[0]["file_ids"] == ["n1", "n2"]
    assert source_sets[0]["derivation"]["lower_threshold"] == 0.5

    bundled_schema = pq.read_schema(bundle.source_parquet_path)
    assert "neuron_label" in bundled_schema.names


def test_project_colormap_payload_restores_image_colormap_kwargs() -> None:
    from napari_swc_viewer.widgets.neuron_viewer import NeuronViewerWidget

    kwargs: dict[str, object] = {}
    restored = NeuronViewerWidget._apply_project_colormap_kwargs(
        SimpleNamespace(),
        kwargs,
        {
            "colormap": {
                "type": "colormap",
                "name": "alpha_red",
                "colors": [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]],
                "controls": [0.0, 1.0],
                "interpolation": "linear",
                "nan_color": [0.0, 0.0, 0.0, 0.0],
            }
        },
    )

    assert restored is True
    assert kwargs["colormap"].name == "alpha_red"
    np.testing.assert_allclose(
        np.asarray(kwargs["colormap"].colors),
        [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]],
    )


def test_project_colormap_payload_restores_labels_colormap_kwargs() -> None:
    from napari_swc_viewer.widgets.neuron_viewer import NeuronViewerWidget

    kwargs: dict[str, object] = {}
    restored = NeuronViewerWidget._apply_project_colormap_kwargs(
        SimpleNamespace(_project_label_key=NeuronViewerWidget._project_label_key),
        kwargs,
        {
            "colormap": {
                "type": "direct_label_colormap",
                "color_dict": [
                    {
                        "label": {"type": "none", "value": None},
                        "color": [0.0, 0.0, 0.0, 0.0],
                    },
                    {
                        "label": {"type": "int", "value": 1},
                        "color": [0.2, 0.4, 0.6, 1.0],
                    },
                ],
            }
        },
    )

    assert restored is True
    np.testing.assert_allclose(
        kwargs["colormap"].color_dict[1],
        [0.2, 0.4, 0.6, 1.0],
    )
