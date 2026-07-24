from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from napari_swc_viewer.db import NeuronDatabase
from napari_swc_viewer.flatmap_parquet import (
    FLATMAP_INVALID_CODE_INVALID_DEPTH,
    FLATMAP_INVALID_CODE_INVALID_FLATMAP,
    FLATMAP_INVALID_CODE_OUT_OF_BOUNDS,
    FLATMAP_INVALID_CODE_VALID,
    FLATMAP_PARQUET_METADATA_KEY,
    FLATMAP_PARQUET_FORMAT_VERSION,
    FLATMAP_V3_AUGMENTED_COLUMNS,
    LEGACY_SINGLE_FLATMAP_PARQUET_FORMAT_VERSION,
    FlatmapParquetCancelledError,
    augment_neuron_parquet_with_flatmap,
    augment_neuron_parquet_with_flatmaps,
    read_flatmap_parquet_transform_info,
)
from napari_swc_viewer.flatmap_profiles import (
    FlatmapLookupCancelledError,
    build_flatmap_lookup_set,
    discover_flatmap_lookup_set,
)

nrrd = pytest.importorskip("nrrd")


def _write_nrrd(path: Path, data: np.ndarray) -> None:
    nrrd.write(str(path), data)


def _volumes(shape: tuple[int, int, int] = (4, 4, 4)) -> tuple[np.ndarray, np.ndarray]:
    grid = np.indices(shape, dtype=float)
    flatmap = np.stack((grid[0] + 0.25, grid[1] + 0.5), axis=-1)
    depth = grid[2] + 100.0
    return flatmap.astype(np.float32), depth.astype(np.float32)


def _write_source_parquet(path: Path) -> None:
    table = pa.Table.from_pydict(
        {
            "file_id": ["a.swc", "a.swc", "b.swc"],
            "node_id": [1, 2, 1],
            "type": [1, 3, 3],
            "x": [10.0, 10.0, 1000.0],
            "y": [20.0, 20.0, 0.0],
            "z": [10.0, 30.0, 30.0],
            "radius": [1.0, 0.5, 0.5],
            "parent_id": [-1, 1, -1],
            "region_id": [1, 1, 0],
            "region_name": ["Region", "Region", ""],
            "region_acronym": ["REG", "REG", ""],
            "subject": ["subject", "subject", "subject"],
            "neuron_id": ["a", "a", "b"],
            "custom_note": ["keep-a", "keep-b", "keep-c"],
        }
    )
    pq.write_table(table, path)


def test_read_flatmap_parquet_transform_info_detects_no_transform(tmp_path) -> None:
    source = tmp_path / "neurons.parquet"
    _write_source_parquet(source)

    info = read_flatmap_parquet_transform_info(source)

    assert info.has_flatmap is False
    assert info.has_depth is False
    assert info.has_full_transform is False
    assert info.present_transform_text == ""
    assert info.metadata is None
    assert info.uses_legacy_mirror_fallback is False


def test_read_flatmap_parquet_transform_info_detects_partial_transforms(tmp_path) -> None:
    source = tmp_path / "neurons.parquet"
    _write_source_parquet(source)
    table = pq.read_table(source)

    flatmap_only = tmp_path / "flatmap_only.parquet"
    pq.write_table(
        table.append_column("x_flat", pa.array([1.0, 2.0, 3.0], type=pa.float32()))
        .append_column("y_flat", pa.array([4.0, 5.0, 6.0], type=pa.float32())),
        flatmap_only,
    )
    depth_only = tmp_path / "depth_only.parquet"
    pq.write_table(
        table.append_column("depth_um", pa.array([7.0, 8.0, 9.0], type=pa.float32())),
        depth_only,
    )

    flatmap_info = read_flatmap_parquet_transform_info(flatmap_only)
    depth_info = read_flatmap_parquet_transform_info(depth_only)

    assert flatmap_info.has_flatmap is True
    assert flatmap_info.has_depth is False
    assert flatmap_info.present_transform_text == "flatmap"
    assert depth_info.has_flatmap is False
    assert depth_info.has_depth is True
    assert depth_info.present_transform_text == "depth"


def test_read_flatmap_parquet_transform_info_flags_version_one_mirror_fallback(
    tmp_path,
) -> None:
    source = tmp_path / "legacy_flatmap.parquet"
    _write_source_parquet(source)
    table = pq.read_table(source)
    table = table.append_column(
        "x_flat",
        pa.array([0.1, 0.2, 0.3], type=pa.float32()),
    ).append_column(
        "y_flat",
        pa.array([0.4, 0.5, 0.6], type=pa.float32()),
    ).append_column(
        "depth_um",
        pa.array([100.0, 200.0, 300.0], type=pa.float32()),
    )
    metadata = dict(table.schema.metadata or {})
    metadata[FLATMAP_PARQUET_METADATA_KEY] = json.dumps(
        {"version": 1, "mirror_fallback": True}
    ).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), source)

    info = read_flatmap_parquet_transform_info(source)

    assert info.has_full_transform is True
    assert info.uses_legacy_mirror_fallback is True


def test_augment_neuron_parquet_adds_flatmap_columns_with_mirror_fallback(
    tmp_path,
) -> None:
    source = tmp_path / "neurons.parquet"
    output = tmp_path / "neurons_flatmap.parquet"
    flatmap_path = tmp_path / "flatmap.nrrd"
    depth_path = tmp_path / "depth.nrrd"
    _write_source_parquet(source)
    flatmap, depth = _volumes()
    depth[1, 2, 3] = -1.0
    _write_nrrd(flatmap_path, flatmap)
    _write_nrrd(depth_path, depth)

    summary = augment_neuron_parquet_with_flatmap(
        source,
        output,
        flatmap_path,
        depth_path,
        mirror_midline=15.0,
        batch_size=2,
    )

    assert summary.rows == 3
    assert summary.direct_rows == 1
    assert summary.mirrored_depth_rows == 1
    assert summary.mirrored_rows == 0
    assert summary.unmapped_rows == 1

    table = pq.read_table(output)
    assert table.num_rows == 3
    assert "custom_note" in table.column_names
    assert table.column("custom_note").to_pylist() == ["keep-a", "keep-b", "keep-c"]

    out = table.to_pandas()
    assert out["flatmap_lookup_mode"].tolist() == [
        "direct",
        "mirrored_depth",
        "unmapped",
    ]
    assert out["flatmap_projection_valid"].tolist() == [True, True, False]
    assert out["flatmap_invalid_code"].tolist() == [
        FLATMAP_INVALID_CODE_VALID,
        FLATMAP_INVALID_CODE_VALID,
        FLATMAP_INVALID_CODE_OUT_OF_BOUNDS,
    ]
    assert out.loc[0, "x_flat"] == pytest.approx(1.25)
    assert out.loc[0, "y_flat"] == pytest.approx(2.5)
    assert out.loc[0, "depth_um"] == pytest.approx(101.0)
    assert out.loc[1, "depth_um"] == pytest.approx(100.0)
    assert np.isnan(out.loc[2, "x_flat"])
    assert np.isnan(out.loc[2, "depth_um"])

    metadata = table.schema.metadata or {}
    assert FLATMAP_PARQUET_METADATA_KEY in metadata
    payload = json.loads(metadata[FLATMAP_PARQUET_METADATA_KEY].decode("utf-8"))
    assert (
        payload["version"]
        == LEGACY_SINGLE_FLATMAP_PARQUET_FORMAT_VERSION
        == 2
    )
    assert payload["flatmap_nrrd"]["path"] == str(flatmap_path.resolve())
    assert payload["depth_nrrd"]["path"] == str(depth_path.resolve())
    assert payload["mirror_fallback"] is True
    assert payload["mirror_fallback_strategy"] == (
        "preserve_original_flatmap_then_mirror_depth_then_full_lookup"
    )
    assert payload["lookup_modes"] == [
        "direct",
        "mirrored_depth",
        "mirrored",
        "unmapped",
    ]
    assert payload["mirror_coord_axis"] == 2
    assert payload["mirror_midline"] == 15.0
    info = read_flatmap_parquet_transform_info(output)
    assert info.has_flatmap is True
    assert info.has_depth is True
    assert info.has_full_transform is True
    assert info.present_transform_text == "flatmap and depth"
    assert info.metadata == payload
    assert info.uses_legacy_mirror_fallback is False

    db = NeuronDatabase(output)
    try:
        rows = db.query("SELECT COUNT(*) AS n FROM neurons")
    finally:
        db.close()
    assert rows["n"].tolist() == [3]


def test_augment_neuron_parquet_preserves_bilateral_flatmap_coordinates(
    tmp_path,
) -> None:
    source = tmp_path / "bilateral_neurons.parquet"
    output = tmp_path / "bilateral_neurons_flatmap.parquet"
    flatmap_path = tmp_path / "flatmap_both_shaped.nrrd"
    depth_path = tmp_path / "depth.nrrd"
    pq.write_table(
        pa.Table.from_pydict(
            {
                "file_id": ["left.swc", "right.swc"],
                "node_id": [1, 1],
                "type": [1, 1],
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "z": [0.0, 30.0],
                "parent_id": [-1, -1],
            }
        ),
        source,
    )
    flatmap = np.full((1, 1, 4, 2), -1.0, dtype=np.float32)
    flatmap[0, 0, 0] = (0.1, 0.5)
    flatmap[0, 0, 3] = (1.9, 0.5)
    depth = np.full((1, 1, 4), -1.0, dtype=np.float32)
    depth[0, 0, 3] = 100.0
    _write_nrrd(flatmap_path, flatmap)
    _write_nrrd(depth_path, depth)

    summary = augment_neuron_parquet_with_flatmap(
        source,
        output,
        flatmap_path,
        depth_path,
        mirror_midline=15.0,
    )

    out = pq.read_table(output).to_pandas()
    assert out["x_flat"].tolist() == pytest.approx([0.1, 1.9])
    assert out["depth_um"].tolist() == pytest.approx([100.0, 100.0])
    assert out["flatmap_lookup_mode"].tolist() == [
        "mirrored_depth",
        "direct",
    ]
    assert summary.rows == 2
    assert summary.direct_rows == 1
    assert summary.mirrored_depth_rows == 1
    assert summary.mirrored_rows == 0
    assert summary.unmapped_rows == 0


def test_augment_neuron_parquet_file_ids_none_writes_all_rows(tmp_path) -> None:
    source = tmp_path / "neurons.parquet"
    output = tmp_path / "neurons_flatmap.parquet"
    flatmap_path = tmp_path / "flatmap.nrrd"
    depth_path = tmp_path / "depth.nrrd"
    _write_source_parquet(source)
    flatmap, depth = _volumes()
    _write_nrrd(flatmap_path, flatmap)
    _write_nrrd(depth_path, depth)

    summary = augment_neuron_parquet_with_flatmap(
        source,
        output,
        flatmap_path,
        depth_path,
        file_ids=None,
    )

    table = pq.read_table(output)
    assert summary.rows == 3
    assert table.num_rows == 3


def test_augment_neuron_parquet_filters_file_ids_before_projection(tmp_path) -> None:
    source = tmp_path / "neurons.parquet"
    output = tmp_path / "neurons_flatmap.parquet"
    flatmap_path = tmp_path / "flatmap.nrrd"
    depth_path = tmp_path / "depth.nrrd"
    _write_source_parquet(source)
    flatmap, depth = _volumes()
    _write_nrrd(flatmap_path, flatmap)
    _write_nrrd(depth_path, depth)

    summary = augment_neuron_parquet_with_flatmap(
        source,
        output,
        flatmap_path,
        depth_path,
        file_ids=["a.swc"],
        batch_size=1,
    )

    table = pq.read_table(output)
    out = table.to_pandas()
    assert summary.rows == 2
    assert table.num_rows == 2
    assert out["file_id"].tolist() == ["a.swc", "a.swc"]
    assert out["custom_note"].tolist() == ["keep-a", "keep-b"]
    assert {"x_flat", "y_flat", "depth_um"}.issubset(out.columns)

    metadata = table.schema.metadata or {}
    payload = json.loads(metadata[FLATMAP_PARQUET_METADATA_KEY].decode("utf-8"))
    assert payload["file_ids_filter_count"] == 1


def test_augment_neuron_parquet_rejects_empty_file_ids(tmp_path) -> None:
    with pytest.raises(ValueError, match="file_ids"):
        augment_neuron_parquet_with_flatmap(
            tmp_path / "neurons.parquet",
            tmp_path / "neurons_flatmap.parquet",
            tmp_path / "flatmap.nrrd",
            tmp_path / "depth.nrrd",
            file_ids=[],
        )


def test_augment_neuron_parquet_can_disable_mirror_fallback(tmp_path) -> None:
    source = tmp_path / "neurons.parquet"
    output = tmp_path / "neurons_flatmap.parquet"
    flatmap_path = tmp_path / "flatmap.nrrd"
    depth_path = tmp_path / "depth.nrrd"
    _write_source_parquet(source)
    flatmap, depth = _volumes()
    depth[1, 2, 3] = -1.0
    _write_nrrd(flatmap_path, flatmap)
    _write_nrrd(depth_path, depth)

    summary = augment_neuron_parquet_with_flatmap(
        source,
        output,
        flatmap_path,
        depth_path,
        mirror_fallback=False,
        mirror_midline=15.0,
    )

    out = pq.read_table(output).to_pandas()
    assert summary.direct_rows == 1
    assert summary.mirrored_depth_rows == 0
    assert summary.mirrored_rows == 0
    assert summary.unmapped_rows == 2
    assert out["flatmap_lookup_mode"].tolist() == ["direct", "unmapped", "unmapped"]
    assert out["flatmap_invalid_code"].tolist() == [
        FLATMAP_INVALID_CODE_VALID,
        FLATMAP_INVALID_CODE_INVALID_DEPTH,
        FLATMAP_INVALID_CODE_OUT_OF_BOUNDS,
    ]


def test_augment_neuron_parquet_replaces_existing_flatmap_columns(tmp_path) -> None:
    source = tmp_path / "neurons.parquet"
    output = tmp_path / "neurons_flatmap.parquet"
    flatmap_path = tmp_path / "flatmap.nrrd"
    depth_path = tmp_path / "depth.nrrd"
    _write_source_parquet(source)
    table = pq.read_table(source).append_column(
        "x_flat",
        pa.array([-99.0, -99.0, -99.0], type=pa.float32()),
    )
    pq.write_table(table, source)
    flatmap, depth = _volumes()
    _write_nrrd(flatmap_path, flatmap)
    _write_nrrd(depth_path, depth)

    augment_neuron_parquet_with_flatmap(source, output, flatmap_path, depth_path)

    out = pq.read_table(output).to_pandas()
    assert out["x_flat"].tolist()[:2] == pytest.approx([1.25, 1.25])


def _write_bilateral_lookup_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    shaped = np.full((1, 1, 4, 2), -1.0, dtype=np.float32)
    shaped[0, 0, 1] = (0.1, 0.5)
    shaped[0, 0, 3] = (1.9, 0.5)
    square = np.full((1, 1, 4, 2), -1.0, dtype=np.float32)
    square[0, 0, 0] = (10.1, 2.5)
    square[0, 0, 1] = (10.5, 2.5)
    square[0, 0, 3] = (11.9, 2.5)
    depth = np.full((1, 1, 4), -1.0, dtype=np.float32)
    depth[0, 0, 3] = 100.0
    _write_nrrd(path / "flatmap_both_shaped.nrrd", shaped)
    _write_nrrd(path / "flatmap_both_square.nrrd", square)
    _write_nrrd(path / "depth.nrrd", depth)


def _write_bilateral_source(path: Path) -> None:
    table = pa.Table.from_pydict(
        {
            "file_id": ["left.swc", "right.swc"],
            "node_id": [1, 1],
            "type": [1, 1],
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "z": [0.0, 30.0],
            "parent_id": [-1, -1],
            "custom_note": ["preserve-left", "preserve-right"],
        }
    ).replace_schema_metadata({b"custom.schema.key": b"preserved"})
    pq.write_table(table, path)


def test_v3_whole_parquet_adds_both_styles_and_shared_mirrored_depth(
    tmp_path,
) -> None:
    lookup_dir = tmp_path / "lookup"
    _write_bilateral_lookup_dir(lookup_dir)
    source = tmp_path / "neurons.parquet"
    output = tmp_path / "neurons_flatmap.parquet"
    _write_bilateral_source(source)
    lookup_set = discover_flatmap_lookup_set(
        lookup_dir,
        lookup_resolution_um=10.0,
    )

    summary = augment_neuron_parquet_with_flatmaps(
        source,
        output,
        lookup_set,
        batch_size=1,
    )

    table = pq.read_table(output)
    out = table.to_pandas()
    assert set(FLATMAP_V3_AUGMENTED_COLUMNS).issubset(table.column_names)
    assert out["custom_note"].tolist() == ["preserve-left", "preserve-right"]
    assert np.isnan(out.loc[0, "x_flat_shaped"])
    assert out.loc[1, "x_flat_shaped"] == pytest.approx(1.9)
    assert out["x_flat_square"].tolist() == pytest.approx([10.1, 11.9])
    assert out["depth_um"].tolist() == pytest.approx([100.0, 100.0])
    assert out["depth_lookup_mode"].tolist() == ["mirrored_depth", "direct"]
    assert out["flatmap_shaped_lookup_mode"].tolist() == ["unmapped", "direct"]
    assert out["flatmap_square_lookup_mode"].tolist() == ["direct", "direct"]
    assert out["flatmap_shaped_projection_valid"].tolist() == [False, True]
    assert out["flatmap_square_projection_valid"].tolist() == [True, True]
    assert out["flatmap_shaped_invalid_code"].tolist() == [
        FLATMAP_INVALID_CODE_INVALID_FLATMAP,
        FLATMAP_INVALID_CODE_VALID,
    ]
    assert summary.lookup_set_id == lookup_set.lookup_set_id
    assert summary.rows == 2
    assert summary.direct_rows == 1
    assert summary.mirrored_depth_rows == 1
    assert summary.shaped_valid_rows == 1
    assert summary.square_valid_rows == 2

    metadata = table.schema.metadata or {}
    assert metadata[b"custom.schema.key"] == b"preserved"
    payload = json.loads(metadata[FLATMAP_PARQUET_METADATA_KEY].decode("utf-8"))
    assert payload["version"] == FLATMAP_PARQUET_FORMAT_VERSION == 3
    assert payload["lookup_set_id"] == lookup_set.lookup_set_id
    assert payload["lookup_set"]["source_sha256"] == dict(
        lookup_set.source_sha256
    )
    assert payload["canonical_bounds"]["both_shaped"]["x"] == pytest.approx(
        [0.1, 1.9]
    )
    assert payload["column_mapping"]["both_square"]["x"] == "x_flat_square"
    assert payload["shared_depth_definition"]["policy"] == (
        "original_voxel_then_mirror_depth_voxel_if_invalid"
    )

    info = read_flatmap_parquet_transform_info(output)
    assert info.has_full_transform is True
    assert info.available_styles == ("both_shaped", "both_square")
    assert info.lookup_set_id == lookup_set.lookup_set_id
    assert info.lookup_set.lookup_set_id == lookup_set.lookup_set_id
    assert info.has_style("shaped") is True
    assert info.column_mapping("square")["x"] == "x_flat_square"
    assert info.grid_spec("both_shaped").x_bounds == pytest.approx((0.1, 1.9))


def test_lookup_set_id_is_stable_after_relocation(tmp_path) -> None:
    original = tmp_path / "original"
    relocated = tmp_path / "relocated"
    _write_bilateral_lookup_dir(original)
    relocated.mkdir()
    for source in original.glob("*.nrrd"):
        shutil.copy2(source, relocated / source.name)

    first = discover_flatmap_lookup_set(original, lookup_resolution_um=10.0)
    second = discover_flatmap_lookup_set(relocated, lookup_resolution_um=10.0)

    assert first.lookup_set_id == second.lookup_set_id
    assert first.shaped_grid.grid_spec_id == second.shaped_grid.grid_spec_id
    assert first.to_dict()["source_paths"] != second.to_dict()["source_paths"]


def test_lookup_set_requires_explicit_resolution_without_transform(tmp_path) -> None:
    lookup_dir = tmp_path / "lookup"
    _write_bilateral_lookup_dir(lookup_dir)

    with pytest.raises(ValueError, match="lookup_resolution_um"):
        discover_flatmap_lookup_set(lookup_dir)


def test_lookup_set_discovery_honors_cancellation(tmp_path) -> None:
    lookup_dir = tmp_path / "lookup"
    _write_bilateral_lookup_dir(lookup_dir)

    with pytest.raises(FlatmapLookupCancelledError, match="cancelled"):
        discover_flatmap_lookup_set(
            lookup_dir,
            lookup_resolution_um=10.0,
            cancel_callback=lambda: True,
        )


def test_lookup_set_rejects_spatial_shape_mismatch(tmp_path) -> None:
    lookup_dir = tmp_path / "lookup"
    _write_bilateral_lookup_dir(lookup_dir)
    wrong_square = np.zeros((1, 1, 5, 2), dtype=np.float32)
    _write_nrrd(lookup_dir / "flatmap_both_square.nrrd", wrong_square)

    with pytest.raises(ValueError, match="share the same 3D atlas grid"):
        build_flatmap_lookup_set(
            lookup_dir / "flatmap_both_shaped.nrrd",
            lookup_dir / "flatmap_both_square.nrrd",
            lookup_dir / "depth.nrrd",
            lookup_resolution_um=10.0,
        )


def test_v3_cancellation_preserves_existing_output_atomically(tmp_path) -> None:
    lookup_dir = tmp_path / "lookup"
    _write_bilateral_lookup_dir(lookup_dir)
    lookup_set = discover_flatmap_lookup_set(
        lookup_dir,
        lookup_resolution_um=10.0,
    )
    source = tmp_path / "neurons.parquet"
    output = tmp_path / "neurons_flatmap.parquet"
    _write_bilateral_source(source)
    pq.write_table(pa.Table.from_pydict({"sentinel": [42]}), output)
    callback_calls = 0

    def cancel_after_first_batch() -> bool:
        nonlocal callback_calls
        callback_calls += 1
        return callback_calls >= 2

    with pytest.raises(FlatmapParquetCancelledError, match="cancelled"):
        augment_neuron_parquet_with_flatmaps(
            source,
            output,
            lookup_set,
            batch_size=1,
            cancel_callback=cancel_after_first_batch,
        )

    assert pq.read_table(output).to_pydict() == {"sentinel": [42]}
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_v3_can_atomically_replace_source_parquet_in_place(tmp_path) -> None:
    lookup_dir = tmp_path / "lookup"
    _write_bilateral_lookup_dir(lookup_dir)
    lookup_set = discover_flatmap_lookup_set(
        lookup_dir,
        lookup_resolution_um=10.0,
    )
    source = tmp_path / "neurons.parquet"
    _write_bilateral_source(source)

    summary = augment_neuron_parquet_with_flatmaps(
        source,
        source,
        lookup_set,
        batch_size=1,
    )

    table = pq.read_table(source)
    assert summary.rows == table.num_rows == 2
    assert set(FLATMAP_V3_AUGMENTED_COLUMNS).issubset(table.column_names)
    assert table.column("custom_note").to_pylist() == [
        "preserve-left",
        "preserve-right",
    ]
    assert list(tmp_path.glob(f".{source.name}.*.tmp")) == []
