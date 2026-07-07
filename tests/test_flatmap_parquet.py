from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from napari_swc_viewer.db import NeuronDatabase
from napari_swc_viewer.flatmap_parquet import (
    FLATMAP_INVALID_CODE_INVALID_DEPTH,
    FLATMAP_INVALID_CODE_OUT_OF_BOUNDS,
    FLATMAP_INVALID_CODE_VALID,
    FLATMAP_PARQUET_METADATA_KEY,
    augment_neuron_parquet_with_flatmap,
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
    assert summary.mirrored_rows == 1
    assert summary.unmapped_rows == 1

    table = pq.read_table(output)
    assert table.num_rows == 3
    assert "custom_note" in table.column_names
    assert table.column("custom_note").to_pylist() == ["keep-a", "keep-b", "keep-c"]

    out = table.to_pandas()
    assert out["flatmap_lookup_mode"].tolist() == ["direct", "mirrored", "unmapped"]
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
    assert payload["flatmap_nrrd"]["path"] == str(flatmap_path.resolve())
    assert payload["depth_nrrd"]["path"] == str(depth_path.resolve())
    assert payload["mirror_fallback"] is True
    assert payload["mirror_coord_axis"] == 2
    assert payload["mirror_midline"] == 15.0

    db = NeuronDatabase(output)
    try:
        rows = db.query("SELECT COUNT(*) AS n FROM neurons")
    finally:
        db.close()
    assert rows["n"].tolist() == [3]


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
