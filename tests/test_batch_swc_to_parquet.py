"""Tests for batch SWC-to-Parquet conversion and CLI reporting."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from napari_swc_viewer.db import NeuronDatabase
from napari_swc_viewer.parquet import NEURON_SCHEMA, batch_convert_swc_to_parquet


def _load_script_module():
    """Load the batch conversion CLI script as a module."""
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "convert_swc_to_parquet.py"
    spec = importlib.util.spec_from_file_location("convert_swc_to_parquet_script", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_swc_file(path: Path, nodes: list[tuple[int, int, float, float, float, float, int]]) -> None:
    """Write a simple SWC file from node tuples."""
    lines = [
        "# sample SWC",
        "# id type x y z radius parent",
    ]
    lines.extend(
        f"{node_id} {node_type} {x:.1f} {y:.1f} {z:.1f} {radius:.1f} {parent_id}"
        for node_id, node_type, x, y, z, radius, parent_id in nodes
    )
    path.write_text("\n".join(lines) + "\n")


def _write_two_node_swc(path: Path, soma_z: float, child_z: float) -> None:
    """Write a minimal SWC with one soma and one child node."""
    _write_swc_file(
        path,
        [
            (1, 1, 10.0, 20.0, soma_z, 5.0, -1),
            (2, 3, 11.0, 21.0, child_z, 1.0, 1),
        ],
    )


def _read_parquet_rows(path: Path):
    """Read parquet rows into a sorted pandas DataFrame."""
    return (
        pq.read_table(path)
        .to_pandas()
        .sort_values(["file_id", "node_id"])
        .reset_index(drop=True)
    )


def test_batch_convert_fast_path_writes_plugin_compatible_parquet(tmp_path):
    """Raw mode should write the existing neuron schema with blank region columns."""
    input_dir = tmp_path / "input"
    nested_dir = input_dir / "nested"
    nested_dir.mkdir(parents=True)

    _write_two_node_swc(input_dir / "1001_2002_test.swc", soma_z=10.0, child_z=20.0)
    _write_two_node_swc(nested_dir / "simple.swc", soma_z=30.0, child_z=40.0)

    output_path = tmp_path / "neurons.parquet"
    summary = batch_convert_swc_to_parquet(input_dir, output_path)

    assert summary.discovered_files == 2
    assert summary.processed_files == 2
    assert summary.failed_files == 0
    assert summary.rows_written == 4

    table = pq.read_table(output_path)
    assert table.schema == NEURON_SCHEMA

    rows = _read_parquet_rows(output_path)
    bil_rows = rows[rows["file_id"] == "1001_2002_test.swc"]
    simple_rows = rows[rows["file_id"] == "simple.swc"]

    assert bil_rows["subject"].unique().tolist() == ["2002"]
    assert bil_rows["neuron_id"].unique().tolist() == ["1001"]
    assert simple_rows["subject"].unique().tolist() == ["simple"]
    assert simple_rows["neuron_id"].unique().tolist() == ["simple"]
    assert set(rows["region_id"]) == {0}
    assert set(rows["region_name"]) == {""}
    assert set(rows["region_acronym"]) == {""}

    with NeuronDatabase(output_path) as db:
        somas = db.get_soma_locations()
    assert somas["file_id"].tolist() == ["1001_2002_test.swc", "simple.swc"]


def test_batch_convert_accepts_explicit_swc_file_lists(tmp_path):
    """The batch helper should support the explicit file list flow used by the UI."""
    first = tmp_path / "first.swc"
    second = tmp_path / "second.swc"
    _write_two_node_swc(first, soma_z=10.0, child_z=20.0)
    _write_two_node_swc(second, soma_z=30.0, child_z=40.0)

    output_path = tmp_path / "selected_files.parquet"
    summary = batch_convert_swc_to_parquet([second, first], output_path)

    assert summary.discovered_files == 2
    assert summary.processed_files == 2

    rows = _read_parquet_rows(output_path)
    assert rows["file_id"].unique().tolist() == ["first.swc", "second.swc"]


def test_batch_convert_aligns_only_files_not_already_in_target_hemisphere(tmp_path):
    """Alignment mode should flip only mismatched files and count midline separately."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    _write_two_node_swc(input_dir / "left.swc", soma_z=10.0, child_z=20.0)
    _write_two_node_swc(input_dir / "right.swc", soma_z=90.0, child_z=80.0)
    _write_two_node_swc(input_dir / "midline.swc", soma_z=50.0, child_z=60.0)

    output_path = tmp_path / "aligned.parquet"
    summary = batch_convert_swc_to_parquet(
        input_dir,
        output_path,
        hemisphere="right",
        midline=50.0,
    )

    assert summary.processed_files == 3
    assert summary.flipped_files == 1
    assert summary.already_target_files == 1
    assert summary.midline_files == 1
    assert summary.failed_files == 0

    rows = _read_parquet_rows(output_path)
    left_rows = rows[rows["file_id"] == "left.swc"]
    right_rows = rows[rows["file_id"] == "right.swc"]
    midline_rows = rows[rows["file_id"] == "midline.swc"]

    assert left_rows["z"].tolist() == [90.0, 80.0]
    assert right_rows["z"].tolist() == [90.0, 80.0]
    assert midline_rows["z"].tolist() == [50.0, 60.0]


def test_cli_reports_skipped_files_but_succeeds_when_some_files_convert(tmp_path, capsys):
    """The CLI should keep going on bad inputs and report the failures."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    _write_two_node_swc(input_dir / "good.swc", soma_z=10.0, child_z=20.0)
    _write_swc_file(
        input_dir / "no_soma.swc",
        [
            (1, 3, 10.0, 20.0, 10.0, 5.0, -1),
            (2, 3, 11.0, 21.0, 20.0, 1.0, 1),
        ],
    )
    (input_dir / "malformed.swc").write_text("not a valid swc\n")

    output_path = tmp_path / "output.parquet"
    script = _load_script_module()

    exit_code = script.main(
        [
            str(input_dir),
            str(output_path),
            "--hemisphere",
            "right",
            "--midline",
            "50.0",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Discovered files: 3" in captured.out
    assert "Processed files: 1" in captured.out
    assert "Failed/skipped: 2" in captured.out
    assert "Skipped malformed.swc" in captured.err
    assert "Skipped no_soma.swc" in captured.err

    rows = _read_parquet_rows(output_path)
    assert rows["file_id"].unique().tolist() == ["good.swc"]
    assert rows["z"].tolist() == [90.0, 80.0]


def test_annotated_mode_uses_aligned_coordinates_before_region_annotation(
    tmp_path,
    monkeypatch,
):
    """Region IDs should be computed from the post-alignment coordinates."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_two_node_swc(input_dir / "left.swc", soma_z=10.0, child_z=20.0)

    seen_coords = []

    def fake_setup_allen_sdk(resolution, cache_dir):
        assert resolution == 25
        assert cache_dir is None
        return None, np.zeros((2, 2, 2), dtype=np.int32), object()

    def fake_build_region_lookup(structure_tree):
        assert structure_tree is not None
        return {
            11: {"name": "Right Region", "acronym": "RR"},
            5: {"name": "Left Region", "acronym": "LR"},
        }

    def fake_get_region_ids_vectorized(coords, annotation_volume, resolution):
        assert annotation_volume.shape == (2, 2, 2)
        assert resolution == 25
        seen_coords.append(coords.copy())
        return np.where(coords[:, 2] > 50.0, 11, 5)

    monkeypatch.setattr("napari_swc_viewer.parquet.setup_allen_sdk", fake_setup_allen_sdk)
    monkeypatch.setattr("napari_swc_viewer.parquet.build_region_lookup", fake_build_region_lookup)
    monkeypatch.setattr(
        "napari_swc_viewer.parquet.get_region_ids_vectorized",
        fake_get_region_ids_vectorized,
    )

    output_path = tmp_path / "annotated.parquet"
    summary = batch_convert_swc_to_parquet(
        input_dir,
        output_path,
        hemisphere="right",
        midline=50.0,
        annotate_regions=True,
        resolution=25,
    )

    assert summary.processed_files == 1
    assert summary.flipped_files == 1
    assert len(seen_coords) == 1
    assert np.all(seen_coords[0][:, 2] > 50.0)

    rows = _read_parquet_rows(output_path)
    assert rows["z"].tolist() == [90.0, 80.0]
    assert set(rows["region_id"]) == {11}
    assert set(rows["region_name"]) == {"Right Region"}
    assert set(rows["region_acronym"]) == {"RR"}
