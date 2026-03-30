"""Tests for batch SWC-to-Parquet conversion and CLI reporting."""

from __future__ import annotations

import importlib.util
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import napari_swc_viewer.parquet as parquet_module
from napari_swc_viewer.db import NeuronDatabase
from napari_swc_viewer.parquet import (
    NEURON_SCHEMA,
    batch_convert_swc_to_parquet,
    swc_files_to_parquet,
)


class _InlineProcessPoolExecutor:
    """Synchronous stand-in for ``ProcessPoolExecutor`` used in tests."""

    def __init__(self, max_workers=None, initializer=None, initargs=()):
        self._initializer = initializer
        self._initargs = initargs
        self._initialized = False

    def __enter__(self):
        self._ensure_initialized()
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def _ensure_initialized(self) -> None:
        if not self._initialized and self._initializer is not None:
            self._initializer(*self._initargs)
        self._initialized = True

    def submit(self, fn, *args, **kwargs):
        self._ensure_initialized()
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - mirrors executor behavior
            future.set_exception(exc)
        return future


def _install_inline_process_pool(monkeypatch) -> None:
    """Route the parallel code path through a synchronous fake executor."""
    monkeypatch.setattr(
        "napari_swc_viewer.parquet.ProcessPoolExecutor",
        _InlineProcessPoolExecutor,
    )


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


def _install_fake_annotation(monkeypatch, seen_coords: list[np.ndarray] | None = None) -> None:
    """Replace atlas setup and region lookup with a tiny deterministic fixture."""

    def fake_setup_allen_sdk(resolution, cache_dir):
        assert resolution == 25
        assert cache_dir is None
        return None, np.zeros((8, 8, 8), dtype=np.int32), object()

    def fake_build_region_lookup(structure_tree):
        assert structure_tree is not None
        return {
            0: {"name": "", "acronym": ""},
            5: {"name": "Left Region", "acronym": "LR"},
            11: {"name": "Right Region", "acronym": "RR"},
        }

    def fake_get_region_ids_vectorized(coords, annotation_volume, resolution):
        assert annotation_volume.shape == (8, 8, 8)
        assert resolution == 25
        if seen_coords is not None:
            seen_coords.append(coords.copy())
        return np.where(coords[:, 2] > 50.0, 11, 5)

    monkeypatch.setattr("napari_swc_viewer.parquet.setup_allen_sdk", fake_setup_allen_sdk)
    monkeypatch.setattr("napari_swc_viewer.parquet.build_region_lookup", fake_build_region_lookup)
    monkeypatch.setattr(
        "napari_swc_viewer.parquet.get_region_ids_vectorized",
        fake_get_region_ids_vectorized,
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
    _install_fake_annotation(monkeypatch, seen_coords)

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


def test_parallel_raw_output_matches_serial_output(tmp_path, monkeypatch):
    """The chunked path should preserve the same raw aligned rows and summary."""
    _install_inline_process_pool(monkeypatch)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_two_node_swc(input_dir / "left.swc", soma_z=10.0, child_z=20.0)
    _write_two_node_swc(input_dir / "right.swc", soma_z=90.0, child_z=80.0)
    _write_two_node_swc(input_dir / "midline.swc", soma_z=50.0, child_z=60.0)

    serial_output = tmp_path / "serial.parquet"
    parallel_output = tmp_path / "parallel.parquet"

    serial = batch_convert_swc_to_parquet(
        input_dir,
        serial_output,
        hemisphere="right",
        midline=50.0,
        batch_size=2,
        n_workers=1,
    )
    parallel = batch_convert_swc_to_parquet(
        input_dir,
        parallel_output,
        hemisphere="right",
        midline=50.0,
        batch_size=2,
        n_workers=2,
    )

    assert parallel.discovered_files == serial.discovered_files == 3
    assert parallel.processed_files == serial.processed_files == 3
    assert parallel.failed_files == serial.failed_files == 0
    assert parallel.flipped_files == serial.flipped_files == 1
    assert parallel.already_target_files == serial.already_target_files == 1
    assert parallel.midline_files == serial.midline_files == 1
    assert parallel.rows_written == serial.rows_written == 6

    serial_rows = _read_parquet_rows(serial_output)
    parallel_rows = _read_parquet_rows(parallel_output)
    assert parallel_rows.equals(serial_rows)


def test_parallel_annotated_output_matches_serial_and_cleans_temp_dir(tmp_path, monkeypatch):
    """Annotated chunked conversion should match serial output and clean scratch data."""
    _install_inline_process_pool(monkeypatch)
    _install_fake_annotation(monkeypatch)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_two_node_swc(input_dir / "left.swc", soma_z=10.0, child_z=20.0)
    _write_two_node_swc(input_dir / "right.swc", soma_z=90.0, child_z=80.0)
    _write_two_node_swc(input_dir / "midline.swc", soma_z=50.0, child_z=60.0)

    temp_root = tmp_path / "scratch"
    temp_root.mkdir()
    serial_output = tmp_path / "serial_annotated.parquet"
    parallel_output = tmp_path / "parallel_annotated.parquet"

    serial = batch_convert_swc_to_parquet(
        input_dir,
        serial_output,
        hemisphere="right",
        midline=50.0,
        annotate_regions=True,
        resolution=25,
        batch_size=2,
        n_workers=1,
    )
    parallel = batch_convert_swc_to_parquet(
        input_dir,
        parallel_output,
        hemisphere="right",
        midline=50.0,
        annotate_regions=True,
        resolution=25,
        batch_size=2,
        n_workers=2,
        temp_dir=temp_root,
    )

    assert parallel.processed_files == serial.processed_files == 3
    assert parallel.failed_files == serial.failed_files == 0
    assert parallel.rows_written == serial.rows_written == 6
    assert _read_parquet_rows(parallel_output).equals(_read_parquet_rows(serial_output))
    assert parquet_module._WORKER_ANNOTATION_VOLUME is None
    assert list(temp_root.iterdir()) == []


def test_parallel_progress_reports_completed_chunks_and_finalization(tmp_path, monkeypatch):
    """Parallel progress messages should reflect completed work, not chunk IDs."""
    _install_inline_process_pool(monkeypatch)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for index, soma_z in enumerate((10.0, 90.0, 10.0, 90.0, 10.0), start=1):
        _write_two_node_swc(input_dir / f"cell_{index}.swc", soma_z=soma_z, child_z=soma_z + 10.0)

    events: list[tuple[str, int, int]] = []
    output_path = tmp_path / "parallel_progress.parquet"

    summary = batch_convert_swc_to_parquet(
        input_dir,
        output_path,
        hemisphere="right",
        midline=50.0,
        batch_size=2,
        n_workers=2,
        progress_callback=lambda message, current, total: events.append((message, current, total)),
    )

    assert summary.processed_files == 5
    assert any("Processed 2/5 files (1/3 chunks)..." == message for message, _, _ in events)
    assert any("Processed 4/5 files (2/3 chunks)..." == message for message, _, _ in events)
    assert any("Processed 5/5 files (3/3 chunks)..." == message for message, _, _ in events)
    assert any(message.startswith("Finalizing Parquet (1/3 shards)...") for message, _, _ in events)
    assert any(message.startswith("Finalizing Parquet (3/3 shards)...") for message, _, _ in events)


def test_parallel_partial_success_still_writes_output_and_cleans_temp_dir(tmp_path, monkeypatch):
    """A chunked run should keep successful rows even when some files are skipped."""
    _install_inline_process_pool(monkeypatch)

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

    temp_root = tmp_path / "scratch"
    temp_root.mkdir()
    output_path = tmp_path / "parallel_output.parquet"

    summary = batch_convert_swc_to_parquet(
        input_dir,
        output_path,
        hemisphere="right",
        midline=50.0,
        batch_size=1,
        n_workers=2,
        temp_dir=temp_root,
    )

    assert summary.processed_files == 1
    assert summary.failed_files == 2
    rows = _read_parquet_rows(output_path)
    assert rows["file_id"].unique().tolist() == ["good.swc"]
    assert list(temp_root.iterdir()) == []


def test_swc_files_to_parquet_delegates_to_batch_converter(monkeypatch, tmp_path):
    """The legacy wrapper should now forward to the unified batch helper."""
    calls: dict[str, object] = {}

    def fake_batch_convert(
        input_path,
        output_path,
        *,
        recursive,
        annotate_regions,
        resolution,
        cache_dir,
        batch_size,
        n_workers,
        **_kwargs,
    ):
        calls["input_path"] = input_path
        calls["output_path"] = output_path
        calls["recursive"] = recursive
        calls["annotate_regions"] = annotate_regions
        calls["resolution"] = resolution
        calls["cache_dir"] = cache_dir
        calls["batch_size"] = batch_size
        calls["n_workers"] = n_workers
        return type("Summary", (), {"processed_files": 7})()

    monkeypatch.setattr("napari_swc_viewer.parquet.batch_convert_swc_to_parquet", fake_batch_convert)

    processed = swc_files_to_parquet(
        tmp_path / "input",
        tmp_path / "output.parquet",
        resolution=50,
        cache_dir=tmp_path / "cache",
        recursive=False,
        n_workers=4,
        batch_size=12,
    )

    assert processed == 7
    assert calls["recursive"] is False
    assert calls["annotate_regions"] is True
    assert calls["resolution"] == 50
    assert calls["cache_dir"] == tmp_path / "cache"
    assert calls["batch_size"] == 12
    assert calls["n_workers"] == 4


def test_cli_parses_worker_and_temp_dir_options(tmp_path):
    """The CLI should expose the new worker and scratch-directory controls."""
    script = _load_script_module()

    parsed_default = script.parse_args(["input.swc", "out.parquet"])
    parsed_auto = script.parse_args(["input.swc", "out.parquet", "--workers", "auto"])
    parsed_explicit = script.parse_args(
        [
            "input.swc",
            "out.parquet",
            "--workers",
            "4",
            "--temp-dir",
            str(tmp_path / "scratch"),
        ]
    )

    assert parsed_default.batch_size == 25
    assert parsed_default.workers is None
    assert parsed_auto.workers is None
    assert parsed_explicit.workers == 4
    assert parsed_explicit.temp_dir == tmp_path / "scratch"
