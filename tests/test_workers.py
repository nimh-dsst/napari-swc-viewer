"""Tests for background workers."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

from napari_swc_viewer.point_import import PointParquetAppendSummary
from napari_swc_viewer.parquet import BatchParquetConversionSummary


class _BoundSignal:
    """Minimal Qt-like signal implementation for worker tests."""

    def __init__(self) -> None:
        self._callbacks: list = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, *args) -> None:
        for callback in list(self._callbacks):
            callback(*args)


class _Signal:
    """Descriptor that mimics ``qtpy.QtCore.Signal`` enough for tests."""

    def __init__(self, *_args, **_kwargs) -> None:
        self._storage_name = ""

    def __set_name__(self, _owner, name: str) -> None:
        self._storage_name = f"__signal_{name}"

    def __get__(self, instance, _owner):
        if instance is None:
            return self
        if self._storage_name not in instance.__dict__:
            instance.__dict__[self._storage_name] = _BoundSignal()
        return instance.__dict__[self._storage_name]


class _QObject:
    """Minimal QObject stand-in."""

    def __init__(self, *_args, **_kwargs) -> None:
        return None


def _import_workers_module():
    """Import the workers module with a stubbed QtCore dependency."""
    qtpy_module = types.ModuleType("qtpy")
    qtcore_module = types.ModuleType("qtpy.QtCore")
    qtcore_module.QObject = _QObject
    qtcore_module.Signal = _Signal
    qtpy_module.QtCore = qtcore_module

    sys.modules["qtpy"] = qtpy_module
    sys.modules["qtpy.QtCore"] = qtcore_module
    sys.modules.pop("napari_swc_viewer.workers", None)

    return importlib.import_module("napari_swc_viewer.workers")


def test_convert_worker_uses_batch_conversion_with_alignment(monkeypatch, tmp_path):
    """ConvertWorker should delegate to the batch helper with UI-selected options."""
    workers = _import_workers_module()
    ConvertWorker = workers.ConvertWorker
    calls: dict[str, object] = {}

    def fake_batch_convert_swc_to_parquet(
        input_path,
        output_path,
        *,
        recursive,
        hemisphere,
        atlas_name,
        coord_axis,
        annotate_regions,
        resolution,
        progress_callback,
        **_kwargs,
    ):
        calls["input_path"] = input_path
        calls["output_path"] = output_path
        calls["recursive"] = recursive
        calls["hemisphere"] = hemisphere
        calls["atlas_name"] = atlas_name
        calls["coord_axis"] = coord_axis
        calls["annotate_regions"] = annotate_regions
        calls["resolution"] = resolution
        progress_callback("Processing a.swc...", 0, 2)
        return BatchParquetConversionSummary(
            discovered_files=2,
            processed_files=2,
            flipped_files=1,
            rows_written=4,
        )

    monkeypatch.setattr(
        "napari_swc_viewer.parquet.batch_convert_swc_to_parquet",
        fake_batch_convert_swc_to_parquet,
    )

    output_path = tmp_path / "neurons.parquet"
    worker = ConvertWorker(
        ["a.swc", "b.swc"],
        str(output_path),
        resolution=25,
        hemisphere="right",
        atlas_name="allen_mouse_10um",
    )

    progress_events: list[tuple[str, int, int]] = []
    finished: list[tuple[str, BatchParquetConversionSummary]] = []
    errors: list[str] = []
    worker.progress.connect(
        lambda message, current, total: progress_events.append((message, current, total))
    )
    worker.finished.connect(lambda path, summary: finished.append((path, summary)))
    worker.error.connect(lambda message: errors.append(message))

    worker.run()

    assert [Path(p).name for p in calls["input_path"]] == ["a.swc", "b.swc"]
    assert Path(calls["output_path"]) == output_path
    assert calls["recursive"] is False
    assert calls["hemisphere"] == "right"
    assert calls["atlas_name"] == "allen_mouse_10um"
    assert calls["coord_axis"] == 2
    assert calls["annotate_regions"] is True
    assert calls["resolution"] == 25
    assert progress_events[0] == ("Preparing SWC-to-Parquet conversion...", 0, 2)
    assert progress_events[-1] == ("Finalizing Parquet...", 2, 2)
    assert not errors
    assert finished[0][0] == str(output_path)
    assert finished[0][1].flipped_files == 1


def test_append_point_file_worker_routes_csv_to_append_helper(monkeypatch, tmp_path):
    """AppendPointFileWorker should route CSV input to the CSV append helper."""
    workers = _import_workers_module()
    AppendPointFileWorker = workers.AppendPointFileWorker
    calls: dict[str, object] = {}

    def fake_append_point_csv_to_parquet(
        csv_path,
        mapping_path,
        parquet_path,
        output_path,
    ):
        calls["csv_path"] = csv_path
        calls["mapping_path"] = mapping_path
        calls["parquet_path"] = parquet_path
        calls["output_path"] = output_path
        return PointParquetAppendSummary(appended_rows=2, total_rows=5)

    monkeypatch.setattr(
        "napari_swc_viewer.point_import.append_point_csv_to_parquet",
        fake_append_point_csv_to_parquet,
    )

    worker = AppendPointFileWorker(
        str(tmp_path / "points.csv"),
        str(tmp_path / "mapping.json"),
        str(tmp_path / "points.parquet"),
        str(tmp_path / "points_out.parquet"),
    )
    progress_events: list[tuple[str, int, int]] = []
    finished: list[tuple[str, PointParquetAppendSummary]] = []
    errors: list[str] = []
    worker.progress.connect(
        lambda message, current, total: progress_events.append((message, current, total))
    )
    worker.finished.connect(lambda path, summary: finished.append((path, summary)))
    worker.error.connect(lambda message: errors.append(message))

    worker.run()

    assert Path(calls["csv_path"]).name == "points.csv"
    assert Path(calls["mapping_path"]).name == "mapping.json"
    assert Path(calls["parquet_path"]).name == "points.parquet"
    assert Path(calls["output_path"]).name == "points_out.parquet"
    assert progress_events[0] == ("Validating point CSV and target Parquet...", 1, 3)
    assert progress_events[-1] == ("Done", 3, 3)
    assert not errors
    assert finished[0][0].endswith("points_out.parquet")
    assert finished[0][1].appended_rows == 2
    assert finished[0][1].total_rows == 5


def test_append_point_file_worker_routes_parquet_to_append_helper(monkeypatch, tmp_path):
    """AppendPointFileWorker should route Parquet input to the Parquet append helper."""
    workers = _import_workers_module()
    AppendPointFileWorker = workers.AppendPointFileWorker
    calls: dict[str, object] = {}

    def fake_append_point_parquet_to_parquet(
        input_parquet_path,
        parquet_path,
        output_path,
    ):
        calls["input_parquet_path"] = input_parquet_path
        calls["parquet_path"] = parquet_path
        calls["output_path"] = output_path
        return PointParquetAppendSummary(appended_rows=3, total_rows=8)

    monkeypatch.setattr(
        "napari_swc_viewer.point_import.append_point_parquet_to_parquet",
        fake_append_point_parquet_to_parquet,
    )

    worker = AppendPointFileWorker(
        str(tmp_path / "points_in.parquet"),
        None,
        str(tmp_path / "points.parquet"),
        str(tmp_path / "points_out.parquet"),
    )
    progress_events: list[tuple[str, int, int]] = []
    finished: list[tuple[str, PointParquetAppendSummary]] = []
    errors: list[str] = []
    worker.progress.connect(
        lambda message, current, total: progress_events.append((message, current, total))
    )
    worker.finished.connect(lambda path, summary: finished.append((path, summary)))
    worker.error.connect(lambda message: errors.append(message))

    worker.run()

    assert Path(calls["input_parquet_path"]).name == "points_in.parquet"
    assert Path(calls["parquet_path"]).name == "points.parquet"
    assert Path(calls["output_path"]).name == "points_out.parquet"
    assert progress_events[0] == ("Validating point Parquet schemas...", 1, 3)
    assert not errors
    assert finished[0][1].appended_rows == 3
    assert finished[0][1].total_rows == 8


def test_append_point_file_worker_emits_error(monkeypatch, tmp_path):
    """AppendPointFileWorker should surface append failures through error."""
    workers = _import_workers_module()
    AppendPointFileWorker = workers.AppendPointFileWorker

    def fake_append_point_csv_to_parquet(
        _csv_path,
        _mapping_path,
        _parquet_path,
        _output_path,
    ):
        raise ValueError("schema mismatch")

    monkeypatch.setattr(
        "napari_swc_viewer.point_import.append_point_csv_to_parquet",
        fake_append_point_csv_to_parquet,
    )

    worker = AppendPointFileWorker(
        str(tmp_path / "points.csv"),
        str(tmp_path / "mapping.json"),
        str(tmp_path / "points.parquet"),
        str(tmp_path / "points_out.parquet"),
    )
    finished: list[tuple[str, PointParquetAppendSummary]] = []
    errors: list[str] = []
    worker.finished.connect(lambda path, summary: finished.append((path, summary)))
    worker.error.connect(lambda message: errors.append(message))

    worker.run()

    assert not finished
    assert errors == ["schema mismatch"]


def test_convert_point_csv_worker_delegates_to_batch_helper(monkeypatch, tmp_path):
    """ConvertPointCSVWorker should call the batch point CSV helper."""
    workers = _import_workers_module()
    ConvertPointCSVWorker = workers.ConvertPointCSVWorker
    calls: dict[str, object] = {}

    def fake_convert_point_csv_files_to_parquet(
        csv_paths,
        output_path,
        mapping_path,
        progress_callback,
    ):
        calls["csv_paths"] = csv_paths
        calls["output_path"] = output_path
        calls["mapping_path"] = mapping_path
        progress_callback("Processing point CSV 1/2: one.csv", 0, 2)
        return types.SimpleNamespace(
            discovered_files=2,
            processed_files=2,
            rows_written=4,
        )

    monkeypatch.setattr(
        "napari_swc_viewer.point_import.convert_point_csv_files_to_parquet",
        fake_convert_point_csv_files_to_parquet,
    )

    worker = ConvertPointCSVWorker(
        [str(tmp_path / "one.csv"), str(tmp_path / "two.csv")],
        str(tmp_path / "points.parquet"),
        str(tmp_path / "mapping.json"),
    )
    progress_events: list[tuple[str, int, int]] = []
    finished: list[tuple[str, object]] = []
    errors: list[str] = []
    worker.progress.connect(
        lambda message, current, total: progress_events.append((message, current, total))
    )
    worker.finished.connect(lambda path, summary: finished.append((path, summary)))
    worker.error.connect(lambda message: errors.append(message))

    worker.run()

    assert [Path(path).name for path in calls["csv_paths"]] == ["one.csv", "two.csv"]
    assert Path(calls["output_path"]).name == "points.parquet"
    assert Path(calls["mapping_path"]).name == "mapping.json"
    assert progress_events[0] == ("Processing point CSV 1/2: one.csv", 0, 2)
    assert not errors
    assert finished[0][0].endswith("points.parquet")
    assert finished[0][1].processed_files == 2
    assert finished[0][1].rows_written == 4


def test_convert_point_csv_worker_emits_error(monkeypatch, tmp_path):
    """ConvertPointCSVWorker should surface point conversion failures."""
    workers = _import_workers_module()
    ConvertPointCSVWorker = workers.ConvertPointCSVWorker

    def fake_convert_point_csv_files_to_parquet(
        _csv_paths,
        _output_path,
        _mapping_path,
        progress_callback=None,
    ):
        raise ValueError("bad headers")

    monkeypatch.setattr(
        "napari_swc_viewer.point_import.convert_point_csv_files_to_parquet",
        fake_convert_point_csv_files_to_parquet,
    )

    worker = ConvertPointCSVWorker(
        [str(tmp_path / "one.csv")],
        str(tmp_path / "points.parquet"),
        None,
    )
    finished: list[tuple[str, object]] = []
    errors: list[str] = []
    worker.finished.connect(lambda path, summary: finished.append((path, summary)))
    worker.error.connect(lambda message: errors.append(message))

    worker.run()

    assert not finished
    assert errors == ["bad headers"]
