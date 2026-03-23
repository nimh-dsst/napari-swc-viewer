"""Tests for background workers."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

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
