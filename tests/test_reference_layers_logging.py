"""Tests for reference-layer startup timing logs."""

from __future__ import annotations

import importlib.util
import logging
import types
from pathlib import Path

import numpy as np


def _import_reference_layers_module():
    """Import ``reference_layers.py`` without importing the widgets package."""
    module_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "napari_swc_viewer"
        / "widgets"
        / "reference_layers.py"
    )
    spec = importlib.util.spec_from_file_location(
        "napari_swc_viewer.widgets.reference_layers_test_module",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeAtlas:
    def __init__(self) -> None:
        self.reference = np.zeros((2, 3, 4), dtype=np.float32)


class _FakeViewer:
    def __init__(self) -> None:
        self.calls = []

    def add_image(self, data, **kwargs):
        self.calls.append((data, kwargs))
        return types.SimpleNamespace(data=data, kwargs=kwargs)


def test_add_allen_template_logs_reference_and_add_image_timings(caplog):
    """Template timing should isolate atlas access from napari layer creation."""
    module = _import_reference_layers_module()
    viewer = _FakeViewer()
    atlas = _FakeAtlas()

    with caplog.at_level(logging.DEBUG, logger=module.logger.name):
        layer = module.add_allen_template(viewer, atlas, name="Template")

    assert layer.data is atlas.reference
    assert viewer.calls[0][1]["name"] == "Template"

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "startup_timing event=add_allen_template_phase status=ok" in message
        and "phase=atlas.reference" in message
        and "shape=2x3x4" in message
        and "dtype=float32" in message
        for message in messages
    )
    assert any(
        "startup_timing event=add_allen_template_phase status=ok" in message
        and "phase=viewer.add_image" in message
        and "shape=2x3x4" in message
        and "dtype=float32" in message
        for message in messages
    )
