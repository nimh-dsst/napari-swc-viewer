from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd


class _FakeWidget:
    def __init__(self, *_args, **_kwargs) -> None:
        return None


class _FakeFileDialog:
    @staticmethod
    def getOpenFileName(*_args, **_kwargs):
        return "", ""

    @staticmethod
    def getSaveFileName(*_args, **_kwargs):
        return "", ""


def _load_flatmap_widget_module(monkeypatch):
    fake_qtwidgets = types.ModuleType("qtpy.QtWidgets")
    for name in (
        "QCheckBox",
        "QComboBox",
        "QGroupBox",
        "QHBoxLayout",
        "QLabel",
        "QPushButton",
        "QScrollArea",
        "QSpinBox",
        "QVBoxLayout",
        "QWidget",
    ):
        setattr(fake_qtwidgets, name, _FakeWidget)
    fake_qtwidgets.QFileDialog = _FakeFileDialog
    fake_qtpy = types.ModuleType("qtpy")
    fake_qtpy.QtWidgets = fake_qtwidgets
    fake_notifications = types.ModuleType("napari.utils.notifications")
    fake_notifications.show_info = lambda *_args, **_kwargs: None
    fake_notifications.show_warning = lambda *_args, **_kwargs: None
    fake_utils = types.ModuleType("napari.utils")
    fake_utils.notifications = fake_notifications
    fake_napari = types.ModuleType("napari")
    fake_napari.utils = fake_utils

    monkeypatch.setitem(sys.modules, "qtpy", fake_qtpy)
    monkeypatch.setitem(sys.modules, "qtpy.QtWidgets", fake_qtwidgets)
    monkeypatch.setitem(sys.modules, "napari", fake_napari)
    monkeypatch.setitem(sys.modules, "napari.utils", fake_utils)
    monkeypatch.setitem(
        sys.modules,
        "napari.utils.notifications",
        fake_notifications,
    )

    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "napari_swc_viewer"
        / "widgets"
        / "flatmap.py"
    )
    module_name = "napari_swc_viewer.widgets.flatmap_test_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DummyLabel:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, text: str) -> None:
        self.text = text


class _DummyLayer:
    def __init__(self, data, **kwargs) -> None:
        self.data = np.asarray(data)
        self.name = kwargs["name"]
        self.metadata = kwargs.get("metadata", {})
        self.edge_color = np.asarray(kwargs.get("edge_color", []))
        self.edge_width = kwargs.get("edge_width")
        self.face_color = np.asarray(kwargs.get("face_color", []))
        self.size = kwargs.get("size")
        self.contrast_limits = kwargs.get("contrast_limits")
        self.colormap = kwargs.get("colormap")
        self.blending = kwargs.get("blending")
        self.rendering = kwargs.get("rendering")
        self.opacity = kwargs.get("opacity")
        self.visible = True
        self.refresh_count = 0

    def refresh(self) -> None:
        self.refresh_count += 1


class _DummyViewer:
    def __init__(self) -> None:
        self.layers: list[_DummyLayer] = []
        self.dims = types.SimpleNamespace(ndisplay=3)
        self.camera = types.SimpleNamespace(center=None, zoom=None)

    def add_shapes(self, data, **kwargs) -> _DummyLayer:
        layer = _DummyLayer(data, **kwargs)
        self.layers.append(layer)
        return layer

    def add_image(self, data, **kwargs) -> _DummyLayer:
        layer = _DummyLayer(data, **kwargs)
        self.layers.append(layer)
        return layer

    def add_points(self, data, **kwargs) -> _DummyLayer:
        layer = _DummyLayer(data, **kwargs)
        self.layers.append(layer)
        return layer


def _widget(module):
    widget = module.FlatmapProjectionWidget.__new__(module.FlatmapProjectionWidget)
    widget._viewer = _DummyViewer()
    widget._projection_layer = None
    widget._flatmap_path = Path("flatmap_both_shaped.nrrd")
    widget._depth_path = Path("depth.nrrd")
    widget._status_label = _DummyLabel()
    widget._color_map_provider = lambda: {
        "a.swc": [1.0, 0.0, 0.0, 1.0],
        "b.swc": [0.0, 1.0, 0.0, 0.5],
    }
    return widget


def test_file_ids_for_source_uses_selected_then_all_fallback(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._table_file_ids_provider = lambda: ["a.swc", "b.swc", "a.swc"]
    widget._selected_file_ids_provider = lambda: ["b.swc", "b.swc"]

    assert widget._file_ids_for_source("selected") == ["b.swc"]
    assert widget._file_ids_for_source("all") == ["a.swc", "b.swc"]

    widget._selected_file_ids_provider = lambda: []
    assert widget._file_ids_for_source("selected") == ["a.swc", "b.swc"]


def test_create_heatmap_layer_uses_metadata_and_3d_focus(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    projected = pd.DataFrame({"file_id": ["a.swc", "b.swc"]})
    volume = np.zeros((2, 4, 4), dtype=np.float32)
    volume[0, 1, 2] = 1.0
    volume[1, 3, 0] = 2.0
    render_summary = module.FlatmapRenderSummary(
        3,
        3,
        2,
        1,
        3,
        0,
        2,
        2,
        4,
        2,
        25.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        25.0,
        True,
    )
    render_result = module.FlatmapRenderResult(
        projected_nodes=projected,
        volume=volume,
        points=np.asarray([[0.0, 1.0, 2.0], [1.0, 3.0, 0.0]]),
        point_file_ids=["a.swc", "b.swc"],
        summary=render_summary,
    )
    summary = module.ProjectionSummary(3, 2, 1, 0, 1, 0, 0, 2, 1)
    old_layer = _DummyLayer([], name="Isocortex Flatmap Traces")
    widget._viewer.layers.append(old_layer)

    layer = widget._create_or_update_render_layer(
        render_result,
        summary,
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="heatmap",
    )

    assert layer.name == "Isocortex Flatmap Heatmap"
    assert layer.metadata["projection_kind"] == "isocortex_flatmap"
    assert layer.metadata["flatmap_render_mode"] == "heatmap"
    assert layer.metadata["render_summary"]["rendered_nodes"] == 3
    assert layer.contrast_limits == (0.0, 2.0)
    assert layer._napari_swc_flatmap_projected_nodes is projected
    assert layer._napari_swc_flatmap_summary is summary
    assert layer._napari_swc_flatmap_render_summary is render_summary
    assert old_layer not in widget._viewer.layers
    assert widget._viewer.dims.ndisplay == 3
    assert widget._viewer.camera.center == (0.5, 2.0, 1.0)
    assert widget._viewer.camera.zoom == 300.0


def test_create_points_layer_uses_table_colors(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    render_summary = module.FlatmapRenderSummary(
        2,
        2,
        1,
        1,
        2,
        0,
        2,
        2,
        8,
        2,
        25.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        25.0,
        True,
    )
    render_result = module.FlatmapRenderResult(
        projected_nodes=pd.DataFrame({"file_id": ["a.swc", "b.swc"]}),
        volume=np.zeros((2, 8, 8), dtype=np.float32),
        points=np.asarray([[0.0, 1.0, 2.0], [1.0, 3.0, 4.0]]),
        point_file_ids=["a.swc", "b.swc"],
        summary=render_summary,
    )
    summary = module.ProjectionSummary(2, 1, 0, 0, 1, 0, 0, 2, 1)

    layer = widget._create_or_update_render_layer(
        render_result,
        summary,
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="points",
    )

    assert layer.name == "Isocortex Flatmap Points"
    np.testing.assert_allclose(
        layer.face_color,
        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.5]],
    )
    np.testing.assert_allclose(layer.data, render_result.points)
    assert layer.metadata["flatmap_render_mode"] == "points"


def test_export_current_projection_to_path_writes_csv(monkeypatch, tmp_path) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._last_projected_nodes = pd.DataFrame(
        {
            "file_id": ["a.swc"],
            "node_id": [1],
            "valid": [True],
            "x_flat": [0.25],
            "y_flat": [0.5],
            "flatmap_valid": [True],
            "depth_valid": [True],
            "render_valid": [True],
            "x_flat_bin": [10],
            "y_flat_bin": [20],
            "depth_bin": [1],
            "depth_bin_label": ["0-25 um"],
        }
    )

    output = widget._export_current_projection_to_path(tmp_path / "projection.csv")

    assert output.exists()
    exported = pd.read_csv(output)
    assert exported["file_id"].tolist() == ["a.swc"]
    assert "depth_um" in exported.columns
    assert "coordinate_mode" in exported.columns
    assert "render_valid" in exported.columns
    assert exported["x_flat_bin"].tolist() == [10]
    assert "Exported flatmap projection" in widget._status_label.text
