from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_PATCHED_MODULE_NAMES = [
    "napari",
    "napari.utils",
    "napari.utils.notifications",
    "qtpy",
    "qtpy.QtCore",
    "qtpy.QtWidgets",
    "napari_swc_viewer.widgets",
    "napari_swc_viewer.widgets.analysis_tab",
    "napari_swc_viewer.widgets.mask_layer_selector",
    "napari_swc_viewer.widgets.neuron_table",
    "napari_swc_viewer.widgets.region_selector",
    "napari_swc_viewer.widgets.reference_layers",
    "napari_swc_viewer.widgets.neuron_viewer",
    "napari_swc_viewer.widgets.slice_projection",
]
_ORIGINAL_MODULES = {
    name: sys.modules.get(name) for name in _PATCHED_MODULE_NAMES
}


class _DummyAnalysisSignal:
    def __init__(self) -> None:
        self._callbacks: list = []

    def connect(self, *_args, **_kwargs) -> None:
        if _args:
            self._callbacks.append(_args[0])

    def emit(self, *args, **kwargs) -> None:
        for callback in list(self._callbacks):
            callback(*args, **kwargs)


class _Signal:
    """Minimal descriptor stand-in for ``qtpy.QtCore.Signal``."""

    def __init__(self, *_args, **_kwargs) -> None:
        self._storage_name = ""

    def __set_name__(self, _owner, name: str) -> None:
        self._storage_name = f"__signal_{name}"

    def __get__(self, instance, _owner):
        if instance is None:
            return self
        if self._storage_name not in instance.__dict__:
            instance.__dict__[self._storage_name] = _DummyAnalysisSignal()
        return instance.__dict__[self._storage_name]


class _FakeAnalysisTabWidget:
    def __init__(self, *_args, **_kwargs) -> None:
        self.cluster_colors_updated = _DummyAnalysisSignal()

    def set_slice_projector(self, *_args, **_kwargs) -> None:
        return None

    def set_database(self, *_args, **_kwargs) -> None:
        return None

    def set_atlas(self, *_args, **_kwargs) -> None:
        return None

    def apply_cluster_colors(self) -> None:
        return None


class _FakeQTimer:
    def __init__(self, *_args, **_kwargs) -> None:
        self.timeout = _DummyAnalysisSignal()

    def setSingleShot(self, *_args, **_kwargs) -> None:
        return None

    def setInterval(self, *_args, **_kwargs) -> None:
        return None

    def start(self, *_args, **_kwargs) -> None:
        return None

    def stop(self, *_args, **_kwargs) -> None:
        return None


class _FakeQt:
    Horizontal = 1


class _FakeWidget:
    def __init__(self, *_args, **_kwargs) -> None:
        return None


class _FakeApplication(_FakeWidget):
    @staticmethod
    def processEvents() -> None:
        return None


fake_qtcore = types.ModuleType("qtpy.QtCore")
fake_qtcore.Qt = _FakeQt
fake_qtcore.QThread = _FakeWidget
fake_qtcore.QTimer = _FakeQTimer
fake_qtcore.Signal = _Signal
sys.modules["qtpy.QtCore"] = fake_qtcore

fake_qtwidgets = types.ModuleType("qtpy.QtWidgets")
for _name, _value in {
    "QAbstractItemView": _FakeWidget,
    "QApplication": _FakeApplication,
    "QCheckBox": _FakeWidget,
    "QComboBox": _FakeWidget,
    "QDoubleSpinBox": _FakeWidget,
    "QFileDialog": _FakeWidget,
    "QGroupBox": _FakeWidget,
    "QHeaderView": _FakeWidget,
    "QHBoxLayout": _FakeWidget,
    "QLabel": _FakeWidget,
    "QListWidget": _FakeWidget,
    "QProgressBar": _FakeWidget,
    "QPushButton": _FakeWidget,
    "QScrollArea": _FakeWidget,
    "QSlider": _FakeWidget,
    "QSpinBox": _FakeWidget,
    "QStackedWidget": _FakeWidget,
    "QTableWidget": _FakeWidget,
    "QTableWidgetItem": _FakeWidget,
    "QTabWidget": _FakeWidget,
    "QVBoxLayout": _FakeWidget,
    "QWidget": _FakeWidget,
}.items():
    setattr(fake_qtwidgets, _name, _value)
sys.modules["qtpy.QtWidgets"] = fake_qtwidgets

fake_qtpy = types.ModuleType("qtpy")
fake_qtpy.QtCore = fake_qtcore
fake_qtpy.QtWidgets = fake_qtwidgets
sys.modules["qtpy"] = fake_qtpy

fake_napari_notifications = types.ModuleType("napari.utils.notifications")
fake_napari_notifications.show_info = lambda *args, **kwargs: None
fake_napari_notifications.show_warning = lambda *args, **kwargs: None
sys.modules["napari.utils.notifications"] = fake_napari_notifications

fake_napari_utils = types.ModuleType("napari.utils")
fake_napari_utils.notifications = fake_napari_notifications
sys.modules["napari.utils"] = fake_napari_utils

fake_napari = types.ModuleType("napari")
fake_napari.utils = fake_napari_utils
sys.modules["napari"] = fake_napari

fake_analysis_module = types.ModuleType("napari_swc_viewer.widgets.analysis_tab")
fake_analysis_module.AnalysisTabWidget = _FakeAnalysisTabWidget
sys.modules["napari_swc_viewer.widgets.analysis_tab"] = fake_analysis_module

fake_mask_selector_module = types.ModuleType(
    "napari_swc_viewer.widgets.mask_layer_selector"
)
fake_mask_selector_module.MaskLayerSelectorWidget = _FakeWidget
sys.modules["napari_swc_viewer.widgets.mask_layer_selector"] = fake_mask_selector_module

fake_neuron_table_module = types.ModuleType("napari_swc_viewer.widgets.neuron_table")
fake_neuron_table_module.NeuronTableWidget = _FakeWidget
sys.modules["napari_swc_viewer.widgets.neuron_table"] = fake_neuron_table_module

fake_region_selector_module = types.ModuleType(
    "napari_swc_viewer.widgets.region_selector"
)
fake_region_selector_module.RegionSelectorWidget = _FakeWidget
sys.modules["napari_swc_viewer.widgets.region_selector"] = fake_region_selector_module

fake_reference_layers_module = types.ModuleType(
    "napari_swc_viewer.widgets.reference_layers"
)
for _name in (
    "add_allen_template",
    "add_brain_outline",
    "add_region_mesh",
    "add_region_segmentation",
    "remove_region_layers",
    "remove_region_segmentation",
):
    setattr(fake_reference_layers_module, _name, lambda *args, **kwargs: None)
sys.modules["napari_swc_viewer.widgets.reference_layers"] = fake_reference_layers_module

sys.modules.pop("napari_swc_viewer.widgets.neuron_viewer", None)
widgets_package = types.ModuleType("napari_swc_viewer.widgets")
widgets_package.__path__ = [
    str(
        Path(__file__).resolve().parents[1]
        / "src"
        / "napari_swc_viewer"
        / "widgets"
    )
]
sys.modules["napari_swc_viewer.widgets"] = widgets_package

NeuronViewerWidget = importlib.import_module(
    "napari_swc_viewer.widgets.neuron_viewer"
).NeuronViewerWidget
SomaSliceProjector = importlib.import_module(
    "napari_swc_viewer.widgets.slice_projection"
).SomaSliceProjector

for _name, _module in _ORIGINAL_MODULES.items():
    if _module is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _module


class _DummySignal:
    def __init__(self) -> None:
        self._callbacks: list = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def disconnect(self, callback) -> None:
        if callback in self._callbacks:
            self._callbacks.remove(callback)


class _DummyDims:
    def __init__(
        self,
        ndisplay: int = 2,
        not_displayed: tuple[int, ...] = (0,),
        point: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.ndisplay = ndisplay
        self.not_displayed = not_displayed
        self.point = point
        self.order = (0, 1, 2)
        self.events = types.SimpleNamespace(
            current_step=_DummySignal(),
            ndisplay=_DummySignal(),
            order=_DummySignal(),
        )


class _DummyBlocker:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _DummyLayerEvents:
    def __init__(self) -> None:
        self.highlight = _DummySignal()

    def blocker_all(self) -> _DummyBlocker:
        return _DummyBlocker()


class _DummyPointsLayer:
    def __init__(self, data: np.ndarray, **kwargs) -> None:
        self.data = np.asarray(data, dtype=float)
        face_color = kwargs.get("face_color")
        self.face_color = (
            np.asarray(face_color, dtype=float) if face_color is not None else None
        )
        self.name = kwargs["name"]
        self.scale = kwargs.get("scale")
        self.metadata = kwargs.get("metadata", {})
        self.opacity = kwargs.get("opacity", 1.0)
        self.visible = True
        self.size = kwargs.get("size")
        self.mode = "pan_zoom"
        self.border_color = kwargs.get("border_color")
        self.border_width = kwargs.get("border_width")
        self.events = _DummyLayerEvents()
        self.selected_data: set[int] = set()
        self.refresh_count = 0

    def refresh(self) -> None:
        self.refresh_count += 1


class _DummyShapesLayer:
    def __init__(self, data: np.ndarray, **kwargs) -> None:
        self.data = np.asarray(data, dtype=float)
        self.name = kwargs["name"]
        self.scale = kwargs.get("scale")
        self.metadata = kwargs.get("metadata", {})
        self.opacity = kwargs.get("opacity", 1.0)
        self.visible = True
        self.edge_color = kwargs.get("edge_color")
        self.edge_width = kwargs.get("edge_width")


class _DummyImageLayer:
    def __init__(self, data: np.ndarray, **kwargs) -> None:
        self.data = np.asarray(data, dtype=float)
        self.name = kwargs["name"]
        self.metadata = kwargs.get("metadata", {})
        self.opacity = kwargs.get("opacity", 1.0)
        self.colormap = kwargs.get("colormap")
        self.blending = kwargs.get("blending")
        self.rendering = kwargs.get("rendering")
        self.contrast_limits = kwargs.get("contrast_limits")
        self.visible = kwargs.get("visible", True)


class _DummyViewer:
    def __init__(
        self,
        ndisplay: int = 2,
        not_displayed: tuple[int, ...] = (0,),
        point: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.layers: list[_DummyPointsLayer] = []
        self.dims = _DummyDims(
            ndisplay=ndisplay,
            not_displayed=not_displayed,
            point=point,
        )
        self.status = ""

    def add_points(self, data: np.ndarray, **kwargs) -> _DummyPointsLayer:
        layer = _DummyPointsLayer(data, **kwargs)
        self.layers.append(layer)
        return layer

    def add_shapes(self, data: np.ndarray, **kwargs) -> _DummyShapesLayer:
        layer = _DummyShapesLayer(data, **kwargs)
        self.layers.append(layer)
        return layer

    def add_image(self, data: np.ndarray, **kwargs) -> _DummyImageLayer:
        layer = _DummyImageLayer(data, **kwargs)
        self.layers.append(layer)
        return layer


class _DummyCheckBox:
    def __init__(self, checked: bool) -> None:
        self._checked = checked

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool) -> None:
        self._checked = checked


class _DummyComboBox:
    def __init__(self, text: str) -> None:
        self._text = text

    def currentText(self) -> str:
        return self._text


class _DummyLabel:
    def __init__(self) -> None:
        self.visible = None
        self.text = ""

    def setVisible(self, value: bool) -> None:
        self.visible = value

    def setText(self, value: str) -> None:
        self.text = value


class _DummyButton:
    def __init__(self, text: str = "") -> None:
        self.enabled = True
        self.visible = True
        self.text = text

    def setEnabled(self, value: bool) -> None:
        self.enabled = value

    def setVisible(self, value: bool) -> None:
        self.visible = value

    def setText(self, value: str) -> None:
        self.text = value


class _DummyStack:
    def __init__(self) -> None:
        self.index = None

    def setCurrentIndex(self, index: int) -> None:
        self.index = index


class _DummyProgressBar:
    def __init__(self) -> None:
        self.visible = False
        self.range = (0, 0)
        self.value = 0

    def setRange(self, minimum: int, maximum: int) -> None:
        self.range = (minimum, maximum)

    def setValue(self, value: int) -> None:
        self.value = value

    def setVisible(self, value: bool) -> None:
        self.visible = value


class _DummyValueControl:
    def __init__(self, value) -> None:
        self._value = value

    def value(self):
        return self._value


def _make_soma_projector(
    viewer: _DummyViewer,
    *,
    tolerance: float = 5.0,
    point_size: int = 7,
) -> SomaSliceProjector:
    projector = SomaSliceProjector.__new__(SomaSliceProjector)
    projector._viewer = viewer
    projector._tolerance = tolerance
    projector._point_size = point_size
    projector._highlight_callback = None
    projector._source_data = {}
    projector._projection_layer = None
    projector._scale = None
    projector._enabled = True
    projector._connected = False
    projector._all_coords = None
    projector._all_colors = None
    projector._all_file_ids = None
    projector._axis_index = {}
    projector._last_result_key = None
    projector._last_result = None
    projector._update_timer = MagicMock()
    return projector


def _bind_projection_helpers(widget) -> None:
    widget._soma_projection_active_in_2d = types.MethodType(
        NeuronViewerWidget._soma_projection_active_in_2d,
        widget,
    )
    widget._set_neuron_points_soma_visibility = types.MethodType(
        NeuronViewerWidget._set_neuron_points_soma_visibility,
        widget,
    )
    widget._sync_soma_projection_overlay_state = types.MethodType(
        NeuronViewerWidget._sync_soma_projection_overlay_state,
        widget,
    )


def _bind_scene_helpers(widget) -> None:
    widget._current_scene_file_ids = types.MethodType(
        NeuronViewerWidget._current_scene_file_ids,
        widget,
    )
    widget._build_soma_projection_batch = types.MethodType(
        NeuronViewerWidget._build_soma_projection_batch,
        widget,
    )
    widget._clear_neuron_layers = types.MethodType(
        NeuronViewerWidget._clear_neuron_layers,
        widget,
    )
    widget._render_selected_with_mode = types.MethodType(
        NeuronViewerWidget._render_selected_with_mode,
        widget,
    )


@pytest.mark.parametrize("slice_axis", [0, 1, 2])
def test_soma_slice_projector_flattens_points_onto_active_slice(slice_axis: int) -> None:
    viewer = _DummyViewer()
    projector = _make_soma_projector(viewer, tolerance=1.5)
    coords_a = np.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]])
    coords_b = np.array([[40.0, 50.0, 60.0]])

    projector.add_soma_data_batch(
        {
            "neuron-a": (coords_a, (1.0, 0.0, 0.0, 1.0)),
            "neuron-b": (coords_b, (0.0, 1.0, 0.0, 1.0)),
        }
    )

    slice_position = float(coords_a[0, slice_axis])
    points, colors, file_ids = projector._compute_slice_projection(
        slice_position,
        slice_axis,
    )

    assert points is not None
    assert colors is not None
    assert file_ids == ["neuron-a", "neuron-a"]
    np.testing.assert_allclose(points[:, slice_axis], slice_position)
    kept_axes = [axis for axis in range(3) if axis != slice_axis]
    np.testing.assert_allclose(points[:, kept_axes], coords_a[:, kept_axes])


def test_soma_slice_projector_returns_none_when_slice_has_no_hits() -> None:
    viewer = _DummyViewer()
    projector = _make_soma_projector(viewer, tolerance=0.5)
    projector.add_soma_data("neuron-a", np.array([[10.0, 20.0, 30.0]]))

    points, colors, file_ids = projector._compute_slice_projection(99.0, 0)

    assert points is None
    assert colors is None
    assert file_ids is None


def test_soma_slice_projector_updates_colors_without_rebuilding_geometry() -> None:
    viewer = _DummyViewer()
    projector = _make_soma_projector(viewer)
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    projector.add_soma_data("neuron-a", coords, (0.0, 1.0, 1.0, 1.0))

    projector.update_neuron_colors({"neuron-a": [1.0, 0.0, 0.0, 1.0]})

    assert projector._source_data["neuron-a"][1] == (1.0, 0.0, 0.0, 1.0)
    np.testing.assert_array_almost_equal(
        projector._all_colors,
        np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]),
    )


def test_soma_slice_projector_updates_layer_and_clear_removes_it() -> None:
    viewer = _DummyViewer(
        ndisplay=2,
        not_displayed=(0,),
        point=(10.0, 0.0, 0.0),
    )
    projector = _make_soma_projector(viewer, tolerance=1.0, point_size=9)
    projector.add_soma_data("neuron-a", np.array([[10.0, 5.0, 5.0]]))

    projector._do_update_projection()

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.name == "Soma Slice Projection"
    assert layer.metadata["file_ids"] == ["neuron-a"]
    assert layer.size == 9
    assert layer.border_color == "#39ff14"
    assert layer.border_width == 0.15

    projector.clear()

    assert viewer.layers == []
    assert projector._projection_layer is None


def test_build_soma_projection_batch_uses_rendered_soma_points_when_available() -> None:
    widget = types.SimpleNamespace(_db=MagicMock())
    points_df = pd.DataFrame(
        {
            "file_id": ["n1", "n1", "n1"],
            "type": [1, 2, 1],
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
            "z": [7.0, 8.0, 9.0],
        }
    )

    batch = NeuronViewerWidget._build_soma_projection_batch(
        widget,
        file_ids=["n1"],
        neuron_colors=[[0.1, 0.2, 0.3, 0.4]],
        points_df=points_df,
    )

    widget._db.get_soma_points.assert_not_called()
    coords, color = batch["n1"]
    np.testing.assert_allclose(coords, np.array([[1.0, 4.0, 7.0], [3.0, 6.0, 9.0]]))
    assert color == (0.1, 0.2, 0.3, 0.4)


def test_build_soma_projection_batch_queries_db_for_lines_only_rendering() -> None:
    soma_df = pd.DataFrame(
        {
            "file_id": ["n1", "n2"],
            "neuron_id": ["N1", "N2"],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "z": [5.0, 6.0],
        }
    )
    widget = types.SimpleNamespace(_db=MagicMock())
    widget._db.get_soma_points.return_value = soma_df

    batch = NeuronViewerWidget._build_soma_projection_batch(
        widget,
        file_ids=["n1", "n2"],
        neuron_colors=[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
        points_df=None,
    )

    widget._db.get_soma_points.assert_called_once_with(["n1", "n2"])
    assert set(batch) == {"n1", "n2"}
    np.testing.assert_allclose(batch["n1"][0], np.array([[1.0, 3.0, 5.0]]))
    assert batch["n2"][1] == (0.0, 1.0, 0.0, 1.0)


def test_build_soma_projection_batch_queries_only_missing_soma_only_ids() -> None:
    points_df = pd.DataFrame(
        {
            "file_id": ["n1", "n1"],
            "type": [1, 2],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "z": [5.0, 6.0],
        }
    )
    soma_df = pd.DataFrame(
        {
            "file_id": ["n2"],
            "neuron_id": ["N2"],
            "x": [7.0],
            "y": [8.0],
            "z": [9.0],
        }
    )
    widget = types.SimpleNamespace(_db=MagicMock())
    widget._db.get_soma_points.return_value = soma_df

    batch = NeuronViewerWidget._build_soma_projection_batch(
        widget,
        file_ids=["n1", "n2"],
        neuron_colors=[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
        points_df=points_df,
    )

    widget._db.get_soma_points.assert_called_once_with(["n2"])
    np.testing.assert_allclose(batch["n1"][0], np.array([[1.0, 3.0, 5.0]]))
    np.testing.assert_allclose(batch["n2"][0], np.array([[7.0, 8.0, 9.0]]))
    assert batch["n2"][1] == (0.0, 1.0, 0.0, 1.0)


def test_render_selected_soma_only_rebuilds_scene_with_soma_mode() -> None:
    widget = types.SimpleNamespace(
        _scene_render_modes={"n1": "full"},
        _neuron_table=MagicMock(),
        _render_status_label=_DummyLabel(),
        _render_scene=MagicMock(),
    )
    widget._neuron_table.get_selected_file_ids.return_value = ["n1", "n2"]
    _bind_scene_helpers(widget)

    NeuronViewerWidget._render_selected_soma_only(widget)

    assert widget._scene_render_modes == {"n1": "soma", "n2": "soma"}
    widget._render_scene.assert_called_once_with()


def test_render_selected_neurons_switches_soma_only_back_to_full() -> None:
    widget = types.SimpleNamespace(
        _scene_render_modes={"n1": "soma"},
        _neuron_table=MagicMock(),
        _render_status_label=_DummyLabel(),
        _render_scene=MagicMock(),
    )
    widget._neuron_table.get_selected_file_ids.return_value = ["n1"]
    _bind_scene_helpers(widget)

    NeuronViewerWidget._render_selected_neurons(widget)

    assert widget._scene_render_modes == {"n1": "full"}
    widget._render_scene.assert_called_once_with()


def test_remove_selected_neurons_preserves_remaining_render_modes() -> None:
    widget = types.SimpleNamespace(
        _scene_render_modes={"n1": "full", "n2": "soma"},
        _neuron_table=MagicMock(),
        _render_status_label=_DummyLabel(),
        _render_scene=MagicMock(),
        _capture_depth_state=MagicMock(return_value=None),
        _restore_depth_state=MagicMock(),
        _clear_neuron_layers=MagicMock(),
    )
    widget._neuron_table.get_selected_file_ids.return_value = ["n1"]
    widget._current_scene_file_ids = types.MethodType(
        NeuronViewerWidget._current_scene_file_ids,
        widget,
    )

    NeuronViewerWidget._remove_selected_neurons(widget)

    assert widget._scene_render_modes == {"n2": "soma"}
    widget._render_scene.assert_called_once_with()
    widget._clear_neuron_layers.assert_not_called()


def test_render_scene_queries_full_trace_data_only_for_full_mode_neurons() -> None:
    viewer = _DummyViewer(ndisplay=3)
    widget = types.SimpleNamespace(
        _scene_render_modes={"n1": "full", "n2": "soma"},
        _db=MagicMock(),
        viewer=viewer,
        _render_btn=_DummyButton(),
        _render_soma_only_btn=_DummyButton(),
        _remove_selected_btn=_DummyButton(),
        _render_progress=_DummyProgressBar(),
        _render_status_label=_DummyLabel(),
        _render_mode_combo=_DummyComboBox("Both"),
        _opacity_slider=_DummyValueControl(100),
        _point_size_spin=_DummyValueControl(5),
        _line_width_spin=_DummyValueControl(4),
        _color_by_type_cb=_DummyCheckBox(False),
        _show_slice_projection_cb=_DummyCheckBox(False),
        _neuron_table=types.SimpleNamespace(
            get_color=lambda fid: [1.0, 0.0, 0.0, 1.0],
            set_added_file_ids=MagicMock(),
        ),
        _atlas=None,
        _slice_projector=MagicMock(),
        _soma_slice_projector=MagicMock(),
        _analysis_tab=MagicMock(),
        _current_neuron_layers=[],
        _capture_depth_state=MagicMock(return_value=None),
        _restore_depth_state=MagicMock(),
        _maybe_auto_center_slice=MagicMock(return_value=False),
        _use_auto_centering=lambda: False,
        _apply_layer_visibility=MagicMock(),
        _last_soma_selection=set(),
        _on_soma_selected=MagicMock(),
    )
    _bind_scene_helpers(widget)

    widget._db.get_neuron_lines_batch.return_value = {
        "n1": (
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float),
            np.array([[0, 1]], dtype=np.int32),
        )
    }
    widget._db.get_neurons_for_rendering.return_value = pd.DataFrame(
        {
            "file_id": ["n1", "n1"],
            "type": [1, 2],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "z": [5.0, 6.0],
        }
    )
    widget._db.get_soma_locations.return_value = pd.DataFrame(
        {
            "file_id": ["n1", "n2"],
            "neuron_id": ["N1", "N2"],
            "x": [1.0, 7.0],
            "y": [3.0, 8.0],
            "z": [5.0, 9.0],
        }
    )
    widget._db.get_soma_points.return_value = pd.DataFrame(
        {
            "file_id": ["n2"],
            "neuron_id": ["N2"],
            "x": [7.0],
            "y": [8.0],
            "z": [9.0],
        }
    )

    NeuronViewerWidget._render_scene(widget)

    widget._db.get_neuron_lines_batch.assert_called_once_with(["n1"])
    widget._db.get_neurons_for_rendering.assert_called_once_with(["n1"])
    widget._db.get_soma_locations.assert_called_once_with(["n1", "n2"])
    widget._db.get_soma_points.assert_called_once_with(["n2"])
    widget._slice_projector.add_neuron_data_batch.assert_called_once()
    widget._soma_slice_projector.add_soma_data_batch.assert_called_once()
    assert {layer.name for layer in widget._current_neuron_layers} == {
        "Neuron Lines",
        "Neuron Points",
        "Soma Labels",
    }


def test_apply_layer_visibility_hides_duplicate_soma_markers_in_2d_points_mode() -> None:
    viewer = _DummyViewer(ndisplay=2)
    neuron_points = _DummyPointsLayer(
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        name="Neuron Points",
        face_color=np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
            ]
        ),
        metadata={
            "file_ids_per_point": ["n1", "n1", "n2"],
            "point_types": [1, 2, 1],
            "base_face_colors": np.array(
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
    )
    soma_labels = _DummyPointsLayer(
        np.array([[0.0, 0.0, 0.0]]),
        name="Soma Labels",
        face_color=np.array([[1.0, 0.0, 0.0, 1.0]]),
        metadata={"file_ids": ["n1"]},
    )
    neuron_lines = types.SimpleNamespace(name="Neuron Lines", visible=True)
    widget = types.SimpleNamespace(
        viewer=viewer,
        _show_slice_projection_cb=_DummyCheckBox(True),
        _render_mode_combo=_DummyComboBox("Points"),
        _current_neuron_layers=[neuron_lines, neuron_points, soma_labels],
    )
    _bind_projection_helpers(widget)

    NeuronViewerWidget._apply_layer_visibility(widget, False)

    assert neuron_lines.visible is False
    assert neuron_points.visible is True
    assert soma_labels.visible is False
    assert neuron_points.face_color[0, 3] == 0.0
    assert neuron_points.face_color[1, 3] == 1.0
    assert neuron_points.face_color[2, 3] == 0.0

    widget._show_slice_projection_cb = _DummyCheckBox(False)
    NeuronViewerWidget._apply_layer_visibility(widget, False)

    assert soma_labels.visible is True
    np.testing.assert_allclose(
        neuron_points.face_color,
        neuron_points.metadata["base_face_colors"],
    )


def test_toggle_slice_projection_updates_both_projectors_and_reapplies_2d_visibility() -> None:
    viewer = _DummyViewer(ndisplay=2)
    line_projector = types.SimpleNamespace(enabled=False)
    soma_projector = types.SimpleNamespace(enabled=False)
    widget = types.SimpleNamespace(
        viewer=viewer,
        _slice_projector=line_projector,
        _soma_slice_projector=soma_projector,
        _slice_warning_label=_DummyLabel(),
        _current_neuron_layers=[object()],
        _apply_layer_visibility=MagicMock(),
    )

    NeuronViewerWidget._toggle_slice_projection(widget, 1)

    assert line_projector.enabled is True
    assert soma_projector.enabled is True
    assert widget._slice_warning_label.visible is True
    widget._apply_layer_visibility.assert_called_once_with(False)


def test_update_layer_colors_updates_both_projection_systems() -> None:
    neuron_points = _DummyPointsLayer(
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        name="Neuron Points",
        face_color=np.array([[0.2, 0.2, 0.2, 1.0], [0.3, 0.3, 0.3, 1.0]]),
        metadata={
            "file_ids_per_point": ["n1", "n2"],
            "point_types": [1, 2],
        },
    )
    soma_labels = _DummyPointsLayer(
        np.array([[0.0, 0.0, 0.0]]),
        name="Soma Labels",
        face_color=np.array([[0.2, 0.2, 0.2, 1.0]]),
        metadata={"file_ids": ["n1"]},
    )
    widget = types.SimpleNamespace(
        _current_neuron_layers=[neuron_points, soma_labels],
        _slice_projector=MagicMock(),
        _soma_slice_projector=MagicMock(),
        _sync_soma_projection_overlay_state=MagicMock(),
    )

    NeuronViewerWidget._update_layer_colors(
        widget,
        {"n1": [1.0, 0.0, 0.0, 1.0], "n2": [0.0, 1.0, 0.0, 1.0]},
    )

    np.testing.assert_allclose(
        neuron_points.metadata["base_face_colors"],
        np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]),
    )
    widget._slice_projector.update_neuron_colors.assert_called_once()
    widget._soma_slice_projector.update_neuron_colors.assert_called_once()
    widget._sync_soma_projection_overlay_state.assert_called_once()


def test_on_soma_selected_uses_metadata_file_ids_for_projected_layer() -> None:
    table = MagicMock()
    widget = types.SimpleNamespace(_last_soma_selection=set(), _neuron_table=table)
    layer = types.SimpleNamespace(
        selected_data={0, 2},
        metadata={"file_ids": ["n1", "n2", "n3"]},
    )
    event = types.SimpleNamespace(source=layer)

    NeuronViewerWidget._on_soma_selected(widget, event)

    table.select_file_ids.assert_called_once_with(["n1", "n3"])


@pytest.mark.parametrize(
    ("handler_name", "expected_message", "expected_soma_only"),
    [
        (
            "_query_atlas_neurons_any_node",
            "Searching for neurons with any node in selected atlas regions. Please wait...",
            False,
        ),
        (
            "_query_atlas_neurons_soma",
            "Searching for neurons with soma in selected atlas regions. Please wait...",
            True,
        ),
    ],
)
def test_atlas_query_handlers_set_wait_message_before_dispatch(
    handler_name: str,
    expected_message: str,
    expected_soma_only: bool,
) -> None:
    observed: dict[str, str] = {}

    widget = types.SimpleNamespace(
        _regions_status_label=_DummyLabel(),
    )

    def _record_status(*, soma_only: bool = False) -> None:
        observed["status"] = widget._regions_status_label.text
        observed["soma_only"] = soma_only

    widget._query_neurons_by_region = _record_status

    getattr(NeuronViewerWidget, handler_name)(widget)

    assert observed["status"] == expected_message
    assert observed["soma_only"] is expected_soma_only


@pytest.mark.parametrize(
    ("handler_name", "expected_message", "expected_soma_only"),
    [
        (
            "_query_mask_neurons_any_node",
            "Searching for neurons with any node in selected mask layers. Please wait...",
            False,
        ),
        (
            "_query_mask_neurons_soma",
            "Searching for neurons with soma in selected mask layers. Please wait...",
            True,
        ),
    ],
)
def test_mask_query_handlers_set_wait_message_before_dispatch(
    handler_name: str,
    expected_message: str,
    expected_soma_only: bool,
) -> None:
    observed: dict[str, str] = {}

    widget = types.SimpleNamespace(
        _regions_status_label=_DummyLabel(),
    )

    def _record_status(*, soma_only: bool = False) -> None:
        observed["status"] = widget._regions_status_label.text
        observed["soma_only"] = soma_only

    widget._query_neurons_by_mask = _record_status

    getattr(NeuronViewerWidget, handler_name)(widget)

    assert observed["status"] == expected_message
    assert observed["soma_only"] is expected_soma_only


def test_on_region_query_source_changed_shows_relevant_button_pair() -> None:
    widget = types.SimpleNamespace(
        _region_query_source="Atlas Regions",
        _region_query_stack=_DummyStack(),
        _regions_status_label=_DummyLabel(),
        _atlas_query_any_node_btn=_DummyButton(),
        _atlas_query_soma_btn=_DummyButton(),
        _mask_query_any_node_btn=_DummyButton(),
        _mask_query_soma_btn=_DummyButton(),
    )
    widget._atlas_region_query_buttons = types.MethodType(
        NeuronViewerWidget._atlas_region_query_buttons,
        widget,
    )
    widget._mask_layer_query_buttons = types.MethodType(
        NeuronViewerWidget._mask_layer_query_buttons,
        widget,
    )

    NeuronViewerWidget._on_region_query_source_changed(widget, "Atlas Regions")

    assert widget._region_query_stack.index == 0
    assert widget._atlas_query_any_node_btn.visible is True
    assert widget._atlas_query_soma_btn.visible is True
    assert widget._mask_query_any_node_btn.visible is False
    assert widget._mask_query_soma_btn.visible is False

    NeuronViewerWidget._on_region_query_source_changed(widget, "Mask Layer")

    assert widget._region_query_stack.index == 1
    assert widget._atlas_query_any_node_btn.visible is False
    assert widget._atlas_query_soma_btn.visible is False
    assert widget._mask_query_any_node_btn.visible is True
    assert widget._mask_query_soma_btn.visible is True


def test_refresh_neuron_table_summary_formats_counts_and_clusters() -> None:
    summary = types.SimpleNamespace(
        table_count=4,
        added_count=2,
        visible_count=3,
        cluster_counts=((1, 2), (3, 1), (None, 1)),
    )
    widget = types.SimpleNamespace(
        _neuron_table=types.SimpleNamespace(summary=lambda: summary),
        _neuron_table_summary_label=_DummyLabel(),
    )

    NeuronViewerWidget._refresh_neuron_table_summary(widget)

    assert widget._neuron_table_summary_label.text == (
        "In table: 4 | Added to scene: 2 | Visible: 3\n"
        "Clusters: Cluster 1: 2, Cluster 3: 1, Unclustered: 1"
    )


def test_clear_neuron_table_preserves_scene_render_modes() -> None:
    widget = types.SimpleNamespace(
        _highlighted_file_ids=None,
        _current_neuron_layers=[object()],
        _neuron_table=types.SimpleNamespace(
            clear=MagicMock(),
            _entries={"n1": types.SimpleNamespace(color=[1.0, 0.0, 0.0, 1.0], visible=True)},
        ),
        _last_soma_selection={"n1"},
        _refresh_cluster_filter_controls=MagicMock(),
        _refresh_neuron_table_summary=MagicMock(),
        _render_status_label=_DummyLabel(),
        _regions_status_label=_DummyLabel(),
        _scene_render_modes={"n1": "full"},
    )

    NeuronViewerWidget._clear_neuron_table(widget)

    widget._neuron_table.clear.assert_called_once_with()
    widget._refresh_cluster_filter_controls.assert_called_once_with()
    widget._refresh_neuron_table_summary.assert_called_once_with()
    assert widget._scene_render_modes == {"n1": "full"}
    assert widget._last_soma_selection == set()
    assert widget._render_status_label.text == "Cleared neuron table."
    assert widget._regions_status_label.text == "Cleared neuron table."


def test_clear_neuron_table_clears_highlight_without_recoloring_scene_to_gray() -> None:
    entry = types.SimpleNamespace(color=[0.2, 0.3, 0.4, 1.0], visible=True)
    widget = types.SimpleNamespace(
        _highlighted_file_ids={"n1"},
        _current_neuron_layers=[object()],
        _neuron_table=types.SimpleNamespace(
            clear=MagicMock(),
            _entries={"n1": entry},
        ),
        _last_soma_selection=set(),
        _refresh_cluster_filter_controls=MagicMock(),
        _refresh_neuron_table_summary=MagicMock(),
        _render_status_label=_DummyLabel(),
        _regions_status_label=_DummyLabel(),
        _update_layer_colors=MagicMock(),
    )
    widget._build_effective_color_map = types.MethodType(
        NeuronViewerWidget._build_effective_color_map,
        widget,
    )

    NeuronViewerWidget._clear_neuron_table(widget)

    widget._update_layer_colors.assert_called_once_with(
        {"n1": [0.2, 0.3, 0.4, 1.0]}
    )
    assert widget._highlighted_file_ids is None


def test_selected_neuron_heatmap_layer_name_appends_suffixes() -> None:
    viewer = _DummyViewer(ndisplay=3)
    viewer.layers.extend(
        [
            types.SimpleNamespace(name="Neuron Heatmap: n1"),
            types.SimpleNamespace(name="Neuron Heatmap: n1 (2)"),
        ]
    )
    widget = types.SimpleNamespace(viewer=viewer)
    widget._iter_viewer_layers = types.MethodType(
        NeuronViewerWidget._iter_viewer_layers,
        widget,
    )
    widget._unique_layer_name = types.MethodType(
        NeuronViewerWidget._unique_layer_name,
        widget,
    )
    widget._selected_neuron_heatmap_base_name = types.MethodType(
        NeuronViewerWidget._selected_neuron_heatmap_base_name,
        widget,
    )

    layer_name = NeuronViewerWidget._selected_neuron_heatmap_layer_name(
        widget,
        ["n1"],
    )

    assert layer_name == "Neuron Heatmap: n1 (3)"


def test_add_selected_neuron_heatmap_layer_sets_single_selection_metadata() -> None:
    viewer = _DummyViewer(ndisplay=3)
    widget = types.SimpleNamespace(
        viewer=viewer,
        _db=types.SimpleNamespace(parquet_path=Path("/tmp/neurons.parquet")),
        _atlas=types.SimpleNamespace(atlas_name="fake_atlas"),
        _opacity_slider=_DummyValueControl(80),
    )
    widget._iter_viewer_layers = types.MethodType(
        NeuronViewerWidget._iter_viewer_layers,
        widget,
    )
    widget._unique_layer_name = types.MethodType(
        NeuronViewerWidget._unique_layer_name,
        widget,
    )
    widget._selected_neuron_heatmap_base_name = types.MethodType(
        NeuronViewerWidget._selected_neuron_heatmap_base_name,
        widget,
    )
    widget._selected_neuron_heatmap_layer_name = types.MethodType(
        NeuronViewerWidget._selected_neuron_heatmap_layer_name,
        widget,
    )
    widget._current_atlas_name = types.MethodType(
        NeuronViewerWidget._current_atlas_name,
        widget,
    )

    layer = NeuronViewerWidget._add_selected_neuron_heatmap_layer(
        widget,
        np.array([[[0.0, 5.0]]], dtype=np.float32),
        ["n1"],
    )

    assert layer.name == "Neuron Heatmap: n1"
    assert layer.contrast_limits == (0.0, 5.0)
    assert layer.metadata["heatmap_kind"] == "selected_neurons"
    assert layer.metadata["atlas_name"] == "fake_atlas"
    assert layer.metadata["source_path"] == str(Path("/tmp/neurons.parquet"))
    assert layer.metadata["file_ids"] == ["n1"]
    assert layer.metadata["selection_count"] == 1
    assert layer.metadata["heatmap_source"] is True
    assert layer.metadata["heatmap_native_grid"] is True


def test_selected_neuron_heatmap_finished_adds_unique_multi_selection_layer() -> None:
    viewer = _DummyViewer(ndisplay=3)
    viewer.layers.extend(
        [
            types.SimpleNamespace(name="Neuron Heatmap: 2 selected neurons"),
            types.SimpleNamespace(name="Neuron Heatmap: 2 selected neurons (2)"),
        ]
    )
    widget = types.SimpleNamespace(
        viewer=viewer,
        _db=types.SimpleNamespace(parquet_path=Path("/tmp/neurons.parquet")),
        _atlas=types.SimpleNamespace(atlas_name="fake_atlas"),
        _opacity_slider=_DummyValueControl(75),
        _render_progress=_DummyProgressBar(),
        _render_status_label=_DummyLabel(),
        _selected_heatmap_request_file_ids=("n1", "n2"),
        _refresh_heatmap_layer_list=MagicMock(),
        _refresh_histogram_layer_list=MagicMock(),
        _refresh_mask_layer_options=MagicMock(),
    )
    widget._iter_viewer_layers = types.MethodType(
        NeuronViewerWidget._iter_viewer_layers,
        widget,
    )
    widget._unique_layer_name = types.MethodType(
        NeuronViewerWidget._unique_layer_name,
        widget,
    )
    widget._selected_neuron_heatmap_base_name = types.MethodType(
        NeuronViewerWidget._selected_neuron_heatmap_base_name,
        widget,
    )
    widget._selected_neuron_heatmap_layer_name = types.MethodType(
        NeuronViewerWidget._selected_neuron_heatmap_layer_name,
        widget,
    )
    widget._current_atlas_name = types.MethodType(
        NeuronViewerWidget._current_atlas_name,
        widget,
    )
    widget._add_selected_neuron_heatmap_layer = types.MethodType(
        NeuronViewerWidget._add_selected_neuron_heatmap_layer,
        widget,
    )

    NeuronViewerWidget._on_selected_neuron_heatmap_finished(
        widget,
        np.ones((2, 2, 2), dtype=np.float32),
    )

    created_layer = viewer.layers[-1]
    assert created_layer.name == "Neuron Heatmap: 2 selected neurons (3)"
    assert created_layer.metadata["file_ids"] == ["n1", "n2"]
    assert created_layer.metadata["selection_count"] == 2
    widget._refresh_heatmap_layer_list.assert_called_once_with()
    widget._refresh_histogram_layer_list.assert_called_once_with()
    widget._refresh_mask_layer_options.assert_called_once_with()
    assert "Neuron Heatmap: 2 selected neurons (3)" in widget._render_status_label.text
