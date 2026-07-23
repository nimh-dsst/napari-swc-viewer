from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest

from napari_swc_viewer.flatmap_labels import (
    FlatmapRegionLabelsResult,
    FlatmapRegionLabelsSummary,
)


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


class _DummySignal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, *args, **kwargs) -> None:
        for callback in list(self._callbacks):
            callback(*args, **kwargs)


def _load_flatmap_widget_module(monkeypatch):
    fake_qtwidgets = types.ModuleType("qtpy.QtWidgets")
    for name in (
        "QCheckBox",
        "QComboBox",
        "QDoubleSpinBox",
        "QGroupBox",
        "QHBoxLayout",
        "QLabel",
        "QProgressBar",
        "QPushButton",
        "QScrollArea",
        "QSpinBox",
        "QVBoxLayout",
        "QWidget",
    ):
        setattr(fake_qtwidgets, name, _FakeWidget)
    fake_qtwidgets.QFileDialog = _FakeFileDialog
    fake_qtcore = types.ModuleType("qtpy.QtCore")
    fake_qtcore.QThread = _FakeWidget
    fake_qtpy = types.ModuleType("qtpy")
    fake_qtpy.QtWidgets = fake_qtwidgets
    fake_qtpy.QtCore = fake_qtcore
    fake_notifications = types.ModuleType("napari.utils.notifications")
    fake_notifications.show_info = lambda *_args, **_kwargs: None
    fake_notifications.show_warning = lambda *_args, **_kwargs: None
    fake_utils = types.ModuleType("napari.utils")
    fake_utils.notifications = fake_notifications
    fake_napari = types.ModuleType("napari")
    fake_napari.utils = fake_utils

    monkeypatch.setitem(sys.modules, "qtpy", fake_qtpy)
    monkeypatch.setitem(sys.modules, "qtpy.QtWidgets", fake_qtwidgets)
    monkeypatch.setitem(sys.modules, "qtpy.QtCore", fake_qtcore)
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


class _DummyButton:
    def __init__(self) -> None:
        self.enabled = True

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)


class _DummyValueControl:
    def __init__(self, value=0, *, checked: bool = False) -> None:
        self.value = value
        self.checked = bool(checked)
        self.enabled = True

    def setValue(self, value) -> None:
        self.value = value

    def setChecked(self, checked: bool) -> None:
        self.checked = bool(checked)

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def isChecked(self) -> bool:
        return self.checked


class _DummyProgressBar:
    def __init__(self) -> None:
        self.visible = False
        self.range = (0, 1)
        self.value = 0

    def setVisible(self, visible: bool) -> None:
        self.visible = bool(visible)

    def setRange(self, minimum: int, maximum: int) -> None:
        self.range = (int(minimum), int(maximum))

    def setValue(self, value: int) -> None:
        self.value = int(value)


class _DummyCombo:
    def __init__(self, text: str = "allen_mouse_10um") -> None:
        self.text = text
        self.enabled = True

    def currentText(self) -> str:
        return self.text

    def setCurrentText(self, text: str) -> None:
        self.text = str(text)

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)


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
        self.ndim = self.data.ndim if self.data.ndim else 0
        self.contrast_limits_range = kwargs.get(
            "contrast_limits_range",
            self.contrast_limits,
        )
        self._keep_auto_contrast = False
        self._slice_input = types.SimpleNamespace(ndisplay=2)
        self.visible = True
        self.refresh_count = 0
        self.thumbnail_updates = 0
        self.slice_updates: list[object] = []
        self.slice_dims_calls = []
        self.raise_status_error = False
        self.status_error_message = (
            "too many indices for array: array is 2-dimensional, but 3 were indexed"
        )

    def refresh(self) -> None:
        self.refresh_count += 1

    def _update_thumbnail(self) -> None:
        self.thumbnail_updates += 1

    def reset_contrast_limits(self, mode=None) -> None:
        self.contrast_limits = (-1.0, float(mode or -1.0))

    def reset_contrast_limits_range(self, mode=None) -> None:
        self.contrast_limits_range = (-2.0, float(mode or -2.0))

    def _update_slice_response(self, response) -> object:
        self.slice_updates.append(response)
        self.contrast_limits = (4.0, 5.0)
        self.contrast_limits_range = (4.0, 5.0)
        return response

    def _slice_dims(self, dims, force: bool = False) -> None:
        self.slice_dims_calls.append((dims, force))

    def _get_source_info(self) -> dict[str, str]:
        return {
            "layer_name": self.name,
            "layer_base": self.name,
            "source_type": "",
            "plugin": "",
        }

    def get_status(
        self,
        position=None,
        *,
        view_direction=None,
        dims_displayed=None,
        world=False,
        value=None,
    ) -> dict[str, str]:
        if self.raise_status_error:
            raise IndexError(self.status_error_message)
        return {
            "layer_name": self.name,
            "layer_base": self.name,
            "source_type": "",
            "plugin": "",
            "coordinates": "",
            "coords": "",
            "value": "",
        }


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

    def add_labels(self, data, **kwargs) -> _DummyLayer:
        layer = _DummyLayer(data, **kwargs)
        self.layers.append(layer)
        return layer

    def add_surface(self, data, **kwargs) -> _DummyLayer:
        vertices, faces, values = data
        layer = _DummyLayer(vertices, **kwargs)
        layer.faces = np.asarray(faces)
        layer.values = np.asarray(values)
        self.layers.append(layer)
        return layer

    def add_vectors(self, data, **kwargs) -> _DummyLayer:
        layer = _DummyLayer(data, **kwargs)
        self.layers.append(layer)
        return layer


def _widget(module):
    widget = module.FlatmapProjectionWidget.__new__(module.FlatmapProjectionWidget)
    widget._viewer = _DummyViewer()
    widget._projection_layer = None
    widget._region_labels_layer = None
    widget._flatmap_path = Path("flatmap_both_shaped.nrrd")
    widget._depth_path = Path("depth.nrrd")
    widget._status_label = _DummyLabel()
    widget._region_labels_status_label = _DummyLabel()
    widget._project_btn = _DummyButton()
    widget._projection_progress_bar = _DummyProgressBar()
    widget._region_label_atlas_combo = _DummyCombo("allen_mouse_10um")
    widget._region_labels_btn = _DummyButton()
    widget._clear_region_labels_btn = _DummyButton()
    widget._region_label_atlas_cache = {}
    widget._region_label_atlas_load_thread = None
    widget._region_label_atlas_load_worker = None
    widget._pending_region_label_request = False
    widget._color_map_provider = lambda: {
        "a.swc": [1.0, 0.0, 0.0, 1.0],
        "b.swc": [0.0, 1.0, 0.0, 0.5],
    }
    widget._cluster_map_provider = lambda: {}
    widget._atlas_provider = lambda: None
    widget._selected_region_acronyms_provider = lambda: ["VISp"]
    widget._zero_sentinel_cb = types.SimpleNamespace(isChecked=lambda: False)
    widget._negative_one_sentinel_cb = types.SimpleNamespace(isChecked=lambda: True)
    widget._flatmap_correlation_source_changed_callback = None
    return widget


def test_lookup_stats_cache_reuses_matching_file_and_sentinel_settings(
    monkeypatch,
    tmp_path,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    flatmap_path = tmp_path / "flatmap.nrrd"
    depth_path = tmp_path / "depth.nrrd"
    flatmap_path.write_text("flatmap")
    depth_path.write_text("depth")
    volume_set = types.SimpleNamespace(
        flatmap=np.zeros((2, 2, 2, 2), dtype=np.float32),
        depth=np.ones((2, 2, 2), dtype=np.float32),
        flatmap_path=flatmap_path,
        depth_path=depth_path,
    )
    calls = []

    def fake_compute(flatmap, depth, **kwargs):
        calls.append(kwargs)
        return module.FlatmapLookupStats(
            x_bounds=(0.0, 1.0),
            y_bounds=(0.0, 1.0),
            depth_range_um=(0.0, 1.0),
            flatmap_valid_voxels=1,
            depth_valid_voxels=1,
            flatmap_shape=tuple(flatmap.shape),
            depth_shape=tuple(depth.shape),
            flatmap_dtype=str(flatmap.dtype),
            depth_dtype=str(depth.dtype),
            invalid_zero_sentinel=kwargs["invalid_zero_sentinel"],
            invalid_negative_one_sentinel=kwargs["invalid_negative_one_sentinel"],
        )

    monkeypatch.setattr(module, "compute_flatmap_lookup_stats", fake_compute)

    first = widget._lookup_stats_for_volume_set(
        volume_set,
        invalid_zero_sentinel=False,
        invalid_negative_one_sentinel=True,
    )
    second = widget._lookup_stats_for_volume_set(
        volume_set,
        invalid_zero_sentinel=False,
        invalid_negative_one_sentinel=True,
    )
    third = widget._lookup_stats_for_volume_set(
        volume_set,
        invalid_zero_sentinel=True,
        invalid_negative_one_sentinel=True,
    )

    assert first is second
    assert third is not first
    assert len(calls) == 2


def test_project_cache_profile_restore_waits_for_atlas_then_selects_saved_profile(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)

    class _ProfileCombo:
        def __init__(self) -> None:
            self.items: list[tuple[str, object]] = []
            self.index = -1

        def blockSignals(self, _blocked: bool) -> None:
            return None

        def clear(self) -> None:
            self.items.clear()
            self.index = -1

        def addItem(self, label: str, value: object) -> None:
            self.items.append((str(label), value))
            if self.index < 0:
                self.index = 0

        def count(self) -> int:
            return len(self.items)

        def itemData(self, index: int):
            return self.items[index][1]

        def setCurrentIndex(self, index: int) -> None:
            self.index = int(index)

        def currentData(self):
            return self.itemData(self.index) if self.index >= 0 else None

    class _ValueControl:
        def __init__(self) -> None:
            self.value = None
            self.checked = False
            self.enabled = True

        def setValue(self, value) -> None:
            self.value = value

        def setChecked(self, checked: bool) -> None:
            self.checked = bool(checked)

        def setEnabled(self, enabled: bool) -> None:
            self.enabled = bool(enabled)

    class _Profile:
        def __init__(self, profile_id: str, xy_bins: int) -> None:
            self.profile_id = profile_id
            self.atlas = {"name": "allen_mouse_25um"}
            self._grid = {
                "xy_bins": xy_bins,
                "depth_bin_um": 25.0,
            }
            self.compatibility_calls: list[dict[str, object]] = []

        def compatibility_mismatches(self, **requirements):
            self.compatibility_calls.append(requirements)
            return ()

        def style(self, style: str):
            assert style == "both_shaped"
            return types.SimpleNamespace(grid_spec=self._grid)

    first = _Profile("first-profile", 128)
    saved = _Profile("saved-profile", 256)
    cache = types.SimpleNamespace(
        profiles={first.profile_id: first, saved.profile_id: saved}
    )
    import napari_swc_viewer.flatmap_region_cache as cache_module

    monkeypatch.setattr(cache_module, "open_region_cache", lambda _path: cache)
    monkeypatch.setattr(
        module,
        "read_flatmap_parquet_transform_info",
        lambda _path: types.SimpleNamespace(
            format_version=3,
            lookup_set_id="lookup-set",
            metadata={"shared_depth_definition": {"mirror_coord_axis": 2}},
        ),
    )

    widget._region_cache_dir = None
    widget._region_cache = None
    widget._active_cache_profile = None
    widget._pending_cache_profile_id = None
    widget._cache_dir_label = _DummyLabel()
    widget._cache_status_label = _DummyLabel()
    widget._cache_profile_combo = _ProfileCombo()
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._projection_source_combo = types.SimpleNamespace(
        currentData=lambda: module._PROJECTION_SOURCE_PRECOMPUTED
    )
    widget._current_source_parquet_path = lambda: Path("neurons.parquet")
    widget._invalidate_flatmap_grid_layers = lambda: None
    widget._xy_bins_spin = _ValueControl()
    widget._depth_bin_spin = _ValueControl()
    widget._exclude_depth_minus_one_cb = _ValueControl()
    widget._negative_one_sentinel_cb = _ValueControl()
    widget._zero_sentinel_cb = _ValueControl()
    widget._region_surfaces_btn = _DummyButton()
    widget._region_outlines_btn = _DummyButton()
    widget._clear_region_geometry_btn = _DummyButton()
    atlas_holder = {"atlas": None}
    widget._atlas_provider = lambda: atlas_holder["atlas"]
    widget._request_cache_directory_open = (
        lambda path, profile_id=None: widget.set_cache_directory(
            path,
            profile_id=profile_id,
        )
    )

    widget.restore_cache_reference(
        {"path": "/relocated/cache", "profile_id": saved.profile_id}
    )

    assert widget._pending_cache_profile_id == saved.profile_id
    assert widget._active_cache_profile is None
    assert widget._cache_profile_combo.count() == 0

    atlas_holder["atlas"] = types.SimpleNamespace(
        atlas_name="allen_mouse_10um",
        local_version=(1, 2),
        structures={
            1: {
                "id": 1,
                "name": "root",
                "acronym": "root",
                "structure_id_path": [1],
                "rgb_triplet": [0, 0, 0],
            }
        },
    )
    widget.refresh_cache_profiles()

    assert widget._active_cache_profile is saved
    assert widget._cache_profile_combo.currentData() is saved
    assert widget._pending_cache_profile_id == saved.profile_id
    assert widget._xy_bins_spin.value == 256
    assert widget._depth_bin_spin.value == 25.0
    assert saved.compatibility_calls[-1]["atlas_version"] == "1.2"


def test_set_cache_directory_commits_before_closing_superseded_cache(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    events: list[str] = []

    class _Cache:
        def __init__(self, name: str) -> None:
            self.name = name
            self.closed = False

        def close(self) -> None:
            self.closed = True
            events.append(f"close:{self.name}")

    old_cache = _Cache("old")
    new_cache = _Cache("new")
    new_profile = types.SimpleNamespace(profile_id="new-profile")
    import napari_swc_viewer.flatmap_region_cache as cache_module

    monkeypatch.setattr(cache_module, "open_region_cache", lambda _path: new_cache)
    widget._region_cache_dir = Path("old-cache")
    widget._region_cache = old_cache
    widget._active_cache_profile = object()
    widget._pending_cache_profile_id = None
    widget._cache_dir_label = _DummyLabel()
    widget._compatible_cache_profile_entries = lambda _cache: (
        ("new profile", new_profile),
    )
    widget._populate_cache_profile_combo = lambda _entries, _profile: None
    widget._activate_cache_profile = lambda profile, **_kwargs: (
        events.append(f"activate:{profile.profile_id}"),
        setattr(widget, "_active_cache_profile", profile),
    )
    widget._set_cache_grid_locked = lambda _locked: None

    widget.set_cache_directory(Path("new-cache"))

    assert events == ["activate:new-profile", "close:old"]
    assert old_cache.closed is True
    assert widget._region_cache is new_cache
    assert widget._active_cache_profile is new_profile

    widget._active_cache_profile = object()
    widget._deactivate_cache_profile = lambda: events.append("deactivate")
    widget.set_cache_directory(None)

    assert events[-2:] == ["deactivate", "close:new"]
    assert new_cache.closed is True
    assert widget._region_cache is None


def test_cache_build_finished_closes_worker_returned_profile(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    closed: list[bool] = []
    profile = types.SimpleNamespace(
        profile_id="built-profile",
        close=lambda: closed.append(True),
    )
    widget._region_cache_dir = Path("cache")
    requests: list[tuple[Path, str | None]] = []
    widget._request_cache_directory_open = (
        lambda path, profile_id=None: requests.append((Path(path), profile_id))
    )
    widget._cache_status_label = _DummyLabel()

    widget._on_cache_build_finished(profile)

    assert closed == [True]
    assert requests == [(Path("cache"), "built-profile")]


class _CacheProfile:
    def __init__(self, profile_id: str, *, output_shape=(1, 4, 4)) -> None:
        self.profile_id = profile_id
        self.atlas = {"name": "allen_mouse_25um"}
        self.grid = {
            "output_shape": list(output_shape),
            "xy_bins": 4,
            "depth_bins": 1,
            "depth_bin_um": 25.0,
            "x_bounds": [0.0, 1.0],
            "y_bounds": [0.0, 1.0],
            "depth_bounds_um": [0.0, 25.0],
            "includes_depth_minus_one_plane": False,
        }

    def style(self, style: str):
        assert style == "both_shaped"
        return types.SimpleNamespace(grid_spec=self.grid)


def _configure_cache_activation_widget(widget, module) -> _DummyLayer:
    layer = _DummyLayer(
        np.ones((1, 4, 4), dtype=np.float32),
        name="Isocortex Flatmap Heatmap",
        metadata={"flatmap_render_mode": "heatmap"},
    )
    widget._viewer.layers.append(layer)
    widget._projection_layer = layer
    widget._region_surfaces_layers = []
    widget._region_outlines_layers = []
    widget._active_cache_profile = None
    widget._pending_cache_profile_id = None
    widget._region_cache_dir = Path("cache")
    widget._cache_status_label = _DummyLabel()
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._projection_source_combo = types.SimpleNamespace(
        currentData=lambda: module._PROJECTION_SOURCE_PRECOMPUTED
    )
    widget._xy_bins_spin = _DummyValueControl(4)
    widget._depth_bin_spin = _DummyValueControl(25.0)
    widget._exclude_depth_minus_one_cb = _DummyValueControl(checked=False)
    widget._negative_one_sentinel_cb = _DummyValueControl(checked=True)
    widget._zero_sentinel_cb = _DummyValueControl(checked=False)
    widget._region_surfaces_btn = _DummyButton()
    widget._region_outlines_btn = _DummyButton()
    widget._clear_region_geometry_btn = _DummyButton()
    widget._last_projected_nodes = pd.DataFrame({"file_id": ["a.swc"]})
    widget._last_summary = _simple_projection_summary(module)
    widget._last_render_summary = _simple_render_summary(
        module,
        includes_depth_minus_one_plane=False,
    )
    widget._last_render_mode = module._RENDER_HEATMAP
    widget._last_flatmap_style = "both_shaped"
    widget._last_coordinate_mode = "parquet_columns"
    widget._last_volume_shape = (1, 4, 4)
    widget._last_projection_source = module._PROJECTION_SOURCE_PRECOMPUTED
    widget._last_cache_dir = None
    widget._last_cache_profile_id = None
    return layer


def test_choose_cache_directory_only_schedules_background_open(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    layer = _DummyLayer(
        np.ones((1, 4, 4), dtype=np.float32),
        name="Isocortex Flatmap Heatmap",
        metadata={"flatmap_render_mode": "heatmap"},
    )
    widget._viewer.layers.append(layer)
    requests = []
    widget._request_cache_directory_open = (
        lambda path, profile_id=None: requests.append((path, profile_id))
    )
    monkeypatch.setattr(
        module.QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: "/cache/path",
        raising=False,
    )

    widget._choose_cache_directory()

    assert requests == [("/cache/path", None)]
    assert widget._viewer.layers == [layer]
    assert layer.visible is True


def test_matching_cache_profile_preserves_live_heatmap(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    layer = _configure_cache_activation_widget(widget, module)
    profile = _CacheProfile("matching-profile")
    queued = []
    widget._queue_gui_callback = queued.append

    widget._activate_cache_profile(profile, force_transition=True)

    assert widget._viewer.layers == [layer]
    assert layer.visible is True
    assert queued == []
    assert widget._last_cache_dir == "cache"
    assert widget._last_cache_profile_id == "matching-profile"
    assert layer.metadata["cache_path"] == "cache"
    assert layer.metadata["cache_profile_id"] == "matching-profile"
    assert "matching heatmap kept" in widget._cache_status_label.text


def test_matching_cache_profile_only_retires_other_profile_annotations(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    heatmap = _configure_cache_activation_widget(widget, module)
    profile = _CacheProfile("matching-profile")
    matching_labels = _DummyLayer(
        np.ones((1, 4, 4), dtype=np.int32),
        name="Flatmap Region Labels",
        metadata={
            "cache_path": "old-cache",
            "cache_profile_id": "matching-profile",
            "flatmap_style": "both_shaped",
        },
    )
    stale_surface = _DummyLayer(
        np.ones((1, 3), dtype=np.float32),
        name="Flatmap Region Surfaces",
        metadata={
            "cache_path": "old-cache",
            "cache_profile_id": "other-profile",
            "flatmap_style": "both_shaped",
        },
    )
    widget._viewer.layers.extend([matching_labels, stale_surface])
    widget._region_labels_layer = matching_labels
    widget._region_surfaces_layers = [stale_surface]
    queued = []
    widget._queue_gui_callback = queued.append

    widget._activate_cache_profile(profile, force_transition=True)

    assert heatmap.visible is True
    assert matching_labels.visible is True
    assert matching_labels.metadata["cache_path"] == "cache"
    assert widget._region_labels_layer is matching_labels
    assert stale_surface.visible is False
    assert widget._region_surfaces_layers == []
    assert len(queued) == 1

    queued[0]()

    assert widget._viewer.layers == [heatmap, matching_labels]


def test_mismatched_cache_profile_hides_then_defers_heatmap_removal(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    layer = _configure_cache_activation_widget(widget, module)
    profile = _CacheProfile("different-grid", output_shape=(2, 4, 4))
    queued = []
    widget._queue_gui_callback = queued.append

    widget._activate_cache_profile(profile, force_transition=True)

    assert widget._viewer.layers == [layer]
    assert layer.visible is False
    assert len(queued) == 1
    assert widget._last_render_summary is None
    assert "click Project to Flatmap again" in widget._cache_status_label.text

    queued[0]()

    assert widget._viewer.layers == []


def test_failed_candidate_cache_preserves_active_cache_and_heatmap(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    layer = _DummyLayer(
        np.ones((1, 4, 4), dtype=np.float32),
        name="Isocortex Flatmap Heatmap",
        metadata={"flatmap_render_mode": "heatmap"},
    )
    widget._viewer.layers.append(layer)
    closed = []
    old_cache = object()
    old_profile = types.SimpleNamespace(profile_id="old-profile")
    candidate = types.SimpleNamespace(close=lambda: closed.append(True))
    widget._region_cache = old_cache
    widget._region_cache_dir = Path("old-cache")
    widget._active_cache_profile = old_profile
    widget._cache_status_label = _DummyLabel()
    widget._cache_open_request_serial = 3
    widget._cache_open_shutting_down = False
    warnings = []
    monkeypatch.setattr(module, "show_warning", warnings.append)

    def incompatible(_cache):
        raise RuntimeError("no compatible profile")

    widget._compatible_cache_profile_entries = incompatible

    widget._on_cache_open_finished(candidate, 3, Path("bad-cache"), None)

    assert closed == [True]
    assert widget._region_cache is old_cache
    assert widget._active_cache_profile is old_profile
    assert widget._viewer.layers == [layer]
    assert layer.visible is True
    assert warnings and "no compatible profile" in warnings[0]


def test_stale_background_cache_result_is_closed(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    closed = []
    candidate = types.SimpleNamespace(close=lambda: closed.append(True))
    widget._cache_open_request_serial = 9
    widget._cache_open_shutting_down = False

    widget._on_cache_open_finished(candidate, 8, Path("stale-cache"), None)

    assert closed == [True]


def test_refresh_cache_profiles_activates_selected_profile_once(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)

    class _Combo:
        def __init__(self) -> None:
            self.items = []
            self.index = -1

        def blockSignals(self, _blocked) -> None:
            return None

        def clear(self) -> None:
            self.items = []
            self.index = -1

        def addItem(self, label, value) -> None:
            self.items.append((label, value))
            if self.index < 0:
                self.index = 0

        def setCurrentIndex(self, index) -> None:
            self.index = int(index)

        def currentData(self):
            return self.items[self.index][1] if self.index >= 0 else None

    first = _CacheProfile("first")
    selected = _CacheProfile("selected")
    widget._region_cache = object()
    widget._cache_profile_combo = _Combo()
    widget._pending_cache_profile_id = "selected"
    widget._active_cache_profile = None
    widget._compatible_cache_profile_entries = lambda _cache: (
        ("first", first),
        ("selected", selected),
    )
    activated = []
    widget._activate_cache_profile = activated.append

    widget._refresh_cache_profiles()

    assert activated == [selected]
    assert widget._cache_profile_combo.currentData() is selected


def test_file_ids_for_source_uses_selected_then_all_fallback(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._table_file_ids_provider = lambda: ["a.swc", "b.swc", "a.swc"]
    widget._selected_file_ids_provider = lambda: ["b.swc", "b.swc"]

    assert widget._file_ids_for_source("selected") == ["b.swc"]
    assert widget._file_ids_for_source("all") == ["a.swc", "b.swc"]

    widget._selected_file_ids_provider = lambda: []
    assert widget._file_ids_for_source("selected") == ["a.swc", "b.swc"]


def test_region_label_atlas_defaults_to_10um(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)

    assert widget._current_region_label_atlas_name() == "allen_mouse_10um"


def _label_result(region_id: int = 7) -> FlatmapRegionLabelsResult:
    summary = FlatmapRegionLabelsSummary(
        input_voxels=1,
        selected_region_count=1,
        selected_source_voxels=1,
        valid_source_voxels=1,
        labeled_voxels=1,
        collision_voxels=0,
        xy_bins=1,
        depth_bins=1,
        depth_bin_um=25.0,
        x_flat_min=0.0,
        x_flat_max=1.0,
        y_flat_min=0.0,
        y_flat_max=1.0,
        depth_min_um=0.0,
        depth_max_um=25.0,
    )
    return FlatmapRegionLabelsResult(
        labels=np.asarray([[[region_id]]], dtype=np.int32),
        summary=summary,
        selected_region_ids=[region_id],
        represented_region_ids=[region_id],
    )


def _configure_region_label_creation_widget(widget) -> None:
    widget._selected_region_ids_provider = lambda: [7]
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._zero_sentinel_cb = types.SimpleNamespace(isChecked=lambda: False)
    widget._negative_one_sentinel_cb = types.SimpleNamespace(isChecked=lambda: True)
    widget._xy_bins_spin = types.SimpleNamespace(value=lambda: 1)
    widget._depth_bin_spin = types.SimpleNamespace(value=lambda: 25)
    widget._focus_projection_view = lambda *_args, **_kwargs: None
    widget._lookup_stats_for_volume_set = lambda *_args, **_kwargs: object()


def test_create_region_labels_uses_flatmap_selected_atlas_not_main_loaded_atlas(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    _configure_region_label_creation_widget(widget)
    atlas10 = types.SimpleNamespace(
        atlas_name="allen_mouse_10um",
        annotation=np.asarray([[[10]]], dtype=np.int32),
        structures={7: {"rgb_triplet": [255, 0, 0]}},
    )
    atlas25 = types.SimpleNamespace(
        atlas_name="allen_mouse_25um",
        annotation=np.asarray([[[25]]], dtype=np.int32),
        structures={7: {"rgb_triplet": [0, 255, 0]}},
    )
    widget._region_label_atlas_cache = {"allen_mouse_10um": atlas10}
    widget._atlas_provider = lambda: atlas25
    captured = {}

    monkeypatch.setattr(
        module,
        "load_flatmap_volume_set",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            flatmap=np.zeros((1, 1, 1, 2), dtype=np.float32),
            depth=np.zeros((1, 1, 1), dtype=np.float32),
            flatmap_path=Path("flatmap.nrrd"),
            depth_path=Path("depth.nrrd"),
        ),
    )

    def fake_build(annotation, *_args, **_kwargs):
        captured["annotation"] = np.asarray(annotation).copy()
        captured["mirror_depth_fallback"] = _kwargs["mirror_depth_fallback"]
        captured["mirror_coord_axis"] = _kwargs["mirror_coord_axis"]
        return _label_result(7)

    monkeypatch.setattr(module, "build_flatmap_region_label_volume", fake_build)

    widget._create_region_labels_from_current_state()

    np.testing.assert_array_equal(captured["annotation"], atlas10.annotation)
    assert captured["mirror_depth_fallback"] is True
    assert captured["mirror_coord_axis"] == 2
    assert widget._viewer.layers[-1].metadata["atlas_name"] == "allen_mouse_10um"


def _install_fake_region_label_atlas_worker(monkeypatch, module):
    workers = []
    threads = []

    class _FakeThread:
        def __init__(self) -> None:
            self.started = _DummySignal()
            self.finished = _DummySignal()
            self.running = False
            self.started_called = False
            threads.append(self)

        def start(self) -> None:
            self.running = True
            self.started_called = True
            self.started.emit()

        def quit(self, *_args) -> None:
            self.running = False
            self.finished.emit()

        def isRunning(self) -> bool:
            return self.running

        def deleteLater(self) -> None:
            return None

    class _FakeAtlasLoadWorker:
        def __init__(self, atlas_name: str) -> None:
            self.atlas_name = atlas_name
            self.status = _DummySignal()
            self.progress = _DummySignal()
            self.finished = _DummySignal()
            self.error = _DummySignal()
            self.thread = None
            workers.append(self)

        def moveToThread(self, thread) -> None:
            self.thread = thread

        def run(self) -> None:
            return None

        def deleteLater(self) -> None:
            return None

    fake_qtcore = sys.modules["qtpy.QtCore"]
    fake_qtcore.QThread = _FakeThread
    fake_workers = types.ModuleType("napari_swc_viewer.workers")
    fake_workers.AtlasLoadWorker = _FakeAtlasLoadWorker
    monkeypatch.setitem(sys.modules, "napari_swc_viewer.workers", fake_workers)
    return workers, threads


def test_unloaded_region_label_atlas_loads_then_creates_labels(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    workers, threads = _install_fake_region_label_atlas_worker(monkeypatch, module)
    widget = _widget(module)
    _configure_region_label_creation_widget(widget)
    calls = []

    def fake_create_for_atlas(atlas, selected_region_ids):
        calls.append((atlas, selected_region_ids))
        return _label_result(7)

    widget._create_region_labels_for_atlas = fake_create_for_atlas

    widget._create_region_labels()

    assert len(workers) == 1
    assert workers[0].atlas_name == "allen_mouse_10um"
    assert workers[0].thread is threads[0]
    assert threads[0].started_called is True
    assert widget._region_label_atlas_combo.enabled is False
    assert widget._region_labels_btn.enabled is False
    assert "Loading region-label atlas allen_mouse_10um" in widget._status_label.text
    atlas10 = types.SimpleNamespace(
        atlas_name="allen_mouse_10um",
        annotation=np.zeros((1, 1, 1), dtype=np.int32),
        structures={},
    )

    workers[0].finished.emit(atlas10)

    assert widget._region_label_atlas_cache["allen_mouse_10um"] is atlas10
    assert calls == [(atlas10, [7])]
    assert widget._region_label_atlas_combo.enabled is True
    assert widget._region_labels_btn.enabled is True
    assert widget._region_label_atlas_load_thread is None
    assert widget._region_label_atlas_load_worker is None


def test_region_label_atlas_load_error_restores_controls(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    workers, _threads = _install_fake_region_label_atlas_worker(monkeypatch, module)
    widget = _widget(module)
    _configure_region_label_creation_widget(widget)
    calls = []
    widget._create_region_labels_for_atlas = lambda *_args: calls.append("created")

    widget._create_region_labels()
    workers[0].error.emit("download failed")

    assert calls == []
    assert widget._pending_region_label_request is False
    assert widget._region_label_atlas_combo.enabled is True
    assert widget._region_labels_btn.enabled is True
    assert "download failed" in widget._status_label.text


def _configure_augmentation_widget(widget, source_mode: str, source_path: Path) -> None:
    widget._database_provider = lambda: types.SimpleNamespace(parquet_path=source_path)
    widget._table_file_ids_provider = lambda: ["a.swc", "b.swc", "a.swc"]
    widget._selected_file_ids_provider = lambda: ["b.swc", "b.swc"]
    widget._source_combo = types.SimpleNamespace(currentData=lambda: source_mode)
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._coordinate_mode_combo = types.SimpleNamespace(currentData=lambda: "microns")
    widget._zero_sentinel_cb = types.SimpleNamespace(isChecked=lambda: False)
    widget._negative_one_sentinel_cb = types.SimpleNamespace(isChecked=lambda: True)


def test_augment_current_parquet_to_path_passes_selected_file_ids(
    monkeypatch,
    tmp_path,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    source = tmp_path / "source.parquet"
    output = tmp_path / "augmented.parquet"
    _configure_augmentation_widget(widget, module._SOURCE_SELECTED, source)
    captured = {}

    def fake_augment(source_path, output_path, flatmap_path, depth_path, **kwargs):
        captured.update(
            {
                "source_path": source_path,
                "output_path": output_path,
                "flatmap_path": flatmap_path,
                "depth_path": depth_path,
                "kwargs": kwargs,
            }
        )
        return types.SimpleNamespace(
            output_parquet=Path(output_path),
            rows=2,
            direct_rows=1,
            mirrored_rows=1,
            unmapped_rows=0,
        )

    monkeypatch.setattr(module, "augment_neuron_parquet_with_flatmap", fake_augment)

    summary = widget._augment_current_parquet_to_path(output)

    assert summary.rows == 2
    assert captured["source_path"] == source
    assert captured["output_path"] == output
    assert captured["kwargs"]["file_ids"] == ["b.swc"]
    assert "2 rows from 1 file ID" in widget._status_label.text


def test_augment_current_parquet_to_path_passes_all_table_file_ids(
    monkeypatch,
    tmp_path,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    source = tmp_path / "source.parquet"
    output = tmp_path / "augmented.parquet"
    _configure_augmentation_widget(widget, module._SOURCE_ALL, source)
    captured = {}

    def fake_augment(_source_path, _output_path, _flatmap_path, _depth_path, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            output_parquet=output,
            rows=3,
            direct_rows=2,
            mirrored_rows=0,
            unmapped_rows=1,
        )

    monkeypatch.setattr(module, "augment_neuron_parquet_with_flatmap", fake_augment)

    widget._augment_current_parquet_to_path(output)

    assert captured["file_ids"] == ["a.swc", "b.swc"]
    assert "3 rows from 2 file ID" in widget._status_label.text


def _augmented_nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file_id": ["a.swc", "a.swc", "b.swc"],
            "neuron_id": ["a", "a", "b"],
            "subject": ["s", "s", "s"],
            "node_id": [1, 2, 1],
            "parent_id": [-1, 1, -1],
            "type": [1, 3, 1],
            "x": [10.0, 20.0, 30.0],
            "y": [10.0, 20.0, 30.0],
            "z": [10.0, 20.0, 30.0],
            "region_id": [1, 1, 2],
            "region_acronym": ["R1", "R1", "R2"],
            "x_flat": [0.0, 1.0, np.nan],
            "y_flat": [0.0, 1.0, 2.0],
            "depth_um": [0.0, 25.0, 10.0],
            "flatmap_valid": [True, True, False],
            "depth_valid": [True, True, True],
            "flatmap_projection_valid": [True, True, False],
            "flatmap_invalid_code": [0, 0, 3],
            "flatmap_lookup_mode": ["direct", "direct", "unmapped"],
        }
    )


def _v3_augmented_nodes() -> pd.DataFrame:
    nodes = _augmented_nodes().drop(
        columns=[
            "x_flat",
            "y_flat",
            "flatmap_valid",
            "flatmap_projection_valid",
            "flatmap_invalid_code",
            "flatmap_lookup_mode",
        ]
    )
    nodes["x_flat_shaped"] = [1.0, 2.0, np.nan]
    nodes["y_flat_shaped"] = [3.0, 4.0, np.nan]
    nodes["flatmap_shaped_valid"] = [True, True, False]
    nodes["flatmap_shaped_projection_valid"] = [True, True, False]
    nodes["flatmap_shaped_invalid_code"] = [0, 0, 3]
    nodes["flatmap_shaped_lookup_mode"] = ["direct", "direct", "unmapped"]
    nodes["x_flat_square"] = [11.0, 12.0, np.nan]
    nodes["y_flat_square"] = [13.0, 14.0, np.nan]
    nodes["flatmap_square_valid"] = [True, True, False]
    nodes["flatmap_square_projection_valid"] = [True, True, False]
    nodes["flatmap_square_invalid_code"] = [0, 0, 3]
    nodes["flatmap_square_lookup_mode"] = ["direct", "direct", "unmapped"]
    nodes["depth_invalid_code"] = [0, 0, 0]
    nodes["depth_lookup_mode"] = ["direct", "mirrored_depth", "mirrored_depth"]
    return nodes


def _configure_projection_widget(widget, module, nodes: pd.DataFrame) -> None:
    widget._database_provider = lambda: types.SimpleNamespace(
        get_neurons_for_rendering=lambda file_ids: nodes[
            nodes["file_id"].isin(file_ids)
        ].reset_index(drop=True),
    )
    widget._table_file_ids_provider = lambda: ["a.swc", "b.swc"]
    widget._selected_file_ids_provider = lambda: []
    widget._source_combo = types.SimpleNamespace(currentData=lambda: module._SOURCE_ALL)
    widget._xy_bins_spin = types.SimpleNamespace(value=lambda: 4)
    widget._depth_bin_spin = types.SimpleNamespace(value=lambda: 25)
    widget._exclude_depth_minus_one_cb = types.SimpleNamespace(isChecked=lambda: False)
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._coordinate_mode_combo = types.SimpleNamespace(currentData=lambda: "microns")


def test_project_without_nrrds_uses_augmented_parquet_columns(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._flatmap_path = None
    widget._depth_path = None
    nodes = _augmented_nodes()
    nodes.loc[1, "flatmap_lookup_mode"] = "mirrored_depth"
    _configure_projection_widget(widget, module, nodes)
    captured = {}

    def fake_apply(result, render_result, **kwargs):
        captured["result"] = result
        captured["render_result"] = render_result
        captured["kwargs"] = kwargs

    widget._apply_projection_result = fake_apply

    widget._project()

    assert captured["kwargs"]["flatmap_style"] == "precomputed_parquet"
    assert captured["kwargs"]["coordinate_mode"] == "parquet_columns"
    projected = captured["result"].projected_nodes
    assert projected["valid"].tolist() == [True, True, False]
    assert projected["flatmap_lookup_mode"].tolist() == [
        "direct",
        "mirrored_depth",
        "unmapped",
    ]
    assert projected["invalid_reason"].tolist() == ["", "", "invalid_flatmap"]
    assert captured["render_result"].summary.rendered_nodes == 2
    assert "Parquet flatmap/depth columns" in widget._status_label.text


def test_project_with_nrrds_overrides_augmented_parquet_columns(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    _configure_projection_widget(widget, module, _augmented_nodes())
    calls = {"lookup": 0, "parquet": 0}

    def fake_lookup(self, nodes, **_kwargs):
        calls["lookup"] += 1
        result = types.SimpleNamespace(projected_nodes=nodes, summary=object())
        render = types.SimpleNamespace(
            projected_nodes=nodes,
            summary=types.SimpleNamespace(rendered_nodes=1, total_nodes=1),
        )
        return result, render, None

    def fake_parquet(self, _nodes, **_kwargs):
        calls["parquet"] += 1
        raise AssertionError("Parquet branch should not run when both NRRDs are set")

    widget._project_from_lookup_files = types.MethodType(fake_lookup, widget)
    widget._project_from_parquet_columns = types.MethodType(fake_parquet, widget)
    widget._apply_projection_result = lambda *_args, **_kwargs: None

    widget._project()

    assert calls == {"lookup": 1, "parquet": 0}
    assert "lookup NRRDs" in widget._status_label.text


def test_explicit_precomputed_v3_square_never_loads_selected_nrrds(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    nodes = _v3_augmented_nodes()
    # Unrelated custom columns retained by whole-Parquet preprocessing must not
    # override the namespaced v3 projection validity/reason columns.
    nodes["valid"] = [False, False, True]
    nodes["invalid_reason"] = ["out_of_bounds"] * 3
    _configure_projection_widget(widget, module, nodes)
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_square")
    widget._projection_source_combo = types.SimpleNamespace(
        currentData=lambda: module._PROJECTION_SOURCE_PRECOMPUTED
    )
    # Points render mode drives the per-node precomputed pipeline this test
    # exercises; heatmap mode now uses the DuckDB volume fast path instead.
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_POINTS
    )
    widget._validate_precomputed_parquet_contract = lambda _nodes: None
    widget._flatmap_path = Path("selected-but-unused.nrrd")
    widget._depth_path = Path("selected-but-unused-depth.nrrd")
    widget._canonical_render_bounds = lambda: {
        "x_bounds": (0.0, 20.0),
        "y_bounds": (0.0, 20.0),
        "depth_range_um": (0.0, 50.0),
    }
    monkeypatch.setattr(
        module,
        "load_flatmap_volume_set",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("precomputed projection must not load NRRDs")
        ),
    )
    captured = {}
    widget._apply_projection_result = lambda result, render, **kwargs: captured.update(
        result=result,
        render=render,
        kwargs=kwargs,
    )

    widget._project()

    projected = captured["result"].projected_nodes
    assert projected["x_flat"].tolist()[:2] == [11.0, 12.0]
    assert projected["y_flat"].tolist()[:2] == [13.0, 14.0]
    assert projected["flatmap_lookup_mode"].tolist() == [
        "direct",
        "mirrored_depth",
        "unmapped",
    ]
    assert projected["valid"].tolist() == [True, True, False]
    assert projected["invalid_reason"].tolist() == ["", "", "invalid_flatmap"]
    assert captured["kwargs"]["flatmap_style"] == "both_square"


def test_explicit_precomputed_rejects_partial_v3_columns(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    nodes = _v3_augmented_nodes().drop(columns=["depth_invalid_code"])
    _configure_projection_widget(widget, module, nodes)
    widget._projection_source_combo = types.SimpleNamespace(
        currentData=lambda: module._PROJECTION_SOURCE_PRECOMPUTED
    )
    # Points render mode keeps the per-node validation path under test.
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_POINTS
    )
    applied: list[object] = []
    widget._apply_projection_result = lambda *args, **kwargs: applied.append(
        (args, kwargs)
    )

    widget._project()

    assert applied == []
    assert "missing required flatmap column(s): ['depth_invalid_code']" in (
        widget._status_label.text
    )
    assert "Regenerate it with Prepare Whole Parquet" in widget._status_label.text


@pytest.mark.parametrize(
    ("transform_info", "expected_message"),
    [
        (
            types.SimpleNamespace(
                format_version=0,
                lookup_set_id=None,
                metadata=None,
            ),
            "complete version-3 metadata with a lookup-set ID",
        ),
        (
            types.SimpleNamespace(
                format_version=3,
                lookup_set_id="lookup-set-id",
                metadata={"version": 3, "canonical_bounds": {}},
            ),
            "no valid canonical bounds for both_shaped",
        ),
    ],
)
def test_explicit_precomputed_rejects_incomplete_canonical_metadata(
    monkeypatch,
    transform_info,
    expected_message: str,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    _configure_projection_widget(widget, module, _v3_augmented_nodes())
    widget._projection_source_combo = types.SimpleNamespace(
        currentData=lambda: module._PROJECTION_SOURCE_PRECOMPUTED
    )
    widget._current_source_parquet_path = lambda: Path("neurons.parquet")
    # Points render mode keeps the per-node validation path under test.
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_POINTS
    )
    monkeypatch.setattr(
        module,
        "read_flatmap_parquet_transform_info",
        lambda _path: transform_info,
    )
    applied: list[object] = []
    widget._apply_projection_result = lambda *args, **kwargs: applied.append(
        (args, kwargs)
    )

    widget._project()

    assert applied == []
    assert expected_message in widget._status_label.text


def test_precomputed_cache_grid_values_are_used_without_ui_rounding(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._projection_source_combo = types.SimpleNamespace(
        currentData=lambda: module._PROJECTION_SOURCE_PRECOMPUTED
    )
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    style_cache = types.SimpleNamespace(
        grid_spec={"xy_bins": 3, "depth_bin_um": 12.3456}
    )
    widget._active_cache_profile = types.SimpleNamespace(
        style=lambda _style: style_cache
    )
    # The UI controls cannot represent this small/fractional profile exactly.
    widget._xy_bins_spin = types.SimpleNamespace(value=lambda: 16)
    widget._depth_bin_spin = types.SimpleNamespace(value=lambda: 12.346)

    assert widget._current_xy_bins() == 3
    assert widget._current_depth_bin_um() == 12.3456


def test_project_from_lookup_files_uses_depth_mirror_fallback(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    nodes = pd.DataFrame(
        {
            "file_id": ["a.swc"],
            "neuron_id": ["a"],
            "subject": ["s"],
            "node_id": [1],
            "parent_id": [-1],
            "type": [3],
            "x": [1.0],
            "y": [0.0],
            "z": [0.0],
            "region_id": [1],
            "region_acronym": ["R1"],
        }
    )
    _configure_projection_widget(widget, module, nodes)
    widget._coordinate_mode_combo = types.SimpleNamespace(
        currentData=lambda: module.COORDINATE_MODE_VOXELS
    )

    grid = np.indices((4, 4, 4), dtype=float)
    flatmap = np.stack((grid[0] + 0.25, grid[1] + 0.5), axis=-1).astype(np.float32)
    depth = (grid[2] + 100.0).astype(np.float32)
    depth[1, 0, 0] = -1.0
    volume_set = types.SimpleNamespace(
        flatmap=flatmap,
        depth=depth,
        flatmap_path=Path("flatmap.nrrd"),
        depth_path=Path("depth.nrrd"),
        space_directions=None,
        space_origin=None,
    )
    monkeypatch.setattr(module, "load_flatmap_volume_set", lambda *_args: volume_set)

    result, render_result, _lookup_stats = widget._project_from_lookup_files(nodes)

    projected = result.projected_nodes
    assert projected["flatmap_lookup_mode"].tolist() == ["mirrored_depth"]
    assert projected["voxel_k"].tolist() == [0]
    assert projected["x_flat"].tolist() == pytest.approx([1.25])
    assert projected["depth_um"].tolist() == pytest.approx([103.0])
    assert result.summary.mirrored_depth_lookup_nodes == 1
    assert result.summary.mirrored_lookup_nodes == 0
    assert render_result.summary.rendered_nodes == 1
    summary_text = widget._format_render_summary(
        result.summary,
        render_result.summary,
    )
    assert "Lookup direct/mirrored-depth/mirrored/unmapped" in summary_text
    assert summary_text.endswith("0/1/0/0")


def test_project_updates_progress_bar_and_restores_button(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._flatmap_path = None
    widget._depth_path = None
    _configure_projection_widget(widget, module, _augmented_nodes())
    progress_events = []
    original_set_progress = widget._set_projection_progress

    def record_progress(message: str, current: int, total: int) -> None:
        progress_events.append((message, current, total))
        original_set_progress(message, current, total)

    widget._set_projection_progress = record_progress
    widget._apply_projection_result = lambda *_args, **_kwargs: None

    widget._project()

    assert progress_events[0] == ("Querying neuron rows...", 0, 4)
    assert ("Reading precomputed flatmap columns...", 1, 4) in progress_events
    assert ("Building flatmap render data...", 2, 4) in progress_events
    assert progress_events[-1] == ("Done", 4, 4)
    assert widget._project_btn.enabled is True
    assert widget._projection_progress_bar.visible is False
    assert widget._projection_progress_bar.range == (0, 1)
    assert widget._projection_progress_bar.value == 0


def test_latest_flatmap_correlation_source_requires_heatmap_render(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._last_projected_nodes = pd.DataFrame(
        {
            "file_id": ["a.swc", "b.swc"],
            "render_valid": [True, True],
            "depth_bin": [0, 1],
            "y_flat_bin": [0, 1],
            "x_flat_bin": [0, 1],
        }
    )
    widget._last_render_summary = module.FlatmapRenderSummary(
        total_nodes=2,
        flatmap_valid_nodes=2,
        depth_valid_nodes=2,
        depth_minus_one_nodes=0,
        rendered_nodes=2,
        excluded_depth_minus_one_nodes=0,
        nonzero_voxels=2,
        traces_represented=2,
        xy_bins=4,
        depth_bins=2,
        depth_bin_um=25.0,
        x_flat_min=0.0,
        x_flat_max=1.0,
        y_flat_min=0.0,
        y_flat_max=1.0,
        depth_min_um=0.0,
        depth_max_um=25.0,
        includes_depth_minus_one_plane=False,
    )
    widget._last_volume_shape = (2, 4, 4)
    widget._last_input_file_ids = ("a.swc", "b.swc")
    widget._last_flatmap_style = "both"
    widget._last_coordinate_mode = "microns"
    widget._last_flatmap_path = str(widget._flatmap_path)
    widget._last_depth_path = str(widget._depth_path)
    widget._last_lookup_stats = module.FlatmapLookupStats(
        x_bounds=(0.0, 1.0),
        y_bounds=(0.0, 1.0),
        depth_range_um=(0.0, 25.0),
        flatmap_valid_voxels=2,
        depth_valid_voxels=2,
        flatmap_shape=(1, 1, 1, 2),
        depth_shape=(1, 1, 1),
        flatmap_dtype="float32",
        depth_dtype="float32",
        invalid_zero_sentinel=False,
        invalid_negative_one_sentinel=True,
    )

    widget._last_render_mode = module._RENDER_POINTS
    assert widget.latest_flatmap_correlation_source() is None

    widget._last_render_mode = module._RENDER_HEATMAP
    assert widget.latest_flatmap_correlation_source() is None

    layer = _DummyLayer(
        np.zeros(widget._last_volume_shape, dtype=np.float32),
        name=module._HEATMAP_LAYER_NAME,
        metadata={"flatmap_render_mode": module._RENDER_HEATMAP},
    )
    widget._projection_layer = layer
    widget._viewer.layers.append(layer)
    source = widget.latest_flatmap_correlation_source()

    assert source is not None
    assert source.volume_shape == (2, 4, 4)
    assert source.input_file_ids == ("a.swc", "b.swc")
    assert source.xy_bins == 4
    assert source.depth_bin_um == 25.0
    assert source.mirror_depth_fallback is True
    assert source.mirror_coord_axis == 2


def test_project_without_nrrds_requires_augmented_parquet_columns(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._flatmap_path = None
    widget._depth_path = None
    nodes = _augmented_nodes().drop(columns=["x_flat", "y_flat", "depth_um"])
    _configure_projection_widget(widget, module, nodes)

    widget._project()

    assert (
        "augmented Parquet with x_flat, y_flat, and depth_um"
        in widget._status_label.text
    )


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
    assert layer.metadata["flatmap_heatmap_color_mode"] == "single"
    assert layer.metadata["render_summary"]["rendered_nodes"] == 3
    assert layer.metadata["flatmap_heatmap_contrast_limits"] == (0.0, 2.0)
    assert layer.colormap == "hot"
    assert layer.contrast_limits == (0.0, 2.0)
    assert layer._napari_swc_flatmap_projected_nodes is projected
    assert layer._napari_swc_flatmap_summary is summary
    assert layer._napari_swc_flatmap_render_summary is render_summary
    assert old_layer not in widget._viewer.layers
    assert widget._viewer.dims.ndisplay == 3
    assert layer.slice_dims_calls[-1] == (widget._viewer.dims, True)
    assert widget._viewer.camera.center == (0.5, 2.0, 1.0)
    assert widget._viewer.camera.zoom == 300.0


def test_flatmap_render_layers_use_display_viewer_provider(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    main_viewer = widget._viewer
    display_viewer = _DummyViewer()
    widget._display_viewer_provider = lambda create=True: (
        display_viewer if create else display_viewer
    )
    projected = pd.DataFrame({"file_id": ["a.swc"]})
    volume = np.zeros((1, 4, 4), dtype=np.float32)
    volume[0, 1, 2] = 1.0
    render_result = module.FlatmapRenderResult(
        projected_nodes=projected,
        volume=volume,
        points=np.asarray([[0.0, 1.0, 2.0]]),
        point_file_ids=["a.swc"],
        summary=_simple_render_summary(module),
    )

    layer = widget._create_or_update_render_layer(
        render_result,
        _simple_projection_summary(module),
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="heatmap",
    )

    assert main_viewer.layers == []
    assert display_viewer.layers == [layer]
    assert display_viewer.dims.ndisplay == 3
    assert layer.slice_dims_calls[-1] == (display_viewer.dims, True)
    assert display_viewer.camera.center == (0.0, 1.0, 2.0)


@pytest.mark.parametrize(
    ("render_mode", "heatmap_color_mode"),
    [
        ("heatmap", "single"),
        ("heatmap", "individual"),
        ("points", "single"),
    ],
)
def test_first_render_reports_ready_after_layer_focus(
    monkeypatch,
    render_mode: str,
    heatmap_color_mode: str,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    display_viewer = _DummyViewer()
    widget._display_viewer_provider = lambda create=True: display_viewer
    widget._heatmap_color_mode_combo = types.SimpleNamespace(
        currentData=lambda: heatmap_color_mode
    )
    events = []
    original_focus = widget._focus_projection_view

    def record_focus(layer, data) -> None:
        original_focus(layer, data)
        events.append(("focused", layer))

    def record_ready(viewer, layer) -> None:
        assert viewer is display_viewer
        assert layer in display_viewer.layers
        assert events[-1] == ("focused", layer)
        events.append(("ready", layer))

    widget._focus_projection_view = record_focus
    widget._display_viewer_ready_callback = record_ready

    layer = widget._create_or_update_render_layer(
        _binned_render_result(module),
        _simple_projection_summary(module, total_nodes=4),
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode=render_mode,
    )

    assert layer is not None
    assert events == [("focused", layer), ("ready", layer)]


def test_display_failure_callback_receives_current_viewer(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    display_viewer = _DummyViewer()
    widget._display_viewer_provider = lambda create=True: display_viewer
    calls = []
    widget._display_viewer_failed_callback = (
        lambda viewer, reason: calls.append((viewer, reason))
    )

    widget._notify_display_viewer_failed("projection_failed")

    assert calls == [(display_viewer, "projection_failed")]


def test_release_display_viewer_clears_only_matching_viewer_layer_handles(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    viewer = _DummyViewer()
    other_viewer = _DummyViewer()
    projection_layer = object()
    labels_layer = object()
    surfaces = [object()]
    outlines = [object()]
    projected_nodes = pd.DataFrame({"file_id": ["a.swc"]})
    cache_profile = object()
    widget._last_display_viewer = viewer
    widget._projection_layer = projection_layer
    widget._region_labels_layer = labels_layer
    widget._region_surfaces_layers = surfaces
    widget._region_outlines_layers = outlines
    widget._last_projected_nodes = projected_nodes
    widget._active_cache_profile = cache_profile

    assert widget._release_display_viewer(other_viewer) is False
    assert widget._last_display_viewer is viewer
    assert widget._projection_layer is projection_layer
    assert widget._region_labels_layer is labels_layer
    assert widget._region_surfaces_layers is surfaces
    assert widget._region_outlines_layers is outlines

    assert widget._release_display_viewer(viewer) is True
    assert widget._last_display_viewer is None
    assert widget._projection_layer is None
    assert widget._region_labels_layer is None
    assert widget._region_surfaces_layers == []
    assert widget._region_outlines_layers == []
    assert widget._last_projected_nodes is projected_nodes
    assert widget._active_cache_profile is cache_profile


def _simple_projection_summary(module, total_nodes: int = 1):
    return module.ProjectionSummary(total_nodes, total_nodes, 0, 0, 0, 0, 0, 1, 0)


def _simple_render_summary(
    module,
    total_nodes: int = 1,
    *,
    includes_depth_minus_one_plane: bool = True,
):
    return module.FlatmapRenderSummary(
        total_nodes,
        total_nodes,
        total_nodes,
        0,
        total_nodes,
        0,
        total_nodes,
        1,
        4,
        1,
        25.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        25.0,
        includes_depth_minus_one_plane,
    )


def _binned_render_result(module):
    projected = pd.DataFrame(
        {
            "file_id": ["a.swc", "a.swc", "b.swc", "c.swc"],
            "render_valid": [True, True, True, True],
            "depth_bin": [0, 0, 1, 1],
            "y_flat_bin": [1, 1, 2, 0],
            "x_flat_bin": [2, 2, 0, 1],
        }
    )
    volume = np.zeros((2, 3, 3), dtype=np.float32)
    volume[0, 1, 2] = 2.0
    volume[1, 2, 0] = 1.0
    volume[1, 0, 1] = 1.0
    return module.FlatmapRenderResult(
        projected_nodes=projected,
        volume=volume,
        points=np.asarray(
            [
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [1.0, 2.0, 0.0],
                [1.0, 0.0, 1.0],
            ]
        ),
        point_file_ids=["a.swc", "a.swc", "b.swc", "c.swc"],
        summary=_simple_render_summary(module, total_nodes=4),
    )


def test_individual_heatmap_color_mode_creates_one_layer_per_file_id(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._heatmap_color_mode_combo = types.SimpleNamespace(
        currentData=lambda: "individual"
    )
    summary = _simple_projection_summary(module, total_nodes=4)

    layer = widget._create_or_update_render_layer(
        _binned_render_result(module),
        summary,
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="heatmap",
    )

    assert layer.name == "Isocortex Flatmap Heatmap: a.swc"
    assert [layer.name for layer in widget._viewer.layers] == [
        "Isocortex Flatmap Heatmap: a.swc",
        "Isocortex Flatmap Heatmap: b.swc",
        "Isocortex Flatmap Heatmap: c.swc",
    ]
    first, second, third = widget._viewer.layers
    assert first.data[0, 1, 2] == 2.0
    assert second.data[1, 2, 0] == 1.0
    assert third.data[1, 0, 1] == 1.0
    assert first.metadata["flatmap_heatmap_color_mode"] == "individual"
    assert first.metadata["flatmap_heatmap_group_label"] == "a.swc"
    assert first.metadata["source_file_ids"] == ["a.swc"]
    assert first.metadata["flatmap_heatmap_group_color"] == [1.0, 0.0, 0.0, 1.0]
    assert second.metadata["flatmap_heatmap_group_color"] == [0.0, 1.0, 0.0, 0.5]


def test_cluster_heatmap_color_mode_creates_cluster_and_unclustered_layers(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._heatmap_color_mode_combo = types.SimpleNamespace(
        currentData=lambda: "cluster"
    )
    widget._cluster_map_provider = lambda: {"a.swc": 2, "b.swc": 1}
    summary = _simple_projection_summary(module, total_nodes=4)

    widget._create_or_update_render_layer(
        _binned_render_result(module),
        summary,
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="heatmap",
    )

    assert [layer.name for layer in widget._viewer.layers] == [
        "Isocortex Flatmap Heatmap: Cluster 1",
        "Isocortex Flatmap Heatmap: Cluster 2",
        "Isocortex Flatmap Heatmap: Unclustered",
    ]
    cluster_1, cluster_2, unclustered = widget._viewer.layers
    assert cluster_1.data[1, 2, 0] == 1.0
    assert cluster_2.data[0, 1, 2] == 2.0
    assert unclustered.data[1, 0, 1] == 1.0
    assert cluster_1.metadata["flatmap_heatmap_group_key"] == 1
    assert cluster_1.metadata["source_file_ids"] == ["b.swc"]
    assert cluster_1.metadata["flatmap_heatmap_group_color"] == [
        0.0,
        1.0,
        0.0,
        0.5,
    ]
    assert unclustered.metadata["flatmap_heatmap_group_key"] is None
    assert unclustered.metadata["flatmap_heatmap_group_color"] == [
        0.5,
        0.5,
        0.5,
        1.0,
    ]


def test_switching_heatmap_color_modes_removes_stale_group_layers(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    summary = _simple_projection_summary(module, total_nodes=4)
    render_result = _binned_render_result(module)
    widget._heatmap_color_mode_combo = types.SimpleNamespace(
        currentData=lambda: "individual"
    )
    widget._create_or_update_render_layer(
        render_result,
        summary,
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="heatmap",
    )
    assert len(widget._viewer.layers) == 3

    widget._heatmap_color_mode_combo = types.SimpleNamespace(
        currentData=lambda: "single"
    )
    single = widget._create_or_update_render_layer(
        render_result,
        summary,
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="heatmap",
    )

    assert widget._viewer.layers == [single]
    assert single.name == "Isocortex Flatmap Heatmap"
    assert single.metadata["flatmap_heatmap_color_mode"] == "single"


def test_deleted_heatmap_layer_is_recreated_from_stale_cache(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    summary = _simple_projection_summary(module)
    first_render = module.FlatmapRenderResult(
        projected_nodes=pd.DataFrame({"file_id": ["a.swc"]}),
        volume=np.ones((1, 4, 4), dtype=np.float32),
        points=np.asarray([[0.0, 0.0, 0.0]]),
        point_file_ids=["a.swc"],
        summary=_simple_render_summary(module),
    )
    second_render = module.FlatmapRenderResult(
        projected_nodes=pd.DataFrame({"file_id": ["a.swc"]}),
        volume=np.full((1, 4, 4), 2.0, dtype=np.float32),
        points=np.asarray([[0.0, 1.0, 1.0]]),
        point_file_ids=["a.swc"],
        summary=_simple_render_summary(module),
    )

    first = widget._create_or_update_render_layer(
        first_render,
        summary,
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="heatmap",
    )
    widget._viewer.layers.remove(first)
    assert widget._projection_layer is first

    second = widget._create_or_update_render_layer(
        second_render,
        summary,
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="heatmap",
    )

    assert second is not first
    assert widget._projection_layer is second
    assert widget._viewer.layers == [second]
    np.testing.assert_array_equal(second.data, second_render.volume)


def test_heatmap_layer_is_recreated_when_display_viewer_changes(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    first_viewer = _DummyViewer()
    active_viewer = first_viewer
    widget._display_viewer_provider = lambda create=True: active_viewer
    summary = _simple_projection_summary(module)
    first_render = module.FlatmapRenderResult(
        projected_nodes=pd.DataFrame({"file_id": ["a.swc"]}),
        volume=np.ones((1, 4, 4), dtype=np.float32),
        points=np.asarray([[0.0, 0.0, 0.0]]),
        point_file_ids=["a.swc"],
        summary=_simple_render_summary(module),
    )
    second_render = module.FlatmapRenderResult(
        projected_nodes=pd.DataFrame({"file_id": ["a.swc"]}),
        volume=np.full((1, 4, 4), 2.0, dtype=np.float32),
        points=np.asarray([[0.0, 1.0, 1.0]]),
        point_file_ids=["a.swc"],
        summary=_simple_render_summary(module),
    )

    first = widget._create_or_update_render_layer(
        first_render,
        summary,
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="heatmap",
    )
    second_viewer = _DummyViewer()
    active_viewer = second_viewer

    second = widget._create_or_update_render_layer(
        second_render,
        summary,
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="heatmap",
    )

    assert second is not first
    assert widget._projection_layer is second
    assert first_viewer.layers == [first]
    assert second_viewer.layers == [second]
    np.testing.assert_array_equal(second.data, second_render.volume)


def test_create_region_labels_layer_adds_and_updates_labels(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    summary = FlatmapRegionLabelsSummary(
        input_voxels=4,
        selected_region_count=1,
        selected_source_voxels=2,
        valid_source_voxels=2,
        labeled_voxels=1,
        collision_voxels=1,
        xy_bins=2,
        depth_bins=1,
        depth_bin_um=25.0,
        x_flat_min=0.0,
        x_flat_max=1.0,
        y_flat_min=0.0,
        y_flat_max=1.0,
        depth_min_um=0.0,
        depth_max_um=25.0,
    )
    first = FlatmapRegionLabelsResult(
        labels=np.asarray([[[0, 7], [0, 0]]], dtype=np.int32),
        summary=summary,
        selected_region_ids=[7],
        represented_region_ids=[7],
    )
    metadata = {
        "projection_kind": "flatmap_region_labels",
        "selected_region_ids": [7],
        "summary": summary.to_dict(),
    }

    layer = widget._create_or_update_region_labels_layer(first, metadata)

    assert layer.name == "Flatmap Region Labels"
    np.testing.assert_array_equal(layer.data, first.labels)
    assert layer.metadata["projection_kind"] == "flatmap_region_labels"
    assert layer._napari_swc_flatmap_region_labels_result is first
    assert widget._viewer.layers == [layer]

    second = FlatmapRegionLabelsResult(
        labels=np.asarray([[[8, 0], [0, 0]]], dtype=np.int32),
        summary=summary,
        selected_region_ids=[8],
        represented_region_ids=[8],
    )
    updated = widget._create_or_update_region_labels_layer(second, metadata)

    assert updated is layer
    np.testing.assert_array_equal(layer.data, second.labels)
    assert layer._napari_swc_flatmap_region_labels_result is second
    assert layer.refresh_count == 1


def test_region_labels_layer_uses_display_viewer_provider(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    main_viewer = widget._viewer
    display_viewer = _DummyViewer()
    widget._display_viewer_provider = lambda create=True: (
        display_viewer if create else display_viewer
    )
    summary = FlatmapRegionLabelsSummary(
        input_voxels=4,
        selected_region_count=1,
        selected_source_voxels=2,
        valid_source_voxels=2,
        labeled_voxels=1,
        collision_voxels=0,
        xy_bins=2,
        depth_bins=1,
        depth_bin_um=25.0,
        x_flat_min=0.0,
        x_flat_max=1.0,
        y_flat_min=0.0,
        y_flat_max=1.0,
        depth_min_um=0.0,
        depth_max_um=25.0,
    )
    result = FlatmapRegionLabelsResult(
        labels=np.asarray([[[0, 7], [0, 0]]], dtype=np.int32),
        summary=summary,
        selected_region_ids=[7],
        represented_region_ids=[7],
    )

    layer = widget._create_or_update_region_labels_layer(
        result,
        {"projection_kind": "flatmap_region_labels"},
    )

    assert main_viewer.layers == []
    assert display_viewer.layers == [layer]
    assert layer.name == "Flatmap Region Labels"


def test_cached_region_labels_do_not_access_nrrd_or_atlas_annotation(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    profile = types.SimpleNamespace(profile_id="profile-1")
    widget._active_cache_profile = profile
    widget._region_cache_dir = Path("cache")
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._selected_region_ids_provider = lambda: [10, 11]
    widget._selected_parent_region_ids_provider = lambda: [10]

    class _AtlasWithoutAnnotation:
        structures = {}

        @property
        def annotation(self):
            raise AssertionError("cached labels must not access atlas.annotation")

    widget._atlas_provider = _AtlasWithoutAnnotation
    result = types.SimpleNamespace(
        labels=np.array([[[10, 0], [0, 11]]], dtype=np.int32),
        profile_id="profile-1",
        selected_region_ids=(10, 11),
        represented_region_ids=(10, 11),
        summary=types.SimpleNamespace(
            labeled_bins=2,
            to_dict=lambda: {"labeled_bins": 2},
        ),
    )
    import napari_swc_viewer.flatmap_region_cache as cache_module

    captured = {}
    monkeypatch.setattr(
        cache_module,
        "materialize_region_selection",
        lambda received_profile, region_ids, **kwargs: (
            captured.update(
                profile=received_profile,
                region_ids=region_ids,
                kwargs=kwargs,
            )
            or result
        ),
    )
    monkeypatch.setattr(
        module,
        "load_flatmap_volume_set",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cached labels must not load NRRDs")
        ),
    )
    widget._create_or_update_region_labels_layer = (
        lambda received, _metadata, **_kwargs: types.SimpleNamespace(
            data=received.labels
        )
    )
    widget._focus_projection_view = lambda *_args: None
    widget._set_region_labels_status = lambda message: captured.update(message=message)

    actual = widget._create_cached_region_labels()

    assert actual is result
    assert captured["profile"] is profile
    assert captured["region_ids"] == [10, 11]
    assert captured["kwargs"]["direct_region_ids"] == [10]
    assert captured["kwargs"]["include_surfaces"] is False
    assert captured["kwargs"]["include_outlines"] is False


def test_cached_region_geometry_uses_only_materialized_cache_arrays(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    profile = types.SimpleNamespace(profile_id="profile-1")
    widget._active_cache_profile = profile
    widget._region_cache_dir = Path("cache")
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._selected_parent_region_ids_provider = lambda: [10]
    widget._region_surfaces_layers = []
    widget._region_outlines_layers = []

    class _AtlasWithoutRuntimeGeometry:
        structures = {
            10: {
                "id": 10,
                "acronym": "VISp",
                "rgb_triplet": [12, 34, 56],
            }
        }

        @property
        def annotation(self):
            raise AssertionError("cached geometry must not access atlas.annotation")

        def mesh_from_structure(self, *_args, **_kwargs):
            raise AssertionError("cached geometry must not project atlas meshes")

    widget._atlas_provider = _AtlasWithoutRuntimeGeometry
    widget._set_region_labels_status = lambda _message: None

    fake_colormaps = types.ModuleType("napari.utils.colormaps")
    fake_colormaps.Colormap = lambda colors: np.asarray(colors)
    monkeypatch.setitem(sys.modules, "napari.utils.colormaps", fake_colormaps)

    import napari_swc_viewer.flatmap_region_cache as cache_module

    surface = types.SimpleNamespace(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        component_count=1,
    )
    outlines = types.SimpleNamespace(
        vectors=np.asarray(
            [[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]],
            dtype=np.float32,
        ).reshape(1, 2, 3)
    )
    monkeypatch.setattr(
        cache_module,
        "materialize_region_surface",
        lambda received_profile, region_id, **kwargs: surface,
    )
    monkeypatch.setattr(
        cache_module,
        "materialize_region_outlines",
        lambda received_profile, region_id, **kwargs: outlines,
    )
    monkeypatch.setattr(
        module,
        "load_flatmap_volume_set",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cached geometry must not load NRRDs")
        ),
    )

    widget._create_region_surfaces()
    widget._create_region_outlines()

    assert [layer.name for layer in widget._viewer.layers] == [
        "Flatmap Region Surfaces",
        "Flatmap Region Outlines",
    ]
    assert widget._viewer.layers[0].metadata["source"] == "precomputed_cache"
    assert widget._viewer.layers[1].metadata["source"] == "precomputed_cache"


def test_heatmap_workaround_swallows_thumbnail_rank_mismatch(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)

    class _CrashLayer(_DummyLayer):
        def _update_thumbnail(self) -> None:
            raise RuntimeError("sequence argument must have length equal to input rank")

    layer = _CrashLayer(
        np.zeros((2, 4, 4), dtype=np.float32),
        name="Isocortex Flatmap Heatmap",
        metadata={"flatmap_heatmap_contrast_limits": (0.0, 5.0)},
        contrast_limits=(0.0, 5.0),
    )

    widget._install_heatmap_layer_workarounds(layer)

    layer._update_thumbnail()
    assert layer._napari_swc_flatmap_thumbnail_warning_logged is True


def test_heatmap_workaround_keeps_stable_limits_during_3d_slice_update(
    monkeypatch,
) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    layer = _DummyLayer(
        np.zeros((2, 4, 4), dtype=np.float32),
        name="Isocortex Flatmap Heatmap",
        metadata={"flatmap_heatmap_contrast_limits": (0.0, 11.0)},
        contrast_limits=(0.0, 11.0),
    )
    layer._keep_auto_contrast = True
    layer._slice_input = types.SimpleNamespace(ndisplay=3)

    widget._install_heatmap_layer_workarounds(layer)
    response = types.SimpleNamespace(
        slice_input=types.SimpleNamespace(ndisplay=3),
        payload={"slice": 1},
    )
    result = layer._update_slice_response(response)

    assert result is response
    assert layer.slice_updates == [response]
    assert layer._keep_auto_contrast is True
    assert layer.contrast_limits == (0.0, 11.0)
    assert layer.contrast_limits_range == (0.0, 11.0)


def test_heatmap_workaround_preserves_2d_auto_contrast(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    layer = _DummyLayer(
        np.zeros((2, 4, 4), dtype=np.float32),
        name="Isocortex Flatmap Heatmap",
        metadata={"flatmap_heatmap_contrast_limits": (0.0, 11.0)},
        contrast_limits=(0.0, 11.0),
    )
    layer._keep_auto_contrast = True
    layer._slice_input = types.SimpleNamespace(ndisplay=2)

    widget._install_heatmap_layer_workarounds(layer)
    response = types.SimpleNamespace(
        slice_input=types.SimpleNamespace(ndisplay=2),
        payload={"slice": 2},
    )
    result = layer._update_slice_response(response)

    assert result is response
    assert layer.slice_updates == [response]
    assert layer.contrast_limits == (4.0, 5.0)
    assert layer.contrast_limits_range == (4.0, 5.0)


def test_heatmap_status_guard_handles_stale_2d_slice(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    layer = _DummyLayer(np.zeros((2, 4, 4)), name="Isocortex Flatmap Heatmap")
    layer._slice = types.SimpleNamespace(
        image=types.SimpleNamespace(raw=np.zeros((4, 4), dtype=np.float32))
    )
    layer.raise_status_error = True

    widget._install_heatmap_status_guard(layer)

    status = layer.get_status(
        np.asarray([1.2, 2.5, 3.6]),
        view_direction=np.asarray([1.0, 0.0, 0.0]),
        dims_displayed=[0, 1, 2],
        world=True,
    )

    assert status["coords"] == " [1 2 4]"
    assert status["coordinates"] == " [1 2 4]: "
    assert status["value"] == ""


def test_heatmap_status_guard_reraises_unrelated_index_errors(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    layer = _DummyLayer(np.zeros((2, 4, 4)), name="Isocortex Flatmap Heatmap")
    layer._slice = types.SimpleNamespace(
        image=types.SimpleNamespace(raw=np.zeros((2, 4, 4), dtype=np.float32))
    )
    layer.raise_status_error = True
    layer.status_error_message = "index 99 is out of bounds for axis 0"

    widget._install_heatmap_status_guard(layer)

    with np.testing.assert_raises(IndexError):
        layer.get_status(
            np.asarray([1.2, 2.5, 3.6]),
            view_direction=np.asarray([1.0, 0.0, 0.0]),
            dims_displayed=[0, 1, 2],
            world=True,
        )


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


def test_deleted_points_layer_is_recreated_from_stale_cache(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    summary = _simple_projection_summary(module)
    first_render = module.FlatmapRenderResult(
        projected_nodes=pd.DataFrame({"file_id": ["a.swc"]}),
        volume=np.zeros((1, 4, 4), dtype=np.float32),
        points=np.asarray([[0.0, 0.0, 0.0]]),
        point_file_ids=["a.swc"],
        summary=_simple_render_summary(module),
    )
    second_render = module.FlatmapRenderResult(
        projected_nodes=pd.DataFrame({"file_id": ["b.swc"]}),
        volume=np.zeros((1, 4, 4), dtype=np.float32),
        points=np.asarray([[0.0, 1.0, 1.0]]),
        point_file_ids=["b.swc"],
        summary=_simple_render_summary(module),
    )

    first = widget._create_or_update_render_layer(
        first_render,
        summary,
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="points",
    )
    widget._viewer.layers.remove(first)
    assert widget._projection_layer is first

    second = widget._create_or_update_render_layer(
        second_render,
        summary,
        flatmap_style="flatmap_both_shaped.nrrd",
        coordinate_mode="microns",
        render_mode="points",
    )

    assert second is not first
    assert widget._projection_layer is second
    assert widget._viewer.layers == [second]
    np.testing.assert_array_equal(second.data, second_render.points)


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
            "flatmap_lookup_mode": ["mirrored_depth"],
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
    assert exported["flatmap_lookup_mode"].tolist() == ["mirrored_depth"]
    assert "render_valid" in exported.columns
    assert exported["x_flat_bin"].tolist() == [10]
    assert "Exported flatmap projection" in widget._status_label.text


def test_precomputed_heatmap_uses_duckdb_fast_path(monkeypatch) -> None:
    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._projection_source_combo = types.SimpleNamespace(
        currentData=lambda: module._PROJECTION_SOURCE_PRECOMPUTED
    )
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_HEATMAP
    )
    called: list[bool] = []
    widget._start_precomputed_heatmap_worker = lambda: called.append(True)

    def _fail_query(_file_ids):
        raise AssertionError("fast path must not load nodes into pandas")

    widget._query_nodes = _fail_query
    widget._apply_projection_result = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("fast path must not use the synchronous apply")
    )

    widget._project()

    assert called == [True]


def test_apply_precomputed_heatmap_result_single_disables_per_node(monkeypatch) -> None:
    from napari_swc_viewer import flatmap_heatmap as fh

    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._summary_label = _DummyLabel()
    widget._export_btn = _DummyButton()
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_HEATMAP
    )
    widget._heatmap_color_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._HEATMAP_COLOR_SINGLE
    )
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._active_cache_profile = None
    widget._region_cache_dir = None
    widget._cache_profile_id = lambda _profile: None
    widget._precomputed_heatmap_file_ids = ["a.swc", "b.swc"]
    notified: list[bool] = []
    widget._notify_flatmap_correlation_source_changed = lambda: notified.append(True)

    volume = np.zeros((2, 4, 4), dtype=np.float32)
    volume[0, 1, 2] = 3.0
    stats = fh.FlatmapAggregateStats(
        total_nodes=10,
        total_traces=2,
        flatmap_valid_nodes=8,
        depth_valid_nodes=6,
        depth_minus_one_nodes=0,
        rendered_nodes=6,
        traces_represented=2,
    )
    result = fh.FlatmapHeatmapVolumeResult(
        color_mode=module._HEATMAP_COLOR_SINGLE,
        volume=volume,
        grouped_volumes=(),
        render_summary=_simple_render_summary(module),
        stats=stats,
        volume_shape=(2, 4, 4),
    )

    widget._apply_precomputed_heatmap_result(result)

    assert widget._last_render_mode == module._RENDER_HEATMAP
    assert widget._last_projected_nodes is None
    assert widget._last_projection_source == module._PROJECTION_SOURCE_PRECOMPUTED
    assert widget._last_coordinate_mode == "parquet_columns"
    assert widget._export_btn.enabled is False
    assert notified == [True]
    layer = widget._projection_layer
    assert layer is not None
    assert layer.name == module._HEATMAP_LAYER_NAME
    assert layer.metadata["flatmap_render_mode"] == "heatmap"
    # Per-node correlation source is unavailable for a volume-only fast render.
    assert widget.latest_flatmap_correlation_source() is None


def test_apply_precomputed_heatmap_result_grouped_creates_layer_per_group(
    monkeypatch,
) -> None:
    from napari_swc_viewer import flatmap_heatmap as fh

    module = _load_flatmap_widget_module(monkeypatch)
    widget = _widget(module)
    widget._summary_label = _DummyLabel()
    widget._export_btn = _DummyButton()
    widget._render_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._RENDER_HEATMAP
    )
    widget._heatmap_color_mode_combo = types.SimpleNamespace(
        currentData=lambda: module._HEATMAP_COLOR_INDIVIDUAL
    )
    widget._style_combo = types.SimpleNamespace(currentData=lambda: "both_shaped")
    widget._active_cache_profile = None
    widget._region_cache_dir = None
    widget._cache_profile_id = lambda _profile: None
    widget._precomputed_heatmap_file_ids = ["a.swc", "b.swc"]
    widget._notify_flatmap_correlation_source_changed = lambda: None

    groups = []
    for name in ("a.swc", "b.swc"):
        volume = np.zeros((2, 4, 4), dtype=np.float32)
        volume[0, 1, 2] = 1.0
        groups.append(
            fh.FlatmapGroupedVolume(
                group_key=name,
                label=name,
                source_file_ids=(name,),
                volume=volume,
                rendered_nodes=1,
                nonzero_voxels=1,
            )
        )
    stats = fh.FlatmapAggregateStats(4, 2, 4, 4, 0, 2, 2)
    result = fh.FlatmapHeatmapVolumeResult(
        color_mode=module._HEATMAP_COLOR_INDIVIDUAL,
        volume=None,
        grouped_volumes=tuple(groups),
        render_summary=_simple_render_summary(module, total_nodes=2),
        stats=stats,
        volume_shape=(2, 4, 4),
    )

    widget._apply_precomputed_heatmap_result(result)

    grouped_layers = [
        layer
        for layer in widget._viewer.layers
        if str(getattr(layer, "name", "")).startswith(
            module._GROUPED_HEATMAP_LAYER_PREFIX
        )
    ]
    assert len(grouped_layers) == 2
    assert widget._export_btn.enabled is False
    assert widget._last_projected_nodes is None
