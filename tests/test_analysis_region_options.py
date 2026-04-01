"""Tests for Analysis tab hierarchical region selection helpers."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


class _BoundSignal:
    """Minimal signal object used by widget stubs."""

    def __init__(self) -> None:
        self._callbacks: list = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, *args) -> None:
        for callback in list(self._callbacks):
            callback(*args)


class _Signal:
    """Descriptor stand-in for ``qtpy.QtCore.Signal``."""

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


class _DummyWidget:
    """Very small QWidget-like stand-in."""

    def __init__(self, *_args, **_kwargs) -> None:
        self._enabled = True
        self._visible = True

    def setEnabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def isEnabled(self) -> bool:
        return self._enabled

    def setVisible(self, visible: bool) -> None:
        self._visible = bool(visible)


class _DummyLayout:
    """Simple layout stub that records added children."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.children: list[object] = []

    def addWidget(self, widget) -> None:
        self.children.append(widget)

    def addLayout(self, layout) -> None:
        self.children.append(layout)

    def addStretch(self, *_args) -> None:
        return None

    def setContentsMargins(self, *_args) -> None:
        return None


class _DummyLabel(_DummyWidget):
    """Small QLabel stand-in."""

    def __init__(self, text: str = "", *_args, **_kwargs) -> None:
        super().__init__()
        self._text = text

    def setText(self, text: str) -> None:
        self._text = text

    def text(self) -> str:
        return self._text


class _DummyButton(_DummyWidget):
    """Small QPushButton stand-in."""

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.clicked = _BoundSignal()


class _DummyCombo(_DummyWidget):
    """Small QComboBox stand-in."""

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.currentTextChanged = _BoundSignal()
        self.currentIndexChanged = _BoundSignal()
        self._items: list[dict[str, object]] = []
        self._current_index = -1
        self._blocked = False
        self._editable = False
        self._edit_text = ""

    def addItems(self, items: list[str]) -> None:
        for item in items:
            self.addItem(item)

    def addItem(self, *args) -> None:
        if len(args) == 1:
            text = str(args[0])
        else:
            text = str(args[1])
        self._items.append({"text": text, "data": None})
        if self._current_index < 0:
            self._current_index = 0

    def clear(self) -> None:
        self._items = []
        self._current_index = -1
        self._edit_text = ""

    def setEditable(self, editable: bool) -> None:
        self._editable = bool(editable)

    def currentText(self) -> str:
        if 0 <= self._current_index < len(self._items):
            return str(self._items[self._current_index]["text"])
        return self._edit_text

    def setCurrentText(self, text: str) -> None:
        for index, item in enumerate(self._items):
            if item["text"] == text:
                self._current_index = index
                break
        else:
            self._edit_text = text
        if not self._blocked:
            self.currentTextChanged.emit(text)

    def setEditText(self, text: str) -> None:
        self._edit_text = text

    def blockSignals(self, blocked: bool) -> None:
        self._blocked = bool(blocked)

    def currentIndex(self) -> int:
        return self._current_index

    def setCurrentIndex(self, index: int) -> None:
        self._current_index = int(index)
        if not self._blocked:
            self.currentIndexChanged.emit(index)
            self.currentTextChanged.emit(self.currentText())

    def itemData(self, index: int):
        return self._items[index]["data"]

    def setItemData(self, index: int, data) -> None:
        self._items[index]["data"] = data

    def count(self) -> int:
        return len(self._items)


class _DummySpinBox(_DummyWidget):
    """Small spinbox stand-in."""

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.valueChanged = _BoundSignal()
        self._value = 0

    def setRange(self, *_args) -> None:
        return None

    def setValue(self, value) -> None:
        self._value = value

    def value(self):
        return self._value

    def setSuffix(self, *_args) -> None:
        return None

    def setDecimals(self, *_args) -> None:
        return None

    def setToolTip(self, *_args) -> None:
        return None


class _DummyProgressBar(_DummyWidget):
    """Small progress bar stand-in."""

    def setRange(self, *_args) -> None:
        return None

    def setValue(self, *_args) -> None:
        return None


class _DummyFigure:
    """Minimal figure stand-in."""

    def __init__(self, *_args, **_kwargs) -> None:
        return None


class _DummyCanvas(_DummyWidget):
    """Minimal canvas stand-in."""

    def __init__(self, figure, *_args, **_kwargs) -> None:
        super().__init__()
        self.figure = figure

    def setMinimumHeight(self, *_args) -> None:
        return None

    def draw(self) -> None:
        return None


class _DummyColormap:
    """Minimal napari Colormap stand-in."""

    def __init__(self, colors, name: str) -> None:
        self.colors = colors
        self.name = name


class _DummyCollapsibleSection(_DummyWidget):
    """Simple collapsible section stub that records constructor args."""

    instances: list["_DummyCollapsibleSection"] = []

    def __init__(self, title: str, *, expanded: bool = True, parent=None) -> None:
        super().__init__(parent)
        self.title = title
        self.expanded = expanded
        self._layout = _DummyLayout()
        self.__class__.instances.append(self)

    def content_layout(self) -> _DummyLayout:
        return self._layout


class _DummyRegionSelector(_DummyWidget):
    """Simple region selector stub used for Analysis tab tests."""

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.selection_changed = _BoundSignal()
        self.allowed_ids: set[int] | None = None
        self.atlas = None
        self.selected: tuple[int, str] | None = None
        self._structure_map: dict[int, dict] = {}
        self.clear_calls = 0

    def get_single_selected_region(self) -> tuple[int, str] | None:
        return self.selected

    def set_allowed_structure_ids(self, structure_ids: set[int] | None) -> None:
        self.allowed_ids = None if structure_ids is None else set(structure_ids)

    def set_atlas(self, atlas) -> None:
        self.atlas = atlas
        self._structure_map = getattr(atlas, "structures", {})

    def select_region_by_id(self, region_id: int | None) -> None:
        if region_id is None:
            self.selected = None
            return
        struct = self._structure_map.get(region_id, {})
        acronym = str(struct.get("acronym", f"R{region_id}"))
        self.selected = (int(region_id), acronym)

    def clear(self) -> None:
        self.clear_calls += 1
        self.selected = None
        self._structure_map = {}


class _DummyQColor:
    """Minimal QColor stand-in."""

    @staticmethod
    def fromRgbF(*_args):
        return None


class _DummyQPixmap:
    """Minimal QPixmap stand-in."""

    def __init__(self, *_args, **_kwargs) -> None:
        return None

    def fill(self, *_args) -> None:
        return None


class _DummyQIcon:
    """Minimal QIcon stand-in."""

    def __init__(self, *_args, **_kwargs) -> None:
        return None


class _DummyThread:
    """Minimal QThread stand-in."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.started = _BoundSignal()
        self.finished = _BoundSignal()

    def isRunning(self) -> bool:
        return False

    def start(self) -> None:
        return None

    def quit(self) -> None:
        return None

    def deleteLater(self) -> None:
        return None


class _DummyOrderSignal:
    """Dummy napari dims order event."""

    def connect(self, *_args, **_kwargs) -> None:
        return None


class _DummyViewer:
    """Very small viewer stand-in."""

    def __init__(self) -> None:
        self.dims = types.SimpleNamespace(
            not_displayed=(0,),
            events=types.SimpleNamespace(order=_DummyOrderSignal()),
        )
        self.layers = []
        self.last_image_kwargs: dict[str, object] | None = None

    def add_image(self, data, **kwargs):
        layer = _DummyImageLayer(data, **kwargs)
        self.layers.append(layer)
        self.last_image_kwargs = dict(kwargs)
        return layer


class _DummyImageLayer:
    """Minimal image layer stand-in for analysis heatmap tests."""

    def __init__(self, data, **kwargs) -> None:
        self.data = np.asarray(data)
        self.name = str(kwargs["name"])
        self.metadata = dict(kwargs.get("metadata", {}))
        self.contrast_limits = tuple(kwargs.get("contrast_limits", (0.0, 1.0)))
        self.contrast_limits_range = tuple(
            kwargs.get("contrast_limits_range", self.contrast_limits)
        )
        self.rendering = kwargs.get("rendering")
        self.blending = kwargs.get("blending")
        self.opacity = kwargs.get("opacity")
        self.visible = kwargs.get("visible", True)
        self.scale = list(kwargs.get("scale", []))
        self._keep_auto_contrast = False
        self._slice_input = types.SimpleNamespace(ndisplay=2)
        self.thumbnail_updates = 0
        self.slice_updates: list[object] = []

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


class _FakeAtlas:
    """Atlas-like stand-in for ancestry calculations."""

    def __init__(self, structures: dict[int, dict]) -> None:
        self.structures = structures
        self.resolution = (25.0, 25.0, 25.0)
        self.annotation = types.SimpleNamespace(shape=(4, 4, 4))


def _import_analysis_tab_module():
    """Import ``analysis_tab.py`` with stubbed UI dependencies."""
    backend_module = types.ModuleType("matplotlib.backends.backend_qtagg")
    backend_module.FigureCanvasQTAgg = _DummyCanvas

    figure_module = types.ModuleType("matplotlib.figure")
    figure_module.Figure = _DummyFigure

    napari_module = types.ModuleType("napari")
    napari_module.__path__ = []
    napari_utils_module = types.ModuleType("napari.utils")
    napari_utils_module.__path__ = []
    napari_colormaps_module = types.ModuleType("napari.utils.colormaps")
    napari_colormaps_module.Colormap = _DummyColormap
    napari_module.utils = napari_utils_module
    napari_utils_module.colormaps = napari_colormaps_module

    qtcore_module = types.ModuleType("qtpy.QtCore")
    qtcore_module.QThread = _DummyThread
    qtcore_module.Signal = _Signal

    qtgui_module = types.ModuleType("qtpy.QtGui")
    qtgui_module.QColor = _DummyQColor
    qtgui_module.QIcon = _DummyQIcon
    qtgui_module.QPixmap = _DummyQPixmap

    qtwidgets_module = types.ModuleType("qtpy.QtWidgets")
    for name, value in {
        "QComboBox": _DummyCombo,
        "QDoubleSpinBox": _DummySpinBox,
        "QGroupBox": _DummyWidget,
        "QHBoxLayout": _DummyLayout,
        "QLabel": _DummyLabel,
        "QProgressBar": _DummyProgressBar,
        "QPushButton": _DummyButton,
        "QSpinBox": _DummySpinBox,
        "QVBoxLayout": _DummyLayout,
        "QWidget": _DummyWidget,
    }.items():
        setattr(qtwidgets_module, name, value)

    repo_root = Path(__file__).resolve().parent.parent
    package_root = repo_root / "src" / "napari_swc_viewer"
    widgets_root = package_root / "widgets"

    napari_package = types.ModuleType("napari_swc_viewer")
    napari_package.__path__ = [str(package_root)]

    widgets_package = types.ModuleType("napari_swc_viewer.widgets")
    widgets_package.__path__ = [str(widgets_root)]

    collapsible_module = types.ModuleType(
        "napari_swc_viewer.widgets.collapsible_section"
    )
    collapsible_module.CollapsibleSection = _DummyCollapsibleSection

    region_selector_module = types.ModuleType(
        "napari_swc_viewer.widgets.region_selector"
    )
    region_selector_module.RegionSelectorWidget = _DummyRegionSelector

    replacements = {
        "matplotlib.pyplot": types.ModuleType("matplotlib.pyplot"),
        "seaborn": types.ModuleType("seaborn"),
        "matplotlib.backends.backend_qtagg": backend_module,
        "matplotlib.figure": figure_module,
        "napari": napari_module,
        "napari.utils": napari_utils_module,
        "napari.utils.colormaps": napari_colormaps_module,
        "qtpy.QtCore": qtcore_module,
        "qtpy.QtGui": qtgui_module,
        "qtpy.QtWidgets": qtwidgets_module,
        "napari_swc_viewer": napari_package,
        "napari_swc_viewer.widgets": widgets_package,
        "napari_swc_viewer.widgets.collapsible_section": collapsible_module,
        "napari_swc_viewer.widgets.region_selector": region_selector_module,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    module_name = "napari_swc_viewer.widgets.analysis_tab"
    previous_module = sys.modules.get(module_name)

    try:
        sys.modules.update(replacements)
        sys.modules.pop(module_name, None)

        module_path = widgets_root / "analysis_tab.py"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        assert spec is not None
        assert spec.loader is not None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module

        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def _make_selector(
    selected: tuple[int, str] | None = None,
    structure_map: dict[int, dict] | None = None,
):
    selector = _DummyRegionSelector()
    selector.selected = selected
    selector._structure_map = structure_map or {}
    return selector


def _install_fake_napari_colormaps():
    """Install a minimal ``napari.utils.colormaps`` module for one test."""
    napari_module = types.ModuleType("napari")
    napari_module.__path__ = []
    napari_utils_module = types.ModuleType("napari.utils")
    napari_utils_module.__path__ = []
    napari_colormaps_module = types.ModuleType("napari.utils.colormaps")
    napari_colormaps_module.Colormap = _DummyColormap
    napari_module.utils = napari_utils_module
    napari_utils_module.colormaps = napari_colormaps_module

    replacements = {
        "napari": napari_module,
        "napari.utils": napari_utils_module,
        "napari.utils.colormaps": napari_colormaps_module,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    return previous


def _restore_modules(previous: dict[str, object | None]) -> None:
    """Restore modules temporarily replaced during a test."""
    for name, original in previous.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


def test_analysis_region_sections_are_collapsed_by_default():
    """Analysis tab should expose collapsible top-level sections and compact region trees."""
    _DummyCollapsibleSection.instances = []
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget

    AnalysisTabWidget(_DummyViewer())

    titles = [section.title for section in _DummyCollapsibleSection.instances]
    expanded = [section.expanded for section in _DummyCollapsibleSection.instances]

    assert titles == [
        "Clustering",
        "Select Target Region",
        "Node Count Heatmap",
        "Select Heatmap Region",
        "Progress",
        "Clustermap",
    ]
    assert expanded == [True, False, True, False, True, False]


def test_analysis_allowed_structure_ids_include_dataset_regions_and_ancestors():
    """Dataset-backed Analysis trees should expose represented leaves plus ancestors."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._atlas = _FakeAtlas(
        {
            997: {"id": 997, "acronym": "root", "name": "root", "structure_id_path": [997]},
            315: {"id": 315, "acronym": "ISO", "name": "Isocortex", "structure_id_path": [997, 315]},
            184: {"id": 184, "acronym": "FRP", "name": "Frontal pole", "structure_id_path": [997, 315, 184]},
            68: {"id": 68, "acronym": "FRP1", "name": "FRP1", "structure_id_path": [997, 315, 184, 68]},
        }
    )
    widget._dataset_region_ids = {68}

    assert widget._analysis_allowed_structure_ids() == {68, 184, 315, 997}


def test_refresh_analysis_region_selectors_waits_for_atlas_and_dataset():
    """Selectors should stay empty until both atlas and represented dataset IDs exist."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._cluster_region_selector = _make_selector()
    widget._heat_region_selector = _make_selector()
    widget._cluster_region_summary_label = _DummyLabel()
    widget._heat_region_summary_label = _DummyLabel()
    widget._atlas = None
    widget._dataset_region_ids = {68}

    widget._refresh_analysis_region_selectors()

    assert widget._cluster_region_selector.clear_calls == 1
    assert widget._heat_region_selector.clear_calls == 1
    assert widget._cluster_region_summary_label.text() == "None selected"
    assert widget._heat_region_summary_label.text() == "All regions"


def test_refresh_analysis_region_selectors_populates_both_selectors_from_dataset():
    """Both Analysis selectors should receive the same dataset-backed visible tree IDs."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._cluster_region_selector = _make_selector()
    widget._heat_region_selector = _make_selector()
    widget._cluster_region_summary_label = _DummyLabel()
    widget._heat_region_summary_label = _DummyLabel()
    widget._atlas = _FakeAtlas(
        {
            997: {"id": 997, "acronym": "root", "name": "root", "structure_id_path": [997]},
            315: {"id": 315, "acronym": "ISO", "name": "Isocortex", "structure_id_path": [997, 315]},
            184: {"id": 184, "acronym": "FRP", "name": "Frontal pole", "structure_id_path": [997, 315, 184]},
            68: {"id": 68, "acronym": "FRP1", "name": "FRP1", "structure_id_path": [997, 315, 184, 68]},
        }
    )
    widget._dataset_region_ids = {68}

    widget._refresh_analysis_region_selectors()

    expected = {68, 184, 315, 997}
    assert widget._cluster_region_selector.allowed_ids == expected
    assert widget._heat_region_selector.allowed_ids == expected
    assert widget._cluster_region_selector.atlas is widget._atlas
    assert widget._heat_region_selector.atlas is widget._atlas


def test_refresh_analysis_region_selectors_preserves_independent_selections():
    """Clustering and heatmap selectors should keep separate valid selections."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    cluster_selector = _make_selector(
        selected=(184, "FRP"),
        structure_map={184: {"acronym": "FRP"}},
    )
    heat_selector = _make_selector(
        selected=(500, "CP"),
        structure_map={500: {"acronym": "CP"}},
    )

    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._cluster_region_selector = cluster_selector
    widget._heat_region_selector = heat_selector
    widget._cluster_region_summary_label = _DummyLabel()
    widget._heat_region_summary_label = _DummyLabel()
    widget._atlas = _FakeAtlas(
        {
            997: {"id": 997, "acronym": "root", "name": "root", "structure_id_path": [997]},
            184: {"id": 184, "acronym": "FRP", "name": "Frontal pole", "structure_id_path": [997, 184]},
            500: {"id": 500, "acronym": "CP", "name": "Caudoputamen", "structure_id_path": [997, 500]},
        }
    )
    widget._dataset_region_ids = {184, 500}

    widget._refresh_analysis_region_selectors()

    assert widget._selected_cluster_region() == (184, "FRP")
    assert widget._selected_heat_region() == (500, "CP")


def test_represented_region_ids_for_selection_expands_parent_to_dataset_descendants():
    """Parent selection for heatmaps should expand to represented descendant region IDs."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._atlas = _FakeAtlas(
        {
            997: {"id": 997, "acronym": "root", "name": "root", "structure_id_path": [997]},
            184: {"id": 184, "acronym": "FRP", "name": "Frontal pole", "structure_id_path": [997, 184]},
            68: {"id": 68, "acronym": "FRP1", "name": "FRP1", "structure_id_path": [997, 184, 68]},
            667: {"id": 667, "acronym": "FRP2/3", "name": "FRP2/3", "structure_id_path": [997, 184, 667]},
            500: {"id": 500, "acronym": "CP", "name": "Caudoputamen", "structure_id_path": [997, 500]},
        }
    )
    widget._dataset_region_ids = {68, 667, 500}

    assert widget._represented_region_ids_for_selection(184) == [68, 667]
    assert widget._represented_region_ids_for_selection(500) == [500]


def test_update_region_summary_labels_uses_selected_region_and_blank_heatmap():
    """Summary labels should show the selected clustering region and blank heatmap state."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._cluster_region_selector = _make_selector(
        selected=(184, "FRP"),
        structure_map={184: {"name": "Frontal pole", "acronym": "FRP"}},
    )
    widget._heat_region_selector = _make_selector(selected=None)
    widget._cluster_region_summary_label = _DummyLabel()
    widget._heat_region_summary_label = _DummyLabel()

    widget._update_region_summary_labels()

    assert widget._cluster_region_summary_label.text() == "FRP (Frontal pole)"
    assert widget._heat_region_summary_label.text() == "All regions"


def test_on_heatmap_finished_adds_stable_analysis_contrast_limits():
    """Analysis heatmaps should be added with explicit full-volume contrast limits."""
    module = _import_analysis_tab_module()
    AnalysisTabWidget = module.AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._viewer = _DummyViewer()
    widget._progress_bar = _DummyProgressBar()
    widget._progress_label = _DummyLabel()
    widget._update_button_states = lambda: None
    widget._atlas = types.SimpleNamespace(atlas_name="fake_atlas")
    widget._pending_heatmap_cluster = 1
    widget._pending_heatmap_region = "CH"
    widget._pending_heatmap_depth_bin = 3
    widget._pending_heatmap_depth_axis = 1
    widget._cluster_label_colors = {1: [0.1, 0.2, 0.3, 1.0]}

    volume = np.zeros((2, 3, 4), dtype=np.float32)
    volume[1, 2, 3] = 7.0

    previous = _install_fake_napari_colormaps()
    try:
        widget._on_heatmap_finished(volume)
    finally:
        _restore_modules(previous)

    layer = widget._heatmap_layer
    assert layer.name == "Cluster 1 CH Heatmap"
    assert layer.contrast_limits == (0.0, 7.0)
    assert layer.contrast_limits_range == (0.0, 7.0)
    assert layer.scale == [1.0, 3.0, 1.0]
    assert layer.metadata["heatmap_kind"] == "analysis"
    assert layer.metadata["heatmap_contrast_limits"] == (0.0, 7.0)
    assert layer.metadata["heatmap_autocontrast_policy"] == "stable_full_volume"


def test_analysis_heatmap_workaround_swallows_thumbnail_rank_mismatch():
    """Known napari thumbnail rank errors should be suppressed for analysis heatmaps."""
    module = _import_analysis_tab_module()

    class _CrashLayer(_DummyImageLayer):
        def _update_thumbnail(self) -> None:
            raise RuntimeError("sequence argument must have length equal to input rank")

    layer = _CrashLayer(
        np.zeros((2, 2, 2), dtype=np.float32),
        name="Cluster 1 CH Heatmap",
        metadata={
            "heatmap_kind": "analysis",
            "heatmap_contrast_limits": (0.0, 5.0),
        },
        contrast_limits=(0.0, 5.0),
        contrast_limits_range=(0.0, 5.0),
    )

    module._install_analysis_heatmap_layer_workarounds(layer)

    layer._update_thumbnail()
    assert layer._analysis_heatmap_thumbnail_warning_logged is True


def test_analysis_heatmap_workaround_resets_to_stored_limits():
    """Reset hooks should restore stored full-volume contrast limits."""
    module = _import_analysis_tab_module()
    layer = _DummyImageLayer(
        np.zeros((2, 2, 2), dtype=np.float32),
        name="Cluster 1 CH Heatmap",
        metadata={
            "heatmap_kind": "analysis",
            "heatmap_contrast_limits": (0.0, 9.0),
        },
        contrast_limits=(1.0, 2.0),
        contrast_limits_range=(1.0, 2.0),
    )
    layer._slice_input = types.SimpleNamespace(ndisplay=3)

    module._install_analysis_heatmap_layer_workarounds(layer)
    layer.reset_contrast_limits()

    assert layer.contrast_limits == (0.0, 9.0)
    assert layer.contrast_limits_range == (0.0, 9.0)

    layer.contrast_limits_range = (3.0, 4.0)
    layer.reset_contrast_limits_range()

    assert layer.contrast_limits_range == (0.0, 9.0)


def test_analysis_heatmap_workaround_keeps_stable_limits_during_slice_updates():
    """Continuous auto-contrast should not replace stored full-volume heatmap limits."""
    module = _import_analysis_tab_module()
    layer = _DummyImageLayer(
        np.zeros((2, 2, 2), dtype=np.float32),
        name="Cluster 1 CH Heatmap",
        metadata={
            "heatmap_kind": "analysis",
            "heatmap_contrast_limits": (0.0, 11.0),
        },
        contrast_limits=(0.0, 11.0),
        contrast_limits_range=(0.0, 11.0),
    )
    layer._keep_auto_contrast = True
    layer._slice_input = types.SimpleNamespace(ndisplay=3)

    module._install_analysis_heatmap_layer_workarounds(layer)
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


def test_analysis_heatmap_workaround_preserves_2d_auto_contrast_behavior():
    """2D continuous auto-contrast should still use the original napari path."""
    module = _import_analysis_tab_module()
    layer = _DummyImageLayer(
        np.zeros((2, 2, 2), dtype=np.float32),
        name="Cluster 1 CH Heatmap",
        metadata={
            "heatmap_kind": "analysis",
            "heatmap_contrast_limits": (0.0, 11.0),
        },
        contrast_limits=(0.0, 11.0),
        contrast_limits_range=(0.0, 11.0),
    )
    layer._keep_auto_contrast = True
    layer._slice_input = types.SimpleNamespace(ndisplay=2)

    module._install_analysis_heatmap_layer_workarounds(layer)
    response = types.SimpleNamespace(
        slice_input=types.SimpleNamespace(ndisplay=2),
        payload={"slice": 2},
    )
    result = layer._update_slice_response(response)

    assert result is response
    assert layer.slice_updates == [response]
    assert layer.contrast_limits == (4.0, 5.0)
    assert layer.contrast_limits_range == (4.0, 5.0)

    layer.reset_contrast_limits()
    assert layer.contrast_limits == (-1.0, -1.0)


def test_analysis_heatmap_contrast_limits_fallback_for_zero_volume():
    """All-zero analysis heatmaps should keep the default visible contrast range."""
    module = _import_analysis_tab_module()

    limits = module._analysis_heatmap_contrast_limits(
        np.zeros((3, 3, 3), dtype=np.float32)
    )

    assert limits == (0.0, 1.0)
