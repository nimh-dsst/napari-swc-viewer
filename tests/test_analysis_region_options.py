"""Tests for Analysis tab hierarchical region selection helpers."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from napari_swc_viewer.analysis.clustering import ClusterResult


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
        self.geometry_updates = 0

    def setEnabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def isEnabled(self) -> bool:
        return self._enabled

    def setVisible(self, visible: bool) -> None:
        self._visible = bool(visible)

    def updateGeometry(self) -> None:
        self.geometry_updates += 1


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

    def __init__(self, text: str = "", *_args, **_kwargs) -> None:
        super().__init__()
        self._text = text
        self.clicked = _BoundSignal()


class _DummyLineEdit(_DummyWidget):
    """Small QLineEdit stand-in."""

    def __init__(self, text: str = "", *_args, **_kwargs) -> None:
        super().__init__()
        self._text = text

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:
        self._text = str(text)


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


class _DummyScrollArea(_DummyWidget):
    """Small scroll-area stand-in."""

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.widget_resizable = False
        self.widget = None

    def setWidgetResizable(self, resizable: bool) -> None:
        self.widget_resizable = bool(resizable)

    def setWidget(self, widget) -> None:
        self.widget = widget


class _DummyFigure:
    """Minimal figure stand-in."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.cleared = 0
        self.canvas = None
        self.dpi = 100.0
        self.size_inches: tuple[float, float] | None = (6.0, 6.0)
        self.set_size_inches_calls = 0

    def clear(self) -> None:
        self.cleared += 1

    def add_subplot(self, *_args, **_kwargs):
        return types.SimpleNamespace(text=lambda *a, **k: None)

    def set_canvas(self, canvas) -> None:
        self.canvas = canvas

    def get_dpi(self) -> float:
        return self.dpi

    def get_size_inches(self):
        return self.size_inches

    def set_size_inches(
        self, width: float, height: float, forward: bool = True
    ) -> None:
        self.set_size_inches_calls += 1
        self.size_inches = (float(width), float(height), bool(forward))


class _DummyCanvas(_DummyWidget):
    """Minimal canvas stand-in."""

    def __init__(self, figure, *_args, **_kwargs) -> None:
        super().__init__()
        self.figure = figure
        self._width = 600
        self._height = 400
        self.device_pixel_ratio = 1.0
        self._physical_width: int | None = None
        self._physical_height: int | None = None

    def setMinimumHeight(self, *_args) -> None:
        return None

    def get_width_height(self, *, physical: bool = False):
        if physical:
            if self._physical_width is not None and self._physical_height is not None:
                return (self._physical_width, self._physical_height)
            return (
                int(round(self._width * float(self.device_pixel_ratio))),
                int(round(self._height * float(self.device_pixel_ratio))),
            )
        return (self._width, self._height)

    def draw(self) -> None:
        return None


class _DummyScrollArea(_DummyWidget):
    """Minimal QScrollArea stand-in."""

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.widget_resizable = False
        self.widget = None

    def setWidgetResizable(self, enabled: bool) -> None:
        self.widget_resizable = bool(enabled)

    def setWidget(self, widget) -> None:
        self.widget = widget


class _DummyColormap:
    """Minimal napari Colormap stand-in."""

    def __init__(self, colors, name: str) -> None:
        self.colors = colors
        self.name = name


class _DummyCollapsibleSection(_DummyWidget):
    """Simple collapsible section stub that records constructor args."""

    instances: list["_DummyCollapsibleSection"] = []

    def __init__(
        self, title: str, *, expanded: bool = True, parent=None
    ) -> None:
        super().__init__(parent)
        self.title = title
        self.expanded = expanded
        self.expanded_changed = _BoundSignal()
        self._layout = _DummyLayout()
        self.__class__.instances.append(self)

    def content_layout(self) -> _DummyLayout:
        return self._layout

    def emit_expanded(self, expanded: bool) -> None:
        self.expanded = bool(expanded)
        self.expanded_changed.emit(bool(expanded))


class _DummyRegionSelector(_DummyWidget):
    """Simple region selector stub used for Analysis tab tests."""

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.selection_changed = _BoundSignal()
        self.allowed_ids: set[int] | None = None
        self.atlas = None
        self.selected: tuple[int, str] | None = None
        self.selected_ids: list[int] = []
        self._structure_map: dict[int, dict] = {}
        self.clear_calls = 0

    def get_single_selected_region(self) -> tuple[int, str] | None:
        if self.selected is not None:
            return self.selected
        if self.selected_ids:
            struct_id = int(self.selected_ids[0])
            struct = self._structure_map.get(struct_id, {})
            acronym = str(struct.get("acronym", f"R{struct_id}"))
            return (struct_id, acronym)
        return self.selected

    def get_selected_ids(self, include_children: bool = True) -> list[int]:
        _ = include_children
        if self.selected_ids:
            return list(self.selected_ids)
        if self.selected is not None:
            return [int(self.selected[0])]
        return []

    def set_allowed_structure_ids(
        self, structure_ids: set[int] | None
    ) -> None:
        self.allowed_ids = (
            None if structure_ids is None else set(structure_ids)
        )

    def set_atlas(self, atlas) -> None:
        self.atlas = atlas
        self._structure_map = getattr(atlas, "structures", {})

    def select_region_by_id(self, region_id: int | None) -> None:
        if region_id is None:
            self.selected = None
            self.selected_ids = []
            return
        struct = self._structure_map.get(region_id, {})
        acronym = str(struct.get("acronym", f"R{region_id}"))
        self.selected = (int(region_id), acronym)
        self.selected_ids = [int(region_id)]

    def select_regions(self, acronyms: list[str]) -> None:
        selected_ids = [
            int(struct_id)
            for struct_id, struct in self._structure_map.items()
            if struct.get("acronym") in set(acronyms)
        ]
        self.selected_ids = selected_ids
        self.selected = self.get_single_selected_region()

    def clear(self) -> None:
        self.clear_calls += 1
        self.selected = None
        self.selected_ids = []
        self._structure_map = {}


class _DummyFileDialog:
    """Minimal QFileDialog stand-in."""

    @staticmethod
    def getSaveFileName(*_args, **_kwargs):
        return ("", "")


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


class _DummyQTimer:
    """Minimal QTimer stand-in with immediate singleShot execution."""

    @staticmethod
    def singleShot(_interval: int, callback) -> None:
        callback()


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
    qtcore_module.QTimer = _DummyQTimer
    qtcore_module.Signal = _Signal

    qtgui_module = types.ModuleType("qtpy.QtGui")
    qtgui_module.QColor = _DummyQColor
    qtgui_module.QIcon = _DummyQIcon
    qtgui_module.QPixmap = _DummyQPixmap

    qtwidgets_module = types.ModuleType("qtpy.QtWidgets")
    for name, value in {
        "QComboBox": _DummyCombo,
        "QDoubleSpinBox": _DummySpinBox,
        "QFileDialog": _DummyFileDialog,
        "QGroupBox": _DummyWidget,
        "QHBoxLayout": _DummyLayout,
        "QLabel": _DummyLabel,
        "QLineEdit": _DummyLineEdit,
        "QProgressBar": _DummyProgressBar,
        "QPushButton": _DummyButton,
        "QScrollArea": _DummyScrollArea,
        "QScrollArea": _DummyScrollArea,
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
    selected_ids: list[int] | None = None,
):
    selector = _DummyRegionSelector()
    selector.selected = selected
    selector.selected_ids = list(
        selected_ids or ([] if selected is None else [selected[0]])
    )
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

    widget = AnalysisTabWidget(_DummyViewer())

    titles = [section.title for section in _DummyCollapsibleSection.instances]
    expanded = [
        section.expanded for section in _DummyCollapsibleSection.instances
    ]

    assert titles == [
        "Clustering",
        "Select Target Region",
        "Node Count Heatmap",
        "Select Heatmap Region",
        "Progress",
        "Clustermap",
        "Export Results",
    ]
    assert expanded == [True, False, True, False, True, False, False]


def test_analysis_tab_wraps_content_in_scroll_area():
    """Analysis tab should use a scroll area so the dock can stay bounded."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())

    assert isinstance(widget._scroll_area, _DummyScrollArea)
    assert widget._scroll_area.widget_resizable is True
    assert widget._scroll_area.widget is widget._scroll_content


def test_analysis_allowed_structure_ids_include_dataset_regions_and_ancestors():
    """Dataset-backed Analysis trees should expose represented leaves plus ancestors."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._atlas = _FakeAtlas(
        {
            997: {
                "id": 997,
                "acronym": "root",
                "name": "root",
                "structure_id_path": [997],
            },
            315: {
                "id": 315,
                "acronym": "ISO",
                "name": "Isocortex",
                "structure_id_path": [997, 315],
            },
            184: {
                "id": 184,
                "acronym": "FRP",
                "name": "Frontal pole",
                "structure_id_path": [997, 315, 184],
            },
            68: {
                "id": 68,
                "acronym": "FRP1",
                "name": "FRP1",
                "structure_id_path": [997, 315, 184, 68],
            },
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
            997: {
                "id": 997,
                "acronym": "root",
                "name": "root",
                "structure_id_path": [997],
            },
            315: {
                "id": 315,
                "acronym": "ISO",
                "name": "Isocortex",
                "structure_id_path": [997, 315],
            },
            184: {
                "id": 184,
                "acronym": "FRP",
                "name": "Frontal pole",
                "structure_id_path": [997, 315, 184],
            },
            68: {
                "id": 68,
                "acronym": "FRP1",
                "name": "FRP1",
                "structure_id_path": [997, 315, 184, 68],
            },
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
            997: {
                "id": 997,
                "acronym": "root",
                "name": "root",
                "structure_id_path": [997],
            },
            184: {
                "id": 184,
                "acronym": "FRP",
                "name": "Frontal pole",
                "structure_id_path": [997, 184],
            },
            500: {
                "id": 500,
                "acronym": "CP",
                "name": "Caudoputamen",
                "structure_id_path": [997, 500],
            },
        }
    )
    widget._dataset_region_ids = {184, 500}

    widget._refresh_analysis_region_selectors()

    assert widget._selected_cluster_region() == (184, "FRP")
    assert widget._selected_heat_region() == (500, "CP")


def test_refresh_analysis_region_selectors_preserves_multiple_cluster_selections():
    """Clustering selector should retain multiple direct selections after refresh."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    cluster_selector = _make_selector(
        structure_map={
            184: {"acronym": "FRP"},
            500: {"acronym": "CP"},
        },
        selected_ids=[184, 500],
    )
    heat_selector = _make_selector(
        selected=(500, "CP"), structure_map={500: {"acronym": "CP"}}
    )

    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._cluster_region_selector = cluster_selector
    widget._heat_region_selector = heat_selector
    widget._cluster_region_summary_label = _DummyLabel()
    widget._heat_region_summary_label = _DummyLabel()
    widget._atlas = _FakeAtlas(
        {
            997: {
                "id": 997,
                "acronym": "root",
                "name": "root",
                "structure_id_path": [997],
            },
            184: {
                "id": 184,
                "acronym": "FRP",
                "name": "Frontal pole",
                "structure_id_path": [997, 184],
            },
            500: {
                "id": 500,
                "acronym": "CP",
                "name": "Caudoputamen",
                "structure_id_path": [997, 500],
            },
        }
    )
    widget._dataset_region_ids = {184, 500}

    widget._refresh_analysis_region_selectors()

    assert widget._selected_cluster_regions() == [(184, "FRP"), (500, "CP")]


def test_represented_region_ids_for_selection_expands_parent_to_dataset_descendants():
    """Parent selection for heatmaps should expand to represented descendant region IDs."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._atlas = _FakeAtlas(
        {
            997: {
                "id": 997,
                "acronym": "root",
                "name": "root",
                "structure_id_path": [997],
            },
            184: {
                "id": 184,
                "acronym": "FRP",
                "name": "Frontal pole",
                "structure_id_path": [997, 184],
            },
            68: {
                "id": 68,
                "acronym": "FRP1",
                "name": "FRP1",
                "structure_id_path": [997, 184, 68],
            },
            667: {
                "id": 667,
                "acronym": "FRP2/3",
                "name": "FRP2/3",
                "structure_id_path": [997, 184, 667],
            },
            500: {
                "id": 500,
                "acronym": "CP",
                "name": "Caudoputamen",
                "structure_id_path": [997, 500],
            },
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


def test_update_region_summary_labels_compacts_multiple_cluster_regions():
    """Cluster summary should compact long direct-selection lists."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._cluster_region_selector = _make_selector(
        selected_ids=[184, 500, 315],
        structure_map={
            184: {"name": "Frontal pole", "acronym": "FRP"},
            500: {"name": "Caudoputamen", "acronym": "CP"},
            315: {"name": "Isocortex", "acronym": "ISO"},
        },
    )
    widget._heat_region_selector = _make_selector(selected=None)
    widget._cluster_region_summary_label = _DummyLabel()
    widget._heat_region_summary_label = _DummyLabel()

    widget._update_region_summary_labels()

    assert (
        widget._cluster_region_summary_label.text()
        == "FRP (Frontal pole), CP (Caudoputamen) +1 more"
    )


def test_update_button_states_enables_export_controls_after_clustering():
    """Export controls should enable only when a clustering result exists and the UI is idle."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    widget._db = object()
    widget._atlas = object()
    widget._last_cluster_result = object()

    widget._update_button_states()

    assert widget._render_clustermap_btn.isEnabled()
    assert widget._save_cluster_workbook_btn.isEnabled()
    assert widget._save_distance_workbook_btn.isEnabled()
    assert widget._save_extended_parquet_btn.isEnabled()
    assert widget._save_dendrogram_btn.isEnabled()


def test_update_button_states_disables_export_controls_without_result():
    """Export controls should stay disabled until clustering has completed."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget(_DummyViewer())
    widget._db = object()
    widget._atlas = object()
    widget._last_cluster_result = None

    widget._update_button_states()

    assert not widget._render_clustermap_btn.isEnabled()
    assert not widget._save_cluster_workbook_btn.isEnabled()
    assert not widget._save_distance_workbook_btn.isEnabled()
    assert not widget._save_extended_parquet_btn.isEnabled()
    assert not widget._save_dendrogram_btn.isEnabled()


def test_on_correlation_finished_leaves_clustermap_unrendered():
    """Clustering completion should not auto-render the dendrogram preview."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._n_clusters_spin = _DummySpinBox()
    widget._n_clusters_spin.setValue(2)
    widget._progress_bar = _DummyProgressBar()
    widget._progress_label = _DummyLabel()
    widget._update_button_states = lambda: None
    widget._update_cluster_filter_combo = lambda: None
    placeholder_messages: list[str] = []
    widget._show_clustermap_message = lambda message: (
        placeholder_messages.append(message)
    )
    draw_calls: list[object] = []
    widget._draw_clustermap = lambda result: draw_calls.append(result)
    emitted: list[tuple[object, dict]] = []
    widget.cluster_colors_updated.connect(
        lambda result, color_map: emitted.append((result, color_map))
    )

    result = types.SimpleNamespace(
        neuron_ids=["n1", "n2"],
        labels=np.array([1, 2], dtype=np.int32),
    )

    widget._on_correlation_finished(result)

    assert widget._last_cluster_result is result
    assert draw_calls == []
    assert emitted == []
    assert placeholder_messages == [
        "Clustering complete. Click Render Dendrogram to view."
    ]


def test_render_clustermap_requires_button_press():
    """Render button handler should draw only when a result is available."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    result = types.SimpleNamespace()
    widget._last_cluster_result = result
    draw_calls: list[object] = []
    widget._draw_clustermap = lambda cluster_result: draw_calls.append(
        cluster_result
    )

    widget._render_clustermap_requested()

    assert draw_calls == [result]


def test_color_neurons_by_cluster_emits_updates_only_on_button_press():
    """Explicit cluster-color application should drive table updates and its own message."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._viewer = _DummyViewer()
    widget._slice_projector = None
    widget._progress_bar = _DummyProgressBar()
    widget._progress_label = _DummyLabel()
    widget._last_cluster_result = types.SimpleNamespace(
        neuron_ids=["n1", "n2"]
    )
    widget._cluster_color_map = {
        "n1": [0.12, 0.47, 0.71, 1.0],
        "n2": [0.84, 0.15, 0.16, 1.0],
    }
    widget._actual_n_clusters = 2
    emitted: list[tuple[object, dict]] = []
    widget.cluster_colors_updated.connect(
        lambda result, color_map: emitted.append((result, color_map))
    )
    widget._flush_progress_updates = lambda: None

    widget._color_neurons_by_cluster()

    assert emitted == [
        (widget._last_cluster_result, widget._cluster_color_map)
    ]
    assert (
        widget._progress_label.text()
        == "Applied cluster colors: table 2 clustered neurons (2 clusters)"
    )


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
    assert (
        layer.metadata["heatmap_autocontrast_policy"] == "stable_full_volume"
    )


def test_analysis_heatmap_workaround_swallows_thumbnail_rank_mismatch():
    """Known napari thumbnail rank errors should be suppressed for analysis heatmaps."""
    module = _import_analysis_tab_module()

    class _CrashLayer(_DummyImageLayer):
        def _update_thumbnail(self) -> None:
            raise RuntimeError(
                "sequence argument must have length equal to input rank"
            )

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


def test_draw_clustermap_emits_debug_logs(caplog):
    """Clustermap drawing should use backend-managed figure size once."""
    module = _import_analysis_tab_module()
    AnalysisTabWidget = module.AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._clustermap_rendered = False
    widget._clustermap_section = _DummyCollapsibleSection(
        "Clustermap", expanded=True
    )
    widget._figure = _DummyFigure()
    widget._figure.size_inches = (5.0, 3.0)
    widget._canvas = _DummyCanvas(widget._figure)
    widget._canvas.draw = MagicMock()
    widget._canvas.draw_idle = MagicMock()
    widget._cluster_color_map = {
        "n1": [0.1, 0.2, 0.3, 1.0],
        "n2": [0.2, 0.3, 0.4, 1.0],
    }
    populate_calls: list[tuple[object, object, object, tuple[float, float], int]] = []

    def _fake_populate(
        figure, result, cluster_color_map, *, figsize, dpi
    ):
        populate_calls.append(
            (figure, result, cluster_color_map, tuple(figsize), int(dpi))
        )
        return figure

    module._populate_embedded_clustermap_figure = _fake_populate
    result = ClusterResult(
        correlation_matrix=np.eye(2, dtype=np.float32),
        distance_matrix=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        linkage_matrix=np.array([[0.0, 1.0, 1.0, 2.0]], dtype=np.float64),
        neuron_ids=["n1", "n2"],
        reorder_indices=np.array([0, 1], dtype=np.intp),
        labels=np.array([1, 2], dtype=np.int32),
    )

    with caplog.at_level(
        logging.DEBUG, logger="napari_swc_viewer.widgets.analysis_tab"
    ):
        widget._draw_clustermap(result)

    messages = [record.getMessage() for record in caplog.records]
    assert any("_draw_clustermap start" in message for message in messages)
    assert any(
        "populate_clustermap_figure complete" in message
        for message in messages
    )
    assert any(
        "_draw_clustermap canvas draw complete" in message
        for message in messages
    )
    assert populate_calls == [
        (
            widget._figure,
            result,
            widget._cluster_color_map,
            (5.0, 3.0),
            100,
        )
    ]
    widget._canvas.draw.assert_called_once_with()
    widget._canvas.draw_idle.assert_not_called()
    assert widget._figure.canvas is None
    assert widget._figure.size_inches == (5.0, 3.0)
    assert widget._figure.set_size_inches_calls == 0


def test_draw_clustermap_uses_physical_canvas_size_when_figure_size_unavailable():
    """Preview sizing should fall back to physical canvas pixels on HiDPI displays."""
    module = _import_analysis_tab_module()
    AnalysisTabWidget = module.AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._clustermap_rendered = False
    widget._figure = _DummyFigure()
    widget._figure.size_inches = None
    widget._canvas = _DummyCanvas(widget._figure)
    widget._canvas._width = 600
    widget._canvas._height = 400
    widget._canvas._physical_width = 1200
    widget._canvas._physical_height = 800
    widget._canvas.draw = MagicMock()
    widget._cluster_color_map = None
    populate_calls: list[tuple[object, object, object, tuple[float, float], int]] = []

    def _fake_populate(
        figure, result, cluster_color_map, *, figsize, dpi
    ):
        populate_calls.append(
            (figure, result, cluster_color_map, tuple(figsize), int(dpi))
        )
        return figure

    module._populate_embedded_clustermap_figure = _fake_populate
    result = ClusterResult(
        correlation_matrix=np.eye(2, dtype=np.float32),
        distance_matrix=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        linkage_matrix=np.array([[0.0, 1.0, 1.0, 2.0]], dtype=np.float64),
        neuron_ids=["n1", "n2"],
        reorder_indices=np.array([0, 1], dtype=np.intp),
        labels=np.array([1, 2], dtype=np.int32),
    )

    widget._draw_clustermap(result)

    assert populate_calls == [
        (
            widget._figure,
            result,
            None,
            (12.0, 8.0),
            100,
        )
    ]
    widget._canvas.draw.assert_called_once_with()
    assert widget._figure.set_size_inches_calls == 0


def test_build_clustermap_on_demand_draws_cached_result_and_logs(caplog):
    """Opt-in dendrogram rendering should draw only when the button handler runs."""
    module = _import_analysis_tab_module()
    AnalysisTabWidget = module.AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._build_clustermap_btn = _DummyButton("Build Dendrogram")
    widget._clustermap_status_label = _DummyLabel()
    widget._update_button_states = MagicMock()
    widget._draw_clustermap = MagicMock()
    result = ClusterResult(
        correlation_matrix=np.eye(2, dtype=np.float32),
        distance_matrix=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        linkage_matrix=np.array([[0.0, 1.0, 1.0, 2.0]], dtype=np.float64),
        neuron_ids=["n1", "n2"],
        reorder_indices=np.array([0, 1], dtype=np.intp),
        labels=np.array([1, 2], dtype=np.int32),
    )
    widget._last_cluster_result = result

    with caplog.at_level(
        logging.DEBUG, logger="napari_swc_viewer.widgets.analysis_tab"
    ):
        widget._build_clustermap_on_demand()

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "_build_clustermap_on_demand start" in message for message in messages
    )
    assert any(
        "_build_clustermap_on_demand complete" in message
        for message in messages
    )
    widget._draw_clustermap.assert_called_once_with(result)
    widget._update_button_states.assert_called_once_with()
    assert (
        widget._clustermap_status_label.text()
        == "Dendrogram ready for 2 neurons."
    )


def test_on_correlation_finished_emits_debug_logs(caplog):
    """UI completion handler should defer dendrogram rendering until a button click."""
    module = _import_analysis_tab_module()
    AnalysisTabWidget = module.AnalysisTabWidget
    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._last_cluster_result = None
    widget._cluster_color_map = {"n1": [0.1, 0.2, 0.3, 1.0]}
    widget._actual_n_clusters = 2
    widget._n_clusters_spin = types.SimpleNamespace(value=lambda: 5)
    widget._progress_bar = _DummyProgressBar()
    widget._progress_label = _DummyLabel()
    widget._clustermap_status_label = _DummyLabel()
    widget._build_clustermap_btn = _DummyButton("Build Dendrogram")
    widget._update_button_states = MagicMock()
    widget._update_cluster_filter_combo = lambda: None
    widget._build_cluster_color_map = MagicMock()
    widget._draw_clustermap = MagicMock()
    emitted: list[tuple[object, dict[str, list[float]]]] = []
    widget.cluster_colors_updated = types.SimpleNamespace(
        emit=lambda result, color_map: emitted.append((result, color_map))
    )
    result = ClusterResult(
        correlation_matrix=np.eye(2, dtype=np.float32),
        distance_matrix=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        linkage_matrix=np.array([[0.0, 1.0, 1.0, 2.0]], dtype=np.float64),
        neuron_ids=["n1", "n2"],
        reorder_indices=np.array([0, 1], dtype=np.intp),
        labels=np.array([1, 2], dtype=np.int32),
    )

    with caplog.at_level(
        logging.DEBUG, logger="napari_swc_viewer.widgets.analysis_tab"
    ):
        widget._on_correlation_finished(result)

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "_on_correlation_finished start" in message for message in messages
    )
    assert any("color map built" in message for message in messages)
    assert any(
        "clustermap render deferred until button click" in message
        for message in messages
    )
    widget._build_cluster_color_map.assert_called_once_with()
    widget._update_button_states.assert_called_once_with()
    widget._draw_clustermap.assert_not_called()
    assert emitted == [(result, {"n1": [0.1, 0.2, 0.3, 1.0]})]
    assert widget._clustermap_status_label.text() == (
        "Clustering complete. Click 'Build Dendrogram' to render the cluster map."
    )
