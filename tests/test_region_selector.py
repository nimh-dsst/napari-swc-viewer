"""Tests for region selector hierarchy and query expansion behavior."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


class _Signal:
    """Minimal stand-in for qtpy.QtCore.Signal."""

    def __init__(self, *_args, **_kwargs) -> None:
        return None


class _DummyQtObject:
    """Minimal stand-in for Qt widget classes."""

    def __init__(self, *_args, **_kwargs) -> None:
        return None


class _DummyCheckBox:
    """Small stand-in for the checkbox API used by RegionSelectorWidget."""

    def __init__(self, checked: bool) -> None:
        self._checked = checked

    def isChecked(self) -> bool:
        return self._checked


def _import_region_selector_module():
    """Import ``region_selector.py`` with stubbed Qt dependencies."""
    qtcore_module = types.ModuleType("qtpy.QtCore")
    qtcore_module.Qt = types.SimpleNamespace(
        Checked=2,
        Unchecked=0,
        ItemIsUserCheckable=1,
        UserRole=32,
    )
    qtcore_module.Signal = _Signal

    qtwidgets_module = types.ModuleType("qtpy.QtWidgets")
    for name in (
        "QCheckBox",
        "QHBoxLayout",
        "QLabel",
        "QLineEdit",
        "QPushButton",
        "QTreeWidget",
        "QTreeWidgetItem",
        "QVBoxLayout",
        "QWidget",
    ):
        setattr(qtwidgets_module, name, _DummyQtObject)

    replacements = {
        "qtpy.QtCore": qtcore_module,
        "qtpy.QtWidgets": qtwidgets_module,
    }
    previous = {name: sys.modules.get(name) for name in replacements}

    try:
        sys.modules.update(replacements)
        module_path = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "napari_swc_viewer"
            / "widgets"
            / "region_selector.py"
        )
        spec = importlib.util.spec_from_file_location(
            "region_selector_test_module",
            module_path,
        )
        assert spec is not None
        assert spec.loader is not None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_get_child_structure_ids_uses_structure_id_path_when_parent_missing():
    """Children should still be found when BrainGlobe omits parent_structure_id."""
    module = _import_region_selector_module()
    RegionSelectorWidget = module.RegionSelectorWidget

    widget = RegionSelectorWidget.__new__(RegionSelectorWidget)
    widget._structure_map = {
        315: {"id": 315, "acronym": "Isocortex", "structure_id_path": [997, 315]},
        184: {"id": 184, "acronym": "FRP", "structure_id_path": [997, 315, 184]},
        68: {"id": 68, "acronym": "FRP1", "structure_id_path": [997, 315, 184, 68]},
        667: {
            "id": 667,
            "acronym": "FRP2/3",
            "structure_id_path": [997, 315, 184, 667],
        },
    }

    assert sorted(widget._get_child_structure_ids(184)) == [68, 667]


def test_get_query_acronyms_uses_include_children_checkbox_state():
    """Query expansion should follow the checkbox rather than always including children."""
    module = _import_region_selector_module()
    RegionSelectorWidget = module.RegionSelectorWidget

    widget = RegionSelectorWidget.__new__(RegionSelectorWidget)
    widget._include_children_cb = _DummyCheckBox(False)
    calls: list[bool] = []

    def fake_get_selected_acronyms(*, include_children: bool = True) -> list[str]:
        calls.append(include_children)
        if include_children:
            return ["FRP", "FRP1"]
        return ["FRP"]

    widget.get_selected_acronyms = fake_get_selected_acronyms

    assert widget.get_query_acronyms() == ["FRP"]

    widget._include_children_cb = _DummyCheckBox(True)
    assert widget.get_query_acronyms() == ["FRP", "FRP1"]
    assert calls == [False, True]
