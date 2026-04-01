"""Tests for region selector hierarchy and Analysis-specific behavior."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


class _Signal:
    """Minimal stand-in for ``qtpy.QtCore.Signal``."""

    def __init__(self, *_args, **_kwargs) -> None:
        return None


class _DummyQtObject:
    """Minimal stand-in for simple Qt widget classes."""

    def __init__(self, *_args, **_kwargs) -> None:
        return None


class _DummyCheckBox:
    """Small stand-in for the checkbox API used by RegionSelectorWidget."""

    def __init__(self, checked: bool) -> None:
        self._checked = checked
        self.visible = True
        self.enabled = True

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool) -> None:
        self._checked = bool(checked)

    def setVisible(self, visible: bool) -> None:
        self.visible = bool(visible)

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)


class _DummyTree:
    """Small tree widget stand-in for hierarchy tests."""

    def __init__(self) -> None:
        self.items: list[_DummyTreeItem] = []
        self.blocked = False

    def blockSignals(self, blocked: bool) -> None:
        self.blocked = bool(blocked)

    def clear(self) -> None:
        self.items = []

    def topLevelItemCount(self) -> int:
        return len(self.items)

    def topLevelItem(self, index: int) -> "_DummyTreeItem":
        return self.items[index]


class _DummyTreeItem:
    """Minimal tree item used for selection and population tests."""

    def __init__(self, parent, labels: list[str]) -> None:
        self._labels = labels
        self._flags = 0
        self._check_state = 0
        self._data: dict[tuple[int, int], object] = {}
        self._children: list[_DummyTreeItem] = []
        self.hidden = False
        self.expanded = False

        if isinstance(parent, _DummyTree):
            parent.items.append(self)
        else:
            parent._children.append(self)

    def flags(self) -> int:
        return self._flags

    def setFlags(self, flags: int) -> None:
        self._flags = flags

    def setCheckState(self, _column: int, state: int) -> None:
        self._check_state = state

    def checkState(self, _column: int) -> int:
        return self._check_state

    def setData(self, column: int, role: int, value) -> None:
        self._data[(column, role)] = value

    def data(self, column: int, role: int):
        return self._data.get((column, role))

    def text(self, column: int) -> str:
        return self._labels[column]

    def childCount(self) -> int:
        return len(self._children)

    def child(self, index: int) -> "_DummyTreeItem":
        return self._children[index]

    def setHidden(self, hidden: bool) -> None:
        self.hidden = bool(hidden)

    def setExpanded(self, expanded: bool) -> None:
        self.expanded = bool(expanded)


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
    for name, value in {
        "QCheckBox": _DummyQtObject,
        "QHBoxLayout": _DummyQtObject,
        "QLabel": _DummyQtObject,
        "QLineEdit": _DummyQtObject,
        "QPushButton": _DummyQtObject,
        "QTreeWidget": _DummyTree,
        "QTreeWidgetItem": _DummyTreeItem,
        "QVBoxLayout": _DummyQtObject,
        "QWidget": _DummyQtObject,
    }.items():
        setattr(qtwidgets_module, name, value)

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
    widget._force_include_children = False
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


def test_on_item_changed_enforces_single_selection():
    """Analysis-mode selectors should clear any previously checked item."""
    module = _import_region_selector_module()
    RegionSelectorWidget = module.RegionSelectorWidget
    qt = module.Qt

    widget = RegionSelectorWidget.__new__(RegionSelectorWidget)
    widget._single_select = True
    widget._selection_change_depth = 0
    widget._tree = _DummyTree()
    widget._update_selection_label = lambda: None
    widget._emit_selection_changed = lambda: None

    item_a = _DummyTreeItem(widget._tree, ["FRP", "FRP"])
    item_b = _DummyTreeItem(widget._tree, ["CP", "CP"])
    item_a.setCheckState(0, qt.Checked)
    item_b.setCheckState(0, qt.Checked)
    widget._items_by_id = {184: item_a, 500: item_b}

    widget._on_item_changed(item_a, 0)

    assert item_a.checkState(0) == qt.Checked
    assert item_b.checkState(0) == qt.Unchecked


def test_populate_tree_limits_visible_nodes_to_allowed_ids():
    """Filtered trees should only contain explicitly allowed nodes."""
    module = _import_region_selector_module()
    RegionSelectorWidget = module.RegionSelectorWidget

    widget = RegionSelectorWidget.__new__(RegionSelectorWidget)
    widget._tree = _DummyTree()
    widget._items_by_id = {}
    widget._structure_map = {}
    widget._allowed_structure_ids = {315, 184, 68}
    widget._update_selection_label = lambda: None
    widget._atlas = types.SimpleNamespace(
        structures={
            997: {"id": 997, "acronym": "root", "structure_id_path": [997]},
            315: {"id": 315, "acronym": "ISO", "structure_id_path": [997, 315]},
            184: {"id": 184, "acronym": "FRP", "structure_id_path": [997, 315, 184]},
            68: {"id": 68, "acronym": "FRP1", "structure_id_path": [997, 315, 184, 68]},
            500: {"id": 500, "acronym": "CP", "structure_id_path": [997, 500]},
        }
    )

    widget._populate_tree()

    assert set(widget._items_by_id) == {68, 184, 315}
    assert widget._tree.topLevelItemCount() == 1
    assert widget._tree.topLevelItem(0).text(1) == "ISO"


def test_force_include_children_hides_checkbox_and_overrides_toggle():
    """Analysis-mode selectors should always include descendants and hide the toggle."""
    module = _import_region_selector_module()
    RegionSelectorWidget = module.RegionSelectorWidget

    widget = RegionSelectorWidget.__new__(RegionSelectorWidget)
    widget._show_include_children = True
    widget._force_include_children = False
    widget._include_children_cb = _DummyCheckBox(False)

    widget.set_include_children_controls(
        show_include_children=False,
        force_include_children=True,
    )

    assert widget._include_children_cb.visible is False
    assert widget._include_children_cb.enabled is False
    assert widget._include_children_cb.isChecked() is True
    assert widget.include_children_enabled() is True


def test_get_single_selected_region_returns_direct_selection():
    """Single-region helpers should return the directly checked item only."""
    module = _import_region_selector_module()
    RegionSelectorWidget = module.RegionSelectorWidget
    qt = module.Qt

    widget = RegionSelectorWidget.__new__(RegionSelectorWidget)
    widget._tree = _DummyTree()
    widget._items_by_id = {}
    widget._structure_map = {184: {"acronym": "FRP"}}

    item = _DummyTreeItem(widget._tree, ["FRP", "FRP"])
    item.setData(0, qt.UserRole, 184)
    item.setCheckState(0, qt.Checked)
    widget._items_by_id[184] = item

    assert widget.get_single_selected_region() == (184, "FRP")
