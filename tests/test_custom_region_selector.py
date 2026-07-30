from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest

from napari_swc_viewer.isocortex_layers import (
    CustomRegionHierarchy,
    CustomRegionHierarchyNode,
)


class _BoundSignal:
    def __init__(self) -> None:
        self.emissions: list[tuple[object, ...]] = []

    def emit(self, *args) -> None:
        self.emissions.append(args)


class _Signal:
    def __init__(self, *_args, **_kwargs) -> None:
        self._name = ""

    def __set_name__(self, _owner, name: str) -> None:
        self._name = f"_{name}_signal"

    def __get__(self, instance, _owner):
        if instance is None:
            return self
        if not hasattr(instance, self._name):
            setattr(instance, self._name, _BoundSignal())
        return getattr(instance, self._name)


class _DummyObject:
    def __init__(self, *_args, **_kwargs) -> None:
        return None


class _DummyLabel:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, text: str) -> None:
        self.text = str(text)


class _DummyTree:
    def __init__(self) -> None:
        self.items: list[_DummyTreeItem] = []
        self.blocked = False

    def blockSignals(self, blocked: bool) -> None:
        self.blocked = bool(blocked)

    def topLevelItemCount(self) -> int:
        return len(self.items)

    def topLevelItem(self, index: int) -> "_DummyTreeItem":
        return self.items[index]


class _DummyTreeItem:
    def __init__(self, parent, columns: list[str]) -> None:
        self._parent = None if isinstance(parent, _DummyTree) else parent
        self._columns = columns
        self._children: list[_DummyTreeItem] = []
        self._check_state = 0
        self._flags = 0
        self._data: dict[tuple[int, int], object] = {}
        self.hidden = False
        self.expanded = False
        if isinstance(parent, _DummyTree):
            parent.items.append(self)
        else:
            parent._children.append(self)

    def flags(self) -> int:
        return self._flags

    def setFlags(self, flags: int) -> None:
        self._flags = int(flags)

    def setCheckState(self, _column: int, state: int) -> None:
        self._check_state = int(state)

    def checkState(self, _column: int) -> int:
        return self._check_state

    def setData(self, column: int, role: int, value: object) -> None:
        self._data[(column, role)] = value

    def data(self, column: int, role: int):
        return self._data.get((column, role))

    def childCount(self) -> int:
        return len(self._children)

    def child(self, index: int) -> "_DummyTreeItem":
        return self._children[index]

    def parent(self):
        return self._parent

    def text(self, column: int) -> str:
        return self._columns[column]

    def setHidden(self, hidden: bool) -> None:
        self.hidden = bool(hidden)

    def setExpanded(self, expanded: bool) -> None:
        self.expanded = bool(expanded)


def _load_module():
    qtcore = types.ModuleType("qtpy.QtCore")
    qtcore.Qt = types.SimpleNamespace(
        Checked=2,
        PartiallyChecked=1,
        Unchecked=0,
        ItemIsUserCheckable=1,
        UserRole=32,
    )
    qtcore.Signal = _Signal
    qtwidgets = types.ModuleType("qtpy.QtWidgets")
    for name, value in {
        "QHeaderView": types.SimpleNamespace(Stretch=1, ResizeToContents=2),
        "QHBoxLayout": _DummyObject,
        "QLabel": _DummyObject,
        "QLineEdit": _DummyObject,
        "QPushButton": _DummyObject,
        "QTreeWidget": _DummyTree,
        "QTreeWidgetItem": _DummyTreeItem,
        "QVBoxLayout": _DummyObject,
        "QWidget": _DummyObject,
    }.items():
        setattr(qtwidgets, name, value)

    replacements = {"qtpy.QtCore": qtcore, "qtpy.QtWidgets": qtwidgets}
    previous = {name: sys.modules.get(name) for name in replacements}
    try:
        sys.modules.update(replacements)
        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "napari_swc_viewer"
            / "widgets"
            / "custom_region_selector.py"
        )
        spec = importlib.util.spec_from_file_location(
            "napari_swc_viewer.widgets.custom_region_selector_test_module",
            path,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def _hierarchy() -> CustomRegionHierarchy:
    return CustomRegionHierarchy(
        root=CustomRegionHierarchyNode(
            label="Isocortex Layers",
            children=(
                CustomRegionHierarchyNode(
                    label="L1",
                    children=(
                        CustomRegionHierarchyNode(
                            label="Alpha area, layer 1",
                            acronym="AAA1",
                            region_id=101,
                        ),
                        CustomRegionHierarchyNode(
                            label="Beta area, layer 1",
                            acronym="BBB1",
                            region_id=102,
                        ),
                    ),
                ),
                CustomRegionHierarchyNode(
                    label="L2/3",
                    children=(
                        CustomRegionHierarchyNode(
                            label="Gamma area, layer 2/3",
                            acronym="GGG2/3",
                            region_id=201,
                        ),
                    ),
                ),
            ),
        ),
        atlas_name="allen_mouse_25um",
        atlas_version="1.2",
    )


def _selector(module):
    selector = module.CustomRegionSelectorWidget.__new__(
        module.CustomRegionSelectorWidget
    )
    selector._hierarchy = _hierarchy()
    selector._terminal_items_by_id = {}
    selector._all_items = []
    selector._selection_change_depth = 0
    selector._tree = _DummyTree()
    selector._selection_label = _DummyLabel()
    root = selector._add_node(selector._hierarchy.root, None)
    return selector, root


def test_group_selection_propagates_and_ancestors_become_partial() -> None:
    module = _load_module()
    selector, root = _selector(module)
    first_layer = root.child(0)
    second_layer = root.child(1)

    first_layer.setCheckState(0, module.Qt.Checked)
    selector._on_item_changed(first_layer, 0)

    assert selector.get_selected_region_ids() == [101, 102]
    assert root.checkState(0) == module.Qt.PartiallyChecked
    assert selector._selection_label.text == "Selected: 2 terminal regions"

    second_layer.setCheckState(0, module.Qt.Checked)
    selector._on_item_changed(second_layer, 0)
    assert selector.get_selected_region_ids() == [101, 102, 201]
    assert root.checkState(0) == module.Qt.Checked

    first_layer.child(0).setCheckState(0, module.Qt.Unchecked)
    selector._on_item_changed(first_layer.child(0), 0)
    assert selector.get_selected_region_ids() == [102, 201]
    assert first_layer.checkState(0) == module.Qt.PartiallyChecked
    assert root.checkState(0) == module.Qt.PartiallyChecked


def test_root_selection_and_clear_return_exact_deduplicated_ids() -> None:
    module = _load_module()
    selector, root = _selector(module)

    root.setCheckState(0, module.Qt.Checked)
    selector._on_item_changed(root, 0)

    assert selector.get_selected_region_ids() == [101, 102, 201]
    groups = selector.get_selected_region_groups()
    assert [group.label for group in groups] == ["L1", "L2/3"]
    assert groups[0].region_ids == (101, 102)
    assert groups[0].acronyms == ("AAA1", "BBB1")
    assert groups[1].region_ids == (201,)
    assert groups[1].acronyms == ("GGG2/3",)

    selector.clear_selection()
    assert selector.get_selected_region_ids() == []
    assert selector.get_selected_region_groups() == ()
    assert all(
        item.checkState(0) == module.Qt.Unchecked for item in selector._all_items
    )


def test_partial_layer_selection_returns_only_checked_terminal_records() -> None:
    module = _load_module()
    selector, root = _selector(module)
    first_layer = root.child(0)

    first_layer.child(1).setCheckState(0, module.Qt.Checked)
    selector._on_item_changed(first_layer.child(1), 0)

    groups = selector.get_selected_region_groups()
    assert len(groups) == 1
    assert groups[0].label == "L1"
    assert groups[0].region_ids == (102,)
    assert groups[0].acronyms == ("BBB1",)


def test_search_keeps_matching_leaf_and_ancestors_visible() -> None:
    module = _load_module()
    selector, root = _selector(module)

    selector._on_search_changed("beta")

    assert root.hidden is False
    assert root.child(0).hidden is False
    assert root.child(0).child(0).hidden is True
    assert root.child(0).child(1).hidden is False
    assert root.child(1).hidden is True
    assert root.expanded is True
    assert root.child(0).expanded is True


def test_set_hierarchy_rejects_duplicate_terminal_ids() -> None:
    module = _load_module()
    selector = module.CustomRegionSelectorWidget.__new__(
        module.CustomRegionSelectorWidget
    )
    duplicate = CustomRegionHierarchy(
        root=CustomRegionHierarchyNode(
            label="Root",
            children=(
                CustomRegionHierarchyNode(
                    label="A",
                    acronym="A",
                    region_id=1,
                ),
                CustomRegionHierarchyNode(
                    label="B",
                    acronym="B",
                    region_id=1,
                ),
            ),
        ),
        atlas_name="test",
    )

    with pytest.raises(ValueError, match="duplicate terminal region IDs"):
        selector.set_hierarchy(duplicate)
