from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


def _import_neuron_table_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "napari_swc_viewer"
        / "widgets"
        / "neuron_table.py"
    )
    package_name = "napari_swc_viewer.widgets"
    original_package = sys.modules.get(package_name)
    if original_package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(module_path.parent)]
        sys.modules[package_name] = package

    spec = importlib.util.spec_from_file_location(
        "napari_swc_viewer.widgets.neuron_table",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["napari_swc_viewer.widgets.neuron_table"] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("napari_swc_viewer.widgets.neuron_table", None)
        if original_package is None:
            sys.modules.pop(package_name, None)


class _DummySignal:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def emit(self, *args) -> None:
        self.calls.append(args)


class _DummyItem:
    def __init__(self, file_id, user_role) -> None:
        self._file_id = file_id
        self._user_role = user_role

    def data(self, role):
        return self._file_id if role == self._user_role else None

    def text(self) -> str:
        return str(self._file_id)


class _DummyTable:
    def __init__(self, file_ids: list[object], user_role) -> None:
        self._file_ids = list(file_ids)
        self._user_role = user_role
        self._sorting_enabled = True
        self._signals_blocked = False
        self.sort_calls: list[tuple[int, object]] = []

    def rowCount(self) -> int:
        return len(self._file_ids)

    def item(self, row: int, column: int):
        if column != 2:
            return None
        if row < 0 or row >= len(self._file_ids):
            return None
        return _DummyItem(self._file_ids[row], self._user_role)

    def isSortingEnabled(self) -> bool:
        return self._sorting_enabled

    def setSortingEnabled(self, enabled: bool) -> None:
        self._sorting_enabled = bool(enabled)

    def blockSignals(self, blocked: bool) -> bool:
        previous = self._signals_blocked
        self._signals_blocked = bool(blocked)
        return previous

    def clearSelection(self) -> None:
        return None

    def clearContents(self) -> None:
        self._file_ids = []

    def setRowCount(self, count: int) -> None:
        if count <= 0:
            self._file_ids = []
            return
        if len(self._file_ids) < count:
            self._file_ids.extend([None] * (count - len(self._file_ids)))
        else:
            self._file_ids = self._file_ids[:count]

    def set_file_id(self, row: int, file_id: object) -> None:
        self._file_ids[row] = file_id

    def sortByColumn(self, column: int, order) -> None:
        self.sort_calls.append((column, order))


def _make_widget(module, entries_by_file_id: dict[object, object]):
    widget = module.NeuronTableWidget.__new__(module.NeuronTableWidget)
    widget._entries = dict(entries_by_file_id)
    widget._table = _DummyTable(
        list(entries_by_file_id.keys()),
        module.Qt.UserRole,
    )
    widget.selection_changed = _DummySignal()
    widget.state_changed = _DummySignal()

    def _populate_row(row: int, entry) -> None:
        widget._table.set_file_id(row, entry.file_id)

    widget._populate_row = _populate_row
    return widget


def test_neuron_table_retain_file_ids_preserves_survivor_state() -> None:
    module = _import_neuron_table_module()
    entry_a = module.NeuronEntry(
        file_id="n1",
        subject="s1",
        color=[0.1, 0.2, 0.3, 1.0],
        cluster_id=7,
        visible=False,
        added_to_scene=True,
    )
    entry_b = module.NeuronEntry(file_id="n2", subject="s2")
    entry_c = module.NeuronEntry(
        file_id="n3",
        subject="s3",
        cluster_id=3,
    )
    widget = _make_widget(
        module,
        {"n1": entry_a, "n2": entry_b, "n3": entry_c},
    )

    widget.retain_file_ids(["n1", "n3"])

    assert widget.file_ids() == ["n1", "n3"]
    assert widget.get_color("n1") == [0.1, 0.2, 0.3, 1.0]
    assert widget.get_visibility_map() == {"n1": False, "n3": True}
    assert widget.summary().table_count == 2
    assert widget.summary().added_count == 1
    assert widget.summary().visible_count == 1
    assert widget.available_cluster_ids() == [3, 7]
    assert widget.selection_changed.calls == [([],)]
    assert len(widget.state_changed.calls) == 1


def test_neuron_table_remove_file_ids_preserves_remaining_entry_state() -> None:
    module = _import_neuron_table_module()
    entry_a = module.NeuronEntry(file_id="n1", subject="s1")
    entry_b = module.NeuronEntry(
        file_id="n2",
        subject="s2",
        color=[0.6, 0.4, 0.2, 1.0],
        cluster_id=5,
        visible=False,
        added_to_scene=True,
    )
    entry_c = module.NeuronEntry(file_id="n3", subject="s3")
    widget = _make_widget(
        module,
        {"n1": entry_a, "n2": entry_b, "n3": entry_c},
    )

    widget.remove_file_ids(["n1", "n3"])

    assert widget.file_ids() == ["n2"]
    assert widget.get_color("n2") == [0.6, 0.4, 0.2, 1.0]
    assert widget.get_visibility_map() == {"n2": False}
    assert widget.summary().table_count == 1
    assert widget.summary().added_count == 1
    assert widget.summary().visible_count == 0
    assert widget.available_cluster_ids() == [5]
    assert widget.selection_changed.calls == [([],)]
    assert len(widget.state_changed.calls) == 1


def test_neuron_table_sort_by_cluster_delegates_to_cluster_column_sort() -> None:
    module = _import_neuron_table_module()
    widget = _make_widget(
        module,
        {
            "n1": module.NeuronEntry(file_id="n1", subject="s1", cluster_id=2),
            "n2": module.NeuronEntry(file_id="n2", subject="s2", cluster_id=1),
        },
    )

    widget.sort_by_cluster()

    assert widget._table.sort_calls == [
        (module.COL_CLUSTER, module.Qt.AscendingOrder)
    ]
