from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np

from napari_swc_viewer.analysis.clustering import ClusterResult
from napari_swc_viewer.cluster_assignments import ClusterAssignmentStore


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


class _DummyIndex:
    def __init__(self, row: int, column: int = 0) -> None:
        self._row = int(row)
        self._column = int(column)

    def row(self) -> int:
        return self._row

    def column(self) -> int:
        return self._column


class _DummyModel:
    def index(self, row: int, column: int) -> _DummyIndex:
        return _DummyIndex(row, column)


class _DummyItemSelection:
    def __init__(self) -> None:
        self.ranges: list[tuple[int, int]] = []

    def select(self, top_left: _DummyIndex, bottom_right: _DummyIndex) -> None:
        self.ranges.append((top_left.row(), bottom_right.row()))


class _DummyItemSelectionModel:
    ClearAndSelect = 1
    Rows = 2

    def __init__(self, table: "_DummyTable") -> None:
        self._table = table

    def select(self, selection: _DummyItemSelection, _flags: int) -> None:
        self._table._selected_rows.clear()
        for start, stop in selection.ranges:
            self._table._selected_rows.update(range(start, stop + 1))


class _DummyTable:
    def __init__(
        self,
        file_ids: list[object],
        user_role,
        neuron_id_column: int,
    ) -> None:
        self._file_ids = list(file_ids)
        self._user_role = user_role
        self._neuron_id_column = neuron_id_column
        self._sorting_enabled = True
        self._signals_blocked = False
        self.sort_calls: list[tuple[int, object]] = []
        self.hidden_rows: dict[int, bool] = {}
        self._selected_rows: set[int] = set()
        self._column_count = 7
        self.headers: list[str] = []
        self.resize_calls: list[tuple[int, object]] = []

    def rowCount(self) -> int:
        return len(self._file_ids)

    def columnCount(self) -> int:
        return self._column_count

    def setColumnCount(self, count: int) -> None:
        self._column_count = int(count)

    def setHorizontalHeaderLabels(self, headers: list[str]) -> None:
        self.headers = list(headers)

    def horizontalHeader(self):
        return self

    def setSectionResizeMode(self, column: int, mode) -> None:
        self.resize_calls.append((int(column), mode))

    def model(self) -> _DummyModel:
        return _DummyModel()

    def selectionModel(self) -> _DummyItemSelectionModel:
        return _DummyItemSelectionModel(self)

    def selectedIndexes(self) -> list[_DummyIndex]:
        return [_DummyIndex(row) for row in sorted(self._selected_rows)]

    def item(self, row: int, column: int):
        if column != self._neuron_id_column:
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
        self._selected_rows.clear()

    def clearContents(self) -> None:
        self._file_ids = []
        self._selected_rows.clear()

    def setRowCount(self, count: int) -> None:
        if count <= 0:
            self._file_ids = []
            self._selected_rows.clear()
            return
        if len(self._file_ids) < count:
            self._file_ids.extend([None] * (count - len(self._file_ids)))
        else:
            self._file_ids = self._file_ids[:count]
            self._selected_rows = {
                row for row in self._selected_rows if row < len(self._file_ids)
            }

    def set_file_id(self, row: int, file_id: object) -> None:
        self._file_ids[row] = file_id

    def sortByColumn(self, column: int, order) -> None:
        self.sort_calls.append((column, order))

    def setRowHidden(self, row: int, hidden: bool) -> None:
        self.hidden_rows[int(row)] = bool(hidden)

    def isRowHidden(self, row: int) -> bool:
        return self.hidden_rows.get(int(row), False)


def _make_widget(module, entries_by_file_id: dict[object, object]):
    widget = module.NeuronTableWidget.__new__(module.NeuronTableWidget)
    widget._entries = dict(entries_by_file_id)
    widget._table = _DummyTable(
        list(entries_by_file_id.keys()),
        module.Qt.UserRole,
        module.COL_NEURON_ID,
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
        heatmap_layer_names=("alpha Heatmap",),
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
    assert widget.get_cluster_map() == {"n1": 7, "n3": 3}
    assert widget.get_visibility_map() == {"n1": False, "n3": True}
    assert widget.summary().table_count == 2
    assert widget.summary().added_count == 1
    assert widget.summary().visible_count == 1
    assert widget.available_cluster_ids() == [3, 7]
    assert widget._entries["n1"].heatmap_layer_names == ("alpha Heatmap",)
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

    assert widget._table.sort_calls == [(module.COL_CLUSTER, module.Qt.AscendingOrder)]


def test_update_cluster_assignments_clears_unassigned_rows() -> None:
    module = _import_neuron_table_module()
    widget = _make_widget(
        module,
        {
            "n1": module.NeuronEntry(file_id="n1", subject="s1", cluster_id=9),
            "n2": module.NeuronEntry(file_id="n2", subject="s2", cluster_id=8),
            "n3": module.NeuronEntry(file_id="n3", subject="s3", cluster_id=7),
        },
    )
    cluster_cells: list[tuple[int, int | None]] = []
    widget._set_cluster_cell = lambda row, cluster_id: cluster_cells.append(
        (row, cluster_id)
    )
    result = ClusterResult(
        correlation_matrix=np.eye(1, dtype=np.float32),
        distance_matrix=np.zeros((1, 1), dtype=np.float32),
        linkage_matrix=np.empty((0, 4), dtype=np.float64),
        neuron_ids=["n1"],
        reorder_indices=np.array([0], dtype=np.intp),
        labels=np.array([2], dtype=np.int32),
        unassigned_neuron_ids=["n2"],
    )

    widget.update_cluster_assignments(result)

    assert widget._entries["n1"].cluster_id == 2
    assert widget._entries["n2"].cluster_id is None
    assert widget._entries["n3"].cluster_id == 7
    assert cluster_cells == [(1, None), (0, 2)]
    assert len(widget.state_changed.calls) == 1


def test_neuron_table_set_heatmap_layer_names_updates_entries_with_string_fallback() -> (
    None
):
    module = _import_neuron_table_module()
    widget = _make_widget(
        module,
        {
            1: module.NeuronEntry(file_id=1, subject="s1"),
            "n2": module.NeuronEntry(file_id="n2", subject="s2"),
        },
    )
    heatmap_cells: list[tuple[int, tuple[str, ...]]] = []
    widget._set_heatmap_cell = lambda row, layer_names: heatmap_cells.append(
        (row, tuple(layer_names))
    )

    widget.set_heatmap_layers_by_file_id(
        {
            "1": ["alpha Heatmap", "beta Heatmap"],
        }
    )

    assert widget._entries[1].heatmap_layer_names == (
        "alpha Heatmap",
        "beta Heatmap",
    )
    assert widget._entries["n2"].heatmap_layer_names == ()
    assert heatmap_cells == [
        (0, ("alpha Heatmap", "beta Heatmap")),
        (1, ()),
    ]
    assert len(widget.state_changed.calls) == 1


def test_neuron_table_export_state_includes_label_schema_fields() -> None:
    module = _import_neuron_table_module()
    widget = _make_widget(
        module,
        {
            "n1": module.NeuronEntry(
                file_id="n1",
                subject="s1",
                color=[0.1, 0.2, 0.3, 1.0],
                cluster_id=4,
                visible=False,
                label="projection",
                group="A",
                tags=("axon", "reviewed"),
                notes="keep",
            )
        },
    )

    state = widget.export_state()

    assert state["entries"] == [
        {
            "file_id": "n1",
            "subject": "s1",
            "color": [0.1, 0.2, 0.3, 1.0],
            "cluster_id": 4,
            "visible": False,
            "added_to_scene": False,
            "heatmap_layer_names": [],
            "label": "projection",
            "group": "A",
            "tags": ["axon", "reviewed"],
            "notes": "keep",
        }
    ]


def test_neuron_table_apply_state_restores_matching_label_fields() -> None:
    module = _import_neuron_table_module()
    widget = _make_widget(
        module,
        {
            "n1": module.NeuronEntry(file_id="n1", subject="s1"),
            "n2": module.NeuronEntry(file_id="n2", subject="s2"),
        },
    )
    widget._update_visibility_checkbox = lambda row, visible: None
    widget._update_color_swatch_for_row = lambda row, color: None
    widget._set_cluster_cell = lambda row, cluster_id: None
    widget._set_text_cell = lambda row, column, text, editable: None
    widget.colors_changed = _DummySignal()
    widget.visibility_changed = _DummySignal()

    widget.apply_state(
        {
            "entries": [
                {
                    "file_id": "n1",
                    "label": "projection",
                    "group": "A",
                    "tags": ["axon"],
                    "notes": "keep",
                    "cluster_id": 9,
                    "visible": False,
                    "color": [0.9, 0.8, 0.7, 1.0],
                }
            ]
        }
    )

    entry = widget._entries["n1"]
    assert entry.label == "projection"
    assert entry.group == "A"
    assert entry.tags == ("axon",)
    assert entry.notes == "keep"
    assert entry.cluster_id == 9
    assert entry.visible is False
    assert entry.color == [0.9, 0.8, 0.7, 1.0]
    assert widget._entries["n2"].label == ""


def test_neuron_table_apply_filters_intersects_cluster_and_heatmap_filters() -> None:
    module = _import_neuron_table_module()
    widget = _make_widget(
        module,
        {
            "n1": module.NeuronEntry(file_id="n1", subject="s1", cluster_id=1),
            "n2": module.NeuronEntry(file_id="n2", subject="s2", cluster_id=1),
            "n3": module.NeuronEntry(file_id="n3", subject="s3", cluster_id=2),
        },
    )

    widget.apply_filters(module.ClusterFilterSelection({1}), ["n2", "n3"])

    assert widget._table.hidden_rows == {
        0: True,
        1: False,
        2: True,
    }


def test_neuron_table_apply_filters_manual_heatmap_uses_string_fallback() -> None:
    module = _import_neuron_table_module()
    widget = _make_widget(
        module,
        {
            1: module.NeuronEntry(file_id=1, subject="s1"),
            "n2": module.NeuronEntry(file_id="n2", subject="s2"),
        },
    )

    widget.apply_filters(module.ClusterFilterSelection(), ["1"])

    assert widget._table.hidden_rows == {
        0: False,
        1: True,
    }


def test_neuron_table_select_file_ids_selects_multiple_visible_rows_once() -> None:
    module = _import_neuron_table_module()
    module.QItemSelection = _DummyItemSelection
    module.QItemSelectionModel = _DummyItemSelectionModel
    widget = _make_widget(
        module,
        {
            "n1": module.NeuronEntry(file_id="n1", subject="s1"),
            "n2": module.NeuronEntry(file_id="n2", subject="s2"),
            "n3": module.NeuronEntry(file_id="n3", subject="s3"),
        },
    )
    widget._table.setRowHidden(1, True)

    widget.select_file_ids(["n1", "n2", "n3", "n1", "missing"])

    assert widget.get_selected_file_ids() == ["n1", "n3"]
    assert widget.selection_changed.calls == [(["n1", "n3"],)]


def test_neuron_table_selected_file_ids_excludes_rows_hidden_after_selection() -> None:
    module = _import_neuron_table_module()
    widget = _make_widget(
        module,
        {
            "n1": module.NeuronEntry(file_id="n1", subject="s1"),
            "n2": module.NeuronEntry(file_id="n2", subject="s2"),
        },
    )
    widget._table._selected_rows = {0, 1}
    widget._table.setRowHidden(0, True)

    assert widget.get_selected_file_ids() == ["n2"]


def test_neuron_table_named_assignment_columns_preserve_prior_run_and_active_map() -> (
    None
):
    module = _import_neuron_table_module()
    store = ClusterAssignmentStore()
    first = store.add(
        name="Soma Location 1",
        assignments={"n1": 1, "n2": 2, "n3": 2},
        input_file_ids=["n1", "n2", "n3"],
    )
    store.add(
        name="Voxel Correlation 1",
        assignments={"n2": 1, "n3": 2},
        input_file_ids=["n2", "n3"],
        parent_assignment_id=first.assignment_id,
        parent_cluster_ids=[2],
    )
    widget = _make_widget(
        module,
        {
            "n1": module.NeuronEntry(file_id="n1", subject="s1"),
            "n2": module.NeuronEntry(file_id="n2", subject="s2"),
            "n3": module.NeuronEntry(file_id="n3", subject="s3"),
        },
    )
    widget._assignment_store = store
    widget._cluster_column_by_id = {}

    widget.refresh_cluster_assignments()
    widget.sort_by_cluster()

    assert widget._table.headers[9:] == [
        "Soma Location 1",
        "Voxel Correlation 1 (active)",
        "Color",
    ]
    assert widget.get_cluster_map() == {"n1": None, "n2": 1, "n3": 2}
    assert widget._table.sort_calls[-1] == (10, module.Qt.AscendingOrder)
    state = widget.export_state()
    assert state["version"] == 2
    assert len(state["cluster_assignments"]["sets"]) == 2
