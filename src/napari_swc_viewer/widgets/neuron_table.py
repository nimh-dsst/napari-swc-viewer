"""Neuron selection table widget.

Provides an interactive table for viewing and controlling neurons:
- Toggle neuron visibility on/off
- Track whether a neuron is currently added to the scene
- View neuron ID, subject, and cluster assignment
- Edit neuron colors via color picker
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..neuron_table_ops import (
    GRAY_RGBA,
    NeuronTableSummary,
    added_flags,
    cluster_filter_matches,
    cluster_ids_available,
    cluster_sort_value,
    recolor_cluster_turbo,
    summarize_neuron_table,
    visibility_for_selected_cluster,
)

if TYPE_CHECKING:
    from ..analysis.clustering import ClusterResult

logger = logging.getLogger(__name__)

# Column indices
COL_VISIBLE = 0
COL_ADDED = 1
COL_NEURON_ID = 2
COL_SUBJECT = 3
COL_CLUSTER = 4
COL_COLOR = 5


class _NumericSortItem(QTableWidgetItem):
    """Table item that sorts by numeric key stored in ``Qt.UserRole``."""

    def __lt__(self, other: QTableWidgetItem) -> bool:
        left = self.data(Qt.UserRole)
        right = other.data(Qt.UserRole)
        if left is not None and right is not None:
            return float(left) < float(right)
        return super().__lt__(other)


@dataclass
class NeuronEntry:
    """Per-neuron state tracked by the neuron selection table."""

    file_id: object
    subject: str
    color: list[float] = field(default_factory=lambda: list(GRAY_RGBA))
    cluster_id: int | None = None
    visible: bool = True
    added_to_scene: bool = False


class NeuronTableWidget(QWidget):
    """Interactive table for neuron selection, color editing, and visibility.

    Signals
    -------
    colors_changed : dict
        Emitted when neuron colors change. Payload is ``{file_id: [r,g,b,a]}``.
    visibility_changed : dict
        Emitted when neuron visibility changes. Payload is ``{file_id: bool}``.
    state_changed
        Emitted when tracked table state changes in a way that affects summary UI.
    """

    colors_changed = Signal(dict)
    visibility_changed = Signal(dict)
    selection_changed = Signal(list)
    state_changed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._entries: dict[object, NeuronEntry] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["Vis", "Added", "Neuron ID", "Subject", "Cluster", "Color"]
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.ExtendedSelection)

        # Column sizing
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(COL_VISIBLE, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_ADDED, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_NEURON_ID, QHeaderView.Stretch)
        header.setSectionResizeMode(COL_SUBJECT, QHeaderView.Stretch)
        header.setSectionResizeMode(COL_CLUSTER, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_COLOR, QHeaderView.ResizeToContents)

        self._table.verticalHeader().setVisible(False)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.setSortingEnabled(True)

        layout.addWidget(self._table)

    def populate(self, neurons: list[tuple[str, str]]) -> None:
        """Fill the table with neurons from a query result.

        Parameters
        ----------
        neurons : list[tuple[str, str]]
            List of (file_id, subject) tuples.
        """
        sorting_enabled = self._table.isSortingEnabled()
        self._table.setSortingEnabled(False)

        self._table.setRowCount(0)
        self._entries.clear()

        n = len(neurons)
        cmap = plt.get_cmap("turbo")
        colors = [list(cmap(t)) for t in np.linspace(0, 1, max(n, 1))]

        self._table.setRowCount(n)

        for row, (file_id, subject) in enumerate(neurons):
            color = colors[row] if row < len(colors) else list(GRAY_RGBA)
            entry = NeuronEntry(file_id=file_id, subject=subject, color=color)
            self._entries[file_id] = entry
            self._populate_row(row, entry)

        self._table.setSortingEnabled(sorting_enabled)
        self.state_changed.emit()

    def _populate_row(self, row: int, entry: NeuronEntry) -> None:
        """Populate a single table row from a NeuronEntry."""
        # Visible checkbox
        cb = QCheckBox()
        cb.setChecked(entry.visible)
        cb.stateChanged.connect(partial(self._on_visibility_toggled, entry.file_id))
        cb_widget = QWidget()
        cb_layout = QHBoxLayout(cb_widget)
        cb_layout.addWidget(cb)
        cb_layout.setAlignment(Qt.AlignCenter)
        cb_layout.setContentsMargins(0, 0, 0, 0)
        self._table.setCellWidget(row, COL_VISIBLE, cb_widget)

        # Added state
        self._set_added_cell(row, entry.added_to_scene)

        # Neuron ID
        id_item = QTableWidgetItem(str(entry.file_id))
        id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
        id_item.setData(Qt.UserRole, entry.file_id)
        self._table.setItem(row, COL_NEURON_ID, id_item)

        # Subject
        subj_item = QTableWidgetItem(entry.subject)
        subj_item.setFlags(subj_item.flags() & ~Qt.ItemIsEditable)
        self._table.setItem(row, COL_SUBJECT, subj_item)

        # Cluster
        self._set_cluster_cell(row, entry.cluster_id)

        # Color swatch button
        btn = QPushButton()
        btn.setFixedSize(24, 24)
        self._apply_color_style(btn, entry.color)
        btn.clicked.connect(partial(self._on_color_clicked, entry.file_id))
        self._table.setCellWidget(row, COL_COLOR, btn)

    def _set_added_cell(self, row: int, added: bool) -> None:
        """Set the Added column cell text and sort key."""
        item = self._table.item(row, COL_ADDED)
        if item is None:
            item = QTableWidgetItem()
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, COL_ADDED, item)

        item.setText("Yes" if added else "No")
        item.setData(Qt.UserRole, 1 if added else 0)

    def _set_cluster_cell(self, row: int, cluster_id: int | None) -> None:
        """Set cluster cell text and numeric sort key."""
        item = self._table.item(row, COL_CLUSTER)
        if item is None or not isinstance(item, _NumericSortItem):
            item = _NumericSortItem()
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, COL_CLUSTER, item)

        item.setText("" if cluster_id is None else str(cluster_id))
        item.setData(Qt.UserRole, cluster_sort_value(cluster_id))

    def _apply_color_style(self, btn: QPushButton, color: list[float]) -> None:
        """Set the button background to the given RGBA color."""
        r, g, b, a = [int(c * 255) for c in color[:4]]
        btn.setStyleSheet(
            f"background-color: rgba({r},{g},{b},{a}); border: 1px solid gray;"
        )

    def _on_color_clicked(self, file_id: str) -> None:
        """Open a color dialog when a color swatch is clicked."""
        entry = self._entries.get(file_id)
        if entry is None:
            return

        current = QColor.fromRgbF(*entry.color[:4])
        new_color = QColorDialog.getColor(
            current, self, "Choose Neuron Color", QColorDialog.ShowAlphaChannel
        )
        if not new_color.isValid():
            return

        rgba = [
            new_color.redF(),
            new_color.greenF(),
            new_color.blueF(),
            new_color.alphaF(),
        ]
        entry.color = rgba
        self._update_color_swatch(file_id)
        self.colors_changed.emit({file_id: rgba})

    def _on_selection_changed(self) -> None:
        """Emit selection_changed when table row selection changes."""
        self.selection_changed.emit(self.get_selected_file_ids())

    def _on_visibility_toggled(self, file_id: str, state: int) -> None:
        """Handle a visibility checkbox state change."""
        entry = self._entries.get(file_id)
        if entry is None:
            return

        entry.visible = bool(state)
        self.visibility_changed.emit({fid: e.visible for fid, e in self._entries.items()})
        self.state_changed.emit()

    def _file_id_from_row(self, row: int) -> object | None:
        """Resolve a row to its file_id using item metadata."""
        item = self._table.item(row, COL_NEURON_ID)
        if item is None:
            return None

        file_id = item.data(Qt.UserRole)
        if file_id is not None:
            return file_id

        text = item.text()
        return text if text else None

    def _iter_rows_with_file_ids(self) -> list[tuple[int, object]]:
        """Return all table rows that map to a neuron file_id."""
        out: list[tuple[int, object]] = []
        for row in range(self._table.rowCount()):
            file_id = self._file_id_from_row(row)
            if file_id is not None:
                out.append((row, file_id))
        return out

    def _file_id_to_row_map(self) -> dict[object, int]:
        """Build a map from file_id to current table row."""
        return {file_id: row for row, file_id in self._iter_rows_with_file_ids()}

    def _file_id_to_row(self, file_id: object) -> int | None:
        """Get the current table row for a given file_id."""
        return self._file_id_to_row_map().get(file_id)

    def _update_visibility_checkbox(self, row: int, visible: bool) -> None:
        """Set row visibility checkbox state without emitting checkbox signals."""
        cb_widget = self._table.cellWidget(row, COL_VISIBLE)
        if cb_widget is None:
            return

        cb = cb_widget.findChild(QCheckBox)
        if cb is None:
            return

        was_blocked = cb.blockSignals(True)
        cb.setChecked(visible)
        cb.blockSignals(was_blocked)

    def _update_color_swatch(self, file_id: str) -> None:
        """Update the color swatch button for a given neuron."""
        entry = self._entries.get(file_id)
        if entry is None:
            return

        row = self._file_id_to_row(file_id)
        if row is None:
            return

        self._update_color_swatch_for_row(row, entry.color)

    def _update_color_swatch_for_row(self, row: int, color: list[float]) -> None:
        """Update the color swatch button for one known table row."""
        btn = self._table.cellWidget(row, COL_COLOR)
        if btn is not None:
            self._apply_color_style(btn, color)

    # --- Public API ---

    def get_selected_file_ids(self) -> list[object]:
        """Return the file_ids of the currently selected rows."""
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()})
        selected: list[object] = []
        for row in rows:
            file_id = self._file_id_from_row(row)
            if file_id is not None:
                selected.append(file_id)
        return selected

    def get_color(self, file_id: str) -> list[float]:
        """Return the RGBA color for a neuron."""
        entry = self._entries.get(file_id)
        return list(entry.color) if entry else list(GRAY_RGBA)

    def get_full_color_map(self) -> dict[str, list[float]]:
        """Return a mapping of all file_ids to their current RGBA colors."""
        return {fid: list(e.color) for fid, e in self._entries.items()}

    def get_visibility_map(self) -> dict[str, bool]:
        """Return a mapping of all file_ids to their visibility state."""
        return {fid: e.visible for fid, e in self._entries.items()}

    def summary(self) -> NeuronTableSummary:
        """Return summary counts for the current table contents."""
        return summarize_neuron_table(
            {fid: entry.cluster_id for fid, entry in self._entries.items()},
            {fid: entry.added_to_scene for fid, entry in self._entries.items()},
            {fid: entry.visible for fid, entry in self._entries.items()},
        )

    def clear(self) -> None:
        """Clear all table rows and tracked neuron state."""
        sorting_enabled = self._table.isSortingEnabled()
        signals_blocked = self._table.blockSignals(True)
        self._table.setSortingEnabled(False)
        try:
            self._table.clearContents()
            self._table.setRowCount(0)
            self._entries.clear()
        finally:
            self._table.setSortingEnabled(sorting_enabled)
            self._table.blockSignals(signals_blocked)

        self.state_changed.emit()

    def set_added_file_ids(self, file_ids_in_scene: set[object] | list[object]) -> None:
        """Set whether each neuron is currently added to the scene."""
        flags = added_flags(self._entries.keys(), file_ids_in_scene)
        row_map = self._file_id_to_row_map()

        sorting_enabled = self._table.isSortingEnabled()
        self._table.setSortingEnabled(False)
        try:
            for file_id, added in flags.items():
                entry = self._entries.get(file_id)
                if entry is None:
                    continue

                entry.added_to_scene = added
                row = row_map.get(file_id)
                if row is not None:
                    self._set_added_cell(row, added)
        finally:
            self._table.setSortingEnabled(sorting_enabled)
        self.state_changed.emit()

    def available_cluster_ids(self) -> list[int]:
        """Return sorted unique cluster IDs in the table."""
        cluster_map = {fid: entry.cluster_id for fid, entry in self._entries.items()}
        return cluster_ids_available(cluster_map)

    def apply_cluster_filter(self, cluster_id: int | None) -> None:
        """Hide rows not in ``cluster_id``. ``None`` means show all rows."""
        cluster_map = {fid: entry.cluster_id for fid, entry in self._entries.items()}
        matches = cluster_filter_matches(cluster_map, cluster_id)
        for row, file_id in self._iter_rows_with_file_ids():
            self._table.setRowHidden(row, not matches.get(file_id, True))

    def hide_all_not_in_cluster(self, cluster_id: int) -> None:
        """Set visibility off for all neurons not in ``cluster_id``."""
        cluster_map = {fid: entry.cluster_id for fid, entry in self._entries.items()}
        visibility = visibility_for_selected_cluster(cluster_map, cluster_id)

        row_map = self._file_id_to_row_map()
        changed = False
        for file_id, visible in visibility.items():
            entry = self._entries.get(file_id)
            if entry is None:
                continue
            if entry.visible != visible:
                changed = True
            entry.visible = visible
            row = row_map.get(file_id)
            if row is not None:
                self._update_visibility_checkbox(row, visible)

        if changed:
            self.visibility_changed.emit(self.get_visibility_map())
            self.state_changed.emit()

    def set_all_visible(self) -> None:
        """Set visibility on for all neurons."""
        row_map = self._file_id_to_row_map()
        changed = False
        for file_id, entry in self._entries.items():
            if not entry.visible:
                changed = True
            entry.visible = True
            row = row_map.get(file_id)
            if row is not None:
                self._update_visibility_checkbox(row, True)

        if changed:
            self.visibility_changed.emit(self.get_visibility_map())
            self.state_changed.emit()

    def recolor_cluster_turbo(self, cluster_id: int, gray_others: bool = True) -> None:
        """Recolor selected cluster with turbo; optionally gray non-selected neurons."""
        cluster_map = {fid: entry.cluster_id for fid, entry in self._entries.items()}
        updates = recolor_cluster_turbo(cluster_map, cluster_id, gray_others=gray_others)
        if not updates:
            return

        row_map = self._file_id_to_row_map()
        changed: dict[str, list[float]] = {}
        for file_id, rgba in updates.items():
            entry = self._entries.get(file_id)
            if entry is None:
                continue
            if list(entry.color) == list(rgba):
                continue
            entry.color = list(rgba)
            row = row_map.get(file_id)
            if row is not None:
                self._update_color_swatch(file_id)
            changed[file_id] = list(rgba)

        if changed:
            self.colors_changed.emit(changed)

    def update_cluster_assignments(self, result: ClusterResult) -> None:
        """Update the Cluster column from a ClusterResult.

        Parameters
        ----------
        result : ClusterResult
            Clustering result containing neuron_ids and labels.
        """
        row_map = self._file_id_to_row_map()

        sorting_enabled = self._table.isSortingEnabled()
        self._table.setSortingEnabled(False)
        try:
            for neuron_id, label in zip(result.neuron_ids, result.labels):
                entry = self._entries.get(neuron_id)
                if entry is None:
                    continue
                entry.cluster_id = int(label)
                row = row_map.get(neuron_id)
                if row is not None:
                    self._set_cluster_cell(row, entry.cluster_id)
        finally:
            self._table.setSortingEnabled(sorting_enabled)
        self.state_changed.emit()

    def select_file_ids(self, file_ids: list[str]) -> None:
        """Programmatically select table rows matching *file_ids*.

        Temporarily blocks the ``selection_changed`` signal to avoid
        feedback loops (e.g. soma click → table select → signal → …).
        """
        row_map = self._file_id_to_row_map()

        self._table.blockSignals(True)
        try:
            self._table.clearSelection()
            for fid in file_ids:
                row = row_map.get(fid)
                if row is not None and not self._table.isRowHidden(row):
                    self._table.selectRow(row)
        finally:
            self._table.blockSignals(False)
        # Emit once after all rows are selected
        self.selection_changed.emit(self.get_selected_file_ids())

    def update_colors(
        self,
        color_map: dict[str, list[float]],
        *,
        emit_signal: bool = True,
    ) -> None:
        """Batch-update neuron colors from a color map.

        Emits a single ``colors_changed`` signal at the end when requested.

        Parameters
        ----------
        color_map : dict[str, list[float]]
            Mapping of file_id to RGBA color.
        """
        row_map = self._file_id_to_row_map()
        sorting_enabled = self._table.isSortingEnabled()
        self._table.setSortingEnabled(False)
        changed = {}
        try:
            for file_id, rgba in color_map.items():
                entry = self._entries.get(file_id)
                if entry is None:
                    continue
                new_color = list(rgba)
                if list(entry.color) == new_color:
                    continue
                entry.color = new_color
                row = row_map.get(file_id)
                if row is not None:
                    self._update_color_swatch_for_row(row, entry.color)
                changed[file_id] = list(entry.color)
        finally:
            self._table.setSortingEnabled(sorting_enabled)

        if emit_signal and changed:
            self.colors_changed.emit(changed)
