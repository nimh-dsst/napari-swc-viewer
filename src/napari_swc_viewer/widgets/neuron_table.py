"""Neuron selection table widget.

Provides an interactive table for viewing and controlling neurons:
- Toggle neuron visibility on/off
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

if TYPE_CHECKING:
    from ..analysis.clustering import ClusterResult

logger = logging.getLogger(__name__)

# Column indices
COL_VISIBLE = 0
COL_NEURON_ID = 1
COL_SUBJECT = 2
COL_CLUSTER = 3
COL_COLOR = 4


@dataclass
class NeuronEntry:
    """Per-neuron state tracked by the neuron selection table."""

    file_id: str
    subject: str
    color: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 1.0])
    cluster_id: int | None = None
    visible: bool = True


class NeuronTableWidget(QWidget):
    """Interactive table for neuron selection, color editing, and visibility.

    Signals
    -------
    colors_changed : dict
        Emitted when neuron colors change. Payload is ``{file_id: [r,g,b,a]}``.
    visibility_changed : dict
        Emitted when neuron visibility changes. Payload is ``{file_id: bool}``.
    """

    colors_changed = Signal(dict)
    visibility_changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._entries: dict[str, NeuronEntry] = {}
        self._row_to_file_id: list[str] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Vis", "Neuron ID", "Subject", "Cluster", "Color"]
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.ExtendedSelection)

        # Column sizing
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(COL_VISIBLE, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_NEURON_ID, QHeaderView.Stretch)
        header.setSectionResizeMode(COL_SUBJECT, QHeaderView.Stretch)
        header.setSectionResizeMode(COL_CLUSTER, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(COL_COLOR, QHeaderView.ResizeToContents)

        self._table.verticalHeader().setVisible(False)

        layout.addWidget(self._table)

    def populate(self, neurons: list[tuple[str, str]]) -> None:
        """Fill the table with neurons from a query result.

        Parameters
        ----------
        neurons : list[tuple[str, str]]
            List of (file_id, subject) tuples.
        """
        self._table.setRowCount(0)
        self._entries.clear()
        self._row_to_file_id.clear()

        n = len(neurons)
        cmap = plt.get_cmap("turbo")
        colors = [list(cmap(t)) for t in np.linspace(0, 1, max(n, 1))]

        self._table.setRowCount(n)

        for row, (file_id, subject) in enumerate(neurons):
            color = colors[row] if row < len(colors) else [0.5, 0.5, 0.5, 1.0]
            entry = NeuronEntry(
                file_id=file_id, subject=subject, color=color
            )
            self._entries[file_id] = entry
            self._row_to_file_id.append(file_id)
            self._populate_row(row, entry)

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

        # Neuron ID
        id_item = QTableWidgetItem(entry.file_id)
        id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
        self._table.setItem(row, COL_NEURON_ID, id_item)

        # Subject
        subj_item = QTableWidgetItem(entry.subject)
        subj_item.setFlags(subj_item.flags() & ~Qt.ItemIsEditable)
        self._table.setItem(row, COL_SUBJECT, subj_item)

        # Cluster
        cluster_text = str(entry.cluster_id) if entry.cluster_id is not None else ""
        cluster_item = QTableWidgetItem(cluster_text)
        cluster_item.setFlags(cluster_item.flags() & ~Qt.ItemIsEditable)
        self._table.setItem(row, COL_CLUSTER, cluster_item)

        # Color swatch button
        btn = QPushButton()
        btn.setFixedSize(24, 24)
        self._apply_color_style(btn, entry.color)
        btn.clicked.connect(partial(self._on_color_clicked, entry.file_id))
        self._table.setCellWidget(row, COL_COLOR, btn)

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

        rgba = [new_color.redF(), new_color.greenF(), new_color.blueF(), new_color.alphaF()]
        entry.color = rgba
        self._update_color_swatch(file_id)
        self.colors_changed.emit({file_id: rgba})

    def _on_visibility_toggled(self, file_id: str, state: int) -> None:
        """Handle a visibility checkbox state change."""
        entry = self._entries.get(file_id)
        if entry is None:
            return

        entry.visible = state == Qt.Checked
        self.visibility_changed.emit(
            {fid: e.visible for fid, e in self._entries.items()}
        )

    def _update_color_swatch(self, file_id: str) -> None:
        """Update the color swatch button for a given neuron."""
        entry = self._entries.get(file_id)
        if entry is None:
            return

        row = self._file_id_to_row(file_id)
        if row is None:
            return

        btn = self._table.cellWidget(row, COL_COLOR)
        if btn is not None:
            self._apply_color_style(btn, entry.color)

    def _file_id_to_row(self, file_id: str) -> int | None:
        """Get the table row index for a given file_id."""
        try:
            return self._row_to_file_id.index(file_id)
        except ValueError:
            return None

    # --- Public API ---

    def get_selected_file_ids(self) -> list[str]:
        """Return the file_ids of the currently selected rows."""
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()})
        return [self._row_to_file_id[r] for r in rows if r < len(self._row_to_file_id)]

    def get_color(self, file_id: str) -> list[float]:
        """Return the RGBA color for a neuron."""
        entry = self._entries.get(file_id)
        return list(entry.color) if entry else [0.5, 0.5, 0.5, 1.0]

    def get_full_color_map(self) -> dict[str, list[float]]:
        """Return a mapping of all file_ids to their current RGBA colors."""
        return {fid: list(e.color) for fid, e in self._entries.items()}

    def get_visibility_map(self) -> dict[str, bool]:
        """Return a mapping of all file_ids to their visibility state."""
        return {fid: e.visible for fid, e in self._entries.items()}

    def update_cluster_assignments(self, result: ClusterResult) -> None:
        """Update the Cluster column from a ClusterResult.

        Parameters
        ----------
        result : ClusterResult
            Clustering result containing neuron_ids and labels.
        """
        for neuron_id, label in zip(result.neuron_ids, result.labels):
            entry = self._entries.get(neuron_id)
            if entry is None:
                continue
            entry.cluster_id = int(label)
            row = self._file_id_to_row(neuron_id)
            if row is not None:
                item = self._table.item(row, COL_CLUSTER)
                if item is not None:
                    item.setText(str(int(label)))

    def update_colors(self, color_map: dict[str, list[float]]) -> None:
        """Batch-update neuron colors from a color map.

        Emits a single ``colors_changed`` signal at the end.

        Parameters
        ----------
        color_map : dict[str, list[float]]
            Mapping of file_id to RGBA color.
        """
        changed = {}
        for file_id, rgba in color_map.items():
            entry = self._entries.get(file_id)
            if entry is None:
                continue
            entry.color = list(rgba)
            self._update_color_swatch(file_id)
            changed[file_id] = list(rgba)

        if changed:
            self.colors_changed.emit(changed)
