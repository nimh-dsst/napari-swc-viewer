"""Termini section for the Data tab.

Finds childless nodes of the selected types and adds them as a points layer.
Its second job is a data-quality gate on the neuron table: reconstructions that
produce no termini — overwhelmingly the ones whose neurites are all typed
``Undefined`` — are the neurons to drop before clustering, so the section can
select either group directly in the table.

The two correctness rules from ``terminals.py`` still apply and are enforced
there: the node-type restriction narrows only which termini are *reported*,
never the child lookup, and every graph computation is scoped per ``file_id``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import QThread
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..swc import NodeType
from .collapsible_section import CollapsibleSection
from .node_type_selector import NodeTypeSelectorComboBox

if TYPE_CHECKING:
    import napari
    from brainglobe_atlasapi import BrainGlobeAtlas

    from ..db import NeuronDatabase

logger = logging.getLogger(__name__)

TERMINI_SCOPE_WHOLE = "whole"
TERMINI_SCOPE_CURRENT = "current"
TERMINI_SCOPE_SELECTED = "selected"

TERMINI_GROUP_WITH = "with"
TERMINI_GROUP_WITHOUT = "without"

_GROUP_LABELS = {
    TERMINI_GROUP_WITH: "with termini",
    TERMINI_GROUP_WITHOUT: "lacking termini",
}

# Some neurons in isocortex_total_right_brainglobe_flatmap.parquet have
# dendritic projections typed 2 (axon), so an "Axon" result is really an
# axon-typed result. See AGENTS.md and USE_CASES.md UC-010.
TERMINI_TYPE_CAUTION = (
    "Caution: 'Axon' here means axon-typed, not verified axon. Some neurons "
    "have dendritic projections typed as Axon, so their dendrite tips are "
    "reported as axon termini. Treat counts as an upper bound and check the "
    "points visually before drawing a conclusion."
)
TERMINI_TYPE_CAUTION_DETAIL = (
    "Found 2026-08-05 by visually inspecting termini in napari. This is a "
    "defect in the source SWC annotations, not in the detection: a reported "
    "node genuinely has no children, but the type on it does not reliably "
    "identify the compartment. The number of affected neurons has not been "
    "measured. Node types do not partition these neurons into clean axon and "
    "dendrite subtrees. See AGENTS.md and USE_CASES.md UC-010."
)
# Amber reads as a warning against both the light and dark napari themes.
_CAUTION_STYLE = "color: #d9822b; font-weight: bold;"


class TerminiSectionWidget(QWidget):
    """Collapsible Termini section wired to the neuron table.

    The widget owns its own worker thread and status readout so it can live on
    the Data tab, next to the table it filters, without depending on the
    Analysis tab's progress section.
    """

    def __init__(self, viewer: "napari.Viewer", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._db: NeuronDatabase | None = None
        self._parquet_path: str | None = None
        self._atlas: BrainGlobeAtlas | None = None

        self._worker_thread: QThread | None = None
        self._current_worker: object | None = None

        self._current_table_file_ids_provider = None
        self._selected_table_file_ids_provider = None
        self._table_color_map_provider = None
        self._select_table_file_ids_callback = None

        self._skipped_terminus_file_ids: list[str] = []
        # ``None`` means the run covered every cell in the Parquet.
        self._termini_analyzed_file_ids: set[str] | None = None
        self._termini_file_ids_with: set[str] = set()
        self._termini_run_complete = False

        self._setup_ui()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    def set_database(self, db: NeuronDatabase) -> None:
        """Set the database connection backing terminus detection."""
        self._db = db
        self._parquet_path = str(db.parquet_path)
        self._reset_selection_state()
        self._update_button_states()

    def set_atlas(self, atlas: BrainGlobeAtlas) -> None:
        """Set the atlas used to scale the termini points layer."""
        self._atlas = atlas

    def set_current_table_file_ids_provider(self, provider) -> None:
        """Set a callback returning the current neuron-table file IDs."""
        self._current_table_file_ids_provider = provider

    def set_selected_table_file_ids_provider(self, provider) -> None:
        """Set a callback returning explicitly selected table-row file IDs."""
        self._selected_table_file_ids_provider = provider

    def set_table_color_map_provider(self, provider) -> None:
        """Set a callback returning the table's ``file_id`` to RGBA mapping."""
        self._table_color_map_provider = provider

    def set_select_table_file_ids_callback(self, callback) -> None:
        """Set the callback that selects table rows for the given file IDs."""
        self._select_table_file_ids_callback = callback

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        self._section = CollapsibleSection("Termini", expanded=False)
        layout = self._section.content_layout()

        intro = QLabel(
            "Finds childless nodes of the selected types. Neurons whose "
            "neurites are all Undefined contribute none and are reported as "
            "skipped. Use Select in Table to select the neurons with or "
            "without termini, then Remove Selected From Table to drop them "
            "before clustering."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # The node types in these datasets are not trustworthy compartment
        # labels, so the result must not be read as "these are axons".
        self._termini_caution_label = QLabel(TERMINI_TYPE_CAUTION)
        self._termini_caution_label.setWordWrap(True)
        self._termini_caution_label.setStyleSheet(_CAUTION_STYLE)
        self._termini_caution_label.setToolTip(TERMINI_TYPE_CAUTION_DETAIL)
        layout.addWidget(self._termini_caution_label)

        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Neurons:"))
        self._termini_scope_combo = QComboBox()
        self._termini_scope_combo.addItem("Whole Parquet", TERMINI_SCOPE_WHOLE)
        self._termini_scope_combo.addItem("Current Table", TERMINI_SCOPE_CURRENT)
        self._termini_scope_combo.addItem("Selected Rows", TERMINI_SCOPE_SELECTED)
        # The section exists to prune the table, so default to the table.
        self._termini_scope_combo.setCurrentIndex(1)
        scope_row.addWidget(self._termini_scope_combo)
        layout.addLayout(scope_row)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Node types:"))
        self._termini_node_type_combo = NodeTypeSelectorComboBox()
        self._termini_node_type_combo.set_selected_node_types((NodeType.AXON,))
        type_row.addWidget(self._termini_node_type_combo)
        layout.addLayout(type_row)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Point size:"))
        self._termini_point_size_spin = QDoubleSpinBox()
        self._termini_point_size_spin.setRange(1.0, 500.0)
        self._termini_point_size_spin.setValue(20.0)
        self._termini_point_size_spin.setDecimals(1)
        size_row.addWidget(self._termini_point_size_spin)
        layout.addLayout(size_row)

        self._find_termini_btn = QPushButton("Find Termini")
        self._find_termini_btn.setEnabled(False)
        self._find_termini_btn.clicked.connect(self._run_terminus_detection)
        layout.addWidget(self._find_termini_btn)

        self._termini_coverage_label = QLabel("")
        self._termini_coverage_label.setWordWrap(True)
        layout.addWidget(self._termini_coverage_label)

        select_row = QHBoxLayout()
        select_row.addWidget(QLabel("Select in table:"))
        self._termini_group_combo = QComboBox()
        self._termini_group_combo.addItem("Neurons with termini", TERMINI_GROUP_WITH)
        self._termini_group_combo.addItem(
            "Neurons lacking termini", TERMINI_GROUP_WITHOUT
        )
        select_row.addWidget(self._termini_group_combo)
        self._select_termini_group_btn = QPushButton("Select in Table")
        self._select_termini_group_btn.setEnabled(False)
        self._select_termini_group_btn.clicked.connect(self._select_terminus_group)
        select_row.addWidget(self._select_termini_group_btn)
        layout.addLayout(select_row)

        self._copy_skipped_termini_btn = QPushButton("Copy Skipped Neuron IDs")
        self._copy_skipped_termini_btn.setEnabled(False)
        self._copy_skipped_termini_btn.clicked.connect(
            self._copy_skipped_terminus_file_ids
        )
        layout.addWidget(self._copy_skipped_termini_btn)

        self._termini_progress_bar = QProgressBar()
        self._termini_progress_bar.setVisible(False)
        layout.addWidget(self._termini_progress_bar)

        self._termini_status_label = QLabel("")
        self._termini_status_label.setWordWrap(True)
        layout.addWidget(self._termini_status_label)

        outer_layout.addWidget(self._section)

    def _update_button_states(self) -> None:
        """Enable or disable the section's buttons for the current state."""
        busy = self._worker_thread is not None and self._worker_thread.isRunning()
        # Terminus detection is pure topology, so it needs no atlas.
        self._find_termini_btn.setEnabled(self._db is not None and not busy)
        self._select_termini_group_btn.setEnabled(
            self._termini_run_complete and not busy
        )

    # ------------------------------------------------------------------
    # Table providers
    # ------------------------------------------------------------------
    def _current_table_file_ids(self) -> list[object]:
        """Return the file IDs currently present in the neuron table."""
        provider = self._current_table_file_ids_provider
        if provider is None:
            return []
        return list(provider())

    def _selected_table_file_ids(self) -> list[str]:
        """Return the explicitly selected table-row file IDs."""
        provider = self._selected_table_file_ids_provider
        if provider is None:
            return []
        return list(dict.fromkeys(str(value) for value in provider()))

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def _selected_terminus_scope(self) -> str:
        """Return the neuron scope selected for terminus detection."""
        combo = getattr(self, "_termini_scope_combo", None)
        data_getter = getattr(combo, "currentData", None)
        if callable(data_getter):
            value = data_getter()
            if value in (
                TERMINI_SCOPE_WHOLE,
                TERMINI_SCOPE_CURRENT,
                TERMINI_SCOPE_SELECTED,
            ):
                return str(value)
        return TERMINI_SCOPE_CURRENT

    def _selected_terminus_node_types(self) -> tuple[int, ...] | None:
        """Return the node-type filter for terminus detection."""
        combo = getattr(self, "_termini_node_type_combo", None)
        getter = getattr(combo, "selected_node_types", None)
        if callable(getter):
            return getter()
        return (NodeType.AXON,)

    def _selected_terminus_group(self) -> str:
        """Return which detected group the table selection should target."""
        combo = getattr(self, "_termini_group_combo", None)
        data_getter = getattr(combo, "currentData", None)
        if callable(data_getter):
            value = data_getter()
            if value in (TERMINI_GROUP_WITH, TERMINI_GROUP_WITHOUT):
                return str(value)
        return TERMINI_GROUP_WITH

    def _resolve_terminus_file_ids(self) -> tuple[bool, list[str] | None]:
        """Resolve which neurons terminus detection should cover."""
        scope = self._selected_terminus_scope()
        if scope == TERMINI_SCOPE_WHOLE:
            return True, None

        if scope == TERMINI_SCOPE_SELECTED:
            file_ids = self._selected_table_file_ids()
            label = "No table rows are selected"
        else:
            file_ids = [str(value) for value in self._current_table_file_ids()]
            label = "Current table is empty"

        if file_ids:
            return True, file_ids

        self._termini_status_label.setText(
            f"{label}; switch the termini scope to Whole Parquet or populate "
            "the table first."
        )
        return False, None

    def _reset_selection_state(self) -> None:
        """Discard results from a previous run."""
        self._termini_run_complete = False
        self._termini_file_ids_with = set()
        self._termini_analyzed_file_ids = None
        self._skipped_terminus_file_ids = []

    def _run_terminus_detection(self) -> None:
        """Start terminus detection in a background thread."""
        if self._db is None or self._parquet_path is None:
            return
        if self._worker_thread is not None and self._worker_thread.isRunning():
            return

        node_types = self._selected_terminus_node_types()
        if node_types is not None and not node_types:
            self._termini_status_label.setText(
                "Select at least one node type before finding termini."
            )
            return

        proceed, file_ids = self._resolve_terminus_file_ids()
        if not proceed:
            return

        from ..workers import TerminusWorker

        self._termini_coverage_label.setText("")
        self._copy_skipped_termini_btn.setEnabled(False)
        self._reset_selection_state()
        # Remember the analyzed scope so the group selection can say which
        # table rows the run never covered instead of miscounting them.
        self._termini_analyzed_file_ids = (
            None if file_ids is None else {str(value) for value in file_ids}
        )

        worker = TerminusWorker(
            parquet_path=self._parquet_path,
            file_ids=file_ids,
            node_types=(list(node_types) if node_types is not None else None),
        )
        self._termini_status_label.setText("Finding termini...")
        self._start_background_worker(worker, self._on_termini_finished)

    def _start_background_worker(self, worker, finished_slot) -> None:
        """Wire up and start a background worker in a QThread."""
        thread = QThread()
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(finished_slot)
        worker.finished.connect(thread.quit)
        worker.error.connect(self._on_error)
        worker.error.connect(thread.quit)
        thread.finished.connect(self._on_thread_finished)
        thread.finished.connect(self._update_button_states)
        thread.finished.connect(thread.deleteLater)

        # Keep references to prevent garbage collection
        self._worker_thread = thread
        self._current_worker = worker

        self._termini_progress_bar.setVisible(True)
        self._termini_progress_bar.setRange(0, 0)  # indeterminate
        self._update_button_states()

        thread.start()

    def _on_progress(self, step_name: str, current: int, total: int) -> None:
        """Handle progress updates from the terminus worker."""
        self._termini_status_label.setText(f"Step {current}/{total}: {step_name}")
        self._termini_progress_bar.setRange(0, total)
        self._termini_progress_bar.setValue(current)

    def _on_error(self, message: str) -> None:
        """Handle worker errors."""
        self._termini_progress_bar.setVisible(False)
        self._termini_status_label.setText(f"Error: {message}")
        self._update_button_states()
        logger.error(f"Terminus detection error: {message}")

    def _on_thread_finished(self) -> None:
        """Clear worker references after the thread has stopped."""
        self._worker_thread = None
        self._current_worker = None

    def _on_termini_finished(self, frame, coverage) -> None:
        """Add a points layer for the detected termini and report coverage."""
        self._termini_progress_bar.setVisible(False)
        self._termini_status_label.setText("Terminus detection complete.")

        # Always show coverage: a node-type filter silently drops neurons that
        # never use those types, and that exclusion must stay visible.
        summary = coverage.summary()
        skipped = list(getattr(coverage, "file_ids_without", []))
        if getattr(coverage, "file_ids_without_truncated", False):
            summary += (
                f" Only the first {len(skipped):,} skipped neuron IDs are listed."
            )
        self._termini_coverage_label.setText(summary)
        self._skipped_terminus_file_ids = skipped
        self._copy_skipped_termini_btn.setEnabled(bool(skipped))

        # An empty result is still a completed run: every analyzed neuron then
        # belongs to the "lacking termini" group, which is the group worth
        # removing. Derive that group from the frame rather than from
        # ``coverage.file_ids_without``, which is capped at 200 entries.
        empty = frame is None or len(frame) == 0
        file_ids = [] if empty else frame["file_id"].astype(str).tolist()
        self._termini_file_ids_with = set(file_ids)
        self._termini_run_complete = True
        self._update_button_states()

        if empty:
            return

        coords = frame[["x", "y", "z"]].to_numpy(dtype=float)

        # A whole-Parquet run yields millions of points, so only build a
        # per-point color array when table colors actually vary.
        color_map = self._table_color_map() or {}
        default_rgba = [1.0, 0.35, 0.0, 1.0]
        if color_map:
            colors = np.array(
                [
                    list(color_map.get(file_id, default_rgba))[:4]
                    for file_id in file_ids
                ],
                dtype=float,
            )
        else:
            colors = default_rgba

        scale = None
        if self._atlas is not None:
            scale = [1.0 / res for res in self._atlas.resolution]

        name = "Termini"
        node_types = self._selected_terminus_node_types()
        selection_text = NodeTypeSelectorComboBox.selection_text(node_types)
        if selection_text:
            name = f"Termini ({selection_text})"

        existing = [layer for layer in self._viewer.layers if layer.name == name]
        for layer in existing:
            self._viewer.layers.remove(layer)

        self._viewer.add_points(
            coords,
            size=float(self._termini_point_size_spin.value()),
            face_color=colors,
            border_color="white",
            border_width=0.1,
            name=name,
            opacity=0.9,
            scale=scale,
            metadata={
                "file_ids_per_point": file_ids,
                "node_ids": frame["node_id"].astype(int).tolist(),
                "point_types": frame["type"].astype(int).tolist(),
                "node_type_labels": NodeTypeSelectorComboBox.metadata_labels(
                    node_types
                ),
                "coverage_summary": summary,
                "skipped_file_ids": skipped,
            },
        )

    def _table_color_map(self) -> dict[str, list[float]]:
        """Return the table's current per-neuron colors, keyed by string ID."""
        provider = self._table_color_map_provider
        if provider is None:
            return {}
        return {str(key): value for key, value in dict(provider()).items()}

    # ------------------------------------------------------------------
    # Table selection
    # ------------------------------------------------------------------
    def _select_terminus_group(self) -> None:
        """Select the table rows for the chosen terminus group."""
        if not self._termini_run_complete:
            self._termini_status_label.setText(
                "Run Find Termini before selecting neurons in the table."
            )
            return

        callback = self._select_table_file_ids_callback
        if callback is None:
            return

        table_file_ids = self._current_table_file_ids()
        if not table_file_ids:
            self._termini_status_label.setText(
                "The neuron table is empty; nothing to select."
            )
            return

        group = self._selected_terminus_group()
        want_termini = group == TERMINI_GROUP_WITH
        analyzed = self._termini_analyzed_file_ids

        # Keep the table's own file_id objects: ``select_file_ids`` matches by
        # exact equality, so stringified IDs from the frame would match nothing.
        matches: list[object] = []
        out_of_scope = 0
        for file_id in table_file_ids:
            key = str(file_id)
            if analyzed is not None and key not in analyzed:
                out_of_scope += 1
                continue
            if (key in self._termini_file_ids_with) == want_termini:
                matches.append(file_id)

        label = _GROUP_LABELS[group]
        scope_note = (
            ""
            if not out_of_scope
            else (
                f" {out_of_scope:,} table row(s) were outside the analyzed "
                "scope; re-run Find Termini to cover them."
            )
        )

        if not matches:
            self._termini_status_label.setText(
                f"No table neurons are {label}; the selection is unchanged.{scope_note}"
            )
            return

        callback(matches)

        # Rows hidden by the cluster filter cannot be selected, so report the
        # count Qt actually applied rather than the count requested.
        selected = len(self._selected_table_file_ids())
        message = (
            f"Selected {selected:,} of {len(table_file_ids):,} table neurons {label}."
        )
        if selected < len(matches):
            message += (
                f" {len(matches) - selected:,} matching row(s) are hidden by a "
                "filter and were not selected."
            )
        self._termini_status_label.setText(message + scope_note)

    def _copy_skipped_terminus_file_ids(self) -> None:
        """Put the skipped neuron IDs on the clipboard for inspection."""
        skipped = list(getattr(self, "_skipped_terminus_file_ids", []))
        if not skipped:
            return

        from qtpy.QtWidgets import QApplication

        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText("\n".join(skipped))
        self._termini_status_label.setText(
            f"Copied {len(skipped):,} skipped neuron IDs to the clipboard."
        )
