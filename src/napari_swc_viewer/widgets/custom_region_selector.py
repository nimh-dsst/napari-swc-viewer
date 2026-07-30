"""Searchable tri-state selector for synthetic custom region hierarchies."""

from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..isocortex_layers import (
    CustomRegionHierarchy,
    CustomRegionHierarchyNode,
    CustomRegionSelectionGroup,
)


class CustomRegionSelectorWidget(QWidget):
    """Display and select exact terminal regions in a synthetic hierarchy."""

    selection_changed = Signal(list)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._hierarchy: CustomRegionHierarchy | None = None
        self._terminal_items_by_id: dict[int, QTreeWidgetItem] = {}
        self._all_items: list[QTreeWidgetItem] = []
        self._selection_change_depth = 0
        self._empty_message = ""
        self._setup_ui()
        self.clear_with_message(
            "Load a compatible Allen atlas to inspect custom regions."
        )

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._message_label = QLabel("")
        self._message_label.setWordWrap(True)
        layout.addWidget(self._message_label)

        search_layout = QHBoxLayout()
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText(
            "Search custom regions by name, acronym, or ID..."
        )
        self._search_input.textChanged.connect(self._on_search_changed)
        search_layout.addWidget(self._search_input)

        self._clear_search_btn = QPushButton("Clear")
        self._clear_search_btn.clicked.connect(self._search_input.clear)
        search_layout.addWidget(self._clear_search_btn)
        layout.addLayout(search_layout)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Region", "Acronym", "ID"])
        header = self._tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._tree)

        info_layout = QHBoxLayout()
        self._selection_label = QLabel("Selected: 0 terminal regions")
        info_layout.addWidget(self._selection_label)

        self._clear_selection_btn = QPushButton("Clear Selection")
        self._clear_selection_btn.clicked.connect(self.clear_selection)
        info_layout.addWidget(self._clear_selection_btn)
        layout.addLayout(info_layout)

    def set_hierarchy(self, hierarchy: CustomRegionHierarchy) -> None:
        """Populate the tree from an immutable custom hierarchy."""
        terminal_ids = hierarchy.terminal_region_ids
        if len(terminal_ids) != len(set(terminal_ids)):
            raise ValueError("Custom hierarchy contains duplicate terminal region IDs.")
        self._hierarchy = hierarchy
        self._empty_message = ""
        self._selection_change_depth += 1
        self._tree.blockSignals(True)
        try:
            self._tree.clear()
            self._terminal_items_by_id.clear()
            self._all_items.clear()
            root_item = self._add_node(hierarchy.root, None)
            root_item.setExpanded(True)
        finally:
            self._tree.blockSignals(False)
            self._selection_change_depth -= 1

        version_suffix = (
            f" v{hierarchy.atlas_version}" if hierarchy.atlas_version else ""
        )
        self._message_label.setText(
            f"{hierarchy.atlas_name}{version_suffix}: "
            f"{hierarchy.terminal_region_count} terminal regions."
        )
        self._tree.setEnabled(True)
        self._search_input.setEnabled(True)
        self._clear_search_btn.setEnabled(True)
        self._clear_selection_btn.setEnabled(True)
        self._search_input.clear()
        self._update_selection_label()
        self._emit_selection_changed()

    def clear_with_message(self, message: str) -> None:
        """Clear the hierarchy and show an explanatory empty-state message."""
        self._hierarchy = None
        self._empty_message = str(message)
        self._tree.blockSignals(True)
        try:
            self._tree.clear()
        finally:
            self._tree.blockSignals(False)
        self._terminal_items_by_id.clear()
        self._all_items.clear()
        self._message_label.setText(str(message))
        self._tree.setEnabled(False)
        self._search_input.clear()
        self._search_input.setEnabled(False)
        self._clear_search_btn.setEnabled(False)
        self._clear_selection_btn.setEnabled(False)
        self._update_selection_label()
        self._emit_selection_changed()

    def has_hierarchy(self) -> bool:
        """Return whether a custom hierarchy is currently available."""
        return self._hierarchy is not None

    def unavailable_message(self) -> str:
        """Return the current empty-state explanation."""
        return self._empty_message

    def _add_node(
        self,
        node: CustomRegionHierarchyNode,
        parent_item: QTreeWidgetItem | None,
    ) -> QTreeWidgetItem:
        terminal_ids = node.terminal_region_ids
        label = (
            node.label if node.is_terminal else f"{node.label} ({len(terminal_ids)})"
        )
        columns = [
            label,
            node.acronym if node.is_terminal else "",
            str(node.region_id) if node.region_id is not None else "",
        ]
        if parent_item is None:
            item = QTreeWidgetItem(self._tree, columns)
        else:
            item = QTreeWidgetItem(parent_item, columns)

        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(0, Qt.Unchecked)
        item.setData(0, Qt.UserRole, node.region_id)
        self._all_items.append(item)
        if node.region_id is not None:
            normalized_id = int(node.region_id)
            if normalized_id in self._terminal_items_by_id:
                raise ValueError(
                    f"Custom hierarchy contains duplicate region ID {normalized_id}."
                )
            self._terminal_items_by_id[normalized_id] = item

        for child in node.children:
            self._add_node(child, item)
        return item

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0 or self._selection_change_depth > 0:
            return

        self._selection_change_depth += 1
        self._tree.blockSignals(True)
        try:
            state = item.checkState(0)
            if item.childCount() and state in (Qt.Checked, Qt.Unchecked):
                self._set_descendant_state(item, state)
            self._update_ancestor_states(item)
        finally:
            self._tree.blockSignals(False)
            self._selection_change_depth -= 1

        self._update_selection_label()
        self._emit_selection_changed()

    def _set_descendant_state(
        self,
        item: QTreeWidgetItem,
        state: Qt.CheckState,
    ) -> None:
        for index in range(item.childCount()):
            child = item.child(index)
            child.setCheckState(0, state)
            self._set_descendant_state(child, state)

    def _update_ancestor_states(self, item: QTreeWidgetItem) -> None:
        parent = item.parent()
        while parent is not None:
            states = [
                parent.child(index).checkState(0)
                for index in range(parent.childCount())
            ]
            if states and all(state == Qt.Checked for state in states):
                parent_state = Qt.Checked
            elif states and all(state == Qt.Unchecked for state in states):
                parent_state = Qt.Unchecked
            else:
                parent_state = Qt.PartiallyChecked
            parent.setCheckState(0, parent_state)
            parent = parent.parent()

    def clear_selection(self) -> None:
        """Uncheck every hierarchy node and emit one selection update."""
        self._selection_change_depth += 1
        self._tree.blockSignals(True)
        try:
            for item in self._all_items:
                item.setCheckState(0, Qt.Unchecked)
        finally:
            self._tree.blockSignals(False)
            self._selection_change_depth -= 1
        self._update_selection_label()
        self._emit_selection_changed()

    def get_selected_region_ids(self) -> list[int]:
        """Return sorted, deduplicated checked terminal region IDs."""
        return sorted(
            region_id
            for region_id, item in self._terminal_items_by_id.items()
            if item.checkState(0) == Qt.Checked
        )

    def get_selected_region_groups(
        self,
    ) -> tuple[CustomRegionSelectionGroup, ...]:
        """Return checked terminal regions grouped in hierarchy order."""
        hierarchy = self._hierarchy
        if hierarchy is None:
            return ()

        selected_ids = set(self.get_selected_region_ids())
        groups: list[CustomRegionSelectionGroup] = []
        for group_node in hierarchy.root.children:
            selected_leaves = tuple(
                leaf
                for leaf in group_node.children
                if leaf.region_id is not None
                and int(leaf.region_id) in selected_ids
            )
            if not selected_leaves:
                continue
            groups.append(
                CustomRegionSelectionGroup(
                    label=group_node.label,
                    region_ids=tuple(
                        int(leaf.region_id)
                        for leaf in selected_leaves
                        if leaf.region_id is not None
                    ),
                    acronyms=tuple(leaf.acronym for leaf in selected_leaves),
                )
            )
        return tuple(groups)

    def _update_selection_label(self) -> None:
        count = len(self.get_selected_region_ids())
        self._selection_label.setText(f"Selected: {count} terminal regions")

    def _emit_selection_changed(self) -> None:
        self.selection_changed.emit(self.get_selected_region_ids())

    def _on_search_changed(self, text: str) -> None:
        search_text = str(text).strip().casefold()

        def update_visibility(item: QTreeWidgetItem) -> bool:
            child_visible = False
            for index in range(item.childCount()):
                if update_visibility(item.child(index)):
                    child_visible = True
            matches = not search_text or any(
                search_text in item.text(column).casefold() for column in range(3)
            )
            visible = matches or child_visible
            item.setHidden(not visible)
            if search_text and child_visible:
                item.setExpanded(True)
            return visible

        for index in range(self._tree.topLevelItemCount()):
            update_visibility(self._tree.topLevelItem(index))
