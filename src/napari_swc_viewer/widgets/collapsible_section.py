"""Shared collapsible section widget for dense plugin layouts."""

from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """Simple collapsible section for dense tab layouts."""

    expanded_changed = Signal(bool)

    def __init__(
        self,
        title: str,
        *,
        expanded: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._title = title

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._toggle_button = QPushButton()
        self._toggle_button.setCheckable(True)
        self._toggle_button.setChecked(expanded)
        self._toggle_button.setFlat(True)
        self._toggle_button.setStyleSheet(
            "text-align: left; font-weight: bold; padding: 4px 0;"
        )
        self._toggle_button.toggled.connect(self._set_expanded)
        layout.addWidget(self._toggle_button)

        self._content_widget = QWidget()
        self._content_layout = QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(12, 0, 0, 0)
        layout.addWidget(self._content_widget)

        self._set_expanded(expanded)

    def content_layout(self) -> QVBoxLayout:
        """Return the layout used for the section content."""
        return self._content_layout

    def is_expanded(self) -> bool:
        """Return whether the section content is currently shown."""
        return bool(self._toggle_button.isChecked())

    def set_expanded(self, expanded: bool) -> None:
        """Expand or collapse the section, keeping the toggle button in sync."""
        expanded = bool(expanded)
        if self._toggle_button.isChecked() == expanded:
            return
        # Drives ``_set_expanded`` through the button's ``toggled`` signal.
        self._toggle_button.setChecked(expanded)

    def _set_expanded(self, expanded: bool) -> None:
        """Show or hide the section content."""
        prefix = "[-]" if expanded else "[+]"
        self._toggle_button.setText(f"{prefix} {self._title}")
        self._content_widget.setVisible(expanded)
        self.expanded_changed.emit(bool(expanded))
