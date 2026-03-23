"""Tests for Analysis tab region dropdown population."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd


class _Signal:
    """Minimal stand-in for qtpy.QtCore.Signal."""

    def __init__(self, *_args, **_kwargs) -> None:
        return None


class _DummyQtObject:
    """Minimal stand-in for Qt widget/core/gui classes."""

    def __init__(self, *_args, **_kwargs) -> None:
        return None


def _import_analysis_tab_module():
    """Import ``analysis_tab.py`` with stubbed Qt/matplotlib dependencies."""
    backend_module = types.ModuleType("matplotlib.backends.backend_qtagg")
    backend_module.FigureCanvasQTAgg = _DummyQtObject

    figure_module = types.ModuleType("matplotlib.figure")
    figure_module.Figure = _DummyQtObject

    qtcore_module = types.ModuleType("qtpy.QtCore")
    qtcore_module.QThread = _DummyQtObject
    qtcore_module.Qt = types.SimpleNamespace()
    qtcore_module.Signal = _Signal

    qtgui_module = types.ModuleType("qtpy.QtGui")
    qtgui_module.QColor = _DummyQtObject
    qtgui_module.QIcon = _DummyQtObject
    qtgui_module.QPixmap = _DummyQtObject

    qtwidgets_module = types.ModuleType("qtpy.QtWidgets")
    for name in (
        "QComboBox",
        "QDoubleSpinBox",
        "QGroupBox",
        "QHBoxLayout",
        "QLabel",
        "QProgressBar",
        "QPushButton",
        "QSpinBox",
        "QVBoxLayout",
        "QWidget",
    ):
        setattr(qtwidgets_module, name, _DummyQtObject)

    replacements = {
        "matplotlib.pyplot": types.ModuleType("matplotlib.pyplot"),
        "seaborn": types.ModuleType("seaborn"),
        "matplotlib.backends.backend_qtagg": backend_module,
        "matplotlib.figure": figure_module,
        "qtpy.QtCore": qtcore_module,
        "qtpy.QtGui": qtgui_module,
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
            / "analysis_tab.py"
        )
        spec = importlib.util.spec_from_file_location(
            "analysis_tab_test_module",
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


class _DummyCombo:
    """Small stand-in for the QComboBox API used by AnalysisTabWidget."""

    def __init__(self, current_text: str = "") -> None:
        self.items: list[str] = []
        self._current_text = current_text

    def currentText(self) -> str:
        return self._current_text

    def blockSignals(self, _blocked: bool) -> None:
        return None

    def clear(self) -> None:
        self.items = []

    def addItems(self, items: list[str]) -> None:
        self.items.extend(items)

    def setCurrentText(self, text: str) -> None:
        self._current_text = text

    def setEditText(self, text: str) -> None:
        self._current_text = text


def test_set_available_regions_limits_analysis_dropdowns():
    """Only supplied regions should populate the target/filter dropdowns."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget

    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._region_combo = _DummyCombo(current_text="CP")
    widget._heat_region_combo = _DummyCombo(current_text="VISp")

    widget.set_available_regions(["VISp", "CP", "VISp", "", None])

    assert widget._available_regions == ["CP", "VISp"]
    assert widget._region_combo.items == ["CP", "VISp"]
    assert widget._region_combo.currentText() == "CP"
    assert widget._heat_region_combo.items == ["", "CP", "VISp"]
    assert widget._heat_region_combo.currentText() == "VISp"


def test_set_available_regions_resets_invalid_current_values():
    """When the current region is no longer allowed, the combos should reset cleanly."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget

    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._region_combo = _DummyCombo(current_text="MOp")
    widget._heat_region_combo = _DummyCombo(current_text="MOp")

    widget.set_available_regions(["CP", "VISp"])

    assert widget._region_combo.items == ["CP", "VISp"]
    assert widget._region_combo.currentText() == "CP"
    assert widget._heat_region_combo.items == ["", "CP", "VISp"]
    assert widget._heat_region_combo.currentText() == ""


def test_set_database_populates_regions_from_loaded_parquet_dataset():
    """Database-backed region controls should list only acronyms present in the parquet."""
    AnalysisTabWidget = _import_analysis_tab_module().AnalysisTabWidget

    widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
    widget._region_combo = _DummyCombo(current_text="")
    widget._heat_region_combo = _DummyCombo(current_text="")
    widget._available_regions = []
    widget._update_button_states = lambda: None

    class _DummyDb:
        parquet_path = Path("/tmp/example.parquet")

        @staticmethod
        def get_unique_regions() -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "region_acronym": ["VISp", "CP", "", None, "VISp"],
                }
            )

    widget.set_database(_DummyDb())

    assert Path(widget._parquet_path) == _DummyDb.parquet_path
    assert widget._available_regions == ["CP", "VISp"]
    assert widget._region_combo.items == ["CP", "VISp"]
    assert widget._heat_region_combo.items == ["", "CP", "VISp"]
