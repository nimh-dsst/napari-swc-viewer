from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from napari_swc_viewer.point_import import (
    PointImportError,
    convert_point_csv_to_parquet,
    format_atlas_validation_summary,
    load_column_mapping,
    load_standard_point_parquet,
    standardize_point_dataframe,
    validate_point_metadata_against_atlas,
)


class FakeAtlas:
    def __init__(self) -> None:
        self.annotation = np.zeros((5, 5, 5), dtype=np.int32)
        self.annotation[4, 3, 2] = 101
        self.annotation[2, 2, 1] = 202
        self.resolution = (25.0, 25.0, 25.0)
        self.shape = self.annotation.shape
        self.atlas_name = "fake_atlas"
        self.structures = {
            101: {"name": "Region, One", "acronym": "R1"},
            202: {"name": "Region Two", "acronym": "R2"},
        }


@pytest.fixture
def fake_atlas() -> FakeAtlas:
    return FakeAtlas()


class _DummyEvent:
    def connect(self, *_args, **_kwargs) -> None:
        return None

    def disconnect(self, *_args, **_kwargs) -> None:
        return None


class _DummyDims:
    def __init__(self) -> None:
        self.ndisplay = 3
        self.not_displayed = (0,)
        self.point = (0.0, 0.0, 0.0)
        self.order = (0, 1, 2)
        self.events = types.SimpleNamespace(
            ndisplay=_DummyEvent(),
            order=_DummyEvent(),
            current_step=_DummyEvent(),
        )


class _DummyPointsLayer:
    def __init__(self, data: np.ndarray, **kwargs) -> None:
        self.data = np.asarray(data)
        self.name = kwargs["name"]
        self.scale = np.asarray(kwargs.get("scale"))
        self.properties = {
            key: np.asarray(value, dtype=object)
            for key, value in kwargs.get("properties", {}).items()
        }
        self.metadata = kwargs.get("metadata", {})


class _DummyViewer:
    def __init__(self) -> None:
        self.layers: list[_DummyPointsLayer] = []
        self.dims = _DummyDims()

    def add_points(self, data: np.ndarray, **kwargs) -> _DummyPointsLayer:
        layer = _DummyPointsLayer(data, **kwargs)
        self.layers.append(layer)
        return layer


def _import_neuron_viewer_widget():
    import importlib

    from qtpy.QtWidgets import QApplication, QWidget

    app = QApplication.instance()
    if app is None:
        QApplication([])

    class _DummySignal:
        def connect(self, *_args, **_kwargs) -> None:
            return None

    class _FakeAnalysisTabWidget(QWidget):
        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__()
            self.cluster_colors_updated = _DummySignal()

        def set_slice_projector(self, *_args, **_kwargs) -> None:
            return None

        def set_database(self, *_args, **_kwargs) -> None:
            return None

        def set_atlas(self, *_args, **_kwargs) -> None:
            return None

        def apply_cluster_colors(self) -> None:
            return None

    fake_analysis_module = types.ModuleType(
        "napari_swc_viewer.widgets.analysis_tab"
    )
    fake_analysis_module.AnalysisTabWidget = _FakeAnalysisTabWidget
    sys.modules["napari_swc_viewer.widgets.analysis_tab"] = fake_analysis_module
    sys.modules.pop("napari_swc_viewer.widgets.neuron_viewer", None)
    sys.modules.pop("napari_swc_viewer.widgets", None)

    module = importlib.import_module("napari_swc_viewer.widgets.neuron_viewer")
    return module.NeuronViewerWidget


def test_load_column_mapping_requires_required_targets(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps({"label": "marker", "x": "atlas_x"}))

    with pytest.raises(PointImportError, match="Missing required mapping"):
        load_column_mapping(mapping_path)


def test_convert_point_csv_to_parquet_preserves_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "points.csv"
    mapping_path = tmp_path / "mapping.json"
    output_path = tmp_path / "points.parquet"

    pd.DataFrame(
        {
            "marker": ["A", "B"],
            "atlas_x": [50.0, 25.0],
            "atlas_y": [75.0, 50.0],
            "atlas_z": [100.0, 50.0],
            "score": [0.5, 0.9],
            "region_acronym": ["R1", "R2"],
        }
    ).to_csv(csv_path, index=False)
    mapping_path.write_text(
        json.dumps(
            {
                "label": "marker",
                "x": "atlas_x",
                "y": "atlas_y",
                "z": "atlas_z",
                "acronym": "region_acronym",
            }
        )
    )

    standardized = convert_point_csv_to_parquet(csv_path, mapping_path, output_path)
    loaded = pd.read_parquet(output_path)

    assert list(standardized.columns) == [
        "label",
        "x",
        "y",
        "z",
        "region_name",
        "acronym",
        "id",
        "hemisphere",
        "score",
    ]
    assert list(loaded.columns) == list(standardized.columns)
    assert loaded["score"].tolist() == [0.5, 0.9]
    assert loaded["acronym"].tolist() == ["R1", "R2"]
    assert loaded["region_name"].isna().all()


def test_convert_point_csv_to_parquet_rejects_invalid_coordinates(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "points.csv"
    mapping_path = tmp_path / "mapping.json"
    output_path = tmp_path / "points.parquet"

    pd.DataFrame(
        {
            "marker": ["A"],
            "atlas_x": ["bad"],
            "atlas_y": [75.0],
            "atlas_z": [100.0],
        }
    ).to_csv(csv_path, index=False)
    mapping_path.write_text(
        json.dumps(
            {"label": "marker", "x": "atlas_x", "y": "atlas_y", "z": "atlas_z"}
        )
    )

    with pytest.raises(PointImportError, match="Column 'x' has 1 invalid value"):
        convert_point_csv_to_parquet(csv_path, mapping_path, output_path)


def test_validate_point_metadata_against_atlas_uses_world_xyz_order(
    fake_atlas: FakeAtlas,
) -> None:
    df = pd.DataFrame(
        {
            "label": ["A"],
            "x": [50.0],
            "y": [75.0],
            "z": [100.0],
            "region_name": ["Region/One"],
            "acronym": ["R1"],
            "id": [101],
            "hemisphere": ["right"],
        }
    )

    summary = validate_point_metadata_against_atlas(df, fake_atlas)

    assert not summary.has_mismatches
    assert summary.checked_fields == ("id", "acronym", "region_name", "hemisphere")


def test_validate_point_metadata_against_atlas_reports_field_counts(
    fake_atlas: FakeAtlas,
) -> None:
    df = pd.DataFrame(
        {
            "label": ["A"],
            "x": [50.0],
            "y": [75.0],
            "z": [100.0],
            "region_name": ["Wrong Region"],
            "acronym": ["BAD"],
            "id": [999],
            "hemisphere": ["left"],
        }
    )

    summary = validate_point_metadata_against_atlas(df, fake_atlas)
    message = format_atlas_validation_summary(summary)

    assert summary.total_mismatched_rows == 1
    assert summary.mismatch_counts == {
        "id": 1,
        "acronym": 1,
        "region_name": 1,
        "hemisphere": 1,
    }
    assert "row 1 label=A" in message


def test_standardize_point_dataframe_rejects_conflicting_extra_columns() -> None:
    raw_df = pd.DataFrame(
        {
            "marker": ["A"],
            "atlas_x": [1.0],
            "atlas_y": [2.0],
            "atlas_z": [3.0],
            "label": ["conflict"],
        }
    )

    with pytest.raises(PointImportError, match="conflict with standardized"):
        standardize_point_dataframe(
            raw_df,
            {"label": "marker", "x": "atlas_x", "y": "atlas_y", "z": "atlas_z"},
        )


def test_convert_point_csv_script(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = tmp_path / "points.csv"
    mapping_path = tmp_path / "mapping.json"
    output_path = tmp_path / "points.parquet"

    pd.DataFrame(
        {
            "marker": ["A"],
            "atlas_x": [50.0],
            "atlas_y": [75.0],
            "atlas_z": [100.0],
        }
    ).to_csv(csv_path, index=False)
    mapping_path.write_text(
        json.dumps(
            {"label": "marker", "x": "atlas_x", "y": "atlas_y", "z": "atlas_z"}
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/convert_point_csv.py",
            str(csv_path),
            str(mapping_path),
            str(output_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    loaded = load_standard_point_parquet(output_path)
    assert loaded["label"].tolist() == ["A"]


@pytest.mark.skip(reason="Qt runtime is unavailable in the current test sandbox.")
def test_widget_import_point_parquet_requires_loaded_atlas(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    NeuronViewerWidget = _import_neuron_viewer_widget()
    monkeypatch.setattr(NeuronViewerWidget, "_toggle_template", lambda self, state: None)
    warnings: list[str] = []
    monkeypatch.setattr(
        "napari_swc_viewer.widgets.neuron_viewer.show_warning",
        lambda message: warnings.append(message),
    )

    viewer = _DummyViewer()
    widget = NeuronViewerWidget(viewer)

    parquet_path = tmp_path / "points.parquet"
    pd.DataFrame({"label": ["A"], "x": [1.0], "y": [2.0], "z": [3.0]}).to_parquet(
        parquet_path,
        index=False,
    )

    widget._load_point_parquet_file(str(parquet_path))

    assert warnings == ["Load an atlas before importing point Parquet."]
    assert widget._point_import_status_label.text() == (
        "Load an atlas before importing point Parquet."
    )
    assert len(viewer.layers) == 0


@pytest.mark.skip(reason="Qt runtime is unavailable in the current test sandbox.")
def test_widget_import_point_parquet_creates_layers_and_warns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_atlas: FakeAtlas,
) -> None:
    NeuronViewerWidget = _import_neuron_viewer_widget()
    monkeypatch.setattr(NeuronViewerWidget, "_toggle_template", lambda self, state: None)
    warnings: list[str] = []
    monkeypatch.setattr(
        "napari_swc_viewer.widgets.neuron_viewer.show_warning",
        lambda message: warnings.append(message),
    )

    viewer = _DummyViewer()
    widget = NeuronViewerWidget(viewer)
    widget._atlas = fake_atlas

    parquet_path = tmp_path / "points.parquet"
    pd.DataFrame(
        {
            "label": ["A", "A", "B"],
            "x": [50.0, 25.0, 50.0],
            "y": [75.0, 50.0, 75.0],
            "z": [100.0, 50.0, 100.0],
            "region_name": ["Region/One", pd.NA, "Wrong Region"],
            "acronym": ["R1", pd.NA, "BAD"],
            "id": [101, pd.NA, 999],
            "hemisphere": ["right", pd.NA, "left"],
            "score": [0.1, 0.2, 0.3],
        }
    ).to_parquet(parquet_path, index=False)

    widget._load_point_parquet_file(str(parquet_path))

    layer_names = [layer.name for layer in viewer.layers]
    assert layer_names == ["Points: A", "Points: B"]
    assert warnings
    assert "mismatched point(s)" in warnings[0]
    assert "3 point(s) across 2 label(s)" in widget._point_import_status_label.text()
    assert "mismatched row(s)" in widget._point_import_status_label.text()

    layer_a = viewer.layers[0]
    assert np.allclose(layer_a.scale, [1 / 25.0, 1 / 25.0, 1 / 25.0])
    assert list(layer_a.properties["score"]) == [0.1, 0.2]
    assert list(layer_a.properties["acronym"]) == ["R1", None]
