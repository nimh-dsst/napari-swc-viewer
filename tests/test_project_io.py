from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from napari_swc_viewer.project_io import (
    PROJECT_BUNDLE_FORMAT,
    ENHANCED_NEURON_COLUMNS,
    export_enhanced_neuron_parquet,
    load_project_bundle,
    read_enhanced_parquet_metadata,
    save_project_bundle,
)


def _write_source_parquet(path: Path) -> None:
    pd.DataFrame(
        {
            "file_id": ["n1", "n1", "n2"],
            "node_id": [1, 2, 1],
            "type": [1, 3, 1],
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
            "z": [7.0, 8.0, 9.0],
            "radius": [1.0, 1.0, 1.0],
            "parent_id": [-1, 1, -1],
            "region_id": [1, 1, 2],
            "region_name": ["A", "A", "B"],
            "region_acronym": ["A", "A", "B"],
            "subject": ["s1", "s1", "s2"],
            "neuron_id": ["neuron1", "neuron1", "neuron2"],
        }
    ).to_parquet(path, index=False)


def _table_state() -> dict[str, object]:
    return {
        "version": 1,
        "entries": [
            {
                "file_id": "n1",
                "subject": "s1",
                "label": "projection",
                "group": "A",
                "tags": ["axon", "reviewed"],
                "notes": "keep",
                "cluster_id": 3,
                "color": [0.1, 0.2, 0.3, 1.0],
                "visible": False,
            }
        ],
    }


@dataclass
class _DummyLayer:
    name: str
    data: np.ndarray
    metadata: dict[str, object]
    opacity: float = 0.6
    visible: bool = True
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0)
    blending: str = "additive"
    rendering: str = "mip"
    contrast_limits: tuple[float, float] = (0.0, 1.0)
    colormap: object = field(default_factory=lambda: SimpleNamespace(name="hot"))


class _DummyColormap:
    def __init__(self, name: str, colors: list[list[float]]) -> None:
        self.name = name
        self.colors = np.asarray(colors, dtype=np.float32)
        self.controls = np.linspace(0.0, 1.0, len(colors), dtype=np.float32)
        self.interpolation = "linear"
        self.low_color = None
        self.high_color = None
        self.nan_color = [0.0, 0.0, 0.0, 0.0]


def test_export_enhanced_neuron_parquet_round_trips_labels_and_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "enhanced.parquet"
    _write_source_parquet(source)

    export_enhanced_neuron_parquet(
        source,
        output,
        table_state=_table_state(),
        metadata={"atlas_name": "fake_atlas"},
    )

    loaded = pd.read_parquet(output).sort_values(["file_id", "node_id"]).reset_index(drop=True)
    assert len(loaded) == 3
    assert all(column in loaded.columns for column in ENHANCED_NEURON_COLUMNS)
    assert loaded.loc[loaded["file_id"] == "n1", "neuron_label"].unique().tolist() == [
        "projection"
    ]
    cluster_values = (
        loaded.loc[loaded["file_id"] == "n1", "cluster_assignment"]
        .dropna()
        .unique()
        .tolist()
    )
    assert cluster_values == [3]
    assert loaded.loc[loaded["file_id"] == "n2", "cluster_assignment"].isna().all()

    payload = read_enhanced_parquet_metadata(output)
    assert payload["metadata"]["atlas_name"] == "fake_atlas"
    assert payload["table_state"]["entries"][0]["label"] == "projection"
    assert payload["enhanced_columns"] == list(ENHANCED_NEURON_COLUMNS)


def test_read_enhanced_parquet_metadata_accepts_canonical_parquet(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    _write_source_parquet(source)

    payload = read_enhanced_parquet_metadata(source)

    assert payload["has_project_metadata"] is False
    assert payload["enhanced_columns"] == []
    assert payload["table_state"]["entries"] == []


def test_save_and_load_project_bundle_preserves_mask_array_and_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    bundle_path = tmp_path / "saved.swcv"
    _write_source_parquet(source)
    progress_events: list[tuple[str, int, int]] = []

    heatmap = _DummyLayer(
        name="alpha Heatmap",
        data=np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        metadata={
            "heatmap_source": True,
            "heatmap_kind": "selected_neurons",
            "file_ids": ["n1"],
            "source_path": str(source),
            "color": [1.0, 0.0, 0.0, 1.0],
            "atlas_name": "fake_atlas",
        },
        colormap=_DummyColormap(
            "alpha_red",
            [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]],
        ),
    )
    mask = _DummyLayer(
        name="Mask: alpha Heatmap",
        data=np.array([[[0, 1], [1, 0]]], dtype=np.uint8),
        metadata={
            "mask_query_source": True,
            "source_heatmap_layers": ["alpha Heatmap"],
            "query_excluded_file_ids": ["n1", "n2", "n1"],
            "lower_threshold": 0.5,
            "upper_threshold": 2.0,
            "merge_mode": "separate",
            "atlas_name": "fake_atlas",
        },
    )
    ignored = _DummyLayer(
        name="Neuron Lines",
        data=np.zeros((1, 2, 3), dtype=float),
        metadata={"file_ids": ["n1"]},
    )

    save_project_bundle(
        bundle_path,
        source_parquet_path=source,
        table_state=_table_state(),
        layers=[heatmap, mask, ignored],
        atlas_name="fake_atlas",
        progress_callback=lambda message, current, total: progress_events.append(
            (message, current, total)
        ),
    )

    assert progress_events[0] == ("Preparing project bundle...", 0, 6)
    assert progress_events[-1] == ("Done", 6, 6)
    assert any("Saving layer 1/2: alpha Heatmap" in event[0] for event in progress_events)
    assert any("Saving layer 2/2: Mask: alpha Heatmap" in event[0] for event in progress_events)

    bundle = load_project_bundle(bundle_path)
    assert bundle.manifest["format"] == PROJECT_BUNDLE_FORMAT
    assert bundle.source_parquet_path.exists()
    assert bundle.table_state["entries"][0]["file_id"] == "n1"
    assert len(bundle.layers) == 2
    restored_heatmap = next(
        layer for layer in bundle.layers if layer.metadata["layer_type"] == "image"
    )
    colormap = restored_heatmap.metadata["display"]["colormap"]
    assert colormap["name"] == "alpha_red"
    assert colormap["colors"] == [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]]
    restored_mask = next(
        layer for layer in bundle.layers if layer.metadata["layer_type"] == "labels"
    )
    np.testing.assert_array_equal(restored_mask.data, mask.data)
    source_sets = restored_mask.metadata["source_neuron_sets"]
    assert source_sets[0]["count"] == 2
    assert source_sets[0]["file_ids"] == ["n1", "n2"]
    assert source_sets[0]["derivation"]["lower_threshold"] == 0.5

    bundled_schema = pq.read_schema(bundle.source_parquet_path)
    assert "neuron_label" in bundled_schema.names


def test_project_colormap_payload_restores_image_colormap_kwargs() -> None:
    from napari_swc_viewer.widgets.neuron_viewer import NeuronViewerWidget

    kwargs: dict[str, object] = {}
    restored = NeuronViewerWidget._apply_project_colormap_kwargs(
        SimpleNamespace(),
        kwargs,
        {
            "colormap": {
                "type": "colormap",
                "name": "alpha_red",
                "colors": [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]],
                "controls": [0.0, 1.0],
                "interpolation": "linear",
                "nan_color": [0.0, 0.0, 0.0, 0.0],
            }
        },
    )

    assert restored is True
    assert kwargs["colormap"].name == "alpha_red"
    np.testing.assert_allclose(
        np.asarray(kwargs["colormap"].colors),
        [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]],
    )


def test_project_colormap_payload_restores_labels_colormap_kwargs() -> None:
    from napari_swc_viewer.widgets.neuron_viewer import NeuronViewerWidget

    kwargs: dict[str, object] = {}
    restored = NeuronViewerWidget._apply_project_colormap_kwargs(
        SimpleNamespace(_project_label_key=NeuronViewerWidget._project_label_key),
        kwargs,
        {
            "colormap": {
                "type": "direct_label_colormap",
                "color_dict": [
                    {
                        "label": {"type": "none", "value": None},
                        "color": [0.0, 0.0, 0.0, 0.0],
                    },
                    {
                        "label": {"type": "int", "value": 1},
                        "color": [0.2, 0.4, 0.6, 1.0],
                    },
                ],
            }
        },
    )

    assert restored is True
    np.testing.assert_allclose(
        kwargs["colormap"].color_dict[1],
        [0.2, 0.4, 0.6, 1.0],
    )
