"""Tests for Analysis heatmap region filtering."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from napari_neuron_navigator.analysis.heatmap import build_node_counts_volume


class _FakeAtlas:
    """Atlas-like object providing the shape and resolution used by heatmaps."""

    def __init__(self) -> None:
        self.annotation = np.zeros((2, 2, 2), dtype=np.int32)
        self.resolution = (25.0, 25.0, 25.0)


class _LargeFakeAtlas:
    """Larger atlas-like object for soma-radius heatmap tests."""

    def __init__(self) -> None:
        self.annotation = np.zeros((20, 2, 2), dtype=np.int32)
        self.resolution = (25.0, 25.0, 25.0)


def _write_heatmap_parquet(path: Path) -> None:
    df = pd.DataFrame(
        {
            "file_id": ["n1", "n2", "n3", "n4"],
            "type": [1, 2, 3, 4],
            "region_id": [184, 68, 667, 500],
            "x": [0.0, 25.0, 25.0, 0.0],
            "y": [0.0, 0.0, 25.0, 25.0],
            "z": [0.0, 0.0, 0.0, 0.0],
        }
    )
    df.to_parquet(path, index=False)


def _write_radius_heatmap_parquet(path: Path) -> None:
    df = pd.DataFrame(
        {
            "file_id": [
                "file_a",
                "file_a",
                "file_a",
                "file_b",
                "file_b",
                "file_b",
                "file_b",
                "file_c",
            ],
            "type": [1, 3, 4, 1, 1, 3, 4, 3],
            "region_id": [1, 1, 1, 2, 2, 2, 2, 3],
            "x": [0.0, 50.0, 125.0, 200.0, 300.0, 320.0, 380.0, 25.0],
            "y": [0.0] * 8,
            "z": [0.0] * 8,
        }
    )
    df.to_parquet(path, index=False)


def test_build_node_counts_volume_filters_leaf_region_ids(tmp_path: Path) -> None:
    """Leaf-region heatmaps should only include nodes with the selected region ID."""
    parquet_path = tmp_path / "heatmap.parquet"
    _write_heatmap_parquet(parquet_path)

    conn = duckdb.connect()
    try:
        volume = build_node_counts_volume(
            conn,
            str(parquet_path),
            _FakeAtlas(),
            region_ids=[68],
        )
    finally:
        conn.close()

    assert float(volume.sum()) == 1.0
    assert float(volume[1, 0, 0]) == 1.0


def test_build_node_counts_volume_expands_parent_selection_via_region_ids(
    tmp_path: Path,
) -> None:
    """Parent-region heatmaps should include all represented descendant IDs."""
    parquet_path = tmp_path / "heatmap.parquet"
    _write_heatmap_parquet(parquet_path)

    conn = duckdb.connect()
    try:
        volume = build_node_counts_volume(
            conn,
            str(parquet_path),
            _FakeAtlas(),
            region_ids=[184, 68, 667],
        )
    finally:
        conn.close()

    assert float(volume.sum()) == 3.0
    assert float(volume[0, 1, 0]) == 0.0


def test_build_node_counts_volume_without_region_filter_uses_all_nodes(
    tmp_path: Path,
) -> None:
    """Blank Analysis heatmap selection should leave the heatmap unfiltered."""
    parquet_path = tmp_path / "heatmap.parquet"
    _write_heatmap_parquet(parquet_path)

    conn = duckdb.connect()
    try:
        volume = build_node_counts_volume(
            conn,
            str(parquet_path),
            _FakeAtlas(),
            region_ids=None,
        )
    finally:
        conn.close()

    assert float(volume.sum()) == 4.0


def test_build_node_counts_volume_filters_one_selected_file_id(
    tmp_path: Path,
) -> None:
    """Selected-neuron heatmaps should support a single file_id filter."""
    parquet_path = tmp_path / "heatmap.parquet"
    _write_heatmap_parquet(parquet_path)

    conn = duckdb.connect()
    try:
        volume = build_node_counts_volume(
            conn,
            str(parquet_path),
            _FakeAtlas(),
            file_ids=["n2"],
        )
    finally:
        conn.close()

    assert float(volume.sum()) == 1.0
    assert float(volume[1, 0, 0]) == 1.0


def test_build_node_counts_volume_filters_multiple_selected_file_ids(
    tmp_path: Path,
) -> None:
    """Selected-neuron heatmaps should combine counts across multiple file_ids."""
    parquet_path = tmp_path / "heatmap.parquet"
    _write_heatmap_parquet(parquet_path)

    conn = duckdb.connect()
    try:
        volume = build_node_counts_volume(
            conn,
            str(parquet_path),
            _FakeAtlas(),
            file_ids=["n1", "n3"],
        )
    finally:
        conn.close()

    assert float(volume.sum()) == 2.0
    assert float(volume[0, 0, 0]) == 1.0
    assert float(volume[1, 1, 0]) == 1.0


def test_build_node_counts_volume_filters_single_node_type(
    tmp_path: Path,
) -> None:
    """Node-type filtering should count only matching SWC type rows."""
    parquet_path = tmp_path / "heatmap.parquet"
    _write_heatmap_parquet(parquet_path)

    conn = duckdb.connect()
    try:
        volume = build_node_counts_volume(
            conn,
            str(parquet_path),
            _FakeAtlas(),
            node_types=[2],
        )
    finally:
        conn.close()

    assert float(volume.sum()) == 1.0
    assert float(volume[1, 0, 0]) == 1.0


def test_build_node_counts_volume_filters_combined_dendrite_types(
    tmp_path: Path,
) -> None:
    """Basal and apical dendrite selections should combine into one heatmap."""
    parquet_path = tmp_path / "heatmap.parquet"
    _write_heatmap_parquet(parquet_path)

    conn = duckdb.connect()
    try:
        volume = build_node_counts_volume(
            conn,
            str(parquet_path),
            _FakeAtlas(),
            node_types=[3, 4],
        )
    finally:
        conn.close()

    assert float(volume.sum()) == 2.0
    assert float(volume[1, 1, 0]) == 1.0
    assert float(volume[0, 1, 0]) == 1.0


def test_build_node_counts_volume_combines_type_region_and_file_filters(
    tmp_path: Path,
) -> None:
    """Node-type filters should compose with existing region and file filters."""
    parquet_path = tmp_path / "heatmap.parquet"
    _write_heatmap_parquet(parquet_path)

    conn = duckdb.connect()
    try:
        volume = build_node_counts_volume(
            conn,
            str(parquet_path),
            _FakeAtlas(),
            region_ids=[667, 500],
            file_ids=["n3"],
            node_types=[3, 4],
        )
    finally:
        conn.close()

    assert float(volume.sum()) == 1.0
    assert float(volume[1, 1, 0]) == 1.0


def test_build_node_counts_volume_filters_nodes_by_soma_radius(
    tmp_path: Path,
) -> None:
    """Soma radius should keep selected nodes near each file's soma centroid."""
    parquet_path = tmp_path / "radius_heatmap.parquet"
    _write_radius_heatmap_parquet(parquet_path)

    conn = duckdb.connect()
    try:
        volume = build_node_counts_volume(
            conn,
            str(parquet_path),
            _LargeFakeAtlas(),
            node_types=[3, 4],
            soma_radius_um=100.0,
        )
    finally:
        conn.close()

    assert float(volume.sum()) == 2.0
    assert float(volume[2, 0, 0]) == 1.0
    assert float(volume[12, 0, 0]) == 1.0


def test_build_node_counts_volume_only_requires_soma_when_radius_enabled(
    tmp_path: Path,
) -> None:
    """Files without soma should be excluded only by the soma-radius mode."""
    parquet_path = tmp_path / "radius_heatmap.parquet"
    _write_radius_heatmap_parquet(parquet_path)

    conn = duckdb.connect()
    try:
        unbounded = build_node_counts_volume(
            conn,
            str(parquet_path),
            _LargeFakeAtlas(),
            node_types=[3, 4],
        )
        radius_limited = build_node_counts_volume(
            conn,
            str(parquet_path),
            _LargeFakeAtlas(),
            node_types=[3, 4],
            soma_radius_um=100.0,
        )
    finally:
        conn.close()

    assert float(unbounded.sum()) == 5.0
    assert float(unbounded[1, 0, 0]) == 1.0
    assert float(radius_limited.sum()) == 2.0
    assert float(radius_limited[1, 0, 0]) == 0.0
