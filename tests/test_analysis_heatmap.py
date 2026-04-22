"""Tests for Analysis heatmap region filtering."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from napari_swc_viewer.analysis.heatmap import build_node_counts_volume


class _FakeAtlas:
    """Atlas-like object providing the shape and resolution used by heatmaps."""

    def __init__(self) -> None:
        self.annotation = np.zeros((2, 2, 2), dtype=np.int32)
        self.resolution = (25.0, 25.0, 25.0)


def _write_heatmap_parquet(path: Path) -> None:
    df = pd.DataFrame(
        {
            "file_id": ["n1", "n2", "n3", "n4"],
            "region_id": [184, 68, 667, 500],
            "x": [0.0, 25.0, 25.0, 0.0],
            "y": [0.0, 0.0, 25.0, 25.0],
            "z": [0.0, 0.0, 0.0, 0.0],
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
