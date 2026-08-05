"""Tests for CCF voxel-correlation input preparation."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from napari_swc_viewer.analysis.correlation import count_correlation_input_nodes


def test_count_correlation_input_nodes_supports_unfiltered_file_scope(tmp_path) -> None:
    path = tmp_path / "nodes.parquet"
    pd.DataFrame(
        {
            "file_id": ["n1", "n1", "n2", "n2", "n3"],
            "x": [10.0, 30.0, 10.0, 55.0, np.nan],
            "y": [10.0, 30.0, 10.0, 55.0, 10.0],
            "z": [10.0, 30.0, 10.0, 55.0, 10.0],
        }
    ).to_parquet(path, index=False)
    conn = duckdb.connect()
    try:
        count = count_correlation_input_nodes(
            conn,
            str(path),
            None,
            25.0,
            file_ids=["n1", "n3"],
        )
    finally:
        conn.close()

    assert count == 2


def test_count_correlation_input_nodes_applies_region_lookup(tmp_path) -> None:
    path = tmp_path / "nodes.parquet"
    pd.DataFrame(
        {
            "file_id": ["inside", "outside"],
            "x": [10.0, 55.0],
            "y": [10.0, 55.0],
            "z": [10.0, 55.0],
        }
    ).to_parquet(path, index=False)
    voxel_id_map = np.full((4, 4, 4), -1, dtype=np.int32)
    voxel_id_map[0, 0, 0] = 0
    conn = duckdb.connect()
    try:
        count = count_correlation_input_nodes(
            conn,
            str(path),
            voxel_id_map,
            25.0,
        )
    finally:
        conn.close()

    assert count == 1
