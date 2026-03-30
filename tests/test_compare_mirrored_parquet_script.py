"""Tests for the mirrored parquet comparison script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _load_script_module():
    """Load the mirrored parquet comparison CLI script as a module."""
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "compare_mirrored_parquet.py"
    spec = importlib.util.spec_from_file_location("compare_mirrored_parquet_script", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_parquet(path: Path, data: dict) -> None:
    """Write a small parquet file with the standard neuron schema."""
    pq.write_table(pa.Table.from_pydict(data), path)


def test_collect_comparison_stats_reports_expected_mirror_summary(tmp_path):
    """The script should detect mirrored z coordinates and region remaps."""
    module = _load_script_module()

    left_path = tmp_path / "left.parquet"
    right_path = tmp_path / "right.parquet"
    left_data = {
        "file_id": ["a.swc", "a.swc", "a.swc"],
        "node_id": [1, 2, 3],
        "type": [1, 3, 3],
        "x": [10.0, 11.0, 12.0],
        "y": [20.0, 21.0, 22.0],
        "z": [100.0, 200.0, 300.0],
        "radius": [1.0, 1.0, 1.0],
        "parent_id": [-1, 1, 2],
        "region_id": [5, 5, 5],
        "region_name": ["Left", "Left", "Left"],
        "region_acronym": ["L", "L", "L"],
        "subject": ["s"] * 3,
        "neuron_id": ["n"] * 3,
    }
    right_data = {
        "file_id": ["a.swc", "a.swc", "a.swc"],
        "node_id": [1, 2, 3],
        "type": [1, 3, 3],
        "x": [10.0, 11.0, 12.0],
        "y": [20.0, 21.0, 22.0],
        "z": [11275.0, 11175.0, 11075.0],
        "radius": [1.0, 1.0, 1.0],
        "parent_id": [-1, 1, 2],
        "region_id": [5, 11, 5],
        "region_name": ["Left", "Right", "Left"],
        "region_acronym": ["L", "R", "L"],
        "subject": ["s"] * 3,
        "neuron_id": ["n"] * 3,
    }
    _write_parquet(left_path, left_data)
    _write_parquet(right_path, right_data)

    stats = module.collect_comparison_stats(
        left_path,
        right_path,
        expected_sum=11375.0,
        top_region_remaps=5,
        sample_region_mismatches=5,
    )

    assert stats["comparison"]["row_count_match"] is True
    assert stats["comparison"]["schema_match"] is True
    assert stats["comparison"]["diff_counts"]["x"] == 0
    assert stats["comparison"]["diff_counts"]["y"] == 0
    assert stats["comparison"]["diff_counts"]["z"] == 3
    assert stats["comparison"]["diff_counts"]["region_id"] == 1
    assert stats["comparison"]["bad_mirror_sum"] == 0
    assert stats["comparison"]["min_axis_sum"] == 11375.0
    assert stats["comparison"]["max_axis_sum"] == 11375.0
    assert stats["top_region_remaps"] == [{"left_region": "L", "right_region": "R", "count": 1}]
    assert len(stats["sample_region_mismatches"]) == 1
