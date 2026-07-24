from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

nrrd = pytest.importorskip("nrrd")


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/benchmark_flatmap_pipeline.py"


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    parquet_path = tmp_path / "neurons.parquet"
    flatmap_path = tmp_path / "flatmap_both_shaped.nrrd"
    depth_path = tmp_path / "depth.nrrd"

    rows = []
    for file_id, node_count in (("small.swc", 2), ("large.swc", 4)):
        for node_id in range(node_count):
            rows.append(
                {
                    "file_id": file_id,
                    "neuron_id": file_id.removesuffix(".swc"),
                    "subject": "subject",
                    "node_id": node_id + 1,
                    "type": 3,
                    "x": float(node_id * 10),
                    "y": 0.0,
                    "z": 0.0,
                    "radius": 1.0,
                    "parent_id": node_id if node_id else -1,
                    "region_id": 1,
                    "region_name": "Region",
                    "region_acronym": "REG",
                }
            )
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)

    shape = (4, 4, 4)
    grid = np.indices(shape, dtype=np.float32)
    flatmap = np.stack((grid[0], grid[1]), axis=-1)
    depth = np.full(shape, 50.0, dtype=np.float32)
    nrrd.write(str(flatmap_path), flatmap)
    nrrd.write(str(depth_path), depth)
    return parquet_path, flatmap_path, depth_path


def _run_script(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, SCRIPT, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_benchmark_flatmap_pipeline_help() -> None:
    result = _run_script(["--help"])

    assert result.returncode == 0
    assert "Profile memory and runtime" in result.stdout
    assert "--parquet" in result.stdout
    assert "--npy-cache-dir" in result.stdout
    assert "--no-npy-cache" in result.stdout


def test_benchmark_flatmap_pipeline_writes_json_and_csv(tmp_path: Path) -> None:
    parquet_path, flatmap_path, depth_path = _write_inputs(tmp_path)
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"

    result = _run_script(
        [
            "--parquet",
            str(parquet_path),
            "--flatmap",
            str(flatmap_path),
            "--depth",
            str(depth_path),
            "--file-id",
            "small.swc",
            "--xy-bins",
            "8",
            "--depth-bin-um",
            "25",
            "--output-json",
            str(json_path),
            "--output-csv",
            str(csv_path),
        ]
    )

    assert result.returncode == 0, result.stderr
    assert "START open_database" in result.stderr
    assert "DONE  build_flatmap_render_data" in result.stderr
    assert "render.compute_lookup_stats_chunked" in result.stderr
    assert "render.build_flatmap_render_data_with_stats" in result.stderr
    assert "rows=2" in result.stderr
    report = json.loads(json_path.read_text())
    assert report["status"] == "ok"
    assert report["options"]["live_progress"] is True
    assert report["options"]["use_npy_cache"] is True
    assert report["selected_file_id"] == "small.swc"
    assert report["selected_node_count"] == 2
    assert report["projection_summary"]["total_nodes"] == 2
    assert report["projection_summary"]["mirrored_depth_lookup_nodes"] == 0
    assert report["render_summary"]["rendered_nodes"] == 2
    assert report["object_sizes"]["heatmap_volume_bytes"] > 0
    load_stage = next(
        stage for stage in report["stages"] if stage["name"] == "load_flatmap_volumes"
    )
    assert load_stage["details"]["flatmap_npy_path"].endswith(".float32.npy")
    assert load_stage["details"]["depth_npy_path"].endswith(".float32.npy")
    assert {stage["name"] for stage in report["stages"]} >= {
        "open_database",
        "select_file_id",
        "query_neuron_rows",
        "load_flatmap_volumes",
        "project_nodes_to_flatmap",
        "build_projected_segments",
        "render.compute_lookup_stats_chunked",
        "render.build_flatmap_render_data_with_stats",
    }

    with csv_path.open(newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    assert rows
    assert rows[0]["selected_file_id"] == "small.swc"


def test_benchmark_flatmap_pipeline_auto_selects_largest_file_id(
    tmp_path: Path,
) -> None:
    parquet_path, flatmap_path, depth_path = _write_inputs(tmp_path)
    json_path = tmp_path / "report.json"

    result = _run_script(
        [
            "--parquet",
            str(parquet_path),
            "--flatmap",
            str(flatmap_path),
            "--depth",
            str(depth_path),
            "--candidate-limit",
            "2",
            "--output-json",
            str(json_path),
        ]
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(json_path.read_text())
    assert report["selected_file_id"] == "large.swc"
    assert report["selected_node_count"] == 4
    assert report["selection"]["mode"] == "largest_by_row_count"
    assert report["selection"]["candidates"][0] == {
        "file_id": "large.swc",
        "node_count": 4,
    }


def test_benchmark_flatmap_pipeline_rss_limit_exits_nonzero_after_writing_report(
    tmp_path: Path,
) -> None:
    parquet_path, flatmap_path, depth_path = _write_inputs(tmp_path)
    json_path = tmp_path / "report.json"

    result = _run_script(
        [
            "--parquet",
            str(parquet_path),
            "--flatmap",
            str(flatmap_path),
            "--depth",
            str(depth_path),
            "--file-id",
            "small.swc",
            "--max-rss-gb",
            "0",
            "--output-json",
            str(json_path),
        ]
    )

    assert result.returncode == 2, result.stdout + result.stderr
    report = json.loads(json_path.read_text())
    assert report["status"] == "rss_limit_exceeded"
    assert report["rss_limit"]["exceeded"] is True
