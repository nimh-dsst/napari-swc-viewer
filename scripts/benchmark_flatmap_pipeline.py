#!/usr/bin/env python
"""Profile the real parquet-to-flatmap heatmap pipeline for one neuron."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import csv
import json
import os
import platform
import sys
import threading
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    import resource
except ImportError:  # pragma: no cover - Windows fallback
    resource = None

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from napari_neuron_navigator.db import NeuronDatabase
from napari_neuron_navigator.flatmap_heatmap import (
    DEFAULT_FLATMAP_DEPTH_BIN_UM,
    DEFAULT_FLATMAP_Y_BINS,
    FlatmapRenderResult,
    build_flatmap_render_data,
    compute_flatmap_lookup_stats,
)
from napari_neuron_navigator.flatmap_loader import load_flatmap_volume_set
from napari_neuron_navigator.flatmap_projection import (
    COORDINATE_MODE_MICRONS,
    COORDINATE_MODE_VOXELS,
    DEFAULT_CCFV3_MIRROR_MIDLINE_UM,
    DEFAULT_CCF_RESOLUTION_UM,
    FlatmapProjectionResult,
    build_projected_segments,
    project_neuron_nodes_to_flatmap,
    summarize_projection,
)


EXIT_RSS_LIMIT_EXCEEDED = 2
BYTES_PER_GIB = 1024**3


@dataclass
class StageMetric:
    """Runtime and memory counters for one benchmark stage."""

    name: str
    elapsed_s: float
    rss_before_bytes: int | None
    rss_after_bytes: int | None
    rss_delta_bytes: int | None
    rss_peak_bytes: int | None
    resource_peak_before_bytes: int | None
    resource_peak_after_bytes: int | None
    resource_peak_delta_bytes: int | None
    tracemalloc_current_before_bytes: int
    tracemalloc_current_after_bytes: int
    tracemalloc_peak_bytes: int
    tracemalloc_peak_delta_bytes: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "elapsed_s": self.elapsed_s,
            "rss_before_bytes": self.rss_before_bytes,
            "rss_after_bytes": self.rss_after_bytes,
            "rss_delta_bytes": self.rss_delta_bytes,
            "rss_peak_bytes": self.rss_peak_bytes,
            "resource_peak_before_bytes": self.resource_peak_before_bytes,
            "resource_peak_after_bytes": self.resource_peak_after_bytes,
            "resource_peak_delta_bytes": self.resource_peak_delta_bytes,
            "tracemalloc_current_before_bytes": (
                self.tracemalloc_current_before_bytes
            ),
            "tracemalloc_current_after_bytes": self.tracemalloc_current_after_bytes,
            "tracemalloc_peak_bytes": self.tracemalloc_peak_bytes,
            "tracemalloc_peak_delta_bytes": self.tracemalloc_peak_delta_bytes,
            "details": self.details,
        }


class RssSampler:
    """Sample process RSS with psutil when available."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self._interval_s = float(interval_s)
        self._process = psutil.Process(os.getpid()) if psutil is not None else None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._peak_rss_bytes = self.current_rss_bytes()
        self._stage_peak_rss_bytes = self._peak_rss_bytes

    @property
    def method(self) -> str:
        return "psutil" if self._process is not None else "resource"

    def start(self) -> None:
        if self._process is None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            self.sample()

    def current_rss_bytes(self) -> int | None:
        if self._process is None:
            return None
        return int(self._process.memory_info().rss)

    def sample(self) -> int | None:
        rss = self.current_rss_bytes()
        if rss is None:
            return None
        with self._lock:
            if self._peak_rss_bytes is None or rss > self._peak_rss_bytes:
                self._peak_rss_bytes = rss
            if (
                self._stage_peak_rss_bytes is None
                or rss > self._stage_peak_rss_bytes
            ):
                self._stage_peak_rss_bytes = rss
        return rss

    def reset_stage_peak(self) -> None:
        rss = self.sample()
        with self._lock:
            self._stage_peak_rss_bytes = rss

    @property
    def peak_rss_bytes(self) -> int | None:
        self.sample()
        with self._lock:
            return self._peak_rss_bytes

    @property
    def stage_peak_rss_bytes(self) -> int | None:
        self.sample()
        with self._lock:
            return self._stage_peak_rss_bytes


class Benchmarker:
    """Measure elapsed time and memory for named stages."""

    def __init__(self, *, live: bool = True) -> None:
        self.live = bool(live)
        self.sampler = RssSampler()
        self.stages: list[StageMetric] = []
        tracemalloc.start()
        self.sampler.start()

    def close(self) -> None:
        self.sampler.stop()
        tracemalloc.stop()

    def start_group(self, name: str) -> float:
        start = time.perf_counter()
        if self.live:
            _print_live(
                f"START {name} | rss={_format_bytes(self.sampler.sample())} | "
                f"process_peak={_format_bytes(self.sampler.peak_rss_bytes)} | "
                f"resource_peak={_format_bytes(_resource_peak_rss_bytes())}"
            )
        return start

    def finish_group(
        self,
        name: str,
        start: float,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        if not self.live:
            return
        detail_summary = _stage_detail_summary(details or {})
        suffix = f" | {detail_summary}" if detail_summary else ""
        _print_live(
            f"DONE  {name} | {time.perf_counter() - start:.3f}s | "
            f"rss={_format_bytes(self.sampler.sample())} | "
            f"process_peak={_format_bytes(self.sampler.peak_rss_bytes)} | "
            f"resource_peak={_format_bytes(_resource_peak_rss_bytes())}"
            f"{suffix}"
        )

    def measure(
        self,
        name: str,
        func: Callable[[], Any],
        *,
        details: Callable[[Any], dict[str, Any]] | None = None,
    ) -> Any:
        rss_before = self.sampler.sample()
        resource_before = _resource_peak_rss_bytes()
        trace_before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        self.sampler.reset_stage_peak()

        self._print_stage_start(
            name,
            rss_before=rss_before,
            resource_peak=resource_before,
            python_allocated=trace_before,
        )
        start = time.perf_counter()
        try:
            result = func()
        except BaseException:
            elapsed = time.perf_counter() - start
            self._print_stage_failed(name, elapsed_s=elapsed)
            raise
        elapsed = time.perf_counter() - start

        rss_after = self.sampler.sample()
        stage_peak = self.sampler.stage_peak_rss_bytes
        resource_after = _resource_peak_rss_bytes()
        trace_after, trace_peak = tracemalloc.get_traced_memory()
        metric = StageMetric(
            name=name,
            elapsed_s=elapsed,
            rss_before_bytes=rss_before,
            rss_after_bytes=rss_after,
            rss_delta_bytes=_delta(rss_before, rss_after),
            rss_peak_bytes=stage_peak,
            resource_peak_before_bytes=resource_before,
            resource_peak_after_bytes=resource_after,
            resource_peak_delta_bytes=_delta(resource_before, resource_after),
            tracemalloc_current_before_bytes=int(trace_before),
            tracemalloc_current_after_bytes=int(trace_after),
            tracemalloc_peak_bytes=int(trace_peak),
            tracemalloc_peak_delta_bytes=max(int(trace_peak - trace_before), 0),
            details=details(result) if details is not None else {},
        )
        self.stages.append(metric)
        self._print_stage_done(metric)
        return result

    def _print_stage_start(
        self,
        name: str,
        *,
        rss_before: int | None,
        resource_peak: int | None,
        python_allocated: int,
    ) -> None:
        if not self.live:
            return
        _print_live(
            f"START {name} | rss={_format_bytes(rss_before)} | "
            f"resource_peak={_format_bytes(resource_peak)} | "
            f"python_allocated={_format_bytes(python_allocated)}"
        )

    def _print_stage_done(self, metric: StageMetric) -> None:
        if not self.live:
            return
        detail_summary = _stage_detail_summary(metric.details)
        suffix = f" | {detail_summary}" if detail_summary else ""
        _print_live(
            f"DONE  {metric.name} | {metric.elapsed_s:.3f}s | "
            f"rss={_format_bytes(metric.rss_after_bytes)} | "
            f"delta={_format_bytes(metric.rss_delta_bytes)} | "
            f"stage_peak={_format_bytes(metric.rss_peak_bytes)} | "
            f"python_peak={_format_bytes(metric.tracemalloc_peak_delta_bytes)}"
            f"{suffix}"
        )

    def _print_stage_failed(self, name: str, *, elapsed_s: float) -> None:
        if not self.live:
            return
        _print_live(
            f"FAILED {name} | {elapsed_s:.3f}s | "
            f"rss={_format_bytes(self.sampler.sample())} | "
            f"process_peak={_format_bytes(self.sampler.peak_rss_bytes)} | "
            f"resource_peak={_format_bytes(_resource_peak_rss_bytes())}"
        )


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile memory and runtime for the real neuron parquet to "
            "isocortex flatmap heatmap pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        required=True,
        help="Neuron parquet file to query.",
    )
    parser.add_argument(
        "--flatmap",
        type=Path,
        required=True,
        help="Flatmap lookup NRRD path.",
    )
    parser.add_argument(
        "--depth",
        type=Path,
        required=True,
        help="Depth NRRD path matching the flatmap lookup grid.",
    )
    parser.add_argument(
        "--npy-cache-dir",
        type=Path,
        help=(
            "Optional directory for normalized float32 .npy lookup caches. "
            "Defaults to writing next to each NRRD."
        ),
    )
    parser.add_argument(
        "--no-npy-cache",
        action="store_true",
        help="Disable normalized .npy cache reads and writes.",
    )
    parser.add_argument(
        "--file-id",
        help=(
            "Single neuron file_id to profile. If omitted, the script profiles "
            "the largest file_id by row count."
        ),
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=10,
        help="Number of largest file_id candidates to record when auto-selecting.",
    )
    parser.add_argument(
        "--y-bins",
        type=int,
        default=DEFAULT_FLATMAP_Y_BINS,
        help=(
            "Number of flatmap heatmap bins along the Y axis. The X count "
            "is derived from the flat map aspect ratio so bins stay square."
        ),
    )
    parser.add_argument(
        "--depth-bin-um",
        type=float,
        default=DEFAULT_FLATMAP_DEPTH_BIN_UM,
        help="Depth bin size in microns for the flatmap heatmap.",
    )
    parser.add_argument(
        "--coordinate-mode",
        choices=(COORDINATE_MODE_MICRONS, COORDINATE_MODE_VOXELS),
        default=COORDINATE_MODE_MICRONS,
        help="Interpret input coordinates as microns or lookup voxel indices.",
    )
    parser.add_argument(
        "--resolution-um",
        type=float,
        default=DEFAULT_CCF_RESOLUTION_UM,
        help="CCF voxel resolution used when coordinate mode is microns.",
    )
    parser.add_argument(
        "--no-mirror-fallback",
        action="store_true",
        help=(
            "Disable both mirrored-depth recovery and full opposite-hemisphere "
            "retry for invalid direct lookup rows."
        ),
    )
    parser.add_argument(
        "--mirror-axis",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="Coordinate axis mirrored across the CCFv3 midline (default: 2).",
    )
    parser.add_argument(
        "--mirror-midline",
        type=float,
        default=None,
        help=(
            "Override the mirror midline. Defaults to "
            f"{DEFAULT_CCFV3_MIRROR_MIDLINE_UM:g} microns in micron mode "
            "or the lookup-grid center in voxel mode."
        ),
    )
    parser.add_argument(
        "--treat-zero-flatmap-invalid",
        action="store_true",
        help="Treat flatmap lookup coordinates (0, 0) as invalid sentinels.",
    )
    parser.add_argument(
        "--allow-negative-one-flatmap",
        action="store_true",
        help="Do not treat flatmap lookup coordinates (-1, -1) as invalid.",
    )
    parser.add_argument(
        "--exclude-depth-minus-one",
        action="store_true",
        help="Exclude depth -1 nodes from the rendered sentinel plane.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write the full benchmark report as JSON.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path to write one CSV row per measured stage.",
    )
    parser.add_argument(
        "--max-rss-gb",
        type=float,
        help="Optional peak RSS limit. The script writes reports before exiting nonzero.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress live per-stage progress messages.",
    )
    return parser.parse_args(args)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    benchmark = Benchmarker(live=not args.quiet)
    db: NeuronDatabase | None = None
    try:
        db = benchmark.measure(
            "open_database",
            lambda: NeuronDatabase(args.parquet),
            details=lambda _: {"parquet_path": str(args.parquet)},
        )
        selection = benchmark.measure(
            "select_file_id",
            lambda: _select_file_id(
                db,
                requested_file_id=args.file_id,
                candidate_limit=args.candidate_limit,
            ),
        )
        file_id = selection["selected_file_id"]
        nodes = benchmark.measure(
            "query_neuron_rows",
            lambda: db.get_neurons_for_rendering([file_id]),
            details=_dataframe_details,
        )
        volume_set = benchmark.measure(
            "load_flatmap_volumes",
            lambda: load_flatmap_volume_set(
                args.flatmap,
                args.depth,
                use_npy_cache=not args.no_npy_cache,
                create_npy_cache=not args.no_npy_cache,
                npy_cache_dir=args.npy_cache_dir,
            ),
            details=lambda loaded: {
                "flatmap_shape": list(loaded.flatmap.shape),
                "flatmap_dtype": str(loaded.flatmap.dtype),
                "flatmap_nbytes": int(loaded.flatmap.nbytes),
                "flatmap_npy_path": (
                    None
                    if loaded.flatmap_npy_path is None
                    else str(loaded.flatmap_npy_path)
                ),
                "flatmap_loaded_from_cache": bool(loaded.flatmap_loaded_from_cache),
                "depth_shape": list(loaded.depth.shape),
                "depth_dtype": str(loaded.depth.dtype),
                "depth_nbytes": int(loaded.depth.nbytes),
                "depth_npy_path": (
                    None
                    if loaded.depth_npy_path is None
                    else str(loaded.depth_npy_path)
                ),
                "depth_loaded_from_cache": bool(loaded.depth_loaded_from_cache),
                "has_spatial_transform": bool(
                    loaded.space_directions is not None
                    and loaded.space_origin is not None
                ),
            },
        )
        projected_nodes = benchmark.measure(
            "project_nodes_to_flatmap",
            lambda: project_neuron_nodes_to_flatmap(
                nodes,
                volume_set.flatmap,
                volume_set.depth,
                flatmap_style=args.flatmap.name,
                coordinate_mode=args.coordinate_mode,
                invalid_zero_sentinel=args.treat_zero_flatmap_invalid,
                invalid_negative_one_sentinel=(
                    not args.allow_negative_one_flatmap
                ),
                resolution_um=args.resolution_um,
                space_directions=volume_set.space_directions,
                space_origin=volume_set.space_origin,
                mirror_fallback=not args.no_mirror_fallback,
                mirror_coord_axis=args.mirror_axis,
                mirror_midline=args.mirror_midline,
            ),
            details=_dataframe_details,
        )
        segments = benchmark.measure(
            "build_projected_segments",
            lambda: build_projected_segments(projected_nodes),
            details=lambda value: {
                "segment_count": int(len(value.data)),
                "segment_data_shape": list(value.data.shape),
                "segment_data_nbytes": int(value.data.nbytes),
                "file_id_count": int(len(value.file_ids)),
            },
        )
        projection_summary = summarize_projection(projected_nodes, segments)
        projection_result = FlatmapProjectionResult(
            projected_nodes=projected_nodes,
            segments=segments,
            summary=projection_summary,
        )
        render_result = _build_flatmap_render_data_profiled(
            benchmark,
            projection_result.projected_nodes,
            volume_set.flatmap,
            volume_set.depth,
            y_bins=args.y_bins,
            depth_bin_um=args.depth_bin_um,
            include_depth_minus_one=not args.exclude_depth_minus_one,
            invalid_zero_sentinel=args.treat_zero_flatmap_invalid,
            invalid_negative_one_sentinel=not args.allow_negative_one_flatmap,
        )

        object_sizes = {
            "nodes_dataframe_bytes": _dataframe_memory_bytes(nodes),
            "flatmap_volume_bytes": int(volume_set.flatmap.nbytes),
            "depth_volume_bytes": int(volume_set.depth.nbytes),
            "projected_dataframe_bytes": _dataframe_memory_bytes(projected_nodes),
            "segments_data_bytes": int(segments.data.nbytes),
            "render_projected_dataframe_bytes": _dataframe_memory_bytes(
                render_result.projected_nodes
            ),
            "heatmap_volume_bytes": int(render_result.volume.nbytes),
            "render_points_bytes": int(render_result.points.nbytes),
        }

        completed_at = datetime.now(timezone.utc)
        report = {
            "status": "ok",
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "elapsed_s": (completed_at - started_at).total_seconds(),
            "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
            "environment": _environment(),
            "inputs": {
                "parquet": str(args.parquet),
                "flatmap": str(args.flatmap),
                "depth": str(args.depth),
            },
            "options": {
                "y_bins": int(args.y_bins),
                "depth_bin_um": float(args.depth_bin_um),
                "coordinate_mode": args.coordinate_mode,
                "resolution_um": float(args.resolution_um),
                "mirror_fallback": bool(not args.no_mirror_fallback),
                "mirror_axis": int(args.mirror_axis),
                "mirror_midline": (
                    None if args.mirror_midline is None else float(args.mirror_midline)
                ),
                "use_npy_cache": bool(not args.no_npy_cache),
                "npy_cache_dir": (
                    None if args.npy_cache_dir is None else str(args.npy_cache_dir)
                ),
                "treat_zero_flatmap_invalid": bool(
                    args.treat_zero_flatmap_invalid
                ),
                "treat_negative_one_flatmap_invalid": bool(
                    not args.allow_negative_one_flatmap
                ),
                "include_depth_minus_one": bool(not args.exclude_depth_minus_one),
                "candidate_limit": int(args.candidate_limit),
                "live_progress": bool(not args.quiet),
            },
            "selected_file_id": file_id,
            "selected_node_count": int(selection["selected_node_count"]),
            "selection": selection,
            "stages": [stage.to_dict() for stage in benchmark.stages],
            "object_sizes": object_sizes,
            "projection_summary": projection_summary.to_dict(),
            "render_summary": render_result.summary.to_dict(),
            "process_peak_rss_bytes": benchmark.sampler.peak_rss_bytes,
            "resource_peak_rss_bytes": _resource_peak_rss_bytes(),
            "memory_sampler": benchmark.sampler.method,
        }
        _apply_rss_limit(report, args.max_rss_gb)
        return _json_safe(report)
    finally:
        if db is not None:
            db.close()
        benchmark.close()


def _build_flatmap_render_data_profiled(
    benchmark: Benchmarker,
    projected_nodes: pd.DataFrame,
    flatmap_volume: np.ndarray,
    depth_volume: np.ndarray,
    *,
    y_bins: int,
    depth_bin_um: float,
    include_depth_minus_one: bool,
    invalid_zero_sentinel: bool,
    invalid_negative_one_sentinel: bool,
) -> FlatmapRenderResult:
    group_start = benchmark.start_group("build_flatmap_render_data")
    lookup_stats = benchmark.measure(
        "render.compute_lookup_stats_chunked",
        lambda: compute_flatmap_lookup_stats(
            flatmap_volume,
            depth_volume,
            invalid_zero_sentinel=invalid_zero_sentinel,
            invalid_negative_one_sentinel=invalid_negative_one_sentinel,
        ),
        details=lambda value: value.to_dict(),
    )
    render_result = benchmark.measure(
        "render.build_flatmap_render_data_with_stats",
        lambda: build_flatmap_render_data(
            projected_nodes,
            flatmap_volume,
            depth_volume,
            y_bins=y_bins,
            depth_bin_um=depth_bin_um,
            include_depth_minus_one=include_depth_minus_one,
            invalid_zero_sentinel=invalid_zero_sentinel,
            invalid_negative_one_sentinel=invalid_negative_one_sentinel,
            lookup_stats=lookup_stats,
        ),
        details=lambda value: {
            "projected_nodes_memory_bytes": _dataframe_memory_bytes(
                value.projected_nodes
            ),
            "volume_shape": list(value.volume.shape),
            "volume_dtype": str(value.volume.dtype),
            "volume_nbytes": int(value.volume.nbytes),
            "points_shape": list(value.points.shape),
            "points_dtype": str(value.points.dtype),
            "points_nbytes": int(value.points.nbytes),
            "point_file_id_count": int(len(value.point_file_ids)),
        },
    )
    benchmark.finish_group(
        "build_flatmap_render_data",
        group_start,
        details={
            "rows": int(len(render_result.projected_nodes)),
            "volume_nbytes": int(render_result.volume.nbytes),
            "points_nbytes": int(render_result.points.nbytes),
        },
    )
    return render_result

def _select_file_id(
    db: NeuronDatabase,
    *,
    requested_file_id: str | None,
    candidate_limit: int,
) -> dict[str, Any]:
    if requested_file_id:
        count_df = db.query(
            "SELECT COUNT(*) AS node_count FROM neurons WHERE file_id = ?",
            [requested_file_id],
        )
        count = int(count_df.iloc[0]["node_count"])
        if count <= 0:
            raise ValueError(f"file_id not found in parquet: {requested_file_id!r}")
        return {
            "mode": "requested",
            "selected_file_id": requested_file_id,
            "selected_node_count": count,
            "candidates": [],
        }

    limit = max(1, int(candidate_limit))
    candidates = db.query(
        """
        SELECT file_id, COUNT(*) AS node_count
        FROM neurons
        GROUP BY file_id
        ORDER BY node_count DESC, file_id
        LIMIT ?
        """,
        [limit],
    )
    if candidates.empty:
        raise ValueError("No file_id values were found in the parquet file.")
    records = [
        {"file_id": row["file_id"], "node_count": int(row["node_count"])}
        for _, row in candidates.iterrows()
    ]
    selected = records[0]
    return {
        "mode": "largest_by_row_count",
        "selected_file_id": selected["file_id"],
        "selected_node_count": selected["node_count"],
        "candidates": records,
    }


def _dataframe_details(df: Any) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "memory_bytes": _dataframe_memory_bytes(df),
    }


def _dataframe_memory_bytes(df: Any) -> int:
    return int(df.memory_usage(deep=True).sum())


def _resource_peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak = int(usage.ru_maxrss)
    if sys.platform == "darwin":
        return peak
    return peak * 1024


def _delta(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return int(after - before)


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pid": os.getpid(),
        "psutil_available": psutil is not None,
    }


def _apply_rss_limit(report: dict[str, Any], max_rss_gb: float | None) -> None:
    if max_rss_gb is None:
        report["rss_limit"] = None
        return
    max_rss_bytes = int(float(max_rss_gb) * BYTES_PER_GIB)
    observed = report.get("process_peak_rss_bytes")
    if observed is None:
        observed = report.get("resource_peak_rss_bytes")
    exceeded = observed is not None and int(observed) > max_rss_bytes
    report["rss_limit"] = {
        "max_rss_gb": float(max_rss_gb),
        "max_rss_bytes": max_rss_bytes,
        "observed_peak_rss_bytes": observed,
        "exceeded": bool(exceeded),
    }
    if exceeded:
        report["status"] = "rss_limit_exceeded"


def write_json(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def write_stage_csv(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "selected_file_id",
        "selected_node_count",
        "stage",
        "elapsed_s",
        "rss_before_bytes",
        "rss_after_bytes",
        "rss_delta_bytes",
        "rss_peak_bytes",
        "resource_peak_before_bytes",
        "resource_peak_after_bytes",
        "resource_peak_delta_bytes",
        "tracemalloc_current_before_bytes",
        "tracemalloc_current_after_bytes",
        "tracemalloc_peak_bytes",
        "tracemalloc_peak_delta_bytes",
        "process_peak_rss_bytes",
        "resource_peak_rss_bytes",
    ]
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for stage in report["stages"]:
            writer.writerow(
                {
                    "status": report["status"],
                    "selected_file_id": report["selected_file_id"],
                    "selected_node_count": report["selected_node_count"],
                    "stage": stage["name"],
                    "elapsed_s": stage["elapsed_s"],
                    "rss_before_bytes": stage["rss_before_bytes"],
                    "rss_after_bytes": stage["rss_after_bytes"],
                    "rss_delta_bytes": stage["rss_delta_bytes"],
                    "rss_peak_bytes": stage["rss_peak_bytes"],
                    "resource_peak_before_bytes": (
                        stage["resource_peak_before_bytes"]
                    ),
                    "resource_peak_after_bytes": (
                        stage["resource_peak_after_bytes"]
                    ),
                    "resource_peak_delta_bytes": (
                        stage["resource_peak_delta_bytes"]
                    ),
                    "tracemalloc_current_before_bytes": (
                        stage["tracemalloc_current_before_bytes"]
                    ),
                    "tracemalloc_current_after_bytes": (
                        stage["tracemalloc_current_after_bytes"]
                    ),
                    "tracemalloc_peak_bytes": stage["tracemalloc_peak_bytes"],
                    "tracemalloc_peak_delta_bytes": (
                        stage["tracemalloc_peak_delta_bytes"]
                    ),
                    "process_peak_rss_bytes": report["process_peak_rss_bytes"],
                    "resource_peak_rss_bytes": report["resource_peak_rss_bytes"],
                }
            )


def _print_live(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def _stage_detail_summary(details: dict[str, Any]) -> str:
    summary_parts: list[str] = []
    if "rows" in details:
        summary_parts.append(f"rows={int(details['rows']):,}")
    if "memory_bytes" in details:
        summary_parts.append(f"df={_format_bytes(details['memory_bytes'])}")
    if "shape" in details:
        summary_parts.append(f"shape={details['shape']}")
    if "dtype" in details:
        summary_parts.append(f"dtype={details['dtype']}")
    if "nbytes" in details:
        summary_parts.append(f"array={_format_bytes(details['nbytes'])}")
    if "true_count" in details:
        summary_parts.append(f"true={int(details['true_count']):,}")
    if "count" in details:
        summary_parts.append(f"count={int(details['count']):,}")
    if "flatmap_nbytes" in details:
        summary_parts.append(f"flatmap={_format_bytes(details['flatmap_nbytes'])}")
    if "flatmap_loaded_from_cache" in details:
        summary_parts.append(
            "flatmap_cache="
            f"{'hit' if details['flatmap_loaded_from_cache'] else 'miss'}"
        )
    if "depth_nbytes" in details:
        summary_parts.append(f"depth={_format_bytes(details['depth_nbytes'])}")
    if "depth_loaded_from_cache" in details:
        summary_parts.append(
            "depth_cache="
            f"{'hit' if details['depth_loaded_from_cache'] else 'miss'}"
        )
    if "segment_count" in details:
        summary_parts.append(f"segments={int(details['segment_count']):,}")
    if "segment_data_nbytes" in details:
        summary_parts.append(
            f"segment_data={_format_bytes(details['segment_data_nbytes'])}"
        )
    if "volume_nbytes" in details:
        summary_parts.append(f"heatmap={_format_bytes(details['volume_nbytes'])}")
    if "points_nbytes" in details:
        summary_parts.append(f"points={_format_bytes(details['points_nbytes'])}")
    if "voxel_count" in details:
        summary_parts.append(f"voxels={int(details['voxel_count']):,}")
    return " | ".join(summary_parts)


def print_report(report: dict[str, Any]) -> None:
    print("Flatmap Pipeline Benchmark")
    print("=" * 72)
    print(f"Status: {report['status']}")
    print(f"Parquet: {report['inputs']['parquet']}")
    print(f"Flatmap: {report['inputs']['flatmap']}")
    print(f"Depth: {report['inputs']['depth']}")
    print(
        "NPY cache: "
        f"{'enabled' if report['options']['use_npy_cache'] else 'disabled'}"
    )
    print(
        "Selected neuron: "
        f"{report['selected_file_id']} "
        f"({report['selected_node_count']:,} rows)"
    )
    print(
        "Peak RSS: "
        f"{_format_bytes(report['process_peak_rss_bytes'])} "
        f"(resource peak {_format_bytes(report['resource_peak_rss_bytes'])})"
    )
    print("")
    print(
        f"{'Stage':<30} {'Time':>9} {'RSS Before':>12} {'RSS After':>12} "
        f"{'RSS Peak':>12} {'Py Peak':>12}"
    )
    print("-" * 92)
    for stage in report["stages"]:
        print(
            f"{stage['name']:<30} "
            f"{stage['elapsed_s']:>8.3f}s "
            f"{_format_bytes(stage['rss_before_bytes']):>12} "
            f"{_format_bytes(stage['rss_after_bytes']):>12} "
            f"{_format_bytes(stage['rss_peak_bytes']):>12} "
            f"{_format_bytes(stage['tracemalloc_peak_delta_bytes']):>12}"
        )
    print("")
    print("Major object sizes")
    for key, value in report["object_sizes"].items():
        print(f"  {key}: {_format_bytes(value)}")
    print("")
    projection = report["projection_summary"]
    render = report["render_summary"]
    print(
        "Projection: "
        f"{projection['valid_nodes']:,}/{projection['total_nodes']:,} valid nodes, "
        f"{projection['rendered_segments']:,} rendered segments"
    )
    print(
        "Render: "
        f"{render['rendered_nodes']:,} rendered nodes, "
        f"{render['nonzero_voxels']:,} nonzero voxels, "
        f"{render['depth_bins']}x{render['y_bins']}x{render['x_bins']} volume"
    )
    if report.get("rss_limit"):
        limit = report["rss_limit"]
        print(
            "RSS limit: "
            f"{limit['max_rss_gb']:.3f} GiB, "
            f"observed {_format_bytes(limit['observed_peak_rss_bytes'])}, "
            f"exceeded={limit['exceeded']}"
        )


def _format_bytes(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{sign}{value:.0f} {unit}"
            return f"{sign}{value:.1f} {unit}"
        value /= 1024.0
    return f"{sign}{value:.1f} TiB"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except ImportError:  # pragma: no cover
        pass
    return value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_benchmark(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.output_json is not None:
        write_json(report, args.output_json)
    if args.output_csv is not None:
        write_stage_csv(report, args.output_csv)
    print_report(report)

    limit = report.get("rss_limit")
    if limit and limit.get("exceeded"):
        return EXIT_RSS_LIMIT_EXCEEDED
    return 0


if __name__ == "__main__":
    sys.exit(main())
