#!/usr/bin/env python
"""Compare two mirrored neuron parquet files using DuckDB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import duckdb

VALID_AXES = ("x", "y", "z")
NON_COORD_COLUMNS = (
    "file_id",
    "node_id",
    "type",
    "radius",
    "parent_id",
    "region_id",
    "region_name",
    "region_acronym",
    "subject",
    "neuron_id",
)
COORD_COLUMNS = ("x", "y", "z")


def _quote_path(path: Path | str) -> str:
    """Return a DuckDB-safe single-quoted parquet path."""
    return str(Path(path)).replace("\\", "/").replace("'", "''")


def _sum_if(condition: str) -> str:
    """Build an integer-valued conditional sum expression."""
    return f"SUM(CASE WHEN {condition} THEN 1 ELSE 0 END)"


def _read_schema(con: duckdb.DuckDBPyConnection, path: Path | str) -> list[dict[str, str]]:
    """Read parquet schema metadata."""
    escaped = _quote_path(path)
    rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{escaped}')").fetchall()
    return [{"name": row[0], "type": row[1]} for row in rows]


def _read_row_count(con: duckdb.DuckDBPyConnection, path: Path | str) -> int:
    """Read parquet row count."""
    escaped = _quote_path(path)
    return int(con.execute(f"SELECT COUNT(*) FROM read_parquet('{escaped}')").fetchone()[0])


def _read_ranges(con: duckdb.DuckDBPyConnection, path: Path | str) -> dict[str, list[float]]:
    """Read coordinate ranges for x, y, z."""
    escaped = _quote_path(path)
    row = con.execute(
        f"""
        SELECT
            MIN(x), MAX(x),
            MIN(y), MAX(y),
            MIN(z), MAX(z)
        FROM read_parquet('{escaped}')
        """
    ).fetchone()
    return {
        "x": [float(row[0]), float(row[1])],
        "y": [float(row[2]), float(row[3])],
        "z": [float(row[4]), float(row[5])],
    }


def _comparison_query(left_path: Path | str, right_path: Path | str, axis: str, tolerance: float) -> str:
    """Build the main positional comparison query."""
    left_escaped = _quote_path(left_path)
    right_escaped = _quote_path(right_path)
    return f"""
        SELECT
            COUNT(*) AS total_rows,
            {_sum_if("l.file_id IS DISTINCT FROM r.file_id")} AS diff_file_id,
            {_sum_if("l.node_id IS DISTINCT FROM r.node_id")} AS diff_node_id,
            {_sum_if("l.type IS DISTINCT FROM r.type")} AS diff_type,
            {_sum_if(f"ABS(l.x - r.x) > {tolerance}")} AS diff_x,
            {_sum_if(f"ABS(l.y - r.y) > {tolerance}")} AS diff_y,
            {_sum_if(f"ABS(l.z - r.z) > {tolerance}")} AS diff_z,
            {_sum_if(f"ABS(l.{axis} - r.{axis}) <= {tolerance}")} AS equal_axis_rows,
            MIN(l.{axis} + r.{axis}) AS min_axis_sum,
            MAX(l.{axis} + r.{axis}) AS max_axis_sum,
            {_sum_if(f"ABS(l.radius - r.radius) > {tolerance}")} AS diff_radius,
            {_sum_if("l.parent_id IS DISTINCT FROM r.parent_id")} AS diff_parent_id,
            {_sum_if("l.region_id IS DISTINCT FROM r.region_id")} AS diff_region_id,
            {_sum_if("l.region_name IS DISTINCT FROM r.region_name")} AS diff_region_name,
            {_sum_if("l.region_acronym IS DISTINCT FROM r.region_acronym")} AS diff_region_acronym,
            {_sum_if("l.subject IS DISTINCT FROM r.subject")} AS diff_subject,
            {_sum_if("l.neuron_id IS DISTINCT FROM r.neuron_id")} AS diff_neuron_id
        FROM read_parquet('{left_escaped}') l
        POSITIONAL JOIN read_parquet('{right_escaped}') r
    """


def _mirror_error_query(
    left_path: Path | str,
    right_path: Path | str,
    axis: str,
    tolerance: float,
    expected_sum: float,
) -> str:
    """Build the mirror-sum verification query."""
    left_escaped = _quote_path(left_path)
    right_escaped = _quote_path(right_path)
    return f"""
        SELECT
            {_sum_if(f"ABS((l.{axis} + r.{axis}) - {expected_sum}) > {tolerance}")} AS bad_mirror_sum,
            MAX(ABS((l.{axis} + r.{axis}) - {expected_sum})) AS max_mirror_error
        FROM read_parquet('{left_escaped}') l
        POSITIONAL JOIN read_parquet('{right_escaped}') r
    """


def _top_region_remaps_query(
    left_path: Path | str,
    right_path: Path | str,
    limit: int,
) -> str:
    """Build the top region remap query."""
    left_escaped = _quote_path(left_path)
    right_escaped = _quote_path(right_path)
    return f"""
        SELECT
            l.region_acronym AS left_region,
            r.region_acronym AS right_region,
            COUNT(*) AS count
        FROM read_parquet('{left_escaped}') l
        POSITIONAL JOIN read_parquet('{right_escaped}') r
        WHERE l.region_id IS DISTINCT FROM r.region_id
        GROUP BY 1, 2
        ORDER BY count DESC, left_region, right_region
        LIMIT {int(limit)}
    """


def _sample_region_mismatch_query(
    left_path: Path | str,
    right_path: Path | str,
    axis: str,
    limit: int,
) -> str:
    """Build a small mismatch sample query."""
    left_escaped = _quote_path(left_path)
    right_escaped = _quote_path(right_path)
    return f"""
        SELECT
            l.file_id,
            l.node_id,
            l.neuron_id,
            l.region_id AS left_region_id,
            r.region_id AS right_region_id,
            l.region_acronym AS left_region,
            r.region_acronym AS right_region,
            l.{axis} AS left_axis,
            r.{axis} AS right_axis,
            l.{axis} + r.{axis} AS axis_sum
        FROM read_parquet('{left_escaped}') l
        POSITIONAL JOIN read_parquet('{right_escaped}') r
        WHERE l.region_id IS DISTINCT FROM r.region_id
        LIMIT {int(limit)}
    """


def collect_comparison_stats(
    left_path: Path | str,
    right_path: Path | str,
    *,
    axis: str = "z",
    tolerance: float = 1e-6,
    expected_sum: float | None = None,
    top_region_remaps: int = 20,
    sample_region_mismatches: int = 10,
) -> dict[str, Any]:
    """Collect mirrored parquet comparison stats."""
    if axis not in VALID_AXES:
        raise ValueError(f"axis must be one of {VALID_AXES}, got {axis!r}")

    left_path = Path(left_path)
    right_path = Path(right_path)
    if not left_path.exists():
        raise FileNotFoundError(f"Left parquet not found: {left_path}")
    if not right_path.exists():
        raise FileNotFoundError(f"Right parquet not found: {right_path}")

    con = duckdb.connect()
    try:
        left_schema = _read_schema(con, left_path)
        right_schema = _read_schema(con, right_path)
        left_rows = _read_row_count(con, left_path)
        right_rows = _read_row_count(con, right_path)
        left_ranges = _read_ranges(con, left_path)
        right_ranges = _read_ranges(con, right_path)

        comparison_columns = [
            "total_rows",
            "diff_file_id",
            "diff_node_id",
            "diff_type",
            "diff_x",
            "diff_y",
            "diff_z",
            "equal_axis_rows",
            "min_axis_sum",
            "max_axis_sum",
            "diff_radius",
            "diff_parent_id",
            "diff_region_id",
            "diff_region_name",
            "diff_region_acronym",
            "diff_subject",
            "diff_neuron_id",
        ]
        comparison_values = con.execute(
            _comparison_query(left_path, right_path, axis, tolerance)
        ).fetchone()
        comparison = dict(zip(comparison_columns, comparison_values, strict=True))

        inferred_sum = 0.5 * (
            float(comparison["min_axis_sum"]) + float(comparison["max_axis_sum"])
        )
        if expected_sum is None:
            expected_sum = inferred_sum

        mirror_error_columns = ["bad_mirror_sum", "max_mirror_error"]
        mirror_error_values = con.execute(
            _mirror_error_query(left_path, right_path, axis, tolerance, expected_sum)
        ).fetchone()
        mirror_error = dict(zip(mirror_error_columns, mirror_error_values, strict=True))

        top_remaps_rows = con.execute(
            _top_region_remaps_query(left_path, right_path, top_region_remaps)
        ).fetchall()
        sample_rows = con.execute(
            _sample_region_mismatch_query(left_path, right_path, axis, sample_region_mismatches)
        ).fetchall()
    finally:
        con.close()

    total_rows = int(comparison["total_rows"])
    diff_counts = {
        "file_id": int(comparison["diff_file_id"]),
        "node_id": int(comparison["diff_node_id"]),
        "type": int(comparison["diff_type"]),
        "x": int(comparison["diff_x"]),
        "y": int(comparison["diff_y"]),
        "z": int(comparison["diff_z"]),
        "radius": int(comparison["diff_radius"]),
        "parent_id": int(comparison["diff_parent_id"]),
        "region_id": int(comparison["diff_region_id"]),
        "region_name": int(comparison["diff_region_name"]),
        "region_acronym": int(comparison["diff_region_acronym"]),
        "subject": int(comparison["diff_subject"]),
        "neuron_id": int(comparison["diff_neuron_id"]),
    }

    return {
        "left": {
            "path": str(left_path),
            "rows": left_rows,
            "schema": left_schema,
            "ranges": left_ranges,
        },
        "right": {
            "path": str(right_path),
            "rows": right_rows,
            "schema": right_schema,
            "ranges": right_ranges,
        },
        "comparison": {
            "axis": axis,
            "tolerance": tolerance,
            "total_rows": total_rows,
            "schema_match": left_schema == right_schema,
            "row_count_match": left_rows == right_rows,
            "diff_counts": diff_counts,
            "equal_axis_rows": int(comparison["equal_axis_rows"]),
            "expected_axis_sum": float(expected_sum),
            "inferred_axis_sum": float(inferred_sum),
            "min_axis_sum": float(comparison["min_axis_sum"]),
            "max_axis_sum": float(comparison["max_axis_sum"]),
            "axis_sum_span": float(comparison["max_axis_sum"]) - float(comparison["min_axis_sum"]),
            "bad_mirror_sum": int(mirror_error["bad_mirror_sum"]),
            "max_mirror_error": float(mirror_error["max_mirror_error"] or 0.0),
            "region_mismatch_pct": (
                float(diff_counts["region_id"]) / float(total_rows) if total_rows else 0.0
            ),
        },
        "top_region_remaps": [
            {"left_region": row[0], "right_region": row[1], "count": int(row[2])}
            for row in top_remaps_rows
        ],
        "sample_region_mismatches": [
            {
                "file_id": row[0],
                "node_id": int(row[1]),
                "neuron_id": row[2],
                "left_region_id": int(row[3]),
                "right_region_id": int(row[4]),
                "left_region": row[5],
                "right_region": row[6],
                "left_axis": float(row[7]),
                "right_axis": float(row[8]),
                "axis_sum": float(row[9]),
            }
            for row in sample_rows
        ],
    }


def _format_range(name: str, bounds: list[float]) -> str:
    """Format a coordinate range line."""
    return f"{name}[{bounds[0]:.6f}, {bounds[1]:.6f}]"


def render_report(stats: dict[str, Any]) -> str:
    """Render a human-readable comparison report."""
    left = stats["left"]
    right = stats["right"]
    comparison = stats["comparison"]
    diff_counts = comparison["diff_counts"]

    lines = [
        "Files",
        f"  left: {left['path']}",
        f"  right: {right['path']}",
        f"  row_count_match: {comparison['row_count_match']} ({left['rows']} vs {right['rows']})",
        f"  schema_match: {comparison['schema_match']}",
        "",
        "Ranges",
        "  left:  "
        + " ".join(_format_range(axis, left["ranges"][axis]) for axis in COORD_COLUMNS),
        "  right: "
        + " ".join(_format_range(axis, right["ranges"][axis]) for axis in COORD_COLUMNS),
        "",
        "Comparison",
        f"  total_rows: {comparison['total_rows']}",
        f"  mirror_axis: {comparison['axis']}",
        f"  expected_axis_sum: {comparison['expected_axis_sum']:.6f}",
        f"  inferred_axis_sum: {comparison['inferred_axis_sum']:.6f}",
        f"  axis_sum_min_max: [{comparison['min_axis_sum']:.6f}, {comparison['max_axis_sum']:.6f}]",
        f"  axis_sum_span: {comparison['axis_sum_span']:.6f}",
        f"  bad_mirror_sum: {comparison['bad_mirror_sum']}",
        f"  max_mirror_error: {comparison['max_mirror_error']:.6f}",
        f"  equal_axis_rows: {comparison['equal_axis_rows']}",
        f"  region_mismatch_pct: {comparison['region_mismatch_pct']:.6%}",
        "  diff_counts:",
    ]

    for column in NON_COORD_COLUMNS + COORD_COLUMNS:
        lines.append(f"    {column}: {diff_counts[column]}")

    if stats["top_region_remaps"]:
        lines.append("")
        lines.append("Top Region Remaps")
        for row in stats["top_region_remaps"]:
            lines.append(
                f"  {row['left_region']} -> {row['right_region']}: {row['count']}"
            )

    if stats["sample_region_mismatches"]:
        lines.append("")
        lines.append("Sample Region Mismatches")
        for row in stats["sample_region_mismatches"]:
            lines.append(
                "  "
                f"{row['file_id']} node={row['node_id']} neuron={row['neuron_id']} "
                f"{row['left_region']}({row['left_region_id']}) -> "
                f"{row['right_region']}({row['right_region_id']}) "
                f"axis=({row['left_axis']:.6f}, {row['right_axis']:.6f}) "
                f"sum={row['axis_sum']:.6f}"
            )

    return "\n".join(lines)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compare two mirrored neuron parquet files using DuckDB.",
    )
    parser.add_argument("left_parquet", type=Path, help="Left-aligned parquet path")
    parser.add_argument("right_parquet", type=Path, help="Right-aligned parquet path")
    parser.add_argument(
        "--axis",
        choices=VALID_AXES,
        default="z",
        help="Mirrored coordinate axis (default: z)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Floating-point comparison tolerance (default: 1e-6)",
    )
    parser.add_argument(
        "--expected-sum",
        type=float,
        default=None,
        help="Expected constant sum across the mirror axis; defaults to the inferred midpoint",
    )
    parser.add_argument(
        "--top-region-remaps",
        type=int,
        default=20,
        help="Number of top region remaps to show (default: 20)",
    )
    parser.add_argument(
        "--sample-region-mismatches",
        type=int,
        default=10,
        help="Number of mismatch sample rows to show (default: 10)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a text report",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """Run the mirrored parquet comparison CLI."""
    parsed = parse_args(args)

    try:
        stats = collect_comparison_stats(
            parsed.left_parquet,
            parsed.right_parquet,
            axis=parsed.axis,
            tolerance=parsed.tolerance,
            expected_sum=parsed.expected_sum,
            top_region_remaps=parsed.top_region_remaps,
            sample_region_mismatches=parsed.sample_region_mismatches,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if parsed.json:
        print(json.dumps(stats, indent=2, sort_keys=True))
    else:
        print(render_report(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
