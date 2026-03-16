#!/usr/bin/env python
"""Normalize atlas-registered point CSVs into standardized Parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from napari_swc_viewer.point_import import (
    OPTIONAL_POINT_COLUMNS,
    REQUIRED_POINT_COLUMNS,
    convert_point_csv_to_parquet,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Convert a raw atlas-registered point CSV plus a JSON column mapping "
            "into standardized point Parquet."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Required standardized columns:
  {", ".join(REQUIRED_POINT_COLUMNS)}

Optional standardized columns:
  {", ".join(OPTIONAL_POINT_COLUMNS)}

Mapping JSON format:
  {{"label":"marker","x":"atlas_x","y":"atlas_y","z":"atlas_z"}}

Example:
  %(prog)s raw_points.csv mapping.json points_standardized.parquet
""",
    )
    parser.add_argument("input_csv", type=Path, help="Raw input CSV file")
    parser.add_argument(
        "mapping_json",
        type=Path,
        help="JSON file mapping standard column names to source CSV headers",
    )
    parser.add_argument(
        "output_parquet",
        type=Path,
        help="Output standardized Parquet path",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """Run the CSV-to-Parquet conversion."""

    parsed = parse_args(args)
    try:
        standardized = convert_point_csv_to_parquet(
            parsed.input_csv,
            parsed.mapping_json,
            parsed.output_parquet,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote {len(standardized):,} standardized point rows to "
        f"{parsed.output_parquet}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
