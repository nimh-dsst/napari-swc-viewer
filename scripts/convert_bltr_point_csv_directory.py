#!/usr/bin/env python
"""Convert a directory of BLTR point CSVs into one standardized Parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from napari_neuron_navigator.point_import import (
    BLTR_EXTRA_COLUMNS,
    OPTIONAL_POINT_COLUMNS,
    REQUIRED_POINT_COLUMNS,
    convert_bltr_point_csv_directory_to_parquet,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Convert a directory of BLTR-format two-row-header point CSV files into "
            "one standardized point Parquet with an added origin_csv column."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Required standardized columns:
  {", ".join(REQUIRED_POINT_COLUMNS)}

Optional standardized columns:
  {", ".join(OPTIONAL_POINT_COLUMNS)}

Preserved BLTR columns:
  {", ".join(BLTR_EXTRA_COLUMNS)}

Example:
  %(prog)s "bltr cases" bltr_combined.parquet
""",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing top-level BLTR CSV files",
    )
    parser.add_argument(
        "output_parquet",
        type=Path,
        help="Output standardized Parquet path",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """Run the BLTR directory-to-Parquet conversion."""

    parsed = parse_args(args)
    try:
        summary = convert_bltr_point_csv_directory_to_parquet(
            parsed.input_dir,
            parsed.output_parquet,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote {summary.rows_written:,} standardized point rows from "
        f"{summary.processed_files} BLTR CSV file(s) to {parsed.output_parquet}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
