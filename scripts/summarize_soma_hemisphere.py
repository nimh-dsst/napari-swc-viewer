#!/usr/bin/env python
"""Print a hemisphere summary for soma nodes in a Parquet file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from napari_swc_viewer.soma_summary import (
    DEFAULT_COORD_AXIS,
    DEFAULT_MIDLINE,
    DEFAULT_TOLERANCE,
    format_soma_hemisphere_summary,
    summarize_soma_hemispheres,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the hemisphere location of all soma nodes in a Parquet file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s pfc_left.parquet
  %(prog)s isocortex_total.parquet --midline {DEFAULT_MIDLINE} --coord-axis 2
""",
    )
    parser.add_argument(
        "parquet_path",
        type=Path,
        help="Input Parquet file path",
    )
    parser.add_argument(
        "--coord-axis",
        type=int,
        default=DEFAULT_COORD_AXIS,
        choices=[0, 1, 2],
        help="Coordinate axis to treat as left-right (default: 2, z)",
    )
    parser.add_argument(
        "--midline",
        type=float,
        default=DEFAULT_MIDLINE,
        help=(
            "Midline position in microns "
            f"(default: {DEFAULT_MIDLINE:.1f}, Allen Mouse 10 um on axis 2)"
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="Midline tolerance in microns (default: 1.0)",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """Run the soma hemisphere summary CLI."""
    parsed = parse_args(args)

    try:
        summary = summarize_soma_hemispheres(
            parsed.parquet_path,
            coord_axis=parsed.coord_axis,
            midline=parsed.midline,
            tolerance=parsed.tolerance,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(format_soma_hemisphere_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
