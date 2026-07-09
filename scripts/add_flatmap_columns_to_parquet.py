#!/usr/bin/env python
"""Add flatmap/depth coordinate columns to a neuron Parquet file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from napari_swc_viewer.flatmap_parquet import (
    DEFAULT_CCFV3_MIRROR_MIDLINE_UM,
    DEFAULT_FLATMAP_PARQUET_BATCH_SIZE,
    augment_neuron_parquet_with_flatmap,
)
from napari_swc_viewer.flatmap_projection import COORDINATE_MODE_MICRONS


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add NRRD-derived flatmap/depth projection columns to a neuron "
            "Parquet file."
        ),
    )
    parser.add_argument("source_parquet", type=Path)
    parser.add_argument("output_parquet", type=Path)
    parser.add_argument("--flatmap", required=True, type=Path, help="Flatmap NRRD path.")
    parser.add_argument("--depth", required=True, type=Path, help="Depth NRRD path.")
    parser.add_argument(
        "--file-id",
        action="append",
        dest="file_ids",
        default=None,
        help=(
            "Only augment rows with this file_id. Repeat to include multiple "
            "file IDs. Defaults to the whole source Parquet."
        ),
    )
    parser.add_argument(
        "--flatmap-style",
        default="",
        help="Flatmap style label stored in metadata. Defaults to flatmap filename.",
    )
    parser.add_argument(
        "--coordinate-mode",
        default=COORDINATE_MODE_MICRONS,
        choices=("microns", "voxels"),
        help="Interpret source x/y/z as CCF microns or lookup voxel indices.",
    )
    parser.add_argument(
        "--no-mirror-fallback",
        action="store_true",
        help="Disable opposite-hemisphere retry for invalid direct lookup rows.",
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
        help="Treat flatmap lookup coordinates (0, 0) as invalid.",
    )
    parser.add_argument(
        "--allow-negative-one-flatmap",
        action="store_true",
        help="Do not treat flatmap lookup coordinates (-1, -1) as invalid.",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=DEFAULT_FLATMAP_PARQUET_BATCH_SIZE,
        help=f"Parquet rows per processing batch (default: {DEFAULT_FLATMAP_PARQUET_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        help="Parquet compression codec for output (default: zstd).",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)

    try:
        summary = augment_neuron_parquet_with_flatmap(
            parsed.source_parquet,
            parsed.output_parquet,
            parsed.flatmap,
            parsed.depth,
            file_ids=parsed.file_ids,
            coordinate_mode=parsed.coordinate_mode,
            flatmap_style=parsed.flatmap_style,
            mirror_fallback=not parsed.no_mirror_fallback,
            mirror_coord_axis=parsed.mirror_axis,
            mirror_midline=parsed.mirror_midline,
            invalid_zero_sentinel=parsed.treat_zero_flatmap_invalid,
            invalid_negative_one_sentinel=not parsed.allow_negative_one_flatmap,
            batch_size=parsed.batch_size,
            compression=parsed.compression,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Rows written: {summary.rows:,}")
    print(f"Direct lookup rows: {summary.direct_rows:,}")
    print(f"Mirrored lookup rows: {summary.mirrored_rows:,}")
    print(f"Unmapped rows: {summary.unmapped_rows:,}")
    print(f"Output path: {summary.output_parquet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
