#!/usr/bin/env python
"""CLI script to convert SWC files to annotated Parquet format.

Usage:
    python scripts/swc_to_parquet.py input_dir output.parquet
    python scripts/swc_to_parquet.py input_dir output.parquet --resolution 10
    python scripts/swc_to_parquet.py input_dir output.parquet --workers 4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from napari_swc_viewer.parquet import (
    get_parquet_summary,
    swc_files_to_parquet,
)


def main():
    parser = argparse.ArgumentParser(
        description="Convert SWC files to annotated Parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert all SWC files in a directory
    python swc_to_parquet.py test_data/ neurons.parquet

    # Use higher resolution annotation (10um)
    python swc_to_parquet.py test_data/ neurons.parquet --resolution 10

    # Process in parallel with 4 workers
    python swc_to_parquet.py test_data/ neurons.parquet --workers 4

    # Non-recursive search
    python swc_to_parquet.py test_data/ neurons.parquet --no-recursive
        """,
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to SWC file or directory of SWC files",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output Parquet file path",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=25,
        choices=[10, 25, 50, 100],
        help="Allen CCF resolution in microns (default: 25)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for Allen SDK cache (default: ~/.allen_sdk_cache)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary of existing Parquet file and exit",
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Summary mode
    if args.summary:
        if not args.output.exists():
            logger.error(f"Parquet file not found: {args.output}")
            sys.exit(1)

        summary = get_parquet_summary(args.output)
        print(f"\nParquet file: {args.output}")
        print(f"  Total rows: {summary['n_rows']:,}")
        print(f"  Files: {summary['n_files']:,}")
        print(f"  Subjects: {summary['n_subjects']:,}")
        print(f"  Regions: {summary['n_regions']:,}")
        print("\n  Top 10 regions by node count:")
        for acronym, count in list(summary["regions"].items())[:10]:
            print(f"    {acronym}: {count:,}")
        return

    # Validate input
    if not args.input.exists():
        logger.error(f"Input path not found: {args.input}")
        sys.exit(1)

    # Check if output already exists
    if args.output.exists():
        response = input(f"Output file {args.output} exists. Overwrite? [y/N] ")
        if response.lower() != "y":
            logger.info("Aborted")
            sys.exit(0)

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Run conversion
    logger.info(f"Converting SWC files from {args.input} to {args.output}")
    logger.info(f"Resolution: {args.resolution}um, Workers: {args.workers}")

    n_processed = swc_files_to_parquet(
        input_path=args.input,
        output_path=args.output,
        resolution=args.resolution,
        cache_dir=args.cache_dir,
        recursive=not args.no_recursive,
        n_workers=args.workers,
    )

    if n_processed == 0:
        logger.warning("No files were processed")
        sys.exit(1)

    # Print summary
    summary = get_parquet_summary(args.output)
    logger.info(f"\nConversion complete!")
    logger.info(f"  Processed: {n_processed} files")
    logger.info(f"  Total rows: {summary['n_rows']:,}")
    logger.info(f"  Unique regions: {summary['n_regions']}")


if __name__ == "__main__":
    main()
