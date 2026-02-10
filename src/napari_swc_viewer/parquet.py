"""Parquet schema and SWC-to-Parquet conversion with brain region annotation.

This module provides functionality to:
1. Define the Parquet schema for annotated neuron data
2. Convert SWC files to Parquet format with region annotations
3. Batch process multiple SWC files with parallel processing
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .region import (
    build_region_lookup,
    get_region_ids_vectorized,
    setup_allen_sdk,
)
from .swc import parse_swc

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Parquet schema for annotated neuron data
NEURON_SCHEMA = pa.schema(
    [
        pa.field("file_id", pa.string()),  # Source SWC filename
        pa.field("node_id", pa.int32()),  # Node ID within file
        pa.field("type", pa.int32()),  # Node type (1=soma, 2=axon, etc.)
        pa.field("x", pa.float64()),  # X coordinate (microns)
        pa.field("y", pa.float64()),  # Y coordinate (microns)
        pa.field("z", pa.float64()),  # Z coordinate (microns)
        pa.field("radius", pa.float64()),  # Node radius
        pa.field("parent_id", pa.int32()),  # Parent node ID
        pa.field("region_id", pa.int32()),  # Allen CCF region ID
        pa.field("region_name", pa.string()),  # Full region name
        pa.field("region_acronym", pa.string()),  # e.g., "VISp"
        pa.field("subject", pa.string()),  # Subject ID (from filename)
        pa.field("neuron_id", pa.string()),  # Neuron identifier (from filename)
    ]
)


def parse_filename_metadata(filename: str) -> dict[str, str]:
    """Extract subject and neuron ID from SWC filename.

    Attempts to parse common naming conventions:
    - BIL format: {neuron_id}_{subject}_{slice}-X{x}-Y{y}.swc
    - Simple format: {subject}_{suffix}.swc

    Parameters
    ----------
    filename : str
        The SWC filename (without path).

    Returns
    -------
    dict[str, str]
        Dictionary with 'subject' and 'neuron_id' keys.
    """
    stem = Path(filename).stem

    # Try BIL format: 1059281710_18462_6029-X10270-Y8859
    bil_match = re.match(r"(\d+)_(\d+)_", stem)
    if bil_match:
        return {
            "neuron_id": bil_match.group(1),
            "subject": bil_match.group(2),
        }

    # Try format: H19.03.315.11.12.01.01_1024476468_m
    h19_match = re.match(r"(H\d+\.\d+\.\d+\.\d+\.\d+\.\d+\.\d+)_(\d+)", stem)
    if h19_match:
        return {
            "subject": h19_match.group(1),
            "neuron_id": h19_match.group(2),
        }

    # Fallback: use whole stem as both
    return {
        "subject": stem,
        "neuron_id": stem,
    }


def swc_to_annotated_rows(
    swc_path: Path,
    annotation_volume: NDArray[np.int32],
    structure_tree,
    region_lookup: dict[int, dict],
    resolution: int = 25,
) -> list[dict]:
    """Convert a single SWC file to annotated row dictionaries.

    Parameters
    ----------
    swc_path : Path
        Path to the SWC file.
    annotation_volume : NDArray[np.int32]
        3D annotation volume from Allen SDK.
    structure_tree : StructureTree
        Allen SDK structure tree.
    region_lookup : dict[int, dict]
        Precomputed mapping from region ID to region info.
    resolution : int, default=25
        Resolution of the annotation volume in microns.

    Returns
    -------
    list[dict]
        List of row dictionaries matching NEURON_SCHEMA.
    """
    swc_data = parse_swc(swc_path)
    filename = swc_path.name
    metadata = parse_filename_metadata(filename)

    # Get region IDs for all coordinates (vectorized)
    region_ids = get_region_ids_vectorized(
        swc_data.coords, annotation_volume, resolution
    )

    rows = []
    for i in range(swc_data.n_nodes):
        region_id = int(region_ids[i])
        region_info = region_lookup.get(region_id, {})

        rows.append(
            {
                "file_id": filename,
                "node_id": int(swc_data.ids[i]),
                "type": int(swc_data.types[i]),
                "x": float(swc_data.coords[i, 0]),
                "y": float(swc_data.coords[i, 1]),
                "z": float(swc_data.coords[i, 2]),
                "radius": float(swc_data.radii[i]),
                "parent_id": int(swc_data.parents[i]),
                "region_id": region_id,
                "region_name": region_info.get("name", ""),
                "region_acronym": region_info.get("acronym", ""),
                "subject": metadata["subject"],
                "neuron_id": metadata["neuron_id"],
            }
        )

    return rows


def _process_single_swc(args: tuple) -> list[dict]:
    """Worker function for parallel processing. Handles its own Allen SDK setup."""
    swc_path, resolution, cache_dir = args

    # Each worker loads its own copy of Allen SDK data
    _, annotation_volume, structure_tree = setup_allen_sdk(resolution, cache_dir)
    region_lookup = build_region_lookup(structure_tree)

    return swc_to_annotated_rows(
        swc_path, annotation_volume, structure_tree, region_lookup, resolution
    )


def discover_swc_files(input_path: Path, recursive: bool = True) -> list[Path]:
    """Discover SWC files in a directory.

    Parameters
    ----------
    input_path : Path
        Path to a directory or single SWC file.
    recursive : bool, default=True
        If True, search subdirectories recursively.

    Returns
    -------
    list[Path]
        List of paths to SWC files.
    """
    input_path = Path(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() == ".swc":
            return [input_path]
        return []

    if recursive:
        return sorted(input_path.rglob("*.swc"))
    return sorted(input_path.glob("*.swc"))


def swc_files_to_parquet(
    input_path: Path | str,
    output_path: Path | str,
    resolution: int = 25,
    cache_dir: Path | str | None = None,
    recursive: bool = True,
    n_workers: int = 1,
    batch_size: int = 100,
) -> int:
    """Convert SWC files to a single annotated Parquet file.

    Parameters
    ----------
    input_path : Path or str
        Path to a directory of SWC files or a single SWC file.
    output_path : Path or str
        Path for the output Parquet file.
    resolution : int, default=25
        Allen CCF resolution in microns.
    cache_dir : Path or str, optional
        Directory to cache Allen SDK data.
    recursive : bool, default=True
        If True, search subdirectories recursively.
    n_workers : int, default=1
        Number of parallel workers. Use 1 for serial processing.
    batch_size : int, default=100
        Number of files to process before writing to disk.

    Returns
    -------
    int
        Number of SWC files processed.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Discover SWC files
    swc_files = discover_swc_files(input_path, recursive)
    if not swc_files:
        logger.warning(f"No SWC files found in {input_path}")
        return 0

    logger.info(f"Found {len(swc_files)} SWC files to process")

    # Set up Allen SDK (for serial processing or building lookup)
    _, annotation_volume, structure_tree = setup_allen_sdk(resolution, cache_dir)
    region_lookup = build_region_lookup(structure_tree)

    all_rows: list[dict] = []
    processed = 0

    if n_workers > 1:
        # Parallel processing
        args_list = [
            (swc_path, resolution, str(cache_dir) if cache_dir else None)
            for swc_path in swc_files
        ]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_single_swc, args): args[0]
                for args in args_list
            }

            for future in as_completed(futures):
                swc_path = futures[future]
                try:
                    rows = future.result()
                    all_rows.extend(rows)
                    processed += 1

                    if processed % 10 == 0:
                        logger.info(f"Processed {processed}/{len(swc_files)} files")

                except Exception as e:
                    logger.error(f"Error processing {swc_path}: {e}")
    else:
        # Serial processing
        for i, swc_path in enumerate(swc_files):
            try:
                rows = swc_to_annotated_rows(
                    swc_path,
                    annotation_volume,
                    structure_tree,
                    region_lookup,
                    resolution,
                )
                all_rows.extend(rows)
                processed += 1

                if processed % 10 == 0:
                    logger.info(f"Processed {processed}/{len(swc_files)} files")

            except Exception as e:
                logger.error(f"Error processing {swc_path}: {e}")

    # Write to Parquet
    if all_rows:
        table = pa.Table.from_pylist(all_rows, schema=NEURON_SCHEMA)
        pq.write_table(table, output_path, compression="snappy")
        logger.info(
            f"Wrote {len(all_rows)} rows from {processed} files to {output_path}"
        )

    return processed


def append_to_parquet(
    existing_path: Path | str,
    new_swc_path: Path | str,
    resolution: int = 25,
    cache_dir: Path | str | None = None,
) -> int:
    """Append new SWC files to an existing Parquet file.

    Parameters
    ----------
    existing_path : Path or str
        Path to existing Parquet file.
    new_swc_path : Path or str
        Path to new SWC file or directory.
    resolution : int, default=25
        Allen CCF resolution in microns.
    cache_dir : Path or str, optional
        Directory to cache Allen SDK data.

    Returns
    -------
    int
        Number of new files appended.
    """
    existing_path = Path(existing_path)
    new_swc_path = Path(new_swc_path)

    # Load existing data
    existing_table = pq.read_table(existing_path)
    existing_files = set(existing_table.column("file_id").to_pylist())

    # Discover new files
    new_files = discover_swc_files(new_swc_path)
    new_files = [f for f in new_files if f.name not in existing_files]

    if not new_files:
        logger.info("No new files to append")
        return 0

    # Process new files
    _, annotation_volume, structure_tree = setup_allen_sdk(resolution, cache_dir)
    region_lookup = build_region_lookup(structure_tree)

    new_rows: list[dict] = []
    for swc_path in new_files:
        try:
            rows = swc_to_annotated_rows(
                swc_path,
                annotation_volume,
                structure_tree,
                region_lookup,
                resolution,
            )
            new_rows.extend(rows)
        except Exception as e:
            logger.error(f"Error processing {swc_path}: {e}")

    if new_rows:
        new_table = pa.Table.from_pylist(new_rows, schema=NEURON_SCHEMA)
        combined = pa.concat_tables([existing_table, new_table])
        pq.write_table(combined, existing_path, compression="snappy")
        logger.info(f"Appended {len(new_files)} files to {existing_path}")

    return len(new_files)


def get_parquet_summary(parquet_path: Path | str) -> dict:
    """Get summary statistics for a Parquet file.

    Parameters
    ----------
    parquet_path : Path or str
        Path to the Parquet file.

    Returns
    -------
    dict
        Summary with keys: n_rows, n_files, n_subjects, n_regions, regions
    """
    import duckdb

    conn = duckdb.connect()
    path_str = str(parquet_path)

    stats = {}

    # Row count
    result = conn.execute(
        f"SELECT COUNT(*) as n FROM read_parquet('{path_str}')"
    ).fetchone()
    stats["n_rows"] = result[0]

    # File count
    result = conn.execute(
        f"SELECT COUNT(DISTINCT file_id) as n FROM read_parquet('{path_str}')"
    ).fetchone()
    stats["n_files"] = result[0]

    # Subject count
    result = conn.execute(
        f"SELECT COUNT(DISTINCT subject) as n FROM read_parquet('{path_str}')"
    ).fetchone()
    stats["n_subjects"] = result[0]

    # Region count and list
    result = conn.execute(
        f"""
        SELECT region_acronym, COUNT(*) as n
        FROM read_parquet('{path_str}')
        WHERE region_acronym != ''
        GROUP BY region_acronym
        ORDER BY n DESC
        """
    ).fetchall()
    stats["n_regions"] = len(result)
    stats["regions"] = {row[0]: row[1] for row in result}

    conn.close()
    return stats
