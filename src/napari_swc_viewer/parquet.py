"""Parquet schema and SWC-to-Parquet conversion helpers."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .hemisphere import Hemisphere, detect_soma_hemisphere, flip_swc, get_atlas_midline
from .region import build_region_lookup, get_region_ids_vectorized, setup_allen_sdk
from .swc import SWCData, parse_swc

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


@dataclass
class BatchParquetConversionSummary:
    """Summary statistics for a batch SWC-to-Parquet conversion."""

    discovered_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    flipped_files: int = 0
    already_target_files: int = 0
    midline_files: int = 0
    rows_written: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)


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


def _require_nonempty_swc(swc_data: SWCData, source: Path | str) -> None:
    """Raise when an SWC file parses to zero valid nodes."""
    if swc_data.n_nodes == 0:
        raise ValueError(f"No valid SWC nodes found in {source}")


def swc_data_to_table(
    swc_data: SWCData,
    filename: str,
    region_ids: NDArray[np.int32] | None = None,
    region_lookup: dict[int, dict] | None = None,
) -> pa.Table:
    """Convert parsed SWC data into a plugin-compatible Parquet table."""
    n_nodes = swc_data.n_nodes
    metadata = parse_filename_metadata(filename)

    if region_ids is None:
        region_id_values = np.zeros(n_nodes, dtype=np.int32)
        region_names = [""] * n_nodes
        region_acronyms = [""] * n_nodes
    else:
        region_id_values = np.asarray(region_ids, dtype=np.int32)
        if region_id_values.shape != (n_nodes,):
            raise ValueError("region_ids must have one entry per SWC node")

        lookup = region_lookup or {}
        region_names = [
            lookup.get(int(region_id), {}).get("name", "")
            for region_id in region_id_values.tolist()
        ]
        region_acronyms = [
            lookup.get(int(region_id), {}).get("acronym", "")
            for region_id in region_id_values.tolist()
        ]

    columns = {
        "file_id": [filename] * n_nodes,
        "node_id": swc_data.ids,
        "type": swc_data.types,
        "x": swc_data.coords[:, 0],
        "y": swc_data.coords[:, 1],
        "z": swc_data.coords[:, 2],
        "radius": swc_data.radii,
        "parent_id": swc_data.parents,
        "region_id": region_id_values,
        "region_name": region_names,
        "region_acronym": region_acronyms,
        "subject": [metadata["subject"]] * n_nodes,
        "neuron_id": [metadata["neuron_id"]] * n_nodes,
    }

    return pa.Table.from_pydict(columns, schema=NEURON_SCHEMA)


def swc_to_annotated_table(
    swc_path: Path,
    annotation_volume: NDArray[np.int32],
    region_lookup: dict[int, dict],
    resolution: int = 25,
) -> pa.Table:
    """Convert a single SWC file to an annotated pyarrow table."""
    swc_data = parse_swc(swc_path)
    _require_nonempty_swc(swc_data, swc_path)

    region_ids = np.asarray(
        get_region_ids_vectorized(swc_data.coords, annotation_volume, resolution),
        dtype=np.int32,
    )
    return swc_data_to_table(
        swc_data,
        swc_path.name,
        region_ids=region_ids,
        region_lookup=region_lookup,
    )


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
    _ = structure_tree
    return swc_to_annotated_table(
        swc_path,
        annotation_volume,
        region_lookup,
        resolution=resolution,
    ).to_pylist()


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


def _resolve_swc_files(
    input_source: Path | str | Iterable[Path | str],
    recursive: bool = True,
) -> list[Path]:
    """Resolve either a path-like input or explicit SWC paths to file paths."""
    if isinstance(input_source, (str, Path)):
        return discover_swc_files(Path(input_source), recursive=recursive)

    resolved: list[Path] = []
    seen: set[Path] = set()
    for item in input_source:
        path = Path(item)
        if path.suffix.lower() != ".swc":
            continue
        normalized = path.resolve(strict=False)
        if normalized in seen:
            continue
        seen.add(normalized)
        resolved.append(path)
    return sorted(resolved)


def _normalize_target_hemisphere(
    hemisphere: Hemisphere | str | None,
) -> Hemisphere | None:
    """Normalize an optional target hemisphere value."""
    if hemisphere is None:
        return None

    if isinstance(hemisphere, Hemisphere):
        target = hemisphere
    else:
        target = Hemisphere(str(hemisphere).lower())

    if target == Hemisphere.MIDLINE:
        raise ValueError("Target hemisphere must be 'left' or 'right'")

    return target


def _write_table_batch(
    writer: pq.ParquetWriter | None,
    output_path: Path,
    tables: list[pa.Table],
) -> pq.ParquetWriter | None:
    """Write a batch of tables to an output Parquet file."""
    if not tables:
        return writer

    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    if writer is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(
            output_path,
            NEURON_SCHEMA,
            compression="snappy",
        )
    writer.write_table(table)
    return writer


def batch_convert_swc_to_parquet(
    input_path: Path | str | Iterable[Path | str],
    output_path: Path | str,
    *,
    recursive: bool = True,
    hemisphere: Hemisphere | str | None = None,
    atlas_name: str = "allen_mouse_10um",
    coord_axis: int = 2,
    midline: float | None = None,
    annotate_regions: bool = False,
    resolution: int = 25,
    cache_dir: Path | str | None = None,
    batch_size: int = 100,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> BatchParquetConversionSummary:
    """Convert SWC files into one Parquet file with optional alignment."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    output_path = Path(output_path)
    summary = BatchParquetConversionSummary()
    swc_files = _resolve_swc_files(input_path, recursive=recursive)
    summary.discovered_files = len(swc_files)
    total_files = len(swc_files)

    target_hemisphere = _normalize_target_hemisphere(hemisphere)

    atlas = None
    if target_hemisphere is not None and midline is None:
        from brainglobe_atlasapi import BrainGlobeAtlas

        atlas = BrainGlobeAtlas(atlas_name)
        midline = get_atlas_midline(atlas, coord_axis)

    annotation_volume = None
    region_lookup: dict[int, dict] | None = None
    if annotate_regions:
        _, annotation_volume, structure_tree = setup_allen_sdk(resolution, cache_dir)
        region_lookup = build_region_lookup(structure_tree)

    writer: pq.ParquetWriter | None = None
    buffered_tables: list[pa.Table] = []

    try:
        for swc_path in swc_files:
            try:
                if progress_callback is not None:
                    progress_callback(
                        f"Processing {swc_path.name}...",
                        summary.processed_files + summary.failed_files,
                        total_files,
                    )
                swc_data = parse_swc(swc_path)
                _require_nonempty_swc(swc_data, swc_path)

                if target_hemisphere is not None:
                    detected = detect_soma_hemisphere(
                        swc_data,
                        atlas=atlas,
                        atlas_name=atlas_name,
                        midline=midline,
                        coord_axis=coord_axis,
                        validate=False,
                    )

                    if detected == Hemisphere.MIDLINE:
                        summary.midline_files += 1
                    elif detected == target_hemisphere:
                        summary.already_target_files += 1
                    else:
                        swc_data = flip_swc(
                            swc_data,
                            atlas=atlas,
                            atlas_name=atlas_name,
                            midline=midline,
                            coord_axis=coord_axis,
                        )
                        summary.flipped_files += 1

                region_ids = None
                if annotate_regions:
                    assert annotation_volume is not None
                    region_ids = np.asarray(
                        get_region_ids_vectorized(
                            swc_data.coords,
                            annotation_volume,
                            resolution,
                        ),
                        dtype=np.int32,
                    )

                table = swc_data_to_table(
                    swc_data,
                    swc_path.name,
                    region_ids=region_ids,
                    region_lookup=region_lookup,
                )
                buffered_tables.append(table)
                summary.processed_files += 1
                summary.rows_written += table.num_rows

                if len(buffered_tables) >= batch_size:
                    writer = _write_table_batch(writer, output_path, buffered_tables)
                    buffered_tables = []

            except Exception as exc:
                summary.failed_files += 1
                summary.failures.append((str(swc_path), str(exc)))
                logger.error("Error processing %s: %s", swc_path, exc)

        writer = _write_table_batch(writer, output_path, buffered_tables)
    finally:
        if writer is not None:
            writer.close()

    return summary


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
        logger.warning("No SWC files found in %s", input_path)
        return 0

    logger.info("Found %d SWC files to process", len(swc_files))

    # Set up Allen SDK (for serial processing or building lookup)
    _, annotation_volume, structure_tree = setup_allen_sdk(resolution, cache_dir)
    region_lookup = build_region_lookup(structure_tree)

    all_rows: list[dict] = []
    processed = 0

    if n_workers > 1:
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
                        logger.info(
                            "Processed %d/%d files",
                            processed,
                            len(swc_files),
                        )

                except Exception as exc:
                    logger.error("Error processing %s: %s", swc_path, exc)
    else:
        for swc_path in swc_files:
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
                    logger.info("Processed %d/%d files", processed, len(swc_files))

            except Exception as exc:
                logger.error("Error processing %s: %s", swc_path, exc)

    if all_rows:
        table = pa.Table.from_pylist(all_rows, schema=NEURON_SCHEMA)
        pq.write_table(table, output_path, compression="snappy")
        logger.info(
            "Wrote %d rows from %d files to %s",
            len(all_rows),
            processed,
            output_path,
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

    existing_table = pq.read_table(existing_path)
    existing_files = set(existing_table.column("file_id").to_pylist())

    new_files = discover_swc_files(new_swc_path)
    new_files = [f for f in new_files if f.name not in existing_files]

    if not new_files:
        logger.info("No new files to append")
        return 0

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
        except Exception as exc:
            logger.error("Error processing %s: %s", swc_path, exc)

    if new_rows:
        new_table = pa.Table.from_pylist(new_rows, schema=NEURON_SCHEMA)
        combined = pa.concat_tables([existing_table, new_table])
        pq.write_table(combined, existing_path, compression="snappy")
        logger.info("Appended %d files to %s", len(new_files), existing_path)

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

    result = conn.execute(
        f"SELECT COUNT(*) as n FROM read_parquet('{path_str}')"
    ).fetchone()
    stats["n_rows"] = result[0]

    result = conn.execute(
        f"SELECT COUNT(DISTINCT file_id) as n FROM read_parquet('{path_str}')"
    ).fetchone()
    stats["n_files"] = result[0]

    result = conn.execute(
        f"SELECT COUNT(DISTINCT subject) as n FROM read_parquet('{path_str}')"
    ).fetchone()
    stats["n_subjects"] = result[0]

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
