"""Parquet schema and SWC-to-Parquet conversion helpers."""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
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


@dataclass(frozen=True)
class _BatchWorkerConfig:
    """Shared worker configuration for chunked conversion."""

    target_hemisphere: Hemisphere | None
    atlas_name: str
    coord_axis: int
    midline: float | None
    annotate_regions: bool
    resolution: int


@dataclass
class _FileConversionResult:
    """Result for one successfully converted SWC file."""

    table: pa.Table
    flipped_file: bool = False
    already_target_file: bool = False
    midline_file: bool = False


@dataclass
class _ChunkConversionResult:
    """Summary returned from a worker chunk."""

    chunk_index: int
    processed_files: int = 0
    failed_files: int = 0
    flipped_files: int = 0
    already_target_files: int = 0
    midline_files: int = 0
    rows_written: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
    shard_path: str | None = None


_WORKER_CONFIG: _BatchWorkerConfig | None = None
_WORKER_ANNOTATION_VOLUME: NDArray[np.int32] | None = None
_WORKER_REGION_LOOKUP: dict[int, dict] | None = None


def _clear_batch_worker_state() -> None:
    """Release any process-local worker cache that can pin scratch files on Windows."""
    global _WORKER_CONFIG, _WORKER_ANNOTATION_VOLUME, _WORKER_REGION_LOOKUP

    annotation_volume = _WORKER_ANNOTATION_VOLUME
    _WORKER_CONFIG = None
    _WORKER_REGION_LOOKUP = None
    _WORKER_ANNOTATION_VOLUME = None

    if isinstance(annotation_volume, np.memmap):
        mmap_handle = getattr(annotation_volume, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()


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
        unique_region_ids, inverse = np.unique(region_id_values, return_inverse=True)
        region_names_lookup = np.asarray(
            [
                lookup.get(int(region_id), {}).get("name", "")
                for region_id in unique_region_ids.tolist()
            ],
            dtype=object,
        )
        region_acronyms_lookup = np.asarray(
            [
                lookup.get(int(region_id), {}).get("acronym", "")
                for region_id in unique_region_ids.tolist()
            ],
            dtype=object,
        )
        region_names = region_names_lookup[inverse].tolist()
        region_acronyms = region_acronyms_lookup[inverse].tolist()

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


def _resolve_worker_count(
    n_workers: int | None,
    total_files: int,
    batch_size: int,
) -> int:
    """Resolve worker count from explicit or automatic configuration."""
    if n_workers is None:
        if total_files <= batch_size:
            return 1
        return min(8, os.cpu_count() or 1)

    if n_workers < 1:
        raise ValueError("n_workers must be at least 1")

    return n_workers


def _open_parquet_writer(output_path: Path) -> pq.ParquetWriter:
    """Open a Parquet writer using the canonical neuron schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return pq.ParquetWriter(
        output_path,
        NEURON_SCHEMA,
        compression="snappy",
    )


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
        writer = _open_parquet_writer(output_path)
    writer.write_table(table)
    return writer


def _convert_swc_file_to_table(
    swc_path: Path,
    *,
    target_hemisphere: Hemisphere | None,
    atlas_name: str,
    coord_axis: int,
    midline: float | None,
    annotate_regions: bool,
    resolution: int,
    annotation_volume: NDArray[np.int32] | None,
    region_lookup: dict[int, dict] | None,
) -> _FileConversionResult:
    """Parse, align, annotate, and materialize a single SWC file."""
    swc_data = parse_swc(swc_path)
    _require_nonempty_swc(swc_data, swc_path)

    flipped_file = False
    already_target_file = False
    midline_file = False

    if target_hemisphere is not None:
        detected = detect_soma_hemisphere(
            swc_data,
            atlas_name=atlas_name,
            midline=midline,
            coord_axis=coord_axis,
            validate=False,
        )

        if detected == Hemisphere.MIDLINE:
            midline_file = True
        elif detected == target_hemisphere:
            already_target_file = True
        else:
            swc_data = flip_swc(
                swc_data,
                atlas_name=atlas_name,
                midline=midline,
                coord_axis=coord_axis,
            )
            flipped_file = True

    region_ids = None
    if annotate_regions:
        if annotation_volume is None:
            raise ValueError("annotation_volume is required when annotate_regions=True")
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
    return _FileConversionResult(
        table=table,
        flipped_file=flipped_file,
        already_target_file=already_target_file,
        midline_file=midline_file,
    )


def _add_file_result_to_summary(
    summary: BatchParquetConversionSummary,
    result: _FileConversionResult,
) -> None:
    """Update a batch summary from one successful file result."""
    summary.processed_files += 1
    summary.rows_written += result.table.num_rows
    if result.flipped_file:
        summary.flipped_files += 1
    if result.already_target_file:
        summary.already_target_files += 1
    if result.midline_file:
        summary.midline_files += 1


def _init_batch_worker(
    config: _BatchWorkerConfig,
    region_lookup: dict[int, dict] | None,
    annotation_volume_path: str | None,
) -> None:
    """Initialize process-local shared state for chunk workers."""
    global _WORKER_CONFIG, _WORKER_ANNOTATION_VOLUME, _WORKER_REGION_LOOKUP

    _clear_batch_worker_state()
    _WORKER_CONFIG = config
    _WORKER_REGION_LOOKUP = region_lookup

    if annotation_volume_path is not None:
        _WORKER_ANNOTATION_VOLUME = np.load(annotation_volume_path, mmap_mode="r")


def _process_swc_chunk(args: tuple[int, tuple[str, ...], str]) -> _ChunkConversionResult:
    """Process one chunk of SWC files and write one shard file."""
    if _WORKER_CONFIG is None:
        raise RuntimeError("SWC conversion worker was not initialized")

    chunk_index, swc_paths, shard_dir = args
    result = _ChunkConversionResult(chunk_index=chunk_index)
    shard_path = Path(shard_dir) / f"chunk_{chunk_index:06d}.parquet"
    writer: pq.ParquetWriter | None = None

    try:
        for swc_path_str in swc_paths:
            swc_path = Path(swc_path_str)
            try:
                file_result = _convert_swc_file_to_table(
                    swc_path,
                    target_hemisphere=_WORKER_CONFIG.target_hemisphere,
                    atlas_name=_WORKER_CONFIG.atlas_name,
                    coord_axis=_WORKER_CONFIG.coord_axis,
                    midline=_WORKER_CONFIG.midline,
                    annotate_regions=_WORKER_CONFIG.annotate_regions,
                    resolution=_WORKER_CONFIG.resolution,
                    annotation_volume=_WORKER_ANNOTATION_VOLUME,
                    region_lookup=_WORKER_REGION_LOOKUP,
                )
                writer = _write_table_batch(writer, shard_path, [file_result.table])
                result.processed_files += 1
                result.rows_written += file_result.table.num_rows
                if file_result.flipped_file:
                    result.flipped_files += 1
                if file_result.already_target_file:
                    result.already_target_files += 1
                if file_result.midline_file:
                    result.midline_files += 1
            except Exception as exc:
                result.failed_files += 1
                result.failures.append((str(swc_path), str(exc)))

        if writer is not None:
            result.shard_path = str(shard_path)
    finally:
        if writer is not None:
            writer.close()

    return result


def _merge_parquet_shards(
    shard_paths: list[Path],
    output_path: Path,
    *,
    progress_callback: Callable[[str, int, int], None] | None = None,
    total_files: int | None = None,
) -> None:
    """Merge shard parquet files into one deterministic output file."""
    writer: pq.ParquetWriter | None = None

    try:
        total_shards = len(shard_paths)
        for shard_index, shard_path in enumerate(shard_paths, start=1):
            if progress_callback is not None:
                progress_callback(
                    f"Finalizing Parquet ({shard_index}/{total_shards} shards)...",
                    total_files or 0,
                    total_files or 0,
                )
            if writer is None:
                writer = _open_parquet_writer(output_path)
            with pq.ParquetFile(shard_path) as parquet_file:
                for batch in parquet_file.iter_batches():
                    writer.write_batch(batch)
    finally:
        if writer is not None:
            writer.close()


def _run_serial_batch_conversion(
    swc_files: list[Path],
    output_path: Path,
    *,
    target_hemisphere: Hemisphere | None,
    atlas_name: str,
    coord_axis: int,
    midline: float | None,
    annotate_regions: bool,
    resolution: int,
    annotation_volume: NDArray[np.int32] | None,
    region_lookup: dict[int, dict] | None,
    batch_size: int,
    progress_callback: Callable[[str, int, int], None] | None,
) -> BatchParquetConversionSummary:
    """Convert SWCs serially while reusing the shared per-file logic."""
    summary = BatchParquetConversionSummary(discovered_files=len(swc_files))
    writer: pq.ParquetWriter | None = None
    buffered_tables: list[pa.Table] = []
    total_files = len(swc_files)

    try:
        for swc_path in swc_files:
            try:
                if progress_callback is not None:
                    progress_callback(
                        f"Processing {swc_path.name}...",
                        summary.processed_files + summary.failed_files,
                        total_files,
                    )

                file_result = _convert_swc_file_to_table(
                    swc_path,
                    target_hemisphere=target_hemisphere,
                    atlas_name=atlas_name,
                    coord_axis=coord_axis,
                    midline=midline,
                    annotate_regions=annotate_regions,
                    resolution=resolution,
                    annotation_volume=annotation_volume,
                    region_lookup=region_lookup,
                )
                _add_file_result_to_summary(summary, file_result)
                buffered_tables.append(file_result.table)

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


def _run_parallel_batch_conversion(
    swc_files: list[Path],
    output_path: Path,
    *,
    worker_count: int,
    target_hemisphere: Hemisphere | None,
    atlas_name: str,
    coord_axis: int,
    midline: float | None,
    annotate_regions: bool,
    resolution: int,
    annotation_volume: NDArray[np.int32] | None,
    region_lookup: dict[int, dict] | None,
    batch_size: int,
    temp_dir: Path | str | None,
    progress_callback: Callable[[str, int, int], None] | None,
) -> BatchParquetConversionSummary:
    """Convert SWCs in parallel using chunk-local shard files."""
    summary = BatchParquetConversionSummary(discovered_files=len(swc_files))
    total_files = len(swc_files)

    if total_files == 0:
        return summary

    temp_root = Path(temp_dir) if temp_dir is not None else output_path.parent
    temp_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="swc_to_parquet_", dir=temp_root))
    shard_dir = work_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    staged_output_path = work_dir / "merged_output.parquet"

    annotation_volume_path: Path | None = None
    if annotate_regions:
        if annotation_volume is None:
            raise ValueError("annotation_volume is required when annotate_regions=True")
        annotation_volume_path = work_dir / "annotation_volume.npy"
        np.save(annotation_volume_path, annotation_volume, allow_pickle=False)
        annotation_volume = None

    chunks = [
        tuple(str(path) for path in swc_files[start : start + batch_size])
        for start in range(0, total_files, batch_size)
    ]
    worker_count = min(worker_count, len(chunks))
    worker_config = _BatchWorkerConfig(
        target_hemisphere=target_hemisphere,
        atlas_name=atlas_name,
        coord_axis=coord_axis,
        midline=midline,
        annotate_regions=annotate_regions,
        resolution=resolution,
    )
    chunk_results: dict[int, _ChunkConversionResult] = {}
    completed_files = 0
    completed_chunks = 0

    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_batch_worker,
            initargs=(
                worker_config,
                region_lookup,
                str(annotation_volume_path) if annotation_volume_path is not None else None,
            ),
        ) as executor:
            futures = {
                executor.submit(_process_swc_chunk, (chunk_index, chunk, str(shard_dir))): chunk_index
                for chunk_index, chunk in enumerate(chunks)
            }

            pending_futures = futures.copy()
            while pending_futures:
                done_futures, _ = wait(
                    tuple(pending_futures),
                    return_when=FIRST_COMPLETED,
                )
                ready_chunks = sorted(
                    ((pending_futures.pop(future), future) for future in done_futures),
                    key=lambda item: item[0],
                )
                for chunk_index, future in ready_chunks:
                    chunk_result = future.result()
                    chunk_results[chunk_index] = chunk_result
                    completed_files += chunk_result.processed_files + chunk_result.failed_files
                    completed_chunks += 1
                    if progress_callback is not None:
                        progress_callback(
                            (
                                f"Processed {completed_files}/{total_files} files "
                                f"({completed_chunks}/{len(chunks)} chunks)..."
                            ),
                            completed_files,
                            total_files,
                        )

        shard_paths: list[Path] = []
        for chunk_index in range(len(chunks)):
            chunk_result = chunk_results[chunk_index]
            summary.processed_files += chunk_result.processed_files
            summary.failed_files += chunk_result.failed_files
            summary.flipped_files += chunk_result.flipped_files
            summary.already_target_files += chunk_result.already_target_files
            summary.midline_files += chunk_result.midline_files
            summary.rows_written += chunk_result.rows_written
            summary.failures.extend(chunk_result.failures)
            if chunk_result.shard_path is not None:
                shard_paths.append(Path(chunk_result.shard_path))

        if shard_paths:
            _merge_parquet_shards(
                shard_paths,
                staged_output_path,
                progress_callback=progress_callback,
                total_files=total_files,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            staged_output_path.replace(output_path)
    finally:
        _clear_batch_worker_state()
        shutil.rmtree(work_dir, ignore_errors=True)

    return summary


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
    batch_size: int = 25,
    n_workers: int | None = None,
    temp_dir: Path | str | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> BatchParquetConversionSummary:
    """Convert SWC files into one Parquet file with optional alignment."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    output_path = Path(output_path)
    swc_files = _resolve_swc_files(input_path, recursive=recursive)
    total_files = len(swc_files)

    target_hemisphere = _normalize_target_hemisphere(hemisphere)

    if target_hemisphere is not None and midline is None:
        from brainglobe_atlasapi import BrainGlobeAtlas

        atlas = BrainGlobeAtlas(atlas_name)
        midline = get_atlas_midline(atlas, coord_axis)

    annotation_volume = None
    region_lookup: dict[int, dict] | None = None
    if annotate_regions:
        _, annotation_volume, structure_tree = setup_allen_sdk(resolution, cache_dir)
        region_lookup = build_region_lookup(structure_tree)

    worker_count = _resolve_worker_count(n_workers, total_files, batch_size)
    if worker_count <= 1 or total_files == 0:
        return _run_serial_batch_conversion(
            swc_files,
            output_path,
            target_hemisphere=target_hemisphere,
            atlas_name=atlas_name,
            coord_axis=coord_axis,
            midline=midline,
            annotate_regions=annotate_regions,
            resolution=resolution,
            annotation_volume=annotation_volume,
            region_lookup=region_lookup,
            batch_size=batch_size,
            progress_callback=progress_callback,
        )

    return _run_parallel_batch_conversion(
        swc_files,
        output_path,
        worker_count=worker_count,
        target_hemisphere=target_hemisphere,
        atlas_name=atlas_name,
        coord_axis=coord_axis,
        midline=midline,
        annotate_regions=annotate_regions,
        resolution=resolution,
        annotation_volume=annotation_volume,
        region_lookup=region_lookup,
        batch_size=batch_size,
        temp_dir=temp_dir,
        progress_callback=progress_callback,
    )


def swc_files_to_parquet(
    input_path: Path | str,
    output_path: Path | str,
    resolution: int = 25,
    cache_dir: Path | str | None = None,
    recursive: bool = True,
    n_workers: int = 1,
    batch_size: int = 25,
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
    summary = batch_convert_swc_to_parquet(
        input_path,
        output_path,
        recursive=recursive,
        annotate_regions=True,
        resolution=resolution,
        cache_dir=cache_dir,
        batch_size=batch_size,
        n_workers=n_workers,
    )
    return summary.processed_files


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
