"""Parquet augmentation helpers for flatmap/depth coordinates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .flatmap_loader import load_flatmap_volume_set
from .flatmap_projection import (
    COORDINATE_MODE_MICRONS,
    REQUIRED_NODE_COLUMNS,
    project_neuron_nodes_to_flatmap,
)

FLATMAP_PARQUET_METADATA_KEY = b"napari_swc_viewer.flatmap_projection_json"
FLATMAP_PARQUET_FORMAT_VERSION = 1
DEFAULT_CCFV3_MIRROR_MIDLINE_UM = 5695.0
DEFAULT_FLATMAP_PARQUET_BATCH_SIZE = 250_000

FLATMAP_AUGMENTED_COLUMNS = (
    "x_flat",
    "y_flat",
    "depth_um",
    "flatmap_valid",
    "depth_valid",
    "flatmap_projection_valid",
    "flatmap_invalid_code",
    "flatmap_lookup_mode",
)

FLATMAP_INVALID_CODE_VALID = 0
FLATMAP_INVALID_CODE_MISSING_INPUT = 1
FLATMAP_INVALID_CODE_OUT_OF_BOUNDS = 2
FLATMAP_INVALID_CODE_INVALID_FLATMAP = 3
FLATMAP_INVALID_CODE_INVALID_DEPTH = 4

_INVALID_REASON_CODES = {
    "": FLATMAP_INVALID_CODE_VALID,
    "missing_input": FLATMAP_INVALID_CODE_MISSING_INPUT,
    "out_of_bounds": FLATMAP_INVALID_CODE_OUT_OF_BOUNDS,
    "invalid_flatmap": FLATMAP_INVALID_CODE_INVALID_FLATMAP,
    "invalid_depth": FLATMAP_INVALID_CODE_INVALID_DEPTH,
}

_ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class FlatmapParquetAugmentationSummary:
    """Summary of one flatmap Parquet augmentation run."""

    source_parquet: Path
    output_parquet: Path
    flatmap_path: Path
    depth_path: Path
    rows: int
    direct_rows: int
    mirrored_rows: int
    unmapped_rows: int


def _source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _require_source_columns(schema: pa.Schema) -> None:
    missing = [name for name in REQUIRED_NODE_COLUMNS if name not in schema.names]
    if missing:
        raise ValueError(
            "Neuron Parquet is missing required flatmap projection column(s): "
            f"{missing}"
        )


def _normalise_file_ids(file_ids: list[object] | tuple[object, ...] | None) -> list[object] | None:
    if file_ids is None:
        return None

    out: list[object] = []
    seen: set[object] = set()
    for file_id in file_ids:
        key = file_id
        try:
            already_seen = key in seen
        except TypeError:
            key = str(file_id)
            already_seen = key in seen
        if already_seen:
            continue
        seen.add(key)
        out.append(file_id)

    if not out:
        raise ValueError("file_ids must contain at least one value when provided.")
    return out


def _file_id_filter_expression(
    schema: pa.Schema,
    file_ids: list[object],
) -> ds.Expression:
    file_id_type = schema.field("file_id").type
    try:
        values = pa.array(file_ids, type=file_id_type)
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
        values = pa.array([str(file_id) for file_id in file_ids], type=file_id_type)
    return ds.field("file_id").isin(values)


def _mirror_chunk_coordinates(
    nodes: pd.DataFrame,
    *,
    mirror_coord_axis: int,
    mirror_midline: float,
) -> pd.DataFrame:
    mirrored = nodes.copy()
    coord_columns = ["x", "y", "z"]
    coord_column = coord_columns[mirror_coord_axis]
    values = pd.to_numeric(mirrored[coord_column], errors="coerce").to_numpy(
        dtype=float
    )
    mirrored.loc[:, coord_column] = (2.0 * float(mirror_midline)) - values
    return mirrored


def _project_chunk_with_mirror(
    nodes: pd.DataFrame,
    flatmap: np.ndarray,
    depth: np.ndarray,
    *,
    flatmap_style: str,
    coordinate_mode: str,
    invalid_zero_sentinel: bool,
    invalid_negative_one_sentinel: bool,
    mirror_fallback: bool,
    mirror_coord_axis: int,
    mirror_midline: float,
    space_directions: np.ndarray | None,
    space_origin: np.ndarray | None,
) -> pd.DataFrame:
    direct = project_neuron_nodes_to_flatmap(
        nodes,
        flatmap,
        depth,
        flatmap_style=flatmap_style,
        coordinate_mode=coordinate_mode,
        invalid_zero_sentinel=invalid_zero_sentinel,
        invalid_negative_one_sentinel=invalid_negative_one_sentinel,
        space_directions=space_directions,
        space_origin=space_origin,
    )

    selected = direct.reset_index(drop=True)
    lookup_mode = np.full(len(selected), "unmapped", dtype=object)
    direct_valid = selected["valid"].to_numpy(dtype=bool)
    lookup_mode[direct_valid] = "direct"

    if mirror_fallback and (~direct_valid).any():
        retry_positions = np.flatnonzero(~direct_valid)
        retry_nodes = nodes.iloc[retry_positions].reset_index(drop=True)
        mirrored_nodes = _mirror_chunk_coordinates(
            retry_nodes,
            mirror_coord_axis=mirror_coord_axis,
            mirror_midline=mirror_midline,
        )
        mirrored = project_neuron_nodes_to_flatmap(
            mirrored_nodes,
            flatmap,
            depth,
            flatmap_style=flatmap_style,
            coordinate_mode=coordinate_mode,
            invalid_zero_sentinel=invalid_zero_sentinel,
            invalid_negative_one_sentinel=invalid_negative_one_sentinel,
            space_directions=space_directions,
            space_origin=space_origin,
        ).reset_index(drop=True)

        projection_columns = (
            "x_flat",
            "y_flat",
            "depth_um",
            "flatmap_valid",
            "depth_valid",
            "valid",
            "invalid_reason",
        )
        selected.loc[retry_positions, projection_columns] = mirrored.loc[
            :,
            projection_columns,
        ].to_numpy()
        mirrored_valid = mirrored["valid"].to_numpy(dtype=bool)
        lookup_mode[retry_positions[mirrored_valid]] = "mirrored"

    selected.loc[:, "flatmap_lookup_mode"] = lookup_mode
    return selected


def _invalid_codes(projected: pd.DataFrame) -> np.ndarray:
    valid = projected["valid"].to_numpy(dtype=bool)
    reasons = projected["invalid_reason"].fillna("").astype(str).to_numpy()
    codes = np.asarray(
        [
            FLATMAP_INVALID_CODE_VALID
            if is_valid
            else _INVALID_REASON_CODES.get(reason, FLATMAP_INVALID_CODE_OUT_OF_BOUNDS)
            for is_valid, reason in zip(valid, reasons)
        ],
        dtype=np.int8,
    )
    return codes


def _augmentation_arrays(projected: pd.DataFrame) -> list[tuple[str, pa.Array]]:
    return [
        (
            "x_flat",
            pa.array(projected["x_flat"].to_numpy(dtype=np.float32), type=pa.float32()),
        ),
        (
            "y_flat",
            pa.array(projected["y_flat"].to_numpy(dtype=np.float32), type=pa.float32()),
        ),
        (
            "depth_um",
            pa.array(
                projected["depth_um"].to_numpy(dtype=np.float32),
                type=pa.float32(),
            ),
        ),
        (
            "flatmap_valid",
            pa.array(projected["flatmap_valid"].to_numpy(dtype=bool), type=pa.bool_()),
        ),
        (
            "depth_valid",
            pa.array(projected["depth_valid"].to_numpy(dtype=bool), type=pa.bool_()),
        ),
        (
            "flatmap_projection_valid",
            pa.array(projected["valid"].to_numpy(dtype=bool), type=pa.bool_()),
        ),
        (
            "flatmap_invalid_code",
            pa.array(_invalid_codes(projected), type=pa.int8()),
        ),
        (
            "flatmap_lookup_mode",
            pa.array(projected["flatmap_lookup_mode"].astype(str), type=pa.string()),
        ),
    ]


def _append_augmentation_columns(
    source_table: pa.Table,
    projected: pd.DataFrame,
) -> pa.Table:
    existing_augmented = [
        name for name in FLATMAP_AUGMENTED_COLUMNS if name in source_table.column_names
    ]
    out = source_table.drop(existing_augmented) if existing_augmented else source_table
    for name, array in _augmentation_arrays(projected):
        out = out.append_column(name, array)
    return out


def _schema_metadata(
    source_metadata: dict[bytes, bytes],
    *,
    source_parquet: Path,
    flatmap_path: Path,
    depth_path: Path,
    flatmap_shape: tuple[int, ...],
    depth_shape: tuple[int, ...],
    flatmap_style: str,
    coordinate_mode: str,
    invalid_zero_sentinel: bool,
    invalid_negative_one_sentinel: bool,
    mirror_fallback: bool,
    mirror_coord_axis: int,
    mirror_midline: float,
    file_ids_filter_count: int | None,
    space_directions: np.ndarray | None,
    space_origin: np.ndarray | None,
) -> dict[bytes, bytes]:
    metadata = dict(source_metadata)
    payload = {
        "format": "napari_swc_viewer.flatmap_projection",
        "version": FLATMAP_PARQUET_FORMAT_VERSION,
        "source_parquet": _source_signature(source_parquet),
        "flatmap_nrrd": _source_signature(flatmap_path),
        "depth_nrrd": _source_signature(depth_path),
        "flatmap_shape": flatmap_shape,
        "depth_shape": depth_shape,
        "flatmap_style": flatmap_style,
        "coordinate_mode": coordinate_mode,
        "invalid_zero_sentinel": bool(invalid_zero_sentinel),
        "invalid_negative_one_sentinel": bool(invalid_negative_one_sentinel),
        "mirror_fallback": bool(mirror_fallback),
        "mirror_coord_axis": int(mirror_coord_axis),
        "mirror_midline": float(mirror_midline),
        "file_ids_filter_count": file_ids_filter_count,
        "space_directions": space_directions,
        "space_origin": space_origin,
        "columns": list(FLATMAP_AUGMENTED_COLUMNS),
        "invalid_codes": {
            "valid": FLATMAP_INVALID_CODE_VALID,
            "missing_input": FLATMAP_INVALID_CODE_MISSING_INPUT,
            "out_of_bounds": FLATMAP_INVALID_CODE_OUT_OF_BOUNDS,
            "invalid_flatmap": FLATMAP_INVALID_CODE_INVALID_FLATMAP,
            "invalid_depth": FLATMAP_INVALID_CODE_INVALID_DEPTH,
        },
    }
    metadata[FLATMAP_PARQUET_METADATA_KEY] = json.dumps(
        _json_safe(payload),
        sort_keys=True,
    ).encode("utf-8")
    return metadata


def augment_neuron_parquet_with_flatmap(
    source_parquet: str | Path,
    output_parquet: str | Path,
    flatmap_path: str | Path,
    depth_path: str | Path,
    *,
    file_ids: list[object] | tuple[object, ...] | None = None,
    coordinate_mode: str = COORDINATE_MODE_MICRONS,
    flatmap_style: str = "",
    mirror_fallback: bool = True,
    mirror_coord_axis: int = 2,
    mirror_midline: float | None = None,
    invalid_zero_sentinel: bool = False,
    invalid_negative_one_sentinel: bool = True,
    batch_size: int = DEFAULT_FLATMAP_PARQUET_BATCH_SIZE,
    compression: str = "zstd",
    progress_callback: _ProgressCallback | None = None,
) -> FlatmapParquetAugmentationSummary:
    """Write a neuron Parquet with added flatmap/depth projection columns."""
    source_path = Path(source_parquet)
    output_path = Path(output_parquet)
    flatmap_source = Path(flatmap_path)
    depth_source = Path(depth_path)
    selected_file_ids = _normalise_file_ids(file_ids)

    if mirror_coord_axis not in (0, 1, 2):
        raise ValueError("mirror_coord_axis must be 0, 1, or 2.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    resolved_midline = (
        DEFAULT_CCFV3_MIRROR_MIDLINE_UM
        if mirror_midline is None
        else float(mirror_midline)
    )
    style = flatmap_style or flatmap_source.name

    volume_set = load_flatmap_volume_set(flatmap_source, depth_source)
    source_file = pq.ParquetFile(source_path)
    _require_source_columns(source_file.schema_arrow)
    total_rows = int(source_file.metadata.num_rows)
    if selected_file_ids is None:
        batches = source_file.iter_batches(batch_size=batch_size)
    else:
        source_dataset = ds.dataset(source_path, format="parquet")
        file_id_filter = _file_id_filter_expression(
            source_file.schema_arrow,
            selected_file_ids,
        )
        batches = source_dataset.scanner(
            filter=file_id_filter,
            batch_size=batch_size,
        ).to_batches()

    write_target = output_path
    replace_output = False
    if source_path.resolve() == output_path.resolve():
        write_target = output_path.with_name(f"{output_path.name}.tmp")
        replace_output = True
        if write_target.exists():
            write_target.unlink()
    write_target.parent.mkdir(parents=True, exist_ok=True)

    source_metadata = dict(source_file.schema_arrow.metadata or {})
    output_metadata = _schema_metadata(
        source_metadata,
        source_parquet=source_path,
        flatmap_path=flatmap_source,
        depth_path=depth_source,
        flatmap_shape=tuple(int(size) for size in volume_set.flatmap.shape),
        depth_shape=tuple(int(size) for size in volume_set.depth.shape),
        flatmap_style=style,
        coordinate_mode=coordinate_mode,
        invalid_zero_sentinel=invalid_zero_sentinel,
        invalid_negative_one_sentinel=invalid_negative_one_sentinel,
        mirror_fallback=mirror_fallback,
        mirror_coord_axis=mirror_coord_axis,
        mirror_midline=resolved_midline,
        file_ids_filter_count=(
            None if selected_file_ids is None else len(selected_file_ids)
        ),
        space_directions=volume_set.space_directions,
        space_origin=volume_set.space_origin,
    )

    rows_written = 0
    direct_rows = 0
    mirrored_rows = 0
    unmapped_rows = 0
    writer: pq.ParquetWriter | None = None

    try:
        for batch in batches:
            source_table = pa.Table.from_batches([batch])
            nodes = source_table.to_pandas()
            projected = _project_chunk_with_mirror(
                nodes,
                volume_set.flatmap,
                volume_set.depth,
                flatmap_style=style,
                coordinate_mode=coordinate_mode,
                invalid_zero_sentinel=invalid_zero_sentinel,
                invalid_negative_one_sentinel=invalid_negative_one_sentinel,
                mirror_fallback=mirror_fallback,
                mirror_coord_axis=mirror_coord_axis,
                mirror_midline=resolved_midline,
                space_directions=volume_set.space_directions,
                space_origin=volume_set.space_origin,
            )
            augmented = _append_augmentation_columns(source_table, projected)

            if writer is None:
                output_schema = augmented.schema.with_metadata(output_metadata)
                writer = pq.ParquetWriter(
                    write_target,
                    output_schema,
                    compression=compression,
                )
            writer.write_table(augmented.cast(writer.schema))

            modes = projected["flatmap_lookup_mode"].astype(str)
            direct_rows += int((modes == "direct").sum())
            mirrored_rows += int((modes == "mirrored").sum())
            unmapped_rows += int((modes == "unmapped").sum())
            rows_written += int(source_table.num_rows)
            if progress_callback is not None:
                progress_callback(
                    "Adding flatmap/depth columns...",
                    rows_written,
                    total_rows,
                )
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        empty_schema = pa.schema(
            [
                field
                for field in source_file.schema_arrow
                if field.name not in FLATMAP_AUGMENTED_COLUMNS
            ]
        )
        for field in (
            pa.field("x_flat", pa.float32()),
            pa.field("y_flat", pa.float32()),
            pa.field("depth_um", pa.float32()),
            pa.field("flatmap_valid", pa.bool_()),
            pa.field("depth_valid", pa.bool_()),
            pa.field("flatmap_projection_valid", pa.bool_()),
            pa.field("flatmap_invalid_code", pa.int8()),
            pa.field("flatmap_lookup_mode", pa.string()),
        ):
            if field.name not in empty_schema.names:
                empty_schema = empty_schema.append(field)
        empty_schema = empty_schema.with_metadata(output_metadata)
        with pq.ParquetWriter(
            write_target,
            empty_schema,
            compression=compression,
        ):
            pass

    if replace_output:
        write_target.replace(output_path)

    return FlatmapParquetAugmentationSummary(
        source_parquet=source_path,
        output_parquet=output_path,
        flatmap_path=flatmap_source,
        depth_path=depth_source,
        rows=rows_written,
        direct_rows=direct_rows,
        mirrored_rows=mirrored_rows,
        unmapped_rows=unmapped_rows,
    )
