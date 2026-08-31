"""Parquet augmentation helpers for flatmap/depth coordinates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .flatmap_loader import (
    FlatmapLookupLoadCancelledError,
    load_flatmap_volume_set,
)
from .flatmap_profiles import (
    BILATERAL_FLATMAP_STYLES,
    FlatmapGridSpec,
    FlatmapLookupSet,
    normalize_bilateral_style,
)
from .flatmap_projection import (
    COORDINATE_MODE_MICRONS,
    DEFAULT_CCFV3_MIRROR_MIDLINE_UM as _DEFAULT_CCFV3_MIRROR_MIDLINE_UM,
    FLATMAP_LOOKUP_DIRECT,
    FLATMAP_LOOKUP_MIRRORED,
    FLATMAP_LOOKUP_MIRRORED_DEPTH,
    FLATMAP_LOOKUP_UNMAPPED,
    REQUIRED_NODE_COLUMNS,
    coordinates_to_voxel_indices,
    project_neuron_nodes_to_flatmap,
    resolve_flatmap_mirror_midline,
)

FLATMAP_PARQUET_METADATA_KEY = b"napari_neuron_navigator.flatmap_projection_json"
LEGACY_FLATMAP_PARQUET_METADATA_KEY = b"napari_swc_viewer.flatmap_projection_json"
FLATMAP_PARQUET_FORMAT_VERSION = 3
LEGACY_SINGLE_FLATMAP_PARQUET_FORMAT_VERSION = 2
DEFAULT_CCFV3_MIRROR_MIDLINE_UM = _DEFAULT_CCFV3_MIRROR_MIDLINE_UM
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

FLATMAP_V3_STYLE_COLUMN_MAPPING: dict[str, dict[str, str]] = {
    "both_shaped": {
        "x": "x_flat_shaped",
        "y": "y_flat_shaped",
        "valid": "flatmap_shaped_valid",
        "projection_valid": "flatmap_shaped_projection_valid",
        "invalid_code": "flatmap_shaped_invalid_code",
        "lookup_mode": "flatmap_shaped_lookup_mode",
    },
    "both_square": {
        "x": "x_flat_square",
        "y": "y_flat_square",
        "valid": "flatmap_square_valid",
        "projection_valid": "flatmap_square_projection_valid",
        "invalid_code": "flatmap_square_invalid_code",
        "lookup_mode": "flatmap_square_lookup_mode",
    },
}
FLATMAP_V3_DEPTH_COLUMN_MAPPING = {
    "depth": "depth_um",
    "valid": "depth_valid",
    "invalid_code": "depth_invalid_code",
    "lookup_mode": "depth_lookup_mode",
}
FLATMAP_V3_AUGMENTED_COLUMNS = tuple(
    column
    for style in BILATERAL_FLATMAP_STYLES
    for column in FLATMAP_V3_STYLE_COLUMN_MAPPING[style].values()
) + tuple(FLATMAP_V3_DEPTH_COLUMN_MAPPING.values())

FLATMAP_COORDINATE_COLUMNS = ("x_flat", "y_flat")
DEPTH_COORDINATE_COLUMNS = ("depth_um",)

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
_INVALID_CODE_REASONS = {code: reason for reason, code in _INVALID_REASON_CODES.items()}

_ProgressCallback = Callable[[str, int, int], None]
_CancelCallback = Callable[[], bool]


class FlatmapParquetCancelledError(RuntimeError):
    """Raised when a cancellable whole-Parquet augmentation is stopped."""


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
    mirrored_depth_rows: int = 0
    lookup_set_id: str | None = None
    square_flatmap_path: Path | None = None
    shaped_valid_rows: int = 0
    square_valid_rows: int = 0


@dataclass(frozen=True)
class FlatmapParquetTransformInfo:
    """Flatmap/depth transform columns detected in a neuron Parquet schema."""

    path: Path
    has_flatmap: bool
    has_depth: bool
    metadata: dict[str, Any] | None
    available_styles: tuple[str, ...] = ()
    has_v3_depth: bool = False

    @property
    def has_full_transform(self) -> bool:
        """Return True when the parquet can render flatmap/depth without NRRDs."""
        if self.format_version >= FLATMAP_PARQUET_FORMAT_VERSION:
            return bool(
                set(BILATERAL_FLATMAP_STYLES).issubset(self.available_styles)
                and self.has_v3_depth
            )
        return bool(self.has_flatmap and self.has_depth)

    @property
    def format_version(self) -> int:
        """Return the stored transform format version, or zero if unavailable."""
        try:
            return int((self.metadata or {}).get("version", 0))
        except (TypeError, ValueError):
            return 0

    @property
    def lookup_set_id(self) -> str | None:
        """Return the portable v3 lookup-set identifier when present."""
        raw = (self.metadata or {}).get("lookup_set_id")
        if raw is None:
            lookup_set = (self.metadata or {}).get("lookup_set")
            if isinstance(lookup_set, dict):
                raw = lookup_set.get("lookup_set_id")
        return None if raw is None else str(raw)

    def has_style(self, style: str) -> bool:
        """Return whether this Parquet contains the requested flatmap style."""
        try:
            normalized = normalize_bilateral_style(style)
        except ValueError:
            return False
        return normalized in self.available_styles

    def column_mapping(self, style: str) -> dict[str, str] | None:
        """Return stored v3 column names for one style."""
        return flatmap_parquet_style_column_mapping(self.metadata, style)

    def grid_spec(self, style: str) -> FlatmapGridSpec | None:
        """Return canonical bounds/transform metadata for one style."""
        return flatmap_grid_spec_from_parquet_metadata(self.metadata, style)

    @property
    def lookup_set(self) -> FlatmapLookupSet | None:
        """Return the portable v3 lookup-set definition when valid."""
        return flatmap_lookup_set_from_parquet_metadata(self.metadata)

    @property
    def uses_legacy_mirror_fallback(self) -> bool:
        """Return whether stored coordinates use the version-1 mirror strategy."""
        metadata = self.metadata
        if not isinstance(metadata, dict):
            return False
        try:
            version = int(metadata.get("version", 0))
        except (TypeError, ValueError):
            return False
        return version == 1 and bool(metadata.get("mirror_fallback", False))

    @property
    def present_transform_text(self) -> str:
        """Return a compact human-readable description of present transforms."""
        return format_flatmap_parquet_transform_presence(
            has_flatmap=self.has_flatmap,
            has_depth=self.has_depth,
        )


def format_flatmap_parquet_transform_presence(
    *,
    has_flatmap: bool,
    has_depth: bool,
) -> str:
    """Return display text for detected flatmap/depth transform columns."""
    if has_flatmap and has_depth:
        return "flatmap and depth"
    if has_flatmap:
        return "flatmap"
    if has_depth:
        return "depth"
    return ""


def flatmap_invalid_code_to_reason(code: object) -> str:
    """Return the projection invalid reason represented by an augmented code."""
    try:
        normalized = int(code)
    except (TypeError, ValueError):
        return "out_of_bounds"
    return _INVALID_CODE_REASONS.get(normalized, "out_of_bounds")


def _decode_flatmap_projection_metadata(
    metadata: dict[bytes, bytes],
) -> dict[str, Any] | None:
    raw = metadata.get(FLATMAP_PARQUET_METADATA_KEY)
    if raw is None:
        raw = metadata.get(LEGACY_FLATMAP_PARQUET_METADATA_KEY)
    if raw is None:
        return None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"raw": raw.hex()}
    return payload if isinstance(payload, dict) else {"value": payload}


def decode_flatmap_parquet_metadata(
    metadata: dict[bytes, bytes] | None,
) -> dict[str, Any] | None:
    """Decode flatmap metadata from an Arrow schema metadata mapping."""
    return _decode_flatmap_projection_metadata(dict(metadata or {}))


def flatmap_parquet_style_column_mapping(
    metadata: dict[str, Any] | None,
    style: str,
) -> dict[str, str] | None:
    """Return the v3 column mapping for one bilateral style."""
    try:
        normalized = normalize_bilateral_style(style)
    except ValueError:
        return None
    mapping = (metadata or {}).get("column_mapping")
    if isinstance(mapping, dict) and isinstance(mapping.get(normalized), dict):
        return {
            str(key): str(value)
            for key, value in mapping[normalized].items()
        }
    try:
        version = int((metadata or {}).get("version", 0) or 0)
    except (TypeError, ValueError):
        version = 0
    if version >= 3:
        return dict(FLATMAP_V3_STYLE_COLUMN_MAPPING[normalized])
    return None


def flatmap_grid_spec_from_parquet_metadata(
    metadata: dict[str, Any] | None,
    style: str,
) -> FlatmapGridSpec | None:
    """Parse canonical global bounds for one style from v3 metadata."""
    try:
        normalized = normalize_bilateral_style(style)
    except ValueError:
        return None
    lookup_set = (metadata or {}).get("lookup_set")
    if not isinstance(lookup_set, dict):
        return None
    styles = lookup_set.get("styles")
    if not isinstance(styles, dict) or not isinstance(styles.get(normalized), dict):
        return None
    try:
        return FlatmapGridSpec.from_dict(styles[normalized])
    except (KeyError, TypeError, ValueError):
        return None


def flatmap_lookup_set_from_parquet_metadata(
    metadata: dict[str, Any] | None,
) -> FlatmapLookupSet | None:
    """Parse the complete v3 lookup-set definition from Parquet metadata."""
    lookup_set = (metadata or {}).get("lookup_set")
    if not isinstance(lookup_set, dict):
        return None
    try:
        return FlatmapLookupSet.from_dict(lookup_set)
    except (KeyError, TypeError, ValueError):
        return None


def read_flatmap_parquet_transform_info(
    parquet_path: str | Path,
) -> FlatmapParquetTransformInfo:
    """Inspect a neuron Parquet for reusable flatmap/depth coordinate columns."""
    path = Path(parquet_path)
    schema = pq.read_schema(path)
    names = set(schema.names)
    metadata = _decode_flatmap_projection_metadata(dict(schema.metadata or {}))
    available_styles = tuple(
        style
        for style in BILATERAL_FLATMAP_STYLES
        if set(FLATMAP_V3_STYLE_COLUMN_MAPPING[style].values()).issubset(names)
    )
    has_legacy_flatmap = set(FLATMAP_COORDINATE_COLUMNS).issubset(names)
    has_v3_depth = set(FLATMAP_V3_DEPTH_COLUMN_MAPPING.values()).issubset(names)
    return FlatmapParquetTransformInfo(
        path=path,
        has_flatmap=bool(has_legacy_flatmap or available_styles),
        has_depth=set(DEPTH_COORDINATE_COLUMNS).issubset(names),
        metadata=metadata,
        available_styles=available_styles,
        has_v3_depth=has_v3_depth,
    )


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


def _invalid_codes(projected) -> np.ndarray:
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


def _augmentation_arrays(projected) -> list[tuple[str, pa.Array]]:
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
    projected,
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
        "format": "napari_neuron_navigator.flatmap_projection",
        "version": LEGACY_SINGLE_FLATMAP_PARQUET_FORMAT_VERSION,
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
        "mirror_fallback_strategy": (
            "preserve_original_flatmap_then_mirror_depth_then_full_lookup"
        ),
        "mirror_coord_axis": int(mirror_coord_axis),
        "mirror_midline": float(mirror_midline),
        "file_ids_filter_count": file_ids_filter_count,
        "space_directions": space_directions,
        "space_origin": space_origin,
        "columns": list(FLATMAP_AUGMENTED_COLUMNS),
        "lookup_modes": [
            FLATMAP_LOOKUP_DIRECT,
            FLATMAP_LOOKUP_MIRRORED_DEPTH,
            FLATMAP_LOOKUP_MIRRORED,
            FLATMAP_LOOKUP_UNMAPPED,
        ],
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


def _v3_schema_metadata(
    source_metadata: dict[bytes, bytes],
    *,
    source_parquet: Path,
    lookup_set: FlatmapLookupSet,
    coordinate_mode: str,
    mirror_coord_axis: int,
) -> dict[bytes, bytes]:
    metadata = dict(source_metadata)
    shaped_grid = lookup_set.shaped_grid
    payload = {
        "format": "napari_neuron_navigator.flatmap_projection",
        "version": FLATMAP_PARQUET_FORMAT_VERSION,
        "lookup_set_id": lookup_set.lookup_set_id,
        "lookup_set": lookup_set.to_dict(include_paths=True),
        "source_parquet": _source_signature(source_parquet),
        "coordinate_mode": coordinate_mode,
        "coordinate_order": {
            "lookup": list(shaped_grid.lookup_coordinate_order),
            "flatmap": list(shaped_grid.flatmap_coordinate_order),
            "render": list(shaped_grid.render_coordinate_order),
        },
        "spatial_transform": {
            "lookup_resolution_um": list(shaped_grid.lookup_resolution_um),
            "space_directions": [list(row) for row in shaped_grid.space_directions],
            "space_origin": list(shaped_grid.space_origin),
            "spatial_shape": list(shaped_grid.spatial_shape),
        },
        "canonical_bounds": {
            style: {
                "x": list(lookup_set.grid_for_style(style).x_bounds),
                "y": list(lookup_set.grid_for_style(style).y_bounds),
                "depth_um": list(
                    lookup_set.grid_for_style(style).depth_bounds_um
                ),
            }
            for style in BILATERAL_FLATMAP_STYLES
        },
        "column_mapping": {
            **{
                style: dict(FLATMAP_V3_STYLE_COLUMN_MAPPING[style])
                for style in BILATERAL_FLATMAP_STYLES
            },
            "depth": dict(FLATMAP_V3_DEPTH_COLUMN_MAPPING),
        },
        "shared_depth_definition": {
            "columns": dict(FLATMAP_V3_DEPTH_COLUMN_MAPPING),
            "bounds_um": list(shaped_grid.depth_bounds_um),
            "invalid_below_um": shaped_grid.depth_invalid_below_um,
            "mirror_coord_axis": int(mirror_coord_axis),
            "policy": "original_voxel_then_mirror_depth_voxel_if_invalid",
        },
        "validity": {
            "invalid_zero_sentinel": shaped_grid.invalid_zero_sentinel,
            "invalid_negative_one_sentinel": (
                shaped_grid.invalid_negative_one_sentinel
            ),
            "bilateral_xy_policy": "original_voxel_only",
        },
        "algorithms": {
            "format": "dual_bilateral_flatmap_v1",
            "coordinate_to_voxel": "nearest_floor_plus_half_v1",
            "depth_recovery": "mirror_voxel_axis_v1",
            "lookup_set_identity": "canonical_json_sha256_v1",
        },
        "columns": list(FLATMAP_V3_AUGMENTED_COLUMNS),
        "lookup_modes": {
            "flatmap": [FLATMAP_LOOKUP_DIRECT, FLATMAP_LOOKUP_UNMAPPED],
            "depth": [
                FLATMAP_LOOKUP_DIRECT,
                FLATMAP_LOOKUP_MIRRORED_DEPTH,
                FLATMAP_LOOKUP_UNMAPPED,
            ],
        },
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
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return metadata


def _temporary_parquet_path(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    ) as temporary_file:
        return Path(temporary_file.name)


def _v3_invalid_codes(
    *,
    finite: np.ndarray,
    in_bounds: np.ndarray,
    flatmap_valid: np.ndarray | None = None,
    depth_valid: np.ndarray | None = None,
) -> np.ndarray:
    codes = np.full(len(finite), FLATMAP_INVALID_CODE_VALID, dtype=np.int8)
    codes[~finite] = FLATMAP_INVALID_CODE_MISSING_INPUT
    codes[finite & ~in_bounds] = FLATMAP_INVALID_CODE_OUT_OF_BOUNDS
    if flatmap_valid is not None:
        codes[in_bounds & ~flatmap_valid] = FLATMAP_INVALID_CODE_INVALID_FLATMAP
        if depth_valid is not None:
            codes[in_bounds & flatmap_valid & ~depth_valid] = (
                FLATMAP_INVALID_CODE_INVALID_DEPTH
            )
    elif depth_valid is not None:
        codes[in_bounds & ~depth_valid] = FLATMAP_INVALID_CODE_INVALID_DEPTH
    return codes


def _project_v3_batch(
    source_table: pa.Table,
    *,
    lookup_set: FlatmapLookupSet,
    shaped_volume: np.ndarray,
    square_volume: np.ndarray,
    depth_volume: np.ndarray,
    coordinate_mode: str,
    mirror_coord_axis: int,
) -> tuple[pa.Table, dict[str, int]]:
    nodes = source_table.select(list(REQUIRED_NODE_COLUMNS)).to_pandas()
    coords = nodes[["x", "y", "z"]].to_numpy(dtype=float)
    voxels, finite = coordinates_to_voxel_indices(
        coords,
        coordinate_mode=coordinate_mode,
        resolution_um=float(lookup_set.lookup_resolution_um[0]),
        space_directions=np.asarray(lookup_set.space_directions, dtype=float),
        space_origin=np.asarray(lookup_set.space_origin, dtype=float),
    )
    spatial_shape = np.asarray(lookup_set.spatial_shape, dtype=np.int64)
    in_bounds = (
        finite
        & np.all(voxels >= 0, axis=1)
        & np.all(voxels < spatial_shape, axis=1)
    )
    row_count = len(nodes)

    depth_values = np.full(row_count, np.nan, dtype=np.float32)
    if in_bounds.any():
        idx = voxels[in_bounds]
        depth_values[in_bounds] = depth_volume[idx[:, 0], idx[:, 1], idx[:, 2]]
    depth_valid = in_bounds & np.isfinite(depth_values) & (depth_values >= 0.0)
    depth_modes = np.full(row_count, FLATMAP_LOOKUP_UNMAPPED, dtype=object)
    depth_modes[depth_valid] = FLATMAP_LOOKUP_DIRECT

    retry_depth = in_bounds & ~depth_valid
    if retry_depth.any():
        mirrored = voxels[retry_depth].copy()
        mirrored[:, mirror_coord_axis] = (
            spatial_shape[mirror_coord_axis]
            - 1
            - mirrored[:, mirror_coord_axis]
        )
        mirrored_values = depth_volume[
            mirrored[:, 0], mirrored[:, 1], mirrored[:, 2]
        ]
        recovered = np.isfinite(mirrored_values) & (mirrored_values >= 0.0)
        if recovered.any():
            retry_positions = np.flatnonzero(retry_depth)
            recovered_positions = retry_positions[recovered]
            depth_values[recovered_positions] = mirrored_values[recovered]
            depth_valid[recovered_positions] = True
            depth_modes[recovered_positions] = FLATMAP_LOOKUP_MIRRORED_DEPTH

    arrays: list[tuple[str, pa.Array]] = []
    counts: dict[str, int] = {
        "depth_direct": int((depth_modes == FLATMAP_LOOKUP_DIRECT).sum()),
        "depth_mirrored": int(
            (depth_modes == FLATMAP_LOOKUP_MIRRORED_DEPTH).sum()
        ),
        "depth_unmapped": int((depth_modes == FLATMAP_LOOKUP_UNMAPPED).sum()),
    }
    for style, volume in (
        ("both_shaped", shaped_volume),
        ("both_square", square_volume),
    ):
        mapping = FLATMAP_V3_STYLE_COLUMN_MAPPING[style]
        xy = np.full((row_count, 2), np.nan, dtype=np.float32)
        if in_bounds.any():
            idx = voxels[in_bounds]
            xy[in_bounds] = volume[idx[:, 0], idx[:, 1], idx[:, 2]]
        xy_valid = in_bounds & np.all(np.isfinite(xy), axis=1)
        grid = lookup_set.grid_for_style(style)
        if grid.invalid_negative_one_sentinel:
            xy_valid &= ~((xy[:, 0] == -1.0) & (xy[:, 1] == -1.0))
        if grid.invalid_zero_sentinel:
            xy_valid &= ~((xy[:, 0] == 0.0) & (xy[:, 1] == 0.0))
        projection_valid = xy_valid & depth_valid
        xy[~xy_valid] = np.nan
        lookup_modes = np.full(row_count, FLATMAP_LOOKUP_UNMAPPED, dtype=object)
        lookup_modes[xy_valid] = FLATMAP_LOOKUP_DIRECT
        arrays.extend(
            [
                (mapping["x"], pa.array(xy[:, 0], type=pa.float32())),
                (mapping["y"], pa.array(xy[:, 1], type=pa.float32())),
                (mapping["valid"], pa.array(xy_valid, type=pa.bool_())),
                (
                    mapping["projection_valid"],
                    pa.array(projection_valid, type=pa.bool_()),
                ),
                (
                    mapping["invalid_code"],
                    pa.array(
                        _v3_invalid_codes(
                            finite=finite,
                            in_bounds=in_bounds,
                            flatmap_valid=xy_valid,
                            depth_valid=depth_valid,
                        ),
                        type=pa.int8(),
                    ),
                ),
                (mapping["lookup_mode"], pa.array(lookup_modes, type=pa.string())),
            ]
        )
        counts[f"{style}_valid"] = int(xy_valid.sum())

    depth_values[~depth_valid] = np.nan
    arrays.extend(
        [
            (
                FLATMAP_V3_DEPTH_COLUMN_MAPPING["depth"],
                pa.array(depth_values, type=pa.float32()),
            ),
            (
                FLATMAP_V3_DEPTH_COLUMN_MAPPING["valid"],
                pa.array(depth_valid, type=pa.bool_()),
            ),
            (
                FLATMAP_V3_DEPTH_COLUMN_MAPPING["invalid_code"],
                pa.array(
                    _v3_invalid_codes(
                        finite=finite,
                        in_bounds=in_bounds,
                        depth_valid=depth_valid,
                    ),
                    type=pa.int8(),
                ),
            ),
            (
                FLATMAP_V3_DEPTH_COLUMN_MAPPING["lookup_mode"],
                pa.array(depth_modes, type=pa.string()),
            ),
        ]
    )

    existing = [
        name for name in FLATMAP_V3_AUGMENTED_COLUMNS if name in source_table.column_names
    ]
    augmented = source_table.drop(existing) if existing else source_table
    for name, array in arrays:
        augmented = augmented.append_column(name, array)
    return augmented, counts


def _empty_v3_schema(source_schema: pa.Schema) -> pa.Schema:
    schema = pa.schema(
        [field for field in source_schema if field.name not in FLATMAP_V3_AUGMENTED_COLUMNS]
    )
    fields = {
        **{
            mapping["x"]: pa.float32()
            for mapping in FLATMAP_V3_STYLE_COLUMN_MAPPING.values()
        },
        **{
            mapping["y"]: pa.float32()
            for mapping in FLATMAP_V3_STYLE_COLUMN_MAPPING.values()
        },
        **{
            mapping["valid"]: pa.bool_()
            for mapping in FLATMAP_V3_STYLE_COLUMN_MAPPING.values()
        },
        **{
            mapping["projection_valid"]: pa.bool_()
            for mapping in FLATMAP_V3_STYLE_COLUMN_MAPPING.values()
        },
        **{
            mapping["invalid_code"]: pa.int8()
            for mapping in FLATMAP_V3_STYLE_COLUMN_MAPPING.values()
        },
        **{
            mapping["lookup_mode"]: pa.string()
            for mapping in FLATMAP_V3_STYLE_COLUMN_MAPPING.values()
        },
        "depth_um": pa.float32(),
        "depth_valid": pa.bool_(),
        "depth_invalid_code": pa.int8(),
        "depth_lookup_mode": pa.string(),
    }
    for name in FLATMAP_V3_AUGMENTED_COLUMNS:
        if name not in schema.names:
            schema = schema.append(pa.field(name, fields[name]))
    return schema


def augment_neuron_parquet_with_flatmaps(
    source_parquet: str | Path,
    output_parquet: str | Path,
    lookup_set: FlatmapLookupSet,
    *,
    coordinate_mode: str = COORDINATE_MODE_MICRONS,
    mirror_coord_axis: int = 2,
    batch_size: int = DEFAULT_FLATMAP_PARQUET_BATCH_SIZE,
    compression: str = "zstd",
    npy_cache_dir: str | Path | None = None,
    progress_callback: _ProgressCallback | None = None,
    cancel_callback: _CancelCallback | None = None,
) -> FlatmapParquetAugmentationSummary:
    """Atomically append bilateral shaped/square coordinates to every row.

    XY coordinates always come from the original lookup voxel. Shared depth is
    read there first and, only when invalid, retried at the voxel mirrored
    across ``mirror_coord_axis``.
    """
    source_path = Path(source_parquet)
    output_path = Path(output_parquet)
    if mirror_coord_axis not in (0, 1, 2):
        raise ValueError("mirror_coord_axis must be 0, 1, or 2.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    try:
        shaped_set = load_flatmap_volume_set(
            lookup_set.shaped_path,
            lookup_set.depth_path,
            npy_cache_dir=npy_cache_dir,
            mmap_npy=True,
            cancel_callback=cancel_callback,
        )
        square_set = load_flatmap_volume_set(
            lookup_set.square_path,
            lookup_set.depth_path,
            npy_cache_dir=npy_cache_dir,
            mmap_npy=True,
            cancel_callback=cancel_callback,
        )
    except FlatmapLookupLoadCancelledError as exc:
        raise FlatmapParquetCancelledError(
            "Flatmap Parquet augmentation was cancelled."
        ) from exc
    if (
        tuple(shaped_set.flatmap.shape[:3]) != lookup_set.spatial_shape
        or tuple(square_set.flatmap.shape[:3]) != lookup_set.spatial_shape
        or tuple(shaped_set.depth.shape) != lookup_set.spatial_shape
    ):
        raise ValueError("Loaded lookup arrays no longer match the lookup-set shape.")

    source_file = pq.ParquetFile(source_path)
    _require_source_columns(source_file.schema_arrow)
    total_rows = int(source_file.metadata.num_rows)
    output_metadata = _v3_schema_metadata(
        dict(source_file.schema_arrow.metadata or {}),
        source_parquet=source_path,
        lookup_set=lookup_set,
        coordinate_mode=coordinate_mode,
        mirror_coord_axis=mirror_coord_axis,
    )
    write_target = _temporary_parquet_path(output_path)
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    counters = {
        "depth_direct": 0,
        "depth_mirrored": 0,
        "depth_unmapped": 0,
        "both_shaped_valid": 0,
        "both_square_valid": 0,
    }
    try:
        for batch in source_file.iter_batches(batch_size=batch_size):
            if cancel_callback is not None and cancel_callback():
                raise FlatmapParquetCancelledError(
                    "Flatmap Parquet augmentation was cancelled."
                )
            source_table = pa.Table.from_batches([batch])
            augmented, batch_counts = _project_v3_batch(
                source_table,
                lookup_set=lookup_set,
                shaped_volume=shaped_set.flatmap,
                square_volume=square_set.flatmap,
                depth_volume=shaped_set.depth,
                coordinate_mode=coordinate_mode,
                mirror_coord_axis=mirror_coord_axis,
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    write_target,
                    augmented.schema.with_metadata(output_metadata),
                    compression=compression,
                )
            writer.write_table(augmented.cast(writer.schema))
            rows_written += int(source_table.num_rows)
            for key, value in batch_counts.items():
                counters[key] += int(value)
            if progress_callback is not None:
                progress_callback(
                    "Adding bilateral flatmap/depth columns...",
                    rows_written,
                    total_rows,
                )

        if writer is None:
            empty_schema = _empty_v3_schema(source_file.schema_arrow).with_metadata(
                output_metadata
            )
            writer = pq.ParquetWriter(
                write_target,
                empty_schema,
                compression=compression,
            )
        writer.close()
        writer = None
        if cancel_callback is not None and cancel_callback():
            raise FlatmapParquetCancelledError(
                "Flatmap Parquet augmentation was cancelled."
            )
        source_file.close()
        write_target.replace(output_path)
    except BaseException:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        try:
            source_file.close()
        except Exception:
            pass
        try:
            write_target.unlink()
        except FileNotFoundError:
            pass
        raise

    return FlatmapParquetAugmentationSummary(
        source_parquet=source_path,
        output_parquet=output_path,
        flatmap_path=lookup_set.shaped_path,
        depth_path=lookup_set.depth_path,
        rows=rows_written,
        direct_rows=counters["depth_direct"],
        mirrored_rows=0,
        unmapped_rows=counters["depth_unmapped"],
        mirrored_depth_rows=counters["depth_mirrored"],
        lookup_set_id=lookup_set.lookup_set_id,
        square_flatmap_path=lookup_set.square_path,
        shaped_valid_rows=counters["both_shaped_valid"],
        square_valid_rows=counters["both_square_valid"],
    )


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

    style = flatmap_style or flatmap_source.name

    volume_set = load_flatmap_volume_set(flatmap_source, depth_source)
    resolved_midline = resolve_flatmap_mirror_midline(
        coordinate_mode=coordinate_mode,
        flatmap_shape=tuple(int(size) for size in volume_set.flatmap.shape[:3]),
        mirror_coord_axis=mirror_coord_axis,
        mirror_midline=mirror_midline,
    )
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
    mirrored_depth_rows = 0
    mirrored_rows = 0
    unmapped_rows = 0
    writer: pq.ParquetWriter | None = None

    try:
        for batch in batches:
            source_table = pa.Table.from_batches([batch])
            nodes = source_table.to_pandas()
            projected = project_neuron_nodes_to_flatmap(
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
            direct_rows += int((modes == FLATMAP_LOOKUP_DIRECT).sum())
            mirrored_depth_rows += int(
                (modes == FLATMAP_LOOKUP_MIRRORED_DEPTH).sum()
            )
            mirrored_rows += int((modes == FLATMAP_LOOKUP_MIRRORED).sum())
            unmapped_rows += int((modes == FLATMAP_LOOKUP_UNMAPPED).sum())
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
        mirrored_depth_rows=mirrored_depth_rows,
        mirrored_rows=mirrored_rows,
        unmapped_rows=unmapped_rows,
    )
