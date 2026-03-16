"""Standard point Parquet import and CSV normalization utilities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .hemisphere import get_atlas_midline

REQUIRED_POINT_COLUMNS = ("label", "x", "y", "z")
OPTIONAL_POINT_COLUMNS = ("region_name", "acronym", "id", "hemisphere")
STANDARD_POINT_COLUMNS = REQUIRED_POINT_COLUMNS + OPTIONAL_POINT_COLUMNS

_STRING_OPTIONAL_COLUMNS = ("region_name", "acronym", "hemisphere")


class PointImportError(ValueError):
    """Raised when a point import file or mapping is invalid."""


@dataclass(frozen=True)
class AtlasValidationSummary:
    """Summary of optional metadata validation against an atlas."""

    total_points: int
    checked_fields: tuple[str, ...]
    mismatch_counts: dict[str, int]
    mismatches: pd.DataFrame

    @property
    def total_mismatched_rows(self) -> int:
        return int(len(self.mismatches))

    @property
    def has_mismatches(self) -> bool:
        return self.total_mismatched_rows > 0


def _empty_string_series(length: int) -> pd.Series:
    return pd.Series(pd.array([pd.NA] * length, dtype="string"))


def _empty_int_series(length: int) -> pd.Series:
    return pd.Series(pd.array([pd.NA] * length, dtype="Int64"))


def _normalize_string_series(series: pd.Series) -> pd.Series:
    def normalize(value: Any) -> Any:
        if pd.isna(value):
            return pd.NA
        return str(value)

    return series.map(normalize).astype("string")


def _nonempty_string_mask(series: pd.Series) -> pd.Series:
    normalized = _normalize_string_series(series)
    return normalized.notna() & normalized.str.strip().ne("")


def _normalize_region_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().replace("/", ", ")
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return text


def _compare_string_series(left: pd.Series, right: pd.Series) -> pd.Series:
    left_norm = _normalize_string_series(left).str.strip()
    right_norm = _normalize_string_series(right).str.strip()
    return left_norm == right_norm


def _compare_region_name_series(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.map(_normalize_region_name) == right.map(_normalize_region_name)


def _compare_hemisphere_series(left: pd.Series, right: pd.Series) -> pd.Series:
    return (
        _normalize_string_series(left).str.strip().str.lower()
        == _normalize_string_series(right).str.strip().str.lower()
    )


def load_column_mapping(mapping_path: str | Path) -> dict[str, str]:
    """Load and validate a target-to-source JSON column mapping."""

    path = Path(mapping_path)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise PointImportError(f"Mapping file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PointImportError(f"Invalid JSON in mapping file: {path}") from exc

    if not isinstance(data, dict):
        raise PointImportError("Mapping file must contain a single JSON object.")

    unknown_targets = sorted(set(data) - set(STANDARD_POINT_COLUMNS))
    if unknown_targets:
        targets = ", ".join(unknown_targets)
        raise PointImportError(f"Unknown mapping target column(s): {targets}")

    mapping: dict[str, str] = {}
    for target, source in data.items():
        if not isinstance(source, str) or not source.strip():
            raise PointImportError(
                f"Mapping for '{target}' must be a non-empty source column name."
            )
        mapping[str(target)] = source.strip()

    missing_required = [
        column for column in REQUIRED_POINT_COLUMNS if column not in mapping
    ]
    if missing_required:
        columns = ", ".join(missing_required)
        raise PointImportError(f"Missing required mapping(s): {columns}")

    duplicate_sources = sorted(
        source
        for source in set(mapping.values())
        if list(mapping.values()).count(source) > 1
    )
    if duplicate_sources:
        sources = ", ".join(duplicate_sources)
        raise PointImportError(
            f"Source columns cannot be mapped more than once: {sources}"
        )

    return mapping


def validate_standard_point_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a standardized point dataframe."""

    missing_required = [
        column for column in REQUIRED_POINT_COLUMNS if column not in df.columns
    ]
    if missing_required:
        columns = ", ".join(missing_required)
        raise PointImportError(f"Missing required point column(s): {columns}")

    normalized = df.copy()
    extras = [
        column for column in normalized.columns if column not in STANDARD_POINT_COLUMNS
    ]

    labels = _normalize_string_series(normalized["label"]).str.strip()
    invalid_labels = labels.isna() | labels.eq("")
    if invalid_labels.any():
        count = int(invalid_labels.sum())
        raise PointImportError(f"Column 'label' has {count} empty value(s).")
    normalized["label"] = labels

    for column in ("x", "y", "z"):
        values = pd.to_numeric(normalized[column], errors="coerce")
        invalid = values.isna()
        if invalid.any():
            count = int(invalid.sum())
            raise PointImportError(f"Column '{column}' has {count} invalid value(s).")
        normalized[column] = values.astype(float)

    for column in _STRING_OPTIONAL_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = _empty_string_series(len(normalized))
        else:
            normalized[column] = _normalize_string_series(normalized[column])

    if "id" not in normalized.columns:
        normalized["id"] = _empty_int_series(len(normalized))
    else:
        id_series = normalized["id"]
        if pd.api.types.is_string_dtype(id_series) or id_series.dtype == object:
            blanks = _normalize_string_series(id_series).str.strip().eq("")
            id_series = id_series.where(~blanks, pd.NA)
        values = pd.to_numeric(id_series, errors="coerce")
        invalid = pd.Series(False, index=normalized.index)
        invalid |= values.isna() & id_series.notna()
        if invalid.any():
            count = int(invalid.sum())
            raise PointImportError(f"Column 'id' has {count} invalid value(s).")
        normalized["id"] = values.astype("Int64")

    ordered_columns = [*STANDARD_POINT_COLUMNS, *extras]
    return normalized[ordered_columns]


def standardize_point_dataframe(
    raw_df: pd.DataFrame,
    mapping: dict[str, str],
) -> pd.DataFrame:
    """Apply a target-to-source mapping to produce a standardized dataframe."""

    missing_sources = sorted(set(mapping.values()) - set(raw_df.columns))
    if missing_sources:
        columns = ", ".join(missing_sources)
        raise PointImportError(f"Mapped source column(s) not found in CSV: {columns}")

    mapped_sources = set(mapping.values())
    conflicting_extras = sorted(
        column
        for column in raw_df.columns
        if column not in mapped_sources and column in STANDARD_POINT_COLUMNS
    )
    if conflicting_extras:
        columns = ", ".join(conflicting_extras)
        raise PointImportError(
            "Unmapped source columns conflict with standardized column names: "
            f"{columns}"
        )

    standardized = pd.DataFrame(index=raw_df.index)
    for column in REQUIRED_POINT_COLUMNS:
        standardized[column] = raw_df[mapping[column]]

    for column in OPTIONAL_POINT_COLUMNS:
        if column in mapping:
            standardized[column] = raw_df[mapping[column]]
        elif column == "id":
            standardized[column] = _empty_int_series(len(raw_df))
        else:
            standardized[column] = _empty_string_series(len(raw_df))

    for column in raw_df.columns:
        if column not in mapped_sources:
            standardized[column] = raw_df[column]

    return validate_standard_point_dataframe(standardized)


def load_raw_point_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a raw point CSV file."""

    path = Path(csv_path)
    try:
        return pd.read_csv(path)
    except FileNotFoundError as exc:
        raise PointImportError(f"CSV file not found: {path}") from exc


def convert_point_csv_to_parquet(
    csv_path: str | Path,
    mapping_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Convert a raw point CSV plus mapping JSON into standardized Parquet."""

    raw_df = load_raw_point_csv(csv_path)
    mapping = load_column_mapping(mapping_path)
    standardized = standardize_point_dataframe(raw_df, mapping)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    standardized.to_parquet(output_path, index=False)
    return standardized


def load_standard_point_parquet(parquet_path: str | Path) -> pd.DataFrame:
    """Load and validate a standardized point Parquet file."""

    path = Path(parquet_path)
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError as exc:
        raise PointImportError(f"Point Parquet file not found: {path}") from exc
    except Exception as exc:
        raise PointImportError(f"Failed to read Point Parquet file: {path}") from exc
    return validate_standard_point_dataframe(df)


def _build_region_lookup(atlas: Any) -> dict[int, dict[str, Any]]:
    """Build a simple region metadata lookup from a BrainGlobe atlas."""

    lookup: dict[int, dict[str, Any]] = {}
    structures = getattr(atlas, "structures", {})
    for key, structure in structures.items():
        if not isinstance(key, int):
            continue
        try:
            name = structure["name"]
        except Exception:
            name = getattr(structure, "name", "")
        try:
            acronym = structure["acronym"]
        except Exception:
            acronym = getattr(structure, "acronym", "")
        lookup[int(key)] = {
            "id": int(key),
            "name": str(name),
            "acronym": str(acronym),
        }
    return lookup


def _world_coords_to_atlas_region_ids(
    coords_xyz: np.ndarray,
    atlas: Any,
) -> np.ndarray:
    """Map world-space XYZ micron coordinates to atlas annotation region IDs."""

    annotation = np.asarray(atlas.annotation)
    resolution = np.asarray(atlas.resolution, dtype=float)
    lookup_coords = np.asarray(coords_xyz, dtype=float)[:, [2, 1, 0]]
    voxel_coords = np.round(lookup_coords / resolution).astype(int)

    region_ids = np.zeros(len(voxel_coords), dtype=np.int64)
    in_bounds = np.all(
        (voxel_coords >= 0) & (voxel_coords < np.asarray(annotation.shape)),
        axis=1,
    )
    valid = voxel_coords[in_bounds]
    if len(valid) > 0:
        region_ids[in_bounds] = annotation[valid[:, 0], valid[:, 1], valid[:, 2]]
    return region_ids


def validate_point_metadata_against_atlas(
    df: pd.DataFrame,
    atlas: Any,
) -> AtlasValidationSummary:
    """Validate optional point metadata columns against atlas-derived values."""

    standardized = validate_standard_point_dataframe(df)
    coords = standardized[["x", "y", "z"]].to_numpy(dtype=float, copy=False)

    region_ids = _world_coords_to_atlas_region_ids(coords, atlas)
    lookup = _build_region_lookup(atlas)
    derived_region_name = pd.Series(
        [lookup.get(int(region_id), {}).get("name", "") for region_id in region_ids],
        dtype="string",
    )
    derived_acronym = pd.Series(
        [lookup.get(int(region_id), {}).get("acronym", "") for region_id in region_ids],
        dtype="string",
    )

    midline = get_atlas_midline(atlas, coord_axis=2)
    hemisphere_values = np.full(len(standardized), "midline", dtype=object)
    hemisphere_values[coords[:, 2] < midline - 1.0] = "left"
    hemisphere_values[coords[:, 2] > midline + 1.0] = "right"
    derived_hemisphere = pd.Series(hemisphere_values, dtype="string")

    compared_fields: list[str] = []
    mismatch_counts: dict[str, int] = {}
    row_mask = pd.Series(False, index=standardized.index)
    mismatch_fields: dict[int, list[str]] = {}

    def record_mismatches(field: str, mask: pd.Series) -> None:
        mismatch_counts[field] = int(mask.sum())
        if not mask.any():
            return
        for index in standardized.index[mask]:
            mismatch_fields.setdefault(int(index), []).append(field)

    if "id" in standardized.columns:
        supplied = standardized["id"].notna()
        if supplied.any():
            compared_fields.append("id")
            mismatch = supplied & (standardized["id"].astype("Int64") != region_ids)
            row_mask |= mismatch
            record_mismatches("id", mismatch)

    if "acronym" in standardized.columns:
        supplied = _nonempty_string_mask(standardized["acronym"])
        if supplied.any():
            compared_fields.append("acronym")
            mismatch = supplied & ~_compare_string_series(
                standardized["acronym"],
                derived_acronym,
            )
            row_mask |= mismatch
            record_mismatches("acronym", mismatch)

    if "region_name" in standardized.columns:
        supplied = _nonempty_string_mask(standardized["region_name"])
        if supplied.any():
            compared_fields.append("region_name")
            mismatch = supplied & ~_compare_region_name_series(
                standardized["region_name"],
                derived_region_name,
            )
            row_mask |= mismatch
            record_mismatches("region_name", mismatch)

    if "hemisphere" in standardized.columns:
        supplied = _nonempty_string_mask(standardized["hemisphere"])
        if supplied.any():
            compared_fields.append("hemisphere")
            mismatch = supplied & ~_compare_hemisphere_series(
                standardized["hemisphere"],
                derived_hemisphere,
            )
            row_mask |= mismatch
            record_mismatches("hemisphere", mismatch)

    mismatch_df = standardized.loc[row_mask].copy()
    mismatch_df["atlas_region_name"] = derived_region_name[row_mask].to_numpy()
    mismatch_df["atlas_acronym"] = derived_acronym[row_mask].to_numpy()
    mismatch_df["atlas_id"] = region_ids[row_mask.to_numpy()]
    mismatch_df["atlas_hemisphere"] = derived_hemisphere[row_mask].to_numpy()
    mismatch_df["mismatch_fields"] = [
        ",".join(mismatch_fields[int(index)])
        for index in mismatch_df.index
    ]

    return AtlasValidationSummary(
        total_points=len(standardized),
        checked_fields=tuple(compared_fields),
        mismatch_counts=mismatch_counts,
        mismatches=mismatch_df,
    )


def format_atlas_validation_summary(
    summary: AtlasValidationSummary,
    max_examples: int = 5,
) -> str:
    """Format a concise user-facing summary of atlas validation mismatches."""

    if not summary.has_mismatches:
        return (
            f"Atlas validation checked {summary.total_points} point(s) and found "
            "no mismatches."
        )

    count_bits = ", ".join(
        f"{field}: {count}"
        for field, count in summary.mismatch_counts.items()
        if count > 0
    )
    message = (
        f"Atlas validation found {summary.total_mismatched_rows} mismatched point(s) "
        f"out of {summary.total_points}"
    )
    if count_bits:
        message += f" ({count_bits})"
    message += "."

    examples: list[str] = []
    for index, row in summary.mismatches.head(max_examples).iterrows():
        label = row["label"]
        fields = row["mismatch_fields"]
        examples.append(f"row {int(index) + 1} label={label} [{fields}]")

    if examples:
        message += " Examples: " + "; ".join(examples)

    return message


def dataframe_to_point_properties(df: pd.DataFrame) -> dict[str, list[Any]]:
    """Convert a point dataframe into napari point properties."""

    properties: dict[str, list[Any]] = {}
    for column in df.columns:
        if column in {"x", "y", "z"}:
            continue
        series = df[column]
        values = series.astype(object).where(series.notna(), None).tolist()
        properties[column] = values
    return properties

