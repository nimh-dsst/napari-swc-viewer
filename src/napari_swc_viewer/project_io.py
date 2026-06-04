"""Lossless project bundle and enhanced Parquet I/O helpers."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_BUNDLE_FORMAT = "napari_swc_viewer.project_bundle"
PROJECT_FORMAT_VERSION = "1"
PROJECT_METADATA_PREFIX = "napari_swc_viewer.project."

ENHANCED_NEURON_COLUMNS = (
    "neuron_label",
    "neuron_group",
    "neuron_tags_json",
    "neuron_notes",
    "cluster_assignment",
)
_TEXT_ENHANCED_COLUMNS = (
    "neuron_label",
    "neuron_group",
    "neuron_tags_json",
    "neuron_notes",
)
_SOURCE_FILE_ID_INLINE_LIMIT = 10_000
_ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class ProjectLayer:
    """One layer restored from a project bundle."""

    layer_id: str
    data: np.ndarray
    metadata: dict[str, Any]
    data_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class ProjectBundle:
    """Loaded project bundle contents."""

    path: Path
    manifest: dict[str, Any]
    source_parquet_path: Path
    table_state: dict[str, Any]
    layers: tuple[ProjectLayer, ...]


def _json_safe(value: Any) -> Any:
    """Return a JSON-serializable representation of common scientific values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, bool, int, float)):
        return enum_value
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def _emit_progress(
    progress_callback: _ProgressCallback | None,
    message: str,
    current: int,
    total: int,
) -> None:
    if progress_callback is not None:
        progress_callback(message, current, total)


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_table_state(
    table_state: Mapping[str, Any] | Sequence[Any] | None,
) -> dict[str, Any]:
    if table_state is None:
        return {"version": PROJECT_FORMAT_VERSION, "entries": []}
    if isinstance(table_state, Mapping):
        entries = table_state.get("entries", [])
        return {
            **dict(table_state),
            "version": str(table_state.get("version", PROJECT_FORMAT_VERSION)),
            "entries": list(entries) if not isinstance(entries, str) else [],
        }
    return {"version": PROJECT_FORMAT_VERSION, "entries": list(table_state)}


def _entry_file_id(entry: Mapping[str, Any]) -> str | None:
    file_id = entry.get("file_id")
    if file_id is None:
        return None
    text = str(file_id)
    return text if text else None


def _table_state_by_file_id(
    table_state: Mapping[str, Any] | Sequence[Any] | None,
) -> dict[str, Mapping[str, Any]]:
    normalized = _normalise_table_state(table_state)
    out: dict[str, Mapping[str, Any]] = {}
    for raw_entry in normalized.get("entries", []):
        if not isinstance(raw_entry, Mapping):
            continue
        file_id = _entry_file_id(raw_entry)
        if file_id is not None:
            out[file_id] = raw_entry
    return out


def _tags_json_from_entry(entry: Mapping[str, Any]) -> str | None:
    if "neuron_tags_json" in entry and entry["neuron_tags_json"] not in (None, ""):
        return str(entry["neuron_tags_json"])
    tags = entry.get("tags", ())
    if tags is None or tags == "":
        return None
    if isinstance(tags, str):
        values = [part.strip() for part in tags.split(",") if part.strip()]
    else:
        try:
            values = [str(part).strip() for part in tags if str(part).strip()]
        except TypeError:
            values = [str(tags).strip()]
    return json.dumps(values, sort_keys=True)


def _cluster_from_entry(entry: Mapping[str, Any]) -> int | None:
    value = entry.get("cluster_id", entry.get("cluster_assignment"))
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _enhanced_column_values(
    file_ids: Iterable[Any],
    table_state: Mapping[str, Any] | Sequence[Any] | None,
) -> dict[str, list[Any]]:
    by_file_id = _table_state_by_file_id(table_state)
    values: dict[str, list[Any]] = {column: [] for column in ENHANCED_NEURON_COLUMNS}
    for file_id in file_ids:
        entry = by_file_id.get(str(file_id), {})
        values["neuron_label"].append(entry.get("label", entry.get("neuron_label")))
        values["neuron_group"].append(entry.get("group", entry.get("neuron_group")))
        values["neuron_tags_json"].append(_tags_json_from_entry(entry))
        values["neuron_notes"].append(entry.get("notes", entry.get("neuron_notes")))
        values["cluster_assignment"].append(_cluster_from_entry(entry))
    return values


def _append_or_replace_column(table: pa.Table, name: str, array: pa.Array) -> pa.Table:
    if name in table.column_names:
        return table.set_column(table.column_names.index(name), name, array)
    return table.append_column(name, array)


def _enhanced_parquet_payload(
    *,
    source_parquet_path: Path,
    table_state: Mapping[str, Any] | Sequence[Any] | None,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "version": PROJECT_FORMAT_VERSION,
        "source_parquet_path": str(source_parquet_path),
        "source_parquet_sha256": _sha256_file(source_parquet_path),
        "table_state": _normalise_table_state(table_state),
        "metadata": dict(metadata or {}),
    }


def export_enhanced_neuron_parquet(
    source_parquet_path: str | Path,
    output_path: str | Path,
    *,
    table_state: Mapping[str, Any] | Sequence[Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a neuron Parquet with optional per-neuron labels and JSON metadata."""
    source_path = Path(source_parquet_path)
    output = Path(output_path)
    table = pq.read_table(source_path)
    if "file_id" not in table.column_names:
        raise ValueError("Enhanced neuron Parquet requires a file_id column.")

    values = _enhanced_column_values(table.column("file_id").to_pylist(), table_state)
    for column in _TEXT_ENHANCED_COLUMNS:
        table = _append_or_replace_column(
            table,
            column,
            pa.array(values[column], type=pa.string()),
        )
    table = _append_or_replace_column(
        table,
        "cluster_assignment",
        pa.array(values["cluster_assignment"], type=pa.int32()),
    )

    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata[f"{PROJECT_METADATA_PREFIX}version".encode("utf-8")] = (
        PROJECT_FORMAT_VERSION.encode("utf-8")
    )
    schema_metadata[f"{PROJECT_METADATA_PREFIX}metadata_json".encode("utf-8")] = (
        json.dumps(
            _json_safe(
                _enhanced_parquet_payload(
                    source_parquet_path=source_path,
                    table_state=table_state,
                    metadata=metadata,
                )
            ),
            sort_keys=True,
        ).encode("utf-8")
    )
    table = table.replace_schema_metadata(schema_metadata)

    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output, compression="snappy")
    return output


def _duckdb_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _quote_identifier(name: str) -> str:
    """Return a DuckDB-safe quoted identifier."""
    return '"' + name.replace('"', '""') + '"'


def _text_or_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _selected_table_rows(table_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return current-table rows for filtered project parquet export."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_entry in table_state.get("entries", []):
        if not isinstance(raw_entry, Mapping):
            continue
        file_id = _entry_file_id(raw_entry)
        if file_id is None or file_id in seen:
            continue
        seen.add(file_id)
        rows.append(
            {
                "file_id": file_id,
                "row_order": len(rows),
                "neuron_label": _text_or_none(
                    raw_entry.get("label", raw_entry.get("neuron_label"))
                ),
                "neuron_group": _text_or_none(
                    raw_entry.get("group", raw_entry.get("neuron_group"))
                ),
                "neuron_tags_json": _tags_json_from_entry(raw_entry),
                "neuron_notes": _text_or_none(
                    raw_entry.get("notes", raw_entry.get("neuron_notes"))
                ),
                "cluster_assignment": _cluster_from_entry(raw_entry),
            }
        )
    return rows


def _selected_rows_arrow_table(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    """Build the small selected-neuron table registered with DuckDB."""
    return pa.Table.from_pydict(
        {
            "file_id": [str(row["file_id"]) for row in rows],
            "row_order": [int(row["row_order"]) for row in rows],
            "neuron_label": [row.get("neuron_label") for row in rows],
            "neuron_group": [row.get("neuron_group") for row in rows],
            "neuron_tags_json": [row.get("neuron_tags_json") for row in rows],
            "neuron_notes": [row.get("neuron_notes") for row in rows],
            "cluster_assignment": [row.get("cluster_assignment") for row in rows],
        },
        schema=pa.schema(
            [
                ("file_id", pa.string()),
                ("row_order", pa.int32()),
                ("neuron_label", pa.string()),
                ("neuron_group", pa.string()),
                ("neuron_tags_json", pa.string()),
                ("neuron_notes", pa.string()),
                ("cluster_assignment", pa.int32()),
            ]
        ),
    )


def _source_parquet_provenance(path: Path) -> dict[str, Any]:
    """Return lightweight source-file provenance without hashing large files."""
    stat = path.stat()
    return {
        "original_path": str(path),
        "original_size_bytes": int(stat.st_size),
        "original_mtime_ns": int(stat.st_mtime_ns),
        "original_mtime": datetime.fromtimestamp(
            stat.st_mtime,
            timezone.utc,
        ).isoformat(),
    }


def export_filtered_project_neuron_parquet(
    source_parquet_path: str | Path,
    output_path: str | Path,
    *,
    table_state: Mapping[str, Any] | Sequence[Any] | None = None,
) -> Path:
    """Write only current-table neurons from a source Parquet for project bundles."""
    source_path = Path(source_parquet_path)
    output = Path(output_path)
    table_state_payload = _normalise_table_state(table_state)
    selected_rows = _selected_table_rows(table_state_payload)
    if not selected_rows:
        raise ValueError("Save Project requires at least one neuron in the data table.")

    schema = pq.read_schema(source_path)
    if "file_id" not in schema.names:
        raise ValueError("Project neuron Parquet requires a file_id column.")

    source_columns = [
        name for name in schema.names if name not in ENHANCED_NEURON_COLUMNS
    ]
    select_parts = [
        f"src.{_quote_identifier(name)} AS {_quote_identifier(name)}"
        for name in source_columns
    ]
    select_parts.extend(
        f"sel.{_quote_identifier(column)} AS {_quote_identifier(column)}"
        for column in ENHANCED_NEURON_COLUMNS
    )

    order_parts = ["sel.row_order", 'CAST(src."file_id" AS VARCHAR)']
    if "node_id" in schema.names:
        order_parts.append(f"src.{_quote_identifier('node_id')}")

    output.parent.mkdir(parents=True, exist_ok=True)
    write_target = output
    replace_output = False
    if source_path.resolve() == output.resolve():
        write_target = output.with_name(f"{output.name}.tmp")
        replace_output = True
        if write_target.exists():
            write_target.unlink()

    conn = duckdb.connect()
    try:
        conn.register("selected_neurons", _selected_rows_arrow_table(selected_rows))
        conn.execute(
            f"""
            COPY (
                SELECT {", ".join(select_parts)}
                FROM read_parquet('{_duckdb_path(source_path)}') AS src
                INNER JOIN selected_neurons AS sel
                    ON CAST(src.{_quote_identifier("file_id")} AS VARCHAR) = sel.file_id
                ORDER BY {", ".join(order_parts)}
            )
            TO '{_duckdb_path(write_target)}'
            (FORMAT PARQUET, COMPRESSION 'SNAPPY')
            """
        )
    finally:
        conn.close()

    if replace_output:
        write_target.replace(output)
    return output


def _table_state_from_enhanced_columns(parquet_path: Path, schema: pa.Schema) -> dict[str, Any]:
    available = [column for column in ENHANCED_NEURON_COLUMNS if column in schema.names]
    if not available or "file_id" not in schema.names:
        return {"version": PROJECT_FORMAT_VERSION, "entries": []}

    select_parts = ["CAST(file_id AS VARCHAR) AS file_id"]
    if "subject" in schema.names:
        select_parts.append("MAX(CAST(subject AS VARCHAR)) AS subject")
    else:
        select_parts.append("NULL AS subject")
    for column in _TEXT_ENHANCED_COLUMNS:
        if column in schema.names:
            select_parts.append(f"MAX(CAST({column} AS VARCHAR)) AS {column}")
        else:
            select_parts.append(f"NULL AS {column}")
    if "cluster_assignment" in schema.names:
        select_parts.append("MAX(CAST(cluster_assignment AS INTEGER)) AS cluster_assignment")
    else:
        select_parts.append("NULL AS cluster_assignment")

    conn = duckdb.connect()
    try:
        df = conn.execute(
            f"""
            SELECT {", ".join(select_parts)}
            FROM read_parquet('{_duckdb_path(parquet_path)}')
            GROUP BY file_id
            ORDER BY file_id
            """
        ).fetchdf()
    finally:
        conn.close()

    entries: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        tags_json = row.get("neuron_tags_json")
        tags: list[str] = []
        if tags_json not in (None, ""):
            try:
                decoded = json.loads(str(tags_json))
                if isinstance(decoded, list):
                    tags = [str(value) for value in decoded]
            except json.JSONDecodeError:
                tags = [str(tags_json)]
        cluster = row.get("cluster_assignment")
        if cluster is not None and isinstance(cluster, float) and math.isnan(cluster):
            cluster = None
        entries.append(
            {
                "file_id": row.get("file_id"),
                "subject": row.get("subject") or "",
                "label": row.get("neuron_label") or "",
                "group": row.get("neuron_group") or "",
                "tags": tags,
                "notes": row.get("neuron_notes") or "",
                "cluster_id": None if cluster is None else int(cluster),
            }
        )
    return {"version": PROJECT_FORMAT_VERSION, "entries": entries}


def read_enhanced_parquet_metadata(parquet_path: str | Path) -> dict[str, Any]:
    """Read project metadata from an enhanced Parquet, if present."""
    path = Path(parquet_path)
    schema = pq.read_schema(path)
    metadata = dict(schema.metadata or {})
    json_key = f"{PROJECT_METADATA_PREFIX}metadata_json".encode("utf-8")
    if json_key in metadata:
        payload = json.loads(metadata[json_key].decode("utf-8"))
    else:
        payload = {"version": PROJECT_FORMAT_VERSION}

    if "table_state" not in payload:
        payload["table_state"] = _table_state_from_enhanced_columns(path, schema)
    payload["has_project_metadata"] = json_key in metadata
    payload["enhanced_columns"] = [
        column for column in ENHANCED_NEURON_COLUMNS if column in schema.names
    ]
    return payload


def _is_app_created_layer(layer: Any) -> bool:
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, Mapping):
        return False
    return bool(metadata.get("heatmap_source") or metadata.get("mask_query_source"))


def _layer_kind(layer: Any) -> str:
    metadata = getattr(layer, "metadata", None)
    class_name = layer.__class__.__name__.lower()
    if isinstance(metadata, Mapping) and metadata.get("mask_query_source"):
        return "labels"
    if "labels" in class_name:
        return "labels"
    return "image"


def _sequence_or_none(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return [value.decode("utf-8") if isinstance(value, bytes) else value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _first_present(metadata: Mapping[str, Any], keys: Sequence[str]) -> list[Any]:
    out: list[Any] = []
    for key in keys:
        values = _sequence_or_none(metadata.get(key))
        if not values:
            continue
        out.extend(values)
    seen: set[str] = set()
    deduplicated: list[Any] = []
    for value in out:
        marker = str(value)
        if marker in seen:
            continue
        seen.add(marker)
        deduplicated.append(value)
    return deduplicated


def _write_source_file_ids(
    layer_dir: Path,
    layer_id: str,
    file_ids: list[Any],
) -> dict[str, Any]:
    if len(file_ids) <= _SOURCE_FILE_ID_INLINE_LIMIT:
        return {"file_ids": _json_safe(file_ids)}

    sidecar = layer_dir / "source_file_ids.parquet"
    pq.write_table(
        pa.Table.from_pydict({"file_id": [str(file_id) for file_id in file_ids]}),
        sidecar,
        compression="snappy",
    )
    return {
        "file_ids_path": f"layers/{layer_id}/source_file_ids.parquet",
        "file_ids_count": len(file_ids),
    }


def _source_neuron_sets(
    metadata: Mapping[str, Any],
    *,
    source_parquet_path: Path | None,
    layer_dir: Path,
    layer_id: str,
) -> list[dict[str, Any]]:
    file_ids = _first_present(
        metadata,
        ("query_excluded_file_ids", "source_file_ids", "file_ids"),
    )
    if not file_ids:
        return []

    derivation_keys = (
        "threshold_mode",
        "threshold_value",
        "lower_threshold",
        "upper_threshold",
        "bounds_source",
        "sigma",
        "blur_sigma",
        "merge_mode",
        "heatmap_kind",
        "heatmap_region",
        "heatmap_selected_region_id",
        "heatmap_selected_region_acronym",
        "heatmap_selected_region_ids",
        "heatmap_selected_region_acronyms",
        "heatmap_region_ids",
        "heatmap_include_child_regions",
    )
    payload: dict[str, Any] = {
        "role": "source_neurons",
        "source_parquet_path": metadata.get("source_path")
        or (str(source_parquet_path) if source_parquet_path is not None else None),
        "source_layer_names": metadata.get("source_heatmap_layers", []),
        "source_heatmap_kind": metadata.get("heatmap_kind"),
        "count": len(file_ids),
        "derivation": {
            key: metadata.get(key)
            for key in derivation_keys
            if key in metadata
        },
    }
    payload.update(_write_source_file_ids(layer_dir, layer_id, file_ids))
    return [payload]


def _label_key_payload(key: Any) -> dict[str, Any]:
    """Return a JSON-safe label-key payload that can preserve None and integers."""
    if key is None:
        return {"type": "none", "value": None}
    if isinstance(key, np.generic):
        key = key.item()
    if isinstance(key, int):
        return {"type": "int", "value": int(key)}
    return {"type": "str", "value": str(key)}


def _color_value(value: Any) -> list[float] | None:
    """Return a color value as a float list, if possible."""
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size == 0:
        return None
    return [float(component) for component in array.tolist()]


def _serialize_colormap(colormap: Any) -> dict[str, Any] | None:
    """Return a JSON-safe colormap payload for image and labels layers."""
    if colormap is None:
        return None

    color_dict = getattr(colormap, "color_dict", None)
    if isinstance(color_dict, Mapping):
        return {
            "type": "direct_label_colormap",
            "name": getattr(colormap, "name", None),
            "color_dict": [
                {
                    "label": _label_key_payload(key),
                    "color": _color_value(value),
                }
                for key, value in color_dict.items()
            ],
            "default_color": _color_value(getattr(colormap, "default_color", None)),
            "use_selection": bool(getattr(colormap, "use_selection", False)),
        }

    colors = getattr(colormap, "colors", None)
    if colors is None:
        return {"type": "named_colormap", "name": getattr(colormap, "name", str(colormap))}

    interpolation = getattr(colormap, "interpolation", None)
    return {
        "type": "colormap",
        "name": getattr(colormap, "name", None),
        "colors": _json_safe(np.asarray(colors, dtype=float)),
        "controls": _json_safe(getattr(colormap, "controls", None)),
        "interpolation": _json_safe(interpolation),
        "low_color": _color_value(getattr(colormap, "low_color", None)),
        "high_color": _color_value(getattr(colormap, "high_color", None)),
        "nan_color": _color_value(getattr(colormap, "nan_color", None)),
    }


def _layer_display_metadata(layer: Any) -> dict[str, Any]:
    display: dict[str, Any] = {}
    for attr_name in (
        "visible",
        "opacity",
        "scale",
        "translate",
        "blending",
        "rendering",
        "contrast_limits",
        "contrast_limits_range",
    ):
        if hasattr(layer, attr_name):
            display[attr_name] = getattr(layer, attr_name)

    colormap = getattr(layer, "colormap", None)
    if colormap is not None:
        display["colormap_name"] = getattr(colormap, "name", str(colormap))
        serialized = _serialize_colormap(colormap)
        if serialized is not None:
            display["colormap"] = serialized
    return display


def _layer_metadata_payload(
    layer: Any,
    *,
    layer_id: str,
    layer_dir: Path,
    source_parquet_path: Path | None,
) -> dict[str, Any]:
    metadata = getattr(layer, "metadata", None)
    metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    return {
        "version": PROJECT_FORMAT_VERSION,
        "id": layer_id,
        "layer_type": _layer_kind(layer),
        "name": str(getattr(layer, "name", layer_id)),
        "display": _layer_display_metadata(layer),
        "metadata": metadata,
        "source_neuron_sets": _source_neuron_sets(
            metadata,
            source_parquet_path=source_parquet_path,
            layer_dir=layer_dir,
            layer_id=layer_id,
        ),
    }


def save_project_bundle(
    bundle_path: str | Path,
    *,
    source_parquet_path: str | Path,
    table_state: Mapping[str, Any] | Sequence[Any] | None = None,
    layers: Iterable[Any] = (),
    atlas_name: str | None = None,
    analysis_metadata: Mapping[str, Any] | None = None,
    progress_callback: _ProgressCallback | None = None,
) -> Path:
    """Save a project bundle directory for app-created layer state."""
    bundle = Path(bundle_path)
    app_layers = [layer for layer in layers if _is_app_created_layer(layer)]
    total_steps = 4 + len(app_layers)
    _emit_progress(progress_callback, "Preparing project bundle...", 0, total_steps)

    source_path = Path(source_parquet_path)
    table_state_payload = _normalise_table_state(table_state)
    selected_rows = _selected_table_rows(table_state_payload)
    if not selected_rows:
        raise ValueError("Save Project requires at least one neuron in the data table.")

    data_dir = bundle / "data"
    layers_dir = bundle / "layers"
    data_dir.mkdir(parents=True, exist_ok=True)
    layers_dir.mkdir(parents=True, exist_ok=True)

    bundled_parquet = data_dir / "source_neurons.parquet"
    _emit_progress(
        progress_callback,
        "Writing filtered neuron Parquet...",
        1,
        total_steps,
    )
    export_filtered_project_neuron_parquet(
        source_path,
        bundled_parquet,
        table_state=table_state_payload,
    )

    table_state_path = data_dir / "table_state.json"
    _emit_progress(progress_callback, "Writing table state...", 2, total_steps)
    _write_json(table_state_path, table_state_payload)

    manifest_layers: list[dict[str, Any]] = []
    for index, layer in enumerate(app_layers):
        layer_id = f"layer_{index:04d}"
        layer_dir = layers_dir / layer_id
        layer_dir.mkdir(parents=True, exist_ok=True)
        layer_name = str(getattr(layer, "name", layer_id))
        _emit_progress(
            progress_callback,
            f"Saving layer {index + 1}/{len(app_layers)}: {layer_name}",
            3 + index,
            total_steps,
        )

        data_path = layer_dir / "data.npy"
        np.save(data_path, np.asarray(getattr(layer, "data")))

        metadata_payload = _layer_metadata_payload(
            layer,
            layer_id=layer_id,
            layer_dir=layer_dir,
            source_parquet_path=source_path,
        )
        metadata_path = layer_dir / "metadata.json"
        _write_json(metadata_path, metadata_payload)

        manifest_layers.append(
            {
                "id": layer_id,
                "layer_type": metadata_payload["layer_type"],
                "name": metadata_payload["name"],
                "data_path": f"layers/{layer_id}/data.npy",
                "metadata_path": f"layers/{layer_id}/metadata.json",
            }
        )

    _emit_progress(progress_callback, "Writing project manifest...", total_steps - 1, total_steps)
    manifest = {
        "format": PROJECT_BUNDLE_FORMAT,
        "version": PROJECT_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_parquet": {
            "path": "data/source_neurons.parquet",
            "sha256": _sha256_file(bundled_parquet),
            "is_filtered_subset": True,
            "filter": {
                "type": "current_table_file_ids",
                "file_id_count": len(selected_rows),
            },
            **_source_parquet_provenance(source_path),
        },
        "table_state_path": "data/table_state.json",
        "atlas_name": atlas_name,
        "analysis_metadata": dict(analysis_metadata or {}),
        "layers": manifest_layers,
    }
    _write_json(bundle / "manifest.json", manifest)
    _emit_progress(progress_callback, "Done", total_steps, total_steps)
    return bundle


def load_project_bundle(bundle_path: str | Path) -> ProjectBundle:
    """Load a project bundle manifest, table state, and saved layer arrays."""
    bundle = Path(bundle_path)
    manifest = _read_json(bundle / "manifest.json")
    if manifest.get("format") != PROJECT_BUNDLE_FORMAT:
        raise ValueError(f"Unsupported project bundle format: {manifest.get('format')}")

    source_info = manifest.get("source_parquet")
    if not isinstance(source_info, Mapping) or not source_info.get("path"):
        raise ValueError("Project bundle manifest is missing source_parquet.path.")
    source_parquet = bundle / str(source_info["path"])
    if not source_parquet.exists():
        raise FileNotFoundError(f"Project source Parquet not found: {source_parquet}")

    table_state_ref = manifest.get("table_state_path", "data/table_state.json")
    table_state = _read_json(bundle / str(table_state_ref))

    loaded_layers: list[ProjectLayer] = []
    for layer_ref in manifest.get("layers", []):
        if not isinstance(layer_ref, Mapping):
            continue
        layer_id = str(layer_ref.get("id", f"layer_{len(loaded_layers):04d}"))
        data_path = bundle / str(layer_ref.get("data_path", ""))
        metadata_path = bundle / str(layer_ref.get("metadata_path", ""))
        if not data_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Project layer files are incomplete for {layer_id}.")
        loaded_layers.append(
            ProjectLayer(
                layer_id=layer_id,
                data=np.load(data_path, allow_pickle=False),
                metadata=_read_json(metadata_path),
                data_path=data_path,
                metadata_path=metadata_path,
            )
        )

    return ProjectBundle(
        path=bundle,
        manifest=manifest,
        source_parquet_path=source_parquet,
        table_state=table_state,
        layers=tuple(loaded_layers),
    )


def copy_project_source_parquet(
    source_parquet_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Copy a source Parquet file without changing its schema or metadata."""
    source = Path(source_parquet_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    return output
