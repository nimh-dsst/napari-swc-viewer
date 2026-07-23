"""Flatmap-space voxel correlation for clustered neurons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .clustering import ClusterResult, compute_clustermap_data
from ..flatmap_heatmap import FlatmapLookupStats


@dataclass(frozen=True)
class FlatmapVoxelCorrelationSource:
    """Binned flatmap heatmap data used as a clustering source."""

    projected_nodes: pd.DataFrame
    volume_shape: tuple[int, int, int]
    input_file_ids: tuple[str, ...]
    xy_bins: int
    depth_bin_um: float
    include_depth_minus_one: bool
    flatmap_style: str | None = None
    coordinate_mode: str | None = None
    flatmap_path: str | None = None
    depth_path: str | None = None
    invalid_zero_sentinel: bool = False
    invalid_negative_one_sentinel: bool = True
    lookup_stats: FlatmapLookupStats | None = None
    mirror_depth_fallback: bool = True
    mirror_coord_axis: int = 2
    cache_dir: str | None = None
    cache_profile_id: str | None = None
    cache_style: str | None = None


@dataclass(frozen=True)
class FlatmapCountMatrix:
    """Dense count matrix and provenance for flatmap voxel correlation."""

    neuron_ids: list[str]
    voxel_ids: np.ndarray
    count_matrix: np.ndarray
    rendered_node_count: int
    unassigned_neuron_ids: list[str]


def _unique_strings_in_order(values) -> tuple[str, ...]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return tuple(unique)


def _validate_region_mask(
    region_mask: np.ndarray | None,
    volume_shape: tuple[int, int, int],
) -> np.ndarray | None:
    if region_mask is None:
        return None
    mask = np.asarray(region_mask, dtype=bool)
    if mask.shape != tuple(volume_shape):
        raise ValueError(
            "region_mask shape must match the flatmap heatmap volume; "
            f"got {mask.shape} and {tuple(volume_shape)}."
        )
    return mask


def _rendered_binned_nodes(
    projected_nodes: pd.DataFrame,
    volume_shape: tuple[int, int, int],
    *,
    region_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    required = ("file_id", "render_valid", "depth_bin", "y_flat_bin", "x_flat_bin")
    missing = [column for column in required if column not in projected_nodes.columns]
    if missing:
        raise ValueError(f"Projected nodes are missing render column(s): {missing}")
    if len(volume_shape) != 3:
        raise ValueError(f"volume_shape must be 3D; got {volume_shape}.")

    depth_size, y_size, x_size = (int(size) for size in volume_shape)
    mask = _validate_region_mask(region_mask, volume_shape)
    table = projected_nodes.copy()

    render_valid = table["render_valid"].fillna(False).astype(bool).to_numpy()
    depth_bins = pd.to_numeric(table["depth_bin"], errors="coerce").to_numpy(dtype=float)
    y_bins = pd.to_numeric(table["y_flat_bin"], errors="coerce").to_numpy(dtype=float)
    x_bins = pd.to_numeric(table["x_flat_bin"], errors="coerce").to_numpy(dtype=float)
    finite_bins = np.isfinite(depth_bins) & np.isfinite(y_bins) & np.isfinite(x_bins)
    in_bounds = (
        finite_bins
        & (depth_bins >= 0)
        & (depth_bins < depth_size)
        & (y_bins >= 0)
        & (y_bins < y_size)
        & (x_bins >= 0)
        & (x_bins < x_size)
    )
    valid = render_valid & in_bounds
    if mask is not None and valid.any():
        d = depth_bins[valid].astype(np.int64)
        y = y_bins[valid].astype(np.int64)
        x = x_bins[valid].astype(np.int64)
        region_valid = np.zeros(len(table), dtype=bool)
        region_valid[np.flatnonzero(valid)] = mask[d, y, x]
        valid &= region_valid

    rendered = table.loc[valid].copy()
    if rendered.empty:
        return rendered

    for column in ("depth_bin", "y_flat_bin", "x_flat_bin"):
        rendered.loc[:, column] = pd.to_numeric(
            rendered[column], errors="coerce"
        ).astype(np.int64)
    rendered.loc[:, "_flatmap_file_id"] = rendered["file_id"].map(str)
    rendered.loc[:, "_flatmap_linear_voxel"] = (
        rendered["depth_bin"].to_numpy(dtype=np.int64) * y_size * x_size
        + rendered["y_flat_bin"].to_numpy(dtype=np.int64) * x_size
        + rendered["x_flat_bin"].to_numpy(dtype=np.int64)
    )
    return rendered


def build_flatmap_count_matrix(
    source: FlatmapVoxelCorrelationSource,
    *,
    region_mask: np.ndarray | None = None,
) -> FlatmapCountMatrix:
    """Build per-neuron node-count vectors in flatmap heatmap voxel space."""
    rendered = _rendered_binned_nodes(
        source.projected_nodes,
        source.volume_shape,
        region_mask=region_mask,
    )
    input_file_ids = tuple(source.input_file_ids) or _unique_strings_in_order(
        source.projected_nodes["file_id"].tolist()
        if "file_id" in source.projected_nodes.columns
        else []
    )

    if rendered.empty:
        return FlatmapCountMatrix(
            neuron_ids=[],
            voxel_ids=np.empty(0, dtype=np.int64),
            count_matrix=np.empty((0, 0), dtype=np.float32),
            rendered_node_count=0,
            unassigned_neuron_ids=list(input_file_ids),
        )

    rendered_ids = set(rendered["_flatmap_file_id"].tolist())
    neuron_ids = [file_id for file_id in input_file_ids if file_id in rendered_ids]
    if not neuron_ids:
        neuron_ids = list(_unique_strings_in_order(rendered["_flatmap_file_id"].tolist()))

    unassigned = [file_id for file_id in input_file_ids if file_id not in rendered_ids]
    counts = (
        rendered.groupby(["_flatmap_file_id", "_flatmap_linear_voxel"], sort=False)
        .size()
        .reset_index(name="count")
    )
    voxel_ids = np.sort(counts["_flatmap_linear_voxel"].unique().astype(np.int64))
    row_index = {file_id: index for index, file_id in enumerate(neuron_ids)}
    col_index = {int(voxel_id): index for index, voxel_id in enumerate(voxel_ids)}
    matrix = np.zeros((len(neuron_ids), len(voxel_ids)), dtype=np.float32)
    for file_id, voxel_id, count in counts.itertuples(index=False):
        row = row_index.get(str(file_id))
        if row is None:
            continue
        matrix[row, col_index[int(voxel_id)]] = float(count)

    return FlatmapCountMatrix(
        neuron_ids=neuron_ids,
        voxel_ids=voxel_ids,
        count_matrix=matrix,
        rendered_node_count=int(len(rendered)),
        unassigned_neuron_ids=unassigned,
    )


def pearson_correlation_from_counts(count_matrix: np.ndarray) -> np.ndarray:
    """Return a stable Pearson correlation matrix from count vectors."""
    counts = np.asarray(count_matrix, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(f"count_matrix must be 2D; got shape {counts.shape}.")
    n_neurons = counts.shape[0]
    if n_neurons == 0:
        return np.empty((0, 0), dtype=np.float32)

    centered = counts - counts.mean(axis=1, keepdims=True)
    norm = np.sqrt(np.sum(centered * centered, axis=1))
    denom = np.outer(norm, norm)
    corr = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    valid = denom > 0.0
    corr[valid] = (centered @ centered.T)[valid] / denom[valid]
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0).astype(np.float32)


def query_flatmap_soma_coordinates(
    parquet_path: str,
    *,
    style: str,
    file_ids: list[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Return per-neuron soma coordinates in flatmap + depth space.

    Averages the flatmap ``x``/``y`` and ``depth_um`` of each neuron's soma
    nodes (``type == 1``) that have valid flatmap and depth projections, using
    the version-3 column family selected by ``style``.  Returns ``(file_ids,
    coords)`` where ``coords`` has one ``(x_flat, y_flat, depth_um)`` row per
    file, ordered to match ``file_ids``.
    """
    import duckdb

    from ..flatmap_heatmap import (
        _combine_where,
        _duckdb_column_names,
        _duckdb_source_path,
        _file_id_filter,
        _flatmap_sql_expressions,
        _sql_identifier,
        _style_suffix,
    )

    suffix = _style_suffix(style)
    file_filter = _file_id_filter(file_ids)
    if file_filter is None:
        return [], np.empty((0, 3), dtype=float)
    file_filter_sql, params = file_filter

    conn = duckdb.connect()
    try:
        source_sql = f"read_parquet('{_duckdb_source_path(parquet_path)}')"
        column_names = _duckdb_column_names(conn, source_sql)
        # Only the flatmap_valid / depth_valid expressions are used here; the
        # bin math is irrelevant for soma coordinates, so pass placeholder
        # bounds and bins that keep _flatmap_sql_expressions valid.
        expressions = _flatmap_sql_expressions(
            column_names,
            suffix=suffix,
            x_lower=0.0,
            x_upper=1.0,
            y_lower=0.0,
            y_upper=1.0,
            depth_lower=0.0,
            depth_bin_um=1.0,
            xy_bins=1,
            valid_depth_bins=1,
            sentinel_offset=0,
            include_depth_minus_one=False,
        )
        x_ref = _sql_identifier(f"x_flat_{suffix}")
        y_ref = _sql_identifier(f"y_flat_{suffix}")
        soma_where = (
            f"type = 1 AND ({expressions['flatmap_valid']}) "
            f"AND ({expressions['depth_valid']})"
        )
        where_sql = _combine_where(soma_where, file_filter_sql)
        query = f"""
            SELECT
                file_id,
                AVG({x_ref}) AS x_flat,
                AVG({y_ref}) AS y_flat,
                AVG(depth_um) AS depth_um
            FROM {source_sql}
            WHERE {where_sql}
            GROUP BY file_id
            ORDER BY file_id
        """
        soma_df = (
            conn.execute(query, params).fetchdf()
            if params
            else conn.execute(query).fetchdf()
        )
    finally:
        conn.close()

    if soma_df.empty:
        return [], np.empty((0, 3), dtype=float)

    coords = soma_df[["x_flat", "y_flat", "depth_um"]].to_numpy(dtype=float)
    finite = np.all(np.isfinite(coords), axis=1)
    coords = coords[finite]
    ids = soma_df["file_id"].astype(str).to_numpy()[finite].tolist()
    return ids, coords


@dataclass(frozen=True)
class FlatmapParquetCorrelationProvenance:
    """Binning provenance for a parquet-driven flatmap correlation run."""

    style: str
    xy_bins: int
    depth_bin_um: float
    include_depth_minus_one: bool
    volume_shape: tuple[int, int, int]
    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]
    depth_range_um: tuple[float, float]


def _resolve_flatmap_render_bounds(
    parquet_path: str,
    style: str,
    suffix: str,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return canonical (x, y, depth) render bounds for one flatmap style.

    Prefers the canonical bounds recorded in version-3 Parquet metadata so
    partial queries stay aligned to the same grid.  Falls back to data-derived
    min/max for legacy Parquets that never recorded canonical bounds.
    """
    import duckdb

    from ..flatmap_heatmap import _duckdb_source_path
    from ..flatmap_parquet import read_flatmap_parquet_transform_info

    info = read_flatmap_parquet_transform_info(parquet_path)
    grid = info.grid_spec(style)
    if grid is not None:
        return grid.x_bounds, grid.y_bounds, grid.depth_bounds_um

    x_column = f"x_flat_{suffix}"
    y_column = f"y_flat_{suffix}"
    conn = duckdb.connect()
    try:
        source_sql = f"read_parquet('{_duckdb_source_path(parquet_path)}')"
        row = conn.execute(
            f"""
            SELECT
                MIN("{x_column}") FILTER (WHERE isfinite("{x_column}")),
                MAX("{x_column}") FILTER (WHERE isfinite("{x_column}")),
                MIN("{y_column}") FILTER (WHERE isfinite("{y_column}")),
                MAX("{y_column}") FILTER (WHERE isfinite("{y_column}")),
                MIN(depth_um) FILTER (WHERE isfinite(depth_um) AND depth_um >= 0.0),
                MAX(depth_um) FILTER (WHERE isfinite(depth_um) AND depth_um >= 0.0)
            FROM {source_sql}
            """
        ).fetchone()
    finally:
        conn.close()
    if row is None or any(value is None for value in row):
        raise ValueError(
            "Parquet does not contain valid flatmap/depth coordinates for style "
            f"{style!r}; cannot resolve render bounds."
        )
    return (
        (float(row[0]), float(row[1])),
        (float(row[2]), float(row[3])),
        (float(row[4]), float(row[5])),
    )


def build_flatmap_count_matrix_from_bin_counts(
    counts: pd.DataFrame,
    volume_shape: tuple[int, int, int],
    *,
    input_file_ids: tuple[str, ...],
) -> FlatmapCountMatrix:
    """Build per-neuron node-count vectors from DuckDB per-file bin counts.

    ``counts`` must have ``file_id``, ``depth_bin``, ``y_bin``, ``x_bin`` and
    ``node_count`` columns, exactly as produced by
    :func:`napari_swc_viewer.flatmap_heatmap._query_flatmap_bin_counts` with
    ``include_file_id=True``.
    """
    depth_size, y_size, x_size = (int(size) for size in volume_shape)
    ordered_inputs = tuple(str(file_id) for file_id in input_file_ids)
    if counts.empty:
        return FlatmapCountMatrix(
            neuron_ids=[],
            voxel_ids=np.empty(0, dtype=np.int64),
            count_matrix=np.empty((0, 0), dtype=np.float32),
            rendered_node_count=0,
            unassigned_neuron_ids=list(ordered_inputs),
        )

    file_ids = counts["file_id"].map(str).to_numpy()
    linear = (
        counts["depth_bin"].to_numpy(dtype=np.int64) * y_size * x_size
        + counts["y_bin"].to_numpy(dtype=np.int64) * x_size
        + counts["x_bin"].to_numpy(dtype=np.int64)
    )
    node_counts = counts["node_count"].to_numpy(dtype=np.float64)

    rendered_ids = set(file_ids.tolist())
    neuron_ids = [file_id for file_id in ordered_inputs if file_id in rendered_ids]
    if not neuron_ids:
        neuron_ids = list(_unique_strings_in_order(file_ids.tolist()))
    unassigned = [file_id for file_id in ordered_inputs if file_id not in rendered_ids]

    voxel_ids = np.unique(linear.astype(np.int64))
    row_index = {file_id: index for index, file_id in enumerate(neuron_ids)}
    col_index = {int(voxel_id): index for index, voxel_id in enumerate(voxel_ids)}
    matrix = np.zeros((len(neuron_ids), len(voxel_ids)), dtype=np.float32)
    for file_id, voxel_id, count in zip(file_ids, linear, node_counts):
        row = row_index.get(str(file_id))
        if row is None:
            continue
        matrix[row, col_index[int(voxel_id)]] += float(count)

    return FlatmapCountMatrix(
        neuron_ids=neuron_ids,
        voxel_ids=voxel_ids,
        count_matrix=matrix,
        rendered_node_count=int(node_counts.sum()),
        unassigned_neuron_ids=unassigned,
    )


def compute_flatmap_voxel_correlation_from_parquet(
    parquet_path: str,
    *,
    style: str,
    xy_bins: int,
    depth_bin_um: float,
    include_depth_minus_one: bool = True,
    method: str = "average",
    n_clusters: int = 5,
    file_ids: list[str] | None = None,
) -> tuple[ClusterResult, FlatmapCountMatrix, FlatmapParquetCorrelationProvenance]:
    """Cluster neurons by flatmap-space voxel correlation straight from Parquet.

    Reads only the precomputed flatmap/depth columns via DuckDB and bins them
    with a ``GROUP BY`` (mirroring the heatmap fast path), so no rendered
    heatmap is required.  Requires the version-3 bilateral ``*_shaped`` /
    ``*_square`` coordinate columns.
    """
    import duckdb

    from ..flatmap_heatmap import (
        MAX_FLATMAP_HEATMAP_VOXELS,
        _combine_where,
        _depth_bin_count,
        _duckdb_column_names,
        _duckdb_source_path,
        _file_id_filter,
        _flatmap_sql_expressions,
        _nondegenerate_bounds,
        _query_flatmap_bin_counts,
        _style_suffix,
        _validate_resolution,
    )

    suffix = _style_suffix(style)
    xy_bins, depth_bin_um = _validate_resolution(xy_bins, depth_bin_um)

    x_bounds, y_bounds, depth_range = _resolve_flatmap_render_bounds(
        parquet_path, style, suffix
    )
    x_lower, x_upper = _nondegenerate_bounds(x_bounds[0], x_bounds[1])
    y_lower, y_upper = _nondegenerate_bounds(y_bounds[0], y_bounds[1])
    depth_lower, depth_upper = _nondegenerate_bounds(depth_range[0], depth_range[1])

    valid_depth_bins = _depth_bin_count((depth_lower, depth_upper), depth_bin_um)
    sentinel_offset = 1 if include_depth_minus_one else 0
    total_depth_bins = valid_depth_bins + sentinel_offset
    volume_shape = (total_depth_bins, xy_bins, xy_bins)

    voxel_count = int(total_depth_bins * xy_bins * xy_bins)
    if voxel_count > MAX_FLATMAP_HEATMAP_VOXELS:
        raise ValueError(
            "Flatmap voxel grid is too large: "
            f"{total_depth_bins}x{xy_bins}x{xy_bins} voxels. "
            "Use fewer XY bins or a larger depth bin."
        )

    file_filter = _file_id_filter(file_ids)
    if file_filter is None:
        raise ValueError(
            "Flatmap voxel correlation requires at least one neuron; the "
            "selected file set is empty."
        )
    file_filter_sql, file_params = file_filter

    conn = duckdb.connect()
    try:
        source_sql = f"read_parquet('{_duckdb_source_path(parquet_path)}')"
        column_names = _duckdb_column_names(conn, source_sql)
        expressions = _flatmap_sql_expressions(
            column_names,
            suffix=suffix,
            x_lower=x_lower,
            x_upper=x_upper,
            y_lower=y_lower,
            y_upper=y_upper,
            depth_lower=depth_lower,
            depth_bin_um=depth_bin_um,
            xy_bins=xy_bins,
            valid_depth_bins=valid_depth_bins,
            sentinel_offset=sentinel_offset,
            include_depth_minus_one=include_depth_minus_one,
        )
        where_sql = _combine_where(expressions["render_where"], file_filter_sql)
        counts = _query_flatmap_bin_counts(
            conn,
            source_sql,
            expressions,
            where_sql,
            file_params,
            include_file_id=True,
        )
    finally:
        conn.close()

    if file_ids is None:
        input_file_ids = _unique_strings_in_order(counts["file_id"].tolist())
    else:
        input_file_ids = tuple(str(file_id) for file_id in file_ids)

    count_data = build_flatmap_count_matrix_from_bin_counts(
        counts,
        volume_shape,
        input_file_ids=input_file_ids,
    )
    if len(count_data.neuron_ids) < 2:
        raise ValueError(
            "Flatmap voxel correlation requires at least 2 neurons with valid "
            "flatmap/depth coordinates."
        )
    if count_data.count_matrix.shape[1] == 0:
        raise ValueError(
            "Flatmap voxel correlation requires at least one occupied voxel."
        )

    corr = pearson_correlation_from_counts(count_data.count_matrix)
    result = compute_clustermap_data(
        corr,
        count_data.neuron_ids,
        method=method,
        n_clusters=n_clusters,
    )
    result.unassigned_neuron_ids = list(count_data.unassigned_neuron_ids)
    provenance = FlatmapParquetCorrelationProvenance(
        style=style,
        xy_bins=int(xy_bins),
        depth_bin_um=float(depth_bin_um),
        include_depth_minus_one=bool(include_depth_minus_one),
        volume_shape=volume_shape,
        x_bounds=(x_lower, x_upper),
        y_bounds=(y_lower, y_upper),
        depth_range_um=(depth_lower, depth_upper),
    )
    return result, count_data, provenance


def compute_flatmap_voxel_correlation_result(
    source: FlatmapVoxelCorrelationSource,
    *,
    method: str = "average",
    n_clusters: int = 5,
    region_mask: np.ndarray | None = None,
) -> tuple[ClusterResult, FlatmapCountMatrix]:
    """Cluster neurons by Pearson correlation in flatmap heatmap voxel space."""
    count_data = build_flatmap_count_matrix(source, region_mask=region_mask)
    if len(count_data.neuron_ids) < 2:
        raise ValueError(
            "Flatmap voxel correlation requires at least 2 rendered neurons "
            "after applying the selected filters."
        )
    if count_data.count_matrix.shape[1] == 0:
        raise ValueError(
            "Flatmap voxel correlation requires at least one occupied voxel."
        )

    corr = pearson_correlation_from_counts(count_data.count_matrix)
    result = compute_clustermap_data(
        corr,
        count_data.neuron_ids,
        method=method,
        n_clusters=n_clusters,
    )
    result.unassigned_neuron_ids = list(count_data.unassigned_neuron_ids)
    return result, count_data
