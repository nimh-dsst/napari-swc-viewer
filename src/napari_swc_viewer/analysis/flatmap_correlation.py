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
