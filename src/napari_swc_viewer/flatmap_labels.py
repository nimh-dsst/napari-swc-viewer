"""Region-label overlays in depth-aware flatmap space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .flatmap_heatmap import (
    DEFAULT_FLATMAP_DEPTH_BIN_UM,
    DEFAULT_FLATMAP_XY_BINS,
    DEFAULT_LOOKUP_STATS_CHUNK_VOXELS,
    MAX_FLATMAP_HEATMAP_VOXELS,
    FlatmapLookupStats,
    _bin_flat_values,
    _depth_bin_count,
    _flatmap_valid_mask,
    _spatial_chunk_slices,
    _validate_depth_volume,
    _validate_flatmap_volume,
    _validate_lookup_stats,
    compute_flatmap_lookup_stats,
)


@dataclass(frozen=True)
class FlatmapRegionLabelsSummary:
    """Counts and grid settings for one flatmap region-label remap."""

    input_voxels: int
    selected_region_count: int
    selected_source_voxels: int
    valid_source_voxels: int
    labeled_voxels: int
    collision_voxels: int
    xy_bins: int
    depth_bins: int
    depth_bin_um: float
    x_flat_min: float
    x_flat_max: float
    y_flat_min: float
    y_flat_max: float
    depth_min_um: float
    depth_max_um: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-safe dictionary."""
        return {
            "input_voxels": int(self.input_voxels),
            "selected_region_count": int(self.selected_region_count),
            "selected_source_voxels": int(self.selected_source_voxels),
            "valid_source_voxels": int(self.valid_source_voxels),
            "labeled_voxels": int(self.labeled_voxels),
            "collision_voxels": int(self.collision_voxels),
            "xy_bins": int(self.xy_bins),
            "depth_bins": int(self.depth_bins),
            "depth_bin_um": float(self.depth_bin_um),
            "x_flat_min": float(self.x_flat_min),
            "x_flat_max": float(self.x_flat_max),
            "y_flat_min": float(self.y_flat_min),
            "y_flat_max": float(self.y_flat_max),
            "depth_min_um": float(self.depth_min_um),
            "depth_max_um": float(self.depth_max_um),
        }


@dataclass(frozen=True)
class FlatmapRegionLabelsResult:
    """Region-label volume and metadata for a flatmap overlay."""

    labels: np.ndarray
    summary: FlatmapRegionLabelsSummary
    selected_region_ids: list[int]
    represented_region_ids: list[int]


def _normalise_selected_region_ids(region_ids: Iterable[int]) -> np.ndarray:
    ids = sorted({int(region_id) for region_id in region_ids if int(region_id) > 0})
    if not ids:
        raise ValueError("Select at least one atlas region before creating labels.")
    return np.asarray(ids, dtype=np.int32)


def _pack_pairs(linear_bins: np.ndarray, region_ids: np.ndarray) -> np.ndarray:
    bins = np.asarray(linear_bins, dtype=np.uint64)
    ids = np.asarray(region_ids, dtype=np.uint64) & np.uint64(0xFFFFFFFF)
    return (bins << np.uint64(32)) | ids


def _unpack_pair_bins(packed: np.ndarray) -> np.ndarray:
    return (np.asarray(packed, dtype=np.uint64) >> np.uint64(32)).astype(np.int64)


def _unpack_pair_region_ids(packed: np.ndarray) -> np.ndarray:
    return (np.asarray(packed, dtype=np.uint64) & np.uint64(0xFFFFFFFF)).astype(
        np.int32
    )


def _merge_pair_counts(
    packed_chunks: list[np.ndarray],
    count_chunks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if not packed_chunks:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int64)

    packed = np.concatenate(packed_chunks)
    counts = np.concatenate(count_chunks).astype(np.int64, copy=False)
    unique_pairs, inverse = np.unique(packed, return_inverse=True)
    merged_counts = np.zeros(len(unique_pairs), dtype=np.int64)
    np.add.at(merged_counts, inverse, counts)
    return unique_pairs, merged_counts


def _choose_majority_labels(
    pair_keys: np.ndarray,
    pair_counts: np.ndarray,
    *,
    output_shape: tuple[int, int, int],
) -> tuple[np.ndarray, int]:
    labels = np.zeros(output_shape, dtype=np.int32)
    if pair_keys.size == 0:
        return labels, 0

    linear_bins = _unpack_pair_bins(pair_keys)
    region_ids = _unpack_pair_region_ids(pair_keys)
    counts = np.asarray(pair_counts, dtype=np.int64)

    order = np.lexsort((region_ids, -counts, linear_bins))
    sorted_bins = linear_bins[order]
    sorted_ids = region_ids[order]
    _unique_bins, first_indices, region_counts = np.unique(
        sorted_bins,
        return_index=True,
        return_counts=True,
    )
    winner_bins = sorted_bins[first_indices]
    winner_ids = sorted_ids[first_indices]
    labels.reshape(-1)[winner_bins] = winner_ids
    collision_voxels = int((region_counts > 1).sum())
    return labels, collision_voxels


def _validate_label_grid_size(depth_bins: int, xy_bins: int) -> None:
    voxel_count = int(depth_bins * xy_bins * xy_bins)
    if voxel_count > MAX_FLATMAP_HEATMAP_VOXELS:
        raise ValueError(
            "Flatmap region labels are too large: "
            f"{depth_bins}x{xy_bins}x{xy_bins} voxels. "
            "Use fewer XY bins or a larger depth bin."
        )


def build_flatmap_region_label_volume(
    annotation_volume: np.ndarray,
    flatmap_volume: np.ndarray,
    depth_volume: np.ndarray,
    *,
    selected_region_ids: Iterable[int],
    xy_bins: int = DEFAULT_FLATMAP_XY_BINS,
    depth_bin_um: float = DEFAULT_FLATMAP_DEPTH_BIN_UM,
    invalid_zero_sentinel: bool = False,
    invalid_negative_one_sentinel: bool = True,
    lookup_stats: FlatmapLookupStats | None = None,
    lookup_stats_chunk_voxels: int = DEFAULT_LOOKUP_STATS_CHUNK_VOXELS,
) -> FlatmapRegionLabelsResult:
    """Build a depth-aware flatmap labels volume from atlas annotations."""
    xy_bins = int(xy_bins)
    depth_bin_um = float(depth_bin_um)
    if xy_bins <= 0:
        raise ValueError("xy_bins must be positive.")
    if depth_bin_um <= 0.0:
        raise ValueError("depth_bin_um must be positive.")

    annotation = np.asarray(annotation_volume)
    flatmap = _validate_flatmap_volume(flatmap_volume)
    depth = _validate_depth_volume(depth_volume)
    if annotation.shape != flatmap.shape[:3]:
        raise ValueError(
            "Atlas annotation shape must match the flatmap/depth lookup grid; "
            f"got annotation {annotation.shape} and lookup grid {flatmap.shape[:3]}."
        )
    if depth.shape != flatmap.shape[:3]:
        raise ValueError(
            "depth_volume shape must match the flatmap lookup grid; "
            f"got depth {depth.shape} and flatmap grid {flatmap.shape[:3]}."
        )

    selected_ids = _normalise_selected_region_ids(selected_region_ids)
    if lookup_stats is None:
        lookup_stats = compute_flatmap_lookup_stats(
            flatmap,
            depth,
            invalid_zero_sentinel=invalid_zero_sentinel,
            invalid_negative_one_sentinel=invalid_negative_one_sentinel,
            chunk_voxels=lookup_stats_chunk_voxels,
        )
    else:
        _validate_lookup_stats(
            lookup_stats,
            flatmap,
            depth,
            invalid_zero_sentinel=invalid_zero_sentinel,
            invalid_negative_one_sentinel=invalid_negative_one_sentinel,
        )

    depth_bins = _depth_bin_count(lookup_stats.depth_range_um, depth_bin_um)
    _validate_label_grid_size(depth_bins, xy_bins)
    output_shape = (depth_bins, xy_bins, xy_bins)

    packed_chunks: list[np.ndarray] = []
    count_chunks: list[np.ndarray] = []
    selected_source_voxels = 0
    valid_source_voxels = 0

    for chunk_slice in _spatial_chunk_slices(
        annotation.shape,
        chunk_voxels=lookup_stats_chunk_voxels,
    ):
        annotation_chunk = annotation[chunk_slice]
        selected = np.isin(annotation_chunk, selected_ids)
        if not selected.any():
            continue

        selected_source_voxels += int(selected.sum())
        flat_xy = np.asarray(flatmap[chunk_slice], dtype=float)
        depth_values = np.asarray(depth[chunk_slice], dtype=float)
        flat_valid = _flatmap_valid_mask(
            flat_xy.reshape(-1, 2),
            invalid_zero_sentinel=invalid_zero_sentinel,
            invalid_negative_one_sentinel=invalid_negative_one_sentinel,
        ).reshape(annotation_chunk.shape)
        depth_valid = np.isfinite(depth_values) & (depth_values >= 0.0)
        valid = selected & flat_valid & depth_valid
        if not valid.any():
            continue

        valid_source_voxels += int(valid.sum())
        x_bins = _bin_flat_values(
            flat_xy[..., 0][valid],
            lookup_stats.x_bounds,
            xy_bins,
        )
        y_bins = _bin_flat_values(
            flat_xy[..., 1][valid],
            lookup_stats.y_bounds,
            xy_bins,
        )
        depth_bins_for_voxels = np.floor(
            (depth_values[valid] - lookup_stats.depth_range_um[0]) / depth_bin_um
        ).astype(np.int64)
        depth_bins_for_voxels = np.clip(depth_bins_for_voxels, 0, depth_bins - 1)
        linear_bins = (
            (depth_bins_for_voxels * xy_bins * xy_bins)
            + (y_bins * xy_bins)
            + x_bins
        )
        region_ids = annotation_chunk[valid].astype(np.int32, copy=False)
        packed = _pack_pairs(linear_bins, region_ids)
        unique_pairs, counts = np.unique(packed, return_counts=True)
        packed_chunks.append(unique_pairs)
        count_chunks.append(counts.astype(np.int64, copy=False))

    pair_keys, pair_counts = _merge_pair_counts(packed_chunks, count_chunks)
    labels, collision_voxels = _choose_majority_labels(
        pair_keys,
        pair_counts,
        output_shape=output_shape,
    )
    represented_ids = sorted(
        int(region_id) for region_id in np.unique(labels) if int(region_id) > 0
    )
    summary = FlatmapRegionLabelsSummary(
        input_voxels=int(annotation.size),
        selected_region_count=int(len(selected_ids)),
        selected_source_voxels=int(selected_source_voxels),
        valid_source_voxels=int(valid_source_voxels),
        labeled_voxels=int(np.count_nonzero(labels)),
        collision_voxels=int(collision_voxels),
        xy_bins=int(xy_bins),
        depth_bins=int(depth_bins),
        depth_bin_um=float(depth_bin_um),
        x_flat_min=float(lookup_stats.x_bounds[0]),
        x_flat_max=float(lookup_stats.x_bounds[1]),
        y_flat_min=float(lookup_stats.y_bounds[0]),
        y_flat_max=float(lookup_stats.y_bounds[1]),
        depth_min_um=float(lookup_stats.depth_range_um[0]),
        depth_max_um=float(lookup_stats.depth_range_um[1]),
    )
    return FlatmapRegionLabelsResult(
        labels=labels,
        summary=summary,
        selected_region_ids=[int(region_id) for region_id in selected_ids.tolist()],
        represented_region_ids=represented_ids,
    )
