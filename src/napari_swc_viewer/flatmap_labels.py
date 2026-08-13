"""Region-label overlays in depth-aware flatmap space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .flatmap_heatmap import (
    DEFAULT_FLATMAP_DEPTH_BIN_UM,
    DEFAULT_FLATMAP_Y_BINS,
    DEFAULT_LOOKUP_STATS_CHUNK_VOXELS,
    MAX_FLATMAP_HEATMAP_VOXELS,
    FlatmapLookupStats,
    _bin_flat_values,
    _check_bin_count_inputs,
    _depth_bin_count,
    _flatmap_valid_mask,
    _resolve_axis_bin_counts,
    _spatial_chunk_slices,
    _validate_depth_volume,
    _validate_flatmap_volume,
    _validate_lookup_stats,
    compute_flatmap_lookup_stats,
)


@dataclass(frozen=True, kw_only=True)
class FlatmapRegionLabelsSummary:
    """Counts and grid settings for one flatmap region-label remap.

    Keyword-only so the per-axis bin-count split cannot shift positional
    arguments at a construction site.
    """

    input_voxels: int
    selected_region_count: int
    selected_source_voxels: int
    valid_source_voxels: int
    labeled_voxels: int
    collision_voxels: int
    y_bins: int
    x_bins: int
    depth_bins: int
    depth_bin_um: float
    x_flat_min: float
    x_flat_max: float
    y_flat_min: float
    y_flat_max: float
    depth_min_um: float
    depth_max_um: float
    mirrored_depth_source_voxels: int = 0

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-safe dictionary."""
        return {
            "input_voxels": int(self.input_voxels),
            "selected_region_count": int(self.selected_region_count),
            "selected_source_voxels": int(self.selected_source_voxels),
            "valid_source_voxels": int(self.valid_source_voxels),
            "mirrored_depth_source_voxels": int(self.mirrored_depth_source_voxels),
            "labeled_voxels": int(self.labeled_voxels),
            "collision_voxels": int(self.collision_voxels),
            "y_bins": int(self.y_bins),
            "x_bins": int(self.x_bins),
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


def _validate_label_grid_size(depth_bins: int, y_bins: int, x_bins: int) -> None:
    voxel_count = int(depth_bins * y_bins * x_bins)
    if voxel_count > MAX_FLATMAP_HEATMAP_VOXELS:
        raise ValueError(
            "Flatmap region labels are too large: "
            f"{depth_bins}x{y_bins}x{x_bins} voxels. "
            "Use fewer Y bins or a larger depth bin."
        )


def _fill_missing_depth_from_mirror(
    depth_values: np.ndarray,
    depth_volume: np.ndarray,
    missing_depth: np.ndarray,
    *,
    chunk_slice: slice,
    mirror_coord_axis: int,
) -> int:
    """Fill selected invalid depths from mirrored source voxels in-place."""
    local_coords = np.argwhere(missing_depth)
    if local_coords.size == 0:
        return 0

    global_coords = local_coords.copy()
    global_coords[:, 0] += int(chunk_slice.start or 0)
    global_coords[:, mirror_coord_axis] = (
        int(depth_volume.shape[mirror_coord_axis])
        - 1
        - global_coords[:, mirror_coord_axis]
    )
    mirrored_values = np.asarray(
        depth_volume[tuple(global_coords.T)],
        dtype=float,
    )
    mirrored_valid = np.isfinite(mirrored_values) & (mirrored_values >= 0.0)
    if not mirrored_valid.any():
        return 0

    rescued_local = local_coords[mirrored_valid]
    depth_values[tuple(rescued_local.T)] = mirrored_values[mirrored_valid]
    return int(mirrored_valid.sum())


def build_flatmap_region_label_volume(
    annotation_volume: np.ndarray,
    flatmap_volume: np.ndarray,
    depth_volume: np.ndarray,
    *,
    selected_region_ids: Iterable[int],
    y_bins: int = DEFAULT_FLATMAP_Y_BINS,
    x_bins: int | None = None,
    depth_bin_um: float = DEFAULT_FLATMAP_DEPTH_BIN_UM,
    invalid_zero_sentinel: bool = False,
    invalid_negative_one_sentinel: bool = True,
    lookup_stats: FlatmapLookupStats | None = None,
    lookup_stats_chunk_voxels: int = DEFAULT_LOOKUP_STATS_CHUNK_VOXELS,
    mirror_depth_fallback: bool = True,
    mirror_coord_axis: int = 2,
) -> FlatmapRegionLabelsResult:
    """Build a depth-aware flatmap labels volume from atlas annotations.

    ``x_bins`` defaults to the derived square-bin count.  A caller projecting a
    region mask onto an existing render must pass that render's stored count, so
    the mask and the heatmap share one grid.
    """
    _check_bin_count_inputs(y_bins, x_bins, depth_bin_um)
    if mirror_coord_axis not in (0, 1, 2):
        raise ValueError("mirror_coord_axis must be 0, 1, or 2.")

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

    y_bins, x_bins, depth_bin_um = _resolve_axis_bin_counts(
        x_bounds=lookup_stats.x_bounds,
        y_bounds=lookup_stats.y_bounds,
        y_bins=y_bins,
        x_bins=x_bins,
        depth_bin_um=depth_bin_um,
    )
    depth_bins = _depth_bin_count(lookup_stats.depth_range_um, depth_bin_um)
    _validate_label_grid_size(depth_bins, y_bins, x_bins)
    output_shape = (depth_bins, y_bins, x_bins)

    packed_chunks: list[np.ndarray] = []
    count_chunks: list[np.ndarray] = []
    selected_source_voxels = 0
    valid_source_voxels = 0
    mirrored_depth_source_voxels = 0

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
        if mirror_depth_fallback:
            mirrored_depth_source_voxels += _fill_missing_depth_from_mirror(
                depth_values,
                depth,
                selected & flat_valid & ~depth_valid,
                chunk_slice=chunk_slice,
                mirror_coord_axis=mirror_coord_axis,
            )
            depth_valid = np.isfinite(depth_values) & (depth_values >= 0.0)
        valid = selected & flat_valid & depth_valid
        if not valid.any():
            continue

        valid_source_voxels += int(valid.sum())
        x_bin_indices = _bin_flat_values(
            flat_xy[..., 0][valid],
            lookup_stats.x_bounds,
            x_bins,
        )
        y_bin_indices = _bin_flat_values(
            flat_xy[..., 1][valid],
            lookup_stats.y_bounds,
            y_bins,
        )
        depth_bin_indices = np.floor(
            (depth_values[valid] - lookup_stats.depth_range_um[0]) / depth_bin_um
        ).astype(np.int64)
        depth_bin_indices = np.clip(depth_bin_indices, 0, depth_bins - 1)
        linear_bins = (
            (depth_bin_indices * y_bins * x_bins)
            + (y_bin_indices * x_bins)
            + x_bin_indices
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
        mirrored_depth_source_voxels=int(mirrored_depth_source_voxels),
        labeled_voxels=int(np.count_nonzero(labels)),
        collision_voxels=int(collision_voxels),
        y_bins=int(y_bins),
        x_bins=int(x_bins),
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
