"""Region mask extraction and dilation using BrainGlobe Atlas API.

Ported from swc-mapper/create_dilated_gpe_ids.py, replacing Allen SDK
with BrainGlobe for mask extraction. The dilation algorithm (EDT + binary
search) is preserved verbatim.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def smooth_heatmap_volume(
    volume: NDArray[np.floating] | np.ndarray,
    sigma: float = 1.0,
) -> NDArray[np.float32]:
    """Smooth a 3D heatmap volume with a Gaussian kernel."""
    arr = np.asarray(volume, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {arr.shape}")
    if sigma <= 0:
        return arr.copy()
    return gaussian_filter(arr, sigma=float(sigma), mode="nearest").astype(np.float32)


def merge_heatmap_volumes(
    volumes: list[NDArray[np.floating] | np.ndarray],
) -> NDArray[np.float32]:
    """Sum multiple heatmap volumes voxelwise."""
    if not volumes:
        raise ValueError("At least one heatmap volume is required.")

    arrays = [np.asarray(volume, dtype=np.float32) for volume in volumes]
    first_shape = arrays[0].shape
    if any(arr.ndim != 3 for arr in arrays):
        raise ValueError("All heatmap volumes must be 3D.")
    if any(arr.shape != first_shape for arr in arrays[1:]):
        raise ValueError("All heatmap volumes must have the same shape.")
    return np.sum(arrays, axis=0, dtype=np.float32)


def isolate_heatmap_volume_to_region_ids(
    volume: NDArray[np.floating] | np.ndarray,
    atlas: BrainGlobeAtlas,
    region_ids: list[int] | tuple[int, ...],
) -> NDArray[np.float32]:
    """Return a heatmap copy with values outside selected atlas regions set to zero."""
    arr = np.asarray(volume, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {arr.shape}")

    selected_ids = [int(region_id) for region_id in region_ids]
    if not selected_ids:
        raise ValueError("At least one region ID is required.")

    annotation = np.asarray(atlas.annotation)
    if annotation.shape != arr.shape:
        raise ValueError(
            "Heatmap shape does not match atlas annotation shape: "
            f"{arr.shape} != {annotation.shape}"
        )

    region_mask = np.isin(annotation, selected_ids)
    isolated = np.zeros_like(arr, dtype=np.float32)
    isolated[region_mask] = arr[region_mask]
    return isolated


def otsu_threshold_positive(
    volume: NDArray[np.floating] | np.ndarray,
    bins: int = 256,
) -> float:
    """Compute an Otsu threshold using only positive voxels."""
    positive = np.asarray(volume, dtype=np.float32)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if positive.size == 0:
        return 0.0

    min_value = float(positive.min())
    max_value = float(positive.max())
    if max_value <= min_value:
        return min_value

    hist, bin_edges = np.histogram(positive, bins=min(int(bins), int(positive.size)))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    weight1 = np.cumsum(hist)
    weight2 = hist.sum() - weight1
    weighted = hist * bin_centers
    mean1 = np.divide(
        np.cumsum(weighted),
        weight1,
        out=np.zeros_like(bin_centers),
        where=weight1 > 0,
    )
    mean2 = np.divide(
        np.cumsum(weighted[::-1])[::-1],
        weight2,
        out=np.zeros_like(bin_centers),
        where=weight2 > 0,
    )
    between = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    if between.size == 0:
        return min_value
    return float(bin_centers[int(np.argmax(between))])


def build_binary_mask_from_heatmap(
    volume: NDArray[np.floating] | np.ndarray,
    sigma: float = 1.0,
    threshold_mode: str = "otsu",
    manual_threshold: float | None = None,
) -> tuple[NDArray[np.uint8], float, NDArray[np.float32]]:
    """Create a binary mask from a heatmap via smoothing and thresholding."""
    smoothed = smooth_heatmap_volume(volume, sigma=sigma)
    mode = str(threshold_mode).strip().lower()
    if mode == "manual":
        if manual_threshold is None:
            raise ValueError("Manual threshold mode requires a threshold value.")
        threshold = float(manual_threshold)
    elif mode == "otsu":
        threshold = otsu_threshold_positive(smoothed)
    else:
        raise ValueError(f"Unknown threshold mode: {threshold_mode}")

    if mode == "otsu" and not np.any(smoothed > 0):
        mask = np.zeros(smoothed.shape, dtype=np.uint8)
    else:
        mask = build_binary_mask_from_threshold_range(
            smoothed,
            lower_threshold=threshold,
        )
    return mask, threshold, smoothed


def build_binary_mask_from_threshold_range(
    volume: NDArray[np.floating] | np.ndarray,
    lower_threshold: float,
    upper_threshold: float | None = None,
) -> NDArray[np.uint8]:
    """Create a binary mask from explicit lower and optional upper bounds."""
    arr = np.asarray(volume, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {arr.shape}")

    lower = float(lower_threshold)
    upper = None if upper_threshold is None else float(upper_threshold)
    if upper is not None and upper < lower:
        raise ValueError("Upper threshold must be greater than or equal to lower threshold.")

    mask = arr >= lower
    if upper is not None:
        mask &= arr <= upper
    return mask.astype(np.uint8, copy=False)


def get_region_mask(atlas: BrainGlobeAtlas, acronym: str) -> NDArray[np.bool_]:
    """Get binary mask for a brain region using BrainGlobe Atlas API.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        A loaded BrainGlobe atlas instance.
    acronym : str
        Region acronym (e.g., "GPe", "CP").

    Returns
    -------
    NDArray[np.bool_]
        3D boolean mask where True indicates voxels belonging to the region
        (including all sub-structures).
    """
    logger.info(f"Extracting mask for region '{acronym}' from atlas")
    raw_mask = atlas.get_structure_mask(acronym)
    mask = raw_mask > 0
    logger.info(
        f"Region '{acronym}' mask: {mask.sum():,} voxels, "
        f"shape {mask.shape}"
    )
    return mask


def get_regions_mask(
    atlas: BrainGlobeAtlas,
    acronyms: list[str],
) -> NDArray[np.bool_]:
    """Return the voxelwise union of one or more selected atlas regions."""
    selected = [str(acronym).strip() for acronym in acronyms if str(acronym).strip()]
    if not selected:
        raise ValueError("At least one region acronym is required.")

    masks = [get_region_mask(atlas, acronym) for acronym in selected]
    combined = np.logical_or.reduce(masks)
    logger.info(
        "Combined %d region mask(s): %s -> %,d voxels",
        len(selected),
        ", ".join(selected),
        int(combined.sum()),
    )
    return combined


def dilate_mask_to_volume_increase(
    mask: NDArray[np.bool_],
    increase_fraction: float = 0.20,
    voxel_spacing_um: tuple[float, ...] = (10.0, 10.0, 10.0),
    tol_fraction: float = 0.001,
    max_iters: int = 60,
) -> NDArray[np.bool_]:
    """Dilate a 3D binary mask so its volume increases by a target fraction.

    Uses the Euclidean distance transform (EDT) on the background, then
    binary search for the distance threshold that yields the desired
    volume increase.

    Ported verbatim from swc-mapper/create_dilated_gpe_ids.py.

    Parameters
    ----------
    mask : NDArray[np.bool_]
        3D binary mask. Shape: (Z, Y, X) or any consistent 3D order.
    increase_fraction : float
        Desired fractional volume increase (e.g., 0.20 => +20%).
    voxel_spacing_um : tuple of float
        Physical voxel spacing for EDT sampling. For Allen 10um: (10,10,10).
    tol_fraction : float
        Acceptable relative error on the target volume (0.001 = 0.1%).
    max_iters : int
        Maximum binary search iterations.

    Returns
    -------
    NDArray[np.bool_]
        Dilated binary mask, same shape as input.
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}")

    m = mask.astype(bool, copy=False)
    v0 = int(m.sum())
    if v0 == 0:
        raise ValueError("Mask is empty (volume=0); cannot dilate.")

    target = v0 * (1.0 + float(increase_fraction))
    target_lo = target * (1.0 - float(tol_fraction))
    target_hi = target * (1.0 + float(tol_fraction))

    logger.info(
        f"Dilating mask: {v0:,} voxels, target +{increase_fraction*100:.0f}% "
        f"= {int(target):,} voxels"
    )

    # EDT on background: distance from each background voxel to nearest object
    bg = ~m
    dist_bg = distance_transform_edt(bg, sampling=voxel_spacing_um)

    def vol_at(t_um: float) -> int:
        return int((m | (dist_bg <= t_um)).sum())

    # Bracket the upper bound
    t_lo = 0.0
    t_hi = max(voxel_spacing_um)
    v_hi = vol_at(t_hi)

    expand_steps = 0
    while v_hi < target and expand_steps < 50:
        t_hi *= 2.0
        v_hi = vol_at(t_hi)
        expand_steps += 1

    if v_hi < target:
        raise RuntimeError(
            "Could not reach target volume increase. "
            "Mask may be too large relative to the image bounds."
        )

    # Binary search for threshold
    best_t = t_hi
    best_err = abs(v_hi - target)

    for _ in range(max_iters):
        t_mid = 0.5 * (t_lo + t_hi)
        v_mid = vol_at(t_mid)

        err = abs(v_mid - target)
        if err < best_err:
            best_err = err
            best_t = t_mid

        if target_lo <= v_mid <= target_hi:
            best_t = t_mid
            break

        if v_mid < target:
            t_lo = t_mid
        else:
            t_hi = t_mid

    new_mask = (m | (dist_bg <= best_t)).astype(bool)
    logger.info(
        f"Dilation complete: {new_mask.sum():,} voxels "
        f"(+{(new_mask.sum() / v0 - 1) * 100:.1f}%)"
    )
    return new_mask


def get_expanded_region_voxel_ids_for_regions(
    atlas: BrainGlobeAtlas,
    acronyms: list[str],
    increase_fraction: float = 0.2,
) -> NDArray[np.int32]:
    """Create a voxel ID map for the union of selected regions."""
    mask = get_regions_mask(atlas, acronyms)
    resolution = tuple(float(r) for r in atlas.resolution)

    if increase_fraction > 0:
        exp_mask = dilate_mask_to_volume_increase(
            mask,
            increase_fraction=increase_fraction,
            voxel_spacing_um=resolution,
        )
    else:
        exp_mask = mask

    id_map = np.full(exp_mask.shape, -1, dtype=np.int32)
    id_map[exp_mask] = np.arange(exp_mask.sum(), dtype=np.int32)

    logger.info(
        "Voxel ID map for %s (+%.0f%%): %,d voxels with IDs",
        ", ".join(str(acronym) for acronym in acronyms),
        increase_fraction * 100.0,
        int(exp_mask.sum()),
    )
    return id_map


def get_expanded_region_voxel_ids(
    atlas: BrainGlobeAtlas,
    acronym: str,
    increase_fraction: float = 0.2,
) -> NDArray[np.int32]:
    """Create a voxel ID map for an expanded brain region.

    Extracts the region mask, dilates it, then assigns sequential integer
    IDs to each voxel inside the expanded region.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        A loaded BrainGlobe atlas instance.
    acronym : str
        Region acronym (e.g., "GPe", "CP").
    increase_fraction : float
        Fractional volume increase for dilation (e.g., 0.2 = +20%).

    Returns
    -------
    NDArray[np.int32]
        3D array where expanded-region voxels have sequential IDs (0, 1, 2, ...)
        and voxels outside the expanded region have value -1.
    """
    return get_expanded_region_voxel_ids_for_regions(
        atlas,
        [acronym],
        increase_fraction=increase_fraction,
    )
