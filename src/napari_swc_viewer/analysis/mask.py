"""Region mask extraction and dilation using BrainGlobe Atlas API.

Ported from swc-mapper/create_dilated_gpe_ids.py, replacing Allen SDK
with BrainGlobe for mask extraction. The dilation algorithm (EDT + binary
search) is preserved verbatim.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import distance_transform_edt

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


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
    mask = get_region_mask(atlas, acronym)
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
        f"Voxel ID map for '{acronym}' (+{increase_fraction*100:.0f}%): "
        f"{exp_mask.sum():,} voxels with IDs"
    )
    return id_map
