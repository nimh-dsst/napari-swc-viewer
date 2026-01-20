"""Hemisphere detection and coordinate flipping using BrainGlobe Atlas API.

This module provides functionality to:
1. Detect which hemisphere SWC coordinates are in using BrainGlobe atlases
2. Flip coordinates from one hemisphere to the other

All coordinate operations use numpy vectorized operations for efficient
processing of large SWC files (10,000+ nodes).
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas

from .swc import SWCData

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Hemisphere(Enum):
    """Brain hemisphere identifiers."""

    LEFT = "left"
    RIGHT = "right"
    MIDLINE = "midline"


def get_atlas_midline(atlas: BrainGlobeAtlas) -> float:
    """Get the midline coordinate (in microns) for an atlas.

    The midline is at the center of the atlas along the
    left-right axis (typically the first axis in BrainGlobe atlases).

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        A BrainGlobe atlas instance.

    Returns
    -------
    float
        The midline coordinate in microns.
    """
    # BrainGlobe atlases have shape (AP, DV, LR) or similar
    # The midline is at the center of the left-right axis
    # Shape is in voxels, resolution converts to microns
    lr_axis = 2  # Left-right is typically the third axis (index 2)
    shape_voxels = atlas.shape[lr_axis]
    resolution_um = atlas.resolution[lr_axis]
    midline_um = (shape_voxels * resolution_um) / 2.0
    return midline_um


def detect_hemisphere(
    coords: NDArray[np.float64],
    atlas: BrainGlobeAtlas | None = None,
    atlas_name: str = "allen_mouse_25um",
    midline: float | None = None,
    coord_axis: int = 0,
) -> Hemisphere:
    """Detect which hemisphere a coordinate or set of coordinates is in.

    Parameters
    ----------
    coords : NDArray[np.float64]
        Coordinates to check. Can be a single point (3,) or multiple points (N, 3).
    atlas : BrainGlobeAtlas, optional
        Pre-loaded atlas instance. If None, will load using atlas_name.
    atlas_name : str, default="allen_mouse_25um"
        Name of the BrainGlobe atlas to use if atlas is not provided.
    midline : float, optional
        Override the atlas midline with a custom value in microns.
    coord_axis : int, default=0
        Which coordinate axis (0=x, 1=y, 2=z) corresponds to the left-right axis.

    Returns
    -------
    Hemisphere
        The hemisphere the coordinates are in (LEFT, RIGHT, or MIDLINE).

    Notes
    -----
    For multiple coordinates, returns the hemisphere of the centroid.
    Coordinates exactly at the midline return MIDLINE.
    """
    coords = np.atleast_2d(coords)

    # Get midline value
    if midline is None:
        if atlas is None:
            atlas = BrainGlobeAtlas(atlas_name)
        midline = get_atlas_midline(atlas)

    # Calculate mean position along left-right axis
    mean_lr = np.mean(coords[:, coord_axis])

    # Determine hemisphere with small tolerance for midline
    tolerance = 1.0  # 1 micron tolerance
    if abs(mean_lr - midline) < tolerance:
        return Hemisphere.MIDLINE
    elif mean_lr < midline:
        return Hemisphere.LEFT
    else:
        return Hemisphere.RIGHT


def detect_soma_hemisphere(
    swc_data: SWCData,
    atlas: BrainGlobeAtlas | None = None,
    atlas_name: str = "allen_mouse_25um",
    midline: float | None = None,
    coord_axis: int = 0,
) -> Hemisphere:
    """Detect which hemisphere the soma of an SWC morphology is in.

    Parameters
    ----------
    swc_data : SWCData
        Parsed SWC morphology data.
    atlas : BrainGlobeAtlas, optional
        Pre-loaded atlas instance.
    atlas_name : str, default="allen_mouse_25um"
        Name of the BrainGlobe atlas to use.
    midline : float, optional
        Override the atlas midline with a custom value.
    coord_axis : int, default=0
        Which coordinate axis corresponds to left-right.

    Returns
    -------
    Hemisphere
        The hemisphere the soma is in.

    Raises
    ------
    ValueError
        If no soma nodes are found in the SWC data.
    """
    soma_coords = swc_data.soma_coords
    if len(soma_coords) == 0:
        raise ValueError("No soma nodes found in SWC data")

    return detect_hemisphere(
        soma_coords,
        atlas=atlas,
        atlas_name=atlas_name,
        midline=midline,
        coord_axis=coord_axis,
    )


def flip_coordinates(
    coords: NDArray[np.float64],
    atlas: BrainGlobeAtlas | None = None,
    atlas_name: str = "allen_mouse_25um",
    midline: float | None = None,
    coord_axis: int = 0,
) -> NDArray[np.float64]:
    """Flip coordinates from one hemisphere to the other.

    This function mirrors coordinates across the atlas midline along the
    specified axis. Uses vectorized numpy operations for efficient processing
    of large coordinate arrays.

    Parameters
    ----------
    coords : NDArray[np.float64]
        Coordinates to flip. Shape (N, 3) or (3,).
    atlas : BrainGlobeAtlas, optional
        Pre-loaded atlas instance.
    atlas_name : str, default="allen_mouse_25um"
        Name of the BrainGlobe atlas to use.
    midline : float, optional
        Override the atlas midline with a custom value.
    coord_axis : int, default=0
        Which coordinate axis (0=x, 1=y, 2=z) to flip across.

    Returns
    -------
    NDArray[np.float64]
        Flipped coordinates with the same shape as input.

    Notes
    -----
    The flip operation is: new_coord = 2 * midline - old_coord
    This mirrors the coordinate across the midline plane.

    This operation is fully vectorized and can efficiently process
    arrays with 10,000+ coordinates.
    """
    coords = np.asarray(coords, dtype=np.float64)
    original_shape = coords.shape
    coords = np.atleast_2d(coords)

    # Get midline value
    if midline is None:
        if atlas is None:
            atlas = BrainGlobeAtlas(atlas_name)
        midline = get_atlas_midline(atlas)

    # Vectorized flip: reflect across midline
    # new_x = midline - (old_x - midline) = 2 * midline - old_x
    flipped = coords.copy()
    flipped[:, coord_axis] = 2.0 * midline - coords[:, coord_axis]

    # Return with original shape
    if len(original_shape) == 1:
        return flipped[0]
    return flipped


def flip_swc(
    swc_data: SWCData,
    atlas: BrainGlobeAtlas | None = None,
    atlas_name: str = "allen_mouse_25um",
    midline: float | None = None,
    coord_axis: int = 0,
    in_place: bool = False,
) -> SWCData:
    """Flip all coordinates in an SWC morphology to the opposite hemisphere.

    Parameters
    ----------
    swc_data : SWCData
        The SWC morphology data to flip.
    atlas : BrainGlobeAtlas, optional
        Pre-loaded atlas instance.
    atlas_name : str, default="allen_mouse_25um"
        Name of the BrainGlobe atlas to use.
    midline : float, optional
        Override the atlas midline with a custom value.
    coord_axis : int, default=0
        Which coordinate axis to flip across.
    in_place : bool, default=False
        If True, modify the input SWCData. If False, return a copy.

    Returns
    -------
    SWCData
        SWC data with flipped coordinates.

    Notes
    -----
    This operation uses vectorized numpy operations and can efficiently
    process SWC files with 10,000+ nodes in a single array operation.
    """
    if in_place:
        result = swc_data
    else:
        result = swc_data.copy()

    # Flip all coordinates in a single vectorized operation
    result.coords = flip_coordinates(
        result.coords,
        atlas=atlas,
        atlas_name=atlas_name,
        midline=midline,
        coord_axis=coord_axis,
    )

    return result


def flip_swc_batch(
    swc_data_list: list[SWCData],
    atlas: BrainGlobeAtlas | None = None,
    atlas_name: str = "allen_mouse_25um",
    midline: float | None = None,
    coord_axis: int = 0,
) -> list[SWCData]:
    """Flip multiple SWC morphologies to the opposite hemisphere.

    This function pre-loads the atlas once and reuses it for all
    morphologies, providing better performance for batch operations.

    Parameters
    ----------
    swc_data_list : list[SWCData]
        List of SWC morphology data to flip.
    atlas : BrainGlobeAtlas, optional
        Pre-loaded atlas instance.
    atlas_name : str, default="allen_mouse_25um"
        Name of the BrainGlobe atlas to use.
    midline : float, optional
        Override the atlas midline with a custom value.
    coord_axis : int, default=0
        Which coordinate axis to flip across.

    Returns
    -------
    list[SWCData]
        List of SWC data with flipped coordinates.
    """
    # Load atlas once for all operations
    if atlas is None and midline is None:
        atlas = BrainGlobeAtlas(atlas_name)
        midline = get_atlas_midline(atlas)

    return [
        flip_swc(
            swc_data,
            atlas=atlas,
            midline=midline,
            coord_axis=coord_axis,
        )
        for swc_data in swc_data_list
    ]
