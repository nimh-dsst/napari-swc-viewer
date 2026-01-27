"""Allen SDK setup and coordinate-to-region mapping.

This module provides functionality to:
1. Set up Allen SDK ReferenceSpaceCache for accessing annotation volumes
2. Map coordinates to brain regions using the Allen CCF
3. Get hierarchical region information from the structure tree

Note: This module requires the allensdk package, which can be installed with:
    pip install allensdk

Due to dependency conflicts with scipy, allensdk is an optional dependency.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Check if allensdk is available
try:
    from allensdk.core.reference_space_cache import ReferenceSpaceCache

    ALLENSDK_AVAILABLE = True
except ImportError:
    ALLENSDK_AVAILABLE = False
    ReferenceSpaceCache = None

# Default cache directory for Allen SDK data
DEFAULT_CACHE_DIR = Path.home() / ".allen_sdk_cache"

# Allen CCF resolution options in microns
RESOLUTIONS = [10, 25, 50, 100]


def setup_allen_sdk(
    resolution: int = 25,
    cache_dir: Path | str | None = None,
) -> tuple:
    """Set up Allen SDK ReferenceSpaceCache and load annotation volume.

    Parameters
    ----------
    resolution : int, default=25
        Resolution in microns. Must be one of [10, 25, 50, 100].
    cache_dir : Path or str, optional
        Directory to cache downloaded data. Defaults to ~/.allen_sdk_cache.

    Returns
    -------
    tuple
        (rsp, annotation_volume, structure_tree) where:
        - rsp: ReferenceSpaceCache instance
        - annotation_volume: 3D array of region IDs
        - structure_tree: StructureTree for region metadata

    Raises
    ------
    ImportError
        If allensdk is not installed.
    """
    if not ALLENSDK_AVAILABLE:
        raise ImportError(
            "allensdk is required for region mapping. "
            "Install it with: pip install allensdk"
        )

    if resolution not in RESOLUTIONS:
        raise ValueError(f"Resolution must be one of {RESOLUTIONS}, got {resolution}")

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create the reference space cache
    rsp = ReferenceSpaceCache(
        resolution,
        reference_space_key="annotation/ccf_2017",
        manifest=str(cache_dir / "manifest.json"),
    )

    # Download and load annotation volume
    logger.info(f"Loading annotation volume at {resolution}um resolution...")
    annotation_volume, meta = rsp.get_annotation_volume()

    # Get structure tree for region names
    structure_tree = rsp.get_structure_tree()

    logger.info(
        f"Annotation volume shape: {annotation_volume.shape}, "
        f"Structure tree loaded with {len(structure_tree.get_structures_by_set_id([]))} structures"
    )

    return rsp, annotation_volume, structure_tree


@lru_cache(maxsize=1)
def _get_cached_allen_data(
    resolution: int = 25,
    cache_dir: str | None = None,
) -> tuple:
    """Get cached Allen SDK data. Returns same tuple as setup_allen_sdk."""
    return setup_allen_sdk(resolution, cache_dir)


def get_region_at_coords(
    coords: NDArray[np.float64] | tuple[float, float, float],
    annotation_volume: NDArray[np.int32],
    structure_tree,
    resolution: int = 25,
) -> dict | None:
    """Get brain region information for a single coordinate.

    Parameters
    ----------
    coords : array-like
        Coordinate in microns (x, y, z) or PIR format matching the atlas.
    annotation_volume : NDArray[np.int32]
        3D annotation volume from Allen SDK.
    structure_tree : StructureTree
        Allen SDK structure tree for region metadata.
    resolution : int, default=25
        Resolution of the annotation volume in microns.

    Returns
    -------
    dict or None
        Region info with keys: id, name, acronym, structure_id_path, color_hex_triplet
        Returns None if coordinate is outside the brain.
    """
    # Convert microns to voxel indices
    coords = np.asarray(coords)
    voxel_coords = np.round(coords / resolution).astype(int)

    # Check bounds
    if not all(0 <= voxel_coords[i] < annotation_volume.shape[i] for i in range(3)):
        return None

    # Get region ID from annotation volume
    region_id = annotation_volume[voxel_coords[0], voxel_coords[1], voxel_coords[2]]

    if region_id == 0:
        return None  # Outside brain

    # Get region info from structure tree
    structure = structure_tree.get_structures_by_id([region_id])
    if not structure:
        return None

    structure = structure[0]
    return {
        "id": region_id,
        "name": structure["name"],
        "acronym": structure["acronym"],
        "structure_id_path": structure["structure_id_path"],
        "color_hex_triplet": structure.get("color_hex_triplet", ""),
    }


def get_regions_for_coords(
    coords: NDArray[np.float64],
    annotation_volume: NDArray[np.int32],
    structure_tree,
    resolution: int = 25,
) -> list[dict | None]:
    """Get brain region information for multiple coordinates.

    Parameters
    ----------
    coords : NDArray[np.float64]
        Array of coordinates in microns, shape (N, 3).
    annotation_volume : NDArray[np.int32]
        3D annotation volume from Allen SDK.
    structure_tree : StructureTree
        Allen SDK structure tree for region metadata.
    resolution : int, default=25
        Resolution of the annotation volume in microns.

    Returns
    -------
    list[dict | None]
        List of region info dicts, one per coordinate.
    """
    coords = np.atleast_2d(coords)
    return [
        get_region_at_coords(coord, annotation_volume, structure_tree, resolution)
        for coord in coords
    ]


def get_region_ids_vectorized(
    coords: NDArray[np.float64],
    annotation_volume: NDArray[np.int32],
    resolution: int = 25,
) -> NDArray[np.int32]:
    """Get region IDs for multiple coordinates using vectorized operations.

    Parameters
    ----------
    coords : NDArray[np.float64]
        Array of coordinates in microns, shape (N, 3).
    annotation_volume : NDArray[np.int32]
        3D annotation volume from Allen SDK.
    resolution : int, default=25
        Resolution of the annotation volume in microns.

    Returns
    -------
    NDArray[np.int32]
        Array of region IDs, 0 for coordinates outside the brain.
    """
    coords = np.atleast_2d(coords)
    n_coords = len(coords)

    # Convert microns to voxel indices
    voxel_coords = np.round(coords / resolution).astype(int)

    # Initialize result with zeros (outside brain)
    region_ids = np.zeros(n_coords, dtype=np.int32)

    # Check bounds for all coordinates
    in_bounds = np.all(
        (voxel_coords >= 0) & (voxel_coords < np.array(annotation_volume.shape)),
        axis=1,
    )

    # Get region IDs for in-bounds coordinates
    valid_voxels = voxel_coords[in_bounds]
    if len(valid_voxels) > 0:
        region_ids[in_bounds] = annotation_volume[
            valid_voxels[:, 0], valid_voxels[:, 1], valid_voxels[:, 2]
        ]

    return region_ids


def build_region_lookup(structure_tree) -> dict[int, dict]:
    """Build a lookup table from region ID to region info.

    Parameters
    ----------
    structure_tree : StructureTree
        Allen SDK structure tree.

    Returns
    -------
    dict[int, dict]
        Mapping from region ID to dict with name, acronym, etc.
    """
    all_structures = structure_tree.get_structures_by_set_id([])
    return {
        s["id"]: {
            "id": s["id"],
            "name": s["name"],
            "acronym": s["acronym"],
            "structure_id_path": s["structure_id_path"],
            "color_hex_triplet": s.get("color_hex_triplet", ""),
            "parent_structure_id": s.get("parent_structure_id"),
        }
        for s in all_structures
    }


def get_region_hierarchy(structure_tree) -> dict[int, list[int]]:
    """Get the full hierarchy of regions.

    Parameters
    ----------
    structure_tree : StructureTree
        Allen SDK structure tree.

    Returns
    -------
    dict[int, list[int]]
        Mapping from region ID to list of child region IDs.
    """
    all_structures = structure_tree.get_structures_by_set_id([])
    children: dict[int, list[int]] = {}

    for s in all_structures:
        parent_id = s.get("parent_structure_id")
        if parent_id is not None:
            if parent_id not in children:
                children[parent_id] = []
            children[parent_id].append(s["id"])

    return children


def get_all_descendant_ids(
    structure_tree,
    region_id: int,
) -> set[int]:
    """Get all descendant region IDs for a given region.

    Parameters
    ----------
    structure_tree : StructureTree
        Allen SDK structure tree.
    region_id : int
        The region ID to get descendants for.

    Returns
    -------
    set[int]
        Set of all descendant region IDs, including the input region.
    """
    descendants = {region_id}
    children = structure_tree.child_ids([region_id])[0]

    for child_id in children:
        descendants.update(get_all_descendant_ids(structure_tree, child_id))

    return descendants


def get_structures_with_ancestors(
    structure_tree,
    region_ids: list[int] | set[int],
) -> dict[int, list[dict]]:
    """Get structure info with full ancestor path for multiple regions.

    Parameters
    ----------
    structure_tree : StructureTree
        Allen SDK structure tree.
    region_ids : list or set of int
        Region IDs to get ancestry for.

    Returns
    -------
    dict[int, list[dict]]
        Mapping from region ID to list of ancestor structures
        (from root to leaf).
    """
    all_structures = structure_tree.get_structures_by_id(list(region_ids))
    lookup = {s["id"]: s for s in all_structures}

    result = {}
    for region_id in region_ids:
        if region_id not in lookup:
            continue
        structure = lookup[region_id]
        path_ids = structure.get("structure_id_path", [])
        ancestors = structure_tree.get_structures_by_id(path_ids)
        result[region_id] = sorted(ancestors, key=lambda x: len(x["structure_id_path"]))

    return result
