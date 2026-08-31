"""Shared atlas coordinate conversion helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def atlas_resolution_array(atlas_or_resolution: Any) -> np.ndarray:
    """Return atlas resolution as a float XYZ array."""
    resolution = getattr(atlas_or_resolution, "resolution", atlas_or_resolution)
    return np.asarray(resolution, dtype=float)


def world_coords_xyz_to_atlas_voxels(
    coords_xyz: np.ndarray,
    atlas_or_resolution: Any,
) -> np.ndarray:
    """Convert world-space XYZ micron coordinates to atlas ZYX voxel indices."""
    resolution = atlas_resolution_array(atlas_or_resolution)
    coords = np.asarray(coords_xyz, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(1, 3)
    lookup_coords = coords[:, [2, 1, 0]]
    return np.round(lookup_coords / resolution).astype(int)


def swc_coords_xyz_to_atlas_voxels(
    coords_xyz: np.ndarray,
    atlas_or_resolution: Any,
) -> np.ndarray:
    """Convert SWC/parquet XYZ micron coordinates to native atlas-grid indices."""
    resolution = atlas_resolution_array(atlas_or_resolution)
    coords = np.asarray(coords_xyz, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(1, 3)
    return np.floor(coords / resolution).astype(int)


def mask_to_world_xyz_bounds(
    mask_volume: np.ndarray,
    atlas_or_resolution: Any,
    padding_voxels: float = 0.5,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return conservative XYZ world bounds for a nonzero atlas mask."""
    nonzero = np.argwhere(np.asarray(mask_volume) > 0)
    if len(nonzero) == 0:
        return None

    resolution = atlas_resolution_array(atlas_or_resolution)
    lower_zyx = np.maximum(nonzero.min(axis=0).astype(float) - padding_voxels, 0.0)
    upper_zyx = nonzero.max(axis=0).astype(float) + padding_voxels
    lower_xyz = lower_zyx[[2, 1, 0]] * resolution
    upper_xyz = upper_zyx[[2, 1, 0]] * resolution
    return lower_xyz, upper_xyz


def mask_to_swc_xyz_bounds(
    mask_volume: np.ndarray,
    atlas_or_resolution: Any,
    padding_voxels: float = 0.5,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return conservative SWC/parquet XYZ world bounds for a nonzero mask."""
    nonzero = np.argwhere(np.asarray(mask_volume) > 0)
    if len(nonzero) == 0:
        return None

    resolution = atlas_resolution_array(atlas_or_resolution)
    lower_xyz = np.maximum(nonzero.min(axis=0).astype(float) - padding_voxels, 0.0)
    upper_xyz = nonzero.max(axis=0).astype(float) + padding_voxels
    return lower_xyz * resolution, upper_xyz * resolution
