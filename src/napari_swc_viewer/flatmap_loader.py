"""NRRD loading helpers for isocortex flatmap projection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FLATMAP_STYLE_FILENAMES: dict[str, str] = {
    "both_shaped": "flatmap_both_shaped.nrrd",
    "both_square": "flatmap_both_square.nrrd",
    "single_shaped": "flatmap_shaped.nrrd",
    "single_square": "flatmap_square.nrrd",
}


@dataclass(frozen=True)
class FlatmapVolumeSet:
    """Loaded flatmap lookup and depth volumes with source metadata."""

    flatmap: np.ndarray
    depth: np.ndarray
    flatmap_header: dict[str, Any]
    depth_header: dict[str, Any]
    flatmap_path: Path
    depth_path: Path
    space_directions: np.ndarray | None = None
    space_origin: np.ndarray | None = None


def _read_nrrd(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read one NRRD file as a NumPy array and metadata dictionary."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"NRRD file not found: {source}")

    try:
        import nrrd
    except ImportError as exc:  # pragma: no cover - exercised without dependency
        raise RuntimeError(
            "pynrrd is required to read flatmap NRRD files. "
            "Install the project dependencies with pixi."
        ) from exc

    data, header = nrrd.read(str(source))
    return np.asarray(data), dict(header)


def normalize_flatmap_volume(data: np.ndarray) -> np.ndarray:
    """Return flatmap data normalized to ``(nx, ny, nz, 2)`` float64."""
    array = np.asarray(data)
    if array.ndim != 4:
        raise ValueError(
            "Flatmap NRRD must be a 4D volume with one coordinate axis of length 2; "
            f"got shape {array.shape}."
        )

    if array.shape[-1] == 2:
        normalized = array
    elif array.shape[0] == 2:
        normalized = np.moveaxis(array, 0, -1)
    else:
        coordinate_axes = [axis for axis, size in enumerate(array.shape) if size == 2]
        if len(coordinate_axes) != 1:
            raise ValueError(
                "Flatmap NRRD must have exactly one coordinate axis of length 2 "
                "when the coordinate axis is not first or last; "
                f"got shape {array.shape}."
            )
        normalized = np.moveaxis(array, coordinate_axes[0], -1)

    if normalized.shape[-1] != 2 or any(size <= 0 for size in normalized.shape[:3]):
        raise ValueError(f"Invalid flatmap volume shape: {array.shape}.")
    return np.asarray(normalized, dtype=np.float64)


def _flatmap_coordinate_axis(data: np.ndarray) -> int:
    """Return the non-spatial vector axis for one flatmap array."""
    array = np.asarray(data)
    if array.ndim != 4:
        raise ValueError(
            "Flatmap NRRD must be a 4D volume with one coordinate axis of length 2; "
            f"got shape {array.shape}."
        )
    if array.shape[-1] == 2:
        return array.ndim - 1
    if array.shape[0] == 2:
        return 0

    coordinate_axes = [axis for axis, size in enumerate(array.shape) if size == 2]
    if len(coordinate_axes) != 1:
        raise ValueError(
            "Flatmap NRRD must have exactly one coordinate axis of length 2 "
            "when the coordinate axis is not first or last; "
            f"got shape {array.shape}."
        )
    return coordinate_axes[0]


def normalize_depth_volume(data: np.ndarray) -> np.ndarray:
    """Return depth data normalized to a 3D float64 volume."""
    array = np.asarray(data)
    if array.ndim != 3:
        raise ValueError(f"Depth NRRD must be a 3D volume; got shape {array.shape}.")
    if any(size <= 0 for size in array.shape):
        raise ValueError(f"Invalid depth volume shape: {array.shape}.")
    return np.asarray(array, dtype=np.float64)


def _header_vector(value: object) -> np.ndarray | None:
    """Return one finite 3D header vector, or ``None`` for non-spatial axes."""
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.size != 3 or not np.all(np.isfinite(vector)):
        return None
    return vector


def spatial_transform_from_header(
    header: dict[str, Any],
    *,
    ndim: int,
    coordinate_axis: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract NRRD spatial axis vectors and origin from a header.

    NRRD stores one 3D direction vector per data axis. Flatmap arrays may have
    an extra vector-valued coordinate axis; that axis is omitted here so the
    result maps normalized 3D lookup indices to CCF physical coordinates.
    """
    raw_directions = header.get("space directions")
    raw_origin = header.get("space origin")
    if raw_directions is None or raw_origin is None:
        return None, None

    try:
        directions_values = list(raw_directions)
    except TypeError:
        return None, None
    if len(directions_values) != ndim:
        return None, None

    directions: list[np.ndarray] = []
    for axis, value in enumerate(directions_values):
        if coordinate_axis is not None and axis == coordinate_axis:
            continue
        vector = _header_vector(value)
        if vector is not None:
            directions.append(vector)
    if len(directions) != 3:
        return None, None

    origin = _header_vector(raw_origin)
    if origin is None:
        return None, None
    return np.vstack(directions), origin


def validate_flatmap_depth_shapes(flatmap: np.ndarray, depth: np.ndarray) -> None:
    """Raise when flatmap and depth volumes do not share a lookup grid."""
    if tuple(flatmap.shape[:3]) != tuple(depth.shape):
        raise ValueError(
            "Flatmap and depth volumes must share the same 3D atlas grid; "
            f"got flatmap grid {flatmap.shape[:3]} and depth shape {depth.shape}."
        )


def load_flatmap_volume_set(
    flatmap_path: str | Path,
    depth_path: str | Path,
) -> FlatmapVolumeSet:
    """Load and validate a flatmap lookup NRRD and matching depth NRRD."""
    flatmap_data, flatmap_header = _read_nrrd(flatmap_path)
    depth_data, depth_header = _read_nrrd(depth_path)

    flatmap_coordinate_axis = _flatmap_coordinate_axis(flatmap_data)
    flatmap = normalize_flatmap_volume(flatmap_data)
    depth = normalize_depth_volume(depth_data)
    validate_flatmap_depth_shapes(flatmap, depth)
    space_directions, space_origin = spatial_transform_from_header(
        depth_header,
        ndim=np.asarray(depth_data).ndim,
    )
    if space_directions is None or space_origin is None:
        space_directions, space_origin = spatial_transform_from_header(
            flatmap_header,
            ndim=np.asarray(flatmap_data).ndim,
            coordinate_axis=flatmap_coordinate_axis,
        )

    return FlatmapVolumeSet(
        flatmap=flatmap,
        depth=depth,
        flatmap_header=flatmap_header,
        depth_header=depth_header,
        flatmap_path=Path(flatmap_path),
        depth_path=Path(depth_path),
        space_directions=space_directions,
        space_origin=space_origin,
    )
