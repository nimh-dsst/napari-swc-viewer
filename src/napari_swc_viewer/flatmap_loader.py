"""NRRD loading helpers for isocortex flatmap projection."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
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

_NPY_CACHE_VERSION = 1
_NPY_CACHE_SUFFIX = ".float32.npy"
_NPY_CACHE_WRITE_CHUNK_VALUES = 16_000_000


class FlatmapLookupLoadCancelledError(RuntimeError):
    """Raised when normalized lookup materialization is cancelled."""


def _check_load_cancelled(cancel_callback: Callable[[], bool] | None) -> None:
    if cancel_callback is not None and cancel_callback():
        raise FlatmapLookupLoadCancelledError("Flatmap lookup loading cancelled.")


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
    flatmap_npy_path: Path | None = None
    depth_npy_path: Path | None = None
    flatmap_loaded_from_cache: bool = False
    depth_loaded_from_cache: bool = False


@dataclass(frozen=True)
class _LoadedVolume:
    data: np.ndarray
    header: dict[str, Any]
    source_ndim: int
    coordinate_axis: int | None
    npy_path: Path | None
    loaded_from_cache: bool


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


def _read_nrrd_header(path: str | Path) -> dict[str, Any]:
    """Read one NRRD header without loading the array payload."""
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

    return dict(nrrd.read_header(str(source)))


def _npy_cache_paths(
    source_path: str | Path,
    cache_dir: str | Path | None,
) -> tuple[Path, Path]:
    source = Path(source_path)
    if cache_dir is None:
        cache_path = source.with_name(f"{source.stem}{_NPY_CACHE_SUFFIX}")
    else:
        source_digest = hashlib.sha256(
            str(source.resolve()).encode("utf-8")
        ).hexdigest()[:12]
        cache_path = (
            Path(cache_dir) / f"{source.stem}.{source_digest}{_NPY_CACHE_SUFFIX}"
        )
    return cache_path, Path(f"{cache_path}.json")


def _source_signature(source_path: Path) -> dict[str, Any]:
    stat = source_path.stat()
    return {
        "path": str(source_path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _cache_metadata_matches_source(
    metadata: dict[str, Any],
    *,
    source_path: Path,
    kind: str,
) -> bool:
    if metadata.get("cache_version") != _NPY_CACHE_VERSION:
        return False
    if metadata.get("kind") != kind:
        return False
    return metadata.get("source") == _source_signature(source_path)


def _read_cache_metadata(metadata_path: Path) -> dict[str, Any] | None:
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return metadata if isinstance(metadata, dict) else None


def _load_npy_cache(
    source_path: Path,
    *,
    kind: str,
    cache_path: Path,
    metadata_path: Path,
    mmap_npy: bool,
) -> _LoadedVolume | None:
    if not cache_path.exists() or not metadata_path.exists():
        return None

    metadata = _read_cache_metadata(metadata_path)
    if metadata is None or not _cache_metadata_matches_source(
        metadata,
        source_path=source_path,
        kind=kind,
    ):
        return None

    try:
        array = np.load(
            cache_path,
            mmap_mode="r" if mmap_npy else None,
            allow_pickle=False,
        )
    except (OSError, ValueError):
        return None

    if array.dtype != np.float32:
        return None

    normalized_shape = metadata.get("normalized_shape")
    if not isinstance(normalized_shape, list) or tuple(array.shape) != tuple(
        normalized_shape
    ):
        return None

    if kind == "flatmap":
        if array.ndim != 4 or array.shape[-1] != 2:
            return None
        raw_coordinate_axis = metadata.get("coordinate_axis")
        if raw_coordinate_axis is None:
            return None
        coordinate_axis = int(raw_coordinate_axis)
    else:
        if array.ndim != 3:
            return None
        coordinate_axis = None

    raw_source_ndim = metadata.get("source_ndim")
    if raw_source_ndim is None:
        return None

    return _LoadedVolume(
        data=array,
        header=_read_nrrd_header(source_path),
        source_ndim=int(raw_source_ndim),
        coordinate_axis=coordinate_axis,
        npy_path=cache_path,
        loaded_from_cache=True,
    )


def _write_npy_cache(
    data: np.ndarray,
    *,
    source_path: Path,
    kind: str,
    cache_path: Path,
    metadata_path: Path,
    source_shape: tuple[int, ...],
    source_ndim: int,
    coordinate_axis: int | None,
    cancel_callback: Callable[[], bool] | None,
) -> Path | None:
    """Persist one normalized float32 lookup array and cache metadata.

    Cache writes are opportunistic so read-only data locations do not break
    flatmap projection in the GUI.
    """
    metadata = {
        "cache_version": _NPY_CACHE_VERSION,
        "kind": kind,
        "source": _source_signature(source_path),
        "source_shape": [int(size) for size in source_shape],
        "source_ndim": int(source_ndim),
        "coordinate_axis": (
            None if coordinate_axis is None else int(coordinate_axis)
        ),
        "normalized_shape": [int(size) for size in data.shape],
        "dtype": "float32",
    }

    token = uuid.uuid4().hex
    cache_tmp = cache_path.with_name(f".{cache_path.name}.{token}.tmp")
    metadata_tmp = metadata_path.with_name(f".{metadata_path.name}.{token}.tmp")
    try:
        _check_load_cancelled(cancel_callback)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        output = np.lib.format.open_memmap(
            cache_tmp,
            mode="w+",
            dtype=np.float32,
            shape=data.shape,
        )
        plane_values = max(1, int(np.prod(data.shape[1:], dtype=np.int64)))
        chunk_size = max(1, _NPY_CACHE_WRITE_CHUNK_VALUES // plane_values)
        for start in range(0, int(data.shape[0]), chunk_size):
            _check_load_cancelled(cancel_callback)
            stop = min(start + chunk_size, int(data.shape[0]))
            output[start:stop] = data[start:stop]
        output.flush()
        del output
        cache_tmp.replace(cache_path)
        metadata_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        metadata_tmp.replace(metadata_path)
    except (OSError, ValueError):
        for path in (cache_tmp, metadata_tmp):
            try:
                path.unlink()
            except OSError:
                pass
        return None
    except BaseException:
        for path in (cache_tmp, metadata_tmp):
            try:
                path.unlink()
            except OSError:
                pass
        raise

    return cache_path


def _load_flatmap_volume(
    path: str | Path,
    *,
    use_npy_cache: bool,
    create_npy_cache: bool,
    npy_cache_dir: str | Path | None,
    mmap_npy: bool,
    cancel_callback: Callable[[], bool] | None,
) -> _LoadedVolume:
    source_path = Path(path)
    _check_load_cancelled(cancel_callback)
    if use_npy_cache:
        cache_path, metadata_path = _npy_cache_paths(source_path, npy_cache_dir)
        cached = _load_npy_cache(
            source_path,
            kind="flatmap",
            cache_path=cache_path,
            metadata_path=metadata_path,
            mmap_npy=mmap_npy,
        )
        if cached is not None:
            return cached
    else:
        cache_path = metadata_path = None

    flatmap_data, flatmap_header = _read_nrrd(source_path)
    _check_load_cancelled(cancel_callback)
    coordinate_axis = _flatmap_coordinate_axis(flatmap_data)
    source_shape = tuple(int(size) for size in flatmap_data.shape)
    source_ndim = int(np.asarray(flatmap_data).ndim)
    flatmap_view = _normalized_flatmap_view(flatmap_data)

    npy_path = None
    if use_npy_cache and create_npy_cache:
        npy_path = _write_npy_cache(
            flatmap_view,
            source_path=source_path,
            kind="flatmap",
            cache_path=cache_path,
            metadata_path=metadata_path,
            source_shape=source_shape,
            source_ndim=source_ndim,
            coordinate_axis=coordinate_axis,
            cancel_callback=cancel_callback,
        )
        if npy_path is not None:
            del flatmap_view
            del flatmap_data
            cached = _load_npy_cache(
                source_path,
                kind="flatmap",
                cache_path=cache_path,
                metadata_path=metadata_path,
                mmap_npy=mmap_npy,
            )
            if cached is not None:
                return _LoadedVolume(
                    data=cached.data,
                    header=flatmap_header,
                    source_ndim=source_ndim,
                    coordinate_axis=coordinate_axis,
                    npy_path=npy_path,
                    loaded_from_cache=False,
                )
            flatmap_data, _unused_header = _read_nrrd(source_path)
            flatmap_view = _normalized_flatmap_view(flatmap_data)

    flatmap = np.asarray(flatmap_view, dtype=np.float32)
    del flatmap_view
    del flatmap_data

    return _LoadedVolume(
        data=flatmap,
        header=flatmap_header,
        source_ndim=source_ndim,
        coordinate_axis=coordinate_axis,
        npy_path=npy_path,
        loaded_from_cache=False,
    )


def _load_depth_volume(
    path: str | Path,
    *,
    use_npy_cache: bool,
    create_npy_cache: bool,
    npy_cache_dir: str | Path | None,
    mmap_npy: bool,
    cancel_callback: Callable[[], bool] | None,
) -> _LoadedVolume:
    source_path = Path(path)
    _check_load_cancelled(cancel_callback)
    if use_npy_cache:
        cache_path, metadata_path = _npy_cache_paths(source_path, npy_cache_dir)
        cached = _load_npy_cache(
            source_path,
            kind="depth",
            cache_path=cache_path,
            metadata_path=metadata_path,
            mmap_npy=mmap_npy,
        )
        if cached is not None:
            return cached
    else:
        cache_path = metadata_path = None

    depth_data, depth_header = _read_nrrd(source_path)
    _check_load_cancelled(cancel_callback)
    source_shape = tuple(int(size) for size in depth_data.shape)
    source_ndim = int(np.asarray(depth_data).ndim)
    depth_view = _normalized_depth_view(depth_data)

    npy_path = None
    if use_npy_cache and create_npy_cache:
        npy_path = _write_npy_cache(
            depth_view,
            source_path=source_path,
            kind="depth",
            cache_path=cache_path,
            metadata_path=metadata_path,
            source_shape=source_shape,
            source_ndim=source_ndim,
            coordinate_axis=None,
            cancel_callback=cancel_callback,
        )
        if npy_path is not None:
            del depth_view
            del depth_data
            cached = _load_npy_cache(
                source_path,
                kind="depth",
                cache_path=cache_path,
                metadata_path=metadata_path,
                mmap_npy=mmap_npy,
            )
            if cached is not None:
                return _LoadedVolume(
                    data=cached.data,
                    header=depth_header,
                    source_ndim=source_ndim,
                    coordinate_axis=None,
                    npy_path=npy_path,
                    loaded_from_cache=False,
                )
            depth_data, _unused_header = _read_nrrd(source_path)
            depth_view = _normalized_depth_view(depth_data)

    depth = np.asarray(depth_view, dtype=np.float32)
    del depth_view
    del depth_data

    return _LoadedVolume(
        data=depth,
        header=depth_header,
        source_ndim=source_ndim,
        coordinate_axis=None,
        npy_path=npy_path,
        loaded_from_cache=False,
    )


def _normalized_flatmap_view(data: np.ndarray) -> np.ndarray:
    """Return a validated ``(nx, ny, nz, 2)`` view without dtype copying."""
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
    return normalized


def normalize_flatmap_volume(data: np.ndarray) -> np.ndarray:
    """Return flatmap data normalized to ``(nx, ny, nz, 2)`` float32."""
    return np.asarray(_normalized_flatmap_view(data), dtype=np.float32)


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


def _normalized_depth_view(data: np.ndarray) -> np.ndarray:
    """Return a validated 3D depth view without dtype copying."""
    array = np.asarray(data)
    if array.ndim != 3:
        raise ValueError(f"Depth NRRD must be a 3D volume; got shape {array.shape}.")
    if any(size <= 0 for size in array.shape):
        raise ValueError(f"Invalid depth volume shape: {array.shape}.")
    return array


def normalize_depth_volume(data: np.ndarray) -> np.ndarray:
    """Return depth data normalized to a 3D float32 volume."""
    return np.asarray(_normalized_depth_view(data), dtype=np.float32)


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
    *,
    use_npy_cache: bool = True,
    create_npy_cache: bool = True,
    npy_cache_dir: str | Path | None = None,
    mmap_npy: bool = True,
    cancel_callback: Callable[[], bool] | None = None,
) -> FlatmapVolumeSet:
    """Load and validate a flatmap lookup volume and matching depth volume.

    When ``use_npy_cache`` is true, normalized float32 ``.npy`` files are used
    when their metadata still matches the source NRRDs. Missing caches are
    created opportunistically when ``create_npy_cache`` is also true.
    """
    flatmap_volume = _load_flatmap_volume(
        flatmap_path,
        use_npy_cache=use_npy_cache,
        create_npy_cache=create_npy_cache,
        npy_cache_dir=npy_cache_dir,
        mmap_npy=mmap_npy,
        cancel_callback=cancel_callback,
    )
    depth_volume = _load_depth_volume(
        depth_path,
        use_npy_cache=use_npy_cache,
        create_npy_cache=create_npy_cache,
        npy_cache_dir=npy_cache_dir,
        mmap_npy=mmap_npy,
        cancel_callback=cancel_callback,
    )

    validate_flatmap_depth_shapes(flatmap_volume.data, depth_volume.data)
    space_directions, space_origin = spatial_transform_from_header(
        depth_volume.header,
        ndim=depth_volume.source_ndim,
    )
    if space_directions is None or space_origin is None:
        space_directions, space_origin = spatial_transform_from_header(
            flatmap_volume.header,
            ndim=flatmap_volume.source_ndim,
            coordinate_axis=flatmap_volume.coordinate_axis,
        )

    return FlatmapVolumeSet(
        flatmap=flatmap_volume.data,
        depth=depth_volume.data,
        flatmap_header=flatmap_volume.header,
        depth_header=depth_volume.header,
        flatmap_path=Path(flatmap_path),
        depth_path=Path(depth_path),
        space_directions=space_directions,
        space_origin=space_origin,
        flatmap_npy_path=flatmap_volume.npy_path,
        depth_npy_path=depth_volume.npy_path,
        flatmap_loaded_from_cache=flatmap_volume.loaded_from_cache,
        depth_loaded_from_cache=depth_volume.loaded_from_cache,
    )
