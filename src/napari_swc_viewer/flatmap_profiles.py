"""Portable metadata contracts for bilateral flatmap lookup sets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .flatmap_heatmap import compute_flatmap_lookup_stats
from .flatmap_loader import (
    FLATMAP_STYLE_FILENAMES,
    FlatmapLookupLoadCancelledError,
    FlatmapVolumeSet,
    load_flatmap_volume_set,
    spatial_transform_from_header,
)

BILATERAL_FLATMAP_STYLES = ("both_shaped", "both_square")
FLATMAP_LOOKUP_SET_ALGORITHM_VERSION = 1
DEFAULT_DEPTH_LOOKUP_FILENAME = "depth.nrrd"

_ProgressCallback = Callable[[str, int, int], None]
_CancelCallback = Callable[[], bool]


class FlatmapLookupCancelledError(RuntimeError):
    """Raised when lookup-set discovery/hashing is cancelled."""


def _check_cancel(cancel_callback: _CancelCallback | None) -> None:
    if cancel_callback is not None and cancel_callback():
        raise FlatmapLookupCancelledError("Flatmap lookup-set preparation cancelled.")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_file(
    path: str | Path,
    *,
    chunk_bytes: int = 8 * 1024 * 1024,
    cancel_callback: _CancelCallback | None = None,
    progress_callback: _ProgressCallback | None = None,
) -> str:
    """Return a streaming SHA-256 digest for *path*."""
    source = Path(path)
    digest = hashlib.sha256()
    total = int(source.stat().st_size)
    completed = 0
    with source.open("rb") as input_file:
        while True:
            _check_cancel(cancel_callback)
            chunk = input_file.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
            completed += len(chunk)
            if progress_callback is not None:
                progress_callback(f"Hashing {source.name}...", completed, total)
    return digest.hexdigest()


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}-{hashlib.sha256(_canonical_json(payload)).hexdigest()}"


def _float_tuple(values: Sequence[object], *, length: int) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if len(normalized) != length or not np.all(np.isfinite(normalized)):
        raise ValueError(f"Expected {length} finite values; got {values!r}.")
    return normalized


def _matrix_tuple(values: object) -> tuple[tuple[float, float, float], ...]:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("space_directions must be a finite 3 by 3 matrix.")
    return tuple(tuple(float(value) for value in row) for row in matrix)


@dataclass(frozen=True)
class FlatmapGridSpec:
    """Canonical lookup/grid metadata shared by Parquet and region caches."""

    grid_spec_id: str
    style: str
    lookup_coordinate_order: tuple[str, str, str]
    flatmap_coordinate_order: tuple[str, str]
    render_coordinate_order: tuple[str, str, str]
    spatial_shape: tuple[int, int, int]
    flatmap_shape: tuple[int, int, int, int]
    depth_shape: tuple[int, int, int]
    lookup_resolution_um: tuple[float, float, float]
    space_directions: tuple[tuple[float, float, float], ...]
    space_origin: tuple[float, float, float]
    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]
    depth_bounds_um: tuple[float, float]
    invalid_zero_sentinel: bool
    invalid_negative_one_sentinel: bool
    depth_invalid_below_um: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-safe representation."""
        return {
            "grid_spec_id": self.grid_spec_id,
            "style": self.style,
            "lookup_coordinate_order": list(self.lookup_coordinate_order),
            "flatmap_coordinate_order": list(self.flatmap_coordinate_order),
            "render_coordinate_order": list(self.render_coordinate_order),
            "spatial_shape": list(self.spatial_shape),
            "flatmap_shape": list(self.flatmap_shape),
            "depth_shape": list(self.depth_shape),
            "lookup_resolution_um": list(self.lookup_resolution_um),
            "space_directions": [list(row) for row in self.space_directions],
            "space_origin": list(self.space_origin),
            "x_bounds": list(self.x_bounds),
            "y_bounds": list(self.y_bounds),
            "depth_bounds_um": list(self.depth_bounds_um),
            "validity": {
                "invalid_zero_sentinel": self.invalid_zero_sentinel,
                "invalid_negative_one_sentinel": self.invalid_negative_one_sentinel,
                "depth_invalid_below_um": self.depth_invalid_below_um,
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FlatmapGridSpec:
        """Parse a grid spec stored in version-3 metadata or a cache manifest."""
        validity = payload.get("validity", {})
        if not isinstance(validity, Mapping):
            validity = {}
        return cls(
            grid_spec_id=str(payload["grid_spec_id"]),
            style=str(payload["style"]),
            lookup_coordinate_order=tuple(payload["lookup_coordinate_order"]),  # type: ignore[arg-type]
            flatmap_coordinate_order=tuple(payload["flatmap_coordinate_order"]),  # type: ignore[arg-type]
            render_coordinate_order=tuple(payload["render_coordinate_order"]),  # type: ignore[arg-type]
            spatial_shape=tuple(int(v) for v in payload["spatial_shape"]),  # type: ignore[arg-type]
            flatmap_shape=tuple(int(v) for v in payload["flatmap_shape"]),  # type: ignore[arg-type]
            depth_shape=tuple(int(v) for v in payload["depth_shape"]),  # type: ignore[arg-type]
            lookup_resolution_um=_float_tuple(
                payload["lookup_resolution_um"], length=3  # type: ignore[arg-type]
            ),
            space_directions=_matrix_tuple(payload["space_directions"]),
            space_origin=_float_tuple(payload["space_origin"], length=3),  # type: ignore[arg-type]
            x_bounds=_float_tuple(payload["x_bounds"], length=2),  # type: ignore[arg-type]
            y_bounds=_float_tuple(payload["y_bounds"], length=2),  # type: ignore[arg-type]
            depth_bounds_um=_float_tuple(payload["depth_bounds_um"], length=2),  # type: ignore[arg-type]
            invalid_zero_sentinel=bool(
                validity.get("invalid_zero_sentinel", False)
            ),
            invalid_negative_one_sentinel=bool(
                validity.get("invalid_negative_one_sentinel", True)
            ),
            depth_invalid_below_um=float(
                validity.get("depth_invalid_below_um", 0.0)
            ),
        )


@dataclass(frozen=True)
class FlatmapLookupSet:
    """Portable definition of the shaped/square/depth bilateral lookup trio."""

    lookup_set_id: str
    shaped_path: Path
    square_path: Path
    depth_path: Path
    source_sha256: Mapping[str, str]
    shaped_grid: FlatmapGridSpec
    square_grid: FlatmapGridSpec
    algorithm_version: int = FLATMAP_LOOKUP_SET_ALGORITHM_VERSION

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        return self.shaped_grid.spatial_shape

    @property
    def space_directions(self) -> tuple[tuple[float, float, float], ...]:
        return self.shaped_grid.space_directions

    @property
    def space_origin(self) -> tuple[float, float, float]:
        return self.shaped_grid.space_origin

    @property
    def lookup_resolution_um(self) -> tuple[float, float, float]:
        return self.shaped_grid.lookup_resolution_um

    def grid_for_style(self, style: str) -> FlatmapGridSpec:
        """Return the shaped or square grid metadata."""
        normalized = normalize_bilateral_style(style)
        return self.shaped_grid if normalized == "both_shaped" else self.square_grid

    def to_dict(self, *, include_paths: bool = True) -> dict[str, Any]:
        """Return JSON-safe metadata; paths are provenance, never identity."""
        payload: dict[str, Any] = {
            "lookup_set_id": self.lookup_set_id,
            "algorithm_version": self.algorithm_version,
            "source_sha256": dict(sorted(self.source_sha256.items())),
            "styles": {
                "both_shaped": self.shaped_grid.to_dict(),
                "both_square": self.square_grid.to_dict(),
            },
        }
        if include_paths:
            payload["source_paths"] = {
                "both_shaped": str(self.shaped_path.resolve()),
                "both_square": str(self.square_path.resolve()),
                "depth": str(self.depth_path.resolve()),
            }
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FlatmapLookupSet:
        """Parse a lookup-set definition from Parquet/cache metadata."""
        styles = payload.get("styles")
        hashes = payload.get("source_sha256")
        paths = payload.get("source_paths", {})
        if not isinstance(styles, Mapping) or not isinstance(hashes, Mapping):
            raise ValueError("Lookup-set metadata is missing styles or source hashes.")
        if not isinstance(paths, Mapping):
            paths = {}
        shaped = FlatmapGridSpec.from_dict(styles["both_shaped"])
        square = FlatmapGridSpec.from_dict(styles["both_square"])
        return cls(
            lookup_set_id=str(payload["lookup_set_id"]),
            shaped_path=Path(
                str(paths.get("both_shaped", FLATMAP_STYLE_FILENAMES["both_shaped"]))
            ),
            square_path=Path(
                str(paths.get("both_square", FLATMAP_STYLE_FILENAMES["both_square"]))
            ),
            depth_path=Path(str(paths.get("depth", DEFAULT_DEPTH_LOOKUP_FILENAME))),
            source_sha256={str(key): str(value) for key, value in hashes.items()},
            shaped_grid=shaped,
            square_grid=square,
            algorithm_version=int(
                payload.get(
                    "algorithm_version", FLATMAP_LOOKUP_SET_ALGORITHM_VERSION
                )
            ),
        )


def normalize_bilateral_style(style: str) -> str:
    aliases = {
        "shaped": "both_shaped",
        "square": "both_square",
        "bilateral_shaped": "both_shaped",
        "bilateral_square": "both_square",
    }
    normalized = aliases.get(str(style), str(style))
    if normalized not in BILATERAL_FLATMAP_STYLES:
        raise ValueError(
            "style must identify a bilateral shaped or square flatmap; "
            f"got {style!r}."
        )
    return normalized


def _explicit_resolution(
    value: float | Sequence[float] | None,
) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        resolution = (float(value),) * 3
    else:
        resolution = _float_tuple(value, length=3)
    if any(component <= 0.0 for component in resolution):
        raise ValueError("lookup_resolution_um must contain positive values.")
    return resolution


def _flatmap_coordinate_axis(header: Mapping[str, Any]) -> int | None:
    raw_sizes = header.get("sizes")
    try:
        sizes = tuple(int(value) for value in raw_sizes)
    except (TypeError, ValueError):
        return None
    if len(sizes) != 4:
        return None

    raw_directions = header.get("space directions")
    if raw_directions is not None:
        try:
            directions = list(raw_directions)
        except TypeError:
            directions = []
        if len(directions) == 4:
            nonspatial: list[int] = []
            for axis, value in enumerate(directions):
                try:
                    vector = np.asarray(value, dtype=float).reshape(-1)
                except (TypeError, ValueError):
                    nonspatial.append(axis)
                    continue
                if vector.size != 3 or not np.all(np.isfinite(vector)):
                    nonspatial.append(axis)
            if len(nonspatial) == 1:
                return nonspatial[0]

    candidates = [axis for axis, size in enumerate(sizes) if size == 2]
    if len(candidates) == 1:
        return candidates[0]
    if sizes[-1] == 2:
        return 3
    if sizes[0] == 2:
        return 0
    return None


def _volume_transform(
    volume: FlatmapVolumeSet,
    *,
    kind: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    if kind == "depth":
        header = volume.depth_header
        raw_sizes = header.get("sizes")
        try:
            ndim = len(raw_sizes)
        except TypeError:
            ndim = 3
        directions, origin = spatial_transform_from_header(header, ndim=ndim)
    else:
        header = volume.flatmap_header
        raw_sizes = header.get("sizes")
        try:
            ndim = len(raw_sizes)
        except TypeError:
            ndim = 4
        directions, origin = spatial_transform_from_header(
            header,
            ndim=ndim,
            coordinate_axis=_flatmap_coordinate_axis(header),
        )
    if directions is None or origin is None:
        return None
    directions = np.asarray(directions, dtype=float)
    origin = np.asarray(origin, dtype=float)
    if (
        directions.shape != (3, 3)
        or origin.shape != (3,)
        or not np.all(np.isfinite(directions))
        or not np.all(np.isfinite(origin))
        or abs(float(np.linalg.det(directions))) <= np.finfo(float).eps
    ):
        return None
    return directions, origin


def _validated_transform(
    shaped: FlatmapVolumeSet,
    square: FlatmapVolumeSet,
    *,
    lookup_resolution_um: float | Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    explicit_resolution = _explicit_resolution(lookup_resolution_um)
    candidates = [
        ("shaped flatmap", _volume_transform(shaped, kind="flatmap")),
        ("square flatmap", _volume_transform(square, kind="flatmap")),
        ("depth", _volume_transform(shaped, kind="depth")),
    ]
    available = [(name, transform) for name, transform in candidates if transform]
    missing = [name for name, transform in candidates if transform is None]
    if missing and explicit_resolution is None:
        raise ValueError(
            "Lookup NRRD header(s) lack a usable spatial transform: "
            f"{', '.join(missing)}. Provide lookup_resolution_um explicitly."
        )

    if missing:
        assert explicit_resolution is not None
        directions = np.diag(np.asarray(explicit_resolution, dtype=float))
        origin = np.zeros(3, dtype=float)
        for name, transform in available:
            assert transform is not None
            candidate_directions, candidate_origin = transform
            if not np.allclose(
                candidate_directions, directions, rtol=0.0, atol=1e-6
            ) or not np.allclose(candidate_origin, origin, rtol=0.0, atol=1e-6):
                raise ValueError(
                    f"The usable {name} transform does not match the explicit "
                    "axis-aligned lookup_resolution_um fallback."
                )
        return directions, origin, explicit_resolution

    if available:
        reference_name, reference = available[0]
        assert reference is not None
        directions, origin = reference
        for name, transform in available[1:]:
            assert transform is not None
            candidate_directions, candidate_origin = transform
            if not np.allclose(
                candidate_directions, directions, rtol=0.0, atol=1e-6
            ) or not np.allclose(candidate_origin, origin, rtol=0.0, atol=1e-6):
                raise ValueError(
                    "Lookup NRRD spatial transforms do not match: "
                    f"{reference_name} and {name}."
                )
        inferred_resolution = tuple(
            float(value) for value in np.linalg.norm(directions, axis=1)
        )
        if explicit_resolution is not None and not np.allclose(
            inferred_resolution, explicit_resolution, rtol=0.0, atol=1e-6
        ):
            raise ValueError(
                "Explicit lookup_resolution_um does not match the usable NRRD "
                f"transform ({inferred_resolution})."
            )
        resolution = inferred_resolution
    else:  # pragma: no cover - all-missing transforms return above
        raise AssertionError("Unreachable transform validation state.")

    return directions, origin, resolution


def _file_provenance(path: Path, sha256: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256,
    }


def build_flatmap_lookup_set(
    shaped_path: str | Path,
    square_path: str | Path,
    depth_path: str | Path,
    *,
    lookup_resolution_um: float | Sequence[float] | None = None,
    invalid_zero_sentinel: bool = False,
    invalid_negative_one_sentinel: bool = True,
    npy_cache_dir: str | Path | None = None,
    cancel_callback: _CancelCallback | None = None,
    progress_callback: _ProgressCallback | None = None,
) -> FlatmapLookupSet:
    """Load, validate, hash, and describe one bilateral lookup trio."""
    shaped_source = Path(shaped_path)
    square_source = Path(square_path)
    depth_source = Path(depth_path)
    _check_cancel(cancel_callback)
    if progress_callback is not None:
        progress_callback("Loading bilateral shaped lookup...", 0, 3)
    try:
        shaped = load_flatmap_volume_set(
            shaped_source,
            depth_source,
            npy_cache_dir=npy_cache_dir,
            mmap_npy=True,
            cancel_callback=cancel_callback,
        )
    except FlatmapLookupLoadCancelledError as exc:
        raise FlatmapLookupCancelledError(
            "Flatmap lookup-set preparation cancelled."
        ) from exc
    _check_cancel(cancel_callback)
    if progress_callback is not None:
        progress_callback("Loading bilateral square lookup...", 1, 3)
    try:
        square = load_flatmap_volume_set(
            square_source,
            depth_source,
            npy_cache_dir=npy_cache_dir,
            mmap_npy=True,
            cancel_callback=cancel_callback,
        )
    except FlatmapLookupLoadCancelledError as exc:
        raise FlatmapLookupCancelledError(
            "Flatmap lookup-set preparation cancelled."
        ) from exc
    _check_cancel(cancel_callback)
    shaped_shape = tuple(int(size) for size in shaped.flatmap.shape[:3])
    square_shape = tuple(int(size) for size in square.flatmap.shape[:3])
    depth_shape = tuple(int(size) for size in shaped.depth.shape)
    if shaped_shape != square_shape or shaped_shape != depth_shape:
        raise ValueError(
            "Bilateral shaped, square, and depth lookups must have identical "
            f"spatial shapes; got {shaped_shape}, {square_shape}, and {depth_shape}."
        )

    directions, origin, resolution = _validated_transform(
        shaped,
        square,
        lookup_resolution_um=lookup_resolution_um,
    )
    if progress_callback is not None:
        progress_callback("Computing shaped canonical bounds...", 2, 3)
    shaped_stats = compute_flatmap_lookup_stats(
        shaped.flatmap,
        shaped.depth,
        invalid_zero_sentinel=invalid_zero_sentinel,
        invalid_negative_one_sentinel=invalid_negative_one_sentinel,
        cancel_callback=(
            None
            if cancel_callback is None
            else lambda: (_check_cancel(cancel_callback) or False)
        ),
    )
    _check_cancel(cancel_callback)
    if progress_callback is not None:
        progress_callback("Computing square canonical bounds...", 2, 3)
    square_stats = compute_flatmap_lookup_stats(
        square.flatmap,
        square.depth,
        invalid_zero_sentinel=invalid_zero_sentinel,
        invalid_negative_one_sentinel=invalid_negative_one_sentinel,
        cancel_callback=(
            None
            if cancel_callback is None
            else lambda: (_check_cancel(cancel_callback) or False)
        ),
    )
    _check_cancel(cancel_callback)

    hashes = {
        "both_shaped": sha256_file(
            shaped_source,
            cancel_callback=cancel_callback,
            progress_callback=progress_callback,
        ),
        "both_square": sha256_file(
            square_source,
            cancel_callback=cancel_callback,
            progress_callback=progress_callback,
        ),
        "depth": sha256_file(
            depth_source,
            cancel_callback=cancel_callback,
            progress_callback=progress_callback,
        ),
    }
    _check_cancel(cancel_callback)
    common = {
        "lookup_coordinate_order": ["x", "y", "z"],
        "flatmap_coordinate_order": ["x_flat", "y_flat"],
        "render_coordinate_order": ["depth", "y", "x"],
        "spatial_shape": list(shaped_shape),
        "depth_shape": list(depth_shape),
        "lookup_resolution_um": list(resolution),
        "space_directions": np.asarray(directions, dtype=float).tolist(),
        "space_origin": np.asarray(origin, dtype=float).tolist(),
        "validity": {
            "invalid_zero_sentinel": bool(invalid_zero_sentinel),
            "invalid_negative_one_sentinel": bool(invalid_negative_one_sentinel),
            "depth_invalid_below_um": 0.0,
        },
    }

    def make_grid(style: str, stats) -> FlatmapGridSpec:
        identity_payload = {
            "algorithm_version": FLATMAP_LOOKUP_SET_ALGORITHM_VERSION,
            "style": style,
            "flatmap_sha256": hashes[style],
            "depth_sha256": hashes["depth"],
            **common,
            "flatmap_shape": list(stats.flatmap_shape),
            "x_bounds": list(stats.x_bounds),
            "y_bounds": list(stats.y_bounds),
            "depth_bounds_um": list(stats.depth_range_um),
        }
        return FlatmapGridSpec(
            grid_spec_id=_stable_id("fmg1", identity_payload),
            style=style,
            lookup_coordinate_order=("x", "y", "z"),
            flatmap_coordinate_order=("x_flat", "y_flat"),
            render_coordinate_order=("depth", "y", "x"),
            spatial_shape=shaped_shape,
            flatmap_shape=tuple(int(v) for v in stats.flatmap_shape),
            depth_shape=depth_shape,
            lookup_resolution_um=resolution,
            space_directions=_matrix_tuple(directions),
            space_origin=_float_tuple(origin, length=3),
            x_bounds=tuple(float(v) for v in stats.x_bounds),
            y_bounds=tuple(float(v) for v in stats.y_bounds),
            depth_bounds_um=tuple(float(v) for v in stats.depth_range_um),
            invalid_zero_sentinel=bool(invalid_zero_sentinel),
            invalid_negative_one_sentinel=bool(invalid_negative_one_sentinel),
        )

    shaped_grid = make_grid("both_shaped", shaped_stats)
    square_grid = make_grid("both_square", square_stats)
    identity = {
        "algorithm_version": FLATMAP_LOOKUP_SET_ALGORITHM_VERSION,
        "source_sha256": hashes,
        "styles": {
            "both_shaped": shaped_grid.to_dict(),
            "both_square": square_grid.to_dict(),
        },
    }
    return FlatmapLookupSet(
        lookup_set_id=_stable_id("fls1", identity),
        shaped_path=shaped_source,
        square_path=square_source,
        depth_path=depth_source,
        source_sha256=hashes,
        shaped_grid=shaped_grid,
        square_grid=square_grid,
    )


def discover_flatmap_lookup_paths(
    lookup_dir: str | Path,
) -> tuple[Path, Path, Path]:
    """Discover the required bilateral shaped, square, and depth NRRDs."""
    directory = Path(lookup_dir)
    if not directory.is_dir():
        raise ValueError(f"Lookup directory does not exist: {directory}")
    shaped = directory / FLATMAP_STYLE_FILENAMES["both_shaped"]
    square = directory / FLATMAP_STYLE_FILENAMES["both_square"]
    depth = directory / DEFAULT_DEPTH_LOOKUP_FILENAME
    missing = [path.name for path in (shaped, square) if not path.is_file()]
    if not depth.is_file():
        candidates = sorted(directory.glob("*depth*.nrrd"))
        if len(candidates) == 1:
            depth = candidates[0]
        else:
            missing.append(DEFAULT_DEPTH_LOOKUP_FILENAME)
    if missing:
        raise FileNotFoundError(
            f"Lookup directory {directory} is missing required file(s): {missing}"
        )
    return shaped, square, depth


def discover_flatmap_lookup_set(
    lookup_dir: str | Path,
    **kwargs: Any,
) -> FlatmapLookupSet:
    """Discover and build one bilateral lookup-set definition."""
    return build_flatmap_lookup_set(*discover_flatmap_lookup_paths(lookup_dir), **kwargs)
