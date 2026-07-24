from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import napari_swc_viewer.flatmap_loader as flatmap_loader
from napari_swc_viewer.flatmap_loader import (
    FlatmapLookupLoadCancelledError,
    load_flatmap_volume_set,
    normalize_flatmap_volume,
    spatial_transform_from_header,
)

nrrd = pytest.importorskip("nrrd")


def _write_nrrd(path, data: np.ndarray) -> None:
    nrrd.write(str(path), data)


def _install_windows_npy_guard(monkeypatch):
    """Simulate Windows refusing to move or delete mapped NPY files."""
    tracked: list[tuple[Path, np.memmap]] = []
    replaced_sources: list[Path] = []
    unlinked_paths: list[Path] = []
    real_load = flatmap_loader.np.load
    real_open_memmap = flatmap_loader.np.lib.format.open_memmap
    real_replace = Path.replace
    real_unlink = Path.unlink

    def track(path, array):
        if isinstance(array, np.memmap) and not any(
            existing is array for _tracked_path, existing in tracked
        ):
            tracked.append((Path(path), array))
        return array

    def tracked_load(path, *args, **kwargs):
        return track(path, real_load(path, *args, **kwargs))

    def tracked_open_memmap(path, *args, **kwargs):
        return track(path, real_open_memmap(path, *args, **kwargs))

    def is_open(path: Path) -> bool:
        return any(
            tracked_path == Path(path)
            and getattr(array, "_mmap", None) is not None
            and not array._mmap.closed
            for tracked_path, array in tracked
        )

    def sharing_violation(path: Path) -> PermissionError:
        return PermissionError(
            32,
            "The process cannot access the file because it is being used "
            "by another process",
            str(path),
        )

    def guarded_replace(path, target):
        path = Path(path)
        if is_open(path):
            raise sharing_violation(path)
        replaced_sources.append(path)
        return real_replace(path, target)

    def guarded_unlink(path, *args, **kwargs):
        path = Path(path)
        if is_open(path):
            raise sharing_violation(path)
        unlinked_paths.append(path)
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(flatmap_loader.np, "load", tracked_load)
    monkeypatch.setattr(
        flatmap_loader.np.lib.format,
        "open_memmap",
        tracked_open_memmap,
    )
    monkeypatch.setattr(Path, "replace", guarded_replace)
    monkeypatch.setattr(Path, "unlink", guarded_unlink)
    return tracked, replaced_sources, unlinked_paths


def test_load_flatmap_volume_set_normalizes_last_axis_channels(tmp_path) -> None:
    flatmap_path = tmp_path / "flatmap_both_shaped.nrrd"
    depth_path = tmp_path / "depth.nrrd"
    flatmap = np.zeros((2, 3, 4, 2), dtype=np.float32)
    depth = np.ones((2, 3, 4), dtype=np.float32)
    _write_nrrd(flatmap_path, flatmap)
    _write_nrrd(depth_path, depth)

    loaded = load_flatmap_volume_set(flatmap_path, depth_path)

    assert loaded.flatmap.shape == (2, 3, 4, 2)
    assert loaded.depth.shape == (2, 3, 4)
    assert loaded.flatmap.dtype == np.float32
    assert loaded.depth.dtype == np.float32
    assert loaded.flatmap_path == flatmap_path
    assert loaded.depth_path == depth_path


def test_load_flatmap_volume_set_normalizes_first_axis_channels(tmp_path) -> None:
    flatmap_path = tmp_path / "flatmap_square.nrrd"
    depth_path = tmp_path / "depth.nrrd"
    flatmap = np.zeros((2, 3, 4, 5), dtype=np.float32)
    depth = np.ones((3, 4, 5), dtype=np.float32)
    _write_nrrd(flatmap_path, flatmap)
    _write_nrrd(depth_path, depth)

    loaded = load_flatmap_volume_set(flatmap_path, depth_path)

    assert loaded.flatmap.shape == (3, 4, 5, 2)
    assert loaded.depth.shape == (3, 4, 5)


def test_load_flatmap_volume_set_creates_and_uses_npy_cache(
    tmp_path,
    monkeypatch,
) -> None:
    flatmap_path = tmp_path / "flatmap_both_shaped.nrrd"
    depth_path = tmp_path / "depth.nrrd"
    flatmap = np.zeros((2, 3, 4, 2), dtype=np.float32)
    flatmap[..., 0] = 3
    depth = np.ones((2, 3, 4), dtype=np.float32)
    _write_nrrd(flatmap_path, flatmap)
    _write_nrrd(depth_path, depth)

    first = load_flatmap_volume_set(flatmap_path, depth_path)

    assert first.flatmap_loaded_from_cache is False
    assert first.depth_loaded_from_cache is False
    assert isinstance(first.flatmap, np.memmap)
    assert isinstance(first.depth, np.memmap)
    assert first.flatmap_npy_path == tmp_path / "flatmap_both_shaped.float32.npy"
    assert first.depth_npy_path == tmp_path / "depth.float32.npy"
    assert first.flatmap_npy_path.exists()
    assert first.depth_npy_path.exists()
    assert first.flatmap_npy_path.with_suffix(".npy.json").exists()
    assert first.depth_npy_path.with_suffix(".npy.json").exists()

    def _fail_read_nrrd(path):
        raise AssertionError(f"cache miss for {path}")

    monkeypatch.setattr(
        "napari_swc_viewer.flatmap_loader._read_nrrd",
        _fail_read_nrrd,
    )

    second = load_flatmap_volume_set(flatmap_path, depth_path)

    assert second.flatmap_loaded_from_cache is True
    assert second.depth_loaded_from_cache is True
    assert isinstance(second.flatmap, np.memmap)
    assert isinstance(second.depth, np.memmap)
    np.testing.assert_array_equal(second.flatmap, flatmap)
    np.testing.assert_array_equal(second.depth, depth)


def test_load_flatmap_volume_set_reuses_explicit_writable_cache_dir(
    tmp_path,
    monkeypatch,
) -> None:
    lookup_dir = tmp_path / "read-only-lookups"
    lookup_dir.mkdir()
    flatmap_path = lookup_dir / "flatmap_both_shaped.nrrd"
    depth_path = lookup_dir / "depth.nrrd"
    cache_dir = tmp_path / "writable-cache"
    flatmap = np.zeros((2, 3, 4, 2), dtype=np.float32)
    flatmap[..., 0] = 7
    depth = np.full((2, 3, 4), 25, dtype=np.float32)
    _write_nrrd(flatmap_path, flatmap)
    _write_nrrd(depth_path, depth)

    first = load_flatmap_volume_set(
        flatmap_path,
        depth_path,
        npy_cache_dir=cache_dir,
    )

    assert first.flatmap_npy_path is not None
    assert first.depth_npy_path is not None
    assert first.flatmap_npy_path.parent == cache_dir
    assert first.depth_npy_path.parent == cache_dir

    def _fail_read_nrrd(path):
        raise AssertionError(f"cache miss for {path}")

    monkeypatch.setattr(flatmap_loader, "_read_nrrd", _fail_read_nrrd)
    second = load_flatmap_volume_set(
        flatmap_path,
        depth_path,
        npy_cache_dir=cache_dir,
    )

    assert second.flatmap_loaded_from_cache is True
    assert second.depth_loaded_from_cache is True
    np.testing.assert_array_equal(second.flatmap, flatmap)
    np.testing.assert_array_equal(second.depth, depth)


def test_normalized_cache_write_closes_mapping_before_publication(
    tmp_path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "depth.nrrd"
    source_path.write_bytes(b"source")
    cache_dir = tmp_path / "cache"
    cache_path = cache_dir / "depth.float32.npy"
    metadata_path = cache_dir / "depth.float32.npy.json"
    data = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    tracked, replaced_sources, _unlinked_paths = _install_windows_npy_guard(monkeypatch)

    result = flatmap_loader._write_npy_cache(
        data,
        source_path=source_path,
        kind="depth",
        cache_path=cache_path,
        metadata_path=metadata_path,
        source_shape=data.shape,
        source_ndim=data.ndim,
        coordinate_axis=None,
        cancel_callback=None,
    )

    assert result == cache_path
    assert cache_path.exists()
    assert metadata_path.exists()
    assert len(tracked) == 1
    assert tracked[0][1]._mmap.closed
    assert len(replaced_sources) == 2


def test_cancelled_normalized_cache_write_removes_temporary_files(
    tmp_path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "depth.nrrd"
    source_path.write_bytes(b"source")
    cache_dir = tmp_path / "cache"
    cache_path = cache_dir / "depth.float32.npy"
    metadata_path = cache_dir / "depth.float32.npy.json"
    data = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    callback_calls = 0

    def cancel_during_second_chunk() -> bool:
        nonlocal callback_calls
        callback_calls += 1
        return callback_calls >= 3

    tracked, _replaced_sources, unlinked_paths = _install_windows_npy_guard(monkeypatch)
    monkeypatch.setattr(flatmap_loader, "_NPY_CACHE_WRITE_CHUNK_VALUES", 4)

    with pytest.raises(FlatmapLookupLoadCancelledError, match="cancelled"):
        flatmap_loader._write_npy_cache(
            data,
            source_path=source_path,
            kind="depth",
            cache_path=cache_path,
            metadata_path=metadata_path,
            source_shape=data.shape,
            source_ndim=data.ndim,
            coordinate_axis=None,
            cancel_callback=cancel_during_second_chunk,
        )

    assert callback_calls == 3
    assert not cache_path.exists()
    assert not metadata_path.exists()
    assert list(cache_dir.iterdir()) == []
    assert len(tracked) == 1
    assert tracked[0][1]._mmap.closed
    assert unlinked_paths


def test_failed_normalized_cache_write_closes_mapping_and_preserves_error(
    tmp_path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "depth.nrrd"
    source_path.write_bytes(b"source")
    cache_dir = tmp_path / "cache"
    cache_path = cache_dir / "depth.float32.npy"
    metadata_path = cache_dir / "depth.float32.npy.json"

    class FailingArray:
        shape = (3, 2, 2)

        def __getitem__(self, _index):
            raise RuntimeError("injected cache conversion failure")

    tracked, _replaced_sources, _unlinked_paths = _install_windows_npy_guard(
        monkeypatch
    )
    real_close_memmap = flatmap_loader._close_memmap

    def close_then_report_error(array, *, flush=False):
        real_close_memmap(array, flush=flush)
        if not flush:
            raise PermissionError(32, "injected close failure")

    monkeypatch.setattr(flatmap_loader, "_close_memmap", close_then_report_error)

    with pytest.raises(RuntimeError, match="injected cache conversion failure"):
        flatmap_loader._write_npy_cache(
            FailingArray(),
            source_path=source_path,
            kind="depth",
            cache_path=cache_path,
            metadata_path=metadata_path,
            source_shape=FailingArray.shape,
            source_ndim=3,
            coordinate_axis=None,
            cancel_callback=None,
        )

    assert len(tracked) == 1
    assert tracked[0][1]._mmap.closed
    assert not cache_path.exists()
    assert not metadata_path.exists()
    assert list(cache_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("case", "kind", "array", "metadata_overrides"),
    [
        (
            "dtype",
            "depth",
            np.ones((2, 2, 2), dtype=np.float64),
            {},
        ),
        (
            "shape",
            "depth",
            np.ones((2, 2, 2), dtype=np.float32),
            {"normalized_shape": [9, 9, 9]},
        ),
        (
            "dimensionality",
            "depth",
            np.ones((2, 2), dtype=np.float32),
            {"source_ndim": 2},
        ),
        (
            "coordinate-axis",
            "flatmap",
            np.ones((2, 2, 2, 2), dtype=np.float32),
            {"coordinate_axis": "invalid"},
        ),
    ],
)
def test_invalid_mapped_npy_cache_is_closed_before_fallback(
    tmp_path,
    monkeypatch,
    case,
    kind,
    array,
    metadata_overrides,
) -> None:
    source_path = tmp_path / f"{case}.nrrd"
    source_path.write_bytes(b"source")
    cache_path = tmp_path / f"{case}.float32.npy"
    metadata_path = tmp_path / f"{case}.float32.npy.json"
    np.save(cache_path, array, allow_pickle=False)
    metadata = {
        "cache_version": flatmap_loader._NPY_CACHE_VERSION,
        "kind": kind,
        "source": flatmap_loader._source_signature(source_path),
        "source_shape": [int(size) for size in array.shape],
        "source_ndim": int(array.ndim),
        "coordinate_axis": 3 if kind == "flatmap" else None,
        "normalized_shape": [int(size) for size in array.shape],
        "dtype": "float32",
    }
    metadata.update(metadata_overrides)
    metadata_path.write_text(json.dumps(metadata))
    tracked, _replaced_sources, _unlinked_paths = _install_windows_npy_guard(
        monkeypatch
    )

    loaded = flatmap_loader._load_npy_cache(
        source_path,
        kind=kind,
        cache_path=cache_path,
        metadata_path=metadata_path,
        mmap_npy=True,
    )

    assert loaded is None
    if len(tracked) != 1:
        pytest.fail(f"expected one mapped cache array, got {len(tracked)}")
    mapping = tracked[0][1]._mmap
    assert mapping is not None
    assert mapping.closed
    cache_path.unlink()
    assert not cache_path.exists()


def test_spatial_transform_from_header_omits_flatmap_vector_axis() -> None:
    directions = np.asarray(
        [
            [np.nan, np.nan, np.nan],
            [0.0, 0.0, 10.0],
            [0.0, 10.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    origin = np.asarray([1.0, 2.0, 3.0], dtype=float)
    header = {"space directions": directions, "space origin": origin}

    parsed_directions, parsed_origin = spatial_transform_from_header(
        header,
        ndim=4,
        coordinate_axis=0,
    )

    np.testing.assert_allclose(
        parsed_directions,
        [[0.0, 0.0, 10.0], [0.0, 10.0, 0.0], [10.0, 0.0, 0.0]],
    )
    np.testing.assert_allclose(parsed_origin, origin)


def test_normalize_flatmap_volume_rejects_missing_coordinate_axis() -> None:
    with pytest.raises(ValueError, match="4D volume"):
        normalize_flatmap_volume(np.zeros((3, 4, 5), dtype=np.float32))


def test_load_flatmap_volume_set_rejects_depth_shape_mismatch(tmp_path) -> None:
    flatmap_path = tmp_path / "flatmap_both_square.nrrd"
    depth_path = tmp_path / "depth.nrrd"
    _write_nrrd(flatmap_path, np.zeros((2, 3, 4, 2), dtype=np.float32))
    _write_nrrd(depth_path, np.ones((2, 3, 5), dtype=np.float32))

    with pytest.raises(ValueError, match="same 3D atlas grid"):
        load_flatmap_volume_set(flatmap_path, depth_path)


def test_load_flatmap_volume_set_rejects_missing_files(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_flatmap_volume_set(tmp_path / "missing.nrrd", tmp_path / "depth.nrrd")
