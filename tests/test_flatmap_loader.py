from __future__ import annotations

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
