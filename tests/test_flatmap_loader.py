from __future__ import annotations

import numpy as np
import pytest

from napari_swc_viewer.flatmap_loader import (
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
    assert loaded.flatmap.dtype == np.float64
    assert loaded.depth.dtype == np.float64
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
