from __future__ import annotations

import numpy as np
import pytest

from napari_neuron_navigator._reader import (
    napari_get_reader,
    read_nrrd_file,
    read_nrrd_files,
)

nrrd = pytest.importorskip("nrrd")


def _write_nrrd(path, data: np.ndarray) -> None:
    nrrd.write(str(path), data)


def test_read_nrrd_file_splits_last_axis_flatmap_channels(tmp_path) -> None:
    path = tmp_path / "flatmap_last.nrrd"
    flatmap = np.zeros((2, 3, 4, 2), dtype=np.float32)
    flatmap[..., 0] = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    flatmap[..., 1] = flatmap[..., 0] + 100.0
    _write_nrrd(path, flatmap)

    layers = read_nrrd_file(path)

    assert len(layers) == 2
    x_data, x_props, x_type = layers[0]
    y_data, y_props, y_type = layers[1]
    assert x_type == "image"
    assert y_type == "image"
    assert x_props["name"] == "Flatmap X: flatmap_last"
    assert y_props["name"] == "Flatmap Y: flatmap_last"
    np.testing.assert_array_equal(x_data, flatmap[..., 0])
    np.testing.assert_array_equal(y_data, flatmap[..., 1])
    assert x_props["metadata"]["nrrd_role"] == "flatmap"
    assert x_props["metadata"]["flatmap_channel"] == "x"
    assert x_props["metadata"]["source_shape"] == flatmap.shape
    assert x_props["metadata"]["normalized_shape"] == flatmap.shape


def test_read_nrrd_file_splits_first_axis_flatmap_channels(tmp_path) -> None:
    path = tmp_path / "flatmap_first.nrrd"
    flatmap = np.zeros((2, 3, 4, 5), dtype=np.float32)
    flatmap[0] = 7.0
    flatmap[1] = 11.0
    _write_nrrd(path, flatmap)

    layers = read_nrrd_file(path)

    assert len(layers) == 2
    x_data, x_props, _x_type = layers[0]
    y_data, y_props, _y_type = layers[1]
    np.testing.assert_array_equal(x_data, np.full((3, 4, 5), 7.0, dtype=np.float32))
    np.testing.assert_array_equal(y_data, np.full((3, 4, 5), 11.0, dtype=np.float32))
    assert x_props["metadata"]["source_shape"] == flatmap.shape
    assert x_props["metadata"]["normalized_shape"] == (3, 4, 5, 2)
    assert y_props["metadata"]["flatmap_channel_index"] == 1


def test_read_nrrd_file_loads_depth_volume(tmp_path) -> None:
    path = tmp_path / "depth.nrrd"
    depth = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    _write_nrrd(path, depth)

    layers = read_nrrd_file(path)

    assert len(layers) == 1
    data, props, layer_type = layers[0]
    assert layer_type == "image"
    assert props["name"] == "Depth: depth"
    np.testing.assert_array_equal(data, depth)
    assert props["metadata"]["nrrd_role"] == "depth"
    assert props["metadata"]["source_shape"] == depth.shape
    assert props["metadata"]["normalized_shape"] == depth.shape


def test_read_nrrd_files_preserves_file_order(tmp_path) -> None:
    flatmap_path = tmp_path / "flatmap.nrrd"
    depth_path = tmp_path / "depth.nrrd"
    _write_nrrd(flatmap_path, np.zeros((2, 3, 4, 2), dtype=np.float32))
    _write_nrrd(depth_path, np.ones((2, 3, 4), dtype=np.float32))

    layers = read_nrrd_files([flatmap_path, depth_path])

    assert [props["name"] for _data, props, _type in layers] == [
        "Flatmap X: flatmap",
        "Flatmap Y: flatmap",
        "Depth: depth",
    ]


def test_read_nrrd_file_rejects_unsupported_shape(tmp_path) -> None:
    path = tmp_path / "unsupported.nrrd"
    _write_nrrd(path, np.zeros((3, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="Unsupported NRRD shape"):
        read_nrrd_file(path)


def test_napari_get_reader_claims_only_all_nrrd_selections(tmp_path) -> None:
    nrrd_path = tmp_path / "depth.nrrd"
    text_path = tmp_path / "notes.txt"

    assert napari_get_reader(str(nrrd_path)) is read_nrrd_file
    assert napari_get_reader([str(nrrd_path)]) is read_nrrd_files
    assert napari_get_reader([str(nrrd_path), str(text_path)]) is None
