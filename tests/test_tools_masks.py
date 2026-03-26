from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from napari_swc_viewer.analysis.mask import (
    build_binary_mask_from_heatmap,
    build_binary_mask_from_threshold_range,
    merge_heatmap_volumes,
    otsu_threshold_positive,
    smooth_heatmap_volume,
)
from napari_swc_viewer.db import NeuronDatabase


class FakeAtlas:
    def __init__(self) -> None:
        self.annotation = np.zeros((5, 5, 5), dtype=np.int32)
        self.resolution = (25.0, 25.0, 25.0)
        self.atlas_name = "fake_atlas"
        self.structures = {}


def test_smooth_heatmap_volume_sigma_zero_returns_copy() -> None:
    volume = np.zeros((3, 3, 3), dtype=np.float32)
    volume[1, 1, 1] = 5.0

    smoothed = smooth_heatmap_volume(volume, sigma=0.0)

    assert np.array_equal(smoothed, volume)
    assert smoothed is not volume


def test_merge_heatmap_volumes_sums_inputs() -> None:
    left = np.zeros((3, 3, 3), dtype=np.float32)
    right = np.zeros((3, 3, 3), dtype=np.float32)
    left[1, 1, 1] = 2.0
    right[1, 1, 1] = 3.0
    right[2, 1, 1] = 1.0

    merged = merge_heatmap_volumes([left, right])

    assert float(merged[1, 1, 1]) == 5.0
    assert float(merged[2, 1, 1]) == 1.0


def test_otsu_threshold_positive_ignores_zero_background() -> None:
    volume = np.zeros((4, 4, 4), dtype=np.float32)
    volume.ravel()[:4] = 1.0
    volume.ravel()[4:8] = 10.0

    threshold = otsu_threshold_positive(volume)

    assert 1.0 <= threshold <= 10.0


def test_build_binary_mask_from_heatmap_all_zero_volume_stays_empty() -> None:
    volume = np.zeros((3, 3, 3), dtype=np.float32)

    mask, threshold, smoothed = build_binary_mask_from_heatmap(
        volume,
        sigma=1.0,
        threshold_mode="otsu",
    )

    assert threshold == 0.0
    assert smoothed.shape == volume.shape
    assert int(mask.sum()) == 0


def test_build_binary_mask_from_threshold_range_lower_only() -> None:
    volume = np.zeros((3, 3, 3), dtype=np.float32)
    volume[1, 1, 1] = 5.0
    volume[1, 1, 2] = 1.0

    mask = build_binary_mask_from_threshold_range(
        volume,
        lower_threshold=2.0,
    )

    assert int(mask[1, 1, 1]) == 1
    assert int(mask[1, 1, 2]) == 0


def test_build_binary_mask_from_threshold_range_lower_and_upper() -> None:
    volume = np.zeros((3, 3, 3), dtype=np.float32)
    volume[1, 1, 0] = 1.0
    volume[1, 1, 1] = 5.0
    volume[1, 1, 2] = 10.0

    mask = build_binary_mask_from_threshold_range(
        volume,
        lower_threshold=2.0,
        upper_threshold=6.0,
    )

    assert int(mask[1, 1, 0]) == 0
    assert int(mask[1, 1, 1]) == 1
    assert int(mask[1, 1, 2]) == 0


def test_build_binary_mask_from_threshold_range_rejects_upper_below_lower() -> None:
    volume = np.zeros((3, 3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="Upper threshold"):
        build_binary_mask_from_threshold_range(
            volume,
            lower_threshold=2.0,
            upper_threshold=1.0,
        )


def test_build_binary_mask_from_heatmap_manual_threshold() -> None:
    volume = np.zeros((3, 3, 3), dtype=np.float32)
    volume[1, 1, 1] = 5.0
    volume[1, 1, 2] = 1.0

    mask, threshold, _smoothed = build_binary_mask_from_heatmap(
        volume,
        sigma=0.0,
        threshold_mode="manual",
        manual_threshold=2.0,
    )

    assert threshold == 2.0
    assert int(mask[1, 1, 1]) == 1
    assert int(mask[1, 1, 2]) == 0


def test_get_neurons_by_mask_supports_any_node_and_soma_only(tmp_path: Path) -> None:
    atlas = FakeAtlas()
    parquet_path = tmp_path / "neurons.parquet"
    df = pd.DataFrame(
        {
            "file_id": ["file_a", "file_a", "file_b", "file_c"],
            "neuron_id": ["n1", "n1", "n2", "n3"],
            "subject": ["s1", "s1", "s2", "s3"],
            "node_id": [1, 2, 1, 1],
            "parent_id": [-1, 1, -1, -1],
            "x": [0.0, 50.0, 100.0, 75.0],
            "y": [0.0, 75.0, 75.0, 75.0],
            "z": [0.0, 100.0, 100.0, 100.0],
            "type": [1, 2, 1, 1],
            "region_id": [0, 0, 0, 0],
            "region_name": ["", "", "", ""],
            "region_acronym": ["", "", "", ""],
        }
    )
    df.to_parquet(parquet_path, index=False)

    mask = np.zeros(atlas.annotation.shape, dtype=np.uint8)
    mask[4, 3, 2] = 1
    mask[4, 3, 4] = 1

    with NeuronDatabase(parquet_path) as db:
        any_node = db.get_neurons_by_mask(mask, atlas, soma_only=False)
        soma_only = db.get_neurons_by_mask(mask, atlas, soma_only=True)

    assert any_node["file_id"].tolist() == ["file_a", "file_b"]
    assert soma_only["file_id"].tolist() == ["file_b"]
