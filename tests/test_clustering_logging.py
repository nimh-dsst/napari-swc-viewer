"""Tests for clustering debug logging instrumentation."""

from __future__ import annotations

import logging

import numpy as np

from napari_swc_viewer.analysis.clustering import (
    cluster_somas_hierarchical,
    compute_linkage,
)


def test_compute_linkage_emits_debug_phase_logs(caplog) -> None:
    """Linkage helper should log squareform and linkage timing phases."""
    dist = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ],
        dtype=np.float32,
    )

    with caplog.at_level(logging.DEBUG, logger="napari_swc_viewer.analysis.clustering"):
        linkage_matrix = compute_linkage(dist, method="average")

    assert linkage_matrix.shape == (2, 4)
    messages = [record.getMessage() for record in caplog.records]
    assert any("compute_linkage start" in message for message in messages)
    assert any("compute_linkage squareform complete" in message for message in messages)
    assert any("compute_linkage linkage complete" in message for message in messages)


def test_cluster_somas_hierarchical_emits_debug_phase_logs(caplog) -> None:
    """Hierarchical soma clustering should log each major clustering phase."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    neuron_ids = ["n1", "n2", "n3", "n4"]

    with caplog.at_level(logging.DEBUG, logger="napari_swc_viewer.analysis.clustering"):
        result = cluster_somas_hierarchical(
            coords,
            neuron_ids,
            method="ward",
            n_clusters=2,
        )

    assert result.labels.shape == (4,)
    messages = [record.getMessage() for record in caplog.records]
    assert any("cluster_somas_hierarchical start" in message for message in messages)
    assert any("Built Euclidean distance matrix" in message for message in messages)
    assert any("distance build complete" in message for message in messages)
    assert any("fcluster complete" in message for message in messages)
    assert any("leaves_list complete" in message for message in messages)
