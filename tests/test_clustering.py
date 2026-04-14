"""Tests for clustering helpers."""

from __future__ import annotations

import numpy as np

from napari_swc_viewer.analysis import clustering


def test_cluster_somas_hierarchical_uses_condensed_distances_for_linkage(monkeypatch) -> None:
    """Ward soma clustering should link directly from condensed distances."""
    calls: dict[str, object] = {}

    def fake_pdist(coords, metric):
        calls["pdist_shape"] = tuple(coords.shape)
        calls["pdist_metric"] = metric
        return np.array([10.0, 20.0, 30.0], dtype=np.float64)

    def fake_linkage(values, method):
        calls["linkage_ndim"] = int(np.asarray(values).ndim)
        calls["linkage_dtype"] = np.asarray(values).dtype
        calls["linkage_method"] = method
        return np.array([[0.0, 1.0, 10.0, 2.0], [2.0, 3.0, 20.0, 3.0]], dtype=np.float64)

    def fake_squareform(values, checks=False):
        calls["squareform_ndim"] = int(np.asarray(values).ndim)
        calls["squareform_checks"] = checks
        return np.array(
            [
                [0.0, 10.0, 20.0],
                [10.0, 0.0, 30.0],
                [20.0, 30.0, 0.0],
            ],
            dtype=np.float32,
        )

    monkeypatch.setattr(clustering, "pdist", fake_pdist)
    monkeypatch.setattr(clustering, "linkage", fake_linkage)
    monkeypatch.setattr(clustering, "squareform", fake_squareform)
    monkeypatch.setattr(
        clustering,
        "fcluster",
        lambda _linkage_matrix, t, criterion: np.array([1, 1, 2], dtype=np.int32),
    )
    monkeypatch.setattr(
        clustering,
        "leaves_list",
        lambda _linkage_matrix: np.array([2, 1, 0], dtype=np.intp),
    )

    result = clustering.cluster_somas_hierarchical(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        ["n1", "n2", "n3"],
        method="ward",
        n_clusters=2,
    )

    assert calls["pdist_shape"] == (3, 3)
    assert calls["pdist_metric"] == "euclidean"
    assert calls["linkage_ndim"] == 1
    assert calls["linkage_method"] == "ward"
    assert calls["squareform_ndim"] == 1
    assert calls["squareform_checks"] is False
    assert result.distance_matrix.shape == (3, 3)
    assert result.reorder_indices.tolist() == [2, 1, 0]
    assert result.labels.tolist() == [1, 1, 2]
