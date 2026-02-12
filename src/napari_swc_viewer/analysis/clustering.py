"""Hierarchical clustering and dendrogram comparison.

Ported from swc-mapper/compare_cluster_grids.py. Provides clustering
of neurons based on their pairwise correlation matrix, plus tools for
comparing different clustering solutions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Container for clustering results."""

    correlation_matrix: NDArray[np.float32]
    distance_matrix: NDArray[np.float32]
    linkage_matrix: NDArray[np.float64]
    neuron_ids: list[str]
    reorder_indices: NDArray[np.intp]
    labels: NDArray[np.int32] = field(default_factory=lambda: np.array([], dtype=np.int32))


def compute_linkage(
    distance_matrix: NDArray[np.float32],
    method: str = "average",
) -> NDArray[np.float64]:
    """Compute hierarchical clustering linkage from a square distance matrix.

    Parameters
    ----------
    distance_matrix : NDArray[np.float32]
        Square symmetric distance matrix with zero diagonal.
    method : str
        Linkage method: 'average', 'ward', 'complete', 'single'.

    Returns
    -------
    NDArray[np.float64]
        Scipy linkage matrix.
    """
    condensed = squareform(distance_matrix, checks=False)
    return linkage(condensed, method=method)


def compute_clustermap_data(
    corr_matrix: NDArray[np.float32],
    neuron_ids: list[str],
    method: str = "average",
    n_clusters: int = 5,
) -> ClusterResult:
    """Compute full clustering from a correlation matrix.

    Converts correlation to distance (1 - r), computes hierarchical
    clustering, extracts cluster labels, and determines the dendrogram
    reorder for display.

    Parameters
    ----------
    corr_matrix : NDArray[np.float32]
        Square symmetric correlation matrix with 1.0 diagonal.
    neuron_ids : list[str]
        Ordered neuron identifiers matching matrix rows/columns.
    method : str
        Linkage method.
    n_clusters : int
        Number of clusters to extract via fcluster.

    Returns
    -------
    ClusterResult
        Complete clustering result.
    """
    # Clip correlation to [-1, 1] and convert to distance
    r = np.clip(corr_matrix, -1.0, 1.0)
    dist = 1.0 - r
    np.fill_diagonal(dist, 0.0)

    # Ensure non-negative distances
    dist = np.maximum(dist, 0.0)

    logger.info(
        f"Computing {method} linkage for {len(neuron_ids)} neurons, "
        f"distance range [{dist[dist > 0].min():.3f}, {dist.max():.3f}]"
    )

    Z = compute_linkage(dist, method=method)

    # Extract cluster labels
    labels = fcluster(Z, t=n_clusters, criterion="maxclust").astype(np.int32)
    actual_k = int(len(np.unique(labels)))

    # Get dendrogram leaf order (reorder indices)
    from scipy.cluster.hierarchy import leaves_list

    reorder = leaves_list(Z)

    if actual_k < n_clusters:
        logger.warning(
            f"Requested {n_clusters} clusters but fcluster produced only "
            f"{actual_k}: the dendrogram does not support that many distinct "
            f"groups. Label distribution: "
            f"{dict(zip(*np.unique(labels, return_counts=True)))}"
        )
    else:
        logger.info(
            f"Clustering complete: {actual_k} clusters, "
            f"sizes: {dict(zip(*np.unique(labels, return_counts=True)))}"
        )

    return ClusterResult(
        correlation_matrix=corr_matrix,
        distance_matrix=dist.astype(np.float32),
        linkage_matrix=Z,
        neuron_ids=neuron_ids,
        reorder_indices=reorder,
        labels=labels,
    )


def extract_clusters(
    linkage_matrix: NDArray[np.float64],
    n_clusters: int,
) -> NDArray[np.int32]:
    """Extract flat cluster labels from a linkage matrix.

    Parameters
    ----------
    linkage_matrix : NDArray[np.float64]
        Scipy linkage matrix.
    n_clusters : int
        Desired number of clusters.

    Returns
    -------
    NDArray[np.int32]
        Cluster label for each sample (1-indexed).
    """
    return fcluster(linkage_matrix, t=n_clusters, criterion="maxclust").astype(np.int32)


def cophenetic_spearman(
    Z1: NDArray[np.float64],
    Z2: NDArray[np.float64],
) -> tuple[float, float]:
    """Compare two dendrograms via cophenetic distance correlation.

    Parameters
    ----------
    Z1, Z2 : NDArray[np.float64]
        Linkage matrices from two different clustering solutions.

    Returns
    -------
    tuple[float, float]
        (Spearman r, p-value) comparing the cophenetic distances.
    """
    c1 = cophenet(Z1)
    c2 = cophenet(Z2)
    r, p = spearmanr(c1, c2)
    return float(r), float(p)


def compare_partitions(
    Z1: NDArray[np.float64],
    Z2: NDArray[np.float64],
    ks: range = range(2, 21),
) -> list[tuple[int, float, float]]:
    """Compare cluster partitions at multiple k values using ARI and NMI.

    Requires scikit-learn. Returns empty list if not installed.

    Parameters
    ----------
    Z1, Z2 : NDArray[np.float64]
        Linkage matrices to compare.
    ks : range
        Range of cluster counts to evaluate.

    Returns
    -------
    list[tuple[int, float, float]]
        List of (k, ARI, NMI) tuples.
    """
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    except ImportError:
        logger.warning("scikit-learn not installed; skipping partition comparison")
        return []

    results = []
    for k in ks:
        lab1 = fcluster(Z1, t=k, criterion="maxclust")
        lab2 = fcluster(Z2, t=k, criterion="maxclust")
        ari = adjusted_rand_score(lab1, lab2)
        nmi = normalized_mutual_info_score(lab1, lab2)
        results.append((k, float(ari), float(nmi)))

    return results
