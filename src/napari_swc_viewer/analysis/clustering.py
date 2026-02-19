"""Hierarchical clustering and dendrogram comparison.

Ported from swc-mapper/compare_cluster_grids.py. Provides clustering
of neurons based on their pairwise correlation matrix, plus tools for
comparing different clustering solutions.

Also provides soma-location-based clustering using hierarchical,
k-means, and DBSCAN algorithms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.cluster.hierarchy import cophenet, fcluster, leaves_list, linkage
from scipy.spatial.distance import pdist, squareform
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


# ---------------------------------------------------------------------------
# Soma-location clustering
# ---------------------------------------------------------------------------


def _euclidean_distance_matrix(
    coords: NDArray[np.float64],
) -> NDArray[np.float32]:
    """Compute a square Euclidean distance matrix from 3-D coordinates.

    Parameters
    ----------
    coords : NDArray[np.float64]
        (N, 3) array of soma coordinates in microns.

    Returns
    -------
    NDArray[np.float32]
        (N, N) symmetric distance matrix with zero diagonal.
    """
    return squareform(pdist(coords, metric="euclidean")).astype(np.float32)


def cluster_somas_hierarchical(
    coords: NDArray[np.float64],
    neuron_ids: list[str],
    method: str = "ward",
    n_clusters: int = 5,
) -> ClusterResult:
    """Cluster neurons by soma location using hierarchical clustering.

    Parameters
    ----------
    coords : NDArray[np.float64]
        (N, 3) soma coordinates in microns.
    neuron_ids : list[str]
        Neuron identifiers matching rows of *coords*.
    method : str
        Linkage method (ward, average, complete, single).
    n_clusters : int
        Number of flat clusters to extract.

    Returns
    -------
    ClusterResult
    """
    dist = _euclidean_distance_matrix(coords)
    Z = compute_linkage(dist, method=method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust").astype(np.int32)
    reorder = leaves_list(Z)

    actual_k = int(len(np.unique(labels)))
    logger.info(
        f"Soma hierarchical clustering: {len(neuron_ids)} neurons, "
        f"{actual_k} clusters, method={method}"
    )

    return ClusterResult(
        correlation_matrix=dist,
        distance_matrix=dist,
        linkage_matrix=Z,
        neuron_ids=neuron_ids,
        reorder_indices=reorder,
        labels=labels,
    )


def cluster_somas_kmeans(
    coords: NDArray[np.float64],
    neuron_ids: list[str],
    n_clusters: int = 5,
) -> ClusterResult:
    """Cluster neurons by soma location using k-means.

    Requires scikit-learn.

    Parameters
    ----------
    coords : NDArray[np.float64]
        (N, 3) soma coordinates in microns.
    neuron_ids : list[str]
        Neuron identifiers matching rows of *coords*.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    ClusterResult
    """
    from sklearn.cluster import KMeans

    dist = _euclidean_distance_matrix(coords)
    # Linkage is computed for clustermap dendrogram visualisation only
    Z = compute_linkage(dist, method="average")
    reorder = leaves_list(Z)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(coords).astype(np.int32) + 1  # 1-indexed like fcluster

    actual_k = int(len(np.unique(labels)))
    logger.info(
        f"Soma k-means clustering: {len(neuron_ids)} neurons, "
        f"{actual_k} clusters"
    )

    return ClusterResult(
        correlation_matrix=dist,
        distance_matrix=dist,
        linkage_matrix=Z,
        neuron_ids=neuron_ids,
        reorder_indices=reorder,
        labels=labels,
    )


def cluster_somas_dbscan(
    coords: NDArray[np.float64],
    neuron_ids: list[str],
    eps: float = 100.0,
    min_samples: int = 5,
) -> ClusterResult:
    """Cluster neurons by soma location using DBSCAN.

    Requires scikit-learn.  Noise points (label == -1 from DBSCAN)
    are assigned to cluster label 0 so all labels are non-negative.

    Parameters
    ----------
    coords : NDArray[np.float64]
        (N, 3) soma coordinates in microns.
    neuron_ids : list[str]
        Neuron identifiers matching rows of *coords*.
    eps : float
        Maximum distance between samples for DBSCAN (in microns).
    min_samples : int
        Minimum samples in a neighbourhood for DBSCAN.

    Returns
    -------
    ClusterResult
    """
    from sklearn.cluster import DBSCAN

    dist = _euclidean_distance_matrix(coords)
    # Linkage is computed for clustermap dendrogram visualisation only
    Z = compute_linkage(dist, method="average")
    reorder = leaves_list(Z)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    raw_labels = db.fit_predict(dist)

    # Shift so that labels start at 1 (noise=-1 becomes 0)
    labels = (raw_labels + 2).astype(np.int32)  # noise→1, cluster0→2, ...

    actual_k = int(len(np.unique(labels)))
    n_noise = int((raw_labels == -1).sum())
    logger.info(
        f"Soma DBSCAN clustering: {len(neuron_ids)} neurons, "
        f"{actual_k} clusters (incl. noise), {n_noise} noise points, "
        f"eps={eps}, min_samples={min_samples}"
    )

    return ClusterResult(
        correlation_matrix=dist,
        distance_matrix=dist,
        linkage_matrix=Z,
        neuron_ids=neuron_ids,
        reorder_indices=reorder,
        labels=labels,
    )
