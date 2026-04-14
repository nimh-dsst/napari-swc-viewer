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
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from scipy.cluster.hierarchy import cophenet, fcluster, leaves_list, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _format_nbytes(num_bytes: int) -> str:
    """Return a compact human-readable size string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.2f} KiB"
    if num_bytes < 1024**3:
        return f"{num_bytes / (1024**2):.2f} MiB"
    return f"{num_bytes / (1024**3):.2f} GiB"


@dataclass(frozen=True)
class ClusterRegionSelection:
    """Direct region selections and represented dataset descendants."""

    selected_region_ids: list[int] = field(default_factory=list)
    selected_region_acronyms: list[str] = field(default_factory=list)
    represented_region_ids: list[int] = field(default_factory=list)
    represented_region_acronyms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe mapping for export metadata."""
        return {
            "selected_region_ids": [
                int(value) for value in self.selected_region_ids
            ],
            "selected_region_acronyms": [
                str(value) for value in self.selected_region_acronyms
            ],
            "represented_region_ids": [
                int(value) for value in self.represented_region_ids
            ],
            "represented_region_acronyms": [
                str(value) for value in self.represented_region_acronyms
            ],
        }


@dataclass(frozen=True)
class ClusterRunMetadata:
    """Parameters and provenance required to reproduce a clustering run."""

    analysis_method: str
    clustering_algorithm: str
    distance_metric: str
    clustering_linkage: str | None
    dendrogram_linkage: str | None
    selected_region_ids: list[int] = field(default_factory=list)
    selected_region_acronyms: list[str] = field(default_factory=list)
    represented_region_ids: list[int] = field(default_factory=list)
    represented_region_acronyms: list[str] = field(default_factory=list)
    dilation_fraction: float = 0.0
    requested_cluster_count: int | None = None
    actual_cluster_count: int = 0
    dbscan_eps: float | None = None
    dbscan_min_samples: int | None = None
    atlas_name: str | None = None
    atlas_resolution_um: tuple[float, ...] = ()
    source_parquet_path: str | None = None
    dendrogram_leaf_order: list[int] = field(default_factory=list)

    @classmethod
    def from_region_selection(
        cls,
        *,
        region_selection: ClusterRegionSelection,
        analysis_method: str,
        clustering_algorithm: str,
        distance_metric: str,
        clustering_linkage: str | None,
        dendrogram_linkage: str | None,
        dilation_fraction: float,
        requested_cluster_count: int | None,
        actual_cluster_count: int,
        dbscan_eps: float | None,
        dbscan_min_samples: int | None,
        atlas_name: str | None,
        atlas_resolution_um: tuple[float, ...],
        source_parquet_path: str | None,
        dendrogram_leaf_order: list[int],
    ) -> "ClusterRunMetadata":
        """Build metadata from a worker-region selection payload."""
        return cls(
            analysis_method=analysis_method,
            clustering_algorithm=clustering_algorithm,
            distance_metric=distance_metric,
            clustering_linkage=clustering_linkage,
            dendrogram_linkage=dendrogram_linkage,
            selected_region_ids=list(region_selection.selected_region_ids),
            selected_region_acronyms=list(
                region_selection.selected_region_acronyms
            ),
            represented_region_ids=list(
                region_selection.represented_region_ids
            ),
            represented_region_acronyms=list(
                region_selection.represented_region_acronyms
            ),
            dilation_fraction=float(dilation_fraction),
            requested_cluster_count=requested_cluster_count,
            actual_cluster_count=int(actual_cluster_count),
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            atlas_name=atlas_name,
            atlas_resolution_um=tuple(
                float(value) for value in atlas_resolution_um
            ),
            source_parquet_path=source_parquet_path,
            dendrogram_leaf_order=[
                int(value) for value in dendrogram_leaf_order
            ],
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe mapping for workbook/parquet exports."""
        return {
            "analysis_method": self.analysis_method,
            "clustering_algorithm": self.clustering_algorithm,
            "distance_metric": self.distance_metric,
            "clustering_linkage": self.clustering_linkage,
            "dendrogram_linkage": self.dendrogram_linkage,
            "selected_region_ids": [
                int(value) for value in self.selected_region_ids
            ],
            "selected_region_acronyms": [
                str(value) for value in self.selected_region_acronyms
            ],
            "represented_region_ids": [
                int(value) for value in self.represented_region_ids
            ],
            "represented_region_acronyms": [
                str(value) for value in self.represented_region_acronyms
            ],
            "dilation_fraction": float(self.dilation_fraction),
            "requested_cluster_count": self.requested_cluster_count,
            "actual_cluster_count": int(self.actual_cluster_count),
            "dbscan_eps": self.dbscan_eps,
            "dbscan_min_samples": self.dbscan_min_samples,
            "atlas_name": self.atlas_name,
            "atlas_resolution_um": [
                float(value) for value in self.atlas_resolution_um
            ],
            "source_parquet_path": self.source_parquet_path,
            "dendrogram_leaf_order": [
                int(value) for value in self.dendrogram_leaf_order
            ],
        }


@dataclass
class ClusterResult:
    """Container for clustering results."""

    correlation_matrix: NDArray[np.float32]
    distance_matrix: NDArray[np.float32]
    linkage_matrix: NDArray[np.float64]
    neuron_ids: list[str]
    reorder_indices: NDArray[np.intp]
    labels: NDArray[np.int32] = field(
        default_factory=lambda: np.array([], dtype=np.int32)
    )
    metadata: ClusterRunMetadata | None = None

    def neuron_ids_in_leaf_order(self) -> list[str]:
        """Return neuron IDs in dendrogram leaf order."""
        return [
            self.neuron_ids[int(index)]
            for index in self.reorder_indices.tolist()
        ]

    def labels_in_leaf_order(self) -> list[int]:
        """Return cluster labels in dendrogram leaf order."""
        return [
            int(self.labels[int(index)])
            for index in self.reorder_indices.tolist()
        ]


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
    logger.debug(
        "compute_linkage start: method=%s, distance_matrix shape=%s dtype=%s nbytes=%s",
        method,
        distance_matrix.shape,
        distance_matrix.dtype,
        _format_nbytes(int(distance_matrix.nbytes)),
    )
    squareform_start = perf_counter()
    condensed = squareform(distance_matrix, checks=False)
    squareform_elapsed = perf_counter() - squareform_start
    logger.debug(
        "compute_linkage squareform complete: condensed shape=%s dtype=%s nbytes=%s elapsed=%.3fs",
        condensed.shape,
        condensed.dtype,
        _format_nbytes(int(condensed.nbytes)),
        squareform_elapsed,
    )
    linkage_start = perf_counter()
    logger.debug(
        "compute_linkage calling scipy.linkage with method=%s", method
    )
    result = linkage(condensed, method=method)
    linkage_elapsed = perf_counter() - linkage_start
    logger.debug(
        "compute_linkage linkage complete: linkage shape=%s dtype=%s nbytes=%s elapsed=%.3fs",
        result.shape,
        result.dtype,
        _format_nbytes(int(result.nbytes)),
        linkage_elapsed,
    )
    return result


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
    return fcluster(linkage_matrix, t=n_clusters, criterion="maxclust").astype(
        np.int32
    )


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
        from sklearn.metrics import (
            adjusted_rand_score,
            normalized_mutual_info_score,
        )
    except ImportError:
        logger.warning(
            "scikit-learn not installed; skipping partition comparison"
        )
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


def _euclidean_condensed_distances(
    coords: NDArray[np.float64],
) -> NDArray[np.float32]:
    """Compute condensed pairwise Euclidean distances from 3-D coordinates."""
    return pdist(coords, metric="euclidean").astype(np.float32, copy=False)


def compute_linkage_from_condensed(
    condensed_distances: NDArray[np.floating],
    method: str = "average",
) -> NDArray[np.float64]:
    """Compute linkage directly from a condensed distance vector."""
    return linkage(np.asarray(condensed_distances), method=method)


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
    total_start = perf_counter()
    logger.debug(
        "cluster_somas_hierarchical start: neurons=%d coords shape=%s dtype=%s method=%s n_clusters=%d",
        len(neuron_ids),
        coords.shape,
        coords.dtype,
        method,
        n_clusters,
    )
    distance_start = perf_counter()
    condensed = _euclidean_condensed_distances(coords)
    dist = squareform(condensed, checks=False).astype(np.float32, copy=False)
    logger.debug(
        "Built Euclidean distance matrix for %d soma coordinates",
        len(neuron_ids),
    )
    distance_elapsed = perf_counter() - distance_start
    logger.debug(
        "cluster_somas_hierarchical distance build complete: elapsed=%.3fs distance_nbytes=%s",
        distance_elapsed,
        _format_nbytes(int(dist.nbytes)),
    )
    linkage_start = perf_counter()
    Z = compute_linkage_from_condensed(condensed, method=method)
    linkage_elapsed = perf_counter() - linkage_start
    logger.debug(
        "cluster_somas_hierarchical linkage complete: elapsed=%.3fs linkage_nbytes=%s",
        linkage_elapsed,
        _format_nbytes(int(Z.nbytes)),
    )
    fcluster_start = perf_counter()
    labels = fcluster(Z, t=n_clusters, criterion="maxclust").astype(np.int32)
    fcluster_elapsed = perf_counter() - fcluster_start
    logger.debug(
        "cluster_somas_hierarchical fcluster complete: elapsed=%.3fs labels shape=%s dtype=%s",
        fcluster_elapsed,
        labels.shape,
        labels.dtype,
    )
    reorder_start = perf_counter()
    reorder = leaves_list(Z)
    reorder_elapsed = perf_counter() - reorder_start
    logger.debug(
        "cluster_somas_hierarchical leaves_list complete: elapsed=%.3fs reorder shape=%s dtype=%s",
        reorder_elapsed,
        reorder.shape,
        reorder.dtype,
    )

    actual_k = int(len(np.unique(labels)))
    logger.info(
        f"Soma hierarchical clustering: {len(neuron_ids)} neurons, "
        f"{actual_k} clusters, method={method}"
    )
    logger.debug(
        "cluster_somas_hierarchical complete: total_elapsed=%.3fs cluster_sizes=%s",
        perf_counter() - total_start,
        dict(zip(*np.unique(labels, return_counts=True))),
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

    condensed = _euclidean_condensed_distances(coords)
    # Linkage is computed for clustermap dendrogram visualisation only
    Z = compute_linkage_from_condensed(condensed, method="average")
    reorder = leaves_list(Z)
    dist = squareform(condensed, checks=False).astype(np.float32, copy=False)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = (
        km.fit_predict(coords).astype(np.int32) + 1
    )  # 1-indexed like fcluster

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

    condensed = _euclidean_condensed_distances(coords)
    # Linkage is computed for clustermap dendrogram visualisation only
    Z = compute_linkage_from_condensed(condensed, method="average")
    reorder = leaves_list(Z)
    dist = squareform(condensed, checks=False).astype(np.float32, copy=False)

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
