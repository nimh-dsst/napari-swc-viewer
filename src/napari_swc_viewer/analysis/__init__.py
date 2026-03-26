"""Spatial analysis pipeline for neuron clustering and heatmap generation.

Ported from the swc-mapper repository, adapted to use BrainGlobe Atlas API
instead of Allen SDK.
"""

from .clustering import (
    ClusterResult,
    cluster_somas_dbscan,
    cluster_somas_hierarchical,
    cluster_somas_kmeans,
    compute_clustermap_data,
    compute_linkage,
)
from .correlation import (
    compute_pearson_correlation_matrix,
    correlation_long_to_matrix,
)
from .heatmap import build_node_counts_volume
from .mask import (
    build_binary_mask_from_heatmap,
    build_binary_mask_from_threshold_range,
    dilate_mask_to_volume_increase,
    get_expanded_region_voxel_ids,
    get_region_mask,
    merge_heatmap_volumes,
    otsu_threshold_positive,
    smooth_heatmap_volume,
)

__all__ = [
    "get_region_mask",
    "dilate_mask_to_volume_increase",
    "get_expanded_region_voxel_ids",
    "smooth_heatmap_volume",
    "merge_heatmap_volumes",
    "otsu_threshold_positive",
    "build_binary_mask_from_heatmap",
    "build_binary_mask_from_threshold_range",
    "compute_pearson_correlation_matrix",
    "correlation_long_to_matrix",
    "build_node_counts_volume",
    "compute_linkage",
    "compute_clustermap_data",
    "cluster_somas_hierarchical",
    "cluster_somas_kmeans",
    "cluster_somas_dbscan",
    "ClusterResult",
]
