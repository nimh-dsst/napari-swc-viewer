"""Spatial analysis pipeline for neuron clustering and heatmap generation.

Ported from the swc-mapper repository, adapted to use BrainGlobe Atlas API
instead of Allen SDK.
"""

from .clustering import ClusterResult, compute_clustermap_data, compute_linkage
from .correlation import (
    compute_pearson_correlation_matrix,
    correlation_long_to_matrix,
)
from .heatmap import build_node_counts_volume
from .mask import (
    dilate_mask_to_volume_increase,
    get_expanded_region_voxel_ids,
    get_region_mask,
)

__all__ = [
    "get_region_mask",
    "dilate_mask_to_volume_increase",
    "get_expanded_region_voxel_ids",
    "compute_pearson_correlation_matrix",
    "correlation_long_to_matrix",
    "build_node_counts_volume",
    "compute_linkage",
    "compute_clustermap_data",
    "ClusterResult",
]
