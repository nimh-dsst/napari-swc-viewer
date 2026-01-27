"""napari-swc-viewer: A napari plugin for viewing SWC files."""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .db import NeuronDatabase
from .hemisphere import (
    Hemisphere,
    detect_hemisphere,
    detect_soma_hemisphere,
    flip_coordinates,
    flip_swc,
    flip_swc_batch,
    get_atlas_midline,
)
from .parquet import (
    NEURON_SCHEMA,
    get_parquet_summary,
    swc_files_to_parquet,
)
from .region import (
    build_region_lookup,
    get_region_at_coords,
    get_region_ids_vectorized,
    get_regions_for_coords,
    setup_allen_sdk,
)
from .swc import NodeType, SWCData, parse_swc, write_swc

__all__ = [
    "__version__",
    # SWC parsing
    "NodeType",
    "SWCData",
    "parse_swc",
    "write_swc",
    # Hemisphere operations
    "Hemisphere",
    "detect_hemisphere",
    "detect_soma_hemisphere",
    "flip_coordinates",
    "flip_swc",
    "flip_swc_batch",
    "get_atlas_midline",
    # Region mapping
    "setup_allen_sdk",
    "get_region_at_coords",
    "get_regions_for_coords",
    "get_region_ids_vectorized",
    "build_region_lookup",
    # Parquet conversion
    "NEURON_SCHEMA",
    "swc_files_to_parquet",
    "get_parquet_summary",
    # Database
    "NeuronDatabase",
]
