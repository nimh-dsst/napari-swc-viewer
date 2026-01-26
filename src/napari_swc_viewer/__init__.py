"""napari-swc-viewer: A napari plugin for viewing SWC files."""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .hemisphere import (
    Hemisphere,
    detect_hemisphere,
    detect_soma_hemisphere,
    flip_coordinates,
    flip_swc,
    flip_swc_batch,
    get_atlas_midline,
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
]
