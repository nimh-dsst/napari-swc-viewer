"""napari-swc-viewer: A napari plugin for viewing SWC files."""

import logging

from .logging_utils import configure_debug_logging, startup_timing

configure_debug_logging()
logger = logging.getLogger(__name__)

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

with startup_timing(logger, "package_import", module="swc"):
    from .swc import NodeType, SWCData, parse_swc, write_swc

with startup_timing(logger, "package_import", module="hemisphere"):
    from .hemisphere import (
        Hemisphere,
        detect_hemisphere,
        detect_soma_hemisphere,
        flip_coordinates,
        flip_swc,
        flip_swc_batch,
        get_atlas_midline,
    )

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
