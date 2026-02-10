"""napari reader hook implementation for SWC and Parquet files.

This module provides napari reader hooks for:
1. SWC files - rendered as shapes (lines) or points
2. Parquet files - rendered with neurons as shapes layers
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .swc import NodeType, parse_swc

if TYPE_CHECKING:
    from napari.types import LayerDataTuple


def napari_get_reader(
    path: str | list[str],
) -> Callable[[str | list[str]], list[LayerDataTuple]] | None:
    """Return a reader function if the path is a supported file type.

    Parameters
    ----------
    path : str or list of str
        Path(s) to the file(s) to read.

    Returns
    -------
    callable or None
        Reader function if the file type is supported, None otherwise.
    """
    if isinstance(path, list):
        # Handle multiple files
        paths = [Path(p) for p in path]
        suffixes = {p.suffix.lower() for p in paths}

        if suffixes == {".swc"}:
            return read_swc_files
        if suffixes == {".parquet"}:
            return read_parquet_file

        return None

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".swc":
        return read_swc_file
    if suffix == ".parquet":
        return read_parquet_file

    return None


def read_swc_file(path: str | list[str]) -> list[LayerDataTuple]:
    """Read a single SWC file and return napari layer data.

    Parameters
    ----------
    path : str or list of str
        Path to the SWC file.

    Returns
    -------
    list of LayerDataTuple
        List of layer data tuples for napari.
    """
    if isinstance(path, list):
        path = path[0]

    filepath = Path(path)
    swc_data = parse_swc(filepath)

    # Build lines from parent-child relationships
    lines = []
    id_to_idx = {node_id: idx for idx, node_id in enumerate(swc_data.ids)}

    for idx, parent_id in enumerate(swc_data.parents):
        if parent_id in id_to_idx:
            parent_idx = id_to_idx[parent_id]
            lines.append([swc_data.coords[parent_idx], swc_data.coords[idx]])

    if not lines:
        # Fall back to points if no lines can be constructed
        return [
            (
                swc_data.coords,
                {
                    "size": 2,
                    "face_color": "cyan",
                    "name": filepath.stem,
                },
                "points",
            )
        ]

    # Create shapes layer with lines
    layer_data: LayerDataTuple = (
        lines,
        {
            "shape_type": "line",
            "edge_width": 1,
            "edge_color": "cyan",
            "name": filepath.stem,
        },
        "shapes",
    )

    return [layer_data]


def read_swc_files(paths: str | list[str]) -> list[LayerDataTuple]:
    """Read multiple SWC files and return napari layer data.

    Parameters
    ----------
    paths : str or list of str
        Paths to the SWC files.

    Returns
    -------
    list of LayerDataTuple
        List of layer data tuples for napari.
    """
    if isinstance(paths, str):
        paths = [paths]

    layers = []
    for path in paths:
        layers.extend(read_swc_file(path))

    return layers


def read_parquet_file(path: str | list[str]) -> list[LayerDataTuple]:
    """Read a Parquet file with neuron data and return napari layer data.

    This function reads the Parquet file and creates a points layer
    showing all soma locations. The full neuron data can be explored
    using the NeuronViewerWidget.

    Parameters
    ----------
    path : str or list of str
        Path to the Parquet file.

    Returns
    -------
    list of LayerDataTuple
        List of layer data tuples for napari.
    """
    if isinstance(path, list):
        path = path[0]

    filepath = Path(path)

    from .db import NeuronDatabase

    try:
        db = NeuronDatabase(filepath)
    except FileNotFoundError:
        return []

    # Get soma locations for initial display
    somas = db.get_soma_locations()
    db.close()

    if somas.empty:
        return []

    coords = somas[["x", "y", "z"]].values

    # Create points layer for somas
    layer_data: LayerDataTuple = (
        coords,
        {
            "size": 10,
            "face_color": "red",
            "name": f"Somas: {filepath.stem}",
            "properties": {
                "file_id": somas["file_id"].tolist(),
                "neuron_id": somas["neuron_id"].tolist(),
                "region": somas["region_acronym"].tolist(),
            },
        },
        "points",
    )

    return [layer_data]


def swc_to_shapes_data(
    path: str | Path,
    color_by_type: bool = True,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Convert SWC file to shapes layer data with optional coloring.

    Parameters
    ----------
    path : str or Path
        Path to the SWC file.
    color_by_type : bool, default=True
        If True, color lines by node type.

    Returns
    -------
    tuple
        (lines, properties) for napari shapes layer.
    """
    swc_data = parse_swc(path)

    # Build lines and colors
    lines = []
    colors = []
    id_to_idx = {node_id: idx for idx, node_id in enumerate(swc_data.ids)}

    # Node type colors
    type_colors = {
        NodeType.SOMA: "red",
        NodeType.AXON: "blue",
        NodeType.BASAL_DENDRITE: "green",
        NodeType.APICAL_DENDRITE: "yellow",
        NodeType.UNDEFINED: "gray",
        NodeType.CUSTOM: "purple",
        NodeType.UNSPECIFIED_NEURITE: "orange",
        NodeType.GLIA: "pink",
    }

    for idx, parent_id in enumerate(swc_data.parents):
        if parent_id in id_to_idx:
            parent_idx = id_to_idx[parent_id]
            lines.append(
                np.array([swc_data.coords[parent_idx], swc_data.coords[idx]])
            )
            if color_by_type:
                node_type = swc_data.types[idx]
                colors.append(type_colors.get(node_type, "gray"))
            else:
                colors.append("cyan")

    properties = {
        "shape_type": "line",
        "edge_width": 1,
        "edge_color": colors if color_by_type else "cyan",
        "name": Path(path).stem,
    }

    return lines, properties
