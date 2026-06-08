"""SWC file parsing and data structures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable
import warnings

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# SWC node type constants
class NodeType:
    """SWC node type identifiers."""

    UNDEFINED = 0
    SOMA = 1
    AXON = 2
    BASAL_DENDRITE = 3
    APICAL_DENDRITE = 4
    CUSTOM = 5
    UNSPECIFIED_NEURITE = 6
    GLIA = 7


STANDARD_NODE_TYPE_OPTIONS = (
    (NodeType.SOMA, "Soma"),
    (NodeType.AXON, "Axon"),
    (NodeType.BASAL_DENDRITE, "Basal dendrite"),
    (NodeType.APICAL_DENDRITE, "Apical dendrite"),
)

NODE_TYPE_LABELS = {
    NodeType.UNDEFINED: "Undefined",
    NodeType.SOMA: "Soma",
    NodeType.AXON: "Axon",
    NodeType.BASAL_DENDRITE: "Basal dendrite",
    NodeType.APICAL_DENDRITE: "Apical dendrite",
    NodeType.CUSTOM: "Custom",
    NodeType.UNSPECIFIED_NEURITE: "Unspecified neurite",
    NodeType.GLIA: "Glia processes",
}


def normalize_node_types(
    node_types: Iterable[int] | None,
) -> tuple[int, ...] | None:
    """Return sorted unique SWC node type IDs, or ``None`` for no filter."""
    if node_types is None:
        return None
    return tuple(sorted({int(node_type) for node_type in node_types}))


def node_type_label(node_type: int) -> str:
    """Return a display label for one SWC node type ID."""
    value = int(node_type)
    return NODE_TYPE_LABELS.get(value, f"Type {value}")


def node_type_labels(node_types: Iterable[int]) -> list[str]:
    """Return display labels for SWC node type IDs."""
    normalized = normalize_node_types(node_types)
    if normalized is None:
        return []
    return [node_type_label(node_type) for node_type in normalized]


@dataclass
class SWCData:
    """Container for SWC morphology data.

    Attributes
    ----------
    ids : NDArray[np.int32]
        Node identifiers (1-indexed in standard SWC).
    types : NDArray[np.int32]
        Node type codes (1=soma, 2=axon, 3=basal dendrite, etc.).
    coords : NDArray[np.float64]
        Node coordinates as (N, 3) array with columns [x, y, z].
    radii : NDArray[np.float64]
        Node radii.
    parents : NDArray[np.int32]
        Parent node identifiers (-1 for root nodes).
    """

    ids: NDArray[np.int32]
    types: NDArray[np.int32]
    coords: NDArray[np.float64]
    radii: NDArray[np.float64]
    parents: NDArray[np.int32]

    @property
    def n_nodes(self) -> int:
        """Return the number of nodes."""
        return len(self.ids)

    @property
    def soma_mask(self) -> NDArray[np.bool_]:
        """Return boolean mask for soma nodes."""
        return self.types == NodeType.SOMA

    @property
    def soma_coords(self) -> NDArray[np.float64]:
        """Return coordinates of soma nodes."""
        return self.coords[self.soma_mask]

    @property
    def root_mask(self) -> NDArray[np.bool_]:
        """Return boolean mask for root nodes (parent == -1)."""
        return self.parents == -1

    def copy(self) -> SWCData:
        """Return a deep copy of this SWC data."""
        return SWCData(
            ids=self.ids.copy(),
            types=self.types.copy(),
            coords=self.coords.copy(),
            radii=self.radii.copy(),
            parents=self.parents.copy(),
        )


def _empty_swc_data() -> SWCData:
    """Return an empty SWC container with stable array shapes."""
    return SWCData(
        ids=np.array([], dtype=np.int32),
        types=np.array([], dtype=np.int32),
        coords=np.empty((0, 3), dtype=np.float64),
        radii=np.array([], dtype=np.float64),
        parents=np.array([], dtype=np.int32),
    )


def _swc_array_to_data(data: NDArray[np.float64]) -> SWCData:
    """Convert a numeric SWC matrix into ``SWCData`` arrays."""
    if data.size == 0:
        return _empty_swc_data()

    if data.ndim == 1:
        data = data[np.newaxis, :]

    return SWCData(
        ids=np.asarray(data[:, 0], dtype=np.int32),
        types=np.asarray(data[:, 1], dtype=np.int32),
        coords=np.asarray(data[:, 2:5], dtype=np.float64),
        radii=np.asarray(data[:, 5], dtype=np.float64),
        parents=np.asarray(data[:, 6], dtype=np.int32),
    )


def _parse_swc_fast(filepath: Path) -> SWCData:
    """Parse a regular SWC using NumPy's text loader."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        data = np.loadtxt(
            filepath,
            comments="#",
            usecols=(0, 1, 2, 3, 4, 5, 6),
            dtype=np.float64,
        )

    return _swc_array_to_data(data)


def _parse_swc_tolerant(filepath: Path) -> SWCData:
    """Parse SWC text line-by-line, skipping short or empty rows."""
    ids = []
    types = []
    coords = []
    radii = []
    parents = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 7:
                continue

            ids.append(int(parts[0]))
            types.append(int(parts[1]))
            coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
            radii.append(float(parts[5]))
            parents.append(int(parts[6]))

    if not ids:
        return _empty_swc_data()

    return SWCData(
        ids=np.array(ids, dtype=np.int32),
        types=np.array(types, dtype=np.int32),
        coords=np.array(coords, dtype=np.float64),
        radii=np.array(radii, dtype=np.float64),
        parents=np.array(parents, dtype=np.int32),
    )


def parse_swc(filepath: str | Path) -> SWCData:
    """Parse an SWC file into structured data.

    Parameters
    ----------
    filepath : str or Path
        Path to the SWC file.

    Returns
    -------
    SWCData
        Parsed SWC morphology data.

    Notes
    -----
    SWC format: each data line contains space-separated values:
        id type x y z radius parent_id

    Comment lines start with '#' and are ignored.
    """
    filepath = Path(filepath)

    try:
        return _parse_swc_fast(filepath)
    except (IndexError, ValueError):
        return _parse_swc_tolerant(filepath)


def write_swc(swc_data: SWCData, filepath: str | Path) -> None:
    """Write SWC data to a file.

    Parameters
    ----------
    swc_data : SWCData
        The SWC data to write.
    filepath : str or Path
        Output file path.
    """
    filepath = Path(filepath)

    with open(filepath, "w") as f:
        f.write("# SWC file generated by napari-swc-viewer\n")
        f.write("# id type x y z radius parent\n")

        for i in range(swc_data.n_nodes):
            f.write(
                f"{swc_data.ids[i]} {swc_data.types[i]} "
                f"{swc_data.coords[i, 0]:.3f} {swc_data.coords[i, 1]:.3f} "
                f"{swc_data.coords[i, 2]:.3f} {swc_data.radii[i]:.3f} "
                f"{swc_data.parents[i]}\n"
            )
