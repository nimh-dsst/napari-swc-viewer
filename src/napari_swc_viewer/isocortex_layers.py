"""Allen Isocortex laminar-region classification helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any

ALLEN_ISOCORTEX_LAYER_KEYS = ("1", "2/3", "4", "5", "6a", "6b")
ALLEN_ISOCORTEX_LAYER_LABELS = ("L1", "L2/3", "L4", "L5", "L6a", "L6b")

_LAYER_NAME_PATTERN = re.compile(
    r"(?:,|/)\s*(?:layer\s+)?(2/3|6a|6b|1|4|5)\s*$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class AllenIsocortexLayerMap:
    """Atlas-derived mapping from terminal Isocortex region IDs to layers."""

    atlas_name: str
    isocortex_region_id: int
    region_to_layer_index: dict[int, int]
    region_ids_by_layer: tuple[tuple[int, ...], ...]
    layer_keys: tuple[str, ...] = ALLEN_ISOCORTEX_LAYER_KEYS
    layer_labels: tuple[str, ...] = ALLEN_ISOCORTEX_LAYER_LABELS
    atlas_version: str = ""

    @property
    def region_count(self) -> int:
        """Return the number of terminal regions assigned to a layer."""
        return len(self.region_to_layer_index)


@dataclass(frozen=True)
class CustomRegionHierarchyNode:
    """One synthetic group or terminal atlas region in a custom hierarchy."""

    label: str
    acronym: str = ""
    region_id: int | None = None
    children: tuple[CustomRegionHierarchyNode, ...] = ()

    @property
    def is_terminal(self) -> bool:
        """Return whether this node represents one exact atlas region."""
        return self.region_id is not None

    @property
    def terminal_region_ids(self) -> tuple[int, ...]:
        """Return the exact terminal atlas IDs represented by this node."""
        if self.region_id is not None:
            return (int(self.region_id),)
        return tuple(
            region_id
            for child in self.children
            for region_id in child.terminal_region_ids
        )


@dataclass(frozen=True)
class CustomRegionHierarchy:
    """Atlas-derived custom region hierarchy and its provenance."""

    root: CustomRegionHierarchyNode
    atlas_name: str
    atlas_version: str = ""

    @property
    def terminal_region_ids(self) -> tuple[int, ...]:
        """Return every exact atlas region represented by the hierarchy."""
        return self.root.terminal_region_ids

    @property
    def terminal_region_count(self) -> int:
        """Return the number of exact terminal atlas regions."""
        return len(self.terminal_region_ids)


@dataclass(frozen=True)
class CustomRegionSelectionGroup:
    """Selected terminal atlas regions belonging to one synthetic layer."""

    label: str
    region_ids: tuple[int, ...]
    acronyms: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.region_ids) != len(self.acronyms):
            raise ValueError(
                "Custom region selection IDs and acronyms must have equal length."
            )
        if len(self.region_ids) != len(set(self.region_ids)):
            raise ValueError(
                "Custom region selection groups cannot contain duplicate IDs."
            )

    @property
    def region_count(self) -> int:
        """Return the number of selected terminal regions in this layer."""
        return len(self.region_ids)


def _structure_items(structures: object) -> list[tuple[object, Mapping[str, Any]]]:
    if isinstance(structures, Mapping):
        raw_items = structures.items()
    elif isinstance(structures, (list, tuple)):
        raw_items = enumerate(structures)
    else:
        raise ValueError("The loaded atlas does not expose a usable structure catalog.")
    return [(key, value) for key, value in raw_items if isinstance(value, Mapping)]


def _structure_id(key: object, structure: Mapping[str, Any]) -> int | None:
    try:
        return int(structure.get("id", key))
    except (TypeError, ValueError):
        return None


def _structure_path(structure: Mapping[str, Any]) -> tuple[int, ...]:
    raw_path = structure.get("structure_id_path", ()) or ()
    if isinstance(raw_path, str):
        parts = [part for part in raw_path.strip("/").split("/") if part]
    else:
        try:
            parts = list(raw_path)
        except TypeError:
            return ()
    try:
        return tuple(int(value) for value in parts)
    except (TypeError, ValueError):
        return ()


def _layer_key_from_name(name: object) -> str | None:
    match = _LAYER_NAME_PATTERN.search(str(name or "").strip())
    if match is None:
        return None
    return match.group(1).lower()


def build_allen_isocortex_layer_map(
    structures: object,
    *,
    atlas_name: str = "",
    atlas_version: str = "",
) -> AllenIsocortexLayerMap:
    """Build the six-layer mapping from an Allen-compatible structure catalog.

    Only terminal descendants of the catalog's ``Isocortex`` structure are
    eligible. Allen naming variants such as ``, layer 1``, ``/Layer 1``, and
    ``, 6a`` are recognized.
    """
    items = _structure_items(structures)
    records: dict[int, tuple[Mapping[str, Any], tuple[int, ...]]] = {}
    isocortex_ids: list[int] = []
    for key, structure in items:
        region_id = _structure_id(key, structure)
        if region_id is None:
            continue
        path = _structure_path(structure)
        records[region_id] = (structure, path)
        acronym = str(structure.get("acronym", "") or "").strip().lower()
        name = str(structure.get("name", "") or "").strip().lower()
        if acronym == "isocortex" or name == "isocortex":
            isocortex_ids.append(region_id)

    if len(set(isocortex_ids)) != 1:
        raise ValueError(
            "The loaded atlas must contain exactly one Isocortex structure."
        )
    isocortex_id = int(isocortex_ids[0])

    descendants = {
        region_id: (structure, path)
        for region_id, (structure, path) in records.items()
        if isocortex_id in path
    }
    if not descendants:
        raise ValueError("The loaded atlas does not contain descendants of Isocortex.")

    parent_ids = {
        ancestor
        for _region_id, (_structure, path) in descendants.items()
        for ancestor in path[:-1]
        if ancestor in descendants
    }
    layer_index_by_key = {
        key: index for index, key in enumerate(ALLEN_ISOCORTEX_LAYER_KEYS)
    }
    region_to_layer: dict[int, int] = {}
    ids_by_layer: list[list[int]] = [[] for _label in ALLEN_ISOCORTEX_LAYER_LABELS]
    for region_id, (structure, _path) in descendants.items():
        if region_id in parent_ids:
            continue
        layer_key = _layer_key_from_name(structure.get("name"))
        if layer_key is None:
            continue
        layer_index = layer_index_by_key[layer_key]
        region_to_layer[region_id] = layer_index
        ids_by_layer[layer_index].append(region_id)

    missing = [
        ALLEN_ISOCORTEX_LAYER_LABELS[index]
        for index, region_ids in enumerate(ids_by_layer)
        if not region_ids
    ]
    if missing:
        raise ValueError(
            "The loaded atlas does not contain terminal Isocortex regions for "
            f"layer(s): {', '.join(missing)}."
        )

    return AllenIsocortexLayerMap(
        atlas_name=str(atlas_name or ""),
        isocortex_region_id=isocortex_id,
        region_to_layer_index=region_to_layer,
        region_ids_by_layer=tuple(
            tuple(sorted(region_ids)) for region_ids in ids_by_layer
        ),
        atlas_version=str(atlas_version or ""),
    )


def _atlas_version_text(atlas: object) -> str:
    value = (
        getattr(atlas, "local_version", None)
        or getattr(atlas, "atlas_version", None)
        or getattr(atlas, "version", None)
    )
    if isinstance(value, (tuple, list)):
        return ".".join(str(part).strip() for part in value)
    return str(value or "").strip()


def layer_map_from_atlas(atlas: object | None) -> AllenIsocortexLayerMap:
    """Return an Isocortex layer map for a loaded atlas object."""
    if atlas is None:
        raise ValueError(
            "Load an Allen mouse atlas before rendering an Allen layer heatmap."
        )
    atlas_name = str(getattr(atlas, "atlas_name", "") or "")
    structures = getattr(atlas, "structures", None)
    return build_allen_isocortex_layer_map(
        structures,
        atlas_name=atlas_name,
        atlas_version=_atlas_version_text(atlas),
    )


def build_isocortex_layer_hierarchy(
    structures: object,
    layer_map: AllenIsocortexLayerMap,
) -> CustomRegionHierarchy:
    """Build ``Isocortex Layers → layer → terminal region`` hierarchy data."""
    structures_by_id: dict[int, Mapping[str, Any]] = {}
    for key, structure in _structure_items(structures):
        region_id = _structure_id(key, structure)
        if region_id is not None:
            structures_by_id[region_id] = structure

    layer_nodes: list[CustomRegionHierarchyNode] = []
    seen_region_ids: set[int] = set()
    for layer_label, region_ids in zip(
        layer_map.layer_labels,
        layer_map.region_ids_by_layer,
        strict=True,
    ):
        leaves: list[CustomRegionHierarchyNode] = []
        for region_id in region_ids:
            normalized_id = int(region_id)
            if normalized_id in seen_region_ids:
                raise ValueError(
                    f"Region ID {normalized_id} appears in more than one layer."
                )
            structure = structures_by_id.get(normalized_id)
            if structure is None:
                raise ValueError(
                    f"Mapped region ID {normalized_id} is absent from the "
                    "loaded atlas structure catalog."
                )
            name = str(structure.get("name", "") or "").strip()
            acronym = str(structure.get("acronym", "") or "").strip()
            if not name or not acronym:
                raise ValueError(
                    f"Mapped region ID {normalized_id} has no usable name or acronym."
                )
            leaves.append(
                CustomRegionHierarchyNode(
                    label=name,
                    acronym=acronym,
                    region_id=normalized_id,
                )
            )
            seen_region_ids.add(normalized_id)
        leaves.sort(
            key=lambda node: (
                node.label.casefold(),
                node.acronym.casefold(),
                int(node.region_id or -1),
            )
        )
        layer_nodes.append(
            CustomRegionHierarchyNode(
                label=layer_label,
                children=tuple(leaves),
            )
        )

    expected_ids = set(layer_map.region_to_layer_index)
    if seen_region_ids != expected_ids:
        missing = sorted(expected_ids - seen_region_ids)
        unexpected = sorted(seen_region_ids - expected_ids)
        raise ValueError(
            "Custom hierarchy does not match the Allen layer mapping "
            f"(missing={missing}, unexpected={unexpected})."
        )

    root = CustomRegionHierarchyNode(
        label="Isocortex Layers",
        children=tuple(layer_nodes),
    )
    return CustomRegionHierarchy(
        root=root,
        atlas_name=layer_map.atlas_name,
        atlas_version=layer_map.atlas_version,
    )


def isocortex_layer_hierarchy_from_atlas(
    atlas: object | None,
) -> CustomRegionHierarchy:
    """Build the custom Isocortex layer hierarchy from one loaded atlas."""
    layer_map = layer_map_from_atlas(atlas)
    structures = getattr(atlas, "structures", None)
    return build_isocortex_layer_hierarchy(structures, layer_map)
