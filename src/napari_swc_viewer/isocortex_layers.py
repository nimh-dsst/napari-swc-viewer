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
