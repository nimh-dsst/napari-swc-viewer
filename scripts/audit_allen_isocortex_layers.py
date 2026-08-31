#!/usr/bin/env python
"""Print the terminal regions assigned to each Allen Isocortex layer.

Run from the repository root with:

    pixi run python scripts/audit_allen_isocortex_layers.py
"""

from __future__ import annotations

from collections.abc import Mapping
import sys
from typing import Any

from brainglobe_atlasapi import BrainGlobeAtlas

from napari_neuron_navigator.isocortex_layers import (
    AllenIsocortexLayerMap,
    layer_map_from_atlas,
)

ATLAS_NAME = "allen_mouse_25um"


def _structures_by_id(structures: object) -> dict[int, Mapping[str, Any]]:
    """Return structure records keyed by integer region ID."""
    if isinstance(structures, Mapping):
        values = structures.values()
    elif isinstance(structures, (list, tuple)):
        values = structures
    else:
        raise ValueError("The atlas does not expose a usable structure catalog.")

    result: dict[int, Mapping[str, Any]] = {}
    for structure in values:
        if not isinstance(structure, Mapping):
            continue
        try:
            region_id = int(structure["id"])
        except (KeyError, TypeError, ValueError):
            continue
        result[region_id] = structure
    return result


def format_layer_report(
    structures: object,
    layer_map: AllenIsocortexLayerMap,
) -> str:
    """Format all matched regions as a human-readable layer audit."""
    structures_by_id = _structures_by_id(structures)
    version_suffix = f" v{layer_map.atlas_version}" if layer_map.atlas_version else ""
    lines = [
        "Allen Isocortex layer region audit",
        f"Atlas: {layer_map.atlas_name or ATLAS_NAME}{version_suffix}",
        f"Isocortex region ID: {layer_map.isocortex_region_id}",
        f"Matched terminal regions: {layer_map.region_count}",
    ]
    layer_counts: list[tuple[str, int]] = []

    for label, region_ids in zip(
        layer_map.layer_labels,
        layer_map.region_ids_by_layer,
        strict=True,
    ):
        records: list[tuple[str, int, str, str]] = []
        for region_id in region_ids:
            structure = structures_by_id.get(region_id)
            if structure is None:
                raise RuntimeError(
                    f"Mapped region ID {region_id} is absent from the atlas catalog."
                )
            acronym = str(structure.get("acronym", "") or "")
            name = str(structure.get("name", "") or "")
            records.append((name.casefold(), region_id, acronym, name))
        records.sort()
        id_width = max(
            len("ID"),
            *(
                len(str(region_id))
                for _sort_name, region_id, _acronym, _name in records
            ),
        )
        acronym_width = max(
            len("ACRONYM"),
            *(len(acronym) for _sort_name, _region_id, acronym, _name in records),
        )

        lines.extend(
            [
                "",
                f"{label} ({len(records)} regions)",
                "-" * (len(label) + len(str(len(records))) + 11),
                f"  {'ID':>{id_width}}  {'ACRONYM':<{acronym_width}}  NAME",
            ]
        )
        for _sort_name, region_id, acronym, name in records:
            lines.append(
                f"  {region_id:>{id_width}}  {acronym:<{acronym_width}}  {name}"
            )
        layer_counts.append((label, len(records)))

    lines.extend(["", "Summary", "-------"])
    lines.extend(f"  {label:<4} {count:>3}" for label, count in layer_counts)
    lines.append(f"  {'Total':<4} {sum(count for _label, count in layer_counts):>3}")
    return "\n".join(lines)


def main() -> int:
    """Load the 25 µm Allen atlas and print its layer assignments."""
    try:
        atlas = BrainGlobeAtlas(ATLAS_NAME, check_latest=False)
        layer_map = layer_map_from_atlas(atlas)
        print(format_layer_report(atlas.structures, layer_map))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
