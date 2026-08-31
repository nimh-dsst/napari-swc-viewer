from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from napari_neuron_navigator.isocortex_layers import (
    ALLEN_ISOCORTEX_LAYER_LABELS,
    CustomRegionSelectionGroup,
    build_isocortex_layer_hierarchy,
    build_allen_isocortex_layer_map,
    isocortex_layer_hierarchy_from_atlas,
    layer_map_from_atlas,
)


def _structure(
    region_id: int,
    acronym: str,
    name: str,
    path: list[int],
) -> dict[str, object]:
    return {
        "id": region_id,
        "acronym": acronym,
        "name": name,
        "structure_id_path": path,
    }


def _catalog() -> dict[int, dict[str, object]]:
    return {
        997: _structure(997, "root", "root", [997]),
        315: _structure(315, "Isocortex", "Isocortex", [997, 315]),
        500: _structure(500, "AREA", "Example area", [997, 315, 500]),
        101: _structure(
            101,
            "AREA1",
            "Example area, layer 1",
            [997, 315, 500, 101],
        ),
        102: _structure(
            102,
            "AREA2/3",
            "Example area/Layer 2/3",
            [997, 315, 500, 102],
        ),
        103: _structure(
            103,
            "AREA4",
            "Example area, Layer 4",
            [997, 315, 500, 103],
        ),
        104: _structure(
            104,
            "AREA5",
            "Example area, layer 5",
            [997, 315, 500, 104],
        ),
        105: _structure(
            105,
            "AREA6a",
            "Example area, 6a",
            [997, 315, 500, 105],
        ),
        106: _structure(
            106,
            "AREA6b",
            "Example area, 6b",
            [997, 315, 500, 106],
        ),
        700: _structure(
            700,
            "OTHER1",
            "Unrelated region, layer 1",
            [997, 700],
        ),
    }


def test_build_allen_layer_map_uses_terminal_isocortex_regions() -> None:
    layer_map = build_allen_isocortex_layer_map(
        _catalog(),
        atlas_name="allen_mouse_25um",
    )

    assert layer_map.layer_labels == ALLEN_ISOCORTEX_LAYER_LABELS
    assert layer_map.atlas_name == "allen_mouse_25um"
    assert layer_map.isocortex_region_id == 315
    assert layer_map.region_count == 6
    assert layer_map.region_to_layer_index == {
        101: 0,
        102: 1,
        103: 2,
        104: 3,
        105: 4,
        106: 5,
    }
    assert 500 not in layer_map.region_to_layer_index
    assert 700 not in layer_map.region_to_layer_index


def test_build_allen_layer_map_rejects_incomplete_catalog() -> None:
    catalog = _catalog()
    del catalog[103]

    with pytest.raises(ValueError, match="L4"):
        build_allen_isocortex_layer_map(catalog)


def test_layer_map_from_atlas_requires_loaded_atlas() -> None:
    with pytest.raises(ValueError, match="Load an Allen mouse atlas"):
        layer_map_from_atlas(None)


def test_layer_map_from_atlas_records_atlas_identity() -> None:
    class Atlas:
        atlas_name = "allen_mouse_25um"
        local_version = (1, 2, 3)
        structures = _catalog()

    layer_map = layer_map_from_atlas(Atlas())

    assert layer_map.atlas_name == "allen_mouse_25um"
    assert layer_map.atlas_version == "1.2.3"


def test_build_isocortex_layer_hierarchy_uses_exact_terminal_regions() -> None:
    catalog = _catalog()
    catalog[107] = _structure(
        107,
        "AAA1",
        "Aardvark area, layer 1",
        [997, 315, 501, 107],
    )
    catalog[501] = _structure(
        501,
        "AAA",
        "Aardvark area",
        [997, 315, 501],
    )
    layer_map = build_allen_isocortex_layer_map(
        catalog,
        atlas_name="allen_mouse_25um",
        atlas_version="1.2",
    )

    hierarchy = build_isocortex_layer_hierarchy(catalog, layer_map)

    assert hierarchy.root.label == "Isocortex Layers"
    assert [node.label for node in hierarchy.root.children] == list(
        ALLEN_ISOCORTEX_LAYER_LABELS
    )
    assert [leaf.label for leaf in hierarchy.root.children[0].children] == [
        "Aardvark area, layer 1",
        "Example area, layer 1",
    ]
    for layer_node, expected_ids in zip(
        hierarchy.root.children,
        layer_map.region_ids_by_layer,
        strict=True,
    ):
        assert set(layer_node.terminal_region_ids) == set(expected_ids)
    assert set(hierarchy.terminal_region_ids) == set(layer_map.region_to_layer_index)
    assert 315 not in hierarchy.terminal_region_ids
    assert 500 not in hierarchy.terminal_region_ids
    assert 700 not in hierarchy.terminal_region_ids
    assert hierarchy.terminal_region_count == 7
    assert hierarchy.atlas_name == "allen_mouse_25um"
    assert hierarchy.atlas_version == "1.2"


def test_isocortex_layer_hierarchy_from_atlas_uses_loaded_catalog() -> None:
    class Atlas:
        atlas_name = "allen_mouse_25um"
        local_version = (1, 2)
        structures = _catalog()

    hierarchy = isocortex_layer_hierarchy_from_atlas(Atlas())

    assert hierarchy.terminal_region_count == 6
    assert hierarchy.root.children[0].children[0].region_id == 101


def test_custom_region_selection_group_is_immutable_and_aligned() -> None:
    group = CustomRegionSelectionGroup(
        label="L1",
        region_ids=(101, 102),
        acronyms=("AREA1", "OTHER1"),
    )

    assert group.region_count == 2
    with pytest.raises(FrozenInstanceError):
        group.label = "L2/3"
    with pytest.raises(ValueError, match="equal length"):
        CustomRegionSelectionGroup(
            label="L1",
            region_ids=(101,),
            acronyms=(),
        )
