from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from napari_swc_viewer.flatmap_region_cache import (
    REGION_CACHE_MANIFEST_FILENAME,
    RegionCacheCancelled,
    RegionCacheValidationError,
    build_region_cache_profile,
    materialize_region_outlines,
    materialize_region_selection,
    materialize_region_surface,
    open_region_cache,
    structure_catalog_id,
)


def _lookup_arrays():
    shape = (2, 2, 2)
    shaped = np.empty((*shape, 2), dtype=np.float32)
    square = np.empty((*shape, 2), dtype=np.float32)
    for first in range(shape[0]):
        for second in range(shape[1]):
            shaped[first, second, :, 0] = first
            shaped[first, second, :, 1] = second
            square[first, second, :, 0] = first * 2
            square[first, second, :, 1] = second * 3
    depth = np.zeros(shape, dtype=np.float32)
    depth[:, :, 1] = 10.0
    # Region 3 at this voxel recovers depth zero from the opposite hemisphere.
    depth[0, 0, 1] = -1.0
    # A region-4 source voxel is deliberately outside both flatmap lookups.
    shaped[1, 0, 1] = (-1.0, -1.0)
    square[1, 0, 1] = (-1.0, -1.0)
    annotation = np.array(
        [
            [[2, 3], [2, 0]],
            [[3, 4], [0, 4]],
        ],
        dtype=np.int32,
    )
    return annotation, shaped, square, depth


def _build(cache_dir: Path, **kwargs):
    annotation, shaped, square, depth = _lookup_arrays()
    return build_region_cache_profile(
        cache_dir,
        annotation=annotation,
        shaped_flatmap=shaped,
        square_flatmap=square,
        depth=depth,
        lookup_set_id="portable-lookup-set",
        atlas_name="test_mouse",
        atlas_version="1.2",
        atlas_resolution_um=(10, 10, 10),
        region_descendants={
            1: {2, 3},
            2: {2},
            3: {3},
            4: {4},
            5: {5},
        },
        xy_bins=2,
        depth_bin_um=10,
        chunk_voxels=2,
        **kwargs,
    )


def test_region_cache_round_trip_and_parent_geometry(tmp_path):
    profile = _build(tmp_path / "cache")
    assert profile.lookup_set_id == "portable-lookup-set"
    assert set(profile.styles) == {"shaped", "square"}
    assert profile.style("both_shaped").output_shape == (2, 2, 2)
    assert profile.style("square").grid_spec["x_bounds"] == [0.0, 2.0]

    cache = open_region_cache(tmp_path / "cache")
    assert list(cache.profiles) == [profile.profile_id]
    style = cache.profile().style("shaped")
    assert isinstance(style.array("occupancy_linear_bins"), np.memmap)
    assert style._data["counts"]["mirrored_depth_source_voxels"] == 1

    result = materialize_region_selection(
        cache,
        [2, 3],
        style="shaped",
        direct_region_ids=[1, 5],
    )
    assert result.labels.shape == (2, 2, 2)
    # Region 2 and 3 each contribute one voxel to bin zero: smaller ID wins.
    assert result.labels.reshape(-1)[0] == 2
    assert result.summary.collision_bins == 1
    assert result.represented_region_ids == (2, 3)
    assert [surface.region_id for surface in result.surfaces] == [1, 5]
    assert result.surfaces[0].vertices.shape[1:] == (3,)
    assert result.surfaces[0].faces.shape[1:] == (3,)
    assert result.surfaces[1].vertices.shape == (0, 3)
    assert result.outlines[0].vectors.shape[1:] == (2, 3)
    assert result.outlines[1].vectors.shape == (0, 2, 3)
    assert np.all(result.surfaces[0].faces >= 0)
    assert np.max(result.surfaces[0].faces) < len(result.surfaces[0].vertices)

    assert materialize_region_surface(profile, 1, style="shaped").component_count >= 1
    assert materialize_region_outlines(profile, 1, style="shaped").vectors.shape[
        1:
    ] == (2, 3)


def test_surface_and_outline_are_voxel_faithful_for_adjacent_bins():
    from napari_swc_viewer.flatmap_region_cache import (
        _outlines_for_bins,
        _surface_for_bins,
    )

    vertices, faces = _surface_for_bins(np.array([0, 1]), (1, 2, 2))
    outlines = _outlines_for_bins(np.array([0, 1]), (1, 2, 2))
    # Two cubes sharing one face expose ten quads and have six XY perimeter edges.
    assert vertices.shape == (12, 3)
    assert faces.shape == (20, 3)
    assert outlines.shape == (6, 2, 3)
    undirected_edges = np.sort(
        np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0),
        axis=1,
    )
    _edges, counts = np.unique(undirected_edges, axis=0, return_counts=True)
    assert np.all(counts == 2)


def test_multiple_profiles_duplicate_rejection_and_relocation(tmp_path):
    root = tmp_path / "cache"
    first = _build(root)
    with pytest.raises(FileExistsError, match="already exists"):
        _build(root)

    annotation, shaped, square, depth = _lookup_arrays()
    second = build_region_cache_profile(
        root,
        annotation=annotation,
        shaped_flatmap=shaped,
        square_flatmap=square,
        depth=depth,
        lookup_set_id="portable-lookup-set",
        atlas_name="test_mouse",
        atlas_version="1.2",
        atlas_resolution_um=10,
        region_descendants={1: {2, 3}},
        xy_bins=3,
        depth_bin_um=10,
        chunk_voxels=2,
    )
    assert second.profile_id != first.profile_id
    assert len(open_region_cache(root).profiles) == 2

    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    reopened = open_region_cache(relocated)
    assert set(reopened.profiles) == {first.profile_id, second.profile_id}
    assert reopened.profile(first.profile_id).directory.is_relative_to(relocated)


def test_profile_reports_exact_compatibility_mismatches(tmp_path):
    profile = _build(tmp_path / "cache")
    assert not profile.compatibility_mismatches(
        lookup_set_id="portable-lookup-set",
        atlas_name="test_mouse",
        atlas_version="1.2",
        atlas_resolution_um=(10, 10, 10),
        annotation_shape=(2, 2, 2),
        style="both_square",
    )
    mismatches = profile.compatibility_mismatches(
        lookup_set_id="different",
        atlas_name="other",
        atlas_version="2",
        annotation_shape=(3, 2, 2),
    )
    assert any("lookup_set_id" in message for message in mismatches)
    assert any("atlas family" in message for message in mismatches)
    assert any("atlas version" in message for message in mismatches)
    assert any("annotation shape" in message for message in mismatches)


def test_depth_mirror_policy_affects_profile_identity_and_compatibility(tmp_path):
    root = tmp_path / "cache"
    default = _build(root)
    different_axis = _build(root, mirror_coord_axis=1)
    no_fallback = _build(root, mirror_depth_fallback=False)

    assert len(
        {default.profile_id, different_axis.profile_id, no_fallback.profile_id}
    ) == 3
    assert not default.compatibility_mismatches(
        mirror_depth_fallback=True,
        mirror_coord_axis=2,
    )

    mismatches = default.compatibility_mismatches(
        mirror_depth_fallback=False,
        mirror_coord_axis=1,
    )
    assert any("depth mirror fallback" in message for message in mismatches)
    assert any("depth mirror axis" in message for message in mismatches)


def test_structure_catalog_and_atlas_family_are_resolution_independent(tmp_path):
    annotation, shaped, square, depth = _lookup_arrays()
    structures = {
        "root": {
            "id": 1,
            "acronym": "ROOT",
            "name": "Root",
            "structure_id_path": [1],
            "rgb_triplet": [1, 2, 3],
        },
        "child": {
            "id": 2,
            "acronym": "C",
            "name": "Child",
            "structure_id_path": [1, 2],
            "rgb_triplet": [4, 5, 6],
        },
    }
    profile = build_region_cache_profile(
        tmp_path / "cache",
        annotation=annotation,
        shaped_flatmap=shaped,
        square_flatmap=square,
        depth=depth,
        lookup_set_id="lookup",
        atlas_name="allen_mouse_10um",
        atlas_version="1.2",
        atlas_resolution_um=10,
        atlas_structures=structures,
        xy_bins=2,
        depth_bin_um=10,
        chunk_voxels=2,
    )
    catalog_id = structure_catalog_id(structures)
    assert profile.atlas["family"] == "allen_mouse"
    assert profile.atlas["structure_catalog_id"] == catalog_id
    # Viewing may use a different atlas resolution when its family/catalog match.
    assert not profile.compatibility_mismatches(
        atlas_name="allen_mouse_25um",
        atlas_version="1.2",
        structure_catalog_id=catalog_id,
    )
    assert any(
        "structure catalog" in message
        for message in profile.compatibility_mismatches(
            atlas_name="allen_mouse_25um",
            structure_catalog_id="different",
        )
    )


def test_open_rejects_missing_corrupt_and_invalid_arrays(tmp_path):
    root = tmp_path / "cache"
    profile = _build(root)
    manifest = json.loads((root / REGION_CACHE_MANIFEST_FILENAME).read_text())
    style_data = manifest["profiles"][profile.profile_id]["styles"]["shaped"]
    profile_dir = (
        root / manifest["profiles"][profile.profile_id]["directory"] / "shaped"
    )
    offsets_path = (
        profile_dir / style_data["arrays"]["occupancy_region_offsets"]["path"]
    )
    offsets = np.load(offsets_path, allow_pickle=False)
    offsets[0] = 1
    np.save(offsets_path, offsets, allow_pickle=False)
    with pytest.raises(RegionCacheValidationError, match="offsets"):
        open_region_cache(root)

    offsets[0] = 0
    np.save(offsets_path, offsets, allow_pickle=False)
    faces_path = profile_dir / style_data["arrays"]["surface_faces"]["path"]
    faces = np.load(faces_path, allow_pickle=False)
    faces[0, 0] = np.iinfo(np.int32).max
    np.save(faces_path, faces, allow_pickle=False)
    with pytest.raises(RegionCacheValidationError, match="face index"):
        open_region_cache(root)

    faces_path.unlink()
    with pytest.raises(RegionCacheValidationError, match="missing"):
        open_region_cache(root)


def test_open_rejects_contract_dtype_even_when_file_matches_manifest(tmp_path):
    root = tmp_path / "cache"
    profile = _build(root)
    manifest_path = root / REGION_CACHE_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    style_data = manifest["profiles"][profile.profile_id]["styles"]["shaped"]
    profile_dir = (
        root / manifest["profiles"][profile.profile_id]["directory"] / "shaped"
    )
    region_ids_spec = style_data["arrays"]["occupancy_region_ids"]
    region_ids_path = profile_dir / region_ids_spec["path"]
    region_ids = np.load(region_ids_path, allow_pickle=False).astype(np.int64)
    np.save(region_ids_path, region_ids, allow_pickle=False)
    region_ids_spec["dtype"] = "int64"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(
        RegionCacheValidationError,
        match="occupancy_region_ids.*must use int32",
    ):
        open_region_cache(root)


def test_open_rejects_incorrect_validation_count_metadata(tmp_path):
    root = tmp_path / "cache"
    profile = _build(root)
    manifest_path = root / REGION_CACHE_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    style_data = manifest["profiles"][profile.profile_id]["styles"]["shaped"]
    style_data["counts"]["surface_face_count"] += 1
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(
        RegionCacheValidationError,
        match="validation count 'surface_face_count' differs",
    ):
        open_region_cache(root)


def test_cancelled_build_keeps_existing_manifest_and_cleans_temporary_profile(tmp_path):
    root = tmp_path / "cache"
    _build(root)
    manifest_path = root / REGION_CACHE_MANIFEST_FILENAME
    original = manifest_path.read_bytes()
    annotation, shaped, square, depth = _lookup_arrays()
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        return calls >= 2

    with pytest.raises(RegionCacheCancelled):
        build_region_cache_profile(
            root,
            annotation=annotation,
            shaped_flatmap=shaped,
            square_flatmap=square,
            depth=depth,
            lookup_set_id="other-lookup",
            atlas_name="test_mouse",
            atlas_version="1.2",
            atlas_resolution_um=10,
            xy_bins=3,
            depth_bin_um=10,
            chunk_voxels=2,
            cancel_callback=cancel,
        )
    assert manifest_path.read_bytes() == original
    assert not list((root / "profiles").glob("*.tmp"))
    assert len(open_region_cache(root).profiles) == 1


def test_build_rejects_annotation_lookup_shape_mismatch(tmp_path):
    _annotation, shaped, square, depth = _lookup_arrays()
    with pytest.raises(ValueError, match="exactly match"):
        build_region_cache_profile(
            tmp_path / "cache",
            annotation=np.zeros((3, 2, 2), dtype=np.int32),
            shaped_flatmap=shaped,
            square_flatmap=square,
            depth=depth,
            lookup_set_id="lookup",
            atlas_name="test_mouse",
            atlas_resolution_um=10,
            xy_bins=2,
            depth_bin_um=10,
        )


def test_build_uses_lookup_set_grid_contract_and_requires_exact_resolution(
    tmp_path,
):
    from napari_swc_viewer.flatmap_profiles import FlatmapGridSpec

    annotation, shaped, square, depth = _lookup_arrays()

    def grid(style: str, x_upper: float, y_upper: float) -> FlatmapGridSpec:
        return FlatmapGridSpec(
            grid_spec_id=f"grid-{style}",
            style=style,
            lookup_coordinate_order=("x", "y", "z"),
            flatmap_coordinate_order=("x_flat", "y_flat"),
            render_coordinate_order=("depth", "y", "x"),
            spatial_shape=(2, 2, 2),
            flatmap_shape=(2, 2, 2, 2),
            depth_shape=(2, 2, 2),
            lookup_resolution_um=(10.0, 10.0, 10.0),
            space_directions=((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)),
            space_origin=(0.0, 0.0, 0.0),
            x_bounds=(0.0, x_upper),
            y_bounds=(0.0, y_upper),
            depth_bounds_um=(0.0, 10.0),
            invalid_zero_sentinel=False,
            invalid_negative_one_sentinel=True,
        )

    lookup_set = SimpleNamespace(
        lookup_set_id="lookup-contract",
        lookup_resolution_um=(10.0, 10.0, 10.0),
        spatial_shape=(2, 2, 2),
        shaped_grid=grid("both_shaped", 1.0, 1.0),
        square_grid=grid("both_square", 2.0, 3.0),
    )
    profile = build_region_cache_profile(
        tmp_path / "cache",
        lookup_set,
        annotation=annotation,
        shaped_flatmap=shaped,
        square_flatmap=square,
        depth=depth,
        atlas_name="test_mouse_10um",
        atlas_resolution_um=10,
        xy_bins=2,
        depth_bin_um=10,
        chunk_voxels=2,
    )
    style_cache = profile.style("shaped")
    assert style_cache.lookup_grid_spec == lookup_set.shaped_grid
    assert style_cache.grid_spec["x_bounds"] == [0.0, 1.0]

    with pytest.raises(ValueError, match="resolution match"):
        build_region_cache_profile(
            tmp_path / "bad-cache",
            lookup_set,
            annotation=annotation,
            shaped_flatmap=shaped,
            square_flatmap=square,
            depth=depth,
            atlas_name="test_mouse_25um",
            atlas_resolution_um=25,
            xy_bins=2,
            depth_bin_um=10,
        )


def test_empty_annotation_regions_round_trip(tmp_path):
    annotation, shaped, square, depth = _lookup_arrays()
    profile = build_region_cache_profile(
        tmp_path / "cache",
        annotation=np.zeros_like(annotation),
        shaped_flatmap=shaped,
        square_flatmap=square,
        depth=depth,
        lookup_set_id="empty-lookup",
        atlas_name="test_mouse",
        atlas_resolution_um=10,
        region_descendants={1: {1}},
        xy_bins=2,
        depth_bin_um=10,
        chunk_voxels=2,
    )
    result = materialize_region_selection(
        profile,
        [1],
        style="shaped",
        direct_region_ids=[1],
    )
    assert not np.any(result.labels)
    assert result.summary.labeled_bins == 0
    assert result.surfaces[0].vertices.shape == (0, 3)
    assert result.outlines[0].vectors.shape == (0, 2, 3)
