"""Allen template and region mesh rendering for napari.

This module provides functions to add Allen CCF reference images and
brain region meshes to a napari viewer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import napari
    from brainglobe_atlasapi import BrainGlobeAtlas

logger = logging.getLogger(__name__)


def add_allen_template(
    viewer: napari.Viewer,
    atlas: BrainGlobeAtlas,
    name: str = "Allen Template",
    opacity: float = 0.5,
    colormap: str = "gray",
    visible: bool = True,
) -> napari.layers.Image:
    """Add the Allen CCF template image to the viewer.

    All layers are in voxel/pixel space (not microns) to match brainrender-napari.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    atlas : BrainGlobeAtlas
        The atlas to use for the template.
    name : str, default="Allen Template"
        Name for the layer.
    opacity : float, default=0.5
        Layer opacity.
    colormap : str, default="gray"
        Colormap for the image.
    visible : bool, default=True
        Whether the layer is visible by default.

    Returns
    -------
    napari.layers.Image
        The created image layer.
    """
    # Get reference image (in voxel space)
    reference = atlas.reference

    layer = viewer.add_image(
        reference,
        name=name,
        opacity=opacity,
        colormap=colormap,
        visible=visible,
        blending="additive",
    )

    return layer


def add_annotation_volume(
    viewer: napari.Viewer,
    atlas: BrainGlobeAtlas,
    name: str = "Allen Annotations",
    opacity: float = 0.3,
    visible: bool = False,
) -> napari.layers.Labels:
    """Add the Allen CCF annotation volume as a labels layer.

    All layers are in voxel/pixel space (not microns) to match brainrender-napari.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    atlas : BrainGlobeAtlas
        The atlas to use for annotations.
    name : str, default="Allen Annotations"
        Name for the layer.
    opacity : float, default=0.3
        Layer opacity.
    visible : bool, default=False
        Whether the layer is visible by default.

    Returns
    -------
    napari.layers.Labels
        The created labels layer.
    """
    # Annotation volume (in voxel space)
    annotation = atlas.annotation

    layer = viewer.add_labels(
        annotation,
        name=name,
        opacity=opacity,
        visible=visible,
    )

    return layer


def add_region_segmentation(
    viewer: napari.Viewer,
    atlas: BrainGlobeAtlas,
    acronyms: list[str],
    name: str = "Region Segmentation",
    opacity: float = 0.3,
    visible: bool = True,
) -> napari.layers.Labels | None:
    """Add a filtered annotation volume showing only selected brain regions.

    Each selected region (and all its descendants in the annotation hierarchy)
    is shown with the region's atlas-defined RGB color. Voxels outside the
    selected regions are transparent (label 0).

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    atlas : BrainGlobeAtlas
        The atlas to use for annotations and structure metadata.
    acronyms : list[str]
        List of region acronyms to display (parent-level).
    name : str, default="Region Segmentation"
        Name for the layer.
    opacity : float, default=0.3
        Layer opacity.
    visible : bool, default=True
        Whether the layer is visible by default.

    Returns
    -------
    napari.layers.Labels or None
        The created labels layer, or None if no valid regions found.
    """
    if not acronyms:
        return None

    annotation = atlas.annotation

    # For each selected parent acronym, collect all descendant annotation IDs
    # and map them to the parent's atlas-defined color.
    all_selected_ids: set[int] = set()
    color_dict: dict[int, tuple[float, float, float]] = {}

    for acronym in acronyms:
        try:
            structure = atlas.structures[acronym]
        except KeyError:
            logger.warning(f"Region '{acronym}' not found in atlas structures")
            continue

        parent_id = structure["id"]
        rgb = structure.get("rgb_triplet", [128, 128, 128])
        rgb_normalized = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)

        # Collect all descendants: any structure whose structure_id_path
        # contains this parent_id is a descendant (or the region itself).
        descendant_ids: set[int] = set()
        for key, struct in atlas.structures.items():
            if isinstance(key, int):
                path = struct.get("structure_id_path", [])
                if parent_id in path:
                    descendant_ids.add(key)

        all_selected_ids.update(descendant_ids)

        # Map each descendant ID to the parent's color
        for did in descendant_ids:
            color_dict[did] = rgb_normalized

    if not all_selected_ids:
        logger.warning("No valid annotation IDs found for selected regions")
        return None

    # Build filtered annotation volume (keep only selected region voxels)
    id_array = np.array(sorted(all_selected_ids), dtype=annotation.dtype)
    mask = np.isin(annotation, id_array)
    filtered = np.where(mask, annotation, np.zeros_like(annotation))

    logger.info(
        f"Region segmentation: {len(acronyms)} regions, "
        f"{len(all_selected_ids)} annotation IDs, "
        f"{mask.sum():,} voxels"
    )

    layer = viewer.add_labels(
        filtered,
        name=name,
        opacity=opacity,
        visible=visible,
        color=color_dict,
    )

    return layer


def remove_region_segmentation(
    viewer: napari.Viewer,
    name: str = "Region Segmentation",
) -> bool:
    """Remove the region segmentation layer if it exists.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    name : str, default="Region Segmentation"
        The layer name to look for.

    Returns
    -------
    bool
        True if a layer was removed, False otherwise.
    """
    for layer in viewer.layers:
        if layer.name == name:
            viewer.layers.remove(layer)
            return True
    return False


def add_region_mesh(
    viewer: napari.Viewer,
    atlas: BrainGlobeAtlas,
    acronym: str,
    opacity: float = 0.4,
    color: str | tuple | None = None,
    name: str | None = None,
    visible: bool = True,
) -> napari.layers.Surface | None:
    """Add a brain region mesh to the viewer.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    atlas : BrainGlobeAtlas
        The atlas to use for the mesh.
    acronym : str
        The region acronym (e.g., "VISp").
    opacity : float, default=0.4
        Mesh opacity.
    color : str or tuple, optional
        Mesh color. If None, uses the atlas's color for the region.
    name : str, optional
        Name for the layer. If None, uses "Region: {acronym}".
    visible : bool, default=True
        Whether the layer is visible by default.

    Returns
    -------
    napari.layers.Surface or None
        The created surface layer, or None if the region mesh is not available.
    """
    # Get structure info - StructuresDict supports direct acronym access via []
    try:
        structure = atlas.structures[acronym]
    except KeyError:
        logger.warning(f"Region '{acronym}' not found in atlas structures")
        return None

    # Get mesh using BrainGlobe API
    try:
        mesh = atlas.mesh_from_structure(acronym)
        logger.info(f"Loaded mesh for '{acronym}' with {len(mesh.points)} vertices")
    except (KeyError, FileNotFoundError) as e:
        logger.warning(f"Could not load mesh for '{acronym}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading mesh for '{acronym}': {e}")
        return None

    # Get vertices and faces (meshio format)
    # Convert to float32/int32 for better vispy/OpenGL compatibility
    vertices = mesh.points.astype(np.float32)
    faces = mesh.cells[0].data.astype(np.int32)

    # Scale mesh from microns to pixel/voxel space to match the reference image
    scale = [1.0 / res for res in atlas.resolution]

    # Create layer name
    if name is None:
        name = f"Region: {acronym}"

    # Determine vertex colors
    if color is None:
        rgb = structure.get("rgb_triplet", [128, 128, 128])
    else:
        rgb = [int(c * 255) if isinstance(c, float) and c <= 1 else c for c in color]

    # Create vertex colors array (RGB 0-1 for each vertex, float32 for vispy)
    vertex_colors = np.repeat(
        [[float(c) / 255 for c in rgb]], len(vertices), axis=0
    ).astype(np.float32)

    logger.info(f"Creating surface layer '{name}': {len(vertices)} vertices, {len(faces)} faces")
    layer = viewer.add_surface(
        (vertices, faces),
        scale=scale,
        name=name,
        opacity=opacity,
        blending="translucent_no_depth",
        vertex_colors=vertex_colors,
        visible=visible,
    )

    logger.info(f"Added region mesh layer: {layer}")
    return layer


def add_region_meshes(
    viewer: napari.Viewer,
    atlas: BrainGlobeAtlas,
    acronyms: list[str],
    opacity: float = 0.3,
    use_atlas_colors: bool = True,
    visible: bool = True,
) -> list[napari.layers.Surface]:
    """Add multiple brain region meshes to the viewer.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    atlas : BrainGlobeAtlas
        The atlas to use for meshes.
    acronyms : list[str]
        List of region acronyms to add.
    opacity : float, default=0.3
        Mesh opacity.
    use_atlas_colors : bool, default=True
        If True, use the atlas's colors for each region.
    visible : bool, default=True
        Whether the layers are visible by default.

    Returns
    -------
    list[napari.layers.Surface]
        List of created surface layers.
    """
    layers = []
    for acronym in acronyms:
        layer = add_region_mesh(
            viewer,
            atlas,
            acronym,
            opacity=opacity,
            color=None if use_atlas_colors else (0.5, 0.5, 0.5),
            visible=visible,
        )
        if layer is not None:
            layers.append(layer)

    return layers


def add_brain_outline(
    viewer: napari.Viewer,
    atlas: BrainGlobeAtlas,
    opacity: float = 0.2,
    name: str = "Brain Outline",
    visible: bool = True,
) -> napari.layers.Surface | None:
    """Add the whole brain outline mesh to the viewer.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    atlas : BrainGlobeAtlas
        The atlas to use.
    opacity : float, default=0.2
        Mesh opacity.
    name : str, default="Brain Outline"
        Name for the layer.
    visible : bool, default=True
        Whether the layer is visible by default.

    Returns
    -------
    napari.layers.Surface or None
        The created surface layer, or None if not available.
    """
    # Get the root mesh using BrainGlobe API
    try:
        mesh = atlas.mesh_from_structure("root")
        logger.info(f"Loaded root mesh with {len(mesh.points)} vertices")
    except (KeyError, FileNotFoundError) as e:
        logger.warning(f"Could not load root mesh: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading root mesh: {e}")
        return None

    # Get vertices and faces (meshio format)
    # Convert to float32/int32 for better vispy/OpenGL compatibility
    vertices = mesh.points.astype(np.float32)
    faces = mesh.cells[0].data.astype(np.int32)

    # Scale mesh from microns to pixel/voxel space to match the reference image
    scale = [1.0 / res for res in atlas.resolution]

    # Gray color for outline (float32 for vispy)
    vertex_colors = np.repeat([[0.5, 0.5, 0.5]], len(vertices), axis=0).astype(np.float32)

    logger.info(f"Creating brain outline surface: {len(vertices)} vertices, {len(faces)} faces")

    layer = viewer.add_surface(
        (vertices, faces),
        scale=scale,
        name=name,
        opacity=opacity,
        blending="translucent_no_depth",
        vertex_colors=vertex_colors,
        visible=visible,
    )

    logger.info(f"Added brain outline layer: {layer}")
    return layer


def remove_region_layers(
    viewer: napari.Viewer,
    prefix: str = "Region:",
) -> int:
    """Remove all region mesh layers from the viewer.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    prefix : str, default="Region:"
        Prefix to match for layer names.

    Returns
    -------
    int
        Number of layers removed.
    """
    to_remove = [layer for layer in viewer.layers if layer.name.startswith(prefix)]

    for layer in to_remove:
        viewer.layers.remove(layer)

    return len(to_remove)
