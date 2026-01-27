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
    # Get reference image
    reference = atlas.reference

    # Create scale to convert voxels to microns
    scale = atlas.resolution

    layer = viewer.add_image(
        reference,
        name=name,
        scale=scale,
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
    annotation = atlas.annotation

    # Create scale to convert voxels to microns
    scale = atlas.resolution

    layer = viewer.add_labels(
        annotation,
        name=name,
        scale=scale,
        opacity=opacity,
        visible=visible,
    )

    return layer


def add_region_mesh(
    viewer: napari.Viewer,
    atlas: BrainGlobeAtlas,
    acronym: str,
    opacity: float = 0.3,
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
    opacity : float, default=0.3
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
    # Get structure info - atlas.structures is keyed by ID, so we need to look up by acronym
    structure = None
    for struct_id, struct in atlas.structures.items():
        if isinstance(struct_id, int) and struct.get("acronym") == acronym:
            structure = struct
            break

    if structure is None:
        logger.warning(f"Region '{acronym}' not found in atlas structures")
        return None

    # Get mesh using BrainGlobe API (accepts acronym directly)
    try:
        mesh = atlas.mesh_from_structure(acronym)
        logger.info(f"Loaded mesh for '{acronym}' with {len(mesh.points)} vertices")
    except (KeyError, FileNotFoundError) as e:
        logger.warning(f"Could not load mesh for '{acronym}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading mesh for '{acronym}': {e}")
        return None

    # Get vertices and faces
    vertices = mesh.points
    # meshio returns faces as cells - extract triangles
    faces = None
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            faces = cell_block.data
            break

    if faces is None:
        logger.warning(f"No triangle faces found in mesh for '{acronym}'")
        return None

    # Determine color
    if color is None:
        rgb = structure.get("rgb_triplet", [128, 128, 128])
        color = tuple(c / 255 for c in rgb)

    # Create layer name
    if name is None:
        name = f"Region: {acronym}"

    # Create values array for coloring
    values = np.ones(len(vertices))

    logger.info(f"Creating surface layer '{name}': {len(vertices)} vertices, {len(faces)} faces")
    layer = viewer.add_surface(
        (vertices, faces, values),
        name=name,
        opacity=opacity,
        colormap="gray",
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
    opacity: float = 0.1,
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
    opacity : float, default=0.1
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

    vertices = mesh.points

    # meshio returns faces as cells - extract triangles
    faces = None
    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            faces = cell_block.data
            break

    if faces is None:
        logger.warning("No triangle faces found in root mesh")
        return None

    logger.info(f"Creating brain outline surface: {len(vertices)} vertices, {len(faces)} faces")
    values = np.ones(len(vertices))

    layer = viewer.add_surface(
        (vertices, faces, values),
        name=name,
        opacity=opacity,
        colormap="gray",
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
