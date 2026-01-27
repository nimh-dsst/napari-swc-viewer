"""Allen template and region mesh rendering for napari.

This module provides functions to add Allen CCF reference images and
brain region meshes to a napari viewer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import napari
    from brainglobe_atlasapi import BrainGlobeAtlas


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
    # Get structure info
    if acronym not in atlas.structures:
        return None

    structure = atlas.structures[acronym]
    struct_id = structure["id"]

    # Get mesh
    try:
        mesh = atlas.meshes[struct_id]
    except (KeyError, FileNotFoundError):
        return None

    # Get vertices and faces
    vertices = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:4]  # Convert from VTK format

    # Convert to microns (BrainGlobe meshes are in atlas space)
    # Meshes are already in microns for BrainGlobe atlases

    # Determine color
    if color is None:
        color_hex = structure.get("color_hex_triplet", "808080")
        color = tuple(int(color_hex[i : i + 2], 16) / 255 for i in (0, 2, 4))

    # Create layer name
    if name is None:
        name = f"Region: {acronym}"

    # Create values array for coloring
    values = np.ones(len(vertices))

    layer = viewer.add_surface(
        (vertices, faces, values),
        name=name,
        opacity=opacity,
        colormap="gray",
        visible=visible,
    )

    # Set single color
    layer.colormap = "gray"

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
    # The root structure is usually ID 997 in Allen CCF
    try:
        mesh = atlas.meshes[997]
    except (KeyError, FileNotFoundError):
        # Try to find root by looking for "root" or "brain" structure
        root_id = None
        for struct_id, struct in atlas.structures.items():
            if isinstance(struct_id, int):
                name_lower = struct.get("name", "").lower()
                if "root" in name_lower or name_lower == "brain":
                    root_id = struct_id
                    break

        if root_id is None:
            return None

        try:
            mesh = atlas.meshes[root_id]
        except (KeyError, FileNotFoundError):
            return None

    vertices = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:4]
    values = np.ones(len(vertices))

    layer = viewer.add_surface(
        (vertices, faces, values),
        name=name,
        opacity=opacity,
        colormap="gray",
        visible=visible,
    )

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
