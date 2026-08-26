"""Flatmap axis annotations checked against a real napari viewer model.

``tests/test_flatmap_widget.py`` covers these methods through the full render
path, but with a stubbed ``napari`` module and a hand-written viewer double.
That double cannot tell us whether napari still spells the overlay fields the
way this code expects, nor that ``viewer.dims.axis_labels`` — rather than
``layer.axis_labels`` — is what the slider caption and axes overlay read. This
module fills that gap with ``ViewerModel``, which needs no Qt.

napari is imported inside the fixtures, never at module scope: importing it
during collection initializes Qt far enough that the widget tests in
``tests/test_analysis_region_options.py`` abort rather than construct their
widgets without a QApplication.
"""

from __future__ import annotations

import numpy as np
import pytest

# ``FlatmapProjectionWidget`` is a QWidget, and PyQt refuses attribute access on
# an instance whose C++ base was never constructed. Bind just the annotation
# methods onto a plain object so the real code runs without a QApplication.
_ANNOTATION_METHODS = (
    "_apply_display_axis_annotations",
    "_capture_display_axis_annotation_state",
    "_clear_display_axis_annotations",
    "_connect_display_dims_events",
    "_current_display_viewer",
    "_flatmap_axis_labels_for_layer",
    "_on_display_dims_step_changed",
    "_plane_count_for_layer",
    "_plane_labels_for_layer",
    "_resolve_display_viewer",
)


@pytest.fixture
def widget_class():
    from napari_swc_viewer.widgets.flatmap import FlatmapProjectionWidget

    return FlatmapProjectionWidget


@pytest.fixture
def viewer():
    from napari.components import ViewerModel

    return ViewerModel()


@pytest.fixture
def host(widget_class, viewer):
    namespace = {name: widget_class.__dict__[name] for name in _ANNOTATION_METHODS}
    namespace.update(
        _viewer=viewer,
        _display_viewer_provider=None,
        _display_axis_annotation_state=None,
    )
    return type("_AnnotationHost", (), namespace)()


def _allen_layer_image(widget_class, viewer):
    volume = np.zeros((6, 8, 8), dtype=np.float32)
    volume[0, 1, 2] = 1.0
    layer = viewer.add_image(
        volume,
        name="Isocortex Flatmap Allen Layers",
        axis_labels=widget_class._allen_layer_axis_labels(),
        metadata={
            "flatmap_plane_mode": "allen_layers",
            "allen_layer_labels": ["L1", "L2/3", "L4", "L5", "L6a", "L6b"],
        },
    )
    viewer.dims.ndisplay = 2
    return layer


def test_annotations_set_the_viewer_state_napari_renders(
    widget_class,
    viewer,
    host,
) -> None:
    layer = _allen_layer_image(widget_class, viewer)

    host._apply_display_axis_annotations(layer)

    assert viewer.dims.axis_labels == ("Allen layer", "Flatmap Y", "Flatmap X")
    assert viewer.scene.overlays.axes.visible is True
    assert viewer.scene.overlays.axes.labels is True
    # The two labels napari's axes overlay draws for the displayed axes.
    displayed = [viewer.dims.axis_labels[axis] for axis in viewer.dims.displayed[::-1]]
    assert displayed == ["Flatmap X", "Flatmap Y"]
    assert viewer.canvas.overlays.text.visible is True
    assert viewer.canvas.overlays.text.position == "top_left"
    assert viewer.canvas.overlays.text.font_size == 12


def test_plane_label_follows_a_real_dims_slider(widget_class, viewer, host) -> None:
    layer = _allen_layer_image(widget_class, viewer)

    host._apply_display_axis_annotations(layer)

    # napari opens a new six-plane axis at its middle position, not at zero.
    assert viewer.dims.current_step[0] == 2
    assert viewer.canvas.overlays.text.text == "Allen layer: L4  (plane 3 of 6)"

    viewer.dims.set_current_step(0, 0)
    assert viewer.canvas.overlays.text.text == "Allen layer: L1  (plane 1 of 6)"

    viewer.dims.set_current_step(0, 5)
    assert viewer.canvas.overlays.text.text == "Allen layer: L6b  (plane 6 of 6)"


def test_clearing_restores_the_viewer_and_stops_following(
    widget_class,
    viewer,
    host,
) -> None:
    layer = _allen_layer_image(widget_class, viewer)

    host._apply_display_axis_annotations(layer)
    host._clear_display_axis_annotations()

    # Clearing plugin-managed overlays does not override napari 0.9's labels;
    # the still-present layer remains their source of truth.
    assert viewer.dims.axis_labels == ("Allen layer", "Flatmap Y", "Flatmap X")
    assert viewer.scene.overlays.axes.visible is False
    assert viewer.canvas.overlays.text.visible is False
    assert viewer.canvas.overlays.text.text == ""

    viewer.dims.set_current_step(0, 4)

    assert viewer.canvas.overlays.text.text == ""


def test_depth_planes_are_named_by_micron_range(widget_class, viewer, host) -> None:
    layer = viewer.add_image(
        np.zeros((3, 8, 8), dtype=np.float32),
        name="Isocortex Flatmap Heatmap",
        axis_labels=widget_class._depth_axis_labels(),
        metadata={
            "flatmap_plane_mode": "depth",
            "render_summary": {
                "depth_bins": 3,
                "depth_bin_um": 25.0,
                "depth_min_um": 0.0,
                "includes_depth_minus_one_plane": False,
            },
        },
    )

    host._apply_display_axis_annotations(layer)
    viewer.dims.set_current_step(0, 2)

    assert viewer.dims.axis_labels == ("Depth bin", "Flatmap Y", "Flatmap X")
    assert viewer.canvas.overlays.text.text == "Depth bin: 50-75 um  (plane 3 of 3)"


def test_flat_render_names_only_two_axes(widget_class, viewer, host) -> None:
    image = np.zeros((8, 8), dtype=np.float32)
    image[3, 5] = 1.0
    layer = viewer.add_image(
        image,
        name="Isocortex Flatmap 2D Heatmap",
        axis_labels=widget_class._flat_axis_labels(),
        metadata={"flatmap_plane_mode": "flat"},
    )
    viewer.dims.ndisplay = 2

    host._apply_display_axis_annotations(layer)

    assert viewer.dims.axis_labels == ("Flatmap Y", "Flatmap X")
    assert viewer.scene.overlays.axes.visible is True
    # A collapsed render has no plane axis, so there is no plane to caption.
    assert viewer.canvas.overlays.text.visible is False
    assert viewer.canvas.overlays.text.text == ""


def test_flat_render_retires_a_previous_plane_caption(
    widget_class,
    viewer,
    host,
) -> None:
    stack = _allen_layer_image(widget_class, viewer)
    host._apply_display_axis_annotations(stack)
    assert viewer.canvas.overlays.text.text.startswith("Allen layer")

    viewer.layers.remove(stack)
    flat = viewer.add_image(
        np.ones((8, 8), dtype=np.float32),
        name="Isocortex Flatmap 2D Heatmap",
        axis_labels=widget_class._flat_axis_labels(),
        metadata={"flatmap_plane_mode": "flat"},
    )
    host._apply_display_axis_annotations(flat)

    assert viewer.canvas.overlays.text.visible is False
    assert viewer.canvas.overlays.text.text == ""


def test_soma_points_layer_keeps_the_allen_plane_caption(
    widget_class,
    viewer,
    host,
) -> None:
    stack = _allen_layer_image(widget_class, viewer)
    host._apply_display_axis_annotations(stack)
    viewer.dims.set_current_step(0, 1)
    assert viewer.canvas.overlays.text.text == "Allen layer: L2/3  (plane 2 of 6)"

    # napari 0.9 propagates axis_labels from a Points layer; that keeps the
    # soma overlay from reading as a foreign layer.
    somas = viewer.add_points(
        np.asarray([[1.0, 2.0, 3.0]]),
        name="Isocortex Flatmap Somas",
        axis_labels=widget_class._allen_layer_axis_labels(),
        metadata={
            "flatmap_plane_mode": "allen_layers",
            "allen_layer_labels": ["L1", "L2/3", "L4", "L5", "L6a", "L6b"],
        },
    )

    host._apply_display_axis_annotations(somas)

    assert viewer.dims.axis_labels == ("Allen layer", "Flatmap Y", "Flatmap X")
    assert viewer.scene.overlays.axes.visible is True
    assert viewer.canvas.overlays.text.visible is True
    assert viewer.canvas.overlays.text.text == "Allen layer: L2/3  (plane 2 of 6)"


def test_flat_vector_layer_keeps_two_axis_labels(widget_class, viewer, host) -> None:
    layer = viewer.add_vectors(
        np.asarray([[[0.0, 0.0], [3.0, 4.0]]], dtype=np.float32),
        name="Isocortex Flatmap 2D Vectors",
        axis_labels=widget_class._flat_axis_labels(),
        vector_style="line",
        metadata={"flatmap_plane_mode": "flat"},
    )
    viewer.dims.ndisplay = 2

    host._apply_display_axis_annotations(layer)

    assert viewer.dims.axis_labels == ("Flatmap Y", "Flatmap X")
    assert viewer.canvas.overlays.text.visible is False


def test_collapsed_region_overlays_get_two_axes_and_no_plane_caption(
    widget_class,
    viewer,
    host,
) -> None:
    """The 2D region overlays occupy one plane, so nothing should caption it."""
    labels = viewer.add_labels(
        np.asarray([[315, 0], [0, 315]], dtype=np.int32),
        name="Flatmap Region Labels 2D",
        axis_labels=widget_class._flat_axis_labels(),
        metadata={"flatmap_plane_mode": "flat"},
    )
    outlines = viewer.add_vectors(
        np.asarray([[[0.5, 0.5], [0.0, 1.0]]], dtype=np.float32),
        name="Flatmap Region Outlines 2D: Isocortex (315)",
        axis_labels=widget_class._flat_axis_labels(),
        vector_style="line",
        metadata={"flatmap_plane_mode": "flat"},
    )
    viewer.dims.ndisplay = 2

    for layer in (labels, outlines):
        host._apply_display_axis_annotations(layer)

        assert host._plane_labels_for_layer(layer) is None
        assert viewer.dims.axis_labels == ("Flatmap Y", "Flatmap X")
        assert viewer.scene.overlays.axes.visible is True
        assert viewer.canvas.overlays.text.visible is False
        assert viewer.canvas.overlays.text.text == ""


def test_foreign_layer_axis_labels_are_not_copied(viewer, host) -> None:
    points = viewer.add_points(
        np.zeros((2, 3), dtype=float),
        name="Isocortex Flatmap Points",
    )

    host._apply_display_axis_annotations(points)

    assert viewer.dims.axis_labels == ("-3", "-2", "-1")
    assert viewer.scene.overlays.axes.visible is False
    assert viewer.canvas.overlays.text.text == ""


def test_main_viewer_scene_swap_preserves_real_layer_objects(viewer) -> None:
    from napari_swc_viewer.widgets.neuron_viewer import _MainViewerFlatmapScene

    anatomy = viewer.add_image(
        np.zeros((3, 4, 5), dtype=np.float32),
        name="Anatomy",
        axis_labels=("Anterior", "Dorsal", "Left"),
    )
    points = viewer.add_points(np.asarray([[1.0, 2.0, 3.0]]), name="Somas")
    viewer.layers.selection.active = points
    viewer.scene.camera.center = (7.0, 8.0, 9.0)
    viewer.scene.camera.zoom = 3.0
    scene = _MainViewerFlatmapScene(viewer)

    assert scene.enter() is True
    assert list(viewer.layers) == []
    transient = viewer.add_image(
        np.zeros((2, 6, 8), dtype=np.float32),
        name="Flatmap",
        axis_labels=("Depth bin", "Flatmap Y", "Flatmap X"),
        metadata={"napari_swc_viewer_space": "flatmap"},
    )
    viewer.scene.camera.center = (1.0, 2.0, 3.0)

    assert scene.restore() is True
    assert list(viewer.layers) == [anatomy, points]
    assert all(layer is not transient for layer in viewer.layers)
    assert viewer.layers.selection.active is points
    assert viewer.scene.camera.center == (7.0, 8.0, 9.0)
    assert viewer.scene.camera.zoom == 3.0
