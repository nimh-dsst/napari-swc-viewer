"""Dynamic 2D slice projection for neuron line segments.

This module provides functionality to display neuron line segments in napari's
2D slice view. Since thin line segments rarely intersect the exact slice plane,
this projector shows all line segments within a configurable Z tolerance.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import QTimer

if TYPE_CHECKING:
    import napari

logger = logging.getLogger(__name__)


class NeuronSliceProjector:
    """Projects neuron line segments onto the current 2D slice.

    This class maintains a cache of neuron line data and dynamically updates
    a 2D Vectors layer to show line segments within a configurable Z tolerance
    of the current slice position.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    tolerance : float
        Z tolerance in microns. Lines within this distance of the slice are shown.
    """

    def __init__(self, viewer: napari.Viewer, tolerance: float = 50.0, edge_width: int = 4):
        self._viewer = viewer
        self._tolerance = tolerance
        self._edge_width = edge_width
        self._source_data: dict[str, tuple[np.ndarray, np.ndarray, tuple]] = {}
        self._projection_layer = None
        self._scale: list[float] | None = None
        self._enabled = False
        self._connected = False

        # Precomputed arrays for vectorized projection (rebuilt on data change)
        self._all_p1: np.ndarray | None = None  # (M, 3) start points
        self._all_p2: np.ndarray | None = None  # (M, 3) end points
        self._all_colors: np.ndarray | None = None  # (M, 4) RGBA per segment

        # Per-axis sorted spatial index for fast slice queries
        self._axis_index: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}

        # Single-entry result cache: (axis, position, tolerance) → (lines, colors)
        self._last_result_key: tuple | None = None
        self._last_result: tuple | None = None

        # Debounce timer for slice updates
        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(50)  # 50ms debounce
        self._update_timer.timeout.connect(self._do_update_projection)

    @property
    def tolerance(self) -> float:
        """Get the current Z tolerance in microns."""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: float) -> None:
        """Set the Z tolerance and trigger an update."""
        self._tolerance = value
        self._invalidate_cache()
        if self._enabled:
            self._schedule_update()

    @property
    def edge_width(self) -> int:
        """Get the current edge width."""
        return self._edge_width

    @edge_width.setter
    def edge_width(self, value: int) -> None:
        """Set the edge width and update the projection layer."""
        self._edge_width = value
        if self._projection_layer is not None:
            self._projection_layer.edge_width = value

    @property
    def enabled(self) -> bool:
        """Check if the projector is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable the projector."""
        self._enabled = value
        if value:
            self._connect_events()
            self._schedule_update()
        else:
            self._disconnect_events()
            self._remove_projection_layer()

    def set_scale(self, scale: list[float] | None) -> None:
        """Set the coordinate scale for the projection layer.

        Parameters
        ----------
        scale : list[float] | None
            Scale factors for each dimension, or None for no scaling.
        """
        self._scale = scale

    def add_neuron_data(
        self,
        file_id: str,
        coords: np.ndarray,
        edges: np.ndarray,
        color: tuple = (1, 1, 0, 1),
    ) -> None:
        """Add or update neuron line data for projection.

        Parameters
        ----------
        file_id : str
            Unique identifier for the neuron.
        coords : np.ndarray
            Node coordinates array with shape (N, 3) in ZYX order.
        edges : np.ndarray
            Edge array with shape (M, 2) containing node index pairs.
        color : tuple
            RGBA color tuple for this neuron's lines.
        """
        self._source_data[file_id] = (coords.copy(), edges.copy(), color)
        self._rebuild_arrays()
        if self._enabled:
            self._schedule_update()

    def add_neuron_data_batch(
        self,
        data: dict[str, tuple[np.ndarray, np.ndarray, tuple]],
    ) -> None:
        """Add multiple neurons at once, rebuilding arrays only once.

        Parameters
        ----------
        data : dict[str, tuple[ndarray, ndarray, tuple]]
            Mapping of file_id to (coords, edges, color).
        """
        for file_id, (coords, edges, color) in data.items():
            self._source_data[file_id] = (coords.copy(), edges.copy(), color)
        self._rebuild_arrays()
        if self._enabled:
            self._schedule_update()

    def update_neuron_colors(
        self,
        color_map: dict[str, list[float]],
    ) -> None:
        """Update the color for each neuron without changing geometry.

        Parameters
        ----------
        color_map : dict[str, list[float]]
            Mapping of file_id to RGBA color list.
        """
        changed = False
        for file_id, (coords, edges, old_color) in list(self._source_data.items()):
            if file_id in color_map:
                new_color = tuple(color_map[file_id][:4])
                if new_color != old_color:
                    self._source_data[file_id] = (coords, edges, new_color)
                    changed = True
        if changed:
            self._rebuild_colors_only()
            self._invalidate_cache()
            if self._enabled:
                self._schedule_update()

    def remove_neuron_data(self, file_id: str) -> None:
        """Remove neuron data from projection.

        Parameters
        ----------
        file_id : str
            Unique identifier for the neuron to remove.
        """
        if file_id in self._source_data:
            del self._source_data[file_id]
            self._rebuild_arrays()
            if self._enabled:
                self._schedule_update()

    def clear(self) -> None:
        """Clear all neuron data and remove the projection layer."""
        self._source_data.clear()
        self._all_p1 = None
        self._all_p2 = None
        self._all_colors = None
        self._axis_index.clear()
        self._invalidate_cache()
        self._remove_projection_layer()

    def _rebuild_arrays(self) -> None:
        """Precompute flat arrays of all line segment endpoints and colors.

        Called when neuron data is added or removed. Converts the per-neuron
        data into contiguous arrays for fast vectorized slicing, then builds
        the per-axis spatial index.
        """
        if not self._source_data:
            self._all_p1 = None
            self._all_p2 = None
            self._all_colors = None
            self._axis_index.clear()
            self._invalidate_cache()
            return

        p1_list = []
        p2_list = []
        color_list = []
        for coords, edges, color in self._source_data.values():
            if len(edges) == 0:
                continue
            p1_list.append(coords[edges[:, 0]])
            p2_list.append(coords[edges[:, 1]])
            color_arr = np.empty((len(edges), len(color)))
            color_arr[:] = color
            color_list.append(color_arr)

        if p1_list:
            self._all_p1 = np.concatenate(p1_list)
            self._all_p2 = np.concatenate(p2_list)
            self._all_colors = np.concatenate(color_list)
        else:
            self._all_p1 = None
            self._all_p2 = None
            self._all_colors = None

        self._rebuild_axis_index()
        self._invalidate_cache()

    def _rebuild_axis_index(self) -> None:
        """Build a per-axis sorted spatial index for fast slice queries.

        For each axis, stores sorted_indices, sorted_z_min, sorted_z_max,
        and max_span so that slice queries can use binary search instead of
        scanning all segments.
        """
        self._axis_index.clear()
        if self._all_p1 is None:
            return

        for axis in range(3):
            v1 = self._all_p1[:, axis]
            v2 = self._all_p2[:, axis]
            z_min = np.minimum(v1, v2)
            z_max = np.maximum(v1, v2)
            order = np.argsort(z_min)
            sorted_z_min = z_min[order]
            sorted_z_max = z_max[order]
            max_span = float((z_max - z_min).max()) if len(z_min) > 0 else 0.0
            self._axis_index[axis] = (order, sorted_z_min, sorted_z_max, max_span)

    def _rebuild_colors_only(self) -> None:
        """Rebuild only the color array from source data.

        Skips geometry and index rebuild — use when only colors have changed.
        """
        if not self._source_data or self._all_p1 is None:
            return

        color_list = []
        for coords, edges, color in self._source_data.values():
            if len(edges) == 0:
                continue
            color_arr = np.empty((len(edges), len(color)))
            color_arr[:] = color
            color_list.append(color_arr)

        if color_list:
            self._all_colors = np.concatenate(color_list)

    def _invalidate_cache(self) -> None:
        """Invalidate the single-entry result cache."""
        self._last_result_key = None
        self._last_result = None

    def _connect_events(self) -> None:
        """Connect to viewer dimension events."""
        if not self._connected:
            self._viewer.dims.events.current_step.connect(self._on_dims_changed)
            self._viewer.dims.events.ndisplay.connect(self._on_ndisplay_changed)
            self._viewer.dims.events.order.connect(self._on_dims_changed)
            self._connected = True

    def _disconnect_events(self) -> None:
        """Disconnect from viewer dimension events."""
        if self._connected:
            try:
                self._viewer.dims.events.current_step.disconnect(self._on_dims_changed)
                self._viewer.dims.events.ndisplay.disconnect(self._on_ndisplay_changed)
                self._viewer.dims.events.order.disconnect(self._on_dims_changed)
            except (TypeError, RuntimeError):
                pass  # Event not connected
            self._connected = False

    def _on_dims_changed(self, event) -> None:
        """Handle dimension/slice changes."""
        if self._enabled and self._viewer.dims.ndisplay == 2:
            self._schedule_update()

    def _on_ndisplay_changed(self, event) -> None:
        """Handle display mode changes (2D/3D toggle)."""
        if not self._enabled:
            return

        if self._viewer.dims.ndisplay == 2:
            self._schedule_update()
        else:
            # In 3D mode, hide the projection layer
            if self._projection_layer is not None:
                self._projection_layer.visible = False

    def _schedule_update(self) -> None:
        """Schedule a debounced projection update."""
        self._update_timer.start()

    def _do_update_projection(self) -> None:
        """Actually perform the projection update."""
        if not self._enabled:
            return

        # Only show projection in 2D mode
        if self._viewer.dims.ndisplay != 2:
            if self._projection_layer is not None:
                self._projection_layer.visible = False
            return

        # Determine which axis is being sliced (the non-displayed dimension)
        not_displayed = self._viewer.dims.not_displayed
        if not not_displayed:
            return
        slice_axis = not_displayed[0]

        # Get current position in world coordinates, then convert to data
        # coordinates (microns). dims.point gives the world coordinate;
        # dividing by the layer scale converts back to data space.
        slice_world = self._viewer.dims.point[slice_axis]
        if self._scale is not None:
            slice_position_microns = slice_world / self._scale[slice_axis]
        else:
            slice_position_microns = slice_world

        # Compute lines within the slab
        lines, colors = self._compute_slice_projection(
            slice_position_microns, slice_axis
        )

        if lines is None:
            self._remove_projection_layer()
            return

        # Create or update the projection layer
        self._update_projection_layer(lines, colors)

    def _compute_slice_projection(
        self, slice_position: float, slice_axis: int
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Compute line segments within the slab, flattened onto the slice.

        Uses a per-axis sorted spatial index so that each query examines only
        a narrow candidate band via binary search, instead of scanning all
        segments.

        Parameters
        ----------
        slice_position : float
            Current position along the slicing axis in microns.
        slice_axis : int
            The data dimension index being sliced (0, 1, or 2).

        Returns
        -------
        lines : ndarray or None
            Array of shape (N, 2, 3) in vectors format [start, direction], or None.
        colors : ndarray or None
            Array of shape (N, 4) with RGBA colors, or None.
        """
        if self._all_p1 is None:
            return None, None

        # Check single-entry cache
        cache_key = (slice_axis, slice_position, self._tolerance)
        if self._last_result_key == cache_key:
            return self._last_result

        slab_min = slice_position - self._tolerance
        slab_max = slice_position + self._tolerance

        if slice_axis in self._axis_index:
            # Fast path: use spatial index
            order, sorted_z_min, sorted_z_max, max_span = self._axis_index[slice_axis]

            # Binary search for the candidate window
            left = np.searchsorted(sorted_z_min, slab_min - max_span, side="left")
            right = np.searchsorted(sorted_z_min, slab_max, side="right")

            if left >= right:
                result = (None, None)
                self._last_result_key = cache_key
                self._last_result = result
                return result

            # Filter candidates: keep segments whose z_max >= slab_min
            sub_z_max = sorted_z_max[left:right]
            sub_mask = sub_z_max >= slab_min
            hit_indices = order[left:right][sub_mask]
        else:
            # Fallback: brute-force (should not happen after _rebuild_axis_index)
            v1 = self._all_p1[:, slice_axis]
            v2 = self._all_p2[:, slice_axis]
            mask = (np.maximum(v1, v2) >= slab_min) & (np.minimum(v1, v2) <= slab_max)
            hit_indices = np.nonzero(mask)[0]

        if len(hit_indices) == 0:
            result = (None, None)
            self._last_result_key = cache_key
            self._last_result = result
            return result

        # Extract matching segments and flatten onto slice plane
        p1 = self._all_p1[hit_indices].copy()
        p2 = self._all_p2[hit_indices].copy()
        p1[:, slice_axis] = slice_position
        p2[:, slice_axis] = slice_position

        # Stack into (N, 2, 3) for napari vectors: [start, direction]
        lines = np.stack([p1, p2 - p1], axis=1)
        colors = self._all_colors[hit_indices]

        result = (lines, colors)
        self._last_result_key = cache_key
        self._last_result = result
        return result

    def _update_projection_layer(
        self, lines: np.ndarray, colors: np.ndarray
    ) -> None:
        """Update or create the projection vectors layer.

        Parameters
        ----------
        lines : np.ndarray
            Array of shape (N, 2, 3) in vectors format [start, direction].
        colors : np.ndarray
            Array of shape (N, 4) with RGBA colors per segment.
        """
        layer_name = "Neuron Slice Projection"

        if self._projection_layer is None:
            # Check if layer exists in viewer (may have been created before)
            for layer in self._viewer.layers:
                if layer.name == layer_name:
                    self._projection_layer = layer
                    break

        if self._projection_layer is None:
            # Create new layer
            self._projection_layer = self._viewer.add_vectors(
                lines,
                edge_width=self._edge_width,
                edge_color=colors,
                name=layer_name,
                opacity=1.0,
                scale=self._scale,
                vector_style="line",
            )
        else:
            # Update existing layer
            self._projection_layer.data = lines
            self._projection_layer.edge_color = colors
            self._projection_layer.visible = True
            # Update scale if changed
            if self._scale is not None:
                self._projection_layer.scale = self._scale

    def _remove_projection_layer(self) -> None:
        """Remove the projection layer from the viewer."""
        if self._projection_layer is not None:
            try:
                self._viewer.layers.remove(self._projection_layer)
            except ValueError:
                pass  # Layer already removed
            self._projection_layer = None

    def cleanup(self) -> None:
        """Clean up resources when the widget is destroyed."""
        self._update_timer.stop()
        self._disconnect_events()
        self._remove_projection_layer()
        self._source_data.clear()
