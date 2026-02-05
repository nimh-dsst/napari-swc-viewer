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
    a 2D Shapes layer to show line segments within a configurable Z tolerance
    of the current slice position.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    tolerance : float
        Z tolerance in microns. Lines within this distance of the slice are shown.
    """

    def __init__(self, viewer: napari.Viewer, tolerance: float = 50.0):
        self._viewer = viewer
        self._tolerance = tolerance
        self._source_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._projection_layer = None
        self._scale: list[float] | None = None
        self._enabled = True
        self._connected = False

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
        if self._enabled:
            self._schedule_update()

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
        self, file_id: str, coords: np.ndarray, edges: np.ndarray
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
        """
        self._source_data[file_id] = (coords.copy(), edges.copy())
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
            if self._enabled:
                self._schedule_update()

    def clear(self) -> None:
        """Clear all neuron data and remove the projection layer."""
        self._source_data.clear()
        self._remove_projection_layer()

    def _connect_events(self) -> None:
        """Connect to viewer dimension events."""
        if not self._connected:
            self._viewer.dims.events.current_step.connect(self._on_dims_changed)
            self._viewer.dims.events.ndisplay.connect(self._on_ndisplay_changed)
            self._connected = True

    def _disconnect_events(self) -> None:
        """Disconnect from viewer dimension events."""
        if self._connected:
            try:
                self._viewer.dims.events.current_step.disconnect(self._on_dims_changed)
                self._viewer.dims.events.ndisplay.disconnect(self._on_ndisplay_changed)
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

        # Get current Z position (first dimension in napari's ZYX order)
        z_position = self._viewer.dims.current_step[0]

        # Apply scale to Z position if we have scaling
        if self._scale is not None:
            z_position_microns = z_position * (1.0 / self._scale[0])
        else:
            z_position_microns = z_position

        # Compute lines within the slab
        lines = self._compute_slice_projection(z_position_microns)

        if not lines:
            self._remove_projection_layer()
            return

        # Create or update the projection layer
        self._update_projection_layer(lines)

    def _compute_slice_projection(self, z_position: float) -> list:
        """Compute line segments within the Z slab.

        Parameters
        ----------
        z_position : float
            Current Z position in microns.

        Returns
        -------
        list
            List of line segments [[start, end], ...] for lines within tolerance.
        """
        z_min = z_position - self._tolerance
        z_max = z_position + self._tolerance

        lines = []
        for coords, edges in self._source_data.values():
            for i, j in edges:
                z1, z2 = coords[i][0], coords[j][0]
                # Include segment if any part is within the slab
                if max(z1, z2) >= z_min and min(z1, z2) <= z_max:
                    lines.append([coords[i], coords[j]])

        return lines

    def _update_projection_layer(self, lines: list) -> None:
        """Update or create the projection shapes layer.

        Parameters
        ----------
        lines : list
            List of line segments to display.
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
            self._projection_layer = self._viewer.add_shapes(
                lines,
                shape_type="line",
                edge_width=2,
                edge_color="yellow",
                name=layer_name,
                opacity=0.8,
                scale=self._scale,
            )
        else:
            # Update existing layer
            self._projection_layer.data = lines
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
