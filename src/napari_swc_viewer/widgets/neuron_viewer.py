"""Main neuron viewer widget combining all components.

This widget provides a unified interface for:
1. Loading neuron data from Parquet files
2. Selecting brain regions to filter neurons
3. Visualizing neurons as points or lines
4. Displaying Allen CCF reference layers
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas
from napari.utils.notifications import show_info
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..db import NeuronDatabase
from .reference_layers import (
    add_allen_template,
    add_brain_outline,
    add_region_mesh,
    remove_region_layers,
)
from .region_selector import RegionSelectorWidget

if TYPE_CHECKING:
    import napari

logger = logging.getLogger(__name__)


class NeuronViewerWidget(QWidget):
    """Main widget for viewing neurons with brain region filtering.

    This widget integrates:
    - Parquet file loading and database querying
    - Hierarchical region selection
    - Neuron visualization (points or lines)
    - Allen CCF reference layers

    Parameters
    ----------
    napari_viewer : napari.Viewer
        The napari viewer instance.
    """

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._db: NeuronDatabase | None = None
        self._atlas: BrainGlobeAtlas | None = None
        self._current_neuron_layers: list = []
        self._current_region_layers: list = []

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)

        # Tabs for organization
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Data tab
        data_tab = QWidget()
        tabs.addTab(data_tab, "Data")
        self._setup_data_tab(data_tab)

        # Regions tab
        regions_tab = QWidget()
        tabs.addTab(regions_tab, "Regions")
        self._setup_regions_tab(regions_tab)

        # Visualization tab
        viz_tab = QWidget()
        tabs.addTab(viz_tab, "Visualization")
        self._setup_viz_tab(viz_tab)

        # Reference tab
        ref_tab = QWidget()
        tabs.addTab(ref_tab, "Reference")
        self._setup_reference_tab(ref_tab)

    def _setup_data_tab(self, parent: QWidget) -> None:
        """Set up the data loading tab."""
        layout = QVBoxLayout(parent)

        # File selection
        file_group = QGroupBox("Parquet Data")
        file_layout = QVBoxLayout(file_group)

        file_row = QHBoxLayout()
        self._file_label = QLabel("No file loaded")
        self._file_label.setWordWrap(True)
        file_row.addWidget(self._file_label)

        load_btn = QPushButton("Load...")
        load_btn.clicked.connect(self._load_parquet)
        file_row.addWidget(load_btn)
        file_layout.addLayout(file_row)

        # Stats
        self._stats_label = QLabel("")
        file_layout.addWidget(self._stats_label)

        layout.addWidget(file_group)

        # Atlas selection
        atlas_group = QGroupBox("Atlas")
        atlas_layout = QHBoxLayout(atlas_group)

        atlas_layout.addWidget(QLabel("Atlas:"))
        self._atlas_combo = QComboBox()
        self._atlas_combo.addItems(
            [
                "allen_mouse_10um",
                "allen_mouse_25um",
                "allen_mouse_50um",
            ]
        )
        self._atlas_combo.setCurrentText("allen_mouse_25um")
        atlas_layout.addWidget(self._atlas_combo)

        load_atlas_btn = QPushButton("Load Atlas")
        load_atlas_btn.clicked.connect(self._load_atlas)
        atlas_layout.addWidget(load_atlas_btn)

        layout.addWidget(atlas_group)

        # Atlas status label
        self._atlas_status_label = QLabel("Atlas: Not loaded")
        layout.addWidget(self._atlas_status_label)

        # Selected neurons list
        neurons_group = QGroupBox("Selected Neurons")
        neurons_layout = QVBoxLayout(neurons_group)

        self._neuron_list = QListWidget()
        self._neuron_list.setSelectionMode(QListWidget.ExtendedSelection)
        neurons_layout.addWidget(self._neuron_list)

        neuron_btn_row = QHBoxLayout()
        self._render_btn = QPushButton("Render Selected")
        self._render_btn.clicked.connect(self._render_selected_neurons)
        neuron_btn_row.addWidget(self._render_btn)

        self._clear_neurons_btn = QPushButton("Clear")
        self._clear_neurons_btn.clicked.connect(self._clear_neuron_layers)
        neuron_btn_row.addWidget(self._clear_neurons_btn)

        neurons_layout.addLayout(neuron_btn_row)

        layout.addWidget(neurons_group)
        layout.addStretch()

    def _setup_regions_tab(self, parent: QWidget) -> None:
        """Set up the region selection tab."""
        layout = QVBoxLayout(parent)

        # Region selector widget
        self._region_selector = RegionSelectorWidget()
        self._region_selector.selection_changed.connect(self._on_regions_selected)
        layout.addWidget(self._region_selector)

        # Query button
        btn_row = QHBoxLayout()
        self._query_btn = QPushButton("Find Neurons in Selected Regions")
        self._query_btn.clicked.connect(self._query_neurons_by_region)
        self._query_btn.setEnabled(False)
        btn_row.addWidget(self._query_btn)
        layout.addLayout(btn_row)

    def _setup_viz_tab(self, parent: QWidget) -> None:
        """Set up the visualization settings tab."""
        layout = QVBoxLayout(parent)

        # Render mode
        mode_group = QGroupBox("Render Mode")
        mode_layout = QVBoxLayout(mode_group)

        self._render_mode_combo = QComboBox()
        self._render_mode_combo.addItems(["Lines", "Points", "Both"])
        self._render_mode_combo.setCurrentText("Lines")
        mode_layout.addWidget(self._render_mode_combo)

        layout.addWidget(mode_group)

        # Point settings
        point_group = QGroupBox("Point Settings")
        point_layout = QVBoxLayout(point_group)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Size:"))
        self._point_size_spin = QSpinBox()
        self._point_size_spin.setRange(1, 50)
        self._point_size_spin.setValue(5)
        size_row.addWidget(self._point_size_spin)
        point_layout.addLayout(size_row)

        self._color_by_type_cb = QCheckBox("Color by node type")
        self._color_by_type_cb.setChecked(True)
        point_layout.addWidget(self._color_by_type_cb)

        layout.addWidget(point_group)

        # Line settings
        line_group = QGroupBox("Line Settings")
        line_layout = QVBoxLayout(line_group)

        width_row = QHBoxLayout()
        width_row.addWidget(QLabel("Width:"))
        self._line_width_spin = QSpinBox()
        self._line_width_spin.setRange(1, 20)
        self._line_width_spin.setValue(2)
        width_row.addWidget(self._line_width_spin)
        line_layout.addLayout(width_row)

        layout.addWidget(line_group)

        # Opacity
        opacity_group = QGroupBox("Opacity")
        opacity_layout = QHBoxLayout(opacity_group)

        self._opacity_slider = QSlider(Qt.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(80)
        opacity_layout.addWidget(self._opacity_slider)

        self._opacity_label = QLabel("80%")
        self._opacity_slider.valueChanged.connect(
            lambda v: self._opacity_label.setText(f"{v}%")
        )
        opacity_layout.addWidget(self._opacity_label)

        layout.addWidget(opacity_group)

        layout.addStretch()

    def _setup_reference_tab(self, parent: QWidget) -> None:
        """Set up the reference layers tab."""
        layout = QVBoxLayout(parent)

        # Template
        template_group = QGroupBox("Reference Template")
        template_layout = QVBoxLayout(template_group)

        self._show_template_cb = QCheckBox("Show template")
        self._show_template_cb.setChecked(False)
        self._show_template_cb.stateChanged.connect(self._toggle_template)
        template_layout.addWidget(self._show_template_cb)

        template_opacity_row = QHBoxLayout()
        template_opacity_row.addWidget(QLabel("Opacity:"))
        self._template_opacity_slider = QSlider(Qt.Horizontal)
        self._template_opacity_slider.setRange(0, 100)
        self._template_opacity_slider.setValue(30)
        self._template_opacity_slider.valueChanged.connect(self._update_template_opacity)
        template_opacity_row.addWidget(self._template_opacity_slider)
        template_layout.addLayout(template_opacity_row)

        layout.addWidget(template_group)

        # Brain outline
        outline_group = QGroupBox("Brain Outline")
        outline_layout = QVBoxLayout(outline_group)

        self._show_outline_cb = QCheckBox("Show brain outline")
        self._show_outline_cb.setChecked(False)
        self._show_outline_cb.stateChanged.connect(self._toggle_outline)
        outline_layout.addWidget(self._show_outline_cb)

        layout.addWidget(outline_group)

        # Region meshes
        mesh_group = QGroupBox("Region Meshes")
        mesh_layout = QVBoxLayout(mesh_group)

        self._show_region_meshes_cb = QCheckBox("Show selected region meshes")
        self._show_region_meshes_cb.setChecked(False)
        self._show_region_meshes_cb.stateChanged.connect(self._toggle_region_meshes)
        mesh_layout.addWidget(self._show_region_meshes_cb)

        mesh_opacity_row = QHBoxLayout()
        mesh_opacity_row.addWidget(QLabel("Opacity:"))
        self._mesh_opacity_slider = QSlider(Qt.Horizontal)
        self._mesh_opacity_slider.setRange(0, 100)
        self._mesh_opacity_slider.setValue(30)
        mesh_opacity_row.addWidget(self._mesh_opacity_slider)
        mesh_layout.addLayout(mesh_opacity_row)

        layout.addWidget(mesh_group)

        layout.addStretch()

    def _load_parquet(self) -> None:
        """Open file dialog and load Parquet file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Parquet File",
            "",
            "Parquet Files (*.parquet);;All Files (*)",
        )

        if not filepath:
            return

        try:
            self._db = NeuronDatabase(filepath)
            self._file_label.setText(Path(filepath).name)

            # Update stats
            stats = self._db.get_statistics()
            self._stats_label.setText(
                f"Nodes: {stats['n_nodes']:,} | "
                f"Files: {stats['n_files']:,} | "
                f"Subjects: {stats['n_subjects']:,} | "
                f"Regions: {stats['n_regions']:,}"
            )

            self._query_btn.setEnabled(True)
            logger.info(f"Loaded Parquet file: {filepath}")

        except Exception as e:
            logger.error(f"Failed to load Parquet file: {e}")
            self._file_label.setText(f"Error: {e}")

    def _load_atlas(self) -> None:
        """Load the selected BrainGlobe atlas."""
        atlas_name = self._atlas_combo.currentText()

        self._atlas_status_label.setText(f"Atlas: Loading {atlas_name}...")
        # Force UI update
        self._atlas_status_label.repaint()

        try:
            self._atlas = BrainGlobeAtlas(atlas_name)
            self._region_selector.set_atlas(self._atlas)
            self._atlas_status_label.setText(
                f"Atlas: {atlas_name} ({len(self._atlas.structures)} structures)"
            )
            logger.info(f"Loaded atlas: {atlas_name}")

        except Exception as e:
            logger.error(f"Failed to load atlas: {e}")
            self._atlas_status_label.setText(f"Atlas: Error - {e}")

    def _on_regions_selected(self, acronyms: list[str]) -> None:
        """Handle region selection changes."""
        # Update region meshes if enabled
        if self._show_region_meshes_cb.isChecked():
            self._update_region_meshes(acronyms)

    def _query_neurons_by_region(self) -> None:
        """Query neurons in selected regions."""
        if self._db is None:
            return

        acronyms = self._region_selector.get_selected_acronyms(include_children=True)
        if not acronyms:
            return

        try:
            result = self._db.get_neurons_by_region(acronyms)

            # Update neuron list
            self._neuron_list.clear()
            for _, row in result.iterrows():
                item = QListWidgetItem(f"{row['file_id']} ({row['subject']})")
                item.setData(Qt.UserRole, row["file_id"])
                self._neuron_list.addItem(item)

            logger.info(f"Found {len(result)} neurons in selected regions")

        except Exception as e:
            logger.error(f"Query failed: {e}")

    def _render_selected_neurons(self) -> None:
        """Render the selected neurons from the list."""
        selected_items = self._neuron_list.selectedItems()
        if not selected_items or self._db is None:
            return

        file_ids = [item.data(Qt.UserRole) for item in selected_items]

        # Clear existing neuron layers
        self._clear_neuron_layers()

        render_mode = self._render_mode_combo.currentText()
        opacity = self._opacity_slider.value() / 100.0

        for file_id in file_ids:
            if render_mode in ("Lines", "Both"):
                self._add_neuron_lines(file_id, opacity)
            if render_mode in ("Points", "Both"):
                self._add_neuron_points(file_id, opacity)

    def _add_neuron_lines(self, file_id: str, opacity: float) -> None:
        """Add a neuron as a shapes layer with lines."""
        coords, edges = self._db.get_neuron_lines(file_id)

        if len(coords) == 0 or len(edges) == 0:
            return

        # Create line segments
        lines = []
        for edge in edges:
            lines.append([coords[edge[0]], coords[edge[1]]])

        layer = self.viewer.add_shapes(
            lines,
            shape_type="line",
            edge_width=self._line_width_spin.value(),
            edge_color="cyan",
            name=f"Lines: {file_id}",
            opacity=opacity,
        )

        self._current_neuron_layers.append(layer)

    def _add_neuron_points(self, file_id: str, opacity: float) -> None:
        """Add a neuron as a points layer."""
        df = self._db.get_neurons_for_rendering([file_id])

        if df.empty:
            return

        coords = df[["x", "y", "z"]].values

        # Color by node type if enabled
        if self._color_by_type_cb.isChecked():
            # Node type colors
            type_colors = {
                1: [1, 0, 0, 1],  # Soma - red
                2: [0, 0, 1, 1],  # Axon - blue
                3: [0, 1, 0, 1],  # Basal dendrite - green
                4: [1, 1, 0, 1],  # Apical dendrite - yellow
            }
            colors = np.array(
                [type_colors.get(t, [0.5, 0.5, 0.5, 1]) for t in df["type"].values]
            )
        else:
            colors = "magenta"

        layer = self.viewer.add_points(
            coords,
            size=self._point_size_spin.value(),
            face_color=colors,
            name=f"Points: {file_id}",
            opacity=opacity,
        )

        self._current_neuron_layers.append(layer)

    def _clear_neuron_layers(self) -> None:
        """Remove all current neuron layers."""
        for layer in self._current_neuron_layers:
            try:
                self.viewer.layers.remove(layer)
            except ValueError:
                pass  # Layer already removed

        self._current_neuron_layers.clear()

    def _toggle_template(self, state: int) -> None:
        """Toggle the template layer visibility."""
        if self._atlas is None:
            self._load_atlas()
            if self._atlas is None:
                self._show_template_cb.setChecked(False)
                return

        layer_name = "Allen Template"

        if state == Qt.Checked:
            # Check if layer already exists
            existing = [l for l in self.viewer.layers if l.name == layer_name]
            if not existing:
                opacity = self._template_opacity_slider.value() / 100.0
                add_allen_template(self.viewer, self._atlas, opacity=opacity)
        else:
            # Remove template layer
            for layer in self.viewer.layers:
                if layer.name == layer_name:
                    self.viewer.layers.remove(layer)
                    break

    def _update_template_opacity(self, value: int) -> None:
        """Update the template layer opacity."""
        opacity = value / 100.0
        for layer in self.viewer.layers:
            if layer.name == "Allen Template":
                layer.opacity = opacity
                break

    def _toggle_outline(self, state: int) -> None:
        """Toggle the brain outline visibility."""
        if self._atlas is None:
            self._load_atlas()
            if self._atlas is None:
                self._show_outline_cb.setChecked(False)
                return

        layer_name = "Brain Outline"

        if state == Qt.Checked:
            existing = [l for l in self.viewer.layers if l.name == layer_name]
            if not existing:
                # Switch to 3D mode for mesh viewing
                if self.viewer.dims.ndisplay == 2:
                    self.viewer.dims.ndisplay = 3
                    show_info("Switched to 3D view for mesh display")
                add_brain_outline(self.viewer, self._atlas)
        else:
            for layer in self.viewer.layers:
                if layer.name == layer_name:
                    self.viewer.layers.remove(layer)
                    break

    def _toggle_region_meshes(self, state: int) -> None:
        """Toggle region mesh visibility."""
        if state == Qt.Checked:
            acronyms = self._region_selector.get_selected_acronyms(include_children=False)
            self._update_region_meshes(acronyms)
        else:
            remove_region_layers(self.viewer)

    def _update_region_meshes(self, acronyms: list[str]) -> None:
        """Update displayed region meshes."""
        if self._atlas is None:
            self._load_atlas()
            if self._atlas is None:
                return

        # Remove existing region meshes
        remove_region_layers(self.viewer)

        if not self._show_region_meshes_cb.isChecked():
            return

        if not acronyms:
            return

        # Switch to 3D mode for mesh viewing
        if self.viewer.dims.ndisplay == 2:
            self.viewer.dims.ndisplay = 3
            show_info("Switched to 3D view for mesh display")

        # Add new meshes
        opacity = self._mesh_opacity_slider.value() / 100.0
        for acronym in acronyms:
            add_region_mesh(self.viewer, self._atlas, acronym, opacity=opacity)
