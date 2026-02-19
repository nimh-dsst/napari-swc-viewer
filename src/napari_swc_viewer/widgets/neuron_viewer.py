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

import matplotlib.pyplot as plt
import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas
from napari.utils.notifications import show_info
from qtpy.QtCore import Qt, QThread, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
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
from .analysis_tab import AnalysisTabWidget
from .region_selector import RegionSelectorWidget
from .slice_projection import NeuronSliceProjector

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

        # Slice projection for 2D viewing
        self._slice_projector = NeuronSliceProjector(napari_viewer, tolerance=100.0)

        # Conversion worker state
        self._convert_thread: QThread | None = None
        self._convert_worker = None

        self._setup_ui()

        # Auto-hide neuron line layers in 2D mode
        self.viewer.dims.events.ndisplay.connect(self._on_ndisplay_changed)

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

        # Analysis tab
        self._analysis_tab = AnalysisTabWidget(self.viewer)
        self._analysis_tab.set_slice_projector(self._slice_projector)
        tabs.addTab(self._analysis_tab, "Analysis")

    def _setup_data_tab(self, parent: QWidget) -> None:
        """Set up the data loading tab."""
        layout = QVBoxLayout(parent)

        # SWC to Parquet conversion
        convert_group = QGroupBox("Convert SWC to Parquet")
        convert_layout = QVBoxLayout(convert_group)

        convert_btn_row = QHBoxLayout()
        convert_dir_btn = QPushButton("From Directory...")
        convert_dir_btn.clicked.connect(self._convert_from_directory)
        convert_btn_row.addWidget(convert_dir_btn)

        convert_files_btn = QPushButton("From Files...")
        convert_files_btn.clicked.connect(self._convert_from_files)
        convert_btn_row.addWidget(convert_files_btn)
        convert_layout.addLayout(convert_btn_row)

        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Resolution (μm):"))
        self._convert_resolution_spin = QSpinBox()
        self._convert_resolution_spin.setRange(10, 100)
        self._convert_resolution_spin.setValue(25)
        res_row.addWidget(self._convert_resolution_spin)
        convert_layout.addLayout(res_row)

        self._convert_progress = QProgressBar()
        self._convert_progress.setVisible(False)
        convert_layout.addWidget(self._convert_progress)

        self._convert_status_label = QLabel("")
        convert_layout.addWidget(self._convert_status_label)

        layout.addWidget(convert_group)

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

        self._render_progress = QProgressBar()
        self._render_progress.setVisible(False)
        neurons_layout.addWidget(self._render_progress)

        self._render_status_label = QLabel("")
        neurons_layout.addWidget(self._render_status_label)

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
        self._line_width_spin.setValue(4)
        self._line_width_spin.valueChanged.connect(self._update_line_width)
        width_row.addWidget(self._line_width_spin)
        line_layout.addLayout(width_row)

        layout.addWidget(line_group)

        # 2D Slice Projection settings
        slice_group = QGroupBox("2D Slice Projection")
        slice_layout = QVBoxLayout(slice_group)

        self._show_slice_projection_cb = QCheckBox("Show in 2D slices")
        self._show_slice_projection_cb.setChecked(False)
        self._show_slice_projection_cb.stateChanged.connect(self._toggle_slice_projection)
        slice_layout.addWidget(self._show_slice_projection_cb)

        self._slice_warning_label = QLabel(
            "Warning: Slice navigation is slower when projection is on."
        )
        self._slice_warning_label.setStyleSheet("color: #cc7700; font-style: italic;")
        self._slice_warning_label.setWordWrap(True)
        self._slice_warning_label.setVisible(False)
        slice_layout.addWidget(self._slice_warning_label)

        thickness_row = QHBoxLayout()
        thickness_row.addWidget(QLabel("Slice thickness (μm):"))
        self._slice_thickness_spin = QSpinBox()
        self._slice_thickness_spin.setRange(10, 2500)
        self._slice_thickness_spin.setValue(100)
        self._slice_thickness_spin.valueChanged.connect(self._update_slice_thickness)
        thickness_row.addWidget(self._slice_thickness_spin)
        slice_layout.addLayout(thickness_row)

        layout.addWidget(slice_group)

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
            self._analysis_tab.set_database(self._db)
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
            self._analysis_tab.set_atlas(self._atlas)
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
        """Render the selected neurons from the list.

        All neurons are batched into a single shapes layer (lines) and/or a
        single points layer instead of one layer per neuron, which is vastly
        faster for large selections.
        """
        selected_items = self._neuron_list.selectedItems()
        if not selected_items or self._db is None:
            return

        file_ids = [item.data(Qt.UserRole) for item in selected_items]
        n = len(file_ids)

        # Show progress UI
        self._render_btn.setEnabled(False)
        self._render_progress.setRange(0, n)
        self._render_progress.setValue(0)
        self._render_progress.setVisible(True)
        self._render_status_label.setText(f"Querying {n} neurons...")
        QApplication.processEvents()

        # Clear existing neuron layers
        self._clear_neuron_layers()

        render_mode = self._render_mode_combo.currentText()
        opacity = self._opacity_slider.value() / 100.0

        # Sample turbo colormap at regular intervals for per-neuron colors
        cmap = plt.get_cmap("turbo")
        neuron_colors = [list(cmap(t)) for t in np.linspace(0, 1, n)]

        # Scale to match atlas mesh (coordinates are in microns)
        scale = None
        if self._atlas is not None:
            scale = [1.0 / res for res in self._atlas.resolution]

        # --- Lines ---
        if render_mode in ("Lines", "Both"):
            # Single batch query for all neurons
            all_data = self._db.get_neuron_lines_batch(file_ids)

            self._render_status_label.setText(f"Building line segments for {n} neurons...")
            QApplication.processEvents()

            all_lines = []
            all_edge_colors = []
            projector_batch = {}
            rendered_file_ids = []
            segments_per_neuron = []

            for i, (file_id, color) in enumerate(zip(file_ids, neuron_colors)):
                if file_id not in all_data:
                    continue
                coords, edges = all_data[file_id]
                if len(edges) == 0:
                    continue

                # Vectorized line segment building
                segments = np.stack(
                    [coords[edges[:, 0]], coords[edges[:, 1]]], axis=1
                )
                all_lines.append(segments)

                color_arr = np.empty((len(edges), 4))
                color_arr[:] = color[:4]
                all_edge_colors.append(color_arr)

                projector_batch[file_id] = (coords, edges, tuple(color))
                rendered_file_ids.append(file_id)
                segments_per_neuron.append(len(edges))

                self._render_progress.setValue(i + 1)
                if (i + 1) % 10 == 0:
                    QApplication.processEvents()

            if all_lines:
                merged_lines = np.concatenate(all_lines)
                merged_colors = np.concatenate(all_edge_colors)

                total_segs = len(merged_lines)
                self._render_status_label.setText(
                    f"Adding {total_segs:,} line segments to viewer..."
                )
                self._render_progress.setRange(0, 0)  # indeterminate
                QApplication.processEvents()

                layer = self.viewer.add_shapes(
                    merged_lines,
                    shape_type="line",
                    edge_width=self._line_width_spin.value(),
                    edge_color=merged_colors,
                    name="Neuron Lines",
                    opacity=opacity,
                    scale=scale,
                    metadata={
                        "file_ids": rendered_file_ids,
                        "segments_per_neuron": segments_per_neuron,
                    },
                )
                self._current_neuron_layers.append(layer)

            # Batch update slice projector (single rebuild)
            self._slice_projector.set_scale(scale)
            self._slice_projector.add_neuron_data_batch(projector_batch)

        # --- Points ---
        if render_mode in ("Points", "Both"):
            self._render_status_label.setText("Querying point data...")
            self._render_progress.setRange(0, 0)  # indeterminate
            QApplication.processEvents()

            # Single batch query for all neurons
            df = self._db.get_neurons_for_rendering(file_ids)

            if not df.empty:
                self._render_status_label.setText(
                    f"Adding {len(df):,} points to viewer..."
                )
                QApplication.processEvents()

                coords = df[["x", "y", "z"]].values

                if self._color_by_type_cb.isChecked():
                    type_colors = {
                        1: [1, 0, 0, 1],  # Soma - red
                        2: [0, 0, 1, 1],  # Axon - blue
                        3: [0, 1, 0, 1],  # Basal dendrite - green
                        4: [1, 1, 0, 1],  # Apical dendrite - yellow
                    }
                    colors = np.array(
                        [
                            type_colors.get(t, [0.5, 0.5, 0.5, 1])
                            for t in df["type"].values
                        ]
                    )
                else:
                    # Per-point color based on which neuron each point belongs to
                    color_map = dict(zip(file_ids, neuron_colors))
                    colors = np.array(
                        [color_map[fid][:4] for fid in df["file_id"].values]
                    )

                layer = self.viewer.add_points(
                    coords,
                    size=self._point_size_spin.value(),
                    face_color=colors,
                    name="Neuron Points",
                    opacity=opacity,
                    scale=scale,
                    metadata={
                        "file_ids_per_point": df["file_id"].values.tolist(),
                    },
                )
                self._current_neuron_layers.append(layer)

        # Re-apply cluster colors if a clustering result exists
        self._analysis_tab.apply_cluster_colors()

        # Hide progress UI
        self._render_progress.setVisible(False)
        self._render_status_label.setText(f"Rendered {n} neurons.")
        self._render_btn.setEnabled(True)

    def _clear_neuron_layers(self) -> None:
        """Remove all current neuron layers."""
        for layer in self._current_neuron_layers:
            try:
                self.viewer.layers.remove(layer)
            except ValueError:
                pass  # Layer already removed

        self._current_neuron_layers.clear()

        # Clear slice projector data
        self._slice_projector.clear()

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

    def _on_ndisplay_changed(self, event) -> None:
        """Auto-hide neuron line/point layers in 2D to keep slice scrubbing fast."""
        if not self._current_neuron_layers:
            return

        is_2d = self.viewer.dims.ndisplay == 2
        if is_2d:
            self.viewer.status = "Switching to 2D view..."
        else:
            self.viewer.status = "Rendering 3D neuron layers..."
        # Defer the heavy work so the status bar paints first
        QTimer.singleShot(0, lambda: self._apply_layer_visibility(not is_2d))

    def _apply_layer_visibility(self, visible: bool) -> None:
        """Set visibility on all neuron layers and clear the status message."""
        for layer in self._current_neuron_layers:
            layer.visible = visible
        self.viewer.status = "Ready"

    def _toggle_slice_projection(self, state: int) -> None:
        """Toggle the 2D slice projection visibility."""
        enabled = state == Qt.Checked
        self._slice_projector.enabled = enabled
        self._slice_warning_label.setVisible(enabled)

    def _update_slice_thickness(self, value: int) -> None:
        """Update the slice projection thickness/tolerance."""
        self._slice_projector.tolerance = float(value)

    def _update_line_width(self, value: int) -> None:
        """Update line width for both neuron layers and projection."""
        for layer in self._current_neuron_layers:
            if hasattr(layer, "edge_width"):
                layer.edge_width = value
        self._slice_projector.edge_width = value

    # --- SWC-to-Parquet conversion ---

    def _convert_from_directory(self) -> None:
        """Pick a directory of SWC files and convert to Parquet."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory of SWC Files"
        )
        if not directory:
            return

        swc_files = sorted(Path(directory).rglob("*.swc"))
        if not swc_files:
            self._convert_status_label.setText("No SWC files found in directory.")
            return

        self._prompt_output_and_convert([str(f) for f in swc_files])

    def _convert_from_files(self) -> None:
        """Pick individual SWC files and convert to Parquet."""
        filepaths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select SWC Files",
            "",
            "SWC Files (*.swc);;All Files (*)",
        )
        if not filepaths:
            return

        self._prompt_output_and_convert(filepaths)

    def _prompt_output_and_convert(self, swc_paths: list[str]) -> None:
        """Ask for output path and start conversion."""
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Parquet File",
            "neurons.parquet",
            "Parquet Files (*.parquet)",
        )
        if not output_path:
            return

        self._start_conversion(swc_paths, output_path)

    def _start_conversion(self, swc_paths: list[str], output_path: str) -> None:
        """Launch the background conversion worker."""
        from ..workers import ConvertWorker

        resolution = self._convert_resolution_spin.value()

        self._convert_progress.setVisible(True)
        self._convert_progress.setRange(0, len(swc_paths))
        self._convert_progress.setValue(0)
        self._convert_status_label.setText(
            f"Converting {len(swc_paths)} SWC files..."
        )

        self._convert_thread = QThread()
        self._convert_worker = ConvertWorker(swc_paths, output_path, resolution)
        self._convert_worker.moveToThread(self._convert_thread)

        self._convert_thread.started.connect(self._convert_worker.run)
        self._convert_worker.progress.connect(self._on_convert_progress)
        self._convert_worker.finished.connect(self._on_convert_finished)
        self._convert_worker.error.connect(self._on_convert_error)
        self._convert_worker.finished.connect(self._convert_thread.quit)
        self._convert_worker.error.connect(self._convert_thread.quit)

        self._convert_thread.start()

    def _on_convert_progress(self, message: str, current: int, total: int) -> None:
        """Handle conversion progress updates."""
        self._convert_progress.setValue(current)
        self._convert_status_label.setText(message)

    def _on_convert_finished(self, output_path: str, n_files: int) -> None:
        """Handle conversion completion."""
        self._convert_progress.setVisible(False)
        self._convert_status_label.setText(
            f"Done! Converted {n_files} files → {Path(output_path).name}"
        )
        logger.info(f"SWC-to-Parquet conversion complete: {output_path}")

    def _on_convert_error(self, error_msg: str) -> None:
        """Handle conversion error."""
        self._convert_progress.setVisible(False)
        self._convert_status_label.setText(f"Error: {error_msg}")
        logger.error(f"SWC-to-Parquet conversion failed: {error_msg}")
