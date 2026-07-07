"""Qt widget for lookup-based isocortex flatmap projection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from napari.utils.notifications import show_info, show_warning
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..flatmap_export import export_projected_nodes_csv
from ..flatmap_heatmap import (
    DEFAULT_FLATMAP_DEPTH_BIN_UM,
    DEFAULT_FLATMAP_XY_BINS,
    FlatmapLookupStats,
    FlatmapRenderResult,
    FlatmapRenderSummary,
    build_flatmap_render_data,
    compute_flatmap_lookup_stats,
)
from ..flatmap_loader import FLATMAP_STYLE_FILENAMES, load_flatmap_volume_set
from ..flatmap_parquet import augment_neuron_parquet_with_flatmap
from ..flatmap_projection import (
    COORDINATE_MODE_MICRONS,
    COORDINATE_MODE_VOXELS,
    DEFAULT_CCF_RESOLUTION_UM,
    FlatmapProjectionResult,
    ProjectionSummary,
    project_and_build_segments,
)

logger = logging.getLogger(__name__)

_SOURCE_SELECTED = "selected"
_SOURCE_ALL = "all"
_RENDER_HEATMAP = "heatmap"
_RENDER_POINTS = "points"
_OLD_SHAPES_LAYER_NAME = "Isocortex Flatmap Traces"
_HEATMAP_LAYER_NAME = "Isocortex Flatmap Heatmap"
_POINTS_LAYER_NAME = "Isocortex Flatmap Points"
_FLATMAP_RENDER_LAYER_NAMES = {
    _OLD_SHAPES_LAYER_NAME,
    _HEATMAP_LAYER_NAME,
    _POINTS_LAYER_NAME,
}
_DEFAULT_TRACE_COLOR = np.asarray([0.5, 0.5, 0.5, 1.0], dtype=float)


class FlatmapProjectionWidget(QWidget):
    """Project loaded neuron rows into precomputed isocortex flatmap space."""

    def __init__(
        self,
        viewer,
        *,
        database_provider: Callable[[], object | None],
        selected_file_ids_provider: Callable[[], list[object]],
        table_file_ids_provider: Callable[[], list[object]],
        color_map_provider: Callable[[], dict[object, list[float]]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._database_provider = database_provider
        self._selected_file_ids_provider = selected_file_ids_provider
        self._table_file_ids_provider = table_file_ids_provider
        self._color_map_provider = color_map_provider

        self._flatmap_path: Path | None = None
        self._depth_path: Path | None = None
        self._projection_layer = None
        self._last_projected_nodes: pd.DataFrame | None = None
        self._last_summary: ProjectionSummary | None = None
        self._last_render_summary: FlatmapRenderSummary | None = None
        self._lookup_stats_cache_key: tuple[object, ...] | None = None
        self._lookup_stats_cache: FlatmapLookupStats | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Build the tab UI."""
        parent_layout = QVBoxLayout(self)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        parent_layout.addWidget(scroll_area)

        content = QWidget()
        scroll_area.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)

        files_group = QGroupBox("Flatmap Lookup Files")
        files_layout = QVBoxLayout(files_group)

        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Style:"))
        self._style_combo = QComboBox()
        self._style_combo.addItem("Both hemispheres, shaped", "both_shaped")
        self._style_combo.addItem("Both hemispheres, square", "both_square")
        self._style_combo.addItem("Single hemisphere, shaped", "single_shaped")
        self._style_combo.addItem("Single hemisphere, square", "single_square")
        self._style_combo.currentIndexChanged.connect(
            self._update_expected_filename_label
        )
        style_row.addWidget(self._style_combo)
        files_layout.addLayout(style_row)

        self._expected_filename_label = QLabel("")
        self._expected_filename_label.setWordWrap(True)
        files_layout.addWidget(self._expected_filename_label)

        flatmap_row = QHBoxLayout()
        self._flatmap_path_label = QLabel("No flatmap selected")
        self._flatmap_path_label.setWordWrap(True)
        flatmap_row.addWidget(self._flatmap_path_label, stretch=1)
        flatmap_btn = QPushButton("Choose Flatmap...")
        flatmap_btn.clicked.connect(self._choose_flatmap_path)
        flatmap_row.addWidget(flatmap_btn)
        files_layout.addLayout(flatmap_row)

        depth_row = QHBoxLayout()
        self._depth_path_label = QLabel("No depth selected")
        self._depth_path_label.setWordWrap(True)
        depth_row.addWidget(self._depth_path_label, stretch=1)
        depth_btn = QPushButton("Choose Depth...")
        depth_btn.clicked.connect(self._choose_depth_path)
        depth_row.addWidget(depth_btn)
        files_layout.addLayout(depth_row)
        layout.addWidget(files_group)

        options_group = QGroupBox("Projection Options")
        options_layout = QVBoxLayout(options_group)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Input:"))
        self._source_combo = QComboBox()
        self._source_combo.addItem("Selected table rows, otherwise all", _SOURCE_SELECTED)
        self._source_combo.addItem("All table rows", _SOURCE_ALL)
        source_row.addWidget(self._source_combo)
        options_layout.addLayout(source_row)

        coordinate_row = QHBoxLayout()
        coordinate_row.addWidget(QLabel("Coordinates:"))
        self._coordinate_mode_combo = QComboBox()
        self._coordinate_mode_combo.addItem(
            "CCF microns from NRRD header",
            COORDINATE_MODE_MICRONS,
        )
        self._coordinate_mode_combo.addItem("10 um voxel indices", COORDINATE_MODE_VOXELS)
        coordinate_row.addWidget(self._coordinate_mode_combo)
        options_layout.addLayout(coordinate_row)

        render_row = QHBoxLayout()
        render_row.addWidget(QLabel("Render:"))
        self._render_mode_combo = QComboBox()
        self._render_mode_combo.addItem("3D Heatmap", _RENDER_HEATMAP)
        self._render_mode_combo.addItem("3D Points", _RENDER_POINTS)
        render_row.addWidget(self._render_mode_combo)
        options_layout.addLayout(render_row)

        xy_bins_row = QHBoxLayout()
        xy_bins_row.addWidget(QLabel("XY bins:"))
        self._xy_bins_spin = QSpinBox()
        self._xy_bins_spin.setRange(16, 1024)
        self._xy_bins_spin.setSingleStep(16)
        self._xy_bins_spin.setValue(DEFAULT_FLATMAP_XY_BINS)
        xy_bins_row.addWidget(self._xy_bins_spin)
        options_layout.addLayout(xy_bins_row)

        depth_bin_row = QHBoxLayout()
        depth_bin_row.addWidget(QLabel("Depth bin:"))
        self._depth_bin_spin = QSpinBox()
        self._depth_bin_spin.setRange(1, 1000)
        self._depth_bin_spin.setSingleStep(5)
        self._depth_bin_spin.setSuffix(" um")
        self._depth_bin_spin.setValue(int(DEFAULT_FLATMAP_DEPTH_BIN_UM))
        depth_bin_row.addWidget(self._depth_bin_spin)
        options_layout.addLayout(depth_bin_row)

        self._negative_one_sentinel_cb = QCheckBox("Treat flatmap (-1, -1) as invalid")
        self._negative_one_sentinel_cb.setChecked(True)
        options_layout.addWidget(self._negative_one_sentinel_cb)
        self._zero_sentinel_cb = QCheckBox("Treat flatmap (0, 0) as invalid")
        options_layout.addWidget(self._zero_sentinel_cb)
        self._exclude_depth_minus_one_cb = QCheckBox("Exclude depth -1 nodes")
        self._exclude_depth_minus_one_cb.setChecked(False)
        options_layout.addWidget(self._exclude_depth_minus_one_cb)
        layout.addWidget(options_group)

        actions_row = QHBoxLayout()
        self._project_btn = QPushButton("Project to Flatmap")
        self._project_btn.clicked.connect(self._project)
        actions_row.addWidget(self._project_btn)
        self._export_btn = QPushButton("Export CSV...")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_csv)
        actions_row.addWidget(self._export_btn)
        self._augment_parquet_btn = QPushButton("Save Augmented Parquet...")
        self._augment_parquet_btn.clicked.connect(self._augment_parquet)
        actions_row.addWidget(self._augment_parquet_btn)
        layout.addLayout(actions_row)

        summary_group = QGroupBox("Projection Summary")
        summary_layout = QVBoxLayout(summary_group)
        self._summary_label = QLabel("No projection run yet.")
        self._summary_label.setWordWrap(True)
        summary_layout.addWidget(self._summary_label)
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        summary_layout.addWidget(self._status_label)
        layout.addWidget(summary_group)

        warning = QLabel(
            "Flatmap coordinates are for visualization and indexing only. "
            "Keep neurite length, branch geometry, and 3D distance calculations "
            "in the original CCF coordinate space."
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)
        layout.addStretch()

        self._update_expected_filename_label()

    def set_flatmap_path(self, path: str | Path | None) -> None:
        """Set the flatmap path, primarily for tests and scripted use."""
        self._flatmap_path = Path(path) if path else None
        text = str(self._flatmap_path) if self._flatmap_path else "No flatmap selected"
        self._flatmap_path_label.setText(text)

    def set_depth_path(self, path: str | Path | None) -> None:
        """Set the depth path, primarily for tests and scripted use."""
        self._depth_path = Path(path) if path else None
        text = str(self._depth_path) if self._depth_path else "No depth selected"
        self._depth_path_label.setText(text)

    def _current_style_key(self) -> str:
        key = self._style_combo.currentData()
        return str(key or "both_shaped")

    def _current_style_filename(self) -> str:
        return FLATMAP_STYLE_FILENAMES.get(
            self._current_style_key(),
            FLATMAP_STYLE_FILENAMES["both_shaped"],
        )

    def _current_coordinate_mode(self) -> str:
        mode = self._coordinate_mode_combo.currentData()
        return str(mode or COORDINATE_MODE_MICRONS)

    def _current_render_mode(self) -> str:
        mode = self._render_mode_combo.currentData()
        return str(mode or _RENDER_HEATMAP)

    def _current_xy_bins(self) -> int:
        return int(self._xy_bins_spin.value())

    def _current_depth_bin_um(self) -> float:
        return float(self._depth_bin_spin.value())

    def _current_source_mode(self) -> str:
        mode = self._source_combo.currentData()
        return str(mode or _SOURCE_SELECTED)

    def _update_expected_filename_label(self) -> None:
        filename = self._current_style_filename()
        self._expected_filename_label.setText(
            f"Expected Zenodo v4.1 flatmap file for this style: {filename}"
        )

    def _choose_flatmap_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Flatmap NRRD",
            "",
            "NRRD Files (*.nrrd);;All Files (*)",
        )
        if path:
            self.set_flatmap_path(path)

    def _choose_depth_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Depth NRRD",
            "",
            "NRRD Files (*.nrrd);;All Files (*)",
        )
        if path:
            self.set_depth_path(path)

    @staticmethod
    def _deduplicate_file_ids(file_ids: list[object]) -> list[object]:
        out: list[object] = []
        seen: set[object] = set()
        for file_id in file_ids:
            if file_id in seen:
                continue
            seen.add(file_id)
            out.append(file_id)
        return out

    def _file_ids_for_source(self, source_mode: str | None = None) -> list[object]:
        mode = source_mode or self._current_source_mode()
        table_ids = self._deduplicate_file_ids(list(self._table_file_ids_provider() or []))
        if mode == _SOURCE_ALL:
            return table_ids

        selected_ids = self._deduplicate_file_ids(
            list(self._selected_file_ids_provider() or [])
        )
        return selected_ids if selected_ids else table_ids

    def _query_nodes(self, file_ids: list[object]) -> pd.DataFrame:
        db = self._database_provider()
        if db is None:
            raise RuntimeError("Load a neuron Parquet before projecting to flatmap.")
        if not file_ids:
            raise RuntimeError("No neurons are available to project.")

        getter = getattr(db, "get_neurons_for_rendering", None)
        if not callable(getter):
            raise RuntimeError("Loaded neuron database does not support rendering queries.")
        nodes = getter(file_ids)
        if nodes is None or nodes.empty:
            raise RuntimeError("No neuron rows matched the requested file IDs.")
        return nodes

    def _projection_request_ready(self) -> None:
        if self._flatmap_path is None:
            raise RuntimeError("Choose a flatmap NRRD file before projecting.")
        if self._depth_path is None:
            raise RuntimeError("Choose depth.nrrd before projecting.")

    def _project(self) -> None:
        """Run projection from the current UI state and render the layer."""
        try:
            self._projection_request_ready()
            file_ids = self._file_ids_for_source()
            nodes = self._query_nodes(file_ids)
            volume_set = load_flatmap_volume_set(self._flatmap_path, self._depth_path)
            result = project_and_build_segments(
                nodes,
                volume_set.flatmap,
                volume_set.depth,
                flatmap_style=self._current_style_filename(),
                coordinate_mode=self._current_coordinate_mode(),
                invalid_zero_sentinel=self._zero_sentinel_cb.isChecked(),
                invalid_negative_one_sentinel=(
                    self._negative_one_sentinel_cb.isChecked()
                ),
                resolution_um=DEFAULT_CCF_RESOLUTION_UM,
                space_directions=volume_set.space_directions,
                space_origin=volume_set.space_origin,
            )
            render_result = build_flatmap_render_data(
                result.projected_nodes,
                volume_set.flatmap,
                volume_set.depth,
                xy_bins=self._current_xy_bins(),
                depth_bin_um=self._current_depth_bin_um(),
                include_depth_minus_one=(
                    not self._exclude_depth_minus_one_cb.isChecked()
                ),
                invalid_zero_sentinel=self._zero_sentinel_cb.isChecked(),
                invalid_negative_one_sentinel=(
                    self._negative_one_sentinel_cb.isChecked()
                ),
                lookup_stats=self._lookup_stats_for_volume_set(
                    volume_set,
                    invalid_zero_sentinel=self._zero_sentinel_cb.isChecked(),
                    invalid_negative_one_sentinel=(
                        self._negative_one_sentinel_cb.isChecked()
                    ),
                ),
            )
            self._apply_projection_result(result, render_result)
            self._status_label.setText(
                f"Rendered {render_result.summary.rendered_nodes:,} of "
                f"{render_result.summary.total_nodes:,} projected node(s)."
            )
            show_info("Flatmap projection complete.")
        except Exception as exc:
            logger.exception("Flatmap projection failed")
            self._status_label.setText(f"Flatmap projection failed: {exc}")
            show_warning(f"Flatmap projection failed: {exc}")

    @staticmethod
    def _path_signature(path: Path) -> tuple[str, int | None, int | None]:
        try:
            stat = path.stat()
        except OSError:
            return (str(path), None, None)
        return (str(path), int(stat.st_size), int(stat.st_mtime_ns))

    def _lookup_stats_cache_key_for(
        self,
        volume_set,
        *,
        invalid_zero_sentinel: bool,
        invalid_negative_one_sentinel: bool,
    ) -> tuple[object, ...]:
        return (
            self._path_signature(Path(volume_set.flatmap_path)),
            self._path_signature(Path(volume_set.depth_path)),
            bool(invalid_zero_sentinel),
            bool(invalid_negative_one_sentinel),
            tuple(volume_set.flatmap.shape),
            str(volume_set.flatmap.dtype),
            tuple(volume_set.depth.shape),
            str(volume_set.depth.dtype),
        )

    def _lookup_stats_for_volume_set(
        self,
        volume_set,
        *,
        invalid_zero_sentinel: bool,
        invalid_negative_one_sentinel: bool,
    ) -> FlatmapLookupStats:
        key = self._lookup_stats_cache_key_for(
            volume_set,
            invalid_zero_sentinel=invalid_zero_sentinel,
            invalid_negative_one_sentinel=invalid_negative_one_sentinel,
        )
        cached_key = getattr(self, "_lookup_stats_cache_key", None)
        cached_stats = getattr(self, "_lookup_stats_cache", None)
        if cached_key == key and cached_stats is not None:
            return cached_stats

        stats = compute_flatmap_lookup_stats(
            volume_set.flatmap,
            volume_set.depth,
            invalid_zero_sentinel=invalid_zero_sentinel,
            invalid_negative_one_sentinel=invalid_negative_one_sentinel,
        )
        self._lookup_stats_cache_key = key
        self._lookup_stats_cache = stats
        return stats

    def _apply_projection_result(
        self,
        result: FlatmapProjectionResult,
        render_result: FlatmapRenderResult,
    ) -> None:
        self._last_projected_nodes = render_result.projected_nodes
        self._last_summary = result.summary
        self._last_render_summary = render_result.summary
        self._summary_label.setText(
            self._format_render_summary(result.summary, render_result.summary)
        )
        self._create_or_update_render_layer(
            render_result,
            result.summary,
            flatmap_style=self._current_style_filename(),
            coordinate_mode=self._current_coordinate_mode(),
            render_mode=self._current_render_mode(),
        )
        self._export_btn.setEnabled(not render_result.projected_nodes.empty)

    def _color_for_file_id(
        self,
        file_id: object,
        color_map: dict[object, list[float]],
    ) -> np.ndarray:
        raw = color_map.get(file_id)
        if raw is None:
            raw = color_map.get(str(file_id))
        if raw is None:
            return _DEFAULT_TRACE_COLOR.copy()

        color = np.asarray(raw, dtype=float).reshape(-1)
        if color.size < 4:
            color = np.pad(color, (0, 4 - color.size), constant_values=1.0)
        return np.clip(color[:4], 0.0, 1.0)

    def _colors_for_file_ids(self, file_ids: list[object]) -> np.ndarray:
        color_map = self._color_map_provider() or {}
        if len(file_ids) == 0:
            return np.empty((0, 4), dtype=float)
        return np.vstack(
            [self._color_for_file_id(file_id, color_map) for file_id in file_ids]
        )

    @staticmethod
    def _heatmap_contrast_limits(volume: np.ndarray) -> tuple[float, float]:
        upper = float(np.nanmax(volume)) if volume.size else 0.0
        if not np.isfinite(upper) or upper <= 0.0:
            return (0.0, 1.0)
        return (0.0, upper)

    @staticmethod
    def _render_layer_name(render_mode: str) -> str:
        return _POINTS_LAYER_NAME if render_mode == _RENDER_POINTS else _HEATMAP_LAYER_NAME

    def _find_layer_by_name(self, name: str):
        layers = getattr(self._viewer, "layers", ())
        for layer in layers:
            if getattr(layer, "name", None) == name:
                return layer
        return None

    def _remove_projection_layer(self, *, except_name: str | None = None) -> None:
        layers = getattr(self._viewer, "layers", None)
        if layers is None:
            self._projection_layer = None
            return
        for layer in list(layers):
            name = getattr(layer, "name", None)
            if name not in _FLATMAP_RENDER_LAYER_NAMES or name == except_name:
                continue
            try:
                layers.remove(layer)
            except ValueError:
                pass
            if layer is self._projection_layer:
                self._projection_layer = None
        if (
            self._projection_layer is not None
            and getattr(self._projection_layer, "name", None) != except_name
        ):
            self._projection_layer = None

    def _render_metadata(
        self,
        projection_summary: ProjectionSummary,
        render_summary: FlatmapRenderSummary,
        *,
        flatmap_style: str,
        coordinate_mode: str,
        render_mode: str,
    ) -> dict[str, object]:
        return {
            "projection_kind": "isocortex_flatmap",
            "flatmap_render_mode": render_mode,
            "flatmap_style": flatmap_style,
            "coordinate_mode": coordinate_mode,
            "projection_summary": projection_summary.to_dict(),
            "render_summary": render_summary.to_dict(),
            "flatmap_path": str(self._flatmap_path) if self._flatmap_path else "",
            "depth_path": str(self._depth_path) if self._depth_path else "",
        }

    def _set_layer_state(
        self,
        layer,
        projected_nodes: pd.DataFrame,
        summary: ProjectionSummary,
        render_summary: FlatmapRenderSummary,
    ) -> None:
        setattr(layer, "_napari_swc_flatmap_projected_nodes", projected_nodes)
        setattr(layer, "_napari_swc_flatmap_summary", summary)
        setattr(layer, "_napari_swc_flatmap_render_summary", render_summary)

    @staticmethod
    def _format_render_summary(
        projection_summary: ProjectionSummary,
        render_summary: FlatmapRenderSummary,
    ) -> str:
        depth_minus_one_action = (
            "rendered"
            if render_summary.includes_depth_minus_one_plane
            else "excluded"
        )
        return (
            f"Input nodes: {projection_summary.total_nodes:,}\n"
            f"Flatmap-valid nodes: {render_summary.flatmap_valid_nodes:,}\n"
            f"Depth-valid nodes: {render_summary.depth_valid_nodes:,}\n"
            f"Depth -1 nodes {depth_minus_one_action}: "
            f"{render_summary.depth_minus_one_nodes:,}\n"
            f"Rendered nodes: {render_summary.rendered_nodes:,}\n"
            f"Nonzero heatmap voxels: {render_summary.nonzero_voxels:,}\n"
            f"Represented traces: {render_summary.traces_represented:,} "
            f"of {projection_summary.total_traces:,}\n"
            f"Invalid flatmap/depth: "
            f"{projection_summary.invalid_flatmap_nodes:,}/"
            f"{projection_summary.invalid_depth_nodes:,}"
        )

    def _create_or_update_render_layer(
        self,
        render_result: FlatmapRenderResult,
        projection_summary: ProjectionSummary,
        *,
        flatmap_style: str,
        coordinate_mode: str,
        render_mode: str,
    ):
        """Create or update the napari depth-aware flatmap render layer."""
        if render_result.summary.rendered_nodes == 0:
            self._remove_projection_layer()
            return None

        layer_name = self._render_layer_name(render_mode)
        self._remove_projection_layer(except_name=layer_name)
        metadata = self._render_metadata(
            projection_summary,
            render_result.summary,
            flatmap_style=flatmap_style,
            coordinate_mode=coordinate_mode,
            render_mode=render_mode,
        )
        layer = self._projection_layer or self._find_layer_by_name(layer_name)

        if render_mode == _RENDER_POINTS:
            layer = self._create_or_update_points_layer(layer, render_result, metadata)
        else:
            layer = self._create_or_update_heatmap_layer(layer, render_result, metadata)

        self._projection_layer = layer
        self._set_layer_state(
            layer,
            render_result.projected_nodes,
            projection_summary,
            render_result.summary,
        )
        data = render_result.points if render_mode == _RENDER_POINTS else render_result.volume
        self._focus_projection_view(layer, data)
        return layer

    def _create_or_update_heatmap_layer(
        self,
        layer,
        render_result: FlatmapRenderResult,
        metadata: dict[str, object],
    ):
        volume = render_result.volume
        contrast_limits = self._heatmap_contrast_limits(volume)
        if layer is None:
            return self._viewer.add_image(
                volume,
                name=_HEATMAP_LAYER_NAME,
                colormap="hot",
                blending="additive",
                rendering="mip",
                opacity=0.8,
                contrast_limits=contrast_limits,
                metadata=metadata,
            )

        blocker = getattr(getattr(layer, "events", None), "blocker_all", None)
        if callable(blocker):
            with blocker():
                layer.data = volume
                layer.metadata = metadata
                layer.contrast_limits = contrast_limits
                layer.visible = True
        else:
            layer.data = volume
            layer.metadata = metadata
            layer.contrast_limits = contrast_limits
            layer.visible = True
        refresh = getattr(layer, "refresh", None)
        if callable(refresh):
            refresh()
        return layer

    def _create_or_update_points_layer(
        self,
        layer,
        render_result: FlatmapRenderResult,
        metadata: dict[str, object],
    ):
        points = render_result.points
        colors = self._colors_for_file_ids(render_result.point_file_ids)
        if layer is None:
            return self._viewer.add_points(
                points,
                name=_POINTS_LAYER_NAME,
                size=2.0,
                face_color=colors,
                border_width=0.0,
                blending="translucent",
                metadata=metadata,
            )

        blocker = getattr(getattr(layer, "events", None), "blocker_all", None)
        if callable(blocker):
            with blocker():
                layer.data = points
                layer.face_color = colors
                layer.metadata = metadata
                layer.visible = True
        else:
            layer.data = points
            layer.face_color = colors
            layer.metadata = metadata
            layer.visible = True
        refresh = getattr(layer, "refresh", None)
        if callable(refresh):
            refresh()
        return layer

    def _focus_projection_view(self, layer, data: np.ndarray) -> None:
        """Switch to 3D and center the camera on the flatmap render bounds."""
        try:
            layer.visible = True
        except Exception:
            pass

        dims = getattr(self._viewer, "dims", None)
        if dims is not None and getattr(dims, "ndisplay", None) != 3:
            try:
                dims.ndisplay = 3
            except Exception:
                logger.debug("Failed to switch viewer to 3D display.", exc_info=True)

        layers = getattr(self._viewer, "layers", None)
        selection = getattr(layers, "selection", None)
        if selection is not None:
            try:
                selection.active = layer
            except Exception:
                logger.debug("Failed to activate flatmap layer.", exc_info=True)

        array = np.asarray(data, dtype=float)
        if array.ndim == 3:
            coords = np.argwhere(array > 0)
            if len(coords) == 0:
                lower = np.zeros(3, dtype=float)
                upper = np.asarray(array.shape, dtype=float) - 1.0
            else:
                lower = np.min(coords, axis=0).astype(float)
                upper = np.max(coords, axis=0).astype(float)
        else:
            coords = array.reshape(-1, 3)
            finite_mask = np.all(np.isfinite(coords), axis=1)
            if not finite_mask.any():
                return
            finite = coords[finite_mask]
            lower = np.min(finite, axis=0)
            upper = np.max(finite, axis=0)
        center = tuple(((lower + upper) / 2.0).tolist())
        span = float(np.max(upper - lower))

        camera = getattr(self._viewer, "camera", None)
        if camera is None:
            reset_view = getattr(self._viewer, "reset_view", None)
            if callable(reset_view):
                reset_view()
            return

        try:
            camera.center = center
        except Exception:
            logger.debug("Failed to center camera on flatmap layer.", exc_info=True)

        if span > 0.0 and np.isfinite(span):
            try:
                camera.zoom = float(np.clip(600.0 / span, 0.01, 10_000.0))
            except Exception:
                logger.debug("Failed to zoom camera to flatmap layer.", exc_info=True)

    def _export_csv(self) -> None:
        if self._last_projected_nodes is None or self._last_projected_nodes.empty:
            show_warning("Run a flatmap projection before exporting CSV.")
            return

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Flatmap Projection CSV",
            "flatmap_projection.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if output_path:
            self._export_current_projection_to_path(output_path)

    def _export_current_projection_to_path(self, output_path: str | Path) -> Path:
        """Export the current projected node table to a specific CSV path."""
        if self._last_projected_nodes is None or self._last_projected_nodes.empty:
            raise RuntimeError("Run a flatmap projection before exporting CSV.")
        saved = export_projected_nodes_csv(self._last_projected_nodes, output_path)
        self._status_label.setText(f"Exported flatmap projection to {saved}.")
        show_info(f"Exported flatmap projection to {saved}")
        return saved

    def _current_source_parquet_path(self) -> Path:
        db = self._database_provider()
        if db is None:
            raise RuntimeError("Load a neuron Parquet before saving augmented Parquet.")
        parquet_path = getattr(db, "parquet_path", None)
        if parquet_path is None:
            raise RuntimeError("Loaded neuron database does not expose a Parquet path.")
        return Path(parquet_path)

    def _augment_parquet(self) -> None:
        try:
            self._projection_request_ready()
            source_path = self._current_source_parquet_path()
        except Exception as exc:
            show_warning(str(exc))
            return

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Augmented Flatmap Parquet",
            str(source_path.with_name(f"{source_path.stem}_flatmap.parquet")),
            "Parquet Files (*.parquet);;All Files (*)",
        )
        if not output_path:
            return

        try:
            self._augment_current_parquet_to_path(output_path)
        except Exception as exc:
            logger.exception("Flatmap Parquet augmentation failed")
            self._status_label.setText(f"Flatmap Parquet augmentation failed: {exc}")
            show_warning(f"Flatmap Parquet augmentation failed: {exc}")

    def _augment_current_parquet_to_path(self, output_path: str | Path):
        """Save a Parquet file augmented with NRRD-derived flatmap columns."""
        self._projection_request_ready()
        source_path = self._current_source_parquet_path()
        file_ids = self._file_ids_for_source()
        if not file_ids:
            raise RuntimeError("No neurons are available to save.")
        summary = augment_neuron_parquet_with_flatmap(
            source_path,
            output_path,
            self._flatmap_path,
            self._depth_path,
            file_ids=file_ids,
            flatmap_style=self._current_style_filename(),
            coordinate_mode=self._current_coordinate_mode(),
            invalid_zero_sentinel=self._zero_sentinel_cb.isChecked(),
            invalid_negative_one_sentinel=self._negative_one_sentinel_cb.isChecked(),
        )
        self._status_label.setText(
            "Saved augmented Parquet to "
            f"{summary.output_parquet} "
            f"({summary.rows:,} rows from {len(file_ids):,} file ID(s); "
            f"{summary.direct_rows:,} direct, "
            f"{summary.mirrored_rows:,} mirrored, "
            f"{summary.unmapped_rows:,} unmapped)."
        )
        show_info(f"Saved augmented Parquet to {summary.output_parquet}")
        return summary
