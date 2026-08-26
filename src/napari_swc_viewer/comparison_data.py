"""Data access and render preparation for comparison-board cells."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import duckdb
import numpy as np

from .analysis.clustering import query_ccf_soma_coordinates
from .analysis.flatmap_correlation import query_flatmap_soma_coordinates
from .cluster_assignments import ClusterAssignmentSet, ClusterAssignmentStore
from .comparison import (
    REDUCTION_PROJECTION,
    SOURCE_CCF_HEATMAP,
    SOURCE_CCF_SOMAS,
    SOURCE_FLATMAP_ARBOR_HEATMAP,
    SOURCE_FLATMAP_SOMAS,
    ClusterLabelMatch,
    ComparisonCellSpec,
    assignment_display_colors,
    ccf_plane_axes,
    comparison_source_id,
    heatmap_filter_signature,
    project_ccf_points,
    reduce_ccf_volume,
)
from .flatmap_heatmap import (
    DEFAULT_FLATMAP_DEPTH_BIN_UM,
    FLATMAP_HEATMAP_COLOR_CLUSTER,
    build_flatmap_heatmap_volume_result,
    resolve_flatmap_bin_counts,
)
from .flatmap_parquet import read_flatmap_parquet_transform_info


@dataclass(frozen=True)
class ComparisonHeatmapSource:
    """One app-created CCF heatmap available to the comparison board."""

    source_id: str
    name: str
    data: np.ndarray
    metadata: Mapping[str, object]
    scale: tuple[float, float, float]
    translate: tuple[float, float, float]

    @property
    def assignment_id(self) -> str | None:
        value = self.metadata.get("comparison_assignment_id")
        return None if value in (None, "") else str(value)

    @property
    def cluster_label(self) -> int | None:
        value = self.metadata.get("heatmap_cluster")
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @property
    def atlas_resolution_um(self) -> tuple[float, float, float] | None:
        provenance = self.metadata.get("comparison_atlas_provenance")
        if not isinstance(provenance, Mapping):
            return None
        raw = provenance.get("resolution_um")
        try:
            values = tuple(float(value) for value in raw)  # type: ignore[union-attr]
        except (TypeError, ValueError):
            return None
        if len(values) != 3 or not np.all(np.isfinite(values)):
            return None
        if any(value <= 0.0 for value in values):
            return None
        return values

    def atlas_provenance_key(
        self,
        fallback_resolution: tuple[float, float, float],
    ) -> tuple[object, ...]:
        provenance = self.metadata.get("comparison_atlas_provenance")
        if isinstance(provenance, Mapping):
            version = provenance.get("atlas_version")
            return (
                provenance.get("atlas_name") or self.metadata.get("atlas_name"),
                None if version is None else str(version),
                self.atlas_resolution_um or fallback_resolution,
            )
        return (
            self.metadata.get("atlas_name"),
            None,
            fallback_resolution,
        )


@dataclass(frozen=True)
class ComparisonHeatmapGroup:
    """One selectable, compatible CCF heatmap source or cluster set."""

    label: str
    source_ids: tuple[str, ...]
    assignment_id: str | None
    filter_signature: tuple[object, ...]


@dataclass
class ComparisonRenderData:
    """Qt-independent data needed to draw one comparison cell."""

    cell_id: str
    title: str
    source_kind: str
    assigned_count: int
    omitted_count: int
    subtitle: str
    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]
    compatibility_key: tuple[object, ...]
    coordinate_provenance: dict[str, object] = field(default_factory=dict)
    intensity_key: tuple[object, ...] | None = None
    heatmaps: dict[int, np.ndarray] = field(default_factory=dict)
    colors: dict[int, list[float]] = field(default_factory=dict)
    points: np.ndarray | None = None
    point_colors: np.ndarray | None = None
    matches: tuple[ClusterLabelMatch, ...] = ()
    observed_intensity_max: float = 0.0
    intensity_max: float | None = None
    provenance: dict[str, object] = field(default_factory=dict)

    def to_provenance(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "title": self.title,
            "source_kind": self.source_kind,
            "assigned_neurons": int(self.assigned_count),
            "omitted_or_unassigned_neurons": int(self.omitted_count),
            "subtitle": self.subtitle,
            "x_bounds": list(self.x_bounds),
            "y_bounds": list(self.y_bounds),
            "coordinate_provenance": dict(self.coordinate_provenance),
            "intensity_max": self.intensity_max,
            "cluster_matches": [match.to_state() for match in self.matches],
            **self.provenance,
        }


def _three_floats(value: object, *, default: float) -> tuple[float, float, float]:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        array = np.asarray([], dtype=float)
    if array.size < 3 or not np.all(np.isfinite(array[:3])):
        return (default, default, default)
    return tuple(float(item) for item in array[:3])


def _assignment_counts(assignment: ClusterAssignmentSet) -> tuple[int, int]:
    assigned_ids = {str(value) for value in assignment.assignments}
    cohort = {str(value) for value in assignment.input_file_ids}
    cohort.update(str(value) for value in assignment.unassigned_neuron_ids)
    return len(assigned_ids), max(0, len(cohort - assigned_ids))


def _assignment_provenance(
    assignment: ClusterAssignmentSet,
    colors: Mapping[int, Sequence[float]],
) -> dict[str, object]:
    """Return saved and display-only cluster identities for an export."""
    cluster_ids = sorted({int(label) for label in assignment.assignments.values()})
    return {
        "assignment_id": assignment.assignment_id,
        "assignment_name": assignment.name,
        "original_cluster_ids": cluster_ids,
        "saved_palette": {
            str(label): list(assignment.label_colors.get(label, ()))
            for label in cluster_ids
        },
        "display_palette": {
            str(label): list(colors.get(label, ())) for label in cluster_ids
        },
    }


def _finite_bounds(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (0.0, 1.0)
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if upper <= lower:
        pad = max(abs(lower) * 0.01, 0.5)
        return (lower - pad, upper + pad)
    pad = (upper - lower) * 0.02
    return (lower - pad, upper + pad)


class ComparisonDataProvider:
    """Resolve comparison recipes against the current loaded project."""

    def __init__(
        self,
        *,
        database_provider: Callable[[], object | None],
        assignment_store_provider: Callable[[], ClusterAssignmentStore],
        viewer_layers_provider: Callable[[], Sequence[object]],
        atlas_provider: Callable[[], object | None],
    ) -> None:
        self._database_provider = database_provider
        self._assignment_store_provider = assignment_store_provider
        self._viewer_layers_provider = viewer_layers_provider
        self._atlas_provider = atlas_provider
        self._flatmap_cache: dict[tuple[object, ...], ComparisonRenderData] = {}
        self._heatmap_source_snapshot: tuple[ComparisonHeatmapSource, ...] = ()
        self.refresh_heatmap_sources()

    def clear_cache(self) -> None:
        self._flatmap_cache.clear()

    def parquet_path(self) -> Path | None:
        database = self._database_provider()
        path = getattr(database, "parquet_path", None)
        return Path(path) if path is not None else None

    def source_signature(self) -> dict[str, object] | None:
        """Return the inexpensive file identity also used by render caching."""
        path = self.parquet_path()
        if path is None:
            return None
        try:
            stat = path.stat()
        except OSError:
            return {"path": str(path)}
        return {
            "path": str(path.resolve()),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    def assignments(self) -> tuple[ClusterAssignmentSet, ...]:
        return self._assignment_store_provider().sets()

    def assignment(self, assignment_id: str | None) -> ClusterAssignmentSet | None:
        return self._assignment_store_provider().get(assignment_id)

    def _reference_assignment(
        self, assignment_id: str | None
    ) -> ClusterAssignmentSet | None:
        return self.assignment(assignment_id)

    def heatmap_sources(self) -> tuple[ComparisonHeatmapSource, ...]:
        """Return the main-thread snapshot of available Analysis heatmaps."""
        return self._heatmap_source_snapshot

    def refresh_heatmap_sources(self) -> tuple[ComparisonHeatmapSource, ...]:
        """Snapshot napari heatmap sources before background rendering starts."""
        sources: list[ComparisonHeatmapSource] = []
        for layer in self._viewer_layers_provider():
            source_id = comparison_source_id(layer)
            metadata = getattr(layer, "metadata", None)
            data = getattr(layer, "data", None)
            if (
                source_id is None
                or not isinstance(metadata, Mapping)
                or metadata.get("heatmap_kind") != "analysis"
                or data is None
            ):
                continue
            array = np.asarray(data)
            if array.ndim != 3:
                continue
            sources.append(
                ComparisonHeatmapSource(
                    source_id=source_id,
                    name=str(getattr(layer, "name", source_id)),
                    data=array,
                    metadata=metadata,
                    scale=_three_floats(getattr(layer, "scale", None), default=1.0),
                    translate=_three_floats(
                        getattr(layer, "translate", None), default=0.0
                    ),
                )
            )
        self._heatmap_source_snapshot = tuple(sources)
        return self._heatmap_source_snapshot

    def heatmap_groups(self) -> tuple[ComparisonHeatmapGroup, ...]:
        grouped: dict[tuple[object, ...], list[ComparisonHeatmapSource]] = {}
        singles: list[ComparisonHeatmapGroup] = []
        for source in self.refresh_heatmap_sources():
            if source.assignment_id is None or source.cluster_label is None:
                singles.append(
                    ComparisonHeatmapGroup(
                        label=source.name,
                        source_ids=(source.source_id,),
                        assignment_id=source.assignment_id,
                        filter_signature=heatmap_filter_signature(source.metadata),
                    )
                )
                continue
            signature = heatmap_filter_signature(source.metadata)
            grouped.setdefault(signature, []).append(source)

        assignment_by_id = {
            assignment.assignment_id: assignment for assignment in self.assignments()
        }
        sets: list[ComparisonHeatmapGroup] = []
        for signature, sources in grouped.items():
            sources.sort(key=lambda item: int(item.cluster_label or 0))
            assignment_id = sources[0].assignment_id
            assignment = assignment_by_id.get(str(assignment_id))
            assignment_name = (
                assignment.name if assignment is not None else str(assignment_id)
            )
            sets.append(
                ComparisonHeatmapGroup(
                    label=f"{assignment_name} — {len(sources)} cluster heatmaps",
                    source_ids=tuple(source.source_id for source in sources),
                    assignment_id=assignment_id,
                    filter_signature=signature,
                )
            )
        return tuple([*sets, *singles])

    def prepare_spec(self, cell: ComparisonCellSpec) -> ComparisonCellSpec:
        """Fill missing authoritative flatmap grid values without replacing any."""
        if cell.source_kind not in {
            SOURCE_FLATMAP_SOMAS,
            SOURCE_FLATMAP_ARBOR_HEATMAP,
        }:
            return cell
        path = self.parquet_path()
        if path is None:
            return cell
        info = read_flatmap_parquet_transform_info(path)
        grid = info.grid_spec(cell.flatmap_style)
        if grid is None:
            return cell
        x_bounds = cell.x_bounds or tuple(grid.x_bounds)
        y_bounds = cell.y_bounds or tuple(grid.y_bounds)
        x_bins = cell.x_bins
        if x_bins is None:
            # This is the one policy derivation point for a new recipe.  Once
            # saved, ComparisonCellSpec.x_bins is passed through verbatim.
            x_bins = resolve_flatmap_bin_counts(
                x_bounds=x_bounds,
                y_bounds=y_bounds,
                y_bins=cell.y_bins,
            ).x_bins
        return replace(
            cell,
            x_bounds=(float(x_bounds[0]), float(x_bounds[1])),
            y_bounds=(float(y_bounds[0]), float(y_bounds[1])),
            x_bins=int(x_bins),
        )

    def render_cell(
        self,
        cell: ComparisonCellSpec,
        *,
        reference_assignment_id: str | None,
    ) -> tuple[ComparisonCellSpec, ComparisonRenderData]:
        prepared = self.prepare_spec(cell)
        if prepared.source_kind == SOURCE_FLATMAP_SOMAS:
            result = self._render_flatmap_somas(prepared, reference_assignment_id)
        elif prepared.source_kind == SOURCE_FLATMAP_ARBOR_HEATMAP:
            result = self._render_flatmap_heatmap(prepared, reference_assignment_id)
        elif prepared.source_kind == SOURCE_CCF_SOMAS:
            result = self._render_ccf_somas(prepared, reference_assignment_id)
        elif prepared.source_kind == SOURCE_CCF_HEATMAP:
            result = self._render_ccf_heatmap(prepared, reference_assignment_id)
        else:  # pragma: no cover - ComparisonCellSpec normalizes source kinds
            raise ValueError(f"Unsupported comparison source: {prepared.source_kind}")
        return prepared, result

    def _require_assignment(self, cell: ComparisonCellSpec) -> ClusterAssignmentSet:
        assignment = self.assignment(cell.assignment_id)
        if assignment is None:
            raise ValueError(
                "The saved cluster assignment for this comparison cell is missing."
            )
        return assignment

    def _colors_and_matches(
        self,
        assignment: ClusterAssignmentSet,
        reference_assignment_id: str | None,
    ) -> tuple[dict[int, list[float]], tuple[ClusterLabelMatch, ...]]:
        reference = self._reference_assignment(reference_assignment_id)
        return assignment_display_colors(assignment, reference=reference)

    def _render_flatmap_somas(
        self,
        cell: ComparisonCellSpec,
        reference_assignment_id: str | None,
    ) -> ComparisonRenderData:
        path = self.parquet_path()
        if path is None:
            raise ValueError("Load a neuron Parquet before rendering a comparison.")
        assignment = self._require_assignment(cell)
        colors, matches = self._colors_and_matches(assignment, reference_assignment_id)
        file_ids, coordinates = query_flatmap_soma_coordinates(
            str(path),
            style=cell.flatmap_style,
            file_ids=list(assignment.assignments),
        )
        labels = np.asarray(
            [assignment.label_for(file_id) for file_id in file_ids], dtype=object
        )
        retained = np.asarray([label is not None for label in labels], dtype=bool)
        coordinates = np.asarray(coordinates, dtype=float)[retained]
        labels = labels[retained]
        point_colors = np.asarray([colors[int(label)] for label in labels], dtype=float)
        assigned_count, omitted_count = _assignment_counts(assignment)
        x_bounds = cell.x_bounds or _finite_bounds(coordinates[:, 0])
        y_bounds = cell.y_bounds or _finite_bounds(coordinates[:, 1])
        compatibility = (
            "flatmap",
            cell.flatmap_style,
            tuple(x_bounds),
            tuple(y_bounds),
        )
        return ComparisonRenderData(
            cell_id=cell.cell_id,
            title=cell.title,
            source_kind=cell.source_kind,
            assigned_count=assigned_count,
            omitted_count=omitted_count,
            subtitle=(
                f"{assignment.name} · {len(coordinates):,} somas · "
                f"{cell.flatmap_style.replace('_', ' ')}"
            ),
            x_bounds=tuple(x_bounds),
            y_bounds=tuple(y_bounds),
            compatibility_key=compatibility,
            coordinate_provenance={
                "space": "flatmap",
                "style": cell.flatmap_style,
                "x_bounds": list(x_bounds),
                "y_bounds": list(y_bounds),
                "y_bins": int(cell.y_bins),
                "x_bins": int(cell.x_bins) if cell.x_bins is not None else None,
            },
            points=coordinates[:, :2],
            point_colors=point_colors,
            colors=colors,
            matches=matches,
            provenance={
                **_assignment_provenance(assignment, colors),
                "flatmap_style": cell.flatmap_style,
                "y_bins": cell.y_bins,
                "x_bins": cell.x_bins,
                "rendered_somas": int(len(coordinates)),
            },
        )

    def _render_flatmap_heatmap(
        self,
        cell: ComparisonCellSpec,
        reference_assignment_id: str | None,
    ) -> ComparisonRenderData:
        path = self.parquet_path()
        if path is None:
            raise ValueError("Load a neuron Parquet before rendering a comparison.")
        assignment = self._require_assignment(cell)
        if cell.x_bounds is None or cell.y_bounds is None or cell.x_bins is None:
            raise ValueError(
                "Flatmap comparison metadata is incomplete; regenerate the cell."
            )
        colors, matches = self._colors_and_matches(assignment, reference_assignment_id)
        cache_key = (
            str(path.resolve()),
            path.stat().st_size,
            path.stat().st_mtime_ns,
            assignment.assignment_id,
            cell.flatmap_style,
            cell.y_bins,
            cell.x_bins,
            cell.x_bounds,
            cell.y_bounds,
        )
        cached = self._flatmap_cache.get(cache_key)
        if cached is not None:
            provenance = dict(cached.provenance)
            provenance.update(_assignment_provenance(assignment, colors))
            return replace(
                cached,
                cell_id=cell.cell_id,
                title=cell.title,
                subtitle=(
                    f"{assignment.name} · "
                    f"{int(cached.provenance.get('rendered_nodes', 0)):,} nodes · "
                    f"{cell.flatmap_style.replace('_', ' ')}"
                ),
                colors=colors,
                matches=matches,
                provenance=provenance,
            )

        info = read_flatmap_parquet_transform_info(path)
        grid = info.grid_spec(cell.flatmap_style)
        if grid is None:
            raise ValueError(
                f"The loaded Parquet has no {cell.flatmap_style} flatmap grid."
            )
        connection = duckdb.connect()
        try:
            heatmap = build_flatmap_heatmap_volume_result(
                connection,
                str(path),
                style_key=cell.flatmap_style,
                color_mode=FLATMAP_HEATMAP_COLOR_CLUSTER,
                x_bounds=cell.x_bounds,
                y_bounds=cell.y_bounds,
                depth_range_um=tuple(grid.depth_bounds_um),
                y_bins=cell.y_bins,
                # Stored counts are authoritative; do not rederive them.
                x_bins=cell.x_bins,
                depth_bin_um=DEFAULT_FLATMAP_DEPTH_BIN_UM,
                include_depth_minus_one=True,
                file_ids=list(assignment.assignments),
                cluster_map=assignment.assignments,
                collapse_depth=True,
            )
        finally:
            connection.close()
        volumes = {
            int(group.group_key): np.asarray(group.volume, dtype=np.float32)
            for group in heatmap.grouped_volumes
            if group.group_key is not None
        }
        observed = max(
            (float(np.max(volume)) for volume in volumes.values() if volume.size),
            default=0.0,
        )
        assigned_count, omitted_count = _assignment_counts(assignment)
        compatibility = (
            "flatmap",
            cell.flatmap_style,
            cell.x_bounds,
            cell.y_bounds,
        )
        result = ComparisonRenderData(
            cell_id=cell.cell_id,
            title=cell.title,
            source_kind=cell.source_kind,
            assigned_count=assigned_count,
            omitted_count=omitted_count,
            subtitle=(
                f"{assignment.name} · {heatmap.render_summary.rendered_nodes:,} "
                f"nodes · {cell.flatmap_style.replace('_', ' ')}"
            ),
            x_bounds=cell.x_bounds,
            y_bounds=cell.y_bounds,
            compatibility_key=compatibility,
            coordinate_provenance={
                "space": "flatmap",
                "style": cell.flatmap_style,
                "x_bounds": list(cell.x_bounds),
                "y_bounds": list(cell.y_bounds),
                "y_bins": int(cell.y_bins),
                "x_bins": int(cell.x_bins),
            },
            intensity_key=(
                SOURCE_FLATMAP_ARBOR_HEATMAP,
                *compatibility,
                cell.y_bins,
                cell.x_bins,
            ),
            heatmaps=volumes,
            colors=colors,
            matches=matches,
            observed_intensity_max=observed,
            provenance={
                **_assignment_provenance(assignment, colors),
                "flatmap_style": cell.flatmap_style,
                "y_bins": int(cell.y_bins),
                "x_bins": int(cell.x_bins),
                "rendered_nodes": int(heatmap.render_summary.rendered_nodes),
                "cluster_count": len(volumes),
            },
        )
        self._flatmap_cache[cache_key] = result
        return result

    def _atlas_resolution(self) -> tuple[float, float, float]:
        atlas = self._atlas_provider()
        resolution = getattr(atlas, "resolution", (1.0, 1.0, 1.0))
        return _three_floats(resolution, default=1.0)

    def _atlas_name(self) -> str | None:
        atlas = self._atlas_provider()
        name = getattr(atlas, "atlas_name", None)
        return None if name is None else str(name)

    def _atlas_provenance_key(self) -> tuple[object, ...]:
        atlas = self._atlas_provider()
        version = getattr(atlas, "version", None)
        return (
            self._atlas_name(),
            None if version is None else str(version),
            self._atlas_resolution(),
        )

    def _atlas_plane_bounds(
        self,
        plane: str,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        atlas = self._atlas_provider()
        annotation = getattr(atlas, "annotation", None)
        raw_shape = getattr(annotation, "shape", None)
        try:
            shape = tuple(int(value) for value in raw_shape)
        except (TypeError, ValueError):
            return None
        if len(shape) != 3 or any(value <= 0 for value in shape):
            return None
        resolution = self._atlas_resolution()
        _hidden, vertical, horizontal = ccf_plane_axes(plane)
        return (
            (
                -resolution[horizontal] / 2.0,
                (shape[horizontal] - 0.5) * resolution[horizontal],
            ),
            (
                -resolution[vertical] / 2.0,
                (shape[vertical] - 0.5) * resolution[vertical],
            ),
        )

    def _render_ccf_somas(
        self,
        cell: ComparisonCellSpec,
        reference_assignment_id: str | None,
    ) -> ComparisonRenderData:
        path = self.parquet_path()
        if path is None:
            raise ValueError("Load a neuron Parquet before rendering a comparison.")
        assignment = self._require_assignment(cell)
        colors, matches = self._colors_and_matches(assignment, reference_assignment_id)
        resolution = self._atlas_resolution()
        file_ids, coordinates, _node_count = query_ccf_soma_coordinates(
            str(path),
            resolution=resolution[0],
            file_ids=list(assignment.assignments),
        )
        projection = project_ccf_points(
            coordinates,
            plane=cell.ccf_plane,
            reduction=cell.reduction,
            slice_position_um=cell.slice_position_um,
            slab_thickness_um=cell.slab_thickness_um,
            default_slab_thickness_um=resolution[
                {"coronal": 0, "horizontal": 1, "sagittal": 2}[cell.ccf_plane]
            ],
        )
        retained_ids = np.asarray(file_ids, dtype=object)[projection.retained]
        labels = [assignment.label_for(file_id) for file_id in retained_ids]
        point_colors = np.asarray(
            [colors[int(label)] for label in labels if label is not None], dtype=float
        )
        points = projection.points
        # query_ccf_soma_coordinates was already scoped to assigned file IDs, so
        # all finite retained rows have labels.  Keep the defensive alignment in
        # case a malformed imported assignment changes that invariant.
        if len(point_colors) != len(points):
            labelled = np.asarray([label is not None for label in labels], dtype=bool)
            points = points[labelled]
        assigned_count, omitted_count = _assignment_counts(assignment)
        atlas_bounds = self._atlas_plane_bounds(cell.ccf_plane)
        if atlas_bounds is not None:
            x_bounds, y_bounds = atlas_bounds
        else:
            x_bounds = _finite_bounds(points[:, 0]) if len(points) else (0.0, 1.0)
            y_bounds = _finite_bounds(points[:, 1]) if len(points) else (0.0, 1.0)
        atlas_name = self._atlas_name()
        compatibility = (
            "ccf",
            self._atlas_provenance_key(),
            cell.ccf_plane,
            cell.reduction,
        )
        mode_text = (
            "full projection"
            if cell.reduction == REDUCTION_PROJECTION
            else f"{cell.slab_thickness_um or resolution[projection.hidden_axis]:g} µm slab"
        )
        return ComparisonRenderData(
            cell_id=cell.cell_id,
            title=cell.title,
            source_kind=cell.source_kind,
            assigned_count=assigned_count,
            omitted_count=omitted_count,
            subtitle=(
                f"{assignment.name} · {len(points):,} somas · "
                f"{cell.ccf_plane} {mode_text}"
            ),
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            compatibility_key=compatibility,
            coordinate_provenance={
                "space": "CCFv3",
                "atlas_name": atlas_name,
                "atlas_provenance_key": list(self._atlas_provenance_key()),
            },
            points=points,
            point_colors=point_colors,
            colors=colors,
            matches=matches,
            provenance={
                **_assignment_provenance(assignment, colors),
                "atlas_name": atlas_name,
                "ccf_plane": cell.ccf_plane,
                "reduction": cell.reduction,
                "slice_position_um": cell.slice_position_um,
                "slab_thickness_um": cell.slab_thickness_um,
                "rendered_somas": int(len(points)),
            },
        )

    def _selected_heatmap_sources(
        self, source_ids: Sequence[str]
    ) -> tuple[ComparisonHeatmapSource, ...]:
        by_id = {source.source_id: source for source in self.heatmap_sources()}
        missing = [source_id for source_id in source_ids if source_id not in by_id]
        if missing:
            raise ValueError(
                "Comparison heatmap source is missing: " + ", ".join(missing)
            )
        return tuple(by_id[source_id] for source_id in source_ids)

    @staticmethod
    def _validate_heatmap_sources(
        sources: Sequence[ComparisonHeatmapSource],
    ) -> None:
        if not sources:
            raise ValueError("Choose at least one existing Analysis heatmap.")
        first = sources[0]
        first_signature = heatmap_filter_signature(first.metadata)
        cluster_labels = [
            source.cluster_label
            for source in sources
            if source.cluster_label is not None
        ]
        if len(cluster_labels) != len(set(cluster_labels)):
            raise ValueError("Selected heatmaps repeat a cluster label.")
        if cluster_labels and len(cluster_labels) != len(sources):
            raise ValueError(
                "Selected heatmaps mix cluster-specific and population layers."
            )
        for source in sources[1:]:
            if source.data.shape != first.data.shape:
                raise ValueError("Selected heatmaps do not share the same shape.")
            if not np.allclose(source.scale, first.scale, rtol=0.0, atol=1e-9):
                raise ValueError("Selected heatmaps do not share the same scale.")
            if not np.allclose(source.translate, first.translate, rtol=0.0, atol=1e-9):
                raise ValueError("Selected heatmaps do not share the same origin.")
            if heatmap_filter_signature(source.metadata) != first_signature:
                raise ValueError(
                    "Selected heatmaps do not share assignment/filter provenance."
                )

    def _render_ccf_heatmap(
        self,
        cell: ComparisonCellSpec,
        reference_assignment_id: str | None,
    ) -> ComparisonRenderData:
        sources = self._selected_heatmap_sources(cell.comparison_source_ids)
        self._validate_heatmap_sources(sources)
        source_assignment_id = sources[0].assignment_id
        if (
            cell.assignment_id is not None
            and source_assignment_id is not None
            and cell.assignment_id != source_assignment_id
        ):
            raise ValueError(
                "The comparison cell and heatmap record different assignments."
            )
        assignment_id = cell.assignment_id or sources[0].assignment_id
        assignment = self.assignment(assignment_id)
        if assignment_id is not None and assignment is None:
            raise ValueError(
                "The saved cluster assignment recorded by this heatmap is missing."
            )
        if assignment is not None:
            colors, matches = self._colors_and_matches(
                assignment, reference_assignment_id
            )
            assigned_count, omitted_count = _assignment_counts(assignment)
            assignment_name = assignment.name
            assignment_labels = set(assignment.assignments.values())
            source_labels = {
                source.cluster_label
                for source in sources
                if source.cluster_label is not None
            }
            if not source_labels.issubset(assignment_labels):
                raise ValueError(
                    "A selected heatmap cluster label is absent from its assignment."
                )
        else:
            colors = {}
            matches = ()
            source_file_ids: set[str] = set()
            for source in sources:
                raw_ids = source.metadata.get("source_file_ids")
                if isinstance(raw_ids, (list, tuple)):
                    source_file_ids.update(str(value) for value in raw_ids)
            assigned_count = len(source_file_ids)
            omitted_count = 0
            assignment_name = "Existing heatmap"

        atlas_resolution = sources[0].atlas_resolution_um or self._atlas_resolution()
        base_spacing = tuple(
            atlas_resolution[index] * sources[0].scale[index] for index in range(3)
        )
        base_origin = tuple(
            sources[0].translate[index] * atlas_resolution[index] for index in range(3)
        )
        volumes: dict[int, np.ndarray] = {}
        reduction_result = None
        next_unlabelled = 0
        for source in sources:
            cluster_label = source.cluster_label
            if cluster_label is None:
                while next_unlabelled in volumes:
                    next_unlabelled += 1
                cluster_label = next_unlabelled
                colors.setdefault(cluster_label, [1.0, 0.4, 0.1, 1.0])
            reduction_result = reduce_ccf_volume(
                source.data,
                plane=cell.ccf_plane,
                reduction=cell.reduction,
                spacing_um=base_spacing,
                origin_um=base_origin,
                slice_position_um=cell.slice_position_um,
                slab_thickness_um=cell.slab_thickness_um,
            )
            volumes[int(cluster_label)] = reduction_result.data
        assert reduction_result is not None
        all_display_colors = colors
        rendered_labels = set(volumes)
        colors = {
            label: color
            for label, color in all_display_colors.items()
            if label in rendered_labels
        }
        matches = tuple(
            match for match in matches if match.candidate_label in rendered_labels
        )
        observed = max(
            (float(np.max(volume)) for volume in volumes.values() if volume.size),
            default=0.0,
        )
        atlas_name = sources[0].metadata.get("atlas_name") or self._atlas_name()
        compatibility = (
            "ccf",
            sources[0].atlas_provenance_key(atlas_resolution),
            cell.ccf_plane,
            cell.reduction,
        )
        # Assignment identity is intentionally omitted: equivalent heatmaps from
        # different runs must share an intensity range.
        raw_filter_signature = heatmap_filter_signature(sources[0].metadata)
        comparable_filter = (
            raw_filter_signature[0],
            *raw_filter_signature[2:],
        )
        return ComparisonRenderData(
            cell_id=cell.cell_id,
            title=cell.title,
            source_kind=cell.source_kind,
            assigned_count=assigned_count,
            omitted_count=omitted_count,
            subtitle=(
                f"{assignment_name} · {len(sources)} heatmap layer(s) · "
                f"{cell.ccf_plane} {cell.reduction}"
            ),
            x_bounds=reduction_result.x_bounds_um,
            y_bounds=reduction_result.y_bounds_um,
            compatibility_key=compatibility,
            coordinate_provenance={
                "space": "CCFv3",
                "atlas_name": atlas_name,
                "atlas_provenance_key": list(
                    sources[0].atlas_provenance_key(atlas_resolution)
                ),
            },
            intensity_key=(
                SOURCE_CCF_HEATMAP,
                comparable_filter,
                *compatibility,
                sources[0].data.shape,
                base_spacing,
                base_origin,
                reduction_result.included_index_range,
            ),
            heatmaps=volumes,
            colors=colors,
            matches=matches,
            observed_intensity_max=observed,
            provenance={
                "assignment_id": assignment_id,
                "assignment_name": assignment_name,
                **(
                    _assignment_provenance(assignment, all_display_colors)
                    if assignment is not None
                    else {}
                ),
                "comparison_source_ids": list(cell.comparison_source_ids),
                "atlas_name": atlas_name,
                "ccf_plane": cell.ccf_plane,
                "reduction": cell.reduction,
                "slice_position_um": cell.slice_position_um,
                "slab_thickness_um": cell.slab_thickness_um,
                "included_index_range": list(reduction_result.included_index_range),
                "heatmap_filter_signature": [
                    list(value) if isinstance(value, tuple) else value
                    for value in raw_filter_signature
                ],
            },
        )
