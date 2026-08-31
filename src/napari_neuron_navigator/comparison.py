"""State and rendering primitives for the interactive comparison board.

This module deliberately has no Qt or napari imports.  The comparison window
uses these functions, while tests can verify the scientific parts of a board
without constructing an OpenGL canvas.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
from scipy.optimize import linear_sum_assignment

from .cluster_assignments import ClusterAssignmentSet
from .neuron_palette import neuron_palette


COMPARISON_BOARD_STATE_VERSION = 1
COMPARISON_EXPORT_VERSION = 2
MAX_COMPARISON_ROWS = 4
MAX_COMPARISON_COLUMNS = 4
MAX_COMPARISON_CELLS = MAX_COMPARISON_ROWS * MAX_COMPARISON_COLUMNS

SOURCE_FLATMAP_SOMAS = "flatmap_somas"
SOURCE_FLATMAP_ARBOR_HEATMAP = "flatmap_arbor_heatmap"
SOURCE_CCF_SOMAS = "ccf_somas"
SOURCE_CCF_HEATMAP = "ccf_heatmap"
COMPARISON_SOURCE_KINDS = (
    SOURCE_FLATMAP_SOMAS,
    SOURCE_FLATMAP_ARBOR_HEATMAP,
    SOURCE_CCF_SOMAS,
    SOURCE_CCF_HEATMAP,
)

CCF_PLANE_CORONAL = "coronal"
CCF_PLANE_HORIZONTAL = "horizontal"
CCF_PLANE_SAGITTAL = "sagittal"
CCF_PLANES = (
    CCF_PLANE_CORONAL,
    CCF_PLANE_HORIZONTAL,
    CCF_PLANE_SAGITTAL,
)

REDUCTION_PROJECTION = "projection"
REDUCTION_SLICE = "slice"
CCF_REDUCTIONS = (REDUCTION_PROJECTION, REDUCTION_SLICE)

# CCF coordinates and atlas arrays use the repository's established order:
# (rostral-caudal, dorsal-ventral, left-right).  Each tuple is
# (hidden axis, vertical display axis, horizontal display axis).
_CCF_PLANE_AXES = {
    CCF_PLANE_CORONAL: (0, 1, 2),
    CCF_PLANE_HORIZONTAL: (1, 0, 2),
    CCF_PLANE_SAGITTAL: (2, 1, 0),
}


def _bounded_int(value: object, *, lower: int, upper: int, default: int) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        normalized = int(default)
    return max(lower, min(normalized, upper))


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    return normalized if np.isfinite(normalized) else None


def _pair(value: object) -> tuple[float, float] | None:
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        values = tuple(float(item) for item in value)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return None
    if len(values) != 2 or not np.all(np.isfinite(values)):
        return None
    return values


def _rect(value: object) -> tuple[float, float, float, float] | None:
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        values = tuple(float(item) for item in value)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return None
    if len(values) != 4 or not np.all(np.isfinite(values)):
        return None
    return values


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (str(value),) if str(value) else ()
    try:
        return tuple(dict.fromkeys(str(item) for item in value if str(item)))
    except TypeError:
        return ()


@dataclass
class ComparisonCellSpec:
    """Serializable recipe for one cell in a comparison board."""

    cell_id: str = field(default_factory=lambda: uuid4().hex)
    title: str = "Comparison"
    source_kind: str = SOURCE_FLATMAP_SOMAS
    assignment_id: str | None = None
    comparison_source_ids: tuple[str, ...] = ()
    flatmap_style: str = "both_shaped"
    y_bins: int = 256
    x_bins: int | None = None
    x_bounds: tuple[float, float] | None = None
    y_bounds: tuple[float, float] | None = None
    coordinate_provenance: dict[str, object] = field(default_factory=dict)
    ccf_plane: str = CCF_PLANE_CORONAL
    reduction: str = REDUCTION_PROJECTION
    slice_position_um: float | None = None
    slab_thickness_um: float | None = None
    point_size: float = 6.0
    opacity: float = 0.85
    intensity_max_override: float | None = None
    camera_linked: bool = True
    camera_rect: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        self.cell_id = str(self.cell_id or uuid4().hex)
        self.title = str(self.title or "Comparison")
        if self.source_kind not in COMPARISON_SOURCE_KINDS:
            self.source_kind = SOURCE_FLATMAP_SOMAS
        self.assignment_id = (
            str(self.assignment_id) if self.assignment_id not in (None, "") else None
        )
        self.comparison_source_ids = _string_tuple(self.comparison_source_ids)
        self.flatmap_style = str(self.flatmap_style or "both_shaped")
        self.y_bins = _bounded_int(
            self.y_bins,
            lower=1,
            upper=16_384,
            default=256,
        )
        if self.x_bins is not None:
            self.x_bins = _bounded_int(
                self.x_bins,
                lower=1,
                upper=32_768,
                default=self.y_bins,
            )
        self.x_bounds = _pair(self.x_bounds)
        self.y_bounds = _pair(self.y_bounds)
        self.coordinate_provenance = (
            dict(self.coordinate_provenance)
            if isinstance(self.coordinate_provenance, Mapping)
            else {}
        )
        if self.ccf_plane not in CCF_PLANES:
            self.ccf_plane = CCF_PLANE_CORONAL
        if self.reduction not in CCF_REDUCTIONS:
            self.reduction = REDUCTION_PROJECTION
        self.slice_position_um = _optional_float(self.slice_position_um)
        self.slab_thickness_um = _optional_float(self.slab_thickness_um)
        if self.slab_thickness_um is not None and self.slab_thickness_um <= 0:
            self.slab_thickness_um = None
        self.point_size = max(1.0, float(self.point_size))
        self.opacity = float(np.clip(float(self.opacity), 0.0, 1.0))
        self.intensity_max_override = _optional_float(self.intensity_max_override)
        if self.intensity_max_override is not None and self.intensity_max_override <= 0:
            self.intensity_max_override = None
        self.camera_linked = bool(self.camera_linked)
        self.camera_rect = _rect(self.camera_rect)

    @property
    def is_heatmap(self) -> bool:
        return self.source_kind in {
            SOURCE_FLATMAP_ARBOR_HEATMAP,
            SOURCE_CCF_HEATMAP,
        }

    @property
    def is_assignment_backed(self) -> bool:
        return self.source_kind != SOURCE_CCF_HEATMAP or self.assignment_id is not None

    def to_state(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "title": self.title,
            "source_kind": self.source_kind,
            "assignment_id": self.assignment_id,
            "comparison_source_ids": list(self.comparison_source_ids),
            "flatmap_style": self.flatmap_style,
            "y_bins": int(self.y_bins),
            # A stored x count is authoritative and must never be rederived.
            "x_bins": self.x_bins,
            "x_bounds": list(self.x_bounds) if self.x_bounds is not None else None,
            "y_bounds": list(self.y_bounds) if self.y_bounds is not None else None,
            "coordinate_provenance": dict(self.coordinate_provenance),
            "ccf_plane": self.ccf_plane,
            "reduction": self.reduction,
            "slice_position_um": self.slice_position_um,
            "slab_thickness_um": self.slab_thickness_um,
            "point_size": float(self.point_size),
            "opacity": float(self.opacity),
            "intensity_max_override": self.intensity_max_override,
            "camera_linked": bool(self.camera_linked),
            "camera_rect": (
                list(self.camera_rect) if self.camera_rect is not None else None
            ),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> "ComparisonCellSpec":
        return cls(
            cell_id=str(state.get("cell_id") or uuid4().hex),
            title=str(state.get("title") or "Comparison"),
            source_kind=str(state.get("source_kind") or SOURCE_FLATMAP_SOMAS),
            assignment_id=(
                str(state["assignment_id"])
                if state.get("assignment_id") not in (None, "")
                else None
            ),
            comparison_source_ids=_string_tuple(state.get("comparison_source_ids", ())),
            flatmap_style=str(state.get("flatmap_style") or "both_shaped"),
            y_bins=_bounded_int(
                state.get("y_bins"), lower=1, upper=16_384, default=256
            ),
            x_bins=(
                None
                if state.get("x_bins") in (None, "")
                else _bounded_int(
                    state.get("x_bins"), lower=1, upper=32_768, default=256
                )
            ),
            x_bounds=_pair(state.get("x_bounds")),
            y_bounds=_pair(state.get("y_bounds")),
            coordinate_provenance=(
                dict(state["coordinate_provenance"])
                if isinstance(state.get("coordinate_provenance"), Mapping)
                else {}
            ),
            ccf_plane=str(state.get("ccf_plane") or CCF_PLANE_CORONAL),
            reduction=str(state.get("reduction") or REDUCTION_PROJECTION),
            slice_position_um=_optional_float(state.get("slice_position_um")),
            slab_thickness_um=_optional_float(state.get("slab_thickness_um")),
            point_size=float(
                6.0 if state.get("point_size") is None else state["point_size"]
            ),
            opacity=float(0.85 if state.get("opacity") is None else state["opacity"]),
            intensity_max_override=_optional_float(state.get("intensity_max_override")),
            camera_linked=bool(state.get("camera_linked", True)),
            camera_rect=_rect(state.get("camera_rect")),
        )


@dataclass
class ComparisonBoardState:
    """Serializable state for the project's single comparison board."""

    rows: int = 2
    columns: int = 2
    cells: list[ComparisonCellSpec] = field(default_factory=list)
    reference_assignment_id: str | None = None
    shared_intensity: bool = True
    version: int = COMPARISON_BOARD_STATE_VERSION

    def __post_init__(self) -> None:
        self.rows = _bounded_int(
            self.rows, lower=1, upper=MAX_COMPARISON_ROWS, default=2
        )
        self.columns = _bounded_int(
            self.columns, lower=1, upper=MAX_COMPARISON_COLUMNS, default=2
        )
        normalized: list[ComparisonCellSpec] = []
        seen: set[str] = set()
        for raw_cell in list(self.cells)[:MAX_COMPARISON_CELLS]:
            cell = (
                raw_cell
                if isinstance(raw_cell, ComparisonCellSpec)
                else ComparisonCellSpec.from_state(raw_cell)  # type: ignore[arg-type]
            )
            if cell.cell_id in seen:
                cell.cell_id = uuid4().hex
            seen.add(cell.cell_id)
            normalized.append(cell)
        self.cells = normalized[: self.capacity]
        self.reference_assignment_id = (
            str(self.reference_assignment_id)
            if self.reference_assignment_id not in (None, "")
            else None
        )
        self.shared_intensity = bool(self.shared_intensity)
        self.version = COMPARISON_BOARD_STATE_VERSION

    @property
    def capacity(self) -> int:
        return int(self.rows * self.columns)

    def resize(self, rows: int, columns: int) -> None:
        self.rows = _bounded_int(
            rows, lower=1, upper=MAX_COMPARISON_ROWS, default=self.rows
        )
        self.columns = _bounded_int(
            columns,
            lower=1,
            upper=MAX_COMPARISON_COLUMNS,
            default=self.columns,
        )
        del self.cells[self.capacity :]

    def add_cell(self, cell: ComparisonCellSpec | None = None) -> ComparisonCellSpec:
        """Append one cell without exceeding the configured grid."""
        if len(self.cells) >= self.capacity:
            raise ValueError("The comparison grid is full.")
        resolved = cell or ComparisonCellSpec()
        if any(existing.cell_id == resolved.cell_id for existing in self.cells):
            resolved = ComparisonCellSpec.from_state(resolved.to_state())
            resolved.cell_id = uuid4().hex
        self.cells.append(resolved)
        return resolved

    def duplicate_cell(self, cell_id: str) -> ComparisonCellSpec:
        """Insert a recipe copy after the identified cell with a fresh ID."""
        if len(self.cells) >= self.capacity:
            raise ValueError("The comparison grid is full.")
        index = self._cell_index(cell_id)
        source = self.cells[index]
        duplicate = ComparisonCellSpec.from_state(source.to_state())
        duplicate.cell_id = uuid4().hex
        duplicate.title = f"{source.title} copy"
        self.cells.insert(index + 1, duplicate)
        return duplicate

    def remove_cell(self, cell_id: str) -> ComparisonCellSpec:
        """Remove and return one identified cell."""
        return self.cells.pop(self._cell_index(cell_id))

    def move_cell(self, cell_id: str, offset: int) -> int:
        """Move one cell by *offset* and return its resulting index."""
        index = self._cell_index(cell_id)
        target = max(0, min(index + int(offset), len(self.cells) - 1))
        if target != index:
            self.cells[index], self.cells[target] = (
                self.cells[target],
                self.cells[index],
            )
        return target

    def _cell_index(self, cell_id: str) -> int:
        for index, cell in enumerate(self.cells):
            if cell.cell_id == str(cell_id):
                return index
        raise KeyError(str(cell_id))

    def to_state(self) -> dict[str, object]:
        return {
            "version": COMPARISON_BOARD_STATE_VERSION,
            "rows": int(self.rows),
            "columns": int(self.columns),
            "reference_assignment_id": self.reference_assignment_id,
            "shared_intensity": bool(self.shared_intensity),
            "cells": [cell.to_state() for cell in self.cells],
        }

    @classmethod
    def from_state(cls, state: Mapping[str, object] | None) -> "ComparisonBoardState":
        if not isinstance(state, Mapping):
            return cls()
        raw_cells = state.get("cells", ())
        if isinstance(raw_cells, (str, bytes, Mapping)):
            raw_cells = ()
        cells = [
            ComparisonCellSpec.from_state(item)
            for item in raw_cells  # type: ignore[union-attr]
            if isinstance(item, Mapping)
        ]
        return cls(
            rows=_bounded_int(
                state.get("rows"), lower=1, upper=MAX_COMPARISON_ROWS, default=2
            ),
            columns=_bounded_int(
                state.get("columns"),
                lower=1,
                upper=MAX_COMPARISON_COLUMNS,
                default=2,
            ),
            cells=cells,
            reference_assignment_id=(
                str(state["reference_assignment_id"])
                if state.get("reference_assignment_id") not in (None, "")
                else None
            ),
            shared_intensity=bool(state.get("shared_intensity", True)),
        )


@dataclass(frozen=True)
class ClusterLabelMatch:
    """One candidate cluster's display relationship to a reference cluster."""

    candidate_label: int
    reference_label: int | None
    shared_file_ids: int
    candidate_shared_cohort: int

    @property
    def overlap_fraction(self) -> float:
        if self.candidate_shared_cohort <= 0:
            return 0.0
        return self.shared_file_ids / self.candidate_shared_cohort

    def to_state(self) -> dict[str, object]:
        return {
            "candidate_label": int(self.candidate_label),
            "reference_label": self.reference_label,
            "shared_file_ids": int(self.shared_file_ids),
            "candidate_shared_cohort": int(self.candidate_shared_cohort),
            "overlap_fraction": float(self.overlap_fraction),
        }


@dataclass(frozen=True)
class ClusterMembershipCohortCounts:
    """Coverage counts for two assignment sets aligned by ``file_id``."""

    reference_cohort: int
    candidate_cohort: int
    shared_cohort: int
    reference_cohort_only: int
    candidate_cohort_only: int
    assigned_in_both: int
    reference_assigned_only: int
    candidate_assigned_only: int
    unassigned_in_both: int

    def to_state(self) -> dict[str, int]:
        return {
            "reference_cohort": int(self.reference_cohort),
            "candidate_cohort": int(self.candidate_cohort),
            "shared_cohort": int(self.shared_cohort),
            "reference_cohort_only": int(self.reference_cohort_only),
            "candidate_cohort_only": int(self.candidate_cohort_only),
            "assigned_in_both": int(self.assigned_in_both),
            "reference_assigned_only": int(self.reference_assigned_only),
            "candidate_assigned_only": int(self.candidate_assigned_only),
            "unassigned_in_both": int(self.unassigned_in_both),
        }


@dataclass(frozen=True)
class ClusterPairSimilarity:
    """One optimal cluster pairing, including explicit unmatched clusters."""

    reference_label: int | None
    candidate_label: int | None
    intersection: int
    reference_size: int
    candidate_size: int
    union: int

    @property
    def jaccard(self) -> float:
        if self.union <= 0:
            return 0.0
        return float(self.intersection / self.union)

    def to_state(self) -> dict[str, object]:
        return {
            "reference_cluster_id": self.reference_label,
            "candidate_cluster_id": self.candidate_label,
            "intersection": int(self.intersection),
            "reference_size": int(self.reference_size),
            "candidate_size": int(self.candidate_size),
            "union": int(self.union),
            "jaccard": float(self.jaccard),
        }


@dataclass(frozen=True)
class ClusterMembershipComparison:
    """Basic, label-invariant membership statistics for two saved assignments."""

    reference_assignment_id: str
    reference_assignment_name: str
    candidate_assignment_id: str
    candidate_assignment_name: str
    status: str
    status_message: str
    cohort_counts: ClusterMembershipCohortCounts
    adjusted_rand_index: float | None
    normalized_mutual_information: float | None
    matched_agreement: float | None
    matched_file_ids: int
    reference_cluster_ids: tuple[int, ...]
    candidate_cluster_ids: tuple[int, ...]
    overlap_counts: tuple[tuple[int, ...], ...]
    cluster_matches: tuple[ClusterPairSimilarity, ...]

    def to_state(self) -> dict[str, object]:
        return {
            "status": self.status,
            "status_message": self.status_message,
            "reference_assignment": {
                "assignment_id": self.reference_assignment_id,
                "name": self.reference_assignment_name,
            },
            "candidate_assignment": {
                "assignment_id": self.candidate_assignment_id,
                "name": self.candidate_assignment_name,
            },
            "cohort_counts": self.cohort_counts.to_state(),
            "metrics": {
                "adjusted_rand_index": self.adjusted_rand_index,
                "normalized_mutual_information": (self.normalized_mutual_information),
                "matched_agreement": self.matched_agreement,
                "matched_file_ids": int(self.matched_file_ids),
                "jointly_assigned_file_ids": int(self.cohort_counts.assigned_in_both),
            },
            "overlap_matrix": {
                "reference_cluster_ids": list(self.reference_cluster_ids),
                "candidate_cluster_ids": list(self.candidate_cluster_ids),
                "counts": [list(row) for row in self.overlap_counts],
            },
            "cluster_matches": [match.to_state() for match in self.cluster_matches],
        }


def _maximum_overlap_pairs(
    contingency: np.ndarray,
    reference_labels: Sequence[int],
    candidate_labels: Sequence[int],
) -> dict[int, tuple[int, int]]:
    """Return positive-overlap candidate-to-reference Hungarian pairings."""
    if contingency.size == 0:
        return {}
    rows, columns = linear_sum_assignment(-contingency)
    paired: dict[int, tuple[int, int]] = {}
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        overlap = int(contingency[row, column])
        if overlap > 0:
            paired[int(candidate_labels[column])] = (
                int(reference_labels[row]),
                overlap,
            )
    return paired


def match_cluster_labels(
    reference_assignments: Mapping[object, object],
    candidate_assignments: Mapping[object, object],
) -> tuple[ClusterLabelMatch, ...]:
    """Optimally pair candidate labels with reference labels by ``file_id``.

    Only the shared ``file_id`` universe contributes to the contingency table.
    Labels are paired one-to-one with the Hungarian algorithm; labels left over
    after a split or a different requested ``k`` remain explicitly unmatched.
    """
    reference = {
        str(file_id): int(label)
        for file_id, label in reference_assignments.items()
        if label not in (None, "")
    }
    candidate = {
        str(file_id): int(label)
        for file_id, label in candidate_assignments.items()
        if label not in (None, "")
    }
    candidate_labels = sorted(set(candidate.values()))
    if not candidate_labels:
        return ()

    shared_ids = sorted(set(reference).intersection(candidate))
    reference_labels = sorted({reference[file_id] for file_id in shared_ids})
    candidate_shared_counts = {
        label: sum(candidate[file_id] == label for file_id in shared_ids)
        for label in candidate_labels
    }
    if not shared_ids or not reference_labels:
        return tuple(
            ClusterLabelMatch(label, None, 0, candidate_shared_counts[label])
            for label in candidate_labels
        )

    reference_index = {label: index for index, label in enumerate(reference_labels)}
    candidate_index = {label: index for index, label in enumerate(candidate_labels)}
    contingency = np.zeros(
        (len(reference_labels), len(candidate_labels)), dtype=np.int64
    )
    for file_id in shared_ids:
        contingency[
            reference_index[reference[file_id]],
            candidate_index[candidate[file_id]],
        ] += 1

    paired = _maximum_overlap_pairs(
        contingency,
        reference_labels,
        candidate_labels,
    )

    return tuple(
        ClusterLabelMatch(
            candidate_label=label,
            reference_label=(paired[label][0] if label in paired else None),
            shared_file_ids=(paired[label][1] if label in paired else 0),
            candidate_shared_cohort=candidate_shared_counts[label],
        )
        for label in candidate_labels
    )


def _assignment_cohort(assignment: ClusterAssignmentSet) -> set[str]:
    """Return the complete declared ``file_id`` universe for an assignment."""
    cohort = {str(file_id) for file_id in assignment.input_file_ids}
    cohort.update(str(file_id) for file_id in assignment.unassigned_neuron_ids)
    cohort.update(str(file_id) for file_id in assignment.assignments)
    return cohort


def compare_cluster_memberships(
    reference: ClusterAssignmentSet,
    candidate: ClusterAssignmentSet,
) -> ClusterMembershipComparison:
    """Compare two saved cluster memberships over their shared ``file_id`` cohort."""
    reference_assignments = {
        str(file_id): int(label) for file_id, label in reference.assignments.items()
    }
    candidate_assignments = {
        str(file_id): int(label) for file_id, label in candidate.assignments.items()
    }
    reference_cohort = _assignment_cohort(reference)
    candidate_cohort = _assignment_cohort(candidate)
    shared_cohort = reference_cohort.intersection(candidate_cohort)
    reference_assigned = shared_cohort.intersection(reference_assignments)
    candidate_assigned = shared_cohort.intersection(candidate_assignments)
    jointly_assigned = reference_assigned.intersection(candidate_assigned)

    cohort_counts = ClusterMembershipCohortCounts(
        reference_cohort=len(reference_cohort),
        candidate_cohort=len(candidate_cohort),
        shared_cohort=len(shared_cohort),
        reference_cohort_only=len(reference_cohort - candidate_cohort),
        candidate_cohort_only=len(candidate_cohort - reference_cohort),
        assigned_in_both=len(jointly_assigned),
        reference_assigned_only=len(reference_assigned - candidate_assigned),
        candidate_assigned_only=len(candidate_assigned - reference_assigned),
        unassigned_in_both=len(shared_cohort - reference_assigned - candidate_assigned),
    )

    reference_labels = tuple(
        sorted({reference_assignments[file_id] for file_id in reference_assigned})
    )
    candidate_labels = tuple(
        sorted({candidate_assignments[file_id] for file_id in candidate_assigned})
    )
    reference_index = {label: index for index, label in enumerate(reference_labels)}
    candidate_index = {label: index for index, label in enumerate(candidate_labels)}
    contingency = np.zeros(
        (len(reference_labels), len(candidate_labels)),
        dtype=np.int64,
    )
    for file_id in jointly_assigned:
        contingency[
            reference_index[reference_assignments[file_id]],
            candidate_index[candidate_assignments[file_id]],
        ] += 1

    paired = _maximum_overlap_pairs(
        contingency,
        reference_labels,
        candidate_labels,
    )
    reference_sizes = Counter(
        reference_assignments[file_id] for file_id in reference_assigned
    )
    candidate_sizes = Counter(
        candidate_assignments[file_id] for file_id in candidate_assigned
    )
    candidate_by_reference = {
        reference_label: candidate_label
        for candidate_label, (reference_label, _overlap) in paired.items()
    }
    cluster_matches: list[ClusterPairSimilarity] = []
    for reference_label in reference_labels:
        candidate_label = candidate_by_reference.get(reference_label)
        intersection = paired[candidate_label][1] if candidate_label is not None else 0
        reference_size = reference_sizes[reference_label]
        candidate_size = (
            candidate_sizes[candidate_label] if candidate_label is not None else 0
        )
        cluster_matches.append(
            ClusterPairSimilarity(
                reference_label=reference_label,
                candidate_label=candidate_label,
                intersection=intersection,
                reference_size=reference_size,
                candidate_size=candidate_size,
                union=reference_size + candidate_size - intersection,
            )
        )
    matched_candidates = set(paired)
    for candidate_label in candidate_labels:
        if candidate_label in matched_candidates:
            continue
        candidate_size = candidate_sizes[candidate_label]
        cluster_matches.append(
            ClusterPairSimilarity(
                reference_label=None,
                candidate_label=candidate_label,
                intersection=0,
                reference_size=0,
                candidate_size=candidate_size,
                union=candidate_size,
            )
        )

    adjusted_rand_index: float | None = None
    normalized_mutual_information: float | None = None
    matched_agreement: float | None = None
    matched_file_ids = sum(overlap for _reference, overlap in paired.values())
    if jointly_assigned:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        ordered_file_ids = sorted(jointly_assigned)
        reference_values = [reference_assignments[value] for value in ordered_file_ids]
        candidate_values = [candidate_assignments[value] for value in ordered_file_ids]
        adjusted_rand_index = float(
            adjusted_rand_score(reference_values, candidate_values)
        )
        normalized_mutual_information = float(
            normalized_mutual_info_score(
                reference_values,
                candidate_values,
                average_method="arithmetic",
            )
        )
        matched_agreement = float(matched_file_ids / len(jointly_assigned))
        status = "ok"
        status_message = "Compared neurons assigned in both saved assignments."
    else:
        status = "no_joint_assignments"
        status_message = (
            "The assignments have no shared file_id values assigned in both runs."
        )

    return ClusterMembershipComparison(
        reference_assignment_id=reference.assignment_id,
        reference_assignment_name=reference.name,
        candidate_assignment_id=candidate.assignment_id,
        candidate_assignment_name=candidate.name,
        status=status,
        status_message=status_message,
        cohort_counts=cohort_counts,
        adjusted_rand_index=adjusted_rand_index,
        normalized_mutual_information=normalized_mutual_information,
        matched_agreement=matched_agreement,
        matched_file_ids=matched_file_ids,
        reference_cluster_ids=reference_labels,
        candidate_cluster_ids=candidate_labels,
        overlap_counts=tuple(
            tuple(int(value) for value in row) for row in contingency.tolist()
        ),
        cluster_matches=tuple(cluster_matches),
    )


def assignment_display_colors(
    assignment: ClusterAssignmentSet,
    *,
    reference: ClusterAssignmentSet | None = None,
) -> tuple[dict[int, list[float]], tuple[ClusterLabelMatch, ...]]:
    """Return display-only cluster colors and the reference matching table."""
    labels = sorted(set(int(value) for value in assignment.assignments.values()))
    matches = (
        match_cluster_labels(reference.assignments, assignment.assignments)
        if reference is not None
        else ()
    )
    match_by_label = {match.candidate_label: match for match in matches}
    palette = neuron_palette(max(32, len(labels) * 4))
    colors: dict[int, list[float]] = {}
    used_colors: set[tuple[float, ...]] = {
        tuple(
            np.round(
                np.asarray(reference.label_colors[match.reference_label])[:4],
                12,
            )
        )
        for match in matches
        if reference is not None
        and match.reference_label is not None
        and match.reference_label in reference.label_colors
    }
    for index, label in enumerate(labels):
        match = match_by_label.get(label)
        color = None
        if match is not None and match.reference_label is not None:
            color = reference.label_colors.get(match.reference_label)
        if color is None and reference is None:
            color = assignment.label_colors.get(label)
        if color is None:
            color = next(
                (
                    candidate
                    for candidate in palette
                    if tuple(np.round(np.asarray(candidate)[:4], 12)) not in used_colors
                ),
                palette[index % len(palette)],
            )
        normalized = list(np.asarray(color, dtype=float)[:4])
        colors[label] = normalized
        used_colors.add(tuple(np.round(normalized, 12)))
    return colors, matches


@dataclass(frozen=True)
class CCFReductionResult:
    """A count-preserving 2D reduction of one CCF volume."""

    data: np.ndarray
    plane: str
    reduction: str
    hidden_axis: int
    vertical_axis: int
    horizontal_axis: int
    x_bounds_um: tuple[float, float]
    y_bounds_um: tuple[float, float]
    included_index_range: tuple[int, int]


def ccf_plane_axes(plane: str) -> tuple[int, int, int]:
    try:
        return _CCF_PLANE_AXES[str(plane)]
    except KeyError as exc:
        raise ValueError(f"Unknown CCF plane: {plane!r}") from exc


def _validated_axis_geometry(
    values: Sequence[object], *, name: str
) -> tuple[float, float, float]:
    normalized = tuple(float(value) for value in values)
    if len(normalized) != 3 or not np.all(np.isfinite(normalized)):
        raise ValueError(f"{name} must contain three finite values.")
    if name == "spacing_um" and any(value <= 0.0 for value in normalized):
        raise ValueError("spacing_um must contain positive values.")
    return normalized


def _ccf_reduction_indices(
    size: int,
    *,
    origin_um: float,
    spacing_um: float,
    reduction: str,
    slice_position_um: float | None,
    slab_thickness_um: float | None,
) -> np.ndarray:
    if reduction == REDUCTION_PROJECTION:
        return np.arange(size, dtype=np.intp)
    if reduction != REDUCTION_SLICE:
        raise ValueError(f"Unknown CCF reduction: {reduction!r}")
    centers = origin_um + np.arange(size, dtype=float) * spacing_um
    position = (
        float(slice_position_um)
        if slice_position_um is not None and np.isfinite(slice_position_um)
        else float(centers[size // 2])
    )
    thickness = (
        float(slab_thickness_um)
        if slab_thickness_um is not None
        and np.isfinite(slab_thickness_um)
        and slab_thickness_um > 0.0
        else spacing_um
    )
    selected = np.flatnonzero(np.abs(centers - position) <= thickness / 2.0)
    if selected.size == 0:
        selected = np.asarray([int(np.argmin(np.abs(centers - position)))])
    return selected.astype(np.intp, copy=False)


def reduce_ccf_volume(
    volume: np.ndarray,
    *,
    plane: str,
    reduction: str = REDUCTION_PROJECTION,
    spacing_um: Sequence[object] = (1.0, 1.0, 1.0),
    origin_um: Sequence[object] = (0.0, 0.0, 0.0),
    slice_position_um: float | None = None,
    slab_thickness_um: float | None = None,
) -> CCFReductionResult:
    """Return a physical-axis-aware slice/slab or sum projection."""
    data = np.asarray(volume)
    if data.ndim != 3:
        raise ValueError(f"CCF heatmap volume must be 3D; got shape {data.shape}.")
    spacing = _validated_axis_geometry(spacing_um, name="spacing_um")
    origin = _validated_axis_geometry(origin_um, name="origin_um")
    hidden, vertical, horizontal = ccf_plane_axes(plane)
    selected = _ccf_reduction_indices(
        data.shape[hidden],
        origin_um=origin[hidden],
        spacing_um=spacing[hidden],
        reduction=reduction,
        slice_position_um=slice_position_um,
        slab_thickness_um=slab_thickness_um,
    )
    reduced = np.take(data, selected, axis=hidden).sum(axis=hidden, dtype=np.float64)
    remaining_axes = [axis for axis in range(3) if axis != hidden]
    if remaining_axes != [vertical, horizontal]:
        reduced = np.transpose(reduced)
    reduced = np.asarray(reduced, dtype=np.float32)

    x_bounds = (
        origin[horizontal] - spacing[horizontal] / 2.0,
        origin[horizontal] + (data.shape[horizontal] - 0.5) * spacing[horizontal],
    )
    y_bounds = (
        origin[vertical] - spacing[vertical] / 2.0,
        origin[vertical] + (data.shape[vertical] - 0.5) * spacing[vertical],
    )
    return CCFReductionResult(
        data=reduced,
        plane=str(plane),
        reduction=str(reduction),
        hidden_axis=hidden,
        vertical_axis=vertical,
        horizontal_axis=horizontal,
        x_bounds_um=x_bounds,
        y_bounds_um=y_bounds,
        included_index_range=(int(selected.min()), int(selected.max())),
    )


@dataclass(frozen=True)
class CCFPointProjection:
    """Projected CCF soma points with a mask into the original input rows."""

    points: np.ndarray
    retained: np.ndarray
    hidden_axis: int
    vertical_axis: int
    horizontal_axis: int


def project_ccf_points(
    coordinates_um: np.ndarray,
    *,
    plane: str,
    reduction: str = REDUCTION_PROJECTION,
    slice_position_um: float | None = None,
    slab_thickness_um: float | None = None,
    default_slab_thickness_um: float = 1.0,
) -> CCFPointProjection:
    """Project ``(RC, DV, LR)`` soma coordinates into one anatomical plane."""
    coordinates = np.asarray(coordinates_um, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(
            "coordinates_um must have shape (n, 3) in CCF coordinate order."
        )
    hidden, vertical, horizontal = ccf_plane_axes(plane)
    retained = np.all(np.isfinite(coordinates), axis=1)
    if reduction == REDUCTION_SLICE:
        finite_hidden = coordinates[retained, hidden]
        if slice_position_um is None or not np.isfinite(slice_position_um):
            position = float(np.median(finite_hidden)) if finite_hidden.size else 0.0
        else:
            position = float(slice_position_um)
        thickness = (
            float(slab_thickness_um)
            if slab_thickness_um is not None
            and np.isfinite(slab_thickness_um)
            and slab_thickness_um > 0.0
            else max(float(default_slab_thickness_um), np.finfo(float).eps)
        )
        retained &= np.abs(coordinates[:, hidden] - position) <= thickness / 2.0
    elif reduction != REDUCTION_PROJECTION:
        raise ValueError(f"Unknown CCF reduction: {reduction!r}")
    points = np.column_stack(
        [coordinates[retained, horizontal], coordinates[retained, vertical]]
    )
    return CCFPointProjection(
        points=points,
        retained=retained,
        hidden_axis=hidden,
        vertical_axis=vertical,
        horizontal_axis=horizontal,
    )


def compose_tinted_heatmaps(
    volumes: Mapping[int, np.ndarray],
    colors: Mapping[int, Sequence[float]],
    *,
    intensity_max: float | None = None,
    opacity: float = 0.85,
) -> tuple[np.ndarray, float]:
    """Compose count arrays into one additive RGBA image.

    All cluster arrays share one count range.  The returned floating-point image
    is in ``[0, 1]`` and can be sent directly to pyqtgraph or a PNG exporter.
    """
    if not volumes:
        return np.zeros((0, 0, 4), dtype=np.float32), 1.0
    ordered = sorted((int(label), np.asarray(data)) for label, data in volumes.items())
    shape = ordered[0][1].shape
    if len(shape) != 2:
        raise ValueError("Tinted comparison heatmaps must be two-dimensional.")
    if any(data.shape != shape for _label, data in ordered):
        raise ValueError("All heatmap cluster arrays must have the same shape.")
    observed_max = max(
        (float(np.nanmax(data)) if data.size and np.any(np.isfinite(data)) else 0.0)
        for _label, data in ordered
    )
    resolved_max = (
        float(intensity_max)
        if intensity_max is not None
        and np.isfinite(intensity_max)
        and intensity_max > 0.0
        else observed_max
    )
    if not np.isfinite(resolved_max) or resolved_max <= 0.0:
        resolved_max = 1.0

    rgba = np.zeros((*shape, 4), dtype=np.float32)
    opacity = float(np.clip(opacity, 0.0, 1.0))
    for index, (label, data) in enumerate(ordered):
        raw_color = colors.get(label)
        if raw_color is None:
            raw_color = neuron_palette(index + 1)[-1]
        color = np.asarray(raw_color, dtype=float).reshape(-1)
        if color.size < 4:
            color = np.pad(color, (0, 4 - color.size), constant_values=1.0)
        normalized = np.nan_to_num(
            np.asarray(data, dtype=float) / resolved_max,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        normalized = np.clip(normalized, 0.0, 1.0)
        contribution_alpha = normalized * opacity * float(np.clip(color[3], 0, 1))
        rgba[..., :3] += contribution_alpha[..., None] * np.clip(color[:3], 0, 1)
        rgba[..., 3] += contribution_alpha
    return np.clip(rgba, 0.0, 1.0), resolved_max


def shared_intensity_maxima(
    entries: Sequence[tuple[tuple[object, ...] | None, float, float | None]],
) -> dict[tuple[object, ...], float]:
    """Resolve one maximum per comparable group, excluding cell overrides."""
    maxima: dict[tuple[object, ...], float] = {}
    for key, observed_max, override in entries:
        if key is None or override is not None:
            continue
        value = float(observed_max)
        if not np.isfinite(value):
            continue
        maxima[key] = max(maxima.get(key, 0.0), value)
    return maxima


def compatible_camera_groups(
    cells: Sequence[ComparisonCellSpec],
    compatibility_by_cell: Mapping[str, tuple[object, ...]],
) -> tuple[tuple[str, ...], ...]:
    """Group opted-in cells that share exact coordinate provenance."""
    grouped: dict[tuple[object, ...], list[str]] = {}
    for cell in cells:
        if not cell.camera_linked:
            continue
        key = compatibility_by_cell.get(cell.cell_id)
        if key is None:
            continue
        grouped.setdefault(key, []).append(cell.cell_id)
    return tuple(tuple(cell_ids) for cell_ids in grouped.values())


def heatmap_filter_signature(metadata: Mapping[str, object]) -> tuple[object, ...]:
    """Return the source/filter identity needed to group Analysis heatmaps."""

    def freeze(value: object) -> object:
        if isinstance(value, Mapping):
            return tuple(
                (str(key), freeze(item))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            )
        if isinstance(value, (list, tuple)):
            return tuple(freeze(item) for item in value)
        return value

    normalized = metadata.get("comparison_filter_signature")
    if isinstance(normalized, Mapping):
        return (
            freeze(normalized),
            metadata.get("comparison_assignment_id"),
        )

    region_ids = metadata.get("heatmap_region_ids")
    node_types = metadata.get("heatmap_node_types")
    atlas_provenance = metadata.get("comparison_atlas_provenance")
    atlas_identity = (
        freeze(atlas_provenance)
        if isinstance(atlas_provenance, Mapping)
        else metadata.get("atlas_name")
    )
    return (
        atlas_identity,
        metadata.get("comparison_assignment_id"),
        tuple(region_ids) if isinstance(region_ids, (list, tuple)) else region_ids,
        tuple(node_types) if isinstance(node_types, (list, tuple)) else node_types,
        metadata.get("heatmap_soma_radius_um"),
        metadata.get("depth_axis"),
        metadata.get("depth_bin_factor"),
    )


def comparison_source_id(layer: Any) -> str | None:
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("comparison_source_id")
    return str(value) if value not in (None, "") else None


def comparison_membership_provenance(
    board: ComparisonBoardState,
    *,
    assignments: Sequence[ClusterAssignmentSet],
    assignment_id_by_cell: Mapping[str, str | None] | None = None,
) -> dict[str, object]:
    """Build reference-versus-board membership records for a JSON export."""
    by_id = {assignment.assignment_id: assignment for assignment in assignments}
    resolved_by_cell = dict(assignment_id_by_cell or {})
    candidate_cells: dict[str, list[str]] = {}
    cells_without_assignment: list[str] = []
    for cell in board.cells:
        assignment_id = cell.assignment_id or resolved_by_cell.get(cell.cell_id)
        if assignment_id in (None, ""):
            cells_without_assignment.append(cell.cell_id)
            continue
        candidate_cells.setdefault(str(assignment_id), []).append(cell.cell_id)

    payload: dict[str, object] = {
        "status": "ok",
        "status_message": "",
        "alignment_key": "file_id",
        "metric_cohort": "shared cohort, assigned in both",
        "label_matching": "maximum_overlap_hungarian_one_to_one",
        "nmi_average_method": "arithmetic",
        "reference_assignment_id": board.reference_assignment_id,
        "cells_without_assignment": cells_without_assignment,
        "comparisons": [],
    }
    reference_id = board.reference_assignment_id
    if reference_id is None:
        payload["status"] = "no_reference"
        payload["status_message"] = (
            "No reference assignment was selected when the board was exported."
        )
        return payload
    reference = by_id.get(reference_id)
    if reference is None:
        payload["status"] = "reference_unavailable"
        payload["status_message"] = (
            f"The reference assignment {reference_id!r} is unavailable."
        )
        return payload

    comparisons: list[dict[str, object]] = []
    for candidate_id, cell_ids in candidate_cells.items():
        if candidate_id == reference_id:
            continue
        candidate = by_id.get(candidate_id)
        if candidate is None:
            comparisons.append(
                {
                    "status": "candidate_unavailable",
                    "status_message": (
                        f"The candidate assignment {candidate_id!r} is unavailable."
                    ),
                    "reference_assignment": {
                        "assignment_id": reference.assignment_id,
                        "name": reference.name,
                    },
                    "candidate_assignment": {
                        "assignment_id": candidate_id,
                        "name": None,
                    },
                    "source_cell_ids": list(cell_ids),
                }
            )
            continue
        record = compare_cluster_memberships(reference, candidate).to_state()
        record["source_cell_ids"] = list(cell_ids)
        comparisons.append(record)
    payload["comparisons"] = comparisons
    if not comparisons:
        payload["status"] = "no_candidate_assignments"
        payload["status_message"] = (
            "The board contains no non-reference assignment to compare."
        )
    return payload


def comparison_provenance(
    board: ComparisonBoardState,
    *,
    cells: Sequence[Mapping[str, object]],
    source_parquet: str | None,
    source_signature: Mapping[str, object] | None = None,
    reference_assignment: Mapping[str, object] | None = None,
    membership_comparisons: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Return the JSON sidecar payload for one exported board."""
    return {
        "format": "napari_neuron_navigator.comparison_board",
        "version": COMPARISON_EXPORT_VERSION,
        "source_parquet": source_parquet,
        "source_signature": dict(source_signature or {}),
        "reference_assignment": dict(reference_assignment or {}),
        "board": board.to_state(),
        "rendered_cells": [dict(cell) for cell in cells],
        "membership_comparisons": dict(membership_comparisons or {}),
    }
