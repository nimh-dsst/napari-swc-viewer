"""Named, persistent cluster-assignment sets.

The analysis pipeline produces one :class:`~napari_neuron_navigator.analysis.clustering.ClusterResult`
at a time.  This module keeps the comparatively small, durable part of those
results (labels and provenance) independently from the potentially large
correlation, distance, and linkage matrices.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
import colorsys
from dataclasses import dataclass, field
import re
from typing import Any
from uuid import uuid4


CLUSTER_ASSIGNMENT_STATE_VERSION = 2


def _unique_strings(values: Iterable[object]) -> tuple[str, ...]:
    """Return non-empty strings in first-seen order."""
    if values is None:  # type: ignore[comparison-overlap]
        return ()
    return tuple(dict.fromkeys(str(value) for value in values if str(value)))


def _coerce_color(value: object) -> list[float]:
    """Return one normalized RGBA color."""
    try:
        color = [float(component) for component in value]  # type: ignore[union-attr]
    except (TypeError, ValueError):
        color = [0.5, 0.5, 0.5, 1.0]
    while len(color) < 4:
        color.append(1.0)
    return color[:4]


def _slug(value: str) -> str:
    """Return a conservative Parquet-column slug."""
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return normalized or "assignment"


def _default_label_colors(labels: Iterable[int]) -> dict[int, list[float]]:
    """Return a stable, distinct fallback palette for integer labels."""
    palette: dict[int, list[float]] = {}
    for index, label in enumerate(sorted({int(value) for value in labels})):
        red, green, blue = colorsys.hsv_to_rgb(
            (index * 0.618033988749895) % 1.0,
            0.68,
            0.92,
        )
        palette[label] = [red, green, blue, 1.0]
    return palette


@dataclass
class ClusterAssignmentSet:
    """One named, sparse cluster assignment produced by one analysis run."""

    assignment_id: str
    name: str
    column_name: str
    assignments: dict[str, int] = field(default_factory=dict)
    input_file_ids: tuple[str, ...] = ()
    unassigned_neuron_ids: tuple[str, ...] = ()
    label_colors: dict[int, list[float]] = field(default_factory=dict)
    run_metadata: dict[str, object] = field(default_factory=dict)
    input_scope: str = "whole"
    coordinate_space: str = "CCFv3 Coordinates"
    parent_assignment_id: str | None = None
    parent_cluster_ids: tuple[int, ...] = ()
    created_order: int = 0
    runtime_result: Any | None = field(default=None, repr=False, compare=False)

    def label_for(self, file_id: object) -> int | None:
        """Return the assigned label for *file_id*, if any."""
        value = self.assignments.get(str(file_id))
        return None if value is None else int(value)

    def file_ids_for_label(self, label: int) -> tuple[str, ...]:
        """Return file IDs assigned to *label* in stable input order."""
        target = int(label)
        ordered = [
            file_id
            for file_id in self.input_file_ids
            if self.assignments.get(file_id) == target
        ]
        seen = set(ordered)
        ordered.extend(
            file_id
            for file_id, value in self.assignments.items()
            if value == target and file_id not in seen
        )
        return tuple(ordered)

    def to_state(self, *, include_runtime: bool = False) -> dict[str, object]:
        """Return a JSON-safe representation of this assignment set."""
        state: dict[str, object] = {
            "assignment_id": self.assignment_id,
            "name": self.name,
            "column_name": self.column_name,
            "assignments": {
                str(file_id): int(label) for file_id, label in self.assignments.items()
            },
            "input_file_ids": list(self.input_file_ids),
            "unassigned_neuron_ids": list(self.unassigned_neuron_ids),
            "label_colors": {
                str(label): list(color) for label, color in self.label_colors.items()
            },
            "run_metadata": dict(self.run_metadata),
            "input_scope": self.input_scope,
            "coordinate_space": self.coordinate_space,
            "parent_assignment_id": self.parent_assignment_id,
            "parent_cluster_ids": list(self.parent_cluster_ids),
            "created_order": int(self.created_order),
        }
        if include_runtime:
            state["runtime_result"] = self.runtime_result
        return state

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> "ClusterAssignmentSet":
        """Restore an assignment set from serialized state."""
        raw_assignments = state.get("assignments", {})
        assignments: dict[str, int] = {}
        if isinstance(raw_assignments, Mapping):
            for file_id, label in raw_assignments.items():
                if label in (None, ""):
                    continue
                try:
                    assignments[str(file_id)] = int(label)
                except (TypeError, ValueError):
                    continue

        raw_colors = state.get("label_colors", {})
        label_colors: dict[int, list[float]] = {}
        if isinstance(raw_colors, Mapping):
            for label, color in raw_colors.items():
                try:
                    label_colors[int(label)] = _coerce_color(color)
                except (TypeError, ValueError):
                    continue

        raw_metadata = state.get("run_metadata", {})
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        assignment_id = str(state.get("assignment_id") or uuid4().hex)
        name = str(state.get("name") or "Cluster Assignment")
        column_name = str(state.get("column_name") or f"cluster_{_slug(name)}")
        return cls(
            assignment_id=assignment_id,
            name=name,
            column_name=column_name,
            assignments=assignments,
            input_file_ids=_unique_strings(state.get("input_file_ids", ())),  # type: ignore[arg-type]
            unassigned_neuron_ids=_unique_strings(
                state.get("unassigned_neuron_ids", ())  # type: ignore[arg-type]
            ),
            label_colors=label_colors,
            run_metadata=metadata,
            input_scope=str(state.get("input_scope") or "whole"),
            coordinate_space=str(state.get("coordinate_space") or "CCFv3 Coordinates"),
            parent_assignment_id=(
                str(state["parent_assignment_id"])
                if state.get("parent_assignment_id") not in (None, "")
                else None
            ),
            parent_cluster_ids=tuple(
                int(value)
                for value in state.get("parent_cluster_ids", ())  # type: ignore[union-attr]
            ),
            created_order=int(state.get("created_order") or 0),
        )


class ClusterAssignmentStore:
    """Ordered collection of cluster-assignment sets with one active set."""

    def __init__(self) -> None:
        self._sets: OrderedDict[str, ClusterAssignmentSet] = OrderedDict()
        self._active_assignment_id: str | None = None
        self._next_created_order = 1

    @property
    def active_assignment_id(self) -> str | None:
        return self._active_assignment_id

    @property
    def active(self) -> ClusterAssignmentSet | None:
        return self.get(self._active_assignment_id)

    def __len__(self) -> int:
        return len(self._sets)

    def sets(self) -> tuple[ClusterAssignmentSet, ...]:
        """Return assignment sets in creation order."""
        return tuple(self._sets.values())

    def get(self, assignment_id: str | None) -> ClusterAssignmentSet | None:
        if assignment_id is None:
            return None
        return self._sets.get(str(assignment_id))

    def _unique_name(self, base_name: str) -> str:
        existing = {assignment.name.casefold() for assignment in self._sets.values()}
        if base_name.casefold() not in existing:
            return base_name
        index = 2
        while f"{base_name} {index}".casefold() in existing:
            index += 1
        return f"{base_name} {index}"

    def _next_run_index(self, method_name: str) -> int:
        """Return the next per-method run index, independent of renames."""
        prefix = str(method_name).strip() or "Cluster Assignment"
        pattern = re.compile(rf"^{re.escape(prefix)}\s+(\d+)$", re.IGNORECASE)
        used: list[int] = []
        for assignment in self._sets.values():
            stored_method = str(assignment.run_metadata.get("clustering_method", ""))
            stored_index = assignment.run_metadata.get("method_run_index")
            if stored_method.casefold() == prefix.casefold():
                try:
                    used.append(int(stored_index))
                    continue
                except (TypeError, ValueError):
                    pass
            match = pattern.match(assignment.name)
            if match is not None:
                used.append(int(match.group(1)))
        return max(used, default=0) + 1

    def next_run_name(self, method_name: str) -> str:
        """Return the next ``<method> N`` display name."""
        prefix = str(method_name).strip() or "Cluster Assignment"
        index = self._next_run_index(prefix)
        return self._unique_name(f"{prefix} {index}")

    def _unique_column_name(self, display_name: str) -> str:
        base = f"cluster_{_slug(display_name)}"
        existing = {assignment.column_name for assignment in self._sets.values()}
        if base not in existing:
            return base
        index = 2
        while f"{base}_{index}" in existing:
            index += 1
        return f"{base}_{index}"

    def add(
        self,
        *,
        name: str,
        assignments: Mapping[object, object],
        input_file_ids: Iterable[object],
        unassigned_neuron_ids: Iterable[object] = (),
        label_colors: Mapping[object, object] | None = None,
        run_metadata: Mapping[str, object] | None = None,
        input_scope: str = "whole",
        coordinate_space: str = "CCFv3 Coordinates",
        parent_assignment_id: str | None = None,
        parent_cluster_ids: Iterable[int] = (),
        runtime_result: Any | None = None,
        assignment_id: str | None = None,
        column_name: str | None = None,
        activate: bool = True,
    ) -> ClusterAssignmentSet:
        """Create and optionally activate one assignment set."""
        normalized_assignments: dict[str, int] = {}
        for file_id, label in assignments.items():
            if label in (None, ""):
                continue
            try:
                normalized_assignments[str(file_id)] = int(label)
            except (TypeError, ValueError):
                continue
        normalized_colors: dict[int, list[float]] = {}
        for label, color in (label_colors or {}).items():
            try:
                normalized_colors[int(label)] = _coerce_color(color)
            except (TypeError, ValueError):
                continue
        fallback_colors = _default_label_colors(normalized_assignments.values())
        for label, color in fallback_colors.items():
            normalized_colors.setdefault(label, color)

        stable_id = str(assignment_id or uuid4().hex)
        if stable_id in self._sets:
            raise ValueError(f"Duplicate cluster assignment ID: {stable_id}")
        display_name = self._unique_name(str(name).strip() or "Cluster Assignment")
        assignment = ClusterAssignmentSet(
            assignment_id=stable_id,
            name=display_name,
            column_name=(
                str(column_name)
                if column_name
                else self._unique_column_name(display_name)
            ),
            assignments=normalized_assignments,
            input_file_ids=_unique_strings(input_file_ids),
            unassigned_neuron_ids=_unique_strings(unassigned_neuron_ids),
            label_colors=normalized_colors,
            run_metadata=dict(run_metadata or {}),
            input_scope=str(input_scope),
            coordinate_space=str(coordinate_space),
            parent_assignment_id=(
                str(parent_assignment_id) if parent_assignment_id else None
            ),
            parent_cluster_ids=tuple(
                sorted({int(value) for value in parent_cluster_ids})
            ),
            created_order=self._next_created_order,
            runtime_result=runtime_result,
        )
        self._next_created_order += 1
        self._sets[stable_id] = assignment
        if activate:
            self._active_assignment_id = stable_id
        return assignment

    def add_result(
        self,
        result: Any,
        *,
        method_name: str,
        input_file_ids: Iterable[object] | None = None,
        label_colors: Mapping[object, object] | None = None,
        run_metadata: Mapping[str, object] | None = None,
        input_scope: str = "whole",
        coordinate_space: str = "CCFv3 Coordinates",
        parent_assignment_id: str | None = None,
        parent_cluster_ids: Iterable[int] = (),
    ) -> ClusterAssignmentSet:
        """Create an assignment set from a clustering result."""
        raw_neuron_ids = getattr(result, "neuron_ids", ())
        raw_labels = getattr(result, "labels", ())
        neuron_ids = list(()) if raw_neuron_ids is None else list(raw_neuron_ids)
        labels = list(()) if raw_labels is None else list(raw_labels)
        assignments = dict(zip(neuron_ids, labels))
        unassigned = list(getattr(result, "unassigned_neuron_ids", ()) or ())
        cohort = (
            list(input_file_ids)
            if input_file_ids is not None
            else [*neuron_ids, *unassigned]
        )
        method_display = str(method_name).strip() or "Cluster Assignment"
        run_index = self._next_run_index(method_display)
        durable_metadata = dict(run_metadata or {})
        durable_metadata.update(
            {
                "clustering_method": method_display,
                "method_run_index": run_index,
            }
        )
        return self.add(
            name=self._unique_name(f"{method_display} {run_index}"),
            assignments=assignments,
            input_file_ids=cohort,
            unassigned_neuron_ids=unassigned,
            label_colors=label_colors,
            run_metadata=durable_metadata,
            input_scope=input_scope,
            coordinate_space=coordinate_space,
            parent_assignment_id=parent_assignment_id,
            parent_cluster_ids=parent_cluster_ids,
            runtime_result=result,
        )

    def set_active(self, assignment_id: str | None) -> None:
        """Activate a set, or clear the active set with ``None``."""
        if assignment_id is not None and str(assignment_id) not in self._sets:
            raise KeyError(f"Unknown cluster assignment ID: {assignment_id}")
        self._active_assignment_id = (
            None if assignment_id is None else str(assignment_id)
        )

    def rename(self, assignment_id: str, name: str) -> str:
        """Rename a set without changing its stable Parquet column."""
        assignment = self._sets[str(assignment_id)]
        requested = str(name).strip()
        if not requested:
            raise ValueError("Cluster assignment name cannot be empty.")
        other_names = {
            item.name.casefold()
            for key, item in self._sets.items()
            if key != str(assignment_id)
        }
        candidate = requested
        suffix = 2
        while candidate.casefold() in other_names:
            candidate = f"{requested} {suffix}"
            suffix += 1
        assignment.name = candidate
        return candidate

    def delete(self, assignment_id: str) -> ClusterAssignmentSet:
        """Delete one set and activate the most recent remaining set."""
        removed = self._sets.pop(str(assignment_id))
        if self._active_assignment_id == str(assignment_id):
            remaining = max(
                self._sets.values(),
                key=lambda item: item.created_order,
                default=None,
            )
            self._active_assignment_id = (
                remaining.assignment_id if remaining is not None else None
            )
        return removed

    def active_map(
        self, file_ids: Iterable[object] | None = None
    ) -> dict[object, int | None]:
        """Return active labels, optionally including explicit blank rows."""
        active = self.active
        if file_ids is None:
            return {} if active is None else dict(active.assignments)
        return {
            file_id: (None if active is None else active.label_for(file_id))
            for file_id in file_ids
        }

    def to_state(self) -> dict[str, object]:
        """Return serializable state without runtime matrices."""
        return {
            "version": CLUSTER_ASSIGNMENT_STATE_VERSION,
            "active_assignment_id": self._active_assignment_id,
            "sets": [assignment.to_state() for assignment in self._sets.values()],
        }

    def load_state(self, state: Mapping[str, object]) -> None:
        """Replace store contents from version-2 state."""
        self._sets.clear()
        self._active_assignment_id = None
        self._next_created_order = 1
        raw_sets = state.get("sets", ())
        if isinstance(raw_sets, (str, bytes, Mapping)):
            raw_sets = ()
        for raw_set in raw_sets:  # type: ignore[union-attr]
            if not isinstance(raw_set, Mapping):
                continue
            assignment = ClusterAssignmentSet.from_state(raw_set)
            if assignment.assignment_id in self._sets:
                continue
            self._sets[assignment.assignment_id] = assignment
            self._next_created_order = max(
                self._next_created_order,
                assignment.created_order + 1,
            )
        requested_active = state.get("active_assignment_id")
        if requested_active is not None and str(requested_active) in self._sets:
            self._active_assignment_id = str(requested_active)
        elif self._sets:
            self._active_assignment_id = max(
                self._sets.values(), key=lambda item: item.created_order
            ).assignment_id

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> "ClusterAssignmentStore":
        store = cls()
        store.load_state(state)
        return store

    def import_legacy(
        self,
        assignments: Mapping[object, object],
        *,
        activate: bool = True,
    ) -> ClusterAssignmentSet | None:
        """Create one set from legacy single-cluster values."""
        normalized: dict[str, int] = {}
        for file_id, label in assignments.items():
            if label in (None, ""):
                continue
            try:
                normalized[str(file_id)] = int(label)
            except (TypeError, ValueError):
                continue
        if not normalized:
            return None
        return self.add(
            name="Imported Cluster Assignment",
            assignments=normalized,
            input_file_ids=assignments.keys(),
            input_scope="imported",
            activate=activate,
        )
