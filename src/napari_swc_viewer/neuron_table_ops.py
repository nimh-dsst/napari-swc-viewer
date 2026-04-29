"""Pure helpers for neuron table filtering, visibility, recoloring, and summary."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable, Mapping

import numpy as np
from matplotlib import colormaps

GRAY_RGBA = [0.5, 0.5, 0.5, 1.0]


@dataclass(frozen=True)
class ClusterFilterSelection:
    """Selected cluster groups for filtering table rows."""

    cluster_ids: frozenset[int] = field(default_factory=frozenset)
    include_unclustered: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cluster_ids",
            frozenset(int(cluster_id) for cluster_id in self.cluster_ids),
        )
        object.__setattr__(
            self,
            "include_unclustered",
            bool(self.include_unclustered),
        )

    @property
    def is_all(self) -> bool:
        """Return whether this selection represents all table rows."""
        return not self.cluster_ids and not self.include_unclustered


@dataclass(frozen=True)
class NeuronTableSummary:
    """Summary counts for the current neuron table contents."""

    table_count: int
    added_count: int
    visible_count: int
    cluster_counts: tuple[tuple[int | None, int], ...]


def cluster_sort_value(cluster_id: int | None) -> int:
    """Return a stable numeric sort key for cluster values."""
    return int(cluster_id) if cluster_id is not None else 10**9


def cluster_ids_available(cluster_by_file: Mapping[str, int | None]) -> list[int]:
    """Return sorted unique cluster IDs present in the map."""
    return sorted(
        {int(cluster) for cluster in cluster_by_file.values() if cluster is not None}
    )


def has_unclustered_entries(cluster_by_file: Mapping[str, int | None]) -> bool:
    """Return whether the table contains rows without a cluster assignment."""
    return any(cluster is None for cluster in cluster_by_file.values())


def _coerce_cluster_selection(
    selection: ClusterFilterSelection | int | None,
) -> ClusterFilterSelection:
    """Return a cluster filter selection from the new or legacy API."""
    if isinstance(selection, ClusterFilterSelection):
        return selection
    if selection is None:
        return ClusterFilterSelection()
    return ClusterFilterSelection(frozenset({int(selection)}))


def added_flags(
    file_ids: Iterable[object],
    file_ids_in_scene: Iterable[object],
) -> dict[object, bool]:
    """Return per-neuron added-state flags.

    Membership is checked by exact value first, then by ``str(...)`` fallback
    to tolerate mixed identifier types across table/db/layer metadata.
    """
    in_scene = set(file_ids_in_scene)
    in_scene_str = {str(file_id) for file_id in in_scene}
    return {
        file_id: (file_id in in_scene) or (str(file_id) in in_scene_str)
        for file_id in file_ids
    }


def summarize_neuron_table(
    cluster_by_file: Mapping[object, int | None],
    added_by_file: Mapping[object, bool],
    visible_by_file: Mapping[object, bool],
) -> NeuronTableSummary:
    """Return summary counts for the current neuron table state."""
    file_ids = list(cluster_by_file)
    cluster_counts_by_id: dict[int | None, int] = {}
    for file_id in file_ids:
        cluster_id = cluster_by_file.get(file_id)
        cluster_counts_by_id[cluster_id] = cluster_counts_by_id.get(cluster_id, 0) + 1

    cluster_counts = [
        (int(cluster_id), count)
        for cluster_id, count in cluster_counts_by_id.items()
        if cluster_id is not None
    ]
    cluster_counts.sort(key=lambda item: int(item[0]))
    if None in cluster_counts_by_id:
        cluster_counts.append((None, cluster_counts_by_id[None]))

    return NeuronTableSummary(
        table_count=len(file_ids),
        added_count=sum(bool(added_by_file.get(file_id, False)) for file_id in file_ids),
        visible_count=sum(bool(visible_by_file.get(file_id, False)) for file_id in file_ids),
        cluster_counts=tuple(cluster_counts),
    )


def cluster_filter_matches(
    cluster_by_file: Mapping[str, int | None],
    selection: ClusterFilterSelection | int | None,
) -> dict[str, bool]:
    """Return per-neuron row-visibility for a cluster filter."""
    selected = _coerce_cluster_selection(selection)
    if selected.is_all:
        return {file_id: True for file_id in cluster_by_file}
    return {
        file_id: (
            (cluster is None and selected.include_unclustered)
            or (cluster is not None and int(cluster) in selected.cluster_ids)
        )
        for file_id, cluster in cluster_by_file.items()
    }


def visibility_for_selected_cluster(
    cluster_by_file: Mapping[str, int | None],
    selection: ClusterFilterSelection | int | None,
) -> dict[str, bool]:
    """Return per-neuron visibility map where only selected cluster is visible."""
    return cluster_filter_matches(cluster_by_file, selection)


def recolor_cluster_turbo(
    cluster_by_file: Mapping[str, int | None],
    selection: ClusterFilterSelection | int | None,
    gray_others: bool = True,
) -> dict[str, list[float]]:
    """Return color updates for a selected cluster using turbo linear sampling."""
    selected = _coerce_cluster_selection(selection)
    if selected.is_all:
        return {}

    matches = cluster_filter_matches(cluster_by_file, selected)
    member_ids = sorted(
        (file_id for file_id, is_match in matches.items() if is_match),
        key=str,
    )
    if not member_ids and not gray_others:
        return {}

    cmap = colormaps["turbo"]
    samples = np.linspace(0.0, 1.0, max(len(member_ids), 1))

    updates: dict[str, list[float]] = {}
    for file_id, t in zip(member_ids, samples):
        updates[file_id] = [float(c) for c in cmap(float(t))]

    if gray_others:
        for file_id in cluster_by_file:
            if file_id not in updates:
                updates[file_id] = list(GRAY_RGBA)

    return updates
