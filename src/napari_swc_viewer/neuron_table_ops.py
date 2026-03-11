"""Pure helpers for neuron table filtering, visibility, and recoloring."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
from matplotlib import colormaps

GRAY_RGBA = [0.5, 0.5, 0.5, 1.0]


def cluster_sort_value(cluster_id: int | None) -> int:
    """Return a stable numeric sort key for cluster values."""
    return int(cluster_id) if cluster_id is not None else 10**9


def cluster_ids_available(cluster_by_file: Mapping[str, int | None]) -> list[int]:
    """Return sorted unique cluster IDs present in the map."""
    return sorted({int(cluster) for cluster in cluster_by_file.values() if cluster is not None})


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


def cluster_filter_matches(
    cluster_by_file: Mapping[str, int | None],
    cluster_id: int | None,
) -> dict[str, bool]:
    """Return per-neuron row-visibility for a cluster filter."""
    if cluster_id is None:
        return {file_id: True for file_id in cluster_by_file}
    return {file_id: cluster == cluster_id for file_id, cluster in cluster_by_file.items()}


def visibility_for_selected_cluster(
    cluster_by_file: Mapping[str, int | None],
    cluster_id: int | None,
) -> dict[str, bool]:
    """Return per-neuron visibility map where only selected cluster is visible."""
    if cluster_id is None:
        return {file_id: True for file_id in cluster_by_file}
    return {file_id: cluster == cluster_id for file_id, cluster in cluster_by_file.items()}


def recolor_cluster_turbo(
    cluster_by_file: Mapping[str, int | None],
    cluster_id: int | None,
    gray_others: bool = True,
) -> dict[str, list[float]]:
    """Return color updates for a selected cluster using turbo linear sampling."""
    if cluster_id is None:
        return {}

    member_ids = sorted(
        file_id
        for file_id, cluster in cluster_by_file.items()
        if cluster == cluster_id
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
