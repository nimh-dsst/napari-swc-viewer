"""Tests for pure neuron-table helper logic."""

import numpy as np
from matplotlib import colormaps

from napari_swc_viewer.neuron_table_ops import (
    GRAY_RGBA,
    added_flags,
    cluster_filter_matches,
    cluster_ids_available,
    cluster_sort_value,
    recolor_cluster_turbo,
    visibility_for_selected_cluster,
)


def test_added_flags_transition() -> None:
    """Added flags track current scene membership only."""
    file_ids = ["n1", "n2", "n3"]

    initial = added_flags(file_ids, set())
    after_add = added_flags(file_ids, {"n1", "n3"})
    after_clear = added_flags(file_ids, set())

    assert initial == {"n1": False, "n2": False, "n3": False}
    assert after_add == {"n1": True, "n2": False, "n3": True}
    assert after_clear == {"n1": False, "n2": False, "n3": False}


def test_added_flags_tolerates_mixed_id_types() -> None:
    """Added flags match even when IDs differ only by Python type."""
    file_ids = [1, 2, 3]
    in_scene = {"1", "3"}

    result = added_flags(file_ids, in_scene)

    assert result == {1: True, 2: False, 3: True}


def test_cluster_sort_is_numeric() -> None:
    """Cluster sorting uses numeric values (e.g. -1, 2, 10)."""
    values = [10, -1, 2, None]
    sorted_values = sorted(values, key=cluster_sort_value)
    assert sorted_values == [-1, 2, 10, None]


def test_cluster_filter_matches() -> None:
    """Filtering returns only selected cluster rows, or all for None."""
    cluster_by_file = {"n1": 1, "n2": 2, "n3": 1}
    only_cluster_1 = cluster_filter_matches(cluster_by_file, 1)
    all_rows = cluster_filter_matches(cluster_by_file, None)

    assert only_cluster_1 == {"n1": True, "n2": False, "n3": True}
    assert all(all_rows.values())


def test_visibility_for_selected_cluster_and_show_all() -> None:
    """Visibility map hides non-cluster members and restores all for None."""
    cluster_by_file = {"n1": 1, "n2": 2, "n3": 1}
    hidden = visibility_for_selected_cluster(cluster_by_file, 1)
    shown = visibility_for_selected_cluster(cluster_by_file, None)

    assert hidden == {"n1": True, "n2": False, "n3": True}
    assert shown == {"n1": True, "n2": True, "n3": True}


def test_recolor_cluster_turbo_grays_others() -> None:
    """Selected cluster gets turbo ramp while non-selected become gray."""
    cluster_by_file = {"n10": 2, "n2": 1, "n1": 1}
    colors = recolor_cluster_turbo(cluster_by_file, 1, gray_others=True)

    # Deterministic order is by file_id ascending within the selected cluster.
    selected_ids = sorted(["n1", "n2"])
    expected_samples = np.linspace(0.0, 1.0, 2)
    expected_colors = [
        [float(c) for c in colormaps["turbo"](float(expected_samples[0]))],
        [float(c) for c in colormaps["turbo"](float(expected_samples[1]))],
    ]

    assert colors[selected_ids[0]] == expected_colors[0]
    assert colors[selected_ids[1]] == expected_colors[1]
    assert colors["n10"] == list(GRAY_RGBA)


def test_available_cluster_ids_sorted_unique() -> None:
    """Cluster IDs are unique and sorted ascending, excluding None."""
    cluster_by_file = {"a": 3, "b": None, "c": -1, "d": 3, "e": 2}
    assert cluster_ids_available(cluster_by_file) == [-1, 2, 3]
