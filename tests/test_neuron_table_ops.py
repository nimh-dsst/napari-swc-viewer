"""Tests for pure neuron-table helper logic."""

import numpy as np
from matplotlib import colormaps

from napari_swc_viewer.neuron_table_ops import (
    ClusterFilterSelection,
    GRAY_RGBA,
    NeuronTableSummary,
    added_flags,
    cluster_filter_matches,
    cluster_ids_available,
    cluster_sort_value,
    has_unclustered_entries,
    recolor_cluster_turbo,
    summarize_neuron_table,
    turbo_colors_for_file_ids,
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


def test_cluster_filter_selection_matches_multiple_groups() -> None:
    """Filtering supports multiple clusters plus unclustered rows."""
    cluster_by_file = {"n1": 1, "n2": 2, "n3": None, "n4": 3}
    selection = ClusterFilterSelection({1, 3}, include_unclustered=True)

    matches = cluster_filter_matches(cluster_by_file, selection)

    assert matches == {"n1": True, "n2": False, "n3": True, "n4": True}
    assert selection.is_all is False


def test_cluster_filter_selection_all_rows_when_empty() -> None:
    """An empty explicit selection means all rows."""
    cluster_by_file = {"n1": 1, "n2": None}
    selection = ClusterFilterSelection()

    matches = cluster_filter_matches(cluster_by_file, selection)

    assert selection.is_all is True
    assert matches == {"n1": True, "n2": True}


def test_visibility_for_selected_cluster_and_show_all() -> None:
    """Visibility map hides non-cluster members and restores all for None."""
    cluster_by_file = {"n1": 1, "n2": 2, "n3": 1}
    hidden = visibility_for_selected_cluster(cluster_by_file, 1)
    shown = visibility_for_selected_cluster(cluster_by_file, None)

    assert hidden == {"n1": True, "n2": False, "n3": True}
    assert shown == {"n1": True, "n2": True, "n3": True}


def test_visibility_for_cluster_selection_with_unclustered() -> None:
    """Visibility can keep selected clusters and unclustered rows visible."""
    cluster_by_file = {"n1": 1, "n2": 2, "n3": None, "n4": 3}
    selection = ClusterFilterSelection({2}, include_unclustered=True)

    visibility = visibility_for_selected_cluster(cluster_by_file, selection)

    assert visibility == {"n1": False, "n2": True, "n3": True, "n4": False}


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


def test_recolor_cluster_turbo_handles_multi_selection_and_unclustered() -> None:
    """Turbo recoloring samples across all selected groups together."""
    cluster_by_file = {"n3": None, "n2": 2, "n1": 1, "n4": 3}
    selection = ClusterFilterSelection({1, 2}, include_unclustered=True)

    colors = recolor_cluster_turbo(cluster_by_file, selection, gray_others=True)

    selected_ids = sorted(["n1", "n2", "n3"])
    expected_samples = np.linspace(0.0, 1.0, 3)
    expected_colors = [
        [float(c) for c in colormaps["turbo"](float(sample))]
        for sample in expected_samples
    ]
    for file_id, expected_color in zip(selected_ids, expected_colors):
        assert colors[file_id] == expected_color
    assert colors["n4"] == list(GRAY_RGBA)


def test_turbo_colors_for_file_ids_is_stable_and_deduplicates() -> None:
    colors = turbo_colors_for_file_ids(["n2", "n10", "n2", "n1"])

    expected_ids = ["n1", "n10", "n2"]
    expected_samples = np.linspace(0.0, 1.0, 3)
    assert list(colors) == expected_ids
    for file_id, sample in zip(expected_ids, expected_samples):
        assert colors[file_id] == [
            float(channel) for channel in colormaps["turbo"](float(sample))
        ]


def test_turbo_colors_for_file_ids_handles_empty_and_singleton_inputs() -> None:
    assert turbo_colors_for_file_ids([]) == {}
    assert turbo_colors_for_file_ids(["only"]) == {
        "only": [float(channel) for channel in colormaps["turbo"](0.0)]
    }


def test_recolor_cluster_turbo_uses_shared_turbo_palette() -> None:
    cluster_by_file = {"n3": 2, "n2": 1, "n1": 1}

    colors = recolor_cluster_turbo(cluster_by_file, 1, gray_others=False)

    assert colors == turbo_colors_for_file_ids(["n1", "n2"])


def test_available_cluster_ids_sorted_unique() -> None:
    """Cluster IDs are unique and sorted ascending, excluding None."""
    cluster_by_file = {"a": 3, "b": None, "c": -1, "d": 3, "e": 2}
    assert cluster_ids_available(cluster_by_file) == [-1, 2, 3]


def test_has_unclustered_entries() -> None:
    """Unclustered availability detects rows without cluster assignment."""
    assert has_unclustered_entries({"a": 1, "b": None}) is True
    assert has_unclustered_entries({"a": 1, "b": 2}) is False


def test_summarize_neuron_table_empty() -> None:
    """Empty table summary reports zeros and no clusters."""
    summary = summarize_neuron_table({}, {}, {})

    assert summary == NeuronTableSummary(
        table_count=0,
        added_count=0,
        visible_count=0,
        cluster_counts=(),
    )


def test_summarize_neuron_table_tracks_counts_and_cluster_breakdown() -> None:
    """Summary reflects table membership, added flags, visibility, and clusters."""
    summary = summarize_neuron_table(
        {"n1": 2, "n2": None, "n3": 1, "n4": 2},
        {"n1": True, "n2": False, "n3": True, "n4": False},
        {"n1": True, "n2": False, "n3": True, "n4": True},
    )

    assert summary.table_count == 4
    assert summary.added_count == 2
    assert summary.visible_count == 3
    assert summary.cluster_counts == ((1, 1), (2, 2), (None, 1))
