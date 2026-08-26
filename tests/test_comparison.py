from __future__ import annotations

import json

import numpy as np
import pytest

from napari_swc_viewer.cluster_assignments import ClusterAssignmentStore
from napari_swc_viewer.comparison import (
    CCF_PLANE_CORONAL,
    CCF_PLANE_HORIZONTAL,
    CCF_PLANE_SAGITTAL,
    REDUCTION_PROJECTION,
    REDUCTION_SLICE,
    SOURCE_FLATMAP_ARBOR_HEATMAP,
    ComparisonBoardState,
    ComparisonCellSpec,
    assignment_display_colors,
    compare_cluster_memberships,
    compatible_camera_groups,
    comparison_membership_provenance,
    comparison_provenance,
    compose_tinted_heatmaps,
    match_cluster_labels,
    project_ccf_points,
    reduce_ccf_volume,
    shared_intensity_maxima,
)


def _assignment(
    store: ClusterAssignmentStore,
    name: str,
    assignments: dict[str, int],
    *,
    colors: dict[int, list[float]] | None = None,
    input_file_ids: list[str] | None = None,
    unassigned_file_ids: list[str] | None = None,
):
    return store.add(
        name=name,
        assignments=assignments,
        input_file_ids=input_file_ids or list(assignments),
        unassigned_neuron_ids=unassigned_file_ids or (),
        label_colors=colors,
    )


def test_overlap_matching_uses_file_id_with_duplicate_display_neuron_ids():
    records = [
        {"file_id": "subject-a/42.swc", "subject": "A", "neuron_id": "42"},
        {"file_id": "subject-b/42.swc", "subject": "B", "neuron_id": "42"},
        {"file_id": "subject-c/9.swc", "subject": "C", "neuron_id": "9"},
        {"file_id": "subject-d/10.swc", "subject": "D", "neuron_id": "10"},
    ]
    file_ids = [record["file_id"] for record in records]
    reference = {
        file_ids[0]: 2,
        file_ids[1]: 8,
        file_ids[2]: 8,
        file_ids[3]: 2,
    }
    candidate = {
        file_ids[0]: 7,
        file_ids[1]: 9,
        file_ids[2]: 10,
        file_ids[3]: 7,
        "subject-e/11.swc": 11,
    }

    matches = {
        match.candidate_label: match
        for match in match_cluster_labels(reference, candidate)
    }

    assert records[0]["neuron_id"] == records[1]["neuron_id"]
    assert matches[7].reference_label == 2
    assert matches[7].shared_file_ids == 2
    assert {matches[9].reference_label, matches[10].reference_label} == {8, None}
    assert matches[11].reference_label is None
    assert matches[11].candidate_shared_cohort == 0

    store = ClusterAssignmentStore()
    comparison = compare_cluster_memberships(
        _assignment(store, "Reference", reference),
        _assignment(store, "Candidate", candidate),
    )
    assert comparison.cohort_counts.assigned_in_both == 4
    assert comparison.matched_file_ids == 3


def test_assignment_display_colors_preserve_reference_and_separate_unmatched():
    store = ClusterAssignmentStore()
    reference = _assignment(
        store,
        "Reference",
        {"f1": 2, "f2": 2, "f3": 8, "f4": 8},
        colors={2: [1.0, 0.0, 0.0, 1.0], 8: [0.0, 1.0, 0.0, 1.0]},
    )
    candidate = _assignment(
        store,
        "Candidate",
        {"f1": 7, "f2": 7, "f3": 9, "f4": 10, "f5": 11},
    )

    colors, matches = assignment_display_colors(candidate, reference=reference)
    by_label = {match.candidate_label: match for match in matches}

    assert colors[7] == reference.label_colors[2]
    matched_split = next(
        label for label in (9, 10) if by_label[label].reference_label == 8
    )
    assert colors[matched_split] == reference.label_colors[8]
    unmatched = [
        label for label in (9, 10, 11) if by_label[label].reference_label is None
    ]
    assert len({tuple(colors[label]) for label in unmatched}) == len(unmatched)
    assert not {tuple(colors[label]) for label in unmatched}.intersection(
        {tuple(reference.label_colors[2]), tuple(reference.label_colors[8])}
    )

    saved_colors, no_matches = assignment_display_colors(candidate, reference=None)
    assert no_matches == ()
    assert saved_colors == candidate.label_colors


def test_membership_metrics_are_perfect_after_label_permutation():
    store = ClusterAssignmentStore()
    reference = _assignment(
        store,
        "Reference",
        {"f1": 1, "f2": 1, "f3": 2, "f4": 2},
        input_file_ids=["f1", "f2", "f3", "f4", "f5", "f6"],
        unassigned_file_ids=["f5", "f6"],
    )
    candidate = _assignment(
        store,
        "Candidate",
        {"f1": 7, "f2": 7, "f3": 8, "f4": 8},
        input_file_ids=["f1", "f2", "f3", "f4", "f5", "g1"],
        unassigned_file_ids=["f5", "g1"],
    )

    comparison = compare_cluster_memberships(reference, candidate)

    assert comparison.adjusted_rand_index == pytest.approx(1.0)
    assert comparison.normalized_mutual_information == pytest.approx(1.0)
    assert comparison.matched_agreement == pytest.approx(1.0)
    assert comparison.matched_file_ids == 4
    assert comparison.cohort_counts.to_state() == {
        "reference_cohort": 6,
        "candidate_cohort": 6,
        "shared_cohort": 5,
        "reference_cohort_only": 1,
        "candidate_cohort_only": 1,
        "assigned_in_both": 4,
        "reference_assigned_only": 0,
        "candidate_assigned_only": 0,
        "unassigned_in_both": 1,
    }
    assert comparison.reference_cluster_ids == (1, 2)
    assert comparison.candidate_cluster_ids == (7, 8)
    assert comparison.overlap_counts == ((2, 0), (0, 2))
    assert [match.jaccard for match in comparison.cluster_matches] == [1.0, 1.0]


def test_membership_metrics_report_negative_ari_and_split_overlap():
    store = ClusterAssignmentStore()
    reference = _assignment(
        store,
        "Reference",
        {"f1": 1, "f2": 1, "f3": 2, "f4": 2},
    )
    candidate = _assignment(
        store,
        "Candidate",
        {"f1": 7, "f2": 8, "f3": 7, "f4": 8},
    )

    comparison = compare_cluster_memberships(reference, candidate)

    assert comparison.adjusted_rand_index == pytest.approx(-0.5)
    assert comparison.normalized_mutual_information == pytest.approx(0.0)
    assert comparison.matched_agreement == pytest.approx(0.5)
    assert comparison.overlap_counts == ((1, 1), (1, 1))
    assert all(
        match.jaccard == pytest.approx(1 / 3) for match in comparison.cluster_matches
    )


def test_membership_jaccard_includes_one_sided_assignments_in_shared_cohort():
    store = ClusterAssignmentStore()
    reference = _assignment(
        store,
        "Reference",
        {"f1": 1, "f2": 1},
        input_file_ids=["f1", "f2", "f3", "f4"],
        unassigned_file_ids=["f3", "f4"],
    )
    candidate = _assignment(
        store,
        "Candidate",
        {"f1": 7, "f3": 7},
        input_file_ids=["f1", "f2", "f3", "f4"],
        unassigned_file_ids=["f2", "f4"],
    )

    comparison = compare_cluster_memberships(reference, candidate)

    assert comparison.cohort_counts.assigned_in_both == 1
    assert comparison.cohort_counts.reference_assigned_only == 1
    assert comparison.cohort_counts.candidate_assigned_only == 1
    assert comparison.cohort_counts.unassigned_in_both == 1
    assert comparison.cluster_matches[0].reference_size == 2
    assert comparison.cluster_matches[0].candidate_size == 2
    assert comparison.cluster_matches[0].jaccard == pytest.approx(1 / 3)


def test_membership_metrics_are_null_without_joint_assignments():
    store = ClusterAssignmentStore()
    reference = _assignment(
        store,
        "Reference",
        {"f1": 1},
        input_file_ids=["f1", "f2"],
        unassigned_file_ids=["f2"],
    )
    candidate = _assignment(
        store,
        "Candidate",
        {"f2": 7},
        input_file_ids=["f1", "f2"],
        unassigned_file_ids=["f1"],
    )

    comparison = compare_cluster_memberships(reference, candidate)
    payload = comparison.to_state()

    assert comparison.status == "no_joint_assignments"
    assert comparison.adjusted_rand_index is None
    assert comparison.normalized_mutual_information is None
    assert comparison.matched_agreement is None
    assert payload["metrics"]["adjusted_rand_index"] is None
    assert json.loads(json.dumps(payload))["metrics"]["matched_agreement"] is None


def test_membership_metrics_keep_zero_overlap_clusters_explicitly_unmatched():
    store = ClusterAssignmentStore()
    cohort = ["f1", "f2", "f3", "f4", "f5", "f6"]
    reference = _assignment(
        store,
        "Reference",
        {"f1": 1, "f2": 2, "f3": 3, "f5": 4},
        input_file_ids=cohort,
        unassigned_file_ids=["f4", "f6"],
    )
    candidate = _assignment(
        store,
        "Candidate",
        {"f1": 7, "f2": 8, "f3": 9, "f6": 11},
        input_file_ids=cohort,
        unassigned_file_ids=["f4", "f5"],
    )

    comparison = compare_cluster_memberships(reference, candidate)

    assert any(
        match.reference_label == 4 and match.candidate_label is None
        for match in comparison.cluster_matches
    )
    assert any(
        match.reference_label is None and match.candidate_label == 11
        for match in comparison.cluster_matches
    )


@pytest.mark.parametrize(
    ("plane", "expected", "expected_shape", "x_bounds", "y_bounds"),
    [
        (
            CCF_PLANE_CORONAL,
            lambda data: data.sum(axis=0),
            (3, 4),
            (998.0, 1014.0),
            (197.5, 212.5),
        ),
        (
            CCF_PLANE_HORIZONTAL,
            lambda data: data.sum(axis=1),
            (2, 4),
            (998.0, 1014.0),
            (45.0, 65.0),
        ),
        (
            CCF_PLANE_SAGITTAL,
            lambda data: data.sum(axis=2).T,
            (3, 2),
            (45.0, 65.0),
            (197.5, 212.5),
        ),
    ],
)
def test_ccf_sum_projection_preserves_counts_and_anatomical_axes(
    plane, expected, expected_shape, x_bounds, y_bounds
):
    volume = np.arange(24, dtype=np.int16).reshape(2, 3, 4)

    result = reduce_ccf_volume(
        volume,
        plane=plane,
        reduction=REDUCTION_PROJECTION,
        spacing_um=(10.0, 5.0, 4.0),
        origin_um=(50.0, 200.0, 1000.0),
    )

    assert result.data.shape == expected_shape
    np.testing.assert_array_equal(result.data, expected(volume))
    assert result.data.sum() == volume.sum()
    assert result.x_bounds_um == x_bounds
    assert result.y_bounds_um == y_bounds


def test_ccf_physical_slab_uses_spacing_and_sums_selected_planes():
    volume = np.ones((4, 2, 3), dtype=np.int16)
    volume[:, :, :] *= np.arange(1, 5, dtype=np.int16)[:, None, None]

    result = reduce_ccf_volume(
        volume,
        plane=CCF_PLANE_CORONAL,
        reduction=REDUCTION_SLICE,
        spacing_um=(10.0, 5.0, 2.0),
        origin_um=(100.0, 0.0, 0.0),
        slice_position_um=115.0,
        slab_thickness_um=20.0,
    )

    assert result.included_index_range == (1, 2)
    np.testing.assert_array_equal(result.data, volume[1:3].sum(axis=0))
    assert result.data.sum() == volume[1:3].sum()


def test_ccf_point_planes_and_physical_slab_are_oriented_like_volumes():
    coordinates = np.asarray(
        [
            [100.0, 200.0, 300.0],
            [120.0, 210.0, 330.0],
            [150.0, 220.0, 360.0],
        ]
    )

    coronal = project_ccf_points(coordinates, plane=CCF_PLANE_CORONAL)
    horizontal = project_ccf_points(coordinates, plane=CCF_PLANE_HORIZONTAL)
    sagittal = project_ccf_points(coordinates, plane=CCF_PLANE_SAGITTAL)
    slab = project_ccf_points(
        coordinates,
        plane=CCF_PLANE_CORONAL,
        reduction=REDUCTION_SLICE,
        slice_position_um=125.0,
        slab_thickness_um=20.0,
    )

    np.testing.assert_array_equal(coronal.points, coordinates[:, [2, 1]])
    np.testing.assert_array_equal(horizontal.points, coordinates[:, [2, 0]])
    np.testing.assert_array_equal(sagittal.points, coordinates[:, [0, 1]])
    assert slab.retained.tolist() == [False, True, False]


def test_comparison_state_preserves_rectangular_grid_and_cell_operations():
    state = ComparisonBoardState(rows=4, columns=4)
    first = state.add_cell(
        ComparisonCellSpec(
            title="Shaped",
            source_kind=SOURCE_FLATMAP_ARBOR_HEATMAP,
            flatmap_style="both_shaped",
            y_bins=256,
            x_bins=491,
            x_bounds=(-0.9, 0.9),
            y_bounds=(-0.47, 0.47),
            coordinate_provenance={
                "space": "flatmap",
                "style": "both_shaped",
            },
            opacity=0.0,
        )
    )
    for index in range(15):
        state.add_cell(ComparisonCellSpec(title=f"Cell {index + 2}"))
    assert len(state.cells) == 16
    with pytest.raises(ValueError, match="full"):
        state.add_cell()

    restored = ComparisonBoardState.from_state(state.to_state())
    restored_first = restored.cells[0]
    assert restored_first.x_bins == 491
    assert restored_first.y_bins == 256
    assert restored_first.x_bounds == (-0.9, 0.9)
    assert restored_first.coordinate_provenance["space"] == "flatmap"
    assert restored_first.opacity == 0.0

    restored.remove_cell(restored.cells[-1].cell_id)
    duplicate = restored.duplicate_cell(first.cell_id)
    assert duplicate.cell_id != first.cell_id
    assert duplicate.x_bins == 491
    assert restored.move_cell(duplicate.cell_id, 10) == 11
    restored.resize(1, 1)
    assert len(restored.cells) == 1


def test_every_layout_through_four_by_four_has_the_expected_capacity():
    for rows in range(1, 5):
        for columns in range(1, 5):
            state = ComparisonBoardState(rows=rows, columns=columns)
            for _index in range(rows * columns):
                state.add_cell()
            assert len(state.cells) == rows * columns
            assert state.capacity == rows * columns


def test_shared_intensity_groups_compatible_cells_and_excludes_overrides():
    geometry_a = ("flatmap", "both_shaped", (-0.9, 0.9), (-0.47, 0.47), 256, 491)
    geometry_b = ("flatmap", "both_square", (-1.0, 1.0), (-0.5, 0.5), 256, 512)

    maxima = shared_intensity_maxima(
        [
            (geometry_a, 7.0, None),
            (geometry_a, 12.0, None),
            (geometry_a, 100.0, 25.0),
            (geometry_b, 4.0, None),
            (None, 999.0, None),
        ]
    )

    assert maxima == {geometry_a: 12.0, geometry_b: 4.0}


def test_camera_groups_link_only_exact_compatible_opted_in_cells():
    flat_a = ComparisonCellSpec(cell_id="flat-a")
    flat_b = ComparisonCellSpec(cell_id="flat-b")
    flat_override = ComparisonCellSpec(cell_id="flat-override", camera_linked=False)
    ccf = ComparisonCellSpec(cell_id="ccf")
    keys = {
        "flat-a": ("flatmap", "both_shaped", (-0.9, 0.9), (-0.47, 0.47)),
        "flat-b": ("flatmap", "both_shaped", (-0.9, 0.9), (-0.47, 0.47)),
        "flat-override": (
            "flatmap",
            "both_shaped",
            (-0.9, 0.9),
            (-0.47, 0.47),
        ),
        "ccf": ("ccf", "allen_mouse_25um", "coronal", "projection"),
    }

    groups = compatible_camera_groups(
        [flat_a, flat_b, flat_override, ccf],
        keys,
    )

    assert groups == (("flat-a", "flat-b"), ("ccf",))


def test_tinted_heatmap_composition_uses_one_count_range():
    red = np.asarray([[0.0, 2.0], [4.0, 0.0]])
    green = np.asarray([[1.0, 0.0], [0.0, 8.0]])

    rgba, intensity_max = compose_tinted_heatmaps(
        {2: red, 7: green},
        {2: [1.0, 0.0, 0.0, 1.0], 7: [0.0, 1.0, 0.0, 1.0]},
        intensity_max=8.0,
        opacity=1.0,
    )

    assert intensity_max == 8.0
    assert rgba[1, 0, 0] == pytest.approx(0.5)
    assert rgba[1, 1, 1] == pytest.approx(1.0)


def test_comparison_export_provenance_is_complete_json():
    cell = ComparisonCellSpec(
        title="Run B",
        assignment_id="assignment-b",
        x_bins=491,
    )
    board = ComparisonBoardState(cells=[cell], reference_assignment_id="assignment-a")

    payload = comparison_provenance(
        board,
        cells=[
            {
                "cell_id": cell.cell_id,
                "assignment_id": "assignment-b",
                "assigned_neurons": 10,
                "omitted_or_unassigned_neurons": 2,
                "cluster_matches": [{"candidate_label": 7, "reference_label": 2}],
                "intensity_max": 42.0,
                "comparison_source_ids": ["source-1"],
            }
        ],
        source_parquet="source.parquet",
        source_signature={"size_bytes": 123, "mtime_ns": 456},
        reference_assignment={
            "assignment_id": "assignment-a",
            "saved_palette": {"2": [1.0, 0.0, 0.0, 1.0]},
        },
        membership_comparisons={
            "status": "ok",
            "alignment_key": "file_id",
            "comparisons": [],
        },
    )

    encoded = json.dumps(payload)
    assert '"version": 2' in encoded
    assert payload["board"]["cells"][0]["x_bins"] == 491
    assert payload["source_signature"]["size_bytes"] == 123
    assert payload["reference_assignment"]["assignment_id"] == "assignment-a"
    assert payload["rendered_cells"][0]["intensity_max"] == 42.0
    assert payload["membership_comparisons"]["alignment_key"] == "file_id"


def test_export_membership_metrics_deduplicate_assignments_and_keep_missing_ids():
    store = ClusterAssignmentStore()
    reference = _assignment(
        store,
        "Reference",
        {"f1": 1, "f2": 1, "f3": 2, "f4": 2},
    )
    candidate = _assignment(
        store,
        "Candidate",
        {"f1": 7, "f2": 7, "f3": 8, "f4": 8},
    )
    board = ComparisonBoardState(
        rows=2,
        columns=2,
        reference_assignment_id=reference.assignment_id,
        cells=[
            ComparisonCellSpec(
                cell_id="reference", assignment_id=reference.assignment_id
            ),
            ComparisonCellSpec(
                cell_id="candidate-a", assignment_id=candidate.assignment_id
            ),
            ComparisonCellSpec(
                cell_id="candidate-b", assignment_id=candidate.assignment_id
            ),
            ComparisonCellSpec(cell_id="missing", assignment_id="deleted-assignment"),
        ],
    )

    payload = comparison_membership_provenance(
        board,
        assignments=store.sets(),
    )

    assert payload["status"] == "ok"
    assert payload["alignment_key"] == "file_id"
    comparisons = payload["comparisons"]
    assert len(comparisons) == 2
    assert (
        comparisons[0]["candidate_assignment"]["assignment_id"]
        == candidate.assignment_id
    )
    assert comparisons[0]["source_cell_ids"] == ["candidate-a", "candidate-b"]
    assert comparisons[0]["metrics"]["adjusted_rand_index"] == pytest.approx(1.0)
    assert comparisons[1]["status"] == "candidate_unavailable"
    assert (
        comparisons[1]["candidate_assignment"]["assignment_id"] == "deleted-assignment"
    )
    json.dumps(payload)


def test_export_membership_metrics_explain_missing_reference():
    board = ComparisonBoardState(
        cells=[ComparisonCellSpec(cell_id="candidate", assignment_id="candidate")]
    )

    payload = comparison_membership_provenance(board, assignments=())

    assert payload["status"] == "no_reference"
    assert payload["comparisons"] == []
