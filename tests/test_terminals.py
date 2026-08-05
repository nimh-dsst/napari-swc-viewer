"""Tests for terminus (childless node) detection."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from napari_swc_viewer.db import NeuronDatabase
from napari_swc_viewer.swc import NodeType
from napari_swc_viewer.terminals import (
    TerminusCoverage,
    build_terminus_sql,
    childless_mask,
    list_source_file_ids,
    query_termini,
    resolve_terminus_columns,
    summarize_coverage,
    terminus_mask,
)


# --- childless_mask: pure topology -----------------------------------------


def test_straight_chain_has_exactly_one_terminus() -> None:
    ids = np.array([1, 2, 3, 4])
    parents = np.array([-1, 1, 2, 3])
    assert childless_mask(ids, parents).tolist() == [False, False, False, True]


def test_y_branch_has_two_termini() -> None:
    #   1 -> 2 -> 3
    #          -> 4
    ids = np.array([1, 2, 3, 4])
    parents = np.array([-1, 1, 2, 2])
    assert childless_mask(ids, parents).tolist() == [False, False, True, True]


def test_non_contiguous_node_ids_are_resolved() -> None:
    # Identifiers skip values and do not start at 1.
    ids = np.array([100, 250, 375, 900])
    parents = np.array([-1, 100, 250, 250])
    assert childless_mask(ids, parents).tolist() == [False, False, True, True]


def test_parent_id_greater_than_node_id_is_resolved() -> None:
    # Row order and identifier order both disagree with tree order.
    ids = np.array([4, 3, 2, 1])
    parents = np.array([3, 2, 1, -1])
    # Node 4 is the tip; node 1 is the root.
    assert childless_mask(ids, parents).tolist() == [True, False, False, False]


def test_single_isolated_node_is_childless() -> None:
    assert childless_mask(np.array([1]), np.array([-1])).tolist() == [True]


def test_empty_input_returns_empty_mask() -> None:
    assert childless_mask(np.array([]), np.array([])).tolist() == []


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="same shape"):
        childless_mask(np.array([1, 2]), np.array([-1]))


def test_dangling_parent_reference_does_not_mark_a_child() -> None:
    # Node 2 points at a parent that is absent, so node 1 keeps no child.
    ids = np.array([1, 2])
    parents = np.array([-1, 99])
    assert childless_mask(ids, parents).tolist() == [True, True]


# --- terminus_mask: type restriction applied after the childless test -------


def test_terminus_mask_restricts_reported_types_only() -> None:
    #  soma 1 -> axon 2 -> axon 3 (tip)
    #         -> dend 4 -> dend 5 (tip)
    ids = np.array([1, 2, 3, 4, 5])
    parents = np.array([-1, 1, 2, 1, 4])
    types = np.array(
        [
            NodeType.SOMA,
            NodeType.AXON,
            NodeType.AXON,
            NodeType.BASAL_DENDRITE,
            NodeType.BASAL_DENDRITE,
        ]
    )

    axon = terminus_mask(ids, parents, types, (NodeType.AXON,))
    assert axon.tolist() == [False, False, True, False, False]

    dend = terminus_mask(ids, parents, types, (NodeType.BASAL_DENDRITE,))
    assert dend.tolist() == [False, False, False, False, True]

    # None means "every childless node", regardless of type.
    every = terminus_mask(ids, parents, types, None)
    assert every.tolist() == [False, False, True, False, True]

    # An explicit empty selection reports nothing.
    none = terminus_mask(ids, parents, types, ())
    assert none.tolist() == [False] * 5


def test_axon_node_whose_only_child_is_a_dendrite_is_not_a_terminus() -> None:
    """Guards the trap of filtering by type before the childless test."""
    # axon 2 has exactly one child, and that child is typed as dendrite.
    ids = np.array([1, 2, 3])
    parents = np.array([-1, 1, 2])
    types = np.array([NodeType.SOMA, NodeType.AXON, NodeType.BASAL_DENDRITE])

    axon = terminus_mask(ids, parents, types, (NodeType.AXON,))
    assert axon.tolist() == [False, False, False], (
        "node 2 has a child, so it is not a terminus even though the child "
        "carries a different type"
    )


# --- coverage reporting -----------------------------------------------------


def test_summarize_coverage_counts_skipped_cells() -> None:
    tallies = [
        ("axon.swc", NodeType.AXON, 5),
        ("mixed.swc", NodeType.AXON, 3),
        ("mixed.swc", NodeType.BASAL_DENDRITE, 2),
        ("undef.swc", NodeType.UNDEFINED, 7),
    ]
    coverage = summarize_coverage(tallies, (NodeType.AXON,))

    assert coverage.cells_requested == 3
    assert coverage.cells_with_selected_types == 2
    assert coverage.cells_without_selected_types == 1
    assert coverage.termini_found == 8
    # 2 dendrite tips + 7 undefined tips are childless but not reported.
    assert coverage.childless_nodes_excluded == 9
    assert coverage.file_ids_without == ["undef.swc"]
    assert coverage.has_exclusions is True


def test_summarize_coverage_counts_requested_cells_with_no_rows() -> None:
    coverage = summarize_coverage(
        [("a.swc", NodeType.AXON, 2)],
        (NodeType.AXON,),
        requested_file_ids=["a.swc", "absent.swc"],
    )
    assert coverage.cells_requested == 2
    assert coverage.cells_without_selected_types == 1
    assert coverage.file_ids_without == ["absent.swc"]


def test_coverage_summary_mentions_skipped_neurons() -> None:
    coverage = TerminusCoverage(
        node_types=(NodeType.AXON,),
        cells_requested=10,
        cells_with_selected_types=7,
        cells_without_selected_types=3,
        termini_found=1234,
        childless_nodes_excluded=56,
    )
    text = coverage.summary()
    assert "1,234 termini" in text
    assert "Axon" in text
    assert "7 of 10 neurons" in text
    assert "3 neurons skipped" in text
    assert "56 childless nodes" in text


def test_coverage_summary_has_no_exclusion_clause_when_complete() -> None:
    coverage = TerminusCoverage(
        node_types=(NodeType.AXON,),
        cells_requested=4,
        cells_with_selected_types=4,
        termini_found=10,
    )
    assert "skipped" not in coverage.summary()
    assert coverage.has_exclusions is False


def test_file_ids_without_is_capped() -> None:
    tallies = [(f"cell{i:04d}.swc", NodeType.UNDEFINED, 1) for i in range(250)]
    coverage = summarize_coverage(tallies, (NodeType.AXON,))
    assert coverage.cells_without_selected_types == 250
    assert len(coverage.file_ids_without) == 200
    assert coverage.file_ids_without_truncated is True


# --- DuckDB path ------------------------------------------------------------


def _write_source_parquet(path: Path) -> None:
    """Two axon-labelled cells sharing node ids, plus an undefined-only cell."""
    frame = pd.DataFrame(
        {
            "file_id": [
                # a.swc: soma -> axon chain that forks, plus a dendrite branch
                "a.swc", "a.swc", "a.swc", "a.swc", "a.swc", "a.swc",
                # b.swc: reuses node ids 1..3, one axon tip
                "b.swc", "b.swc", "b.swc",
                # u.swc: every neurite typed 0
                "u.swc", "u.swc", "u.swc",
            ],
            "node_id": [1, 2, 3, 4, 5, 6, 1, 2, 3, 1, 2, 3],
            "parent_id": [-1, 1, 2, 2, 1, 5, -1, 1, 2, -1, 1, 2],
            "type": [
                NodeType.SOMA,
                NodeType.AXON,
                NodeType.AXON,
                NodeType.AXON,
                NodeType.BASAL_DENDRITE,
                NodeType.BASAL_DENDRITE,
                NodeType.SOMA,
                NodeType.AXON,
                NodeType.AXON,
                NodeType.SOMA,
                NodeType.UNDEFINED,
                NodeType.UNDEFINED,
            ],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "y": [0.0] * 12,
            "z": [0.0] * 12,
            "region_acronym": ["R1", "R1", "R2", "R1", "R1", "R1",
                               "R1", "R1", "R2", "R1", "R1", "R1"],
            # neuron_id deliberately repeats across files: it is not a key.
            "neuron_id": ["n1"] * 6 + ["n1"] * 3 + ["n2"] * 3,
        }
    )
    frame.to_parquet(path, index=False)


def test_query_termini_finds_axon_tips_only(tmp_path) -> None:
    path = tmp_path / "neurons.parquet"
    _write_source_parquet(path)
    conn = duckdb.connect()
    try:
        frame, coverage = query_termini(
            conn, f"read_parquet('{path.as_posix()}')"
        )
    finally:
        conn.close()

    # a.swc axon tips are nodes 3 and 4; b.swc axon tip is node 3.
    assert list(zip(frame["file_id"], frame["node_id"])) == [
        ("a.swc", 3),
        ("a.swc", 4),
        ("b.swc", 3),
    ]
    assert set(frame["type"]) == {NodeType.AXON}
    assert coverage.termini_found == 3
    assert coverage.cells_with_selected_types == 2
    assert coverage.cells_without_selected_types == 1
    assert coverage.file_ids_without == ["u.swc"]
    # a.swc dendrite tip (node 6) and u.swc undefined tip (node 3).
    assert coverage.childless_nodes_excluded == 2


def test_duplicate_node_ids_across_files_stay_separated(tmp_path) -> None:
    """Guards against grouping on neuron_id, which repeats across cells."""
    path = tmp_path / "neurons.parquet"
    _write_source_parquet(path)
    conn = duckdb.connect()
    try:
        frame, _ = query_termini(conn, f"read_parquet('{path.as_posix()}')")
    finally:
        conn.close()

    # a.swc and b.swc both share neuron_id "n1" and node ids 1..3. b.swc's
    # node 3 is a real tip and must not be masked by a.swc's node 3 having a
    # child in a different cell.
    assert ("b.swc", 3) in list(zip(frame["file_id"], frame["node_id"]))


def test_region_filter_after_detection_does_not_invent_termini(tmp_path) -> None:
    """Guards the trap of restricting the source before the childless test."""
    path = tmp_path / "neurons.parquet"
    _write_source_parquet(path)
    source = f"read_parquet('{path.as_posix()}')"
    conn = duckdb.connect()
    try:
        frame, _ = query_termini(conn, source)
        # Correct: detect first, then narrow to a region.
        in_region = frame[frame["region_acronym"] == "R1"]

        # Wrong: narrowing the source first strands b.swc node 2, whose only
        # child (node 3) lives in R2, and reports it as a terminus.
        naive = conn.execute(
            f"""
            WITH src AS (SELECT * FROM {source} WHERE region_acronym = 'R1')
            SELECT file_id, node_id FROM src n
            WHERE n.type = {int(NodeType.AXON)}
              AND NOT EXISTS (
                SELECT 1 FROM src c
                WHERE c.file_id = n.file_id AND c.parent_id = n.node_id
              )
            ORDER BY file_id, node_id
            """
        ).fetchall()
    finally:
        conn.close()

    found = list(zip(in_region["file_id"], in_region["node_id"]))
    assert found == [("a.swc", 4)]
    # Sanity: the naive query is expected to be wrong.
    assert ("b.swc", 2) in naive
    assert ("b.swc", 2) not in found


def test_query_termini_honours_file_id_selection(tmp_path) -> None:
    path = tmp_path / "neurons.parquet"
    _write_source_parquet(path)
    conn = duckdb.connect()
    try:
        frame, coverage = query_termini(
            conn, f"read_parquet('{path.as_posix()}')", file_ids=["b.swc"]
        )
    finally:
        conn.close()

    assert frame["file_id"].tolist() == ["b.swc"]
    assert coverage.cells_requested == 1
    assert coverage.termini_found == 1


def test_query_termini_empty_file_ids_returns_nothing(tmp_path) -> None:
    path = tmp_path / "neurons.parquet"
    _write_source_parquet(path)
    conn = duckdb.connect()
    try:
        frame, coverage = query_termini(
            conn, f"read_parquet('{path.as_posix()}')", file_ids=[]
        )
    finally:
        conn.close()

    assert frame.empty
    assert coverage.cells_requested == 0
    assert coverage.termini_found == 0


def test_query_termini_all_node_types(tmp_path) -> None:
    path = tmp_path / "neurons.parquet"
    _write_source_parquet(path)
    conn = duckdb.connect()
    try:
        frame, coverage = query_termini(
            conn, f"read_parquet('{path.as_posix()}')", node_types=None
        )
    finally:
        conn.close()

    assert list(zip(frame["file_id"], frame["node_id"])) == [
        ("a.swc", 3),
        ("a.swc", 4),
        ("a.swc", 6),
        ("b.swc", 3),
        ("u.swc", 3),
    ]
    assert coverage.cells_without_selected_types == 0
    assert coverage.childless_nodes_excluded == 0


def test_resolve_terminus_columns_requires_base_columns() -> None:
    with pytest.raises(ValueError, match="missing"):
        resolve_terminus_columns(["file_id", "node_id"])


def test_resolve_terminus_columns_includes_optional_when_present() -> None:
    columns = resolve_terminus_columns(
        [
            "file_id", "neuron_id", "node_id", "type", "x", "y", "z",
            "depth_um", "region_acronym", "unrelated",
        ]
    )
    assert "depth_um" in columns
    assert "region_acronym" in columns
    assert "unrelated" not in columns


def test_database_get_termini_matches_query_termini(tmp_path) -> None:
    path = tmp_path / "neurons.parquet"
    _write_source_parquet(path)
    db = NeuronDatabase(path)
    try:
        frame, coverage = db.get_termini(["a.swc", "b.swc", "u.swc"])
    finally:
        db.close()

    assert frame["node_id"].tolist() == [3, 4, 3]
    assert coverage.termini_found == 3
    assert coverage.file_ids_without == ["u.swc"]


def test_database_get_termini_defaults_to_all_files(tmp_path) -> None:
    path = tmp_path / "neurons.parquet"
    _write_source_parquet(path)
    db = NeuronDatabase(path)
    try:
        frame, coverage = db.get_termini()
    finally:
        db.close()

    assert coverage.cells_requested == 3
    assert len(frame) == 3


# --- batching ---------------------------------------------------------------


def test_batching_matches_a_single_statement(tmp_path) -> None:
    """Every cell lands wholly in one batch, so results must be identical."""
    path = tmp_path / "neurons.parquet"
    _write_source_parquet(path)
    source = f"read_parquet('{path.as_posix()}')"
    conn = duckdb.connect()
    try:
        whole, whole_cov = query_termini(conn, source, batch_cells=0)
        # One cell per statement: the most aggressive split possible.
        split, split_cov = query_termini(conn, source, batch_cells=1)
    finally:
        conn.close()

    assert list(zip(split["file_id"], split["node_id"])) == list(
        zip(whole["file_id"], whole["node_id"])
    )
    assert split_cov.termini_found == whole_cov.termini_found
    assert split_cov.cells_requested == whole_cov.cells_requested
    assert split_cov.cells_without_selected_types == (
        whole_cov.cells_without_selected_types
    )
    assert split_cov.childless_nodes_excluded == (
        whole_cov.childless_nodes_excluded
    )
    assert split_cov.file_ids_without == whole_cov.file_ids_without


def test_batching_reports_progress_over_every_cell(tmp_path) -> None:
    path = tmp_path / "neurons.parquet"
    _write_source_parquet(path)
    seen: list[tuple[int, int]] = []
    conn = duckdb.connect()
    try:
        query_termini(
            conn,
            f"read_parquet('{path.as_posix()}')",
            batch_cells=2,
            progress_callback=lambda done, total: seen.append((done, total)),
        )
    finally:
        conn.close()

    assert seen[-1] == (3, 3)
    assert [done for done, _ in seen] == sorted(done for done, _ in seen)


def test_batching_respects_an_explicit_file_id_selection(tmp_path) -> None:
    path = tmp_path / "neurons.parquet"
    _write_source_parquet(path)
    conn = duckdb.connect()
    try:
        frame, coverage = query_termini(
            conn,
            f"read_parquet('{path.as_posix()}')",
            file_ids=["a.swc", "u.swc"],
            batch_cells=1,
        )
    finally:
        conn.close()

    assert list(zip(frame["file_id"], frame["node_id"])) == [
        ("a.swc", 3),
        ("a.swc", 4),
    ]
    assert coverage.cells_requested == 2
    assert coverage.file_ids_without == ["u.swc"]


def test_list_source_file_ids_returns_every_cell(tmp_path) -> None:
    path = tmp_path / "neurons.parquet"
    _write_source_parquet(path)
    conn = duckdb.connect()
    try:
        found = list_source_file_ids(conn, f"read_parquet('{path.as_posix()}')")
    finally:
        conn.close()
    assert found == ["a.swc", "b.swc", "u.swc"]


def test_terminus_sql_does_not_widen_the_child_lookup(tmp_path) -> None:
    """The child scan must not select every column of every node."""
    query, params = build_terminus_sql(
        "src", ["file_id", "node_id", "type", "x", "y", "z"], (2,), ["a.swc"]
    )
    # A SELECT * over the source would materialize all columns for all nodes.
    assert "SELECT *" not in query
    assert "SELECT 1 FROM src c" in query
    # File params bind twice (outer and child scan), node types once between.
    assert params == ["a.swc", 2, "a.swc"]


def test_empty_source_returns_nothing_without_invalid_sql(tmp_path) -> None:
    """An empty source must not generate ``file_id IN ()``."""
    path = tmp_path / "empty.parquet"
    pd.DataFrame(
        {
            "file_id": pd.Series([], dtype="object"),
            "node_id": pd.Series([], dtype="int32"),
            "parent_id": pd.Series([], dtype="int32"),
            "type": pd.Series([], dtype="int32"),
            "x": pd.Series([], dtype="float64"),
            "y": pd.Series([], dtype="float64"),
            "z": pd.Series([], dtype="float64"),
            "neuron_id": pd.Series([], dtype="object"),
        }
    ).to_parquet(path, index=False)
    conn = duckdb.connect()
    try:
        frame, coverage = query_termini(conn, f"read_parquet('{path.as_posix()}')")
    finally:
        conn.close()

    assert frame.empty
    assert coverage.cells_requested == 0
    assert coverage.termini_found == 0


def test_empty_file_id_batch_compiles_to_a_false_predicate() -> None:
    query, params = build_terminus_sql("src", ["file_id", "node_id"], (2,), [])
    assert "IN ()" not in query
    assert "FALSE" in query
    assert params == [2]
