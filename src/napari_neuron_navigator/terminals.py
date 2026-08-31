"""Terminus (childless node) detection for SWC morphologies.

A terminus is a node with no children. Restricting termini to a node type
selects that compartment's tips, so axon termini are the childless nodes typed
:data:`~napari_neuron_navigator.swc.NodeType.AXON`.

Two rules keep the result correct, and both come from the same principle: the
childless test must see the whole tree.

1. The node-type restriction applies only to which termini are *reported*, never
   to the child lookup. A node whose only child carries a different type is not
   childless.
2. Region or other spatial restrictions must be applied after detection, for the
   same reason: a node whose only child falls outside the region is not
   childless.

Node identifiers are unique only within a ``file_id``, so every operation here
is scoped per file. ``neuron_id`` is *not* a usable key -- it repeats across
subjects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

from .swc import NodeType, node_type_labels, normalize_node_types

if TYPE_CHECKING:
    import duckdb
    from numpy.typing import NDArray


#: Default compartment for terminus detection.
TERMINUS_NODE_TYPES: tuple[int, ...] = (NodeType.AXON,)

#: Columns returned by :func:`query_termini` when the source provides them.
TERMINUS_BASE_COLUMNS: tuple[str, ...] = (
    "file_id",
    "neuron_id",
    "node_id",
    "type",
    "x",
    "y",
    "z",
)

#: Optional columns included when present on the source.
TERMINUS_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "subject",
    "radius",
    "region_id",
    "region_name",
    "region_acronym",
    "x_flat_shaped",
    "y_flat_shaped",
    "x_flat_square",
    "y_flat_square",
    "depth_um",
)

#: Cap on how many skipped ``file_id`` values a coverage report retains.
MAX_REPORTED_FILE_IDS = 200

#: Cells per SQL statement. The child lookup's anti-join build side scales with
#: the nodes under consideration, so a whole large file in one statement runs out
#: of memory. Batching bounds that cost without changing the result.
DEFAULT_BATCH_CELLS = 400


def childless_mask(
    ids: NDArray[np.integer] | Sequence[int],
    parents: NDArray[np.integer] | Sequence[int],
) -> NDArray[np.bool_]:
    """Return a boolean mask of nodes that no other node claims as its parent.

    Makes no assumption about node ordering: identifiers need not be contiguous,
    start at 1, or exceed their parent's identifier.

    Parameters
    ----------
    ids : array-like of int
        Node identifiers for a single morphology.
    parents : array-like of int
        Parent identifier per node; negative values mark roots.

    Returns
    -------
    NDArray[np.bool_]
        ``True`` where the node has no children.

    Raises
    ------
    ValueError
        If ``ids`` and ``parents`` have different lengths.
    """
    id_array = np.asarray(ids)
    parent_array = np.asarray(parents)
    if id_array.shape != parent_array.shape:
        raise ValueError(
            "ids and parents must have the same shape; got "
            f"{id_array.shape} and {parent_array.shape}"
        )

    n_nodes = id_array.size
    if n_nodes == 0:
        return np.zeros(0, dtype=bool)

    # Resolve each parent reference to the index of the node it points at.
    order = np.argsort(id_array, kind="stable")
    sorted_ids = id_array[order]
    has_parent = parent_array >= 0
    positions = np.clip(np.searchsorted(sorted_ids, parent_array), 0, n_nodes - 1)
    resolved = has_parent & (sorted_ids[positions] == parent_array)

    child_counts = np.zeros(n_nodes, dtype=np.int64)
    if resolved.any():
        referenced = order[positions[resolved]]
        child_counts = np.bincount(referenced, minlength=n_nodes)
    return child_counts == 0


def terminus_mask(
    ids: NDArray[np.integer] | Sequence[int],
    parents: NDArray[np.integer] | Sequence[int],
    types: NDArray[np.integer] | Sequence[int],
    node_types: Iterable[int] | None = TERMINUS_NODE_TYPES,
) -> NDArray[np.bool_]:
    """Return a mask of childless nodes restricted to ``node_types``.

    The type restriction is applied only after the childless test, so a node
    whose only child carries a different type is never reported.

    Parameters
    ----------
    ids, parents : array-like of int
        Node and parent identifiers for a single morphology.
    types : array-like of int
        SWC node type per node.
    node_types : iterable of int, optional
        Types to report. ``None`` reports every childless node; an empty
        iterable reports none.
    """
    childless = childless_mask(ids, parents)
    selected = normalize_node_types(node_types)
    if selected is None:
        return childless
    if not selected:
        return np.zeros(childless.size, dtype=bool)
    type_array = np.asarray(types)
    return childless & np.isin(type_array, np.asarray(selected))


@dataclass
class TerminusCoverage:
    """Which cells contributed termini and which were skipped.

    A node-type restriction silently drops cells that never use those types, so
    this report exists to make that exclusion visible. In particular, cells
    whose neurites are all typed ``0`` (undefined) yield no axon termini.
    """

    node_types: tuple[int, ...] | None
    cells_requested: int = 0
    cells_with_selected_types: int = 0
    cells_without_selected_types: int = 0
    termini_found: int = 0
    childless_nodes_excluded: int = 0
    file_ids_without: list[str] = field(default_factory=list)
    file_ids_without_truncated: bool = False

    @property
    def has_exclusions(self) -> bool:
        """Return whether any cell or childless node was left out."""
        return bool(self.cells_without_selected_types or self.childless_nodes_excluded)

    def summary(self) -> str:
        """Return a one-line, user-facing description of the result."""
        types = (
            "all node types"
            if self.node_types is None
            else " / ".join(node_type_labels(self.node_types)) or "no node types"
        )
        text = (
            f"{self.termini_found:,} termini ({types}) in "
            f"{self.cells_with_selected_types:,} of {self.cells_requested:,} neurons"
        )
        if self.cells_without_selected_types:
            text += (
                f" — {self.cells_without_selected_types:,} neurons skipped "
                "(no nodes of the selected types)"
            )
        if self.childless_nodes_excluded:
            text += (
                f"; {self.childless_nodes_excluded:,} childless nodes of other "
                "types not counted"
            )
        return text


def _sql_identifier(name: str) -> str:
    """Quote a column name for safe inline use in SQL."""
    escaped = str(name).replace('"', '""')
    return f'"{escaped}"'


def resolve_terminus_columns(available: Iterable[str]) -> list[str]:
    """Return the terminus output columns supported by a source table."""
    present = {str(name) for name in available}
    missing = [name for name in TERMINUS_BASE_COLUMNS if name not in present]
    if missing:
        raise ValueError(
            "Terminus detection requires columns "
            f"{', '.join(TERMINUS_BASE_COLUMNS)}; missing {', '.join(missing)}."
        )
    columns = list(TERMINUS_BASE_COLUMNS)
    columns.extend(name for name in TERMINUS_OPTIONAL_COLUMNS if name in present)
    return columns


def _file_filter(
    file_ids: Sequence[str] | None,
    alias: str,
) -> tuple[str, list[object]]:
    """Return ``(sql, params)`` for an optional ``file_id IN (...)`` clause.

    The clause is qualified with ``alias``. An empty selection becomes ``FALSE``
    rather than ``IN ()``, which is not valid SQL.
    """
    if file_ids is None:
        return "", []
    if len(file_ids) == 0:
        return "FALSE", []
    placeholders = ", ".join(["?"] * len(file_ids))
    return (
        f"{alias}.file_id IN ({placeholders})",
        [str(value) for value in file_ids],
    )


def build_terminus_sql(
    source_sql: str,
    columns: Sequence[str],
    node_types: tuple[int, ...] | None,
    file_ids: Sequence[str] | None = None,
) -> tuple[str, list[object]]:
    """Build the anti-join that selects childless nodes of ``node_types``.

    The source is referenced twice with narrow projections rather than through a
    ``SELECT *`` CTE, so the scan feeding the child lookup reads only
    ``file_id``/``parent_id``. A wide CTE would materialize every column of
    every node and exhaust memory on large files.

    ``file_ids`` restricts which cells are considered and is applied to both
    sides identically, so it narrows the set of trees rather than truncating
    any tree. The node-type restriction is applied only to the outer select,
    keeping the child lookup over every node of each selected cell.

    Parameters
    ----------
    source_sql : str
        A table name or ``read_parquet(...)`` expression.
    columns : sequence of str
        Output columns, from :func:`resolve_terminus_columns`.
    node_types : tuple of int or None
        Types to report; ``None`` reports every childless node.
    file_ids : sequence of str, optional
        Restrict to these cells. ``None`` considers every cell.

    Returns
    -------
    tuple[str, list[object]]
        The query and its positional parameters.
    """
    outer_sql, outer_params = _file_filter(file_ids, "n")
    child_sql, child_params = _file_filter(file_ids, "c")
    select_list = ", ".join(f"n.{_sql_identifier(name)}" for name in columns)

    outer: list[str] = []
    params: list[object] = []
    if outer_sql:
        outer.append(outer_sql)
        params.extend(outer_params)
    if node_types is not None:
        placeholders = ", ".join(["?"] * len(node_types))
        outer.append(f"n.type IN ({placeholders})")
        params.extend(int(value) for value in node_types)

    # The child lookup is deliberately unrestricted by type: a node whose only
    # child carries a different type still has a child. The file restriction is
    # repeated here only so the scan can skip row groups; correlating on
    # ``c.file_id = n.file_id`` already confines it to the same cell.
    child_conditions = ["c.file_id = n.file_id", "c.parent_id = n.node_id"]
    if child_sql:
        child_conditions.append(child_sql)
    outer.append(
        "NOT EXISTS (SELECT 1 FROM {source} c WHERE {conds})".format(
            source=source_sql, conds=" AND ".join(child_conditions)
        )
    )
    params.extend(child_params)

    query = f"""
        SELECT {select_list}
        FROM {source_sql} n
        WHERE {" AND ".join(outer)}
        ORDER BY n.file_id, n.node_id
    """
    return query, params


def build_coverage_sql(
    source_sql: str,
    file_ids: Sequence[str] | None = None,
) -> tuple[str, list[object]]:
    """Build the per-cell childless-node-by-type tally for the coverage report.

    Returns the query and its positional parameters.
    """
    outer_sql, outer_params = _file_filter(file_ids, "n")
    child_sql, child_params = _file_filter(file_ids, "c")
    outer: list[str] = []
    params: list[object] = []
    if outer_sql:
        outer.append(outer_sql)
        params.extend(outer_params)

    child_conditions = ["c.file_id = n.file_id", "c.parent_id = n.node_id"]
    if child_sql:
        child_conditions.append(child_sql)
    outer.append(
        "NOT EXISTS (SELECT 1 FROM {source} c WHERE {conds})".format(
            source=source_sql, conds=" AND ".join(child_conditions)
        )
    )
    params.extend(child_params)

    query = f"""
        SELECT n.file_id, n.type, COUNT(*) AS n_childless
        FROM {source_sql} n
        WHERE {" AND ".join(outer)}
        GROUP BY n.file_id, n.type
    """
    return query, params


def summarize_coverage(
    tallies: Iterable[tuple[str, int, int]],
    node_types: tuple[int, ...] | None,
    requested_file_ids: Iterable[str] | None = None,
) -> TerminusCoverage:
    """Turn per-cell childless tallies into a :class:`TerminusCoverage`.

    Parameters
    ----------
    tallies : iterable of (file_id, type, count)
        Rows from :func:`build_coverage_sql`.
    node_types : tuple of int or None
        The types being reported.
    requested_file_ids : iterable of str, optional
        Cells the caller asked about. Cells listed here but absent from
        ``tallies`` are counted as skipped, which covers selections that match
        no rows at all.
    """
    selected = set(node_types) if node_types is not None else None
    per_cell: dict[str, list[int]] = {}
    for file_id, node_type, count in tallies:
        matched, other = per_cell.setdefault(str(file_id), [0, 0])
        if selected is None or int(node_type) in selected:
            matched += int(count)
        else:
            other += int(count)
        per_cell[str(file_id)] = [matched, other]

    if requested_file_ids is not None:
        for file_id in requested_file_ids:
            per_cell.setdefault(str(file_id), [0, 0])

    coverage = TerminusCoverage(node_types=node_types)
    coverage.cells_requested = len(per_cell)
    for file_id in sorted(per_cell):
        matched, other = per_cell[file_id]
        coverage.termini_found += matched
        coverage.childless_nodes_excluded += other
        if matched:
            coverage.cells_with_selected_types += 1
        else:
            coverage.cells_without_selected_types += 1
            if len(coverage.file_ids_without) < MAX_REPORTED_FILE_IDS:
                coverage.file_ids_without.append(file_id)
            else:
                coverage.file_ids_without_truncated = True
    return coverage


def list_source_file_ids(
    conn: duckdb.DuckDBPyConnection,
    source_sql: str,
) -> list[str]:
    """Return every ``file_id`` in a source, in order."""
    rows = conn.execute(
        f"SELECT DISTINCT file_id FROM {source_sql} ORDER BY file_id"
    ).fetchall()
    return [str(row[0]) for row in rows]


def query_termini(
    conn: duckdb.DuckDBPyConnection,
    source_sql: str,
    node_types: Iterable[int] | None = TERMINUS_NODE_TYPES,
    file_ids: Sequence[str] | None = None,
    batch_cells: int = DEFAULT_BATCH_CELLS,
    progress_callback=None,
):
    """Return ``(DataFrame, TerminusCoverage)`` of termini for a DuckDB source.

    Cells are processed in batches of ``batch_cells``. The child lookup is an
    anti-join whose build side is every node under consideration, so running a
    whole large file in one statement exhausts memory; batching keeps that build
    side proportional to the batch instead of the file. Each cell is wholly
    inside exactly one batch, so batching never splits a tree and the result is
    identical to an unbatched run.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Open connection.
    source_sql : str
        A table/view name or ``read_parquet('...')`` expression.
    node_types : iterable of int, optional
        Types to report. ``None`` reports every childless node; an empty
        sequence reports none.
    file_ids : sequence of str, optional
        Restrict to these cells. ``None`` uses every cell in the source.
    batch_cells : int, optional
        Cells per statement. Values below 1 disable batching and run a single
        statement, which is only safe for small sources.
    progress_callback : callable, optional
        Called as ``(cells_done, cells_total)`` after each batch.
    """
    import pandas as pd

    selected = normalize_node_types(node_types)
    relation = conn.execute(f"SELECT * FROM {source_sql} LIMIT 0")
    available = [str(description[0]) for description in relation.description]
    columns = resolve_terminus_columns(available)

    requested: list[str] | None = None
    if file_ids is not None:
        requested = list(dict.fromkeys(str(value) for value in file_ids))
        if not requested:
            return (
                pd.DataFrame(columns=columns),
                TerminusCoverage(node_types=selected),
            )

    empty_selection = selected is not None and not selected

    if batch_cells is None or batch_cells < 1:
        batches: list[Sequence[str] | None] = [requested]
        total = len(requested) if requested is not None else 0
    else:
        if requested is not None:
            cells = requested
        else:
            cells = list_source_file_ids(conn, source_sql)
        if not cells:
            return (
                pd.DataFrame(columns=columns),
                TerminusCoverage(node_types=selected),
            )
        total = len(cells)
        batches = [
            cells[start : start + batch_cells]
            for start in range(0, len(cells), batch_cells)
        ]

    frames = []
    tallies: list[tuple[str, int, int]] = []
    done = 0
    for batch in batches:
        if not empty_selection:
            query, params = build_terminus_sql(source_sql, columns, selected, batch)
            frames.append(conn.execute(query, params).fetchdf())

        coverage_query, coverage_params = build_coverage_sql(source_sql, batch)
        tallies.extend(conn.execute(coverage_query, coverage_params).fetchall())

        done += len(batch) if batch is not None else total
        if callable(progress_callback):
            progress_callback(min(done, total), total)

    if frames:
        frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    else:
        frame = pd.DataFrame(columns=columns)

    coverage = summarize_coverage(tallies, selected, requested)
    return frame, coverage
