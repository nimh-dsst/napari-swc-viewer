"""Lookup-based isocortex flatmap projection for neuron node tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

COORDINATE_MODE_MICRONS = "microns"
COORDINATE_MODE_VOXELS = "voxels"
VALID_COORDINATE_MODES = {COORDINATE_MODE_MICRONS, COORDINATE_MODE_VOXELS}
DEFAULT_CCF_RESOLUTION_UM = 10.0
DEFAULT_CCFV3_MIRROR_MIDLINE_UM = 5695.0
FLATMAP_LOOKUP_DIRECT = "direct"
FLATMAP_LOOKUP_MIRRORED_DEPTH = "mirrored_depth"
FLATMAP_LOOKUP_MIRRORED = "mirrored"
FLATMAP_LOOKUP_UNMAPPED = "unmapped"

REQUIRED_NODE_COLUMNS = ("file_id", "node_id", "parent_id", "type", "x", "y", "z")


@dataclass(frozen=True)
class ProjectedSegments:
    """Projected parent-child line segments and their source identifiers."""

    data: np.ndarray
    file_ids: list[object]
    source_node_ids: list[object]
    target_node_ids: list[object]


@dataclass(frozen=True)
class ProjectionSummary:
    """Counts describing one flatmap projection run."""

    total_nodes: int
    valid_nodes: int
    out_of_bounds_nodes: int
    invalid_flatmap_nodes: int
    invalid_depth_nodes: int
    missing_input_nodes: int
    rendered_segments: int
    total_traces: int
    traces_with_partial_projection: int
    direct_lookup_nodes: int = 0
    mirrored_lookup_nodes: int = 0
    unmapped_lookup_nodes: int = 0
    mirrored_depth_lookup_nodes: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-safe dictionary."""
        return {
            "total_nodes": int(self.total_nodes),
            "valid_nodes": int(self.valid_nodes),
            "out_of_bounds_nodes": int(self.out_of_bounds_nodes),
            "invalid_flatmap_nodes": int(self.invalid_flatmap_nodes),
            "invalid_depth_nodes": int(self.invalid_depth_nodes),
            "missing_input_nodes": int(self.missing_input_nodes),
            "rendered_segments": int(self.rendered_segments),
            "total_traces": int(self.total_traces),
            "traces_with_partial_projection": int(self.traces_with_partial_projection),
            "direct_lookup_nodes": int(self.direct_lookup_nodes),
            "mirrored_depth_lookup_nodes": int(self.mirrored_depth_lookup_nodes),
            "mirrored_lookup_nodes": int(self.mirrored_lookup_nodes),
            "unmapped_lookup_nodes": int(self.unmapped_lookup_nodes),
        }


@dataclass(frozen=True)
class FlatmapProjectionResult:
    """Projected node table, valid line segments, and summary counts."""

    projected_nodes: pd.DataFrame
    segments: ProjectedSegments
    summary: ProjectionSummary


def _require_columns(nodes: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in nodes.columns]
    if missing:
        raise ValueError(f"Neuron table is missing required column(s): {missing}")


def coordinates_to_voxel_indices(
    coords_xyz: np.ndarray,
    *,
    coordinate_mode: str = COORDINATE_MODE_MICRONS,
    resolution_um: float = DEFAULT_CCF_RESOLUTION_UM,
    space_directions: np.ndarray | None = None,
    space_origin: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert XYZ coordinates to nearest lookup voxel indices.

    Returns
    -------
    tuple
        ``(voxel_indices, finite_mask)`` where invalid/non-finite coordinates
        have voxel index ``-1``.
    """
    if coordinate_mode not in VALID_COORDINATE_MODES:
        raise ValueError(
            f"coordinate_mode must be one of {sorted(VALID_COORDINATE_MODES)}; "
            f"got {coordinate_mode!r}."
        )
    if resolution_um <= 0:
        raise ValueError("resolution_um must be positive.")

    coords = np.asarray(coords_xyz, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(1, 3)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected an (N, 3) coordinate array; got {coords.shape}.")

    finite_mask = np.all(np.isfinite(coords), axis=1)
    voxels = np.full(coords.shape, -1, dtype=np.int64)
    if not finite_mask.any():
        return voxels, finite_mask

    values = coords[finite_mask]
    if coordinate_mode == COORDINATE_MODE_MICRONS:
        if space_directions is not None and space_origin is not None:
            directions = np.asarray(space_directions, dtype=float)
            origin = np.asarray(space_origin, dtype=float).reshape(-1)
            if directions.shape != (3, 3):
                raise ValueError(
                    "space_directions must have shape (3, 3) when provided; "
                    f"got {directions.shape}."
                )
            if origin.shape != (3,):
                raise ValueError(
                    "space_origin must have shape (3,) when provided; "
                    f"got {origin.shape}."
                )
            values = (values - origin) @ np.linalg.inv(directions)
        else:
            values = values / float(resolution_um)
    voxels[finite_mask] = np.floor(values + 0.5).astype(np.int64)
    return voxels, finite_mask


def _copy_optional_column(
    nodes: pd.DataFrame,
    column: str,
    default: object,
) -> pd.Series:
    """Return an input column or a default-filled Series."""
    if column in nodes.columns:
        return nodes[column].reset_index(drop=True)
    return pd.Series([default] * len(nodes), index=range(len(nodes)))


def resolve_flatmap_mirror_midline(
    *,
    coordinate_mode: str,
    flatmap_shape: tuple[int, ...],
    mirror_coord_axis: int = 2,
    mirror_midline: float | None = None,
) -> float:
    """Return the mirror midline for a flatmap/depth lookup grid."""
    if mirror_coord_axis not in (0, 1, 2):
        raise ValueError("mirror_coord_axis must be 0, 1, or 2.")
    if coordinate_mode not in VALID_COORDINATE_MODES:
        raise ValueError(
            f"coordinate_mode must be one of {sorted(VALID_COORDINATE_MODES)}; "
            f"got {coordinate_mode!r}."
        )
    if mirror_midline is not None:
        return float(mirror_midline)
    if coordinate_mode == COORDINATE_MODE_VOXELS:
        if len(flatmap_shape) < 3:
            raise ValueError(
                f"flatmap_shape must have at least 3 axes; got {flatmap_shape}."
            )
        return (float(flatmap_shape[mirror_coord_axis]) - 1.0) / 2.0
    return DEFAULT_CCFV3_MIRROR_MIDLINE_UM


def _mirror_node_coordinates(
    nodes: pd.DataFrame,
    *,
    mirror_coord_axis: int,
    mirror_midline: float,
) -> pd.DataFrame:
    mirrored = nodes.copy()
    coord_column = ("x", "y", "z")[mirror_coord_axis]
    values = pd.to_numeric(mirrored[coord_column], errors="coerce").to_numpy(
        dtype=float
    )
    mirrored.loc[:, coord_column] = (2.0 * float(mirror_midline)) - values
    return mirrored


def _project_neuron_nodes_to_flatmap_direct(
    nodes: pd.DataFrame,
    flatmap_volume: np.ndarray,
    depth_volume: np.ndarray,
    *,
    flatmap_style: str = "",
    coordinate_mode: str = COORDINATE_MODE_MICRONS,
    invalid_zero_sentinel: bool = False,
    invalid_negative_one_sentinel: bool = True,
    resolution_um: float = DEFAULT_CCF_RESOLUTION_UM,
    space_directions: np.ndarray | None = None,
    space_origin: np.ndarray | None = None,
) -> pd.DataFrame:
    """Project neuron nodes directly into flatmap coordinates."""
    _require_columns(nodes, REQUIRED_NODE_COLUMNS)
    flatmap = np.asarray(flatmap_volume)
    depth = np.asarray(depth_volume)
    if flatmap.ndim != 4 or flatmap.shape[-1] != 2:
        raise ValueError(
            f"flatmap_volume must have shape (nx, ny, nz, 2); got {flatmap.shape}."
        )
    if depth.shape != flatmap.shape[:3]:
        raise ValueError(
            "depth_volume shape must match the flatmap lookup grid; "
            f"got depth {depth.shape} and flatmap grid {flatmap.shape[:3]}."
        )

    source = nodes.reset_index(drop=True)
    n_nodes = len(source)
    coords = source[["x", "y", "z"]].to_numpy(dtype=float)
    voxels, finite_coords = coordinates_to_voxel_indices(
        coords,
        coordinate_mode=coordinate_mode,
        resolution_um=resolution_um,
        space_directions=space_directions,
        space_origin=space_origin,
    )

    spatial_shape = np.asarray(flatmap.shape[:3], dtype=np.int64)
    in_bounds = (
        finite_coords
        & np.all(voxels >= 0, axis=1)
        & np.all(voxels < spatial_shape, axis=1)
    )

    flat_xy = np.full((n_nodes, 2), np.nan, dtype=np.float64)
    depth_um = np.full(n_nodes, np.nan, dtype=np.float64)
    if in_bounds.any():
        idx = voxels[in_bounds]
        flat_xy[in_bounds] = flatmap[idx[:, 0], idx[:, 1], idx[:, 2]]
        depth_um[in_bounds] = depth[idx[:, 0], idx[:, 1], idx[:, 2]]

    flatmap_valid = np.all(np.isfinite(flat_xy), axis=1)
    if invalid_negative_one_sentinel:
        flatmap_valid &= ~((flat_xy[:, 0] == -1.0) & (flat_xy[:, 1] == -1.0))
    if invalid_zero_sentinel:
        flatmap_valid &= ~((flat_xy[:, 0] == 0.0) & (flat_xy[:, 1] == 0.0))
    depth_valid = np.isfinite(depth_um) & (depth_um >= 0.0)

    valid = in_bounds & flatmap_valid & depth_valid
    invalid_reason = np.full(n_nodes, "", dtype=object)
    invalid_reason[~finite_coords] = "missing_input"
    invalid_reason[finite_coords & ~in_bounds] = "out_of_bounds"
    invalid_flatmap = in_bounds & ~flatmap_valid
    invalid_reason[invalid_flatmap] = "invalid_flatmap"
    invalid_depth = in_bounds & flatmap_valid & ~depth_valid
    invalid_reason[invalid_depth] = "invalid_depth"

    out = pd.DataFrame(
        {
            "file_id": source["file_id"].reset_index(drop=True),
            "neuron_id": _copy_optional_column(source, "neuron_id", ""),
            "subject": _copy_optional_column(source, "subject", ""),
            "node_id": source["node_id"].reset_index(drop=True),
            "parent_id": source["parent_id"].reset_index(drop=True),
            "type": source["type"].reset_index(drop=True),
            "x_um": coords[:, 0],
            "y_um": coords[:, 1],
            "z_um": coords[:, 2],
            "voxel_i": voxels[:, 0],
            "voxel_j": voxels[:, 1],
            "voxel_k": voxels[:, 2],
            "x_flat": flat_xy[:, 0],
            "y_flat": flat_xy[:, 1],
            "depth_um": depth_um,
            "flatmap_valid": in_bounds & flatmap_valid,
            "depth_valid": depth_valid,
            "valid": valid,
            "invalid_reason": invalid_reason,
            "region_id": _copy_optional_column(source, "region_id", pd.NA),
            "region_acronym": _copy_optional_column(source, "region_acronym", ""),
            "flatmap_style": flatmap_style,
            "coordinate_mode": coordinate_mode,
        }
    )
    return out


def project_neuron_nodes_to_flatmap(
    nodes: pd.DataFrame,
    flatmap_volume: np.ndarray,
    depth_volume: np.ndarray,
    *,
    flatmap_style: str = "",
    coordinate_mode: str = COORDINATE_MODE_MICRONS,
    invalid_zero_sentinel: bool = False,
    invalid_negative_one_sentinel: bool = True,
    resolution_um: float = DEFAULT_CCF_RESOLUTION_UM,
    space_directions: np.ndarray | None = None,
    space_origin: np.ndarray | None = None,
    mirror_fallback: bool = False,
    mirror_coord_axis: int = 2,
    mirror_midline: float | None = None,
) -> pd.DataFrame:
    """Project neuron nodes into flatmap coordinates with validity metadata."""
    direct = _project_neuron_nodes_to_flatmap_direct(
        nodes,
        flatmap_volume,
        depth_volume,
        flatmap_style=flatmap_style,
        coordinate_mode=coordinate_mode,
        invalid_zero_sentinel=invalid_zero_sentinel,
        invalid_negative_one_sentinel=invalid_negative_one_sentinel,
        resolution_um=resolution_um,
        space_directions=space_directions,
        space_origin=space_origin,
    ).reset_index(drop=True)

    selected = direct.copy()
    lookup_mode = np.full(len(selected), FLATMAP_LOOKUP_UNMAPPED, dtype=object)
    direct_valid = selected["valid"].to_numpy(dtype=bool)
    lookup_mode[direct_valid] = FLATMAP_LOOKUP_DIRECT

    if mirror_fallback and (~direct_valid).any():
        flatmap = np.asarray(flatmap_volume)
        resolved_midline = resolve_flatmap_mirror_midline(
            coordinate_mode=coordinate_mode,
            flatmap_shape=tuple(int(size) for size in flatmap.shape[:3]),
            mirror_coord_axis=mirror_coord_axis,
            mirror_midline=mirror_midline,
        )
        retry_positions = np.flatnonzero(~direct_valid)
        retry_nodes = (
            nodes.reset_index(drop=True).iloc[retry_positions].reset_index(drop=True)
        )
        mirrored_nodes = _mirror_node_coordinates(
            retry_nodes,
            mirror_coord_axis=mirror_coord_axis,
            mirror_midline=resolved_midline,
        )
        mirrored = _project_neuron_nodes_to_flatmap_direct(
            mirrored_nodes,
            flatmap_volume,
            depth_volume,
            flatmap_style=flatmap_style,
            coordinate_mode=coordinate_mode,
            invalid_zero_sentinel=invalid_zero_sentinel,
            invalid_negative_one_sentinel=invalid_negative_one_sentinel,
            resolution_um=resolution_um,
            space_directions=space_directions,
            space_origin=space_origin,
        ).reset_index(drop=True)

        direct_flatmap_valid = selected["flatmap_valid"].to_numpy(dtype=bool)
        direct_depth_valid = selected["depth_valid"].to_numpy(dtype=bool)
        depth_only_retry = (
            direct_flatmap_valid[retry_positions] & ~direct_depth_valid[retry_positions]
        )
        mirrored_depth_valid = (
            mirrored["depth_valid"].to_numpy(dtype=bool) & depth_only_retry
        )
        if mirrored_depth_valid.any():
            mirrored_depth_positions = retry_positions[mirrored_depth_valid]
            selected.loc[mirrored_depth_positions, "depth_um"] = mirrored.loc[
                mirrored_depth_valid,
                "depth_um",
            ].to_numpy()
            selected.loc[mirrored_depth_positions, "depth_valid"] = True
            selected.loc[mirrored_depth_positions, "valid"] = True
            selected.loc[mirrored_depth_positions, "invalid_reason"] = ""
            lookup_mode[mirrored_depth_positions] = FLATMAP_LOOKUP_MIRRORED_DEPTH

        full_mirror_valid = ~direct_flatmap_valid[retry_positions] & mirrored[
            "valid"
        ].to_numpy(dtype=bool)
        if full_mirror_valid.any():
            mirrored_positions = retry_positions[full_mirror_valid]
            projection_columns = (
                "voxel_i",
                "voxel_j",
                "voxel_k",
                "x_flat",
                "y_flat",
                "depth_um",
                "flatmap_valid",
                "depth_valid",
                "valid",
                "invalid_reason",
            )
            selected.loc[mirrored_positions, projection_columns] = mirrored.loc[
                full_mirror_valid,
                projection_columns,
            ].to_numpy()
            lookup_mode[mirrored_positions] = FLATMAP_LOOKUP_MIRRORED

    selected.loc[:, "flatmap_lookup_mode"] = lookup_mode
    return selected


def build_projected_segments(projected_nodes: pd.DataFrame) -> ProjectedSegments:
    """Build 2D parent-child line segments where both endpoints are valid."""
    _require_columns(
        projected_nodes,
        ("file_id", "node_id", "parent_id", "x_flat", "y_flat", "valid"),
    )

    segment_arrays: list[np.ndarray] = []
    file_ids: list[object] = []
    source_node_ids: list[object] = []
    target_node_ids: list[object] = []

    for file_id, group in projected_nodes.groupby("file_id", sort=False):
        child = group.reset_index(drop=True).loc[
            :,
            ["node_id", "parent_id", "x_flat", "y_flat", "valid"],
        ]
        parent = group.reset_index(drop=True).loc[
            :,
            ["node_id", "x_flat", "y_flat", "valid"],
        ]
        parent = parent.rename(
            columns={
                "node_id": "parent_id",
                "x_flat": "parent_x_flat",
                "y_flat": "parent_y_flat",
                "valid": "parent_valid",
            }
        )
        merged = child.merge(parent, on="parent_id", how="left", sort=False)
        valid_edges = merged["valid"].eq(True) & merged["parent_valid"].eq(True)
        if not bool(valid_edges.any()):
            continue

        edges = merged.loc[valid_edges]
        data = np.empty((len(edges), 2, 2), dtype=np.float64)
        data[:, 0, 0] = edges["parent_x_flat"].to_numpy(dtype=np.float64)
        data[:, 0, 1] = edges["parent_y_flat"].to_numpy(dtype=np.float64)
        data[:, 1, 0] = edges["x_flat"].to_numpy(dtype=np.float64)
        data[:, 1, 1] = edges["y_flat"].to_numpy(dtype=np.float64)
        segment_arrays.append(data)
        file_ids.extend([file_id] * len(edges))
        source_node_ids.extend(edges["parent_id"].tolist())
        target_node_ids.extend(edges["node_id"].tolist())

    data = (
        np.concatenate(segment_arrays, axis=0)
        if segment_arrays
        else np.empty((0, 2, 2), dtype=np.float64)
    )
    return ProjectedSegments(
        data=data,
        file_ids=file_ids,
        source_node_ids=source_node_ids,
        target_node_ids=target_node_ids,
    )


def summarize_projection(
    projected_nodes: pd.DataFrame,
    segments: ProjectedSegments,
) -> ProjectionSummary:
    """Return summary counts for projected nodes and rendered segments."""
    if projected_nodes.empty:
        return ProjectionSummary(0, 0, 0, 0, 0, 0, int(len(segments.data)), 0, 0)

    valid = projected_nodes["valid"].astype(bool)
    reasons = projected_nodes["invalid_reason"].astype(str)
    if "flatmap_lookup_mode" in projected_nodes.columns:
        lookup_modes = projected_nodes["flatmap_lookup_mode"].fillna("").astype(str)
    else:
        lookup_modes = pd.Series(
            np.where(
                valid.to_numpy(dtype=bool),
                FLATMAP_LOOKUP_DIRECT,
                FLATMAP_LOOKUP_UNMAPPED,
            ),
            index=projected_nodes.index,
        )
    partial_traces = 0
    total_traces = 0
    for _file_id, group in projected_nodes.groupby("file_id", sort=False):
        total_traces += 1
        group_valid = group["valid"].astype(bool)
        if bool(group_valid.any()) and not bool(group_valid.all()):
            partial_traces += 1

    return ProjectionSummary(
        total_nodes=int(len(projected_nodes)),
        valid_nodes=int(valid.sum()),
        out_of_bounds_nodes=int((reasons == "out_of_bounds").sum()),
        invalid_flatmap_nodes=int((reasons == "invalid_flatmap").sum()),
        invalid_depth_nodes=int((reasons == "invalid_depth").sum()),
        missing_input_nodes=int((reasons == "missing_input").sum()),
        rendered_segments=int(len(segments.data)),
        total_traces=int(total_traces),
        traces_with_partial_projection=int(partial_traces),
        direct_lookup_nodes=int((lookup_modes == FLATMAP_LOOKUP_DIRECT).sum()),
        mirrored_depth_lookup_nodes=int(
            (lookup_modes == FLATMAP_LOOKUP_MIRRORED_DEPTH).sum()
        ),
        mirrored_lookup_nodes=int((lookup_modes == FLATMAP_LOOKUP_MIRRORED).sum()),
        unmapped_lookup_nodes=int((lookup_modes == FLATMAP_LOOKUP_UNMAPPED).sum()),
    )


def project_and_build_segments(
    nodes: pd.DataFrame,
    flatmap_volume: np.ndarray,
    depth_volume: np.ndarray,
    *,
    flatmap_style: str = "",
    coordinate_mode: str = COORDINATE_MODE_MICRONS,
    invalid_zero_sentinel: bool = False,
    invalid_negative_one_sentinel: bool = True,
    resolution_um: float = DEFAULT_CCF_RESOLUTION_UM,
    space_directions: np.ndarray | None = None,
    space_origin: np.ndarray | None = None,
    mirror_fallback: bool = False,
    mirror_coord_axis: int = 2,
    mirror_midline: float | None = None,
) -> FlatmapProjectionResult:
    """Project nodes, build valid segments, and summarize the result."""
    projected = project_neuron_nodes_to_flatmap(
        nodes,
        flatmap_volume,
        depth_volume,
        flatmap_style=flatmap_style,
        coordinate_mode=coordinate_mode,
        invalid_zero_sentinel=invalid_zero_sentinel,
        invalid_negative_one_sentinel=invalid_negative_one_sentinel,
        resolution_um=resolution_um,
        space_directions=space_directions,
        space_origin=space_origin,
        mirror_fallback=mirror_fallback,
        mirror_coord_axis=mirror_coord_axis,
        mirror_midline=mirror_midline,
    )
    segments = build_projected_segments(projected)
    summary = summarize_projection(projected, segments)
    return FlatmapProjectionResult(projected, segments, summary)


def format_projection_summary(summary: ProjectionSummary) -> str:
    """Return a compact multiline display string for projection counts."""
    return (
        f"Input nodes: {summary.total_nodes:,}\n"
        f"Projected nodes: {summary.valid_nodes:,}\n"
        f"Rendered segments: {summary.rendered_segments:,}\n"
        f"Out of bounds: {summary.out_of_bounds_nodes:,}\n"
        f"Invalid flatmap/depth: "
        f"{summary.invalid_flatmap_nodes:,}/{summary.invalid_depth_nodes:,}\n"
        f"Lookup direct/mirrored-depth/mirrored/unmapped: "
        f"{summary.direct_lookup_nodes:,}/"
        f"{summary.mirrored_depth_lookup_nodes:,}/"
        f"{summary.mirrored_lookup_nodes:,}/"
        f"{summary.unmapped_lookup_nodes:,}\n"
        f"Missing input: {summary.missing_input_nodes:,}\n"
        f"Partial traces: {summary.traces_with_partial_projection:,} "
        f"of {summary.total_traces:,}"
    )
