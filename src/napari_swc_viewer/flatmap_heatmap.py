"""Depth-aware flatmap render helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_FLATMAP_XY_BINS = 256
DEFAULT_FLATMAP_DEPTH_BIN_UM = 25.0
MAX_FLATMAP_HEATMAP_VOXELS = 100_000_000


@dataclass(frozen=True)
class FlatmapRenderSummary:
    """Counts and ranges for a depth-aware flatmap render."""

    total_nodes: int
    flatmap_valid_nodes: int
    depth_valid_nodes: int
    depth_minus_one_nodes: int
    rendered_nodes: int
    excluded_depth_minus_one_nodes: int
    nonzero_voxels: int
    traces_represented: int
    xy_bins: int
    depth_bins: int
    depth_bin_um: float
    x_flat_min: float
    x_flat_max: float
    y_flat_min: float
    y_flat_max: float
    depth_min_um: float
    depth_max_um: float
    includes_depth_minus_one_plane: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        """Return a JSON-safe dictionary."""
        return {
            "total_nodes": int(self.total_nodes),
            "flatmap_valid_nodes": int(self.flatmap_valid_nodes),
            "depth_valid_nodes": int(self.depth_valid_nodes),
            "depth_minus_one_nodes": int(self.depth_minus_one_nodes),
            "rendered_nodes": int(self.rendered_nodes),
            "excluded_depth_minus_one_nodes": int(
                self.excluded_depth_minus_one_nodes
            ),
            "nonzero_voxels": int(self.nonzero_voxels),
            "traces_represented": int(self.traces_represented),
            "xy_bins": int(self.xy_bins),
            "depth_bins": int(self.depth_bins),
            "depth_bin_um": float(self.depth_bin_um),
            "x_flat_min": float(self.x_flat_min),
            "x_flat_max": float(self.x_flat_max),
            "y_flat_min": float(self.y_flat_min),
            "y_flat_max": float(self.y_flat_max),
            "depth_min_um": float(self.depth_min_um),
            "depth_max_um": float(self.depth_max_um),
            "includes_depth_minus_one_plane": bool(
                self.includes_depth_minus_one_plane
            ),
        }


@dataclass(frozen=True)
class FlatmapRenderResult:
    """Binned flatmap render data and the projected table used to create it."""

    projected_nodes: pd.DataFrame
    volume: np.ndarray
    points: np.ndarray
    point_file_ids: list[object]
    summary: FlatmapRenderSummary


def _validate_resolution(xy_bins: int, depth_bin_um: float) -> tuple[int, float]:
    xy_bins = int(xy_bins)
    depth_bin_um = float(depth_bin_um)
    if xy_bins <= 0:
        raise ValueError("xy_bins must be positive.")
    if depth_bin_um <= 0.0:
        raise ValueError("depth_bin_um must be positive.")
    return xy_bins, depth_bin_um


def _nondegenerate_bounds(lower: float, upper: float) -> tuple[float, float]:
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Flatmap bounds must be finite.")
    if upper > lower:
        return float(lower), float(upper)
    pad = max(abs(float(lower)) * 0.01, 0.5)
    return float(lower - pad), float(upper + pad)


def _flatmap_valid_mask(
    flat_xy: np.ndarray,
    *,
    invalid_zero_sentinel: bool,
    invalid_negative_one_sentinel: bool,
) -> np.ndarray:
    finite = np.all(np.isfinite(flat_xy), axis=1)
    valid = finite.copy()
    if invalid_negative_one_sentinel:
        valid &= ~((flat_xy[:, 0] == -1.0) & (flat_xy[:, 1] == -1.0))
    if invalid_zero_sentinel:
        valid &= ~((flat_xy[:, 0] == 0.0) & (flat_xy[:, 1] == 0.0))
    return valid


def compute_flatmap_xy_bounds(
    flatmap_volume: np.ndarray,
    *,
    invalid_zero_sentinel: bool = False,
    invalid_negative_one_sentinel: bool = True,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return finite x/y flatmap coordinate bounds from a lookup volume."""
    flatmap = np.asarray(flatmap_volume, dtype=float)
    if flatmap.ndim != 4 or flatmap.shape[-1] != 2:
        raise ValueError(
            "flatmap_volume must have shape (nx, ny, nz, 2); "
            f"got {flatmap.shape}."
        )

    flat_xy = flatmap.reshape(-1, 2)
    valid = _flatmap_valid_mask(
        flat_xy,
        invalid_zero_sentinel=invalid_zero_sentinel,
        invalid_negative_one_sentinel=invalid_negative_one_sentinel,
    )
    if not valid.any():
        raise ValueError("Flatmap volume does not contain valid x/y lookup values.")

    valid_xy = flat_xy[valid]
    x_bounds = _nondegenerate_bounds(
        float(np.min(valid_xy[:, 0])),
        float(np.max(valid_xy[:, 0])),
    )
    y_bounds = _nondegenerate_bounds(
        float(np.min(valid_xy[:, 1])),
        float(np.max(valid_xy[:, 1])),
    )
    return x_bounds, y_bounds


def compute_depth_range(depth_volume: np.ndarray) -> tuple[float, float]:
    """Return the finite non-negative depth range from a depth lookup volume."""
    depth = np.asarray(depth_volume, dtype=float)
    if depth.ndim != 3:
        raise ValueError(f"depth_volume must be 3D; got {depth.shape}.")

    valid = np.isfinite(depth) & (depth >= 0.0)
    if not valid.any():
        raise ValueError("Depth volume does not contain valid non-negative depths.")
    values = depth[valid]
    lower = float(np.min(values))
    upper = float(np.max(values))
    if upper <= lower:
        upper = lower + 1.0
    return lower, upper


def _bin_flat_values(
    values: np.ndarray,
    bounds: tuple[float, float],
    bins: int,
) -> np.ndarray:
    lower, upper = _nondegenerate_bounds(bounds[0], bounds[1])
    scaled = (np.asarray(values, dtype=float) - lower) / (upper - lower)
    out = np.floor(scaled * bins).astype(np.int64)
    return np.clip(out, 0, bins - 1)


def _depth_bin_count(
    depth_range_um: tuple[float, float],
    depth_bin_um: float,
) -> int:
    lower, upper = depth_range_um
    return max(1, int(np.floor((upper - lower) / depth_bin_um)) + 1)


def _depth_labels(
    depth_bins: np.ndarray,
    *,
    depth_min_um: float,
    depth_bin_um: float,
    sentinel_offset: int,
) -> list[str]:
    labels: list[str] = []
    for bin_index in depth_bins:
        if bin_index < 0:
            labels.append("")
            continue
        if sentinel_offset and bin_index == 0:
            labels.append("depth -1")
            continue
        valid_bin = int(bin_index) - sentinel_offset
        lower = depth_min_um + valid_bin * depth_bin_um
        upper = lower + depth_bin_um
        labels.append(f"{lower:g}-{upper:g} um")
    return labels


def build_flatmap_render_data(
    projected_nodes: pd.DataFrame,
    flatmap_volume: np.ndarray,
    depth_volume: np.ndarray,
    *,
    xy_bins: int = DEFAULT_FLATMAP_XY_BINS,
    depth_bin_um: float = DEFAULT_FLATMAP_DEPTH_BIN_UM,
    include_depth_minus_one: bool = True,
    invalid_zero_sentinel: bool = False,
    invalid_negative_one_sentinel: bool = True,
) -> FlatmapRenderResult:
    """Build a depth-aware flatmap node-count volume and point coordinates."""
    xy_bins, depth_bin_um = _validate_resolution(xy_bins, depth_bin_um)
    x_bounds, y_bounds = compute_flatmap_xy_bounds(
        flatmap_volume,
        invalid_zero_sentinel=invalid_zero_sentinel,
        invalid_negative_one_sentinel=invalid_negative_one_sentinel,
    )
    depth_range = compute_depth_range(depth_volume)
    valid_depth_bins = _depth_bin_count(depth_range, depth_bin_um)
    sentinel_offset = 1 if include_depth_minus_one else 0
    total_depth_bins = valid_depth_bins + sentinel_offset
    voxel_count = int(total_depth_bins * xy_bins * xy_bins)
    if voxel_count > MAX_FLATMAP_HEATMAP_VOXELS:
        raise ValueError(
            "Flatmap heatmap is too large: "
            f"{total_depth_bins}x{xy_bins}x{xy_bins} voxels. "
            "Use fewer XY bins or a larger depth bin."
        )

    table = projected_nodes.copy()
    for column, default in (
        ("flatmap_valid", False),
        ("depth_valid", False),
    ):
        if column not in table.columns:
            table[column] = default

    flatmap_valid = table["flatmap_valid"].fillna(False).astype(bool).to_numpy()
    depth_valid = table["depth_valid"].fillna(False).astype(bool).to_numpy()
    depth_values = pd.to_numeric(table["depth_um"], errors="coerce").to_numpy(
        dtype=float
    )
    depth_minus_one = np.isfinite(depth_values) & (depth_values == -1.0)
    render_valid = flatmap_valid & (depth_valid | (include_depth_minus_one & depth_minus_one))

    x_bins = np.full(len(table), -1, dtype=np.int64)
    y_bins = np.full(len(table), -1, dtype=np.int64)
    depth_bins = np.full(len(table), -1, dtype=np.int64)

    if render_valid.any():
        x_values = pd.to_numeric(table["x_flat"], errors="coerce").to_numpy(dtype=float)
        y_values = pd.to_numeric(table["y_flat"], errors="coerce").to_numpy(dtype=float)
        x_bins[render_valid] = _bin_flat_values(
            x_values[render_valid],
            x_bounds,
            xy_bins,
        )
        y_bins[render_valid] = _bin_flat_values(
            y_values[render_valid],
            y_bounds,
            xy_bins,
        )

        sentinel_nodes = render_valid & depth_minus_one
        valid_depth_nodes = render_valid & depth_valid
        if sentinel_nodes.any():
            depth_bins[sentinel_nodes] = 0
        if valid_depth_nodes.any():
            raw_depth_bins = np.floor(
                (depth_values[valid_depth_nodes] - depth_range[0]) / depth_bin_um
            ).astype(np.int64)
            raw_depth_bins = np.clip(raw_depth_bins, 0, valid_depth_bins - 1)
            depth_bins[valid_depth_nodes] = raw_depth_bins + sentinel_offset

    volume = np.zeros((total_depth_bins, xy_bins, xy_bins), dtype=np.float32)
    render_indices = np.flatnonzero(render_valid)
    if render_indices.size:
        np.add.at(
            volume,
            (
                depth_bins[render_indices],
                y_bins[render_indices],
                x_bins[render_indices],
            ),
            1.0,
        )

    points = (
        np.column_stack(
            (
                depth_bins[render_indices],
                y_bins[render_indices],
                x_bins[render_indices],
            )
        ).astype(np.float64)
        if render_indices.size
        else np.empty((0, 3), dtype=np.float64)
    )
    point_file_ids = (
        table.iloc[render_indices]["file_id"].tolist()
        if render_indices.size and "file_id" in table.columns
        else []
    )

    table.loc[:, "render_valid"] = render_valid
    table.loc[:, "x_flat_bin"] = x_bins
    table.loc[:, "y_flat_bin"] = y_bins
    table.loc[:, "depth_bin"] = depth_bins
    table.loc[:, "depth_bin_label"] = _depth_labels(
        depth_bins,
        depth_min_um=depth_range[0],
        depth_bin_um=depth_bin_um,
        sentinel_offset=sentinel_offset,
    )

    if render_indices.size and "file_id" in table.columns:
        traces_represented = int(table.iloc[render_indices]["file_id"].nunique())
    else:
        traces_represented = 0
    excluded_depth_minus_one = int(
        (flatmap_valid & depth_minus_one & ~render_valid).sum()
    )

    summary = FlatmapRenderSummary(
        total_nodes=int(len(table)),
        flatmap_valid_nodes=int(flatmap_valid.sum()),
        depth_valid_nodes=int((flatmap_valid & depth_valid).sum()),
        depth_minus_one_nodes=int((flatmap_valid & depth_minus_one).sum()),
        rendered_nodes=int(render_valid.sum()),
        excluded_depth_minus_one_nodes=excluded_depth_minus_one,
        nonzero_voxels=int(np.count_nonzero(volume)),
        traces_represented=traces_represented,
        xy_bins=xy_bins,
        depth_bins=total_depth_bins,
        depth_bin_um=depth_bin_um,
        x_flat_min=x_bounds[0],
        x_flat_max=x_bounds[1],
        y_flat_min=y_bounds[0],
        y_flat_max=y_bounds[1],
        depth_min_um=depth_range[0],
        depth_max_um=depth_range[1],
        includes_depth_minus_one_plane=bool(include_depth_minus_one),
    )
    return FlatmapRenderResult(
        projected_nodes=table,
        volume=volume,
        points=points,
        point_file_ids=point_file_ids,
        summary=summary,
    )
