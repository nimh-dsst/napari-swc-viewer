"""Utilities for one-time depth-slice auto-centering."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def compute_center_of_rendered_neurons(
    line_data: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    points_df,
    soma_df,
) -> np.ndarray | None:
    """Compute a center point with line > points > soma fallback priority."""
    if line_data:
        coord_blocks = [coords for coords, _ in line_data.values() if coords.size > 0]
        if coord_blocks:
            return np.concatenate(coord_blocks, axis=0).mean(axis=0)

    for df in (points_df, soma_df):
        if df is None or df.empty:
            continue
        coords = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
        if coords.size > 0:
            return coords.mean(axis=0)

    return None


def depth_axis_from_not_displayed(not_displayed: Iterable[int] | None) -> int:
    """Return the active depth axis from napari dims.not_displayed."""
    if not_displayed is None:
        return 0

    try:
        axes = list(not_displayed)
    except Exception:
        return 0

    if not axes:
        return 0
    return int(axes[0])


def center_to_depth_world(
    center_xyz: np.ndarray,
    depth_axis: int,
    scale: list[float] | None,
) -> float:
    """Convert a microns-space center coordinate to depth-axis world value."""
    target_world = float(center_xyz[depth_axis])
    if scale is not None and depth_axis < len(scale):
        target_world *= float(scale[depth_axis])
    return target_world


def plan_auto_center_depth(
    applied_once: bool,
    line_data: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    points_df,
    soma_df,
    not_displayed: Iterable[int] | None,
    scale: list[float] | None,
) -> tuple[bool, int | None, float | None]:
    """Plan one auto-center action and return (new_state, axis, world_value)."""
    if applied_once:
        return True, None, None

    center_xyz = compute_center_of_rendered_neurons(
        line_data=line_data,
        points_df=points_df,
        soma_df=soma_df,
    )
    if center_xyz is None:
        return False, None, None

    depth_axis = depth_axis_from_not_displayed(not_displayed)
    target_world = center_to_depth_world(center_xyz, depth_axis, scale)
    return True, depth_axis, target_world
