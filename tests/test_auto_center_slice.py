"""Tests for depth-slice auto-centering utilities."""

import numpy as np
import pandas as pd

from napari_swc_viewer.auto_center import (
    center_to_depth_world,
    compute_center_of_rendered_neurons,
    depth_axis_from_not_displayed,
    plan_auto_center_depth,
)


def test_compute_center_prefers_line_coordinates() -> None:
    """Line data center is used ahead of point/soma fallbacks."""
    line_data = {
        "a": (
            np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
            np.array([[0, 1]], dtype=np.int32),
        ),
        "b": (
            np.array([[4.0, 4.0, 4.0]]),
            np.array([], dtype=np.int32).reshape(0, 2),
        ),
    }
    points_df = pd.DataFrame({"x": [100.0], "y": [100.0], "z": [100.0]})
    soma_df = pd.DataFrame({"x": [200.0], "y": [200.0], "z": [200.0]})

    center = compute_center_of_rendered_neurons(
        line_data=line_data,
        points_df=points_df,
        soma_df=soma_df,
    )

    np.testing.assert_allclose(center, [2.0, 2.0, 2.0])


def test_compute_center_falls_back_to_points_then_soma() -> None:
    """Fallback order is points first, then soma."""
    points_df = pd.DataFrame(
        {"x": [10.0, 14.0], "y": [20.0, 24.0], "z": [30.0, 34.0]}
    )
    soma_df = pd.DataFrame({"x": [5.0, 7.0], "y": [15.0, 17.0], "z": [9.0, 11.0]})

    points_center = compute_center_of_rendered_neurons(
        line_data={},
        points_df=points_df,
        soma_df=soma_df,
    )
    soma_center = compute_center_of_rendered_neurons(
        line_data=None,
        points_df=pd.DataFrame(columns=["x", "y", "z"]),
        soma_df=soma_df,
    )

    np.testing.assert_allclose(points_center, [12.0, 22.0, 32.0])
    np.testing.assert_allclose(soma_center, [6.0, 16.0, 10.0])


def test_center_to_world_uses_depth_axis_scale() -> None:
    """Micron coordinates are converted to world coordinates via axis scale."""
    depth_axis = depth_axis_from_not_displayed([2])
    world_depth = center_to_depth_world(
        center_xyz=np.array([10.0, 20.0, 30.0]),
        depth_axis=depth_axis,
        scale=[1.0, 1.0, 0.5],
    )
    assert world_depth == 15.0


def test_plan_auto_center_runs_once_then_skips() -> None:
    """The planner marks first action complete and skips subsequent calls."""
    line_data = {
        "a": (
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]),
            np.array([[0, 1]], dtype=np.int32),
        )
    }

    applied, axis, target_world = plan_auto_center_depth(
        applied_once=False,
        line_data=line_data,
        points_df=None,
        soma_df=None,
        not_displayed=[1],
        scale=[1.0, 0.5, 1.0],
    )
    assert applied is True
    assert axis == 1
    assert target_world == 1.5

    applied_again, axis_again, target_again = plan_auto_center_depth(
        applied_once=applied,
        line_data=line_data,
        points_df=None,
        soma_df=None,
        not_displayed=[1],
        scale=[1.0, 0.5, 1.0],
    )
    assert applied_again is True
    assert axis_again is None
    assert target_again is None


def test_plan_auto_center_noop_without_coordinates() -> None:
    """No-op planning when no line/point/soma coordinates are available."""
    applied, axis, target_world = plan_auto_center_depth(
        applied_once=False,
        line_data=None,
        points_df=pd.DataFrame(columns=["x", "y", "z"]),
        soma_df=pd.DataFrame(columns=["x", "y", "z"]),
        not_displayed=[0],
        scale=None,
    )

    assert applied is False
    assert axis is None
    assert target_world is None
