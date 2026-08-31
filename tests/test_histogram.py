from __future__ import annotations

import numpy as np

from napari_neuron_navigator.analysis.histogram import (
    _build_histogram_plot_series,
    _build_histogram_series,
    _build_histogram_step_curve,
    _prepare_histogram_values,
)


def test_prepare_histogram_values_filters_nonfinite_and_zero_by_default() -> None:
    volume = np.array([0.0, 1.0, np.nan, np.inf, -2.0], dtype=np.float32)

    values = _prepare_histogram_values(volume)

    assert np.array_equal(values, np.array([1.0, -2.0], dtype=np.float32))


def test_prepare_histogram_values_can_include_zero() -> None:
    volume = np.array([0.0, 1.0, np.nan], dtype=np.float32)

    values = _prepare_histogram_values(volume, include_zero=True)

    assert np.array_equal(values, np.array([0.0, 1.0], dtype=np.float32))


def test_build_histogram_series_uses_stable_common_bin_edges() -> None:
    volume = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    bin_edges, series = _build_histogram_series([("Layer A", volume)], bins=4)

    assert np.allclose(bin_edges, np.array([1.0, 1.75, 2.5, 3.25, 4.0], dtype=np.float32))
    assert np.array_equal(series[0]["hist"], np.array([1, 1, 1, 1], dtype=np.int64))


def test_build_histogram_series_preserves_multi_layer_overlay_input() -> None:
    left = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    right = np.array([0.0, 2.0, 4.0], dtype=np.float32)

    bin_edges, series = _build_histogram_series(
        [("Left", left), ("Right", right)],
        bins=4,
    )

    assert np.allclose(bin_edges, np.array([1.0, 1.75, 2.5, 3.25, 4.0], dtype=np.float32))
    assert [entry["name"] for entry in series] == ["Left", "Right"]
    assert int(series[0]["hist"].sum()) == 2
    assert int(series[1]["hist"].sum()) == 2


def test_build_histogram_series_handles_all_zero_input() -> None:
    volume = np.zeros((2, 2, 2), dtype=np.float32)

    bin_edges, series = _build_histogram_series([("Layer A", volume)], bins=4)

    assert np.allclose(bin_edges, np.linspace(0.0, 1.0, 5, dtype=np.float32))
    assert np.array_equal(series[0]["hist"], np.zeros(4, dtype=np.int64))


def test_build_histogram_step_curve_converts_counts_to_step_points() -> None:
    bin_edges = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    hist = np.array([2, 5], dtype=np.int64)

    x, y = _build_histogram_step_curve(bin_edges, hist)

    assert np.array_equal(x, np.array([1.0, 2.0, 2.0, 3.0], dtype=np.float32))
    assert np.array_equal(y, np.array([2.0, 2.0, 5.0, 5.0], dtype=np.float32))


def test_build_histogram_plot_series_preserves_multi_layer_step_curves() -> None:
    left = np.array([1.0, 2.0, 2.0], dtype=np.float32)
    right = np.array([2.0, 4.0, 4.0], dtype=np.float32)

    bin_edges, series = _build_histogram_plot_series(
        [("Left", left), ("Right", right)],
        bins=4,
    )

    assert np.allclose(bin_edges, np.array([1.0, 1.75, 2.5, 3.25, 4.0], dtype=np.float32))
    assert [entry["name"] for entry in series] == ["Left", "Right"]
    assert np.array_equal(
        series[0]["x"],
        np.array([1.0, 1.75, 1.75, 2.5, 2.5, 3.25, 3.25, 4.0], dtype=np.float32),
    )
    assert np.array_equal(
        series[0]["y"],
        np.array([1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert np.array_equal(
        series[1]["y"],
        np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0], dtype=np.float32),
    )


def test_build_histogram_plot_series_handles_empty_curves() -> None:
    volume = np.zeros((2, 2, 2), dtype=np.float32)

    _, series = _build_histogram_plot_series([("Layer A", volume)], bins=4)

    assert series[0]["x"].size == 0
    assert series[0]["y"].size == 0
