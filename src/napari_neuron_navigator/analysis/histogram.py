"""Internal helpers for heatmap histogram preparation."""

from __future__ import annotations

import numpy as np


def _prepare_histogram_values(
    volume: np.ndarray,
    *,
    include_zero: bool = False,
) -> np.ndarray:
    """Return finite intensity values for histogram plotting."""
    values = np.asarray(volume, dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    if not include_zero:
        values = values[values != 0]
    return values.astype(np.float32, copy=False)


def _build_histogram_series(
    named_volumes: list[tuple[str, np.ndarray]],
    *,
    bins: int = 256,
    include_zero: bool = False,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Build common-bin histogram series for a set of named volumes."""
    if bins <= 0:
        raise ValueError("Histogram bins must be greater than zero.")

    prepared: list[tuple[str, np.ndarray]] = []
    nonempty_values: list[np.ndarray] = []
    for name, volume in named_volumes:
        values = _prepare_histogram_values(volume, include_zero=include_zero)
        prepared.append((str(name), values))
        if values.size > 0:
            nonempty_values.append(values)

    if nonempty_values:
        all_values = np.concatenate(nonempty_values)
        value_min = float(all_values.min())
        value_max = float(all_values.max())
        if np.isclose(value_min, value_max):
            width = max(abs(value_min) * 0.01, 0.5)
            bin_edges = np.linspace(
                value_min - width,
                value_max + width,
                bins + 1,
                dtype=np.float32,
            )
        else:
            bin_edges = np.linspace(
                value_min,
                value_max,
                bins + 1,
                dtype=np.float32,
            )
    else:
        bin_edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float32)

    series: list[dict[str, object]] = []
    for name, values in prepared:
        if values.size > 0:
            hist = np.histogram(values, bins=bin_edges)[0].astype(np.int64, copy=False)
        else:
            hist = np.zeros(bins, dtype=np.int64)
        series.append(
            {
                "name": name,
                "values": values,
                "hist": hist,
            }
        )

    return bin_edges, series


def _build_histogram_step_curve(
    bin_edges: np.ndarray,
    hist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert common-bin histogram counts into step-curve coordinates."""
    edges = np.asarray(bin_edges, dtype=np.float32).reshape(-1)
    counts = np.asarray(hist, dtype=np.int64).reshape(-1)

    if edges.size != counts.size + 1:
        raise ValueError("Histogram bin edges must be exactly one longer than counts.")

    if counts.size == 0 or int(counts.sum()) <= 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    x = np.empty(counts.size * 2, dtype=np.float32)
    y = np.empty(counts.size * 2, dtype=np.float32)
    float_counts = counts.astype(np.float32, copy=False)

    x[0::2] = edges[:-1]
    x[1::2] = edges[1:]
    y[0::2] = float_counts
    y[1::2] = float_counts
    return x, y


def _build_histogram_plot_series(
    named_volumes: list[tuple[str, np.ndarray]],
    *,
    bins: int = 256,
    include_zero: bool = False,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Build plot-ready histogram series with shared bins and step curves."""
    bin_edges, series = _build_histogram_series(
        named_volumes,
        bins=bins,
        include_zero=include_zero,
    )

    plot_series: list[dict[str, object]] = []
    for entry in series:
        x, y = _build_histogram_step_curve(bin_edges, entry["hist"])
        plot_series.append(
            {
                **entry,
                "x": x,
                "y": y,
            }
        )

    return bin_edges, plot_series
