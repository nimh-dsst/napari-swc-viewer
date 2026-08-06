"""Tests for the dark-canvas neuron palette.

The palette exists because sampling a sequential colormap put neurons on
near-black endpoints.  These tests assert the two properties that fix, measured
in CIELAB rather than by eye: every color clears a lightness floor, and no two
colors land closer than a perceptual threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from napari_swc_viewer.neuron_palette import (
    MAX_DISTINCT_COLORS,
    MAX_PALETTE_LIGHTNESS,
    MIN_PALETTE_LIGHTNESS,
    _srgb_to_lab,
    neuron_palette,
)

# Below this CIELAB lightness a color reads as near-black on a dark canvas.
# turbo's endpoints, the colors this palette replaced, sit at ~12.
_VISIBILITY_FLOOR = 35.0
# Pairwise CIELAB distance below which two colors are hard to tell apart.
_DISTINCTNESS_FLOOR = 25.0


def _lightness(colors) -> np.ndarray:
    return _srgb_to_lab(np.asarray(colors, dtype=float)[:, :3])[:, 0]


def _min_pairwise_distance(colors) -> float:
    lab = _srgb_to_lab(np.asarray(colors, dtype=float)[:, :3])
    distances = np.linalg.norm(lab[:, None, :] - lab[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


def test_srgb_to_lab_matches_known_reference_values() -> None:
    # Reference L*a*b* for sRGB white, black, and the primaries under D65.
    lab = _srgb_to_lab(
        np.asarray(
            [
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    np.testing.assert_allclose(lab[0], [100.0, 0.0, 0.0], atol=0.02)
    np.testing.assert_allclose(lab[1], [0.0, 0.0, 0.0], atol=0.02)
    np.testing.assert_allclose(lab[2], [53.24, 80.09, 67.20], atol=0.05)
    np.testing.assert_allclose(lab[3], [87.73, -86.18, 83.18], atol=0.05)
    np.testing.assert_allclose(lab[4], [32.30, 79.19, -107.86], atol=0.05)


@pytest.mark.parametrize("count", [1, 2, 3, 4, 8, 16, 32, 64])
def test_every_color_clears_the_dark_canvas_floor(count) -> None:
    colors = neuron_palette(count)

    assert len(colors) == count
    lightness = _lightness(colors)
    assert lightness.min() >= _VISIBILITY_FLOOR
    # The generator's own bounds are tighter than the readability floor.
    assert lightness.min() >= MIN_PALETTE_LIGHTNESS
    assert lightness.max() <= MAX_PALETTE_LIGHTNESS


@pytest.mark.parametrize("count", [2, 3, 4, 8, 16, 32])
def test_colors_stay_mutually_distinguishable(count) -> None:
    assert _min_pairwise_distance(neuron_palette(count)) >= _DISTINCTNESS_FLOOR


def test_palette_beats_a_sampled_sequential_colormap() -> None:
    """Pin the regression this palette was introduced to fix."""
    from matplotlib import colormaps

    turbo = colormaps["turbo"]
    for count in (2, 4, 16):
        sampled = [list(turbo(float(t))) for t in np.linspace(0.0, 1.0, count)]
        # turbo always includes its near-black endpoint, whatever the count.
        assert _lightness(sampled).min() < _VISIBILITY_FLOOR
        assert _lightness(neuron_palette(count)).min() > _lightness(sampled).min()


def test_palette_is_deterministic_across_calls() -> None:
    assert neuron_palette(12) == neuron_palette(12)


def test_palette_is_prefix_stable_as_the_table_grows() -> None:
    # Adding a neuron must not recolor the neurons already in the table.
    small = neuron_palette(5)
    assert neuron_palette(9)[:5] == small
    assert neuron_palette(200)[:5] == small


def test_palette_cycles_beyond_the_distinct_limit() -> None:
    colors = neuron_palette(MAX_DISTINCT_COLORS + 3)

    assert len(colors) == MAX_DISTINCT_COLORS + 3
    # Past the limit no palette is perceptually distinct, so it repeats rather
    # than emitting colors nobody could tell apart.
    assert colors[MAX_DISTINCT_COLORS:] == colors[:3]
    assert _lightness(colors).min() >= MIN_PALETTE_LIGHTNESS


def test_palette_returns_opaque_rgba_lists() -> None:
    for color in neuron_palette(6):
        assert isinstance(color, list)
        assert len(color) == 4
        assert color[3] == 1.0
        assert all(0.0 <= channel <= 1.0 for channel in color)


@pytest.mark.parametrize("count", [0, -1, -50])
def test_palette_is_empty_for_non_positive_counts(count) -> None:
    assert neuron_palette(count) == []


def test_palette_entries_are_independent_copies() -> None:
    first = neuron_palette(3)
    first[0][0] = 0.123

    assert neuron_palette(3)[0][0] != 0.123
