"""Distinct neuron colors that stay visible on a dark canvas.

Sampling a sequential colormap such as ``turbo`` across ``linspace(0, 1, n)``
puts neurons on that map's endpoints, and turbo's endpoints are near-black
(CIELAB lightness ~12).  Against napari's dark canvas -- especially through the
additive, transparent-to-color tint a per-neuron heatmap uses -- those neurons
are invisible.  Sampling a brighter sub-range fixes visibility but not
distinctness: a continuous ramp cut into 16 pieces leaves neighbors a few Delta-E
apart.

This module builds a categorical palette instead, following the Glasbey et al.
(2007) construction: pick colors greedily from a quantized RGB grid, each time
taking the candidate furthest (in a perceptually uniform space) from everything
already picked.  Seeding that set with the canvas color keeps every pick clear
of the background, and a lightness floor keeps them off the dark rail.

Two properties fall out of the greedy order and are worth preserving:

* it is deterministic, so a neuron's color is reproducible across sessions; and
* it is prefix-stable -- the first ``k`` colors do not change as ``n`` grows, so
  adding a neuron to the table does not recolor the others.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# Lightness bounds in CIELAB.  The floor is what keeps colors off a dark canvas;
# the ceiling keeps them from washing out into near-white.
MIN_PALETTE_LIGHTNESS = 45.0
MAX_PALETTE_LIGHTNESS = 95.0

# Quantization of the RGB cube the greedy search draws from.  16 steps per axis
# leaves ~2900 admissible candidates, which is ample for a palette this size and
# keeps generation in the single-digit milliseconds.
_RGB_GRID_STEPS = 16

# Beyond a few hundred entries no palette is perceptually distinct, so the
# generated palette stops here and callers cycle it.  Pretending 18,000 neurons
# can have 18,000 telltale colors would be a lie either way; cycling at least
# keeps every color individually legible.
MAX_DISTINCT_COLORS = 256

# The canvas the palette has to stand out against, in CIELAB.
_BACKGROUND_LAB = np.array([0.0, 0.0, 0.0])


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB in ``[0, 1]`` to CIELAB under a D65 white point."""
    rgb = np.asarray(rgb, dtype=float)[..., :3]
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    to_xyz = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = linear @ to_xyz.T
    white = np.array([0.95047, 1.0, 1.08883])
    ratio = xyz / white
    delta = 6.0 / 29.0
    f = np.where(
        ratio > delta**3,
        np.cbrt(ratio),
        ratio / (3.0 * delta**2) + 4.0 / 29.0,
    )
    lightness = 116.0 * f[..., 1] - 16.0
    green_red = 500.0 * (f[..., 0] - f[..., 1])
    blue_yellow = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([lightness, green_red, blue_yellow], axis=-1)


def _candidate_colors() -> tuple[np.ndarray, np.ndarray]:
    """Return admissible ``(rgb, lab)`` candidates within the lightness bounds."""
    axis = np.linspace(0.0, 1.0, _RGB_GRID_STEPS)
    red, green, blue = np.meshgrid(axis, axis, axis, indexing="ij")
    rgb = np.stack([red.ravel(), green.ravel(), blue.ravel()], axis=-1)
    lab = _srgb_to_lab(rgb)
    admissible = (lab[:, 0] >= MIN_PALETTE_LIGHTNESS) & (
        lab[:, 0] <= MAX_PALETTE_LIGHTNESS
    )
    return rgb[admissible], lab[admissible]


@lru_cache(maxsize=1)
def _base_palette() -> tuple[tuple[float, float, float, float], ...]:
    """Return the full generated palette, computed once per process."""
    rgb, lab = _candidate_colors()
    if len(rgb) == 0:  # pragma: no cover - only reachable if the bounds exclude all
        return ()

    # Distance to the nearest already-chosen color, seeded by the canvas so the
    # first pick is the candidate that stands out most against the background.
    distance = np.linalg.norm(lab - _BACKGROUND_LAB, axis=1)
    palette: list[tuple[float, float, float, float]] = []
    for _ in range(min(MAX_DISTINCT_COLORS, len(rgb))):
        index = int(np.argmax(distance))
        chosen = rgb[index]
        palette.append(
            (float(chosen[0]), float(chosen[1]), float(chosen[2]), 1.0)
        )
        distance = np.minimum(distance, np.linalg.norm(lab - lab[index], axis=1))
    return tuple(palette)


def neuron_palette(count: int) -> list[list[float]]:
    """Return ``count`` RGBA colors that stay legible on a dark canvas.

    Colors are returned in a deterministic order, and the first ``k`` entries are
    the same for any ``count >= k``, so growing the neuron table never recolors
    the neurons already in it.  Requests beyond :data:`MAX_DISTINCT_COLORS` cycle
    the palette rather than emitting colors too similar to tell apart.
    """
    requested = max(0, int(count))
    if requested == 0:
        return []
    palette = _base_palette()
    if not palette:  # pragma: no cover - defensive
        return [[0.5, 0.5, 0.5, 1.0] for _ in range(requested)]
    return [list(palette[index % len(palette)]) for index in range(requested)]
