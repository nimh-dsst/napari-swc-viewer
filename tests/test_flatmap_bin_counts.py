"""Tests for per-axis flat map bin-count derivation.

The bilateral flat map lays both hemispheres side by side along ``x`` while ``y``
spans one hemisphere, so a single shared bin count makes every bin about twice as
wide as it is tall.  These tests pin the derivation that fixes it, and the first
one is the regression guard that the original square-bin default lacked: it
asserts bin *widths* match across axes rather than bin counts.
"""

from __future__ import annotations

import math

import pytest

from napari_swc_viewer.flatmap_heatmap import (
    DEFAULT_FLATMAP_Y_BINS,
    MAX_FLATMAP_Y_BINS,
    FlatmapBinCounts,
    resolve_flatmap_bin_counts,
)

# Canonical bounds from ``isocortex_total_right_brainglobe_flatmap.parquet``,
# matching the values already checked in at
# tests/test_flatmap_depth_normalization.py.
CANONICAL_BOUNDS = {
    "both_square": ((0.0, 2.0), (0.0, 1.0)),
    "both_shaped": (
        (0.09574997425079346, 1.9042149782180786),
        (0.05073529854416847, 0.9944919943809509),
    ),
}


def _bin_widths(style: str, counts: FlatmapBinCounts) -> tuple[float, float]:
    x_bounds, y_bounds = CANONICAL_BOUNDS[style]
    x_width = (x_bounds[1] - x_bounds[0]) / counts.x_bins
    y_width = (y_bounds[1] - y_bounds[0]) / counts.y_bins
    return x_width, y_width


@pytest.mark.parametrize("style", ["both_square", "both_shaped"])
@pytest.mark.parametrize("y_bins", [17, 64, 256, 512])
def test_derived_grid_has_square_bins(style: str, y_bins: int) -> None:
    """Bin widths must match across axes -- the check the old default failed.

    With a single shared count this ratio is 2.0000 for ``both_square`` and
    1.9162 for ``both_shaped``.  Tolerance is loose enough to absorb the integer
    rounding of the x count, which is worst at small bin counts.
    """
    x_bounds, y_bounds = CANONICAL_BOUNDS[style]
    counts = resolve_flatmap_bin_counts(
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        y_bins=y_bins,
    )
    x_width, y_width = _bin_widths(style, counts)
    # One half-bin of rounding slack on the derived x count.
    tolerance = 0.5 / counts.x_bins
    assert x_width / y_width == pytest.approx(1.0, rel=tolerance * 2)


def test_documented_defaults_are_pinned() -> None:
    """The numbers quoted in USE_CASES/MANUAL must not drift silently."""
    square = resolve_flatmap_bin_counts(
        x_bounds=CANONICAL_BOUNDS["both_square"][0],
        y_bounds=CANONICAL_BOUNDS["both_square"][1],
        y_bins=DEFAULT_FLATMAP_Y_BINS,
    )
    shaped = resolve_flatmap_bin_counts(
        x_bounds=CANONICAL_BOUNDS["both_shaped"][0],
        y_bounds=CANONICAL_BOUNDS["both_shaped"][1],
        y_bins=DEFAULT_FLATMAP_Y_BINS,
    )
    assert square == FlatmapBinCounts(y_bins=256, x_bins=512)
    assert shaped == FlatmapBinCounts(y_bins=256, x_bins=491)


def test_counts_are_never_equal_for_a_bilateral_map() -> None:
    """A direct guard against the old square default coming back."""
    for style, (x_bounds, y_bounds) in CANONICAL_BOUNDS.items():
        counts = resolve_flatmap_bin_counts(
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            y_bins=DEFAULT_FLATMAP_Y_BINS,
        )
        assert counts.x_bins != counts.y_bins, style


def test_square_style_is_exactly_double() -> None:
    """``both_square`` has an exactly 2.0 aspect, so doubling is exact there.

    This is precisely why ``2 * y_bins`` looks correct and must not be
    generalized -- see the shaped case below.
    """
    counts = resolve_flatmap_bin_counts(
        x_bounds=(0.0, 2.0),
        y_bounds=(0.0, 1.0),
        y_bins=256,
    )
    assert counts.x_bins == 2 * counts.y_bins


def test_shaped_style_is_not_double_and_doubling_would_be_anisotropic() -> None:
    """Doubling the shaped style leaves 4.2% anisotropy, so it is wrong."""
    x_bounds, y_bounds = CANONICAL_BOUNDS["both_shaped"]
    counts = resolve_flatmap_bin_counts(
        x_bounds=x_bounds, y_bounds=y_bounds, y_bins=256
    )
    assert counts.x_bins != 2 * counts.y_bins

    y_width = (y_bounds[1] - y_bounds[0]) / counts.y_bins
    doubled_width = (x_bounds[1] - x_bounds[0]) / (2 * counts.y_bins)
    assert doubled_width / y_width == pytest.approx(0.958, abs=0.002)


def test_rounding_is_half_up_not_bankers() -> None:
    """A tie must round up, stably -- this feeds the cache identity digest.

    ``round()`` is non-monotone at ties: ``round(490.5)`` is 490 while
    ``round(491.5)`` is 492.  Bounds below put the derived count exactly on
    ``.5`` so a switch back to ``round()`` fails here.
    """
    # aspect = 2.5 / 1.0, y_bins = 3 -> 7.5 exactly.
    counts = resolve_flatmap_bin_counts(
        x_bounds=(0.0, 2.5),
        y_bounds=(0.0, 1.0),
        y_bins=3,
    )
    assert counts.x_bins == 8
    assert round(7.5) == 8  # sanity: this tie happens to agree
    # aspect = 1.5, y_bins = 3 -> 4.5 exactly, where round() disagrees.
    counts = resolve_flatmap_bin_counts(
        x_bounds=(0.0, 1.5),
        y_bounds=(0.0, 1.0),
        y_bins=3,
    )
    assert counts.x_bins == 5
    assert round(4.5) == 4  # round() would have given 4


def test_narrow_map_never_collapses_x_below_one() -> None:
    """A tall, narrow map must still get at least one x bin."""
    counts = resolve_flatmap_bin_counts(
        x_bounds=(0.0, 0.001),
        y_bounds=(0.0, 1000.0),
        y_bins=1,
    )
    assert counts.x_bins == 1


def test_non_positive_y_bins_is_rejected() -> None:
    for bad in (0, -1):
        with pytest.raises(ValueError, match="y_bins must be positive"):
            resolve_flatmap_bin_counts(
                x_bounds=(0.0, 2.0),
                y_bounds=(0.0, 1.0),
                y_bins=bad,
            )


def test_degenerate_bounds_are_padded_not_fatal() -> None:
    """Reuses ``_nondegenerate_bounds`` so it agrees with ``_bin_flat_values``."""
    counts = resolve_flatmap_bin_counts(
        x_bounds=(1.0, 1.0),
        y_bounds=(0.0, 1.0),
        y_bins=4,
    )
    assert counts.y_bins == 4
    assert counts.x_bins >= 1


def test_default_y_bins_stays_within_the_portable_key_ceiling() -> None:
    """The spin cap must keep the derived x count under the old 4096 limit."""
    widest = max((x[1] - x[0]) / (y[1] - y[0]) for x, y in CANONICAL_BOUNDS.values())
    assert math.floor(MAX_FLATMAP_Y_BINS * widest + 0.5) <= 4096
    assert DEFAULT_FLATMAP_Y_BINS <= MAX_FLATMAP_Y_BINS
