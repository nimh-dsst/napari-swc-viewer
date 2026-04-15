"""Integration tests for hemisphere flipping using vendored BIL data."""

from pathlib import Path

import numpy as np
import pytest

from napari_swc_viewer.hemisphere import Hemisphere, detect_soma_hemisphere, flip_swc
from napari_swc_viewer.swc import parse_swc

FIXTURE_DIR = Path(__file__).parent / "data" / "hemisphere"
ALLEN_MOUSE_10UM_MIDLINE = 5695.0
TEST_FILENAME = "1119749665_17545_3134-X21894-Y19320_reg.swc"
EXPECTED_FLIPPED_FILENAME = "1119749665_17545_3134-X21894-Y19320_reg_right.swc"


def _require_fixture(filename: str) -> Path:
    """Return a committed fixture path or fail with a clear message."""
    path = FIXTURE_DIR / filename
    if not path.exists():
        pytest.fail(f"Expected fixture file not found: {path}")
    return path


@pytest.fixture(scope="module")
def bil_test_file() -> Path:
    """Return the vendored original SWC fixture."""
    return _require_fixture(TEST_FILENAME)


@pytest.fixture(scope="module")
def expected_flipped_file() -> Path:
    """Return the vendored flipped SWC fixture."""
    return _require_fixture(EXPECTED_FLIPPED_FILENAME)


class TestHemisphereFlippingIntegration:
    """Integration tests comparing flipped SWC files against known results."""

    def test_flip_bil_file_matches_expected(
        self, bil_test_file: Path, expected_flipped_file: Path
    ):
        """Test that flipping a BIL file produces expected coordinates."""
        # Parse the original and expected files
        original = parse_swc(bil_test_file)
        expected = parse_swc(expected_flipped_file)

        # Flip the original file using the documented allen_mouse_10um midline.
        flipped = flip_swc(original, midline=ALLEN_MOUSE_10UM_MIDLINE)

        # Compare the number of nodes
        assert flipped.n_nodes == expected.n_nodes, (
            f"Node count mismatch: flipped has {flipped.n_nodes}, "
            f"expected has {expected.n_nodes}"
        )

        # Compare coordinates with tolerance for floating point differences
        np.testing.assert_array_almost_equal(
            flipped.coords,
            expected.coords,
            decimal=3,
            err_msg="Flipped coordinates do not match expected values",
        )

        # Verify non-coordinate fields are preserved
        np.testing.assert_array_equal(
            flipped.ids, expected.ids, err_msg="Node IDs do not match"
        )
        np.testing.assert_array_equal(
            flipped.types, expected.types, err_msg="Node types do not match"
        )
        np.testing.assert_array_equal(
            flipped.parents, expected.parents, err_msg="Parent IDs do not match"
        )
        np.testing.assert_array_almost_equal(
            flipped.radii, expected.radii, decimal=3, err_msg="Radii do not match"
        )

    def test_flipped_file_hemisphere_changed(
        self, bil_test_file: Path, expected_flipped_file: Path
    ):
        """Test that the flipped file is in the opposite hemisphere."""
        original = parse_swc(bil_test_file)
        expected = parse_swc(expected_flipped_file)

        original_hemisphere = detect_soma_hemisphere(
            original,
            midline=ALLEN_MOUSE_10UM_MIDLINE,
        )
        expected_hemisphere = detect_soma_hemisphere(
            expected,
            midline=ALLEN_MOUSE_10UM_MIDLINE,
        )

        # The expected file should be on the opposite hemisphere or original was midline
        if original_hemisphere != Hemisphere.MIDLINE:
            assert original_hemisphere != expected_hemisphere, (
                f"Expected hemisphere to change after flip. "
                f"Original: {original_hemisphere}, Expected: {expected_hemisphere}"
            )

    def test_double_flip_returns_original(self, bil_test_file: Path):
        """Test that flipping twice returns to original coordinates."""
        original = parse_swc(bil_test_file)

        # Flip twice
        flipped_once = flip_swc(original, midline=ALLEN_MOUSE_10UM_MIDLINE)
        flipped_twice = flip_swc(flipped_once, midline=ALLEN_MOUSE_10UM_MIDLINE)

        # Should match original coordinates
        np.testing.assert_array_almost_equal(
            flipped_twice.coords,
            original.coords,
            decimal=3,
            err_msg="Double flip did not return to original coordinates",
        )
