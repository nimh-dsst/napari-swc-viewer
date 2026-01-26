"""Integration tests for hemisphere flipping using real BIL data.

These tests download SWC files from the Brain Image Library and compare
flipped coordinates against known expected results.
"""

import json
import re
import ssl
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from napari_swc_viewer.hemisphere import flip_swc
from napari_swc_viewer.swc import parse_swc

# Test data directories
TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
FLIPPED_DIR = TEST_DATA_DIR / "flipped"

# BIL API endpoints
API_BASE = "https://api.brainimagelibrary.org"
DOWNLOAD_BASE = "https://download.brainimagelibrary.org"

# Known submission UUID for the morphology dataset (from DOI 10.35077/g.73)
MORPHOLOGY_SUBMISSION_UUID = "0fcde5fdd6f7ccb2"

# Test file to download and compare
TEST_FILENAME = "1119749665_17545_3134-X21894-Y19320_reg.swc"
EXPECTED_FLIPPED_FILENAME = "1119749665_17545_3134-X21894-Y19320_reg_right.swc"


def get_ssl_context():
    """Create SSL context that doesn't verify certificates (BIL API requirement)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def api_get(endpoint: str) -> dict:
    """Make a GET request to the BIL API."""
    url = f"{API_BASE}/{endpoint}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=get_ssl_context()) as response:
        return json.loads(response.read().decode())


def find_file_url(filename: str) -> str | None:
    """Find the download URL for a specific file in the BIL dataset.

    Parameters
    ----------
    filename : str
        The filename to search for.

    Returns
    -------
    str or None
        The full download URL if found, None otherwise.
    """
    # Query for bildids in the dataset
    result = api_get(f"query/submission?submission_uuid={MORPHOLOGY_SUBMISSION_UUID}")

    if result.get("success") != "true":
        return None

    bildids = result.get("bildids", [])

    for bildid in bildids:
        retrieve_result = api_get(f"retrieve?bildid={bildid}")

        if retrieve_result.get("success") != "true":
            continue

        for entry in retrieve_result.get("retjson", []):
            dataset = entry.get("Dataset", [{}])[0]
            bildirectory = dataset.get("bildirectory", "")

            if not bildirectory:
                continue

            # Convert bildirectory to download URL path
            match = re.match(r"/bil/data/(.+)", bildirectory)
            if not match:
                continue

            download_path = match.group(1)
            dir_url = f"{DOWNLOAD_BASE}/{download_path}/"

            try:
                req = urllib.request.Request(dir_url)
                with urllib.request.urlopen(req, context=get_ssl_context()) as response:
                    html = response.read().decode()

                # Check if our target file is in this directory
                if f'href="{filename}"' in html:
                    return f"{DOWNLOAD_BASE}/{download_path}/{filename}"
            except Exception:
                continue

    return None


def download_file(url: str, output_path: Path) -> bool:
    """Download a file from URL to output_path."""
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=get_ssl_context()) as response:
            output_path.write_bytes(response.read())
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def bil_test_file() -> Path:
    """Download the test SWC file from BIL if not already present.

    Returns the path to the downloaded file.
    """
    output_path = TEST_DATA_DIR / TEST_FILENAME

    if output_path.exists():
        return output_path

    # Find and download the file
    url = find_file_url(TEST_FILENAME)
    if url is None:
        pytest.skip(f"Could not find {TEST_FILENAME} in BIL dataset")

    if not download_file(url, output_path):
        pytest.skip(f"Failed to download {TEST_FILENAME} from BIL")

    return output_path


@pytest.fixture(scope="module")
def expected_flipped_file() -> Path:
    """Get path to the expected flipped SWC file."""
    path = FLIPPED_DIR / EXPECTED_FLIPPED_FILENAME
    if not path.exists():
        pytest.fail(f"Expected flipped file not found: {path}")
    return path


class TestHemisphereFlippingIntegration:
    """Integration tests comparing flipped SWC files against known results."""

    def test_flip_bil_file_matches_expected(
        self, bil_test_file: Path, expected_flipped_file: Path
    ):
        """Test that flipping a BIL file produces expected coordinates.

        This test downloads a real SWC file from the Brain Image Library,
        flips it using the hemisphere module, and compares the result
        against a known-good flipped file.
        """
        # Parse the original and expected files
        original = parse_swc(bil_test_file)
        expected = parse_swc(expected_flipped_file)

        # Flip the original file using the hemisphere module
        flipped = flip_swc(original, atlas_name="allen_mouse_10um")

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
        from napari_swc_viewer.hemisphere import Hemisphere, detect_soma_hemisphere

        original = parse_swc(bil_test_file)
        expected = parse_swc(expected_flipped_file)

        # Detect hemispheres (use validate=False to avoid atlas download in quick test)
        original_hemisphere = detect_soma_hemisphere(
            original, atlas_name="allen_mouse_10um", validate=False
        )
        expected_hemisphere = detect_soma_hemisphere(
            expected, atlas_name="allen_mouse_10um", validate=False
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
        flipped_once = flip_swc(original, atlas_name="allen_mouse_10um")
        flipped_twice = flip_swc(flipped_once, atlas_name="allen_mouse_10um")

        # Should match original coordinates
        np.testing.assert_array_almost_equal(
            flipped_twice.coords,
            original.coords,
            decimal=3,
            err_msg="Double flip did not return to original coordinates",
        )
