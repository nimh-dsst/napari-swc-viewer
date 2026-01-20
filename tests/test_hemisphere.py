"""Tests for hemisphere detection and coordinate flipping."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from napari_swc_viewer.hemisphere import (
    Hemisphere,
    detect_hemisphere,
    detect_soma_hemisphere,
    flip_coordinates,
    flip_swc,
    flip_swc_batch,
    get_atlas_midline,
)
from napari_swc_viewer.swc import SWCData


@pytest.fixture
def mock_atlas():
    """Create a mock BrainGlobeAtlas."""
    atlas = MagicMock()
    # Mock a 10mm x 10mm x 10mm atlas at 25um resolution
    # Shape in voxels: 400 x 400 x 400
    atlas.shape = (400, 400, 400)
    atlas.resolution = (25.0, 25.0, 25.0)  # microns
    return atlas


@pytest.fixture
def sample_swc_data():
    """Create sample SWC data for testing."""
    return SWCData(
        ids=np.array([1, 2, 3, 4, 5], dtype=np.int32),
        types=np.array([1, 3, 3, 3, 2], dtype=np.int32),
        coords=np.array(
            [
                [1000.0, 2000.0, 3000.0],  # soma - left of midline (midline at 5000)
                [1100.0, 2100.0, 3100.0],
                [1200.0, 2200.0, 3200.0],
                [1300.0, 2300.0, 3300.0],
                [1050.0, 1950.0, 2950.0],
            ],
            dtype=np.float64,
        ),
        radii=np.array([5.0, 2.0, 1.5, 1.0, 2.0], dtype=np.float64),
        parents=np.array([-1, 1, 2, 3, 1], dtype=np.int32),
    )


class TestGetAtlasMidline:
    """Tests for get_atlas_midline function."""

    def test_midline_calculation(self, mock_atlas):
        """Test midline is calculated correctly."""
        # Atlas is 400 voxels at 25um = 10000um total
        # Midline should be at 5000um
        midline = get_atlas_midline(mock_atlas)
        assert midline == 5000.0

    def test_midline_different_resolution(self):
        """Test midline with different resolution."""
        atlas = MagicMock()
        atlas.shape = (500, 500, 1000)  # Different shape for LR axis
        atlas.resolution = (10.0, 10.0, 10.0)  # 10um resolution

        midline = get_atlas_midline(atlas)
        # LR axis (index 2): 1000 voxels * 10um = 10000um, midline at 5000
        assert midline == 5000.0


class TestDetectHemisphere:
    """Tests for detect_hemisphere function."""

    def test_left_hemisphere(self, mock_atlas):
        """Test detection of left hemisphere."""
        coords = np.array([[1000.0, 2000.0, 3000.0]])  # x < midline (5000)
        result = detect_hemisphere(coords, atlas=mock_atlas)
        assert result == Hemisphere.LEFT

    def test_right_hemisphere(self, mock_atlas):
        """Test detection of right hemisphere."""
        coords = np.array([[8000.0, 2000.0, 3000.0]])  # x > midline (5000)
        result = detect_hemisphere(coords, atlas=mock_atlas)
        assert result == Hemisphere.RIGHT

    def test_midline(self, mock_atlas):
        """Test detection at midline."""
        coords = np.array([[5000.0, 2000.0, 3000.0]])  # x == midline
        result = detect_hemisphere(coords, atlas=mock_atlas)
        assert result == Hemisphere.MIDLINE

    def test_single_point(self, mock_atlas):
        """Test with a single point (1D array)."""
        coords = np.array([1000.0, 2000.0, 3000.0])
        result = detect_hemisphere(coords, atlas=mock_atlas)
        assert result == Hemisphere.LEFT

    def test_multiple_points_centroid(self, mock_atlas):
        """Test with multiple points uses centroid."""
        # Coords with mean x = 4000 (left of midline 5000)
        coords = np.array(
            [
                [3000.0, 0.0, 0.0],
                [5000.0, 0.0, 0.0],
            ]
        )
        result = detect_hemisphere(coords, atlas=mock_atlas)
        assert result == Hemisphere.LEFT

    def test_custom_midline(self):
        """Test with custom midline value."""
        coords = np.array([[100.0, 0.0, 0.0]])
        result = detect_hemisphere(coords, midline=50.0)
        assert result == Hemisphere.RIGHT  # 100 > 50

    def test_custom_coord_axis(self):
        """Test with different coordinate axis."""
        coords = np.array([[0.0, 100.0, 0.0]])
        result = detect_hemisphere(coords, midline=50.0, coord_axis=1)
        assert result == Hemisphere.RIGHT  # y=100 > 50


class TestDetectSomaHemisphere:
    """Tests for detect_soma_hemisphere function."""

    def test_soma_left(self, sample_swc_data, mock_atlas):
        """Test soma hemisphere detection."""
        result = detect_soma_hemisphere(sample_swc_data, atlas=mock_atlas)
        assert result == Hemisphere.LEFT

    def test_no_soma_raises(self, mock_atlas):
        """Test that missing soma raises ValueError."""
        swc = SWCData(
            ids=np.array([1, 2], dtype=np.int32),
            types=np.array([2, 3], dtype=np.int32),  # No soma (type 1)
            coords=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64),
            radii=np.array([1.0, 0.5], dtype=np.float64),
            parents=np.array([-1, 1], dtype=np.int32),
        )
        with pytest.raises(ValueError, match="No soma nodes found"):
            detect_soma_hemisphere(swc, atlas=mock_atlas)


class TestFlipCoordinates:
    """Tests for flip_coordinates function."""

    def test_flip_single_point(self, mock_atlas):
        """Test flipping a single coordinate."""
        # Point at x=1000, midline at 5000
        # Flipped: 2*5000 - 1000 = 9000
        coords = np.array([1000.0, 2000.0, 3000.0])
        flipped = flip_coordinates(coords, atlas=mock_atlas)

        np.testing.assert_array_almost_equal(flipped, [9000.0, 2000.0, 3000.0])

    def test_flip_multiple_points(self, mock_atlas):
        """Test flipping multiple coordinates."""
        coords = np.array(
            [
                [1000.0, 2000.0, 3000.0],
                [2000.0, 2000.0, 3000.0],
            ]
        )
        flipped = flip_coordinates(coords, atlas=mock_atlas)

        expected = np.array(
            [
                [9000.0, 2000.0, 3000.0],
                [8000.0, 2000.0, 3000.0],
            ]
        )
        np.testing.assert_array_almost_equal(flipped, expected)

    def test_flip_preserves_y_z(self, mock_atlas):
        """Test that y and z coordinates are unchanged."""
        coords = np.array([[1000.0, 2000.0, 3000.0]])
        flipped = flip_coordinates(coords, atlas=mock_atlas)

        assert flipped[0, 1] == 2000.0
        assert flipped[0, 2] == 3000.0

    def test_flip_custom_midline(self):
        """Test flip with custom midline."""
        coords = np.array([[100.0, 0.0, 0.0]])
        flipped = flip_coordinates(coords, midline=50.0)

        # 2*50 - 100 = 0
        np.testing.assert_array_almost_equal(flipped, [[0.0, 0.0, 0.0]])

    def test_flip_double_returns_original(self, mock_atlas):
        """Test that flipping twice returns original coordinates."""
        coords = np.array([[1234.5, 6789.0, 1111.1]])
        flipped_once = flip_coordinates(coords, atlas=mock_atlas)
        flipped_twice = flip_coordinates(flipped_once, atlas=mock_atlas)

        np.testing.assert_array_almost_equal(coords, flipped_twice)

    def test_flip_large_array_performance(self, mock_atlas):
        """Test that flipping 10000+ coordinates is efficient."""
        # Create array with 10000 coordinates
        n_coords = 10000
        coords = np.random.rand(n_coords, 3) * 10000

        # This should complete quickly with vectorized operations
        flipped = flip_coordinates(coords, atlas=mock_atlas)

        assert flipped.shape == (n_coords, 3)
        # Verify y and z unchanged
        np.testing.assert_array_almost_equal(coords[:, 1:], flipped[:, 1:])


class TestFlipSWC:
    """Tests for flip_swc function."""

    def test_flip_swc_copy(self, sample_swc_data, mock_atlas):
        """Test flipping SWC creates a copy by default."""
        original_coords = sample_swc_data.coords.copy()
        flipped = flip_swc(sample_swc_data, atlas=mock_atlas)

        # Original should be unchanged
        np.testing.assert_array_equal(sample_swc_data.coords, original_coords)
        # Flipped should be different
        assert not np.array_equal(flipped.coords, original_coords)

    def test_flip_swc_in_place(self, sample_swc_data, mock_atlas):
        """Test flipping SWC in place."""
        original_coords = sample_swc_data.coords.copy()
        result = flip_swc(sample_swc_data, atlas=mock_atlas, in_place=True)

        # Result should be the same object
        assert result is sample_swc_data
        # Coords should be modified
        assert not np.array_equal(sample_swc_data.coords, original_coords)

    def test_flip_swc_preserves_structure(self, sample_swc_data, mock_atlas):
        """Test that flipping preserves SWC structure."""
        flipped = flip_swc(sample_swc_data, atlas=mock_atlas)

        np.testing.assert_array_equal(flipped.ids, sample_swc_data.ids)
        np.testing.assert_array_equal(flipped.types, sample_swc_data.types)
        np.testing.assert_array_equal(flipped.radii, sample_swc_data.radii)
        np.testing.assert_array_equal(flipped.parents, sample_swc_data.parents)

    def test_flip_swc_coords_changed(self, sample_swc_data, mock_atlas):
        """Test that coordinates are actually flipped."""
        flipped = flip_swc(sample_swc_data, atlas=mock_atlas)

        # Original soma at x=1000, midline at 5000
        # Flipped soma should be at x=9000
        np.testing.assert_almost_equal(flipped.coords[0, 0], 9000.0)


class TestFlipSWCBatch:
    """Tests for flip_swc_batch function."""

    def test_batch_flip(self, mock_atlas):
        """Test batch flipping multiple SWC files."""
        swc1 = SWCData(
            ids=np.array([1], dtype=np.int32),
            types=np.array([1], dtype=np.int32),
            coords=np.array([[1000.0, 0.0, 0.0]], dtype=np.float64),
            radii=np.array([1.0], dtype=np.float64),
            parents=np.array([-1], dtype=np.int32),
        )
        swc2 = SWCData(
            ids=np.array([1], dtype=np.int32),
            types=np.array([1], dtype=np.int32),
            coords=np.array([[2000.0, 0.0, 0.0]], dtype=np.float64),
            radii=np.array([1.0], dtype=np.float64),
            parents=np.array([-1], dtype=np.int32),
        )

        results = flip_swc_batch([swc1, swc2], atlas=mock_atlas)

        assert len(results) == 2
        np.testing.assert_almost_equal(results[0].coords[0, 0], 9000.0)
        np.testing.assert_almost_equal(results[1].coords[0, 0], 8000.0)

    @patch("napari_swc_viewer.hemisphere.BrainGlobeAtlas")
    def test_batch_loads_atlas_once(self, mock_atlas_class):
        """Test that batch operation loads atlas only once."""
        mock_atlas_class.return_value.shape = (400, 400, 400)
        mock_atlas_class.return_value.resolution = (25.0, 25.0, 25.0)

        swc_list = [
            SWCData(
                ids=np.array([1], dtype=np.int32),
                types=np.array([1], dtype=np.int32),
                coords=np.array([[1000.0, 0.0, 0.0]], dtype=np.float64),
                radii=np.array([1.0], dtype=np.float64),
                parents=np.array([-1], dtype=np.int32),
            )
            for _ in range(5)
        ]

        flip_swc_batch(swc_list, atlas_name="test_atlas")

        # Atlas should be instantiated only once
        assert mock_atlas_class.call_count == 1


class TestHemisphereEnum:
    """Tests for Hemisphere enum."""

    def test_hemisphere_values(self):
        """Test hemisphere enum values."""
        assert Hemisphere.LEFT.value == "left"
        assert Hemisphere.RIGHT.value == "right"
        assert Hemisphere.MIDLINE.value == "midline"
