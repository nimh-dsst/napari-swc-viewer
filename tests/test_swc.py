"""Tests for SWC file parsing."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from napari_swc_viewer.swc import NodeType, SWCData, parse_swc, write_swc


@pytest.fixture
def sample_swc_content():
    """Sample SWC file content."""
    return """# Sample SWC file
# id type x y z radius parent
1 1 100.0 200.0 300.0 5.0 -1
2 3 110.0 210.0 310.0 2.0 1
3 3 120.0 220.0 320.0 1.5 2
4 3 130.0 230.0 330.0 1.0 3
5 2 105.0 195.0 295.0 2.0 1
"""


@pytest.fixture
def sample_swc_file(sample_swc_content, tmp_path):
    """Create a temporary SWC file."""
    swc_file = tmp_path / "sample.swc"
    swc_file.write_text(sample_swc_content)
    return swc_file


class TestSWCData:
    """Tests for SWCData class."""

    def test_n_nodes(self):
        """Test n_nodes property."""
        swc = SWCData(
            ids=np.array([1, 2, 3], dtype=np.int32),
            types=np.array([1, 3, 3], dtype=np.int32),
            coords=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64),
            radii=np.array([1.0, 0.5, 0.5], dtype=np.float64),
            parents=np.array([-1, 1, 2], dtype=np.int32),
        )
        assert swc.n_nodes == 3

    def test_soma_mask(self):
        """Test soma_mask property."""
        swc = SWCData(
            ids=np.array([1, 2, 3], dtype=np.int32),
            types=np.array([1, 3, 3], dtype=np.int32),
            coords=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64),
            radii=np.array([1.0, 0.5, 0.5], dtype=np.float64),
            parents=np.array([-1, 1, 2], dtype=np.int32),
        )
        mask = swc.soma_mask
        assert mask[0] is np.True_
        assert mask[1] is np.False_
        assert mask[2] is np.False_

    def test_soma_coords(self):
        """Test soma_coords property."""
        swc = SWCData(
            ids=np.array([1, 2, 3], dtype=np.int32),
            types=np.array([1, 3, 3], dtype=np.int32),
            coords=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64),
            radii=np.array([1.0, 0.5, 0.5], dtype=np.float64),
            parents=np.array([-1, 1, 2], dtype=np.int32),
        )
        soma_coords = swc.soma_coords
        assert soma_coords.shape == (1, 3)
        np.testing.assert_array_equal(soma_coords[0], [0, 0, 0])

    def test_root_mask(self):
        """Test root_mask property."""
        swc = SWCData(
            ids=np.array([1, 2, 3], dtype=np.int32),
            types=np.array([1, 3, 3], dtype=np.int32),
            coords=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64),
            radii=np.array([1.0, 0.5, 0.5], dtype=np.float64),
            parents=np.array([-1, 1, 2], dtype=np.int32),
        )
        mask = swc.root_mask
        assert mask[0] is np.True_
        assert mask[1] is np.False_
        assert mask[2] is np.False_

    def test_copy(self):
        """Test copy method creates independent copy."""
        swc = SWCData(
            ids=np.array([1, 2], dtype=np.int32),
            types=np.array([1, 3], dtype=np.int32),
            coords=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64),
            radii=np.array([1.0, 0.5], dtype=np.float64),
            parents=np.array([-1, 1], dtype=np.int32),
        )
        copy = swc.copy()

        # Modify original
        swc.coords[0, 0] = 999

        # Copy should be unchanged
        assert copy.coords[0, 0] == 0


class TestParseSWC:
    """Tests for parse_swc function."""

    def test_parse_basic(self, sample_swc_file):
        """Test parsing a basic SWC file."""
        swc = parse_swc(sample_swc_file)

        assert swc.n_nodes == 5
        np.testing.assert_array_equal(swc.ids, [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(swc.types, [1, 3, 3, 3, 2])
        np.testing.assert_array_equal(swc.parents, [-1, 1, 2, 3, 1])

    def test_parse_coords(self, sample_swc_file):
        """Test that coordinates are parsed correctly."""
        swc = parse_swc(sample_swc_file)

        np.testing.assert_array_almost_equal(swc.coords[0], [100.0, 200.0, 300.0])
        np.testing.assert_array_almost_equal(swc.coords[1], [110.0, 210.0, 310.0])

    def test_parse_radii(self, sample_swc_file):
        """Test that radii are parsed correctly."""
        swc = parse_swc(sample_swc_file)

        np.testing.assert_array_almost_equal(swc.radii, [5.0, 2.0, 1.5, 1.0, 2.0])

    def test_parse_string_path(self, sample_swc_file):
        """Test parsing with string path."""
        swc = parse_swc(str(sample_swc_file))
        assert swc.n_nodes == 5

    def test_parse_skips_comments(self, tmp_path):
        """Test that comment lines are skipped."""
        content = """# Comment 1
## Another comment
# Yet another
1 1 0.0 0.0 0.0 1.0 -1
"""
        swc_file = tmp_path / "comments.swc"
        swc_file.write_text(content)

        swc = parse_swc(swc_file)
        assert swc.n_nodes == 1


class TestWriteSWC:
    """Tests for write_swc function."""

    def test_write_basic(self, tmp_path):
        """Test writing a basic SWC file."""
        swc = SWCData(
            ids=np.array([1, 2], dtype=np.int32),
            types=np.array([1, 3], dtype=np.int32),
            coords=np.array([[100.0, 200.0, 300.0], [110.0, 210.0, 310.0]]),
            radii=np.array([5.0, 2.0], dtype=np.float64),
            parents=np.array([-1, 1], dtype=np.int32),
        )

        output_file = tmp_path / "output.swc"
        write_swc(swc, output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "100.000" in content
        assert "200.000" in content

    def test_roundtrip(self, sample_swc_file, tmp_path):
        """Test that parse -> write -> parse gives same result."""
        swc1 = parse_swc(sample_swc_file)

        output_file = tmp_path / "roundtrip.swc"
        write_swc(swc1, output_file)

        swc2 = parse_swc(output_file)

        np.testing.assert_array_equal(swc1.ids, swc2.ids)
        np.testing.assert_array_equal(swc1.types, swc2.types)
        np.testing.assert_array_almost_equal(swc1.coords, swc2.coords, decimal=3)
        np.testing.assert_array_almost_equal(swc1.radii, swc2.radii, decimal=3)
        np.testing.assert_array_equal(swc1.parents, swc2.parents)


class TestNodeType:
    """Tests for NodeType constants."""

    def test_node_types(self):
        """Test that node type constants have expected values."""
        assert NodeType.UNDEFINED == 0
        assert NodeType.SOMA == 1
        assert NodeType.AXON == 2
        assert NodeType.BASAL_DENDRITE == 3
        assert NodeType.APICAL_DENDRITE == 4
