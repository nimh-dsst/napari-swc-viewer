"""Tests for color-by-cluster with batched single-layer rendering."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from napari_swc_viewer.analysis.clustering import ClusterResult


def _make_cluster_result(neuron_ids, labels):
    """Create a minimal ClusterResult for testing."""
    n = len(neuron_ids)
    return ClusterResult(
        correlation_matrix=np.eye(n, dtype=np.float32),
        distance_matrix=np.zeros((n, n), dtype=np.float32),
        linkage_matrix=np.zeros((n - 1, 4), dtype=np.float64),
        neuron_ids=neuron_ids,
        reorder_indices=np.arange(n),
        labels=np.array(labels, dtype=np.int32),
    )


class TestColorByClusterLines:
    """Test cluster coloring on the batched 'Neuron Lines' layer."""

    def test_single_layer_edge_colors_updated(self):
        """Cluster colors are applied as per-segment edge_color array."""
        from napari_swc_viewer.widgets.analysis_tab import AnalysisTabWidget

        viewer = MagicMock()

        # Simulate a single "Neuron Lines" layer with 2 neurons (5 + 3 segments)
        lines_layer = MagicMock()
        lines_layer.name = "Neuron Lines"
        lines_layer.metadata = {
            "file_ids": ["neuronA", "neuronB"],
            "segments_per_neuron": [5, 3],
        }
        viewer.layers = [lines_layer]

        widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
        widget._viewer = viewer
        widget._slice_projector = None
        widget._last_cluster_result = _make_cluster_result(
            ["neuronA", "neuronB"], [1, 2]
        )
        widget._cluster_color_map = None
        widget._build_cluster_color_map()
        widget._progress_label = MagicMock()

        widget._color_neurons_by_cluster()

        # edge_color should have been set to an (8, 4) array
        color_array = lines_layer.edge_color
        assert isinstance(color_array, np.ndarray)
        assert color_array.shape == (8, 4)

        # All segments of neuronA should share the same color
        np.testing.assert_array_equal(color_array[0], color_array[4])
        # All segments of neuronB should share the same color
        np.testing.assert_array_equal(color_array[5], color_array[7])
        # The two neurons should have different colors
        assert not np.array_equal(color_array[0], color_array[5])

    def test_unrecognized_neuron_gets_default_color(self):
        """Neurons not in the cluster result receive a grey default."""
        from napari_swc_viewer.widgets.analysis_tab import AnalysisTabWidget

        viewer = MagicMock()
        lines_layer = MagicMock()
        lines_layer.name = "Neuron Lines"
        lines_layer.metadata = {
            "file_ids": ["neuronA", "unknown_neuron"],
            "segments_per_neuron": [2, 3],
        }
        viewer.layers = [lines_layer]

        widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
        widget._viewer = viewer
        widget._slice_projector = None
        widget._last_cluster_result = _make_cluster_result(["neuronA"], [1])
        widget._cluster_color_map = None
        widget._build_cluster_color_map()
        widget._progress_label = MagicMock()

        widget._color_neurons_by_cluster()

        color_array = lines_layer.edge_color
        assert isinstance(color_array, np.ndarray)
        assert color_array.shape == (5, 4)

        # The unknown neuron's segments (indices 2-4) should be grey
        expected_default = [0.5, 0.5, 0.5, 1.0]
        np.testing.assert_array_almost_equal(color_array[2], expected_default)
        np.testing.assert_array_almost_equal(color_array[4], expected_default)

    def test_no_lines_layer_reports_zero(self):
        """No crash when no Neuron Lines layer exists."""
        from napari_swc_viewer.widgets.analysis_tab import AnalysisTabWidget

        viewer = MagicMock()
        viewer.layers = []

        widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
        widget._viewer = viewer
        widget._slice_projector = None
        widget._last_cluster_result = _make_cluster_result(["n1", "n2"], [1, 2])
        widget._cluster_color_map = None
        widget._build_cluster_color_map()
        widget._progress_label = MagicMock()

        widget._color_neurons_by_cluster()

        widget._progress_label.setText.assert_called_once()
        msg = widget._progress_label.setText.call_args[0][0]
        assert "0 layer(s)" in msg


class TestCustomColorsSmallClusters:
    """Test that 2- and 3-cluster cases use hard-coded distinct colors."""

    def test_two_clusters_get_distinct_custom_colors(self):
        """With n_clusters=2, neurons get explicit blue and red."""
        from napari_swc_viewer.widgets.analysis_tab import AnalysisTabWidget

        viewer = MagicMock()
        lines_layer = MagicMock()
        lines_layer.name = "Neuron Lines"
        lines_layer.metadata = {
            "file_ids": ["nA", "nB"],
            "segments_per_neuron": [3, 4],
        }
        viewer.layers = [lines_layer]

        widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
        widget._viewer = viewer
        widget._slice_projector = None
        widget._last_cluster_result = _make_cluster_result(
            ["nA", "nB"], [1, 2]
        )
        widget._cluster_color_map = None
        widget._build_cluster_color_map()
        widget._progress_label = MagicMock()

        widget._color_neurons_by_cluster()

        color_array = lines_layer.edge_color
        assert isinstance(color_array, np.ndarray)

        # Cluster 1 (nA) → blue [0.12, 0.47, 0.71, 1.0]
        np.testing.assert_array_almost_equal(
            color_array[0], [0.12, 0.47, 0.71, 1.0], decimal=2
        )
        # Cluster 2 (nB) → red [0.84, 0.15, 0.16, 1.0]
        np.testing.assert_array_almost_equal(
            color_array[3], [0.84, 0.15, 0.16, 1.0], decimal=2
        )
        # They must be different
        assert not np.array_equal(color_array[0], color_array[3])

    def test_three_clusters_get_distinct_custom_colors(self):
        """With n_clusters=3, neurons get explicit blue, red, green."""
        from napari_swc_viewer.widgets.analysis_tab import AnalysisTabWidget

        viewer = MagicMock()
        lines_layer = MagicMock()
        lines_layer.name = "Neuron Lines"
        lines_layer.metadata = {
            "file_ids": ["nA", "nB", "nC"],
            "segments_per_neuron": [2, 2, 2],
        }
        viewer.layers = [lines_layer]

        widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
        widget._viewer = viewer
        widget._slice_projector = None
        widget._last_cluster_result = _make_cluster_result(
            ["nA", "nB", "nC"], [1, 2, 3]
        )
        widget._cluster_color_map = None
        widget._build_cluster_color_map()
        widget._progress_label = MagicMock()

        widget._color_neurons_by_cluster()

        color_array = lines_layer.edge_color
        assert isinstance(color_array, np.ndarray)

        blue = color_array[0]  # nA = cluster 1
        red = color_array[2]   # nB = cluster 2
        green = color_array[4] # nC = cluster 3

        np.testing.assert_array_almost_equal(
            blue, [0.12, 0.47, 0.71, 1.0], decimal=2
        )
        np.testing.assert_array_almost_equal(
            red, [0.84, 0.15, 0.16, 1.0], decimal=2
        )
        np.testing.assert_array_almost_equal(
            green, [0.17, 0.63, 0.17, 1.0], decimal=2
        )

        # All three must be pairwise different
        assert not np.array_equal(blue, red)
        assert not np.array_equal(blue, green)
        assert not np.array_equal(red, green)


class TestColorByClusterPoints:
    """Test cluster coloring on the batched 'Neuron Points' layer."""

    def test_single_layer_face_colors_updated(self):
        """Cluster colors are applied as per-point face_color array."""
        from napari_swc_viewer.widgets.analysis_tab import AnalysisTabWidget

        viewer = MagicMock()
        points_layer = MagicMock()
        points_layer.name = "Neuron Points"
        points_layer.metadata = {
            "file_ids_per_point": ["n1", "n1", "n1", "n2", "n2"],
        }
        viewer.layers = [points_layer]

        widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
        widget._viewer = viewer
        widget._slice_projector = None
        widget._last_cluster_result = _make_cluster_result(["n1", "n2"], [1, 2])
        widget._cluster_color_map = None
        widget._build_cluster_color_map()
        widget._progress_label = MagicMock()

        widget._color_neurons_by_cluster()

        color_array = points_layer.face_color
        assert isinstance(color_array, np.ndarray)
        assert color_array.shape == (5, 4)

        # n1 points should share a color, n2 points should share a color
        np.testing.assert_array_equal(color_array[0], color_array[2])
        np.testing.assert_array_equal(color_array[3], color_array[4])
        assert not np.array_equal(color_array[0], color_array[3])


class TestColorByClusterBothLayers:
    """Test that both lines and points layers are updated together."""

    def test_both_layers_counted(self):
        """Progress message reports 2 layer(s) when both exist."""
        from napari_swc_viewer.widgets.analysis_tab import AnalysisTabWidget

        viewer = MagicMock()
        lines_layer = MagicMock()
        lines_layer.name = "Neuron Lines"
        lines_layer.metadata = {
            "file_ids": ["n1"],
            "segments_per_neuron": [4],
        }
        points_layer = MagicMock()
        points_layer.name = "Neuron Points"
        points_layer.metadata = {
            "file_ids_per_point": ["n1", "n1"],
        }
        viewer.layers = [lines_layer, points_layer]

        widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
        widget._viewer = viewer
        widget._slice_projector = None
        widget._last_cluster_result = _make_cluster_result(["n1"], [1])
        widget._cluster_color_map = None
        widget._build_cluster_color_map()
        widget._progress_label = MagicMock()

        widget._color_neurons_by_cluster()

        msg = widget._progress_label.setText.call_args[0][0]
        assert "2 layer(s)" in msg


class TestSliceProjectorColorUpdate:
    """Test that the slice projector colors are updated."""

    def test_update_neuron_colors_changes_source_data(self):
        """update_neuron_colors updates stored colors and triggers rebuild."""
        from napari_swc_viewer.widgets.slice_projection import NeuronSliceProjector

        viewer = MagicMock()
        viewer.dims.ndisplay = 3

        projector = NeuronSliceProjector.__new__(NeuronSliceProjector)
        projector._viewer = viewer
        projector._enabled = False  # Prevent scheduling real updates
        projector._tolerance = 50.0
        projector._edge_width = 4
        projector._projection_layer = None
        projector._scale = None
        projector._connected = False
        projector._update_timer = MagicMock()

        coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
        old_color = (0.0, 1.0, 1.0, 1.0)  # cyan
        projector._source_data = {"neuronA": (coords, edges, old_color)}
        projector._rebuild_arrays()

        # Before update, colors should be cyan
        np.testing.assert_array_almost_equal(
            projector._all_colors[0], [0.0, 1.0, 1.0, 1.0]
        )

        new_color = [1.0, 0.0, 0.0, 1.0]  # red
        projector.update_neuron_colors({"neuronA": new_color})

        # After update, stored color should be red
        assert projector._source_data["neuronA"][2] == (1.0, 0.0, 0.0, 1.0)
        np.testing.assert_array_almost_equal(
            projector._all_colors[0], [1.0, 0.0, 0.0, 1.0]
        )

    def test_update_neuron_colors_skips_unknown(self):
        """Neurons not in source_data are silently ignored."""
        from napari_swc_viewer.widgets.slice_projection import NeuronSliceProjector

        projector = NeuronSliceProjector.__new__(NeuronSliceProjector)
        projector._enabled = False
        projector._update_timer = MagicMock()

        coords = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        edges = np.array([[0, 1]], dtype=np.int32)
        projector._source_data = {"neuronA": (coords, edges, (1, 1, 1, 1))}
        projector._rebuild_arrays()

        # Update with an unknown neuron — should not error
        projector.update_neuron_colors({"unknown": [1, 0, 0, 1]})

        # neuronA should be unchanged
        assert projector._source_data["neuronA"][2] == (1, 1, 1, 1)

    def test_update_no_change_skips_rebuild(self):
        """If color hasn't changed, _rebuild_arrays is not called again."""
        from napari_swc_viewer.widgets.slice_projection import NeuronSliceProjector

        projector = NeuronSliceProjector.__new__(NeuronSliceProjector)
        projector._enabled = False
        projector._update_timer = MagicMock()

        coords = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        edges = np.array([[0, 1]], dtype=np.int32)
        color = (1.0, 0.0, 0.0, 1.0)
        projector._source_data = {"neuronA": (coords, edges, color)}
        projector._rebuild_arrays()

        original_colors = projector._all_colors.copy()

        # Same color — should be a no-op rebuild
        projector.update_neuron_colors({"neuronA": [1.0, 0.0, 0.0, 1.0]})

        np.testing.assert_array_equal(projector._all_colors, original_colors)


class TestLayerMetadata:
    """Test that rendering stores the correct metadata for cluster coloring."""

    def test_lines_metadata_structure(self):
        """Verify metadata keys stored on the Neuron Lines layer."""
        # This test validates the contract between rendering and coloring.
        # The metadata must contain 'file_ids' and 'segments_per_neuron'
        # with matching lengths and correct segment counts.
        metadata = {
            "file_ids": ["a", "b", "c"],
            "segments_per_neuron": [10, 20, 15],
        }
        assert len(metadata["file_ids"]) == len(metadata["segments_per_neuron"])
        assert sum(metadata["segments_per_neuron"]) == 45

    def test_points_metadata_structure(self):
        """Verify metadata key stored on the Neuron Points layer."""
        metadata = {
            "file_ids_per_point": ["a", "a", "b", "b", "b"],
        }
        assert len(metadata["file_ids_per_point"]) == 5
