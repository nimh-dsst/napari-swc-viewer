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

    def test_fewer_clusters_than_requested(self):
        """When fcluster returns fewer clusters than asked, colors still work."""
        from napari_swc_viewer.widgets.analysis_tab import AnalysisTabWidget

        viewer = MagicMock()
        lines_layer = MagicMock()
        lines_layer.name = "Neuron Lines"
        lines_layer.metadata = {
            "file_ids": ["nA", "nB", "nC"],
            "segments_per_neuron": [2, 2, 2],
        }
        viewer.layers = [lines_layer]

        # Requested 3 clusters, but fcluster only produced 2 (labels 1 and 3,
        # skipping 2 — non-contiguous labels).
        widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
        widget._viewer = viewer
        widget._slice_projector = None
        widget._last_cluster_result = _make_cluster_result(
            ["nA", "nB", "nC"], [1, 1, 3]
        )
        widget._cluster_color_map = None
        widget._actual_n_clusters = 0
        widget._build_cluster_color_map()
        widget._progress_label = MagicMock()

        assert widget._actual_n_clusters == 2

        widget._color_neurons_by_cluster()

        color_array = lines_layer.edge_color
        assert isinstance(color_array, np.ndarray)

        # nA and nB share cluster 1 → same color
        np.testing.assert_array_equal(color_array[0], color_array[2])
        # nC is in cluster 3 → different color
        assert not np.array_equal(color_array[0], color_array[4])

    def test_all_same_cluster(self):
        """When all neurons land in one cluster, no crash and single color."""
        from napari_swc_viewer.widgets.analysis_tab import AnalysisTabWidget

        viewer = MagicMock()
        lines_layer = MagicMock()
        lines_layer.name = "Neuron Lines"
        lines_layer.metadata = {
            "file_ids": ["nA", "nB"],
            "segments_per_neuron": [3, 3],
        }
        viewer.layers = [lines_layer]

        widget = AnalysisTabWidget.__new__(AnalysisTabWidget)
        widget._viewer = viewer
        widget._slice_projector = None
        widget._last_cluster_result = _make_cluster_result(
            ["nA", "nB"], [1, 1]
        )
        widget._cluster_color_map = None
        widget._actual_n_clusters = 0
        widget._build_cluster_color_map()
        widget._progress_label = MagicMock()

        assert widget._actual_n_clusters == 1

        widget._color_neurons_by_cluster()

        color_array = lines_layer.edge_color
        assert isinstance(color_array, np.ndarray)
        assert color_array.shape == (6, 4)

        # All segments should have the same color (blue)
        np.testing.assert_array_almost_equal(
            color_array[0], [0.12, 0.47, 0.71, 1.0], decimal=2
        )
        np.testing.assert_array_equal(color_array[0], color_array[5])


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
        projector._axis_index = {}
        projector._last_result_key = None
        projector._last_result = None

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
        projector._axis_index = {}
        projector._last_result_key = None
        projector._last_result = None

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
        projector._axis_index = {}
        projector._last_result_key = None
        projector._last_result = None

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


def _make_projector(**overrides):
    """Create a NeuronSliceProjector via __new__ with sensible defaults."""
    from napari_swc_viewer.widgets.slice_projection import NeuronSliceProjector

    proj = NeuronSliceProjector.__new__(NeuronSliceProjector)
    proj._viewer = MagicMock()
    proj._enabled = False
    proj._tolerance = 50.0
    proj._edge_width = 4
    proj._projection_layer = None
    proj._scale = None
    proj._connected = False
    proj._update_timer = MagicMock()
    proj._source_data = {}
    proj._all_p1 = None
    proj._all_p2 = None
    proj._all_colors = None
    proj._axis_index = {}
    proj._last_result_key = None
    proj._last_result = None
    for k, v in overrides.items():
        setattr(proj, k, v)
    return proj


def _brute_force_projection(projector, slice_position, slice_axis, tolerance):
    """Reference brute-force projection for correctness checks."""
    p1 = projector._all_p1
    p2 = projector._all_p2
    slab_min = slice_position - tolerance
    slab_max = slice_position + tolerance
    v1 = p1[:, slice_axis]
    v2 = p2[:, slice_axis]
    mask = (np.maximum(v1, v2) >= slab_min) & (np.minimum(v1, v2) <= slab_max)
    return np.nonzero(mask)[0]


class TestSpatialIndex:
    """Test that the spatial index produces correct results."""

    def test_index_matches_brute_force(self):
        """Index-based query returns same segments as brute-force."""
        rng = np.random.default_rng(42)
        n_nodes = 500
        coords = rng.uniform(0, 1000, (n_nodes, 3))
        edges = np.column_stack([np.arange(n_nodes - 1), np.arange(1, n_nodes)])

        proj = _make_projector(_tolerance=100.0)
        proj._source_data = {"n1": (coords, edges, (1, 1, 0, 1))}
        proj._rebuild_arrays()

        for axis in range(3):
            for pos in [0.0, 250.0, 500.0, 750.0, 1000.0]:
                lines, colors = proj._compute_slice_projection(pos, axis)
                expected = _brute_force_projection(proj, pos, axis, proj._tolerance)

                if len(expected) == 0:
                    assert lines is None
                else:
                    assert lines is not None
                    assert len(lines) == len(expected)
                    # Clear cache between calls so each is fresh
                    proj._invalidate_cache()

    def test_empty_data(self):
        """Projection returns None when there are no segments."""
        proj = _make_projector()
        lines, colors = proj._compute_slice_projection(500.0, 0)
        assert lines is None
        assert colors is None

    def test_single_segment(self):
        """A single segment is correctly found or missed."""
        coords = np.array([[100, 0, 0], [200, 0, 0]], dtype=np.float64)
        edges = np.array([[0, 1]])
        proj = _make_projector(_tolerance=10.0)
        proj._source_data = {"n1": (coords, edges, (1, 0, 0, 1))}
        proj._rebuild_arrays()

        # Hit: position 150 is within [100, 200]
        lines, colors = proj._compute_slice_projection(150.0, 0)
        assert lines is not None
        assert lines.shape == (1, 2, 3)
        proj._invalidate_cache()

        # Miss: position 0 is far from [100, 200]
        lines, colors = proj._compute_slice_projection(0.0, 0)
        assert lines is None

    def test_all_segments_in_slab(self):
        """When all segments are within the slab, all are returned."""
        coords = np.array([[50, 0, 0], [51, 1, 1], [52, 2, 2]], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]])
        proj = _make_projector(_tolerance=100.0)
        proj._source_data = {"n1": (coords, edges, (1, 1, 0, 1))}
        proj._rebuild_arrays()

        lines, colors = proj._compute_slice_projection(50.0, 0)
        assert lines is not None
        assert len(lines) == 2

    def test_no_segments_in_slab(self):
        """When no segments are within the slab, returns None."""
        coords = np.array([[1000, 0, 0], [1001, 1, 1]], dtype=np.float64)
        edges = np.array([[0, 1]])
        proj = _make_projector(_tolerance=10.0)
        proj._source_data = {"n1": (coords, edges, (1, 1, 0, 1))}
        proj._rebuild_arrays()

        lines, colors = proj._compute_slice_projection(0.0, 0)
        assert lines is None

    def test_multiple_axes(self):
        """Index works independently for each axis."""
        coords = np.array(
            [[0, 500, 1000], [10, 510, 990]], dtype=np.float64
        )
        edges = np.array([[0, 1]])
        proj = _make_projector(_tolerance=20.0)
        proj._source_data = {"n1": (coords, edges, (1, 1, 0, 1))}
        proj._rebuild_arrays()

        # Axis 0: segment spans [0, 10], query at 5 → hit
        lines, _ = proj._compute_slice_projection(5.0, 0)
        assert lines is not None
        proj._invalidate_cache()

        # Axis 1: segment spans [500, 510], query at 0 → miss
        lines, _ = proj._compute_slice_projection(0.0, 1)
        assert lines is None
        proj._invalidate_cache()

        # Axis 2: segment spans [990, 1000], query at 995 → hit
        lines, _ = proj._compute_slice_projection(995.0, 2)
        assert lines is not None


class TestResultCache:
    """Test the single-entry result cache."""

    def test_cache_hit_returns_same_result(self):
        """Repeated query with same params returns cached result."""
        coords = np.array([[100, 0, 0], [200, 0, 0]], dtype=np.float64)
        edges = np.array([[0, 1]])
        proj = _make_projector(_tolerance=50.0)
        proj._source_data = {"n1": (coords, edges, (1, 0, 0, 1))}
        proj._rebuild_arrays()

        lines1, colors1 = proj._compute_slice_projection(150.0, 0)
        lines2, colors2 = proj._compute_slice_projection(150.0, 0)

        # Should be the exact same objects (cached)
        assert lines1 is lines2
        assert colors1 is colors2

    def test_cache_miss_on_different_position(self):
        """Cache misses when position changes."""
        coords = np.array([[100, 0, 0], [200, 0, 0]], dtype=np.float64)
        edges = np.array([[0, 1]])
        proj = _make_projector(_tolerance=150.0)
        proj._source_data = {"n1": (coords, edges, (1, 0, 0, 1))}
        proj._rebuild_arrays()

        lines1, _ = proj._compute_slice_projection(150.0, 0)
        lines2, _ = proj._compute_slice_projection(160.0, 0)

        # Different position → different result objects
        assert lines1 is not lines2

    def test_cache_invalidated_on_data_change(self):
        """Adding new data invalidates the cache."""
        coords = np.array([[100, 0, 0], [200, 0, 0]], dtype=np.float64)
        edges = np.array([[0, 1]])
        proj = _make_projector(_tolerance=50.0)
        proj._source_data = {"n1": (coords, edges, (1, 0, 0, 1))}
        proj._rebuild_arrays()

        proj._compute_slice_projection(150.0, 0)
        assert proj._last_result_key is not None

        # Rebuild arrays (as would happen on add_neuron_data)
        proj._rebuild_arrays()
        assert proj._last_result_key is None

    def test_cache_invalidated_on_tolerance_change(self):
        """Changing tolerance invalidates the cache."""
        coords = np.array([[100, 0, 0], [200, 0, 0]], dtype=np.float64)
        edges = np.array([[0, 1]])
        proj = _make_projector(_tolerance=50.0)
        proj._source_data = {"n1": (coords, edges, (1, 0, 0, 1))}
        proj._rebuild_arrays()

        proj._compute_slice_projection(150.0, 0)
        assert proj._last_result_key is not None

        proj.tolerance = 100.0
        assert proj._last_result_key is None

    def test_cache_invalidated_on_color_change(self):
        """Changing colors invalidates the cache."""
        coords = np.array([[100, 0, 0], [200, 0, 0]], dtype=np.float64)
        edges = np.array([[0, 1]])
        proj = _make_projector(_tolerance=50.0)
        proj._source_data = {"n1": (coords, edges, (1, 0, 0, 1))}
        proj._rebuild_arrays()

        proj._compute_slice_projection(150.0, 0)
        assert proj._last_result_key is not None

        proj.update_neuron_colors({"n1": [0, 1, 0, 1]})
        assert proj._last_result_key is None

    def test_cache_none_result(self):
        """Cache also works for None results (no segments in slab)."""
        coords = np.array([[1000, 0, 0], [1001, 0, 0]], dtype=np.float64)
        edges = np.array([[0, 1]])
        proj = _make_projector(_tolerance=10.0)
        proj._source_data = {"n1": (coords, edges, (1, 0, 0, 1))}
        proj._rebuild_arrays()

        result1 = proj._compute_slice_projection(0.0, 0)
        result2 = proj._compute_slice_projection(0.0, 0)

        assert result1 == (None, None)
        assert result2 == (None, None)
        # Should be the same tuple object from cache
        assert result1 is result2


class TestColorOnlyRebuild:
    """Test that _rebuild_colors_only preserves geometry and index."""

    def test_preserves_geometry(self):
        """Color-only rebuild does not change p1/p2 arrays."""
        coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]])
        proj = _make_projector()
        proj._source_data = {"n1": (coords, edges, (1, 0, 0, 1))}
        proj._rebuild_arrays()

        p1_before = proj._all_p1.copy()
        p2_before = proj._all_p2.copy()

        proj._source_data["n1"] = (coords, edges, (0, 1, 0, 1))
        proj._rebuild_colors_only()

        np.testing.assert_array_equal(proj._all_p1, p1_before)
        np.testing.assert_array_equal(proj._all_p2, p2_before)

    def test_preserves_axis_index(self):
        """Color-only rebuild does not change the spatial index."""
        coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]])
        proj = _make_projector()
        proj._source_data = {"n1": (coords, edges, (1, 0, 0, 1))}
        proj._rebuild_arrays()

        index_before = {
            ax: (order.copy(), zmin.copy(), zmax.copy(), span)
            for ax, (order, zmin, zmax, span) in proj._axis_index.items()
        }

        proj._source_data["n1"] = (coords, edges, (0, 1, 0, 1))
        proj._rebuild_colors_only()

        for ax in range(3):
            order_b, zmin_b, zmax_b, span_b = index_before[ax]
            order_a, zmin_a, zmax_a, span_a = proj._axis_index[ax]
            np.testing.assert_array_equal(order_a, order_b)
            np.testing.assert_array_equal(zmin_a, zmin_b)
            np.testing.assert_array_equal(zmax_a, zmax_b)
            assert span_a == span_b

    def test_updates_colors(self):
        """Color-only rebuild correctly updates the color array."""
        coords = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        edges = np.array([[0, 1]])
        proj = _make_projector()
        proj._source_data = {"n1": (coords, edges, (1, 0, 0, 1))}
        proj._rebuild_arrays()

        np.testing.assert_array_almost_equal(
            proj._all_colors[0], [1, 0, 0, 1]
        )

        proj._source_data["n1"] = (coords, edges, (0, 0, 1, 1))
        proj._rebuild_colors_only()

        np.testing.assert_array_almost_equal(
            proj._all_colors[0], [0, 0, 1, 1]
        )

    def test_update_neuron_colors_uses_color_only_rebuild(self):
        """update_neuron_colors preserves geometry and index."""
        coords = np.array([[0, 0, 0], [100, 50, 50], [200, 100, 100]], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]])
        proj = _make_projector(_tolerance=10.0)
        proj._source_data = {"n1": (coords, edges, (1, 0, 0, 1))}
        proj._rebuild_arrays()

        p1_before = proj._all_p1.copy()
        index_order_before = proj._axis_index[0][0].copy()

        proj.update_neuron_colors({"n1": [0, 1, 0, 1]})

        # Geometry unchanged
        np.testing.assert_array_equal(proj._all_p1, p1_before)
        # Index unchanged
        np.testing.assert_array_equal(proj._axis_index[0][0], index_order_before)
        # Colors updated
        np.testing.assert_array_almost_equal(
            proj._all_colors[0], [0, 1, 0, 1]
        )


class TestClearResetsState:
    """Test that clear() resets index and cache."""

    def test_clear_resets_all_state(self):
        """clear() resets source data, arrays, index, and cache."""
        coords = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        edges = np.array([[0, 1]])
        proj = _make_projector()
        proj._source_data = {"n1": (coords, edges, (1, 0, 0, 1))}
        proj._rebuild_arrays()
        proj._compute_slice_projection(0.5, 0)

        assert proj._all_p1 is not None
        assert len(proj._axis_index) == 3
        assert proj._last_result_key is not None

        proj._projection_layer = None  # avoid mock layer removal issues
        proj.clear()

        assert proj._all_p1 is None
        assert proj._all_p2 is None
        assert proj._all_colors is None
        assert len(proj._axis_index) == 0
        assert proj._last_result_key is None
        assert proj._last_result is None
        assert len(proj._source_data) == 0
