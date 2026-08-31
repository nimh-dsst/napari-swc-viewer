"""Basic import tests for napari-neuron-navigator."""


def test_import():
    """Test that the package can be imported."""
    import napari_neuron_navigator

    assert napari_neuron_navigator.__version__ is not None


def test_napari_import():
    """Test that napari can be imported (validates installation)."""
    import napari

    assert napari.__version__ is not None
