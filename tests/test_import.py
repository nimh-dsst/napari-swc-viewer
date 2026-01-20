"""Basic import tests for napari-swc-viewer."""


def test_import():
    """Test that the package can be imported."""
    import napari_swc_viewer

    assert napari_swc_viewer.__version__ is not None


def test_napari_import():
    """Test that napari can be imported (validates installation)."""
    import napari

    assert napari.__version__ is not None
