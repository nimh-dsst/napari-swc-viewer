"""napari widgets for the SWC viewer."""

from .neuron_viewer import NeuronViewerWidget
from .reference_layers import add_allen_template, add_region_mesh
from .region_selector import RegionSelectorWidget
from .slice_projection import NeuronSliceProjector

__all__ = [
    "RegionSelectorWidget",
    "NeuronViewerWidget",
    "NeuronSliceProjector",
    "add_allen_template",
    "add_region_mesh",
]
