"""napari widgets for the SWC viewer."""

from .analysis_tab import AnalysisTabWidget
from .neuron_table import NeuronTableWidget
from .neuron_viewer import NeuronViewerWidget
from .reference_layers import add_allen_template, add_region_mesh, add_region_segmentation
from .region_selector import RegionSelectorWidget
from .slice_projection import NeuronSliceProjector

__all__ = [
    "AnalysisTabWidget",
    "NeuronTableWidget",
    "RegionSelectorWidget",
    "NeuronViewerWidget",
    "NeuronSliceProjector",
    "add_allen_template",
    "add_region_mesh",
    "add_region_segmentation",
]
