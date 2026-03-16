# napari-swc-viewer
A Napari plugin that allows viewing of SWC files in napari

## Prerequisites

### Installing Pixi

This project uses [Pixi](https://pixi.sh) for environment and dependency management. To install Pixi, follow the instructions at:

https://pixi.sh/latest/#installation

## Installation

### Building and Installing the Package

To build and install the package in development mode:

```bash
pixi run build
```

This will install the package in editable mode, allowing you to make changes without reinstalling.

### Running Napari

To launch napari with the plugin installed:

```bash
pixi run napari
```

This command will automatically build the package (if needed) before launching napari.

### Running Tests

To run the test suite:

```bash
pixi run test
```

To run tests with coverage:

```bash
pixi run test-cov
```

## Standard Point Parquet Workflow

This repo now supports atlas-registered point datasets via a standardized
point Parquet schema:

- Required columns: `label`, `x`, `y`, `z`
- Optional columns: `region_name`, `acronym`, `id`, `hemisphere`
- Additional source columns are preserved and imported as point properties

To convert a raw CSV into standardized point Parquet, provide a JSON
mapping from standard target names to source CSV headers:

```json
{
  "label": "marker",
  "x": "atlas_x",
  "y": "atlas_y",
  "z": "atlas_z",
  "region_name": "region_name",
  "acronym": "acronym",
  "id": "id",
  "hemisphere": "hemisphere"
}
```

Run the converter from the repository root:

```bash
pixi run python scripts/convert_point_csv.py raw_points.csv mapping.json points.parquet
```

In the plugin Data tab, use `Import Point Parquet...` to load the standardized
Parquet. The selected atlas must already be loaded. If optional atlas metadata
columns are present, the app validates them against the selected atlas,
warns on mismatches, and still imports one `Points` layer per unique `label`.

## Hemisphere Detection and Coordinate Flipping

This plugin includes functionality to detect which brain hemisphere an SWC morphology is located in and to flip coordinates from one hemisphere to the other. This is useful for standardizing neuron reconstructions to a common hemisphere for analysis.

### Midline Calculation

The midline (midsagittal plane) is calculated based on the Allen Mouse Brain Common Coordinate Framework v3 (CCFv3) convention:

- **Coordinate system**: The CCFv3 defines a reference image with origin at (0, 0, 0), spacing of 10 µm/voxel, and size (1320, 800, 1140) voxels
- **Axes**: Rostral-to-caudal, dorsal-to-ventral, and left-to-right
- **Voxel convention**: Following the ITK convention, voxel positions are defined at voxel **centers**, so position (0, 0, 0) is at the center of the first voxel

The midline coordinate is calculated as the midpoint between the first and last voxel centers:

```
midline = ((shape - 1) * resolution) / 2
```

For the `allen_mouse_10um` atlas (shape=1140 along the left-right axis):
- First voxel center: 0.0 µm
- Last voxel center: (1140 - 1) × 10 = 11390.0 µm
- **Midline: 5695.0 µm**

### Coordinate Flipping

Coordinates are flipped by reflecting across the midline plane:

```
flipped_coordinate = 2 × midline - original_coordinate
```

This operation is vectorized using NumPy for efficient processing of large SWC files (10,000+ nodes).
