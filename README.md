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
- Additional source columns are preserved in the standardized Parquet
- BLTR batch conversions also append `origin_csv` so point provenance can be
  selected at import time

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

To append another raw CSV into an existing standardized point Parquet with the
same schema, use `--append`:

```bash
pixi run python scripts/convert_point_csv.py --append new_points.csv mapping.json points.parquet
```

For BLTR-format two-row-header CSV directories like [`bltr cases`](bltr%20cases),
use the batch converter instead:

```bash
pixi run python scripts/convert_bltr_point_csv_directory.py "bltr cases" bltr_combined.parquet
```

In the plugin Data tab, use `Create From Directory...` or `Create From File(s)...`
to build a new point Parquet from CSV inputs. Point CSV conversions record
provenance in `origin_csv`, including single-file conversions. The conversion flow first tries known header
formats automatically (`label/x/y/z/...` standardized headers or BLTR two-row
headers) and only asks for a mapping JSON if the CSVs cannot be inferred from
headers. Use `Open Point Parquet...` to preview an existing standardized
Parquet, and `Append Point file` to add either a raw CSV or another
standardized point Parquet onto an existing point Parquet and save the combined
result as a new Parquet file. If the input is CSV, the flow auto-detects
standard headers first and only asks for mapping JSON if needed. If the input
is Parquet, its ordered columns and Arrow types must exactly match the target
Parquet schema. If a point Parquet is already loaded in the preview, the append
flow uses that file as the source by default and goes straight to a save-as
destination. When appending CSV onto an older point Parquet that predates
`origin_csv`, the output is upgraded to include the column and legacy rows are
marked as `(not recorded)`. Opening a point Parquet populates a preview table
with `Label`, `Origin CSV`, and `Points`.
Select one or more rows and click `Import Selected Heatmaps` to create only
those heatmap image layers. The selected atlas is only required for heatmap
import and atlas validation. If optional atlas metadata columns are present,
the app validates the selected subset against the loaded atlas and warns on
mismatches. For older point Parquets without `origin_csv`, the table shows
`(not recorded)` and still imports one heatmap layer per selected `label`.

In the `Tools` tab, eligible native-grid heatmaps can be turned into blurred
napari `Image` layers. In the `Histogram` tab, those same eligible heatmaps and
blurred heatmaps can be inspected as overlaid interactive intensity histograms
and then converted into 3D binary mask `Labels` layers using explicit lower and
optional upper intensity bounds. The histogram view supports zooming and panning
so narrow intensity ranges are easier to inspect. A typical flow is: select a
heatmap in `Tools`, create a blurred layer, open `Histogram`, inspect the
distribution, optionally tune the blurred layer's `contrast_limits` in napari,
copy those limits into the threshold bounds, and then create the mask. The
`Regions` tab can then query neurons either by Allen regions or by one or more
of these generated mask layers, using either any-node or soma-only membership.

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

## Third Party Attribution

This repository contains modified code derived from [napari swc editor](https://github.com/LaboratoryOpticsBiosciences/napari-swc-editor) by Clément Caporal.
