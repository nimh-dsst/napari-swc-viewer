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

To enable plugin debug logging for runtime diagnostics, including detached
flatmap viewer lifecycle events:

```bash
NAPARI_SWC_VIEWER_DEBUG=1 pixi run napari
```

To write the debug trace to a custom file:

```bash
NAPARI_SWC_VIEWER_DEBUG=1 NAPARI_SWC_VIEWER_LOG_FILE=/tmp/napari-swc-viewer.log pixi run napari
```

The default file is `~/.napari-swc-viewer/debug.log`. It rotates at 10 MB and
keeps three backups. Plugin DEBUG records are written to the file and console;
focused napari layer-slicer DEBUG records are added to the file only.

Detached viewers are created hidden in 3D and shown only after their first
flatmap layer is configured. A normal first render records
`event=created_hidden`, `event=first_layer_ready`, `event=show_scheduled`, and
`event=shown` in that order. `pending_first_show=false` on the final record
confirms that the viewer left its hidden setup state.

On macOS the detached window is shown through a guarded normal-window path so
it never inherits napari's app-wide saved fullscreen state. A normal render
there reports `event=normal_show_requested`, `event=fullscreen_restore_suppressed`,
and `show_path=normal_qt` on `event=shown` (other platforms report
`show_path=napari`). If a user manually places the detached window in fullscreen
and closes it, the close is briefly deferred so the native surface can return to
normal before teardown: search for `event=fullscreen_close_deferred`,
`event=fullscreen_exit_requested`, `event=fullscreen_exit_complete`, and
`event=fullscreen_close_retried`. An `event=fullscreen_guard_failure` record
marks any guarded fallback (missing Qt/napari private interface or a
fullscreen-exit timeout). Each snapshot also reports `qt_fullscreen`,
`qt_window_state`, `napari_saved_fullscreen`, and `fullscreen_close_state`.

To diagnose a detached flatmap window that does not close, launch with a custom
log path, create a projection, close **SWC Viewer Flatmap**, wait at least two
seconds, and then close the main napari window. Search the retained trace for
`flatmap_viewer_lifecycle event=qt_close`, the subsequent
`event=close_checkpoint` records, `event=qt_deferreddelete`,
`_LayerSlicer.shutdown`, `event=cleanup_complete`, and the three
`event=post_destroy_checkpoint` records. A successful accepted close reports
`cleanup_trigger=deferred_delete`, `cleanup_qt_viewer=closed`, zero retained
layers, `napari_viewer_registered=false`, both plugin viewer-reference fields
as `false`, and `slicer_executor_shutdown=true`. At the 2000 ms post-destroy
checkpoint, both `qt_matching_top_level_widgets` and
`qt_matching_native_windows` should be `empty`.

For a complete CPD2 walkthrough covering clone/install, `pixi run napari`,
left-hemisphere SWC-to-Parquet conversion, atlas loading, region queries, soma
clustering, and GPe-limited cluster heatmaps, see
[docs/cpd2_workflow.md](docs/cpd2_workflow.md).

### Running Tests

To run the test suite:

```bash
pixi run test
```

To run tests with coverage:

```bash
pixi run test-cov
```

### Test Data Attribution

`tests/test_hemisphere_integration.py` uses a vendored SWC test fixture from the
Brain Image Library. The committed fixture files live under
`tests/data/hemisphere/`.

Source dataset DOI: https://doi.org/10.35077/g.73

## Direct Flatmap/Depth NRRD Loading

Flatmap and depth lookup NRRD files can be opened directly through napari's
File Open dialog or drag-and-drop. Shape-based detection is used:

- A 4D flatmap volume with a length-2 coordinate axis is split into two scalar
  image layers: `Flatmap X: <stem>` and `Flatmap Y: <stem>`.
- A 3D depth volume is loaded as one scalar image layer: `Depth: <stem>`.

Unsupported NRRD shapes are rejected with a clear error. Direct NRRD image
layers are displayed in voxel/pixel space, matching the plugin's reference
image layers.

The primary flatmap workflow preprocesses an entire neuron Parquet once. In the
Flatmap tab, choose the directory containing `flatmap_both_shaped.nrrd`,
`flatmap_both_square.nrrd`, and `depth.nrrd`, then use **Prepare Whole
Parquet...**. The suggested output is `<source>_flatmap.parquet`; replacing the
source requires explicit confirmation. Table queries and selections do not
limit preprocessing. The same bilateral conversion is available from the
command line:

```bash
pixi run python scripts/add_flatmap_columns_to_parquet.py neurons.parquet neurons_flatmap.parquet --lookup-dir /path/to/lookups
```

The version-3 output preserves CCFv3 coordinates, custom columns, and schema
metadata. It appends independent `x_flat_*`, `y_flat_*`, validity, invalid-code,
and lookup-mode columns for shaped and square bilateral maps, plus shared
`depth_um` validity/provenance columns. XY always comes from the original voxel;
when necessary, only depth is recovered from the mirrored voxel. Metadata
records canonical bounds, transforms, source hashes, and a portable lookup-set
ID. The old `--flatmap`/`--depth` single-style command remains available for
legacy files.

SWC conversion can perform both steps as one atomic background operation:
enable **Add bilateral flatmap/depth columns** and select **Lookup directory...**
before choosing SWC files. Cancellation or failure removes the temporary output
and does not replace an existing Parquet.

## Precomputed Flatmap Region Cache

Use **Build Cache Profile...** to project an exactly matching BrainGlobe atlas
annotation into fixed shaped and square render grids. A cache directory has a
`flatmap-region-cache.json` manifest and may hold multiple atlas/lookup/grid
profiles. Each profile stores memory-mappable sparse label occupancy, closed
voxel-faithful surfaces, and per-depth outlines. The defaults are 256 XY bins
and 25 µm depth bins.

For viewing, select **Precomputed Parquet + Cache**, click **Choose Cache
Directory...**, and choose a compatible profile. The selected profile fixes and
locks the render bounds and binning, so a small neuron query still overlays the
global region grid exactly. The plugin reads stored neuron columns and cached
region arrays without loading NRRDs, `atlas.annotation`, or BrainGlobe meshes.
It requires a matching atlas/version structure catalog for region selection and
colors, but that viewing atlas may have a different voxel resolution.

Missing or incompatible cache data reports the specific mismatch and never
recomputes automatically. Select **Recompute from NRRDs** explicitly to use the
legacy runtime conversion path. Saved project bundles retain version-3 Parquet
metadata and reference the external cache path/profile; they do not copy the
large cache directory.

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
napari `Image` layers or post-hoc region-isolated heatmaps. The region isolation
panel reuses the atlas region tree and creates new heatmap layers that preserve
source intensities inside selected regions while zeroing values outside them. In
the `Histogram` tab, those same eligible heatmaps and blurred heatmaps can be
inspected as overlaid interactive intensity histograms and then converted into 3D
binary mask `Labels` layers using explicit lower and optional upper intensity
bounds. The histogram view supports zooming and panning so narrow intensity
ranges are easier to inspect. A typical flow is: select a heatmap in `Tools`,
create a blurred layer, open `Histogram`, inspect the distribution, optionally
tune the blurred layer's `contrast_limits` in napari,
copy those limits into the threshold bounds, and then create the mask. The
`Regions` tab can then query neurons either by Allen regions or by one or more
of these generated mask layers, using all node types or a selected combination
of SWC node types (`soma`, `axon`, `basal dendrite`, `apical dendrite`).
Mask queries read the current mask layer voxels, so manual edits to a generated
mask change the query volume. When a mask was created from heatmaps with tracked
source neurons, the `Regions` tab shows their count and can exclude them from
the mask query results.

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
