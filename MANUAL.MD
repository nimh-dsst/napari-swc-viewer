# napari_swc_viewer User Manual

## What this plugin does

`napari-swc-viewer` is a napari plugin for:

- Opening `.swc` neuron morphology files directly
- Opening annotated `.parquet` neuron datasets
- Filtering neurons by Allen brain regions
- Rendering neurons as lines, points, or both
- Showing Allen reference layers (template, outline, region meshes, segmentation)
- Running clustering and heatmap analysis

The plugin adds:

- A file reader for `*.swc` and `*.parquet`
- A dock widget named **Neuron Viewer**

## Prerequisites

- Python `>=3.10`
- napari `>=0.5`
- `pyqt6`
- BrainGlobe atlas API dependencies (for atlas-backed features)
- Internet access for first-time atlas downloads

This repo is configured to use Pixi.

## Installation and launch

From the repository root:

```bash
pixi run build
pixi run napari
```

When napari is open, launch the widget from the Plugins menu and choose **Neuron Viewer**.

To enable debug-level plugin logging for large clustering runs:

```bash
NAPARI_SWC_VIEWER_DEBUG=1 pixi run napari
```

Optional custom log file:

```bash
NAPARI_SWC_VIEWER_DEBUG=1 NAPARI_SWC_VIEWER_LOG_FILE=/tmp/napari-swc-viewer.log pixi run napari
```

## Quick validation commands

Run from the repository root:

```bash
pixi run test
pixi run test-cov
```

If you need sample data used by tests/examples:

```bash
pixi run download-test-data
```

## Supported data inputs

### SWC files

- Open through napari File Open dialog (`.swc`)
- The reader renders parent-child segments as a `Shapes` layer (line segments)
- If no valid parent-child edges exist, it falls back to a `Points` layer

### Parquet files

- Open through napari File Open dialog (`.parquet`) or through the widget Data tab
- Initial reader view shows soma points in a red `Points` layer
- Full querying/rendering workflows are done in the **Neuron Viewer** widget

Recommended Parquet schema columns:

- `file_id`, `node_id`, `type`, `x`, `y`, `z`, `radius`, `parent_id`
- `region_id`, `region_name`, `region_acronym`
- `subject`, `neuron_id`

### Standard point Parquet files

- Produced from raw CSVs using `scripts/convert_point_csv.py`
- Imported through the widget Data tab with `Import Point Parquet...`
- Required columns: `label`, `x`, `y`, `z`
- Optional columns: `region_name`, `acronym`, `id`, `hemisphere`
- Any additional columns are preserved in the standardized Parquet

## Typical workflow

### 1. Open Neuron Viewer

Open the **Neuron Viewer** dock widget.

Tabs:

- Data
- Regions
- Visualization
- Reference
- Analysis
- Tools
- Histogram

### 2. Load data (Data tab)

In **Parquet Data**:

1. Click `Load...`
2. Choose a `.parquet` file
3. Confirm stats appear (nodes/files/subjects/regions)

In **Atlas**:

1. Choose atlas (`allen_mouse_10um`, `25um`, or `50um`)
2. Click `Load Atlas`

In **Point Parquet Import**:

1. Click `Import Point Parquet...`
2. Choose a standardized point `.parquet` file
3. The app adds one heatmap image layer per unique `label`, using a different color for each layer
4. If optional atlas metadata columns are present, the app checks them against the selected atlas and warns on mismatches

Notes:

- Atlas features need a loaded atlas
- First atlas load may download data and take time
- The template toggle attempts to load atlas automatically
- If atlas download fails, see Troubleshooting below

### 3. Select regions (Regions tab)

1. Choose `Query source`
2. For `Atlas Regions`, use search to find structures, check regions in the tree, and optionally enable/disable `Include child regions`
3. Choose `Node types`; leave `All node types` selected for the previous any-node behavior, or select one or more of `Soma`, `Axon`, `Basal dendrite`, and `Apical dendrite`
4. For `Mask Layer`, check one or more generated mask layers
5. Click the query button for the active source

This fills the **Selected Neurons** table in the Data tab.

### 4. Generate blurred heatmaps (Tools tab)

1. Select one or more eligible heatmap layers
2. Choose `Separate layers` or `Merged layer`
3. Set Gaussian sigma in atlas voxels
4. Click `Create Blurred Layer`
5. To isolate heatmaps after they already exist, use `Region Isolation`: choose
   `Separate layers` or `Merged layer`, select one or more atlas regions, choose
   whether child regions are included, and click `Create Isolated Heatmap`

Notes:

- Only app-created heatmaps on the native atlas voxel grid are eligible
- Depth-binned analysis heatmaps are excluded from mask creation in this version
- Generated blurred layers are also eligible native-grid sources in both the `Tools` and `Histogram` tabs
- Generated region-isolated heatmaps are new native-grid image layers and are also eligible sources in the `Tools` and `Histogram` tabs
- Gaussian sigma is in atlas voxels; for example, on `allen_mouse_25um`, `1.0` means `1 voxel = 25 µm`

### 5. Inspect histograms and build masks (Histogram tab)

1. Select one or more eligible heatmap or blurred heatmap layers
2. Review the overlaid interactive histogram traces for the selected layers
3. Optionally enable `Include zero-valued background` to inspect the zero spike
4. Choose `Separate layers` or `Merged layer` for mask creation
5. Set a `Lower bound`, optionally enable `Use upper bound`, and use either `Use Otsu Lower` or `Use Layer Contrast` to populate the bounds
6. Click `Create Mask Layer`

Notes:

- Generated masks are added as binary `Labels` layers and can be used from the `Regions` tab
- Mask-layer queries use the current nonzero voxels in the mask, including manual edits made after creation
- If the mask came from a heatmap with tracked source neurons, the `Regions` tab shows their count and can exclude them from mask-layer query results
- `Use Layer Contrast` requires exactly one selected heatmap or blurred heatmap layer
- Zero-valued background voxels are hidden from the histogram by default
- The histogram plot supports zooming and panning for tighter threshold inspection

### 6. Manage selected neurons (Data tab table)

In the table you can:

- Select rows to choose which neurons to render
- Toggle per-neuron visibility (`Vis`)
- Edit per-neuron color (color swatch)
- Inspect subject and cluster columns

Buttons:

- `Add Full Neurons`: add currently selected neurons to the scene while keeping neurons already added.
- `Add Soma Only`: add currently selected neurons to the scene as soma-only, reusing the shared soma layers without full trace geometry.
- `Remove Selected`: remove only currently selected neurons from the scene.
- `Clear All`: remove all neuron layers.
- `Add Heatmap`: create a heatmap from the currently selected table neurons.
- `Remove Selected From Table`: remove currently selected neuron rows from the table while keeping rendered scene layers.
- `Remove Unselected from Table`: keep selected table rows and remove all other table rows, including rows hidden by active filters.
- `Clear Table`: clear the table while keeping rendered scene layers.
- Centering mode (`Same` | `Auto`): `Same` keeps the current depth slice after add/remove. `Auto` auto-centers once on first add/remove in the widget session.

### 7. Configure rendering (Visualization tab)

Controls:

- Render mode: `Lines`, `Points`, `Both`
- Point size and optional `Color by node type`
- Line width
- Opacity slider
- Shared 2D Slice Projection toggle (`Show in 2D slices`) and slice thickness (µm)

Rendering behavior:

- Batch rendering is used (single layers for performance): `Neuron Lines` (`Shapes`), `Neuron Points` (`Points`), and `Soma Labels` (`Points`, labeled by `neuron_id`)
- `Add Selected` contributes to the full-trace layers based on the current Visualization render mode, while `Add Soma Only` contributes only to the shared soma layers
- Mixed scenes are supported: some neurons can remain full traces while others are rendered soma-only at the same time
- In 2D mode, line/point layers are auto-hidden to keep navigation responsive
- Optional `Neuron Slice Projection` (vectors layer) shows line segments near current slice
- Optional `Soma Slice Projection` (points layer) shows raw soma/body nodes (`type = 1`) near the current slice using the same shared slice thickness
- In 2D with projection enabled, `Soma Labels` are hidden to avoid duplicate soma markers, and `Points` render mode suppresses raw soma-node duplicates from `Neuron Points`

### 8. Add reference context (Reference tab)

Available layers:

- `Show template` -> `Allen Template` image
- `Show brain outline` -> `Brain Outline` surface
- `Show selected region meshes` -> `Region: <acronym>` surfaces
- `Show selected region segmentation` -> `Region Segmentation` labels layer

Notes:

- Mesh display auto-switches napari to 3D mode
- Segmentation is intended for 2D context overlays

### 9. Run analysis (Analysis tab)

### Clustering

Method options:

- `Voxel Correlation`
- `Soma Location`

For Soma Location, algorithm options:

- `Hierarchical`
- `K-Means`
- `DBSCAN`

Common controls:

- Target region acronym
- Dilation %
- Clusters (for non-DBSCAN modes)
- Linkage method (hierarchical modes)
- DBSCAN `eps` and `min_samples` when DBSCAN is selected

Run:

1. Click `Run Clustering`
2. Optional: expand `Clustermap` and click `Build Dendrogram` to render the dendrogram and cluster map
3. Click `Color Neurons by Cluster` to recolor rendered neuron layers

### Node Count Heatmap

Controls:

- Region filter (optional)
- Cluster filter (enabled after clustering)
- Node types (`All node types`, or any combination of soma, axon, basal dendrite, and apical dendrite)
- Soma radius (`0 µm` disables this filter; positive values count only selected node types within that distance of each neuron's soma centroid)
- Depth bin factor

Run:

1. Click `Build Heatmap Volume`
2. A heatmap image layer is added (cluster-specific or full population)

## SWC to Parquet conversion (inside plugin)

Data tab, **Convert SWC to Parquet**:

- `From Directory...` recursively finds `*.swc`
- `From Files...` selects explicit SWC files
- Set atlas annotation resolution (10-100 µm, default 25)
- Choose output `.parquet`

Conversion runs in a background worker with progress updates.

## CSV to standard point Parquet conversion

This repo also provides a converter for atlas-registered point CSV files:

```bash
python scripts/convert_point_csv.py raw_points.csv mapping.json points.parquet
```

Mapping JSON format:

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

Notes:

- Mapping keys are the standard target column names
- `label`, `x`, `y`, and `z` mappings are required
- The converter preserves any unmapped source columns in the output Parquet
- The converter validates numeric `x`, `y`, `z`, and `id` values before writing output

## Command-line hemisphere utility

This repo also provides a script for hemisphere detection and coordinate flipping:

```bash
python scripts/flip_swc.py input.swc -o output_flipped.swc
```

Useful flags:

- `--atlas allen_mouse_10um`
- `--coord-axis 0|1|2` (default 2)
- `--midline <microns>`
- `--info-only`
- `--list-atlases`

## Troubleshooting

### “Atlas failed to load”

- Check internet access for initial atlas download
- Try another atlas resolution (e.g. `allen_mouse_25um`)
- Restart napari after dependency changes

### “No neurons found in selected regions”

- Confirm region acronyms exist in your atlas
- Verify the selected regions match dataset `region_id` / `region_acronym`
- Toggle `Include child regions` and retry
- Confirm parquet stats report non-zero nodes/files/regions

### Clustering errors or empty results

- Ensure both a Parquet file and atlas are loaded
- Use a region containing enough neurons/somas
- For DBSCAN, increase `eps` or reduce `min_samples` if clusters are not found

### “Point Parquet import failed”

- Confirm the file contains standardized `label`, `x`, `y`, `z` columns
- Re-run `scripts/convert_point_csv.py` if you started from a raw CSV
- Load the target atlas before importing point Parquet
- If the app warns about metadata mismatches, the heatmap layers are still imported; review optional `region_name`, `acronym`, `id`, and `hemisphere` values

### “No eligible heatmaps” in Tools or Histogram

- Only app-created native-grid heatmap and blurred heatmap image layers are listed
- The heatmap must match the currently loaded atlas
- Depth-binned analysis heatmaps are excluded because they are not on the native atlas voxel grid

### “No neurons found” for a mask query

- Confirm the selected mask layer contains nonzero voxels
- Try `Any node in mask` before `Soma in mask`
- Check that the mask was created from heatmaps for the same atlas resolution as the loaded dataset

### Slow 2D slicing

- Keep `Show in 2D slices` disabled unless needed for either neuron lines or soma/body-point projection
- Reduce rendered neuron count
- Use `Lines` or `Points` selectively instead of `Both`

## Notes on coordinates and scale

- Node coordinates are expected in microns
- Some layers are displayed in atlas voxel space and use internal scaling for alignment
- For analysis/annotation quality, SWC coordinates should already be aligned to Allen CCF space
