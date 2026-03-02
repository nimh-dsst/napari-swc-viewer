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

## Typical workflow

### 1. Open Neuron Viewer

Open the **Neuron Viewer** dock widget.

Tabs:

- Data
- Regions
- Visualization
- Reference
- Analysis

### 2. Load data (Data tab)

In **Parquet Data**:

1. Click `Load...`
2. Choose a `.parquet` file
3. Confirm stats appear (nodes/files/subjects/regions)

In **Atlas**:

1. Choose atlas (`allen_mouse_10um`, `25um`, or `50um`)
2. Click `Load Atlas`

Notes:

- Atlas features need a loaded atlas
- First atlas load may download data and take time
- The template toggle attempts to load atlas automatically
- If atlas download fails, see Troubleshooting below

### 3. Select regions (Regions tab)

1. Use search to find structures
2. Check regions in the tree
3. Optionally enable/disable `Include child regions`
4. Click `Find Neurons in Selected Regions`

This fills the **Selected Neurons** table in the Data tab.

### 4. Manage selected neurons (Data tab table)

In the table you can:

- Select rows to choose which neurons to render
- Toggle per-neuron visibility (`Vis`)
- Edit per-neuron color (color swatch)
- Inspect subject and cluster columns

Buttons:

- `Render Selected`: render currently selected table rows
- `Clear`: remove rendered neuron layers

### 5. Configure rendering (Visualization tab)

Controls:

- Render mode: `Lines`, `Points`, `Both`
- Point size and optional `Color by node type`
- Line width
- Opacity slider
- 2D Slice Projection toggle (`Show in 2D slices`) and slice thickness (µm)

Rendering behavior:

- Batch rendering is used (single layers for performance): `Neuron Lines` (`Shapes`), `Neuron Points` (`Points`), and `Soma Labels` (`Points`, labeled by `neuron_id`)
- In 2D mode, line/point layers are auto-hidden to keep navigation responsive
- Optional `Neuron Slice Projection` (vectors layer) shows line segments near current slice

### 6. Add reference context (Reference tab)

Available layers:

- `Show template` -> `Allen Template` image
- `Show brain outline` -> `Brain Outline` surface
- `Show selected region meshes` -> `Region: <acronym>` surfaces
- `Show selected region segmentation` -> `Region Segmentation` labels layer

Notes:

- Mesh display auto-switches napari to 3D mode
- Segmentation is intended for 2D context overlays

### 7. Run analysis (Analysis tab)

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
2. Inspect clustermap output
3. Click `Color Neurons by Cluster` to recolor rendered neuron layers

### Node Count Heatmap

Controls:

- Region filter (optional)
- Cluster filter (enabled after clustering)
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

### Slow 2D slicing

- Keep `Show in 2D slices` disabled unless needed
- Reduce rendered neuron count
- Use `Lines` or `Points` selectively instead of `Both`

## Notes on coordinates and scale

- Node coordinates are expected in microns
- Some layers are displayed in atlas voxel space and use internal scaling for alignment
- For analysis/annotation quality, SWC coordinates should already be aligned to Allen CCF space
