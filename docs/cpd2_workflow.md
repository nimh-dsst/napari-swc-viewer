# CPD2 napari SWC Viewer Workflow

This guide walks through cloning the repository, launching the napari plugin with
Pixi, creating a left-hemisphere aligned `cpd2.parquet` file from `cpd2_data`,
and using soma clustering plus GPe-restricted cluster heatmaps to inspect CPD2
striatal projection organization.

The `cpd2_data` folder is expected to live in the repository root and contain
77 SWC files. These neurons have somas in the caudoputamen/striatum (`CP`) and
axonal projections into the globus pallidus, external segment (`GPe`).

## Clone, install, and run napari

1. Clone the repository.

   ```bash
   git clone https://github.com/nimh-dsst/napari-swc-viewer.git
   cd napari-swc-viewer
   ```

   If you use SSH for GitHub, clone with:

   ```bash
   git clone git@github.com:nimh-dsst/napari-swc-viewer.git
   cd napari-swc-viewer
   ```

2. Install Pixi if it is not already installed.

   Follow the Pixi installation instructions at:
   https://pixi.sh/latest/#installation

3. Launch napari with the plugin available.

   ```bash
   pixi run napari
   ```

   Pixi creates and uses the project environment for you on first run. Do not
   create a separate virtual environment or install dependencies with `pip`
   outside Pixi.

   The `napari` Pixi task depends on the `build` task in `pixi.toml`. That means
   `pixi run napari` first runs `pixi run build`, which installs
   `napari_swc_viewer` in editable mode with `pip install -e .` inside the Pixi
   environment. After the editable install finishes, Pixi runs the `napari`
   executable from the locked environment. The current `pixi.lock` resolves
   napari to `0.6.6`, so this command builds the plugin and then starts napari
   0.6.6.

4. In napari, open the plugin from the menu:

   `Plugins` -> `napari-swc-viewer` -> `Neuron Viewer`

## Create left-aligned `cpd2.parquet`

The CPD2 SWCs in `cpd2_data` are registered SWC files. Convert them into one
annotated Parquet file so the plugin can query neurons by Allen regions and
build heatmaps.

### Option A: create it in the plugin UI

1. In the **Neuron Viewer** widget, go to the **Data** tab.

2. Expand **Atlas**. Confirm the atlas dropdown is set to `allen_mouse_25um`.
   This is the 25 um Allen mouse atlas.

3. Click `Load Atlas`.

   The atlas may also load automatically when the widget starts or when a
   reference layer needs it. If this is your first time using
   `allen_mouse_25um`, the status label and progress bar will show the initial
   BrainGlobe download:

   - `Atlas: Checking BrainGlobe cache for allen_mouse_25um...`
   - If it is not cached, `Atlas: allen_mouse_25um was not found... Downloading...`
   - A determinate progress bar advances from 0 to 100 during the download.
   - After download, the bar may become indeterminate while BrainGlobe installs
     the atlas into the local cache.
   - When ready, the label shows `Atlas: allen_mouse_25um (... structures).`

   After the atlas is loaded, the plugin also shows the indicator message:
   `Atlas loaded. Go to the Reference tab to show the template, outline, or selected region meshes.`

4. In the **Data** tab, expand **Convert SWC to Parquet**.

5. Set:

   - `Resolution (μm)`: `25`
   - `Hemisphere alignment`: `Left`

6. Click `From Directory...` and choose the repository-root `cpd2_data` folder.

7. When prompted for the output file, save it as `cpd2.parquet` in the
   repository root.

8. Watch the conversion progress. For this dataset, expect 77 files to be
   processed. When conversion finishes, the status should report that the files
   were converted and show `cpd2.parquet` as the output.

### Option B: create it from the command line

From the repository root:

```bash
pixi run python scripts/convert_swc_to_parquet.py \
  cpd2_data \
  cpd2.parquet \
  --hemisphere left \
  --atlas allen_mouse_25um \
  --resolution 25 \
  --annotate-regions
```

This writes one Parquet row per SWC node, flips neurons as needed so their soma
coordinates are aligned to the left hemisphere, and annotates nodes with Allen
region metadata for 25 um region queries.

## Load CPD2 data and add all neurons to the table

1. In the **Data** tab, expand **SWC Parquet Data**.

2. Click `Load...` and choose `cpd2.parquet`.

3. Confirm the stats line appears. It should show 77 files, plus node, subject,
   and region counts.

4. Make sure `allen_mouse_25um` is loaded in the **Atlas** section. If it is not
   loaded, click `Load Atlas` and wait for the loaded-atlas indicator message.

5. Go to the **Regions** tab.

6. Set:

   - `Query source`: `Atlas Regions`
   - `Search scope`: `Whole Parquet`

7. In the region search box, search for `root`.

8. Select the `root` region. Leave `Include child regions` enabled. Selecting
   `root` is the broadest query and should include all CPD2 neurons.

9. Click `Find Neurons with Any Node in Selected Regions`.

   You must select a region, or `root`, and then click one of the query buttons.
   Loading `cpd2.parquet` alone does not query the file or populate the data
   table.

10. Return to the **Data** tab. The **Selected Neurons** table should now contain
    all 77 neurons.

The **Selected Neurons** table is a holding area for neurons that can be
visualized, analyzed, colored, filtered, or exported. Rows in this table are not
automatically rendered in the napari scene. The table lets you select rows,
toggle `Vis`, inspect `Added`, `Heatmap`, `Neuron ID`, `Subject`, editable
`Label`/`Group`/`Tags`/`Notes`, `Cluster`, and `Color`, and then decide which
rows to render or use for heatmaps.

## Add soma markers for all 77 neurons

1. In the **Data** tab, click inside the **Selected Neurons** table.

2. Select all rows:

   - macOS: `Cmd+A`
   - Windows/Linux: `Ctrl+A`

3. Click `Add Soma Only` button.

This adds soma markers for the selected neurons without rendering full axonal
and dendritic traces. The table `Added` state updates, and napari should show a
`Soma Labels` points layer. The soma-only view is useful here because the goal
is to cluster the 77 CPD2 neurons by soma position before inspecting where their
GPe axonal projections land.

## Cluster all 77 somas into 10 groups

CPD2 neurons are nucleated in the striatum (`CP`) and project to `GPe`. The
working anatomical expectation is that striatal organization is partly
maintained in the globus pallidus: neighboring striatal neurons tend to project
to neighboring or partially overlapping pallidal zones, rather than scattering
terminals uniformly throughout `GPe`.

Use soma-location clustering to assign the 77 striatal neurons to 10 groups,
then build one `GPe`-limited heatmap per cluster.

1. Go to the **Analysis** tab.

2. In **Clustering**, set:

   - `Method`: `Soma Location`
   - `Algorithm`: `K-Means`
   - `Search scope`: `Current Table`
   - `Clusters`: `10`
   - `Dilation %`: `0%`

3. Expand `Select Target Region`.

4. Search for and select `CP`.

   Because the current table contains all 77 CPD2 neurons and their somas are in
   `CP`, this clusters the CPD2 population by striatal soma position. If the
   progress text reports fewer than 77 somas in `CP`, select `root` as the
   target region and run clustering again to cluster every soma in the current
   table.

5. Click `Run Clustering`.

6. Wait for the progress area to report clustering completion. The Data tab
   table is updated with cluster labels and sorted by cluster. Rendered soma
   markers are also colored by cluster when possible.

7. Optional: expand `Clustermap` and click `Build Dendrogram` to inspect the
   cluster map.

## Create GPe-limited cluster heatmaps

1. Stay in the **Analysis** tab.

2. In **Node Count Heatmap**, expand `Select Heatmap Region`.

3. Search for and select `GPe`.

   This limits the heatmap volume to GPe voxels, so the signal you inspect is
   the pallidal projection pattern rather than the whole-neuron distribution.

4. Set `Depth bin factor` to `1` for native 25 um atlas-grid heatmaps.

5. Use either workflow:

   - To inspect one cluster at a time, choose a specific cluster in
     `Cluster filter`, then click `Build Heatmap Volume`.
   - To create the full series, click `Add All Cluster Heatmaps`.

6. When using `Add All Cluster Heatmaps`, wait for the progress text to step
   through all 10 cluster heatmaps. The layers will be named like
   `Cluster <n> GPe Heatmap`.

7. In the napari layer list, toggle cluster heatmaps on and off to compare their
   GPe projection zones.

The result should not look like every cluster is evenly distributed across all
of `GPe`. If striatal organization is preserved, each soma cluster should show a
partially clustered pallidal terminal distribution, with neighboring or
overlapping GPe zones rather than uniform coverage.

## Adjust cluster heatmap contrast

Each cluster heatmap is a napari image layer with independent contrast controls.

1. In the napari layer list, click one `Cluster <n> GPe Heatmap` layer.

2. In the image layer controls, adjust `contrast_limits`.

   A practical starting point is to keep the lower limit at `0` and lower the
   upper limit until sparse GPe signal becomes visible. Raise the upper limit if
   the strongest voxels saturate too much.

3. Adjust `opacity` if multiple cluster heatmaps are visible at once.

4. Repeat for each cluster heatmap you want to compare.

5. Use the layer visibility toggles to compare clusters one at a time, or leave
   several visible with additive blending to inspect overlap.

If a layer becomes hard to interpret, use napari's reset contrast control for
that image layer, then tune the upper contrast limit again.
