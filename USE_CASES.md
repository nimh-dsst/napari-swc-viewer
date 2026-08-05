# SWC Viewer Use Cases

This document describes what the SWC Viewer plugin can do through concrete,
repeatable user workflows. Each use case also serves as a manual test case for
confirming that the documented capability works in napari.

Use cases are added as workflows are described and refined. They document
observable behavior rather than implementation details.

## Running the Manual Tests

Unless a use case says otherwise:

1. From the repository root, launch napari with the development build:

   ```bash
   pixi run napari
   ```

2. Open the **SWC Viewer** plugin from napari's **Plugins** menu.
3. Follow the selected use case from a clean napari session.
4. Record the result in that use case's **Manual verification** section. A use
   case remains `Not run` until someone actually performs it in napari.

## Use-Case Index

| ID | Capability | Manual status |
| --- | --- | --- |
| [UC-001](#uc-001-download-an-allen-mouse-atlas) | Download and cache a supported Allen Mouse Brain Atlas | Not run |
| [UC-002](#uc-002-convert-swc-files-to-parquet) | Convert a folder or selected SWC files into one Parquet file | Not run |
| [UC-003](#uc-003-prepare-a-whole-neuron-parquet-for-flatmap-viewing) | Append bilateral shaped/square flatmap and depth coordinates to a whole Parquet | Not run |
| [UC-004](#uc-004-build-and-reuse-a-flatmap-region-cache) | Build, reopen, parse, and switch shaped/square region-cache data | Passed |
| [UC-005](#uc-005-view-an-allen-isocortex-layer-flatmap-stack) | View flatmap node counts as six Allen Isocortex layer images | Not run |
| [UC-006](#uc-006-inspect-and-query-custom-isocortex-layer-regions) | Inspect and query exact terminal regions grouped by Isocortex layer | Not run |
| [UC-007](#uc-007-refine-and-save-multiple-cluster-assignments) | Preserve a soma clustering and refine selected neurons with a second method | Not run |
| [UC-008](#uc-008-create-combined-and-individual-neuron-heatmaps) | Create one combined heatmap or color-matched heatmaps for individual neurons | Not run |
| [UC-009](#uc-009-save-and-overwrite-the-current-project) | Save changes back to the current SWC Viewer project safely | Not run |
| [UC-010](#uc-010-identify-axon-termini-and-prune-neurons-lacking-them) | Locate termini as childless axon-typed nodes, then select and remove the neurons that have none (see the annotation caution) | Partially run |

### UC-001: Download an Allen Mouse Atlas

**Capability**

The user can download the Allen Adult Mouse Brain Atlas at the resolution
needed for their data. The plugin supports the BrainGlobe atlas identifiers
`allen_mouse_10um`, `allen_mouse_25um`, and `allen_mouse_50um`.

The underlying atlas is the Allen Institute's Adult Mouse Brain Atlas, which
provides the commonly used CCFv3 coordinate space. The plugin obtains it
through the [BrainGlobe Atlas API](https://brainglobe.info/documentation/brainglobe-atlasapi/index.html),
which standardizes the Allen data into a reference image, anatomical annotation
volume, region hierarchy, region meshes, and metadata. BrainGlobe distributes
the packaged atlas archives from its
[dedicated GIN atlas repository](https://gin.g-node.org/BrainGlobe/atlases).
The plugin downloads from that BrainGlobe repository rather than directly from
the Allen Institute during this workflow.

These files are not bundled with the plugin because the volumetric images and
meshes are large and users may need different resolutions. The plugin needs the
selected atlas locally to place registered SWC morphologies and point data in a
shared anatomical coordinate space, identify the brain region at a coordinate,
display reference images, outlines, and region meshes, and support region-aware
queries and analyses. BrainGlobe caches each downloaded atlas so later sessions
can reuse it without another download. By default, the cache is under
`~/.brainglobe`, although the location can be changed in the BrainGlobe
configuration; see BrainGlobe's
[atlas file documentation](https://brainglobe.info/documentation/brainglobe-atlasapi/usage/using-the-files-directly.html).

**Prerequisites**

- The computer has an internet connection and enough free disk space for the
  selected atlas.
- The user can write to the BrainGlobe cache directory.
- At least one of `allen_mouse_10um`, `allen_mouse_25um`, or
  `allen_mouse_50um` is not already present in the configured BrainGlobe cache.
- Start from a clean napari session with the **SWC Viewer** plugin open.

**Steps and expected results**

1. **Action:** Open the **Data** tab and expand **Atlas**.
   **Expected:** The atlas selector offers `allen_mouse_10um`,
   `allen_mouse_25um`, and `allen_mouse_50um`, with a **Load Atlas** button and
   an atlas status area.
2. **Action:** Select an Allen mouse atlas that is not in the local BrainGlobe
   cache, then click **Load Atlas**.
   **Expected:** The plugin checks the cache, displays a text warning that the
   selected atlas is not cached, and asks whether to download it. The prompt
   identifies the selected atlas and explains that it will be downloaded via
   BrainGlobe.
3. **Action:** Cancel the download prompt.
   **Expected:** No atlas is downloaded or loaded. The Atlas controls remain
   available, and the status text makes it clear that the download was
   cancelled.
4. **Action:** Click **Load Atlas** again and confirm the download when
   prompted.
   **Expected:** The plugin starts the download through BrainGlobe. The Atlas
   controls are disabled during the operation, status text identifies the
   current phase, and the progress bar reports download progress before showing
   that the atlas is being installed in the local cache.
5. **Action:** Wait for the download and installation to finish.
   **Expected:** The Atlas controls are enabled again, the progress indicator
   closes, and the status identifies the loaded atlas and its number of
   structures. The plugin reports that the atlas is loaded and that its
   template, outline, or selected region meshes can be shown from the
   **Reference** tab.
6. **Action:** Close napari, start a new session, select the same atlas, and
   click **Load Atlas**.
   **Expected:** The plugin finds and loads the cached atlas without showing the
   download prompt or downloading the files again.

Repeat this use case for each supported resolution that needs to be available
locally.

**Manual verification**

- Status: Not run
- Last verified: Never
- Notes: The uncached-atlas warning and confirmation prompt are tracked in
  `TODO.md`.

### UC-002: Convert SWC Files to Parquet

**Capability**

The user can combine multiple SWC morphology files into one Parquet file from
the **Data** tab. The Parquet stores each SWC node with its source filename,
node type and topology, coordinates, radius, filename-derived neuron metadata,
and Allen atlas region information. This format lets the plugin load and query
large groups of registered neurons as one dataset and supports its downstream
visualization and analysis tools.

The **Convert SWC to Parquet** section offers two ways to choose the input. Both
produce one Parquet file, but they differ in which SWCs are included:

| Input action | Files included | Use it when |
| --- | --- | --- |
| **From Directory...** | Every file with an `.swc` extension, including uppercase variants such as `.SWC`, in the selected folder and all of its subfolders | A folder represents the complete dataset to convert |
| **From Files...** | Only the files explicitly selected in the multi-file picker; other SWCs in the same folder or its subfolders are not discovered automatically | Only a particular group or subset of SWCs should be converted |

`Resolution (μm)` selects the Allen atlas resolution used to annotate node
coordinates with brain regions. `Hemisphere alignment` controls whether source
coordinates are kept as-is (`None`) or morphologies on the opposite side are
reflected into a common `Left` or `Right` hemisphere. The atlas resolution and
the coordinate space of the SWCs must match for region annotations and
hemisphere alignment to be meaningful.

The optional **Add bilateral flatmap/depth columns** setting chains flatmap
preprocessing onto the same background conversion. **Lookup directory...**
selects a directory containing `flatmap_both_shaped.nrrd`,
`flatmap_both_square.nrrd`, and `depth.nrrd`. When enabled, the output is a
version-3 Parquet that can be rendered without reading the NRRDs again.
**Lookup resolution** normally reads the transform from the NRRD headers; enter
an explicit micrometer value when those headers do not contain a usable
transform. **Cancel conversion** cooperatively stops either phase and prevents
a partial output from being published.

**Prerequisites**

- Complete UC-001 for `allen_mouse_25um` so the atlas needed by this test is
  available locally.
- Use valid SWC files registered to the Allen CCF in micrometers.
- To test the optional chained preprocessing, prepare a lookup directory whose
  shaped, square, and depth NRRDs have the same spatial grid and transform.
- Prepare this input layout, where each SWC has a distinct filename:

  ```text
  swc_batch/
  ├── cell_a.swc
  ├── cell_b.swc
  ├── notes.txt
  └── nested/
      └── cell_c.SWC
  ```

- Start from a clean napari session with the **SWC Viewer** plugin open.

**Steps and expected results**

1. **Action:** Open the **Data** tab and expand **Convert SWC to Parquet**.
   **Expected:** The section shows **From Directory...**, **From Files...**,
   `Resolution (μm)`, `Hemisphere alignment`, **Add bilateral flatmap/depth
   columns**, **Lookup directory...**, `Lookup resolution`, **Cancel
   conversion**, a status area, and a progress indicator that is initially
   hidden.
2. **Action:** Set `Resolution (μm)` to `25` and `Hemisphere alignment` to
   `None`.
   **Expected:** Conversion will use the 25 μm Allen annotation while preserving
   the SWC coordinates.
3. **Action:** Click **From Directory...**, select `swc_batch`, and save the
   output as `directory.parquet` when **Save Parquet File** opens.
   **Expected:** The plugin recursively searches the selected folder, discovers
   `cell_a.swc`, `cell_b.swc`, and `nested/cell_c.SWC`, and ignores `notes.txt`.
   The progress indicator and status text report discovery and conversion of
   three files.
4. **Action:** Wait for directory conversion to finish.
   **Expected:** `directory.parquet` is created. The progress indicator closes,
   and the status reports `Done! Converted 3 file(s) -> directory.parquet`.
5. **Action:** Click **From Files...**, multi-select only `cell_a.swc` and
   `cell_b.swc`, and save the output as `selected_files.parquet`.
   **Expected:** The conversion starts with two explicitly selected files. It
   does not discover `nested/cell_c.SWC` or include `notes.txt`.
6. **Action:** Wait for selected-file conversion to finish.
   **Expected:** `selected_files.parquet` is created. The progress indicator
   closes, and the status reports
   `Done! Converted 2 file(s) -> selected_files.parquet`.
7. **Action:** Expand **SWC Parquet Data**, click **Load...**, and open each new
   Parquet in turn.
   **Expected:** The statistics for `directory.parquet` report `Files: 3`,
   while the statistics for `selected_files.parquet` report `Files: 2`. The
   difference confirms that directory selection is recursive and file selection
   is explicit.
8. **Action:** Start either input-selection flow again, but cancel the input
   picker or the **Save Parquet File** dialog.
   **Expected:** No conversion starts and no additional Parquet file is created.
9. **Action:** Enable **Add bilateral flatmap/depth columns**, click **Lookup
   directory...**, select the prepared lookup directory, leave `Lookup
   resolution` at `From NRRD header`, convert the three-file directory again,
   and save it as `directory_flatmap.parquet`.
   **Expected:** SWC conversion and flatmap augmentation run as one progress
   operation. The published Parquet contains all three neurons plus shaped and
   square bilateral flatmap columns and shared depth columns. No intermediate
   Parquet remains beside the output.
10. **Action:** Start the chained conversion once more and cancel while it is
    running.
    **Expected:** The worker stops, temporary artifacts are removed, any
    existing output remains unchanged, and all conversion controls become
    available again.

If an individual SWC cannot be parsed, the converter skips it, keeps processing
the valid files, and includes the skipped count in the completion status. If no
SWC files are found or none can be processed successfully, the status displays
an error instead of reporting successful conversion.

**Manual verification**

- Status: Not run
- Last verified: Never
- Notes: None

### UC-003: Prepare a Whole Neuron Parquet for Flatmap Viewing

**Capability**

The user can preprocess every node in an existing neuron Parquet once and save
bilateral shaped-flatmap, square-flatmap, and shared depth coordinates in a
version-3 Parquet. The output retains the original CCFv3 coordinates, custom
columns, and schema metadata. Later flatmap viewing reads the stored columns
and does not need to load or query the NRRDs.

**Prerequisites**

- Prepare a neuron Parquet containing multiple neurons and at least one custom
  column. Load it through **Data** > **SWC Parquet Data** and use a Regions query
  so that only a subset of its neurons is in the visible data table.
- Prepare one lookup directory containing `flatmap_both_shaped.nrrd`,
  `flatmap_both_square.nrrd`, and `depth.nrrd`. All three files must describe
  the same spatial grid and transform.
- If the NRRD headers do not provide a usable spatial transform, know the
  lookup resolution in micrometers so it can be entered explicitly.
- Start from a clean napari session with the **SWC Viewer** plugin open.

**Steps and expected results**

1. **Action:** Open the **Flatmap** tab, click **Lookup directory...**, and
   select the prepared directory.
   **Expected:** The selected lookup directory is shown. Validation and lookup
   set identification begin with **Prepare Whole Parquet...**, off the UI
   thread.
2. **Action:** Click **Prepare Whole Parquet...** and accept the suggested
   output name ending in `_flatmap.parquet`.
   **Expected:** A background, cancellable operation processes every source row,
   including neurons excluded from the current table query. Progress does not
   depend on which table rows are selected.
3. **Action:** Wait for preprocessing to finish, then load the output through
   **SWC Parquet Data**.
   **Expected:** The output has the same source row count and custom columns.
   It adds `x_flat_shaped`, `y_flat_shaped`, shaped validity/provenance columns,
   matching square columns, and shared `depth_um` validity/provenance columns.
   Its metadata identifies format version 3 and the lookup-set ID.
4. **Action:** In **Flatmap**, choose **Precomputed Parquet + Cache**, switch
   between **Both hemispheres, shaped** and **Both hemispheres, square**, and
   project the same table subset.
   **Expected:** Both styles render from the stored columns on their recorded
   canonical grids. The plugin does not ask for or load the NRRDs.
5. **Action:** Run **Prepare Whole Parquet...** again, choose the source file as
   the destination, and decline the replacement confirmation.
   **Expected:** Preprocessing does not start and the source is unchanged.
6. **Action:** Start preprocessing to a new output and cancel it while running.
   **Expected:** The temporary output is removed and no partial Parquet is
   published. If an output already existed, its previous contents are intact.
7. **Action:** Select a lookup directory whose files have mismatched grids or
   transforms.
   **Expected:** Validation reports the exact mismatch and does not create an
   output. A lookup set without a usable NRRD transform requests an explicit
   resolution instead of silently assuming one.

**Manual verification**

- Status: Not run
- Last verified: Never
- Notes: None

### UC-004: Build and Reuse a Flatmap Region Cache

**Capability**

The user can build a versioned, reusable cache that projects BrainGlobe atlas
regions into the same fixed shaped and square flatmap grids used by a version-3
neuron Parquet. The cache stores label occupancy, closed region surfaces, and
depth-slice outlines. Reopening it avoids loading atlas annotation voxels,
projecting meshes, or converting region coordinates while viewing.

**Prerequisites**

- Complete UC-003 and keep its version-3 Parquet and lookup directory.
- Load the intended BrainGlobe atlas. Its annotation shape and voxel resolution
  must exactly match the lookup grid; its atlas name and version are recorded
  as cache-profile identity.
- Ensure there is enough writable disk space for both shaped and square cache
  profiles. Use the default 256 XY bins and 25 μm depth bins for this test.
- Select at least one parent region with descendants in the **Regions** tab.
- For the detached-window close check, launch with a dedicated trace:
  `NAPARI_SWC_VIEWER_DEBUG=1 NAPARI_SWC_VIEWER_LOG_FILE=/tmp/napari-swc-viewer.log pixi run napari`.

**Steps and expected results**

1. **Action:** Open **Flatmap**, choose the lookup directory, click **Build
   Cache Profile...**, choose a new cache directory, and keep **New profile XY
   bins** at `256` and **Depth bin** at `25 um`.
   **Expected:** The plugin validates the exact atlas/lookup grid match and
   builds one profile containing a shaped/square pair. Progress covers
   annotation scanning, occupancy, surfaces, and outlines. The manifest is
   published only after all referenced arrays are complete.
2. **Action:** Change **New profile XY bins** to `272`, start a second build,
   and cancel it while it is running.
   **Expected:** Temporary files from that build are removed, the prior profile
   remains listed and usable, and the controls are restored.
3. **Action:** Close and reopen napari, load the version-3 Parquet and a
   BrainGlobe atlas with the same atlas/version structure catalog, choose
   **Precomputed Parquet + Cache**, set **Render** to **Heatmap** and its color
   mode to **Single color**, then click **Project to Flatmap**. After the
   heatmap appears in the detached flatmap window, click **Choose Cache
   Directory...** and select the existing cache. Repeat the cache selection
   several times on both Windows and macOS when those systems are available.
   **Expected:** Cache validation runs without freezing either napari window.
   The detached window first becomes visible with its populated 3D heatmap;
   there is no visible blank intermediate flatmap window.
   The plugin parses `flatmap-region-cache.json`, memory-maps its arrays, and
   lists only profiles compatible with the Parquet lookup-set ID and selected
   style. A heatmap whose fixed grid matches the selected profile remains
   visible and is not recreated. Napari does not crash or show an operating
   system crash report, and the plugin does not access atlas annotation or
   region mesh data.
4. **Action:** Select the new cache profile.
   **Expected:** XY bins, depth-bin size, canonical bounds, and exclusion of the
   depth `-1` sentinel plane are set from the profile and locked. If the profile
   grid differs from an existing heatmap, that heatmap is hidden before being
   removed and the status asks the user to click **Project to Flatmap** again.
5. **Action:** Select a parent region, enable child regions, then click **Show
   Region Labels**, **Show Region Surfaces**, and **Show Region Outlines**.
   Then choose **Custom Regions**, select two terminal leaves, and click the
   same actions again.
   **Expected:** Labels apply the include-child-expanded selection. The surface
   and outlines use the selected parent's cached descendant union and atlas
   color. With Custom Regions active, labels and one atlas-colored
   surface/outline layer per exact terminal ID replace the Atlas selection.
   The resulting **Flatmap Region Labels**, **Flatmap Region Surfaces**, and
   **Flatmap Region Outlines** layers align with the neuron heatmap.
6. **Action:** Switch between **Both hemispheres, shaped** and **Both
   hemispheres, square** without changing the Parquet or cache directory, then
   project and show the desired cached region layers again.
   **Expected:** Stale layers from the previous style are cleared. Neurons and
   newly shown cached region layers use the matching profile grid and remain
   aligned, including when the queried neurons occupy only a small part of the
   global flatmap.
7. **Action:** Load a Parquet or atlas that is incompatible with every cached
   profile, or remove one referenced array and then reopen the cache. When
   testing a second directory, keep a valid cache and heatmap active first.
   **Expected:** The plugin reports the exact lookup, atlas, grid, missing-file,
   or validation mismatch. A failed candidate directory does not replace the
   active cache or clear its heatmap. It never starts NRRD recomputation
   automatically.
8. **Action:** Explicitly choose **Recompute from NRRDs**.
   **Expected:** The legacy runtime lookup workflow becomes available only for
   this explicit fallback choice.
9. **Action:** After a projection appears, use the operating-system close
   control on the detached **SWC Viewer Flatmap** window while leaving the main
   napari window open, and accept napari's close confirmation. Wait at least
   two seconds, then click **Project to Flatmap** again. Repeat the accepted
   close and reopen cycle three times, then close the main napari window and
   retain the debug log. Repeat once while cancelling napari's confirmation.
   **Expected:** The detached window closes completely, the main napari window
   remains responsive, and the detached viewer releases its resources. The
   next projection opens exactly one fresh detached flatmap window with the
   expected layers and never exposes a blank setup window. Cancelling the
   confirmation keeps the original detached viewer and its layers usable. Each
   fresh viewer records `event=created_hidden`, `event=first_layer_ready`,
   `event=show_scheduled`, and `event=shown` before the close records, with
   `pending_first_show=false` after it is shown. For each accepted close, the
   log contains
   `event=qt_close`, three `event=close_checkpoint` records,
   `event=qt_deferreddelete`, `_LayerSlicer.shutdown`, and
   `event=cleanup_complete` with `cleanup_trigger=deferred_delete`,
   `cleanup_qt_viewer=closed`, zero layers, `napari_viewer_registered=false`,
   both plugin viewer references false, and `slicer_executor_shutdown=true`.
   Three `event=post_destroy_checkpoint` records follow; the 2000 ms record has
   no matching flat-map QWidget or native QWindow. The trace contains no
   `event=cleanup_failure`.
10. **Action (macOS only):** Put the main napari window in macOS Full Screen,
    then create a flat-map projection. Close and reopen the detached window three
    times while the main viewer stays fullscreen. Then manually place the
    detached **SWC Viewer Flatmap** window in fullscreen and close it; cancel the
    confirmation once and confirm the viewer is still usable, then close it again
    and accept.
    **Expected:** The detached window opens normally with populated content while
    the main window remains fullscreen — the trace shows `show_path=normal_qt`,
    `event=normal_show_requested`, and `event=fullscreen_restore_suppressed`, and
    the main window stays fullscreen and registered. Closing the manually
    fullscreened detached window safely exits fullscreen, presents confirmation,
    and closes completely, recording `event=fullscreen_close_deferred`,
    `event=fullscreen_exit_requested`, `event=fullscreen_exit_complete`, and
    `event=fullscreen_close_retried`, then the normal
    `event=qt_deferreddelete`/`event=cleanup_complete` cleanup. Cancelling leaves
    the viewer usable in normal window mode. There is no ghost window, no crash
    or OS crash report, no `event=fullscreen_guard_failure`, and no change to the
    main viewer's fullscreen state. napari itself writes `window_fullscreen` to
    its global settings on close; the detached viewer must still open normally on
    the next projection regardless of that saved value.

**Manual verification**

- Status: Passed
- Last verified: 2026-07-22
- Notes: Verified manually on macOS (macOS 26.5.1 arm64, napari 0.6.6, PyQt6
  6.8.1). Cache build/reopen/parse and shaped/square switching all behave as
  expected. The detached flatmap window now closes completely while the main
  napari window stays open, and the plugin creates the viewer hidden in 3D,
  showing it only after its first configured layer (no blank intermediate
  window). The macOS fullscreen workflow (step 10) also passed: the detached
  window opens normally even when the main window is in Full Screen
  (`show_path=normal_qt`), and closing a manually fullscreened detached window
  safely exits fullscreen and tears down without a crash — the trace shows
  `fullscreen_close_deferred` → `fullscreen_exit_complete` →
  `cleanup_complete` with `cleanup_status_thread=stopped`, and no
  `QThread: Destroyed while thread is still running` warning. Cancelling the
  close confirmation left the viewer usable, and reopening produced exactly one
  fresh detached window. This remains a macOS-specific workaround for
  napari 0.6.6 whose Qt-dependent behavior can only be validated manually.

### UC-005: View an Allen Isocortex Layer Flatmap Stack

**Capability**

The user can view selected neuron morphology nodes as six planar flatmap
heatmaps, one for each Allen Isocortex layer group: `L1`, `L2/3`, `L4`, `L5`,
`L6a`, and `L6b`. Layer membership comes from each node's Allen `region_id`,
not from a numeric flatmap-depth interval. Each image pixel is the number of
matching nodes in that flatmap XY bin.

The six images are one napari image stack, ordered from superficial to deep.
The default single-color mode creates one stack; the existing individual and
cluster color modes create one six-plane stack per color group.

**Prerequisites**

- Load a supported Allen mouse atlas in **Data** > **Atlas**.
- Load a neuron Parquet with Allen `region_id` annotations and valid flatmap
  coordinates. For **Precomputed Parquet + Cache**, use a version-3 Parquet
  produced by UC-003.
- Complete UC-004 and keep a compatible flatmap region cache. Select at least
  one Atlas Regions parent or Custom Regions terminal leaf so the planar label
  overlay can be checked.
- Include neurons with nodes in at least two Allen Isocortex layers. Include an
  agranular cortical area without layer 4 when test data is available.
- Start from a clean napari session with the **SWC Viewer** plugin open.

**Steps and expected results**

1. **Action:** In the neuron table, select one or more rows. Open **Flatmap**,
   choose **Precomputed Parquet + Cache**, select **Both hemispheres, shaped**,
   choose the compatible cache directory/profile, and set **Render** to
   **Allen Layer Heatmap (2D stack)**.
   **Expected:** **XY bins** remains available unless locked by the active
   cache profile. **Depth bin** and **Exclude depth -1 nodes** are disabled
   because numeric depth does not assign Allen layers. **Show Region Labels**
   is available for the active cache, while cached surfaces and outlines
   remain unavailable because their geometry is depth-based.
2. **Action:** In **Regions**, choose **Atlas Regions**, select a cortical
   parent region, return to **Flatmap**, and click **Show Region Labels**
   before projecting neurons.
   **Expected:** The detached **SWC Viewer Flatmap** can open with a label-only
   2D stack named **Flatmap Region Labels**. It has six planes ordered `L1`,
   `L2/3`, `L4`, `L5`, `L6a`, `L6b`, contains only the selected region's
   terminal Allen-layer descendants, uses atlas colors, and reads only the
   active cache arrays and structure catalog—not NRRDs or `atlas.annotation`.
3. **Action:** Keep **Heatmap colors** at **Single color** and click **Project
   to Flatmap**.
   **Expected:** The detached **SWC Viewer Flatmap** opens in 2D with one image
   layer named **Isocortex Flatmap Allen Layers**. Its first axis identifies
   indices `0` through `5` as `L1`, `L2/3`, `L4`, `L5`, `L6a`, and `L6b`.
   The existing **Flatmap Region Labels** layer remains aligned with it.
4. **Action:** Move the first-axis slider through all six positions.
   **Expected:** The heatmap and Labels layer change planes together. Each
   position shows only nodes and region labels assigned to that terminal Allen
   Isocortex layer. A cortical area without layer 4 is blank in that area on
   the `L4` plane; it is not filled by a depth estimate.
5. **Action:** Choose **Custom Regions**, select terminal leaves from two
   layers, and click **Show Region Labels** again.
   **Expected:** The existing Labels layer is updated rather than duplicated.
   Only the exact checked terminal IDs are included and each appears on its
   mapped Allen plane; the retained Atlas Regions selection does not contribute.
   Regions competing for one plane/XY bin use greatest source-voxel occupancy,
   with the smaller region ID winning ties.
6. **Action:** Compare the projection summary with an independent count of the
   selected Parquet rows by the corresponding Allen layer region IDs.
   **Expected:** The total rendered-node count and all six per-layer counts
   agree. Flatmap-invalid, non-Isocortex, parent-level, unannotated, and
   otherwise non-laminar nodes are reported as excluded and do not appear.
7. **Action:** Repeat with **Both hemispheres, square**, **All table rows**,
   **Individual neurons**, and **Cluster**.
   **Expected:** Shaped and square stacks use their recorded canonical XY
   grids. Input selection is respected. Individual and cluster modes create
   one synchronized six-plane stack per neuron or cluster using existing
   colors. Stale labels from the prior style are removed; showing them again
   uses the new style's cache grid.
8. **Action:** Choose **Recompute from NRRDs**, select the matching flatmap and
   depth NRRDs, and project the same neurons.
   **Expected:** The materialized result has the same layer assignments and
   node-count semantics. **Export CSV...** includes `allen_layer_index` and
   `allen_layer_label` for every classified node. Planar cached labels,
   surfaces, and outlines are unavailable in this explicit fallback mode.
9. **Action:** Switch **Render** back to **3D Heatmap**.
   **Expected:** The categorical stack is removed, numeric depth controls and
   compatible cached-region actions return, and a new projection uses the
   original depth-binned behavior.
10. **Action:** Retry layer rendering without a loaded atlas, then with a
    Parquet missing `region_id`, and finally with selected neurons that have no
    flatmap-valid terminal Isocortex-layer nodes. Also try **Show Region
    Labels** with no active Atlas/Custom selection, a non-Isocortex-only Atlas
    selection, and terminal layer regions with no occupancy in the active cache.
   **Expected:** Each attempt reports a specific corrective message and does
   not leave a blank detached flatmap window or a stale Labels overlay.

**Manual verification**

- Status: Not run
- Last verified: Never
- Notes: None

### UC-006: Inspect and Query Custom Isocortex Layer Regions

**Capability**

The user can visually inspect the terminal Allen regions assigned to each of
the six Isocortex layer groups and query neurons using exactly those displayed
`region_id` values. The synthetic hierarchy is computed from the currently
loaded atlas and remains independent of the ordinary atlas-region selection
while retaining the same Allen identities. While Custom Regions is active, its
checked terminal regions can drive Reference-tab previews and flatmap labels,
surfaces, and outlines.

**Prerequisites**

- Load an Allen mouse atlas in **Data** > **Atlas**. For the count comparison,
  use `allen_mouse_25um`.
- Load a neuron Parquet annotated against that atlas. It must contain
  `file_id`, `type`, and `region_id`; include neurons from several Isocortex
  layers and at least one non-Isocortex region.
- Put a strict subset of the Parquet's neurons in **Selected Neurons** so the
  two search scopes can be distinguished.
- Complete UC-004 and keep its compatible version-3 Parquet and flatmap region
  cache so cached labels, surfaces, and outlines can be exercised.
- From the repository root, optionally run
  `pixi run python scripts/audit_allen_isocortex_layers.py` to print the
  independently inspectable region list and counts for `allen_mouse_25um`.

**Steps and expected results**

1. **Action:** Open **Regions** and inspect **Query source**.
   **Expected:** The menu is ordered **Atlas Regions**, **Custom Regions**,
   **Mask Layer**. Existing Atlas and Mask pages remain available.
2. **Action:** Select **Custom Regions** with **Search scope** set to
   **Whole Parquet**, then expand **Isocortex Layers**.
   **Expected:** The root starts expanded and the six initially collapsed
   groups appear in the order `L1`, `L2/3`, `L4`, `L5`, `L6a`, `L6b`.
   Root and group labels show terminal-region counts. Each leaf shows its full
   atlas name, acronym, and numeric ID and is alphabetized within its layer.
   For `allen_mouse_25um`, the displayed lists and counts match
   `scripts/audit_allen_isocortex_layers.py`; no parent or unrelated region is
   present.
3. **Action:** Check one layer group, uncheck one leaf beneath it, and then
   check that leaf again.
   **Expected:** Checking the group selects all and only its displayed terminal
   leaves. Unchecking a leaf makes the group and root partially checked;
   restoring it makes the layer fully checked. **Selected: N terminal
   regions** always reports the exact number of selected leaves.
4. **Action:** Clear the selection. Search in turn by part of a full name, an
   acronym, and a numeric region ID, then select one matching leaf and leaves
   from a second layer.
   **Expected:** Search retains matching leaves and their ancestors. **Clear**
   restores the full hierarchy. **Clear Selection** removes all checks.
   Selecting across layers yields a sorted, deduplicated set of the visible
   terminal IDs.
5. **Action:** Choose desired **Node types**, click **Find Neurons in Selected
   Custom Regions**, and compare the results with an exact `region_id` query of
   the Parquet.
   **Expected:** The **Selected Neurons** table contains precisely the matching
   neurons. The status reports the selected terminal-region count and node
   membership. No unlisted descendant or parent region contributes matches,
   and non-Isocortex rows do not match.
6. **Action:** Change **Search scope** to **Current Table**, make a different
   custom selection, and query. Switch back to **Whole Parquet**.
   **Expected:** Current Table querying is restricted to the exact file IDs
   already in the table and preserves nonmatching existing rows. The two
   scopes retain separate custom selections, and returning to Whole Parquet
   restores its selection.
7. **Action:** Keep **Custom Regions** active, select a full layer and a
   partial second layer, then enable **Show selected region meshes** and
   **Show selected region segmentation** in **Reference**.
   **Expected:** Napari switches to 3D for meshes and creates at most one
   `Region: Custom <layer>` Surface per selected canonical layer. Each
   terminal mesh retains its Allen color. `Region Segmentation` contains
   exactly the checked terminal IDs with no unlisted descendants. Changing
   Custom checks or Search scope refreshes both visible previews, and their
   opacity controls continue to apply.
8. **Action:** Switch **Query source** between **Atlas Regions** and
   **Custom Regions** while both Reference controls remain enabled.
   **Expected:** The existing preview layers are replaced using the active
   source and scope. Both selectors retain their own selections.
9. **Action:** Keep **Custom Regions** active, select two terminal leaves, open
   **Flatmap**, choose **Precomputed Parquet + Cache**, activate the compatible
   profile, set **Render** to **3D Heatmap**, and click **Show Region Labels**,
   **Show Region Surfaces**, and **Show Region Outlines**.
   **Expected:** **Flatmap Region Labels** contains exactly the checked
   terminal IDs. One atlas-colored surface and one outline layer are created
   per selected terminal region, with its acronym and numeric ID in the layer
   name. No unchecked descendant, synthetic layer union, or Atlas Regions
   selection contributes. Viewing reads cache arrays and the atlas structure
   catalog without loading NRRDs, `atlas.annotation`, or BrainGlobe meshes.
10. **Action:** Change the Custom Regions selection and click each **Show**
    action again; then switch between **Whole Parquet** and **Current Table**
    and repeat with the different retained selection.
    **Expected:** The existing Labels layer is updated instead of duplicated.
    Each geometry family is replaced with one layer per terminal ID from the
    active scope. Selection changes do not rebuild flatmap layers until the
    corresponding **Show** action is clicked.
11. **Action:** Set **Render** to **Allen Layer Heatmap (2D stack)** and click
    **Show Region Labels**, then project neurons and move the first-axis slider.
    Repeat after switching between shaped and square styles.
    **Expected:** The selected custom terminal regions appear only on their
    corresponding `L1`, `L2/3`, `L4`, `L5`, `L6a`, or `L6b` planes. The
    labels and heatmap remain synchronized and aligned on each style's grid.
    Cached surfaces and outlines remain disabled because they are depth-based.
12. **Action:** Switch **Query source** to **Atlas Regions**, show its
    overlays, then switch to **Mask Layer** and try the same actions.
    **Expected:** Atlas actions retain include-child labels and parent-union
    geometry. Mask selections never fall back to a retained Atlas selection
    and instead report that flatmap atlas overlays require Atlas or Custom
    Regions.
13. **Action:** Exercise the error cases: query or render with no custom leaf
    selected; use an empty Current Table; load a Parquet without `region_id`;
    load an atlas whose structure catalog cannot produce all six Isocortex
    layers; and select a valid terminal region with no occupancy or geometry
    in the active cache.
    **Expected:** Each case reports an actionable message and does not perform
    a misleading query. An incompatible or missing atlas leaves the Custom
    Regions page in an explanatory empty state and does not open or download a
    second atlas. A valid query selection with no matching neurons reports zero
    results without error. A valid render selection with no cached data clears
    the stale overlay family and reports that nothing was represented.

**Manual verification**

- Status: Not run
- Last verified: Never
- Notes: None

### UC-007: Refine and Save Multiple Cluster Assignments

**Capability**

The user can retain several named clustering results for the same neurons. A
first run can group neurons by soma location; one soma cluster can then be
selected and clustered by voxel correlation without removing other neurons or
overwriting the soma assignments. One assignment is active at a time for
filtering, colors, flatmap grouping, and Analysis heatmaps. Projects and
Enhanced Parquet exports preserve every assignment and its run provenance.

**Prerequisites**

- Load an Allen mouse atlas and a neuron Parquet containing at least six
  neurons with soma nodes and morphology nodes in one represented atlas region.
- To test Flat map + Depth selected-row clustering and flatmap grouping, use a
  version-3 Parquet with bilateral flatmap/depth columns.
- Populate **Selected Neurons** with the full test population and leave at
  least two neurons in each intended soma cluster.
- To verify the large-run guard, prepare cohorts whose post-filter clustering
  inputs contain exactly 10,000,000 and at least 10,000,001 contributing node
  rows.
- Start with no prior cluster assignments or record their names so the new
  columns can be distinguished.

**Steps and expected results**

1. **Action:** Open **Analysis** > **Clustering**, choose **CCFv3
   Coordinates**, **Soma Location**, and the desired soma algorithm. Set
   **Input neurons** to **Current Table**, click **Clear Selection** under
   **Select Target Region**, choose at least two clusters, and click **Run
   Clustering**.
   **Expected:** The run completes and creates a **Soma Location 1** column in
   the Data-tab table. **Cluster assignment** selects **Soma Location 1**, and
   every clustered neuron has an integer label. **Target region (optional)**
   reports **All regions (optional)**, and all eligible somas in the Current
   Table cohort contribute. The cluster filter, summary, rendered colors,
   flatmap cluster mode, and Analysis heatmap cluster choices use these labels.
2. **Action:** In **Data** > **Selected Neurons**, use **Cluster** to show one
   soma cluster and select all of its visible rows with Ctrl+A or Cmd+A.
   **Expected:** Only rows from that soma cluster are selected. Other neurons
   remain in the table and retain their soma labels.
3. **Action:** Return to **Analysis** > **Clustering**, choose **Voxel
   Correlation**, set **Input neurons** to **Selected Rows**, select the target
   region for this scope, and click **Run Clustering**. Repeat after clicking
   **Clear Selection** for this scope.
   **Expected:** Only the explicitly selected neuron IDs enter the run. A new
   **Voxel Correlation 1** column appears and becomes active. Its selected
   neurons receive local integer labels; every neuron outside the run is blank.
   **Soma Location 1** remains unchanged. The saved provenance records the
   selected cohort and its parent Soma Location assignment and cluster. The
   first run uses the selected region and dilation; the repeated run reports
   **All regions (optional)** and uses all finite CCFv3 nodes for those rows.
4. **Action:** Switch **Cluster assignment** between **Soma Location 1** and
   **Voxel Correlation 1**.
   **Expected:** The active header, cluster filter, summary, sort target,
   rendered colors, flatmap cluster grouping, and Analysis heatmap cluster
   choices all follow the selected assignment. Switching does not modify any
   saved labels. The voxel assignment's blank rows appear under
   **Unclustered**.
5. **Action:** Click **Rename...**, give the voxel assignment a descriptive
   name, and confirm. Then create another selected-row run.
   **Expected:** The table header and selector use the new display name while
   the next run creates another independent column. Existing Enhanced Parquet
   column identifiers remain stable.
6. **Action:** Select an assignment with live run data and build its
   dendrogram or save a single-run Analysis export.
   **Expected:** The output uses the active run. Other assignment columns are
   unaffected.
7. **Action:** In **Data** > **SWC Parquet Data**, click **Save Project As...**
   and **Export Enhanced Parquet...**. Close the session, load the saved
   project, and then load the Enhanced Parquet separately.
   **Expected:** Both reload paths restore every assignment name, sparse label
   map, active assignment, palette, cohort, parent context, and run parameters.
   Enhanced Parquet contains one nullable integer column per assignment and a
   backward-compatible `cluster_assignment` column matching the active set.
   After project reload, assignment-based filters, colors, flatmap grouping,
   and heatmaps work immediately; dendrogram and distance actions report
   **Rerun required** because matrices are not stored.
8. **Action:** With an assignment active, click **Delete**, cancel once, then
   confirm deletion.
   **Expected:** Cancelling changes nothing. Confirming removes only that
   assignment column and its saved result; no neuron rows are removed. The
   most recently created remaining assignment becomes active.
9. **Action:** Set **Input neurons** to **Selected Rows** with no selected
   rows, then with exactly one selected row, and click **Run Clustering** after
   each change. Set **Input neurons** to **Whole Parquet**, clear its target
   region selection, and try again. Repeat a valid selected-row run in **Flat
   map + Depth** for both **Soma Location** and **Voxel Correlation** when
   flatmap columns are available.
   **Expected:** Empty and one-row inputs produce actionable messages and do
   not launch a worker or create a column. Whole Parquet reports **Select at
   least one target region** and does not run. Valid flatmap runs use exactly
   the selected neuron IDs and create new sparse assignments like their CCFv3
   counterparts.
10. **Action:** Run clustering on the cohort with exactly 10,000,000 contributing
    node rows. Then run it on the cohort with at least 10,000,001 contributing
    rows; click **Cancel** in **Large Clustering Run**, repeat, and click
    **Continue**.
    **Expected:** The 10,000,000-node run starts without a warning. The larger
    run displays its exact comma-formatted node count and defaults to **Cancel**.
    Cancelling starts no clustering worker and creates no assignment; the
    repeated confirmation starts the snapshotted run without changing its
    cohort or region filter.

**Manual verification**

- Status: Not run
- Last verified: Never
- Notes: None

### UC-008: Create Combined and Individual Neuron Heatmaps

**Capability**

The user can turn selected rows in the Data-tab neuron table into either one
combined node-count heatmap or one color-matched heatmap per neuron. Individual
layers preserve the selected cohort captured when the run starts and make it
possible to compare neuron occupancy independently. A monochrome cohort is
automatically assigned distinct Turbo colors so its rendered neurons, table
swatches, and heatmaps remain visually associated.

**Prerequisites**

- Have an Allen mouse atlas available locally and a neuron Parquet containing
  at least four neurons with morphology nodes inside the atlas bounds.
- Know three neurons to use as the heatmap cohort and retain a fourth neuron as
  a non-selected color control.
- To exercise the memory warning, use an atlas resolution and selected-neuron
  count whose displayed estimate exceeds 1 GiB.
- Start from a clean napari session with the **SWC Viewer** plugin open.

**Steps and expected results**

1. **Action:** Open **Data** > **Selected Neurons**, click **Add Heatmap**, and
   inspect its menu before loading a neuron Parquet.
   **Expected:** The menu contains exactly **Single Heatmap** and **Individual
   Heatmaps**. Choosing either option reports **Load a neuron Parquet before
   creating a heatmap** and creates no layer.
2. **Action:** Load the test neuron Parquet, add its four neurons to **Selected
   Neurons**, leave the atlas unloaded, select three rows, and choose **Add
   Heatmap** > **Single Heatmap**. Then clear the table selection and try the
   action again after loading the atlas.
   **Expected:** Without an atlas, the first attempt reports **Load an atlas
   before creating a neuron heatmap**. With no selected rows, the second attempt
   reports **Select at least one neuron row to create a heatmap**. Neither
   attempt creates a layer.
3. **Action:** Give the three cohort neurons visibly different colors with
   their color swatches, select their rows, and choose **Add Heatmap** >
   **Single Heatmap**.
   **Expected:** One Greek-named image layer such as **alpha Heatmap** is added
   with the `hot` colormap. Its node counts combine all three selected neurons.
   Each cohort row lists that layer in the **Heatmap** column, the fourth row
   does not, and choosing the layer under **Manual Heatmap** shows only its
   three source rows.
4. **Action:** Keep the same distinctly colored rows selected and choose **Add
   Heatmap** > **Individual Heatmaps**. Change the table selection while the
   sequential progress messages are visible.
   **Expected:** **Add Heatmap** is disabled for the full queue. Three new
   Greek-named layers are added in deterministic order, one for each originally
   selected neuron, regardless of the later selection change. Each layer uses
   a transparent-to-neuron-color colormap, contains only that neuron's counts,
   initially sets its upper contrast limit to 20% of its maximum node count,
   and appears only on that neuron's **Heatmap** cell and **Manual Heatmap**
   filter result. Lower-density structures are more visible than with the full
   contrast range, and the limit remains editable in napari. The existing
   neuron colors do not change.
5. **Action:** Select the same three cohort rows, set all three color swatches
   to the same RGB color, record the fourth neuron's color, and choose **Add
   Heatmap** > **Individual Heatmaps** again.
   **Expected:** The three selected neurons receive distinct Turbo colors in
   stable neuron-ID order, and rendered versions of those neurons update to the
   same colors. The fourth neuron's color is unchanged. Each newly created
   heatmap matches its source neuron's new color.
6. **Action:** Set one cohort neuron back to the same color as another while
   leaving the third different, then create individual heatmaps for all three.
   Repeat with only one selected neuron.
   **Expected:** A partially duplicated but non-monochrome cohort keeps its
   existing colors, including the duplicate. A one-neuron request creates one
   individual layer without recoloring that neuron.
7. **Action:** Select enough neurons for the projected individual heatmap data
   to exceed 1 GiB and choose **Add Heatmap** > **Individual Heatmaps**. Leave
   **Cancel** selected in **Large Individual Heatmap Request** and confirm.
   **Expected:** The warning shows the selected layer count and estimated GiB
   before rendering overhead. Cancelling starts no worker, adds no layers, and
   changes no neuron colors.
8. **Action:** Delete or rename one of the individual heatmap layers, then use
   the table's **Heatmap** column and **Manual Heatmap** selector.
   **Expected:** Layer membership and names stay synchronized. Deleting a layer
   removes its membership; renaming it updates the affected row and selector
   without changing the recorded source neuron.

**Manual verification**

- Status: Not run
- Last verified: Never
- Notes: None

### UC-009: Save and Overwrite the Current Project

**Capability**

The user can create an SWC Viewer project and then save later table, analysis,
and app-created layer changes back to that same project folder. Replacement is
confirmed explicitly and publishes a complete new bundle without retaining
stale files from the previous version.

**Prerequisites**

- Load a neuron Parquet containing at least two neurons into **Data** > **SWC
  Parquet Data**.
- Add both neurons to **Selected Neurons** and create one app-generated heatmap
  or mask layer that will be recognizable after a project reload.
- Choose a writable directory where a new `.swcv` project folder can be
  created.

**Steps and expected results**

1. **Action:** With only the neuron Parquet loaded, inspect the project controls
   under **Data** > **SWC Parquet Data**.
   **Expected:** **Save Project** is disabled. **Save Project As...**, **Load
   Project...**, and **Export Enhanced Parquet...** remain available.
2. **Action:** Click **Save Project As...**, choose a new path named
   `overwrite_test.swcv`, and complete the save.
   **Expected:** The project is created and the status reports **Saved project
   bundle: overwrite_test.swcv**. **Save Project** becomes enabled and its
   tooltip identifies the new project's absolute path.
3. **Action:** Change a table label or note, remove the saved heatmap or mask
   layer, create a different app-generated layer, and click **Save Project**.
   In **Overwrite SWC Viewer Project?**, click **Cancel**.
   **Expected:** The dialog shows the exact current project path, warns that all
   existing project-folder contents will be replaced, and defaults to
   **Cancel**. Cancelling starts no save and leaves the on-disk project
   unchanged.
4. **Action:** Click **Save Project** again and click **Overwrite**.
   **Expected:** Progress is shown while saving, both save actions are disabled
   during serialization, and completion reports **Saved project bundle:
   overwrite_test.swcv**. No save-location picker is shown.
5. **Action:** Close the session, click **Load Project...**, select
   `overwrite_test.swcv`, and inspect its table state and app-generated layers.
   **Expected:** The changed table metadata and replacement layer are restored.
   The layer removed before overwrite does not return, and no stale project
   files affect the loaded session.
6. **Action:** Load a standalone neuron Parquet through **Load...**.
   **Expected:** **Save Project** becomes disabled because the session is no
   longer associated with a current project. **Save Project As...** can create
   another project, and doing so enables **Save Project** for that new folder.
7. **Action:** Load `overwrite_test.swcv`, then rename or move that folder
   outside napari and click **Save Project**.
   **Expected:** The plugin reports that the current project is unavailable or
   unrecognized, disables **Save Project**, and does not create or replace any
   folder. The user can still choose a new destination with **Save Project
   As...**.

**Manual verification**

- Status: Passed
- Last verified: 2026-08-05
- Notes: Verified by removing Heatmap layers. Two clusterings were retained.

### UC-010: Identify Axon Termini and Prune Neurons Lacking Them

**Capability**

The user can locate the tips of traced neurites. A terminus is a node no other
node claims as its parent, so restricting termini to axon-typed nodes yields the
tips of everything the source file typed as axon.

> **Caution: axon-typed does not mean axon.** Visual inspection on 2026-08-05
> found neurons in `isocortex_total_right_brainglobe_flatmap.parquet` whose
> **dendritic** projections are typed `2` (axon). Those dendrite tips are
> therefore reported as axon termini. This is a defect in the source SWC
> annotations, not in the detection: a reported node genuinely has no children,
> but the `type` on it cannot be trusted to identify the compartment. Treat every
> count below as **axon-typed** termini — an upper bound on true axon termini,
> contaminated by an unknown number of dendrite tips. The extent has not been
> quantified. Verify visually before drawing a biological conclusion from these
> points, and do not assume node types partition a neuron into clean axon and
> dendrite subtrees.

The other risk the node-type restriction carries is coverage. Reconstructions
whose neurites are all typed `Undefined` contain no axon-typed nodes, so they
contribute nothing. The plugin therefore always reports how many neurons were
skipped instead of quietly returning a partial answer.

Because those skipped neurons are exactly the ones worth excluding from a
clustering run, the section doubles as a data-quality gate on the neuron table:
after a run it can select either the neurons that produced termini or the ones
that did not, so the unwanted group can be removed with **Remove Selected From
Table** before clustering.

Two rules keep the count correct, and both follow from the childless test needing
to see the whole tree: the node-type restriction narrows only which termini are
reported, and any region restriction is applied after detection. Filtering nodes
away before the test would leave their parents looking childless.

**Prerequisites**

- A neuron Parquet with `file_id`, `node_id`, `parent_id`, `type`, `x`, `y`, and
  `z` columns loaded into **Data** > **SWC Parquet Data**.
- No atlas is required; terminus detection is pure topology.
- For the reference numbers below, use
  `isocortex_total_right_brainglobe_flatmap.parquet` (18,621 neurons,
  728,703,227 nodes). Useful individual neurons in that file:
  - `17099_002_reg.swc` — axon labelled, 295 axon termini, no dendrite nodes.
  - `192309_031_reg.swc` — axon and dendrite labelled, 91 axon termini and 89
    dendrite termini.
  - `212064_001.swc` — every neurite typed `Undefined`, so 0 axon termini.

**Steps and expected results**

1. **Action:** Open the **Data** tab, scroll below **Selected Neurons**, and
   expand **Termini**.
   **Expected:** The section explains that it finds childless nodes of the
   selected types, that neurons typed entirely `Undefined` are reported as
   skipped, and how to use the selection to prune the table. **Neurons:**
   defaults to **Current Table**, **Node types:** shows **Axon**, and **Point
   size:** shows `20.0`. **Find Termini** is enabled once a Parquet is loaded,
   with or without an atlas. **Select in Table** is disabled until a run
   finishes.
2. **Action:** Add `17099_002_reg.swc` to **Selected Neurons**, set **Neurons:**
   to **Selected Rows**, and click **Find Termini**.
   **Expected:** Progress is shown, then a **Termini (Axon)** points layer
   appears. The coverage line reads **295 termini (Axon) in 1 of 1 neurons** with
   no skipped-neuron clause, and **Copy Skipped Neuron IDs** stays disabled.
3. **Action:** Hide every layer except the neuron trace and the new points layer,
   then inspect the arbor in 3D.
   **Expected:** Points sit at the visible free ends of the axon branches, not
   mid-branch and not at branch points. The soma carries no point.
4. **Action:** Select `192309_031_reg.swc` alone and click **Find Termini** with
   **Node types:** still **Axon**.
   **Expected:** The coverage line reads **91 termini (Axon) in 1 of 1 neurons**,
   and it also reports that **89 childless nodes of other types** were not
   counted. Points land on the far-reaching axon branches, none on the compact
   arbor next to the soma.
5. **Action:** Change **Node types:** to **Basal dendrite** only and click **Find
   Termini** again.
   **Expected:** A **Termini (Basal dendrite)** layer replaces nothing — it is a
   separate layer — and reports **89 termini (Basal dendrite) in 1 of 1
   neurons**. For this particular neuron its points form a compact cluster close
   to the soma, distinct from the axon-typed termini of the previous step. Note
   that this separation is a property of how *this* file was annotated, not a
   guarantee: see the caution above and step 6.
6. **Action:** Repeat steps 2-4 on several other neurons, hiding all layers
   except the neuron trace and the **Termini (Axon)** points, and look
   specifically for axon-typed points sitting on short, soma-proximal,
   dendrite-shaped arbors rather than on long-range projections.
   **Expected:** At least some neurons show axon-typed termini on plainly
   dendritic branches. This is the annotation defect described in the caution
   above, not a detection error — the plugin faithfully reports the tips of
   whatever the file typed as axon. Record which neurons show it; the extent
   across the file has not been measured.
7. **Action:** Select `212064_001.swc` alone, set **Node types:** back to
   **Axon**, and click **Find Termini**.
   **Expected:** No points layer is added, and the coverage line still appears,
   reading **0 termini (Axon) in 0 of 1 neurons — 1 neurons skipped (no nodes of
   the selected types)** along with the count of childless nodes not counted.
   **Copy Skipped Neuron IDs** becomes enabled.
8. **Action:** Click **Copy Skipped Neuron IDs** and paste into a text editor.
   **Expected:** The status line reports how many IDs were copied, and the
   clipboard holds `212064_001.swc`.
9. **Action:** Set **Neurons:** to **Whole Parquet**, keep **Node types:** as
   **Axon**, and click **Find Termini**.
   **Expected:** Progress advances in neuron batches, reporting
   **Scanning neurons for termini (N/18,621)**, and napari stays responsive
   throughout. The run takes roughly 70 seconds on a workstation with the file on
   local disk. The coverage line then reads **3,207,618 termini (Axon) in 13,750
   of 18,621 neurons — 4,871 neurons skipped (no nodes of the selected types);
   1,521,610 childless nodes of other types not counted**. Because more than 200
   neurons were skipped, the line also states that only the first 200 skipped
   neuron IDs are listed. Memory use stays bounded: detection runs a batch of
   neurons at a time rather than the whole file in one query.
10. **Action:** Set **Neurons:** to **Selected Rows** with no table rows selected
    and click **Find Termini**.
    **Expected:** No worker starts and no layer is added. The status line reports
    that no table rows are selected and suggests switching to **Whole Parquet** or
    populating the table.
11. **Action:** Select the **Termini (Axon)** layer and inspect its metadata (for
    example through the napari console).
    **Expected:** `file_ids_per_point`, `node_ids`, and `point_types` are present
    and the same length as the point data, so each point traces back to its
    source node. `coverage_summary` and `skipped_file_ids` are also recorded.
12. **Action:** Populate **Selected Neurons** with a mix of neurons including
    `17099_002_reg.swc` and `212064_001.swc`, leave **Neurons:** on **Current
    Table** and **Node types:** on **Axon**, then click **Find Termini**.
    **Expected:** The coverage line reports one skipped neuron, and **Select in
    Table** becomes enabled.
13. **Action:** Set the selection dropdown to **Neurons lacking termini** and
    click **Select in Table**.
    **Expected:** Only `212064_001.swc` is selected in the table, and the status
    line reads **Selected 1 of N table neurons lacking termini.**
14. **Action:** Set the dropdown to **Neurons with termini** and click **Select
    in Table** again, without re-running detection.
    **Expected:** The complement is selected — every table neuron except
    `212064_001.swc` — and the status line reports that count. The dropdown
    switches groups without another detection run.
15. **Action:** Switch back to **Neurons lacking termini**, click **Select in
    Table**, then click **Remove Selected From Table** in **Selected Neurons**.
    **Expected:** `212064_001.swc` leaves the table, the table summary updates,
    and a clustering run started from the **Analysis** tab with **Current Table**
    scope no longer includes it.
16. **Action:** With the table still populated, set **Neurons:** to **Selected
    Rows**, select only a subset of rows, click **Find Termini**, then use
    **Select in Table** on either group.
    **Expected:** Only rows inside the analyzed subset are considered, and the
    status line additionally reports how many table rows were outside the
    analyzed scope, suggesting a re-run to cover them. Rows the run never
    examined are never selected on a guess.
17. **Action:** Clear the neuron table, leave **Neurons:** on **Current Table**,
    and click **Find Termini**.
    **Expected:** No worker starts and no layer is added. The status line reports
    that the current table is empty and suggests switching to **Whole Parquet**
    or populating the table.

**Manual verification**

- Status: Partially run
- Last verified: 2026-08-05
- Notes: **Step 6 found the annotation defect.** Visual inspection in napari
  confirmed that some neurons in
  `isocortex_total_right_brainglobe_flatmap.parquet` have dendritic projections
  typed `2` (axon), so their dendrite tips are reported as axon termini. The
  detection itself is behaving correctly — the source SWC types are wrong. The
  number of affected neurons was not counted; quantifying it is open work.
  Because of this, every "axon termini" count in this use case is really a count
  of **axon-typed** termini.
  The counts in steps 2, 4, 5, 7, and 9 were confirmed against that same file by
  running the detection functions over all 728,703,227 rows, but those UI steps
  have not been exercised. Steps 1 and 12-17 cover the move to the **Data** tab
  and the table-selection dropdown and have not been exercised in napari either.

## Use-Case Template

Copy this section when adding a use case. Remove guidance in parentheses and
replace `XXX` with the next sequential identifier.

### UC-XXX: Descriptive Workflow Name

**Capability**

(Explain what the plugin lets the user accomplish and why the workflow is
useful.)

**Prerequisites**

- (List required input files, atlas data, existing layers, configuration, or
  starting state.)

**Steps and expected results**

1. **Action:** (Describe one user action using the visible UI labels.)
   **Expected:** (Describe the observable result of that action.)
2. **Action:** (Continue until the workflow and its final outcome are
   complete.)
   **Expected:** (Describe the observable result.)

**Manual verification**

- Status: Not run
- Last verified: Never
- Notes: None
