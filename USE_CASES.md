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
| [UC-004](#uc-004-build-and-reuse-a-flatmap-region-cache) | Build, reopen, parse, and switch shaped/square region-cache data | Not run |

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
   BrainGlobe atlas with the same atlas/version structure catalog, then choose
   **Precomputed Parquet + Cache** and click **Choose Cache Directory...**.
   **Expected:** The plugin parses `flatmap-region-cache.json`, memory-maps its
   arrays, and lists only profiles compatible with the Parquet lookup-set ID
   and selected style. It does not access atlas annotation or region mesh data.
4. **Action:** Select the new cache profile.
   **Expected:** XY bins, depth-bin size, canonical bounds, and exclusion of the
   depth `-1` sentinel plane are set from the profile and locked.
5. **Action:** Select a parent region, enable child regions, then click **Show
   Region Labels**, **Show Region Surfaces**, and **Show Region Outlines**.
   **Expected:** Labels apply the include-child-expanded selection. The surface
   and outlines use the selected parent's cached descendant union and atlas
   color. The resulting **Flatmap Region Labels**, **Flatmap Region Surfaces**,
   and **Flatmap Region Outlines** layers align with the neuron heatmap.
6. **Action:** Switch between **Both hemispheres, shaped** and **Both
   hemispheres, square** without changing the Parquet or cache directory, then
   project and show the desired cached region layers again.
   **Expected:** Stale layers from the previous style are cleared. Neurons and
   newly shown cached region layers use the matching profile grid and remain
   aligned, including when the queried neurons occupy only a small part of the
   global flatmap.
7. **Action:** Load a Parquet or atlas that is incompatible with every cached
   profile, or remove one referenced array and then reopen the cache.
   **Expected:** The plugin reports the exact lookup, atlas, grid, missing-file,
   or validation mismatch. It never starts NRRD recomputation automatically.
8. **Action:** Explicitly choose **Recompute from NRRDs**.
   **Expected:** The legacy runtime lookup workflow becomes available only for
   this explicit fallback choice.

**Manual verification**

- Status: Not run
- Last verified: Never
- Notes: None

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
