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

**Prerequisites**

- Complete UC-001 for `allen_mouse_25um` so the atlas needed by this test is
  available locally.
- Use valid SWC files registered to the Allen CCF in micrometers.
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
   `Resolution (μm)`, `Hemisphere alignment`, a status area, and a progress
   indicator that is initially hidden.
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

If an individual SWC cannot be parsed, the converter skips it, keeps processing
the valid files, and includes the skipped count in the completion status. If no
SWC files are found or none can be processed successfully, the status displays
an error instead of reporting successful conversion.

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
