# 3D Neuron Rendering Optimization Plan

## Goal

Make 3D neuron rendering fast and stable for large SWC Parquet datasets while preserving the current user workflows:

- Render selected neurons as full traces in 3D.
- Preserve per-neuron color, visibility, highlighting, and cluster recoloring.
- Keep soma labels and soma-only rendering behavior.
- Keep the existing `Shapes` implementation available as a compatibility fallback.

## Current Bottleneck

The current 3D line path is in `NeuronViewerWidget._render_scene`:

- Query selected neurons with `NeuronDatabase.get_neuron_lines_batch`.
- Expand each parent-child edge into an explicit segment array.
- Concatenate all segments into `merged_lines`.
- Concatenate per-segment colors into `merged_colors`.
- Call `viewer.add_shapes(..., shape_type="line", edge_color=merged_colors)`.
- Immediately populate the 2D slice-projector cache from the same full geometry.

This is slow because napari `Shapes` treats every SWC edge as a separate shape. For 3D line shapes, napari builds thick tube mesh geometry for each segment before rendering. That creates many Python objects, per-shape metadata, mesh vertices, triangles, colors, and renderer buffers.

Local scale check:

- `neurons.parquet`: 26 neurons, 1,739,157 rows, about 1,739,131 parent-child segments.
- The merged line and color arrays alone are about 133 MB before napari builds shape objects and tube meshes.
- On this dataset, DuckDB query plus local segment construction is under 2 seconds; the likely dominant cost is `viewer.add_shapes`.

The larger local files are far beyond practical full-resolution shape rendering:

- `pfc_left.parquet`: about 303 million rows.
- `isocortex_total_left_brainglobe.parquet`: about 729 million rows.

## Design Principles

- Avoid one napari `Shapes` object per SWC edge for large 3D views.
- Avoid building 2D projection caches unless the user enables 2D slice projection.
- Prefer contiguous `float32` geometry buffers for render-only paths.
- Keep the current `Shapes` backend as a fallback for visual parity and debugging.
- Add guardrails before attempting renders that are likely to exhaust memory or freeze the UI.
- Measure each phase separately: database query, geometry construction, layer creation, first draw, and recolor.

## Phase 0: Instrument The Existing Path

Add debug timings around the current renderer before changing behavior:

- Time `get_neuron_lines_batch`.
- Time segment array construction.
- Log selected neuron count, node count, segment count, and array memory.
- Time `viewer.add_shapes`.
- Time projector cache rebuild.
- Log render mode, line width, opacity, atlas scale, and napari `ndisplay`.

Acceptance:

- Debug logs identify whether load time is dominated by query, geometry construction, layer creation, or projector cache construction.
- The status label shows enough progress that a long render does not look dead.

## Phase 1: Low-Risk Fixes

### 1. Defer 2D Projector Cache Construction

Current behavior always builds the `NeuronSliceProjector` source arrays after 3D lines are rendered.

Change:

- Store enough render geometry metadata to build the projector later.
- Only call `self._slice_projector.add_neuron_data_batch(...)` when `Show in 2D slices` is enabled.
- If projection is already enabled, keep current behavior.

Expected benefit:

- Faster 3D-only render startup.
- Lower memory during normal 3D viewing.

### 2. Stop Auto-Enabling 2D Slice Projection After 3D Render

Current behavior turns `Show in 2D slices` on after adding neurons. That adds extra work and can surprise users.

Change:

- Keep the checkbox state user-controlled.
- Do not enable projection automatically after a 3D render.
- If needed, show a short status hint that 2D projection is available from the Visualization tab.

### 3. Use `float32` Render Buffers

Current code builds `float64` coordinate arrays for lines.

Change:

- Convert render-only line buffers to `np.float32` before passing them to napari.
- Keep analysis/database coordinates at their existing precision.

Expected benefit:

- About 50% lower coordinate-buffer memory for render geometry.
- Lower transfer cost into renderer buffers.

### 4. Add Render Guardrails

Before creating a full 3D layer:

- Estimate selected rows and segments.
- Estimate raw render buffer memory.
- Warn or require confirmation above configurable thresholds.
- Offer alternatives: soma-only, points, fast renderer, or lower-resolution preview.

Suggested initial thresholds:

- Warn at 250,000 segments.
- Strong warning at 1,000,000 segments.
- Block legacy `Shapes` rendering by default above 2,000,000 segments unless the user explicitly overrides.

## Phase 2: Introduce Renderer Backends

Create an internal renderer abstraction so the widget can switch implementation without spreading backend-specific logic through `neuron_viewer.py`.

Suggested structure:

```text
src/napari_swc_viewer/rendering/
  __init__.py
  geometry.py
  renderer_base.py
  shapes_renderer.py
  fast_line_renderer.py
```

Suggested responsibilities:

- `geometry.py`: build segment buffers, color buffers, file ID offsets, and per-neuron segment counts.
- `renderer_base.py`: common interface for add, update colors, update visibility, set width, remove, and metadata.
- `shapes_renderer.py`: current implementation, kept as compatibility fallback.
- `fast_line_renderer.py`: optimized render-only implementation.

Minimum backend interface:

```python
class NeuronLineRenderer:
    name: str

    def add_lines(self, viewer, geometry, *, width, opacity, scale): ...
    def update_colors(self, color_map): ...
    def update_width(self, width): ...
    def set_visible(self, visible): ...
    def remove(self): ...
```

The widget should not need to know whether lines are rendered as Shapes, Vectors, or a custom visual.

## Phase 3: Replace The Default 3D Line Backend

Evaluate two optimized backends and choose the default based on measured behavior.

### Option A: Fast Segment Backend

Render all parent-child segments as one or a small number of contiguous line buffers instead of millions of `Shapes`.

Candidate implementations:

- Napari `Vectors` layer using `(N, 2, 3)` `[start, direction]` data.
- Custom VisPy line visual attached as a napari-compatible layer or overlay.

Advantages:

- Avoids per-segment `Shape` objects.
- Avoids tube mesh generation for every SWC edge.
- Fits the actual requirement: display static 3D traces, not edit shapes.

Risks:

- 3D line width support may vary by backend/GPU.
- Direct per-neuron picking may be limited.
- Per-segment direct color updates must be tested with cluster recoloring and visibility changes.

Decision rule:

- Prefer this path if it can render 1.7M segments without freezing and supports direct colors.
- Keep `Shapes` fallback for cases where thick tube rendering is required.

### Option B: Branch Path Backend

Convert each neuron from individual parent-child segments into branch polylines and render those as fewer path shapes.

Advantages:

- Preserves tube-style appearance better than simple line primitives.
- Reduces shape object count from number of edges to number of branches.
- May reduce mesh vertices by sharing polyline vertices along each branch.

Risks:

- Still uses napari `Shapes` and still creates 3D tube meshes.
- Building branches adds complexity.
- May still be too heavy for million-edge scenes.

Decision rule:

- Use as a compatibility/per-quality backend, not the first default for very large datasets.

## Phase 4: Level Of Detail And Chunking

For very large selections, full-resolution rendering should not be the only option.

Add one or more preview modes:

- Segment stride downsampling.
- Spatial voxel/grid downsampling.
- Per-neuron max segment cap.
- Distance-based or screen-space level of detail.
- Soma-first render, then progressive full-trace loading.

Add chunked rendering:

- Split line buffers into chunks by neuron or segment count.
- Add chunks incrementally with progress updates.
- Allow cancellation during long renders.
- Keep chunk metadata so visibility and recoloring still work.

Acceptance:

- Large scenes remain responsive while loading.
- Users can cancel before memory pressure becomes severe.
- A preview appears quickly, even if full resolution continues in the background.

## Phase 5: Optimize Data Retrieval

The current `get_neuron_lines_batch` builds parent-child edges in Python after loading rows from DuckDB.

Potential improvements:

- Query parent-child pairs directly with a DuckDB self-join.
- Return segment endpoints directly instead of coordinates plus edge indices for render-only paths.
- Keep the existing coordinate-plus-edge path for projector and analysis use cases.
- Consider caching selected geometry for repeated add/remove/recolor workflows.

Example direction:

```sql
SELECT
    child.file_id,
    parent.x AS x0,
    parent.y AS y0,
    parent.z AS z0,
    child.x AS x1,
    child.y AS y1,
    child.z AS z1
FROM neurons child
JOIN neurons parent
  ON child.file_id = parent.file_id
 AND child.parent_id = parent.node_id
WHERE child.file_id IN (...)
ORDER BY child.file_id, child.node_id
```

Acceptance:

- Query/render path can build `segments` without Python dictionaries for every neuron.
- Query timings improve or remain comparable while reducing Python object churn.

## Phase 6: UI Controls

Add explicit rendering choices in the Visualization tab:

- `3D renderer`: `Fast lines`, `Smooth tubes (legacy)`, `Auto`.
- `Max segments`: numeric control or presets.
- `Preview mode`: off, stride, spatial.

Default behavior:

- `Auto` selects fast lines above a conservative segment threshold.
- `Smooth tubes (legacy)` remains available for small scenes or publication-quality screenshots.
- 2D slice projection remains opt-in.

## Validation Plan

### Unit Tests

Add tests for:

- Segment-buffer construction uses `float32` for render-only geometry.
- Segment counts and per-neuron offsets match current behavior.
- Color updates produce the same per-segment colors as the existing `Shapes` layer path.
- Visibility maps set alpha consistently.
- Slice-projector geometry is not built until projection is enabled.
- Renderer selection chooses fast backend above threshold and legacy backend below threshold.

### Benchmarks

Add or extend scripts to measure:

- Query time.
- Geometry build time.
- Layer creation time.
- First-draw time, if available.
- Peak memory, where practical.
- Recolor time.
- Visibility toggle time.

Datasets:

- Synthetic 10k, 100k, 1M, and 2M segment datasets.
- Local `neurons.parquet`.
- A sampled subset of `pfc_left.parquet`.

### Manual Napari Checks

Validate in a real napari session:

- 3D render appears correctly.
- Axis scaling aligns with atlas layers.
- Per-neuron colors match the table.
- Cluster recoloring works.
- Visibility toggles work.
- Line width control works or degrades clearly for the fast backend.
- Switching 2D/3D does not rebuild expensive geometry unnecessarily.
- Clearing/removing neurons releases layers and memory.

Note: Qt-dependent rendering checks may not run reliably in the Codex sandbox. Treat manual napari validation as required before release.

## Acceptance Criteria

Initial target for `neurons.parquet` full 26-neuron render:

- Avoid UI freeze during long load.
- Avoid automatic 2D projector cache work.
- Reduce memory pressure versus legacy `Shapes`.
- Reduce full 3D layer creation time substantially compared with legacy `Shapes`.
- Preserve recoloring, visibility, clearing, and soma-label behavior.

Large dataset target:

- App should refuse or preview extremely large full-resolution renders instead of freezing or exhausting memory.
- Users should be guided toward soma-only, sampled preview, or fast renderer modes.

## Suggested Implementation Order

1. Add timing logs and segment/memory estimates.
2. Defer 2D projector cache construction and stop auto-enabling projection.
3. Convert render-only geometry buffers to `float32`.
4. Add render guardrails and warnings.
5. Extract renderer backend interface while keeping the current `Shapes` backend.
6. Prototype fast segment backend with `Vectors`.
7. If `Vectors` is insufficient, prototype a custom VisPy line visual.
8. Add branch-path backend only if smooth tube rendering remains important for medium-size scenes.
9. Add LOD, chunking, and cancellation for very large selections.

## Open Questions

- Is thick tube-like rendering visually required for the primary workflow, or are fast 3D line primitives acceptable?
- Is per-segment picking needed, or is table-driven selection enough?
- What dataset size should be considered the normal target: tens of neurons, hundreds, or thousands?
- Should the default render mode become soma-only or preview for very large region queries?
- Should full-resolution rendering run in a background worker with cancellation, or should the app require a smaller selection first?
