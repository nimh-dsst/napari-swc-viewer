# AGENTS

This repository uses [pixi](https://pixi.sh) as the standard workflow for development, dependency management, launching napari, and running tests.

## Critical: `file_id` Is The Only Per-Neuron Key

**`neuron_id` is not unique. Never group, join, or index per-neuron data by it.**

In `isocortex_total_right_brainglobe_flatmap.parquet` there are 12,425 distinct
`neuron_id` values for 18,621 neurons, because the same `neuron_id` repeats
across `subject` values. Grouping on it silently merges different neurons into
one, and because `node_id` is only unique *within* a `file_id`, the merged
neurons' node identifiers collide. Nothing errors. The results are just wrong:
in terminus detection, one neuron's node with children masks another neuron's
genuine childless node, so real axon termini disappear from the output.

- Group, join, and index by **`file_id`**. Use `subject` + `neuron_id` only if
  `file_id` is genuinely unavailable.
- `neuron_id` is for display only — it is shorter and nicer in labels. That is
  the sole reason it appears in soma labels and similar UI.
- Any node-graph computation (parent/child edges, degree, leaves, branch points,
  path lengths, subtrees) must be scoped per `file_id`.
- `tests/test_terminals.py::test_duplicate_node_ids_across_files_stay_separated`
  guards this. Keep an equivalent test when adding new per-neuron graph code.

Related node-identifier facts verified against that same file, which also break
naive graph code:

- `node_id` is **not** contiguous and does not always start at 1 (13,771 of
  18,621 neurons). Do not use it as an array index.
- `parent_id` is **not** always less than `node_id` (1,643,105 rows violate it).
  Do not assume rows arrive in tree order.
- Root nodes use `parent_id == -1`. Each neuron in that file has exactly one
  root and one soma, and no dangling parent references — but validate rather
  than assume this for other files.

## Data Filtering Rule For Graph Topology

**A childless/leaf test must see the whole tree.** When selecting nodes by
`type`, region, or any spatial bound, apply the restriction to *which nodes are
reported*, never to the node set the child lookup searches. Filtering nodes away
first strands their parents, which then look childless and get reported as
termini. See `src/napari_swc_viewer/terminals.py` for the correct shape and the
two tests that guard both traps.

Note that `type = 2` covers only 13,750 of the 18,621 neurons in that file; the
other 4,871 have every non-soma node typed `0` (undefined). Compartment-filtered
results must report how many neurons were excluded rather than returning a
partial answer silently.

## Flat Map X Spans Both Hemispheres, Y Spans One

**Never give the two flat map axes the same bin count.** The bilateral map lays
the hemispheres side by side along `x`, so `x` covers roughly twice the extent of
`y`. Equal counts make every bin about twice as wide as it is tall, which throws
away `x` detail at twice the rate of `y` and draws the map horizontally squashed
(no `layer.scale` is set on any flat map layer).

Measured from the canonical bounds in
`isocortex_total_right_brainglobe_flatmap.parquet`:

| style | x span | y span | ratio |
|---|---|---|---|
| `both_square` | 2.0000 | 1.0000 | 2.0000 |
| `both_shaped` | 1.8085 | 0.9434 | 1.9169 |

- The user-facing control is **`y_bins`**. Derive `x` with
  `resolve_flatmap_bin_counts()` in `flatmap_heatmap.py`, which every grid
  builder reaches through `_resolve_axis_bin_counts()`. At the default 256 that
  gives `(256, 512)` for square and `(256, 491)` for shaped.
- **Do not write `x_bins = 2 * y_bins`.** That is right only for `both_square`;
  for `both_shaped` it leaves 4.2% anisotropy. `analysis/flatmap_correlation.py`
  already contains a variant of this trap.
- Do not call `y_bins` "bins per hemisphere" — at 256 the shaped style gets 245
  x bins per hemisphere, not 256.
- Rounding is pinned to `floor(v + 0.5)`, not `round()`: banker's rounding is
  non-monotone at ties (`round(490.5) == 490` but `round(491.5) == 492`) and this
  count feeds the region-cache identity digest.
- **Never re-derive a stored `x_bins`.** Entry points take
  `y_bins: int, x_bins: int | None = None`; a cache-backed caller passes the
  profile's recorded value verbatim, because a JSON float round trip could change
  the derived integer at a tie and the render would then be discarded as a grid
  mismatch. Cache validators check internal consistency only, never policy
  conformance.
- The region-cache profile identity key is `y_bins`, never `x_bins`: one profile
  builds both styles and they resolve to different `x_bins` (491 vs 512).
- Because `x_bins == y_bins` used to hold everywhere, any latent `(y, x)` vs
  `(x, y)` transpose was invisible. Treat surprise failures in this area as
  probable pre-existing bugs, and verify renders visually rather than only by
  shape assertions. `tests/test_flatmap_rectangular_grid.py` guards the axis
  order, the vector/heatmap co-registration, and cross-subsystem agreement.

**The soma-clustering metric follows the same policy.**
`resolve_flatmap_depth_normalization` divides *both* flat map axes by the `y`
span, so flat map space is scaled without being distorted. It used to divide `x`
by `x_span / 2` and `y` by `y_span`, which gave `both_shaped` a divisor ratio of
0.9581 — a 4.2% anisotropy. Both subsystems now take `y` as the reference axis:

- The grid's bin width is `y_span / y_bins` on both axes.
- The metric's unit is the `y` span, i.e. one hemisphere *height*.

`FlatmapDepthNormalization` carries **one** `flatmap_divisor`, not one per axis,
so the anisotropy cannot return by editing a single number. Do not reintroduce
per-axis divisors, and do not scale `x` by `x_span / 2` — the `x` extent is not a
per-hemisphere quantity. `depth_um` keeps its own divisor because it is a
different physical quantity; `depth_scale` weights it.

Fixing this changed `both_shaped` soma-clustering results. Measured on all
18,518 somas in `isocortex_total_right_brainglobe_flatmap.parquet` with
hierarchical/ward at `depth_scale=1.0`, versus the old anisotropic metric:

| style | k=5 | k=10 |
|---|---|---|
| `both_shaped` | 37.5% moved (ARI 0.44) | 38.7% moved (ARI 0.47) |
| `both_square` | 0.0% (ARI 1.00) | 0.0% (ARI 1.00) |

**Always quote the algorithm and the neuron count with a change rate.** These
figures are ward at full scale and nothing else reproduces them: seeded k-means
over the same 18,518 somas moves 1.4% at k=5, and ward over an 833-soma subset
moves 1.9%. Ward is agglomerative, so a small metric perturbation flips
near-tied merges and the reordering cascades, and more somas means more
near-ties. A rate from a partial table therefore does **not** bound the
full-dataset rate — the two differ ~20x here. Do not read a small number from a
subset as evidence the metric barely moved.

`both_square` is an exact 2:1 map whose old divisors were already equal
(`x_span / 2 == 1.0`, `y_span == 1.0`), so it is bit-identical — any change
there is a bug, not the fix. That makes a square-style pair the sharpest
available regression control, because it exercises the same code path the fix
changed. The `distance_metric` recorded in exports was renamed to
`euclidean_flatmap_isotropic[_plus_depth]` so results from before the change
cannot be silently compared as if they matched.

`scripts/compare_clustering_exports.py` measures all of this from paired
pre/post workbook exports (`--dir anisotropy_fix_data`, which is gitignored —
this project keeps no data files in version control, so the workbooks must be
regenerated locally). It gates every pair on comparability before
reporting a number, and it matches cluster labels optimally first, because
cluster ids are arbitrary — a raw label diff reports 99.6% for the shaped k=10
pair whose true movement is 38.7%. `ANISOTROPY_FIX_RESULTS.MD` records the full
validation, including the two controls that held.

## Caution: `type` Is Not A Trustworthy Compartment Label

**Some neurons in `isocortex_total_right_brainglobe_flatmap.parquet` have
dendritic projections typed `2` (axon).** This was found on 2026-08-05 by
visually inspecting detected termini in napari: points labelled as axon termini
sat on arbors that are plainly dendritic.

This is a defect in the source SWC annotations, not in this repository's code.
The childless/leaf detection in `src/napari_swc_viewer/terminals.py` is correct —
a reported node genuinely has no children. What is unreliable is the *meaning*
of the `type` column on those nodes.

Consequences for anything that filters by `type`:

- Do not describe `type = 2` results as "axon" anything. They are **axon-typed**
  nodes. A count of axon termini is an upper bound contaminated by an unknown
  number of dendrite tips.
- Do not claim that node types partition a neuron into clean subtrees, that axon
  and dendrite compartments never interleave, or that an axon-typed terminus
  cannot be a dendrite tip. All three are false for this dataset.
- Compartment-based results need visual or geometric verification before they
  support a biological conclusion. Type alone is not evidence.
- The extent is **not quantified**. Nobody has counted how many neurons or nodes
  are mislabelled, and there is no per-neuron flag for it. Do not state or imply
  a rate until someone measures one.

This does not affect purely topological work (childless tests, degree, branch
points, path lengths) as long as it does not filter or interpret by `type`.

## Expected Workflow

- Use `pixi` commands instead of creating a separate virtual environment or invoking `pip`, `pytest`, or `napari` directly.
- Develop `napari_swc_viewer` inside the pixi environment.
- Let pixi manage installation of napari and the other project dependencies.

## Common Commands

- `pixi run build`
  Installs `napari_swc_viewer` in editable mode for development.

- `pixi run napari`
  Launches napari with the plugin available. This task depends on `build`, so the package is installed before napari starts.

- `pixi run test`
  Runs the test suite.

- `pixi run test-cov`
  Runs the test suite with coverage output.

## Notes For Agents

- When setting up or working in this repository, assume pixi is the source of truth for the environment.
- If you need to verify behavior locally, prefer the pixi tasks defined in `pixi.toml`.
- In Codex's sandboxed environment, Qt-dependent tests may not run successfully. Do not rely on Codex to validate Qt tests from inside the sandbox.

## Use-Case Documentation

`USE_CASES.md` is both the user-facing catalog of plugin capabilities and the
repository's set of repeatable manual test cases. When the user describes a
workflow they want documented as a use case, add it to `USE_CASES.md`.

- Give each use case a stable, sequential identifier (`UC-001`, `UC-002`, and
  so on) and a descriptive title.
- Describe the user-visible capability and why someone would use it.
- Record prerequisites and test data precisely enough for another person to
  repeat the workflow.
- Write numbered actions using the labels visible in the napari interface.
- Pair the actions with observable expected results. Include important error,
  cancellation, and boundary behavior when it is part of the workflow.
- Inspect the current implementation when necessary so the documented steps
  and UI labels are accurate. Do not infer that a use case passed merely from
  reading code or running automated tests.
- Record the manual verification status and date. Leave the status as `Not
  run` when the workflow has not actually been exercised in napari.
- Update an existing use case instead of creating a duplicate when the user is
  refining an already documented workflow.
- Keep entries useful as both capability documentation and standalone manual
  tests; do not assume the reader has access to the conversation that produced
  them.
