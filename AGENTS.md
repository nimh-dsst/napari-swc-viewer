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
