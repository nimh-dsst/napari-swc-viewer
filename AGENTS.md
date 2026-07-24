# AGENTS

This repository uses [pixi](https://pixi.sh) as the standard workflow for development, dependency management, launching napari, and running tests.

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
