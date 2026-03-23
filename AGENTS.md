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
