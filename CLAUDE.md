# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

napari-swc-viewer is a napari plugin for viewing SWC files. SWC is a standard file format for representing neuron morphology (neuronal reconstructions with branching structures).

## Package Management

This project uses **pixi** for Python package management. Use `pixi` commands (not `uv`, `pip`, or `conda` directly) for dependency management and running tasks.

## Qt Backend — NEVER use PyQt5

This project uses **PyQt6** exclusively. **Never** add PyQt5 or PySide2 as a dependency. PyQt5 cannot find NVIDIA OpenGL drivers on Windows, causing hours of debugging pain. If napari or any dependency tries to pull in PyQt5, override it with PyQt6. The `napari[pyqt6]` extra in pyproject.toml and `pyqt6` in pixi.toml enforce this.

## Development Context

This is a napari plugin that:
- Uses napari's plugin architecture with entry points defined in pyproject.toml
- Implements reader hooks (napari_get_reader) and widget contributions
- Depends on napari, numpy, duckdb, brainglobe-atlasapi, and other scientific Python packages
- Source code lives in `src/napari_swc_viewer/`
