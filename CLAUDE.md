# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

napari-swc-viewer is a napari plugin for viewing SWC files. SWC is a standard file format for representing neuron morphology (neuronal reconstructions with branching structures).

## Package Management

This project uses **pixi** for Python package management. Use `pixi` commands (not `uv`, `pip`, or `conda` directly) for dependency management and running tasks.

## Development Context

This is a napari plugin that:
- Uses napari's plugin architecture with entry points defined in pyproject.toml
- Implements reader hooks (napari_get_reader) and widget contributions
- Depends on napari, numpy, duckdb, brainglobe-atlasapi, and other scientific Python packages
- Source code lives in `src/napari_swc_viewer/`
