# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

napari-swc-viewer is a napari plugin for viewing SWC files. SWC is a standard file format for representing neuron morphology (neuronal reconstructions with branching structures).

## Project Status

This is a new project. The repository currently contains only the README, LICENSE, and .gitignore. Source code and build configuration have not yet been created.

## Development Context

This will be a napari plugin, which typically:
- Uses napari's plugin architecture with entry points defined in pyproject.toml
- Implements reader hooks (napari_get_reader) and/or widget contributions
- Depends on napari and commonly uses numpy for data handling
