# napari-swc-viewer
A Napari plugin that allows viewing of SWC files in napari

## Prerequisites

### Installing Pixi

This project uses [Pixi](https://pixi.sh) for environment and dependency management. To install Pixi, follow the instructions at:

https://pixi.sh/latest/#installation

## Installation

### Building and Installing the Package

To build and install the package in development mode:

```bash
pixi run build
```

This will install the package in editable mode, allowing you to make changes without reinstalling.

### Running Napari

To launch napari with the plugin installed:

```bash
pixi run napari
```

This command will automatically build the package (if needed) before launching napari.

### Running Tests

To run the test suite:

```bash
pixi run test
```

To run tests with coverage:

```bash
pixi run test-cov
```
