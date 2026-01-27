# napari-swc-viewer

A napari plugin for viewing SWC neuron morphology files with brain region filtering, hemisphere alignment, and scalable Parquet/DuckDB backend.

## Features

- **SWC File Viewing**: Load and visualize neuron morphologies as lines or points
- **Parquet Backend**: Convert large SWC datasets to Parquet for fast querying
- **Brain Region Filtering**: Query neurons by Allen CCF brain regions
- **Hemisphere Alignment**: Detect and flip neurons to a consistent hemisphere
- **Reference Layers**: Overlay Allen CCF template, annotations, and region meshes
- **Interactive Widgets**: Hierarchical region selector with search

## Prerequisites

### Installing Pixi

This project uses [Pixi](https://pixi.sh) for environment and dependency management. To install Pixi:

```bash
# macOS/Linux
curl -fsSL https://pixi.sh/install.sh | bash

# Windows (PowerShell)
iwr -useb https://pixi.sh/install.ps1 | iex
```

## Installation

### Building and Installing the Package

```bash
# Clone the repository
git clone https://github.com/nimh-dsst/napari-swc-viewer.git
cd napari-swc-viewer

# Install dependencies and build
pixi run build
```

### Running Napari

```bash
pixi run napari
```

### Running Tests

```bash
pixi run test
```

---

## Example Workflow: From SWC Files to Interactive Visualization

This walkthrough demonstrates the complete workflow from downloading SWC files to interactive visualization with region filtering.

### Step 1: Download Example SWC Files

Download 5 example neurons from the Brain Image Library (BIL) dataset "Morphological diversity of single neurons in molecularly defined cell types" (DOI: 10.35077/g.73):

```python
#!/usr/bin/env python
"""download_examples.py - Download 5 example SWC files from BIL."""

import json
import re
import ssl
import urllib.request
from pathlib import Path

# Configuration
OUTPUT_DIR = Path("example_data")
NUM_FILES = 5

# BIL API endpoints
API_BASE = "https://api.brainimagelibrary.org"
DOWNLOAD_BASE = "https://download.brainimagelibrary.org"
MORPHOLOGY_SUBMISSION_UUID = "0fcde5fdd6f7ccb2"


def get_ssl_context():
    """Create SSL context for BIL API."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def api_get(endpoint: str) -> dict:
    """Make a GET request to the BIL API."""
    url = f"{API_BASE}/{endpoint}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=get_ssl_context()) as response:
        return json.loads(response.read().decode())


def download_file(url: str, output_path: Path) -> bool:
    """Download a file from URL."""
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=get_ssl_context()) as response:
            output_path.write_bytes(response.read())
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Querying BIL API...")
    result = api_get(f"query/submission?submission_uuid={MORPHOLOGY_SUBMISSION_UUID}")
    bildids = result.get("bildids", [])[:10]  # Check first 10 entries

    swc_urls = []
    for bildid in bildids:
        result = api_get(f"retrieve?bildid={bildid}")
        for entry in result.get("retjson", []):
            dataset = entry.get("Dataset", [{}])[0]
            bildirectory = dataset.get("bildirectory", "")
            if not bildirectory:
                continue

            match = re.match(r"/bil/data/(.+)", bildirectory)
            if not match:
                continue

            download_path = match.group(1)
            dir_url = f"{DOWNLOAD_BASE}/{download_path}/"

            try:
                req = urllib.request.Request(dir_url)
                with urllib.request.urlopen(req, context=get_ssl_context()) as response:
                    html = response.read().decode()

                for swc_match in re.finditer(r'href="([^"]+\.swc)"', html):
                    filename = swc_match.group(1)
                    if not filename.endswith("_reg.swc"):
                        swc_urls.append(f"{DOWNLOAD_BASE}/{download_path}/{filename}")
            except Exception:
                pass

        if len(swc_urls) >= NUM_FILES:
            break

    print(f"Downloading {NUM_FILES} SWC files...")
    for i, url in enumerate(swc_urls[:NUM_FILES]):
        filename = url.split("/")[-1]
        output_path = OUTPUT_DIR / filename
        print(f"  [{i+1}/{NUM_FILES}] {filename}")
        download_file(url, output_path)

    print(f"\nDownloaded files to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
```

Run the script:

```bash
python download_examples.py
```

### Step 2: Align Neurons to the Right Hemisphere

Use the hemisphere detection and flipping functionality to standardize all neurons to the right hemisphere:

```python
#!/usr/bin/env python
"""align_hemispheres.py - Flip all neurons to the right hemisphere."""

from pathlib import Path
from napari_swc_viewer import (
    parse_swc,
    write_swc,
    detect_soma_hemisphere,
    flip_swc,
    Hemisphere,
)

INPUT_DIR = Path("example_data")
OUTPUT_DIR = Path("example_data_aligned")
OUTPUT_DIR.mkdir(exist_ok=True)

# Process each SWC file
for swc_path in sorted(INPUT_DIR.glob("*.swc")):
    print(f"Processing {swc_path.name}...")

    # Parse the SWC file
    swc_data = parse_swc(swc_path)
    print(f"  Nodes: {swc_data.n_nodes}")
    print(f"  Soma nodes: {swc_data.soma_mask.sum()}")

    # Detect which hemisphere the soma is in
    # Uses BrainGlobe atlas for midline calculation
    hemisphere = detect_soma_hemisphere(
        swc_data,
        atlas_name="allen_mouse_10um",
        validate=False,  # Skip atlas validation for speed
    )
    print(f"  Soma hemisphere: {hemisphere.value}")

    # Flip to right hemisphere if needed
    if hemisphere == Hemisphere.LEFT:
        print("  Flipping to right hemisphere...")
        swc_data = flip_swc(
            swc_data,
            atlas_name="allen_mouse_10um",
        )
        output_name = swc_path.stem + "_right.swc"
    else:
        output_name = swc_path.name

    # Write aligned file
    output_path = OUTPUT_DIR / output_name
    write_swc(swc_data, output_path)
    print(f"  Saved to {output_path}")

print(f"\nAligned files saved to {OUTPUT_DIR}/")
```

Run the alignment script:

```bash
python align_hemispheres.py
```

**Example Output:**
```
Processing 1059281710_18462_6029-X10270-Y8859.swc...
  Nodes: 847
  Soma nodes: 1
  Soma hemisphere: left
  Flipping to right hemisphere...
  Saved to example_data_aligned/1059281710_18462_6029-X10270-Y8859_right.swc
Processing 1059286114_18462_5674-X21158-Y9621.swc...
  Nodes: 1203
  Soma nodes: 1
  Soma hemisphere: right
  Saved to example_data_aligned/1059286114_18462_5674-X21158-Y9621.swc
...
```

### Step 3: Convert SWC Files to Parquet with Region Annotations

For large datasets, convert SWC files to Parquet format with brain region annotations:

```python
#!/usr/bin/env python
"""convert_to_parquet.py - Convert SWC files to annotated Parquet."""

from pathlib import Path
from napari_swc_viewer import swc_files_to_parquet, get_parquet_summary

INPUT_DIR = Path("example_data_aligned")
OUTPUT_FILE = Path("neurons.parquet")

print("Converting SWC files to Parquet...")
print("(This will download Allen SDK data on first run)")

# Convert all SWC files to a single Parquet file
# Each node is annotated with its brain region
n_files = swc_files_to_parquet(
    input_path=INPUT_DIR,
    output_path=OUTPUT_FILE,
    resolution=25,  # Allen CCF resolution in microns
)

print(f"\nProcessed {n_files} SWC files")

# Print summary statistics
summary = get_parquet_summary(OUTPUT_FILE)
print(f"\nParquet file summary:")
print(f"  Total nodes: {summary['n_rows']:,}")
print(f"  Files: {summary['n_files']}")
print(f"  Subjects: {summary['n_subjects']}")
print(f"  Brain regions: {summary['n_regions']}")

print("\nTop 5 regions by node count:")
for acronym, count in list(summary['regions'].items())[:5]:
    print(f"  {acronym}: {count:,} nodes")
```

Run the conversion:

```bash
python convert_to_parquet.py
```

**Example Output:**
```
Converting SWC files to Parquet...
Processed 5 SWC files

Parquet file summary:
  Total nodes: 4,523
  Files: 5
  Subjects: 2
  Brain regions: 12

Top 5 regions by node count:
  VISp: 1,847 nodes
  VISl: 892 nodes
  VISpm: 456 nodes
  VISam: 328 nodes
  VISrl: 201 nodes
```

### Step 4: Query Neurons by Brain Region

Use the DuckDB-powered database interface to query neurons:

```python
#!/usr/bin/env python
"""query_neurons.py - Query neurons by brain region."""

from napari_swc_viewer import NeuronDatabase

# Open the Parquet database
db = NeuronDatabase("neurons.parquet")

# Get database statistics
stats = db.get_statistics()
print(f"Database contains:")
print(f"  {stats['n_nodes']:,} nodes")
print(f"  {stats['n_files']} neurons")
print(f"  {stats['n_regions']} brain regions")

# Get all unique regions
print("\nAll brain regions in dataset:")
regions = db.get_unique_regions()
for _, row in regions.iterrows():
    print(f"  {row['region_acronym']}: {row['node_count']:,} nodes")

# Query neurons by region
print("\nNeurons with nodes in primary visual area (VISp):")
visp_neurons = db.get_neurons_by_region(["VISp"])
for _, row in visp_neurons.iterrows():
    print(f"  {row['file_id']} (subject: {row['subject']})")

# Query neurons in multiple regions
print("\nNeurons in any visual area (VIS*):")
visual_regions = ["VISp", "VISl", "VISpm", "VISam", "VISrl", "VISal"]
visual_neurons = db.get_neurons_by_region(visual_regions)
print(f"  Found {len(visual_neurons)} neurons")

# Get soma locations
print("\nSoma locations:")
somas = db.get_soma_locations()
for _, row in somas.iterrows():
    print(f"  {row['neuron_id']}: ({row['x']:.1f}, {row['y']:.1f}, {row['z']:.1f}) - {row['region_acronym']}")

# Custom SQL query
print("\nAxon node count per neuron:")
result = db.query("""
    SELECT file_id, COUNT(*) as axon_nodes
    FROM neurons
    WHERE type = 2  -- axon nodes
    GROUP BY file_id
    ORDER BY axon_nodes DESC
""")
for _, row in result.iterrows():
    print(f"  {row['file_id']}: {row['axon_nodes']} axon nodes")

db.close()
```

Run the query script:

```bash
python query_neurons.py
```

### Step 5: Visualize in napari

#### Option A: Quick Visualization with SWC Reader

Simply drag and drop `.swc` files into napari, or use the command line:

```bash
pixi run napari example_data_aligned/*.swc
```

#### Option B: Interactive Visualization with Python

```python
#!/usr/bin/env python
"""visualize_neurons.py - Interactive napari visualization."""

import napari
import numpy as np
from pathlib import Path
from napari_swc_viewer import parse_swc, NeuronDatabase

# Create napari viewer
viewer = napari.Viewer()

# ============================================
# Method 1: Load individual SWC files
# ============================================

swc_dir = Path("example_data_aligned")
swc_files = sorted(swc_dir.glob("*.swc"))

# Colors for different neurons
colors = ["cyan", "magenta", "yellow", "lime", "orange"]

# Load each neuron as a shapes layer (lines)
for i, swc_path in enumerate(swc_files[:5]):
    swc_data = parse_swc(swc_path)

    # Build parent-child line segments
    id_to_idx = {nid: idx for idx, nid in enumerate(swc_data.ids)}
    lines = []
    for idx, parent_id in enumerate(swc_data.parents):
        if parent_id in id_to_idx:
            parent_idx = id_to_idx[parent_id]
            lines.append([swc_data.coords[parent_idx], swc_data.coords[idx]])

    # Add as shapes layer
    layer = viewer.add_shapes(
        lines,
        shape_type="line",
        edge_width=2,
        edge_color=colors[i % len(colors)],
        name=swc_path.stem,
        visible=True,  # Toggle visibility here
    )

    # Add soma as point
    soma_coords = swc_data.soma_coords
    if len(soma_coords) > 0:
        viewer.add_points(
            soma_coords,
            size=15,
            face_color="red",
            name=f"{swc_path.stem}_soma",
        )

print("Loaded neurons. Toggle visibility in the layer list.")

# ============================================
# Method 2: Load from Parquet database
# ============================================

# Uncomment to use Parquet database instead:
#
# db = NeuronDatabase("neurons.parquet")
#
# # Get neurons in a specific region
# neurons = db.get_neurons_by_region(["VISp"])
#
# for _, row in neurons.iterrows():
#     coords, edges = db.get_neuron_lines(row["file_id"])
#     lines = [[coords[e[0]], coords[e[1]]] for e in edges]
#
#     viewer.add_shapes(
#         lines,
#         shape_type="line",
#         edge_width=2,
#         edge_color="cyan",
#         name=row["file_id"],
#     )
#
# db.close()

napari.run()
```

### Step 6: Use the Neuron Viewer Widget

For the full interactive experience with region filtering:

1. Launch napari:
   ```bash
   pixi run napari
   ```

2. Open the Neuron Viewer widget:
   - Go to **Plugins** → **Neuron Viewer**

3. Load data:
   - In the **Data** tab, click **Load...** and select your `neurons.parquet` file
   - Click **Load Atlas** to load the Allen CCF structure tree

4. Filter by brain region:
   - Go to the **Regions** tab
   - Search for "VIS" to find visual areas
   - Check the regions you want to query (e.g., VISp, VISl)
   - Click **Find Neurons in Selected Regions**

5. Visualize neurons:
   - Select neurons from the list (Ctrl+click for multiple)
   - Click **Render Selected** to display them

6. Customize visualization:
   - In the **Visualization** tab:
     - Choose **Lines**, **Points**, or **Both** render modes
     - Adjust point size and line width
     - Change opacity

7. Add reference layers:
   - In the **Reference** tab:
     - Toggle **Show template** for Allen CCF reference image
     - Toggle **Show brain outline** for 3D brain surface
     - Toggle **Show selected region meshes** to highlight regions

### Step 7: Toggle Neurons for Comparison

Toggle neuron visibility programmatically:

```python
#!/usr/bin/env python
"""toggle_neurons.py - Toggle neuron visibility for comparison."""

import napari
from pathlib import Path
from napari_swc_viewer import parse_swc

viewer = napari.Viewer()

# Load neurons
swc_dir = Path("example_data_aligned")
layers = {}

for swc_path in sorted(swc_dir.glob("*.swc"))[:5]:
    swc_data = parse_swc(swc_path)

    id_to_idx = {nid: idx for idx, nid in enumerate(swc_data.ids)}
    lines = []
    for idx, parent_id in enumerate(swc_data.parents):
        if parent_id in id_to_idx:
            parent_idx = id_to_idx[parent_id]
            lines.append([swc_data.coords[parent_idx], swc_data.coords[idx]])

    layer = viewer.add_shapes(
        lines,
        shape_type="line",
        edge_width=2,
        edge_color="cyan",
        name=swc_path.stem,
    )
    layers[swc_path.stem] = layer


def show_only(names: list[str]):
    """Show only the specified neurons, hide others."""
    for name, layer in layers.items():
        layer.visible = name in names


def show_all():
    """Show all neurons."""
    for layer in layers.values():
        layer.visible = True


def hide_all():
    """Hide all neurons."""
    for layer in layers.values():
        layer.visible = False


def toggle(name: str):
    """Toggle a specific neuron's visibility."""
    if name in layers:
        layers[name].visible = not layers[name].visible


# Example usage:
# show_only(["neuron1", "neuron2"])  # Show only specific neurons
# hide_all()  # Hide everything
# show_all()  # Show everything
# toggle("neuron1")  # Toggle one neuron

print("Functions available:")
print("  show_only(['name1', 'name2'])  - Show only specified neurons")
print("  show_all()                      - Show all neurons")
print("  hide_all()                      - Hide all neurons")
print("  toggle('name')                  - Toggle one neuron")
print()
print("Available neurons:", list(layers.keys()))

napari.run()
```

---

## API Reference

### SWC Parsing

```python
from napari_swc_viewer import parse_swc, write_swc, SWCData, NodeType

# Parse an SWC file
swc_data = parse_swc("neuron.swc")

# Access data
swc_data.ids        # Node IDs
swc_data.types      # Node types (1=soma, 2=axon, 3=basal, 4=apical)
swc_data.coords     # (N, 3) array of [x, y, z] coordinates
swc_data.radii      # Node radii
swc_data.parents    # Parent node IDs (-1 for root)

# Convenience properties
swc_data.n_nodes       # Number of nodes
swc_data.soma_mask     # Boolean mask for soma nodes
swc_data.soma_coords   # Coordinates of soma nodes
swc_data.root_mask     # Boolean mask for root nodes

# Write to file
write_swc(swc_data, "output.swc")
```

### Hemisphere Operations

```python
from napari_swc_viewer import (
    detect_hemisphere,
    detect_soma_hemisphere,
    flip_coordinates,
    flip_swc,
    flip_swc_batch,
    get_atlas_midline,
    Hemisphere,
)

# Detect hemisphere from coordinates
hemisphere = detect_hemisphere(
    coords,                          # (N, 3) array
    atlas_name="allen_mouse_10um",   # BrainGlobe atlas
)

# Detect from SWC soma
hemisphere = detect_soma_hemisphere(swc_data)

# Flip coordinates across midline
flipped_coords = flip_coordinates(coords, atlas_name="allen_mouse_10um")

# Flip entire SWC morphology
flipped_swc = flip_swc(swc_data)

# Batch flip (loads atlas once)
flipped_list = flip_swc_batch([swc1, swc2, swc3])

# Get atlas midline value
midline = get_atlas_midline(atlas, coord_axis=2)  # 5695.0 for allen_mouse_10um
```

### Parquet Conversion

```python
from napari_swc_viewer import swc_files_to_parquet, get_parquet_summary, NEURON_SCHEMA

# Convert SWC files to Parquet with region annotations
n_files = swc_files_to_parquet(
    input_path="swc_directory/",
    output_path="neurons.parquet",
    resolution=25,           # Allen CCF resolution
    n_workers=4,             # Parallel workers
)

# Get summary statistics
summary = get_parquet_summary("neurons.parquet")
print(summary["n_files"], summary["n_regions"])
```

### Database Queries

```python
from napari_swc_viewer import NeuronDatabase

db = NeuronDatabase("neurons.parquet")

# Query by region
neurons = db.get_neurons_by_region(["VISp", "VISl"])

# Get soma locations
somas = db.get_soma_locations()

# Get line segments for rendering
coords, edges = db.get_neuron_lines("neuron.swc")

# Get all unique regions
regions = db.get_unique_regions()

# Custom SQL
result = db.query("SELECT * FROM neurons WHERE type = 1")

db.close()
```

---

## Hemisphere Detection and Coordinate Flipping

### Midline Calculation

The midline (midsagittal plane) is calculated based on the Allen Mouse Brain Common Coordinate Framework v3 (CCFv3) convention:

- **Coordinate system**: The CCFv3 defines a reference image with origin at (0, 0, 0), spacing of 10 µm/voxel, and size (1320, 800, 1140) voxels
- **Axes**: Rostral-to-caudal, dorsal-to-ventral, and left-to-right
- **Voxel convention**: Following the ITK convention, voxel positions are defined at voxel **centers**, so position (0, 0, 0) is at the center of the first voxel

The midline coordinate is calculated as the midpoint between the first and last voxel centers:

```
midline = ((shape - 1) * resolution) / 2
```

For the `allen_mouse_10um` atlas (shape=1140 along the left-right axis):
- First voxel center: 0.0 µm
- Last voxel center: (1140 - 1) × 10 = 11390.0 µm
- **Midline: 5695.0 µm**

### Coordinate Flipping

Coordinates are flipped by reflecting across the midline plane:

```
flipped_coordinate = 2 × midline - original_coordinate
```

This operation is vectorized using NumPy for efficient processing of large SWC files (10,000+ nodes).

---

## Optional Dependencies

### Allen SDK

For full region mapping functionality (required for `swc_files_to_parquet`):

```bash
pip install allensdk
```

Note: Allen SDK has version constraints on scipy. If you encounter conflicts, consider using a separate environment.

---

## License

BSD-3-Clause
