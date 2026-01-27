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