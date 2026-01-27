#!/usr/bin/env python
"""convert_to_parquet.py - Convert SWC files to annotated Parquet.

NOTE: SWC files must have coordinates registered to the Allen CCF coordinate
system (in microns) for brain region annotation to work correctly.
The test_data/ directory contains Allen CCF-registered files.
"""

from pathlib import Path
from napari_swc_viewer import swc_files_to_parquet, get_parquet_summary

# Use test_data which contains Allen CCF-registered coordinates
# SWC files in example_data_aligned may be in original image space
INPUT_DIR = Path("test_data")
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