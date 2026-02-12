#!/usr/bin/env python
"""Benchmark slice projection performance with real neuron data.

Pipeline:
1. Download 77 SWC files from the Brain Image Library
2. Flip all neurons so somas are in the right hemisphere
3. Convert flipped SWCs to a Parquet file (with Allen CCF annotations)
4. Load the Parquet file and feed data to NeuronSliceProjector
5. Simulate scrubbing through 2D slices and measure timing
"""

from __future__ import annotations

import json
import logging
import random
import re
import ssl
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_FILES = 77
ATLAS_NAME = "allen_mouse_10um"
COORD_AXIS = 2  # Z axis for left-right hemisphere
TOLERANCE = 100.0  # microns, same as real usage
NUM_SLICES = 200  # number of slice positions to benchmark

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = REPO_ROOT / "benchmark_data"
SWC_DIR = BENCHMARK_DIR / "swc_raw"
FLIPPED_DIR = BENCHMARK_DIR / "swc_flipped"
PARQUET_PATH = BENCHMARK_DIR / "neurons.parquet"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 1: Download SWC files (adapted from download_test_data.py)
# ---------------------------------------------------------------------------
API_BASE = "https://api.brainimagelibrary.org"
DOWNLOAD_BASE = "https://download.brainimagelibrary.org"
MORPHOLOGY_SUBMISSION_UUID = "0fcde5fdd6f7ccb2"


def get_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def api_get(endpoint: str) -> dict:
    url = f"{API_BASE}/{endpoint}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=get_ssl_context()) as response:
        return json.loads(response.read().decode())


def get_dataset_bildids() -> list[str]:
    result = api_get(f"query/submission?submission_uuid={MORPHOLOGY_SUBMISSION_UUID}")
    if result.get("success") != "true":
        raise RuntimeError(f"API query failed: {result.get('message')}")
    return result.get("bildids", [])


def get_swc_files_for_bildid(bildid: str) -> list[str]:
    result = api_get(f"retrieve?bildid={bildid}")
    if result.get("success") != "true":
        return []
    swc_urls = []
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
        except Exception as e:
            logger.warning(f"  Could not list {dir_url}: {e}")
    return swc_urls


def download_file(url: str, output_path: Path) -> bool:
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=get_ssl_context()) as response:
            output_path.write_bytes(response.read())
        return True
    except Exception as e:
        logger.warning(f"  Error downloading {url}: {e}")
        return False


def download_swc_files() -> list[Path]:
    """Download NUM_FILES SWC files from BIL. Returns list of downloaded paths."""
    existing = sorted(SWC_DIR.glob("*.swc"))
    if len(existing) >= NUM_FILES:
        logger.info(f"[Download] Already have {len(existing)} SWC files, skipping download")
        return existing[:NUM_FILES]

    SWC_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"[Download] Downloading {NUM_FILES} SWC files from BIL...")

    bildids = get_dataset_bildids()
    logger.info(f"[Download] Found {len(bildids)} dataset entries")

    all_urls = []
    sample_bildids = random.sample(bildids, min(100, len(bildids)))

    for i, bildid in enumerate(sample_bildids):
        urls = get_swc_files_for_bildid(bildid)
        all_urls.extend(urls)
        if (i + 1) % 10 == 0:
            logger.info(f"  Checked {i + 1}/{len(sample_bildids)} bildids, {len(all_urls)} files found")
        if len(all_urls) >= NUM_FILES * 3:
            break

    if not all_urls:
        raise RuntimeError("No SWC files found from BIL API")

    to_download = random.sample(all_urls, min(NUM_FILES, len(all_urls)))
    logger.info(f"[Download] Downloading {len(to_download)} files...")

    downloaded = []
    for i, url in enumerate(to_download):
        filename = url.split("/")[-1]
        output_path = SWC_DIR / filename
        if output_path.exists():
            downloaded.append(output_path)
            continue
        if download_file(url, output_path):
            downloaded.append(output_path)
        if (i + 1) % 10 == 0:
            logger.info(f"  Downloaded {i + 1}/{len(to_download)}")

    logger.info(f"[Download] Downloaded {len(downloaded)} SWC files")
    return downloaded


# ---------------------------------------------------------------------------
# Step 2: Flip all neurons to right hemisphere (adapted from flip_swc.py)
# ---------------------------------------------------------------------------
def flip_all_to_right_hemisphere(swc_paths: list[Path]) -> list[Path]:
    """Flip all neurons so somas are in the right hemisphere. Returns flipped paths."""
    FLIPPED_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already done
    existing = sorted(FLIPPED_DIR.glob("*.swc"))
    if len(existing) >= len(swc_paths):
        logger.info(f"[Flip] Already have {len(existing)} flipped files, skipping")
        return existing[:len(swc_paths)]

    from brainglobe_atlasapi import BrainGlobeAtlas

    from napari_swc_viewer import (
        Hemisphere,
        detect_soma_hemisphere,
        flip_swc,
        get_atlas_midline,
        parse_swc,
        write_swc,
    )

    logger.info(f"[Flip] Loading atlas: {ATLAS_NAME}")
    atlas = BrainGlobeAtlas(ATLAS_NAME)
    midline = get_atlas_midline(atlas, COORD_AXIS)
    logger.info(f"[Flip] Midline: {midline:.2f} um")

    flipped_paths = []
    skipped = 0
    flipped_count = 0
    errors = 0

    for i, swc_path in enumerate(swc_paths):
        output_path = FLIPPED_DIR / swc_path.name
        if output_path.exists():
            flipped_paths.append(output_path)
            continue

        try:
            swc_data = parse_swc(swc_path)

            try:
                hemisphere = detect_soma_hemisphere(
                    swc_data, atlas=atlas, midline=midline,
                    coord_axis=COORD_AXIS, validate=False,
                )
            except ValueError:
                # No soma nodes — just copy as-is
                write_swc(swc_data, output_path)
                flipped_paths.append(output_path)
                skipped += 1
                continue

            if hemisphere == Hemisphere.LEFT:
                swc_data = flip_swc(
                    swc_data, atlas=atlas, midline=midline,
                    coord_axis=COORD_AXIS,
                )
                flipped_count += 1
            else:
                skipped += 1

            write_swc(swc_data, output_path)
            flipped_paths.append(output_path)

        except Exception as e:
            logger.warning(f"  Error processing {swc_path.name}: {e}")
            errors += 1

        if (i + 1) % 20 == 0:
            logger.info(f"  Processed {i + 1}/{len(swc_paths)} (flipped={flipped_count}, kept={skipped}, errors={errors})")

    logger.info(
        f"[Flip] Done: {flipped_count} flipped, {skipped} already right, {errors} errors"
    )
    return flipped_paths


# ---------------------------------------------------------------------------
# Step 3: Convert SWCs to Parquet (using plugin's parquet module)
# ---------------------------------------------------------------------------
def convert_to_parquet(swc_dir: Path) -> Path:
    """Convert SWC files to annotated Parquet. Returns parquet path."""
    if PARQUET_PATH.exists():
        logger.info(f"[Parquet] Already exists: {PARQUET_PATH}")
        return PARQUET_PATH

    from napari_swc_viewer.parquet import swc_files_to_parquet

    logger.info(f"[Parquet] Converting SWC files in {swc_dir} to {PARQUET_PATH}...")
    t0 = time.perf_counter()
    n_files = swc_files_to_parquet(
        input_path=swc_dir,
        output_path=PARQUET_PATH,
        resolution=25,
        n_workers=1,
    )
    elapsed = time.perf_counter() - t0
    logger.info(f"[Parquet] Converted {n_files} files in {elapsed:.1f}s")
    return PARQUET_PATH


# ---------------------------------------------------------------------------
# Step 4 & 5: Load data and benchmark slice projection
# ---------------------------------------------------------------------------
def benchmark_projection(parquet_path: Path) -> None:
    """Load neuron data from parquet and benchmark slice projection."""
    from napari_swc_viewer.db import NeuronDatabase
    from napari_swc_viewer.widgets.slice_projection import NeuronSliceProjector
    from unittest.mock import MagicMock

    logger.info("[Benchmark] Loading neuron data from parquet...")
    db = NeuronDatabase(parquet_path)
    stats = db.get_statistics()
    logger.info(f"  Nodes: {stats['n_nodes']:,}, Files: {stats['n_files']}, "
                f"Subjects: {stats['n_subjects']}, Regions: {stats['n_regions']}")

    # Get all file IDs
    file_ids_df = db.query("SELECT DISTINCT file_id FROM neurons ORDER BY file_id")
    file_ids = file_ids_df["file_id"].tolist()
    logger.info(f"  Loading line data for {len(file_ids)} neurons...")

    t0 = time.perf_counter()
    lines_batch = db.get_neuron_lines_batch(file_ids)
    load_time = time.perf_counter() - t0
    logger.info(f"  Loaded in {load_time:.2f}s")

    total_segments = sum(edges.shape[0] for _, edges in lines_batch.values())
    logger.info(f"  Total segments: {total_segments:,}")

    # Create a projector (mocked viewer, we only benchmark the compute)
    projector = NeuronSliceProjector.__new__(NeuronSliceProjector)
    projector._viewer = MagicMock()
    projector._enabled = False
    projector._tolerance = TOLERANCE
    projector._edge_width = 4
    projector._projection_layer = None
    projector._scale = None
    projector._connected = False
    projector._update_timer = MagicMock()
    projector._source_data = {}
    projector._all_p1 = None
    projector._all_p2 = None
    projector._all_colors = None
    projector._axis_index = {}
    projector._last_result_key = None
    projector._last_result = None

    # Feed data into projector
    logger.info("[Benchmark] Building projector data...")
    batch_data = {}
    for file_id, (coords, edges) in lines_batch.items():
        if edges.shape[0] > 0:
            batch_data[file_id] = (coords, edges, (1.0, 1.0, 0.0, 1.0))

    t0 = time.perf_counter()
    for file_id, (coords, edges, color) in batch_data.items():
        projector._source_data[file_id] = (coords.copy(), edges.copy(), color)
    projector._rebuild_arrays()
    build_time = time.perf_counter() - t0

    n_segments = projector._all_p1.shape[0] if projector._all_p1 is not None else 0
    logger.info(f"  Projector built: {n_segments:,} segments in {build_time:.3f}s")
    logger.info(f"  Axis index entries: {len(projector._axis_index)}")

    if n_segments == 0:
        logger.error("  No segments loaded — nothing to benchmark")
        db.close()
        return

    # Determine the Z range (axis 0 in ZYX) of all segments
    z_all = np.concatenate([projector._all_p1[:, 0], projector._all_p2[:, 0]])
    z_min, z_max = float(z_all.min()), float(z_all.max())
    logger.info(f"  Z range: {z_min:.1f} to {z_max:.1f} um (span={z_max - z_min:.1f})")

    slice_positions = np.linspace(z_min, z_max, NUM_SLICES)
    slice_axis = 0  # Z axis in ZYX

    # --- Benchmark: index-based (current implementation) ---
    logger.info(f"\n[Benchmark] Index-based projection ({NUM_SLICES} slices, tolerance={TOLERANCE})...")
    projector._invalidate_cache()
    hits = []
    t0 = time.perf_counter()
    for pos in slice_positions:
        projector._invalidate_cache()  # Force fresh computation each time
        lines, colors = projector._compute_slice_projection(float(pos), slice_axis)
        hits.append(0 if lines is None else len(lines))
    indexed_time = time.perf_counter() - t0
    indexed_per_slice = indexed_time / NUM_SLICES * 1000  # ms

    logger.info(f"  Total: {indexed_time:.3f}s  ({indexed_per_slice:.2f} ms/slice)")
    logger.info(f"  Segments per slice: min={min(hits)}, max={max(hits)}, "
                f"mean={np.mean(hits):.0f}, median={np.median(hits):.0f}")

    # --- Benchmark: brute-force (old implementation) ---
    logger.info(f"\n[Benchmark] Brute-force projection ({NUM_SLICES} slices)...")
    t0 = time.perf_counter()
    for pos in slice_positions:
        _brute_force_projection(projector, float(pos), slice_axis, TOLERANCE)
    brute_time = time.perf_counter() - t0
    brute_per_slice = brute_time / NUM_SLICES * 1000  # ms

    logger.info(f"  Total: {brute_time:.3f}s  ({brute_per_slice:.2f} ms/slice)")

    # --- Benchmark: cache hits ---
    logger.info(f"\n[Benchmark] Cache hit performance ({NUM_SLICES} repeated queries)...")
    # Prime the cache with the first position
    projector._invalidate_cache()
    projector._compute_slice_projection(float(slice_positions[0]), slice_axis)
    t0 = time.perf_counter()
    for _ in range(NUM_SLICES):
        projector._compute_slice_projection(float(slice_positions[0]), slice_axis)
    cache_time = time.perf_counter() - t0
    cache_per_slice = cache_time / NUM_SLICES * 1000  # ms
    logger.info(f"  Total: {cache_time:.6f}s  ({cache_per_slice:.4f} ms/slice)")

    # --- Benchmark: color-only rebuild ---
    logger.info("\n[Benchmark] Color-only rebuild vs full rebuild...")
    t0 = time.perf_counter()
    for _ in range(10):
        projector._rebuild_arrays()
    full_rebuild_time = (time.perf_counter() - t0) / 10

    t0 = time.perf_counter()
    for _ in range(10):
        projector._rebuild_colors_only()
    color_rebuild_time = (time.perf_counter() - t0) / 10

    logger.info(f"  Full rebuild: {full_rebuild_time * 1000:.2f} ms")
    logger.info(f"  Color-only rebuild: {color_rebuild_time * 1000:.2f} ms")
    logger.info(f"  Speedup: {full_rebuild_time / color_rebuild_time:.1f}x")

    # --- Summary ---
    speedup = brute_time / indexed_time if indexed_time > 0 else float("inf")
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Neurons loaded:       {len(file_ids)}")
    logger.info(f"  Total segments:       {n_segments:,}")
    logger.info(f"  Tolerance:            {TOLERANCE} um")
    logger.info(f"  Slices tested:        {NUM_SLICES}")
    logger.info(f"  Brute-force:          {brute_per_slice:.2f} ms/slice")
    logger.info(f"  Index-based:          {indexed_per_slice:.2f} ms/slice")
    logger.info(f"  Projection speedup:   {speedup:.1f}x")
    logger.info(f"  Cache hit:            {cache_per_slice:.4f} ms/slice")
    logger.info(f"  Full rebuild:         {full_rebuild_time * 1000:.2f} ms")
    logger.info(f"  Color-only rebuild:   {color_rebuild_time * 1000:.2f} ms")
    logger.info("=" * 60)

    db.close()


def _brute_force_projection(projector, slice_position, slice_axis, tolerance):
    """Old brute-force implementation for comparison."""
    slab_min = slice_position - tolerance
    slab_max = slice_position + tolerance
    v1 = projector._all_p1[:, slice_axis]
    v2 = projector._all_p2[:, slice_axis]
    mask = (np.maximum(v1, v2) >= slab_min) & (np.minimum(v1, v2) <= slab_max)

    if not mask.any():
        return None, None

    p1 = projector._all_p1[mask].copy()
    p2 = projector._all_p2[mask].copy()
    p1[:, slice_axis] = slice_position
    p2[:, slice_axis] = slice_position
    lines = np.stack([p1, p2], axis=1)
    colors = projector._all_colors[mask]
    return lines, colors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("Slice Projection Performance Benchmark")
    logger.info("=" * 60)
    logger.info(f"  Target neurons: {NUM_FILES}")
    logger.info(f"  Tolerance: {TOLERANCE} um")
    logger.info(f"  Slice positions: {NUM_SLICES}")
    logger.info(f"  Output dir: {BENCHMARK_DIR}")
    logger.info("")

    # Step 1: Download
    swc_paths = download_swc_files()
    if not swc_paths:
        logger.error("No SWC files available")
        return 1

    # Step 2: Flip hemispheres
    flipped_paths = flip_all_to_right_hemisphere(swc_paths)
    if not flipped_paths:
        logger.error("No flipped SWC files produced")
        return 1

    # Step 3: Convert to Parquet
    parquet_path = convert_to_parquet(FLIPPED_DIR)

    # Step 4 & 5: Benchmark
    benchmark_projection(parquet_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
