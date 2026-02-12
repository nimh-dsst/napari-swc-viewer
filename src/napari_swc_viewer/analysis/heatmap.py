"""Node count heatmap volume construction.

Ported from swc-mapper/create_node_counts_heatmap.py. Instead of writing
partitioned parquet and OME-TIFF, builds an in-memory 3D numpy array
suitable for display as a napari Image layer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import duckdb
    from brainglobe_atlasapi import BrainGlobeAtlas
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def build_node_counts_volume(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: str,
    atlas: BrainGlobeAtlas,
    region_acronym: str | None = None,
    file_ids: list[str] | None = None,
) -> NDArray[np.float32]:
    """Build a 3D node-count volume from parquet data.

    Queries DuckDB to count nodes per voxel, then scatters the sparse
    counts into a dense 3D array matching the atlas annotation shape.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Open DuckDB connection.
    parquet_path : str
        Path to parquet file containing neuron node data.
    atlas : BrainGlobeAtlas
        Atlas for volume shape and resolution.
    region_acronym : str, optional
        Filter to nodes in this region only.
    file_ids : list[str], optional
        Filter to specific neurons.

    Returns
    -------
    NDArray[np.float32]
        3D float32 volume matching atlas annotation shape. Each voxel
        contains the count of neuron nodes within it.
    """
    parquet_path_escaped = str(parquet_path).replace("\\", "/")
    resolution = float(atlas.resolution[0])
    shape = atlas.annotation.shape  # (Z, Y, X) in PIR order

    # Build WHERE clauses
    where_clauses = []
    params = []
    if region_acronym is not None:
        where_clauses.append("region_acronym = ?")
        params.append(region_acronym)
    if file_ids is not None and len(file_ids) > 0:
        placeholders = ", ".join(["?"] * len(file_ids))
        where_clauses.append(f"file_id IN ({placeholders})")
        params.extend(file_ids)

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    # Query: convert coordinates to voxel indices and count nodes per voxel.
    # Axis mapping: parquet (x,y,z) in microns → atlas PIR:
    #   z -> xi (atlas dim 2), y -> yi (atlas dim 1), x -> zi (atlas dim 0)
    query = f"""
        SELECT
            CAST(FLOOR(x / {resolution}) AS INTEGER) AS zi,
            CAST(FLOOR(y / {resolution}) AS INTEGER) AS yi,
            CAST(FLOOR(z / {resolution}) AS INTEGER) AS xi,
            COUNT(*) AS node_count
        FROM read_parquet('{parquet_path_escaped}')
        {where_sql}
        GROUP BY zi, yi, xi
        HAVING zi >= 0 AND zi < {shape[0]}
           AND yi >= 0 AND yi < {shape[1]}
           AND xi >= 0 AND xi < {shape[2]}
    """

    logger.info("Querying node counts per voxel...")
    if params:
        df = conn.execute(query, params).fetchdf()
    else:
        df = conn.execute(query).fetchdf()

    logger.info(f"Found {len(df)} non-empty voxels, total nodes: {df['node_count'].sum():,}")

    # Build dense volume
    volume = np.zeros(shape, dtype=np.float32)
    if len(df) > 0:
        zi = df["zi"].values.astype(np.intp)
        yi = df["yi"].values.astype(np.intp)
        xi = df["xi"].values.astype(np.intp)
        counts = df["node_count"].values.astype(np.float32)
        volume[zi, yi, xi] = counts

    logger.info(
        f"Heatmap volume: shape {volume.shape}, "
        f"max count {volume.max():.0f}, "
        f"non-zero voxels {(volume > 0).sum():,}"
    )
    return volume
