"""Node count heatmap volume construction.

Ported from swc-mapper/create_node_counts_heatmap.py. Instead of writing
partitioned parquet and OME-TIFF, builds an in-memory 3D numpy array
suitable for display as a napari Image layer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..swc import normalize_node_types

if TYPE_CHECKING:
    import duckdb
    from brainglobe_atlasapi import BrainGlobeAtlas
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def build_node_counts_volume(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: str,
    atlas: BrainGlobeAtlas,
    region_ids: list[int] | None = None,
    file_ids: list[str] | None = None,
    node_types: list[int] | tuple[int, ...] | None = None,
    soma_radius_um: float | None = None,
    depth_bin_factor: int = 1,
    depth_axis: int = 0,
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
    region_ids : list[int], optional
        Filter to nodes whose annotation region ID is in this list.
    file_ids : list[str], optional
        Filter to specific neurons.
    node_types : list[int], optional
        Filter to SWC node type IDs. ``None`` leaves node types unfiltered.
    soma_radius_um : float, optional
        If positive, keep only nodes within this distance of their file's
        soma centroid. Files without soma nodes are excluded in this mode.
    depth_bin_factor : int
        Merge this many planes along the depth axis into one voxel.
        A factor of 1 (default) gives standard 1:1 voxels.
    depth_axis : int
        Which atlas axis (0, 1, or 2) is the depth/slicing axis.
        Default 0 corresponds to the first (Z) axis.

    Returns
    -------
    NDArray[np.float32]
        3D float32 volume. The depth axis dimension is reduced by
        ``depth_bin_factor``; other axes match the atlas shape.
    """
    parquet_path_escaped = str(parquet_path).replace("\\", "/")
    resolution = float(atlas.resolution[0])
    atlas_shape = list(atlas.annotation.shape)  # (Z, Y, X) in PIR order

    # Compute output shape: depth axis is binned, others unchanged
    out_shape = list(atlas_shape)
    out_shape[depth_axis] = -(-atlas_shape[depth_axis] // depth_bin_factor)  # ceiling division

    volume = np.zeros(out_shape, dtype=np.float32)

    # Resolution per axis for voxel-index computation
    axis_res = [resolution, resolution, resolution]
    axis_res[depth_axis] = resolution * depth_bin_factor

    # Build WHERE clauses
    where_clauses = []
    params: list[object] = []
    if region_ids:
        placeholders = ", ".join(["?"] * len(region_ids))
        where_clauses.append(f"n.region_id IN ({placeholders})")
        params.extend(int(region_id) for region_id in region_ids)
    if file_ids is not None:
        if len(file_ids) == 0:
            return volume
        placeholders = ", ".join(["?"] * len(file_ids))
        where_clauses.append(f"n.file_id IN ({placeholders})")
        params.extend(file_ids)
    normalized_node_types = normalize_node_types(node_types)
    if normalized_node_types is not None:
        if not normalized_node_types:
            return volume
        placeholders = ", ".join(["?"] * len(normalized_node_types))
        where_clauses.append(f"n.type IN ({placeholders})")
        params.extend(int(node_type) for node_type in normalized_node_types)

    radius = None
    if soma_radius_um is not None:
        radius = float(soma_radius_um)
        if radius <= 0.0:
            radius = None
    if radius is not None:
        where_clauses.append(
            """
            SQRT(
                POW(n.x - s.soma_x, 2)
                + POW(n.y - s.soma_y, 2)
                + POW(n.z - s.soma_z, 2)
            ) <= ?
            """
        )
        params.append(radius)

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    source_sql = f"read_parquet('{parquet_path_escaped}') AS n"
    prefix_sql = ""
    if radius is not None:
        prefix_sql = f"""
            WITH soma_centroids AS (
                SELECT
                    file_id,
                    AVG(x) AS soma_x,
                    AVG(y) AS soma_y,
                    AVG(z) AS soma_z
                FROM read_parquet('{parquet_path_escaped}')
                WHERE type = 1
                GROUP BY file_id
            )
        """
        source_sql = (
            f"read_parquet('{parquet_path_escaped}') AS n "
            "INNER JOIN soma_centroids AS s ON n.file_id = s.file_id"
        )

    # Query: convert coordinates to voxel indices and count nodes per voxel.
    # Axis mapping: parquet (x,y,z) in microns → atlas PIR:
    #   z -> xi (atlas dim 2), y -> yi (atlas dim 1), x -> zi (atlas dim 0)
    query = f"""
        {prefix_sql}
        SELECT
            CAST(FLOOR(n.x / {axis_res[0]}) AS INTEGER) AS zi,
            CAST(FLOOR(n.y / {axis_res[1]}) AS INTEGER) AS yi,
            CAST(FLOOR(n.z / {axis_res[2]}) AS INTEGER) AS xi,
            COUNT(*) AS node_count
        FROM {source_sql}
        {where_sql}
        GROUP BY zi, yi, xi
        HAVING zi >= 0 AND zi < {out_shape[0]}
           AND yi >= 0 AND yi < {out_shape[1]}
           AND xi >= 0 AND xi < {out_shape[2]}
    """

    logger.info(
        "Querying node counts per voxel "
        f"(depth_bin_factor={depth_bin_factor}, depth_axis={depth_axis})..."
    )
    if params:
        df = conn.execute(query, params).fetchdf()
    else:
        df = conn.execute(query).fetchdf()

    logger.info(f"Found {len(df)} non-empty voxels, total nodes: {df['node_count'].sum():,}")

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
