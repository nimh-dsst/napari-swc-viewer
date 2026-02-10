"""Pearson cross-correlation matrix computation using DuckDB.

Ported from swc-mapper/pearson_cross_correlation_matrix_egpe_counts.py and
swc-mapper/corr_full_to_matrix.py.

Computes pairwise Pearson correlations between neurons based on their
node counts per voxel within a target brain region.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa

if TYPE_CHECKING:
    import duckdb
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_pearson_correlation_matrix(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: str,
    voxel_id_map: NDArray[np.int32],
    resolution: float,
    progress_callback: callable | None = None,
) -> pd.DataFrame:
    """Compute pairwise Pearson correlation of neuron node counts per voxel.

    For each neuron, counts the number of nodes in each voxel of the target
    region (defined by voxel_id_map), then computes Pearson r between all
    neuron pairs.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        An open DuckDB connection.
    parquet_path : str
        Path to the neuron parquet file.
    voxel_id_map : NDArray[np.int32]
        3D voxel ID map from get_expanded_region_voxel_ids(). Voxels inside
        the target region have sequential IDs >= 0, outside = -1.
    resolution : float
        Voxel resolution in microns (e.g., 25.0).
    progress_callback : callable, optional
        Called with (step_name: str, step_number: int, total_steps: int).

    Returns
    -------
    pd.DataFrame
        Long-form correlation table with columns: swc_id_1, swc_id_2, r.
        Includes both triangles plus the diagonal (r=1).
    """
    parquet_path_escaped = str(parquet_path).replace("\\", "/")
    Z, Y, X = voxel_id_map.shape

    def _progress(name: str, step: int, total: int = 7) -> None:
        logger.info(f"Correlation step {step}/{total}: {name}")
        if progress_callback is not None:
            progress_callback(name, step, total)

    # Register the voxel ID map as a PyArrow lookup table
    _progress("Registering voxel ID lookup table", 1)
    arrow_lut = pa.table(
        {
            "idx": np.arange(voxel_id_map.size, dtype=np.int64),
            "val": voxel_id_map.ravel(),
        }
    )
    conn.register("lut", arrow_lut)

    # Create a view that maps node coordinates to voxel IDs.
    # Axis convention from swc-mapper: parquet has (x, y, z) in microns,
    # atlas is in PIR order (dim0, dim1, dim2).
    # z -> xi (atlas dim 2), y -> yi (atlas dim 1), x -> zi (atlas dim 0)
    # linear index = zi * (Y * X) + yi * X + xi
    _progress("Mapping nodes to voxel IDs", 2)
    conn.execute(f"""
        CREATE OR REPLACE TEMP VIEW region_nodes AS
        WITH base AS (
            SELECT
                file_id AS swc_id,
                CAST(FLOOR(z / {float(resolution)}) AS BIGINT) AS xi,
                CAST(FLOOR(y / {float(resolution)}) AS BIGINT) AS yi,
                CAST(FLOOR(x / {float(resolution)}) AS BIGINT) AS zi
            FROM read_parquet('{parquet_path_escaped}')
        ),
        mapped AS (
            SELECT
                b.swc_id,
                l.val AS voxel_id
            FROM base b
            JOIN lut l
                ON l.idx = (b.zi * ({Y} * {X})::BIGINT + b.yi * {X}::BIGINT + b.xi)
            WHERE b.xi >= 0 AND b.xi < {X}
              AND b.yi >= 0 AND b.yi < {Y}
              AND b.zi >= 0 AND b.zi < {Z}
        )
        SELECT swc_id, CAST(voxel_id AS INTEGER) AS voxel_id
        FROM mapped
        WHERE voxel_id >= 0
    """)

    # Count nodes per neuron per voxel
    _progress("Counting nodes per voxel", 3)
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE counts_by_voxel AS
        SELECT
            swc_id,
            voxel_id,
            COUNT(*)::BIGINT AS c
        FROM region_nodes
        GROUP BY swc_id, voxel_id
    """)

    # Compute voxel universe size
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE voxel_universe AS
        SELECT COUNT(DISTINCT voxel_id)::BIGINT AS V
        FROM counts_by_voxel
    """)

    # Per-neuron sums for Pearson formula
    _progress("Computing per-neuron statistics", 4)
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE per_neuron AS
        SELECT
            swc_id,
            SUM(c)::DOUBLE AS Sx,
            SUM(c * c)::DOUBLE AS Sxx
        FROM counts_by_voxel
        GROUP BY swc_id
    """)

    # Assign numeric IDs for ordering (to compute only upper triangle)
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE swc_numeric AS
        SELECT
            swc_id,
            ROW_NUMBER() OVER (ORDER BY swc_id) AS swc_num
        FROM (SELECT DISTINCT swc_id FROM counts_by_voxel)
    """)

    # Pairwise cross-products via voxel join (upper triangle only)
    _progress("Computing pairwise cross-products", 5)
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE pairwise_xy AS
        SELECT
            a.swc_id AS i,
            b.swc_id AS j,
            SUM(a.c * b.c)::DOUBLE AS Sxy
        FROM counts_by_voxel a
        JOIN counts_by_voxel b
            ON a.voxel_id = b.voxel_id
        JOIN swc_numeric a_num ON a.swc_id = a_num.swc_id
        JOIN swc_numeric b_num ON b.swc_id = b_num.swc_id
        WHERE a_num.swc_num < b_num.swc_num
        GROUP BY a.swc_id, b.swc_id
    """)

    # Pearson correlation: r = (V*Sxy - Sx*Sy) / sqrt((V*Sxx - Sx^2)(V*Syy - Sy^2))
    _progress("Computing Pearson correlations", 6)
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE corr_pairs AS
        WITH VU AS (SELECT V FROM voxel_universe)
        SELECT
            p.i,
            p.j,
            (VU.V * p.Sxy - x.Sx * y.Sx)
            / NULLIF(
                SQRT(
                    (VU.V * x.Sxx - x.Sx * x.Sx)
                    * (VU.V * y.Sxx - y.Sx * y.Sx)
                ),
                0
            ) AS r
        FROM pairwise_xy p
        JOIN per_neuron x ON p.i = x.swc_id
        JOIN per_neuron y ON p.j = y.swc_id
        CROSS JOIN VU
    """)

    # Build symmetric matrix (both triangles + diagonal)
    _progress("Building symmetric correlation table", 7)
    result_df = conn.execute("""
        SELECT i AS swc_id_1, j AS swc_id_2, r FROM corr_pairs
        UNION ALL
        SELECT j AS swc_id_1, i AS swc_id_2, r FROM corr_pairs
        UNION ALL
        SELECT swc_id AS swc_id_1, swc_id AS swc_id_2, 1.0 AS r FROM per_neuron
    """).fetchdf()

    # Clean up temp tables
    for table in [
        "region_nodes", "counts_by_voxel", "voxel_universe",
        "per_neuron", "swc_numeric", "pairwise_xy", "corr_pairs",
    ]:
        try:
            conn.execute(f"DROP VIEW IF EXISTS {table}")
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        except Exception:
            pass

    conn.unregister("lut")

    n_neurons = result_df["swc_id_1"].nunique()
    logger.info(
        f"Correlation matrix computed: {n_neurons} neurons, "
        f"{len(result_df)} entries"
    )
    return result_df


def correlation_long_to_matrix(
    corr_df: pd.DataFrame,
) -> tuple[pd.DataFrame, NDArray[np.float32]]:
    """Pivot long-form correlation DataFrame to a square matrix.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Long-form table with columns: swc_id_1, swc_id_2, r.

    Returns
    -------
    tuple[pd.DataFrame, NDArray[np.float32]]
        (mat_df, mat) where mat_df is a square DataFrame with neuron IDs
        as index and columns, and mat is the dense float32 array.
    """
    ids = pd.Index(
        pd.unique(pd.concat([corr_df["swc_id_1"], corr_df["swc_id_2"]]))
    ).sort_values()

    mat_df = corr_df.pivot(
        index="swc_id_1", columns="swc_id_2", values="r"
    ).reindex(index=ids, columns=ids)

    # Fill any missing pairs with -1 (uncorrelated/missing)
    mat_df.fillna(-1.0, inplace=True)

    mat = mat_df.to_numpy(dtype=np.float32)

    # Sanity checks
    if not np.allclose(mat, mat.T, equal_nan=True):
        logger.warning("Correlation matrix is not perfectly symmetric; forcing symmetry")
        mat = (mat + mat.T) / 2.0

    if not np.allclose(np.diag(mat), 1.0):
        logger.warning("Diagonal is not all 1.0; forcing diagonal to 1.0")
        np.fill_diagonal(mat, 1.0)

    logger.info(
        f"Correlation matrix: {mat.shape[0]}x{mat.shape[1]}, "
        f"range [{mat.min():.3f}, {mat.max():.3f}]"
    )
    return mat_df, mat
