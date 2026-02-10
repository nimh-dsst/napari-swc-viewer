"""DuckDB query interface for neuron data.

This module provides a NeuronDatabase class for efficient querying of
neuron data stored in Parquet format using DuckDB.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray


class NeuronDatabase:
    """DuckDB-based interface for querying neuron data from Parquet files.

    Parameters
    ----------
    parquet_path : Path or str
        Path to the Parquet file containing neuron data.

    Examples
    --------
    >>> db = NeuronDatabase("neurons.parquet")
    >>> # Get all neurons in a specific region
    >>> neurons = db.get_neurons_by_region(["VISp"])
    >>> # Get soma locations for all neurons
    >>> somas = db.get_soma_locations()
    """

    def __init__(self, parquet_path: Path | str):
        self.parquet_path = Path(parquet_path)
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")

        self.conn = duckdb.connect()
        self._setup_view()

    def _setup_view(self) -> None:
        """Create a view for the Parquet file."""
        path_str = str(self.parquet_path).replace("\\", "/")
        self.conn.execute(
            f"CREATE OR REPLACE VIEW neurons AS SELECT * FROM read_parquet('{path_str}')"
        )

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_neurons_by_region(
        self,
        region_acronyms: list[str],
        include_children: bool = False,
    ) -> pd.DataFrame:
        """Get neurons that have nodes in the specified regions.

        Parameters
        ----------
        region_acronyms : list[str]
            List of region acronyms to query (e.g., ["VISp", "VISl"]).
        include_children : bool, default=False
            If True, include child regions in the query.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: file_id, neuron_id, subject
        """
        if not region_acronyms:
            return pd.DataFrame(columns=["file_id", "neuron_id", "subject"])

        placeholders = ", ".join(["?"] * len(region_acronyms))
        query = f"""
            SELECT DISTINCT file_id, neuron_id, subject
            FROM neurons
            WHERE region_acronym IN ({placeholders})
            ORDER BY file_id
        """
        return self.conn.execute(query, region_acronyms).fetchdf()

    def get_neurons_by_region_id(
        self,
        region_ids: list[int],
    ) -> pd.DataFrame:
        """Get neurons that have nodes in the specified region IDs.

        Parameters
        ----------
        region_ids : list[int]
            List of Allen CCF region IDs.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: file_id, neuron_id, subject
        """
        if not region_ids:
            return pd.DataFrame(columns=["file_id", "neuron_id", "subject"])

        placeholders = ", ".join(["?"] * len(region_ids))
        query = f"""
            SELECT DISTINCT file_id, neuron_id, subject
            FROM neurons
            WHERE region_id IN ({placeholders})
            ORDER BY file_id
        """
        return self.conn.execute(query, region_ids).fetchdf()

    def get_unique_regions(self) -> pd.DataFrame:
        """Get all unique regions in the dataset with counts.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: region_id, region_acronym, region_name, node_count
        """
        query = """
            SELECT
                region_id,
                region_acronym,
                MAX(region_name) as region_name,
                COUNT(*) as node_count
            FROM neurons
            WHERE region_id > 0
            GROUP BY region_id, region_acronym
            ORDER BY node_count DESC
        """
        return self.conn.execute(query).fetchdf()

    def get_soma_locations(
        self,
        file_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """Get soma locations for neurons.

        Parameters
        ----------
        file_ids : list[str], optional
            Filter to specific files. If None, return all somas.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: file_id, neuron_id, x, y, z, region_acronym
        """
        if file_ids:
            placeholders = ", ".join(["?"] * len(file_ids))
            query = f"""
                SELECT
                    file_id, neuron_id,
                    AVG(x) as x, AVG(y) as y, AVG(z) as z,
                    MAX(region_acronym) as region_acronym
                FROM neurons
                WHERE type = 1 AND file_id IN ({placeholders})
                GROUP BY file_id, neuron_id
                ORDER BY file_id
            """
            return self.conn.execute(query, file_ids).fetchdf()
        else:
            query = """
                SELECT
                    file_id, neuron_id,
                    AVG(x) as x, AVG(y) as y, AVG(z) as z,
                    MAX(region_acronym) as region_acronym
                FROM neurons
                WHERE type = 1
                GROUP BY file_id, neuron_id
                ORDER BY file_id
            """
            return self.conn.execute(query).fetchdf()

    def get_neurons_for_rendering(
        self,
        file_ids: list[str],
    ) -> pd.DataFrame:
        """Get full neuron data for rendering.

        Parameters
        ----------
        file_ids : list[str]
            List of file IDs to retrieve.

        Returns
        -------
        pd.DataFrame
            DataFrame with all neuron node data for the specified files.
        """
        if not file_ids:
            return pd.DataFrame()

        placeholders = ", ".join(["?"] * len(file_ids))
        query = f"""
            SELECT *
            FROM neurons
            WHERE file_id IN ({placeholders})
            ORDER BY file_id, node_id
        """
        return self.conn.execute(query, file_ids).fetchdf()

    def get_neuron_lines(
        self,
        file_id: str,
    ) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
        """Get line segments for a single neuron.

        Parameters
        ----------
        file_id : str
            The file ID of the neuron.

        Returns
        -------
        tuple[NDArray, NDArray]
            (vertices, edges) where vertices is (N, 3) coordinates and
            edges is (M, 2) indices into vertices.
        """
        query = """
            SELECT node_id, x, y, z, parent_id
            FROM neurons
            WHERE file_id = ?
            ORDER BY node_id
        """
        df = self.conn.execute(query, [file_id]).fetchdf()

        if df.empty:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 2)

        # Build coordinate array
        coords = df[["x", "y", "z"]].values.astype(np.float64)

        # Build node_id to index mapping
        node_ids = df["node_id"].values
        id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        # Build edges
        edges = []
        for idx, parent_id in enumerate(df["parent_id"].values):
            if parent_id in id_to_idx:
                parent_idx = id_to_idx[parent_id]
                edges.append([parent_idx, idx])

        edges = np.array(edges, dtype=np.int32) if edges else np.array([]).reshape(0, 2)

        return coords, edges

    def get_statistics(self) -> dict:
        """Get summary statistics for the database.

        Returns
        -------
        dict
            Dictionary with keys: n_nodes, n_files, n_subjects, n_regions
        """
        stats = {}

        result = self.conn.execute("SELECT COUNT(*) FROM neurons").fetchone()
        stats["n_nodes"] = result[0]

        result = self.conn.execute(
            "SELECT COUNT(DISTINCT file_id) FROM neurons"
        ).fetchone()
        stats["n_files"] = result[0]

        result = self.conn.execute(
            "SELECT COUNT(DISTINCT subject) FROM neurons"
        ).fetchone()
        stats["n_subjects"] = result[0]

        result = self.conn.execute(
            "SELECT COUNT(DISTINCT region_id) FROM neurons WHERE region_id > 0"
        ).fetchone()
        stats["n_regions"] = result[0]

        return stats

    def get_region_neuron_counts(self) -> pd.DataFrame:
        """Get neuron counts per region.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: region_acronym, region_name, neuron_count
        """
        query = """
            SELECT
                region_acronym,
                MAX(region_name) as region_name,
                COUNT(DISTINCT file_id) as neuron_count
            FROM neurons
            WHERE region_acronym != ''
            GROUP BY region_acronym
            ORDER BY neuron_count DESC
        """
        return self.conn.execute(query).fetchdf()

    def query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        """Execute a custom SQL query.

        Parameters
        ----------
        sql : str
            SQL query string. Use 'neurons' as the table name.
        params : list, optional
            Query parameters for placeholders.

        Returns
        -------
        pd.DataFrame
            Query results as a DataFrame.

        Examples
        --------
        >>> db.query("SELECT * FROM neurons WHERE type = 2 LIMIT 10")
        >>> db.query("SELECT * FROM neurons WHERE file_id = ?", ["file.swc"])
        """
        if params:
            return self.conn.execute(sql, params).fetchdf()
        return self.conn.execute(sql).fetchdf()
