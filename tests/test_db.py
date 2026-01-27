"""Unit tests for the NeuronDatabase class."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from napari_swc_viewer.db import NeuronDatabase
from napari_swc_viewer.parquet import NEURON_SCHEMA


@pytest.fixture
def sample_parquet(tmp_path: Path) -> Path:
    """Create a sample Parquet file for testing."""
    rows = [
        # Neuron 1 - soma and axon nodes
        {
            "file_id": "neuron1.swc",
            "node_id": 1,
            "type": 1,  # soma
            "x": 100.0,
            "y": 200.0,
            "z": 300.0,
            "radius": 10.0,
            "parent_id": -1,
            "region_id": 385,
            "region_name": "Primary visual area",
            "region_acronym": "VISp",
            "subject": "subject1",
            "neuron_id": "neuron1",
        },
        {
            "file_id": "neuron1.swc",
            "node_id": 2,
            "type": 2,  # axon
            "x": 110.0,
            "y": 210.0,
            "z": 310.0,
            "radius": 1.0,
            "parent_id": 1,
            "region_id": 385,
            "region_name": "Primary visual area",
            "region_acronym": "VISp",
            "subject": "subject1",
            "neuron_id": "neuron1",
        },
        {
            "file_id": "neuron1.swc",
            "node_id": 3,
            "type": 2,  # axon
            "x": 120.0,
            "y": 220.0,
            "z": 320.0,
            "radius": 0.5,
            "parent_id": 2,
            "region_id": 394,
            "region_name": "Lateral visual area",
            "region_acronym": "VISl",
            "subject": "subject1",
            "neuron_id": "neuron1",
        },
        # Neuron 2 - different region
        {
            "file_id": "neuron2.swc",
            "node_id": 1,
            "type": 1,  # soma
            "x": 500.0,
            "y": 600.0,
            "z": 700.0,
            "radius": 12.0,
            "parent_id": -1,
            "region_id": 669,
            "region_name": "Primary motor area",
            "region_acronym": "MOp",
            "subject": "subject2",
            "neuron_id": "neuron2",
        },
        {
            "file_id": "neuron2.swc",
            "node_id": 2,
            "type": 3,  # dendrite
            "x": 510.0,
            "y": 610.0,
            "z": 710.0,
            "radius": 2.0,
            "parent_id": 1,
            "region_id": 669,
            "region_name": "Primary motor area",
            "region_acronym": "MOp",
            "subject": "subject2",
            "neuron_id": "neuron2",
        },
    ]

    table = pa.Table.from_pylist(rows, schema=NEURON_SCHEMA)
    parquet_path = tmp_path / "test_neurons.parquet"
    pq.write_table(table, parquet_path)

    return parquet_path


class TestNeuronDatabase:
    def test_init_valid_file(self, sample_parquet: Path):
        """Test initialization with a valid Parquet file."""
        db = NeuronDatabase(sample_parquet)
        assert db.parquet_path == sample_parquet
        db.close()

    def test_init_invalid_file(self, tmp_path: Path):
        """Test initialization with a non-existent file."""
        with pytest.raises(FileNotFoundError):
            NeuronDatabase(tmp_path / "nonexistent.parquet")

    def test_context_manager(self, sample_parquet: Path):
        """Test using NeuronDatabase as a context manager."""
        with NeuronDatabase(sample_parquet) as db:
            stats = db.get_statistics()
            assert stats["n_nodes"] == 5

    def test_get_neurons_by_region(self, sample_parquet: Path):
        """Test querying neurons by region acronym."""
        with NeuronDatabase(sample_parquet) as db:
            # Single region
            result = db.get_neurons_by_region(["VISp"])
            assert len(result) == 1
            assert result.iloc[0]["file_id"] == "neuron1.swc"

            # Multiple regions
            result = db.get_neurons_by_region(["VISp", "MOp"])
            assert len(result) == 2

            # Non-existent region
            result = db.get_neurons_by_region(["NONEXISTENT"])
            assert len(result) == 0

            # Empty list
            result = db.get_neurons_by_region([])
            assert len(result) == 0

    def test_get_neurons_by_region_id(self, sample_parquet: Path):
        """Test querying neurons by region ID."""
        with NeuronDatabase(sample_parquet) as db:
            result = db.get_neurons_by_region_id([385])
            assert len(result) == 1
            assert result.iloc[0]["file_id"] == "neuron1.swc"

            result = db.get_neurons_by_region_id([669])
            assert len(result) == 1
            assert result.iloc[0]["file_id"] == "neuron2.swc"

    def test_get_unique_regions(self, sample_parquet: Path):
        """Test getting unique regions."""
        with NeuronDatabase(sample_parquet) as db:
            result = db.get_unique_regions()

            assert len(result) == 3  # VISp, VISl, MOp
            assert "region_id" in result.columns
            assert "region_acronym" in result.columns
            assert "node_count" in result.columns

    def test_get_soma_locations(self, sample_parquet: Path):
        """Test getting soma locations."""
        with NeuronDatabase(sample_parquet) as db:
            result = db.get_soma_locations()

            assert len(result) == 2  # Two neurons with somas
            assert "x" in result.columns
            assert "y" in result.columns
            assert "z" in result.columns

            # Filter by file_id
            result = db.get_soma_locations(["neuron1.swc"])
            assert len(result) == 1

    def test_get_neurons_for_rendering(self, sample_parquet: Path):
        """Test getting full neuron data for rendering."""
        with NeuronDatabase(sample_parquet) as db:
            result = db.get_neurons_for_rendering(["neuron1.swc"])

            assert len(result) == 3  # neuron1 has 3 nodes
            assert "x" in result.columns
            assert "parent_id" in result.columns

            # Multiple files
            result = db.get_neurons_for_rendering(["neuron1.swc", "neuron2.swc"])
            assert len(result) == 5

            # Empty list
            result = db.get_neurons_for_rendering([])
            assert len(result) == 0

    def test_get_neuron_lines(self, sample_parquet: Path):
        """Test getting line segments for a neuron."""
        with NeuronDatabase(sample_parquet) as db:
            coords, edges = db.get_neuron_lines("neuron1.swc")

            assert coords.shape == (3, 3)  # 3 nodes, 3 dimensions
            assert edges.shape == (2, 2)  # 2 edges (node 2->1, 3->2)

            # Non-existent file
            coords, edges = db.get_neuron_lines("nonexistent.swc")
            assert coords.shape == (0, 3)
            assert edges.shape == (0, 2)

    def test_get_statistics(self, sample_parquet: Path):
        """Test getting summary statistics."""
        with NeuronDatabase(sample_parquet) as db:
            stats = db.get_statistics()

            assert stats["n_nodes"] == 5
            assert stats["n_files"] == 2
            assert stats["n_subjects"] == 2
            assert stats["n_regions"] == 3

    def test_get_region_neuron_counts(self, sample_parquet: Path):
        """Test getting neuron counts per region."""
        with NeuronDatabase(sample_parquet) as db:
            result = db.get_region_neuron_counts()

            assert len(result) == 3
            assert "region_acronym" in result.columns
            assert "neuron_count" in result.columns

    def test_custom_query(self, sample_parquet: Path):
        """Test executing custom SQL queries."""
        with NeuronDatabase(sample_parquet) as db:
            # Simple query
            result = db.query("SELECT COUNT(*) as cnt FROM neurons")
            assert result.iloc[0]["cnt"] == 5

            # Parameterized query
            result = db.query(
                "SELECT * FROM neurons WHERE type = ?",
                [1],
            )
            assert len(result) == 2  # Two soma nodes


class TestNeuronDatabaseEdgeCases:
    def test_empty_parquet(self, tmp_path: Path):
        """Test handling an empty Parquet file."""
        rows = []
        table = pa.Table.from_pylist(rows, schema=NEURON_SCHEMA)
        parquet_path = tmp_path / "empty.parquet"
        pq.write_table(table, parquet_path)

        with NeuronDatabase(parquet_path) as db:
            stats = db.get_statistics()
            assert stats["n_nodes"] == 0

            result = db.get_soma_locations()
            assert len(result) == 0

    def test_no_soma_nodes(self, tmp_path: Path):
        """Test handling neurons without soma nodes."""
        rows = [
            {
                "file_id": "neuron.swc",
                "node_id": 1,
                "type": 2,  # axon, not soma
                "x": 100.0,
                "y": 200.0,
                "z": 300.0,
                "radius": 1.0,
                "parent_id": -1,
                "region_id": 385,
                "region_name": "Primary visual area",
                "region_acronym": "VISp",
                "subject": "subject1",
                "neuron_id": "neuron1",
            },
        ]

        table = pa.Table.from_pylist(rows, schema=NEURON_SCHEMA)
        parquet_path = tmp_path / "no_soma.parquet"
        pq.write_table(table, parquet_path)

        with NeuronDatabase(parquet_path) as db:
            result = db.get_soma_locations()
            assert len(result) == 0
