"""Tests for flat map + depth coordinate normalization in soma clustering.

The raw Parquet columns mix units: ``x_flat``/``y_flat`` are normalized floats
spanning the bilateral flat map while ``depth_um`` is raw microns spanning the
cortical thickness.  Clustering them together unweighted lets depth supply
>99.99% of the Euclidean variance, collapsing the result to a laminar partition.
These tests guard the normalization that fixes that, plus the two traps it is
easy to reintroduce: halving the bilateral ``x`` span, and treating "ignore
depth" as a zero weight rather than a dropped axis.
"""

from __future__ import annotations

import json
import types

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from napari_swc_viewer.analysis.flatmap_correlation import (
    DEFAULT_FLATMAP_DEPTH_SCALE,
    FlatmapDepthNormalization,
    normalize_flatmap_soma_coordinates,
    query_flatmap_soma_coordinates,
    resolve_flatmap_depth_normalization,
)
from napari_swc_viewer.flatmap_parquet import FLATMAP_PARQUET_METADATA_KEY

# Canonical bounds taken from the real
# ``isocortex_total_right_brainglobe_flatmap.parquet``: ``x`` spans both
# hemispheres, ``y`` spans one, and depth is the full cortical thickness.
SQUARE_X_BOUNDS = (0.0, 2.0)
SQUARE_Y_BOUNDS = (0.0, 1.0)
SHAPED_X_BOUNDS = (0.09574997425079346, 1.9042149782180786)
SHAPED_Y_BOUNDS = (0.05073529854416847, 0.9944919943809509)
DEPTH_BOUNDS_UM = (0.0, 1856.49658203125)


def _grid_spec_dict(style: str) -> dict:
    """Return a minimal but complete v3 grid spec for one bilateral style."""
    x_bounds = SQUARE_X_BOUNDS if style == "both_square" else SHAPED_X_BOUNDS
    y_bounds = SQUARE_Y_BOUNDS if style == "both_square" else SHAPED_Y_BOUNDS
    return {
        "grid_spec_id": f"fmg1-test-{style}",
        "style": style,
        "lookup_coordinate_order": ["x", "y", "z"],
        "flatmap_coordinate_order": ["x_flat", "y_flat"],
        "render_coordinate_order": ["depth", "y", "x"],
        "spatial_shape": [528, 320, 456],
        "flatmap_shape": [528, 320, 456, 2],
        "depth_shape": [528, 320, 456],
        "lookup_resolution_um": [25.0, 25.0, 25.0],
        "space_directions": [[25.0, 0.0, 0.0], [0.0, 25.0, 0.0], [0.0, 0.0, 25.0]],
        "space_origin": [0.0, 0.0, 0.0],
        "x_bounds": list(x_bounds),
        "y_bounds": list(y_bounds),
        "depth_bounds_um": list(DEPTH_BOUNDS_UM),
        "validity": {
            "invalid_zero_sentinel": False,
            "invalid_negative_one_sentinel": True,
            "depth_invalid_below_um": 0.0,
        },
    }


def _v3_metadata() -> bytes:
    return json.dumps(
        {
            "version": 3,
            "lookup_set": {
                "styles": {
                    style: _grid_spec_dict(style)
                    for style in ("both_shaped", "both_square")
                }
            },
        }
    ).encode("utf-8")


def _soma_frame() -> pd.DataFrame:
    """Build somas spread across one hemisphere with realistic depth microns.

    Neurons are laid out so that flat map position and depth disagree: pairs
    that are adjacent tangentially sit in different layers, and pairs at the
    same depth sit far apart tangentially.  A depth-dominated metric and a
    balanced one therefore produce different groupings.
    """
    # (x_flat_square, y_flat_square, depth_um)
    somas = [
        (1.05, 0.10, 150.0),
        (1.08, 0.13, 1400.0),
        (1.90, 0.90, 160.0),
        (1.93, 0.93, 1420.0),
    ]
    rows: dict[str, list] = {name: [] for name in _COLUMNS}
    node_id = 0
    for index, (xq, yq, depth) in enumerate(somas):
        rows["file_id"].append(f"neuron_{index}")
        rows["node_id"].append(node_id)
        rows["parent_id"].append(-1)
        rows["type"].append(1)
        rows["x"].append(xq * 1000.0)
        rows["y"].append(yq * 1000.0)
        rows["z"].append(depth)
        rows["x_flat_shaped"].append(xq - 0.05)
        rows["y_flat_shaped"].append(yq + 0.05)
        rows["flatmap_shaped_valid"].append(True)
        rows["x_flat_square"].append(xq)
        rows["y_flat_square"].append(yq)
        rows["flatmap_square_valid"].append(True)
        rows["depth_um"].append(depth)
        rows["depth_valid"].append(True)
        node_id += 1
    frame = pd.DataFrame(rows)
    frame["node_id"] = frame["node_id"].astype(np.int32)
    frame["parent_id"] = frame["parent_id"].astype(np.int32)
    frame["type"] = frame["type"].astype(np.int32)
    for name in (
        "x",
        "y",
        "z",
        "x_flat_shaped",
        "y_flat_shaped",
        "x_flat_square",
        "y_flat_square",
        "depth_um",
    ):
        frame[name] = frame[name].astype(np.float32)
    return frame


_COLUMNS = (
    "file_id",
    "node_id",
    "parent_id",
    "type",
    "x",
    "y",
    "z",
    "x_flat_shaped",
    "y_flat_shaped",
    "flatmap_shaped_valid",
    "x_flat_square",
    "y_flat_square",
    "flatmap_square_valid",
    "depth_um",
    "depth_valid",
)


def _write_parquet(tmp_path, *, with_metadata: bool):
    frame = _soma_frame()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if with_metadata:
        schema_metadata = dict(table.schema.metadata or {})
        schema_metadata[FLATMAP_PARQUET_METADATA_KEY] = _v3_metadata()
        table = table.replace_schema_metadata(schema_metadata)
    name = "with_meta.parquet" if with_metadata else "no_meta.parquet"
    path = tmp_path / name
    pq.write_table(table, path)
    return frame, str(path)


@pytest.fixture()
def canonical_parquet(tmp_path):
    return _write_parquet(tmp_path, with_metadata=True)


@pytest.fixture()
def legacy_parquet(tmp_path):
    return _write_parquet(tmp_path, with_metadata=False)


def test_square_hemisphere_is_a_unit_cube(canonical_parquet) -> None:
    """The square style's x divisor must halve the bilateral span."""
    _frame, path = canonical_parquet
    norm = resolve_flatmap_depth_normalization(path, style="both_square")

    assert norm.bounds_source == "canonical"
    # x spans two hemispheres, so one hemisphere is half of it.
    assert norm.x_divisor == pytest.approx(1.0)
    assert norm.y_divisor == pytest.approx(1.0)
    assert norm.depth_divisor_um == pytest.approx(1856.49658203125)


def test_square_x_divisor_is_not_the_full_bilateral_span(canonical_parquet) -> None:
    """Guard the trap that silently gives y twice x's weight.

    Dividing x by the full canonical span (2.0) squeezes each hemisphere to
    half width, so tangential distance along x counts for half as much as the
    same distance along y.
    """
    _frame, path = canonical_parquet
    norm = resolve_flatmap_depth_normalization(path, style="both_square")

    full_span = SQUARE_X_BOUNDS[1] - SQUARE_X_BOUNDS[0]
    assert norm.x_divisor != pytest.approx(full_span)
    assert norm.x_divisor / norm.y_divisor == pytest.approx(1.0)


def test_shaped_style_corrects_its_aspect_ratio(canonical_parquet) -> None:
    """The shaped map's hemisphere is not square and must be corrected."""
    _frame, path = canonical_parquet
    norm = resolve_flatmap_depth_normalization(path, style="both_shaped")

    expected_x = (SHAPED_X_BOUNDS[1] - SHAPED_X_BOUNDS[0]) / 2.0
    expected_y = SHAPED_Y_BOUNDS[1] - SHAPED_Y_BOUNDS[0]
    assert norm.x_divisor == pytest.approx(expected_x)
    assert norm.y_divisor == pytest.approx(expected_y)
    # The raw shaped hemisphere is ~4% off square, which is why it needs its
    # own divisors rather than reusing the square style's.
    assert expected_x / expected_y == pytest.approx(0.958, abs=0.01)


def test_normalization_puts_axes_on_a_comparable_scale(canonical_parquet) -> None:
    """Depth must stop supplying essentially all of the variance."""
    _frame, path = canonical_parquet
    ids, raw = query_flatmap_soma_coordinates(path, style="both_square")
    assert len(ids) == 4

    raw_share = raw.var(axis=0) / raw.var(axis=0).sum()
    assert raw_share[2] > 0.999, "expected raw depth to dominate before scaling"

    norm = resolve_flatmap_depth_normalization(path, style="both_square")
    scaled = normalize_flatmap_soma_coordinates(raw, norm)
    scaled_share = scaled.var(axis=0) / scaled.var(axis=0).sum()
    # No axis may dominate; depth in particular drops from >99.9% to a share
    # comparable with the two flat map axes.
    assert scaled_share.max() < 0.9
    assert scaled_share[2] > 0.01


def test_depth_scale_monotonically_reweights_depth(canonical_parquet) -> None:
    """Raising the depth scale must raise depth's share of the variance."""
    _frame, path = canonical_parquet
    _ids, raw = query_flatmap_soma_coordinates(path, style="both_square")

    shares = []
    for scale in (0.25, 1.0, 4.0):
        norm = resolve_flatmap_depth_normalization(
            path,
            style="both_square",
            depth_scale=scale,
        )
        scaled = normalize_flatmap_soma_coordinates(raw, norm)
        variance = scaled.var(axis=0)
        shares.append(variance[2] / variance.sum())

    assert shares[0] < shares[1] < shares[2]


def test_ignoring_depth_drops_the_axis_entirely(canonical_parquet) -> None:
    """Excluding depth must return 2-D coordinates, not a zeroed third axis.

    A zeroed axis still contributes a column to the distance computation and
    would leave DBSCAN and k-means operating in 3-D with a degenerate
    dimension; dropping it makes the flat-map-only intent explicit.
    """
    _frame, path = canonical_parquet
    _ids, raw = query_flatmap_soma_coordinates(path, style="both_square")

    norm = resolve_flatmap_depth_normalization(
        path,
        style="both_square",
        include_depth=False,
    )
    scaled = normalize_flatmap_soma_coordinates(raw, norm)

    assert norm.axis_count == 2
    assert scaled.shape == (4, 2)


def test_zero_depth_scale_still_keeps_three_axes(canonical_parquet) -> None:
    """A zero weight is distinct from excluding depth and stays 3-D."""
    _frame, path = canonical_parquet
    _ids, raw = query_flatmap_soma_coordinates(path, style="both_square")

    norm = resolve_flatmap_depth_normalization(
        path,
        style="both_square",
        depth_scale=0.0,
    )
    scaled = normalize_flatmap_soma_coordinates(raw, norm)

    assert scaled.shape == (4, 3)
    assert np.allclose(scaled[:, 2], 0.0)


def test_negative_depth_scale_is_rejected(canonical_parquet) -> None:
    _frame, path = canonical_parquet
    with pytest.raises(ValueError, match="depth_scale must be non-negative"):
        resolve_flatmap_depth_normalization(
            path,
            style="both_square",
            depth_scale=-1.0,
        )


def test_legacy_parquet_falls_back_to_observed_bounds(legacy_parquet) -> None:
    """Without canonical bounds the observed span is used unhalved.

    Observed bounds already cover only the hemispheres present in the data, so
    applying the bilateral halving here would stretch the x axis.
    """
    _frame, path = legacy_parquet
    norm = resolve_flatmap_depth_normalization(path, style="both_square")

    assert norm.bounds_source == "observed"
    observed_x_span = 1.93 - 1.05
    assert norm.x_divisor == pytest.approx(observed_x_span, abs=1e-4)


def test_normalization_rejects_wrongly_shaped_coordinates() -> None:
    norm = FlatmapDepthNormalization(
        style="both_square",
        x_divisor=1.0,
        y_divisor=1.0,
        depth_divisor_um=1856.5,
    )
    with pytest.raises(ValueError, match=r"\(N, 3\) array"):
        normalize_flatmap_soma_coordinates(np.zeros((4, 2)), norm)


def test_normalization_metadata_round_trips(canonical_parquet) -> None:
    """Provenance must record the scale actually used, not just the request."""
    _frame, path = canonical_parquet
    norm = resolve_flatmap_depth_normalization(
        path,
        style="both_square",
        depth_scale=2.5,
        include_depth=True,
    )
    payload = norm.to_dict()

    assert payload["depth_scale"] == pytest.approx(2.5)
    assert payload["include_depth"] is True
    assert payload["bounds_source"] == "canonical"
    assert payload["axis_count"] == 3
    assert payload["x_divisor"] == pytest.approx(1.0)
    # Must be JSON-serializable for workbook/parquet exports.
    assert json.loads(json.dumps(payload)) == payload


def _fake_atlas():
    return types.SimpleNamespace(
        atlas_name="test_atlas",
        resolution=(25.0, 25.0, 25.0),
    )


def _run_worker(path, **kwargs):
    from napari_swc_viewer import workers

    finished: list = []
    errors: list = []
    worker = workers.FlatmapSomaClusterWorker(
        parquet_path=path,
        atlas=_fake_atlas(),
        style="both_square",
        algorithm="hierarchical",
        linkage_method="ward",
        n_clusters=2,
        **kwargs,
    )
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)
    worker.run()
    assert errors == []
    assert len(finished) == 1
    return finished[0]


def test_worker_records_normalization_provenance(canonical_parquet) -> None:
    _frame, path = canonical_parquet
    result = _run_worker(path, depth_scale=2.0)

    assert result.metadata is not None
    assert result.metadata.distance_metric == (
        "euclidean_flatmap_depth_unit_hemisphere"
    )
    recorded = result.metadata.extra_metadata["flatmap_normalization"]
    assert recorded["depth_scale"] == pytest.approx(2.0)
    assert recorded["include_depth"] is True
    assert recorded["axis_count"] == 3


def test_worker_can_cluster_on_flatmap_position_only(canonical_parquet) -> None:
    """Ignoring depth must group the tangentially adjacent pairs together.

    The fixture places neurons 0/1 close together in flat map space but in
    different layers, and 2/3 likewise.  A flat-map-only run must therefore
    split {0, 1} from {2, 3} — the grouping a depth-dominated metric cannot
    produce, since it would pair the two shallow somas against the two deep
    ones instead.
    """
    _frame, path = canonical_parquet
    result = _run_worker(path, include_depth=False)

    assert result.metadata is not None
    assert result.metadata.distance_metric == "euclidean_flatmap_xy_unit_hemisphere"
    recorded = result.metadata.extra_metadata["flatmap_normalization"]
    assert recorded["include_depth"] is False
    assert recorded["axis_count"] == 2

    labels = dict(zip(result.neuron_ids, result.labels.tolist()))
    assert labels["neuron_0"] == labels["neuron_1"]
    assert labels["neuron_2"] == labels["neuron_3"]
    assert labels["neuron_0"] != labels["neuron_2"]


def test_high_depth_scale_groups_by_layer_instead(canonical_parquet) -> None:
    """A large depth scale must flip the grouping to laminar pairs.

    This is the control for the previous test: the same fixture, clustered with
    depth weighted heavily, pairs the two shallow somas and the two deep ones
    across the flat map instead.
    """
    _frame, path = canonical_parquet
    result = _run_worker(path, depth_scale=50.0)

    labels = dict(zip(result.neuron_ids, result.labels.tolist()))
    assert labels["neuron_0"] == labels["neuron_2"]
    assert labels["neuron_1"] == labels["neuron_3"]
    assert labels["neuron_0"] != labels["neuron_1"]


def test_default_depth_scale_is_one() -> None:
    """The documented default must stay the unit-hemisphere weighting."""
    assert DEFAULT_FLATMAP_DEPTH_SCALE == 1.0
