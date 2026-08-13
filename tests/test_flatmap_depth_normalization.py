"""Tests for flat map + depth coordinate normalization in soma clustering.

The raw Parquet columns mix units: ``x_flat``/``y_flat`` are normalized floats
spanning the bilateral flat map while ``depth_um`` is raw microns spanning the
cortical thickness.  Clustering them together unweighted lets depth supply
>99.99% of the Euclidean variance, collapsing the result to a laminar partition.
These tests guard the normalization that fixes that, plus the two traps it is
easy to reintroduce: giving the flat map axes separate divisors, which distorts
flat map space, and treating "ignore depth" as a zero weight rather than a
dropped axis.

Both flat map axes share one divisor taken from the ``y`` span, matching how the
voxel grid derives its bin counts. ``tests/test_flatmap_bin_counts.py`` covers
the grid side of that same policy.
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


def test_square_style_divides_by_the_hemisphere_height(canonical_parquet) -> None:
    """The shared divisor is the y span, which is one hemisphere tall."""
    _frame, path = canonical_parquet
    norm = resolve_flatmap_depth_normalization(path, style="both_square")

    assert norm.bounds_source == "canonical"
    assert norm.flatmap_divisor == pytest.approx(1.0)
    assert norm.depth_divisor_um == pytest.approx(1856.49658203125)


def test_square_style_is_unchanged_by_the_shared_divisor(canonical_parquet) -> None:
    """``both_square`` is an exact 2:1 map, so sharing a divisor changes nothing.

    Its old per-axis divisors were ``x_span / 2 == 1.0`` and ``y_span == 1.0``,
    already equal.  Pinning that here separates "the fix" from "a regression":
    any change in square-style clustering results is a bug, not the intended
    effect.
    """
    _frame, path = canonical_parquet
    norm = resolve_flatmap_depth_normalization(path, style="both_square")

    legacy_x_divisor = (SQUARE_X_BOUNDS[1] - SQUARE_X_BOUNDS[0]) / 2.0
    legacy_y_divisor = SQUARE_Y_BOUNDS[1] - SQUARE_Y_BOUNDS[0]
    assert legacy_x_divisor == pytest.approx(legacy_y_divisor)
    assert norm.flatmap_divisor == pytest.approx(legacy_x_divisor)


@pytest.mark.parametrize("style", ["both_square", "both_shaped"])
def test_flat_map_space_is_isotropic(canonical_parquet, style: str) -> None:
    """Equal flat map separations must give equal distances on both axes.

    This is the test the old normalization lacked. It divided each axis by its
    own span, forcing every style's hemisphere into a square bounding box; for
    ``both_shaped`` that stretched x by 4.2% relative to y. Asserting on a
    *distance* rather than on divisor values is what makes it style-agnostic.
    """
    _frame, path = canonical_parquet
    norm = resolve_flatmap_depth_normalization(
        path,
        style=style,
        include_depth=False,
    )

    step = 0.01
    origin = (0.5, 0.5, 0.0)
    moved_x = (0.5 + step, 0.5, 0.0)
    moved_y = (0.5, 0.5 + step, 0.0)
    scaled = normalize_flatmap_soma_coordinates(
        np.asarray([origin, moved_x, moved_y], dtype=float),
        norm,
    )
    x_distance = float(np.linalg.norm(scaled[1] - scaled[0]))
    y_distance = float(np.linalg.norm(scaled[2] - scaled[0]))

    assert x_distance == pytest.approx(y_distance)
    # Under the old per-axis divisors the shaped style failed this by 4.2%.
    assert x_distance > 0.0


def test_shaped_style_no_longer_stretches_x(canonical_parquet) -> None:
    """Pin the removed distortion, so the regression cannot come back quietly.

    The old shaped divisors were ``x_span / 2`` (0.9042) for x and ``y_span``
    (0.9434) for y, a ratio of 0.9581 — x differences were inflated ~4.4%. The
    shared divisor must be the y span, not the old halved x span.
    """
    _frame, path = canonical_parquet
    norm = resolve_flatmap_depth_normalization(path, style="both_shaped")

    y_span = SHAPED_Y_BOUNDS[1] - SHAPED_Y_BOUNDS[0]
    legacy_x_divisor = (SHAPED_X_BOUNDS[1] - SHAPED_X_BOUNDS[0]) / 2.0

    assert norm.flatmap_divisor == pytest.approx(y_span)
    assert norm.flatmap_divisor != pytest.approx(legacy_x_divisor)
    # The anisotropy that used to be present, quantified.
    assert legacy_x_divisor / y_span == pytest.approx(0.958, abs=0.01)


def test_both_styles_share_the_binning_reference_axis(canonical_parquet) -> None:
    """The normalization and the voxel grid must agree on what sets the unit.

    Both take ``y`` as the reference axis: the grid's bin width is
    ``y_span / y_bins`` on both axes, and the metric's unit is the ``y`` span.
    A change to either that broke this pairing would silently put soma
    clustering and the heatmap on differently-proportioned spaces.
    """
    from napari_swc_viewer.flatmap_heatmap import resolve_flatmap_bin_counts

    _frame, path = canonical_parquet
    for style, x_bounds, y_bounds in (
        ("both_square", SQUARE_X_BOUNDS, SQUARE_Y_BOUNDS),
        ("both_shaped", SHAPED_X_BOUNDS, SHAPED_Y_BOUNDS),
    ):
        norm = resolve_flatmap_depth_normalization(path, style=style)
        y_bins = 256
        counts = resolve_flatmap_bin_counts(
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            y_bins=y_bins,
        )
        # The grid's bin width and the metric's unit differ only by y_bins.
        bin_width = (y_bounds[1] - y_bounds[0]) / counts.y_bins
        assert norm.flatmap_divisor / y_bins == pytest.approx(bin_width)
        # And an x step of one metric unit spans exactly x_bins/y_bins bins.
        assert counts.x_bins / counts.y_bins == pytest.approx(
            (x_bounds[1] - x_bounds[0]) / norm.flatmap_divisor,
            rel=1e-3,
        )


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
    """Without canonical bounds the observed y span sets the unit.

    The old code needed a branch here: canonical x spans two hemispheres and
    was halved, while observed x might span one and was not. Taking the unit
    from y removes that guess -- y spans one hemisphere either way -- so this
    path no longer special-cases anything.
    """
    _frame, path = legacy_parquet
    norm = resolve_flatmap_depth_normalization(path, style="both_square")

    assert norm.bounds_source == "observed"
    observed_y_span = 0.93 - 0.10
    assert norm.flatmap_divisor == pytest.approx(observed_y_span, abs=1e-4)


def test_normalization_rejects_wrongly_shaped_coordinates() -> None:
    norm = FlatmapDepthNormalization(
        style="both_square",
        flatmap_divisor=1.0,
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
    assert payload["flatmap_divisor"] == pytest.approx(1.0)
    # One divisor, not two that must be kept equal.
    assert "x_divisor" not in payload
    assert "y_divisor" not in payload
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
    assert result.metadata.distance_metric == ("euclidean_flatmap_isotropic_plus_depth")
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
    assert result.metadata.distance_metric == "euclidean_flatmap_isotropic"
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
