"""Compare paired pre/post clustering exports to measure a metric change.

Written to validate the flat map soma-clustering isotropy fix: the metric used
to divide ``x_flat`` by ``x_span / 2`` and ``y_flat`` by ``y_span``, giving
``both_shaped`` a 4.2% x/y anisotropy.  Both axes now share one divisor.  There
is no ground truth for "correct" cluster assignments, so this script does not
try to score accuracy.  It does two things that *are* decidable without truth:

1. **Verifies the pair is comparable.**  A change rate only means something if
   the two runs clustered the same neurons with the same algorithm and
   parameters, and differ *only* in the metric.  Every pair is gated on that
   before any number is reported.
2. **Measures how much the assignment moved**, permutation-invariantly.

Cluster IDs are arbitrary labels.  Comparing them directly overstates change
wildly -- a run that merely renumbered its clusters would look 100% changed.
Every number here is either permutation-invariant (ARI, AMI) or computed after
optimally matching pre clusters to post clusters (the change fraction).

**The change rate is strongly scale- and algorithm-dependent, so quote it with
both.**  Measured on ``isocortex_total_right_brainglobe_flatmap.parquet`` at
``depth_scale=1.0``, k=5:

===================  ==============  ==============
algorithm            833 neurons     18,518 neurons
===================  ==============  ==============
ward (hierarchical)  1.92%           37.51%
kmeans               0.36%           1.40%
===================  ==============  ==============

Ward is agglomerative: a small metric perturbation flips near-tied merges and
the reordering cascades, and more points means more near-ties.  Seeded k-means
is far more stable.  A change rate from a subset therefore does **not** bound
the full-dataset rate -- the two differ ~20x here.  Do not treat a small number
from a partial table as evidence the metric barely moved.

The workbooks are **not** in version control -- ``anisotropy_fix_data/`` and
``*.xlsx`` are gitignored, because this project keeps no data files in the
repository.  Regenerate them from napari to re-derive the numbers; see
``ANISOTROPY_FIX_RESULTS.MD`` for the settings each pair was exported under.

Usage::

    python scripts/compare_clustering_exports.py
    python scripts/compare_clustering_exports.py --dir some/other/dir
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

#: Metadata fields that MUST match across a pre/post pair.  These define "same
#: experiment": same neurons, same algorithm, same parameters, same source.  A
#: mismatch in any of them means the pair measures something other than the
#: metric change and the comparison is void.
INVARIANT_FIELDS = (
    "analysis_method",
    "clustering_algorithm",
    "clustering_linkage",
    "requested_cluster_count",
    "dbscan_eps",
    "dbscan_min_samples",
    "selected_region_ids",
    "dilation_fraction",
    "atlas_name",
    "source_parquet_path",
)

#: Metadata fields that must differ **for a soma-location pair**, because the
#: fix renamed them.  If these match, the "post" run did not actually use the
#: fixed code -- the most likely real-world failure and the reason this check
#: exists.  Voxel-correlation pairs are exempt: see NULL_CONTROL_METHOD.
EXPECTED_TO_DIFFER = ("distance_metric",)

#: Voxel correlation clusters on ``1 - pearson_r`` between voxelized node
#: densities and never touches the soma normalization, so a voxel pair is a
#: **null control**: it shares the run's neuron set and grid but must come out
#: bit-identical.  A difference here means something other than the metric
#: changed between the two launches, which would invalidate the soma numbers
#: too.  Its ``distance_metric`` is therefore expected to stay the same.
NULL_CONTROL_METHOD = "flatmap_voxel_correlation"

#: ``both_square`` is an exact 2:1 map, so its old per-axis divisors were
#: already ``x_span / 2 == 1.0`` and ``y_span == 1.0`` -- equal.  Sharing one
#: divisor therefore cannot change its results, making a square-style soma pair
#: the sharpest null control available: it exercises the *same* code path the
#: fix changed and must still come out bit-identical.  Any movement here is a
#: bug in the fix rather than the fix working.  Unlike the voxel control, its
#: ``distance_metric`` *does* change, because the rename is unconditional.
IDENTITY_STYLE = "both_square"

#: Where the paired workbooks live by default, relative to the repository root.
#: Gitignored, so this is a local-only directory that each user regenerates.
DEFAULT_EXPORT_DIR = "anisotropy_fix_data"

#: DBSCAN marks unassigned points with a sentinel rather than a cluster id.
#: Both are checked because the export layer may renumber from zero.
NOISE_LABELS = (-1, 0)


@dataclass(frozen=True)
class Export:
    """One clustering export: per-neuron labels plus its run metadata."""

    path: Path
    labels: pd.Series  # indexed by file_id
    meta: dict[str, object]
    extra: dict[str, object]

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def normalization(self) -> dict[str, object]:
        value = self.extra.get("flatmap_normalization")
        return value if isinstance(value, dict) else {}


def load_export(path: Path) -> Export:
    """Read one workbook into labels keyed by ``file_id`` plus metadata.

    Keyed on ``file_id``, never ``neuron_id``: ``neuron_id`` repeats across
    subjects, so joining two exports on it would merge distinct neurons and
    silently corrupt every number downstream.
    """
    clusters = pd.read_excel(path, sheet_name="Clusters")
    for column in ("file_id", "cluster_assignment"):
        if column not in clusters.columns:
            raise ValueError(f"{path.name}: Clusters sheet lacks {column!r}")

    duplicated = clusters["file_id"].duplicated()
    if duplicated.any():
        dupes = sorted(clusters.loc[duplicated, "file_id"].unique())
        raise ValueError(
            f"{path.name}: file_id repeats ({len(dupes)} values, e.g. {dupes[:3]}); "
            "the export is not one row per neuron."
        )

    labels = (
        clusters.set_index("file_id")["cluster_assignment"].astype(int).sort_index()
    )

    meta_frame = pd.read_excel(path, sheet_name="Metadata")
    meta = dict(zip(meta_frame["field"], meta_frame["value"], strict=True))

    extra: dict[str, object] = {}
    raw_extra = meta.get("extra_metadata")
    if isinstance(raw_extra, str) and raw_extra.strip():
        extra = json.loads(raw_extra)

    return Export(path=path, labels=labels, meta=meta, extra=extra)


def is_null_control(export: Export) -> bool:
    """True when the run cannot reach the soma metric at all."""
    return str(export.meta.get("analysis_method")) == NULL_CONTROL_METHOD


def is_identity_style(export: Export) -> bool:
    """True when the style's old per-axis divisors were already equal."""
    return (
        not is_null_control(export)
        and str(export.extra.get("flatmap_style")) == IDENTITY_STYLE
    )


def expects_identical(export: Export) -> bool:
    """True when this pair must come out bit-identical.

    Two independent reasons, both worth keeping distinct in the report: the run
    never uses the metric (voxel), or the metric change is a no-op for that
    style (``both_square``).
    """
    return is_null_control(export) or is_identity_style(export)


def kind(export: Export) -> str:
    """Short tag naming which control a pair is, for the verdict line."""
    return "null control" if is_null_control(export) else "identity control"


def control_reason(export: Export) -> str:
    if is_null_control(export):
        return (
            f"null control: {export.meta.get('analysis_method')} does not use "
            "the soma normalization"
        )
    return (
        f"identity control: {IDENTITY_STYLE} is an exact 2:1 map, so its old "
        "per-axis divisors were already equal"
    )


def _same(left: object, right: object) -> bool:
    """Compare two metadata values, treating NaN as equal to NaN.

    Unset numeric fields arrive as NaN (``dbscan_eps`` on a k-means run), and
    ``nan != nan`` would report every such field as a mismatch.
    """
    left_null = left is None or (isinstance(left, float) and np.isnan(left))
    right_null = right is None or (isinstance(right, float) and np.isnan(right))
    if left_null or right_null:
        return left_null and right_null
    return str(left) == str(right)


def verify_pair(pre: Export, post: Export) -> tuple[list[str], list[str]]:
    """Return ``(failures, notes)`` for one pre/post pair.

    Failures void the comparison.  Notes are observations worth reporting that
    do not invalidate it.
    """
    failures: list[str] = []
    notes: list[str] = []

    pre_ids, post_ids = set(pre.labels.index), set(post.labels.index)
    if pre_ids != post_ids:
        only_pre, only_post = pre_ids - post_ids, post_ids - pre_ids
        failures.append(
            f"neuron sets differ: {len(pre_ids)} pre vs {len(post_ids)} post; "
            f"{len(only_pre)} only-pre, {len(only_post)} only-post "
            f"(e.g. {sorted(only_pre)[:2] or sorted(only_post)[:2]})"
        )
    else:
        notes.append(f"same {len(pre_ids)} neurons (matched on file_id)")

    for field in INVARIANT_FIELDS:
        if not _same(pre.meta.get(field), post.meta.get(field)):
            failures.append(
                f"{field} differs: {pre.meta.get(field)!r} -> {post.meta.get(field)!r}"
            )

    if expects_identical(pre):
        notes.append(f"{control_reason(pre)}, so this pair must be identical")

    if is_null_control(pre):
        # A voxel pair never touches the soma metric, so an unchanged
        # distance_metric is the expected outcome, not a failure.
        for field in EXPECTED_TO_DIFFER:
            if not _same(pre.meta.get(field), post.meta.get(field)):
                failures.append(
                    f"{field} changed on a null control: "
                    f"{pre.meta.get(field)!r} -> {post.meta.get(field)!r}"
                )
    else:
        for field in EXPECTED_TO_DIFFER:
            if _same(pre.meta.get(field), post.meta.get(field)):
                failures.append(
                    f"{field} is unchanged ({pre.meta.get(field)!r}); the post "
                    "run did not use the fixed code path"
                )
            else:
                notes.append(
                    f"{field}: {pre.meta.get(field)} -> {post.meta.get(field)}"
                )

    # The metric change itself: pre must carry per-axis divisors, post must
    # carry exactly one shared divisor equal to the y span.
    pre_norm, post_norm = pre.normalization, post.normalization
    if pre_norm or post_norm:
        if "flatmap_divisor" in pre_norm:
            failures.append("pre export already has flatmap_divisor; it is not pre-fix")
        if {"x_divisor", "y_divisor"} & set(post_norm):
            failures.append("post export still has per-axis divisors; fix not applied")

        x_div, y_div = pre_norm.get("x_divisor"), pre_norm.get("y_divisor")
        shared = post_norm.get("flatmap_divisor")
        if isinstance(x_div, (int, float)) and isinstance(y_div, (int, float)):
            notes.append(
                f"pre divisors x={x_div:.6f} y={y_div:.6f} "
                f"(ratio {x_div / y_div:.4f}, anisotropy "
                f"{abs(1 - x_div / y_div) * 100:.2f}%)"
            )
        if isinstance(shared, (int, float)):
            notes.append(f"post shared divisor {shared:.6f}")
            if isinstance(y_div, (int, float)) and not np.isclose(shared, y_div):
                failures.append(
                    f"post divisor {shared:.6f} is not the pre y span {y_div:.6f}; "
                    "the shared divisor must be the y span"
                )

        for field in ("depth_scale", "include_depth", "axis_count", "style"):
            if not _same(pre_norm.get(field), post_norm.get(field)):
                failures.append(
                    f"normalization.{field} differs: "
                    f"{pre_norm.get(field)!r} -> {post_norm.get(field)!r}"
                )
        for label, norm in (("pre", pre_norm), ("post", post_norm)):
            source = norm.get("bounds_source")
            if source != "canonical":
                failures.append(
                    f"{label} bounds_source is {source!r}, not 'canonical'; "
                    "the runs used different bound sources"
                )

    for label, export in (("pre", pre), ("post", post)):
        style = export.extra.get("flatmap_style")
        if style is not None:
            notes.append(f"{label} style {style}")

    return failures, notes


def changed_fraction(pre: np.ndarray, post: np.ndarray) -> tuple[float, int, int]:
    """Fraction of items whose cluster changed under optimal label matching.

    Cluster ids are arbitrary, so pre cluster 3 and post cluster 1 may be the
    same group.  Hungarian matching on the contingency table finds the label
    correspondence that maximizes agreement, making the leftover disagreement
    the smallest change consistent with the data -- a lower bound on movement,
    which is the honest direction to err for a "how much did this move" number.
    """
    pre_vals, pre_idx = np.unique(pre, return_inverse=True)
    post_vals, post_idx = np.unique(post, return_inverse=True)
    table = np.zeros((len(pre_vals), len(post_vals)), dtype=np.int64)
    np.add.at(table, (pre_idx, post_idx), 1)

    size = max(table.shape)
    square = np.zeros((size, size), dtype=np.int64)
    square[: table.shape[0], : table.shape[1]] = table
    rows, cols = linear_sum_assignment(-square)
    agreed = int(square[rows, cols].sum())
    total = int(pre.size)
    return 1.0 - agreed / total, total - agreed, total


def noise_counts(labels: pd.Series) -> int:
    return int(labels.isin(NOISE_LABELS).sum())


def describe_sizes(labels: pd.Series) -> str:
    sizes = labels.value_counts().sort_index()
    return ", ".join(f"{label}:{count}" for label, count in sizes.items())


def compare(pre: Export, post: Export) -> dict[str, object]:
    """Verify one pair, then measure how far the assignment moved."""
    failures, notes = verify_pair(pre, post)

    print(f"\n{'=' * 74}\n{pre.name}\n  -> {post.name}")
    for note in notes:
        print(f"  ok    {note}")
    for failure in failures:
        print(f"  FAIL  {failure}")

    if failures:
        print("  --> comparison VOID; not reporting a change rate")
        return {"ok": False, "failures": failures}

    shared = pre.labels.index  # verified identical above
    pre_labels = pre.labels.loc[shared].to_numpy()
    post_labels = post.labels.loc[shared].to_numpy()

    fraction, moved, total = changed_fraction(pre_labels, post_labels)
    raw_changed = int((pre_labels != post_labels).sum())
    ari = adjusted_rand_score(pre_labels, post_labels)
    ami = adjusted_mutual_info_score(pre_labels, post_labels)

    print(f"  clusters      pre {pre.labels.nunique()} -> post {post.labels.nunique()}")
    print(f"  sizes pre     {describe_sizes(pre.labels)}")
    print(f"  sizes post    {describe_sizes(post.labels)}")
    pre_noise, post_noise = noise_counts(pre.labels), noise_counts(post.labels)
    if pre_noise or post_noise:
        print(f"  noise/sentinel  pre {pre_noise} -> post {post_noise}")
    print(f"  ARI           {ari:.6f}")
    print(f"  AMI           {ami:.6f}")
    print(
        f"  changed       {moved}/{total} = {fraction * 100:.2f}%  (optimal matching)"
    )
    print(
        f"  (raw label inequality would report "
        f"{raw_changed / total * 100:.2f}% -- inflated by cluster renumbering)"
    )

    control = expects_identical(pre)
    if control:
        if moved == 0 and np.isclose(ari, 1.0):
            print(f"  --> CONTROL HELD: identical, as required ({kind(pre)})")
        elif is_null_control(pre):
            print(
                f"  --> CONTROL BROKEN: {moved} neurons moved in a run that "
                "cannot be affected by the metric; something else differed "
                "between the two launches"
            )
        else:
            print(
                f"  --> CONTROL BROKEN: {moved} neurons moved on "
                f"{IDENTITY_STYLE}, whose divisors were already equal "
                f"({pre.normalization.get('x_divisor')} / "
                f"{pre.normalization.get('y_divisor')}). This is a bug in the "
                "fix, not the fix working."
            )

    return {
        "ok": True,
        "control": control,
        "control_held": bool(control and moved == 0),
        "ari": ari,
        "ami": ami,
        "changed_fraction": fraction,
        "moved": moved,
        "total": total,
        "algorithm": str(pre.meta.get("clustering_algorithm")),
        "k": pre.meta.get("requested_cluster_count"),
        "style": pre.extra.get("flatmap_style"),
        "kind": kind(pre) if control else "",
    }


def find_pairs(directory: Path) -> list[tuple[str, Path, Path]]:
    """Match every ``...pre...`` workbook to its ``post`` counterpart.

    Accepts both ``pre_<tag>.xlsx`` and ``<prefix>_pre_<tag>.xlsx``, so runs can
    be grouped by an arbitrary leading label (``square_pre_...``) without the
    pairing having to know the naming scheme in advance.
    """
    pairs: list[tuple[str, Path, Path]] = []
    for pre_path in sorted(directory.glob("*.xlsx")):
        name = pre_path.name
        if name.startswith("pre_"):
            prefix, rest = "", name[len("pre_") :]
        elif "_pre_" in name:
            head, _, rest = name.partition("_pre_")
            prefix = head + "_"
        else:
            continue

        post_path = pre_path.with_name(f"{prefix}post_{rest}")
        if not post_path.exists():
            print(f"warning: no post export for {name}", file=sys.stderr)
            continue
        tag = (prefix + rest.split("_isocortex")[0]).strip("_")
        pairs.append((tag, pre_path, post_path))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help=(
            "directory holding the pre_*/post_* workbooks "
            f"(default: ./{DEFAULT_EXPORT_DIR}, falling back to cwd)"
        ),
    )
    args = parser.parse_args()

    # Default to the data directory when it exists, but still work when the
    # workbooks sit in the cwd -- that is where napari writes them before they
    # are filed away, so both layouts are worth supporting.
    directory = args.dir
    if directory is None:
        default = Path(DEFAULT_EXPORT_DIR)
        directory = default if default.is_dir() else Path.cwd()

    pairs = find_pairs(directory)
    if not pairs:
        print(f"no pre_/post_ pairs found in {directory}", file=sys.stderr)
        return 1

    results: dict[str, dict[str, object]] = {}
    for tag, pre_path, post_path in pairs:
        results[tag] = compare(load_export(pre_path), load_export(post_path))

    print(f"\n{'=' * 74}\nSUMMARY")
    print(
        f"  {'pair':<20} {'algorithm':<13} {'style':<12} {'k':>3} {'n':>6} "
        f"{'ARI':>8} {'changed':>9}"
    )
    for tag, result in results.items():
        if not result.get("ok"):
            print(f"  {tag:<20} {'VOID -- see failures above':<40}")
            continue
        suffix = ""
        if result.get("control"):
            suffix = f"  <- {result['kind']}" + (
                " HELD" if result.get("control_held") else " BROKEN"
            )
        k = result["k"]
        k_text = "-" if k is None or (isinstance(k, float) and np.isnan(k)) else int(k)
        print(
            f"  {tag:<20} {result['algorithm']:<13} {str(result['style']):<12} "
            f"{k_text!s:>3} {result['total']:>6} {result['ari']:>8.4f} "
            f"{result['changed_fraction'] * 100:>8.2f}%{suffix}"
        )

    voided = [tag for tag, result in results.items() if not result.get("ok")]
    broken = [
        tag
        for tag, result in results.items()
        if result.get("control") and not result.get("control_held")
    ]
    return 1 if voided or broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
