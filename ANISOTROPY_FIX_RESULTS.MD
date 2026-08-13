# Removing X/Y Anisotropy from the Flat Map Soma-Clustering Metric

**Validation report — `napari-swc-viewer`, branch `debug/smoosh`, 2026-08-13**

## Abstract

The flat map soma-clustering metric in `napari-swc-viewer` normalized its two
tangential axes by separate divisors: `x_flat` by half the canonical `x` span
and `y_flat` by the full `y` span. For the `both_shaped` flat map style these
divisors were 0.904233 and 0.943757, a ratio of 0.9581 — a **4.19% anisotropy**
that inflated `x` separations relative to `y` and made equal distances on the
cortical sheet unequal in the metric. We replaced both divisors with a single
shared divisor equal to the `y` span, making the metric isotropic and aligning
it with the policy the voxel grid already followed.

Because no ground truth exists for correct cluster assignments, we validated the
change by invariance and by controls rather than by accuracy. Twelve paired
clustering exports were produced from napari before and after the fix and
compared with permutation-invariant measures. Over all 18,518 somas in
`isocortex_total_right_brainglobe_flatmap.parquet` using hierarchical/ward
clustering, the shaped style moved **37.5%** of somas at k=5 and **38.7%** at
k=10 (adjusted Rand index 0.44 and 0.47). Two independent controls held exactly:
the `both_square` style, whose prior divisors were already equal, moved **0.0%**
(ARI 1.000000) under the most perturbation-sensitive settings tested, and a
voxel-correlation pair, which never reaches the soma normalization, likewise
moved 0.0%. Together these confirm the change is real, correctly scoped to the
styles it should affect, and free of incidental side effects.

We further found the change rate to be strongly dependent on algorithm and
dataset size — ward over an 833-soma subset moves 1.9% where ward at full scale
moves 37.5% — and corrected a documented figure that had been recorded without
those qualifiers.

## 1. Introduction

The bilateral flat map lays the two cortical hemispheres side by side along `x`,
so `x` spans roughly twice the extent of `y`. This asymmetry is a known hazard in
this codebase: giving both axes the same treatment squashes the map, and a prior
change had already corrected it on the voxel-grid side by deriving the `x` bin
count from the measured aspect ratio so that each bin is as wide as it is tall.

The soma-clustering metric was left out of that correction and tracked
separately. It normalized coordinates as

```
x_scaled = x_flat / (x_span / 2)
y_scaled = y_flat / y_span
d_scaled = depth_um / depth_span
```

The intent of halving the `x` span was to make one hemisphere a unit cube, on the
reasoning that `x` covers two hemispheres. For `both_square`, an exact 2:1 map,
this yields divisors of 1.0 and 1.0 and is harmless. For `both_shaped`, whose
canonical spans are 1.8085 and 0.9434, it yields 0.904233 and 0.943757 — unequal,
so an `x` displacement and an equal `y` displacement produced different distances.
The distortion was small but systematic and applied to every pairwise distance in
the clustering.

Two properties of the problem shaped the validation strategy:

1. **There is no ground truth.** The flat map projection distorts the cortical
   surface, and flat map `x`/`y` have no reliable conversion to microns. No
   external reference says which partition of somas is correct, so accuracy
   cannot be scored. Isotropy is adopted as a *prior* — with the true surface
   geometry unknowable, adding no distortion is the conservative choice — and
   the validation can only confirm that the implementation realizes that prior.

2. **The codebase supplies its own control.** Because `both_square` is an exact
   2:1 map, its prior divisors were already equal. Sharing one divisor is
   therefore arithmetically a no-op for that style, while still exercising the
   entire modified code path. Any movement in square-style results indicates a
   defect introduced by the fix rather than the fix working.

This report records what was changed, how it was validated, and what the
measurements support.

## 2. Methods

### 2.1 Implementation

`FlatmapDepthNormalization` in
`src/napari_swc_viewer/analysis/flatmap_correlation.py` was reduced from two
tangential divisors (`x_divisor`, `y_divisor`) to a single `flatmap_divisor`
taken from the `y` span. `y` is the reference axis because it spans one
hemisphere while `x` spans two, so the `x` extent is not a per-hemisphere
quantity and must not set the unit. `depth_um` retains a separate divisor
because it is a physically different quantity in different units, weighted
against tangential distance by the user-facing **Depth scale** control.

Collapsing to one field was deliberate: two numbers required to stay equal can
drift apart through a single edit, whereas one number cannot. The change also
removed a branch that had halved canonical `x` bounds while leaving observed
bounds unhalved; taking the unit from `y` makes that distinction unnecessary,
since `y` spans one hemisphere under either bounds source.

The `distance_metric` string recorded in exports was renamed from
`euclidean_flatmap_depth_unit_hemisphere` to
`euclidean_flatmap_isotropic_plus_depth` (and the depth-excluded variant from
`euclidean_flatmap_xy_unit_hemisphere` to `euclidean_flatmap_isotropic`), so
results produced before and after the change cannot be silently compared as if
they matched. This rename also served as a verification instrument: it is how we
confirmed each "post" export had actually executed the modified code.

### 2.2 Isolating the change

The metric change existed only as uncommitted working-tree modifications, while
the earlier voxel-grid correction was already committed. We confirmed by hunk
inspection that the committed work touched neither
`resolve_flatmap_depth_normalization`, `normalize_flatmap_soma_coordinates`,
`FlatmapDepthNormalization`, nor `FlatmapSomaClusterWorker`. Stashing only the
uncommitted changes therefore reverts the metric while leaving the grid fix in
place, making the divisor policy the sole variable between paired runs. The
package is installed editable, so each stash toggle takes effect on relaunch.

### 2.3 Export generation

Clustering was run in napari before and after the fix and exported to XLSX.
Twelve exports were produced in six pairs, covering three algorithms
(hierarchical/ward, k-means, DBSCAN), two flat map styles, two cluster counts,
and two dataset scales. All runs used **Depth scale** `1.00` with depth
included, no region filtering, and
`isocortex_total_right_brainglobe_flatmap.parquet` as the source.

### 2.4 Analysis

`scripts/compare_clustering_exports.py` reads each pair and reports only after a
comparability gate passes.

**Comparability gate.** Every pair is verified for identical neuron sets and
identical run parameters — `analysis_method`, `clustering_algorithm`,
`clustering_linkage`, `requested_cluster_count`, DBSCAN `eps`/`min_samples`,
region selection, dilation, atlas, and source Parquet path — plus identical
normalization inputs (`depth_scale`, `include_depth`, `axis_count`, `style`).
`bounds_source` is required to be `canonical` on both sides, so that neither run
silently fell back to observed bounds. Soma pairs must show the `distance_metric`
rename; a pair that does not is rejected as not having run the fixed code. Any
failure voids the pair and suppresses its numbers.

Neurons are matched on `file_id`. This is essential rather than incidental:
`neuron_id` repeats across subjects in this dataset, so joining on it would
merge distinct neurons and corrupt every downstream figure.

**Measures.** Cluster identifiers are arbitrary, so we report the adjusted Rand
index and adjusted mutual information, both invariant to relabeling, together
with a change fraction computed after optimally matching pre-fix clusters to
post-fix clusters by Hungarian assignment on the contingency table. The matched
fraction is the smallest movement consistent with the data, which is the
conservative direction for a "how much did this move" statistic.

The necessity of matching is quantitative, not stylistic: for the shaped k=10
pair, a raw comparison of `cluster_assignment` values reports 99.64% changed
against a true movement of 38.70%, because the post-fix run renumbered nearly
every cluster. Naive label diffing overstates movement by more than two orders
of magnitude in the k-means case (99.88% against 0.36%).

**Independent replication.** All figures were also computed directly from the
Parquet file by applying both normalizations to the same soma coordinates
outside the napari application, to distinguish genuine metric effects from
export-layer artifacts.

## 3. Results

### 3.1 The recorded provenance confirms the intended change

Export metadata showed the pre-fix runs carrying `x_divisor = 0.904233` and
`y_divisor = 0.943757` for `both_shaped` — a ratio of 0.9581, i.e. 4.19%
anisotropy — and the post-fix runs carrying a single
`flatmap_divisor = 0.943757` equal to the prior `y` span, with no per-axis
fields present. For `both_square` the pre-fix divisors were 1.0 and 1.0, exactly
as an exact 2:1 map predicts. All runs recorded `bounds_source: canonical`,
`depth_scale: 1.0`, `include_depth: true`, and `axis_count: 3`.

### 3.2 Primary measurements

All pairs passed the comparability gate.

| pair | algorithm | style | k | n | ARI | AMI | moved |
|---|---|---|---|---|---|---|---|
| shaped_soma_ward | ward | `both_shaped` | 10 | 18,518 | 0.4743 | 0.6315 | **38.70%** |
| square_soma_ward | ward | `both_square` | 10 | 18,518 | 1.0000 | 1.0000 | **0.00%** |
| soma_ward | ward | `both_shaped` | 5 | 833 | 0.9541 | 0.9583 | 1.92% |
| soma_kmeans | k-means | `both_shaped` | 5 | 833 | 0.9927 | 0.9874 | 0.36% |
| soma_dbscan | DBSCAN | `both_shaped` | — | 833 | 0.8914 | 0.8104 | 0.36% |
| voxel | ward | `both_shaped` | 5 | 835 | 1.0000 | 1.0000 | **0.00%** |

Direct computation from the Parquet file, independent of the application:

| algorithm | style | k | n | ARI | moved |
|---|---|---|---|---|---|
| ward | `both_shaped` | 5 | 18,518 | 0.4363 | 37.51% |
| ward | `both_shaped` | 10 | 18,518 | 0.4743 | 38.70% |
| k-means | `both_shaped` | 5 | 18,518 | 0.9681 | 1.40% |
| k-means | `both_shaped` | 10 | 18,518 | 0.9645 | 1.85% |

The independent computation reproduced the application's exports exactly where
they overlap — 38.70% and ARI 0.4743 for shaped ward at k=10, and 0.36% with ARI
0.9927 for the 833-soma k-means pair — establishing that the measured movement
originates in the metric and not in the export path.

### 3.3 Both controls held

The **identity control** is the strongest result in this validation. The
`both_square` pair was run under the most perturbation-sensitive configuration
available: the full dataset, ward linkage, and k=10 — precisely the
configuration in which the shaped style moves 38.70%. Ward is agglomerative, so
any change to the distance computation flips near-tied merges and the reordering
cascades; at 18,518 points the opportunity for such flips is maximal. Under
those conditions the square style produced **zero** movement, with all ten
cluster sizes identical member-for-member (689, 1123, 2047, 690, 1637, 1603,
1744, 2458, 5211, 1316) and ARI 1.000000, while its `distance_metric` confirmed
the new code had run.

This control is informative because square and shaped share the entire modified
code path and differ only in their divisor values. A transpose, an axis
confusion, an off-by-one, or any incidental change to the distance computation
would have moved square as well. None did. The result therefore scopes the
change precisely to the anisotropy it targeted, and retroactively supports
attributing the shaped movement to removal of that anisotropy rather than to
unrelated code churn.

The **null control** — a voxel-correlation pair, which clusters on
`1 - pearson_r` over voxelized node densities and never consults the soma
normalization — also produced zero movement (ARI 1.000000, 0/835). This rules
out confounds between the two application launches, such as a differing neuron
set, altered filter state, or stale cache, any of which would have invalidated
the soma comparisons.

Taken together the two controls constrain the result through different
mechanisms: one where the metric is never reached, and one where the metric
change is arithmetically a no-op.

### 3.4 The change rate depends strongly on algorithm and scale

The same fix produces movement spanning two orders of magnitude depending on how
it is measured, at k=5:

| algorithm | 833 somas | 18,518 somas |
|---|---|---|
| ward | 1.92% | 37.51% |
| k-means | 0.36% | 1.40% |

Ward at full scale moves roughly 20 times more than ward on the subset. The
mechanism is ward's agglomerative construction: a small metric perturbation
changes which of several near-tied merges occurs first, and that choice
propagates through all subsequent merges. A larger dataset presents more
near-ties and so more opportunity for divergence. Seeded k-means, which
re-optimizes toward centroids from a fixed initialization, is far more stable.

The practical consequence is that a change rate measured on a partial table does
not bound the full-dataset rate, and a small rate from a subset is not evidence
that the metric barely moved.

### 3.5 A documented figure was incorrect

Repository documentation recorded "~40% of somas land in a different cluster at
k=5 (~25% at k=10)". The k=5 figure is confirmed at 37.5%. The k=10 figure is
not: both the export and the independent computation give 38.7%. The documented
value was low by roughly 14 points and, more consequentially, reversed the trend
— it showed movement falling from k=5 to k=10 where it in fact rises slightly.
A reader reasoning from the original text would have expected the metric change
to matter less at finer partitions. Neither figure had recorded the algorithm or
the neuron count on which it was measured.

`AGENTS.md` and `USE_CASES.md` have been corrected, and both now carry the
algorithm, the dataset size, and the depth scale alongside the rates, plus an
explicit warning against citing a subset rate as a full-dataset one.

### 3.6 Secondary observation

Under the pre-fix metric, shaped ward at k=10 placed 5,616 somas — 30% of the
dataset — in a single cluster, while the largest post-fix cluster holds 3,391
(18%), with sizes distributed more evenly. Removing the anisotropy broke up a
dominant cluster rather than only shifting boundaries. This is consistent with
the distortion having been a genuine artifact, but it is a plausibility signal
only: no ground truth establishes which partition is biologically correct.

## 4. Conclusions

1. **The fix is implemented correctly and is live in the application.** Export
   provenance shows the single shared divisor equal to the `y` span, the removal
   of per-axis fields, and the renamed metric identifier in every post-fix run.

2. **Its effect is precisely scoped.** The square style, whose prior divisors
   were already equal, is bit-identical under the most sensitive configuration
   tested, while the shaped style — the only style that carried anisotropy —
   moves substantially. A voxel-correlation control confirms nothing else
   differed between runs.

3. **The magnitude is large for shaped-style ward clustering**: 37.5% of somas
   at k=5 and 38.7% at k=10 over the full dataset. Results produced before the
   change are not comparable to results produced after it, and the renamed
   `distance_metric` prevents the two from being conflated.

4. **Change rates must be quoted with their algorithm and dataset size.** The
   same fix registers as 0.36% or 37.5% depending on those choices. This is the
   most transferable methodological finding here and the source of the incorrect
   documented figure.

5. **This validation establishes conformance, not biological correctness.** It
   demonstrates that the metric is isotropic and that the change affects only
   what it should. It does not and cannot establish that isotropy is the
   biologically right choice; that rests on the prior that flat map `x`/`y` admit
   no reliable micron conversion, making zero added distortion the conservative
   option. Results in this area should be described accordingly.

### Limitations

Manual verification covered the metric change and the export metadata (UC-012
step 9). The remaining UC-012 steps — tooltip wording, eps unit switching,
DBSCAN boundary behavior, and the voxel-correlation steps — were not exercised
and are recorded as outstanding. Cluster-shape isotropy was not assessed
visually; a 4% distortion is below reliable visual detection, which is why the
validation rests on controls and invariants instead. The `both_square` style was
tested at k=10 only, though its identity result is exact and analytically
expected at any k. Automated coverage of the divisor arithmetic, isotropy as a
distance property, agreement with the voxel grid's reference axis, depth
exclusion, monotonic reweighting, the observed-bounds fallback, and provenance
lives in `tests/test_flatmap_depth_normalization.py`; the full suite passes at
1,041 tests.

### Data availability

The twelve export workbooks behind this report are **not stored in the
repository**. They live in `anisotropy_fix_data/` in the working tree, which is
listed in `.gitignore` along with `*.xlsx`, by deliberate choice: this project
does not keep data files in version control, important or otherwise. Anyone
cloning the repository gets this report and the analysis script but not the
workbooks, and must regenerate them to re-derive the numbers.

To regenerate: run the clustering in napari with the settings recorded in §2.3
and export each result, once with the fix applied and once with it reverted (see
§2.2 for the isolation procedure), naming the files as below. The source Parquet
is `isocortex_total_right_brainglobe_flatmap.parquet`, also untracked.

### Reproducing this analysis

```
python scripts/compare_clustering_exports.py --dir anisotropy_fix_data
```

Exports must be named `<prefix>_pre_<tag>.xlsx` and `<prefix>_post_<tag>.xlsx`,
or `pre_<tag>.xlsx` and `post_<tag>.xlsx`. The script auto-detects voxel and
square-style pairs as controls and applies the appropriate expectation to each,
and it refuses to report a change rate for any pair that fails the comparability
gate in §2.4 — so a regenerated set that differs in neuron scope or parameters
will be rejected rather than silently producing different numbers.

## AI Usage Statement

This analysis, the comparison script
(`scripts/compare_clustering_exports.py`), the documentation corrections to
`AGENTS.md` and `USE_CASES.md`, and this report were produced by **Claude Opus 5
(1M context)** — model ID `claude-opus-5[1m]` — running in Claude Code, with
extended thinking enabled, at the prompting of **@joshlawrimore**.

The human author designed and executed the validation strategy, generated all
twelve paired clustering exports from napari across the algorithms, styles,
cluster counts, and dataset scales reported here, and directed the sequence of
controls —
including the voxel-correlation pair and the decisive full-scale `both_square`
identity control that anchors the conclusions.

The model wrote the analysis code, computed the reported statistics, performed
the independent replication from the Parquet file, and drafted the text. One
model-authored intermediate conclusion was wrong and was corrected against the
data: on seeing 1.92% movement in an 833-soma subset, the model initially judged
the documented ~40% figure to be overstated, before full-scale measurement showed
the documented k=5 value was correct and the discrepancy was one of scale. The
k=10 correction reported in §3.5 survived that scrutiny and independent
replication. All numerical claims in this report derive from the exports and the
Parquet file rather than from model recall, and readers should verify them by
rerunning the script above.
