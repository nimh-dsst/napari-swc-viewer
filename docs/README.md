# Documentation

Generate the CPD2 workflow PDF from the repository root:

```bash
pixi run -e docs cpd2-pdf
```

The task converts `docs/cpd2_workflow.md` and writes:

```text
docs/_build/cpd2_workflow.pdf
```

The `docs` Pixi environment contains the PDF dependencies. Pixi installs that
environment on first use, and `docs/_build/` is ignored by git.

## Data Caution: SWC `type` Does Not Reliably Identify The Compartment

**Some neurons in `isocortex_total_right_brainglobe_flatmap.parquet` have
dendritic projections typed `2` (axon).** This was found on 2026-08-05 by
visually inspecting detected termini in napari.

Anything in these documents that filters or groups by node `type` — axon
termini, axon-only projections, compartment-specific heatmaps — inherits this
problem. The plugin reports exactly what the source files say; the source files
are wrong for an unmeasured subset of neurons.

When reading or writing docs here:

- Say **axon-typed**, not **axon**. A count of axon termini is an upper bound
  contaminated by an unknown number of dendrite tips.
- Do not claim node types cleanly separate axonal from dendritic arbors.
- Verify compartment-based results visually before treating them as biology.
- The extent is unquantified — do not state a rate until someone measures one.

Purely topological results (childless tests, branch points, path lengths) are
unaffected as long as they do not filter or interpret by `type`. See `AGENTS.md`
for the full rule and `USE_CASES.md` UC-010 for the workflow where this surfaced.
