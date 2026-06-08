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
