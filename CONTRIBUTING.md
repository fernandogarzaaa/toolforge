# Contributing to toolforge

The bar for changes is the one the project holds itself to: every number in the README is measured,
and anything not run is labelled untested.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"            # core + test tooling (no torch)
pip install -e ".[train,export]"   # only for the smoke test and real runs
```

## Before opening a PR

```bash
make lint        # ruff check + ruff format --check
make typecheck   # mypy
make test        # fast unit tests, no torch required
make smoke       # tiny random-model end-to-end run (needs [train])
make mutation    # optional: testteeth mutation grade of the core modules
```

- Core logic (scorer, schema validation, data cleaning, pair building, prompt formatting) is pure
  Python and must stay importable without torch. Heavy imports go inside functions.
- New scorer or validation behaviour needs a spec-driven test: assert the exact verdict and reason,
  cover the boundary case and the failure path. `make mutation` shows what a test misses.
- Prompt-format changes must keep `tests/test_template_parity.py` green against the real template:
  `TOOLFORGE_PARITY_MODEL=Qwen/Qwen2.5-0.5B-Instruct pytest -m parity`.
- Results tables are regenerated with `toolforge eval` (`results/RESULTS.md`); always state the run size.

## Releases

Bump `version` in `pyproject.toml` and `src/toolforge/__init__.py`, update `CHANGELOG.md`, tag
`vX.Y.Z`. The release workflow builds the sdist and wheel, attaches them to a GitHub Release and
uploads to PyPI only if the `PYPI_API_TOKEN` secret is configured.
