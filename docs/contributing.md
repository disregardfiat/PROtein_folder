# Contributing & development

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Tests

```bash
pytest tests/
```

Optional TPU/JAX validation: `./run_tests_tpu.sh` (expects a venv with JAX installed).

## Code quality

- **Ruff** — lint (and format if you enable `ruff format` locally).
- **mypy** — `mypy` is configured in `pyproject.toml` for `horizon_physics.proteins` (third-party stubs may require `ignore_missing_imports`).

CI runs pytest on push (see `.github/workflows/ci.yml`).

## Layout

- **`src/horizon_physics/`** — installable package (src layout).
- **`tests/`** — pytest suite.
- **`docs/`** — MkDocs sources (+ this site).
- **`casp_server.py`** — deployment entrypoint at repo root.

## Pull requests

Keep changes **physics-first**: no empirical force-field terms, no PDB-derived potentials, no ML estimators smuggled into the core energy. Benchmarks and optional grading may consume external **structures for scoring only**.

## License

Contributions are accepted under the **MIT + government-use restriction** in `LICENSE`. Do not remove or weaken the restriction without explicit maintainer approval.
