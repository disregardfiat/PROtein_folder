# PROtein — HQIV protein folding

First-principles protein structure from **Horizon-centric Quantum Information and Vacuum (HQIV)**. **No ML**, **no PDB statistics**, **no empirical force fields**. Formal definitions and proofs live in **[hqiv-lean](https://github.com/disregardfiat/hqiv-lean)**.

**License:** MIT **with an explicit government-use restriction** — see [`LICENSE`](LICENSE) and [`GOVERNMENT_USE.md`](GOVERNMENT_USE.md). PyPI metadata includes the same notice.

## Documentation

**[https://disregardfiat.github.io/protein_folder/](https://disregardfiat.github.io/protein_folder/)** — theory (with Lean links), installation, API (mkdocstrings), benchmarks, contributing, live demo.

Build locally: `pip install -e ".[dev]"` then `mkdocs serve`.

## Install

```bash
pip install "protein-folder[full]"
```

Extras: `tpu` (JAX), `grading` (pandas), `demo` (Streamlit + Flask stack). Editable clone:

```bash
git clone https://github.com/disregardfiat/protein_folder.git && cd protein_folder
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Docker: `docker build -t protein-folder:cpu .` — see [`Dockerfile`](Dockerfile) and [`Dockerfile.tpu`](Dockerfile.tpu).

## Quick API

```python
from horizon_physics.proteins import minimize_full_chain, full_chain_to_pdb

result = minimize_full_chain("MKFLNDR", include_sidechains=True)
print(full_chain_to_pdb(result))
```

Package layout: **`src/horizon_physics/`** (import name unchanged). Tests: **`tests/`**. CASP-style server: **`casp_server.py`** at repo root.

## Live demo

```bash
pip install "protein-folder[demo]"
protein-folder-streamlit
```

## Citation

HQIV framework (academic request in LICENSE): [Zenodo](https://doi.org/10.5281/zenodo.18794889).

## Repository layout (high level)

| Path | Role |
|------|------|
| `src/horizon_physics/proteins/` | Folding pipeline, energy, tunnel, CASP hooks |
| `tests/` | Pytest + Hypothesis |
| `docs/` | MkDocs site sources |
| `scripts/` | CASP grading / benchmark drivers |

Details: [documentation site](https://disregardfiat.github.io/protein_folder/) and [`src/horizon_physics/proteins/README.md`](src/horizon_physics/proteins/README.md).
