# PROtein — first-principles folding

**PROtein** (PyPI: [`protein-folder`](https://pypi.org/project/protein-folder/)) predicts protein structure from **HQIV** (Horizon-centric Quantum Information and Vacuum) physics: **no machine learning**, **no PDB-derived statistics**, and **no empirical molecular mechanics force fields**. Numerical targets and discrete structures are tied to the formal proofs in [**hqiv-lean**](https://github.com/disregardfiat/hqiv-lean).

!!! warning "Government-use restriction"
    This software is MIT-licensed **with an additional government-use restriction**. See [Government use & license](government-use.md) and [`GOVERNMENT_USE.md`](https://github.com/disregardfiat/protein_folder/blob/main/GOVERNMENT_USE.md) before deployment.

## Highlights

![HQIV-native sparse pi-phase gate map](assets/images/Uu9Hk.jpg)

Image credit: Grok (from the provided picture)

- **Co-translational ribosome tunnel** — null-cone search, lip plane, binary-tree segment scheduling, optional thermal steps ([`lean_ribosome_tunnel_pipeline.py`](https://github.com/disregardfiat/protein_folder/blob/main/src/horizon_physics/proteins/lean_ribosome_tunnel_pipeline.py)).
- **Analytical horizon gradients** — informational energy and CA+bond models ([`folding_energy.py`](https://github.com/disregardfiat/protein_folder/blob/main/src/horizon_physics/proteins/folding_energy.py)).
- **Deterministic L-BFGS** and hierarchical / JAX-accelerated paths where enabled ([`full_protein_minimizer.py`](https://github.com/disregardfiat/protein_folder/blob/main/src/horizon_physics/proteins/full_protein_minimizer.py)).
- **HQIV-native sparse gate + pruning (optional)** — OSHoracle sparse refinement expands sparse support horizon-causally, applies an HQIV-native sparse pi-phase gate evolution, then prunes to only the flipped support before the CA minimization update ([`osh_oracle_folding.py`](https://github.com/disregardfiat/protein_folder/blob/main/src/horizon_physics/proteins/osh_oracle_folding.py), [`pipeline_interchange.py`](https://github.com/disregardfiat/protein_folder/blob/main/src/horizon_physics/proteins/pipeline_interchange.py)).
- **Lean parity tests** — Python scalars match Lean anchors (e.g. long-range hydrogen-bond proxy, collective modes); see `tests/`.

## Quick install

```bash
pip install "protein-folder[full]"
```

Docker and TPU-oriented images are described in [Installation](installation.md).

## Citation

HQIV framework (academic request in LICENSE): [Zenodo record](https://doi.org/10.5281/zenodo.18794889).

Paper / preprint PDF: available in the [hqiv-lean](https://github.com/disregardfiat/hqiv-lean) GitHub repo (DOI pending on hal.science).

## Navigation

Use the tabs above for theory (with links into **hqiv-lean**), API docs, benchmarks, the Streamlit live demo, and contributing guidelines.
