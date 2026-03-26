# How it works

## Pipelines

1. **Sequence → secondary-structure hints** — basins for φ/ψ and related scalars (`secondary_structure_predictor.py`, `theta_eff_residue`).
2. **Backbone placement** — HQIV backbone geometry (bond lengths, ω) and CA trace initialization (`peptide_backbone.py`, `casp_submission.py`).
3. **Energy minimization** — informational CA energy with optional sequential bond penalties and sidechain packing (`folding_energy.py`, `full_protein_minimizer.py`).
4. **Co-translational path** — ribosome tunnel masking, segment scheduling, extrusion, then post-tunnel refinement (`co_translational_tunnel.py`, `lean_ribosome_tunnel_pipeline.py`).
5. **Hierarchical / JAX** — optional tree-structured minimization for large chains (`hierarchical/`).
6. **Assembly** — multi-chain docking after per-chain folds (`assembly_dock.py`).
7. **OSHoracle sparse refinement (optional)** — sparse support is expanded horizon-causally (`i -> i and i+1`), gate-evolved in a dense reconstruction, then only changed-support residues are updated in CA minimization (`osh_oracle_folding.py`, `pipeline_interchange.py`).

## Lean parity

Tests under `tests/test_*lean*` and `tests/test_collective_*.py` assert numeric agreement with Lean-exported anchors (e.g. `phi_of_shell`, hydrogen-bond proxy coefficients). This is the **regression bridge** between **hqiv-lean** and this engine.

## Key source files

| File | Role |
|------|------|
| [`full_protein_minimizer.py`](https://github.com/disregardfiat/protein_folder/blob/main/src/horizon_physics/proteins/full_protein_minimizer.py) | Main chain minimizer, PDB export |
| [`lean_ribosome_tunnel_pipeline.py`](https://github.com/disregardfiat/protein_folder/blob/main/src/horizon_physics/proteins/lean_ribosome_tunnel_pipeline.py) | Lean-aligned tunnel + refinement |
| [`folding_energy.py`](https://github.com/disregardfiat/protein_folder/blob/main/src/horizon_physics/proteins/folding_energy.py) | Horizon informational energy, gradients |
| [`validation.ipynb`](https://github.com/disregardfiat/protein_folder/blob/main/src/horizon_physics/proteins/validation.ipynb) | Notebook checks vs experimental backbone targets |

## HTTP server (CASP-style)

`casp_server.py` at the repository root exposes a Flask app (typically via gunicorn). Install `protein-folder[demo]` and run as documented in the module docstring. The **Streamlit** UI is a lighter local alternative — see [Live demo](live-demo.md).
