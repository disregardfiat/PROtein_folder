# Benchmarks & validation

## In-repo tooling

- **`scripts/run_casp_fold_and_grade.py`** — fetch CASP-style targets, fold, report Cα-RMSD vs experimental PDB when available.
- **`scripts/run_pipeline_grade.py`** — crambin-native vs ab-initio baselines with honest RMSD reporting.
- **`horizon_physics/proteins/grade_folds.py`** — Kabsch Cα-RMSD and PDB CA loading.
- **Grading extras** — with `protein-folder[grading]`, trajectory vs gold metrics live under `grading/` (see package README).

## Interpreting scores

Ab-initio folding from sequence alone on full CASP targets is **hard**; the codebase emphasizes **physical consistency**, **Lean parity**, and **staged validation** (short chains, crambin, tunnel pipelines). Use `--refine-from-gold` in CASP scripts only to validate the minimizer + grader in the sub-Å regime, not as a production ab-initio claim.

## Planned / roadmap

Broader CASP14/15 and CAMEO tables, TM-score / lDDT, and error bars are tracked in [ROADMAP.md](ROADMAP.md) (Phase 5). This page will link to published tables as they are frozen for each release.

## Note on empirical hooks

Benchmark **targets** (experimental PDBs) are used only for **scoring**, not for training potentials. The energy model remains HQIV-derived — see [Theory](theory.md).
