# Theory & HQIV physics

HQIV treats informational structure on a discrete light cone and enforces **informational monogamy** across shells. Protein folding in this codebase is implemented as **geometric + informational energy minimization** on CA (and optional sidechain) degrees of freedom, with parameters that are intended to align with the Lean formalization rather than with fitted empirical potentials.

## Formal proofs (Lean 4)

The canonical proof repository is **[hqiv-lean](https://github.com/disregardfiat/hqiv-lean)**. When reading the code here, use Lean as the ground truth for definitions of:

- Discrete horizon / shell structure and admissible modes
- Long-range hydrogen-bond proxy scalars (`HQIVLongRange` — see parity tests in `tests/test_hqiv_long_range_lean_parity.py`)
- Collective mode / kink budgets (`HQIVCollectiveModes` — see `tests/test_collective_modes_scalars.py` and `tests/test_collective_kink_ca.py`)

!!! note "Linking theorems to code"
    The Python modules `hqiv_long_range.py`, `collective_modes_scalars.py`, and `folding_energy.py` are written to mirror those definitions numerically. When a symbol changes in Lean, the **parity tests** are the first line of regression detection.

## Representative energy terms

Informational totals combine site and pair contributions derived from horizon poles; a compact schematic:

\\[
E \\sim \\sum_i \\varepsilon_{\\mathrm{site}}(i) + \\sum_{(i,j) \\in \\mathcal{P}} \\varepsilon_{\\mathrm{pair}}(i,j)
\\]

where \\(\\mathcal{P}\\) is built from geometric neighbor structure (see `build_horizon_poles` in [`folding_energy.py`](https://github.com/disregardfiat/protein_folder/blob/main/src/horizon_physics/proteins/folding_energy.py)). Exact discrete formulas are in code and Lean, not in this summary.

## Parameter sensitivity

- **Neighbor cutoff** — widens or narrows pair lists; larger cutoffs increase cost and can change fine balance between local and horizon-mediated terms.
- **Shell indices / atomic numbers** — enter pole strengths and screening; changing them moves the model away from HQIV-aligned values.
- **Tunnel geometry** — cone half-angle and tunnel length set the co-translational feasible set; too tight a cone can over-constrain early residues.

Edge cases: very short peptides (\\(n < 3\\)) have degenerate geometry; minimizers may exit early. Multi-chain jobs require assembly / docking stages documented under [How it works](how-it-works.md).

## Further reading

- Paper / preprint materials under `src/horizon_physics/proteins/paper/` in the repository.
- Zenodo: [HQIV framework](https://doi.org/10.5281/zenodo.18794889).
