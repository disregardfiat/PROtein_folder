# HQIV Proteins Skill – Agent Guidelines

- **horizon_physics/proteins/** implements protein structure and folding from HQIV first principles only. Use when modifying or extending peptide backbone, alpha helix, beta sheet, side chains, folding energy, or CASP submission.
- **Pure first principles**: Every geometry (bond lengths, angles, φ/ψ, residues/turn, pitch, H-bond distances, χ angles) is derived from the discrete null lattice, causal-horizon monogamy, E_tot = Σ m c² + Σ ħ c / Θ_i, horizon scalar φ = 2 c² / Θ_local, and damping f_φ = −γ′ φ / (a_loc + φ/6)² ∇φ. No empirical data, no hard-coded radii, no PDB statistics, no force fields.
- **Style**: Functions return dicts or tuples with units in Å or degrees; comprehensive docstrings explain HQIV derivation steps; each module has a `if __name__ == "__main__"` block that prints results. MIT license, Python 3.10+, numpy only.
- **Bond lengths**: Cα–C 1.53 Å and C–N 1.33 Å emerge from Θ_local(Z, coordination) with Θ ∝ Z^{-1.1}/coord^{1/3}; ω = 180° from peptide planarity and monogamy.
- **Alpha helix**: 3.6 residues/turn, 5.4 Å pitch, H-bond N–O ~ 2.9 Å from diamond volume balance and f_φ.
- **Beta sheet**: Rise, strand spacing, and H-bond distance from lattice layers and same Θ-based H-bond length.
- **Side chains**: All 20 amino acids; χ and rotamer preferences from φ-shell placement and lattice symmetry (60°, −60°, 180°).
- **Folding**: E_tot minimizer (backbone + side-chain + φ damping); no external force field.
- **Deterministic gradient descent only**: No Monte Carlo, no stochastic methods, no random seeds. Use only gradient-based optimization; L-BFGS (two-loop recursion) in pure numpy. Minima at exact rationals: φ = -57°, ψ = -47°, rise = 3/2 Å, pitch = 27/5 Å. Return fractions via fractions.Fraction where possible (rational_alpha_parameters, rational_ramachandran_alpha).
- **gradient_descent_folding.py**: minimize_e_tot_lbfgs(positions_init, z_list, max_iter, m, gtol); rational_alpha_parameters(), rational_ramachandran_alpha().
- **validation.py**: Pure Python validation (no Jupyter); python -m horizon_physics.proteins.validation runs all checks and prints "Exact match to experiment".
- **secondary_structure_predictor.py**: Deterministic SS from E_tot(φ,ψ) basins; no ML. predict_ss(sequence, window=5) → (ss_string, confidence); preferred basin from Θ_eff per residue (lattice Z_eff); H/E/C labels; predict_ss_with_angles() for dist_to_alpha_min, dist_to_beta_min. Feeds casp_submission for SS-aware Cα placement (helix vs extended vs coil).
- **full_protein_minimizer.py**: Full-chain E_tot minimization. minimize_full_chain(sequence, ca_init=None, ss_string=None, max_iter=500, gtol=1e-5, include_sidechains=False) → dict with ca_min, backbone_atoms, E_ca_final, E_backbone_final, info; L-BFGS over Cα then rebuild N,CA,C,O (and optional Cβ). full_chain_to_pdb(result, chain_id="A") → PDB string.
- **CASP**: hqiv_predict_structure(fasta, chain_id="A", ss_string=None) → CASP-format PDB string; if ss_string is None, predict_ss(sequence) is used for SS-aware backbone placement.
- **Public API**: From horizon_physics.proteins import hqiv_predict_structure, alpha_helix_geometry, backbone_geometry, beta_sheet_geometry, side_chain_chi_preferences, e_tot, minimize_e_tot, minimize_e_tot_lbfgs, rational_alpha_parameters, rational_ramachandran_alpha. See __init__.py and README.md.
