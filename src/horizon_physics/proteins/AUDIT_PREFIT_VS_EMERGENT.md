# Audit: Prefit vs emergent parameters

Parameters that are **emergent** are derived in code from HQIV (lattice, Θ, φ, monogamy). **Prefit** (or convention) values are hard-coded or chosen without an in-repo derivation.

---

## Emergent (derived in code)

| Location | Parameter | How it emerges |
|----------|-----------|----------------|
| `_hqiv_base` | Θ_C, Θ_N, r_Cα–C, r_C–N | `theta_local(Z, coord)`, `bond_length_from_theta`; theta0 anchors scale so Θ_C(6,2)=1.53 |
| `peptide_backbone` | Calpha_C, C_N, N_Calpha | From `theta_for_atom` + `bond_length_from_theta` |
| `peptide_backbone` | φ_α, ψ_α = (-57°, -47°), φ_β, ψ_β = (-120°, 120°) | Stated as HQIV landscape minima; used as fixed rationals |
| `alpha_helix` | rise = 3/2 Å, pitch = 27/5 Å, 18/5 res/turn | Rationals from HQIV; `RISE_ANG`, `PITCH_ANG` |
| `alpha_helix` | turn_angle_deg | 360 / RESIDUES_PER_TURN |
| `folding_energy` | R_CA_CA_EQ = 3.8 | Comment: compromise of extended (3.2) and helix (5.4); 3.2/5.4 from other modules |
| `casp_submission` | rise_ext = 3.2 | From `beta_sheet_geometry` |
| Universal | ħc | `HBAR_C_EV_ANG`; physical constant |
| Universal | φ = 2/Θ | `horizon_scalar(theta_ang)` |

---

## Prefit or convention (not derived in this repo)

### 1. Single length-scale / anchor

| Location | Parameter | Note |
|----------|-----------|------|
| `_hqiv_base` | `theta0 = 1.53 * (6**alpha) * (2**(1/3))` | 1.53 Å sets lattice scale so that Θ_C = 1.53. Single reference length. |
| `_hqiv_base` | `alpha = 0.91` | Exponent in Θ ∝ Z^{-α}; stated to yield r_C–N/r_Cα–C; not derived here. |

### 2. Backbone and carbonyl

| Location | Parameter | Note |
|----------|-----------|------|
| `peptide_backbone` | `r_co_carbonyl = 1.23` | Comment: "Θ_O(coord=1) scaled by π resonance"; no formula in code. |
| `peptide_backbone` (ramachandran_map) | `theta_alpha = 2.5`, `theta_beta = 2.2` | Basin depths for E_tot map. |
| `peptide_backbone` (ramachandran_map) | `30`, `40` (in Gaussian widths) | Basin widths in (φ,ψ). |
| `peptide_backbone` (ramachandran_map) | `0.1 * exp(-(...)/180**2)` | Steric/clash term coefficient. |

### 3. Alpha helix

| Location | Parameter | Note |
|----------|-----------|------|
| `alpha_helix` | `hbond_no_ang = 2.90` | Overwrites `r_no` from `bond_length_from_theta`; stated "Θ_N + Θ_O overlap and f_φ". |
| `alpha_helix` | `radius_ang = 2.3` | Overwrites geometric radius; "from diamond stacking". |

### 4. Beta sheet

| Location | Parameter | Note |
|----------|-----------|------|
| `beta_sheet` | `rise_ang = 3.2` | Comment: "backbone step in extended conformation"; not computed from (φ,ψ). |
| `beta_sheet` | `strand_spacing_ang = 4.7` | Comment: "two strands share diamond layer"; not derived. |
| `beta_sheet` | `hbond_no = 2.90` | Same as alpha; hard-coded. |
| `beta_sheet` | `pleat_repeat_ang = 6.4` | Hard-coded. |

### 5. Folding energy and horizon

| Location | Parameter | Note |
|----------|-----------|------|
| `folding_energy` | `r_ref = 2.0` | Local crowding scale in theta_at_position; not derived. |
| `folding_energy` | `R_HORIZON = 15.0`, `CUTOFF = 12.0` | Horizon radius and neighbor-list cutoff; not from screening formula. |
| `folding_energy` | `k_horizon = 0.5 * HBAR_C_EV_ANG` | Horizon force prefactor. |
| `folding_energy` | `R_BOND_MIN = 2.5`, `R_BOND_MAX = 6.0` | Cα–Cα bounds; 2.5/6.0 not derived from Θ. |
| `folding_energy` | `R_CLASH = 2.0` | Non-bonded clash distance. |
| `folding_energy` | `K_BOND = 200 * HBAR_C`, `K_CLASH = 500 * HBAR_C` | Penalty strengths; tuning. |
| `folding_energy` | `lambda_damp = 0.1 * HBAR_C` | Damping weight in E_tot. |
| `folding_energy` | `0.1` in soft bond term | Factor in (r - r_eq)² for r in [r_min, r_max]. |

### 6. _hqiv_base

| Location | Parameter | Note |
|----------|-----------|------|
| `_hqiv_base` | `A_LOC_ANG = 1.0` | Reference scale in damping denominator. |
| `_hqiv_base` | `GAMMA_PRIME = 0.4` | Comment: "consistent with HQIV γ ≈ 0.40"; external reference. |

### 7. Secondary structure

| Location | Parameter | Note |
|----------|-----------|------|
| `secondary_structure_predictor` | `_THETA_ALPHA_BETA_CROSSOVER = 1.35` | Comment: "E_alpha(Θ) = E_beta(Θ)"; value not derived in code. |
| `secondary_structure_predictor` | `scale = max(np.ptp(theta_arr), 0.3)` | 0.3 clamp for confidence. |
| `secondary_structure_predictor` | coil confidence `0.5` | Arbitrary. |

### 8. CASP and minimizer

| Location | Parameter | Note |
|----------|-----------|------|
| `casp_submission` | `rise_coil = 3.0` | Coil segment rise; not from beta_sheet. |
| `full_protein_minimizer` | `step = 0.5 / (g_norm + 1e-6)` | Adaptive step scaling. |
| `full_protein_minimizer` | `k_clash = 500.0` in _pack_lbfgs | Same as K_CLASH; duplicated. |
| `full_protein_minimizer` | `r_clash = 2.0` (default) | Pass-through. |

---

## Summary

- **Emergent:** Bond lengths from Θ (except C=O 1.23), rational helix rise/pitch, φ/ψ minima as fixed rationals, horizon scalar φ = 2/Θ, and use of 3.2/5.4 in the 3.8 compromise.
- **Single scale / anchor:** 1.53 Å (and α = 0.91) in `theta0`; everything else in Å follows from that.
- **Prefit / convention:** C=O 1.23; H-bond 2.9; helix radius 2.3; beta rise 3.2, spacing 4.7, pleat 6.4; r_ref 2.0; R_HORIZON 15, CUTOFF 12; R_BOND_MIN/MAX 2.5/6.0; R_CLASH 2.0; K_BOND 200, K_CLASH 500; lambda_damp 0.1; A_LOC, GAMMA_PRIME; SS crossover 1.35; coil rise 3.0; Ramachandran map constants (theta_alpha/beta, widths, 0.1); adaptive step 0.5; confidence scale 0.3 and coil 0.5.

---

## Recommendations

1. **Document** the single length-scale (1.53 Å) and α = 0.91 in the paper/README as the only “input” scale/exponent; treat everything else as either derived from Θ/φ or as convention/tuning.
2. **Derive where feasible:** e.g. r_co_carbonyl from Θ_O and a stated π-resonance factor; H-bond 2.9 from bond_length_from_theta(theta_N, theta_O) and a short comment if it’s already correct; beta rise from (φ_β, ψ_β) and backbone geometry.
3. **Centralize** bond/clash constants (R_BOND_MIN/MAX, R_CLASH, K_BOND, K_CLASH) and horizon (R_HORIZON, CUTOFF, k_horizon) in one place (e.g. `_hqiv_base` or `folding_energy`) and add one-line comments: “convention” vs “from screening/balance” if derived later.
4. **SS crossover:** Either derive 1.35 from a stated E_alpha(Θ)=E_beta(Θ) condition in code, or label it as a chosen crossover in docs.
