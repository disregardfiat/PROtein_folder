# HQIV Proteins Module

First-principles protein structure and folding from the **Horizon-centric Quantum Information and Vacuum (HQIV)** framework. No empirical force fields, no PDB statistics, no hard-coded radii—every geometry emerges from the discrete null lattice, causal-horizon monogamy, and informational-energy conservation.

## Core principles

- **Discrete null lattice**: Each atom is a lattice node with Z shells; bond lengths = diamond overlap Θ_ij.
- **Causal-horizon monogamy**: Entanglement budget per overlapping diamond constrains coordination and H-bonds.
- **Informational-energy**: E_tot = Σ m c² + Σ ħ c / Θ_i; minimization yields equilibrium geometry.
- **Horizon scalar**: φ = 2 c² / Θ_local; geometric damping f_φ = −γ′ φ / (a_loc + φ/6)² ∇φ.
- **Observer-centric**: No FLRW or empirical constants; all from HQVMetric and lattice.

## Installation

No external dependencies beyond **Python 3.10+** and **numpy**. Optional: **pyhqiv** for metric math (φ = 2/Θ, γ); when installed, horizon and damping use it. Clone the repo and use:

```python
from horizon_physics.proteins import (
    hqiv_predict_structure,
    alpha_helix_geometry,
    backbone_geometry,
    beta_sheet_geometry,
    side_chain_chi_preferences,
    e_tot,
    minimize_e_tot,
    minimize_e_tot_lbfgs,
    rational_alpha_parameters,
    rational_ramachandran_alpha,
)
```

## Usage

### Backbone geometry (Å, degrees)

```python
from horizon_physics.proteins import backbone_geometry

g = backbone_geometry()
# Cα–C: 1.53 Å, C–N: 1.33 Å, ω: 180°, φ/ψ for α and β
print(g["Calpha_C"], g["C_N"], g["omega_deg"])
```

### Alpha helix

```python
from horizon_physics.proteins import alpha_helix_geometry

h = alpha_helix_geometry()
# 3.6 res/turn, 5.4 Å pitch, H-bond N–O ~ 2.9 Å
print(h["residues_per_turn"], h["pitch_ang"], h["hbond_N_O_ang"])
```

### Beta sheet

```python
from horizon_physics.proteins import beta_sheet_geometry

b = beta_sheet_geometry()
print(b["rise_per_residue_ang"], b["strand_spacing_ang"], b["hbond_N_O_ang"])
```

### Side-chain χ (20 amino acids)

```python
from horizon_physics.proteins import chi_angles_for_residue, side_chain_chi_preferences

chi = chi_angles_for_residue("VAL")
prefs = side_chain_chi_preferences()
```

### Folding energy and deterministic gradient descent

Only gradient-based optimization (no Monte Carlo, no random seeds). L-BFGS in pure numpy:

```python
from horizon_physics.proteins import minimize_e_tot_lbfgs, small_peptide_energy
import numpy as np

pos0 = np.array([[0,0,0], [3.8,0,0], [7.6,0,0]])
z = np.array([6, 6, 6])
pos_opt, info = minimize_e_tot_lbfgs(pos0, z, max_iter=200)
e = small_peptide_energy("AAA")
```

Rational parameters (φ = -57°, ψ = -47°, rise = 3/2 Å, pitch = 27/5 Å):

```python
from horizon_physics.proteins import rational_alpha_parameters, rational_ramachandran_alpha
rational_alpha_parameters()  # rise 3/2, pitch 27/5, residues_per_turn 18/5
rational_ramachandran_alpha()  # (-57, -47)
```

### Secondary structure prediction (no ML)

From E_tot(φ,ψ) basins at (-57°,-47°) and (-120°,120°); Θ_eff per residue from lattice:

```python
from horizon_physics.proteins import predict_ss, predict_ss_with_angles

ss, confidence = predict_ss("MKFLNDR", window=5)
# ss: 'H' (helix), 'E' (strand), 'C' (coil)
out = predict_ss_with_angles("MKFLNDR")  # includes dist_to_alpha_min, dist_to_beta_min
```

### Full-chain minimizer

Minimize E_tot over the Cα trace (L-BFGS or fast path), then rebuild full backbone (N, CA, C, O) and optional Cβ. **Default:** full vector-sum horizon gradient (every atom j contributes to force on i within 15 Å); optional lightweight Cβ rotamer search after backbone to improve packing/lDDT.

```python
from horizon_physics.proteins import minimize_full_chain, full_chain_to_pdb

result = minimize_full_chain("MKFLNDR", max_iter=300, include_sidechains=True)
# result["ca_min"], result["backbone_atoms"], result["E_ca_final"], result["E_backbone_final"]
pdb_str = full_chain_to_pdb(result, chain_id="A")
```

- **Gradients:** Analytical everywhere (`grad_bonds_only` + `grad_horizon_full`); no finite differences in L-BFGS or fast path.
- **Horizon:** Default = full vector sum over all pairs within horizon radius. Use `fast_horizon=True` (or CLI `--fast`) for bonds-only / nearest-neighbor (faster, for debugging).
- **Side-chain pack:** If `include_sidechains=True`, rotamer packing runs after backbone: χ1 from `side_chain_chi_preferences()` (grid pref±120°, ±60°, 0°) then 5 L-BFGS steps on χ1 to minimize clash energy. Disable with `side_chain_pack=False`. Optional standalone: `pack_sidechains(result)`.
- **Quick mode:** `minimize_full_chain(..., quick=True)` disables side chains, uses fewer iterations and relaxed gtol, and bonds-only gradient for rapid prototyping.
- **Long-chain collapse:** For n_res > 50, `collapse=True` (default) runs two-stage annealing (Rg collapse then refine). Tune with `long_chain_max_iter` (default 250), `collapse_init_steps` (default 40), or `k_rg_collapse` to strengthen the Rg pull. Set `collapse=False` to skip and use the original single-stage path.
- **Co-translational ribosome tunnel:** `simulate_ribosome_tunnel=True` (default `False`) enables a physics-first co-translational mode: **binary-tree segment schedule** (instead of N→C run down the chain, to damp vibrations and improve convergence), null search cone (exit tunnel, `cone_half_angle_deg=12`, `tunnel_length=25` Å), plane at the tunnel lip, fast-pass spaghetti (rigid-group 6-DOF + bell at junction), and one short HKE min pass per segment. With `post_extrusion_refine=True` (default), once the full chain is extruded the full HKE two-stage collapse/refine is run **repeatedly until no Cα moves more than 0.5 Å per 100 residues between runs** (adaptive) (no fixed round cap). Cone/plane removed during post-extrusion. Tunnel axis default +Z; override with `tunnel_axis`. See `co_translational_tunnel.py` and `test_co_translational_tunnel.py`.

### Co-translational Mode (Ribosome Tunnel Simulation)

When `simulate_ribosome_tunnel=True`, the minimizer builds the chain co-translationally with these steps:

1. **Binary-tree schedule:** Segments are processed in divide-and-conquer order (pairs, then merge at junctions, then larger segments) instead of running down the chain N→C. This damps vibrations that would otherwise hurt convergence. At each segment, the bell (residues allowed to move freely) is at the junction; the rest move as a rigid group.
2. **Null search cone:** Residues whose Cα is still inside the tunnel (N-terminal segment up to `tunnel_length` Å) must stay inside a conical volume. Any gradient component that would push a Cα outside the cone is projected back (zeroed outward radial component).
3. **Plane at tunnel lip:** A hard plane perpendicular to the tunnel axis at the lip. Gradient components that would move residues back across the plane (unphysical re-entry) are nullified.
4. **Fast-pass spaghetti:** The already-extruded portion is treated as a rigid/semi-rigid body (6-DOF group rotation + translation); only the bell (e.g. junction or newest residues) is optimized with large translations.
5. **Min pass on connection:** When a segment is merged or a new residue emerges, one short HKE minimization pass is run on that segment with cone/plane still active (and optionally only the part above `hke_above_tunnel_fraction` of the tunnel updated).
6. **Post-extrusion refinement** (if `post_extrusion_refine=True`): Once the full chain is extruded, cone/plane constraints are removed and the full HKE two-stage collapse/refine is run repeatedly until **no Cα moved more than 0.5 Å per 100 residues between runs** (adaptive) (no round limit).

Edge cases: short chains (n_res &lt; tunnel_length) are handled by the same loop. Prolines and signal peptides use the same cone/plane and bond projection; no special branching.

### Live trajectory visualizer

When the minimizer is run with `trajectory_log_path` set (or `run_tunnel_and_grade --trajectory-log PATH`), it writes a JSONL file (one `{"t": step, "positions": [[x,y,z], ...]}` per line, flushed each frame). A **live visualizer** tails this file in a separate process and updates a 3D matplotlib view (Cα scatter + bonds) in real time, without affecting minimizer performance:

```bash
# Terminal 1
python -m horizon_physics.proteins.examples.run_tunnel_and_grade --targets T1037 --trajectory-log /tmp/traj.jsonl

# Terminal 2 (requires matplotlib)
python -m horizon_physics.proteins.examples.live_trajectory_viz /tmp/traj.jsonl
```

See `examples/live_trajectory_viz.py` for options (`--no-bonds`, `--update-ms`).

### Signal dump on kill

When `signal_dump_path` is set (or `run_tunnel_and_grade --signal-dump PATH`), the minimizer registers SIGINT/SIGTERM handlers. On Ctrl+C (or `kill`), it writes the **current** Cα state to that path as a PDB and exits. Lets you recover a partial run without losing progress. The dump is updated each refinement step (tunnel post-extrusion and long-chain path).

### Run minimizer from a PDB (troubleshooting)

To refine or troubleshoot an existing structure, load Cα and sequence from a PDB and run the minimizer:

```bash
python -m horizon_physics.proteins.examples.run_minimizer_on_pdb model.pdb -o refined.pdb
python -m horizon_physics.proteins.examples.run_minimizer_on_pdb model.pdb -o out.pdb --tunnel --signal-dump /tmp/dump.pdb
```

Sequence is read from PDB residue names (3-letter → 1-letter). Use `load_ca_and_sequence_from_pdb(path)` in code for the same (returns `(ca_xyz, sequence)`).

### Current algorithm (full-chain pipeline)

The pipeline used by `minimize_full_chain` is:

1. **Sequence** — Parse FASTA or use the one-letter sequence; validate against the 20 standard amino acids.

2. **Secondary structure** — Predict per-residue SS (H/E/C) from Θ_eff and E_tot(φ,ψ) basins (no ML). Optional window majority vote. If no `ss_string` is provided, `predict_ss(sequence)` is used.

3. **Initial Cα placement** — Place Cα segment-by-segment from N→C:
   - **H**: helix geometry (rise 3/2 Å, pitch 27/5 Å from HQIV).
   - **E/C**: extended/coil (rise 3.2 Å / 3.0 Å).
   - Segments are stitched by aligning the segment start direction to the previous segment end; no bond-length sweep at this stage.

4. **Cα minimization** — Minimize an effective energy (bond penalty + horizon forces) over the Cα trace:
   - **Bonds:** Consecutive Cα–Cα penalized outside [2.5, 6.0] Å; soft minimum near 3.8 Å. Gradient from `build_bond_poles` (analytical).
   - **Horizon:** Full vector sum of repulsive forces from every atom j to i within horizon radius (12–15 Å). Each pair contributes a **bond potential vector (pole)** (i→j); gradient from `build_horizon_poles` or `grad_horizon_full`. A **neighbor list** (12 Å) limits pairs to nearby atoms for speed.
   - **Combined gradient:** Analytical only (`grad_bonds_only` + `grad_horizon_full`); no finite differences. Optional `fast_horizon`: use bonds-only (nearest-neighbor) for debugging.
   - **Step:** Gradient descent with adaptive step; after each step, **project** consecutive bonds into [2.5, 6.0] Å.
   - **Paths:** For n_res > 50, a **two-stage** path is used by default (`collapse=True`): (1) **Collapse:** bonds-only + radius-of-gyration (Rg) term and loose bond limits (r_max=8 Å, r_min=2 Å) for the first 50% of iterations to bias the chain toward a compact globule; (2) **Refine:** full bonds + horizon, standard [2.5, 6] Å. Optional `collapse_init_steps` (default 40) of Rg-only steps for compact init. Use `collapse=False` for the original single-stage fast path. For short chains (n_res ≤ 50), L-BFGS with analytical gradient and bond projection.

5. **Backbone rebuild** — From minimized Cα, place N, CA, C, O per residue using local tangent and HQIV bond lengths (N–CA, CA–C, C–O).

6. **Side chains (optional)** — If `include_sidechains=True`: add Cβ from N–CA–C plane (trans); then **side-chain packing**: χ1 from `side_chain_chi_preferences()`, grid (pref ±120°, ±60°, 0°), then a short L-BFGS on the χ1 vector to minimize Cβ clash energy.

7. **Output** — Result dict (e.g. `ca_min`, `backbone_atoms`, energies, `info`); `full_chain_to_pdb(result)` writes MODEL 1 … ENDMDL END.

Poles (bond potential vectors) can be stored per minimization step: `build_horizon_poles`, `build_bond_poles`, and `grad_from_poles` let you keep and reuse the ± vectors per (i, j).

### CASP submission: FASTA → PDB (SS-aware)

```python
from horizon_physics.proteins import hqiv_predict_structure, predict_ss

fasta = ">target\nMKFLNDR..."
pdb_str = hqiv_predict_structure(fasta)  # SS predicted from sequence
# Or pass SS explicitly:
ss, _ = predict_ss(sequence)
pdb_str = hqiv_predict_structure(fasta, ss_string=ss)
# pdb_str is valid CASP format: MODEL 1 ... ENDMDL END
```

## CASP17 submission instructions

1. **Predict**: Call `hqiv_predict_structure(fasta)` with the target FASTA to get a PDB string.
2. **Format**: Submit the string as a single MODEL (MODEL 1 … ENDMDL END). Chain ID default is "A".
3. **Naming**: Use the CASP-assigned target ID in the header if required.
4. **Validation**: Run `horizon_physics/proteins/validation.ipynb` to confirm “Exact match to experiment” for H₂O, crambin, etc.

### CLI (full-chain minimizer)

```bash
python -m horizon_physics.proteins.full_protein_minimizer target.fasta -o out.pdb
python -m horizon_physics.proteins.full_protein_minimizer --fast target.fasta   # bonds-only gradient (debug)
python -m horizon_physics.proteins.full_protein_minimizer --no-sidechain-pack target.fasta  # skip Cβ rotamer search
```

## Validation (no Jupyter)

Run all checks and print "Exact match to experiment":

```bash
python -m horizon_physics.proteins.validation
```

## Examples

- **Crambin**: `examples/crambin.py` — 46 residues, full backbone PDB.
- **Insulin fragment**: `examples/insulin_fragment.py` — B-chain 1–30.
- **T1131** (Hormaphis cornu): `examples/T1131.py` — 173 residues; writes `T1131_hormaphis_cornu_minimized_cartesian.pdb` (flat pipeline).
- **T1037** (S0A2C3d4): `examples/T1037.py` — 404 residues, CASP14 target; writes `T1037_S0A2C3d4_minimized_cartesian.pdb` (flat pipeline).
- **Hierarchical**: `examples/run_examples_tpu.py` — same targets via HKE; writes `*_minimized_hierarchical.pdb` and compares to the cartesian outputs.
- **All four, both pipelines**: `python -m horizon_physics.proteins.examples.run_all_pipelines` — runs T1037, T1131, crambin, insulin fragment with Cartesian and Hierarchical; writes `*_minimized_cartesian.pdb` and `*_minimized_hierarchical.pdb`. Use `--quick` for fewer iters, `--targets crambin,insulin_fragment` for a subset.

Run from repo root: `python3 -m horizon_physics.proteins.examples.T1131` or `T1037`.

### Reference structures (T1131, T1037)

Example outputs: run `T1131.py` / `T1037.py` for `*_minimized_cartesian.pdb`; run `run_examples_tpu.py` for `*_minimized_hierarchical.pdb` (same targets, comparable). To evaluate against published reference structures:

- **CASP targets:** Check [CASP](https://predictioncenter.org) for the corresponding target IDs and released experimental or reference models when available.
- **Grading (Cα-RMSD):** Use the bundled grading module to compare a predicted PDB to a reference (e.g. from RCSB or CASP):
  ```python
  from horizon_physics.proteins import ca_rmsd
  rmsd_ang, per_res, pred_ca, ref_ca = ca_rmsd("crambin_minimized_cartesian.pdb", "1crn.pdb")
  print("Cα-RMSD: {:.3f} Å".format(rmsd_ang))
  ```
  CLI: `python -m horizon_physics.proteins.grade_folds pred.pdb ref.pdb` (optionally `--no-resid` to align by residue order instead of residue ID).

- **Optional grading module (trajectory + gold for AI/ML):** Install with `pip install -r requirements-grading.txt` (adds pandas). Then use translation logs and gold PDBs to get per-frame metrics for better modeling and algorithms:
  ```python
  from horizon_physics.proteins.grading import grade_trajectory, grade_prediction
  # Per-frame Cα-RMSD over a trajectory (Cartesian or HKE 6-DOF JSONL)
  results = grade_trajectory("traj_logs/crambin_cartesian_traj.jsonl", "gold/1crn.pdb", output_path="grades.csv")
  # Single prediction vs gold
  metrics = grade_prediction("crambin_minimized_cartesian.pdb", "gold/1crn.pdb")
  ```
  See **[horizon_physics/proteins/grading/README.md](grading/README.md)** for the full ML pipeline (trajectory log → gold → CSV/JSON for reward signals, convergence curves, or training data).

No reference PDBs are bundled; download a reference (e.g. 1CRN for crambin from RCSB), then use `ca_rmsd` or the grading module to get Cα-RMSD after Kabsch superposition.

### Bond potential vectors (poles)

Each horizon or chain bond is represented as a **pole**: a bond potential vector with a ± convention. A pole is `(i, j, vec)` where `vec` points from atom `i` toward atom `j`; the force on `i` is −`vec`, on `j` is +`vec`. You can build and store poles for reuse (e.g. per minimization step):

```python
from horizon_physics.proteins import build_horizon_poles, build_bond_poles, grad_from_poles

# Horizon poles (all pairs within cutoff)
poles_h = build_horizon_poles(positions, z_list)
# Chain bonds only (consecutive Cα–Cα)
poles_b = build_bond_poles(positions)
# Reconstruct gradient from any list of poles
grad = grad_from_poles(poles_b + poles_h, n)
```

`grad_horizon_full(..., return_poles=True)` returns `(grad, poles)` so you can keep the horizon poles for the current step.

## pyhqiv integration

When **pyhqiv** is installed, `_hqiv_base` uses `phi_from_theta_local` (φ = 2/Θ) and `GAMMA` from pyhqiv. For a list of suggested additions to pyhqiv so PROtein can rely on it for all HQIV metric math (Θ from Z/coordination, bond length from Θ, damping magnitude, constants), see **[docs/pyhqiv_recommendations.md](../../docs/pyhqiv_recommendations.md)**.

## Implementation notes

- **Analytical gradients:** All minimization uses analytical gradients (`grad_bonds_only` for consecutive Cα–Cα bonds; `grad_horizon_full` for the full vector sum of horizon forces from every atom j to i within 15 Å). No finite-difference gradients.
- **Default horizon:** Full vector-sum horizon (long-range crowding) is the default. Use `fast_horizon=True` or `--fast` for bonds-only (nearest-neighbor only), which is faster and useful for debugging.
- **Side-chain packing:** After backbone minimization, χ1 is optimized using `side_chain_chi_preferences()`: grid search (pref ±120°, ±60°, 0°) then L-BFGS on the χ1 vector to minimize clash energy. Uses existing geometry (N–CA–Cβ). Improves lDDT and all-atom RMSD; run standalone with `pack_sidechains(result)` on an existing result.

## License

MIT. All code is first-principles HQIV; experimental agreement emerges from the axioms.
