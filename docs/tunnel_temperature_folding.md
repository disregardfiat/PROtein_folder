## Tunnel + Temperature‑Aware Folding Strategy

This document outlines a two‑stage folding strategy that:

- Uses **Cartesian / tunnel geometry** to generate an initial plausible backbone.
- Folds at a **finite temperature** inside the ribosome tunnel (null search space).
- Then unlocks into the **global search space** outside the tunnel for full compaction.
- Uses a **discrete move set** and **per‑DOF energy caching** to keep the search tiny.

---

### 1. Stage 0 – Initial plausible geometry (Cartesian)

1. **Build full chain at once**
   - Use the existing Cartesian / fast minimizer to produce a full N–CA–C–O backbone for the whole sequence.
   - This gives:
     - `pos_cartesian (N_atoms, 3)`
     - `z_list` (atom types / shells)
     - Cα positions as a subset.

2. **Secondary structure prediction**
   - Use `predict_ss(sequence)` to obtain an SS string (H, E, C).
   - Helical segments are natural candidates for **kinetic groups** (rigid units) later.

This stage is run effectively at “0 K” just to get a plausible starting geometry, not to explore the folding landscape.

---

### 2. Stage 1 – Tunnel‑restricted folding at finite temperature

The ribosome tunnel is treated as a **null search space**: residues inside the tunnel must obey geometry constraints, but are allowed to explore local energy basins at finite temperature.

#### 2.1. Map Cartesian backbone into tunnel coordinates

- Use the existing `align_chain_to_tunnel(ca_positions, ptc_origin, axis)` logic to:
  - Place the N‑terminus at the PTC origin.
  - Align the initial chain along the tunnel axis.

#### 2.2. Apply tunnel + lip constraints

- **Cone constraint**:
  - Cα positions inside the tunnel must lie within a conical volume around the axis.
  - Gradient components that move Cα’s outside the cone are masked (zeroed).

- **Lip plane**:
  - Residues past the lip cannot move back through the lip plane.
  - Gradient components pointing back into the tunnel for those residues are masked.

These masks are applied *before* temperature‑dependent acceptance.

#### 2.3. Finite‑temperature HKE / Cα folding

On top of the tunnel constraints, folding is done via a **discrete move set** over Cα‑level DOFs:

- **DOFs**:
  - Per‑residue Cα translations/rotations or HKE group DOFs (rigid helix units, loops, domains).

- **Discrete states per DOF**:
  - For each DOF, define a small set of allowed states (angles / translations).
  - Cache, for each neighbor state:
    - \(\Delta E = E_{\text{new}} - E_{\text{current}}\) using `e_tot_ca_with_bonds` / `grad_full`.

- **Temperature‑aware search**:
  - For each candidate move:
    - If \(\Delta E < 0\), accept.
    - If \(\Delta E > 0\), accept with probability \(\exp(-\Delta E / kT_{\text{eff}})\).
  - `kT_eff` corresponds to the effective folding temperature (e.g. body temperature).

- **Locking DOFs / groups**:
  - If all candidate moves for a DOF have \(\Delta E \gg kT_{\text{eff}}\) or are masked by tunnel/lip/sterics, mark the DOF as **locked**.
  - Group DOFs (helices, domains, loops) become **kinetic groups**:
    - When all DOFs in a group are locked, the group is treated as rigid in subsequent steps.

This stage lets residues **explore local basins** inside the tunnel, while respecting the physical constraints of the ribosome environment.

---

### 3. Stage 2 – Unlock global search space after tunnel exit

Once the last residue has emerged from the tunnel:

1. **Unlock tunnel constraints**
   - Disable cone and lip gradient masking.
   - All residues are now free to move in the full 3D space, subject only to:
     - Bond constraints,
     - Sterics / clashes,
     - Horizon / informational energy.

2. **Global Cα compaction (binary search)**
   - Run a Cα‑level **binary‑tree minimization**:
     - Divide the chain into segments.
     - For each segment, run a short HKE / L‑BFGS pass over its DOFs.
     - Merge segments and repeat, so that the compaction is hierarchical.
   - Continue to:
     - Lock DOFs / groups whose moves are too costly (relative to \(kT_{\text{eff}}\)).
     - Focus the search on flexible loops and mispacked regions.

3. **Optional EM‑field polishing**
   - After the global Cα topology is compact, optionally feed the backbone into the EM‑field pipeline:
     - Small displacement steps,
     - No or weak compaction term,
     - A few iterations to resolve clashes and fine‑tune geometry.

This stage is still at finite temperature, but with a much larger accessible search volume than the tunnel.

---

### 4. Stage 3 – Cryo‑like finalization for comparison to experiment

For comparison to **cryo‑EM / X‑ray** data, run a final **low‑temperature / 0 K** refinement:

1. Set \(T_{\text{eff}} \to 0\) (or very small).
2. Turn off stochastic acceptance; only accept moves that reduce energy:
   - Or run pure gradient/HKE / EM‑field relaxation without noise.
3. Run a small number of steps:
   - Clear remaining clashes.
   - Sharpen local geometry (torsions, bond lengths).

The resulting **cryo state** is the one that should be graded with `grade_folds.py` (Cα‑RMSD, etc.) against experimental structures.

---

### 4.1 Implemented: discrete φ/ψ refinement (post‑tunnel)

After the tunnel + HKE stage, an optional **discrete DOF refinement** step is available:

- **Modules**: `backbone_phi_psi.py` (φ/ψ ↔ Cα/atoms), `discrete_dof.py` (per‑residue φ/ψ states from pyhqiv), `temperature_path_search.py` (Metropolis loop).
- **Flow**: From the tunnel backbone, extract φ/ψ → snap to discrete states → run `run_discrete_refinement(sequence, temperature, n_steps, initial_backbone_atoms=...)` → Metropolis over neighbor moves, rebuild Cα from φ/ψ via `ca_positions_from_phi_psi`, evaluate `e_tot_ca_with_bonds`, lock DOFs periodically.
- **Requires**: `pyhqiv` (e.g. from hqvmpy) for `build_backbone_dofs_for_sequence`. If absent, the pipeline skips discrete refinement and still reports Cartesian vs tunnel and writes tunnel PDBs.
- **Example**: `run_tunnel_temperature_pipeline.py` runs crambin and insulin_fragment with discrete refinement when pyhqiv is available, writes `*_tunnel.pdb` and `*_discrete.pdb`, and grades crambin vs 1CRN.

---

### 5. Why this makes the search tiny and physical

- **Tiny search space**:
  - Each DOF is a finite set of pre‑scored states.
  - Moves are selected from a small set of low‑ΔE candidates.
  - Locked DOFs and kinetic groups remove large parts of the space from consideration.

- **Physical realism**:
  - Tunnel and lip constraints enforce ribosome geometry while residues are inside.
  - Finite temperature allows exploration of local basins and movement across modest barriers.
  - Unlocking after tunnel exit allows global domain wrapping and compaction.
  - Final cryo‑like refinement aligns with how experimental structures are obtained.

This design connects the **initial plausible geometry** (Cartesian), the **co‑translational tunnel regime**, and the **global folding search outside the tunnel**, all under a temperature‑aware, computationally efficient framework.

