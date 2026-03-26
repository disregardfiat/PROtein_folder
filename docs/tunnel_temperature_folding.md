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

---

### 6. Server path vs extrusion path / why a PDB can “blow up”

**Two different pipelines:**

- **Server (CASP) path**: For each submission the server runs **HKE** (`minimize_full_chain_hierarchical`) then **tree‑torque** refinement. There is **no 7K00 tunnel** in this path; it is Cartesian HKE → discrete refinement only.
- **Extrusion path**: Used in scripts like `extrude_into_7K00_tunnel.py`: **extrude through the modeled 7K00 tunnel** → **free minimizer** → **tree‑torque**. That pipeline often gives good outcomes because the tunnel constrains the initial fold.

**Why a PDB can blow up on the server:** For some targets (e.g. longer chains), the HKE minimizer can take bad steps and DOFs/positions can explode. `hierarchical_result_for_pdb` then produces backbone atoms with huge coordinates. If tree‑torque is skipped (e.g. exception) or the code falls back to the raw HKE result, that blown‑up backbone gets written and emailed → invalid PDB with huge numbers.

**Defenses in place:**

- **Final PDB sanity check** before move/send: coordinates must be finite and \|coord\| ≤ 9999 Å; otherwise the job fails and a failure email is sent (requester + CC), and the bad PDB is not moved to outputs.
- **Backbone sanity check right after HKE**: After every `hierarchical_result_for_pdb` (and after `minimize_full_chain` when ligands are used), the server runs `_backbone_sanity_check(backbone_atoms)`. If any backbone coordinate is non‑finite or out of range, it raises before tree‑torque or output, so a blown‑up backbone is never passed on or sent.

- **1‑hour wall‑clock timeout**: Each **entire job** (fast‑pass if needed, then HKE) runs in one **subprocess** with `start_new_session=True` and `timeout=PREDICTION_TIMEOUT_SEC` (default 3600). There is no separate “Phase 1” that blocks: every pending job is started in its own thread, each running one subprocess. If a subprocess does not finish within 1 hour, the **entire process group** is killed and the server **sends the fast‑pass result** by email (if .fast.pdb was written) with a time‑limit note. So the user always gets a result after at most 1 hour, and no single job can block others.

---

### 7. Known CASP targets and experimental refs (equal footing)

When **CASP_KNOWN_TARGETS_CACHE** is set to a directory path, the server:

1. Fetches the list of known targets from **predictioncenter.org** (sequences + PDB codes from the target list) for the round given by **CASP_ROUND** (default CASP16).
2. On each prediction job, matches the request sequence(s) to these targets (exact match).
3. If there is a match and the target has a released experimental structure (PDB code), the server downloads that PDB from RCSB (if not already in the cache) and copies it to **outputs** as `{base}.experimental_ref.pdb` alongside the prediction.

So we are on the same footing as other teams: we use the same experimental data from the Prediction Center, then run our **A+B assembly strategy** (or single-chain pipeline) as usual. The experimental ref in outputs allows direct comparison (e.g. with `grade_folds.ca_rmsd`) without manual lookup.

---

### 8. Post-tunnel short anneal and validation grading

**Short Metropolis anneal (Lean post-extrusion, `em_treetorque` mode):** After 3D EM-field relaxation, you can replace the single discrete tree-torque pass with a **short cool-down** at several temperatures (high → `temperature_k`) using true Metropolis acceptance on the discrete moves. In Python this is `post_extrusion_anneal=True` on `fold_lean_ribosome_tunnel` / `minimize_full_chain`; optional `post_extrusion_anneal_schedule_k` is a tuple of Kelvin values (≥2). On the server, set `CASP_LEAN_POST_ANNEAL=1` and optionally `CASP_LEAN_POST_ANNEAL_SCHEDULE=348,330,318,310` (comma-separated). Any optional Langevin tail uses the **last** schedule temperature as its thermal scale.

**Grading, not just timing:** `scripts/validate_casp_lean_pipeline.py` reports **wall time**, **PDB sanity**, and **scalar metrics** from the minimizer result (`E_ca_final`, `E_backbone_final`, Cα **Rg**). For structure quality vs an experiment or CASP model, pass **`--grade-ref path/to/ref.pdb`** or set **`VALIDATE_LEAN_REF_PDB`**; the script writes the prediction to a temp file and prints **Cα-RMSD** (Kabsch, order-aligned with trim to min length — use a reference that matches the folded sequence order). For full control (residue-ID alignment, chain selection), call `horizon_physics.proteins.grade_folds.ca_rmsd` directly.

**In-tunnel thermal gradient (aligned with post-extrusion Langevin):** After each binary-tree segment’s masked L-BFGS pass, you can run **kT-scaled noisy gradient descent** on that segment using the **same** `grad_func` and **same** cone / lip / `hke_above_tunnel_fraction` masking as the L-BFGS step (`tunnel_thermal_gradient_relax_segment` in `co_translational_tunnel.py`). The thermal scale is **`refinement_temperature_k`** / body **`temperature_k`** on the Lean path. Python: `tunnel_thermal_gradient_steps` on `minimize_full_chain` / `fold_lean_ribosome_tunnel`. Server: **`CASP_LEAN_TUNNEL_THERMAL_STEPS`**, **`CASP_LEAN_TUNNEL_THERMAL_NOISE_FRAC`**, optional **`CASP_LEAN_TUNNEL_THERMAL_SEED`**. Default **0** preserves the old deterministic tunnel. **`quick=True`** caps per-segment thermal steps at **12** to bound cost.

---

### 9. WHIP stepwise A/B vs experimental crambin (1CRN)

To compare **force-carrier (WHIP)** translation knobs on **equal footing**, use **`scripts/crambin_stepwise_side_by_side.py`**: it builds **one shared** tunnel snapshot, then runs several variants (sqrt vs resonant, EM field direction refresh on/off, barrier/leak settings) with identical gradient steps and bond projection. Output JSON includes per-step **gap vs native long-range pairs**, optional subsampled **Cα-RMSD** vs `proteins/1CRN.pdb`, and **large-translation events** (displacement threshold + pair-distance drop). Example:

`python3 scripts/crambin_stepwise_side_by_side.py --out .casp_grade_outputs/iter_small/crambin_stepwise_side_by_side.json` (default **1000** translation steps per variant; override with `--max-steps`.)

For a fast smoke test: `--quick-snapshot --max-steps 15 --rmsd-every 0`.

**Run until motion stalls** (instead of a fixed step budget): `--until-still --safety-max-steps 30000 --record-every 25` stops when scaled carrier motion stays below `--motion-floor` for `--still-patience` consecutive steps, or when `||grad|| < --grad-tol`, or at the safety cap. Each result includes **`fold_analysis`**: Cα clash count, Ramachandran distance to alpha/beta basins vs 1CRN, **missed native contacts** (pairs &lt;7.5 Å in the reference but opened in the prediction), **spurious tight long-range** pairs, **`tuning_hints`** for pruning bad geometry, and **`n_translation_steps`** (every WHIP iteration counts as one translation).

**`stepwise_diagnostics`** (always): summarizes the **translation timeline** — best mean native-pair gap step, first rebound after that optimum, RMSD sample extrema, per tracked pair the step closest to native, and short **narrative** hints. When the run ends on a **step cap**, use this plus **`records`** to see *where* progress toward gold stalled or reversed, then tune WHIP / long-range / refine stages accordingly.

The built-in variant panel uses **larger `carrier_step` / `carrier_span`**, a higher **`wave_leak_floor`**, and slightly higher **`linear_gain`** / barrier drive so terminus-emitted carrier waves **propagate farther and bend the chain faster**. A small **`collective_kink_weight`** on `grad_full` (HQIV Cα kink budget toward the default helix reference) helps keep those stronger waves from drifting into **unphysical** local kink geometry; tune weights if RMSD improves but Ramachandran or clash metrics degrade.

Selection policy: by default (`--variants all`) the script runs the **admissible subset** aligned with Lean `ProteinVariantSelection` (clash-safe, Ramachandran-clean). Other variants are still defined and can be run explicitly via `--variants <name1,name2,...>` for diagnostics.

**WHIP / force-carrier EM direction refresh (production minimizer):** when `ensemble_translation_mix_alpha > 0`, `_minimize_bonds_fast` and the co-translational tunnel passes rebuild the EM-augmented translation direction set whenever a **nonlocal** Cα pair (sequence separation ≥ `ensemble_em_refresh_min_seq_sep`, default 3) crosses **into** the horizon sphere of radius **`ensemble_em_refresh_horizon_ang`** (default **15 Å**, same as `folding_energy.R_HORIZON` / `grad_horizon_full`). Counting uses `folding_energy.count_nonlocal_pairs_entering_horizon`. If **`ensemble_em_refresh_on_horizon_leaving`** is True, a symmetric **`count_nonlocal_pairs_leaving_horizon`** trigger also rebuilds the set when a pair moves **out** of that radius (horizon coupling drops off). Optional **`ensemble_em_refresh_on_large_disp`** restores the older displacement-based trigger. Tunables live on `minimize_full_chain` (`ensemble_em_refresh_*`).

