# Hierarchical Kinematic Engine (HKE)

Optional parallel architecture for protein folding. Use the **flat** minimizer (`minimize_full_chain`) or the **hierarchical** one (`minimize_full_chain_hierarchical`) via a single flag—no changes to existing flat code.

## Switch with one flag

```python
from horizon_physics.proteins import minimize_full_chain, minimize_full_chain_hierarchical

# Flat (existing)
result = minimize_full_chain("MKFLNDR", include_sidechains=True)

# Hierarchical (HKE)
pos, atom_types = minimize_full_chain_hierarchical(
    "MKFLNDR",
    include_sidechains=True,
    device="cpu",  # or "cuda", "tpu" when JAX is available
    grouping_strategy="residue",  # or "ss", "domain", "helix_unit" (tight H/E as single kinetic units)
)
```

## Architecture

- **Atom**: Relative 6DOF (bond length, two bond angles, torsion, optional rigid offset). Forward kinematics: parent (R, t) → world position.
- **RigidGroup**: Rigid body (3×3 R + translation t), list of Atoms or child RigidGroups. Combined energy (horizon, clash, bonds) over the group. Nested tree.
- **Protein**: Root of the tree. Grouping strategies: per-residue, SS-aware, or user-defined domains. Methods:
  - `forward_kinematics()` → flat (N, 3) positions and (N,) z_shell for interop with `e_tot` / `grad_full`
  - `get_dofs()` / `set_dofs()` → vector of free DOFs (group 6DOF + internal torsions φ, ψ)
  - `compute_total_energy()` → reuses existing energy + group-level potentials
  - `minimize_hierarchical()` → staged: coarse rigid-body + torsions → refinement → optional flat Cartesian refinement

## Backend

- **Primary**: JAX (jit, grad, vmap, device placement for GPU/TPU).
- **Fallback**: Pure NumPy if JAX is not installed; heavy loops still run, gradients via finite differences over DOFs.

## Staged minimization

1. **Stage 1–2**: Minimize total energy over DOFs (6 for first residue, φ/ψ for the rest) with L-BFGS (or gradient descent).
2. **Stage 3**: Flat Cartesian refinement using existing `grad_full` on the FK positions for final polish.

The staging is aligned with the **funnel/ribosome as null search space**: the accessible volume is the natural bound so that (1) helices form inside the bound, (2) group-level translations explore within it, and (3) at “exit” (stage 3) the chain can wrap into itself for tertiary contacts. See \texttt{docs/funnel\_null\_search.md}.

All new code lives under `horizon_physics/proteins/hierarchical/`; the flat pipeline is unchanged except for an optional import of `minimize_full_chain_hierarchical` in the package `__init__.py`.
