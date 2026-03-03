# EM-field co-translational pipeline

Lattice-based 3D EM field with tunnel extrusion and force-following relaxation.

## Overview

- **Atom**: Each atom is a full class; radius and Θ from pyhqiv (`theta_for_atom`).
- **EMField**: 3D lattice potential; build/update via local stencil (no n²).
- **CoTranslationalAssembler**: Residues emerge from tunnel; search space rebounds every insertion; atoms relax along field lines.

## Pipeline

1. **Build whole protein at once** via fast assembler (SS-aware Cα placement + backbone). Helices form naturally from geometry; no 1-per-pass extrusion.
2. **Alternating HKE + field**: Relaxation alternates between (a) EM-field force-following step and (b) HKE gradient step on Cα. Field rebuilt after each pair.
3. **Helical kinetic groups**: Helical segments (from SS prediction) move as rigid bodies during relaxation—each helix gets one translation per step.
4. **Post-assembly relaxation**: Run until atoms barely move (coarse-to-fine res: 5→3→1→0.5 Å).
5. **Low-energy pockets** in the field become docking hot-spots for ligands or next protein domains.

## Usage

```python
from horizon_physics.proteins import CoTranslationalAssembler

assembler = CoTranslationalAssembler(batch_size=50)
assembler.run_pipeline("MKFLNDFESKIS...")  # your sequence

spans, end2end = assembler.compactness()
pockets = assembler.low_energy_pockets()  # docking hot-spots
```

## Parameters

- `batch_size`: Residues from fast assembler before switching to 1-per-pass (default 50).
- `field_res`: Lattice resolution in Å (default 0.8).
- `tunnel_exit`, `tunnel_axis`: Tunnel geometry for extrusion.
- `temperature`: Effective temperature (e.g. 310.0 for body temperature). Currently stored on the assembler and available to scale stochastic/annealing moves in the relaxation steps.

## Performance

- **Vectorized stencil**: `add_atom` uses numpy meshgrid instead of Python loops (~100× faster).
- **Tight bounding box**: Field is resized to fit atoms + padding each rebuild (fewer voxels).
- **Batch force computation**: `forces_at_all` computes forces for all atoms at once.
- **Coarse-to-fine resolution**: `field_res_schedule` (e.g. `[5, 3, 1, 0.5]` Å) runs relaxation at low resolution first (few voxels, fast), then refines at higher resolution for the final fit.
- **Refine mode** (`refine=True` or `--refine`): 800 steps, 0.5 Å final resolution, 0.01 Å convergence threshold. Targets 1.5 Å gold-standard fit.
- **Power-law displacement**: Scale uses (|F|/max|F|)^(1/4) instead of linear; subtle, effect doubled by force from both sides of each attraction.
- `compact_until_converged`: Run post-assembly relaxation (default True). Set False to skip.
- `max_disp_threshold`: Stop relaxation when max atom displacement < this (default 0.02 Å).

## Ruler + anchor + drag

- **Ruler**: Residue addition rate calibrates step size. Target displacement per step (`_target_disp_per_step`) adapts `step_size` so progress matches the 1-residue-per-pass pace.
- **Anchor**: Atom with smallest |F| (most equilibrated) is the anchor. Used as pivot for NN-group rotations.
- **Drag**: When an atom moves, bonded neighbors are pulled along (`drag_factor`). Each atom drags its peers.
- **Rotation**: Every 4th relax step, nearest-neighbor group around the anchor rotates slightly around the tunnel axis.

## Post-assembly relaxation

After all residues are added, `relax_to_convergence()` runs by default:

- **Free fold**: No tunnel rebounding—the chain can compact in any direction.
- **High-potential first**: Displacement scaled by |F| so atoms in high-potential (clashing) regions move more.
- **Convergence**: Stops when `max(displacement) < max_disp_threshold` (default 0.02 Å), after at least `min_steps` (default 20).

```python
assembler.run_pipeline(seq, compact_until_converged=True, max_disp_threshold=0.02)
# Or call explicitly:
n_steps = assembler.relax_to_convergence(
    max_disp_threshold=0.02, max_steps=2000, min_steps=20,
    high_potential_first=True, step_size=0.1
)
```
