# Examples & gallery

## Jupyter

- **[validation.ipynb](https://github.com/disregardfiat/protein_folder/blob/main/src/horizon_physics/proteins/validation.ipynb)** — backbone, helix, sheet, and small-system checks ("exact match to experiment" smoke tests).

When running locally, open the notebook from the repo with the environment that has `pip install -e ".[full]"` and ensure the kernel can import `horizon_physics` (the first code cell walks up to the repo and adds `src/`).

## Runnable scripts

Under `src/horizon_physics/proteins/examples/`:

| Script | Description |
|--------|-------------|
| `crambin.py` | Small protein PDB via HQIV predict |
| `insulin_fragment.py` | Insulin B fragment |
| `run_tunnel_and_grade.py` | Tunnel fold + grading hooks |
| `run_minimizer_on_pdb.py` | Refine from existing PDB |
| `extrude_into_7K00_tunnel.py` | Tunnel field extrusion pipeline |

Run as modules from the repo root after install:

```bash
python -m horizon_physics.proteins.examples.crambin
```

## PDB outputs

Example output PDBs ship in `examples/` for regression and visualization (e.g. crambin minimized structures). These are **model outputs**, not experimental structures, unless the filename indicates a gold reference (e.g. `crambin_1CRN.pdb`).

## Visual trajectory

See the root [README](https://github.com/disregardfiat/protein_folder#readme) for `live_trajectory_viz` usage with JSONL trajectory logs.
