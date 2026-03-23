# Optional grading module

Translation logs (trajectory JSONL) plus gold-standard PDBs let you grade folds and feed metrics into AI/ML for better modeling and algorithms.

## Install (optional)

```bash
pip install -r requirements-grading.txt   # adds pandas for CSV/DataFrame export
```

Core grading works with only the base PROtein deps (numpy, scipy); pandas is optional for CSV and DataFrame output.

## Pipeline for ML

1. **Run minimization with trajectory logging**
   ```bash
   python -m horizon_physics.proteins.examples.run_all_pipelines \
     --targets crambin --trajectory-log traj_logs
   ```
   Produces e.g. `traj_logs/crambin_cartesian_traj.jsonl` and `traj_logs/crambin_hierarchical_traj.jsonl`.

2. **Obtain a gold-standard PDB**
   Download the reference (e.g. 1CRN for crambin from RCSB) and place it next to your outputs.

3. **Grade the trajectory**
   ```python
   from horizon_physics.proteins.grading import grade_trajectory

   results = grade_trajectory(
       "traj_logs/crambin_cartesian_traj.jsonl",
       "gold/1crn.pdb",
       output_path="crambin_cartesian_grade.csv",
       output_format="csv",
   )
   # results: per-frame rmsd_ang, per_residue_rmsd_min/max/mean
   ```

4. **Use the metrics**
   - **Convergence:** Plot `frame` vs `rmsd_ang` to see how fast the run approaches the gold structure.
   - **Reward / loss:** Use per-frame or final RMSD as a reward signal or loss for policy or model learning.
   - **Datasets:** Batch `grade_trajectory` over many runs and gold PDBs to build tables for training (e.g. (sequence, trajectory_path, gold_path) → metrics).

## Grade a single prediction

```python
from horizon_physics.proteins.grading import grade_prediction

metrics = grade_prediction("crambin_minimized_cartesian.pdb", "gold/1crn.pdb")
print(metrics)  # ca_rmsd_ang, n_res, per_residue_rmsd_*
```

## Trajectory formats

- **Cartesian:** `positions` = list of `[x, y, z]` (Cα-only, one per residue). From the flat pipeline.
- **HKE (6-DOF):** `positions` = list of 6-element lists `[tx, ty, tz, ez, ey, ex]` per residue. Converted to Cα via `relative_6dof_to_world_backbone` before RMSD.

Both are detected automatically from the JSONL.
