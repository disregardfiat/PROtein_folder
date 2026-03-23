"""
Optional grading module: trajectory logs + gold-standard PDBs → metrics for AI/ML.

Use translation (trajectory) logs and reference structures to evaluate folds and
drive better modeling and algorithms. Install optional deps for DataFrame/CSV
export and richer metrics.

  pip install -r requirements-grading.txt   # optional

Core (no extra deps):
  grade_trajectory(traj_path, gold_pdb_path) → list of {frame, rmsd, ...}
  grade_prediction(pred_pdb, gold_pdb)       → dict of metrics (uses grade_folds)

With optional pandas: results can be returned as DataFrame and written to CSV.
"""

from __future__ import annotations

from .trajectory_grade import grade_trajectory, load_trajectory_frames
from .metrics import grade_prediction

__all__ = [
    "grade_trajectory",
    "load_trajectory_frames",
    "grade_prediction",
]
