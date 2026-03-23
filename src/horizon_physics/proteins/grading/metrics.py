"""
Optional grading metrics: single prediction vs gold.

Wraps grade_folds.ca_rmsd and optionally adds lDDT or other metrics when
extra dependencies are available. Use for final-model evaluation and
for reward/loss design in ML.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def grade_prediction(
    pred_pdb_path: str,
    gold_pdb_path: str,
    align_by_resid: bool = True,
    include_per_residue: bool = True,
) -> Dict[str, Any]:
    """
    Grade a single predicted PDB against a gold-standard PDB.

    Returns a dict suitable for logging or ML: ca_rmsd_ang, and optionally
    per-residue stats (min, max, mean distance). With optional dependencies
    (see requirements-grading.txt), can add lDDT or other metrics later.

    Parameters
    ----------
    pred_pdb_path : str
        Path to predicted PDB (e.g. *_minimized_cartesian.pdb).
    gold_pdb_path : str
        Path to reference PDB.
    align_by_resid : bool
        Match residues by PDB residue ID when True.
    include_per_residue : bool
        Include per_residue_rmsd_min/max/mean in the result.

    Returns
    -------
    dict
        ca_rmsd_ang, n_res, and optionally per_residue_*.
    """
    from ..grade_folds import ca_rmsd

    rmsd_ang, per_res, pred_ca, ref_ca = ca_rmsd(
        pred_pdb_path, gold_pdb_path, align_by_resid=align_by_resid
    )
    out: Dict[str, Any] = {
        "ca_rmsd_ang": rmsd_ang,
        "n_res": len(per_res),
    }
    if include_per_residue and per_res is not None:
        out["per_residue_rmsd_min"] = float(np.min(per_res))
        out["per_residue_rmsd_max"] = float(np.max(per_res))
        out["per_residue_rmsd_mean"] = float(np.mean(per_res))
    return out
