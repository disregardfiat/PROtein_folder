"""
Grade trajectory logs (JSONL) against a gold-standard PDB.

Consumes the translation logs from --trajectory-log and a reference PDB to produce
per-frame Cα-RMSD (and optional stats) for ML: e.g. convergence curves, reward
signals, or dataset generation. Supports both Cartesian (n_res, 3) and HKE
(n_res, 6) trajectory formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Optional pandas for DataFrame and CSV export (ML pipelines)
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore
    _HAS_PANDAS = False


def _ca_from_cartesian_frame(positions: Union[List, np.ndarray]) -> np.ndarray:
    """Positions (n_res, 3) or (n_res*4, 3) → Cα (n_res, 3). Assumes (n_res, 3) is already Cα-only."""
    P = np.asarray(positions, dtype=np.float64)
    if P.ndim == 1:
        P = P.reshape(-1, 3)
    n = P.shape[0]
    # Cartesian pipeline logs Cα-only (n_res, 3); hierarchical after conversion is (n_res*4, 3) N,CA,C,O
    if n % 4 == 0 and n >= 4:
        n_res = n // 4
        # N, CA, C, O per residue → CA at index 1, 5, 9, ...
        ca = P[1::4]
        return ca
    return P


def _ca_from_6dof_frame(positions: Union[List, List[List]], n_res: int) -> np.ndarray:
    """One frame with positions = list of 6-element lists (n_res, 6) → Cα (n_res, 3) via 6-DOF→backbone→CA."""
    try:
        from ..hierarchical import relative_6dof_to_world_backbone
    except ImportError:
        from ..hierarchical.minimize_hierarchical import relative_6dof_to_world_backbone
    dofs = np.asarray(positions, dtype=np.float64)
    if dofs.ndim == 1:
        dofs = dofs.reshape(n_res, 6)
    backbone = relative_6dof_to_world_backbone(dofs)  # (n_res*4, 3)
    ca = backbone[1::4]
    return ca


def load_trajectory_frames(
    path: str,
    n_res: Optional[int] = None,
) -> List[Tuple[int, np.ndarray, str]]:
    """
    Load trajectory JSONL; yield (frame_index, ca_xyz (n_res, 3), format).

    Format is "cartesian" (Cα-only or backbone) or "6dof" (HKE). If n_res is None
    and the first frame is 6-DOF, n_res is inferred from len(positions).
    """
    path = Path(path)
    frames: List[Tuple[int, np.ndarray, str]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            t = rec.get("t", len(frames))
            pos = rec.get("positions", [])
            if not pos:
                continue
            # Detect format: list of 6-element lists → 6dof
            first = pos[0] if pos else []
            if isinstance(first, (list, tuple)) and len(first) == 6:
                if n_res is None:
                    n_res = len(pos)
                ca = _ca_from_6dof_frame(pos, n_res)
                frames.append((int(t), ca, "6dof"))
            else:
                ca = _ca_from_cartesian_frame(pos)
                if n_res is None:
                    n_res = ca.shape[0]
                frames.append((int(t), np.asarray(ca, dtype=np.float64), "cartesian"))
    return frames


def grade_trajectory(
    traj_path: str,
    gold_pdb_path: str,
    n_res: Optional[int] = None,
    output_path: Optional[str] = None,
    output_format: str = "csv",
) -> Union[List[Dict[str, Any]], "pd.DataFrame"]:
    """
    Grade each frame of a trajectory JSONL against a gold-standard PDB.

    Loads gold Cα from the PDB, loads trajectory frames (Cartesian or 6-DOF),
    superposes each frame to gold (Kabsch), and records per-frame Cα-RMSD.
    Optionally writes results to CSV or JSON for ML pipelines. Residues are
    aligned by order (trajectory has no residue IDs).

    Parameters
    ----------
    traj_path : str
        Path to trajectory JSONL (from --trajectory-log).
    gold_pdb_path : str
        Path to reference PDB (e.g. RCSB or CASP).
    n_res : int, optional
        Number of residues (inferred from first frame if None).
    output_path : str, optional
        If set, write results to this path (CSV or JSON by output_format).
    output_format : str
        "csv" or "json". CSV requires pandas.

    Returns
    -------
    results : list of dict or DataFrame
        Per-frame: frame, rmsd_ang, per_residue_rmsd_min, max, mean. DataFrame if pandas.
    """
    from ..grade_folds import load_ca_from_pdb, kabsch_superpose

    gold_ca, gold_res = load_ca_from_pdb(gold_pdb_path)
    frames = load_trajectory_frames(traj_path, n_res=n_res)
    if not frames:
        raise ValueError("No frames in trajectory: {}".format(traj_path))

    n_res = frames[0][1].shape[0]
    # Trajectory JSONL has no residue IDs; align by order. Trim gold to trajectory length if needed.
    if gold_ca.shape[0] < n_res:
        raise ValueError("Gold has fewer Cα ({}) than trajectory ({}).".format(gold_ca.shape[0], n_res))
    gold_ca = gold_ca[:n_res]

    results: List[Dict[str, Any]] = []
    for t, ca, fmt in frames:
        if ca.shape[0] != gold_ca.shape[0]:
            continue
        _, _, pred_aligned = kabsch_superpose(gold_ca, ca)
        diff = pred_aligned - gold_ca
        per_res = np.sqrt(np.sum(diff ** 2, axis=1))
        rmsd_ang = float(np.sqrt(np.mean(per_res ** 2)))
        results.append({
            "frame": t,
            "rmsd_ang": rmsd_ang,
            "format": fmt,
            "per_residue_rmsd_min": float(np.min(per_res)),
            "per_residue_rmsd_max": float(np.max(per_res)),
            "per_residue_rmsd_mean": float(np.mean(per_res)),
        })

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if output_format == "csv" and _HAS_PANDAS:
            df = pd.DataFrame(results)
            df.to_csv(out, index=False)
        elif output_format == "json":
            with open(out, "w") as f:
                json.dump(results, f, indent=2)
        else:
            with open(out, "w") as f:
                json.dump(results, f, indent=2)

    if _HAS_PANDAS:
        return pd.DataFrame(results)
    return results
