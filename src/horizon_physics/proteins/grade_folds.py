"""
Grade predicted folds against a reference structure (e.g. experimental PDB).

Provides Cα superposition (Kabsch) and Cα-RMSD. No external dependencies beyond NumPy.
Use with CASP/RCSB reference PDBs to evaluate minimize_full_chain and HKE outputs.

Usage:
  python -m horizon_physics.proteins.grade_folds pred.pdb ref.pdb
  from horizon_physics.proteins.grade_folds import ca_rmsd, load_ca_from_pdb
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_ca_from_pdb(path: str, chain_id: Optional[str] = None) -> Tuple[np.ndarray, List[int]]:
    """
    Load Cα coordinates and residue numbers from a PDB file.

    Parameters
    ----------
    path : str
        Path to PDB file.
    chain_id : str or None
        If set, only load Cα from this chain (PDB column 22).

    Returns
    -------
    ca_xyz : (N, 3) float
        Cα positions in Å.
    res_ids : list of int
        Residue sequence number for each Cα (from PDB column 23-26).
    """
    ca_xyz: List[Tuple[float, float, float]] = []
    res_ids: List[int] = []
    with open(path) as f:
        for line in f:
            if line.startswith("ATOM ") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue
                if chain_id is not None and len(line) > 21 and line[21].strip() != chain_id:
                    continue
                try:
                    res_id = int(line[22:26].strip())
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    ca_xyz.append((x, y, z))
                    res_ids.append(res_id)
                except (ValueError, IndexError):
                    continue
    return np.array(ca_xyz, dtype=np.float64), res_ids


# Three-letter to one-letter (for loading sequence from PDB)
AA_3to1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def load_ca_and_sequence_from_pdb(path: str) -> Tuple[np.ndarray, str]:
    """
    Load Cα coordinates and one-letter sequence from a PDB file (from CA lines only).
    Residue type from PDB column 17-20 (3-letter code). Unknown codes become 'X'.
    Returns (ca_xyz (N, 3), sequence).
    """
    ca_xyz: List[Tuple[float, float, float]] = []
    seq_list: List[str] = []
    with open(path) as f:
        for line in f:
            if line.startswith("ATOM ") or line.startswith("HETATM"):
                if line[12:16].strip() != "CA":
                    continue
                try:
                    res_3 = line[17:20].strip().upper()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    ca_xyz.append((x, y, z))
                    seq_list.append(AA_3to1.get(res_3, "X"))
                except (ValueError, IndexError):
                    continue
    return np.array(ca_xyz, dtype=np.float64), "".join(seq_list)


ResidueKey = Tuple[str, int, str]


def load_backbone_atoms_ordered_from_pdb(
    path: str,
    chain_id: Optional[str] = None,
    *,
    hetatm: bool = False,
) -> List[Tuple[str, np.ndarray]]:
    """
    Load backbone atoms in **N, CA, C, O** order per residue (repeat for each residue).

    Each residue must have N, CA, and C. If O is missing, O is placed at C (placeholder;
    φ/ψ extraction uses only N, CA, C for dihedrals).

    Parameters
    ----------
    path : str
        PDB path.
    chain_id : str or None
        If set, only this chain (column 22).
    hetatm : bool
        If True, also read HETATM lines (default False: ATOM only).

    Returns
    -------
    List of (atom_name, (x, y, z)) with 4 entries per residue in sequence order.
    """
    # (chain, resseq, icode) -> atom_name -> xyz
    bucket: Dict[ResidueKey, Dict[str, np.ndarray]] = {}
    with open(path) as f:
        for line in f:
            if not (line.startswith("ATOM ") or (hetatm and line.startswith("HETATM"))):
                continue
            try:
                atom_name = line[12:16].strip().upper()
                if atom_name not in ("N", "CA", "C", "O", "OXT"):
                    continue
                if atom_name == "OXT":
                    continue
                ch = line[21] if len(line) > 21 else " "
                if chain_id is not None and ch.strip() != chain_id.strip():
                    continue
                resseq = int(line[22:26].strip())
                icode = line[26].strip() if len(line) > 26 else ""
                resname = line[17:20].strip().upper()
                if resname in ("HOH", "WAT", "SOL"):
                    continue
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue
            key: ResidueKey = (ch, resseq, icode)
            bucket.setdefault(key, {})
            bucket[key][atom_name] = np.array([x, y, z], dtype=np.float64)

    if not bucket:
        raise ValueError("load_backbone_atoms_ordered_from_pdb: no matching backbone atoms.")

    def _key_sort(k: ResidueKey) -> Tuple[str, int, str]:
        return (k[0], k[1], k[2])

    ordered_keys = sorted(bucket.keys(), key=_key_sort)
    out: List[Tuple[str, np.ndarray]] = []
    for key in ordered_keys:
        atoms = bucket[key]
        if "N" not in atoms or "CA" not in atoms or "C" not in atoms:
            continue
        n_xyz = atoms["N"]
        ca_xyz = atoms["CA"]
        c_xyz = atoms["C"]
        o_xyz = atoms.get("O", c_xyz.copy())
        out.append(("N", n_xyz))
        out.append(("CA", ca_xyz))
        out.append(("C", c_xyz))
        out.append(("O", o_xyz))
    if len(out) < 4:
        raise ValueError("load_backbone_atoms_ordered_from_pdb: incomplete backbone.")
    return out


def kabsch_superpose(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find rotation R and translation t so that (R @ Q.T).T + t is best aligned to P (minimize RMSD).

    P, Q: (N, 3) arrays. Centers are moved to origin; then R is the Kabsch rotation.
    Returns (R, t, Q_aligned) where Q_aligned = (R @ Q.T).T + t has same center as P and minimum RMSD to P.
    """
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    Q = np.asarray(Q, dtype=np.float64).reshape(-1, 3)
    n = P.shape[0]
    assert Q.shape[0] == n, "P and Q must have same number of points"
    cen_P = np.mean(P, axis=0)
    cen_Q = np.mean(Q, axis=0)
    P_centered = P - cen_P
    Q_centered = Q - cen_Q
    H = Q_centered.T @ P_centered
    U, _, Vt = np.linalg.svd(H)
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = (U @ Vt).T
    t = cen_P - (R @ cen_Q)
    Q_aligned = (R @ Q.T).T + t
    return R, t, Q_aligned


def ca_rmsd(
    pred_path: str,
    ref_path: str,
    align_by_resid: bool = True,
    ref_chain_id: Optional[str] = None,
    trim_to_min_length: bool = False,
) -> Tuple[float, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Compute Cα-RMSD between a predicted PDB and a reference PDB after Kabsch superposition.

    Parameters
    ----------
    pred_path : str
        Path to predicted PDB (e.g. *_minimized_cartesian.pdb or *_minimized_hierarchical.pdb).
    ref_path : str
        Path to reference PDB (e.g. experimental or CASP model).
    align_by_resid : bool
        If True, match Cα by residue ID (ref and pred must have same residue numbers).
        If False, assume same order and length (no residue-number matching).
    ref_chain_id : str or None
        If set, load only this chain from the reference PDB (e.g. "B" for 4INS B-chain).
    trim_to_min_length : bool
        If True and align_by_resid is False, trim pred and ref to min length (first N residues).

    Returns
    -------
    rmsd_ang : float
        Cα-RMSD in Å after superposition.
    per_residue_rmsd : (N,) or None
        Per-Cα distance (Å) after superposition; None if lengths differ and align_by_resid True.
    pred_ca : (N, 3)
        Predicted Cα after superposition (for optional export).
    ref_ca : (N, 3)
        Reference Cα (same order as pred_ca).
    """
    pred_ca, pred_res = load_ca_from_pdb(pred_path)
    ref_ca, ref_res = load_ca_from_pdb(ref_path, chain_id=ref_chain_id)
    if align_by_resid:
        ref_res_to_idx = {r: i for i, r in enumerate(ref_res)}
        common = [r for r in pred_res if r in ref_res_to_idx]
        if not common:
            raise ValueError("No common residue IDs between pred and ref.")
        pred_idx = [i for i, r in enumerate(pred_res) if r in ref_res_to_idx]
        ref_idx = [ref_res_to_idx[r] for r in pred_res if r in ref_res_to_idx]
        pred_ca = pred_ca[pred_idx]
        ref_ca = ref_ca[ref_idx]
    else:
        if len(pred_ca) != len(ref_ca):
            if trim_to_min_length:
                n = min(len(pred_ca), len(ref_ca))
                pred_ca = pred_ca[:n]
                ref_ca = ref_ca[:n]
            else:
                raise ValueError(
                    "Pred has {} Cα, ref has {} Cα; use align_by_resid=True if residue IDs match, or trim_to_min_length=True.".format(
                        len(pred_ca), len(ref_ca)
                    )
                )
    _, _, pred_aligned = kabsch_superpose(ref_ca, pred_ca)
    diff = pred_aligned - ref_ca
    per_res = np.sqrt(np.sum(diff ** 2, axis=1))
    rmsd_ang = float(np.sqrt(np.mean(per_res ** 2)))
    return rmsd_ang, per_res, pred_aligned, ref_ca


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python -m horizon_physics.proteins.grade_folds <pred.pdb> <ref.pdb> [--no-resid]")
        print("  Compute Cα-RMSD after Kabsch superposition. Use --no-resid to align by order (same length).")
        sys.exit(1)
    pred_path = sys.argv[1]
    ref_path = sys.argv[2]
    align_by_resid = "--no-resid" not in sys.argv
    try:
        rmsd, per_res, _, _ = ca_rmsd(pred_path, ref_path, align_by_resid=align_by_resid)
        print("Cα-RMSD: {:.3f} Å".format(rmsd))
        if per_res is not None and len(per_res) <= 20:
            print("Per-residue Cα distance (Å):", np.round(per_res, 3).tolist())
        elif per_res is not None:
            print("Per-residue: min={:.3f} max={:.3f} Å".format(float(np.min(per_res)), float(np.max(per_res))))
    except Exception as e:
        print("Error:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
