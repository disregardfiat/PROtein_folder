#!/usr/bin/env python3
"""
Analyze where a predicted fold diverges from a reference structure.

Outputs:
- Global Cα RMSD (after Kabsch alignment)
- Worst per-residue Cα deviations
- Local geometry mismatch (virtual bond angle / pseudo-torsion)
- Contact-map disagreement (precision / recall / F1)

Usage:
  python3 scripts/analyze_fold_mismatch.py --pred path/to/pred.pdb --ref path/to/ref.pdb
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from horizon_physics.proteins.grade_folds import ca_rmsd, load_ca_from_pdb


def _safe_unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def _virtual_bond_angles(ca: np.ndarray) -> np.ndarray:
    """Angles at i for triplets (i-1,i,i+1), in radians, length n-2."""
    n = ca.shape[0]
    if n < 3:
        return np.zeros((0,), dtype=float)
    out = np.zeros((n - 2,), dtype=float)
    for i in range(1, n - 1):
        u = _safe_unit(ca[i - 1] - ca[i])
        v = _safe_unit(ca[i + 1] - ca[i])
        c = float(np.clip(np.dot(u, v), -1.0, 1.0))
        out[i - 1] = float(np.arccos(c))
    return out


def _pseudo_torsions(ca: np.ndarray) -> np.ndarray:
    """Dihedral-like torsion on quadruplets (i-1,i,i+1,i+2), radians, length n-3."""
    n = ca.shape[0]
    if n < 4:
        return np.zeros((0,), dtype=float)
    out = np.zeros((n - 3,), dtype=float)
    for i in range(n - 3):
        p0, p1, p2, p3 = ca[i], ca[i + 1], ca[i + 2], ca[i + 3]
        b0 = p1 - p0
        b1 = p2 - p1
        b2 = p3 - p2
        n1 = np.cross(b0, b1)
        n2 = np.cross(b1, b2)
        n1u = _safe_unit(n1)
        n2u = _safe_unit(n2)
        m1 = np.cross(n1u, _safe_unit(b1))
        x = float(np.dot(n1u, n2u))
        y = float(np.dot(m1, n2u))
        out[i] = float(np.arctan2(y, x))
    return out


def _contact_pairs(ca: np.ndarray, cutoff: float, min_seq_sep: int = 3) -> set[Tuple[int, int]]:
    n = ca.shape[0]
    pairs: set[Tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + min_seq_sep, n):
            d = float(np.linalg.norm(ca[j] - ca[i]))
            if d <= cutoff:
                pairs.add((i, j))
    return pairs


def analyze(pred_path: str, ref_path: str, contact_cutoff: float = 8.0) -> Dict[str, Any]:
    rmsd, per_res, pred_aligned, ref_ca = ca_rmsd(
        pred_path,
        ref_path,
        align_by_resid=False,
        trim_to_min_length=True,
    )
    per_res = np.asarray(per_res if per_res is not None else [], dtype=float)
    n = int(per_res.shape[0])

    pred_ca_raw, pred_res_ids = load_ca_from_pdb(pred_path)
    ref_ca_raw, ref_res_ids = load_ca_from_pdb(ref_path)
    n_trim = min(pred_ca_raw.shape[0], ref_ca_raw.shape[0])
    pred_res_ids = pred_res_ids[:n_trim]
    ref_res_ids = ref_res_ids[:n_trim]

    # local geometry
    ang_pred = _virtual_bond_angles(pred_aligned)
    ang_ref = _virtual_bond_angles(ref_ca)
    ang_err = np.abs(ang_pred - ang_ref)

    tor_pred = _pseudo_torsions(pred_aligned)
    tor_ref = _pseudo_torsions(ref_ca)
    tor_diff = np.abs(tor_pred - tor_ref)
    tor_err = np.minimum(tor_diff, 2.0 * np.pi - tor_diff)

    # contact-map agreement
    pred_contacts = _contact_pairs(pred_aligned, contact_cutoff)
    ref_contacts = _contact_pairs(ref_ca, contact_cutoff)
    inter = pred_contacts.intersection(ref_contacts)
    precision = (len(inter) / len(pred_contacts)) if pred_contacts else 0.0
    recall = (len(inter) / len(ref_contacts)) if ref_contacts else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    worst_idx = np.argsort(per_res)[::-1][:10]
    worst_rows: List[Dict[str, Any]] = []
    for k in worst_idx:
        i = int(k)
        worst_rows.append(
            {
                "idx0": i,
                "pred_resid": int(pred_res_ids[i]) if i < len(pred_res_ids) else None,
                "ref_resid": int(ref_res_ids[i]) if i < len(ref_res_ids) else None,
                "ca_error_ang": float(per_res[i]),
            }
        )

    return {
        "pred_path": pred_path,
        "ref_path": ref_path,
        "n_aligned": n,
        "ca_rmsd_ang": float(rmsd),
        "ca_error_mean_ang": float(np.mean(per_res)) if n > 0 else 0.0,
        "ca_error_median_ang": float(np.median(per_res)) if n > 0 else 0.0,
        "ca_error_p90_ang": float(np.percentile(per_res, 90)) if n > 0 else 0.0,
        "virtual_angle_mae_deg": float(np.degrees(np.mean(ang_err))) if ang_err.size > 0 else 0.0,
        "pseudo_torsion_mae_deg": float(np.degrees(np.mean(tor_err))) if tor_err.size > 0 else 0.0,
        "contact_cutoff_ang": float(contact_cutoff),
        "contact_precision": float(precision),
        "contact_recall": float(recall),
        "contact_f1": float(f1),
        "n_pred_contacts": int(len(pred_contacts)),
        "n_ref_contacts": int(len(ref_contacts)),
        "n_shared_contacts": int(len(inter)),
        "worst_residues": worst_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze fold mismatch vs reference.")
    ap.add_argument("--pred", required=True, help="Predicted PDB path")
    ap.add_argument("--ref", required=True, help="Reference PDB path")
    ap.add_argument("--contact-cutoff", type=float, default=8.0, help="Contact cutoff in Angstrom")
    ap.add_argument("--json-out", default=None, help="Optional output JSON path")
    args = ap.parse_args()

    out = analyze(args.pred, args.ref, contact_cutoff=float(args.contact_cutoff))
    print("=== Fold mismatch analysis ===")
    print(f"Cα-RMSD: {out['ca_rmsd_ang']:.3f} Å  (n={out['n_aligned']})")
    print(
        f"Cα error mean/median/p90: {out['ca_error_mean_ang']:.3f} / "
        f"{out['ca_error_median_ang']:.3f} / {out['ca_error_p90_ang']:.3f} Å"
    )
    print(
        f"Local geometry MAE: virtual-angle={out['virtual_angle_mae_deg']:.2f}°, "
        f"pseudo-torsion={out['pseudo_torsion_mae_deg']:.2f}°"
    )
    print(
        f"Contact agreement @{out['contact_cutoff_ang']:.1f}Å: "
        f"P={out['contact_precision']:.3f} R={out['contact_recall']:.3f} F1={out['contact_f1']:.3f} "
        f"(shared={out['n_shared_contacts']}, pred={out['n_pred_contacts']}, ref={out['n_ref_contacts']})"
    )
    print("Worst residues by Cα error:")
    for r in out["worst_residues"]:
        print(
            f"  idx={r['idx0']:>3}  pred_resid={r['pred_resid']}  ref_resid={r['ref_resid']}  "
            f"err={r['ca_error_ang']:.3f} Å"
        )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

