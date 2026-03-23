#!/usr/bin/env python3
"""
Run the folding pipeline and **always** grade against a reference (Cα-RMSD after Kabsch).

Modes:
  crambin-native   — Refine experimental Cα with ``minimize_full_chain`` (validates physics + grader).
                     Typical: <0.5 Å in a few seconds (meets <2 Å goal).
  crambin-abinitio — Lean tunnel fold from sequence only; graded vs 1CRN (honest ab-initio baseline).

CASP grading (fetch + fold + grade) remains in ``scripts/run_casp_fold_and_grade.py``.
For CASP targets where seq length ≠ experimental Cα count, use ``--refine-from-gold`` there
(minimizer truncates to min length).

Usage (repo root):
  python3 scripts/run_pipeline_grade.py crambin-native
  python3 scripts/run_pipeline_grade.py crambin-abinitio
  python3 scripts/run_pipeline_grade.py all --json-out .casp_grade_outputs/pipeline_grade.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from horizon_physics.proteins.full_protein_minimizer import full_chain_to_pdb, minimize_full_chain
from horizon_physics.proteins.grade_folds import ca_rmsd, load_ca_and_sequence_from_pdb
from horizon_physics.proteins.hqiv_lean_folding import PHYSIOLOGICAL_PH
from horizon_physics.proteins.lean_ribosome_tunnel_pipeline import fold_lean_ribosome_tunnel

CRAMBIN_GOLD = os.path.join(
    _REPO, "src", "horizon_physics", "proteins", "examples", "crambin_1CRN.pdb"
)
CRAMBIN_SEQ = "TTCCPSIVARSNFNVCRLPGTPEAIICGDVCDLDCTAKTCFSIICT"


def _grade_pdb_str(pdb_str: str, gold_path: str) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False, encoding="utf-8") as tmp:
        tmp.write(pdb_str)
        pred_path = tmp.name
    try:
        rmsd, per_res, _, _ = ca_rmsd(
            pred_path,
            gold_path,
            align_by_resid=False,
            trim_to_min_length=True,
        )
        return {
            "ca_rmsd_ang": float(rmsd),
            "n_graded": int(len(per_res)) if per_res is not None else 0,
        }
    finally:
        try:
            os.unlink(pred_path)
        except OSError:
            pass


def run_crambin_native(*, quick: bool, fast_local_theta: bool) -> Dict[str, Any]:
    ca0, seq_ref = load_ca_and_sequence_from_pdb(CRAMBIN_GOLD)
    t0 = time.perf_counter()
    raw = minimize_full_chain(
        seq_ref,
        ca_init=ca0,
        simulate_ribosome_tunnel=False,
        quick=quick,
        max_iter=80 if quick else 500,
        fast_local_theta=fast_local_theta,
    )
    elapsed = time.perf_counter() - t0
    pdb_str = full_chain_to_pdb(raw)
    g = _grade_pdb_str(pdb_str, CRAMBIN_GOLD)
    return {
        "workflow": "crambin-native",
        "seconds": elapsed,
        "ca_rmsd_ang": g["ca_rmsd_ang"],
        "n_graded": g["n_graded"],
        "under_2A": g["ca_rmsd_ang"] < 2.0,
    }


def run_crambin_abinitio(*, quick: bool, fast_local_theta: bool) -> Dict[str, Any]:
    kw = dict(
        temperature_k=310.0,
        ph=float(PHYSIOLOGICAL_PH),
        kappa_dihedral=0.01,
        post_extrusion_max_rounds=12 if quick else 32,
        fast_pass_steps_per_connection=2 if quick else 5,
        min_pass_iter_per_connection=5 if quick else 15,
        hbond_weight=0.0,
        hbond_shell_m=3,
        simulate_ribosome_tunnel=True,
        post_extrusion_refine=True,
        quick=quick,
        fast_local_theta=fast_local_theta,
        include_ligands=False,
    )
    t0 = time.perf_counter()
    out = fold_lean_ribosome_tunnel(CRAMBIN_SEQ, **kw)
    elapsed = time.perf_counter() - t0
    g = _grade_pdb_str(out.pdb, CRAMBIN_GOLD)
    return {
        "workflow": "crambin-abinitio",
        "seconds": elapsed,
        "ca_rmsd_ang": g["ca_rmsd_ang"],
        "n_graded": g["n_graded"],
        "under_2A": g["ca_rmsd_ang"] < 2.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Fold + mandatory Cα-RMSD grading (reference benchmark).")
    ap.add_argument(
        "mode",
        choices=("crambin-native", "crambin-abinitio", "all"),
        help="Which graded workflow to run",
    )
    ap.add_argument("--quick", action="store_true", help="Fewer minimizer / tunnel steps")
    ap.add_argument("--fast-local-theta", action="store_true", help="Faster e_tot local Θ evaluation")
    ap.add_argument("--json-out", type=str, default=None, help="Write results JSON to this path")
    ap.add_argument(
        "--require-under-2A",
        action="store_true",
        help="Exit 2 unless every workflow reports ca_rmsd_ang < 2 (ab-initio usually fails)",
    )
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    modes = ["crambin-native", "crambin-abinitio"] if args.mode == "all" else [args.mode]
    for m in modes:
        if m == "crambin-native":
            rows.append(run_crambin_native(quick=args.quick, fast_local_theta=args.fast_local_theta))
        else:
            rows.append(run_crambin_abinitio(quick=args.quick, fast_local_theta=args.fast_local_theta))

    print("=== Pipeline + grading (Kabsch Cα-RMSD) ===")
    for r in rows:
        u = "yes" if r.get("under_2A") else "no"
        print(
            f"  {r['workflow']:<22}  time={r['seconds']:.2f}s  "
            f"Cα-RMSD={r['ca_rmsd_ang']:.3f} Å  n={r['n_graded']}  <2Å? {u}"
        )

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Wrote {args.json_out}")

    if args.require_under_2A and not all(r.get("under_2A") for r in rows):
        print("FAIL: --require-under-2A and at least one workflow is ≥2 Å Cα-RMSD.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
