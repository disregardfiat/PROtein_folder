#!/usr/bin/env python3
"""
Run ``minimize_ca_with_osh_oracle`` with Lean HQIV-native sparse gates
(``use_hqiv_native_gate``) on repo test PDBs and report Cα RMSD vs experimental structures.

Uses the same convention as ``Hqiv/QuantumComputing/OSHoracleHQIVNative.lean``:
pivot from per-residue ``z`` (shell) and ``referenceM``; π phase on one harmonic mode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from horizon_physics.proteins.full_protein_minimizer import _place_full_backbone, full_chain_to_pdb
from horizon_physics.proteins.grade_folds import ca_rmsd, load_ca_from_pdb
from horizon_physics.proteins.osh_oracle_folding import REFERENCE_M_HQIV_NATIVE, minimize_ca_with_osh_oracle


def _run_one(
    target_pdb: str,
    gold_pdb: str,
    *,
    z_shell: int,
    n_iter: int,
    step_size: float,
    ansatz_depth: int,
    gate_mix: float,
) -> Dict[str, Any]:
    ca, _ = load_ca_from_pdb(target_pdb)
    ca0 = np.asarray(ca, dtype=float)
    seq_len = int(ca0.shape[0])
    sequence = "A" * seq_len

    t0 = time.perf_counter()
    ca_min, info = minimize_ca_with_osh_oracle(
        ca0,
        z_shell=int(z_shell),
        n_iter=int(n_iter),
        step_size=float(step_size),
        ansatz_depth=int(ansatz_depth),
        gate_mix=float(gate_mix),
        use_energy_reservoir=False,
        use_hqiv_native_gate=True,
        hqiv_reference_m=int(REFERENCE_M_HQIV_NATIVE),
    )
    wall = float(time.perf_counter() - t0)

    backbone = _place_full_backbone(ca_min, sequence)
    obj = {
        "ca_min": ca_min,
        "backbone_atoms": backbone,
        "sequence": sequence,
        "n_res": seq_len,
    }
    pdb_str = full_chain_to_pdb(obj)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False, encoding="utf-8") as pf:
        pf.write(pdb_str)
        pred_path = pf.name
    try:
        rmsd, *_ = ca_rmsd(pred_path, gold_pdb, align_by_resid=False, trim_to_min_length=True)
    finally:
        os.unlink(pred_path)

    pivot = (int(z_shell) * seq_len + int(REFERENCE_M_HQIV_NATIVE)) % (seq_len + 1)

    return {
        "target_pdb": os.path.abspath(target_pdb),
        "gold_pdb": os.path.abspath(gold_pdb),
        "n_res": seq_len,
        "z_shell": int(z_shell),
        "hqiv_reference_m": int(REFERENCE_M_HQIV_NATIVE),
        "hqiv_pivot_mod_L_plus_1": int(pivot),
        "n_iter": int(n_iter),
        "final_energy_ev": float(info.final_energy_ev),
        "accepted_steps": int(info.accepted_steps),
        "stop_reason": str(info.stop_reason),
        "wall_seconds": wall,
        "ca_rmsd_ang": float(rmsd),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Grade HQIV-native OSHoracle Cα minimization vs PDB.")
    ap.add_argument(
        "--targets",
        nargs="*",
        default=[
            "proteins/1CRN.pdb",
            "proteins/1VII.pdb",
        ],
        help="Initial Cα PDB paths (default: small test proteins).",
    )
    ap.add_argument("--gold-suffix", default=None, help="Override gold path (default: same as target).")
    ap.add_argument("--z-shell", type=int, default=6)
    ap.add_argument("--n-iter", type=int, default=120)
    ap.add_argument("--step-size", type=float, default=0.025)
    ap.add_argument("--ansatz-depth", type=int, default=2)
    ap.add_argument("--gate-mix", type=float, default=0.55)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    for rel in args.targets:
        path = rel if os.path.isabs(rel) else os.path.join(REPO, rel)
        gold = args.gold_suffix or path
        if not os.path.isfile(path):
            print(f"SKIP missing {path}", file=sys.stderr)
            continue
        rows.append(_run_one(path, gold, z_shell=args.z_shell, n_iter=args.n_iter, step_size=args.step_size, ansatz_depth=args.ansatz_depth, gate_mix=args.gate_mix))

    out_path = args.out_json
    if out_path is None:
        out_path = os.path.join(REPO, ".hqiv_native_osh_grade.json")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {"runs": rows, "lean_module": "Hqiv/QuantumComputing/OSHoracleHQIVNative.lean"}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"WROTE {out_path}")
    for r in rows:
        print(
            f"n={r['n_res']:3d}  RMSD={r['ca_rmsd_ang']:.3f} Å  E={r['final_energy_ev']:.2f} eV  "
            f"accepts={r['accepted_steps']}  pivot={r['hqiv_pivot_mod_L_plus_1']}  {os.path.basename(r['target_pdb'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
