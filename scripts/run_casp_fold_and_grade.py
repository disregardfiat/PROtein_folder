#!/usr/bin/env python3
"""
Fetch CASP-style inputs from predictioncenter.org, fold with minimize_full_chain,
and grade predictions vs experimental PDBs (Cα-RMSD after Kabsch).

Grading is always attempted when a released PDB exists; each target prints a
``GRADING`` block with Cα-RMSD.

Ab-initio folding from sequence alone is graded honestly and is typically far
above 2 Å on CASP-sized targets. To validate the minimizer + grader in the
<2 Å regime in reasonable time, use ``--refine-from-gold`` (Cartesian refine
from experimental Cα; sequence is truncated to min(seq, n_Cα) when counts
differ). Combine with ``--fail-if-rmsd-above 2`` for CI.

Usage (repo root):
  python3 scripts/run_casp_fold_and_grade.py --max-targets 3 --max-residues 120 --quick
  python3 scripts/run_casp_fold_and_grade.py --targets T1235 --refine-from-gold --fail-if-rmsd-above 2

Reference benchmark (bundled crambin vs 1CRN): ``scripts/run_pipeline_grade.py``.

Requires network for first run (sequences + target list + PDB download).
Cache dir defaults to ./.casp_grade_cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Repo root
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from horizon_physics.proteins.casp_targets import (
    CASPTarget,
    fetch_known_targets,
    ensure_experimental_ref,
)
from horizon_physics.proteins.full_protein_minimizer import minimize_full_chain, full_chain_to_pdb
from horizon_physics.proteins.grade_folds import load_ca_from_pdb
from horizon_physics.proteins.ligands import parse_ligands, parse_ligands_from_file


def _write_pdb(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def run_one(
    target: CASPTarget,
    *,
    cache_dir: str,
    out_dir: str,
    quick: bool,
    use_tunnel: bool,
    fast_local_theta: bool = False,
    horizon_neighbor_cutoff: Optional[float] = None,
    refine_from_gold: bool = False,
    ligand_file: Optional[str] = None,
    ligand_smiles: Optional[str] = None,
    ligand_chain_id: Optional[str] = None,
) -> Dict[str, Any]:
    seq = target.sequences[0] if target.sequences else ""
    t0 = time.perf_counter()
    err: Optional[str] = None
    pred_path = os.path.join(out_dir, f"{target.target_id}_pred.pdb")
    gold_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    ca_init: Optional[Any] = None
    used_gold_ca_init = False
    refine_truncated_to: Optional[int] = None
    seq_fold = seq

    try:
        if refine_from_gold and target.pdb_code:
            gold_try = ensure_experimental_ref(target, cache_dir)
            if gold_try and os.path.isfile(gold_try):
                ca_xyz, _ = load_ca_from_pdb(gold_try)
                n_ca = int(ca_xyz.shape[0])
                n_seq = len(seq)
                if n_ca > 0 and n_seq > 0:
                    if n_ca != n_seq:
                        n_use = min(n_ca, n_seq)
                        ca_init = ca_xyz[:n_use]
                        seq_fold = seq[:n_use]
                        refine_truncated_to = n_use
                    else:
                        ca_init = ca_xyz
                    used_gold_ca_init = True

        kwargs = dict(
            include_sidechains=False,
            quick=quick,
            max_iter=80 if quick else 500,
        )
        if ca_init is not None:
            kwargs["ca_init"] = ca_init
            kwargs["simulate_ribosome_tunnel"] = False
        if ca_init is None:
            if use_tunnel:
                kwargs.update(
                    simulate_ribosome_tunnel=True,
                    post_extrusion_refine=True,
                    post_extrusion_max_disp_floor=0.25,
                    post_extrusion_max_rounds=16,
                    fast_pass_steps_per_connection=2 if quick else 5,
                    min_pass_iter_per_connection=5 if quick else 15,
                )
            else:
                kwargs["simulate_ribosome_tunnel"] = False

        lig_list = []
        if ligand_file:
            lig_list = parse_ligands_from_file(ligand_file)
        elif ligand_smiles:
            lig_list = parse_ligands(ligand_smiles)
        if lig_list:
            kwargs["include_ligands"] = True
            kwargs["ligands"] = lig_list
        if fast_local_theta:
            kwargs["fast_local_theta"] = True
        if horizon_neighbor_cutoff is not None:
            kwargs["horizon_neighbor_cutoff"] = float(horizon_neighbor_cutoff)

        result = minimize_full_chain(seq_fold, **kwargs)
        pdb_str = full_chain_to_pdb(result, ligand_chain_id=ligand_chain_id if lig_list else None)
        _write_pdb(pred_path, pdb_str)

        if target.pdb_code:
            gold_path = ensure_experimental_ref(target, cache_dir)
            if gold_path and os.path.isfile(gold_path):
                # Experimental auth_seq_id / chain often ≠ model numbering; use order + trim.
                from horizon_physics.proteins.grade_folds import ca_rmsd

                rmsd_ang, per_res, _, _ = ca_rmsd(
                    pred_path,
                    gold_path,
                    align_by_resid=False,
                    trim_to_min_length=True,
                )
                metrics = {
                    "ca_rmsd_ang": float(rmsd_ang),
                    "n_res": int(len(per_res)) if per_res is not None else 0,
                }
    except Exception as e:
        err = str(e)

    elapsed = time.perf_counter() - t0
    return {
        "target_id": target.target_id,
        "n_res": len(seq_fold),
        "n_res_target_full": len(seq),
        "refine_truncated_to": refine_truncated_to,
        "pdb_code": target.pdb_code,
        "seconds": elapsed,
        "pred_pdb": pred_path,
        "gold_pdb": gold_path,
        "error": err,
        "ca_rmsd_ang": metrics.get("ca_rmsd_ang") if metrics else None,
        "n_graded": metrics.get("n_res") if metrics else None,
        "fast_local_theta": bool(fast_local_theta),
        "horizon_neighbor_cutoff": horizon_neighbor_cutoff,
        "refine_from_gold": bool(refine_from_gold),
        "used_gold_ca_init": used_gold_ca_init,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="CASP fetch + fold + Cα-RMSD grade")
    p.add_argument("--casp-round", default="CASP16", help="Prediction Center round folder")
    p.add_argument("--cache-dir", default=os.path.join(_REPO, ".casp_grade_cache"))
    p.add_argument("--out-dir", default=os.path.join(_REPO, ".casp_grade_outputs"))
    p.add_argument("--max-targets", type=int, default=5)
    p.add_argument("--max-residues", type=int, default=150, help="Skip targets longer than this")
    p.add_argument("--min-residues", type=int, default=20)
    p.add_argument("--quick", action="store_true", help="Fewer minimizer iterations / tunnel steps")
    p.add_argument("--no-tunnel", action="store_true", help="Cartesian minimize_full_chain only")
    p.add_argument(
        "--fast-local-theta",
        action="store_true",
        help="Batched nearest-neighbor Θ_i in e_tot (same physics, faster on long chains)",
    )
    p.add_argument(
        "--horizon-neighbor-cutoff",
        type=float,
        default=None,
        metavar="ANG",
        help="Cap horizon neighbor-list radius (Å); prunes long pairs (faster, approximate)",
    )
    p.add_argument(
        "--targets",
        default=None,
        help="Comma-separated target ids to include (subset of fetched list), e.g. T1201,T1202",
    )
    p.add_argument(
        "--ligand-file",
        default=None,
        help="Path to ligand description (PDB HETATM block or SMILES/InChI lines); enables include_ligands",
    )
    p.add_argument(
        "--ligand-smiles",
        default=None,
        help="Inline ligand string (same as parse_ligands: PDB block or newline-separated SMILES/InChI)",
    )
    p.add_argument(
        "--ligand-chain-id",
        default=None,
        help="PDB chain ID for HETATM ligand rows (default: same as protein A)",
    )
    p.add_argument(
        "--refine-from-gold",
        action="store_true",
        help="If experimental PDB has same Cα count as CASP sequence, start minimization from those Cα "
        "(Cartesian refine; tunnel off). Graded vs same PDB — use to validate physics relaxation (<2 Å typical).",
    )
    p.add_argument(
        "--fail-if-rmsd-above",
        type=float,
        default=None,
        metavar="ANG",
        help="Exit 2 if any graded target has Cα-RMSD above this (skipped if no grade)",
    )
    args = p.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Fetching CASP targets ({args.casp_round})...", flush=True)
    try:
        all_t: List[CASPTarget] = fetch_known_targets(
            casp_round=args.casp_round,
            cache_dir=args.cache_dir,
            fill_pdb_codes=True,
        )
    except Exception as e:
        print(f"FAILED to fetch targets: {e}", flush=True)
        return 1

    want = None
    if args.targets:
        want = {x.strip().upper() for x in args.targets.split(",") if x.strip()}

    filtered: List[CASPTarget] = []
    for t in all_t:
        if not t.sequences:
            continue
        L = len(t.sequences[0])
        if L < args.min_residues or L > args.max_residues:
            continue
        if not t.pdb_code:
            continue
        if want and t.target_id.upper() not in want:
            continue
        filtered.append(t)

    filtered = filtered[: args.max_targets]
    if not filtered:
        print(
            "No targets after filter (need pdb_code, length in range). "
            "Try larger --max-residues or omit --targets.",
            flush=True,
        )
        return 1

    use_tunnel = not args.no_tunnel
    tunnel_note = "off (refine-from-gold)" if args.refine_from_gold else ("on" if use_tunnel else "off")
    print(
        f"Running {len(filtered)} target(s): {[t.target_id for t in filtered]} "
        f"(tunnel={tunnel_note}, quick={args.quick}, "
        f"fast_local_theta={args.fast_local_theta}, "
        f"horizon_cutoff={args.horizon_neighbor_cutoff})\n",
        flush=True,
    )

    rows: List[Dict[str, Any]] = []
    for t in filtered:
        print(f"=== {t.target_id} ({len(t.sequences[0])} aa) pdb={t.pdb_code} ===", flush=True)
        row = run_one(
            t,
            cache_dir=args.cache_dir,
            out_dir=args.out_dir,
            quick=args.quick,
            use_tunnel=use_tunnel,
            fast_local_theta=args.fast_local_theta,
            horizon_neighbor_cutoff=args.horizon_neighbor_cutoff,
            refine_from_gold=args.refine_from_gold,
            ligand_file=args.ligand_file,
            ligand_smiles=args.ligand_smiles,
            ligand_chain_id=args.ligand_chain_id,
        )
        rows.append(row)
        if row.get("error"):
            print(f"  ERROR: {row['error']}", flush=True)
        else:
            print("  --- GRADING (Kabsch Cα-RMSD vs experimental) ---", flush=True)
            rmsd = row.get("ca_rmsd_ang")
            if args.refine_from_gold:
                rt = row.get("refine_truncated_to")
                print(
                    f"  refine_from_gold: used_gold_ca_init={row.get('used_gold_ca_init')}"
                    + (f", truncated_to_n={rt} (Cα vs CASP seq length mismatch)" if rt is not None else ""),
                    flush=True,
                )
            print(
                f"  time={row['seconds']:.2f}s  Cα-RMSD={rmsd:.3f} Å (n={row.get('n_graded')})"
                if rmsd is not None
                else f"  time={row['seconds']:.2f}s  (no grade: missing ref?)"
            )

    summary_path = os.path.join(args.out_dir, "casp_fold_grade_summary.json")
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {summary_path}", flush=True)

    print("\n=== Summary ===")
    print(f"{'Target':<12} {'n_res':>6} {'sec':>8} {'Cα-RMSD':>10} {'gold':<6}")
    print("-" * 52)
    for row in rows:
        rmsd = row.get("ca_rmsd_ang")
        rmsd_s = f"{rmsd:.3f}" if rmsd is not None else "—"
        pc = row.get("pdb_code") or "—"
        print(
            f"{row['target_id']:<12} {row['n_res']:>6} {row['seconds']:>8.2f} {rmsd_s:>10} {str(pc):<6}"
        )
    total_t = sum(r["seconds"] for r in rows)
    print("-" * 52)
    print(f"Total wall time (sum of targets): {total_t:.2f}s")

    if any(r.get("error") for r in rows):
        return 1
    lim = args.fail_if_rmsd_above
    if lim is not None:
        for r in rows:
            rv = r.get("ca_rmsd_ang")
            if rv is not None and float(rv) > float(lim):
                print(
                    f"\nFAIL: {r.get('target_id')} Cα-RMSD={rv:.3f} Å > {lim} Å (--fail-if-rmsd-above)",
                    flush=True,
                )
                return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
