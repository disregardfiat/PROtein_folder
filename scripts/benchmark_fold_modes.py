#!/usr/bin/env python3
"""
Compare folding time and Cα-RMSD across flag combinations (same targets as CASP grade script).

Usage (repo root):
  python3 scripts/benchmark_fold_modes.py --max-targets 2 --max-residues 100 --quick

Writes ``.casp_grade_outputs/fold_mode_benchmark.json`` by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from horizon_physics.proteins.casp_targets import CASPTarget, fetch_known_targets, ensure_experimental_ref
from horizon_physics.proteins.full_protein_minimizer import minimize_full_chain, full_chain_to_pdb
from horizon_physics.proteins.grade_folds import ca_rmsd


def _write_pdb(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def _run_profile(
    target: CASPTarget,
    *,
    cache_dir: str,
    out_dir: str,
    quick: bool,
    use_tunnel: bool,
    label: str,
    fast_local_theta: bool,
    horizon_neighbor_cutoff: Optional[float],
) -> Dict[str, Any]:
    seq = target.sequences[0] if target.sequences else ""
    t0 = time.perf_counter()
    err: Optional[str] = None
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:48]
    pred_path = os.path.join(out_dir, f"{target.target_id}_{safe_label}_pred.pdb")
    metrics: Optional[Dict[str, Any]] = None

    try:
        kwargs: Dict[str, Any] = dict(
            include_sidechains=False,
            quick=quick,
            max_iter=80 if quick else 500,
            fast_local_theta=fast_local_theta,
        )
        if horizon_neighbor_cutoff is not None:
            kwargs["horizon_neighbor_cutoff"] = float(horizon_neighbor_cutoff)
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

        result = minimize_full_chain(seq, **kwargs)
        pdb_str = full_chain_to_pdb(result)
        _write_pdb(pred_path, pdb_str)

        if target.pdb_code:
            gold_path = ensure_experimental_ref(target, cache_dir)
            if gold_path and os.path.isfile(gold_path):
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
        "profile": label,
        "target_id": target.target_id,
        "n_res": len(seq),
        "seconds": elapsed,
        "pred_pdb": pred_path,
        "error": err,
        "ca_rmsd_ang": metrics.get("ca_rmsd_ang") if metrics else None,
        "fast_local_theta": fast_local_theta,
        "horizon_neighbor_cutoff": horizon_neighbor_cutoff,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark fold flag combinations vs CASP refs")
    p.add_argument("--casp-round", default="CASP16")
    p.add_argument("--cache-dir", default=os.path.join(_REPO, ".casp_grade_cache"))
    p.add_argument("--out-dir", default=os.path.join(_REPO, ".casp_grade_outputs"))
    p.add_argument("--max-targets", type=int, default=3)
    p.add_argument("--max-residues", type=int, default=120)
    p.add_argument("--min-residues", type=int, default=20)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-tunnel", action="store_true")
    p.add_argument("--targets", default=None, help="Comma-separated target ids")
    p.add_argument(
        "--modes",
        default="baseline,fast_theta,prune8,both",
        help="Comma-separated: baseline | fast_theta | prune8 | both | prune10",
    )
    args = p.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    all_t: List[CASPTarget] = fetch_known_targets(
        casp_round=args.casp_round,
        cache_dir=args.cache_dir,
        fill_pdb_codes=True,
    )
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
        print("No targets after filter.", flush=True)
        return 1

    use_tunnel = not args.no_tunnel
    mode_map: Dict[str, Tuple[bool, Optional[float]]] = {
        "baseline": (False, None),
        "fast_theta": (True, None),
        "prune8": (False, 8.0),
        "prune10": (False, 10.0),
        "both": (True, 8.0),
    }
    want_modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    profiles: List[Tuple[str, bool, Optional[float]]] = []
    for m in want_modes:
        if m not in mode_map:
            print(f"Unknown mode '{m}', known: {list(mode_map)}", flush=True)
            return 1
        flt, hcut = mode_map[m]
        profiles.append((m, flt, hcut))

    rows: List[Dict[str, Any]] = []
    for t in filtered:
        for label, flt, hcut in profiles:
            print(f"=== {t.target_id}  mode={label} ===", flush=True)
            row = _run_profile(
                t,
                cache_dir=args.cache_dir,
                out_dir=args.out_dir,
                quick=args.quick,
                use_tunnel=use_tunnel,
                label=label,
                fast_local_theta=flt,
                horizon_neighbor_cutoff=hcut,
            )
            rows.append(row)
            if row.get("error"):
                print(f"  ERROR: {row['error']}", flush=True)
            else:
                r = row.get("ca_rmsd_ang")
                print(
                    f"  {row['seconds']:.2f}s  Cα-RMSD={r:.3f} Å"
                    if r is not None
                    else f"  {row['seconds']:.2f}s  (no grade)"
                )

    out_path = os.path.join(args.out_dir, "fold_mode_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "tunnel": use_tunnel,
                "quick": args.quick,
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out_path}", flush=True)
    return 1 if any(r.get("error") for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
