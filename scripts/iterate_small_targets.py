#!/usr/bin/env python3
"""
Iterative improvement harness: start with smallest released CASP targets, benchmark configs,
promote winners, and expand.

Why this script:
  - CASP1/2 are not returned by the current Prediction Center endpoint used in this repo.
  - We still follow the same strategy on small released targets (default CASP16).

Usage:
  python3 scripts/iterate_small_targets.py --small-count 3 --max-residues 140 --rounds 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from horizon_physics.proteins.casp_targets import CASPTarget, ensure_experimental_ref, fetch_known_targets
from horizon_physics.proteins.full_protein_minimizer import full_chain_to_pdb, minimize_full_chain
from horizon_physics.proteins.grade_folds import ca_rmsd


def _seed_configs() -> List[Tuple[str, Dict[str, Any]]]:
    return [
        (
            "baseline_quick_tunnel",
            {
                "quick": True,
                "simulate_ribosome_tunnel": True,
                "include_sidechains": False,
                "fast_local_theta": True,
                "horizon_neighbor_cutoff": None,
                "kappa_dihedral": 0.01,
                "hbond_weight": 0.0,
                "fast_pass_steps_per_connection": 2,
                "min_pass_iter_per_connection": 5,
                "post_extrusion_refine": True,
                "post_extrusion_max_rounds": 12,
            },
        ),
        (
            "quick_notunnel",
            {
                "quick": True,
                "simulate_ribosome_tunnel": False,
                "include_sidechains": False,
                "fast_local_theta": True,
                "horizon_neighbor_cutoff": 10.0,
                "kappa_dihedral": 0.01,
                "hbond_weight": 0.0,
            },
        ),
        (
            "refine_tunnel",
            {
                "quick": False,
                "simulate_ribosome_tunnel": True,
                "include_sidechains": False,
                "fast_local_theta": True,
                "horizon_neighbor_cutoff": None,
                "kappa_dihedral": 0.01,
                "hbond_weight": 0.0,
                "fast_pass_steps_per_connection": 5,
                "min_pass_iter_per_connection": 15,
                "post_extrusion_refine": True,
                "post_extrusion_refine_mode": "em_treetorque",
                "post_extrusion_treetorque_phases": 8,
                "post_extrusion_treetorque_n_steps": 200,
            },
        ),
        (
            "refine_tunnel_kink",
            {
                "quick": False,
                "simulate_ribosome_tunnel": True,
                "include_sidechains": False,
                "fast_local_theta": True,
                "horizon_neighbor_cutoff": 10.0,
                "kappa_dihedral": 0.01,
                "hbond_weight": 0.0,
                "collective_kink_weight": 0.005,
                "collective_kink_m": 3,
                "collective_kink_use_ss_mask": True,
                "fast_pass_steps_per_connection": 5,
                "min_pass_iter_per_connection": 15,
                "post_extrusion_refine": True,
                "post_extrusion_refine_mode": "em_treetorque",
                "post_extrusion_treetorque_phases": 8,
                "post_extrusion_treetorque_n_steps": 200,
            },
        ),
    ]


def _mutate_configs(top: List[Tuple[str, Dict[str, Any]]]) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    for name, kw in top:
        out.append((name, dict(kw)))
        for hcut in (None, 12.0, 10.0, 8.0):
            k2 = dict(kw)
            k2["horizon_neighbor_cutoff"] = hcut
            out.append((f"{name}_h{hcut}", k2))
        for ck in (0.0, 0.0025, 0.005, 0.01):
            k2 = dict(kw)
            k2["collective_kink_weight"] = ck
            k2["collective_kink_m"] = 3
            k2["collective_kink_use_ss_mask"] = True
            out.append((f"{name}_ck{ck}", k2))
    # de-duplicate by sorted kwargs
    seen = set()
    uniq: List[Tuple[str, Dict[str, Any]]] = []
    for n, k in out:
        key = tuple(sorted(k.items()))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((n, k))
    return uniq


def _grade(pred_pdb: str, gold_pdb: str) -> Tuple[float, int]:
    r, per, _, _ = ca_rmsd(pred_pdb, gold_pdb, align_by_resid=False, trim_to_min_length=True)
    return float(r), int(len(per)) if per is not None else 0


def _run_cfg_on_target(t: CASPTarget, gold_pdb: str, name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    seq = t.sequences[0]
    t0 = time.perf_counter()
    try:
        res = minimize_full_chain(seq, **kwargs)
        pdb = full_chain_to_pdb(res)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False, encoding="utf-8") as tmp:
            tmp.write(pdb)
            pred = tmp.name
        try:
            rmsd, n_g = _grade(pred, gold_pdb)
        finally:
            try:
                os.unlink(pred)
            except OSError:
                pass
        return {
            "target_id": t.target_id,
            "config": name,
            "seconds": time.perf_counter() - t0,
            "ca_rmsd_ang": rmsd,
            "n_graded": n_g,
            "error": None,
            "kwargs": kwargs,
        }
    except Exception as e:
        return {
            "target_id": t.target_id,
            "config": name,
            "seconds": time.perf_counter() - t0,
            "ca_rmsd_ang": None,
            "n_graded": None,
            "error": str(e),
            "kwargs": kwargs,
        }


def _aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_cfg: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        cfg = r["config"]
        agg = by_cfg.setdefault(
            cfg,
            {"config": cfg, "n": 0, "n_ok": 0, "mean_rmsd": 0.0, "mean_seconds": 0.0, "kwargs": r.get("kwargs")},
        )
        agg["n"] += 1
        agg["mean_seconds"] += float(r.get("seconds", 0.0))
        if r.get("ca_rmsd_ang") is not None:
            agg["n_ok"] += 1
            agg["mean_rmsd"] += float(r["ca_rmsd_ang"])
    out: List[Dict[str, Any]] = []
    for a in by_cfg.values():
        n = max(1, a["n"])
        a["mean_seconds"] /= n
        if a["n_ok"] > 0:
            a["mean_rmsd"] /= a["n_ok"]
        else:
            a["mean_rmsd"] = 1e9
        out.append(a)
    out.sort(key=lambda x: (x["mean_rmsd"], x["mean_seconds"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Iterative small-target improvement loop for CASP folding.")
    ap.add_argument("--casp-round", default="CASP16")
    ap.add_argument("--cache-dir", default=os.path.join(_REPO, ".casp_grade_cache"))
    ap.add_argument("--out-dir", default=os.path.join(_REPO, ".casp_grade_outputs", "iter_small"))
    ap.add_argument("--small-count", type=int, default=3, help="Use this many smallest graded targets each round")
    ap.add_argument("--min-residues", type=int, default=20)
    ap.add_argument("--max-residues", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=2, help="Mutation rounds")
    ap.add_argument("--top-k", type=int, default=4, help="Promote this many configs each round")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    all_t = fetch_known_targets(casp_round=args.casp_round, cache_dir=args.cache_dir, fill_pdb_codes=True)
    cand: List[CASPTarget] = []
    for t in all_t:
        if not t.sequences or not t.pdb_code:
            continue
        L = len(t.sequences[0])
        if L < args.min_residues or L > args.max_residues:
            continue
        cand.append(t)
    cand.sort(key=lambda x: len(x.sequences[0]))
    targets = cand[: args.small_count]
    if not targets:
        print("No graded small targets found.", flush=True)
        return 1

    gold_map: Dict[str, str] = {}
    for t in targets:
        gp = ensure_experimental_ref(t, args.cache_dir)
        if gp and os.path.isfile(gp):
            gold_map[t.target_id] = gp
    targets = [t for t in targets if t.target_id in gold_map]
    if not targets:
        print("No usable references for selected small targets.", flush=True)
        return 1

    cfgs = _seed_configs()
    history: List[Dict[str, Any]] = []
    print(f"Using targets: {[f'{t.target_id}({len(t.sequences[0])})' for t in targets]}", flush=True)

    for rnd in range(1, args.rounds + 1):
        rows: List[Dict[str, Any]] = []
        print(f"\n=== Round {rnd} configs={len(cfgs)} ===", flush=True)
        for t in targets:
            gp = gold_map[t.target_id]
            for name, kw in cfgs:
                r = _run_cfg_on_target(t, gp, name, kw)
                rows.append(r)
                if r.get("error"):
                    print(f"  {t.target_id:>6}  {name:<28}  ERROR: {r['error']}", flush=True)
                else:
                    print(
                        f"  {t.target_id:>6}  {name:<28}  {r['seconds']:7.2f}s  RMSD={r['ca_rmsd_ang']:.3f}",
                        flush=True,
                    )
        agg = _aggregate(rows)
        top = agg[: max(1, args.top_k)]
        print("\nRound leaderboard:", flush=True)
        for i, a in enumerate(top, start=1):
            print(
                f"  {i:>2}. {a['config']:<30} mean_RMSD={a['mean_rmsd']:.3f}  mean_t={a['mean_seconds']:.2f}s",
                flush=True,
            )
        history.append({"round": rnd, "rows": rows, "aggregate": agg, "targets": [t.target_id for t in targets]})
        cfgs = _mutate_configs([(a["config"], dict(a["kwargs"])) for a in top])

    out_path = os.path.join(args.out_dir, "iter_small_history.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

