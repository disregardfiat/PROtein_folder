#!/usr/bin/env python3
"""
Optimize folding pipelines per weekly target under a fixed budget.

Constraint baked in: you can submit up to 5 PDBs/target, and the **first attachment**
is the contest submission. This script writes ranked outputs so `rank_1` is the
email-first file.

Usage:
  python3 scripts/optimize_weekly_targets.py --targets T1235 --budget-seconds 600 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from horizon_physics.proteins.casp_targets import CASPTarget, ensure_experimental_ref, fetch_known_targets
from horizon_physics.proteins.full_protein_minimizer import full_chain_to_pdb, minimize_full_chain
from horizon_physics.proteins.grade_folds import ca_rmsd


@dataclass(frozen=True)
class RunConfig:
    name: str
    kwargs: Dict[str, Any]


def _candidate_grid(*, smoke_mode: bool = False) -> List[RunConfig]:
    cfgs: List[RunConfig] = []
    # Scout: cheap breadth.
    for tunnel in (True, False):
        for hcut in (None, 10.0, 8.0):
            cfgs.append(
                RunConfig(
                    name=f"scout_t{int(tunnel)}_h{hcut}",
                    kwargs={
                        "quick": True,
                        "simulate_ribosome_tunnel": tunnel,
                        "include_sidechains": False,
                        "fast_local_theta": True,
                        "horizon_neighbor_cutoff": hcut,
                        "kappa_dihedral": 0.01,
                        "hbond_weight": 0.0,
                        "fast_pass_steps_per_connection": 2,
                        "min_pass_iter_per_connection": 5,
                        "post_extrusion_refine": True,
                        "post_extrusion_max_rounds": 12,
                    },
                )
            )
    # Refine: better quality candidates.
    if smoke_mode:
        # Keep smoke checks fast and deterministic (no tunnel).
        return [
            RunConfig(
                name="smoke_fast_notunnel",
                kwargs={
                    "quick": True,
                    "simulate_ribosome_tunnel": False,
                    "include_sidechains": False,
                    "fast_local_theta": True,
                    "horizon_neighbor_cutoff": 10.0,
                    "kappa_dihedral": 0.01,
                    "hbond_weight": 0.0,
                },
            ),
            RunConfig(
                name="smoke_refine_notunnel",
                kwargs={
                    "quick": False,
                    "simulate_ribosome_tunnel": False,
                    "include_sidechains": False,
                    "fast_local_theta": True,
                    "horizon_neighbor_cutoff": 10.0,
                    "kappa_dihedral": 0.01,
                    "hbond_weight": 0.0,
                    "collective_kink_weight": 0.005,
                    "collective_kink_m": 3,
                    "collective_kink_use_ss_mask": True,
                },
            ),
        ]
    for tunnel in (True, False):
        for hcut in (None, 10.0):
            for ck in (0.0, 0.005):
                cfgs.append(
                    RunConfig(
                        name=f"refine_t{int(tunnel)}_h{hcut}_ck{ck}",
                        kwargs={
                            "quick": False,
                            "simulate_ribosome_tunnel": tunnel,
                            "include_sidechains": False,
                            "fast_local_theta": True,
                            "horizon_neighbor_cutoff": hcut,
                            "kappa_dihedral": 0.01,
                            "hbond_weight": 0.0,
                            "collective_kink_weight": ck,
                            "collective_kink_m": 3,
                            "collective_kink_use_ss_mask": True,
                            "fast_pass_steps_per_connection": 5,
                            "min_pass_iter_per_connection": 15,
                            "post_extrusion_refine": True,
                            "post_extrusion_refine_mode": "em_treetorque",
                            "post_extrusion_treetorque_phases": 8,
                            "post_extrusion_treetorque_n_steps": 200,
                        },
                    )
                )
    return cfgs


def _score(row: Dict[str, Any]) -> Tuple[float, float]:
    """Lower is better. Prefer RMSD when available; otherwise use proxy score, then time."""
    rmsd = row.get("ca_rmsd_ang")
    if rmsd is not None:
        return (float(rmsd), float(row.get("seconds", 1e9)))
    proxy = row.get("proxy_score")
    if proxy is not None:
        return (float(proxy), float(row.get("seconds", 1e9)))
    return (1e12, float(row.get("seconds", 1e9)))


def _proxy_metrics(ca: Any, res: Dict[str, Any]) -> Dict[str, Any]:
    """
    No-reference ranking metrics for blind targets.

    proxy_score combines:
      - E_ca_final (physics energy),
      - sequential Cα bond RMSE vs 3.8 Å,
      - non-neighbor clash count (< 2.0 Å),
      - compactness prior via Rg deviation from a weak N^0.38 baseline.
    """
    import numpy as np

    ca_np = np.asarray(ca, dtype=float)
    n = int(ca_np.shape[0])
    if n < 2:
        return {
            "proxy_score": 1e12,
            "proxy_e_ca_final": None,
            "proxy_bond_rmse": None,
            "proxy_clash_pairs": None,
            "proxy_rg_ang": None,
            "proxy_rg_target_ang": None,
            "proxy_mode": "blind",
        }

    d = ca_np[1:] - ca_np[:-1]
    bond = np.linalg.norm(d, axis=1)
    bond_rmse = float(np.sqrt(np.mean((bond - 3.8) ** 2)))

    clash_pairs = 0
    for sep in range(2, min(20, n)):
        v = ca_np[sep:] - ca_np[:-sep]
        r = np.linalg.norm(v, axis=1)
        clash_pairs += int(np.sum(r < 2.0))

    com = np.mean(ca_np, axis=0)
    rg = float(np.sqrt(np.mean(np.sum((ca_np - com) ** 2, axis=1))))
    rg_target = float(2.2 * (n ** 0.38))

    e_ca = res.get("E_ca_final")
    e_ca_v = float(e_ca) if e_ca is not None else 0.0

    # Weights chosen to keep terms numerically similar for 50-300 aa chains.
    proxy = (
        0.01 * e_ca_v
        + 120.0 * bond_rmse
        + 35.0 * float(clash_pairs)
        + 3.0 * abs(rg - rg_target)
    )
    return {
        "proxy_score": float(proxy),
        "proxy_e_ca_final": e_ca_v,
        "proxy_bond_rmse": bond_rmse,
        "proxy_clash_pairs": int(clash_pairs),
        "proxy_rg_ang": rg,
        "proxy_rg_target_ang": rg_target,
        "proxy_mode": "blind",
    }


def _run_one_config(
    sequence: str,
    gold_pdb: Optional[str],
    cfg_name: str,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        res = minimize_full_chain(sequence, **kwargs)
        pdb = full_chain_to_pdb(res)
        pm = _proxy_metrics(res.get("ca_min"), res)
        rmsd = None
        n_graded = None
        if gold_pdb and os.path.isfile(gold_pdb):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False, encoding="utf-8") as tmp:
                tmp.write(pdb)
                pred_path = tmp.name
            try:
                rr, per_res, _, _ = ca_rmsd(pred_path, gold_pdb, align_by_resid=False, trim_to_min_length=True)
                rmsd = float(rr)
                n_graded = int(len(per_res)) if per_res is not None else 0
            finally:
                try:
                    os.unlink(pred_path)
                except OSError:
                    pass
        return {
            "config": cfg_name,
            "seconds": time.perf_counter() - t0,
            "ca_rmsd_ang": rmsd,
            "n_graded": n_graded,
            **pm,
            "error": None,
            "kwargs": kwargs,
            "pdb": pdb,
        }
    except Exception as e:
        return {
            "config": cfg_name,
            "seconds": time.perf_counter() - t0,
            "ca_rmsd_ang": None,
            "n_graded": None,
            "proxy_score": None,
            "error": str(e),
            "kwargs": kwargs,
            "pdb": None,
        }


def _select_targets(all_targets: List[CASPTarget], wanted: Optional[str], min_len: int, max_len: int, max_targets: int) -> List[CASPTarget]:
    want = None
    if wanted:
        want = {x.strip().upper() for x in wanted.split(",") if x.strip()}
    out: List[CASPTarget] = []
    for t in all_targets:
        if not t.sequences:
            continue
        seq = t.sequences[0]
        if len(seq) < min_len or len(seq) > max_len:
            continue
        if want and t.target_id.upper() not in want:
            continue
        out.append(t)
    return out[:max_targets]


def optimize_target(
    target: CASPTarget,
    *,
    cache_dir: str,
    out_dir: str,
    budget_seconds: float,
    workers: int,
    smoke_mode: bool = False,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    seq = target.sequences[0]
    gold = ensure_experimental_ref(target, cache_dir) if target.pdb_code else None
    configs = _candidate_grid(smoke_mode=smoke_mode)
    started = time.perf_counter()
    best_rows: List[Dict[str, Any]] = []
    runs_done = 0
    runs_failed = 0

    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        pending = set()
        idx = 0

        def _submit_next() -> bool:
            nonlocal idx
            if idx >= len(configs):
                return False
            c = configs[idx]
            idx += 1
            fut = ex.submit(_run_one_config, seq, gold, c.name, c.kwargs)
            pending.add(fut)
            return True

        # Fill initial queue.
        while len(pending) < workers and _submit_next():
            pass

        while pending:
            remaining = budget_seconds - (time.perf_counter() - started)
            if remaining <= 0:
                for f in list(pending):
                    f.cancel()
                break
            done, pending = wait(pending, timeout=min(3.0, remaining), return_when=FIRST_COMPLETED)
            if not done:
                continue
            for f in done:
                row = f.result()
                runs_done += 1
                if row.get("error"):
                    runs_failed += 1
                else:
                    best_rows.append(row)
                    best_rows.sort(key=_score)
                    best_rows = best_rows[:5]
                # Keep submitting while we still have time and configs.
                if (time.perf_counter() - started) < budget_seconds:
                    _submit_next()

    # Persist top-5 ranked outputs; rank_1 should be attached first in email.
    ranked_paths: List[str] = []
    for rank, row in enumerate(best_rows[:5], start=1):
        pdb = row.get("pdb")
        if not pdb:
            continue
        path = os.path.join(out_dir, f"{target.target_id}_rank_{rank}.pdb")
        with open(path, "w", encoding="utf-8") as f:
            f.write(pdb)
        ranked_paths.append(path)
        row["rank"] = rank
        row["pdb_path"] = path
        row.pop("pdb", None)

    summary = {
        "target_id": target.target_id,
        "pdb_code": target.pdb_code,
        "n_res": len(seq),
        "budget_seconds": budget_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "runs_completed": runs_done,
        "runs_failed": runs_failed,
        "top5": best_rows[:5],
        "submission_policy": {
            "max_attachments": 5,
            "primary_submission_file": ranked_paths[0] if ranked_paths else None,
            "email_attachment_order": ranked_paths,
            "note": "Attach rank_1 first; that is the scored contest submission.",
        },
        "gold_pdb": gold,
    }
    with open(os.path.join(out_dir, f"{target.target_id}_optimization_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Optimize weekly target fold pipeline under per-target budget.")
    ap.add_argument("--casp-round", default="CASP16")
    ap.add_argument("--cache-dir", default=os.path.join(_REPO, ".casp_grade_cache"))
    ap.add_argument("--out-dir", default=os.path.join(_REPO, ".casp_grade_outputs", "weekly_optimizer"))
    ap.add_argument("--targets", default=None, help="Comma-separated target ids (e.g. T1235,H1202)")
    ap.add_argument("--max-targets", type=int, default=5)
    ap.add_argument("--min-residues", type=int, default=20)
    ap.add_argument("--max-residues", type=int, default=300)
    ap.add_argument("--budget-seconds", type=float, default=3 * 60 * 60, help="Per-target budget (default 10800)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--smoke-mode", action="store_true", help="Use quick scout configs only (for fast checks)")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    all_targets = fetch_known_targets(casp_round=args.casp_round, cache_dir=args.cache_dir, fill_pdb_codes=True)
    selected = _select_targets(all_targets, args.targets, args.min_residues, args.max_residues, args.max_targets)
    if not selected:
        print("No targets selected after filtering.", flush=True)
        return 1

    manifest: List[Dict[str, Any]] = []
    for t in selected:
        target_out = os.path.join(args.out_dir, t.target_id)
        os.makedirs(target_out, exist_ok=True)
        print(f"=== Optimize {t.target_id} ({len(t.sequences[0])} aa, budget={args.budget_seconds:.0f}s) ===", flush=True)
        s = optimize_target(
            t,
            cache_dir=args.cache_dir,
            out_dir=target_out,
            budget_seconds=float(args.budget_seconds),
            workers=int(args.workers),
            smoke_mode=bool(args.smoke_mode),
        )
        top = s.get("top5", [])
        if top:
            b = top[0]
            print(
                f"  best: config={b.get('config')}  rmsd={b.get('ca_rmsd_ang')}  time={b.get('seconds'):.2f}s",
                flush=True,
            )
            print(f"  rank_1 (attach first): {s['submission_policy']['primary_submission_file']}", flush=True)
        else:
            print("  no successful runs", flush=True)
        manifest.append(s)

    mpath = os.path.join(args.out_dir, "weekly_optimizer_manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {mpath}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

