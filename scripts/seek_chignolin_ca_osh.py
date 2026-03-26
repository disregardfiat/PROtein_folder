#!/usr/bin/env python3
"""
Random / grid search over minor ``minimize_ca_with_osh_oracle`` knobs on chignolin (1UAO).

Uses one fixed tunnel Cα snapshot so differences come only from oracle hyperparameters.
Scores CA RMSD vs ``proteins/1UAO.pdb`` (same as other benchmarks).

Usage:
  python scripts/seek_chignolin_ca_osh.py --trials 40 --seed 42
  python scripts/seek_chignolin_ca_osh.py --grid  # small deterministic grid
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

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from horizon_physics.proteins.casp_submission import _place_full_backbone  # noqa: E402
from horizon_physics.proteins.full_protein_minimizer import (  # noqa: E402
    minimize_full_chain,
    full_chain_to_pdb,
)
from horizon_physics.proteins.grade_folds import ca_rmsd  # noqa: E402
from horizon_physics.proteins.osh_oracle_folding import minimize_ca_with_osh_oracle  # noqa: E402
from horizon_physics.proteins.very_short_fold_targets import target_by_key  # noqa: E402


def score_ca(ca: np.ndarray, seq: str, ref_path: str) -> float:
    bb = _place_full_backbone(np.asarray(ca, dtype=float), seq)
    obj = {"ca_min": ca, "backbone_atoms": bb, "sequence": seq, "n_res": len(seq)}
    pdb = full_chain_to_pdb(obj)
    p = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
    try:
        p.write(pdb)
        p.close()
        r, _, _, _ = ca_rmsd(p.name, ref_path, align_by_resid=False, trim_to_min_length=True)
        return float(r)
    finally:
        os.unlink(p.name)


def _trial_kwargs(
    *,
    step: float,
    quantile: float,
    res_init: float,
    res_gain: float,
    gate_mix: float,
    use_contacts: bool,
    cutoff: float,
    max_ref: int,
    contact_term_w: int,
    contact_term_scale: float,
) -> Dict[str, Any]:
    kw: Dict[str, Any] = {
        "step_size": float(step),
        "amp_threshold_quantile": float(quantile),
        "gate_mix": float(gate_mix),
        "use_energy_reservoir": True,
        "reservoir_init": float(res_init),
        "reservoir_gain_scale": float(res_gain),
        "stop_when_settled": True,
        "settle_window": 40,
        "settle_min_iter": 60,
        "settle_energy_tol": 1e-5,
        "settle_step_tol": 1e-5,
        "n_iter": 900,
        "use_contact_reflectors": bool(use_contacts),
        "contact_cutoff_ang": float(cutoff),
        "contact_max_reflectors": int(max_ref),
        "contact_grad_coupling": 0.0,
        "contact_weight_gradient": False,
        "contact_terminus_window": int(contact_term_w),
        "contact_terminus_score_scale": float(contact_term_scale),
        "energy_kwargs": {"fast_local_theta": True},
    }
    return kw


def run_random_search(
    ca0: np.ndarray,
    seq: str,
    ref_path: str,
    trials: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    best_rmsd = float("inf")
    best_meta: Dict[str, Any] = {}

    for t in range(int(trials)):
        step = float(rng.choice([0.017, 0.019, 0.0218, 0.024, 0.027, 0.030]))
        quantile = float(rng.uniform(0.44, 0.58))
        res_init = float(rng.uniform(35.0, 85.0))
        res_gain = float(rng.uniform(1.05, 1.38))
        gate_mix = float(rng.uniform(0.48, 0.62))
        use_contacts = bool(rng.random() > 0.35)
        cutoff = float(rng.choice([4.5, 5.0, 5.5, 6.0]))
        max_ref = int(rng.choice([2, 3, 4, 5]))
        tw = int(rng.choice([0, 4, 6, 8]))
        ts = float(rng.uniform(1.0, 1.12)) if tw > 0 else 1.0

        kw = _trial_kwargs(
            step=step,
            quantile=quantile,
            res_init=res_init,
            res_gain=res_gain,
            gate_mix=gate_mix,
            use_contacts=use_contacts,
            cutoff=cutoff,
            max_ref=max_ref,
            contact_term_w=tw,
            contact_term_scale=ts,
        )
        t0 = time.perf_counter()
        ca1, info = minimize_ca_with_osh_oracle(ca0.copy(), **kw)
        dt = time.perf_counter() - t0
        rmsd = score_ca(ca1, seq, ref_path)
        rec = {
            "trial": t + 1,
            "rmsd": rmsd,
            "wall_s": dt,
            "accepted": int(info.accepted_steps),
            "iters_exec": int(info.iterations_executed),
            "stop": str(info.stop_reason),
            "step": step,
            "quantile": quantile,
            "reservoir_init": res_init,
            "reservoir_gain": res_gain,
            "gate_mix": gate_mix,
            "use_contact_reflectors": use_contacts,
            "contact_cutoff": cutoff,
            "contact_max_reflectors": max_ref,
            "contact_terminus_window": tw,
            "contact_terminus_score_scale": ts,
        }
        rows.append(rec)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_meta = dict(rec)
        print(
            f"trial {t+1}/{trials} RMSD={rmsd:.3f} Å  step={step:.4f} q={quantile:.3f} "
            f"res_i={res_init:.1f} contacts={use_contacts}  ({dt:.2f}s)"
        )

    return rows, best_meta


def run_small_grid(ca0: np.ndarray, seq: str, ref_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    steps = [0.019, 0.0218, 0.024]
    quants = [0.48, 0.525, 0.55]
    res_inits = [45.0, 55.0, 65.0]
    rows: List[Dict[str, Any]] = []
    best_rmsd = float("inf")
    best_meta: Dict[str, Any] = {}
    n = 0
    for step in steps:
        for q in quants:
            for ri in res_inits:
                n += 1
                kw = _trial_kwargs(
                    step=step,
                    quantile=q,
                    res_init=ri,
                    res_gain=1.235,
                    gate_mix=0.55,
                    use_contacts=True,
                    cutoff=5.0,
                    max_ref=3,
                    contact_term_w=6,
                    contact_term_scale=1.06,
                )
                t0 = time.perf_counter()
                ca1, info = minimize_ca_with_osh_oracle(ca0.copy(), **kw)
                dt = time.perf_counter() - t0
                rmsd = score_ca(ca1, seq, ref_path)
                rec = {
                    "trial": n,
                    "rmsd": rmsd,
                    "wall_s": dt,
                    "step": step,
                    "quantile": q,
                    "reservoir_init": ri,
                    "accepted": int(info.accepted_steps),
                    "iters_exec": int(info.iterations_executed),
                    "stop": str(info.stop_reason),
                }
                rows.append(rec)
                if rmsd < best_rmsd:
                    best_rmsd = rmsd
                    best_meta = dict(rec)
                print(f"grid {n} RMSD={rmsd:.3f} step={step} q={q} res={ri} ({dt:.2f}s)")
    return rows, best_meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Seek Cα OSH knobs on chignolin vs 1UAO.")
    ap.add_argument("--trials", type=int, default=36, help="Random search trials (ignored with --grid).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid", action="store_true", help="Run small 3×3×3 grid instead of random.")
    ap.add_argument(
        "--out-json",
        default=os.path.join(_REPO, ".casp_grade_outputs", "iter_small", "chignolin_ca_osh_seek.json"),
    )
    args = ap.parse_args()

    tgt = target_by_key("chignolin_1uao")
    seq = tgt.sequence
    ref_path = os.path.join(_REPO, "proteins", tgt.reference_pdb)
    if not os.path.isfile(ref_path):
        print("Missing ref", ref_path, "- run scripts/fetch_very_short_reference_pdbs.py", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    tunnel = minimize_full_chain(
        seq,
        quick=True,
        simulate_ribosome_tunnel=True,
        post_extrusion_refine=False,
        fast_pass_steps_per_connection=2,
        min_pass_iter_per_connection=5,
        fast_local_theta=True,
        horizon_neighbor_cutoff=10.0,
        kappa_dihedral=0.01,
        hbond_weight=0.0,
    )
    ca0 = np.asarray(tunnel["ca_min"], dtype=float)
    t_tunnel = time.perf_counter() - t0
    base_rmsd = score_ca(ca0, seq, ref_path)

    if args.grid:
        rows, best = run_small_grid(ca0, seq, ref_path)
        mode = "grid"
    else:
        rows, best = run_random_search(ca0, seq, ref_path, int(args.trials), int(args.seed))
        mode = "random"

    out = {
        "target": tgt.key,
        "sequence": seq,
        "reference_pdb": ref_path,
        "tunnel_wall_s": t_tunnel,
        "tunnel_ca_rmsd_ang": base_rmsd,
        "mode": mode,
        "best": best,
        "trials": rows,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== Best ===")
    print(json.dumps(best, indent=2))
    print("Wrote", args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
