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

# Crambin (1CRN) sequence — 46 residues (used for crambin-only sweeps).
CRAMBIN_SEQ = "TTCCPSIVARSNFNVCRLPGTPEAIICGDVCDLDCTAKTCFSIICT"


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
        (
            "refine_tunnel_variational_staged",
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
                "variational_pair_weight": 0.05,
                "variational_pair_epsilon": 0.08,
                "variational_pair_sigma": 4.0,
                "variational_pair_dist_cutoff": 10.0,
                "variational_pair_min_seq_sep": 3,
                "variational_pair_max_pairs": 300,
                "variational_staged_opt": True,
                "variational_stage_frac": 0.5,
                "variational_bound_prune": True,
                "variational_bound_prune_margin": 0.0,
                "fast_pass_steps_per_connection": 5,
                "min_pass_iter_per_connection": 15,
                "post_extrusion_refine": True,
                "post_extrusion_refine_mode": "em_treetorque",
                "post_extrusion_treetorque_phases": 8,
                "post_extrusion_treetorque_n_steps": 200,
            },
        ),
        (
            "refine_tunnel_inertial_twist",
            {
                "quick": False,
                "simulate_ribosome_tunnel": True,
                "include_sidechains": False,
                "fast_local_theta": True,
                "horizon_neighbor_cutoff": 10.0,
                "kappa_dihedral": 0.01,
                "hbond_weight": 0.0,
                "inertial_twist_weight": 0.01,
                "inertial_twist_exponent": 1.5,
                "fast_pass_steps_per_connection": 5,
                "min_pass_iter_per_connection": 15,
                "post_extrusion_refine": True,
                "post_extrusion_refine_mode": "em_treetorque",
                "post_extrusion_treetorque_phases": 8,
                "post_extrusion_treetorque_n_steps": 200,
            },
        ),
        (
            "quick_tunnel_ensemble_inner",
            {
                "quick": True,
                "simulate_ribosome_tunnel": True,
                "include_sidechains": False,
                "fast_local_theta": True,
                "horizon_neighbor_cutoff": None,
                "kappa_dihedral": 0.01,
                "hbond_weight": 0.0,
                "ensemble_translation_mix_alpha": 0.05,
                "ensemble_translation_step": 0.35,
                "ensemble_decay_span": 0.25,
                "ensemble_beta": 0.25,
                "ensemble_s2_power": 1.0,
                "ensemble_score_lambda": 0.01,
                "post_extrusion_refine": True,
                "post_extrusion_max_rounds": 12,
            },
        ),
    ]


def _mutate_configs(top: List[Tuple[str, Dict[str, Any]]]) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    for name, kw in top:
        out.append((name, dict(kw)))
        if float(kw.get("ensemble_translation_mix_alpha", 0.0)) > 0.0:
            # Keep this search cheap: only sweep ensemble knobs for ensemble-enabled configs.
            for mix in (0.05, 0.1, 0.15):
                k2 = dict(kw)
                k2["ensemble_translation_mix_alpha"] = float(mix)
                out.append((f"{name}_emt{mix}", k2))
            for tstep in (0.25, 0.35):
                k2 = dict(kw)
                k2["ensemble_translation_step"] = float(tstep)
                out.append((f"{name}_ets{tstep}", k2))
            for beta in (0.25, 0.35):
                k2 = dict(kw)
                k2["ensemble_beta"] = float(beta)
                out.append((f"{name}_eb{beta}", k2))
            for lam in (0.0, 0.01, 0.03):
                k2 = dict(kw)
                k2["ensemble_score_lambda"] = float(lam)
                out.append((f"{name}_el{lam}", k2))
            for pwr in (0.75, 1.0, 1.25):
                k2 = dict(kw)
                k2["ensemble_s2_power"] = float(pwr)
                out.append((f"{name}_es{pwr}", k2))
            continue
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
        for vpw in (0.0, 0.025, 0.05, 0.1):
            k2 = dict(kw)
            k2["variational_pair_weight"] = vpw
            if vpw > 0.0:
                k2["variational_staged_opt"] = True
                k2["variational_bound_prune"] = True
            out.append((f"{name}_vp{vpw}", k2))
        for sfrac in (0.33, 0.5, 0.67):
            k2 = dict(kw)
            if float(k2.get("variational_pair_weight", 0.0)) > 0.0:
                k2["variational_staged_opt"] = True
                k2["variational_stage_frac"] = sfrac
            out.append((f"{name}_sf{sfrac}", k2))
        for itw in (0.0, 0.005, 0.01, 0.02):
            k2 = dict(kw)
            k2["inertial_twist_weight"] = itw
            if itw > 0.0:
                k2["inertial_twist_exponent"] = 1.5
            out.append((f"{name}_itw{itw}", k2))
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


def _mutate_ensemble_configs_fine(top: List[Tuple[str, Dict[str, Any]]]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Fine stage: focus only on ensemble carrier knobs (viscosity/dampening + inertial transport).
    We keep all other kwargs from the parent config and sweep:
      - ensemble_translation_mix_alpha
      - ensemble_translation_step
      - ensemble_beta
      - ensemble_score_lambda
      - ensemble_s2_power
      - ensemble_inertial_dt
      - ensemble_linear_damping
      - ensemble_linear_gain
      - ensemble_damping_mode
      - ensemble_angular_mix_alpha
      - ensemble_angular_damping
      - ensemble_angular_gain
    """
    mixs = (0.03, 0.05, 0.07, 0.09, 0.11, 0.15)
    tsteps = (0.25, 0.275, 0.3, 0.325, 0.35)
    betas = (0.25, 0.275, 0.3, 0.325, 0.35)
    lams = (0.0, 0.005, 0.01, 0.015, 0.02, 0.03)
    pows = (0.75, 0.875, 1.0, 1.125, 1.25)
    amixs = (0.0, 0.03, 0.06, 0.09, 0.12)
    adamps = (0.7, 0.8, 0.9, 0.95)
    agains = (0.05, 0.1, 0.2, 0.3)
    idts = (0.5, 1.0, 1.5)
    ldamps = (0.8, 0.9, 0.95)
    lgains = (0.5, 1.0, 1.5)
    dmodes = ("linear", "sqrt")

    out: List[Tuple[str, Dict[str, Any]]] = []
    for name, kw in top:
        # Always keep the parent itself as a candidate.
        out.append((name, dict(kw)))

        base = dict(kw)
        for mix in mixs:
            if float(mix) <= 0.0:
                continue
            k2 = dict(base)
            k2["ensemble_translation_mix_alpha"] = float(mix)
            out.append((f"{name}_femt{mix}", k2))

        for tstep in tsteps:
            k2 = dict(base)
            k2["ensemble_translation_step"] = float(tstep)
            out.append((f"{name}_fets{tstep}", k2))

        for beta in betas:
            k2 = dict(base)
            k2["ensemble_beta"] = float(beta)
            out.append((f"{name}_febe{beta}", k2))

        for lam in lams:
            k2 = dict(base)
            k2["ensemble_score_lambda"] = float(lam)
            out.append((f"{name}_fela{lam}", k2))

        for pwr in pows:
            k2 = dict(base)
            k2["ensemble_s2_power"] = float(pwr)
            out.append((f"{name}_fes{pwr}", k2))

        for idt in idts:
            k2 = dict(base)
            k2["ensemble_inertial_dt"] = float(idt)
            out.append((f"{name}_feidt{idt}", k2))

        for ld in ldamps:
            k2 = dict(base)
            k2["ensemble_linear_damping"] = float(ld)
            out.append((f"{name}_feld{ld}", k2))

        for lg in lgains:
            k2 = dict(base)
            k2["ensemble_linear_gain"] = float(lg)
            out.append((f"{name}_felg{lg}", k2))

        for dm in dmodes:
            k2 = dict(base)
            k2["ensemble_damping_mode"] = str(dm)
            out.append((f"{name}_fedm{dm}", k2))

        for amix in amixs:
            k2 = dict(base)
            k2["ensemble_angular_mix_alpha"] = float(amix)
            out.append((f"{name}_feam{amix}", k2))

        for adamp in adamps:
            k2 = dict(base)
            k2["ensemble_angular_damping"] = float(adamp)
            out.append((f"{name}_fead{adamp}", k2))

        for again in agains:
            k2 = dict(base)
            k2["ensemble_angular_gain"] = float(again)
            out.append((f"{name}_feag{again}", k2))

    # de-duplicate by sorted kwargs (config name is just a label).
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


def _run_cfg_on_target(
    t: CASPTarget,
    gold_pdb: str,
    name: str,
    kwargs: Dict[str, Any],
    *,
    tunnel_first: bool,
) -> Dict[str, Any]:
    seq = t.sequences[0]
    t0 = time.perf_counter()
    try:
        k = dict(kwargs)
        if tunnel_first:
            k["simulate_ribosome_tunnel"] = True
        res = minimize_full_chain(seq, **k)
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
            "kwargs": k,
        }
    except Exception as e:
        return {
            "target_id": t.target_id,
            "config": name,
            "seconds": time.perf_counter() - t0,
            "ca_rmsd_ang": None,
            "n_graded": None,
            "error": str(e),
            "kwargs": k if "k" in locals() else kwargs,
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
    ap.add_argument(
        "--time-sweep",
        action="store_true",
        help="Run a time-bounded two-stage ensemble sweep (course then fine) on crambin only.",
    )
    ap.add_argument(
        "--crambin-only",
        action="store_true",
        help="When --time-sweep is set, run only on the local 1CRN gold standard (no CASP fetch).",
    )
    ap.add_argument("--course-minutes", type=float, default=15.0, help="Course stage wall clock minutes.")
    ap.add_argument("--fine-hours", type=float, default=6.0, help="Fine stage wall clock hours.")
    ap.add_argument(
        "--stage-max-rounds",
        type=int,
        default=999,
        help="Hard cap on sweep rounds per stage (time limit still enforced).",
    )
    ap.add_argument("--no-two-pass", action="store_true", help="Disable the top-k robustness rerun.")
    ap.add_argument(
        "--tunnel-first",
        action="store_true",
        help="Force simulate_ribosome_tunnel=True for every evaluated configuration.",
    )
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    if bool(args.time_sweep) and bool(args.crambin_only):
        if len(CRAMBIN_SEQ) < args.min_residues or len(CRAMBIN_SEQ) > args.max_residues:
            print(
                f"Crambin length={len(CRAMBIN_SEQ)} outside [{args.min_residues},{args.max_residues}]; nothing to run.",
                flush=True,
            )
            return 1

        targets = [
            CASPTarget(
                target_id="crambin_only",
                target_type="Prot",
                sequences=[CRAMBIN_SEQ],
                pdb_code="1crn",
                description="Local crambin-only sweep target",
            )
        ]
        gold_path = os.path.join(_REPO, "proteins", "1CRN.pdb")
        if not os.path.isfile(gold_path):
            print(f"Missing gold reference: {gold_path}", flush=True)
            return 1
        gold_map: Dict[str, str] = {"crambin_only": gold_path}

        def _run_stage_time_bounded(
            *,
            stage_name: str,
            wall_limit_s: float,
            init_cfgs: List[Tuple[str, Dict[str, Any]]],
            mutate_fn,
        ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
            start = time.perf_counter()
            cfgs = list(init_cfgs)
            rows_all: List[Dict[str, Any]] = []
            rounds_done = 0

            while rounds_done < int(args.stage_max_rounds):
                if time.perf_counter() - start >= float(wall_limit_s):
                    break
                rounds_done += 1

                print(f"\n=== {stage_name} Round {rounds_done} configs={len(cfgs)} ===", flush=True)
                rows_round: List[Dict[str, Any]] = []

                stop_now = False
                for t in targets:
                    gp = gold_map[t.target_id]
                    for name, kw in cfgs:
                        if time.perf_counter() - start >= float(wall_limit_s):
                            stop_now = True
                            break
                        r = _run_cfg_on_target(
                            t,
                            gp,
                            name,
                            kw,
                            tunnel_first=bool(args.tunnel_first),
                        )
                        rows_all.append(r)
                        rows_round.append(r)
                        if r.get("error"):
                            print(f"  {t.target_id:>10}  {name:<40}  ERROR: {r['error']}", flush=True)
                        else:
                            print(
                                f"  {t.target_id:>10}  {name:<40}  {r['seconds']:7.2f}s  RMSD={r['ca_rmsd_ang']:.3f}",
                                flush=True,
                            )
                    if stop_now:
                        break

                if not rows_round:
                    break

                agg_round = _aggregate(rows_round)
                top_round = agg_round[: max(1, int(args.top_k))]
                print("\nStage leaderboard (this round):", flush=True)
                for i, a in enumerate(top_round, start=1):
                    print(
                        f"  {i:>2}. {a['config']:<30} mean_RMSD={a['mean_rmsd']:.3f}  mean_t={a['mean_seconds']:.2f}s",
                        flush=True,
                    )

                # If we stopped due to wall clock, don't mutate further.
                if time.perf_counter() - start >= float(wall_limit_s):
                    break

                cfgs = mutate_fn([(a["config"], dict(a["kwargs"])) for a in top_round])

            agg_final = _aggregate(rows_all) if rows_all else []
            top_final = agg_final[: max(1, int(args.top_k))] if agg_final else []

            if not bool(args.no_two_pass) and top_final:
                print(f"\n=== {stage_name} Two-pass rerun top-k for robustness ===", flush=True)
                # Re-run top configs once more (same kwargs) to stabilize ranking.
                rerun_cfgs = [(a["config"], dict(a["kwargs"])) for a in top_final]
                rows_extra: List[Dict[str, Any]] = []
                for t in targets:
                    gp = gold_map[t.target_id]
                    for name, kw in rerun_cfgs:
                        if time.perf_counter() - start >= float(wall_limit_s):
                            print("Time exceeded during two-pass rerun; stopping reruns.", flush=True)
                            break
                        r = _run_cfg_on_target(
                            t,
                            gp,
                            name,
                            kw,
                            tunnel_first=bool(args.tunnel_first),
                        )
                        rows_all.append(r)
                        rows_extra.append(r)
                if rows_extra:
                    agg_final = _aggregate(rows_all)
                    top_final = agg_final[: max(1, int(args.top_k))]

            return rows_all, agg_final

        course_seconds = float(args.course_minutes) * 60.0
        fine_seconds = float(args.fine_hours) * 3600.0

        cfgs0 = _seed_configs()
        print(f"Running course sweep for ~{args.course_minutes} minutes.", flush=True)
        rows_course, agg_course = _run_stage_time_bounded(
            stage_name="COURSE",
            wall_limit_s=course_seconds,
            init_cfgs=cfgs0,
            mutate_fn=_mutate_configs,
        )

        top_course = agg_course[: max(1, int(args.top_k))] if agg_course else []
        print("\nCourse top configs:", flush=True)
        for i, a in enumerate(top_course, start=1):
            print(
                f"  {i:>2}. {a['config']:<30} mean_RMSD={a['mean_rmsd']:.3f}  mean_t={a['mean_seconds']:.2f}s",
                flush=True,
            )

        # Fine stage focuses on ensemble knobs only.
        fine_init = [(a["config"], dict(a["kwargs"])) for a in top_course]
        print(f"\nRunning fine sweep for ~{args.fine_hours} hours.", flush=True)
        rows_fine, agg_fine = _run_stage_time_bounded(
            stage_name="FINE",
            wall_limit_s=fine_seconds,
            init_cfgs=fine_init,
            mutate_fn=_mutate_ensemble_configs_fine,
        )

        out_path = os.path.join(args.out_dir, "iter_small_time_sweep_history.json")
        history_obj = {
            "targets": [t.target_id for t in targets],
            "gold_path": gold_path,
            "course": {
                "course_seconds": course_seconds,
                "rows": rows_course,
                "aggregate": agg_course,
                "top_k": top_course,
            },
            "fine": {
                "fine_seconds": fine_seconds,
                "rows": rows_fine,
                "aggregate": agg_fine,
            },
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(history_obj, f, indent=2)
        print(f"\nWrote {out_path}", flush=True)
        return 0

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
                r = _run_cfg_on_target(t, gp, name, kw, tunnel_first=bool(args.tunnel_first))
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

