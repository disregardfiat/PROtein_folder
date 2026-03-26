#!/usr/bin/env python3
"""
Tune full heavy-atom backbone OSH (N, CA, C, O) on chignolin so CA RMSD matches the Cα-only seek.

Pipeline: tunnel → Cα OSH (best knobs from ``chignolin_ca_osh_seek.json`` if present) →
idealized backbone from that Cα trace → ``minimize_backbone_with_osh_oracle`` with
optional ``ca_target`` / ``k_ca_target`` in ``energy_kwargs`` (geometry refinement without
drifting the folded Cα map).

Usage:
  python3 scripts/seek_chignolin_backbone_osh.py --trials 30 --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

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
from horizon_physics.proteins.folding_energy import K_BOND, K_CLASH  # noqa: E402
from horizon_physics.proteins.grade_folds import ca_rmsd  # noqa: E402
from horizon_physics.proteins.osh_oracle_backbone import minimize_backbone_with_osh_oracle  # noqa: E402
from horizon_physics.proteins.osh_oracle_folding import minimize_ca_with_osh_oracle  # noqa: E402
from horizon_physics.proteins.very_short_fold_targets import target_by_key  # noqa: E402

_DEFAULT_SEEK = os.path.join(
    _REPO, ".casp_grade_outputs", "iter_small", "chignolin_ca_osh_seek.json"
)


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


def load_ca_best_kwargs(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        b = data.get("best") or {}
        if not b:
            return None
        return {
            "step_size": float(b["step"]),
            "amp_threshold_quantile": float(b["quantile"]),
            "gate_mix": float(b["gate_mix"]),
            "use_energy_reservoir": True,
            "reservoir_init": float(b["reservoir_init"]),
            "reservoir_gain_scale": float(b["reservoir_gain"]),
            "use_contact_reflectors": bool(b.get("use_contact_reflectors", True)),
            "contact_cutoff_ang": float(b.get("contact_cutoff", 5.0)),
            "contact_max_reflectors": int(b.get("contact_max_reflectors", 4)),
            "contact_grad_coupling": 0.0,
            "contact_weight_gradient": False,
            "contact_terminus_window": int(b.get("contact_terminus_window", 4)),
            "contact_terminus_score_scale": float(b.get("contact_terminus_score_scale", 1.03)),
            "n_iter": 900,
            "stop_when_settled": True,
            "settle_window": 40,
            "settle_min_iter": 60,
            "settle_energy_tol": 1e-5,
            "settle_step_tol": 1e-5,
            "energy_kwargs": {"fast_local_theta": True},
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def default_ca_osh_kwargs() -> Dict[str, Any]:
    return {
        "step_size": 0.024,
        "amp_threshold_quantile": 0.503,
        "gate_mix": 0.57,
        "use_energy_reservoir": True,
        "reservoir_init": 53.5,
        "reservoir_gain_scale": 1.356,
        "use_contact_reflectors": True,
        "contact_cutoff_ang": 4.5,
        "contact_max_reflectors": 4,
        "contact_grad_coupling": 0.0,
        "contact_weight_gradient": False,
        "contact_terminus_window": 4,
        "contact_terminus_score_scale": 1.027,
        "n_iter": 900,
        "stop_when_settled": True,
        "settle_window": 40,
        "settle_min_iter": 60,
        "settle_energy_tol": 1e-5,
        "settle_step_tol": 1e-5,
        "energy_kwargs": {"fast_local_theta": True},
    }


def positions_from_ca(ca: np.ndarray, seq: str) -> np.ndarray:
    bb = _place_full_backbone(np.asarray(ca, dtype=float), seq)
    return np.array([xyz for _, xyz in bb], dtype=float)


def run_search(
    ca_ref: np.ndarray,
    seq: str,
    ref_path: str,
    baseline_ca_rmsd: float,
    trials: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    pos0 = positions_from_ca(ca_ref, seq)
    ca_fixed = np.asarray(ca_ref, dtype=float).copy()
    rows: List[Dict[str, Any]] = []
    best_rmsd = float("inf")
    best_meta: Dict[str, Any] = {}

    for t in range(int(trials)):
        step = float(rng.choice([0.008, 0.012, 0.016, 0.02, 0.024]))
        quantile = float(rng.uniform(0.42, 0.62))
        res_init = float(rng.uniform(25.0, 95.0))
        res_gain = float(rng.uniform(1.05, 1.45))
        gate_mix = float(rng.uniform(0.48, 0.62))
        k_ca = float(rng.choice([0.0, 0.15, 0.4, 1.0, 2.5, 6.0, 14.0]))
        k_bond_scale = float(rng.choice([0.65, 1.0, 1.35]))
        include_clash = bool(rng.random() > 0.2)
        k_clash_scale = float(rng.choice([0.35, 0.7, 1.0])) if include_clash else 1.0
        proj_passes = int(rng.choice([2, 3, 4, 5]))
        use_contacts = bool(rng.random() > 0.35)
        cutoff = float(rng.choice([4.5, 5.0, 5.75]))
        max_ref = int(rng.choice([2, 3, 4]))
        tw = int(rng.choice([0, 4, 6]))
        ts = float(rng.uniform(1.0, 1.09)) if tw > 0 else 1.0
        n_iter = int(rng.choice([400, 700, 1000]))

        e_kw: Dict[str, Any] = {
            "fast_local_theta": True,
            "ca_target": ca_fixed,
            "k_ca_target": k_ca,
            "k_bond": float(K_BOND) * k_bond_scale,
            "include_clash": include_clash,
            "k_clash": float(K_CLASH) * k_clash_scale,
        }

        kw: Dict[str, Any] = {
            "sequence": seq,
            "n_iter": n_iter,
            "step_size": step,
            "amp_threshold_quantile": quantile,
            "gate_mix": gate_mix,
            "use_energy_reservoir": True,
            "reservoir_init": res_init,
            "reservoir_gain_scale": res_gain,
            "stop_when_settled": True,
            "settle_window": 30,
            "settle_min_iter": 50,
            "settle_energy_tol": 2e-5,
            "settle_step_tol": 2e-5,
            "use_contact_reflectors": use_contacts,
            "contact_cutoff_ang": cutoff,
            "contact_max_reflectors": max_ref,
            "contact_grad_coupling": 0.0,
            "contact_weight_gradient": False,
            "contact_terminus_window": tw,
            "contact_terminus_score_scale": ts,
            "backbone_projection_passes": proj_passes,
            "harmonic_max_dims": min(120, 12 * len(seq)),
            "energy_kwargs": e_kw,
        }

        t0 = time.perf_counter()
        pos1, info = minimize_backbone_with_osh_oracle(pos0.copy(), **kw)
        dt = time.perf_counter() - t0
        ca1 = np.asarray(pos1[1::4], dtype=float)
        rmsd = score_ca(ca1, seq, ref_path)
        if not math.isfinite(rmsd):
            continue
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
            "k_ca_target": k_ca,
            "k_bond_scale": k_bond_scale,
            "include_clash": include_clash,
            "k_clash_scale": k_clash_scale,
            "backbone_projection_passes": proj_passes,
            "use_contact_reflectors": use_contacts,
            "contact_cutoff": cutoff,
            "contact_max_reflectors": max_ref,
            "contact_terminus_window": tw,
            "contact_terminus_score_scale": ts,
            "n_iter": n_iter,
            "delta_vs_ca_only_ang": float(rmsd - baseline_ca_rmsd),
        }
        rows.append(rec)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_meta = dict(rec)
        print(
            f"trial {t+1}/{trials} RMSD={rmsd:.3f} Å  k_ca={k_ca:.2f} clash={include_clash} "
            f"step={step:.4f}  ({dt:.2f}s)"
        )

    return rows, best_meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Seek backbone OSH knobs on chignolin vs 1UAO.")
    ap.add_argument("--trials", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--ca-seek-json",
        default=_DEFAULT_SEEK,
        help="Cα seek JSON (for best Cα-OSH hyperparameters).",
    )
    ap.add_argument(
        "--out-json",
        default=os.path.join(_REPO, ".casp_grade_outputs", "iter_small", "chignolin_backbone_osh_seek.json"),
    )
    args = ap.parse_args()

    tgt = target_by_key("chignolin_1uao")
    seq = tgt.sequence
    ref_path = os.path.join(_REPO, "proteins", tgt.reference_pdb)
    if not os.path.isfile(ref_path):
        print("Missing ref", ref_path, "- run scripts/fetch_very_short_reference_pdbs.py", file=sys.stderr)
        return 1

    ca_kw = load_ca_best_kwargs(str(args.ca_seek_json)) or default_ca_osh_kwargs()

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
    ca_tunnel = np.asarray(tunnel["ca_min"], dtype=float)
    t_tunnel = time.perf_counter() - t0

    t1 = time.perf_counter()
    ca_ref, ca_info = minimize_ca_with_osh_oracle(ca_tunnel.copy(), **ca_kw)
    t_ca_osh = time.perf_counter() - t1

    baseline_ca_rmsd = score_ca(ca_ref, seq, ref_path)

    rows, best = run_search(
        ca_ref,
        seq,
        ref_path,
        baseline_ca_rmsd,
        int(args.trials),
        int(args.seed),
    )

    out = {
        "target": tgt.key,
        "sequence": seq,
        "reference_pdb": ref_path,
        "ca_seek_json": str(args.ca_seek_json),
        "tunnel_wall_s": t_tunnel,
        "ca_osh_wall_s": t_ca_osh,
        "ca_only_rmsd_after_osh_ang": baseline_ca_rmsd,
        "ca_osh_stop": str(ca_info.stop_reason),
        "best": best,
        "trials": rows,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== Cα-only (after same CA OSH) ===")
    print(f"baseline CA RMSD = {baseline_ca_rmsd:.3f} Å")
    print("\n=== Best backbone OSH ===")
    print(json.dumps(best, indent=2))
    print("Wrote", args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
