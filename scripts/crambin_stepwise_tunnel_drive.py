#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from typing import Dict, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from horizon_physics.proteins.force_carrier_ensemble import choose_best_translation_direction
from horizon_physics.proteins.folding_energy import grad_full
from horizon_physics.proteins.full_protein_minimizer import minimize_full_chain, full_chain_to_pdb
from horizon_physics.proteins.grade_folds import ca_rmsd, load_ca_from_pdb
from horizon_physics.proteins.gradient_descent_folding import _project_bonds


CRAMBIN_SEQ = "TTCCPSIVARSNFNVCRLPGTPEAIICGDVCDLDCTAKTCFSIICT"
CRAMBIN_GOLD = os.path.join(REPO, "proteins", "1CRN.pdb")
OUT_PATH = os.path.join(REPO, ".casp_grade_outputs", "iter_small", "crambin_stepwise_drive.json")

TARGET_PAIRS = {
    "ss_16_26": (16, 26),
    "ss_4_32": (4, 32),
    "ss_3_40": (3, 40),
    "sheet_1_32": (1, 32),
    "sheet_4_35": (4, 35),
}


def pair_dist(ca: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(ca[i - 1] - ca[j - 1]))


def local_s2_angles(ca: np.ndarray) -> np.ndarray:
    n = int(ca.shape[0])
    out = np.zeros((n,), dtype=float)
    if n < 3:
        return out
    for i in range(1, n - 1):
        u = ca[i - 1] - ca[i]
        v = ca[i + 1] - ca[i]
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu < 1e-12 or nv < 1e-12:
            continue
        c = float(np.clip(np.dot(u / nu, v / nv), -1.0, 1.0))
        th = float(np.arccos(c))
        out[i] = float(np.sin(th) ** 2)
    return out


def pair_gap_score(ca: np.ndarray, native: Dict[str, float]) -> float:
    gaps = []
    for k, (i, j) in TARGET_PAIRS.items():
        gaps.append(max(0.0, pair_dist(ca, i, j) - native[k]))
    return float(np.mean(gaps))


def main() -> int:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    ref_ca, _ = load_ca_from_pdb(CRAMBIN_GOLD)
    native_pair = {k: pair_dist(ref_ca, *ij) for k, ij in TARGET_PAIRS.items()}

    # 1) Build tunnel snapshot (no post-extrusion refinement).
    t0 = time.perf_counter()
    snap = minimize_full_chain(
        CRAMBIN_SEQ,
        quick=True,
        simulate_ribosome_tunnel=True,
        post_extrusion_refine=False,
        fast_pass_steps_per_connection=2,
        min_pass_iter_per_connection=5,
        fast_local_theta=True,
        horizon_neighbor_cutoff=10.0,
        kappa_dihedral=0.01,
        hbond_weight=0.0,
        ensemble_translation_mix_alpha=0.03,
        tunnel_free_terminus_steps=10,
        tunnel_free_terminus_window=12,
        tunnel_handedness_bias_weight=0.03,
        tunnel_handedness_target=0.45,
        tunnel_handedness_sign=1.0,
    )
    ca = np.asarray(snap["ca_min"], dtype=float).copy()
    n = int(ca.shape[0])
    z_list = np.full((n,), 6.0, dtype=float)

    # 2) Stepwise drive from snapshot.
    linear_momentum_state = np.zeros((n, 3), dtype=float)
    barrier_budget_state = np.zeros((n,), dtype=float)
    omega_state = np.zeros((3,), dtype=float)
    direction_set = np.array(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        dtype=float,
    )

    grad_kwargs = {"neighbor_cutoff": 10.0, "em_scale": 1.0}
    mix_alpha = 0.03
    max_steps = 120
    bad_patience = 10

    records = []
    base_gap = pair_gap_score(ca, native_pair)
    best_gap = base_gap
    worse_count = 0
    stop_reason = "max_steps"

    for step_idx in range(max_steps):
        g = grad_full(
            ca,
            z_list,
            include_bonds=True,
            include_horizon=True,
            include_clash=True,
            **grad_kwargs,
        )
        g_norm = float(np.linalg.norm(g))
        if g_norm < 1e-7:
            stop_reason = "grad_small"
            break
        step = 0.5 / (g_norm + 1e-6)
        delta_grad = -step * g
        sel = choose_best_translation_direction(
            grad=g,
            positions=ca,
            step=0.35,
            span=0.25,
            p=1.0,
            beta=0.35,
            score_lambda=0.0,
            direction_set=direction_set,
            sources=(0, -1),
            residue_masses=z_list,
            left_anchor_infinite=False,
            linear_momentum_state=linear_momentum_state,
            barrier_budget_state=barrier_budget_state,
            inertial_dt=0.6,
            linear_damping=0.85,
            linear_gain=0.6,
            damping_mode="sqrt",
            barrier_decay=0.98,
            barrier_build=0.01,
            barrier_relief=0.2,
            omega_state=omega_state,
            angular_mix=0.0,
            angular_damping=0.9,
            angular_gain=0.2,
        )
        disp = np.asarray(sel["best_disp"], dtype=float)
        linear_momentum_state = np.asarray(sel["best_linear_momentum_state"], dtype=float).reshape(n, 3)
        barrier_budget_state = np.asarray(sel["best_barrier_budget_state"], dtype=float).reshape(n)
        omega_state = np.asarray(sel["best_omega_state"], dtype=float).reshape(3)

        n_disp = float(np.linalg.norm(disp)) + 1e-14
        n_delta = float(np.linalg.norm(delta_grad)) + 1e-14
        disp_scaled = disp * (n_delta / n_disp)
        ca = ca + (1.0 - mix_alpha) * delta_grad + mix_alpha * disp_scaled
        ca = _project_bonds(ca, r_min=2.5, r_max=6.0)

        gap = pair_gap_score(ca, native_pair)
        if gap < best_gap:
            best_gap = gap
            worse_count = 0
        else:
            worse_count += 1

        pair_d = {k: pair_dist(ca, *ij) for k, ij in TARGET_PAIRS.items()}
        s2 = local_s2_angles(ca)
        records.append(
            {
                "step": int(step_idx + 1),
                "gap_score": float(gap),
                "best_gap_score": float(best_gap),
                "pair_dist_ang": pair_d,
                "s2_mean": float(np.mean(s2)),
                "s2_max": float(np.max(s2)),
                "barrier_mean": float(np.mean(barrier_budget_state)),
                "barrier_max": float(np.max(barrier_budget_state)),
                "mass_profile_mean": float(np.mean(np.asarray(sel["mass_profile"], dtype=float))),
                "bond_left_mass_head": [float(x) for x in np.asarray(sel["bond_left_mass"], dtype=float)[:6]],
                "bond_right_mass_head": [float(x) for x in np.asarray(sel["bond_right_mass"], dtype=float)[:6]],
                "best_score": float(sel["best_score"]),
            }
        )

        # Unexperimental: sustained worsening of long-range closure gap.
        if worse_count >= bad_patience and gap > (base_gap * 1.03):
            stop_reason = "unexperimental_gap_worsening"
            break

    # Final RMSD for the stepped state.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False, encoding="utf-8") as pf:
        from horizon_physics.proteins.full_protein_minimizer import _place_full_backbone, _add_cb
        # make temporary fold object to reuse exporter
        backbone_atoms = _place_full_backbone(ca, CRAMBIN_SEQ)
        obj = {"ca_min": ca, "backbone_atoms": backbone_atoms, "sequence": CRAMBIN_SEQ, "n_res": len(CRAMBIN_SEQ)}
        pf.write(full_chain_to_pdb(obj))
        pred_path = pf.name
    try:
        rmsd, _, _, _ = ca_rmsd(pred_path, CRAMBIN_GOLD, align_by_resid=False, trim_to_min_length=True)
    finally:
        os.unlink(pred_path)

    out = {
        "snapshot_build_seconds": float(time.perf_counter() - t0),
        "target_pairs_native_ang": native_pair,
        "initial_gap_score": float(base_gap),
        "final_gap_score": float(records[-1]["gap_score"] if records else base_gap),
        "stop_reason": stop_reason,
        "n_steps_executed": int(len(records)),
        "final_rmsd_ang": float(rmsd),
        "records": records,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("WROTE", OUT_PATH)
    print("stop_reason", stop_reason, "steps", len(records), "final_rmsd", float(rmsd))
    if records:
        print("initial_gap", float(base_gap), "final_gap", float(records[-1]["gap_score"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

