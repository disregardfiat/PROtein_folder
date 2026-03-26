#!/usr/bin/env python3
"""
Side-by-side stepwise WHIP (force-carrier) minimization vs experimental crambin (1CRN).

All variants start from the *same* tunnel snapshot so differences are purely from
translation hyperparameters and EM field refresh policy.

Outputs JSON with per-step traces, detected "large translation" events, final Cα RMSD,
and (by default) a **fold_analysis** block: clashes, Ramachandran vs native, missed /
spurious long-range contacts, and heuristic **tuning_hints** for pruning bad geometry.

**Run until motion stalls** (instead of a fixed step budget):

  python3 scripts/crambin_stepwise_side_by_side.py --until-still --safety-max-steps 30000 \\
      --record-every 25 --out .casp_grade_outputs/iter_small/crambin_until_still.json

Example (fixed step budget; default is 1000):

  python3 scripts/crambin_stepwise_side_by_side.py --out .casp_grade_outputs/iter_small/crambin_stepwise_ab.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from horizon_physics.proteins.force_carrier_ensemble import (
    build_direction_set_6_axes,
    choose_best_translation_direction,
    maybe_refresh_em_field_direction_set,
)
from horizon_physics.proteins.folding_energy import grad_full
from horizon_physics.proteins.full_protein_minimizer import full_chain_to_pdb, minimize_full_chain
from horizon_physics.proteins.backbone_phi_psi import backbone_phi_psi_from_atoms
from horizon_physics.proteins.grade_folds import (
    ca_rmsd,
    load_backbone_atoms_ordered_from_pdb,
    load_ca_from_pdb,
)
from horizon_physics.proteins.gradient_descent_folding import _project_bonds
from horizon_physics.proteins.peptide_backbone import (
    PHI_ALPHA_DEG,
    PHI_BETA_DEG,
    PSI_ALPHA_DEG,
    PSI_BETA_DEG,
)

CRAMBIN_SEQ = "TTCCPSIVARSNFNVCRLPGTPEAIICGDVCDLDCTAKTCFSIICT"
CRAMBIN_GOLD = os.path.join(REPO, "proteins", "1CRN.pdb")
# Lean anchor from HQIV: etaModePhi is constant at referenceM=4.
REFERENCE_M = 4

# Long-range pairs used in second-fold local refine sweeps (1-based indices).
NATIVE_PAIR_KEYS = ("1-32", "3-40", "8-34", "16-26")
NATIVE_PAIRS_1BASE: List[Tuple[str, int, int]] = [
    ("1-32", 1, 32),
    ("3-40", 3, 40),
    ("8-34", 8, 34),
    ("16-26", 16, 26),
]


@dataclass
class RunLimits:
    """Budget and convergence for the stepwise translation loop (one WHIP step per iteration)."""

    until_still: bool = False
    safety_max_steps: int = 50_000
    motion_floor: float = 1e-4
    still_patience: int = 150
    grad_tol: float = 1e-7
    record_every: int = 1
    disable_gap_early_stop: bool = False


def pair_dist(ca: np.ndarray, i1: int, j1: int) -> float:
    return float(np.linalg.norm(ca[i1 - 1] - ca[j1 - 1]))


def eta_mode_phi_constant(reference_m: int = REFERENCE_M) -> float:
    """Lean `etaModePhi_constant`: 1 / ((referenceM + 2) * (referenceM + 1))."""
    rm = int(reference_m)
    return 1.0 / float((rm + 2) * (rm + 1))


def native_pair_targets(ref_ca: np.ndarray) -> Dict[str, float]:
    return {k: pair_dist(ref_ca, i, j) for k, i, j in NATIVE_PAIRS_1BASE}


def gap_vs_native(ca: np.ndarray, native: Dict[str, float]) -> float:
    gaps = []
    for k, i, j in NATIVE_PAIRS_1BASE:
        gaps.append(max(0.0, pair_dist(ca, i, j) - native[k]))
    return float(np.mean(gaps))


def ca_rmsd_from_state(ca: np.ndarray, seq: str) -> float:
    from horizon_physics.proteins.full_protein_minimizer import _add_cb, _place_full_backbone

    backbone_atoms = _place_full_backbone(ca, seq)
    obj = {"ca_min": ca, "backbone_atoms": backbone_atoms, "sequence": seq, "n_res": len(seq)}
    pdb_str = full_chain_to_pdb(obj)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False, encoding="utf-8") as pf:
        pf.write(pdb_str)
        pred_path = pf.name
    try:
        r, _, _, _ = ca_rmsd(pred_path, CRAMBIN_GOLD, align_by_resid=False, trim_to_min_length=True)
        return float(r)
    finally:
        os.unlink(pred_path)


def ca_to_pdb_str(ca: np.ndarray, seq: str) -> str:
    """Export a full-chain PDB string from a Cα-only backbone state (debug/analysis)."""
    from horizon_physics.proteins.full_protein_minimizer import _place_full_backbone

    backbone_atoms = _place_full_backbone(ca, seq)
    obj = {"ca_min": ca, "backbone_atoms": backbone_atoms, "sequence": seq, "n_res": len(seq)}
    return full_chain_to_pdb(obj)


def _min_dist_ramachandran_basins_deg(phi_deg: np.ndarray, psi_deg: np.ndarray) -> np.ndarray:
    d_a = np.hypot(phi_deg - float(PHI_ALPHA_DEG), psi_deg - float(PSI_ALPHA_DEG))
    d_b = np.hypot(phi_deg - float(PHI_BETA_DEG), psi_deg - float(PSI_BETA_DEG))
    return np.minimum(d_a, d_b)


def count_ca_clashes(
    ca: np.ndarray,
    *,
    min_seq_sep: int = 2,
    clash_cutoff_ang: float = 2.0,
    max_report: int = 40,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Non-bonded Cα–Cα pairs closer than clash_cutoff_ang (Å)."""
    n = int(ca.shape[0])
    hits: List[Dict[str, Any]] = []
    for i in range(n):
        for j in range(i + min_seq_sep, n):
            d = float(np.linalg.norm(ca[j] - ca[i]))
            if d < clash_cutoff_ang:
                hits.append({"i_1based": i + 1, "j_1based": j + 1, "dist_ang": d})
    return int(len(hits)), hits[:max_report]


def analyze_final_fold(
    ca_pred: np.ndarray,
    seq: str,
    gold_pdb: str,
    ref_ca: np.ndarray,
    native_pair_detail: Dict[str, float],
    *,
    ramachandran_outlier_deg: float = 75.0,
    native_ramachandran_tight_deg: float = 55.0,
    contact_min_seq_sep: int = 4,
    native_contact_cutoff_ang: float = 7.5,
    missed_contact_excess_ang: float = 3.0,
    gold_chain_id: Optional[str] = "A",
) -> Dict[str, Any]:
    """
    Compare final Cα trace to experimental: clashes, φ/ψ vs native, long-range contacts,
    tracked native pairs, and heuristic tuning hints.
    """
    from horizon_physics.proteins.full_protein_minimizer import _place_full_backbone

    n = len(seq)
    clash_n, clash_list = count_ca_clashes(ca_pred, min_seq_sep=2, clash_cutoff_ang=2.0)
    rg_pred = float(np.sqrt(np.mean(np.sum((ca_pred - np.mean(ca_pred, axis=0)) ** 2, axis=1))))
    rg_ref = float(np.sqrt(np.mean(np.sum((ref_ca - np.mean(ref_ca, axis=0)) ** 2, axis=1))))

    native_bb = load_backbone_atoms_ordered_from_pdb(gold_pdb, chain_id=gold_chain_id)
    pred_bb = _place_full_backbone(ca_pred, seq)
    phi_n, psi_n = backbone_phi_psi_from_atoms(native_bb)
    phi_p, psi_p = backbone_phi_psi_from_atoms(pred_bb)
    m = min(int(phi_n.shape[0]), int(phi_p.shape[0]), n)
    phi_nd = np.degrees(phi_n[:m])
    psi_nd = np.degrees(psi_n[:m])
    phi_pd = np.degrees(phi_p[:m])
    psi_pd = np.degrees(psi_p[:m])
    d_nat = _min_dist_ramachandran_basins_deg(phi_nd, psi_nd)
    d_pred = _min_dist_ramachandran_basins_deg(phi_pd, psi_pd)
    ram_out_pred: List[int] = []
    ram_worse_than_native: List[int] = []
    for idx in range(1, m - 1):
        if idx < len(seq) and seq[idx] == "G":
            continue
        if d_pred[idx] > ramachandran_outlier_deg:
            ram_out_pred.append(idx + 1)
        if d_pred[idx] > d_nat[idx] + 25.0 and d_nat[idx] < native_ramachandran_tight_deg:
            ram_worse_than_native.append(idx + 1)

    tracked_pairs: List[Dict[str, Any]] = []
    for k, i, j in NATIVE_PAIRS_1BASE:
        d_p = pair_dist(ca_pred, i, j)
        tgt = float(native_pair_detail[k])
        tracked_pairs.append(
            {
                "pair": k,
                "dist_pred_ang": d_p,
                "dist_native_ang": tgt,
                "excess_ang": float(max(0.0, d_p - tgt)),
            }
        )

    missed_contacts: List[Dict[str, Any]] = []
    spurious_tight: List[Dict[str, Any]] = []
    nr = min(int(ref_ca.shape[0]), int(ca_pred.shape[0]))
    for i in range(nr):
        for j in range(i + contact_min_seq_sep, nr):
            d_r = float(np.linalg.norm(ref_ca[j] - ref_ca[i]))
            d_p = float(np.linalg.norm(ca_pred[j] - ca_pred[i]))
            if d_r <= native_contact_cutoff_ang and d_p > d_r + missed_contact_excess_ang:
                missed_contacts.append(
                    {
                        "i_1based": i + 1,
                        "j_1based": j + 1,
                        "dist_native_ang": d_r,
                        "dist_pred_ang": d_p,
                        "excess_ang": d_p - d_r,
                    }
                )
            if d_r >= 12.0 and d_p < 5.0:
                spurious_tight.append(
                    {
                        "i_1based": i + 1,
                        "j_1based": j + 1,
                        "dist_native_ang": d_r,
                        "dist_pred_ang": d_p,
                    }
                )

    missed_contacts.sort(key=lambda x: -x["excess_ang"])
    spurious_tight.sort(key=lambda x: x["dist_pred_ang"])

    tuning_hints: List[str] = []
    if clash_n >= 3:
        tuning_hints.append(
            "Many Cα clashes (<2 Å, non-bonded): strengthen clash gradient in grad_full, "
            "tighten bond projection, or reduce carrier_step / mix_alpha."
        )
    if len(ram_out_pred) > max(5, m // 8):
        tuning_hints.append(
            "Many Ramachandran outliers vs alpha/beta basins: raise kappa_dihedral on minimize_full_chain "
            "or add a post pass with dihedral / discrete φψ refinement."
        )
    if len(missed_contacts) > 15:
        tuning_hints.append(
            "Many native sub-7.5 Å contacts are blown open in the prediction: increase long-range "
            "signal (hbond_weight, variational pairs), WHIP barrier_drive toward native pairs, "
            "or run tree-torque / staged refine after stepwise WHIP."
        )
    if len(spurious_tight) > 8:
        tuning_hints.append(
            "Several residue pairs are <5 Å in pred but >12 Å in native (over-compaction / wrong topology): "
            "prune moves that shrink Rg without improving tracked native gaps; consider Rg or "
            "collective-kink regularizers."
        )
    if tracked_pairs and all(float(tp["excess_ang"]) > 2.0 for tp in tracked_pairs):
        tuning_hints.append(
            "All tracked long-range native pairs remain >2 Å too long: algorithm missed the native fold "
            "topology; prioritize long-range objectives before local polish."
        )

    return {
        "n_ca_clashes": clash_n,
        "ca_clash_examples": clash_list,
        "rg_pred_ang": rg_pred,
        "rg_native_ang": rg_ref,
        "ramachandran_outlier_residues_1based": ram_out_pred[:50],
        "n_ramachandran_outliers": len(ram_out_pred),
        "ramachandran_worse_than_native_1based": ram_worse_than_native[:50],
        "n_ramachandran_worse_than_native": len(ram_worse_than_native),
        "tracked_native_pairs": tracked_pairs,
        "missed_native_contacts": missed_contacts[:80],
        "n_missed_native_contacts": len(missed_contacts),
        "spurious_tight_longrange": spurious_tight[:40],
        "n_spurious_tight_longrange": len(spurious_tight),
        "contact_criteria": {
            "min_seq_sep": contact_min_seq_sep,
            "native_contact_cutoff_ang": native_contact_cutoff_ang,
            "missed_contact_excess_ang": missed_contact_excess_ang,
        },
        "tuning_hints": tuning_hints,
    }


def _push_top_steps_by_disp(bucket: List[Dict[str, Any]], row: Dict[str, Any], k: int = 12) -> None:
    bucket.append(row)
    bucket.sort(key=lambda r: float(r["max_disp_scaled"]), reverse=True)
    del bucket[k:]


def _dihedral_rad(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    b1 = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    b2 = np.asarray(p3, dtype=float) - np.asarray(p2, dtype=float)
    b3 = np.asarray(p4, dtype=float) - np.asarray(p3, dtype=float)
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1n = float(np.linalg.norm(n1))
    n2n = float(np.linalg.norm(n2))
    if n1n < 1e-12 or n2n < 1e-12:
        return 0.0
    n1 /= n1n
    n2 /= n2n
    b2u = b2 / (np.linalg.norm(b2) + 1e-12)
    m1 = np.cross(n1, b2u)
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    return float(np.arctan2(y, x))


def estimate_directional_budget_states(ca: np.ndarray) -> Tuple[float, float, float]:
    """
    Geometry-derived directional budget states in [0, 1]:
    - bend: mean normalized Cα bond-angle deviation from straight
    - torsion: mean |dihedral| normalized by π
    - conjugation: second-difference roughness of bond directions
    """
    pos = np.asarray(ca, dtype=float)
    n = int(pos.shape[0])
    if n < 4:
        return 0.0, 0.0, 0.0
    bonds = pos[1:] - pos[:-1]
    bnorm = np.linalg.norm(bonds, axis=1, keepdims=True) + 1e-12
    u = bonds / bnorm

    bend_vals: List[float] = []
    for i in range(1, n - 1):
        c = float(np.clip(np.dot(u[i - 1], u[i]), -1.0, 1.0))
        ang = float(np.arccos(c))  # [0, pi]
        bend_vals.append(abs(np.pi - ang) / np.pi)
    bend_state = float(np.clip(np.mean(bend_vals) if bend_vals else 0.0, 0.0, 1.0))

    tors_vals: List[float] = []
    for i in range(0, n - 3):
        tors_vals.append(abs(_dihedral_rad(pos[i], pos[i + 1], pos[i + 2], pos[i + 3])) / np.pi)
    tors_state = float(np.clip(np.mean(tors_vals) if tors_vals else 0.0, 0.0, 1.0))

    if u.shape[0] >= 3:
        second = u[2:] - 2.0 * u[1:-1] + u[:-2]
        conj_state = float(
            np.clip(np.mean(np.linalg.norm(second, axis=1)) / np.sqrt(8.0), 0.0, 1.0)
        )
    else:
        conj_state = 0.0
    return bend_state, tors_state, conj_state


def phased_scale(step_n: int, phase_a_steps: int, start_scale: float) -> float:
    """Linear two-phase anneal from start_scale -> 1 across phase_a_steps."""
    if phase_a_steps <= 0:
        return 1.0
    t = float(np.clip(float(step_n) / float(max(1, phase_a_steps)), 0.0, 1.0))
    s0 = float(np.clip(start_scale, 0.0, 1.0))
    return float(s0 + (1.0 - s0) * t)


def compute_stepwise_diagnostics(
    records: List[Dict[str, Any]],
    *,
    initial_gap: float,
    stop_reason: str,
    native_pair: Dict[str, float],
    rebound_margin_ang: float = 0.35,
) -> Dict[str, Any]:
    """
    Summarize the translation timeline for tuning when runs hit a step cap.

    Surfaces *where* mean native-pair excess was best, when it first materially
    rebounded, and how RMSD evolved when sampled — so ``records`` are not the
    only place to look.
    """
    if not records:
        return {"note": "no records"}

    steps = [int(r["step"]) for r in records]
    gaps = [float(r["gap_score"]) for r in records]
    best_i = int(np.argmin(gaps))
    best_step = int(steps[best_i])
    best_gap = float(gaps[best_i])
    final_gap = float(gaps[-1])
    first_step = int(steps[0])

    # First step after the global best index where gap is clearly worse (local rebound).
    first_rebound_after_best: Optional[int] = None
    for j in range(best_i + 1, len(gaps)):
        if gaps[j] > best_gap + float(rebound_margin_ang):
            first_rebound_after_best = int(steps[j])
            break

    n_improved = 0
    for j in range(1, len(gaps)):
        if gaps[j] < gaps[j - 1] - 1e-12:
            n_improved += 1

    longest_worsen_streak = 0
    cur = 0
    for j in range(1, len(gaps)):
        if gaps[j] >= gaps[j - 1] - 1e-12:
            cur += 1
            longest_worsen_streak = max(longest_worsen_streak, cur)
        else:
            cur = 0

    rmsd_points: List[Dict[str, float]] = []
    for r in records:
        rv = r.get("ca_rmsd_ang")
        if rv is not None:
            rmsd_points.append({"step": float(r["step"]), "ca_rmsd_ang": float(rv)})
    rmsd_diag: Dict[str, Any] = {"n_samples": len(rmsd_points)}
    if rmsd_points:
        best_r = min(rmsd_points, key=lambda x: x["ca_rmsd_ang"])
        worst_r = max(rmsd_points, key=lambda x: x["ca_rmsd_ang"])
        rmsd_diag.update(
            {
                "first_sample": rmsd_points[0],
                "last_sample": rmsd_points[-1],
                "best_rmsd_sample": best_r,
                "worst_rmsd_sample": worst_r,
            }
        )

    per_pair_closest: List[Dict[str, Any]] = []
    for k in native_pair:
        tgt = float(native_pair[k])
        best_j = 0
        best_excess = float("inf")
        for j, r in enumerate(records):
            pd = r.get("pair_dist_ang") or {}
            if k not in pd:
                continue
            ex = max(0.0, float(pd[k]) - tgt)
            if ex < best_excess:
                best_excess = ex
                best_j = j
        per_pair_closest.append(
            {
                "pair": k,
                "step_closest_to_native": int(steps[best_j]),
                "excess_ang_at_that_step": float(best_excess),
                "dist_ang_at_that_step": float((records[best_j].get("pair_dist_ang") or {}).get(k, 0.0)),
            }
        )

    hit_cap = stop_reason in ("max_steps", "safety_max_steps")
    narrative: List[str] = []
    if hit_cap:
        narrative.append(
            f"Stopped at step cap ({stop_reason}): best mean native-pair gap {best_gap:.4f} Å at step {best_step}; "
            f"final gap {final_gap:.4f} Å (Δ vs best {final_gap - best_gap:+.4f} Å)."
        )
        if first_rebound_after_best is not None:
            narrative.append(
                f"First rebound ≥{rebound_margin_ang:g} Å above best gap after that optimum: step {first_rebound_after_best} "
                f"— inspect ``records`` (and optional PDB dumps) around steps {best_step}–{first_rebound_after_best}."
            )
        else:
            narrative.append(
                "No clear rebound past margin after the best gap; trajectory may be slowly drifting or stalled "
                "rather than one sharp derail — check RMSD samples and per-pair closest steps."
            )
    else:
        narrative.append(f"Stopped: {stop_reason}. Best gap {best_gap:.4f} Å at step {best_step}; final {final_gap:.4f} Å.")

    return {
        "stop_reason": str(stop_reason),
        "hit_step_cap": bool(hit_cap),
        "first_recorded_step": first_step,
        "last_recorded_step": int(steps[-1]),
        "n_records": len(records),
        "initial_gap_score": float(initial_gap),
        "best_gap_score": best_gap,
        "best_gap_step": best_step,
        "final_gap_score": final_gap,
        "gap_improvement_initial_to_best": float(initial_gap - best_gap),
        "gap_erosion_best_to_final": float(final_gap - best_gap),
        "first_rebound_step_after_best": first_rebound_after_best,
        "rebound_margin_ang": float(rebound_margin_ang),
        "n_record_steps_with_strict_gap_improvement": int(n_improved),
        "longest_monotone_non_improving_gap_run": int(longest_worsen_streak),
        "rmsd": rmsd_diag,
        "tracked_pairs_closest_to_native": per_pair_closest,
        "narrative": narrative,
    }


@dataclass
class WVariant:
    """One WHIP configuration for the shared step loop."""

    name: str
    mix_alpha: float = 0.08
    carrier_step: float = 0.35
    carrier_span: float = 0.25
    carrier_p: float = 1.0
    carrier_beta: float = 0.35
    score_lambda: float = 0.0
    inertial_dt: float = 1.0
    linear_damping: float = 0.88
    linear_gain: float = 1.4
    damping_mode: str = "sqrt"
    barrier_decay: float = 0.98
    barrier_build: float = 0.06
    barrier_relief: float = 0.4
    barrier_drive_gain: float = 0.05
    barrier_floor: float = 0.03
    barrier_trigger_offset: float = 1e-12
    barrier_kick_gain: float = 0.0
    directional_budget_base: float = 0.0
    directional_budget_bend_coeff: float = 0.0
    directional_budget_torsion_coeff: float = 0.0
    directional_budget_conjugation_coeff: float = 0.0
    directional_budget_bend_state: float = 0.0
    directional_budget_torsion_state: float = 0.0
    directional_budget_conjugation_state: float = 0.0
    dynamic_directional_budget_states: bool = False
    budget_phase_a_steps: int = 150
    budget_start_scale: float = 0.2
    phase_a_carrier_step_boost: float = 1.15
    phase_a_carrier_span_boost: float = 1.10
    wave_leak_floor: float = 0.18
    resonance_decay: float = 0.92
    resonance_gain: float = 0.35
    resonance_harmonic_weight: float = 0.5
    resonance_damping_span: float = 0.35
    resonance_gain_boost: float = 0.5
    angular_mix: float = 0.0
    angular_damping: float = 0.9
    angular_gain: float = 0.2
    use_em_field_refresh: bool = True
    # Align with ``folding_energy.R_HORIZON`` / ``grad_horizon_full``: refresh when a nonlocal
    # pair crosses *into* this radius (Å), turning on direct horizon coupling; optionally when
    # a pair *leaves* (coupling drops off).
    em_refresh_on_horizon_crossing: bool = True
    em_refresh_on_horizon_leaving: bool = False
    em_refresh_horizon_ang: float = 15.0
    em_refresh_min_seq_sep: int = 3
    # Optional legacy: also allow refresh on large raw carrier displacement.
    em_refresh_on_large_disp: bool = False
    em_refresh_large_disp_thresh: float = 0.35
    em_max_extra_directions: int = 24
    # Log "large translation + pair drop" events using this displacement threshold (fair across variants).
    event_disp_thresh: float = 0.45
    max_steps: int = 1000
    bad_patience: int = 10
    grad_neighbor_cutoff: float = 10.0
    grad_em_scale: float = 1.0
    # Small HQIV collective kink gradient (FD) on Cα: steers interior virtual bond angles toward
    # the default α-helix reference so aggressive carrier waves do not dominate in unphysical bends.
    collective_kink_weight: float = 0.0
    collective_kink_m: int = 3
    # Update law:
    # - "blend": legacy panel rule (scaled carrier + convex blend with local descent)
    # - "natural_hqiv": Lean ProteinNaturalFolding:
    #       naturalDisp = etaModePhi(m) * localDescentDisp + carrierDisp
    update_rule: str = "blend"
    eta_mode_phi: float = eta_mode_phi_constant()

    def to_serializable(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def all_variants() -> List[WVariant]:
    """All defined variants (kept for explicit A/B requests and reproducibility).

    Non-legacy variants use stronger **carrier_span / carrier_step / wave_leak_floor** so
    terminus-driven waves propagate farther along the chain (faster large-scale bending), plus
    a light **collective_kink_weight** so the gradient still favors HQIV-physical Cα kink geometry.
    """
    return [
        WVariant(
            name="legacy_sqrt_6axis_no_barrier_floor",
            mix_alpha=0.03,
            inertial_dt=0.6,
            linear_damping=0.85,
            linear_gain=0.6,
            damping_mode="sqrt",
            barrier_decay=0.98,
            barrier_build=0.01,
            barrier_relief=0.2,
            barrier_drive_gain=0.0,
            barrier_floor=0.0,
            wave_leak_floor=0.12,
            use_em_field_refresh=False,
        ),
        WVariant(
            name="sqrt_em_refresh_horizon15",
            mix_alpha=0.085,
            carrier_step=0.44,
            carrier_span=0.40,
            inertial_dt=1.05,
            linear_damping=0.88,
            linear_gain=1.62,
            damping_mode="sqrt",
            barrier_decay=0.98,
            barrier_build=0.06,
            barrier_relief=0.48,
            barrier_drive_gain=0.08,
            barrier_floor=0.03,
            wave_leak_floor=0.28,
            use_em_field_refresh=True,
            em_refresh_on_horizon_crossing=True,
            em_refresh_on_horizon_leaving=True,
            em_refresh_horizon_ang=15.0,
            collective_kink_weight=0.03,
            collective_kink_m=3,
        ),
        WVariant(
            name="natural_hqiv_em_refresh_horizon15",
            update_rule="natural_hqiv",
            eta_mode_phi=eta_mode_phi_constant(),
            mix_alpha=0.0,  # unused in natural_hqiv; keep explicit for serialization
            carrier_step=0.42,
            carrier_span=0.42,
            inertial_dt=1.08,
            linear_damping=0.88,
            linear_gain=1.65,
            damping_mode="resonant",
            barrier_decay=0.98,
            barrier_build=0.06,
            barrier_relief=0.48,
            barrier_drive_gain=0.0,
            barrier_floor=0.03,
            barrier_trigger_offset=1.0,
            barrier_kick_gain=0.30,
            directional_budget_base=0.35,
            directional_budget_bend_coeff=0.18,
            directional_budget_torsion_coeff=0.12,
            directional_budget_conjugation_coeff=0.10,
            directional_budget_bend_state=0.0,
            directional_budget_torsion_state=0.0,
            directional_budget_conjugation_state=0.0,
            dynamic_directional_budget_states=True,
            budget_phase_a_steps=150,
            budget_start_scale=0.15,
            phase_a_carrier_step_boost=1.18,
            phase_a_carrier_span_boost=1.12,
            wave_leak_floor=0.30,
            resonance_decay=0.90,
            resonance_gain=0.50,
            resonance_harmonic_weight=0.65,
            resonance_damping_span=0.30,
            resonance_gain_boost=0.70,
            use_em_field_refresh=True,
            em_refresh_on_horizon_crossing=True,
            em_refresh_on_horizon_leaving=True,
            em_refresh_horizon_ang=15.0,
            collective_kink_weight=0.03,
            collective_kink_m=3,
        ),
        WVariant(
            name="resonant_em_refresh_horizon15",
            mix_alpha=0.085,
            carrier_step=0.44,
            carrier_span=0.40,
            inertial_dt=1.05,
            linear_damping=0.88,
            linear_gain=1.62,
            damping_mode="resonant",
            barrier_decay=0.98,
            barrier_build=0.06,
            barrier_relief=0.48,
            barrier_drive_gain=0.08,
            barrier_floor=0.03,
            wave_leak_floor=0.28,
            resonance_decay=0.90,
            resonance_gain=0.48,
            resonance_harmonic_weight=0.62,
            resonance_damping_span=0.30,
            resonance_gain_boost=0.68,
            use_em_field_refresh=True,
            em_refresh_on_horizon_crossing=True,
            em_refresh_on_horizon_leaving=True,
            em_refresh_horizon_ang=15.0,
            collective_kink_weight=0.03,
            collective_kink_m=3,
        ),
        WVariant(
            name="resonant_no_em_refresh",
            mix_alpha=0.085,
            carrier_step=0.44,
            carrier_span=0.40,
            inertial_dt=1.05,
            linear_damping=0.88,
            linear_gain=1.62,
            damping_mode="resonant",
            barrier_decay=0.98,
            barrier_build=0.06,
            barrier_relief=0.48,
            barrier_drive_gain=0.08,
            barrier_floor=0.03,
            wave_leak_floor=0.28,
            resonance_decay=0.90,
            resonance_gain=0.48,
            resonance_harmonic_weight=0.62,
            resonance_damping_span=0.30,
            resonance_gain_boost=0.68,
            use_em_field_refresh=False,
            collective_kink_weight=0.03,
            collective_kink_m=3,
        ),
        WVariant(
            name="sqrt_em_refresh_large_disp035",
            mix_alpha=0.085,
            carrier_step=0.44,
            carrier_span=0.40,
            inertial_dt=1.05,
            linear_damping=0.88,
            linear_gain=1.62,
            damping_mode="sqrt",
            barrier_decay=0.98,
            barrier_build=0.06,
            barrier_relief=0.48,
            barrier_drive_gain=0.08,
            barrier_floor=0.03,
            wave_leak_floor=0.28,
            use_em_field_refresh=True,
            em_refresh_on_horizon_crossing=False,
            em_refresh_on_large_disp=True,
            em_refresh_large_disp_thresh=0.35,
            collective_kink_weight=0.03,
            collective_kink_m=3,
        ),
    ]


def default_variants() -> List[WVariant]:
    """
    Default ensemble policy aligned with Lean `ProteinVariantSelection` admissibility:
    keep variants with low clash risk (`caClashes <= 5` in the benchmark table) and
    zero Ramachandran outliers, while preserving all variant definitions in `all_variants()`.
    """
    keep = {
        "natural_hqiv_em_refresh_horizon15",
        "resonant_em_refresh_horizon15",
        "resonant_no_em_refresh",
        "sqrt_em_refresh_large_disp035",
    }
    return [v for v in all_variants() if v.name in keep]


def run_variant(
    ca0: np.ndarray,
    z_list: np.ndarray,
    native_pair: Dict[str, float],
    base_gap: float,
    v: WVariant,
    ref_ca: np.ndarray,
    *,
    realistic_drop: float,
    rmsd_every: int,
    pdb_every: int,
    pdb_out_dir: Optional[str],
    pdb_on_event: bool,
    run_limits: Optional[RunLimits] = None,
    skip_fold_analysis: bool = False,
) -> Dict[str, Any]:
    rl = run_limits or RunLimits()
    if rl.until_still:
        step_limit = int(rl.safety_max_steps)
    else:
        step_limit = min(int(v.max_steps), int(rl.safety_max_steps))

    ca = np.asarray(ca0, dtype=float).copy()
    n = int(ca.shape[0])
    base_axes = build_direction_set_6_axes()
    direction_set_active = np.asarray(base_axes, dtype=float).copy()

    linear_momentum_state = np.zeros((n, 3), dtype=float)
    barrier_budget_state = np.zeros((n,), dtype=float)
    resonance_state: Optional[float] = 0.0
    omega_state = np.zeros((3,), dtype=float)

    grad_kwargs: Dict[str, Any] = {
        "neighbor_cutoff": float(v.grad_neighbor_cutoff),
        "em_scale": float(v.grad_em_scale),
    }
    if float(v.collective_kink_weight) > 0.0:
        grad_kwargs["collective_kink_weight"] = float(v.collective_kink_weight)
        grad_kwargs["collective_kink_m"] = int(v.collective_kink_m)
    records: List[Dict[str, Any]] = []
    prev_pair = {k: pair_dist(ca, i, j) for k, i, j in NATIVE_PAIRS_1BASE}
    events: List[Dict[str, Any]] = []
    best_gap = base_gap
    worse_count = 0
    stop_reason = "max_steps"

    agg_enter = 0
    agg_leave = 0
    agg_refresh = 0
    top_steps: List[Dict[str, Any]] = []
    still_counter = 0
    last_record: Optional[Dict[str, Any]] = None
    total_translation_steps = 0

    for step_idx in range(step_limit):
        ca_before = ca.copy()
        g = grad_full(
            ca,
            z_list,
            include_bonds=True,
            include_horizon=True,
            include_clash=True,
            **grad_kwargs,
        )
        g_norm = float(np.linalg.norm(g))
        if g_norm < float(rl.grad_tol):
            stop_reason = "grad_small"
            total_translation_steps = int(step_idx + 1)
            break
        step = 0.5 / (g_norm + 1e-6)
        delta_grad = -step * g

        rs = resonance_state if v.damping_mode.strip().lower() == "resonant" else None

        step_n = int(step_idx + 1)
        phase_scale = phased_scale(step_n, int(v.budget_phase_a_steps), float(v.budget_start_scale))
        step_eff = float(v.carrier_step) * (
            1.0 + (float(v.phase_a_carrier_step_boost) - 1.0) * (1.0 - phase_scale)
        )
        span_eff = float(v.carrier_span) * (
            1.0 + (float(v.phase_a_carrier_span_boost) - 1.0) * (1.0 - phase_scale)
        )
        bend_state = float(v.directional_budget_bend_state)
        tors_state = float(v.directional_budget_torsion_state)
        conj_state = float(v.directional_budget_conjugation_state)
        if bool(v.dynamic_directional_budget_states):
            bend_state, tors_state, conj_state = estimate_directional_budget_states(ca)
        sel = choose_best_translation_direction(
            grad=g,
            positions=ca,
            step=step_eff,
            span=span_eff,
            p=float(v.carrier_p),
            beta=float(v.carrier_beta),
            score_lambda=float(v.score_lambda),
            direction_set=direction_set_active,
            sources=(0, -1),
            residue_masses=z_list,
            left_anchor_infinite=False,
            linear_momentum_state=linear_momentum_state,
            barrier_budget_state=barrier_budget_state,
            inertial_dt=float(v.inertial_dt),
            linear_damping=float(v.linear_damping),
            linear_gain=float(v.linear_gain),
            damping_mode=str(v.damping_mode),
            barrier_decay=float(v.barrier_decay),
            barrier_build=float(v.barrier_build),
            barrier_relief=float(v.barrier_relief),
            barrier_drive_gain=float(v.barrier_drive_gain),
            barrier_floor=float(v.barrier_floor),
            barrier_trigger_offset=float(v.barrier_trigger_offset),
            barrier_kick_gain=float(v.barrier_kick_gain),
            directional_budget_base=float(v.directional_budget_base) * phase_scale,
            directional_budget_bend_coeff=float(v.directional_budget_bend_coeff) * phase_scale,
            directional_budget_torsion_coeff=float(v.directional_budget_torsion_coeff) * phase_scale,
            directional_budget_conjugation_coeff=float(v.directional_budget_conjugation_coeff)
            * phase_scale,
            directional_budget_bend_state=float(bend_state),
            directional_budget_torsion_state=float(tors_state),
            directional_budget_conjugation_state=float(conj_state),
            wave_leak_floor=float(v.wave_leak_floor),
            resonance_state=rs,
            resonance_decay=float(v.resonance_decay),
            resonance_gain=float(v.resonance_gain),
            resonance_harmonic_weight=float(v.resonance_harmonic_weight),
            resonance_damping_span=float(v.resonance_damping_span),
            resonance_gain_boost=float(v.resonance_gain_boost),
            omega_state=omega_state,
            angular_mix=float(v.angular_mix),
            angular_damping=float(v.angular_damping),
            angular_gain=float(v.angular_gain),
        )
        disp = np.asarray(sel["best_disp"], dtype=float)
        max_disp_raw = float(np.max(np.linalg.norm(disp, axis=1)))
        linear_momentum_state = np.asarray(sel["best_linear_momentum_state"], dtype=float).reshape(n, 3)
        barrier_budget_state = np.asarray(sel["best_barrier_budget_state"], dtype=float).reshape(n)
        omega_state = np.asarray(sel["best_omega_state"], dtype=float).reshape(3)
        if v.damping_mode.strip().lower() == "resonant":
            resonance_state = float(sel.get("best_resonance_state", 0.0))
        else:
            resonance_state = 0.0

        n_disp = float(np.linalg.norm(disp)) + 1e-14
        n_delta = float(np.linalg.norm(delta_grad)) + 1e-14
        disp_scaled = disp * (n_delta / n_disp)

        update_rule = str(v.update_rule).strip().lower()
        if update_rule == "natural_hqiv":
            # Faithful to Lean ProteinNaturalFolding:
            # naturalDisp = etaModePhi(m) * localDescentDisp + carrierDisp
            eta = float(v.eta_mode_phi)
            nat_disp = eta * delta_grad + disp
            max_disp_scaled = float(np.max(np.linalg.norm(nat_disp, axis=1)))
            ca = ca + nat_disp
        else:
            max_disp_scaled = float(np.max(np.linalg.norm(disp_scaled, axis=1)))
            ca = ca + (1.0 - float(v.mix_alpha)) * delta_grad + float(v.mix_alpha) * disp_scaled
        ca = _project_bonds(ca, r_min=2.5, r_max=6.0)
        total_translation_steps = int(step_idx + 1)

        n_horizon_cross = 0
        n_horizon_leave = 0
        did_refresh = False
        if v.use_em_field_refresh:
            direction_set_active, n_horizon_cross, n_horizon_leave, did_refresh = (
                maybe_refresh_em_field_direction_set(
                    ca_before,
                    ca,
                    g,
                    base_axes,
                    direction_set_active,
                    disp,
                    refresh_on_horizon_crossing=bool(v.em_refresh_on_horizon_crossing),
                    refresh_on_horizon_leaving=bool(v.em_refresh_on_horizon_leaving),
                    horizon_ang=float(v.em_refresh_horizon_ang),
                    min_seq_sep=int(v.em_refresh_min_seq_sep),
                    refresh_on_large_disp=bool(v.em_refresh_on_large_disp),
                    large_disp_thresh=float(v.em_refresh_large_disp_thresh),
                    max_extra_vectors=int(v.em_max_extra_directions),
                )
            )

        agg_enter += int(n_horizon_cross)
        agg_leave += int(n_horizon_leave)
        if did_refresh:
            agg_refresh += 1

        gap = gap_vs_native(ca, native_pair)
        if gap < best_gap:
            best_gap = gap
            worse_count = 0
        else:
            worse_count += 1

        pair_d = {k: pair_dist(ca, i, j) for k, i, j in NATIVE_PAIRS_1BASE}
        ev_thr = float(v.event_disp_thresh)
        step_events_added = 0
        for k, d_now in pair_d.items():
            drop = float(prev_pair[k] - d_now)
            if max_disp_scaled >= ev_thr and drop >= float(realistic_drop):
                events.append(
                    {
                        "step": int(step_idx + 1),
                        "pair": k,
                        "drop": drop,
                        "max_disp": max_disp_scaled,
                    }
                )
                step_events_added += 1
        prev_pair = pair_d

        _push_top_steps_by_disp(
            top_steps,
            {
                "step": step_n,
                "max_disp_scaled": max_disp_scaled,
                "gap_score": float(gap),
                "phase_scale": float(phase_scale),
                "horizon_pairs_entering": int(n_horizon_cross),
                "horizon_pairs_leaving": int(n_horizon_leave),
                "em_directions_refreshed": bool(did_refresh),
            },
            k=12,
        )

        rmsd_val: Optional[float] = None
        if rmsd_every > 0 and (step_n % rmsd_every == 0 or step_n == step_limit):
            rmsd_val = float(ca_rmsd_from_state(ca, CRAMBIN_SEQ))
        rec = {
            "step": step_n,
            "gap_score": float(gap),
            "best_gap_score": float(best_gap),
            "ca_rmsd_ang": rmsd_val,
            "pair_dist_ang": pair_d,
            "max_disp": max_disp_scaled,
            "max_disp_raw": max_disp_raw,
            "grad_norm": float(g_norm),
            "phase_scale": float(phase_scale),
            "carrier_step_eff": float(step_eff),
            "carrier_span_eff": float(span_eff),
            "directional_budget_bend_state": float(bend_state),
            "directional_budget_torsion_state": float(tors_state),
            "directional_budget_conjugation_state": float(conj_state),
            "em_directions_refreshed": bool(did_refresh),
            "horizon_pairs_entering": int(n_horizon_cross),
            "horizon_pairs_leaving": int(n_horizon_leave),
            "n_directions": int(direction_set_active.shape[0]),
            "barrier_mean": float(np.mean(barrier_budget_state)),
            "barrier_max": float(np.max(barrier_budget_state)),
            "resonance_state": float(resonance_state or 0.0),
            "resonance_measure": float(sel.get("best_resonance_measure", 0.0)),
            "best_score": float(sel["best_score"]),
        }
        last_record = rec
        re = int(rl.record_every)
        if re <= 1 or (step_n % re == 0):
            records.append(rec)

        if rl.until_still:
            if max_disp_scaled < float(rl.motion_floor):
                still_counter += 1
                if still_counter >= int(rl.still_patience):
                    stop_reason = "motion_stalled"
                    break
            else:
                still_counter = 0

        if (not rl.disable_gap_early_stop) and worse_count >= int(v.bad_patience) and gap > (base_gap * 1.03):
            stop_reason = "unexperimental_gap_worsening"
            break

        wrote_pdb = False
        if pdb_out_dir is not None and pdb_every > 0:
            if (step_n % int(pdb_every) == 0) or (step_n == step_limit):
                os.makedirs(pdb_out_dir, exist_ok=True)
                pdb_path = os.path.join(pdb_out_dir, f"{v.name}_step_{step_n:04d}.pdb")
                with open(pdb_path, "w", encoding="utf-8") as f:
                    f.write(ca_to_pdb_str(ca, CRAMBIN_SEQ))
                wrote_pdb = True
        if pdb_out_dir is not None and pdb_on_event and step_events_added > 0:
            os.makedirs(pdb_out_dir, exist_ok=True)
            pdb_path = os.path.join(pdb_out_dir, f"{v.name}_event_step_{step_n:04d}.pdb")
            with open(pdb_path, "w", encoding="utf-8") as f:
                f.write(ca_to_pdb_str(ca, CRAMBIN_SEQ))
            wrote_pdb = True

    if last_record is not None:
        if not records or int(records[-1]["step"]) != int(last_record["step"]):
            records.append(last_record)

    if total_translation_steps == 0:
        total_translation_steps = int(step_limit)
    if stop_reason == "max_steps" and total_translation_steps >= step_limit:
        stop_reason = "safety_max_steps" if rl.until_still else "max_steps"

    final_rmsd = float(ca_rmsd_from_state(ca, CRAMBIN_SEQ))
    final_gap = float(last_record["gap_score"]) if last_record is not None else float(base_gap)

    barrier_summary: Dict[str, Any] = {
        "top_steps_by_max_disp_scaled": list(top_steps),
        "sum_horizon_pairs_entering": int(agg_enter),
        "sum_horizon_pairs_leaving": int(agg_leave),
        "n_steps_with_em_refresh": int(agg_refresh),
        "record_every": int(rl.record_every),
        "n_translation_steps": int(total_translation_steps),
    }

    fold_analysis: Optional[Dict[str, Any]] = None
    if not skip_fold_analysis:
        fold_analysis = analyze_final_fold(
            ca,
            CRAMBIN_SEQ,
            CRAMBIN_GOLD,
            np.asarray(ref_ca, dtype=float),
            native_pair,
        )

    stepwise_diagnostics = compute_stepwise_diagnostics(
        records,
        initial_gap=float(base_gap),
        stop_reason=str(stop_reason),
        native_pair=native_pair,
    )

    return {
        "variant": v.name,
        "config": v.to_serializable(),
        "run_limits": {
            "until_still": bool(rl.until_still),
            "safety_max_steps": int(rl.safety_max_steps),
            "motion_floor": float(rl.motion_floor),
            "still_patience": int(rl.still_patience),
            "grad_tol": float(rl.grad_tol),
            "record_every": int(rl.record_every),
            "disable_gap_early_stop": bool(rl.disable_gap_early_stop),
        },
        "stop_reason": stop_reason,
        "n_steps": int(total_translation_steps),
        "n_translation_steps": int(total_translation_steps),
        "n_records_written": len(records),
        "initial_gap_score": float(base_gap),
        "final_gap_score": final_gap,
        "final_rmsd_ang": final_rmsd,
        "final_pair_dist_ang": {k: pair_dist(ca, i, j) for k, i, j in NATIVE_PAIRS_1BASE},
        "events": events,
        "n_large_pair_events": len(events),
        "barrier_summary": barrier_summary,
        "fold_analysis": fold_analysis,
        "stepwise_diagnostics": stepwise_diagnostics,
        "records": records,
    }


def build_snapshot(quick: bool) -> Tuple[np.ndarray, np.ndarray, float]:
    t0 = time.perf_counter()
    if quick:
        snap = minimize_full_chain(
            CRAMBIN_SEQ,
            quick=True,
            simulate_ribosome_tunnel=True,
            post_extrusion_refine=False,
            fast_pass_steps_per_connection=2,
            min_pass_iter_per_connection=3,
            fast_local_theta=True,
            horizon_neighbor_cutoff=10.0,
            kappa_dihedral=0.01,
            hbond_weight=0.0,
            ensemble_translation_mix_alpha=0.03,
            tunnel_free_terminus_steps=6,
            tunnel_free_terminus_window=12,
            tunnel_handedness_bias_weight=0.03,
            tunnel_handedness_target=0.45,
            tunnel_handedness_sign=1.0,
        )
    else:
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
    return ca, z_list, float(time.perf_counter() - t0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Side-by-side WHIP stepwise runs vs 1CRN.")
    ap.add_argument(
        "--out",
        default=os.path.join(REPO, ".casp_grade_outputs", "iter_small", "crambin_stepwise_side_by_side.json"),
    )
    ap.add_argument("--max-steps", type=int, default=1000, help="Translation iterations per variant (unless --until-still).")
    ap.add_argument("--realistic-drop", type=float, default=0.25, help="Min pair-distance drop (Å) to log an event.")
    ap.add_argument(
        "--event-disp-thresh",
        type=float,
        default=0.45,
        help="Displacement magnitude threshold for counting a 'large translation' event.",
    )
    ap.add_argument("--quick-snapshot", action="store_true", help="Faster tunnel snapshot for smoke tests.")
    ap.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variant names, or 'all' for the built-in panel.",
    )
    ap.add_argument(
        "--rmsd-every",
        type=int,
        default=20,
        help="Compute expensive Cα RMSD vs 1CRN every N steps (0 = only at end).",
    )
    ap.add_argument(
        "--write-pdb-every",
        type=int,
        default=0,
        help="If >0, write full PDB snapshots for each variant every N steps (debug only).",
    )
    ap.add_argument(
        "--pdb-out-dir",
        default=None,
        help="Output directory for per-step PDB debug snapshots. Defaults near --out if --write-pdb-every > 0.",
    )
    ap.add_argument(
        "--pdb-on-event",
        action="store_true",
        help="Also write a PDB snapshot at steps where a logged 'large translation' event triggers.",
    )
    ap.add_argument(
        "--until-still",
        action="store_true",
        help="Run until scaled carrier motion stays below --motion-floor for --still-patience steps "
        "(cap: --safety-max-steps). Implies --no-gap-early-stop unless you override.",
    )
    ap.add_argument(
        "--safety-max-steps",
        type=int,
        default=50_000,
        help="Hard cap on WHIP translation iterations (also used as the only limit when --until-still).",
    )
    ap.add_argument(
        "--motion-floor",
        type=float,
        default=1e-4,
        help="With --until-still: max_disp_scaled below this counts as 'no motion'.",
    )
    ap.add_argument(
        "--still-patience",
        type=int,
        default=150,
        help="Consecutive low-motion steps required to stop when --until-still.",
    )
    ap.add_argument(
        "--grad-tol",
        type=float,
        default=1e-7,
        help="Stop if ||grad|| falls below this before taking a step.",
    )
    ap.add_argument(
        "--record-every",
        type=int,
        default=1,
        help="Only append a JSON timeline row every N translation steps (keeps files small for long runs).",
    )
    ap.add_argument(
        "--no-gap-early-stop",
        action="store_true",
        help="Do not stop on unexperimental_gap_worsening (useful for long exploratory runs).",
    )
    ap.add_argument(
        "--allow-gap-early-stop",
        action="store_true",
        help="With --until-still, still allow unexperimental_gap_worsening early stop (default: disabled when --until-still).",
    )
    ap.add_argument(
        "--skip-fold-analysis",
        action="store_true",
        help="Skip clash / Ramachandran / contact analysis on the final structure.",
    )
    args = ap.parse_args()

    ref_ca, _ = load_ca_from_pdb(CRAMBIN_GOLD)
    native_pair = native_pair_targets(np.asarray(ref_ca, dtype=float))

    ca_snap, z_list, snap_s = build_snapshot(bool(args.quick_snapshot))
    base_gap = gap_vs_native(ca_snap, native_pair)

    panel = default_variants()
    all_panel = all_variants()
    for v in all_panel:
        v.max_steps = int(args.max_steps)
        v.event_disp_thresh = float(args.event_disp_thresh)

    run_limits = RunLimits(
        until_still=bool(args.until_still),
        safety_max_steps=int(args.safety_max_steps),
        motion_floor=float(args.motion_floor),
        still_patience=int(args.still_patience),
        grad_tol=float(args.grad_tol),
        record_every=int(args.record_every),
        disable_gap_early_stop=bool(args.no_gap_early_stop)
        or (bool(args.until_still) and not bool(args.allow_gap_early_stop)),
    )

    if str(args.variants).strip().lower() != "all":
        want = {s.strip() for s in str(args.variants).split(",") if s.strip()}
        panel = [v for v in all_panel if v.name in want]
        if not panel:
            print("No matching variants.", file=sys.stderr)
            return 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pdb_out_dir: Optional[str] = None
    if int(args.write_pdb_every) > 0 or bool(args.pdb_on_event):
        if args.pdb_out_dir:
            pdb_out_dir = str(args.pdb_out_dir)
        else:
            pdb_out_dir = os.path.join(os.path.dirname(args.out), "pdb_step_traces")

    results: List[Dict[str, Any]] = []
    for v in panel:
        t0 = time.perf_counter()
        ca_i = np.asarray(ca_snap, dtype=float).copy()
        r = run_variant(
            ca_i,
            z_list,
            native_pair,
            base_gap,
            v,
            np.asarray(ref_ca, dtype=float),
            realistic_drop=float(args.realistic_drop),
            rmsd_every=int(args.rmsd_every),
            pdb_every=int(args.write_pdb_every),
            pdb_out_dir=pdb_out_dir,
            pdb_on_event=bool(args.pdb_on_event),
            run_limits=run_limits,
            skip_fold_analysis=bool(args.skip_fold_analysis),
        )
        r["wall_seconds"] = float(time.perf_counter() - t0)
        results.append(r)
        ntr = int(r.get("n_translation_steps", r.get("n_steps", 0)))
        nrec = int(r.get("n_records_written", 0))
        print(
            f"{v.name:40s}  trans={ntr:<5} records={nrec:<5} final_RMSD={r['final_rmsd_ang']:.3f}  "
            f"events={r['n_large_pair_events']:<3}  {r['stop_reason']}",
            flush=True,
        )

    ranked_rmsd = sorted(results, key=lambda x: float(x["final_rmsd_ang"]))
    ranked_events = sorted(results, key=lambda x: (-int(x["n_large_pair_events"]), float(x["final_rmsd_ang"])))

    out_obj = {
        "gold_pdb": CRAMBIN_GOLD,
        "snapshot_build_seconds": snap_s,
        "native_pair_targets_ang": native_pair,
        "initial_gap_score": float(base_gap),
        "realistic_drop_threshold": float(args.realistic_drop),
        "run_limits": {
            "until_still": bool(args.until_still),
            "safety_max_steps": int(args.safety_max_steps),
            "motion_floor": float(args.motion_floor),
            "still_patience": int(args.still_patience),
            "grad_tol": float(args.grad_tol),
            "record_every": int(args.record_every),
            "disable_gap_early_stop": bool(run_limits.disable_gap_early_stop),
            "max_steps_variant_budget": int(args.max_steps),
        },
        "results": results,
        "ranking_by_final_rmsd": [r["variant"] for r in ranked_rmsd],
        "ranking_by_event_count_then_rmsd": [r["variant"] for r in ranked_events],
        "notes": [
            "Lower final_rmsd_ang is better for overall fold match to 1CRN.",
            "EM direction refresh (when enabled): matches minimize_full_chain — rebuild when a nonlocal Cα pair enters the 15 Å horizon (coupling turns on), and optionally when a pair leaves (coupling drops). See ensemble_em_refresh_on_horizon_leaving.",
            "events: step where max_disp_scaled exceeds event_disp_thresh AND a native pair distance drops by >= realistic_drop.",
            "Tune pruning: variants that spike events without improving gap_score / RMSD are suspect.",
            "n_translation_steps counts every WHIP iteration (one translation + bond projection); n_records_written may be smaller if --record-every > 1.",
            "fold_analysis: Cα clashes, Ramachandran vs 1CRN, missed native contacts (sub-7.5 Å in native, opened in pred), spurious tight long-range pairs, tuning_hints.",
            "With --until-still, the loop stops when max_disp_scaled < motion_floor for still_patience consecutive steps, or grad_norm < grad_tol, or safety_max_steps.",
            "stepwise_diagnostics: when you hit max_steps / safety_max_steps, use best_gap_step, first_rebound_step_after_best, tracked_pairs_closest_to_native, and narrative — then inspect records (or PDB) around those steps to see where the trajectory left the path toward gold.",
        ],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)
    print("WROTE", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
