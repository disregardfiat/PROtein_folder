#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from horizon_physics.proteins.full_protein_minimizer import full_chain_to_pdb, minimize_full_chain
from horizon_physics.proteins.grade_folds import ca_rmsd, load_ca_from_pdb
from horizon_physics.proteins.osh_oracle_folding import minimize_ca_with_osh_oracle

CRAMBIN_SEQ = "TTCCPSIVARSNFNVCRLPGTPEAIICGDVCDLDCTAKTCFSIICT"
CRAMBIN_GOLD = os.path.join(REPO, "proteins", "1CRN.pdb")


def _ca_to_pdb(ca: np.ndarray, seq: str, out_path: str) -> str:
    from horizon_physics.proteins.full_protein_minimizer import _place_full_backbone

    bb = _place_full_backbone(np.asarray(ca, dtype=float), seq)
    obj = {"ca_min": ca, "backbone_atoms": bb, "sequence": seq, "n_res": len(seq)}
    pdb = full_chain_to_pdb(obj)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(pdb)
    return out_path


def _score(pred_pdb_path: str, ref_pdb_path: str) -> float:
    rmsd, _, _, _ = ca_rmsd(pred_pdb_path, ref_pdb_path, align_by_resid=False, trim_to_min_length=True)
    return float(rmsd)


def _pair_dist(ca: np.ndarray, i1: int, j1: int) -> float:
    return float(np.linalg.norm(ca[i1 - 1] - ca[j1 - 1]))


def _native_pair_gap(ca: np.ndarray, ref_ca: np.ndarray) -> float:
    tracked = [(1, 32), (3, 40), (8, 34), (16, 26)]
    gaps = []
    for i, j in tracked:
        tgt = _pair_dist(ref_ca, i, j)
        cur = _pair_dist(ca, i, j)
        gaps.append(max(0.0, cur - tgt))
    return float(np.mean(gaps))


def run_side_by_side(
    sequence: str,
    *,
    quick: bool,
    oracle_iters: int,
    oracle_step: float,
    oracle_gate_mix: float,
    oracle_quantile: float,
    oracle_use_harmonic_metropolis: bool,
    oracle_random_seed: Optional[int],
    oracle_stop_when_settled: bool,
    oracle_settle_window: int,
    oracle_settle_energy_tol: float,
    oracle_settle_step_tol: float,
    oracle_settle_min_iter: int,
    oracle_use_local_rapidity: bool,
    oracle_rapidity_gain: float,
    oracle_rapidity_tangent_weight: float,
    oracle_rapidity_normal_weight: float,
    oracle_inertial_pk_weight: float,
    oracle_inertial_k_potential: float,
    oracle_inertial_k_kinetic: float,
    oracle_inertial_velocity_decay: float,
    oracle_use_energy_reservoir: bool,
    oracle_reservoir_init: float,
    oracle_reservoir_gain_scale: float,
    oracle_use_contact_reflectors: bool,
    oracle_contact_min_seq_sep: int,
    oracle_contact_cutoff_ang: float,
    oracle_contact_max_reflectors: int,
    oracle_contact_grad_coupling: float,
    oracle_contact_weight_gradient: bool,
    oracle_contact_score_mode: str,
    oracle_contact_inverse_power: float,
    oracle_contact_score_min_dist_ang: float,
    oracle_use_resonance_multiplier: bool,
    oracle_resonance_terminus_boost: float,
    oracle_resonance_core_damping: float,
    oracle_resonance_transition_width: int,
    oracle_resonance_compaction_cutoff_ang: float,
    oracle_resonance_compaction_min_seq_sep: int,
    oracle_tunnel_budget_distance_score_mode: str,
    oracle_tunnel_budget_inverse_power: float,
    oracle_tunnel_budget_distance_d0_ang: float,
    oracle_use_end_bias_budget: bool,
    oracle_end_bias_scale: float,
    oracle_end_bias_floor: float,
    oracle_use_mode_shape_participation: bool,
    oracle_mode_shape_fixed_end: str,
    oracle_mode_shape_factor_min: float,
    oracle_mode_shape_factor_max: float,
    oracle_omega_refresh_period: int,
    oracle_use_terminus_gradient_boost: bool,
    oracle_terminus_gradient_boost: float,
    oracle_terminus_gradient_transition_width: int,
    oracle_terminus_gradient_core_scale: float,
    oracle_contact_terminus_window: int,
    oracle_contact_terminus_score_scale: float,
    out_dir: str,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    ref_ca, _ = load_ca_from_pdb(CRAMBIN_GOLD)
    seq = "".join(c for c in sequence.strip().upper() if c.isalpha())
    if not seq:
        raise ValueError("Empty sequence.")

    # Path A: tunnel assembly + extrusion only (no post-extrusion refine).
    t0 = time.perf_counter()
    tunnel = minimize_full_chain(
        seq,
        quick=bool(quick),
        simulate_ribosome_tunnel=True,
        post_extrusion_refine=False,
        fast_pass_steps_per_connection=2,
        min_pass_iter_per_connection=5,
        fast_local_theta=True,
        horizon_neighbor_cutoff=10.0,
        kappa_dihedral=0.01,
        hbond_weight=0.0,
    )
    tunnel_s = float(time.perf_counter() - t0)
    ca_tunnel = np.asarray(tunnel["ca_min"], dtype=float)

    # Path B: same tunnel snapshot + Quantum/OSHoracle sparse minimizer.
    t1 = time.perf_counter()
    ca_oracle, oracle_info = minimize_ca_with_osh_oracle(
        ca_tunnel.copy(),
        n_iter=int(oracle_iters),
        step_size=float(oracle_step),
        gate_mix=float(oracle_gate_mix),
        amp_threshold_quantile=float(oracle_quantile),
        use_harmonic_metropolis=bool(oracle_use_harmonic_metropolis),
        random_seed=oracle_random_seed,
        stop_when_settled=bool(oracle_stop_when_settled),
        settle_window=int(oracle_settle_window),
        settle_energy_tol=float(oracle_settle_energy_tol),
        settle_step_tol=float(oracle_settle_step_tol),
        settle_min_iter=int(oracle_settle_min_iter),
        use_local_rapidity_translation=bool(oracle_use_local_rapidity),
        rapidity_gain=float(oracle_rapidity_gain),
        rapidity_tangent_weight=float(oracle_rapidity_tangent_weight),
        rapidity_normal_weight=float(oracle_rapidity_normal_weight),
        inertial_pk_weight=float(oracle_inertial_pk_weight),
        inertial_k_potential=float(oracle_inertial_k_potential),
        inertial_k_kinetic=float(oracle_inertial_k_kinetic),
        inertial_velocity_decay=float(oracle_inertial_velocity_decay),
        use_energy_reservoir=bool(oracle_use_energy_reservoir),
        reservoir_init=float(oracle_reservoir_init),
        reservoir_gain_scale=float(oracle_reservoir_gain_scale),
        use_contact_reflectors=bool(oracle_use_contact_reflectors),
        contact_min_seq_sep=int(oracle_contact_min_seq_sep),
        contact_cutoff_ang=float(oracle_contact_cutoff_ang),
        contact_max_reflectors=int(oracle_contact_max_reflectors),
        contact_grad_coupling=float(oracle_contact_grad_coupling),
        contact_weight_gradient=bool(oracle_contact_weight_gradient),
        contact_score_mode=str(oracle_contact_score_mode),
        contact_inverse_power=float(oracle_contact_inverse_power),
        contact_score_min_dist_ang=float(oracle_contact_score_min_dist_ang),
        use_resonance_multiplier=bool(oracle_use_resonance_multiplier),
        resonance_terminus_boost=float(oracle_resonance_terminus_boost),
        resonance_core_damping=float(oracle_resonance_core_damping),
        resonance_transition_width=int(oracle_resonance_transition_width),
        resonance_compaction_cutoff_ang=float(oracle_resonance_compaction_cutoff_ang),
        resonance_compaction_min_seq_sep=int(oracle_resonance_compaction_min_seq_sep),
        tunnel_budget_distance_score_mode=str(oracle_tunnel_budget_distance_score_mode),
        tunnel_budget_inverse_power=float(oracle_tunnel_budget_inverse_power),
        tunnel_budget_distance_d0_ang=float(oracle_tunnel_budget_distance_d0_ang),
        use_end_bias_budget=bool(oracle_use_end_bias_budget),
        end_bias_scale=float(oracle_end_bias_scale),
        end_bias_floor=float(oracle_end_bias_floor),
        use_mode_shape_participation=bool(oracle_use_mode_shape_participation),
        mode_shape_fixed_end=str(oracle_mode_shape_fixed_end),
        mode_shape_factor_min=float(oracle_mode_shape_factor_min),
        mode_shape_factor_max=float(oracle_mode_shape_factor_max),
        omega_refresh_period=int(oracle_omega_refresh_period),
        use_terminus_gradient_boost=bool(oracle_use_terminus_gradient_boost),
        terminus_gradient_boost=float(oracle_terminus_gradient_boost),
        terminus_gradient_transition_width=int(oracle_terminus_gradient_transition_width),
        terminus_gradient_core_scale=float(oracle_terminus_gradient_core_scale),
        contact_terminus_window=int(oracle_contact_terminus_window),
        contact_terminus_score_scale=float(oracle_contact_terminus_score_scale),
    )
    oracle_s = float(time.perf_counter() - t1)

    tunnel_pdb = os.path.join(out_dir, "tunnel_extrusion_only.pdb")
    oracle_pdb = os.path.join(out_dir, "quantum_osh_oracle.pdb")
    _ca_to_pdb(ca_tunnel, seq, tunnel_pdb)
    _ca_to_pdb(ca_oracle, seq, oracle_pdb)

    tunnel_rmsd = _score(tunnel_pdb, CRAMBIN_GOLD)
    oracle_rmsd = _score(oracle_pdb, CRAMBIN_GOLD)

    gap_tunnel = _native_pair_gap(ca_tunnel, np.asarray(ref_ca, dtype=float))
    gap_oracle = _native_pair_gap(ca_oracle, np.asarray(ref_ca, dtype=float))

    return {
        "sequence_length": len(seq),
        "reference_pdb": CRAMBIN_GOLD,
        "paths": {
            "tunnel_extrusion": {
                "description": "Ribosome tunnel assembly/extrusion only (no post-extrusion refine)",
                "wall_seconds": tunnel_s,
                "ca_rmsd_ang": tunnel_rmsd,
                "native_pair_gap_ang": gap_tunnel,
                "pdb_path": tunnel_pdb,
            },
            "quantum_osh_oracle": {
                "description": "Same tunnel snapshot + OSHoracle sparse support minimizer",
                "wall_seconds": oracle_s,
                "ca_rmsd_ang": oracle_rmsd,
                "native_pair_gap_ang": gap_oracle,
                "accepted_steps": int(oracle_info.accepted_steps),
                "iterations": int(oracle_info.iterations),
                "iterations_executed": int(oracle_info.iterations_executed),
                "last_flipped_count": int(oracle_info.last_flipped_count),
                "last_step_size": float(oracle_info.last_step_size),
                "natural_harmonic_scale": float(oracle_info.natural_harmonic_scale),
                "metropolis_accepts": int(oracle_info.metropolis_accepts),
                "stop_reason": str(oracle_info.stop_reason),
                "settled": bool(oracle_info.settled),
                "use_local_rapidity_translation": bool(oracle_use_local_rapidity),
                "inertial_energy_final_ev": float(oracle_info.inertial_energy_final_ev),
                "inertial_pk_weight": float(oracle_inertial_pk_weight),
                "reservoir_energy_final_ev": float(oracle_info.reservoir_energy_final_ev),
                "reservoir_uphill_accepts": int(oracle_info.reservoir_uphill_accepts),
                "tunnel_harmonic_budget_final_ev": float(oracle_info.tunnel_harmonic_budget_final_ev),
                "contact_reflector_count": int(oracle_info.contact_reflector_count),
                "omega_refresh_count": int(oracle_info.omega_refresh_count),
                "use_terminus_gradient_boost": bool(oracle_use_terminus_gradient_boost),
                "terminus_gradient_boost": float(oracle_terminus_gradient_boost),
                "terminus_gradient_transition_width": int(oracle_terminus_gradient_transition_width),
                "contact_terminus_window": int(oracle_contact_terminus_window),
                "contact_terminus_score_scale": float(oracle_contact_terminus_score_scale),
                "pdb_path": oracle_pdb,
            },
        },
        "delta": {
            "rmsd_improvement_ang": float(tunnel_rmsd - oracle_rmsd),
            "native_pair_gap_improvement_ang": float(gap_tunnel - gap_oracle),
        },
    }


def main() -> int:
    preset_ap = argparse.ArgumentParser(add_help=False)
    preset_ap.add_argument(
        "--oracle-preset",
        choices=("none", "crambin_tunnel_refine"),
        default="none",
        help=argparse.SUPPRESS,
    )
    preset_ns, argv_rest = preset_ap.parse_known_args(sys.argv[1:])

    ap = argparse.ArgumentParser(
        description=(
            "Side-by-side: tunnel assembly/extrusion vs Quantum/OSHoracle minimizer.\n\n"
            "Oracle preset (recognized anywhere on the command line): "
            "--oracle-preset {none,crambin_tunnel_refine}."
        ),
        epilog=(
            "crambin_tunnel_refine sets defaults: oracle-step 0.0218, oracle-quantile 0.525 "
            "(0.55 leaves a different attractor with this step/gain), energy reservoir on "
            "(init 55, gain 1.235), settle window 70 / min-iter 220 / tight tolerances, "
            "and mild contact-reflector preference for terminal pairs (window 7, scale 1.05) "
            "when --oracle-use-contact-reflectors is on. "
            "Stronger terminus step scaling: --oracle-use-terminus-gradient-boost (can change the attractor). "
            "Override any preset defaults with explicit flags."
        ),
    )
    ap.add_argument("--sequence", default=CRAMBIN_SEQ, help="Protein sequence (1-letter AA).")
    ap.add_argument(
        "--out-json",
        default=os.path.join(REPO, ".casp_grade_outputs", "iter_small", "quantum_osh_side_by_side.json"),
        help="Output JSON summary path.",
    )
    ap.add_argument(
        "--out-dir",
        default=os.path.join(REPO, ".casp_grade_outputs", "iter_small", "quantum_osh_side_by_side"),
        help="Output directory for paired PDBs.",
    )
    ap.add_argument("--quick", action="store_true", help="Use quick tunnel settings.")
    ap.add_argument("--oracle-iters", type=int, default=140, help="OSHoracle minimizer iterations.")
    ap.add_argument("--oracle-step", type=float, default=0.02, help="OSHoracle base step size.")
    ap.add_argument("--oracle-gate-mix", type=float, default=0.55, help="OSHoracle gate mixing factor [0,1].")
    ap.add_argument(
        "--oracle-quantile",
        type=float,
        default=0.55,
        help="Gradient magnitude quantile for sparse support threshold [0,1].",
    )
    ap.add_argument(
        "--oracle-harmonic-metropolis",
        action="store_true",
        help="Enable harmonic-scale Metropolis acceptance for uphill proposals.",
    )
    ap.add_argument("--oracle-seed", type=int, default=123, help="Random seed for Metropolis acceptance.")
    ap.add_argument(
        "--oracle-stop-when-settled",
        action="store_true",
        help="Stop OSHoracle early when energy/step size settles.",
    )
    ap.add_argument("--oracle-settle-window", type=int, default=20, help="Window size for settle detection.")
    ap.add_argument(
        "--oracle-settle-energy-tol",
        type=float,
        default=1e-3,
        help="Energy span tolerance across settle window.",
    )
    ap.add_argument(
        "--oracle-settle-step-tol",
        type=float,
        default=3e-4,
        help="Mean effective step tolerance across settle window.",
    )
    ap.add_argument(
        "--oracle-settle-min-iter",
        type=int,
        default=30,
        help="Minimum iterations before settle checks are active.",
    )
    ap.add_argument(
        "--oracle-use-local-rapidity",
        action="store_true",
        help="Inject local-frame rapidity-like Cartesian translation into OSHoracle updates.",
    )
    ap.add_argument("--oracle-rapidity-gain", type=float, default=0.25, help="Rapidity translation gain.")
    ap.add_argument(
        "--oracle-rapidity-tangent-weight",
        type=float,
        default=0.7,
        help="Rapidity tangent component weight.",
    )
    ap.add_argument(
        "--oracle-rapidity-normal-weight",
        type=float,
        default=0.3,
        help="Rapidity normal component weight.",
    )
    ap.add_argument(
        "--oracle-inertial-pk-weight",
        type=float,
        default=0.0,
        help="Weight on per-atom inertial P/K energy budget.",
    )
    ap.add_argument(
        "--oracle-inertial-k-potential",
        type=float,
        default=1.0,
        help="Potential (P) coefficient for inertial budget.",
    )
    ap.add_argument(
        "--oracle-inertial-k-kinetic",
        type=float,
        default=1.0,
        help="Kinetic (K) coefficient for inertial budget.",
    )
    ap.add_argument(
        "--oracle-inertial-velocity-decay",
        type=float,
        default=0.9,
        help="Velocity decay for inertial state update.",
    )
    ap.add_argument(
        "--oracle-use-energy-reservoir",
        action="store_true",
        help="Enable non-decaying downhill-energy reservoir for uphill acceptance budget.",
    )
    ap.add_argument(
        "--oracle-reservoir-init",
        type=float,
        default=0.0,
        help="Initial reservoir energy budget.",
    )
    ap.add_argument(
        "--oracle-reservoir-gain-scale",
        type=float,
        default=1.0,
        help="Scale factor for converting downhill drop into reservoir budget.",
    )
    ap.add_argument(
        "--oracle-use-contact-reflectors",
        action="store_true",
        help="Add virtual tunnel-budget reflectors at nonlocal Cα contacts (optional gradient weighting).",
    )
    ap.add_argument(
        "--oracle-contact-min-seq-sep",
        type=int,
        default=4,
        help="Minimum |i−j| along sequence for a contact reflector pair.",
    )
    ap.add_argument(
        "--oracle-contact-cutoff-ang",
        type=float,
        default=8.0,
        help="Cα–Cα distance below this (Å) counts as contact for reflectors.",
    )
    ap.add_argument(
        "--oracle-contact-max-reflectors",
        type=int,
        default=16,
        help="Cap on distinct residue indices used as contact reflectors.",
    )
    ap.add_argument(
        "--oracle-contact-grad-coupling",
        type=float,
        default=1.0,
        help="Boost contact score by (normalized ||∇E||) product when gradient weighting is on.",
    )
    ap.add_argument(
        "--oracle-contact-score-mode",
        type=str,
        default="hard_linear",
        choices=("hard_linear", "inverse_square"),
        help="How to score proximity for contact-based virtual reflectors.",
    )
    ap.add_argument(
        "--oracle-contact-inverse-power",
        type=float,
        default=2.0,
        help="Power p in inverse-power contact score (1/d^p growth as d decreases).",
    )
    ap.add_argument(
        "--oracle-contact-score-min-dist-ang",
        type=float,
        default=1.0,
        help="Minimum distance (Å) for inverse-power scoring to avoid blowups.",
    )
    ap.add_argument(
        "--oracle-contact-no-gradient-weight",
        action="store_true",
        help="Rank contacts by geometry only (ignore gradient magnitudes).",
    )
    ap.add_argument(
        "--oracle-use-terminus-gradient-boost",
        action="store_true",
        help="Scale gradient (and rapidity) steps up at chain ends vs core (targets terminal RMSD).",
    )
    ap.add_argument(
        "--oracle-disable-terminus-gradient-boost",
        action="store_true",
        help="Turn off terminus gradient boost even if a preset enabled it.",
    )
    ap.add_argument(
        "--oracle-terminus-gradient-boost",
        type=float,
        default=1.28,
        help="Step multiplier at each terminus when terminus gradient boost is on.",
    )
    ap.add_argument(
        "--oracle-terminus-gradient-transition-width",
        type=int,
        default=8,
        help="Residues from each end over which terminus step scale ramps to core.",
    )
    ap.add_argument(
        "--oracle-terminus-gradient-core-scale",
        type=float,
        default=1.0,
        help="Step scale in the chain interior (usually 1.0).",
    )
    ap.add_argument(
        "--oracle-contact-terminus-window",
        type=int,
        default=0,
        help="If >0, multiply contact reflector pair scores by oracle-contact-terminus-score-scale "
        "when either residue lies within this many residues of an end.",
    )
    ap.add_argument(
        "--oracle-contact-terminus-score-scale",
        type=float,
        default=1.0,
        help="Score multiplier for terminal-involving contact pairs (only if window > 0).",
    )
    ap.add_argument(
        "--oracle-omega-refresh-period",
        type=int,
        default=0,
        help="Re-estimate natural harmonic scale ω every N iterations (0 = once at start).",
    )
    ap.add_argument(
        "--oracle-tunnel-budget-distance-score-mode",
        type=str,
        default="linear",
        choices=("linear", "inverse_power", "inverse_square"),
        help="Mapping from distance-to-reflector -> tunnel-budget scaling.",
    )
    ap.add_argument(
        "--oracle-tunnel-budget-inverse-power",
        type=float,
        default=2.0,
        help="Power p used for inverse_power tunnel-budget mode (1/(d+d0)^p).",
    )
    ap.add_argument(
        "--oracle-tunnel-budget-distance-d0-ang",
        type=float,
        default=1.0,
        help="Shift d0 (Å) to avoid singularity at d=0 in inverse_power tunnel-budget mode.",
    )
    ap.add_argument(
        "--oracle-use-end-budget-bias",
        action="store_true",
        help="Ends-first bias: multiply tunnel budget by 2*(1-nearest_end/other_end).",
    )
    ap.add_argument(
        "--oracle-end-bias-scale",
        type=float,
        default=2.0,
        help="Scale factor for end-budget bias (default matches 2*(1-ratio)).",
    )
    ap.add_argument(
        "--oracle-end-bias-floor",
        type=float,
        default=0.1,
        help="Minimum damping factor for end-budget bias (keeps step scaling nonzero).",
    )
    ap.add_argument(
        "--oracle-use-mode-shape-participation",
        action="store_true",
        help="Ends-first mode-shape weighting: fixed-free first-mode factor scales harmonic step_eff.",
    )
    ap.add_argument(
        "--oracle-mode-shape-fixed-end",
        type=str,
        default="right",
        choices=("left", "right"),
        help="Which end is treated as fixed (other end is free) for mode-shape weighting.",
    )
    ap.add_argument(
        "--oracle-mode-shape-factor-min",
        type=float,
        default=0.5,
        help="Mode factor at fixed end (step_eff scales down toward this).",
    )
    ap.add_argument(
        "--oracle-mode-shape-factor-max",
        type=float,
        default=1.2,
        help="Mode factor at free end (step_eff scales up toward this).",
    )
    ap.add_argument(
        "--oracle-use-resonance-multiplier",
        action="store_true",
        help="Use compaction-based per-residue resonance multiplier to reshape harmonic step scaling.",
    )
    ap.add_argument(
        "--oracle-resonance-terminus-boost",
        type=float,
        default=1.8,
        help="Resonance boost at ends (higher boosts termini).",
    )
    ap.add_argument(
        "--oracle-resonance-core-damping",
        type=float,
        default=0.4,
        help="Resonance damping in compact core (0..1 typical).",
    )
    ap.add_argument(
        "--oracle-resonance-transition-width",
        type=int,
        default=5,
        help="How many residues from each end are treated as terminus-like.",
    )
    ap.add_argument(
        "--oracle-resonance-compaction-cutoff-ang",
        type=float,
        default=8.0,
        help="Distance cutoff (Å) for local compaction/contact density scoring.",
    )
    ap.add_argument(
        "--oracle-resonance-compaction-min-seq-sep",
        type=int,
        default=4,
        help="Minimum |i-j| sequence separation for compaction scoring.",
    )

    if preset_ns.oracle_preset == "crambin_tunnel_refine":
        ap.set_defaults(
            oracle_quantile=0.525,
            oracle_use_energy_reservoir=True,
            oracle_step=0.0218,
            oracle_reservoir_init=55.0,
            oracle_reservoir_gain_scale=1.235,
            oracle_stop_when_settled=True,
            oracle_settle_window=70,
            oracle_settle_min_iter=220,
            oracle_settle_energy_tol=1e-5,
            oracle_settle_step_tol=1e-5,
            oracle_contact_terminus_window=7,
            oracle_contact_terminus_score_scale=1.05,
        )

    args = ap.parse_args(argv_rest)
    args.oracle_preset = preset_ns.oracle_preset

    out_json = str(args.out_json)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    out = run_side_by_side(
        str(args.sequence),
        quick=bool(args.quick),
        oracle_iters=int(args.oracle_iters),
        oracle_step=float(args.oracle_step),
        oracle_gate_mix=float(args.oracle_gate_mix),
        oracle_quantile=float(args.oracle_quantile),
        oracle_use_harmonic_metropolis=bool(args.oracle_harmonic_metropolis),
        oracle_random_seed=int(args.oracle_seed) if args.oracle_seed is not None else None,
        oracle_stop_when_settled=bool(args.oracle_stop_when_settled),
        oracle_settle_window=int(args.oracle_settle_window),
        oracle_settle_energy_tol=float(args.oracle_settle_energy_tol),
        oracle_settle_step_tol=float(args.oracle_settle_step_tol),
        oracle_settle_min_iter=int(args.oracle_settle_min_iter),
        oracle_use_local_rapidity=bool(args.oracle_use_local_rapidity),
        oracle_rapidity_gain=float(args.oracle_rapidity_gain),
        oracle_rapidity_tangent_weight=float(args.oracle_rapidity_tangent_weight),
        oracle_rapidity_normal_weight=float(args.oracle_rapidity_normal_weight),
        oracle_inertial_pk_weight=float(args.oracle_inertial_pk_weight),
        oracle_inertial_k_potential=float(args.oracle_inertial_k_potential),
        oracle_inertial_k_kinetic=float(args.oracle_inertial_k_kinetic),
        oracle_inertial_velocity_decay=float(args.oracle_inertial_velocity_decay),
        oracle_use_energy_reservoir=bool(args.oracle_use_energy_reservoir),
        oracle_reservoir_init=float(args.oracle_reservoir_init),
        oracle_reservoir_gain_scale=float(args.oracle_reservoir_gain_scale),
        oracle_use_contact_reflectors=bool(args.oracle_use_contact_reflectors),
        oracle_contact_min_seq_sep=int(args.oracle_contact_min_seq_sep),
        oracle_contact_cutoff_ang=float(args.oracle_contact_cutoff_ang),
        oracle_contact_max_reflectors=int(args.oracle_contact_max_reflectors),
        oracle_contact_grad_coupling=float(args.oracle_contact_grad_coupling),
        oracle_contact_weight_gradient=not bool(args.oracle_contact_no_gradient_weight),
        oracle_contact_score_mode=str(args.oracle_contact_score_mode),
        oracle_contact_inverse_power=float(args.oracle_contact_inverse_power),
        oracle_contact_score_min_dist_ang=float(args.oracle_contact_score_min_dist_ang),
        oracle_use_resonance_multiplier=bool(args.oracle_use_resonance_multiplier),
        oracle_resonance_terminus_boost=float(args.oracle_resonance_terminus_boost),
        oracle_resonance_core_damping=float(args.oracle_resonance_core_damping),
        oracle_resonance_transition_width=int(args.oracle_resonance_transition_width),
        oracle_resonance_compaction_cutoff_ang=float(args.oracle_resonance_compaction_cutoff_ang),
        oracle_resonance_compaction_min_seq_sep=int(args.oracle_resonance_compaction_min_seq_sep),
        oracle_tunnel_budget_distance_score_mode=str(args.oracle_tunnel_budget_distance_score_mode),
        oracle_tunnel_budget_inverse_power=float(args.oracle_tunnel_budget_inverse_power),
        oracle_tunnel_budget_distance_d0_ang=float(args.oracle_tunnel_budget_distance_d0_ang),
        oracle_use_end_bias_budget=bool(args.oracle_use_end_budget_bias),
        oracle_end_bias_scale=float(args.oracle_end_bias_scale),
        oracle_end_bias_floor=float(args.oracle_end_bias_floor),
        oracle_use_mode_shape_participation=bool(args.oracle_use_mode_shape_participation),
        oracle_mode_shape_fixed_end=str(args.oracle_mode_shape_fixed_end),
        oracle_mode_shape_factor_min=float(args.oracle_mode_shape_factor_min),
        oracle_mode_shape_factor_max=float(args.oracle_mode_shape_factor_max),
        oracle_omega_refresh_period=int(args.oracle_omega_refresh_period),
        oracle_use_terminus_gradient_boost=bool(args.oracle_use_terminus_gradient_boost)
        and not bool(args.oracle_disable_terminus_gradient_boost),
        oracle_terminus_gradient_boost=float(args.oracle_terminus_gradient_boost),
        oracle_terminus_gradient_transition_width=int(args.oracle_terminus_gradient_transition_width),
        oracle_terminus_gradient_core_scale=float(args.oracle_terminus_gradient_core_scale),
        oracle_contact_terminus_window=int(args.oracle_contact_terminus_window),
        oracle_contact_terminus_score_scale=float(args.oracle_contact_terminus_score_scale),
        out_dir=str(args.out_dir),
    )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    t = out["paths"]["tunnel_extrusion"]
    q = out["paths"]["quantum_osh_oracle"]
    d = out["delta"]
    print("WROTE", out_json)
    print(
        "Tunnel RMSD {:.3f} Å | Quantum/OSH RMSD {:.3f} Å | Δ {:.3f} Å".format(
            float(t["ca_rmsd_ang"]), float(q["ca_rmsd_ang"]), float(d["rmsd_improvement_ang"])
        )
    )
    print(
        "Tunnel pair-gap {:.3f} Å | Quantum/OSH pair-gap {:.3f} Å | Δ {:.3f} Å".format(
            float(t["native_pair_gap_ang"]),
            float(q["native_pair_gap_ang"]),
            float(d["native_pair_gap_improvement_ang"]),
        )
    )
    print("PDBs:", t["pdb_path"], q["pdb_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

