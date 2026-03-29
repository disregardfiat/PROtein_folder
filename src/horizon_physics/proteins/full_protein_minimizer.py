"""
Full-chain E_tot minimizer: backbone + optional side-chain from HQIV.

Minimizes E_tot = Σ ħc/Θ_i + U_φ over the Cα trace (deterministic L-BFGS),
with bond-length constraints: consecutive Cα in [2.5, 6.0] Å. After every
step, project so bonds remain valid; clash penalty for non-bonded < 2.0 Å.
Then reconstruct full backbone. No force fields; all from first principles.

Returns: minimized Cα, full backbone coordinates, E_final, and optional PDB.
MIT License. Python 3.10+. Numpy only.
"""

from __future__ import annotations

import functools
import json
import signal
import sys
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple


def write_trajectory_frame(file_handle: Optional[Any], step: int, positions: np.ndarray) -> None:
    """
    Append one frame to a trajectory log (JSONL). Safe to call with file_handle=None.
    Format: {"t": step, "positions": [[x,y,z], ...]} per line, flush after each write
    so that `tail -f` shows progress and the log can feed Manim or other viewers.
    """
    if file_handle is None:
        return
    positions = np.asarray(positions)
    if positions.ndim == 1:
        positions = positions.reshape(-1, 3)
    file_handle.write(json.dumps({"t": step, "positions": positions.tolist()}) + "\n")
    file_handle.flush()

try:
    from scipy.spatial import cKDTree as _cKDTree
    _HAS_SCIPY_SPATIAL = True
except ImportError:
    _cKDTree = None
    _HAS_SCIPY_SPATIAL = False

from .casp_submission import (
    _parse_fasta,
    _place_backbone_ca,
    _place_full_backbone,
    AA_1to3,
)
from .folding_energy import (
    e_tot,
    e_tot_ca_with_bonds,
    count_variational_pairs_ca,
    variational_pair_lower_bound_term,
    grad_full,
    grad_bonds_only,
    rg_squared,
    grad_rg_squared,
    K_RG_DEFAULT,
    K_RG_COLLAPSE,
)
from .gradient_descent_folding import minimize_e_tot_lbfgs, _project_bonds
from . import co_translational_tunnel as _tunnel
from .co_translational_tunnel import rigid_body_force_torque
from .force_carrier_ensemble import (
    build_direction_set_6_axes,
    choose_best_translation_direction,
    maybe_refresh_em_field_direction_set,
)
from .ligands import LigandAgent, parse_ligands
from .pdb_hqiv_header import hqiv_qconsi_empty_pdb_block, hqiv_qconsi_model_lines
from .protein_qc_refinement import grad_qc_soft_clash_protein_fixed
from .peptide_backbone import backbone_bond_lengths
from .side_chain_placement import chi_angles_for_residue, side_chain_chi_preferences

# Z_shell for E_tot over full backbone: N=7, C=6, O=8
Z_N, Z_CA, Z_C, Z_O = 7, 6, 6, 8


def _minimize_bonds_fast(
    ca_init: np.ndarray,
    z_list: np.ndarray,
    max_iter: int = 30,
    fast_horizon: bool = False,
    collapse: bool = False,
    collapse_frac: float = 0.5,
    loose_r_max: float = 8.0,
    collapse_init_steps: int = 0,
    k_rg_collapse: Optional[float] = None,
    trajectory_log: Optional[Any] = None,
    on_step_callback: Optional[Callable[[int, np.ndarray], None]] = None,
    grad_full_extra_kwargs: Optional[Dict[str, Any]] = None,
    grad_extra_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    fast_local_theta: bool = False,
    variational_bound_prune: bool = False,
    variational_bound_prune_margin: float = 0.0,
    ensemble_translation_mix_alpha: float = 0.0,
    ensemble_translation_step: float = 0.35,
    ensemble_decay_span: float = 0.25,
    ensemble_beta: float = 0.35,
    ensemble_s2_power: float = 1.0,
    ensemble_score_lambda: float = 0.0,
    ensemble_inertial_dt: float = 1.0,
    ensemble_linear_damping: float = 0.9,
    ensemble_linear_gain: float = 1.0,
    ensemble_damping_mode: str = "linear",
    ensemble_barrier_decay: float = 0.95,
    ensemble_barrier_build: float = 0.05,
    ensemble_barrier_relief: float = 0.25,
    ensemble_target_transition_only: bool = False,
    ensemble_target_mix_alpha: float = 0.03,
    ensemble_target_mix_tolerance: float = 1e-9,
    ensemble_angular_mix_alpha: float = 0.0,
    ensemble_angular_damping: float = 0.9,
    ensemble_angular_gain: float = 0.2,
    ensemble_direction_set: Optional[np.ndarray] = None,
    ensemble_em_refresh_on_horizon_crossing: bool = True,
    ensemble_em_refresh_on_horizon_leaving: bool = False,
    ensemble_em_refresh_horizon_ang: float = 15.0,
    ensemble_em_refresh_min_seq_sep: int = 3,
    ensemble_em_refresh_on_large_disp: bool = False,
    ensemble_em_refresh_large_disp_thresh: float = 0.35,
    ensemble_em_max_extra_directions: int = 24,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Fast minimization for long chains. Default: bonds + full vector-sum horizon (analytical).
    fast_horizon=True: bonds-only (nearest-neighbor), for debugging/speed.
    fast_local_theta: passed to final ``e_tot_ca_with_bonds`` energy report only.

    If collapse=True: two-stage annealing for globule formation.
    - Optional compact init: collapse_init_steps of Rg-only + bond projection.
    - Stage 1 (first collapse_frac of max_iter): add k_rg * grad(Rg²) to pull chain compact;
      use loose_r_max in bond projection so chain can collapse. k_rg defaults to K_RG_COLLAPSE.
    - Stage 2: standard bonds + horizon, r_max=6.0, refine.
    """
    k_rg = k_rg_collapse if k_rg_collapse is not None else K_RG_COLLAPSE
    pos = np.array(ca_init, dtype=float)
    n = pos.shape[0]
    direction_set = ensemble_direction_set
    if direction_set is None:
        direction_set = build_direction_set_6_axes()
    direction_set_active = np.asarray(direction_set, dtype=float)
    a_mix = float(ensemble_translation_mix_alpha)
    use_target_inertial = (not bool(ensemble_target_transition_only)) or (
        abs(a_mix - float(ensemble_target_mix_alpha)) <= float(ensemble_target_mix_tolerance)
    )
    linear_momentum_state = np.zeros((n, 3), dtype=float)
    barrier_budget_state = np.zeros((n,), dtype=float)
    resonance_state = 0.0
    omega_state = np.zeros(3, dtype=float)
    r_min_tight, r_max_tight = 2.5, 6.0
    # During collapse allow tighter spacing (r_min_loose) so chain can compactify; projection won't re-extend
    r_min_loose = 2.0
    n_collapse = int(collapse_frac * max_iter) if collapse else 0
    n_refine = max_iter - n_collapse
    total_iter = 0

    write_trajectory_frame(trajectory_log, 0, pos)
    if on_step_callback is not None:
        on_step_callback(0, pos.copy())

    # Optional: compact init via Rg-only steps (pull toward COM while keeping bonds)
    if collapse and collapse_init_steps > 0:
        for _ in range(collapse_init_steps):
            pos_before = pos.copy()
            grad = (k_rg * grad_rg_squared(pos)).astype(pos.dtype)
            g_norm = np.linalg.norm(grad)
            if g_norm < 1e-9:
                break
            step = 1.0 / (g_norm + 1e-6)
            delta_grad = -step * grad
            disp_best = None
            if a_mix > 0.0:
                sel = choose_best_translation_direction(
                    grad=grad,
                    positions=pos,
                    step=ensemble_translation_step,
                    span=ensemble_decay_span,
                    p=ensemble_s2_power,
                    beta=ensemble_beta,
                    score_lambda=ensemble_score_lambda,
                    direction_set=direction_set_active,
                    sources=(0, -1),
                    residue_masses=z_list,
                    left_anchor_infinite=False,
                    linear_momentum_state=linear_momentum_state,
                    barrier_budget_state=barrier_budget_state,
                    inertial_dt=ensemble_inertial_dt if use_target_inertial else 1.0,
                    linear_damping=ensemble_linear_damping if use_target_inertial else 0.0,
                    linear_gain=ensemble_linear_gain if use_target_inertial else 1.0,
                    damping_mode=ensemble_damping_mode if use_target_inertial else "linear",
                    barrier_decay=ensemble_barrier_decay if use_target_inertial else 1.0,
                    barrier_build=ensemble_barrier_build if use_target_inertial else 0.0,
                    barrier_relief=ensemble_barrier_relief if use_target_inertial else 0.0,
                    resonance_state=resonance_state,
                    omega_state=omega_state,
                    angular_mix=ensemble_angular_mix_alpha if use_target_inertial else 0.0,
                    angular_damping=ensemble_angular_damping if use_target_inertial else 0.0,
                    angular_gain=ensemble_angular_gain if use_target_inertial else 0.0,
                )
                disp_best = np.asarray(sel["best_disp"], dtype=float)
                linear_momentum_state = np.asarray(
                    sel.get("best_linear_momentum_state", linear_momentum_state), dtype=float
                ).reshape(n, 3)
                barrier_budget_state = np.asarray(
                    sel.get("best_barrier_budget_state", barrier_budget_state), dtype=float
                ).reshape(n,)
                resonance_state = float(sel.get("best_resonance_state", resonance_state))
                omega_state = np.asarray(sel.get("best_omega_state", omega_state), dtype=float).reshape(3,)
                n_disp = float(np.linalg.norm(disp_best)) + 1e-14
                n_delta = float(np.linalg.norm(delta_grad)) + 1e-14
                disp_scaled = disp_best * (n_delta / n_disp)
                pos = pos + (1.0 - a_mix) * delta_grad + a_mix * disp_scaled
            else:
                pos = pos + delta_grad
            pos = _project_bonds(pos, r_min=r_min_loose, r_max=loose_r_max)
            if a_mix > 0.0:
                direction_set_active, _, _, _ = maybe_refresh_em_field_direction_set(
                    pos_before,
                    pos,
                    grad,
                    direction_set,
                    direction_set_active,
                    disp_best,
                    refresh_on_horizon_crossing=bool(ensemble_em_refresh_on_horizon_crossing),
                    refresh_on_horizon_leaving=bool(ensemble_em_refresh_on_horizon_leaving),
                    horizon_ang=float(ensemble_em_refresh_horizon_ang),
                    min_seq_sep=int(ensemble_em_refresh_min_seq_sep),
                    refresh_on_large_disp=bool(ensemble_em_refresh_on_large_disp),
                    large_disp_thresh=float(ensemble_em_refresh_large_disp_thresh),
                    max_extra_vectors=int(ensemble_em_max_extra_directions),
                )
            total_iter += 1
            write_trajectory_frame(trajectory_log, total_iter, pos)
            if on_step_callback is not None:
                on_step_callback(total_iter, pos.copy())

    # Stage 1: collapse (loose bonds + Rg bias). Use bonds-only so horizon repulsion
    # doesn't oppose the Rg pull; allow r_min_loose so projection doesn't re-extend.
    # Larger step during collapse (1.5/g_norm) so chain actually moves toward COM.
    if n_collapse > 0:
        for it in range(n_collapse):
            pos_before = pos.copy()
            grad = grad_bonds_only(pos, r_min=r_min_loose, r_max=loose_r_max)
            grad = grad + k_rg * grad_rg_squared(pos)
            g_norm = np.linalg.norm(grad)
            if g_norm < 1e-4:
                break
            step = 3.0 / (g_norm + 1e-6)  # larger step in collapse phase to reach compact state
            delta_grad = -step * grad
            disp_best = None
            if a_mix > 0.0:
                sel = choose_best_translation_direction(
                    grad=grad,
                    positions=pos,
                    step=ensemble_translation_step,
                    span=ensemble_decay_span,
                    p=ensemble_s2_power,
                    beta=ensemble_beta,
                    score_lambda=ensemble_score_lambda,
                    direction_set=direction_set_active,
                    sources=(0, -1),
                    residue_masses=z_list,
                    left_anchor_infinite=False,
                    linear_momentum_state=linear_momentum_state,
                    barrier_budget_state=barrier_budget_state,
                    inertial_dt=ensemble_inertial_dt if use_target_inertial else 1.0,
                    linear_damping=ensemble_linear_damping if use_target_inertial else 0.0,
                    linear_gain=ensemble_linear_gain if use_target_inertial else 1.0,
                    damping_mode=ensemble_damping_mode if use_target_inertial else "linear",
                    barrier_decay=ensemble_barrier_decay if use_target_inertial else 1.0,
                    barrier_build=ensemble_barrier_build if use_target_inertial else 0.0,
                    barrier_relief=ensemble_barrier_relief if use_target_inertial else 0.0,
                    resonance_state=resonance_state,
                    omega_state=omega_state,
                    angular_mix=ensemble_angular_mix_alpha if use_target_inertial else 0.0,
                    angular_damping=ensemble_angular_damping if use_target_inertial else 0.0,
                    angular_gain=ensemble_angular_gain if use_target_inertial else 0.0,
                )
                disp_best = np.asarray(sel["best_disp"], dtype=float)
                linear_momentum_state = np.asarray(
                    sel.get("best_linear_momentum_state", linear_momentum_state), dtype=float
                ).reshape(n, 3)
                barrier_budget_state = np.asarray(
                    sel.get("best_barrier_budget_state", barrier_budget_state), dtype=float
                ).reshape(n,)
                resonance_state = float(sel.get("best_resonance_state", resonance_state))
                omega_state = np.asarray(sel.get("best_omega_state", omega_state), dtype=float).reshape(3,)
                n_disp = float(np.linalg.norm(disp_best)) + 1e-14
                n_delta = float(np.linalg.norm(delta_grad)) + 1e-14
                disp_scaled = disp_best * (n_delta / n_disp)
                pos = pos + (1.0 - a_mix) * delta_grad + a_mix * disp_scaled
            else:
                pos = pos + delta_grad
            pos = _project_bonds(pos, r_min=r_min_loose, r_max=loose_r_max)
            if a_mix > 0.0:
                direction_set_active, _, _, _ = maybe_refresh_em_field_direction_set(
                    pos_before,
                    pos,
                    grad,
                    direction_set,
                    direction_set_active,
                    disp_best,
                    refresh_on_horizon_crossing=bool(ensemble_em_refresh_on_horizon_crossing),
                    refresh_on_horizon_leaving=bool(ensemble_em_refresh_on_horizon_leaving),
                    horizon_ang=float(ensemble_em_refresh_horizon_ang),
                    min_seq_sep=int(ensemble_em_refresh_min_seq_sep),
                    refresh_on_large_disp=bool(ensemble_em_refresh_on_large_disp),
                    large_disp_thresh=float(ensemble_em_refresh_large_disp_thresh),
                    max_extra_vectors=int(ensemble_em_max_extra_directions),
                )
            total_iter += 1
            write_trajectory_frame(trajectory_log, total_iter, pos)
            if on_step_callback is not None:
                on_step_callback(total_iter, pos.copy())

    g_extra_kw = dict(grad_full_extra_kwargs or {})
    best_e_seen = float("inf")
    # Stage 2: refine (tight bonds, no Rg); project into standard [2.5, 6] Å
    for it in range(n_refine):
        pos_before = pos.copy()
        if fast_horizon:
            grad = grad_bonds_only(pos)
        else:
            gkw_iter = dict(g_extra_kw)
            if variational_bound_prune:
                w_pair = float(gkw_iter.get("variational_pair_weight", 0.0))
                if w_pair > 0.0:
                    # Lean-style pruning: if lower bound for current state cannot beat best seen,
                    # skip expensive variational FD gradient for this step.
                    base_e = float(e_tot_ca_with_bonds(pos, z_list, fast_local_theta=fast_local_theta))
                    n_pairs = count_variational_pairs_ca(
                        pos,
                        dist_cutoff=float(gkw_iter.get("variational_pair_dist_cutoff", 12.0)),
                        min_seq_sep=int(gkw_iter.get("variational_pair_min_seq_sep", 3)),
                        max_pairs=int(gkw_iter.get("variational_pair_max_pairs", 400)),
                    )
                    lb = base_e + variational_pair_lower_bound_term(
                        n_pairs,
                        pair_weight=w_pair,
                        epsilon=float(gkw_iter.get("variational_pair_epsilon", 0.1)),
                    )
                    if np.isfinite(best_e_seen) and lb >= (best_e_seen - float(variational_bound_prune_margin)):
                        gkw_iter["variational_pair_weight"] = 0.0
            grad = grad_full(
                pos,
                z_list,
                include_bonds=True,
                include_horizon=True,
                include_clash=True,
                **gkw_iter,
            )
        if grad_extra_fn is not None:
            grad = grad + grad_extra_fn(pos, z_list)
        g_norm = np.linalg.norm(grad)
        if g_norm < 1e-4:
            break
        step = 0.5 / (g_norm + 1e-6)
        delta_grad = -step * grad
        disp_best = None
        if a_mix > 0.0:
            sel = choose_best_translation_direction(
                grad=grad,
                positions=pos,
                step=ensemble_translation_step,
                span=ensemble_decay_span,
                p=ensemble_s2_power,
                beta=ensemble_beta,
                score_lambda=ensemble_score_lambda,
                direction_set=direction_set_active,
                sources=(0, -1),
                residue_masses=z_list,
                left_anchor_infinite=False,
                linear_momentum_state=linear_momentum_state,
                barrier_budget_state=barrier_budget_state,
                inertial_dt=ensemble_inertial_dt if use_target_inertial else 1.0,
                linear_damping=ensemble_linear_damping if use_target_inertial else 0.0,
                linear_gain=ensemble_linear_gain if use_target_inertial else 1.0,
                damping_mode=ensemble_damping_mode if use_target_inertial else "linear",
                barrier_decay=ensemble_barrier_decay if use_target_inertial else 1.0,
                barrier_build=ensemble_barrier_build if use_target_inertial else 0.0,
                barrier_relief=ensemble_barrier_relief if use_target_inertial else 0.0,
                resonance_state=resonance_state,
                omega_state=omega_state,
                angular_mix=ensemble_angular_mix_alpha if use_target_inertial else 0.0,
                angular_damping=ensemble_angular_damping if use_target_inertial else 0.0,
                angular_gain=ensemble_angular_gain if use_target_inertial else 0.0,
            )
            disp_best = np.asarray(sel["best_disp"], dtype=float)
            linear_momentum_state = np.asarray(
                sel.get("best_linear_momentum_state", linear_momentum_state), dtype=float
            ).reshape(n, 3)
            barrier_budget_state = np.asarray(
                sel.get("best_barrier_budget_state", barrier_budget_state), dtype=float
            ).reshape(n,)
            resonance_state = float(sel.get("best_resonance_state", resonance_state))
            omega_state = np.asarray(sel.get("best_omega_state", omega_state), dtype=float).reshape(3,)
            n_disp = float(np.linalg.norm(disp_best)) + 1e-14
            n_delta = float(np.linalg.norm(delta_grad)) + 1e-14
            disp_scaled = disp_best * (n_delta / n_disp)
            pos = pos + (1.0 - a_mix) * delta_grad + a_mix * disp_scaled
        else:
            pos = pos + delta_grad
        pos = _project_bonds(pos, r_min=r_min_tight, r_max=r_max_tight)
        if a_mix > 0.0:
            direction_set_active, _, _, _ = maybe_refresh_em_field_direction_set(
                pos_before,
                pos,
                grad,
                direction_set,
                direction_set_active,
                disp_best,
                refresh_on_horizon_crossing=bool(ensemble_em_refresh_on_horizon_crossing),
                refresh_on_horizon_leaving=bool(ensemble_em_refresh_on_horizon_leaving),
                horizon_ang=float(ensemble_em_refresh_horizon_ang),
                min_seq_sep=int(ensemble_em_refresh_min_seq_sep),
                refresh_on_large_disp=bool(ensemble_em_refresh_on_large_disp),
                large_disp_thresh=float(ensemble_em_refresh_large_disp_thresh),
                max_extra_vectors=int(ensemble_em_max_extra_directions),
            )
        e_step = float(e_tot_ca_with_bonds(pos, z_list, fast_local_theta=fast_local_theta))
        if e_step < best_e_seen:
            best_e_seen = e_step
        total_iter += 1
        write_trajectory_frame(trajectory_log, total_iter, pos)
        if on_step_callback is not None:
            on_step_callback(total_iter, pos.copy())

    z = np.full(n, 6)
    e_final = float(e_tot_ca_with_bonds(pos, z, fast_local_theta=fast_local_theta))
    msg = "Bonds + horizon (fast path)"
    if collapse:
        msg = "Two-stage collapse + refine (long chain)"
    elif fast_horizon:
        msg = "Bonds-only (fast path)"
    return pos, {
        "e_final": e_final,
        "e_initial": e_final,
        "n_iter": total_iter,
        "success": True,
        "message": msg,
    }


def _z_list_ca(sequence: str) -> np.ndarray:
    """Z for each Cα (6). Length n_res."""
    n = len(sequence)
    return np.full(n, Z_CA)


def _full_backbone_positions_and_z(
    atoms: List[Tuple[str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert list of (atom_name, xyz) to (positions (n,3), z_list (n))."""
    positions = np.array([xyz for _, xyz in atoms])
    z_list = []
    for name, _ in atoms:
        if name == "N":
            z_list.append(Z_N)
        elif name == "CA":
            z_list.append(Z_CA)
        elif name == "C":
            z_list.append(Z_C)
        elif name == "O":
            z_list.append(Z_O)
        else:
            z_list.append(6)
    return positions, np.array(z_list)


# Default max iterations for long-chain path (n_res > 50); two-stage collapse uses this
LONG_CHAIN_MAX_ITER_DEFAULT = 250
LONG_CHAIN_COLLAPSE_INIT_STEPS = 40

LIGAND_REFINE_STEPS = 50
LIGAND_STEP_T = 0.05
LIGAND_STEP_ANG = 0.02
# Lean ``ProteinQCRefinement``: soft clash σ (Å) and weight on clash vs site trace (gradient uses clash only).
LIGAND_QC_SOFT_SIGMA = 3.0
LIGAND_QC_CLASH_WEIGHT = 1.0


def _refine_ligands_6dof(
    pos_bb: np.ndarray,
    z_bb: np.ndarray,
    ligands: List[LigandAgent],
    max_steps: int = LIGAND_REFINE_STEPS,
    step_t: float = LIGAND_STEP_T,
    step_ang: float = LIGAND_STEP_ANG,
    grad_full_kwargs: Optional[Dict[str, Any]] = None,
    *,
    ligand_refinement_mode: str = "lean_qc",
    qc_soft_clash_sigma: float = LIGAND_QC_SOFT_SIGMA,
    qc_clash_weight: float = LIGAND_QC_CLASH_WEIGHT,
) -> None:
    """
    Minimize energy over ligand 6-DOF only; protein positions fixed. Modifies ligands in place.

    ``ligand_refinement_mode``:
      - ``lean_qc``: Lean ``ProteinQCRefinement`` — weighted soft clash vs a single σ (site term is constant).
      - ``horizon``: legacy ``grad_full`` + protein bond restoration on the fixed backbone.
    """
    n_bb = pos_bb.shape[0]
    n_lig_atoms = sum(lig.n_atoms() for lig in ligands)
    if n_lig_atoms == 0:
        return
    z_lig = np.concatenate([lig.z_list for lig in ligands])
    z_list = np.concatenate([z_bb, z_lig])
    indices_per_ligand: List[List[int]] = []
    off = n_bb
    for lig in ligands:
        n = lig.n_atoms()
        indices_per_ligand.append(list(range(off, off + n)))
        off += n

    mode = str(ligand_refinement_mode or "lean_qc").strip().lower()
    use_qc = mode in ("lean_qc", "qc", "protein_qc_refinement")
    sigma = float(qc_soft_clash_sigma)
    w_clash = float(max(0.0, qc_clash_weight))

    # Match ``minimize_full_chain`` ``grad_full`` screening (``em_scale``, …). The
    # ``hBondProxy`` FD term is Cα-chain-only; disable it for protein+ligand atom arrays.
    gkw = dict(grad_full_kwargs or {})
    gkw["hbond_weight"] = 0.0

    for _ in range(max_steps):
        lig_pos = np.concatenate([lig.get_world_positions() for lig in ligands], axis=0)
        combined = np.concatenate([pos_bb, lig_pos], axis=0)
        if use_qc:
            grad = np.zeros_like(combined)
            grad[n_bb:] = w_clash * grad_qc_soft_clash_protein_fixed(pos_bb, lig_pos, sigma)
        else:
            # Bonds only within protein; horizon + clash for full system (no protein–ligand bonds)
            grad = grad_full(
                combined,
                z_list,
                include_bonds=False,
                include_horizon=True,
                **gkw,
            )
            grad[:n_bb] += grad_bonds_only(pos_bb, include_clash=False)
        for lig, indices in zip(ligands, indices_per_ligand):
            F, T = rigid_body_force_torque(combined, grad, indices)
            t, euler = lig.get_6dof()
            lig.set_6dof(t - step_t * F, euler - step_ang * T)


def merged_multichain_backbone_atoms(
    result_and_chain_ids: List[Tuple[Dict[str, object], str]],
    *,
    chain_gap: float = 50.0,
) -> List[Tuple[str, np.ndarray]]:
    """
    Concatenate backbone (N,CA,C,O,…) from several minimizer ``result`` dicts with +x offsets,
    matching the layout used by ``casp_server._merge_pdb_chains`` (chain–chain separation).
    """
    chain_mins: List[float] = []
    for result, _ in result_and_chain_ids:
        bb = result.get("backbone_atoms") or []
        if not bb:
            chain_mins.append(0.0)
        else:
            chain_mins.append(min(float(xyz[0]) for _, xyz in bb))
    out: List[Tuple[str, np.ndarray]] = []
    prev_max_x = float("-inf")
    for ci, (result, _) in enumerate(result_and_chain_ids):
        backbone_atoms = result.get("backbone_atoms") or []
        sequence = str(result.get("sequence") or "")
        include_sidechains = bool(result.get("include_sidechains", False))
        n_res = int(result.get("n_res") or 0)
        if not backbone_atoms or not sequence or n_res <= 0:
            continue
        offset_x = max(0.0, prev_max_x + chain_gap - chain_mins[ci])
        idx = 0
        chain_max_x = float("-inf")
        for res_id in range(1, n_res + 1):
            res_1 = sequence[res_id - 1]
            n_atoms_this = (5 if res_1 != "G" else 4) if include_sidechains else 4
            for _ in range(n_atoms_this):
                name, xyz = backbone_atoms[idx]
                x = float(xyz[0]) + offset_x
                y, z = float(xyz[1]), float(xyz[2])
                chain_max_x = max(chain_max_x, x)
                out.append((name, np.asarray([x, y, z], dtype=np.float64)))
                idx += 1
        prev_max_x = chain_max_x
    return out


def refine_ligands_on_multichain_results(
    result_and_chain_ids: List[Tuple[Dict[str, object], str]],
    ligands: List[LigandAgent],
    *,
    chain_gap: float = 50.0,
    grad_full_kwargs: Optional[Dict[str, Any]] = None,
    ligand_refine_steps: int = LIGAND_REFINE_STEPS,
    ligand_step_t: float = LIGAND_STEP_T,
    ligand_step_ang: float = LIGAND_STEP_ANG,
    ligand_refinement_mode: str = "lean_qc",
    qc_soft_clash_sigma: float = LIGAND_QC_SOFT_SIGMA,
    qc_clash_weight: float = LIGAND_QC_CLASH_WEIGHT,
) -> None:
    """
    Place ligand centroid at the merged-complex backbone COM, then 6-DOF horizon refinement
    against the fixed multichain backbone (same physics as single-chain ``include_ligands``).
    Modifies ``ligands`` in place.
    """
    if not ligands:
        return
    merged = merged_multichain_backbone_atoms(result_and_chain_ids, chain_gap=chain_gap)
    if not merged:
        return
    pos_bb, z_bb = _full_backbone_positions_and_z(merged)
    protein_com = np.mean(pos_bb, axis=0)
    for lig in ligands:
        t, euler = lig.get_6dof()
        lig.set_6dof(protein_com.copy(), euler)
    _refine_ligands_6dof(
        pos_bb,
        z_bb,
        ligands,
        max_steps=ligand_refine_steps,
        step_t=ligand_step_t,
        step_ang=ligand_step_ang,
        grad_full_kwargs=grad_full_kwargs,
        ligand_refinement_mode=str(ligand_refinement_mode),
        qc_soft_clash_sigma=float(qc_soft_clash_sigma),
        qc_clash_weight=float(qc_clash_weight),
    )


def pdb_hetatm_lines_for_ligands(
    ligands: List[LigandAgent],
    *,
    start_atom_id: int,
    ligand_chain_id: str,
) -> Tuple[List[str], int]:
    """PDB HETATM lines for all ligands; returns (lines, next_atom_id)."""
    lines: List[str] = []
    atom_id = start_atom_id

    def _pdb_coord(x: float) -> str:
        s = f"{float(x):8.3f}"
        return s[-8:] if len(s) > 8 else s

    for res_id_het, lig in enumerate(ligands, start=1):
        res_3 = (lig.res_name or "LIG")[:3]
        for a, (name, xyz) in enumerate(zip(lig.atom_names, lig.get_world_positions())):
            elem = (lig.elements[a] if a < len(lig.elements) else "C")[:2].rjust(2)
            line = (
                f"HETATM{atom_id:5d}  {name:2s}  {res_3:3s} {ligand_chain_id}{res_id_het:4d}    "
                f"{_pdb_coord(xyz[0])}{_pdb_coord(xyz[1])}{_pdb_coord(xyz[2])}  1.00  0.00      {elem:>2s}"
            )
            lines.append(line[:80] if len(line) > 80 else line)
            atom_id += 1
    return lines, atom_id


def minimize_full_chain(
    sequence: str,
    ca_init: Optional[np.ndarray] = None,
    ss_string: Optional[str] = None,
    max_iter: int = 500,
    gtol: float = 1e-5,
    include_sidechains: bool = False,
    include_ligands: bool = False,
    ligands: Optional[List[LigandAgent]] = None,
    fast_horizon: bool = False,
    side_chain_pack: bool = True,
    quick: bool = False,
    collapse: bool = True,
    long_chain_max_iter: Optional[int] = None,
    collapse_init_steps: Optional[int] = None,
    k_rg_collapse: Optional[float] = None,
    trajectory_log_path: Optional[str] = None,
    signal_dump_path: Optional[str] = None,
    simulate_ribosome_tunnel: bool = False,
    tunnel_length: float = 25.0,
    cone_half_angle_deg: float = 12.0,
    lip_plane_distance: float = 0.0,
    post_extrusion_refine: bool = True,
    post_extrusion_refine_mode: str = "em_treetorque",
    post_extrusion_em_max_steps: Optional[int] = None,
    post_extrusion_treetorque_phases: int = 8,
    post_extrusion_treetorque_n_steps: int = 200,
    refinement_temperature_k: float = 310.0,
    post_extrusion_discrete_metropolis: bool = False,
    post_extrusion_discrete_seed: Optional[int] = None,
    post_extrusion_anneal: bool = False,
    post_extrusion_anneal_schedule_k: Optional[Tuple[float, ...]] = None,
    post_extrusion_langevin_steps: int = 0,
    post_extrusion_langevin_noise_fraction: float = 0.2,
    post_extrusion_max_disp_floor: float = 0.25,
    post_extrusion_max_rounds: int = 32,
    post_extrusion_osh_hqiv_native: bool = False,
    post_extrusion_osh_n_iter: int = 120,
    post_extrusion_osh_step_size: float = 0.03,
    post_extrusion_osh_ansatz_depth: int = 2,
    post_extrusion_osh_gate_mix: float = 0.55,
    post_extrusion_osh_hqiv_reference_m: int = 4,
    fast_pass_steps_per_connection: int = 5,
    min_pass_iter_per_connection: int = 15,
    tunnel_axis: Optional[np.ndarray] = None,
    hke_above_tunnel_fraction: float = 0.5,
    tunnel_thermal_gradient_steps: int = 0,
    tunnel_thermal_noise_fraction: float = 0.2,
    tunnel_thermal_step_size: Optional[float] = None,
    tunnel_thermal_reference_temperature_k: float = 310.0,
    tunnel_thermal_seed: Optional[int] = None,
    tunnel_free_terminus_steps: int = 0,
    tunnel_free_terminus_window: int = 10,
    tunnel_handedness_bias_weight: float = 0.0,
    tunnel_handedness_target: float = 0.4,
    tunnel_handedness_sign: float = 1.0,
    em_scale: float = 1.0,
    kappa_dihedral: float = 0.0,
    hbond_weight: float = 0.0,
    hbond_shell_m: int = 3,
    hbond_min_seq_sep: int = 3,
    hbond_max_pairs: int = 200,
    hbond_dist_cutoff: float = 15.0,
    fast_local_theta: bool = False,
    horizon_neighbor_cutoff: Optional[float] = None,
    collective_kink_weight: float = 0.0,
    collective_kink_m: int = 3,
    collective_kink_theta_ref_rad: Optional[float] = None,
    collective_kink_use_ss_mask: bool = False,
    variational_pair_weight: float = 0.0,
    variational_pair_epsilon: float = 0.1,
    variational_pair_sigma: float = 4.0,
    variational_pair_dist_cutoff: float = 12.0,
    variational_pair_min_seq_sep: int = 3,
    variational_pair_max_pairs: int = 400,
    variational_staged_opt: bool = False,
    variational_stage_frac: float = 0.5,
    variational_bound_prune: bool = False,
    variational_bound_prune_margin: float = 0.0,
    inertial_twist_weight: float = 0.0,
    inertial_twist_theta_ref_rad: Optional[float] = None,
    inertial_twist_exponent: float = 1.0,
    ensemble_translation_mix_alpha: float = 0.0,
    ensemble_translation_step: float = 0.35,
    ensemble_decay_span: float = 0.25,
    ensemble_beta: float = 0.35,
    ensemble_s2_power: float = 1.0,
    ensemble_score_lambda: float = 0.0,
    ensemble_inertial_dt: float = 1.0,
    ensemble_linear_damping: float = 0.9,
    ensemble_linear_gain: float = 1.0,
    ensemble_damping_mode: str = "linear",
    ensemble_barrier_decay: float = 0.95,
    ensemble_barrier_build: float = 0.05,
    ensemble_barrier_relief: float = 0.25,
    ensemble_target_transition_only: bool = False,
    ensemble_target_mix_alpha: float = 0.03,
    ensemble_target_mix_tolerance: float = 1e-9,
    ensemble_angular_mix_alpha: float = 0.0,
    ensemble_angular_damping: float = 0.9,
    ensemble_angular_gain: float = 0.2,
    ensemble_em_refresh_on_horizon_crossing: bool = True,
    ensemble_em_refresh_on_horizon_leaving: bool = False,
    ensemble_em_refresh_horizon_ang: float = 15.0,
    ensemble_em_refresh_min_seq_sep: int = 3,
    ensemble_em_refresh_on_large_disp: bool = False,
    ensemble_em_refresh_large_disp_thresh: float = 0.35,
    ensemble_em_max_extra_directions: int = 24,
    ligand_refine_steps: int = LIGAND_REFINE_STEPS,
    ligand_step_t: float = LIGAND_STEP_T,
    ligand_step_ang: float = LIGAND_STEP_ANG,
    ligand_refinement_mode: str = "lean_qc",
    qc_soft_clash_sigma: float = LIGAND_QC_SOFT_SIGMA,
    qc_clash_weight: float = LIGAND_QC_CLASH_WEIGHT,
) -> Dict[str, object]:
    """
    Full-chain minimization: minimize E_tot over Cα, then rebuild backbone.

    Args:
        sequence: One-letter amino acid sequence (or will be parsed from FASTA if needed).
        ca_init: Initial Cα positions (n_res, 3) in Å. If None, from SS-aware placement.
        ss_string: Optional SS string (H/E/C). If None and ca_init is None, predict_ss(sequence).
        max_iter: L-BFGS max iterations (short chains).
        gtol: Gradient tolerance for convergence.
        include_sidechains: If True, add Cβ positions (from ideal geometry); no full side chains yet.
        include_ligands: If True and ligands is non-empty, run 6-DOF rigid-body refinement for ligands against the fixed protein (horizon + clash). Default False for pure CAMEO compatibility.
        ligands: List of LigandAgent (from parse_ligands). Used only when include_ligands=True.
        fast_horizon: If True, use bonds-only gradient (nearest-neighbor); faster, for debugging.
        side_chain_pack: If True and include_sidechains, run lightweight Cβ rotamer search after backbone.
        quick: If True, rapid prototyping: no side chains, fewer iterations, relaxed gtol, bonds-only for long chains.
        collapse: If True and n_res > 50, use two-stage annealing (Rg collapse then refine) for globule formation.
        long_chain_max_iter: Max iterations for long-chain path (default 250). Used when n_res > 50.
        collapse_init_steps: Optional Rg-only steps before main loop to get compact init (default 40 when collapse=True).
        k_rg_collapse: Weight for Rg² term during collapse (default K_RG_COLLAPSE). Increase for stronger pull toward compact.
        trajectory_log_path: If set, write each step's Cα positions to this file as JSONL (one {"t": step, "positions": [...]} per line, flush each line) for Manim or other viewers; use with `tail -f` to watch the run.
        signal_dump_path: If set, on SIGINT/SIGTERM (e.g. Ctrl+C) write the current Cα state to this path as a PDB and exit. Lets you recover a partial run.
        simulate_ribosome_tunnel: If True, use co-translational mode: null search cone (exit tunnel), plane at lip to null unphysical re-entry, fast-pass spaghetti (rigid group + bell-end large translations), and a short HKE min pass on each connection event. Default False for backward compatibility.
        tunnel_length: Length of tunnel in Å along extrusion axis (default 25.0). Residues with Cα inside this segment are cone-constrained.
        cone_half_angle_deg: Half-angle of the cone in degrees (default 12.0). Cα inside the tunnel must stay within this cone from the peptidyl transferase center.
        lip_plane_distance: Extra distance in Å beyond tunnel_length for the lip plane (default 0.0). Lip is at tunnel_length + lip_plane_distance along the axis.
        post_extrusion_refine: When simulate_ribosome_tunnel=True, if True (default), run post-extrusion refinement after the chain leaves the cone/plane (see ``post_extrusion_refine_mode``).
        post_extrusion_refine_mode: ``"em_treetorque"`` (default) = 3D EM-field ``relax_to_convergence`` with ``alternate_hke=False`` (no ``grad_full`` Cα step), then discrete ``run_discrete_refinement`` (tree-torque). ``"hke"`` = legacy repeated ``_minimize_bonds_fast`` / L-BFGS rounds. ``"none"`` = skip.
        post_extrusion_em_max_steps: Cap on EM relax steps (``None`` = length-scaled inside ``refine_ca_post_tunnel_em_treetorque``).
        post_extrusion_treetorque_phases: Max phases for ``run_discrete_refinement`` (when mode is ``em_treetorque``).
        post_extrusion_treetorque_n_steps: Inner discrete steps per phase cap (Metropolis loop length).
        refinement_temperature_k: Kelvin-like scale for ``run_discrete_refinement`` and thermal Cα relax when post mode is ``em_treetorque``.
        post_extrusion_discrete_metropolis: If True, discrete φ/ψ moves use Metropolis at ``refinement_temperature_k`` (else greedy downhill).
        post_extrusion_discrete_seed: RNG seed for Metropolis / thermal relax (``None`` = non-deterministic stream).
        post_extrusion_anneal: If True and post mode is ``em_treetorque``, run a short multi-stage Metropolis cool-down instead of one discrete pass (see ``post_extrusion_anneal_schedule_k``).
        post_extrusion_anneal_schedule_k: Optional Kelvin schedule (≥2 values, high→low); last stage should match ``refinement_temperature_k`` for consistent Langevin tail.
        post_extrusion_langevin_steps: After discrete refine, number of ``thermal_gradient_relax_ca`` steps (0 = off).
        post_extrusion_langevin_noise_fraction: Scales RMS of Gaussian noise vs ``step_size`` in thermal relax.
        post_extrusion_max_disp_floor: (``hke`` mode only) Minimum motion threshold (Å) for stopping post-extrusion iterations.
        post_extrusion_max_rounds: (``hke`` mode only) Safety cap on outer rounds (each runs one ``_minimize_bonds_fast`` collapse+refine).
        post_extrusion_osh_hqiv_native: After EM/tree-torque or HKE post-extrusion (when mode is not ``none``), run ``minimize_ca_with_osh_oracle`` with Lean HQIV-native π-phase sparse gates (``OSHoracleHQIVNative``).
        post_extrusion_osh_n_iter / _step_size / _ansatz_depth / _gate_mix / _hqiv_reference_m: Pass-through to that OSHoracle refine.
        fast_pass_steps_per_connection: Co-translational rigid-group + bell gradient steps per segment (set 0 to skip).
        min_pass_iter_per_connection: L-BFGS iterations per segment in the tunnel (connection-triggered “HKE” pass; set 0 to skip).
        tunnel_axis: (3,) unit vector for tunnel axis; default +Z. Chain is aligned so N-terminus is at origin and extrusion is along this axis.
        hke_above_tunnel_fraction: When simulate_ribosome_tunnel=True, HKE (connection-triggered L-BFGS) is applied only to residues above this fraction of the tunnel length (default 0.5 = 50%). The chain is built with the fast method (rigid group + bell-end); only the part above this threshold is updated by the min pass.
        tunnel_thermal_gradient_steps: When simulate_ribosome_tunnel=True and >0, after each binary-tree segment's masked L-BFGS pass, run this many kT-noised gradient steps with the same cone/plane/HKE-fraction masking as L-BFGS. Uses ``refinement_temperature_k`` as the thermal scale. Default 0 = deterministic tunnel (legacy).
        tunnel_thermal_noise_fraction: RMS multiplier for Gaussian Cα noise vs step size (same role as post-extrusion Langevin).
        tunnel_thermal_step_size: Cα step scale; ``None`` → 0.022 if ``quick`` else 0.028.
        tunnel_thermal_reference_temperature_k: Denominator for √(T/T_ref) noise scaling (typically 310 K).
        tunnel_thermal_seed: RNG seed for tunnel thermal steps; ``None`` uses ``post_extrusion_discrete_seed`` if set, else non-deterministic.
        em_scale: Multiplier on horizon (EM) pole strength in grad_full (default 1). Use 1/ε_r for bulk water screening (HQIV Lean `waterDielectricValley`).
        kappa_dihedral: Weight on Lean `foldEnergyWithDihedral` correction κ(1−cos Δθ) on φ/ψ toward the HQIV α Ramachandran basin (default 0 = off).
        hbond_weight: Weight on Lean ``HQIVLongRange`` ``hBondProxy`` gradient (finite differences on Cα; default 0 = off). Expensive for long chains.
        hbond_shell_m: Nuclear/shell index ``m`` for ``K_hbond m``, ``R_hbond m`` (default 3).
        hbond_min_seq_sep, hbond_max_pairs, hbond_dist_cutoff: Pair list for ``total_h_bond_proxy_energy_ca``.
        fast_local_theta: If True, evaluate Σ ħc/Θ_i + damping in ``e_tot_ca_with_bonds`` via batched
            nearest-neighbor distances (same Θ_i as the legacy loop, faster for long chains).
        horizon_neighbor_cutoff: If set, cap the spatial neighbor list radius (Å) in ``grad_full`` horizon
            forces below module default ``CUTOFF`` (prunes weak long-range pairs; faster, approximate).
        collective_kink_weight: Weight on Lean ``HQIVCollectiveModes`` Cα kink budget (0 = off).
        collective_kink_m: Shell index ``m`` for ``K_multipole``.
        collective_kink_theta_ref_rad: Interior Cα bend reference (rad); default from HQIV α-helix trace.
        collective_kink_use_ss_mask: If True, only apply kink budget at residues predicted ``H`` (``predict_ss``).
        variational_pair_weight: Weight on Lean ``VariationalScoreTerms`` bounded nonlocal pair proxy
            (0 = off). Added to fold score as ``w_pair * pair_sum``.
        variational_pair_epsilon: Per-pair depth parameter ε for bounded pair proxy. Lower bound per pair
            is ``-ε`` by construction, matching Lean lower-bound assumptions.
        variational_pair_sigma: LJ-like pair radius scale σ (Å) for bounded pair proxy.
        variational_pair_dist_cutoff: Pair cutoff (Å) for nonlocal variational pair term.
        variational_pair_min_seq_sep: Minimum sequence separation |i-j| for pair term.
        variational_pair_max_pairs: Safety cap on evaluated pair terms.
        variational_staged_opt: If True and variational pair is enabled, run a staged objective:
            first optimize with variational pair off, then refine with full variational weight.
        variational_stage_frac: Fraction of iteration budget used by stage-1 warmup (0..1).
        variational_bound_prune: If True, use Lean-style lower-bound gating in long-chain fast
            refinement to skip expensive variational FD gradients when they cannot beat best seen.
        variational_bound_prune_margin: Safety margin (energy units) for bound gating.
        inertial_twist_weight: Weight on center-heavy local bend penalty so central twists are
            penalized more than terminal twists.
        inertial_twist_theta_ref_rad: Reference local bend angle (rad) for inertial twist term.
        inertial_twist_exponent: Shape exponent for center-heavy weighting profile.
        ligand_refine_steps, ligand_step_t, ligand_step_ang: Rigid-body refinement of ligands when
            ``include_ligands`` is True. Default ``ligand_refinement_mode="lean_qc"`` matches Lean
            ``ProteinQCRefinement`` (soft clash). Use ``"horizon"`` for legacy ``grad_full`` screening
            (``hbond_weight`` forced off).
        ligand_refinement_mode: ``lean_qc`` (default) or ``horizon``.
        qc_soft_clash_sigma, qc_clash_weight: Lean QC soft-clash distance scale (Å) and weight (only for ``lean_qc``).

    Returns:
        dict with:
          ca_min: (n_res, 3) minimized Cα in Å.
          backbone_atoms: list of (atom_name, xyz) per residue (N, CA, C, O).
          E_ca_final: E_tot evaluated on Cα only (eV).
          E_backbone_final: E_tot evaluated on full backbone N,CA,C,O (eV).
          info: from minimize_e_tot_lbfgs (e_final, e_initial, n_iter, success, message).
          n_res: number of residues.
          sequence: sequence used.
    """
    seq = sequence.strip().upper()
    if not seq or not all(c in AA_1to3 for c in seq):
        seq = _parse_fasta(sequence)
    if not seq:
        return {
            "ca_min": np.zeros((0, 3)),
            "backbone_atoms": [],
            "E_ca_final": 0.0,
            "E_backbone_final": 0.0,
            "info": {"e_final": 0, "e_initial": 0, "n_iter": 0, "success": True, "message": "Empty"},
            "n_res": 0,
            "sequence": "",
        }
    n_res = len(seq)
    if ca_init is None:
        ca_init = _place_backbone_ca(seq, ss_string=ss_string)
    else:
        ca_init = np.asarray(ca_init, dtype=float)
        assert ca_init.shape == (n_res, 3)
    z_ca = _z_list_ca(seq)

    collective_kink_ss_mask_np: Optional[np.ndarray] = None
    if collective_kink_weight > 0.0 and collective_kink_use_ss_mask:
        if ss_string is not None and len(ss_string) == n_res:
            sk = ss_string
        else:
            from .casp_submission import predict_ss

            sk, _ = predict_ss(seq, window=5)
        collective_kink_ss_mask_np = np.array([c.upper() == "H" for c in sk], dtype=bool)

    _e_tot_bind_kw: Dict[str, Any] = {"fast_local_theta": bool(fast_local_theta)}
    if collective_kink_weight > 0.0:
        _e_tot_bind_kw["collective_kink_weight"] = float(collective_kink_weight)
        _e_tot_bind_kw["collective_kink_m"] = int(collective_kink_m)
        if collective_kink_theta_ref_rad is not None:
            _e_tot_bind_kw["collective_kink_theta_ref_rad"] = float(collective_kink_theta_ref_rad)
        if collective_kink_ss_mask_np is not None:
            _e_tot_bind_kw["collective_kink_ss_mask"] = collective_kink_ss_mask_np
    if variational_pair_weight > 0.0:
        _e_tot_bind_kw["variational_pair_weight"] = float(variational_pair_weight)
        _e_tot_bind_kw["variational_pair_epsilon"] = float(variational_pair_epsilon)
        _e_tot_bind_kw["variational_pair_sigma"] = float(variational_pair_sigma)
        _e_tot_bind_kw["variational_pair_dist_cutoff"] = float(variational_pair_dist_cutoff)
        _e_tot_bind_kw["variational_pair_min_seq_sep"] = int(variational_pair_min_seq_sep)
        _e_tot_bind_kw["variational_pair_max_pairs"] = int(variational_pair_max_pairs)
    if inertial_twist_weight > 0.0:
        _e_tot_bind_kw["inertial_twist_weight"] = float(inertial_twist_weight)
        if inertial_twist_theta_ref_rad is not None:
            _e_tot_bind_kw["inertial_twist_theta_ref_rad"] = float(inertial_twist_theta_ref_rad)
        _e_tot_bind_kw["inertial_twist_exponent"] = float(inertial_twist_exponent)

    if quick:
        include_sidechains = False
        side_chain_pack = False
        fast_horizon = True
        max_iter = min(100, max_iter)
        gtol = max(1e-4, gtol)
        collapse = False
    long_iter = long_chain_max_iter if long_chain_max_iter is not None else LONG_CHAIN_MAX_ITER_DEFAULT
    init_steps = collapse_init_steps if collapse_init_steps is not None else (LONG_CHAIN_COLLAPSE_INIT_STEPS if collapse and n_res > 50 else 0)
    traj_file = None
    if trajectory_log_path:
        traj_file = open(trajectory_log_path, "w")
    # Mutable ref for signal dump: updated each step so Ctrl+C writes latest state to PDB
    current_for_dump: Dict[str, Any] = {"ca": ca_init.copy(), "sequence": seq}

    def _do_signal_dump(exit_after: bool = True) -> None:
        if signal_dump_path is None or current_for_dump["ca"] is None:
            return
        ca = current_for_dump["ca"]
        seq_d = current_for_dump["sequence"]
        if len(seq_d) == 0 or ca.shape[0] != len(seq_d):
            return
        try:
            backbone_atoms = _place_full_backbone(ca, seq_d)
            result = {
                "ca_min": ca,
                "backbone_atoms": backbone_atoms,
                "sequence": seq_d,
                "n_res": len(seq_d),
            }
            pdb_str = full_chain_to_pdb(result)
            with open(signal_dump_path, "w") as f:
                f.write(pdb_str)
            sys.stderr.write(f"Signal dump: wrote current state to {signal_dump_path}\n")
        except Exception as e:
            sys.stderr.write(f"Signal dump failed: {e}\n")
        if exit_after:
            sys.exit(0)

    if signal_dump_path:
        def _handler_exit(_signum: int, _frame: Any) -> None:
            _do_signal_dump(exit_after=True)
        def _handler_dump_only(_signum: int, _frame: Any) -> None:
            _do_signal_dump(exit_after=False)
        signal.signal(signal.SIGINT, _handler_exit)
        signal.signal(signal.SIGTERM, _handler_exit)
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, _handler_dump_only)

    _gf_kw: Dict[str, Any] = {
        "em_scale": em_scale,
        "hbond_weight": hbond_weight,
        "hbond_shell_m": hbond_shell_m,
        "hbond_min_seq_sep": hbond_min_seq_sep,
        "hbond_max_pairs": hbond_max_pairs,
        "hbond_dist_cutoff": hbond_dist_cutoff,
    }
    if horizon_neighbor_cutoff is not None:
        _gf_kw["neighbor_cutoff"] = float(horizon_neighbor_cutoff)
    if collective_kink_weight > 0.0:
        _gf_kw["collective_kink_weight"] = float(collective_kink_weight)
        _gf_kw["collective_kink_m"] = int(collective_kink_m)
        if collective_kink_theta_ref_rad is not None:
            _gf_kw["collective_kink_theta_ref_rad"] = float(collective_kink_theta_ref_rad)
        if collective_kink_ss_mask_np is not None:
            _gf_kw["collective_kink_ss_mask"] = collective_kink_ss_mask_np
    if variational_pair_weight > 0.0:
        _gf_kw["variational_pair_weight"] = float(variational_pair_weight)
        _gf_kw["variational_pair_epsilon"] = float(variational_pair_epsilon)
        _gf_kw["variational_pair_sigma"] = float(variational_pair_sigma)
        _gf_kw["variational_pair_dist_cutoff"] = float(variational_pair_dist_cutoff)
        _gf_kw["variational_pair_min_seq_sep"] = int(variational_pair_min_seq_sep)
        _gf_kw["variational_pair_max_pairs"] = int(variational_pair_max_pairs)
    if inertial_twist_weight > 0.0:
        _gf_kw["inertial_twist_weight"] = float(inertial_twist_weight)
        if inertial_twist_theta_ref_rad is not None:
            _gf_kw["inertial_twist_theta_ref_rad"] = float(inertial_twist_theta_ref_rad)
        _gf_kw["inertial_twist_exponent"] = float(inertial_twist_exponent)

    _e_ca_for_line_search = functools.partial(e_tot_ca_with_bonds, **_e_tot_bind_kw)

    try:
        def _grad_ca_with_lean(pos: np.ndarray, z: np.ndarray, *, include_horizon: bool = True) -> np.ndarray:
            g = grad_full(
                pos,
                z,
                include_bonds=True,
                include_horizon=include_horizon,
                include_clash=True,
                **_gf_kw,
            )
            # Dihedral / backbone reconstruction needs one Cα per residue; only apply when
            # the coordinate block matches the full sequence (co-translational segments are shorter).
            if kappa_dihedral > 0 and pos.shape[0] == len(seq):
                from .hqiv_lean_folding import grad_dihedral_penalty_ca_fd

                g = g + grad_dihedral_penalty_ca_fd(pos, seq, kappa_dihedral)
            return g

        def _e_ca_with_lean(pos: np.ndarray, z: np.ndarray) -> float:
            e = float(e_tot_ca_with_bonds(pos, z, **_e_tot_bind_kw))
            if kappa_dihedral > 0:
                from .hqiv_lean_folding import dihedral_penalty_from_ca

                e += dihedral_penalty_from_ca(pos, seq, kappa=kappa_dihedral)
            return e

        _grad_extra_opt: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
        if kappa_dihedral > 0:
            from .hqiv_lean_folding import grad_dihedral_penalty_ca_fd as _grad_dihedral_fd

            def _grad_extra_opt(pos: np.ndarray, z: np.ndarray) -> np.ndarray:
                return _grad_dihedral_fd(pos, seq, kappa_dihedral)

        _bond_refine_extras = {
            "grad_full_extra_kwargs": dict(_gf_kw),
            "grad_extra_fn": _grad_extra_opt,
            "variational_bound_prune": bool(variational_bound_prune),
            "variational_bound_prune_margin": float(variational_bound_prune_margin),
            "ensemble_translation_mix_alpha": float(ensemble_translation_mix_alpha),
            "ensemble_translation_step": float(ensemble_translation_step),
            "ensemble_decay_span": float(ensemble_decay_span),
            "ensemble_beta": float(ensemble_beta),
            "ensemble_s2_power": float(ensemble_s2_power),
            "ensemble_score_lambda": float(ensemble_score_lambda),
            "ensemble_inertial_dt": float(ensemble_inertial_dt),
            "ensemble_linear_damping": float(ensemble_linear_damping),
            "ensemble_linear_gain": float(ensemble_linear_gain),
            "ensemble_damping_mode": str(ensemble_damping_mode),
            "ensemble_barrier_decay": float(ensemble_barrier_decay),
            "ensemble_barrier_build": float(ensemble_barrier_build),
            "ensemble_barrier_relief": float(ensemble_barrier_relief),
            "ensemble_target_transition_only": bool(ensemble_target_transition_only),
            "ensemble_target_mix_alpha": float(ensemble_target_mix_alpha),
            "ensemble_target_mix_tolerance": float(ensemble_target_mix_tolerance),
            "ensemble_angular_mix_alpha": float(ensemble_angular_mix_alpha),
            "ensemble_angular_damping": float(ensemble_angular_damping),
            "ensemble_angular_gain": float(ensemble_angular_gain),
            "ensemble_em_refresh_on_horizon_crossing": bool(ensemble_em_refresh_on_horizon_crossing),
            "ensemble_em_refresh_on_horizon_leaving": bool(ensemble_em_refresh_on_horizon_leaving),
            "ensemble_em_refresh_horizon_ang": float(ensemble_em_refresh_horizon_ang),
            "ensemble_em_refresh_min_seq_sep": int(ensemble_em_refresh_min_seq_sep),
            "ensemble_em_refresh_on_large_disp": bool(ensemble_em_refresh_on_large_disp),
            "ensemble_em_refresh_large_disp_thresh": float(ensemble_em_refresh_large_disp_thresh),
            "ensemble_em_max_extra_directions": int(ensemble_em_max_extra_directions),
        }

        # Co-translational ribosome tunnel mode: cone + lip plane, fast-pass spaghetti, connection-triggered HKE
        if simulate_ribosome_tunnel:
            ptc_origin = np.zeros(3)
            axis = _tunnel._normalize_axis(tunnel_axis) if tunnel_axis is not None else _tunnel.DEFAULT_TUNNEL_AXIS.copy()
            grad_func = lambda pos, z: _grad_ca_with_lean(pos, z, include_horizon=True)
            _th_step = tunnel_thermal_step_size
            if _th_step is None:
                _th_step = 0.022 if quick else 0.028
            _th_seed = (
                tunnel_thermal_seed
                if tunnel_thermal_seed is not None
                else post_extrusion_discrete_seed
            )
            ca_min, info = _tunnel.co_translational_minimize(
                ca_init,
                z_ca,
                ptc_origin,
                axis,
                tunnel_length=tunnel_length,
                cone_half_angle_deg=cone_half_angle_deg,
                lip_plane_distance=lip_plane_distance,
                grad_func=grad_func,
                project_bonds=_project_bonds,
                n_bell=2,
                fast_pass_steps_per_connection=fast_pass_steps_per_connection,
                min_pass_iter_per_connection=min_pass_iter_per_connection,
                r_bond_min=2.5,
                r_bond_max=6.0,
                hke_above_tunnel_fraction=hke_above_tunnel_fraction,
                tunnel_thermal_gradient_steps=int(tunnel_thermal_gradient_steps),
                tunnel_thermal_temperature_k=float(refinement_temperature_k),
                tunnel_thermal_reference_temperature_k=float(tunnel_thermal_reference_temperature_k),
                tunnel_thermal_noise_fraction=float(tunnel_thermal_noise_fraction),
                tunnel_thermal_step_size=float(_th_step),
                tunnel_thermal_quick_cap=bool(quick),
                tunnel_thermal_seed=_th_seed,
                tunnel_free_terminus_steps=int(tunnel_free_terminus_steps),
                tunnel_free_terminus_window=int(tunnel_free_terminus_window),
                tunnel_handedness_bias_weight=float(tunnel_handedness_bias_weight),
                tunnel_handedness_target=float(tunnel_handedness_target),
                tunnel_handedness_sign=float(tunnel_handedness_sign),
                energy_func_ca=_e_ca_for_line_search,
                ensemble_translation_mix_alpha=float(ensemble_translation_mix_alpha),
                ensemble_translation_step=float(ensemble_translation_step),
                ensemble_decay_span=float(ensemble_decay_span),
                ensemble_beta=float(ensemble_beta),
                ensemble_s2_power=float(ensemble_s2_power),
                ensemble_score_lambda=float(ensemble_score_lambda),
                ensemble_inertial_dt=float(ensemble_inertial_dt),
                ensemble_linear_damping=float(ensemble_linear_damping),
                ensemble_linear_gain=float(ensemble_linear_gain),
                ensemble_damping_mode=str(ensemble_damping_mode),
                ensemble_barrier_decay=float(ensemble_barrier_decay),
                ensemble_barrier_build=float(ensemble_barrier_build),
                ensemble_barrier_relief=float(ensemble_barrier_relief),
                ensemble_target_transition_only=bool(ensemble_target_transition_only),
                ensemble_target_mix_alpha=float(ensemble_target_mix_alpha),
                ensemble_target_mix_tolerance=float(ensemble_target_mix_tolerance),
                ensemble_angular_mix_alpha=float(ensemble_angular_mix_alpha),
                ensemble_angular_damping=float(ensemble_angular_damping),
                ensemble_angular_gain=float(ensemble_angular_gain),
                ensemble_em_refresh_on_horizon_crossing=bool(ensemble_em_refresh_on_horizon_crossing),
                ensemble_em_refresh_on_horizon_leaving=bool(ensemble_em_refresh_on_horizon_leaving),
                ensemble_em_refresh_horizon_ang=float(ensemble_em_refresh_horizon_ang),
                ensemble_em_refresh_min_seq_sep=int(ensemble_em_refresh_min_seq_sep),
                ensemble_em_refresh_on_large_disp=bool(ensemble_em_refresh_on_large_disp),
                ensemble_em_refresh_large_disp_thresh=float(ensemble_em_refresh_large_disp_thresh),
                ensemble_em_max_extra_directions=int(ensemble_em_max_extra_directions),
            )
            current_for_dump["ca"] = ca_min.copy()
            # Post-extrusion: default EM 3D field + tree-torque (no grad_full HKE); optional legacy HKE loop.
            if post_extrusion_refine:
                mode = (post_extrusion_refine_mode or "em_treetorque").strip().lower()
                if mode not in ("none", "hke", "em_treetorque"):
                    mode = "em_treetorque"
                if mode == "none":
                    info = {
                        **info,
                        "message": "Co-translational tunnel (no post-extrusion refine)",
                        "post_extrusion_refine_mode": "none",
                    }
                elif mode == "hke":
                    total_refine_iter = 0
                    post_extrusion_max_disp = max(
                        float(post_extrusion_max_disp_floor),
                        0.5 * (n_res / 100.0),
                    )
                    post_extrusion_rounds = 0

                    def _update_dump(_step: int, pos: np.ndarray) -> None:
                        current_for_dump["ca"] = pos.copy()

                    while True:
                        ca_prev = ca_min.copy()
                        ca_min, info_refine = _minimize_bonds_fast(
                            ca_min,
                            z_ca,
                            max_iter=long_iter,
                            fast_horizon=fast_horizon,
                            collapse=True,
                            collapse_frac=0.5,
                            loose_r_max=8.0,
                            collapse_init_steps=init_steps,
                            k_rg_collapse=k_rg_collapse,
                            trajectory_log=traj_file,
                            on_step_callback=_update_dump,
                            fast_local_theta=fast_local_theta,
                            **_bond_refine_extras,
                        )
                        current_for_dump["ca"] = ca_min.copy()
                        total_refine_iter += info_refine.get("n_iter", 0)
                        max_disp = np.max(np.linalg.norm(ca_min - ca_prev, axis=1))
                        post_extrusion_rounds += 1
                        if max_disp <= post_extrusion_max_disp:
                            break
                        if post_extrusion_rounds >= post_extrusion_max_rounds:
                            break
                    info = {
                        **info_refine,
                        "message": "Co-translational tunnel + post-extrusion HKE refine",
                        "n_iter": info.get("n_iter", 0) + total_refine_iter,
                        "post_extrusion_rounds": post_extrusion_rounds,
                        "post_extrusion_max_disp_threshold": post_extrusion_max_disp,
                        "post_extrusion_refine_mode": "hke",
                    }
                else:
                    from .post_tunnel_em_treetorque import refine_ca_post_tunnel_em_treetorque

                    ca_min, em_tt_info = refine_ca_post_tunnel_em_treetorque(
                        ca_min,
                        seq,
                        quick=quick,
                        temperature=float(refinement_temperature_k),
                        em_max_steps=post_extrusion_em_max_steps,
                        treetorque_phases=int(post_extrusion_treetorque_phases),
                        treetorque_n_steps=int(post_extrusion_treetorque_n_steps),
                        discrete_use_metropolis=bool(post_extrusion_discrete_metropolis),
                        discrete_seed=post_extrusion_discrete_seed,
                        post_extrusion_anneal=bool(post_extrusion_anneal),
                        post_extrusion_anneal_schedule_k=post_extrusion_anneal_schedule_k,
                        langevin_steps=int(post_extrusion_langevin_steps),
                        langevin_noise_fraction=float(post_extrusion_langevin_noise_fraction),
                        grad_full_extra_kwargs=dict(_gf_kw),
                    )
                    current_for_dump["ca"] = ca_min.copy()
                    info = {
                        **info,
                        "message": em_tt_info.get("message", "post-extrusion EM + tree-torque"),
                        "n_iter": info.get("n_iter", 0),
                        "post_extrusion_refine_mode": "em_treetorque",
                        **{k: v for k, v in em_tt_info.items() if k != "message"},
                    }
                if bool(post_extrusion_osh_hqiv_native) and mode != "none":
                    from .osh_oracle_folding import minimize_ca_with_osh_oracle

                    ca_min, osh_info = minimize_ca_with_osh_oracle(
                        ca_min,
                        sequence=seq,
                        z_shells=np.asarray(z_ca, dtype=np.int32),
                        z_shell=int(z_ca.flat[0]) if z_ca.size else 6,
                        n_iter=int(post_extrusion_osh_n_iter),
                        step_size=float(post_extrusion_osh_step_size),
                        ansatz_depth=int(post_extrusion_osh_ansatz_depth),
                        gate_mix=float(post_extrusion_osh_gate_mix),
                        use_energy_reservoir=False,
                        use_hqiv_native_gate=True,
                        hqiv_reference_m=int(post_extrusion_osh_hqiv_reference_m),
                        energy_kwargs=dict(_gf_kw),
                    )
                    current_for_dump["ca"] = ca_min.copy()
                    info = {
                        **info,
                        "post_extrusion_osh_hqiv_native": True,
                        "osh_oracle_accepted_steps": int(osh_info.accepted_steps),
                        "osh_oracle_final_energy_ev": float(osh_info.final_energy_ev),
                        "osh_oracle_iterations_executed": int(osh_info.iterations_executed),
                    }
            elif traj_file is not None:
                write_trajectory_frame(traj_file, 0, ca_min)
        # Standard HKE path (unchanged): fast path for long chains or L-BFGS for short
        elif n_res > 50:
            def _update_dump_long(_step: int, pos: np.ndarray) -> None:
                current_for_dump["ca"] = pos.copy()
            staged = bool(variational_staged_opt and variational_pair_weight > 0.0)
            if staged:
                frac = float(np.clip(variational_stage_frac, 0.0, 1.0))
                it1 = max(1, int(long_iter * frac))
                it2 = max(1, int(long_iter - it1))
                extras1 = dict(_bond_refine_extras)
                gkw1 = dict(extras1.get("grad_full_extra_kwargs", {}))
                gkw1["variational_pair_weight"] = 0.0
                extras1["grad_full_extra_kwargs"] = gkw1
                ca_warm, info1 = _minimize_bonds_fast(
                    ca_init,
                    z_ca,
                    max_iter=it1,
                    fast_horizon=fast_horizon,
                    collapse=collapse,
                    collapse_frac=0.5,
                    collapse_init_steps=init_steps if collapse else 0,
                    k_rg_collapse=k_rg_collapse,
                    trajectory_log=traj_file,
                    on_step_callback=_update_dump_long if signal_dump_path else None,
                    fast_local_theta=fast_local_theta,
                    **extras1,
                )
                ca_min, info2 = _minimize_bonds_fast(
                    ca_warm,
                    z_ca,
                    max_iter=it2,
                    fast_horizon=fast_horizon,
                    collapse=False,
                    collapse_frac=0.5,
                    collapse_init_steps=0,
                    k_rg_collapse=k_rg_collapse,
                    trajectory_log=traj_file,
                    on_step_callback=_update_dump_long if signal_dump_path else None,
                    fast_local_theta=fast_local_theta,
                    **_bond_refine_extras,
                )
                info = dict(info2)
                info["message"] = "Two-stage variational refine (warmup + full)"
                info["n_iter"] = int(info1.get("n_iter", 0)) + int(info2.get("n_iter", 0))
                info["stage1_variational_weight"] = 0.0
                info["stage2_variational_weight"] = float(variational_pair_weight)
            else:
                ca_min, info = _minimize_bonds_fast(
                    ca_init,
                    z_ca,
                    max_iter=long_iter,
                    fast_horizon=fast_horizon,
                    collapse=collapse,
                    collapse_frac=0.5,
                    collapse_init_steps=init_steps if collapse else 0,
                    k_rg_collapse=k_rg_collapse,
                    trajectory_log=traj_file,
                    on_step_callback=_update_dump_long if signal_dump_path else None,
                    fast_local_theta=fast_local_theta,
                    **_bond_refine_extras,
                )
        else:
            def _traj_cb(step: int, pos: np.ndarray) -> None:
                write_trajectory_frame(traj_file, step, pos)
            staged = bool(variational_staged_opt and variational_pair_weight > 0.0)
            if staged:
                frac = float(np.clip(variational_stage_frac, 0.0, 1.0))
                it1 = max(1, int(max_iter * frac))
                it2 = max(1, int(max_iter - it1))
                e_kw1 = dict(_e_tot_bind_kw)
                g_kw1 = dict(_gf_kw)
                e_kw1["variational_pair_weight"] = 0.0
                g_kw1["variational_pair_weight"] = 0.0
                e1 = functools.partial(e_tot_ca_with_bonds, **e_kw1)
                ca_warm, info1 = minimize_e_tot_lbfgs(
                    ca_init,
                    z_ca,
                    max_iter=it1,
                    gtol=gtol,
                    energy_func=e1,
                    grad_func=lambda pos, z: grad_full(
                        pos,
                        z,
                        include_bonds=True,
                        include_horizon=not fast_horizon,
                        include_clash=True,
                        **g_kw1,
                    ),
                    project_bonds=True,
                    r_bond_min=2.5,
                    r_bond_max=6.0,
                    trajectory_callback=_traj_cb if traj_file else None,
                    ensemble_translation_mix_alpha=float(ensemble_translation_mix_alpha),
                    ensemble_translation_step=float(ensemble_translation_step),
                    ensemble_decay_span=float(ensemble_decay_span),
                    ensemble_beta=float(ensemble_beta),
                    ensemble_s2_power=float(ensemble_s2_power),
                    ensemble_score_lambda=float(ensemble_score_lambda),
                    ensemble_inertial_dt=float(ensemble_inertial_dt),
                    ensemble_linear_damping=float(ensemble_linear_damping),
                    ensemble_linear_gain=float(ensemble_linear_gain),
                    ensemble_damping_mode=str(ensemble_damping_mode),
                    ensemble_barrier_decay=float(ensemble_barrier_decay),
                    ensemble_barrier_build=float(ensemble_barrier_build),
                    ensemble_barrier_relief=float(ensemble_barrier_relief),
                    ensemble_target_transition_only=bool(ensemble_target_transition_only),
                    ensemble_target_mix_alpha=float(ensemble_target_mix_alpha),
                    ensemble_target_mix_tolerance=float(ensemble_target_mix_tolerance),
                    ensemble_angular_mix_alpha=float(ensemble_angular_mix_alpha),
                    ensemble_angular_damping=float(ensemble_angular_damping),
                    ensemble_angular_gain=float(ensemble_angular_gain),
                    ensemble_em_refresh_on_horizon_crossing=bool(ensemble_em_refresh_on_horizon_crossing),
                    ensemble_em_refresh_on_horizon_leaving=bool(ensemble_em_refresh_on_horizon_leaving),
                    ensemble_em_refresh_horizon_ang=float(ensemble_em_refresh_horizon_ang),
                    ensemble_em_refresh_min_seq_sep=int(ensemble_em_refresh_min_seq_sep),
                    ensemble_em_refresh_on_large_disp=bool(ensemble_em_refresh_on_large_disp),
                    ensemble_em_max_extra_directions=int(ensemble_em_max_extra_directions),
                    ensemble_large_translation_trigger=float(ensemble_em_refresh_large_disp_thresh),
                )
                ca_min, info2 = minimize_e_tot_lbfgs(
                    ca_warm,
                    z_ca,
                    max_iter=it2,
                    gtol=gtol,
                    energy_func=_e_ca_with_lean if kappa_dihedral > 0 else _e_ca_for_line_search,
                    grad_func=lambda pos, z: _grad_ca_with_lean(pos, z, include_horizon=not fast_horizon),
                    project_bonds=True,
                    r_bond_min=2.5,
                    r_bond_max=6.0,
                    trajectory_callback=_traj_cb if traj_file else None,
                    ensemble_translation_mix_alpha=float(ensemble_translation_mix_alpha),
                    ensemble_translation_step=float(ensemble_translation_step),
                    ensemble_decay_span=float(ensemble_decay_span),
                    ensemble_beta=float(ensemble_beta),
                    ensemble_s2_power=float(ensemble_s2_power),
                    ensemble_score_lambda=float(ensemble_score_lambda),
                    ensemble_inertial_dt=float(ensemble_inertial_dt),
                    ensemble_linear_damping=float(ensemble_linear_damping),
                    ensemble_linear_gain=float(ensemble_linear_gain),
                    ensemble_damping_mode=str(ensemble_damping_mode),
                    ensemble_barrier_decay=float(ensemble_barrier_decay),
                    ensemble_barrier_build=float(ensemble_barrier_build),
                    ensemble_barrier_relief=float(ensemble_barrier_relief),
                    ensemble_target_transition_only=bool(ensemble_target_transition_only),
                    ensemble_target_mix_alpha=float(ensemble_target_mix_alpha),
                    ensemble_target_mix_tolerance=float(ensemble_target_mix_tolerance),
                    ensemble_angular_mix_alpha=float(ensemble_angular_mix_alpha),
                    ensemble_angular_damping=float(ensemble_angular_damping),
                    ensemble_angular_gain=float(ensemble_angular_gain),
                    ensemble_em_refresh_on_horizon_crossing=bool(ensemble_em_refresh_on_horizon_crossing),
                    ensemble_em_refresh_on_horizon_leaving=bool(ensemble_em_refresh_on_horizon_leaving),
                    ensemble_em_refresh_horizon_ang=float(ensemble_em_refresh_horizon_ang),
                    ensemble_em_refresh_min_seq_sep=int(ensemble_em_refresh_min_seq_sep),
                    ensemble_em_refresh_on_large_disp=bool(ensemble_em_refresh_on_large_disp),
                    ensemble_em_max_extra_directions=int(ensemble_em_max_extra_directions),
                    ensemble_large_translation_trigger=float(ensemble_em_refresh_large_disp_thresh),
                )
                info = dict(info2)
                info["message"] = "Two-stage variational L-BFGS (warmup + full)"
                info["n_iter"] = int(info1.get("n_iter", 0)) + int(info2.get("n_iter", 0))
                info["stage1_variational_weight"] = 0.0
                info["stage2_variational_weight"] = float(variational_pair_weight)
            else:
                ca_min, info = minimize_e_tot_lbfgs(
                    ca_init,
                    z_ca,
                    max_iter=max_iter,
                    gtol=gtol,
                    energy_func=_e_ca_with_lean if kappa_dihedral > 0 else _e_ca_for_line_search,
                    grad_func=lambda pos, z: _grad_ca_with_lean(pos, z, include_horizon=not fast_horizon),
                    project_bonds=True,
                    r_bond_min=2.5,
                    r_bond_max=6.0,
                    trajectory_callback=_traj_cb if traj_file else None,
                    ensemble_translation_mix_alpha=float(ensemble_translation_mix_alpha),
                    ensemble_translation_step=float(ensemble_translation_step),
                    ensemble_decay_span=float(ensemble_decay_span),
                    ensemble_beta=float(ensemble_beta),
                    ensemble_s2_power=float(ensemble_s2_power),
                    ensemble_score_lambda=float(ensemble_score_lambda),
                    ensemble_inertial_dt=float(ensemble_inertial_dt),
                    ensemble_linear_damping=float(ensemble_linear_damping),
                    ensemble_linear_gain=float(ensemble_linear_gain),
                    ensemble_damping_mode=str(ensemble_damping_mode),
                    ensemble_barrier_decay=float(ensemble_barrier_decay),
                    ensemble_barrier_build=float(ensemble_barrier_build),
                    ensemble_barrier_relief=float(ensemble_barrier_relief),
                    ensemble_target_transition_only=bool(ensemble_target_transition_only),
                    ensemble_target_mix_alpha=float(ensemble_target_mix_alpha),
                    ensemble_target_mix_tolerance=float(ensemble_target_mix_tolerance),
                    ensemble_angular_mix_alpha=float(ensemble_angular_mix_alpha),
                    ensemble_angular_damping=float(ensemble_angular_damping),
                    ensemble_angular_gain=float(ensemble_angular_gain),
                    ensemble_em_refresh_on_horizon_crossing=bool(ensemble_em_refresh_on_horizon_crossing),
                    ensemble_em_refresh_on_horizon_leaving=bool(ensemble_em_refresh_on_horizon_leaving),
                    ensemble_em_refresh_horizon_ang=float(ensemble_em_refresh_horizon_ang),
                    ensemble_em_refresh_min_seq_sep=int(ensemble_em_refresh_min_seq_sep),
                    ensemble_em_refresh_on_large_disp=bool(ensemble_em_refresh_on_large_disp),
                    ensemble_em_max_extra_directions=int(ensemble_em_max_extra_directions),
                    ensemble_large_translation_trigger=float(ensemble_em_refresh_large_disp_thresh),
                )
    finally:
        if traj_file is not None:
            traj_file.close()
    backbone_atoms = _place_full_backbone(ca_min, seq)
    pos_bb, z_bb = _full_backbone_positions_and_z(backbone_atoms)
    E_ca_final = float(e_tot(ca_min, z_ca))
    E_backbone_final = float(e_tot(pos_bb, z_bb))
    if include_sidechains:
        backbone_atoms = _add_cb(backbone_atoms, seq)
        if side_chain_pack:
            backbone_atoms = _side_chain_pack_light(backbone_atoms, seq)

    ligand_list: Optional[List[LigandAgent]] = None
    if include_ligands and ligands:
        pos_bb, z_bb = _full_backbone_positions_and_z(backbone_atoms)
        protein_com = np.mean(pos_bb, axis=0)
        for lig in ligands:
            t, euler = lig.get_6dof()
            lig.set_6dof(protein_com.copy(), euler)
        ligand_list = list(ligands)
        _refine_ligands_6dof(
            pos_bb,
            z_bb,
            ligand_list,
            max_steps=int(ligand_refine_steps),
            step_t=float(ligand_step_t),
            step_ang=float(ligand_step_ang),
            grad_full_kwargs=dict(_gf_kw),
            ligand_refinement_mode=str(ligand_refinement_mode),
            qc_soft_clash_sigma=float(qc_soft_clash_sigma),
            qc_clash_weight=float(qc_clash_weight),
        )

    out = {
        "ca_min": ca_min,
        "backbone_atoms": backbone_atoms,
        "E_ca_final": E_ca_final,
        "E_backbone_final": E_backbone_final,
        "info": info,
        "n_res": n_res,
        "sequence": seq,
        "include_sidechains": include_sidechains,
    }
    if ligand_list is not None:
        out["ligands"] = ligand_list
    return out


def pack_sidechains(
    result: Dict[str, object],
    r_clash: float = 2.0,
    lbfgs_steps: int = 5,
) -> Dict[str, object]:
    """
    Run side-chain rotamer packing on an existing minimize_full_chain result.

    Uses side_chain_chi_preferences() for χ1: grid search (pref ±120°, ±60°, 0°) then
    L-BFGS on χ1 to minimize clash energy. Pushes lDDT up and all-atom RMSD down.
    Requires result["include_sidechains"]; otherwise returns result unchanged.
    """
    if not result.get("include_sidechains") or not result.get("backbone_atoms"):
        return result
    seq = result["sequence"]
    packed = _side_chain_pack_light(
        list(result["backbone_atoms"]),
        seq,
        r_clash=r_clash,
        lbfgs_steps=lbfgs_steps,
    )
    out = dict(result)
    out["backbone_atoms"] = packed
    return out


def _rotate_vector_around_axis(v: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    """Rotate vector v around axis by deg (degrees). Axis need not be unit."""
    axis = np.asarray(axis, dtype=float)
    nrm = np.linalg.norm(axis)
    if nrm < 1e-12:
        return v.copy()
    axis = axis / nrm
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return v * c + np.cross(axis, v) * s + axis * (np.dot(axis, v) * (1 - c))


def _side_chain_pack_light(
    backbone_atoms: List[Tuple[str, np.ndarray]],
    sequence: str,
    r_clash: float = 2.0,
    lbfgs_steps: int = 5,
) -> List[Tuple[str, np.ndarray]]:
    """
    Side-chain packing: (1) grid search on χ1 using chi_angles_for_residue preferences;
    (2) 3–5 L-BFGS steps on χ1 vector for fine-tuning. Improves lDDT / surface packing.
    """
    out = list(backbone_atoms)
    all_xyz = np.array([xyz for _, xyz in out])
    n_res = len(sequence)
    bonds = backbone_bond_lengths()
    r_ca_cb = bonds["Calpha_Cbeta"]
    chi_prefs = side_chain_chi_preferences()
    idx = 0
    cb_res_indices: List[int] = []
    cb_atom_indices: List[int] = []
    for res_i in range(n_res):
        aa = sequence[res_i]
        if aa == "G":
            idx += 4
            continue
        i_n, i_ca, i_cb = idx, idx + 1, idx + 2
        idx += 5
        cb_res_indices.append(res_i)
        cb_atom_indices.append(i_cb)
        n_xyz = out[i_n][1]
        ca_xyz = out[i_ca][1]
        c_xyz = out[i_cb + 2][1]
        cb_xyz = out[i_cb][1]
        chi1_pref = chi_prefs.get(AA_1to3.get(aa, "ALA"), {}).get("chi1_deg")
        if chi1_pref is None:
            chi1_pref = 60.0
        grid = [chi1_pref + d for d in (-120.0, -60.0, 0.0, 60.0, 120.0)]
        v0 = _default_cb_vector(n_xyz, ca_xyz, c_xyz, r_ca_cb)
        axis = ca_xyz - n_xyz
        best_cb = cb_xyz
        tree = _cKDTree(all_xyz) if _HAS_SCIPY_SPATIAL else None
        best_count = _count_clashes(cb_xyz, all_xyz, exclude=(i_n, i_ca, i_cb), r_clash=r_clash, tree=tree)
        for chi1 in grid:
            v_rot = _rotate_vector_around_axis(v0, axis, chi1)
            cb_new = ca_xyz + v_rot
            count = _count_clashes(cb_new, all_xyz, exclude=(i_n, i_ca, i_cb), r_clash=r_clash, tree=tree)
            if count < best_count:
                best_count = count
                best_cb = cb_new
        out[i_cb] = ("CB", best_cb)
        all_xyz[i_cb] = best_cb
    if lbfgs_steps > 0 and cb_res_indices:
        chi_vec = _pack_lbfgs(out, sequence, cb_res_indices, cb_atom_indices, r_ca_cb, r_clash, lbfgs_steps)
        out = _apply_chi_vec(out, sequence, chi_vec, cb_res_indices, cb_atom_indices, r_ca_cb)
    return out


def _default_cb_vector(n_xyz: np.ndarray, ca_xyz: np.ndarray, c_xyz: np.ndarray, r_ca_cb: float) -> np.ndarray:
    """Default CA→Cβ vector (trans from N–CA–C plane)."""
    v1 = n_xyz - ca_xyz
    v2 = c_xyz - ca_xyz
    direction = np.cross(v1, v2)
    nrm = np.linalg.norm(direction)
    if nrm < 1e-9:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = direction / nrm
    return r_ca_cb * direction


def _apply_chi_vec(
    atoms: List[Tuple[str, np.ndarray]],
    sequence: str,
    chi_vec: np.ndarray,
    cb_res_indices: List[int],
    cb_atom_indices: List[int],
    r_ca_cb: float,
) -> List[Tuple[str, np.ndarray]]:
    """Apply chi1 vector to update Cβ positions."""
    out = list(atoms)
    for k, res_i in enumerate(cb_res_indices):
        i_cb = cb_atom_indices[k]
        i_n = i_cb - 2
        i_ca = i_cb - 1
        i_c = i_cb + 2
        n_xyz = out[i_n][1]
        ca_xyz = out[i_ca][1]
        c_xyz = out[i_c][1]
        v0 = _default_cb_vector(n_xyz, ca_xyz, c_xyz, r_ca_cb)
        axis = ca_xyz - n_xyz
        v_rot = _rotate_vector_around_axis(v0, axis, float(chi_vec[k]))
        out[i_cb] = ("CB", ca_xyz + v_rot)
    return out


def _pack_lbfgs(
    atoms: List[Tuple[str, np.ndarray]],
    sequence: str,
    cb_res_indices: List[int],
    cb_atom_indices: List[int],
    r_ca_cb: float,
    r_clash: float,
    max_iter: int,
) -> np.ndarray:
    """3–5 L-BFGS steps on χ1 vector to minimize clash energy."""
    n_chi = len(cb_res_indices)
    chi_vec = np.zeros(n_chi)
    for k, res_i in enumerate(cb_res_indices):
        i_cb = cb_atom_indices[k]
        i_n, i_ca, i_c = i_cb - 2, i_cb - 1, i_cb + 2
        n_xyz = atoms[i_n][1]
        ca_xyz = atoms[i_ca][1]
        c_xyz = atoms[i_c][1]
        cb_xyz = atoms[i_cb][1]
        v0 = _default_cb_vector(n_xyz, ca_xyz, c_xyz, r_ca_cb)
        axis = ca_xyz - n_xyz
        v_cb = cb_xyz - ca_xyz
        chi_vec[k] = _chi1_from_vectors(v0, v_cb, axis)
    k_clash = 500.0
    s_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for it in range(max_iter):
        e, grad = _chi_clash_and_grad(atoms, sequence, chi_vec, cb_res_indices, cb_atom_indices, r_ca_cb, r_clash, k_clash)
        g_norm = np.linalg.norm(grad)
        if g_norm < 1e-4:
            break
        if len(s_list) >= 5:
            s_list.pop(0)
            y_list.pop(0)
        if len(s_list) == 0:
            direction = -grad
        else:
            direction = _lbfgs_chi(grad, s_list, y_list)
        step = 1.0
        e_curr = e
        for _ in range(20):
            chi_new = chi_vec + step * direction
            e_new, _ = _chi_clash_and_grad(
                atoms, sequence, chi_new, cb_res_indices, cb_atom_indices, r_ca_cb, r_clash, k_clash,
            )
            if e_new <= e_curr + 1e-4 * step * np.dot(grad, direction):
                break
            step *= 0.5
        _, grad_new = _chi_clash_and_grad(
            atoms, sequence, chi_new, cb_res_indices, cb_atom_indices, r_ca_cb, r_clash, k_clash,
        )
        s_list.append(chi_new - chi_vec)
        y_list.append(grad_new - grad)
        chi_vec = chi_new
    return chi_vec


def _lbfgs_chi(grad: np.ndarray, s_list: list, y_list: list, m: int = 5) -> np.ndarray:
    """L-BFGS two-loop for chi vector."""
    q = -grad.copy()
    n_vec = len(s_list)
    if n_vec == 0:
        return -grad
    alpha_list = []
    for i in range(n_vec - 1, -1, -1):
        rho = 1.0 / (np.dot(y_list[i], s_list[i]) + 1e-14)
        alpha_list.append(rho * np.dot(s_list[i], q))
        q = q - alpha_list[-1] * y_list[i]
    gamma = np.dot(y_list[-1], s_list[-1]) / (np.dot(y_list[-1], y_list[-1]) + 1e-14)
    r = gamma * q
    for i in range(n_vec):
        rho = 1.0 / (np.dot(y_list[i], s_list[i]) + 1e-14)
        beta = rho * np.dot(y_list[i], r)
        r = r + s_list[i] * (alpha_list[n_vec - 1 - i] - beta)
    return r


def _chi1_from_vectors(v0: np.ndarray, v_cb: np.ndarray, axis: np.ndarray) -> float:
    """Recover chi1 (deg) from v0 and current v_cb = CB - CA. Axis = CA - N."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    v0_n = v0 / (np.linalg.norm(v0) + 1e-12)
    v_cb_n = v_cb / (np.linalg.norm(v_cb) + 1e-12)
    dot = np.clip(np.dot(v0_n, v_cb_n), -1, 1)
    angle_rad = np.arccos(dot)
    cross = np.cross(v0_n, v_cb_n)
    sign = 1.0 if np.dot(axis, cross) >= 0 else -1.0
    return float(np.rad2deg(sign * angle_rad))


def _chi_clash_and_grad(
    atoms: List[Tuple[str, np.ndarray]],
    sequence: str,
    chi_vec: np.ndarray,
    cb_res_indices: List[int],
    cb_atom_indices: List[int],
    r_ca_cb: float,
    r_clash: float,
    k_clash: float,
) -> Tuple[float, np.ndarray]:
    """Clash energy and analytical gradient w.r.t. chi_vec (degrees). Uses cKDTree when available for O(n log n)."""
    atoms_applied = _apply_chi_vec(atoms, sequence, chi_vec, cb_res_indices, cb_atom_indices, r_ca_cb)
    all_xyz = np.array([xyz for _, xyz in atoms_applied])
    n_chi = len(cb_res_indices)
    e = 0.0
    grad = np.zeros(n_chi)
    deg2rad = np.pi / 180.0
    tree = _cKDTree(all_xyz) if _HAS_SCIPY_SPATIAL else None
    for k, res_i in enumerate(cb_res_indices):
        i_cb = cb_atom_indices[k]
        i_n, i_ca, i_c = i_cb - 2, i_cb - 1, i_cb + 2
        n_xyz = atoms[i_n][1]
        ca_xyz = atoms[i_ca][1]
        c_xyz = atoms[i_c][1]
        v0 = _default_cb_vector(n_xyz, ca_xyz, c_xyz, r_ca_cb)
        axis = ca_xyz - n_xyz
        axis_n = axis / (np.linalg.norm(axis) + 1e-12)
        chi_rad = chi_vec[k] * deg2rad
        cb_xyz = all_xyz[i_cb]
        dpos_dchi = _drot_dchi(v0, axis_n, chi_rad) * deg2rad
        if tree is not None:
            neighbors = tree.query_ball_point(cb_xyz, r_clash)
        else:
            neighbors = range(len(all_xyz))
        for j in neighbors:
            if j in (i_n, i_ca, i_cb):
                continue
            d = all_xyz[j] - cb_xyz
            r = np.linalg.norm(d)
            if r < 1e-9:
                continue
            if r < r_clash:
                e += k_clash * (r_clash - r) ** 2
                u = d / r
                g = -2 * k_clash * (r_clash - r)
                grad[k] += g * np.dot(u, dpos_dchi)
    return e, grad


def _drot_dchi(v: np.ndarray, axis: np.ndarray, chi_rad: float) -> np.ndarray:
    """d/d(chi_rad) of R(chi)*v for rotation around axis."""
    c, s = np.cos(chi_rad), np.sin(chi_rad)
    return -v * s + np.cross(axis, v) * c + axis * np.dot(axis, v) * s


def _count_clashes(
    xyz: np.ndarray,
    all_xyz: np.ndarray,
    exclude: Tuple[int, ...],
    r_clash: float = 2.0,
    tree: Any = None,
) -> int:
    """Number of atoms in all_xyz (excluding indices in exclude) within r_clash of xyz. Optional tree from cKDTree(all_xyz) for O(log n) per query."""
    if _HAS_SCIPY_SPATIAL and tree is not None:
        idx = tree.query_ball_point(xyz, r_clash)
        return sum(1 for i in idx if i not in exclude)
    d = np.linalg.norm(all_xyz - xyz, axis=1)
    mask = np.ones(len(d), dtype=bool)
    for i in exclude:
        mask[i] = False
    return int(np.sum(mask & (d < r_clash)))


def _add_cb(
    backbone_atoms: List[Tuple[str, np.ndarray]],
    sequence: str,
) -> List[Tuple[str, np.ndarray]]:
    """
    Add Cβ to backbone atom list (except Gly). Uses N–CA–C plane and trans Cβ.
    Inserts Cβ after CA for each residue. Returns new list (order: N, CA, CB?, C, O per res).
    """
    bonds = backbone_bond_lengths()
    r_ca_cb = bonds["Calpha_Cbeta"]
    n_res = len(sequence)
    out = []
    atoms_per_res = 4
    for i in range(n_res):
        aa = sequence[i]
        n_xyz = backbone_atoms[i * atoms_per_res + 0][1]
        ca_xyz = backbone_atoms[i * atoms_per_res + 1][1]
        c_xyz = backbone_atoms[i * atoms_per_res + 2][1]
        o_xyz = backbone_atoms[i * atoms_per_res + 3][1]
        out.append(("N", n_xyz))
        out.append(("CA", ca_xyz))
        if aa != "G":
            # Cβ in trans: direction from (N-CA) × (C-CA), normalized
            v1 = n_xyz - ca_xyz
            v2 = c_xyz - ca_xyz
            direction = np.cross(v1, v2)
            nrm = np.linalg.norm(direction)
            if nrm < 1e-9:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                direction = direction / nrm
            cb_xyz = ca_xyz + r_ca_cb * direction
            out.append(("CB", cb_xyz))
        out.append(("C", c_xyz))
        out.append(("O", o_xyz))
    return out


def full_chain_to_pdb(
    result: Dict[str, object],
    chain_id: str = "A",
    ligands: Optional[List[LigandAgent]] = None,
    ligand_chain_id: Optional[str] = None,
    *,
    model_step: Optional[int] = None,
) -> str:
    """Format minimizer result as PDB string (REMARK HQIV-QConSi-*-step + MODEL ... ENDMDL END). Protein ATOM then ligand HETATM if provided."""
    backbone_atoms = result["backbone_atoms"]
    sequence = result["sequence"]
    include_sidechains = result.get("include_sidechains", False)
    if ligands is None:
        ligands = result.get("ligands") or []
    if not backbone_atoms and not ligands:
        return hqiv_qconsi_empty_pdb_block(step=model_step)
    het_chain = ligand_chain_id if ligand_chain_id is not None else chain_id
    lines = list(hqiv_qconsi_model_lines(step=model_step))
    atom_id = 1
    n_res = result["n_res"]
    idx = 0
    for res_id in range(1, n_res + 1):
        res_1 = sequence[res_id - 1]
        res_3 = AA_1to3.get(res_1, "UNK")
        n_atoms_this = (5 if res_1 != "G" else 4) if include_sidechains else 4
        for _ in range(n_atoms_this):
            name, xyz = backbone_atoms[idx]
            lines.append(
                f"ATOM  {atom_id:5d}  {name:2s}  {res_3:3s} {chain_id}{res_id:4d}    "
                f"{float(xyz[0]):8.3f}{float(xyz[1]):8.3f}{float(xyz[2]):8.3f}  1.00  0.00           "
            )
            atom_id += 1
            idx += 1
    def _pdb_coord(x: float) -> str:
        s = f"{float(x):8.3f}"
        return s[-8:] if len(s) > 8 else s

    for res_id_het, lig in enumerate(ligands, start=1):
        res_3 = (lig.res_name or "LIG")[:3]
        for a, (name, xyz) in enumerate(zip(lig.atom_names, lig.get_world_positions())):
            elem = (lig.elements[a] if a < len(lig.elements) else "C")[:2].rjust(2)
            line = (
                f"HETATM{atom_id:5d}  {name:2s}  {res_3:3s} {het_chain}{res_id_het:4d}    "
                f"{_pdb_coord(xyz[0])}{_pdb_coord(xyz[1])}{_pdb_coord(xyz[2])}  1.00  0.00      {elem:>2s}"
            )
            lines.append(line[:80] if len(line) > 80 else line)
            atom_id += 1
    lines.append("ENDMDL")
    lines.append("END")
    return "\n".join(lines)


def full_chain_to_pdb_complex(
    backbone_a: List[Tuple[str, np.ndarray]],
    backbone_b: List[Tuple[str, np.ndarray]],
    sequence_a: str,
    sequence_b: str,
    chain_id_a: str = "A",
    chain_id_b: str = "B",
    *,
    model_step: Optional[int] = None,
) -> str:
    """Format two backbones as one PDB (HQIV-QConSi header) with chain A then chain B."""
    lines = list(hqiv_qconsi_model_lines(step=model_step))
    atom_id = 1
    for chain_idx, (backbone, sequence, chain_id) in enumerate([
        (backbone_a, sequence_a, chain_id_a),
        (backbone_b, sequence_b, chain_id_b),
    ]):
        n_res = len(sequence)
        idx = 0
        for res_id in range(1, n_res + 1):
            res_1 = sequence[res_id - 1]
            res_3 = AA_1to3.get(res_1, "UNK")
            n_atoms_this = 4  # N, CA, C, O
            for _ in range(n_atoms_this):
                name, xyz = backbone[idx]
                lines.append(
                    f"ATOM  {atom_id:5d}  {name:2s}  {res_3:3s} {chain_id}{res_id:4d}    "
                    f"{float(xyz[0]):8.3f}{float(xyz[1]):8.3f}{float(xyz[2]):8.3f}  1.00  0.00           "
                )
                atom_id += 1
                idx += 1
    lines.append("ENDMDL")
    lines.append("END")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import sys
    from horizon_physics.proteins.casp_submission import _parse_fasta

    parser = argparse.ArgumentParser(description="HQIV full-chain minimizer: FASTA → minimized PDB")
    parser.add_argument("fasta", nargs="?", help="FASTA file or string (default: stdin)")
    parser.add_argument("-o", "--output", help="Output PDB path (default: stdout)")
    parser.add_argument("--fast", action="store_true", help="Use bonds-only gradient (nearest-neighbor; faster, for debugging)")
    parser.add_argument("--no-sidechain-pack", action="store_true", help="Skip Cβ rotamer search after backbone")
    parser.add_argument("--no-sidechains", action="store_true", help="Backbone only (no Cβ)")
    args = parser.parse_args()

    if args.fasta:
        with open(args.fasta) as f:
            fasta = f.read()
    else:
        fasta = sys.stdin.read()
    seq = _parse_fasta(fasta)
    if not seq:
        sys.stderr.write("No sequence found in FASTA\n")
        sys.exit(1)
    result = minimize_full_chain(
        seq,
        max_iter=500,
        include_sidechains=not args.no_sidechains,
        fast_horizon=args.fast,
        side_chain_pack=not args.no_sidechain_pack and not args.no_sidechains,
    )
    pdb = full_chain_to_pdb(result)
    if args.output:
        with open(args.output, "w") as f:
            f.write(pdb)
    else:
        print(pdb)
