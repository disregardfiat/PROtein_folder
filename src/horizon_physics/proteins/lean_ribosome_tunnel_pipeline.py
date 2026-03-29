"""
HQIV Lean–aligned co-translational folding pipeline (PROtien).

Uses the existing ribosome tunnel (`simulate_ribosome_tunnel=True` in `minimize_full_chain`)
with experimentally grounded bulk water (ε_r(T)), physiological pH bookkeeping, and
optional Lean `foldEnergyWithDihedral` correction κ(1−cos Δθ) on φ/ψ.

Post-extrusion (after the chain leaves the cone/plane) defaults to **3D EM-field relaxation**
plus **discrete tree-torque** — not the legacy L-BFGS HKE loop (see ``post_extrusion_refine_mode``).
Connection-triggered passes default to 0/0 gradient steps so extrusion stays fast; raise
``fast_pass_steps_per_connection`` / ``min_pass_iter_per_connection`` to restore in-tunnel HKE.

Lean references: `Hqiv/Physics/HQIVMolecules.lean` (TorqueTree, foldEnergy, foldEnergyWithDihedral),
`Hqiv/Physics/HQIVAtoms.lean` (waterDielectricValley, pH_charge_flip_effect, waterBondAngleDeg),
`Hqiv/Physics/HQIVCollectiveModes.lean` (collective multipole torque / kink budget; Python: ``collective_modes_scalars``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .full_protein_minimizer import full_chain_to_pdb, minimize_full_chain
from .ligands import LigandAgent, ligand_summary, parse_ligands, parse_ligands_from_file
from .hqiv_lean_folding import (
    PHYSIOLOGICAL_PH,
    WATER_BOND_ANGLE_DEG,
    em_scale_aqueous,
    epsilon_r_water,
    ph_em_scale_delta,
)


@dataclass
class LeanTunnelFoldResult:
    sequence: str
    n_res: int
    temperature_k: float
    ph: float
    epsilon_r: float
    em_scale: float
    water_bond_angle_deg: float
    kappa_dihedral: float
    pdb: str
    raw_result: Dict[str, Any]
    include_ligands: bool = False
    ligand_note: str = ""


def fold_lean_ribosome_tunnel(
    sequence: str,
    *,
    temperature_k: float = 310.0,
    ph: float = PHYSIOLOGICAL_PH,
    epsilon_r: Optional[float] = None,
    apply_ph_screening_modulation: bool = True,
    kappa_dihedral: float = 0.01,
    include_sidechains: bool = False,
    simulate_ribosome_tunnel: bool = True,
    post_extrusion_refine: bool = True,
    post_extrusion_refine_mode: str = "em_treetorque",
    post_extrusion_em_max_steps: Optional[int] = None,
    post_extrusion_treetorque_phases: int = 8,
    post_extrusion_treetorque_n_steps: int = 200,
    post_extrusion_discrete_metropolis: bool = False,
    post_extrusion_discrete_seed: Optional[int] = None,
    post_extrusion_anneal: bool = False,
    post_extrusion_anneal_schedule_k: Optional[Tuple[float, ...]] = None,
    post_extrusion_langevin_steps: int = 0,
    post_extrusion_langevin_noise_fraction: float = 0.2,
    tunnel_length: float = 25.0,
    cone_half_angle_deg: float = 12.0,
    lip_plane_distance: float = 0.0,
    tunnel_axis: Optional[np.ndarray] = None,
    hke_above_tunnel_fraction: float = 0.5,
    quick: bool = False,
    post_extrusion_max_disp_floor: float = 0.25,
    post_extrusion_max_rounds: int = 32,
    post_extrusion_osh_hqiv_native: bool = False,
    post_extrusion_osh_n_iter: int = 120,
    post_extrusion_osh_step_size: float = 0.03,
    post_extrusion_osh_ansatz_depth: int = 2,
    post_extrusion_osh_gate_mix: float = 0.55,
    post_extrusion_osh_hqiv_reference_m: int = 4,
    fast_pass_steps_per_connection: int = 0,
    min_pass_iter_per_connection: int = 0,
    tunnel_thermal_gradient_steps: int = 0,
    tunnel_thermal_noise_fraction: float = 0.2,
    tunnel_thermal_step_size: Optional[float] = None,
    tunnel_thermal_reference_temperature_k: float = 310.0,
    tunnel_thermal_seed: Optional[int] = None,
    hbond_weight: float = 0.0,
    hbond_shell_m: int = 3,
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
    include_ligands: bool = False,
    ligands: Optional[List[LigandAgent]] = None,
    ligand_str: Optional[str] = None,
    ligand_file: Optional[str] = None,
    ligand_refine_steps: Optional[int] = None,
    ligand_step_t: Optional[float] = None,
    ligand_step_ang: Optional[float] = None,
    ligand_refinement_mode: str = "lean_qc",
    qc_soft_clash_sigma: float = 3.0,
    qc_clash_weight: float = 1.0,
    ligand_chain_id: Optional[str] = None,
) -> LeanTunnelFoldResult:
    """
    Fold a sequence with the ribosome tunnel and Lean-aligned solvent / dihedral physics.

    - **Water (bulk):** ε_r from `epsilon_r_water(temperature_k)` unless `epsilon_r` is set explicitly.
    - **EM screening:** `em_scale = (1/ε_r) * ph_em_scale_delta(ph)` when modulation is on (mild pH effect).
    - **Dihedral (Lean native fold):** κ(1−cos Δφ) + κ(1−cos Δψ) toward the HQIV α basin; set `kappa_dihedral=0` to disable.
    - **Anchor constant:** `water_bond_angle_deg` (= 104.5°) is recorded for parity with Lean `waterBondAngleDeg` (not used in Cα minimizer).
    - **Post-extrusion:** Default ``post_extrusion_refine_mode="em_treetorque"`` (3D EM relax + discrete tree-torque). Set ``post_extrusion_discrete_metropolis=True`` for true Metropolis at ``temperature_k``; ``post_extrusion_anneal=True`` for a short multi-temperature Metropolis cool-down (replaces the single discrete pass); optional ``post_extrusion_anneal_schedule_k`` (K, high→low). ``post_extrusion_langevin_steps`` adds kT-noised ``grad_full`` steps after the discrete/anneal phase.
    - **In-tunnel passes:** ``fast_pass_steps_per_connection`` / ``min_pass_iter_per_connection`` default to ``0`` (extrusion geometry only). Increase to restore gradient / L-BFGS during growth.
    - **In-tunnel thermal:** ``tunnel_thermal_gradient_steps`` > 0 adds kT-noised masked-gradient steps after each binary-tree segment (same cone/lip/HKE-fraction masking as L-BFGS); thermal scale is ``temperature_k`` (via ``refinement_temperature_k``). Cheap while segments are short.
    - **Long-range proxy (Lean ``HQIVLongRange``):** optional ``hbond_weight`` / ``hbond_shell_m`` → ``grad_full`` additive ``hBondProxy`` term (costly).
    - **Performance flags:** ``fast_local_theta`` (batched Θ_i in ``e_tot_ca_with_bonds``); ``horizon_neighbor_cutoff`` (tighter horizon neighbor list in Å).
    - **Collective kink (``HQIVCollectiveModes``):** ``collective_kink_weight`` / ``collective_kink_m`` / optional ``theta_ref`` / ``collective_kink_use_ss_mask``.
    - **Variational score terms (``VariationalScoreTerms``):** ``variational_pair_weight`` and related
      pair-shape parameters; optional staged optimization (``variational_staged_opt``) and Lean-style
      lower-bound gradient gating (``variational_bound_prune``).
    - **Inertial twist penalty:** ``inertial_twist_weight`` to penalize central bends more than
      terminal bends (center-heavy chain inertia proxy).
    - **Ligands:** pass ``ligands``, or ``ligand_str`` (PDB block / SMILES lines), or ``ligand_file``; set ``include_ligands=True``. Rigid-body refinement uses the same ``em_scale`` as the protein; optional ``ligand_chain_id`` (e.g. ``\"L\"``) for HETATM chain column (default: same as protein chain ``A``).
    """
    seq = "".join(c for c in sequence.strip().upper() if c.isalpha())
    if not seq:
        raise ValueError("fold_lean_ribosome_tunnel: empty sequence.")

    er = float(epsilon_r) if epsilon_r is not None else epsilon_r_water(temperature_k)
    em_scale = em_scale_aqueous(temperature_k, epsilon_r=er)
    if apply_ph_screening_modulation:
        em_scale *= ph_em_scale_delta(ph)

    lig_list: List[LigandAgent] = []
    if ligands:
        lig_list = list(ligands)
    elif ligand_file:
        lig_list = parse_ligands_from_file(ligand_file)
    elif ligand_str:
        lig_list = parse_ligands(ligand_str)

    use_lig = bool(include_ligands and lig_list)
    lig_kw: Dict[str, Any] = {}
    if ligand_refine_steps is not None:
        lig_kw["ligand_refine_steps"] = int(ligand_refine_steps)
    if ligand_step_t is not None:
        lig_kw["ligand_step_t"] = float(ligand_step_t)
    if ligand_step_ang is not None:
        lig_kw["ligand_step_ang"] = float(ligand_step_ang)
    lig_kw["ligand_refinement_mode"] = str(ligand_refinement_mode)
    lig_kw["qc_soft_clash_sigma"] = float(qc_soft_clash_sigma)
    lig_kw["qc_clash_weight"] = float(qc_clash_weight)

    raw = minimize_full_chain(
        seq,
        include_sidechains=include_sidechains,
        simulate_ribosome_tunnel=simulate_ribosome_tunnel,
        post_extrusion_refine=post_extrusion_refine,
        post_extrusion_refine_mode=post_extrusion_refine_mode,
        post_extrusion_em_max_steps=post_extrusion_em_max_steps,
        post_extrusion_treetorque_phases=post_extrusion_treetorque_phases,
        post_extrusion_treetorque_n_steps=post_extrusion_treetorque_n_steps,
        post_extrusion_discrete_metropolis=post_extrusion_discrete_metropolis,
        post_extrusion_discrete_seed=post_extrusion_discrete_seed,
        post_extrusion_anneal=post_extrusion_anneal,
        post_extrusion_anneal_schedule_k=post_extrusion_anneal_schedule_k,
        post_extrusion_langevin_steps=post_extrusion_langevin_steps,
        post_extrusion_langevin_noise_fraction=post_extrusion_langevin_noise_fraction,
        refinement_temperature_k=float(temperature_k),
        tunnel_length=tunnel_length,
        cone_half_angle_deg=cone_half_angle_deg,
        lip_plane_distance=lip_plane_distance,
        tunnel_axis=tunnel_axis,
        hke_above_tunnel_fraction=hke_above_tunnel_fraction,
        em_scale=em_scale,
        kappa_dihedral=kappa_dihedral,
        quick=quick,
        post_extrusion_max_disp_floor=post_extrusion_max_disp_floor,
        post_extrusion_max_rounds=post_extrusion_max_rounds,
        post_extrusion_osh_hqiv_native=post_extrusion_osh_hqiv_native,
        post_extrusion_osh_n_iter=post_extrusion_osh_n_iter,
        post_extrusion_osh_step_size=post_extrusion_osh_step_size,
        post_extrusion_osh_ansatz_depth=post_extrusion_osh_ansatz_depth,
        post_extrusion_osh_gate_mix=post_extrusion_osh_gate_mix,
        post_extrusion_osh_hqiv_reference_m=post_extrusion_osh_hqiv_reference_m,
        fast_pass_steps_per_connection=fast_pass_steps_per_connection,
        min_pass_iter_per_connection=min_pass_iter_per_connection,
        tunnel_thermal_gradient_steps=tunnel_thermal_gradient_steps,
        tunnel_thermal_noise_fraction=tunnel_thermal_noise_fraction,
        tunnel_thermal_step_size=tunnel_thermal_step_size,
        tunnel_thermal_reference_temperature_k=tunnel_thermal_reference_temperature_k,
        tunnel_thermal_seed=tunnel_thermal_seed,
        hbond_weight=hbond_weight,
        hbond_shell_m=hbond_shell_m,
        fast_local_theta=fast_local_theta,
        horizon_neighbor_cutoff=horizon_neighbor_cutoff,
        collective_kink_weight=collective_kink_weight,
        collective_kink_m=collective_kink_m,
        collective_kink_theta_ref_rad=collective_kink_theta_ref_rad,
        collective_kink_use_ss_mask=collective_kink_use_ss_mask,
        variational_pair_weight=variational_pair_weight,
        variational_pair_epsilon=variational_pair_epsilon,
        variational_pair_sigma=variational_pair_sigma,
        variational_pair_dist_cutoff=variational_pair_dist_cutoff,
        variational_pair_min_seq_sep=variational_pair_min_seq_sep,
        variational_pair_max_pairs=variational_pair_max_pairs,
        variational_staged_opt=variational_staged_opt,
        variational_stage_frac=variational_stage_frac,
        variational_bound_prune=variational_bound_prune,
        variational_bound_prune_margin=variational_bound_prune_margin,
        inertial_twist_weight=inertial_twist_weight,
        inertial_twist_theta_ref_rad=inertial_twist_theta_ref_rad,
        inertial_twist_exponent=inertial_twist_exponent,
        include_ligands=use_lig,
        ligands=lig_list if use_lig else None,
        **lig_kw,
    )
    het_chain = ligand_chain_id if use_lig else None
    pdb_str = full_chain_to_pdb(raw, ligand_chain_id=het_chain)
    note = ligand_summary(lig_list) if use_lig else ""
    return LeanTunnelFoldResult(
        sequence=seq,
        n_res=len(seq),
        temperature_k=float(temperature_k),
        ph=float(ph),
        epsilon_r=er,
        em_scale=float(em_scale),
        water_bond_angle_deg=WATER_BOND_ANGLE_DEG,
        kappa_dihedral=float(kappa_dihedral),
        pdb=pdb_str,
        raw_result=raw,
        include_ligands=use_lig,
        ligand_note=note,
    )
