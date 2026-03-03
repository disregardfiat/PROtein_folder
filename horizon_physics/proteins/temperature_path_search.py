"""
Temperature-aware discrete DOF search: Metropolis moves over φ/ψ states with optional locking.

Builds backbone DOFs, proposes moves from candidate_moves(), accepts/rejects with ΔE/kT,
rebuilds Cα from φ/ψ and evaluates E_tot. Used to refine a backbone (e.g. after tunnel or Cartesian).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .backbone_phi_psi import ca_positions_from_phi_psi, backbone_phi_psi_from_atoms
from .casp_submission import _place_backbone_ca, _place_full_backbone
from .discrete_dof import (
    DiscreteDof,
    DofGroup,
    build_backbone_dofs_for_sequence,
)
from .folding_energy import e_tot_ca_with_bonds
from .em_field_pipeline import Atom as _EMAtom, EMField as _EMField, AA_1to3 as _AA_1to3

# k_B in eV/K
K_B_EV_K = 8.617333e-5


def angles_from_dofs(
    phi_dofs: List[DiscreteDof],
    psi_dofs: List[DiscreteDof],
) -> Tuple[np.ndarray, np.ndarray]:
    """Current φ, ψ in radians from DOF states."""
    phi_rad = np.array([d.angle for d in phi_dofs], dtype=float)
    psi_rad = np.array([d.angle for d in psi_dofs], dtype=float)
    return phi_rad, psi_rad


def snap_dofs_to_angles(
    phi_dofs: List[DiscreteDof],
    psi_dofs: List[DiscreteDof],
    phi_rad: np.ndarray,
    psi_rad: np.ndarray,
) -> None:
    """Set each DOF's state_index to the discrete state nearest to the given angle (mod 2π). Modifies in place."""
    def snap_one(dof: DiscreteDof, angle: float) -> None:
        ang = float(angle)
        # wrap to [-pi, pi] for comparison with profile angles
        ang = (ang + math.pi) % (2 * math.pi) - math.pi
        diff = np.abs(dof.angles - ang)
        # handle wraparound
        diff = np.minimum(diff, 2 * math.pi - diff)
        dof.state_index = int(np.argmin(diff))

    for i, d in enumerate(phi_dofs):
        if i < len(phi_rad):
            snap_one(d, phi_rad[i])
    for i, d in enumerate(psi_dofs):
        if i < len(psi_rad):
            snap_one(d, psi_rad[i])


def _backbone_from_dofs(
    phi_dofs: List[DiscreteDof],
    psi_dofs: List[DiscreteDof],
    seq: str,
) -> List[Tuple[str, np.ndarray]]:
    """Rebuild backbone atoms from current φ/ψ DOF state (for re-drawing the field)."""
    phi_rad, psi_rad = angles_from_dofs(phi_dofs, psi_dofs)
    ca = ca_positions_from_phi_psi(phi_rad, psi_rad)
    return _place_full_backbone(ca, seq)


def _em_unfreeze_locked_dofs(
    sequence: str,
    backbone_atoms: List[Tuple[str, np.ndarray]],
    phi_dofs: List[DiscreteDof],
    psi_dofs: List[DiscreteDof],
    *,
    torque_threshold: float = 0.5,
    top_fraction: float = 0.2,
    assembly_mode: bool = False,
) -> bool:
    """
    Tree-torque from the ends (or from COM in assembly mode): use EM field torque,
    add torques along the chain until we have a translation (residues to unfreeze),
    process in order (ends inward or further-from-COM first), re-draw the field after each nudge.

    - Build EM field from current backbone; compute forces.
    - assembly_mode=False (single chain): If termini are free, anchors at N and C,
      combine τ per residue, order from ends inward. If a terminus is buried, single COM anchor.
    - assembly_mode=True (A+B, (A+B)+C): Single COM anchor; order residues to unfreeze
      by distance from COM (further from center of mass first).
    - Unfreeze and nudge one residue, re-build backbone, re-draw field; iterate.

    Returns True if any DOF was unlocked or nudged.
    """
    seq = "".join(c for c in sequence.strip().upper() if c.isalpha())
    n_res = len(seq)
    if n_res == 0 or len(backbone_atoms) != 4 * n_res:
        return False

    # Terminus considered "not free" if within this fraction of R_g from COM
    TERMINUS_FREE_RG_FRACTION = 0.5

    def build_field_and_torques(
        bb_atoms: List[Tuple[str, np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build EM field, get forces, compute torque per residue. Use N/C anchors when
        termini are free and not assembly_mode; else single COM anchor. Returns (pos, tau_norms, to_unlock_sorted)."""
        atoms_list: List[_EMAtom] = []
        for i in range(n_res):
            res_1 = seq[i]
            res_3 = _AA_1to3.get(res_1, "UNK")
            for a in range(4):
                name, xyz = bb_atoms[i * 4 + a]
                atoms_list.append(_EMAtom(name, res_3, i + 1, np.array(xyz, dtype=float)))
        pos = np.array([a.pos for a in atoms_list])
        lo = np.min(pos, axis=0) - 15.0
        hi = np.max(pos, axis=0) + 15.0
        field = _EMField(origin=lo, size=hi - lo, res=2.0)
        for a in atoms_list:
            field.add_atom(a)
        forces_all = field.forces_at_all(pos)
        if not np.isfinite(forces_all).all():
            return pos, np.zeros(n_res, dtype=float), np.array([], dtype=np.int64)

        com = np.mean(pos, axis=0)
        diff = pos - com
        r_g = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
        r_g = max(r_g, 1e-6)

        r_N = np.mean(pos[0:4], axis=0)
        r_C = np.mean(pos[-4:], axis=0)
        d_N = float(np.linalg.norm(r_N - com))
        d_C = float(np.linalg.norm(r_C - com))
        termini_free = (
            not assembly_mode
            and d_N >= TERMINUS_FREE_RG_FRACTION * r_g
            and d_C >= TERMINUS_FREE_RG_FRACTION * r_g
        )

        if termini_free:
            # Tree from ends: torque about N and C, combine
            rel_N = pos - r_N
            rel_C = pos - r_C
            tau_all_N = np.cross(rel_N, forces_all)
            tau_all_C = np.cross(rel_C, forces_all)
            tau_res = np.zeros((n_res, 3), dtype=float)
            for r in range(n_res):
                i0, i1 = 4 * r, 4 * r + 4
                tau_res[r] = np.sum(tau_all_N[i0:i1], axis=0) + np.sum(tau_all_C[i0:i1], axis=0)
        else:
            # Termini wrapped into center: single COM anchor
            rel = pos - com
            tau_all = np.cross(rel, forces_all)
            tau_res = np.zeros((n_res, 3), dtype=float)
            for r in range(n_res):
                i0, i1 = 4 * r, 4 * r + 4
                tau_res[r] = np.sum(tau_all[i0:i1], axis=0)

        tau_norms = np.linalg.norm(tau_res, axis=1)
        t_max = float(np.max(tau_norms))
        if not np.isfinite(t_max) or t_max < 1e-6:
            return pos, tau_norms, np.array([], dtype=np.int64)
        cutoff = max(torque_threshold, (1.0 - top_fraction) * t_max)
        to_unlock = np.where(tau_norms >= cutoff)[0]
        if assembly_mode:
            # Assembly: order by distance from COM (further from center of mass first)
            res_centers = np.array([np.mean(pos[4 * r : 4 * r + 4], axis=0) for r in range(n_res)])
            dist_from_com = np.linalg.norm(res_centers - com, axis=1)
            to_unlock = sorted(to_unlock, key=lambda i: -dist_from_com[i])
        elif termini_free:
            to_unlock = sorted(to_unlock, key=lambda i: min(i, n_res - 1 - i))
        else:
            to_unlock = sorted(to_unlock, key=lambda i: -tau_norms[i])
        return pos, tau_norms, np.array(to_unlock, dtype=np.int64)

    changed = False
    bb = list(backbone_atoms)
    max_redraws = 500  # cap inner iterations per phase (re-draw after each nudge)
    for _ in range(max_redraws):
        _, tau_norms, to_unlock = build_field_and_torques(bb)
        if to_unlock.size == 0:
            break
        res_idx = int(to_unlock[0])
        d_phi = phi_dofs[res_idx] if res_idx < len(phi_dofs) else None
        d_psi = psi_dofs[res_idx] if res_idx < len(psi_dofs) else None
        for d in (d_phi, d_psi):
            if d is None:
                continue
            was_locked = d.locked
            d.locked = False
            moves = d.candidate_moves()
            if moves:
                best_state = min(
                    (new_idx for new_idx, _ in moves),
                    key=lambda idx: float(d.energies[idx]),
                )
                if best_state != d.state_index:
                    d.state_index = best_state
                    changed = True
            if was_locked:
                changed = True
        # Re-draw the field: rebuild backbone from current DOF state for next iteration
        bb = _backbone_from_dofs(phi_dofs, psi_dofs, seq)

    return changed


@dataclass
class DiscreteRefineResult:
    ca_min: np.ndarray
    backbone_atoms: List[Tuple[str, np.ndarray]]
    E_ca_final: float
    n_res: int
    sequence: str
    n_steps: int
    n_accept: int
    info: Dict[str, Any] = field(default_factory=dict)


def run_discrete_refinement(
    sequence: str,
    temperature: float = 310.0,
    n_steps: int = 200,
    initial_backbone_atoms: Optional[List[Tuple[str, np.ndarray]]] = None,
    initial_ca: Optional[np.ndarray] = None,
    lock_every: int = 20,
    barrier_factor: float = 5.0,
    n_states: int = 32,
    seed: Optional[int] = None,
    run_until_converged: bool = False,
    max_phases_cap: int = 50,
    assembly_mode: bool = False,
) -> DiscreteRefineResult:
    """
    Refine a backbone using discrete φ/ψ moves with EM-field tree-torque unfreezing.

    assembly_mode: If True, use single COM anchor and order residues to unfreeze by
    distance from center of mass (further first). Use for A+B / (A+B)+C assembly refinement.

    1) If initial_backbone_atoms is provided, extract φ/ψ and snap DOFs to nearest state.
       Else if initial_ca is provided, build backbone from _place_full_backbone(ca, seq) then extract φ/ψ.
       Else build initial CA via _place_backbone_ca(seq) and same.
    2) Build phi_dofs, psi_dofs; group = all DOFs.
    3) For n_steps: get candidate_moves; pick one at random; Metropolis accept; rebuild ca, E_tot; every lock_every call maybe_lock(kT).
    4) Return final ca, backbone_atoms, E_ca_final.

    Returns
    -------
    DiscreteRefineResult with ca_min, backbone_atoms, E_ca_final, n_accept, etc.
    """
    from .full_protein_minimizer import _z_list_ca

    seq = "".join(c for c in sequence.strip().upper() if c.isalpha())
    n_res = len(seq)
    if n_res == 0:
        raise ValueError("Empty sequence for run_discrete_refinement.")

    z_ca = _z_list_ca(seq)
    kT_ev = K_B_EV_K * temperature

    # Initial backbone and φ/ψ
    if initial_backbone_atoms is not None and len(initial_backbone_atoms) == 4 * n_res:
        backbone_atoms = list(initial_backbone_atoms)
        phi_rad, psi_rad = backbone_phi_psi_from_atoms(backbone_atoms)
    else:
        if initial_ca is not None and initial_ca.shape == (n_res, 3):
            ca = np.asarray(initial_ca, dtype=float)
        else:
            ca = _place_backbone_ca(seq)
        backbone_atoms = _place_full_backbone(ca, seq)
        phi_rad, psi_rad = backbone_phi_psi_from_atoms(backbone_atoms)

    phi_dofs, psi_dofs = build_backbone_dofs_for_sequence(seq, temperature, n_states=n_states)
    snap_dofs_to_angles(phi_dofs, psi_dofs, phi_rad, psi_rad)

    all_dofs = list(phi_dofs) + list(psi_dofs)
    group = DofGroup(dofs=all_dofs)
    n_phi = len(phi_dofs)

    def apply_move(dof_index: int, new_state_index: int) -> None:
        if dof_index < n_phi:
            phi_dofs[dof_index].state_index = new_state_index
        else:
            psi_dofs[dof_index - n_phi].state_index = new_state_index

    phi_rad, psi_rad = angles_from_dofs(phi_dofs, psi_dofs)
    ca = ca_positions_from_phi_psi(phi_rad, psi_rad)
    E_current = float(e_tot_ca_with_bonds(ca, z_ca))

    total_accept = 0
    total_steps = 0
    # Two-tier loop: deterministic discrete DOFs, then EM field unfreeze when plateaued.
    # If run_until_converged: keep going until a phase has 0 accepts and EM unfreeze returns False.
    max_phases = max_phases_cap if run_until_converged else 4
    em_torque_threshold = 0.5
    em_top_fraction = 0.2
    phase_infos: List[Dict[str, Any]] = []
    converged = False

    for phase in range(max_phases):
        phase_accept = 0
        phase_steps = 0

        for step in range(n_steps):
            if group.all_locked():
                break
            moves = group.candidate_moves()
            if not moves:
                break
            # Deterministic choice: pick move with lowest ΔE (best local improvement)
            dof_index, new_state_index, dE = min(moves, key=lambda m: m[2])
            # Pure downhill dynamics on the discrete HQIV profile: only accept if ΔE < 0.
            if dE < 0.0:
                apply_move(dof_index, new_state_index)
                phi_rad, psi_rad = angles_from_dofs(phi_dofs, psi_dofs)
                ca = ca_positions_from_phi_psi(phi_rad, psi_rad)
                E_current = float(e_tot_ca_with_bonds(ca, z_ca))
                phase_accept += 1
                total_accept += 1
            else:
                # No downhill move available from current state: plateau for this phase.
                break
            if (step + 1) % lock_every == 0:
                for d in all_dofs:
                    d.maybe_lock(kT_ev, barrier_factor=barrier_factor)
            phase_steps += 1
            total_steps += 1

        # Record phase stats
        phase_infos.append(
            {
                "phase": phase,
                "steps": phase_steps,
                "accept": phase_accept,
                "all_locked": group.all_locked(),
            }
        )

        # If we made progress in this phase and not all DOFs are locked, continue
        if phase_accept > 0 and not group.all_locked():
            continue

        # Plateau: try EM-field unfreezing. If it can't unlock anything, we're done.
        backbone_atoms = _place_full_backbone(ca, seq)
        changed = _em_unfreeze_locked_dofs(
            seq,
            backbone_atoms,
            phi_dofs,
            psi_dofs,
            torque_threshold=em_torque_threshold,
            top_fraction=em_top_fraction,
            assembly_mode=assembly_mode,
        )
        if not changed:
            converged = True
            break

        # After EM unfreeze / nudge, recompute angles, CA, energy, and keep going.
        phi_rad, psi_rad = angles_from_dofs(phi_dofs, psi_dofs)
        ca = ca_positions_from_phi_psi(phi_rad, psi_rad)
        E_current = float(e_tot_ca_with_bonds(ca, z_ca))

    backbone_atoms = _place_full_backbone(ca, seq)
    return DiscreteRefineResult(
        ca_min=ca,
        backbone_atoms=backbone_atoms,
        E_ca_final=E_current,
        n_res=n_res,
        sequence=seq,
        n_steps=total_steps,
        n_accept=total_accept,
        info={
            "temperature": temperature,
            "lock_every": lock_every,
            "phases": phase_infos,
            "converged": converged,
        },
    )
