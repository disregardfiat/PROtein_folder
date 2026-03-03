"""
Two-chain (or N-chain) assembly: surface match, placement, and complex minimization.

Pipeline: minimize each chain → score surface "match potential" (contact/clash over
compacted surfaces) → place chains at best-fit interface → minimize combined system
(bonds only within each chain, horizon + inter-chain clash) → return each chain PDB
plus one complex PDB.

MIT License. Python 3.10+. Numpy.
"""

from __future__ import annotations

import multiprocessing
import numpy as np
from typing import List, Tuple

from .folding_energy import (
    build_bond_poles_segments,
    build_horizon_poles,
    grad_from_poles,
    grad_horizon_full,
    R_BOND_MIN,
    R_BOND_MAX,
    R_CA_CA_EQ,
    K_BOND,
    R_CLASH,
    K_CLASH,
)
from .gradient_descent_folding import _project_bonds
from .surface_attachment import find_attachment_point, ATTACHMENT_CONTACT_DIST

try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except ImportError:
    cKDTree = None
    _HAS_SCIPY = False

# Contact = A-B pair in [contact_lo, contact_hi]; clash = pair < clash_max
CONTACT_LO = 4.0   # Å
CONTACT_HI = 10.0  # Å
CLASH_MAX = 2.5    # Å
DOCK_COM_MIN = 12.0  # Å min COM distance when placing
DOCK_COM_MAX = 35.0  # Å max COM distance
N_ROTATIONS = 36    # number of rotations to try for chain B
N_TRANSLATIONS = 8  # steps along A→B axis


def _clash_pairs_multi_chain(
    positions: np.ndarray,
    segment_ends: List[int],
    r_clash: float = R_CLASH,
    cutoff: float = 15.0,
) -> List[Tuple[int, int, float, np.ndarray]]:
    """All non-bond pairs (including inter-chain) within r_clash. Bond = consecutive within same segment."""
    n = positions.shape[0]
    out: List[Tuple[int, int, float, np.ndarray]] = []
    bond_set = set()
    start = 0
    for end in segment_ends:
        end = min(end, n)
        for i in range(start, end - 1):
            bond_set.add((i, i + 1))
        start = end
    if _HAS_SCIPY and n > 10:
        tree = cKDTree(positions)
        for i in range(n):
            for j in tree.query_ball_point(positions[i], max(r_clash, cutoff)):
                if j <= i:
                    continue
                if (i, j) in bond_set or (j, i) in bond_set:
                    continue
                d = positions[j] - positions[i]
                r = float(np.linalg.norm(d))
                if r < 1e-12 or r >= r_clash:
                    continue
                out.append((i, j, r, d / r))
    else:
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in bond_set:
                    continue
                d = positions[j] - positions[i]
                r = float(np.linalg.norm(d))
                if r < 1e-12 or r >= r_clash:
                    continue
                out.append((i, j, r, d / r))
    return out


def _contact_clash_score(
    ca_a: np.ndarray,
    ca_b: np.ndarray,
    contact_lo: float = CONTACT_LO,
    contact_hi: float = CONTACT_HI,
    clash_max: float = CLASH_MAX,
) -> Tuple[float, float]:
    """Score for placement: (n_contact, n_clash). Contact = A-B pair in [contact_lo, contact_hi]; clash = < clash_max."""
    n_a, n_b = ca_a.shape[0], ca_b.shape[0]
    if _HAS_SCIPY:
        tree_b = cKDTree(ca_b)
        n_contact = 0
        n_clash = 0
        for i in range(n_a):
            for j in tree_b.query_ball_point(ca_a[i], contact_hi):
                r = float(np.linalg.norm(ca_a[i] - ca_b[j]))
                if r < clash_max:
                    n_clash += 1
                elif contact_lo <= r <= contact_hi:
                    n_contact += 1
        return (float(n_contact), float(n_clash))
    n_contact = 0
    n_clash = 0
    for i in range(n_a):
        for j in range(n_b):
            r = float(np.linalg.norm(ca_a[i] - ca_b[j]))
            if r < clash_max:
                n_clash += 1
            elif contact_lo <= r <= contact_hi:
                n_contact += 1
    return (float(n_contact), float(n_clash))


def _rotation_matrix_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix for axis (unit) and angle in radians."""
    axis = np.asarray(axis, dtype=float)
    if np.linalg.norm(axis) < 1e-12:
        return np.eye(3)
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return c * np.eye(3) + (1 - c) * np.outer(axis, axis) + s * np.array([
        [0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]
    ])


def _place_two_chains_attachment_guided(
    ca_a: np.ndarray,
    ca_b: np.ndarray,
    d_contact: float = ATTACHMENT_CONTACT_DIST,
) -> Tuple[np.ndarray, np.ndarray] | None:
    """
    Place B using surface attachment points: align B's best attachment toward A's.
    Returns (ca_a, ca_b_placed) or None if attachment analysis fails (degenerate).
    """
    if np.any(np.abs(ca_a) > 1e6) or np.any(np.abs(ca_b) > 1e6):
        return None  # Bad input scale; fall back to grid search
    n_a, n_b = ca_a.shape[0], ca_b.shape[0]
    z_a = np.full(n_a, 6)
    z_b = np.full(n_b, 6)
    pos_a, dir_a, _ = find_attachment_point(ca_a, z_a)
    pos_b, dir_b, _ = find_attachment_point(ca_b, z_b)
    dir_a = np.asarray(dir_a, dtype=float)
    dir_b = np.asarray(dir_b, dtype=float)
    if np.linalg.norm(dir_a) < 1e-9:
        dir_a = np.array([1.0, 0.0, 0.0])
    else:
        dir_a = dir_a / np.linalg.norm(dir_a)
    if np.linalg.norm(dir_b) < 1e-9:
        dir_b = np.array([1.0, 0.0, 0.0])
    else:
        dir_b = dir_b / np.linalg.norm(dir_b)
    # Rotate B so dir_b → -dir_a (B's outward points toward A)
    target = -dir_a
    axis = np.cross(dir_b, target)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        # dir_b parallel to target; use 180° around perpendicular
        if np.abs(np.dot(dir_b, target)) > 0.99:
            axis = np.array([1.0, 0.0, 0.0]) if abs(dir_b[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            axis = axis - np.dot(axis, dir_b) * dir_b
            axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-9:
            return None
    axis = axis / axis_norm
    angle = np.arccos(np.clip(np.dot(dir_b, target), -1.0, 1.0))
    R = _rotation_matrix_axis_angle(axis, angle)
    com_b = np.mean(ca_b, axis=0)
    ca_b_centered = ca_b - com_b
    pos_b_rel = pos_b - com_b
    ca_b_rot = (R @ ca_b_centered.T).T
    pos_b_rot = R @ pos_b_rel
    # Translate so B's attachment point is at pos_a + dir_a * d_contact
    trans = pos_a + dir_a * d_contact - pos_b_rot
    ca_b_placed = ca_b_rot + trans
    return ca_a.copy(), ca_b_placed


def place_two_chains(
    ca_a: np.ndarray,
    ca_b: np.ndarray,
    n_rotations: int = N_ROTATIONS,
    n_translations: int = N_TRANSLATIONS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Place chain B relative to chain A. Uses surface attachment analysis (O(n)):
    finds exposed-surface bonding hotspots from horizon φ, aligns B's best
    attachment toward A's, then refines with contact/clash scoring.
    Falls back to grid search if attachment-guided placement fails.
    Returns (ca_a_placed, ca_b_placed) with A unchanged and B placed.
    """
    # 1) Try attachment-guided placement
    result = _place_two_chains_attachment_guided(ca_a, ca_b)
    if result is not None:
        ca_a_fixed, ca_b_placed = result
        n_contact, n_clash = _contact_clash_score(ca_a_fixed, ca_b_placed)
        if n_clash == 0 or n_contact > 0:
            # Refine: small rotations around attachment axis
            pos_a, dir_a, _ = find_attachment_point(ca_a, np.full(ca_a.shape[0], 6))
            com_b = np.mean(ca_b_placed, axis=0)
            ca_b_centered = ca_b_placed - com_b
            best_b = ca_b_placed.copy()
            best_score = n_contact - 10.0 * n_clash
            for twist in range(12):
                angle = 2 * np.pi * twist / 12
                R = _rotation_matrix_axis_angle(dir_a, angle)
                b_twist = (R @ ca_b_centered.T).T + com_b
                nc, ncl = _contact_clash_score(ca_a_fixed, b_twist)
                sc = nc - 10.0 * ncl
                if sc > best_score:
                    best_score = sc
                    best_b = b_twist.copy()
            return ca_a_fixed, best_b
        return ca_a_fixed, ca_b_placed

    # 2) Fallback: original grid search
    com_a = np.mean(ca_a, axis=0)
    com_b = np.mean(ca_b, axis=0)
    ca_b_centered = ca_b - com_b
    best_score = -1e9
    best_b = ca_b.copy()
    np.random.seed(42)
    for rot_idx in range(n_rotations):
        angle = 2 * np.pi * rot_idx / n_rotations
        axis = np.array([0, 0, 1])
        R1 = _rotation_matrix_axis_angle(axis, angle)
        for tilt in [0, 0.5 * np.pi, 1.0 * np.pi]:
            R2 = _rotation_matrix_axis_angle(np.array([1, 0, 0]), tilt)
            R = R2 @ R1
            b_rot = (R @ ca_b_centered.T).T
            com_b_rot = np.mean(b_rot, axis=0)
            direction = com_b_rot - com_a
            if np.linalg.norm(direction) < 1e-9:
                direction = np.array([1.0, 0.0, 0.0])
            direction = direction / np.linalg.norm(direction)
            for t_idx in range(n_translations):
                dist = DOCK_COM_MIN + (DOCK_COM_MAX - DOCK_COM_MIN) * t_idx / max(1, n_translations - 1)
                trans = com_a + direction * dist - np.mean(b_rot, axis=0)
                b_placed = b_rot + trans
                n_contact, n_clash = _contact_clash_score(ca_a, b_placed)
                score = n_contact - 10.0 * n_clash
                if score > best_score:
                    best_score = score
                    best_b = b_placed.copy()
    return ca_a.copy(), best_b


def _project_bonds_segments(
    positions: np.ndarray,
    segment_ends: List[int],
    r_min: float = R_BOND_MIN,
    r_max: float = R_BOND_MAX,
) -> np.ndarray:
    """Project bonds only within each segment (no cross-chain pull)."""
    pos = np.array(positions, dtype=float)
    n = pos.shape[0]
    start = 0
    for end in segment_ends:
        end = min(end, n)
        for i in range(start, end - 1):
            d = pos[i + 1] - pos[i]
            r = np.linalg.norm(d)
            if r < 1e-9:
                d = np.array([1.0, 0.0, 0.0]) if i == start else (pos[i] - pos[i - 1])
                r = np.linalg.norm(d)
                if r < 1e-9:
                    r = 1.0
            if r > r_max:
                pos[i + 1] = pos[i] + (r_max / r) * d
            elif r < r_min:
                pos[i + 1] = pos[i] + (r_min / r) * d
        start = end
    return pos


def minimize_complex(
    ca_a: np.ndarray,
    ca_b: np.ndarray,
    seq_a: str,
    seq_b: str,
    z_list: np.ndarray,
    max_iter: int = 80,
    r_bond_min: float = R_BOND_MIN,
    r_bond_max: float = R_BOND_MAX,
    converge_max_disp_ang: float | None = None,
) -> Tuple[np.ndarray, dict]:
    """
    Minimize the two-chain complex (HKE-style: bonds within chains, horizon + clash; no funnel).
    If converge_max_disp_ang is set (e.g. 0.5), stop when RMS Cα displacement per residue
    over the last step is below that threshold (Å). Returns (ca_combined, info).
    """
    from .gradient_descent_folding import _lbfgs_two_loop
    from .folding_energy import e_tot

    n1, n2 = ca_a.shape[0], ca_b.shape[0]
    segment_ends = [n1, n1 + n2]
    pos = np.vstack([ca_a, ca_b]).astype(float)
    n = pos.shape[0]
    z_list = np.asarray(z_list, dtype=float)
    if z_list.size != n:
        z_list = np.full(n, 6)

    from .folding_energy import R_CA_CA_EQ

    def energy(pos_flat: np.ndarray) -> float:
        p = pos_flat.reshape(n, 3)
        e = e_tot(p, z_list)
        start = 0
        for end in segment_ends:
            end_ = min(end, n)
            for i in range(start, end_ - 1):
                j = i + 1
                r = np.linalg.norm(p[j] - p[i])
                if r < 1e-12:
                    continue
                if r < r_bond_min:
                    e += K_BOND * (r_bond_min - r) ** 2
                elif r > r_bond_max:
                    e += K_BOND * (r - r_bond_max) ** 2
                else:
                    e += K_BOND * 0.1 * (r - R_CA_CA_EQ) ** 2
            start = end_
        for i, j, r, _ in _clash_pairs_multi_chain(p, segment_ends, r_clash=R_CLASH):
            e += K_CLASH * (R_CLASH - r) ** 2
        return float(e)

    def grad(pos_flat: np.ndarray) -> np.ndarray:
        p = pos_flat.reshape(n, 3)
        g = np.zeros_like(p)
        poles_bond = build_bond_poles_segments(
            p, segment_ends, r_min=r_bond_min, r_max=r_bond_max
        )
        g += grad_from_poles(poles_bond, n)
        g += grad_horizon_full(p, z_list)
        for i, j, r, u in _clash_pairs_multi_chain(p, segment_ends, r_clash=R_CLASH):
            f = -2 * K_CLASH * (R_CLASH - r)
            g[i] -= f * u
            g[j] += f * u
        return g.ravel()

    x = pos.ravel()
    s_list: list = []
    y_list: list = []
    m = 10
    gtol = 1e-5
    grad_flat = grad(x)
    it = -1
    for it in range(max_iter):
        g_norm = np.linalg.norm(grad_flat)
        if g_norm <= gtol:
            break
        if len(s_list) >= m:
            s_list.pop(0)
            y_list.pop(0)
        if len(s_list) == 0:
            direction = -grad_flat
        else:
            direction = _lbfgs_two_loop(grad_flat, s_list, y_list, m)
        step = 1.0
        e0 = energy(x)
        for _ in range(40):
            x_new = x + step * direction
            x_new = _project_bonds_segments(
                x_new.reshape(n, 3), segment_ends, r_min=r_bond_min, r_max=r_bond_max
            ).ravel()
            if energy(x_new) <= e0 + 1e-4 * step * np.dot(grad_flat, direction):
                break
            step *= 0.5
        x_prev = x.copy()
        x = x_new
        # Stop when RMS Cα displacement per residue < threshold (e.g. 0.5 Å)
        if converge_max_disp_ang is not None:
            disp = x.reshape(n, 3) - x_prev.reshape(n, 3)
            # Use Cα positions only: residues correspond to indices 1, 1+4, 1+8, ...
            n_res = n  # upper bound; refined below
            if n % 4 == 0:
                n_res = n // 4
            ca_disps: list[float] = []
            for i in range(n_res):
                idx = 4 * i + 1
                if idx >= n:
                    break
                ca_disps.append(float(np.linalg.norm(disp[idx])))
            if ca_disps:
                rms_disp = float(np.sqrt(np.mean(np.square(ca_disps))))
                if rms_disp < converge_max_disp_ang:
                    pos_final = x.reshape(n, 3)
                    return pos_final, {
                        "e_final": energy(x),
                        "n_iter": it + 1,
                        "success": True,
                        "rms_disp_ang": rms_disp,
                    }
        grad_new = grad(x)
        s_list.append(x - x_prev)
        y_list.append(grad_new - grad_flat)
        grad_flat = grad_new
    pos_final = x.reshape(n, 3)
    return pos_final, {"e_final": energy(x), "n_iter": it + 1, "success": True}


def _hke_chain_worker(
    seq: str,
    funnel_radius: float,
    funnel_radius_exit: float,
    funnel_stiffness: float,
    hke_max_iter_s1: int,
    hke_max_iter_s2: int,
    hke_max_iter_s3: int,
    gtol: float = 1e-5,
    grouping_strategy: str = "residue",
) -> dict:
    """Run HKE-with-funnel for one chain. Module-level for multiprocessing pickling. Returns result dict."""
    from .hierarchical import minimize_full_chain_hierarchical, hierarchical_result_for_pdb
    pos, z_list = minimize_full_chain_hierarchical(
        seq,
        include_sidechains=False,
        funnel_radius=funnel_radius,
        funnel_stiffness=funnel_stiffness,
        funnel_radius_exit=funnel_radius_exit,
        max_iter_stage1=hke_max_iter_s1,
        max_iter_stage2=hke_max_iter_s2,
        max_iter_stage3=hke_max_iter_s3,
        gtol=gtol,
        grouping_strategy=grouping_strategy,
    )
    return hierarchical_result_for_pdb(pos, z_list, seq, include_sidechains=False)


def run_two_chain_assembly_hke(
    seq_a: str,
    seq_b: str,
    funnel_radius: float,
    funnel_radius_exit: float,
    funnel_stiffness: float,
    hke_max_iter_s1: int,
    hke_max_iter_s2: int,
    hke_max_iter_s3: int,
    converge_max_disp_per_100_res: float = 0.5,
    max_dock_iter: int = 2000,
    gtol: float = 1e-5,
    grouping_strategy: str = "residue",
) -> Tuple[dict, dict, dict]:
    """
    Run each chain through HKE-with-funnel in its own process; map bond sites (placement);
    then run the complex with HKE (no funnel) until max displacement per residue
    < converge_max_disp_per_100_res * (total_residues / 100) Å (0.5 Å per 100 res:
    twice as many residues, twice as loose for the complex).
    Returns (result_a, result_b, result_complex) with backbone_chain_a/b for PDBs.
    """
    try:
        from .hierarchical import minimize_full_chain_hierarchical, hierarchical_result_for_pdb
    except ImportError:
        raise ImportError("run_two_chain_assembly_hke requires horizon_physics.proteins.hierarchical") from None

    kwargs = {
        "funnel_radius": funnel_radius,
        "funnel_radius_exit": funnel_radius_exit,
        "funnel_stiffness": funnel_stiffness,
        "hke_max_iter_s1": hke_max_iter_s1,
        "hke_max_iter_s2": hke_max_iter_s2,
        "hke_max_iter_s3": hke_max_iter_s3,
        "gtol": gtol,
        "grouping_strategy": grouping_strategy,
    }
    with multiprocessing.Pool(2) as pool:
        async_a = pool.apply_async(_hke_chain_worker, (seq_a,), kwargs)
        async_b = pool.apply_async(_hke_chain_worker, (seq_b,), kwargs)
        try:
            result_a = async_a.get()
            result_b = async_b.get()
        except Exception as e:
            raise e
    return run_two_chain_assembly(
        result_a,
        result_b,
        max_dock_iter=max_dock_iter,
        converge_max_disp_per_100_res=converge_max_disp_per_100_res,
    )


def _ca_from_result(result: dict) -> np.ndarray:
    """Extract Cα positions from result (ca_min if present, else from backbone_atoms)."""
    if "ca_min" in result and result["ca_min"] is not None:
        return np.asarray(result["ca_min"], dtype=float)
    backbone = result["backbone_atoms"]
    n_res = result["n_res"]
    # Backbone order: N, CA, C, O per residue (4 atoms per res)
    return np.array([backbone[4 * i + 1][1] for i in range(n_res)], dtype=float)


def run_two_chain_assembly(
    result_a: dict,
    result_b: dict,
    max_dock_iter: int = 80,
    converge_max_disp_per_100_res: float | None = None,
) -> Tuple[dict, dict, dict]:
    """
    Given two minimize_full_chain (or hierarchical) results for chain A and B:
    1) Place B relative to A using surface contact/clash score.
    2) Minimize the complex (HKE-style, no funnel) until convergence.
       If converge_max_disp_per_100_res is set (e.g. 0.5), stop when max displacement
       per residue < 0.5 * (total_residues / 100) Å (same rule as single-chain: twice as
       many residues, twice as loose for the complex).
    3) Return (result_a, result_b, result_complex) where result_complex has
       backbone_chain_a, backbone_chain_b for the complex PDB.
    """
    from .full_protein_minimizer import _place_full_backbone
    from .casp_submission import AA_1to3

    ca_a = _ca_from_result(result_a)
    ca_b = _ca_from_result(result_b)
    seq_a = result_a["sequence"]
    seq_b = result_b["sequence"]
    n1, n2 = ca_a.shape[0], ca_b.shape[0]
    n_res = n1 + n2
    z_list = np.full(n_res, 6)

    converge_ang = (
        (converge_max_disp_per_100_res * n_res / 100.0)
        if converge_max_disp_per_100_res is not None
        else None
    )
    ca_a_placed, ca_b_placed = place_two_chains(ca_a, ca_b)
    ca_complex, info = minimize_complex(
        ca_a_placed, ca_b_placed, seq_a, seq_b, z_list, max_iter=max_dock_iter,
        converge_max_disp_ang=converge_ang,
    )
    ca_a_final = ca_complex[:n1]
    ca_b_final = ca_complex[n1:]

    backbone_a = _place_full_backbone(ca_a_final, seq_a)
    backbone_b = _place_full_backbone(ca_b_final, seq_b)
    backbone_complex = backbone_a + backbone_b
    seq_combined = seq_a + seq_b  # for metadata; PDB will have chain A and B

    result_complex = {
        "ca_min": ca_complex,
        "backbone_atoms": backbone_complex,
        "backbone_chain_a": backbone_a,
        "backbone_chain_b": backbone_b,
        "sequence": seq_combined,
        "n_res": n_res,
        "chains": [
            {"sequence": seq_a, "n_res": n1, "ca_min": ca_a_final},
            {"sequence": seq_b, "n_res": n2, "ca_min": ca_b_final},
        ],
        "info": info,
    }
    return result_a, result_b, result_complex


def complex_to_single_chain_result(result_complex: dict) -> dict:
    """
    Turn a two-chain complex result into a single result dict that can be passed
    as the first argument to run_two_chain_assembly for (A+B)+C docking.
    """
    backbone = result_complex["backbone_chain_a"] + result_complex["backbone_chain_b"]
    return {
        "ca_min": result_complex["ca_min"],
        "backbone_atoms": backbone,
        "sequence": result_complex["sequence"],
        "n_res": result_complex["n_res"],
    }
