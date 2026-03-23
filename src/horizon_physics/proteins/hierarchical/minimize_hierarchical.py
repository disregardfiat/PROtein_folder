"""
Staged hierarchical minimization: coarse rigid-body + torsions -> refinement -> optional flat Cartesian.

Runs on Protein instance; uses backend (JAX if available) for acceleration.
Stage 1: minimize E over DOFs (6 + 2*(n_res-1)). Stage 2: same with tighter tolerance.
Stage 3: flat Cartesian refinement using existing grad_full on positions.

Funnel/ribosome as null search space: stages 1-2 approximate a confined volume (helices form,
then group translations); stage 3 corresponds to exit, allowing complex self-wrapping.
Optional funnel_radius (soft cone from first to last COM) bounds group COMs in stage 1-2.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional, Tuple

from . import backend as _be
from .protein import Protein


def _write_trajectory_frame(file_handle: Optional[Any], step: int, positions: Any) -> None:
    """JSONL one frame for Manim/tail -f (world Cartesian). Flush after each line."""
    if file_handle is None:
        return
    positions = _be.to_numpy(positions)
    if hasattr(positions, "ndim") and positions.ndim == 1:
        positions = positions.reshape(-1, 3)
    file_handle.write(json.dumps({"t": step, "positions": positions.tolist()}) + "\n")
    file_handle.flush()


def _write_trajectory_frame_6dof(file_handle: Optional[Any], step: int, pos_6dof: Any) -> None:
    """JSONL one frame: relative 6-DOF per residue for HKE. positions = list of 6-element lists [tx, ty, tz, euler_z, euler_y, euler_x] (Å, rad ZYX)."""
    if file_handle is None:
        return
    pos_6dof = _be.to_numpy(pos_6dof)
    if hasattr(pos_6dof, "ndim") and pos_6dof.ndim == 1:
        pos_6dof = pos_6dof.reshape(-1, 6)
    # Ensure each row is a list of 6 floats so consumer detects (n_res, 6) format
    file_handle.write(json.dumps({"t": step, "positions": [row.tolist() for row in pos_6dof]}) + "\n")
    file_handle.flush()

xp = _be.np_or_jnp()
_jax_available = _be.is_jax_available()


def relative_6dof_to_world_backbone(dofs_per_residue: Any) -> Any:
    """
    Convert (n_res, 6) relative 6-DOF per residue to world backbone (n_res*4, 3).

    Each row: [t_x, t_y, t_z, euler_z, euler_y, euler_x] in Å and radians (Z-Y-X).
    Builds local N, CA, C, O (CA at origin; bonds 1.33, 1.53, 1.23 Å as in PROtein),
    then world_i = t + R @ local_i per residue. Output order: N, CA, C, O per residue.
    """
    import numpy as np
    from horizon_physics.proteins.peptide_backbone import backbone_bond_lengths
    dofs = np.asarray(dofs_per_residue, dtype=np.float64)
    if dofs.ndim == 1:
        dofs = dofs.reshape(-1, 6)
    n_res = dofs.shape[0]
    if n_res == 0:
        return np.zeros((0, 3), dtype=np.float64)
    bonds = backbone_bond_lengths()
    r_n_ca, r_ca_c, r_c_o = bonds["N_Calpha"], bonds["Calpha_C"], bonds["C_O"]
    local = np.array([
        [-r_n_ca, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [r_ca_c, 0.0, 0.0],
        [r_ca_c, r_c_o, 0.0],
    ], dtype=np.float64)
    out = np.zeros((n_res * 4, 3), dtype=np.float64)
    for i in range(n_res):
        t = dofs[i, :3]
        ez, ey, ex = dofs[i, 3], dofs[i, 4], dofs[i, 5]
        ca, sa = np.cos(ez), np.sin(ez)
        cb, sb = np.cos(ey), np.sin(ey)
        cc, sc = np.cos(ex), np.sin(ex)
        R = np.array([
            [ca * cb, ca * sb * sc - sa * cc, ca * sb * cc + sa * sc],
            [sa * cb, sa * sb * sc + ca * cc, sa * sb * cc - ca * sc],
            [-sb, cb * sc, cb * cc],
        ], dtype=np.float64)
        for a in range(4):
            out[i * 4 + a] = t + R @ local[a]
    return out

try:
    from scipy.optimize import minimize as _scipy_minimize
    _SCIPY = True
except ImportError:
    _SCIPY = False


def _energy_from_dofs(protein: Protein, dofs: Any, inter_group_weight: float = 0.0, **kwargs: Any) -> float:
    """Set DOFs and return total energy (for scipy/L-BFGS). kwargs passed to compute_total_energy."""
    protein.set_dofs(dofs)
    return protein.compute_total_energy(
        include_bonds=kwargs.get("include_bonds", True),
        include_horizon=kwargs.get("include_horizon", True),
        include_clash=kwargs.get("include_clash", True),
        inter_group_weight=inter_group_weight,
        inter_group_length_scale=kwargs.get("inter_group_length_scale", 8.0),
        funnel_radius=kwargs.get("funnel_radius"),
        funnel_stiffness=kwargs.get("funnel_stiffness", 1.0),
        funnel_radius_exit=kwargs.get("funnel_radius_exit"),
    )


def _grad_dofs_fd(protein: Protein, dofs: Any, eps: float = 1e-6, inter_group_weight: float = 0.0, **kwargs: Any) -> Any:
    """Gradient w.r.t. DOFs by central differences (NumPy/JAX compatible)."""
    import numpy as np
    dofs_flat = np.asarray(_be.to_numpy(dofs)).reshape(-1)
    n = dofs_flat.shape[0]
    grad = np.zeros(n, dtype=np.float64)
    for i in range(n):
        d_plus = dofs_flat.copy()
        d_minus = dofs_flat.copy()
        d_plus[i] = dofs_flat[i] + eps
        d_minus[i] = dofs_flat[i] - eps
        e_plus = _energy_from_dofs(protein, d_plus, inter_group_weight=inter_group_weight, **kwargs)
        e_minus = _energy_from_dofs(protein, d_minus, inter_group_weight=inter_group_weight, **kwargs)
        grad[i] = (e_plus - e_minus) / (2.0 * eps)
    return grad


def _flat_cartesian_refinement(
    positions: Any,
    z_list: Any,
    max_iter: int = 100,
    gtol: float = 1e-5,
    converge_max_disp_ang: Optional[float] = None,
    n_res: Optional[int] = None,
) -> Tuple[Any, float]:
    """Refine positions with existing grad_full (NumPy path).
    If converge_max_disp_ang and n_res are set, stop when RMS Cα displacement per step < threshold (0.5 Å per 100 res).
    """
    from horizon_physics.proteins.folding_energy import e_tot_ca_with_bonds, grad_full
    import numpy as np
    pos = np.asarray(positions, dtype=np.float64).copy()
    pos = np.reshape(pos, (-1, 3))
    # Sanitize: replace nan/inf to avoid cKDTree overflow
    bad = ~np.isfinite(pos)
    if np.any(bad):
        pos[bad] = 0.0
    z_np = np.asarray(z_list, dtype=np.int32).ravel()
    if n_res is None:
        n_res = pos.shape[0] // 4
    step = 0.02
    for it in range(max_iter):
        g = grad_full(pos, z_np, include_bonds=True, include_horizon=True, include_clash=True)
        g_norm = float(np.linalg.norm(g))
        if g_norm < gtol:
            break
        pos_prev = pos.copy()
        pos = pos - step * g
        if converge_max_disp_ang is not None and n_res is not None:
            # RMS displacement per residue (Cα only, same rule as complex minimizer)
            ca_disps = []
            max_res = min(n_res, pos.shape[0] // 4)
            for i in range(max_res):
                idx = 4 * i + 1
                if idx < pos.shape[0]:
                    d = float(np.linalg.norm(pos[idx] - pos_prev[idx]))
                    ca_disps.append(d)
            if ca_disps:
                rms_disp = float(np.sqrt(np.mean(np.square(ca_disps))))
                if rms_disp < converge_max_disp_ang:
                    break
    e_final = float(e_tot_ca_with_bonds(pos, z_np))
    return pos, e_final


# Default weights for two-stage schedule: Stage 1 coarse (high inter-group), Stage 2 fine (low).
INTER_GROUP_WEIGHT_STAGE1 = 2.0
INTER_GROUP_WEIGHT_STAGE2 = 0.2


def run_staged_minimization(
    protein: Protein,
    max_iter_stage1: int = 80,
    max_iter_stage2: int = 120,
    max_iter_stage3: int = 100,
    gtol: float = 1e-5,
    device: Optional[str] = None,
    use_compact_init: bool = True,
    inter_group_weight_stage1: float = INTER_GROUP_WEIGHT_STAGE1,
    inter_group_weight_stage2: float = INTER_GROUP_WEIGHT_STAGE2,
    trajectory_log_path: Optional[str] = None,
    funnel_radius: Optional[float] = None,
    funnel_stiffness: float = 1.0,
    funnel_radius_exit: Optional[float] = None,
    converge_max_disp_ang: Optional[float] = None,
) -> Tuple[Any, dict]:
    """
    Stage 1 (coarse): high inter-group COM attraction + compact init; minimize over all DOFs.
    Stage 2 (fine): lower inter-group weight + full e_tot; refine DOFs.
    Stage 3: flat Cartesian refinement (grad_full on positions); funnel off (exit).

    Funnel as null search space: when funnel_radius is set, a soft cone (axis = first→last COM,
    radius grows from funnel_radius at first COM to funnel_radius_exit at last; default 2× for cone)
    confines group COMs in stages 1-2; stage 3 has no funnel so the chain can wrap.

    Returns (positions (N, 3), info dict with e_final, n_iter, etc.).
    """
    from .protein import generate_compact_start
    info = {"stage1_iter": 0, "stage2_iter": 0, "stage3_iter": 0, "e_final": 0.0, "success": True}
    dofs = protein.get_dofs()
    if dofs.size == 0:
        pos, z = protein.forward_kinematics()
        return _be.to_numpy(pos), info

    traj_file = open(trajectory_log_path, "w") if trajectory_log_path else None
    _step = [0]

    def _log_frame() -> None:
        if traj_file is None:
            return
        # (n_res, 6) per residue: [t_x, t_y, t_z, euler_z, euler_y, euler_x] Å and radians (ZYX)
        pos_6dof = protein.get_6dof_per_residue()
        pos_6dof = _be.to_numpy(pos_6dof)
        _write_trajectory_frame_6dof(traj_file, _step[0], pos_6dof)
        _step[0] += 1

    funnel_kwargs = (
        {
            "funnel_radius": funnel_radius,
            "funnel_stiffness": funnel_stiffness,
            "funnel_radius_exit": funnel_radius_exit,
        }
        if (funnel_radius is not None and funnel_radius > 0)
        else {}
    )

    try:
        # Compact init for Stage 1 (group COMs in a ball; funnel soft-bounds COMs in 1-2 when set)
        if use_compact_init and protein.n_res > 1:
            dofs = generate_compact_start(protein.n_groups(), radius=5.0, seed=42)
            protein.set_dofs(dofs)
            dofs = protein.get_dofs()
        _log_frame()

        # Stage 1: coarse — high inter-group attraction + funnel (if set)
        if _SCIPY:
            def obj1(x):
                return _energy_from_dofs(
                    protein, x, inter_group_weight=inter_group_weight_stage1, **funnel_kwargs
                )

            def jac1(x):
                return _grad_dofs_fd(
                    protein, x, inter_group_weight=inter_group_weight_stage1, **funnel_kwargs
                )

            def cb1(x):
                protein.set_dofs(x)
                _log_frame()

            res1 = _scipy_minimize(
                obj1,
                _be.to_numpy(dofs),
                method="L-BFGS-B",
                jac=jac1,
                callback=cb1,
                options={"maxiter": max_iter_stage1, "gtol": gtol},
            )
            dofs = xp.asarray(res1.x)
            protein.set_dofs(dofs)
            info["stage1_iter"] = res1.nit
        else:
            for _ in range(max_iter_stage1):
                g = _grad_dofs_fd(
                    protein, dofs, inter_group_weight=inter_group_weight_stage1, **funnel_kwargs
                )
                g_norm = float(xp.linalg.norm(g))
                if g_norm < gtol:
                    break
                dofs = dofs - 0.01 * g
                protein.set_dofs(dofs)
                _log_frame()
            info["stage1_iter"] = max_iter_stage1

        # Stage 2: fine — lower inter-group weight, full e_tot + funnel (if set)
        dofs = protein.get_dofs()
        if _SCIPY:
            def obj2(x):
                return _energy_from_dofs(
                    protein, x, inter_group_weight=inter_group_weight_stage2, **funnel_kwargs
                )

            def jac2(x):
                return _grad_dofs_fd(
                    protein, x, inter_group_weight=inter_group_weight_stage2, **funnel_kwargs
                )

            def cb2(x):
                protein.set_dofs(x)
                _log_frame()

            res2 = _scipy_minimize(
                obj2,
                _be.to_numpy(dofs),
                method="L-BFGS-B",
                jac=jac2,
                callback=cb2,
                options={"maxiter": max_iter_stage2, "gtol": gtol * 0.5},
            )
            protein.set_dofs(xp.asarray(res2.x))
            info["stage2_iter"] = res2.nit
        else:
            for _ in range(max_iter_stage2):
                g = _grad_dofs_fd(
                    protein, protein.get_dofs(), inter_group_weight=inter_group_weight_stage2, **funnel_kwargs
                )
                if float(xp.linalg.norm(g)) < gtol * 0.5:
                    break
                dofs = protein.get_dofs() - 0.005 * g
                protein.set_dofs(dofs)
                _log_frame()
            info["stage2_iter"] = max_iter_stage2
        # Stage 3: flat Cartesian refinement (same displacement tolerance as complex: 0.5 Å per 100 res)
        pos, z_list = protein.forward_kinematics()
        pos_np = _be.to_numpy(pos)
        z_np = _be.to_numpy(z_list)
        pos_refined, e_final = _flat_cartesian_refinement(
            pos_np, z_np, max_iter=max_iter_stage3, gtol=gtol,
            converge_max_disp_ang=converge_max_disp_ang, n_res=protein.n_res,
        )
        # Trajectory is 6-DOF only (Stage 1+2); Stage 3 is Cartesian refinement, no extra frame
        info["e_final"] = e_final
        info["stage3_iter"] = max_iter_stage3
    finally:
        if traj_file is not None:
            traj_file.close()
    return pos_refined, info


def minimize_full_chain_hierarchical(
    sequence: str,
    include_sidechains: bool = False,
    device: Optional[str] = None,
    grouping_strategy: str = "residue",
    ss_string: Optional[str] = None,
    domain_ranges: Optional[list] = None,
    max_iter_stage1: int = 80,
    max_iter_stage2: int = 120,
    max_iter_stage3: int = 100,
    gtol: float = 1e-5,
    trajectory_log_path: Optional[str] = None,
    funnel_radius: Optional[float] = None,
    funnel_stiffness: float = 1.0,
    funnel_radius_exit: Optional[float] = None,
    converge_max_disp_per_100_res: Optional[float] = 0.5,
) -> Tuple[Any, Any]:
    """
    Drop-in hierarchical minimizer. Returns (positions (N, 3), atom_types (N,) z_shell).

    Args:
        sequence: One-letter amino acid sequence.
        include_sidechains: If True, Cβ is added after backbone (same as flat pipeline; side-chain packing optional).
        device: 'cuda', 'tpu', or 'cpu' for JAX placement (when JAX available).
        grouping_strategy: 'residue' | 'ss' | 'domain' | 'helix_unit' (tight H/E segments as single kinetic units).
        ss_string: Optional SS string (H/E/C); if None, predicted.
        domain_ranges: For grouping_strategy='domain', list of (start, end).
        max_iter_stage1/2/3: Iteration limits per stage.
        gtol: Gradient tolerance.
        funnel_radius: Optional cone narrow-end radius (Å) for null-search bound in stages 1-2; None = off.
        funnel_stiffness: Soft-wall stiffness when funnel_radius is set.
        funnel_radius_exit: Cone exit radius (Å); default 2× funnel_radius. Set = funnel_radius for cylinder.
        converge_max_disp_per_100_res: If set (default 0.5), stage-3 Cartesian refinement stops when
            RMS Cα displacement per step < 0.5 * n_res/100 Å (same tolerance as complex minimizer).

    Returns:
        pos: (N, 3) NumPy array in Å (backbone N, CA, C, O per residue).
        atom_types: (N,) z_shell for each atom (for e_tot interop).
    """
    from horizon_physics.proteins.casp_submission import _parse_fasta, AA_1to3
    seq = sequence.strip().upper()
    if not seq or not all(c in AA_1to3 for c in seq):
        seq = _parse_fasta(sequence)
    if not seq:
        return xp.zeros((0, 3), dtype=xp.float64), xp.array([], dtype=xp.int32)
    protein = Protein(
        sequence=seq,
        ss_string=ss_string,
        grouping_strategy=grouping_strategy,
        domain_ranges=domain_ranges,
    )
    n_res = len(seq)
    converge_ang = (converge_max_disp_per_100_res * n_res / 100.0) if converge_max_disp_per_100_res is not None else None
    positions, info = protein.minimize_hierarchical(
        max_iter_stage1=max_iter_stage1,
        max_iter_stage2=max_iter_stage2,
        max_iter_stage3=max_iter_stage3,
        gtol=gtol,
        device=device,
        trajectory_log_path=trajectory_log_path,
        funnel_radius=funnel_radius,
        funnel_stiffness=funnel_stiffness,
        funnel_radius_exit=funnel_radius_exit,
        converge_max_disp_ang=converge_ang,
    )
    # Get z_list from final state (same order as positions: N, CA, C, O per residue)
    _, z_list = protein.forward_kinematics()
    z_list = _be.to_numpy(z_list)
    positions = _be.to_numpy(positions)
    if include_sidechains:
        from horizon_physics.proteins.full_protein_minimizer import _add_cb
        import numpy as np
        n_res = len(seq)
        backbone_atoms = []
        for i in range(n_res):
            for j, name in enumerate(["N", "CA", "C", "O"]):
                backbone_atoms.append((name, np.asarray(positions[i * 4 + j])))
        backbone_atoms = _add_cb(backbone_atoms, seq)
        positions = np.array([xyz for _, xyz in backbone_atoms])
        z_map = {"N": 7, "CA": 6, "C": 6, "O": 8, "CB": 6}
        z_list = np.array([z_map.get(name, 6) for name, _ in backbone_atoms], dtype=np.int32)
    return positions, z_list


def hierarchical_result_for_pdb(
    positions: Any,
    z_list: Any,
    sequence: str,
    include_sidechains: bool,
) -> dict:
    """
    Build a result dict compatible with full_chain_to_pdb from hierarchical output.

    positions: (N, 3), z_list: (N,) — order N, CA, [CB], C, O per residue (CB only for non-Gly).
    Atom list is inferred from len(positions) so we never index out of bounds: if N == 4*n_res
    then backbone only; if N == 5*n_res - num_G then backbone + CB. Returns dict with
    backbone_atoms, sequence, n_res, include_sidechains.
    """
    import numpy as np
    from horizon_physics.proteins.casp_submission import AA_1to3
    seq = sequence.strip().upper()
    n_res = len(seq)
    pos = np.asarray(positions).reshape(-1, 3)
    n_atoms = pos.shape[0]
    # Infer names from actual position count to avoid index-out-of-bounds when
    # include_sidechains was True but positions are backbone-only (e.g. early exit path).
    n_backbone_only = 4 * n_res
    if n_atoms == n_backbone_only:
        names = []
        for _ in range(n_res):
            names.extend(["N", "CA", "C", "O"])
        include_sidechains_out = False
    else:
        names = []
        for i in range(n_res):
            names.extend(["N", "CA"])
            if seq[i] != "G":
                names.append("CB")
            names.extend(["C", "O"])
        # Ensure we have exactly n_atoms names so zip with pos is safe
        if len(names) > n_atoms:
            names = names[:n_atoms]
        elif len(names) < n_atoms:
            names = names + ["CA"] * (n_atoms - len(names))
        include_sidechains_out = include_sidechains
    backbone_atoms = [(names[k], pos[k]) for k in range(n_atoms)]
    return {
        "backbone_atoms": backbone_atoms,
        "sequence": seq,
        "n_res": n_res,
        "include_sidechains": include_sidechains_out,
    }
