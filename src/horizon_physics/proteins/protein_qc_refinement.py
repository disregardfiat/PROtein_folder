"""
Post–Cα QC refinement energy aligned with Lean ``Hqiv/ProteinResearch/ProteinQCRefinement.lean``.

- **Site trace:** ``Σ latticeFullModeEnergy(shell_i)`` (independent of coordinates in the refinement layer).
- **Soft clash:** unordered pairs ``i < j``, if ``d < σ`` then ``(σ - d)²``, else ``0``.
- **Total:** ``E = qc_site_energy + w * qc_soft_clash`` (Lean ``qcRefinementEnergy``).

The site term has zero gradient w.r.t. positions; rigid-body ligand refinement uses ``w * ∇ qc_soft_clash``.

MIT License. Python 3.10+. NumPy; optional SciPy (``cKDTree``) for fast protein–ligand QC clash.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .horizon_qed_bookkeeping import phi_of_shell, shell_spatial_mode_count

try:
    from scipy.spatial import cKDTree as _cKDTree

    _HAS_SCIPY = True
except ImportError:
    _cKDTree = None
    _HAS_SCIPY = False


def lattice_full_mode_energy_ev(shell: int) -> float:
    """Lean bridge: ``available_modes * (phi_of_shell / 2)`` (eV-scale bookkeeping)."""
    m = int(max(0, shell))
    return float(shell_spatial_mode_count(m) * (phi_of_shell(m) / 2.0))


def qc_site_energy_trace_ev(z_list: np.ndarray) -> float:
    """``qcSiteEnergy`` from Lean: sum of ``latticeFullModeEnergy`` per site."""
    z = np.asarray(z_list, dtype=np.int64).ravel()
    return float(sum(lattice_full_mode_energy_ev(int(z[i])) for i in range(z.size)))


def soft_repulsion_from_dist(d: float, sigma: float) -> float:
    """Lean ``softRepulsionFromDist``."""
    if d < float(sigma):
        return (float(sigma) - float(d)) ** 2
    return 0.0


def qc_soft_clash_energy(positions: np.ndarray, sigma: float) -> float:
    """Lean ``qcSoftClashEnergy`` over all unordered pairs ``i < j``."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n = int(pos.shape[0])
    if n < 2:
        return 0.0
    sig = float(sigma)
    acc = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pos[j] - pos[i]))
            if d < 1e-12:
                continue
            acc += soft_repulsion_from_dist(d, sig)
    return float(acc)


def grad_qc_soft_clash(positions: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gradient of ``qc_soft_clash_energy`` w.r.t. all site coordinates.

    For ``d < σ``: ``E = (σ - d)²``, ``∂E/∂r_i = 2(σ - d) * (r_j - r_i) / d``.
    """
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n = int(pos.shape[0])
    g = np.zeros_like(pos)
    if n < 2:
        return g
    sig = float(sigma)
    for i in range(n):
        for j in range(i + 1, n):
            dvec = pos[j] - pos[i]
            d = float(np.linalg.norm(dvec))
            if d < 1e-12 or d >= sig:
                continue
            u = dvec / d
            c = 2.0 * (sig - d)
            g[i] += c * u
            g[j] -= c * u
    return g


def qc_soft_clash_energy_protein_ligand(
    pos_protein: np.ndarray,
    pos_ligand: np.ndarray,
    sigma: float,
) -> float:
    """
    Soft clash energy for **fixed protein** + **mobile ligand**: protein–protein terms omitted (constant).

    Includes protein–ligand and ligand–ligand pairs (Lean ``qcSoftClashEnergy`` on the moving subspace).
    """
    pp = np.asarray(pos_protein, dtype=np.float64).reshape(-1, 3)
    pl = np.asarray(pos_ligand, dtype=np.float64).reshape(-1, 3)
    n_p, n_l = int(pp.shape[0]), int(pl.shape[0])
    sig = float(sigma)
    acc = 0.0
    for i in range(n_l):
        ri = pl[i]
        for j in range(n_p):
            d = float(np.linalg.norm(pp[j] - ri))
            if d < 1e-12:
                continue
            acc += soft_repulsion_from_dist(d, sig)
    for i in range(n_l):
        for j in range(i + 1, n_l):
            d = float(np.linalg.norm(pl[j] - pl[i]))
            if d < 1e-12:
                continue
            acc += soft_repulsion_from_dist(d, sig)
    return float(acc)


def grad_qc_soft_clash_protein_fixed(
    pos_protein: np.ndarray,
    pos_ligand: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Gradient of protein–ligand + ligand–ligand soft clash w.r.t. **ligand** coordinates only.

    Use when the protein is rigid and only ligand sites move (same physics as full gradient on the
    combined vector with protein rows zeroed).
    """
    pp = np.asarray(pos_protein, dtype=np.float64).reshape(-1, 3)
    pl = np.asarray(pos_ligand, dtype=np.float64).reshape(-1, 3)
    n_p, n_l = int(pp.shape[0]), int(pl.shape[0])
    g = np.zeros_like(pl)
    if n_l < 1:
        return g
    sig = float(sigma)
    if _HAS_SCIPY and n_p > 32:
        tree = _cKDTree(pp)
        for i in range(n_l):
            for j in tree.query_ball_point(pl[i], r=sig):
                dvec = pp[j] - pl[i]
                d = float(np.linalg.norm(dvec))
                if d < 1e-12 or d >= sig:
                    continue
                u = dvec / d
                g[i] += 2.0 * (sig - d) * u
    else:
        for i in range(n_l):
            for j in range(n_p):
                dvec = pp[j] - pl[i]
                d = float(np.linalg.norm(dvec))
                if d < 1e-12 or d >= sig:
                    continue
                u = dvec / d
                g[i] += 2.0 * (sig - d) * u
    for i in range(n_l):
        for j in range(i + 1, n_l):
            dvec = pl[j] - pl[i]
            d = float(np.linalg.norm(dvec))
            if d < 1e-12 or d >= sig:
                continue
            u = dvec / d
            c = 2.0 * (sig - d)
            g[i] += c * u
            g[j] -= c * u
    return g


def qc_refinement_energy(
    z_list: np.ndarray,
    positions: np.ndarray,
    sigma: float,
    w_clash: float,
) -> float:
    """Lean ``qcRefinementEnergy`` (site + weighted clash)."""
    w = float(max(0.0, w_clash))
    return qc_site_energy_trace_ev(z_list) + w * qc_soft_clash_energy(positions, sigma)


def load_ligand_agents_from_pdb_file(
    path: str,
    *,
    skip_resnames: Optional[Tuple[str, ...]] = ("HOH", "WAT", "SOL"),
) -> List[Any]:
    """
    Build :class:`~horizon_physics.proteins.ligands.LigandAgent` instances from HETATM records.

    One agent per (residue name, residue sequence, chain) hetero group; waters skipped.
    """
    from .ligands import LigandAgent

    skip = {s.upper() for s in (skip_resnames or ())}
    records: List[Tuple[str, int, str, str, np.ndarray, str]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith("HETATM"):
                continue
            if len(line) < 54:
                continue
            try:
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip() or "LIG"
                if res_name.upper() in skip:
                    continue
                res_seq = int(line[22:26].strip() or 0)
                chain = line[21] if len(line) > 21 else " "
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                if len(line) >= 78:
                    elem = line[76:78].strip()
                else:
                    tail = line[54:].split()
                    elem = tail[-1] if tail and tail[-1].isalpha() and len(tail[-1]) <= 2 else ""
                if not elem:
                    elem = "C"
                records.append(
                    (res_name, res_seq, chain, atom_name, np.array([x, y, z], dtype=np.float64), elem)
                )
            except (ValueError, IndexError):
                continue

    if not records:
        return []

    by_key: Dict[Tuple[str, int, str], List[Tuple[str, np.ndarray, str]]] = {}
    for res_name, res_seq, chain, atom_name, xyz, elem in records:
        key = (res_name.strip(), int(res_seq), chain)
        by_key.setdefault(key, []).append((atom_name, xyz, elem))

    agents: List[LigandAgent] = []
    for res_key, atoms in sorted(by_key.items(), key=lambda kv: kv[0]):
        positions = np.array([a[1] for a in atoms])
        centroid = np.mean(positions, axis=0)
        local = positions - centroid
        atom_names = [a[0] for a in atoms]
        elements = [a[2] for a in atoms]
        ag = LigandAgent(
            res_name=res_key[0],
            atom_names=atom_names,
            local_positions=local,
            elements=elements,
        )
        ag.set_6dof(centroid, np.zeros(3))
        agents.append(ag)
    return agents


def ligand_heavy_atom_rmsd(
    pred_xyz: np.ndarray,
    ref_xyz: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    RMSD (Å) after Kabsch alignment of ``pred_xyz`` onto ``ref_xyz`` (same atom count).
    Returns (rmsd, pred_aligned).
    """
    from .grade_folds import kabsch_superpose

    P = np.asarray(ref_xyz, dtype=np.float64).reshape(-1, 3)
    Q = np.asarray(pred_xyz, dtype=np.float64).reshape(-1, 3)
    if P.shape != Q.shape or P.shape[0] == 0:
        raise ValueError("ligand_heavy_atom_rmsd: shape mismatch or empty")
    _R, _t, Qa = kabsch_superpose(P, Q)
    d = np.linalg.norm(P - Qa, axis=1)
    return float(np.sqrt(np.mean(d * d))), Qa
