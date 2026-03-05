"""
Folding energy E_tot from HQIV: minimization for peptides/proteins.

E_tot = Σ m c² + Σ ħ c / Θ_i (informational-energy) plus geometric damping
f_φ = −γ′ φ / (a_loc + φ/6)² ∇φ and solvent-excluded volume (via φ).
Simple rigorous minimizer: gradient descent on E_tot with Θ_i and φ from
lattice positions. No force fields; all from first principles.

Returns: energy in eV (or relative units), minimized coordinates. MIT License, Python 3.10+, numpy only.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from ._hqiv_base import (
    theta_local,
    horizon_scalar,
    damping_force_magnitude,
    A_LOC_ANG,
    HBAR_C_EV_ANG,
)
from .peptide_backbone import backbone_geometry

def theta_at_position(positions: np.ndarray, i: int, z_shell: int, coordination: int = 2) -> float:
    """
    Θ at node i from position (diamond size from nearest-neighbor distances).
    Θ_local = base Θ scaled by local density: higher density → smaller Θ (monogamy).
    """
    base = theta_local(z_shell, coordination)
    if positions.shape[0] < 2:
        return base
    d = np.linalg.norm(positions - positions[i], axis=1)
    d[i] = np.inf
    r_min = np.min(d)
    # Local crowding: Θ effective = base * min(1, r_min / r_ref)
    r_ref = 2.0
    return base * min(1.0, r_min / r_ref)


# Horizon radius for full vector summation (Å); pairs beyond this don't contribute
R_HORIZON = 15.0
CUTOFF = 12.0  # Å — neighbor list; horizon forces decay rapidly beyond this
USE_NEIGHBOR_LIST = True


# Optional scipy for O(n log n) range queries (neighbor list, clash)
try:
    from scipy.spatial import cKDTree as _cKDTree
    _SCIPY_SPATIAL = True
except ImportError:
    _cKDTree = None
    _SCIPY_SPATIAL = False


# Bond potential vector (pole): direction and magnitude along i→j; − at i, + at j
def _pole(i: int, j: int, vec: np.ndarray) -> Tuple[int, int, np.ndarray]:
    return (i, j, np.asarray(vec, dtype=float))


def build_neighbor_list(pos: np.ndarray, cutoff: float = CUTOFF) -> list:
    """Pairs within cutoff; neigh[i] = [(j, r, unit), ...] for j > i only. Uses cKDTree if scipy available."""
    n = len(pos)
    neigh = [[] for _ in range(n)]
    if _SCIPY_SPATIAL and n > 10:
        tree = _cKDTree(pos)
        for i in range(n):
            idx = tree.query_ball_point(pos[i], cutoff)
            for j in idx:
                if j <= i:
                    continue
                d = pos[j] - pos[i]
                r = np.linalg.norm(d)
                if r < 1e-9 or r >= cutoff:
                    continue
                neigh[i].append((j, r, d / r))
    else:
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[j] - pos[i]
                r = np.linalg.norm(d)
                if r < 1e-9 or r >= cutoff:
                    continue
                unit = d / r
                neigh[i].append((j, r, unit))
    return neigh


def build_horizon_poles(
    positions: np.ndarray,
    z_list: np.ndarray,
    r_ref: float = 2.0,
    r_horizon: float = R_HORIZON,
    k_horizon: float = 0.5 * HBAR_C_EV_ANG,
    use_neighbor_list: bool = USE_NEIGHBOR_LIST,
) -> List[Tuple[int, int, np.ndarray]]:
    """
    Build bond potential vectors (poles) for horizon forces. Each pole is (i, j, vec)
    with vec pointing i→j: force on i is −vec, on j is +vec.
    Returns list of poles for all pairs within cutoff.
    """
    n = positions.shape[0]
    base = theta_local(6, 2)
    cutoff = min(r_horizon, CUTOFF) if use_neighbor_list else r_horizon
    poles: List[Tuple[int, int, np.ndarray]] = []
    if use_neighbor_list:
        neigh = build_neighbor_list(positions, cutoff=cutoff)
        for i in range(n):
            for j, r, unit in neigh[i]:
                theta_ij = base * min(1.0, r / r_ref)
                phi = horizon_scalar(theta_ij)
                pot = k_horizon * phi / (theta_ij + 1e-9)
                vec = pot * unit  # i→j: − at i, + at j
                poles.append(_pole(i, j, vec))
    else:
        for i in range(n):
            for j in range(i + 1, n):
                d = positions[j] - positions[i]
                r = np.linalg.norm(d)
                if r < 1e-9 or r > r_horizon:
                    continue
                unit = d / r
                theta_ij = base * min(1.0, r / r_ref)
                phi = horizon_scalar(theta_ij)
                pot = k_horizon * phi / (theta_ij + 1e-9)
                vec = pot * unit
                poles.append(_pole(i, j, vec))
    return poles


def grad_from_poles(poles: List[Tuple[int, int, np.ndarray]], n: int) -> np.ndarray:
    """Accumulate gradient from stored pole vectors: −vec at i, +vec at j."""
    grad = np.zeros((n, 3))
    for i, j, vec in poles:
        grad[i] -= vec
        grad[j] += vec
    return grad


def grad_horizon_full(
    positions: np.ndarray,
    z_list: np.ndarray,
    r_ref: float = 2.0,
    r_horizon: float = R_HORIZON,
    k_horizon: float = 0.5 * HBAR_C_EV_ANG,
    use_neighbor_list: bool = USE_NEIGHBOR_LIST,
    return_poles: bool = False,
):
    """
    Full vector sum of horizon forces: every atom j contributes to i.
    F_i += pot(r_ij) * unit_vector(i→j) with pot from Θ_ij, φ.
    Repulsive crowding: grad[i] -= pot * unit (push i away from close j).
    With neighbor list (default): O(n·k) for k neighbors within cutoff; 3–8× faster on long chains.
    If return_poles=True, returns (grad, poles) where poles is a list of (i, j, vec) bond potential
    vectors (vec points i→j; − at i, + at j).
    """
    poles = build_horizon_poles(
        positions, z_list, r_ref=r_ref, r_horizon=r_horizon,
        k_horizon=k_horizon, use_neighbor_list=use_neighbor_list,
    )
    n = positions.shape[0]
    grad = grad_from_poles(poles, n)
    if return_poles:
        return grad, poles
    return grad


def e_tot_informational(positions: np.ndarray, z_list: np.ndarray) -> float:
    """
    Σ ħ c / Θ_i for all atoms. positions (n, 3) Å, z_list (n) Z_shell.
    """
    n = positions.shape[0]
    e = 0.0
    for i in range(n):
        theta_i = theta_at_position(positions, i, int(z_list[i]))
        if theta_i > 0:
            e += HBAR_C_EV_ANG / theta_i
    return e


def e_tot_damping(positions: np.ndarray, z_list: np.ndarray, a_loc: float = A_LOC_ANG) -> float:
    """
    Contribution from f_φ: potential energy associated with φ gradient.
    U_φ ∝ ∫ f_φ·dr ~ Σ_i φ_i / (a_loc + φ_i/6) over neighbors.
    """
    n = positions.shape[0]
    u = 0.0
    for i in range(n):
        theta_i = theta_at_position(positions, i, int(z_list[i]))
        phi_i = horizon_scalar(theta_i)
        denom = a_loc + phi_i / 6.0
        if denom > 0:
            u += phi_i / denom
    return u


def e_tot(positions: np.ndarray, z_list: np.ndarray) -> float:
    """
    Total folding energy: E_tot = Σ ħc/Θ_i + λ_damp * U_φ.
    """
    e_info = e_tot_informational(positions, z_list)
    e_damp = e_tot_damping(positions, z_list)
    lambda_damp = 0.1 * HBAR_C_EV_ANG
    return e_info + lambda_damp * e_damp


# Bond/clash from HQIV: Cα–Cα step ~3.8 Å (Θ from lattice); bonding range [r_min, r_max]
R_CA_CA_EQ = 3.8   # Å, from extended (3.2) and helix contour (5.4) compromise
R_BOND_MIN = 2.5   # Å, below = clash
R_BOND_MAX = 6.0   # Å, above = broken chain
R_CLASH = 2.0      # Å, non-bonded atoms closer = clash
K_BOND = 200.0 * HBAR_C_EV_ANG  # strong penalty to keep chain intact
K_BOND_EM_RELAX = 2.5  # eV/Å² for EM-field relax step only; from pyhqiv when available
K_CLASH = 500.0 * HBAR_C_EV_ANG

# Radius-of-gyration collapse bias (optional, for long-chain globule formation)
K_RG_DEFAULT = 0.05 * HBAR_C_EV_ANG  # weight for E_rg = k_rg * Rg² (eV/Å²)
# Stronger weight during two-stage collapse so Rg term can overcome horizon repulsion
K_RG_COLLAPSE = 3.0 * HBAR_C_EV_ANG  # ~6 eV/Å²; use in stage 1 for 404-res globule


def rg_squared(positions: np.ndarray) -> float:
    """Rg² = (1/n) Σ_i |r_i - r_com|². Used for collapse bias (minimize to compactify)."""
    n = positions.shape[0]
    if n < 2:
        return 0.0
    com = np.mean(positions, axis=0)
    return float(np.mean(np.sum((positions - com) ** 2, axis=1)))


def grad_rg_squared(positions: np.ndarray) -> np.ndarray:
    """Gradient of Rg² w.r.t. positions: (2/n)(r_i - r_com). Pulls atoms toward COM."""
    n = positions.shape[0]
    if n < 2:
        return np.zeros_like(positions)
    com = np.mean(positions, axis=0)
    return (2.0 / n) * (positions - com)


def _clash_pairs(
    positions: np.ndarray,
    r_clash: float,
    window: int = 20,
) -> List[Tuple[int, int, float, np.ndarray]]:
    """List of (i, j, r, unit) for non-bonded pairs with r < r_clash, j - i in [2, window]. Uses cKDTree when available."""
    n = positions.shape[0]
    window = min(window, n)
    out: List[Tuple[int, int, float, np.ndarray]] = []
    if _SCIPY_SPATIAL and n > 10:
        tree = _cKDTree(positions)
        for i in range(n):
            idx = tree.query_ball_point(positions[i], r_clash)
            for j in idx:
                if j <= i + 1 or j - i > window:
                    continue
                d = positions[j] - positions[i]
                r = np.linalg.norm(d)
                if r < 1e-12 or r >= r_clash:
                    continue
                out.append((i, j, r, d / r))
    else:
        for j in range(2, window + 1):
            for i in range(n - j):
                d = positions[i + j] - positions[i]
                r = np.linalg.norm(d)
                if r < 1e-12 or r >= r_clash:
                    continue
                out.append((i, i + j, r, d / r))
    return out


def e_tot_ca_with_bonds(
    positions: np.ndarray,
    z_list: np.ndarray,
    r_eq: float = R_CA_CA_EQ,
    r_min: float = R_BOND_MIN,
    r_max: float = R_BOND_MAX,
    r_clash: float = R_CLASH,
    k_bond: float = K_BOND,
    k_clash: float = K_CLASH,
) -> float:
    """
    E_tot + bond-length penalty (consecutive Cα) + clash penalty (non-bonded pairs).
    First principles: atoms "close enough to bond" when r in [r_min, r_max];
    below r_clash for non-bonded = clash. Keeps chain from exploding.
    """
    e = e_tot(positions, z_list)
    n = positions.shape[0]
    # Bond penalty
    d = positions[1:] - positions[:-1]
    r = np.linalg.norm(d, axis=1)
    r = np.where(r < 1e-12, 1.0, r)
    terms = np.where(r < r_min, (r_min - r) ** 2, np.where(r > r_max, (r - r_max) ** 2, 0.1 * (r - r_eq) ** 2))
    e += float(k_bond * np.sum(terms))
    # Clash penalty (O(n log n) with cKDTree when scipy available)
    window = min(20, n)
    for i, j, r, _ in _clash_pairs(positions, r_clash, window):
        e += k_clash * (r_clash - r) ** 2
    return e


def build_bond_poles(
    positions: np.ndarray,
    r_eq: float = R_CA_CA_EQ,
    r_min: float = R_BOND_MIN,
    r_max: float = R_BOND_MAX,
    k_bond: float = K_BOND,
) -> List[Tuple[int, int, np.ndarray]]:
    """
    Bond potential vectors (poles) for consecutive Cα–Cα. Each pole (i, j, vec)
    has vec pointing i→j: force on i is −vec, on j is +vec.
    """
    n = positions.shape[0]
    poles: List[Tuple[int, int, np.ndarray]] = []
    for i in range(n - 1):
        j = i + 1
        d = positions[j] - positions[i]
        r = np.linalg.norm(d)
        if r < 1e-12:
            continue
        u = d / r
        if r < r_min:
            g = -2 * k_bond * (r_min - r)
        elif r > r_max:
            g = 2 * k_bond * (r - r_max)
        else:
            g = 2 * k_bond * 0.1 * (r - r_eq)
        vec = g * u  # i→j
        poles.append(_pole(i, j, vec))
    return poles


def build_bond_poles_segments(
    positions: np.ndarray,
    segment_ends: List[int],
    r_eq: float = R_CA_CA_EQ,
    r_min: float = R_BOND_MIN,
    r_max: float = R_BOND_MAX,
    k_bond: float = K_BOND,
) -> List[Tuple[int, int, np.ndarray]]:
    """
    Bond poles only within segments. segment_ends = [n1, n2, ...] gives exclusive
    end indices (chain 1: 0..n1-1, chain 2: n1..n2-1, ...). No bond between n1-1 and n1.
    """
    n = positions.shape[0]
    poles: List[Tuple[int, int, np.ndarray]] = []
    start = 0
    for end in segment_ends:
        end = min(end, n)
        for i in range(start, end - 1):
            j = i + 1
            d = positions[j] - positions[i]
            r = np.linalg.norm(d)
            if r < 1e-12:
                continue
            u = d / r
            if r < r_min:
                g = -2 * k_bond * (r_min - r)
            elif r > r_max:
                g = 2 * k_bond * (r - r_max)
            else:
                g = 2 * k_bond * 0.1 * (r - r_eq)
            poles.append(_pole(i, j, g * u))
        start = end
    return poles


def grad_bonds_only(
    positions: np.ndarray,
    r_eq: float = R_CA_CA_EQ,
    r_min: float = R_BOND_MIN,
    r_max: float = R_BOND_MAX,
    r_clash: float = R_CLASH,
    k_bond: float = K_BOND,
    k_clash: float = K_CLASH,
    include_clash: bool = False,
) -> np.ndarray:
    """
    Analytical gradient of bond penalty (and optionally clash). O(n) for bonds.
    Used for fast minimization of long chains without finite differences.
    """
    n = positions.shape[0]
    poles = build_bond_poles(positions, r_eq=r_eq, r_min=r_min, r_max=r_max, k_bond=k_bond)
    grad = grad_from_poles(poles, n)
    if include_clash:
        window = min(20, n)
        for i, j, r, u in _clash_pairs(positions, r_clash, window):
            g = -2 * k_clash * (r_clash - r)
            grad[i] -= g * u
            grad[j] += g * u
    return grad


def grad_full(
    positions: np.ndarray,
    z_list: np.ndarray,
    include_bonds: bool = True,
    include_horizon: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Combined gradient: bonds (i,i+1) + full vector sum from all j within horizon.
    Use in fast path for long chains so long-range crowding accumulates.
    """
    grad = np.zeros_like(positions)
    if include_bonds:
        grad += grad_bonds_only(positions, **{k: v for k, v in kwargs.items() if k in ("r_eq", "r_min", "r_max", "r_clash", "k_bond", "k_clash", "include_clash")})
    if include_horizon:
        grad += grad_horizon_full(positions, z_list, **{k: v for k, v in kwargs.items() if k in ("r_ref", "r_horizon", "k_horizon")})
    return grad


def minimize_e_tot(
    positions_init: np.ndarray,
    z_list: np.ndarray,
    steps: int = 200,
    step_size: float = 0.01,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Deterministic gradient descent on E_tot (no random seed). positions_init (n, 3) Å, z_list (n).
    Returns (positions_opt, {"e_final": ..., "e_initial": ...}).
    For L-BFGS (recommended), use gradient_descent_folding.minimize_e_tot_lbfgs.
    """
    pos = np.array(positions_init, dtype=float)
    e0 = e_tot(pos, z_list)
    n = pos.shape[0]
    for _ in range(steps):
        grad = np.zeros_like(pos)
        for j in range(n):
            for d in range(3):
                pos[j, d] += 1e-5
                e_plus = e_tot(pos, z_list)
                pos[j, d] -= 2e-5
                e_minus = e_tot(pos, z_list)
                pos[j, d] += 1e-5
                grad[j, d] = (e_plus - e_minus) / (2e-5)
        pos -= step_size * grad
    e_final = e_tot(pos, z_list)
    return pos, {"e_final": e_final, "e_initial": e0}


def small_peptide_energy(sequence: str) -> Dict[str, float]:
    """
    E_tot for a small peptide: backbone-only Cα trace, Z=6 for Cα.
    sequence: one-letter amino acids. Returns {"e_tot": ..., "per_residue": ...}.
    """
    n = len(sequence)
    # Simple linear chain spacing 3.8 Å (extended)
    positions = np.zeros((n, 3))
    positions[:, 0] = np.arange(n) * 3.8
    z_list = np.full(n, 6)
    e = e_tot(positions, z_list)
    return {"e_tot": e, "per_residue": e / n if n else 0.0}


if __name__ == "__main__":
    # 3 Cα chain
    pos0 = np.array([[0.0, 0, 0], [3.8, 0, 0], [7.6, 0, 0]])
    z = np.array([6, 6, 6])
    pos_opt, info = minimize_e_tot(pos0, z, steps=100)
    print("Folding energy (HQIV E_tot minimizer)")
    print(f"  E_initial: {info['e_initial']:.2f} eV, E_final: {info['e_final']:.2f} eV")
    pep = small_peptide_energy("AAA")
    print(f"  Small peptide AAA: E_tot={pep['e_tot']:.2f} eV")
