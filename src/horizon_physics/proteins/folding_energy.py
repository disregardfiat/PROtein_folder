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
from typing import Any, Dict, List, Optional, Tuple

from ._hqiv_base import (
    theta_local,
    horizon_scalar,
    damping_force_magnitude,
    A_LOC_ANG,
    HBAR_C_EV_ANG,
)
from .peptide_backbone import backbone_geometry

def theta_from_nn_distance(r_nn: float, z_shell: int, coordination: int = 2) -> float:
    """
    Θ_i from nearest-neighbor distance r_nn (same crowding rule as ``theta_at_position``).
    """
    base = theta_local(z_shell, coordination)
    r_ref = 2.0
    if not np.isfinite(r_nn) or r_nn >= 1e30:
        return base
    return base * min(1.0, float(r_nn) / r_ref)


def _per_atom_nearest_neighbor_distance(positions: np.ndarray) -> np.ndarray:
    """
    For each atom, distance to the nearest *other* atom. Same as the inner loop of
    ``theta_at_position`` but O(n log n) with cKDTree when scipy is available and n is large.
    """
    n = int(positions.shape[0])
    if n <= 1:
        return np.full(n, np.inf, dtype=float)
    if _SCIPY_SPATIAL and n > 16:
        tree = _cKDTree(positions)
        kq = min(2, n)
        dists, _ = tree.query(positions, k=kq)
        if kq == 1:
            return np.full(n, np.inf, dtype=float)
        return np.asarray(dists[:, 1], dtype=float)
    out = np.empty(n, dtype=float)
    for i in range(n):
        d = np.linalg.norm(positions - positions[i], axis=1)
        d[i] = np.inf
        out[i] = float(np.min(d))
    return out


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
    return theta_from_nn_distance(r_min, z_shell, coordination)


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
    em_scale: float = 1.0,
    neighbor_cutoff: Optional[float] = None,
) -> List[Tuple[int, int, np.ndarray]]:
    """
    Build bond potential vectors (poles) for horizon forces. Each pole is (i, j, vec)
    with vec pointing i→j: force on i is −vec, on j is +vec.
    Returns list of poles for all pairs within cutoff.

    em_scale: Multiplier on the horizon pole strength (default 1). For aqueous solvent,
    use 1/ε_r to match HQIV Lean `waterDielectricValley` (EM term ∝ 1/(ε_r r)).
    neighbor_cutoff: When ``use_neighbor_list`` is True, cap spatial neighbor search at this
    distance (Å); default is module ``CUTOFF``. Smaller values prune weak long pairs (faster).
    """
    n = positions.shape[0]
    base = theta_local(6, 2)
    cap = float(neighbor_cutoff) if neighbor_cutoff is not None else CUTOFF
    cutoff = min(r_horizon, cap) if use_neighbor_list else r_horizon
    poles: List[Tuple[int, int, np.ndarray]] = []
    if use_neighbor_list:
        neigh = build_neighbor_list(positions, cutoff=cutoff)
        for i in range(n):
            for j, r, unit in neigh[i]:
                theta_ij = base * min(1.0, r / r_ref)
                phi = horizon_scalar(theta_ij)
                pot = em_scale * k_horizon * phi / (theta_ij + 1e-9)
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
                pot = em_scale * k_horizon * phi / (theta_ij + 1e-9)
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
    em_scale: float = 1.0,
    neighbor_cutoff: Optional[float] = None,
):
    """
    Full vector sum of horizon forces: every atom j contributes to i.
    F_i += pot(r_ij) * unit_vector(i→j) with pot from Θ_ij, φ.
    Repulsive crowding: grad[i] -= pot * unit (push i away from close j).
    With neighbor list (default): O(n·k) for k neighbors within cutoff; 3–8× faster on long chains.
    If return_poles=True, returns (grad, poles) where poles is a list of (i, j, vec) bond potential
    vectors (vec points i→j; − at i, + at j).

    em_scale: Same as in build_horizon_poles (e.g. 1/ε_r in water).
    neighbor_cutoff: Optional tighter neighbor-list radius (Å); see ``build_horizon_poles``.
    """
    poles = build_horizon_poles(
        positions, z_list, r_ref=r_ref, r_horizon=r_horizon,
        k_horizon=k_horizon, use_neighbor_list=use_neighbor_list, em_scale=em_scale,
        neighbor_cutoff=neighbor_cutoff,
    )
    n = positions.shape[0]
    grad = grad_from_poles(poles, n)
    if return_poles:
        return grad, poles
    return grad


def e_tot_informational(
    positions: np.ndarray,
    z_list: np.ndarray,
    *,
    fast_local_theta: bool = False,
) -> float:
    """
    Σ ħ c / Θ_i for all atoms. positions (n, 3) Å, z_list (n) Z_shell.

    fast_local_theta: Use batched nearest-neighbor distances (same Θ_i values, faster for long chains).
    """
    n = positions.shape[0]
    r_nn = _per_atom_nearest_neighbor_distance(positions) if fast_local_theta and n > 1 else None
    e = 0.0
    for i in range(n):
        if r_nn is not None:
            theta_i = theta_from_nn_distance(r_nn[i], int(z_list[i]))
        else:
            theta_i = theta_at_position(positions, i, int(z_list[i]))
        if theta_i > 0:
            e += HBAR_C_EV_ANG / theta_i
    return e


def e_tot_damping(
    positions: np.ndarray,
    z_list: np.ndarray,
    a_loc: float = A_LOC_ANG,
    *,
    fast_local_theta: bool = False,
) -> float:
    """
    Contribution from f_φ: potential energy associated with φ gradient.
    U_φ ∝ ∫ f_φ·dr ~ Σ_i φ_i / (a_loc + φ_i/6) over neighbors.
    """
    n = positions.shape[0]
    r_nn = _per_atom_nearest_neighbor_distance(positions) if fast_local_theta and n > 1 else None
    u = 0.0
    for i in range(n):
        if r_nn is not None:
            theta_i = theta_from_nn_distance(r_nn[i], int(z_list[i]))
        else:
            theta_i = theta_at_position(positions, i, int(z_list[i]))
        phi_i = horizon_scalar(theta_i)
        denom = a_loc + phi_i / 6.0
        if denom > 0:
            u += phi_i / denom
    return u


def e_tot(
    positions: np.ndarray,
    z_list: np.ndarray,
    *,
    fast_local_theta: bool = False,
) -> float:
    """
    Total folding energy: E_tot = Σ ħc/Θ_i + λ_damp * U_φ.
    """
    e_info = e_tot_informational(positions, z_list, fast_local_theta=fast_local_theta)
    e_damp = e_tot_damping(positions, z_list, fast_local_theta=fast_local_theta)
    lambda_damp = 0.1 * HBAR_C_EV_ANG
    return e_info + lambda_damp * e_damp


def e_atomic_site(
    positions: np.ndarray,
    z_list: np.ndarray,
    i: int,
    a_loc: float = A_LOC_ANG,
) -> float:
    """
    Per-site contribution to ``e_tot``: ħc/Θ_i + λ_damp * (φ_i / (a_loc + φ_i/6)).

    Summing ``e_atomic_site(..., i)`` over i equals ``e_tot(positions, z_list)`` (same formulas as
    ``e_tot_informational`` / ``e_tot_damping``). This is the Python analogue of an atomic / site
    term in ``assembly_foldEnergy_branch_eq`` (Lean ``Hqiv.Physics.HQIVMolecules``).
    """
    theta_i = theta_at_position(positions, i, int(z_list[i]))
    e_info = HBAR_C_EV_ANG / theta_i if theta_i > 0 else 0.0
    phi_i = horizon_scalar(theta_i)
    denom = a_loc + phi_i / 6.0
    u = phi_i / denom if denom > 0 else 0.0
    lambda_damp = 0.1 * HBAR_C_EV_ANG
    return float(e_info + lambda_damp * u)


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


def bond_valley_em_scalar(
    r: float,
    *,
    r_eq: float = R_CA_CA_EQ,
    r_min: float = R_BOND_MIN,
    r_max: float = R_BOND_MAX,
    k_bond: float = K_BOND,
) -> float:
    """
    Scalar Cα–Cα bond penalty for one edge distance r (same piecewise form as in ``e_tot_ca_with_bonds``).

    Matches the HQIVMolecules ``bondValleyEM`` / ``sumValleyPotentialEM`` edge contribution in the
    Python discretization (quadratic well outside [r_min, r_max], softer well inside).
    """
    if r < 1e-12:
        r = 1.0
    if r < r_min:
        return float(k_bond * (r_min - r) ** 2)
    if r > r_max:
        return float(k_bond * (r - r_max) ** 2)
    return float(k_bond * 0.1 * (r - r_eq) ** 2)


def e_sequential_bond_penalty_ca(
    positions: np.ndarray,
    *,
    r_eq: float = R_CA_CA_EQ,
    r_min: float = R_BOND_MIN,
    r_max: float = R_BOND_MAX,
    k_bond: float = K_BOND,
) -> float:
    """Sum of ``bond_valley_em_scalar`` over consecutive Cα pairs (i, i+1)."""
    n = positions.shape[0]
    if n < 2:
        return 0.0
    d = positions[1:] - positions[:-1]
    r = np.linalg.norm(d, axis=1)
    r = np.where(r < 1e-12, 1.0, r)
    terms = np.where(
        r < r_min,
        (r_min - r) ** 2,
        np.where(r > r_max, (r - r_max) ** 2, 0.1 * (r - r_eq) ** 2),
    )
    return float(k_bond * np.sum(terms))


def e_clash_penalty_ca_windowed(
    positions: np.ndarray,
    *,
    r_clash: float = R_CLASH,
    k_clash: float = K_CLASH,
    window: int = 20,
) -> float:
    """Clash penalty matching ``e_tot_ca_with_bonds`` (non-bonded pairs within window)."""
    e = 0.0
    for i, j, r, _ in _clash_pairs(positions, r_clash, window):
        e += k_clash * (r_clash - r) ** 2
    return float(e)


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
    *,
    fast_local_theta: bool = False,
    collective_kink_weight: float = 0.0,
    collective_kink_m: int = 3,
    collective_kink_theta_ref_rad: Optional[float] = None,
    collective_kink_ss_mask: Optional[Any] = None,
) -> float:
    """
    E_tot + bond-length penalty (consecutive Cα) + clash penalty (non-bonded pairs).
    First principles: atoms "close enough to bond" when r in [r_min, r_max];
    below r_clash for non-bonded = clash. Keeps chain from exploding.

    fast_local_theta: Faster Σ ħc/Θ_i + damping (batched NN distances; numerically same Θ_i).

    collective_kink_weight: If > 0, add Lean-aligned Cα kink budget (``HQIVCollectiveModes``):
        weight × Σ ``K_multipole(m) * helixKinkMeasure(|θᵢ − θ_ref|)`` over interior sites
        (optionally SS-masked). Default ``θ_ref`` from ``default_collective_kink_theta_ref_rad``.
    """
    e = e_tot(positions, z_list, fast_local_theta=fast_local_theta)
    n = positions.shape[0]
    e += e_sequential_bond_penalty_ca(
        positions, r_eq=r_eq, r_min=r_min, r_max=r_max, k_bond=k_bond
    )
    window = min(20, n)
    e += e_clash_penalty_ca_windowed(
        positions, r_clash=r_clash, k_clash=k_clash, window=window
    )
    ck_w = float(collective_kink_weight)
    if ck_w != 0.0 and n >= 3:
        from .collective_modes_scalars import (
            default_collective_kink_theta_ref_rad,
            e_ca_collective_kink_sum,
        )

        tr = collective_kink_theta_ref_rad
        if tr is None:
            tr = default_collective_kink_theta_ref_rad()
        e += ck_w * e_ca_collective_kink_sum(
            positions,
            int(collective_kink_m),
            theta_ref_rad=float(tr),
            ss_mask=collective_kink_ss_mask,
        )
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

    Optional HQIV long-range H-bond proxy (Lean ``HQIVLongRange``), additive on Cα:

    - ``hbond_weight`` (default 0): if > 0, add finite-difference gradient of
      ``total_h_bond_proxy_energy_ca`` (``hbond_shell_m``, ``hbond_min_seq_sep``, …).

    Optional collective kink budget (Lean ``HQIVCollectiveModes``): ``collective_kink_weight``,
    ``collective_kink_m``, ``collective_kink_theta_ref_rad``, ``collective_kink_ss_mask``.
    """
    grad = np.zeros_like(positions)
    if include_bonds:
        grad += grad_bonds_only(positions, **{k: v for k, v in kwargs.items() if k in ("r_eq", "r_min", "r_max", "r_clash", "k_bond", "k_clash", "include_clash")})
    if include_horizon:
        grad += grad_horizon_full(
            positions,
            z_list,
            **{k: v for k, v in kwargs.items() if k in ("r_ref", "r_horizon", "k_horizon", "em_scale", "use_neighbor_list", "neighbor_cutoff")},
        )
    hb_w = float(kwargs.get("hbond_weight", 0.0))
    if hb_w != 0.0:
        from .hqiv_long_range import grad_h_bond_proxy_ca_fd

        m_shell = int(kwargs.get("hbond_shell_m", 3))
        grad = grad + hb_w * grad_h_bond_proxy_ca_fd(
            positions,
            m_shell,
            min_seq_sep=int(kwargs.get("hbond_min_seq_sep", 3)),
            max_pairs=int(kwargs.get("hbond_max_pairs", 200)),
            dist_cutoff=float(kwargs.get("hbond_dist_cutoff", 15.0)),
        )
    ck_w = float(kwargs.get("collective_kink_weight", 0.0))
    if ck_w != 0.0 and positions.shape[0] >= 3:
        from .collective_modes_scalars import (
            default_collective_kink_theta_ref_rad,
            grad_e_ca_collective_kink_fd,
        )

        m_ck = int(kwargs.get("collective_kink_m", 3))
        tr = kwargs.get("collective_kink_theta_ref_rad")
        if tr is None:
            tr = default_collective_kink_theta_ref_rad()
        else:
            tr = float(tr)
        mask = kwargs.get("collective_kink_ss_mask")
        grad = grad + ck_w * grad_e_ca_collective_kink_fd(
            positions,
            m_ck,
            theta_ref_rad=tr,
            ss_mask=mask,
        )
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
