"""
Scalar collective-mode hooks aligned with Lean ``Hqiv.Physics.HQIVCollectiveModes``.

Proofs and definitions: ``HQIV_LEAN/Hqiv/Physics/HQIVCollectiveModes.lean`` (``K_multipole ≡ K_hbond``).
Use these numbers in Metropolis / gradient code; RMSD anchors are symbolic only (see Lean module doc).

Cα **kink budget** (optional folding term): per interior residue, measure virtual bond-angle
deviation from a reference (default ~ideal α-helix Cα–Cα–Cα geometry) and accumulate
``K_multipole m * helixKinkMeasure(|Δθ|)``, matching the scalar budget in
``helixMultipoleKinkBudget`` (additive penalty; straight / on-reference segments minimize
budget — cf. ``macroscopic_snap_global_minimum_straight`` in Lean).
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence

import numpy as np

# Definitional anchors in Lean (not dynamical RMSD theorems)
ANNEAL_TEMPERATURE_KELVIN: float = 310.0
KINK_RMSD_UPPER_ANGSTROM: float = 2.0
PDB_9GGO_A_RMSD_UPPER_A: float = 7.0


def k_multipole(m: int) -> float:
    """Same vacuum ladder as long-range contact: ``K_hbond m`` (Lean ``K_multipole``)."""
    from .hqiv_long_range import K_hbond

    return float(K_hbond(m))


def quadrupole_dot_gradient_proxy(
    num_sites: int,
    quadrupole_gradient_coupling: Callable[[float], float],
    *,
    axis_s: float = 0.0,
) -> float:
    """Scalar ``Q * Φ'(axis)`` with ``Q`` proxy = site count (Lean ``computeQuadrupole``)."""
    return float(num_sites) * float(quadrupole_gradient_coupling(float(axis_s)))


def helix_multipole_torque(
    m: int,
    num_sites: int,
    quadrupole_gradient_coupling: Callable[[float], float],
    *,
    axis_s: float = 0.0,
) -> float:
    """Lean ``helixMultipoleTorque``: ``-K_multipole m * quadrupoleDotGradient``."""
    return -k_multipole(m) * quadrupole_dot_gradient_proxy(
        num_sites, quadrupole_gradient_coupling, axis_s=axis_s
    )


def collective_relax_scalar(m: int, coupling: float, dt: float) -> float:
    """Lean ``collectiveRelaxScalar`` for unit-``Q`` proxy: ``(-K_multipole * coupling) * dt``."""
    return (-k_multipole(m) * float(coupling)) * float(dt)


def collective_relax_helix_torques_dt(
    m: int,
    helix_site_counts: Sequence[int],
    quadrupole_gradient_coupling: Callable[[float], float],
    dt: float,
    *,
    axis_s: float = 0.0,
) -> List[float]:
    """Lean ``collectiveRelaxHelixList``: one ``torque * dt`` per helix (linear in list length)."""
    return [
        helix_multipole_torque(m, int(n), quadrupole_gradient_coupling, axis_s=axis_s) * float(dt)
        for n in helix_site_counts
    ]


def helix_kink_measure(delta: float) -> float:
    """Lean ``helixKinkMeasure``: ``max(δ, 0)``."""
    return max(float(delta), 0.0)


def helix_multipole_kink_budget(m: int, delta: float) -> float:
    """Lean ``helixMultipoleKinkBudget``: ``K_multipole m * helixKinkMeasure δ``."""
    return k_multipole(m) * helix_kink_measure(delta)


def ca_virtual_bond_angle_rad(ca: np.ndarray, i: int) -> float:
    """
    Internal angle at Cα ``i`` from segments (i−1)→i and i→(i+1), radians.
    Undefined for chain ends; returns ``nan`` if degenerate vectors.
    """
    ca = np.asarray(ca, dtype=float)
    n = ca.shape[0]
    if i <= 0 or i >= n - 1:
        return float("nan")
    u = ca[i] - ca[i - 1]
    v = ca[i + 1] - ca[i]
    lu, lv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if lu < 1e-14 or lv < 1e-14:
        return float("nan")
    c = float(np.clip(np.dot(u, v) / (lu * lv), -1.0, 1.0))
    return float(math.acos(c))


def e_ca_collective_kink_sum(
    ca: np.ndarray,
    m: int,
    *,
    theta_ref_rad: float,
    ss_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Σ ``helixMultipoleKinkBudget(m, |θᵢ − θ_ref|)`` over interior vertices.

    ``ss_mask``: optional length-``n`` bool; if set, only residues ``i`` with ``mask[i]``
    contribute (e.g. helix sites in the sense of ``HelixSites``).
    """
    ca = np.asarray(ca, dtype=float)
    n = ca.shape[0]
    if n < 3:
        return 0.0
    mask = ss_mask
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        # Segment-level evaluations (e.g., tunnel subproblems) may pass a full-chain mask.
        # In that case, drop the mask rather than crashing; caller can still use unmasked term.
        if mask.shape[0] != n:
            mask = None
    total = 0.0
    for i in range(1, n - 1):
        if mask is not None and not bool(mask[i]):
            continue
        th = ca_virtual_bond_angle_rad(ca, i)
        if not math.isfinite(th):
            continue
        delta = abs(float(th) - float(theta_ref_rad))
        total += helix_multipole_kink_budget(m, delta)
    return float(total)


def default_collective_kink_theta_ref_rad() -> float:
    """
    Reference Cα bend angle from canonical HQIV α-helix geometry (``alpha_helix_xyz``).
    Use as ``theta_ref_rad`` so ideal helical traces sit near the kink minimum.
    """
    from .alpha_helix import alpha_helix_xyz

    ca = alpha_helix_xyz(np.arange(12))
    th = ca_virtual_bond_angle_rad(ca, 5)
    if not math.isfinite(th):
        return math.radians(52.0)
    return float(th)


def grad_e_ca_collective_kink_fd(
    ca: np.ndarray,
    m: int,
    *,
    theta_ref_rad: float,
    ss_mask: Optional[np.ndarray] = None,
    eps: float = 1e-3,
) -> np.ndarray:
    """Central finite-difference gradient of ``e_ca_collective_kink_sum`` w.r.t. Cα."""
    ca = np.asarray(ca, dtype=float)
    n = ca.shape[0]
    g = np.zeros_like(ca)
    for i in range(n):
        for d in range(3):
            ca_p = ca.copy()
            ca_p[i, d] += eps
            ca_m = ca.copy()
            ca_m[i, d] -= eps
            ep = e_ca_collective_kink_sum(ca_p, m, theta_ref_rad=theta_ref_rad, ss_mask=ss_mask)
            em = e_ca_collective_kink_sum(ca_m, m, theta_ref_rad=theta_ref_rad, ss_mask=ss_mask)
            g[i, d] = (ep - em) / (2.0 * eps)
    return g
