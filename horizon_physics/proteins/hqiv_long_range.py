"""
HQIV long-range H-bond / contact proxy (Lean: Hqiv/Physics/HQIVLongRange.lean).

Vacuum ladder only — **no fitted stiffness K**. Scales match Lean:

  - ``K_hbond m = phi_of_shell m / availableModesNat m`` (= ``omegaCasimir m / modes``)
  - ``R_hbond m = R_m m = (m : ℝ) + 1``
  - ``hBondProxy m θ φ dist = -K_hbond m * (cos θ * cos φ) / (1 + (dist / R_hbond m)²)``

``valleyPotentialLongRange = valleyPotentialEM + hBondProxy`` (additive). Fold dihedral ``θFold``
in ``foldEnergyWithDihedral`` is separate from contact angles ``(θ, φ)`` here.

Design notes (see Lean ``long_range_attraction_emergent``): when ``cos θ > 0`` and ``cos φ > 0``,
``hBondProxy < 0`` (strictly attractive contribution). Tetrahedral / 109.5° chemistry is **not**
hard-coded; angle convention is geometric (see ``contact_alignment_angles_ca``).

Multibody / covalent bookkeeping is documented in Lean ``MultibodyAttachment``; Python uses the
same shell indices for ``m`` when pairing atoms to a Casimir shell.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

# Lean: phiTemperatureCoeff = 2 (AuxiliaryField.lean)
PHI_TEMPERATURE_COEFF = 2.0


def lattice_simplex_count(m: int) -> int:
    """Lean ``latticeSimplexCount m = (m + 2) * (m + 1)`` (OctonionicLightCone)."""
    if m < 0:
        raise ValueError("shell index m must be nonnegative")
    return (m + 2) * (m + 1)


def available_modes_nat(m: int) -> int:
    """Lean ``availableModesNat m = 4 * latticeSimplexCount m``."""
    return 4 * lattice_simplex_count(m)


def phi_of_shell(m: int) -> float:
    """Lean closed form ``phi_of_shell m = phiTemperatureCoeff * (m + 1)``."""
    if m < 0:
        raise ValueError("shell index m must be nonnegative")
    return PHI_TEMPERATURE_COEFF * float(m + 1)


def omega_casimir(m: int) -> float:
    """Lean ``omegaCasimir m = phi_of_shell m``."""
    return phi_of_shell(m)


def R_m(m: int) -> float:
    """Lean ``R_m m = (m : ℝ) + 1`` (NuclearAndAtomicSpectra)."""
    if m < 0:
        raise ValueError("shell index m must be nonnegative")
    return float(m) + 1.0


def K_hbond(m: int) -> float:
    """Lean ``K_hbond m = phi_of_shell m / (availableModesNat m : ℝ)`` — no fitted constant."""
    nu = float(available_modes_nat(m))
    if nu <= 0:
        raise ValueError("available_modes_nat must be positive")
    return phi_of_shell(m) / nu


def R_hbond(m: int) -> float:
    """Lean ``R_hbond m = R_m m``."""
    return R_m(m)


def h_bond_proxy(m: int, theta_rad: float, phi_rad: float, dist: float) -> float:
    """
    Lean ``hBondProxy``: Lorentzian radial factor in ``dist / R_hbond m``,
    angular factor ``cos θ * cos φ`` (contact channels; not fold ``θFold``).
    """
    if dist < 0:
        raise ValueError("dist must be nonnegative")
    k = K_hbond(m)
    rh = R_hbond(m)
    if rh <= 0:
        raise ValueError("R_hbond must be positive")
    den = 1.0 + (dist / rh) ** 2
    return -k * (math.cos(theta_rad) * math.cos(phi_rad)) / den


def valley_potential_long_range_scalar(
    valley_em: float,
    m: int,
    theta_rad: float,
    phi_rad: float,
    dist: float,
) -> float:
    """Lean ``valleyPotentialLongRange = valleyPotentialEM + hBondProxy`` (scalar pieces)."""
    return valley_em + h_bond_proxy(m, theta_rad, phi_rad, dist)


def water_dielectric_r_effective(dist: float, epsilon_r: float) -> float:
    """EM path uses ``r ↦ ε_r * r`` (Lean ``water_dielectric_rescaling_eq_EM``); proxy uses ``dist`` as-is in Python EM placeholder."""
    if epsilon_r <= 0:
        raise ValueError("epsilon_r must be positive")
    return epsilon_r * dist


def long_range_valley_with_aqueous_em_scalar(
    valley_em_at_scaled_r: float,
    m: int,
    theta_rad: float,
    phi_rad: float,
    dist: float,
) -> float:
    """
    Lean ``water_dielectric_rescaling_long_range`` (scalar bookkeeping): EM piece is already
    evaluated at the **scaled** bond argument (``ε_r * r`` in the formal statement); add
    ``hBondProxy`` at the **physical** ``dist`` unchanged.

    Pass ``valley_em_at_scaled_r = valleyPotentialEM(..., r_eff)`` with
    ``r_eff = water_dielectric_r_effective(dist, ε_r)`` when coupling to bulk water.
    """
    return float(valley_em_at_scaled_r) + h_bond_proxy(m, theta_rad, phi_rad, dist)


def contact_alignment_angles_ca(
    ca: np.ndarray,
    i: int,
    j: int,
) -> Tuple[float, float, float]:
    """
    Geometric contact angles (radians) for a Cα pair (i, j), i != j.

    θ: angle between incoming segment at i (toward i from i-1) and vector i→j.
    φ: angle between outgoing segment at j (from j toward j+1) and vector j→i.

    Returns (theta_rad, phi_rad, dist). Skips degenerate geometry by returning
    (π/2, π/2, dist) when a tangent is undefined (chain ends).
    """
    ca = np.asarray(ca, dtype=float)
    n = ca.shape[0]
    if n < 2 or i == j or i < 0 or j < 0 or i >= n or j >= n:
        return (math.pi / 2, math.pi / 2, 0.0)
    rij = ca[j] - ca[i]
    dist = float(np.linalg.norm(rij))
    if dist < 1e-12:
        return (math.pi / 2, math.pi / 2, dist)
    rij_u = rij / dist

    if i > 0:
        ui = ca[i] - ca[i - 1]
        nu = np.linalg.norm(ui)
        if nu > 1e-12:
            ui = ui / nu
            cth = float(np.clip(np.dot(ui, rij_u), -1.0, 1.0))
            theta_rad = math.acos(cth)
        else:
            theta_rad = math.pi / 2
    else:
        theta_rad = math.pi / 2

    if j < n - 1:
        uj = ca[j + 1] - ca[j]
        nv = np.linalg.norm(uj)
        if nv > 1e-12:
            uj = uj / nv
            rji_u = -rij_u
            cph = float(np.clip(np.dot(uj, rji_u), -1.0, 1.0))
            phi_rad = math.acos(cph)
        else:
            phi_rad = math.pi / 2
    else:
        phi_rad = math.pi / 2

    return (theta_rad, phi_rad, dist)


def total_h_bond_proxy_energy_ca(
    ca: np.ndarray,
    m: int,
    *,
    min_seq_sep: int = 3,
    max_pairs: int = 500,
    dist_cutoff: float = 15.0,
) -> float:
    """
    Sum ``hBondProxy`` over residue pairs with |i−j| ≥ ``min_seq_sep`` and distance < ``dist_cutoff``.

    For monitoring / research; not yet wired into ``grad_full`` by default (pairwise cost).
    """
    ca = np.asarray(ca, dtype=float)
    n = ca.shape[0]
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + min_seq_sep, n):
            if count >= max_pairs:
                return total
            th, ph, d = contact_alignment_angles_ca(ca, i, j)
            if d > dist_cutoff or d < 1e-9:
                continue
            total += h_bond_proxy(m, th, ph, d)
            count += 1
    return total


def grad_h_bond_proxy_ca_fd(
    ca: np.ndarray,
    m: int,
    eps: float = 1e-3,
    **kwargs,
) -> np.ndarray:
    """
    Total ``h_bond_proxy`` gradient w.r.t. Cα (finite differences, deterministic).
    """
    ca = np.asarray(ca, dtype=float)
    n = ca.shape[0]
    g = np.zeros_like(ca)
    e0 = total_h_bond_proxy_energy_ca(ca, m, **kwargs)
    for i in range(n):
        for d in range(3):
            ca_p = ca.copy()
            ca_p[i, d] += eps
            ca_m = ca.copy()
            ca_m[i, d] -= eps
            ep = total_h_bond_proxy_energy_ca(ca_p, m, **kwargs)
            em = total_h_bond_proxy_energy_ca(ca_m, m, **kwargs)
            g[i, d] = (ep - em) / (2.0 * eps)
    return g


if __name__ == "__main__":
    for m in (0, 1, 2, 3):
        print(
            f"m={m}  K_hbond={K_hbond(m):.6g}  R_hbond={R_hbond(m):.3f}  modes={available_modes_nat(m)}"
        )
    th = math.radians(40.0)
    ph = math.radians(35.0)
    d = 4.0
    hb = h_bond_proxy(2, th, ph, d)
    print(f"hBondProxy(2,40°,35°,4Å) = {hb:.6g}  (expect < 0 if cosines > 0)")
