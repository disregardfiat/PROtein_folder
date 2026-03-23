"""
HQIV Lean alignment helpers (Hqiv/Physics/HQIVMolecules.lean, HQIVAtoms.lean).

Maps proven Lean structures to the Python folding stack:
  - foldEnergyWithDihedral: κ * (1 - cos θ) dihedral correction on φ/ψ toward HQIV α-basin.
  - waterDielectricValley: horizon EM scale 1/ε_r (see folding_energy.grad_horizon_full em_scale).
  - pH / charge bookkeeping: optional δZ shift on effective screening (mild, physiological).

Constants: water O–H–O angle anchor 104.5° (Lean `waterBondAngleDeg`); bulk ε_r(T) for water.
"""

from __future__ import annotations

import numpy as np

from .backbone_phi_psi import backbone_phi_psi_from_atoms
from .casp_submission import _place_full_backbone
from .peptide_backbone import rational_ramachandran_alpha

# Lean HQIVMolecules / HQIVAtoms anchors
WATER_BOND_ANGLE_DEG = 104.5
PHYSIOLOGICAL_PH = 7.4


def epsilon_r_water(T_kelvin: float = 310.0) -> float:
    """
    Relative permittivity of bulk liquid water vs temperature (experimentally sound).

    Uses a simple Malmberg–Maryott–type linearization around 298 K (ε ≈ 78.5)
    with dε/dT ≈ −0.37 K⁻¹ in the 273–320 K range (approximate; adequate for screening).
    """
    T_ref = 298.15
    eps_ref = 78.5
    d_eps_dT = -0.37
    eps = eps_ref + d_eps_dT * (T_kelvin - T_ref)
    return float(np.clip(eps, 55.0, 90.0))


def em_scale_aqueous(T_kelvin: float = 310.0, epsilon_r: float | None = None) -> float:
    """Lean `waterDielectricValley`: EM piece scales as 1/(ε_r r) → multiply poles by 1/ε_r."""
    er = float(epsilon_r) if epsilon_r is not None else epsilon_r_water(T_kelvin)
    if er <= 1e-9:
        return 1.0
    return 1.0 / er


def ph_em_scale_delta(ph: float = PHYSIOLOGICAL_PH, pka_ref: float = 7.0) -> float:
    """
    Mild modulation of screening from acid–base distance to typical protein pKa (Lean `pH_charge_flip_effect` analog).

    At pH = pka_ref scale is 1.0; one-sided excess protons slightly increase effective EM coupling in screening model.
    """
    x = float(ph) - float(pka_ref)
    return float(1.0 + 0.02 * np.tanh(x))


def fold_energy_dihedral_penalty_phi_psi(
    phi_rad: np.ndarray,
    psi_rad: np.ndarray,
    *,
    kappa: float,
    phi0_deg: float | None = None,
    psi0_deg: float | None = None,
) -> float:
    """
    Sum κ * (1 - cos(φ_i - φ0)) + κ * (1 - cos(ψ_i - ψ0)) over valid Ramachandran indices
    (matches Lean `foldEnergyWithDihedral` structure on dihedral correction).
    """
    if kappa <= 0:
        return 0.0
    if phi0_deg is None or psi0_deg is None:
        phi0_deg, psi0_deg = rational_ramachandran_alpha()
    phi0 = np.deg2rad(phi0_deg)
    psi0 = np.deg2rad(psi0_deg)
    n = len(phi_rad)
    s = 0.0
    for i in range(1, n):
        s += kappa * (1.0 - np.cos(phi_rad[i] - phi0))
    for i in range(n - 1):
        s += kappa * (1.0 - np.cos(psi_rad[i] - psi0))
    return float(s)


def dihedral_penalty_from_ca(
    ca: np.ndarray,
    sequence: str,
    *,
    kappa: float,
    phi0_deg: float | None = None,
    psi0_deg: float | None = None,
) -> float:
    """Reconstruct backbone from Cα, extract φ/ψ, return dihedral penalty energy."""
    if kappa <= 0:
        return 0.0
    bb = _place_full_backbone(ca, sequence)
    phi_rad, psi_rad = backbone_phi_psi_from_atoms(bb)
    return fold_energy_dihedral_penalty_phi_psi(
        phi_rad, psi_rad, kappa=kappa, phi0_deg=phi0_deg, psi0_deg=psi0_deg
    )


def grad_dihedral_penalty_ca_fd(
    ca: np.ndarray,
    sequence: str,
    kappa: float,
    eps: float = 2e-3,
    phi0_deg: float | None = None,
    psi0_deg: float | None = None,
) -> np.ndarray:
    """
    ∇_{Cα} E_dihedral via central finite differences (deterministic; no random seed).
    """
    if kappa <= 0:
        return np.zeros_like(ca)
    ca = np.asarray(ca, dtype=float)
    n = ca.shape[0]
    grad = np.zeros_like(ca)
    e0 = dihedral_penalty_from_ca(ca, sequence, kappa=kappa, phi0_deg=phi0_deg, psi0_deg=psi0_deg)
    for i in range(n):
        for d in range(3):
            ca_p = ca.copy()
            ca_p[i, d] += eps
            ca_m = ca.copy()
            ca_m[i, d] -= eps
            ep = dihedral_penalty_from_ca(ca_p, sequence, kappa=kappa, phi0_deg=phi0_deg, psi0_deg=psi0_deg)
            em = dihedral_penalty_from_ca(ca_m, sequence, kappa=kappa, phi0_deg=phi0_deg, psi0_deg=psi0_deg)
            grad[i, d] = (ep - em) / (2.0 * eps)
    return grad

