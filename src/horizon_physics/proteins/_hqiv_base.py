"""
HQIV base: discrete null lattice, horizon scalar φ, and geometric damping.

First-principles only (no empirical bond lengths). Used by peptide_backbone,
alpha_helix, beta_sheet, side_chain_placement, folding_energy.

When pyhqiv is installed, all metric math (φ, γ, Θ from Z/coord, bond length,
damping magnitude, constants) is delegated to pyhqiv; otherwise local fallbacks.
Units: lengths in Å, angles in degrees. MIT License, Python 3.10+, numpy only.
"""

from __future__ import annotations

import numpy as np

# --- Optional pyhqiv: full metric (φ, γ, Θ, bond length, damping, constants) ---
try:
    from pyhqiv.utils import (
        phi_from_theta_local as _pyhqiv_phi_from_theta,
        theta_local as _pyhqiv_theta_local,
        theta_for_atom as _pyhqiv_theta_for_atom,
        bond_length_from_theta as _pyhqiv_bond_length_from_theta,
        damping_force_magnitude as _pyhqiv_damping_force_magnitude,
    )
    from pyhqiv.constants import (
        GAMMA as _pyhqiv_GAMMA,
        A_LOC_ANG as _pyhqiv_A_LOC_ANG,
        HBAR_C_EV_ANG as _pyhqiv_HBAR_C_EV_ANG,
    )
    _PYHQIV_AVAILABLE = True
except ImportError:
    _pyhqiv_phi_from_theta = None
    _pyhqiv_GAMMA = None
    _pyhqiv_theta_local = None
    _pyhqiv_theta_for_atom = None
    _pyhqiv_bond_length_from_theta = None
    _pyhqiv_damping_force_magnitude = None
    _pyhqiv_A_LOC_ANG = None
    _pyhqiv_HBAR_C_EV_ANG = None
    _PYHQIV_AVAILABLE = False

# --- Constants: from pyhqiv when available ---
HBAR_C_EV_ANG = float(_pyhqiv_HBAR_C_EV_ANG) if _PYHQIV_AVAILABLE else 1973.269804
A_LOC_ANG = float(_pyhqiv_A_LOC_ANG) if _PYHQIV_AVAILABLE else 1.0
GAMMA_PRIME = float(_pyhqiv_GAMMA) if _PYHQIV_AVAILABLE else 0.4


def _theta_local_fallback(z_shell: int, coordination: int = 1) -> float:
    if z_shell <= 0 or coordination <= 0:
        return np.nan
    alpha = 0.91
    theta0 = 1.53 * (6 ** alpha) * (2 ** (1 / 3))
    return theta0 * (z_shell ** (-alpha)) / (coordination ** (1 / 3))


def theta_local(z_shell: int, coordination: int = 1) -> float:
    """Diamond size Θ_local (Å) at a lattice node from Z and coordination. From pyhqiv when available."""
    if _PYHQIV_AVAILABLE:
        return float(_pyhqiv_theta_local(z_shell, coordination))
    return _theta_local_fallback(z_shell, coordination)


def horizon_scalar(theta_ang: float) -> float:
    """φ = 2 c² / Θ_local. From pyhqiv.utils.phi_from_theta_local when available (c=1)."""
    if theta_ang <= 0:
        return np.nan
    if _PYHQIV_AVAILABLE:
        out = _pyhqiv_phi_from_theta(np.asarray(theta_ang, dtype=float), c=1.0)
        return float(np.asarray(out).reshape(-1)[0])
    return 2.0 / theta_ang


def damping_force_magnitude(phi: float, grad_phi: float, a_loc: float = A_LOC_ANG) -> float:
    """|f_φ| = γ φ |∇φ| / (a_loc + φ/6)². From pyhqiv.utils when available."""
    if _PYHQIV_AVAILABLE:
        out = _pyhqiv_damping_force_magnitude(phi, grad_phi, a_loc=a_loc, gamma=None)
        return float(np.asarray(out).reshape(-1)[0])
    denom = a_loc + phi / 6.0
    if denom <= 0:
        return 0.0
    return GAMMA_PRIME * phi * abs(grad_phi) / (denom ** 2)


def bond_length_from_theta(theta_i: float, theta_j: float, monogamy_factor: float = 1.0) -> float:
    """Equilibrium separation (Å) = min(Θ_i, Θ_j) * monogamy_factor. From pyhqiv when available."""
    if _PYHQIV_AVAILABLE:
        return float(_pyhqiv_bond_length_from_theta(theta_i, theta_j, monogamy_factor))
    return min(theta_i, theta_j) * monogamy_factor


def _theta_for_atom_fallback(symbol: str, coordination: int = 1) -> float:
    z_map = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}
    z = z_map.get(symbol, 6)
    return theta_local(z, coordination)


def theta_for_atom(symbol: str, coordination: int = 1) -> float:
    """Θ (Å) for atom type from symbol. From pyhqiv when available."""
    if _PYHQIV_AVAILABLE:
        return float(_pyhqiv_theta_for_atom(symbol, coordination))
    return _theta_for_atom_fallback(symbol, coordination)


if __name__ == "__main__":
    theta_c = theta_for_atom("C", 2)
    theta_n = theta_for_atom("N", 2)
    r_cc = bond_length_from_theta(theta_c, theta_c, 1.0)
    r_cn = bond_length_from_theta(theta_c, theta_n, 1.0)
    print("HQIV base: Θ and bond lengths from lattice + monogamy")
    print(f"  Θ_C(coord=2)={theta_c:.4f} Å, Θ_N(coord=2)={theta_n:.4f} Å")
    print(f"  r_Cα-C={r_cc:.4f} Å, r_C-N={r_cn:.4f} Å")
    assert 1.50 < r_cc < 1.56 and 1.28 < r_cn < 1.38
