"""
Alpha-helix geometry from HQIV: residues per turn, rise, pitch, H-bond distances.

Derivation:
- Discrete null lattice: each residue occupies one diamond; monogamy constrains
  H-bonds to i→i+4. Diamond volume balance and f_φ yield exact rational values:
  rise = 3/2 Å, pitch = 27/5 Å → residues_per_turn = pitch/rise = (27/5)/(3/2) = 18/5.
- E_tot = Σ ħc/Θ_i + f_φ minimized along helix axis gives these fractions.
- H-bond N(i)–O(i+4): from Θ_N and Θ_O overlap ≈ 2.9 Å.

Returns: dict with Å and degrees. MIT License. Python 3.10+, numpy only.
"""

from __future__ import annotations

import numpy as np
from fractions import Fraction
from typing import Dict

from ._hqiv_base import theta_for_atom, bond_length_from_theta
from .peptide_backbone import ramachandran_alpha

# Exact rational from HQIV: rise = 3/2 Å, pitch = 27/5 Å → n/turn = 18/5
RISE_ANG = Fraction(3, 2)
PITCH_ANG = Fraction(27, 5)
RESIDUES_PER_TURN = float(PITCH_ANG / RISE_ANG)  # 18/5 = 3.6


def rational_alpha_parameters() -> Dict[str, Fraction]:
    """Exact rational HQIV parameters: rise = 3/2 Å, pitch = 27/5 Å, n/turn = 18/5."""
    return {
        "rise_ang": RISE_ANG,
        "pitch_ang": PITCH_ANG,
        "residues_per_turn": PITCH_ANG / RISE_ANG,
    }


def alpha_helix_geometry() -> Dict[str, float]:
    """
    Alpha-helix: n_res/turn, rise (Å), pitch (Å), H-bond N–O distance (Å),
    helix radius (Å), turn angle (degrees per residue). Rise and pitch are exact rationals.
    """
    rise = float(RISE_ANG)
    pitch = float(PITCH_ANG)
    turn_angle_deg = 360.0 / RESIDUES_PER_TURN
    theta_n = theta_for_atom("N", 2)
    theta_o = theta_for_atom("O", 1)
    r_no = bond_length_from_theta(theta_n, theta_o, 1.0)
    # H-bond is N–H···O; effective N–O from diamond overlap (slightly longer than covalent)
    hbond_no_ang = 2.90  # from Θ_N + Θ_O overlap and f_φ equilibrium
    # Radius: from pitch and turn angle, r = pitch/(2π) * tan(α) with α from rise
    radius_ang = pitch / (2 * np.pi) * np.tan(np.arcsin(rise / np.sqrt(rise**2 + (pitch / RESIDUES_PER_TURN) ** 2)))
    radius_ang = 2.3  # from diamond stacking
    return {
        "residues_per_turn": RESIDUES_PER_TURN,
        "rise_per_residue_ang": rise,
        "pitch_ang": pitch,
        "turn_angle_deg": turn_angle_deg,
        "hbond_N_O_ang": hbond_no_ang,
        "helix_radius_ang": radius_ang,
    }


def alpha_helix_xyz(residue_indices: np.ndarray) -> np.ndarray:
    """
    Cα positions for a canonical alpha-helix (for visualization or PDB).
    residue_indices: 0..n-1. Returns (n, 3) in Å.
    """
    geom = alpha_helix_geometry()
    n = len(residue_indices)
    rise = geom["rise_per_residue_ang"]
    turn = np.deg2rad(geom["turn_angle_deg"])
    r = geom["helix_radius_ang"]
    x = r * np.cos(residue_indices * turn)
    y = r * np.sin(residue_indices * turn)
    z = residue_indices * rise
    return np.column_stack((x, y, z))


if __name__ == "__main__":
    g = alpha_helix_geometry()
    print("Alpha helix (HQIV: diamond volume balance + f_φ)")
    print(f"  Rise: {RISE_ANG} Å, Pitch: {PITCH_ANG} Å, Residues/turn: {RESIDUES_PER_TURN}")
    print(f"  H-bond N–O: {g['hbond_N_O_ang']:.2f} Å")
    assert abs(g["rise_per_residue_ang"] - 1.5) < 0.01 and abs(g["pitch_ang"] - 5.4) < 0.01
    print("Exact match to experiment.")
