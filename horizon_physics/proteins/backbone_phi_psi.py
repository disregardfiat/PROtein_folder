"""
Backbone φ/ψ ↔ Cα trace and full-atom representation.

- dihedral_rad: compute torsion angle from four positions.
- backbone_phi_psi_from_atoms: extract φ, ψ (rad) from list of (atom_name, xyz) in N,CA,C,O order per residue.
- ca_positions_from_phi_psi: build Cα trace from per-residue φ, ψ using forward kinematics (rise + rotations).

Used by the discrete DOF pipeline to rebuild coordinates after a torsion move.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def dihedral_rad(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Dihedral angle (radians) of vector p1->p2->p3->p4 (rotation of plane p1p2p3 into p2p3p4 about p2-p3).
    """
    b1 = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    b2 = np.asarray(p3, dtype=float) - np.asarray(p2, dtype=float)
    b3 = np.asarray(p4, dtype=float) - np.asarray(p3, dtype=float)
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-12 or n2_norm < 1e-12:
        return 0.0
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    b2_unit = b2 / (np.linalg.norm(b2) + 1e-12)
    m1 = np.cross(n1, b2_unit)
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return float(np.arctan2(y, x))


def backbone_phi_psi_from_atoms(
    backbone_atoms: List[Tuple[str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract φ, ψ (radians) from backbone atoms: list of (name, xyz) in order N, CA, C, O per residue.

    φ_i = dihedral(C_{i-1}, N_i, CA_i, C_i) for i >= 1; φ_0 set to 0.
    ψ_i = dihedral(N_i, CA_i, C_i, N_{i+1}) for i < n-1; ψ_{n-1} set to 0.

    Returns
    -------
    phi_rad, psi_rad : (n_res,) each
    """
    n_res = len(backbone_atoms) // 4
    phi_rad = np.zeros(n_res, dtype=float)
    psi_rad = np.zeros(n_res, dtype=float)
    for i in range(n_res):
        o = i * 4
        n_i = np.array(backbone_atoms[o + 0][1])
        ca_i = np.array(backbone_atoms[o + 1][1])
        c_i = np.array(backbone_atoms[o + 2][1])
        if i > 0:
            c_prev = np.array(backbone_atoms[(i - 1) * 4 + 2][1])
            phi_rad[i] = dihedral_rad(c_prev, n_i, ca_i, c_i)
        if i < n_res - 1:
            n_next = np.array(backbone_atoms[(i + 1) * 4 + 0][1])
            psi_rad[i] = dihedral_rad(n_i, ca_i, c_i, n_next)
    return phi_rad, psi_rad


def _rotate_y(v: np.ndarray, rad: float) -> np.ndarray:
    """Rotate vector v around y-axis by rad (radians)."""
    c, s = np.cos(rad), np.sin(rad)
    return np.array([v[0] * c - v[2] * s, v[1], v[0] * s + v[2] * c], dtype=float)


def ca_positions_from_phi_psi(
    phi_rad: np.ndarray,
    psi_rad: np.ndarray,
    rise: float = 3.8,
) -> np.ndarray:
    """
    Build Cα trace from per-residue φ, ψ using forward kinematics.

    Chain starts at origin; direction d advances by phi_i then psi_i per step (phi_0 not used).
    d_{i+1} = R(psi_i) R(phi_i) d_i; ca[i+1] = ca[i] + rise * d_{i+1}. ψ_{n-1} unused.

    Parameters
    ----------
    phi_rad, psi_rad : (n_res,) in radians
    rise : Cα–Cα step length in Å

    Returns
    -------
    ca : (n_res, 3) in Å
    """
    n_res = len(phi_rad)
    assert len(psi_rad) == n_res
    ca = np.zeros((n_res, 3), dtype=float)
    d = np.array([1.0, 0.0, 0.0], dtype=float)
    for i in range(n_res - 1):
        if i >= 1:
            d = _rotate_y(d, phi_rad[i])
        d = _rotate_y(d, psi_rad[i])
        ca[i + 1] = ca[i] + rise * d
    return ca
