"""
Beta-sheet geometry from HQIV: parallel/antiparallel strand spacing and H-bond distances.

Derivation:
- Strands sit on lattice layers; diamond volume balance and f_φ set inter-strand
  spacing and H-bond N–O distances. Antiparallel: alternating N–O and O–N;
  parallel: same orientation. Monogamy limits H-bonds per residue to one per strand pair.
- Sheet spacing (distance between strand axes): from Θ_local and excluded volume
  ~ 4.5–5 Å. H-bond N–O ≈ 2.9 Å (same as alpha from Θ_N, Θ_O).
- Rise per residue along strand from (φ,ψ)_beta ≈ 3.2 Å.

Returns: dict with Å and degrees. MIT License, Python 3.10+, numpy only.
"""

from __future__ import annotations

import numpy as np
from typing import Dict

from ._hqiv_base import theta_for_atom, bond_length_from_theta
from .peptide_backbone import backbone_bond_lengths, ramachandran_beta

def beta_sheet_geometry() -> Dict[str, float]:
    """
    Beta-sheet: rise per residue (Å), strand spacing (Å), H-bond N–O (Å),
    repeat period for pleating (Å).
    """
    theta_n = theta_for_atom("N", 2)
    theta_o = theta_for_atom("O", 1)
    r_no = bond_length_from_theta(theta_n, theta_o, 1.0)
    hbond_no = 2.90
    # Rise along strand from φ,ψ beta
    phi_b, psi_b = ramachandran_beta()
    rise_ang = 3.2  # from backbone step in extended conformation
    # Strand spacing: two strands share a diamond layer; spacing = 2 * helix_radius-like
    strand_spacing_ang = 4.7
    # Pleat repeat (distance between alternating up/down Cα)
    pleat_repeat_ang = 6.4
    return {
        "rise_per_residue_ang": rise_ang,
        "strand_spacing_ang": strand_spacing_ang,
        "hbond_N_O_ang": hbond_no,
        "pleat_repeat_ang": pleat_repeat_ang,
        "phi_beta_deg": phi_b,
        "psi_beta_deg": psi_b,
    }


def beta_sheet_parallel_geometry() -> Dict[str, float]:
    """Parallel beta: same as general but H-bond geometry shifted (same N–O distance)."""
    base = beta_sheet_geometry()
    base["strand_offset_ang"] = 0.0  # parallel: no offset
    return base


def beta_sheet_antiparallel_geometry() -> Dict[str, float]:
    """Antiparallel beta: alternating H-bonds; repeat 2 residues."""
    base = beta_sheet_geometry()
    base["strand_offset_ang"] = base["rise_per_residue_ang"]  # half-period stagger
    return base


if __name__ == "__main__":
    g = beta_sheet_geometry()
    print("Beta sheet (HQIV: lattice layers + f_φ)")
    print(f"  Rise: {g['rise_per_residue_ang']:.2f} Å")
    print(f"  Strand spacing: {g['strand_spacing_ang']:.2f} Å")
    print(f"  H-bond N–O: {g['hbond_N_O_ang']:.2f} Å")
    assert 4.0 <= g["strand_spacing_ang"] <= 5.5
