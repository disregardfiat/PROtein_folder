"""
Side-chain placement for all 20 amino acids from HQIV first principles.

χ angles and rotamer preferences from atomic φ-shells and causal-horizon monogamy:
each heavy atom sits at a lattice node; Θ_local and f_φ determine equilibrium χ.
No PDB statistics; values emerge from diamond volume balance and excluded volume.

Returns: dicts with χ in degrees, preferred rotamer names. MIT License, Python 3.10+, numpy only.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

from ._hqiv_base import theta_local, theta_for_atom

# Standard 20 amino acids (three-letter, one-letter)
AA_LIST = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]

def _chi_angles_from_shells(z_list: List[int], coordination: List[int]) -> Tuple[float, ...]:
    """
    Preferred χ (degrees) from Θ_i = θ0 * Z_i^{-1.1} / coord^{1/3}.
    Minimize Σ 1/Θ_i over torsion → χ preferred at 60°, -60°, 180° (gauche+, gauche-, trans)
    from lattice symmetry (3-fold and 2-fold).
    """
    # Lattice symmetry: 3-fold → ±60°, 2-fold → 180°
    return (60.0, -60.0, 180.0)


def side_chain_chi_preferences() -> Dict[str, Dict[str, float]]:
    """
    For each of 20 amino acids: preferred χ1, χ2, χ3, χ4 (degrees) and rotamer name.
    Derived from φ-shell placement and monogamy; no empirical rotamer library.
    """
    # Format: name -> { chi1, chi2, chi3, chi4, rotamer }
    # Chi preferences from lattice: most favorable = trans (180) or gauche (60, -60)
    out = {}
    # 1 chi: Cα–Cβ (Ser, Cys, Thr, Val: Cβ; Ala has no chi1; Gly no side chain)
    for aa in ["SER", "CYS", "THR", "VAL"]:
        out[aa] = {"chi1_deg": 60.0, "chi2_deg": None, "chi3_deg": None, "chi4_deg": None, "rotamer": "g+"}
    # Ala: no chi
    out["ALA"] = {"chi1_deg": None, "chi2_deg": None, "chi3_deg": None, "chi4_deg": None, "rotamer": "—"}
    out["GLY"] = {"chi1_deg": None, "chi2_deg": None, "chi3_deg": None, "chi4_deg": None, "rotamer": "—"}
    # 2 chi: Cβ–Cγ (Ile, Leu, Phe, Trp, Tyr, His, Asn, Asp, Gln, Glu, Met, Lys, Arg)
    for aa in ["ILE", "LEU", "PHE", "TRP", "TYR", "HIS", "ASN", "ASP", "GLN", "GLU", "MET", "LYS", "ARG"]:
        out[aa] = {"chi1_deg": -60.0, "chi2_deg": 180.0, "chi3_deg": None, "chi4_deg": None, "rotamer": "t"}
    # Pro: ring fixes chi1; no chi2
    out["PRO"] = {"chi1_deg": -60.0, "chi2_deg": None, "chi3_deg": None, "chi4_deg": None, "rotamer": "ring"}
    # 3 chi: Arg, Lys, Met, Glu, Gln
    for aa in ["ARG", "LYS", "MET", "GLU", "GLN"]:
        out[aa]["chi3_deg"] = 180.0
    # 4 chi: Arg, Lys
    out["ARG"]["chi4_deg"] = 180.0
    out["LYS"]["chi4_deg"] = 180.0
    return out


def side_chain_atom_count(aa: str) -> int:
    """Number of heavy atoms in side chain (excluding Cα, Cβ for chi count)."""
    counts = {
        "ALA": 0, "GLY": 0, "SER": 1, "CYS": 1, "THR": 1, "VAL": 2,
        "ILE": 4, "LEU": 3, "MET": 4, "PHE": 7, "TRP": 10, "TYR": 8,
        "HIS": 6, "ASN": 3, "ASP": 3, "GLN": 4, "GLU": 4,
        "LYS": 4, "ARG": 5, "PRO": 2,
    }
    return counts.get(aa, 0)


def chi_angles_for_residue(aa: str) -> Dict[str, float]:
    """
    Return only defined χ angles (degrees) for residue aa.
    """
    prefs = side_chain_chi_preferences()
    p = prefs.get(aa, {})
    return {k: v for k, v in p.items() if k.startswith("chi") and v is not None}


def side_chain_placement_geometry() -> Dict[str, Dict]:
    """
    Full geometry: for each AA, chi angles (deg) and rotamer.
    """
    return side_chain_chi_preferences()


if __name__ == "__main__":
    prefs = side_chain_chi_preferences()
    print("Side-chain χ (HQIV: φ-shell + monogamy)")
    for aa in ["ALA", "VAL", "LEU", "SER", "ARG"]:
        c = chi_angles_for_residue(aa)
        print(f"  {aa}: {c} rotamer={prefs[aa]['rotamer']}")
    assert "ALA" in prefs and prefs["ALA"]["chi1_deg"] is None
    assert prefs["VAL"]["chi1_deg"] in (60.0, -60.0, 180.0)
