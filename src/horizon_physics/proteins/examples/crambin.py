"""
Crambin (46 residues) example: full small protein from HQIV.

Uses hqiv_predict_structure on crambin FASTA to produce a CASP-format PDB.
Secondary structure: crambin has 3 helices and 3 strands; here we use
alpha-helix geometry for the backbone trace (full SS prediction can be added).
MIT License, Python 3.10+, numpy only.
"""

from __future__ import annotations

import os
import sys

_p = os.path.dirname(os.path.abspath(__file__))
while _p != os.path.dirname(_p):
    if os.path.isfile(os.path.join(_p, "pyproject.toml")):
        _src = os.path.join(_p, "src")
        if _src not in sys.path:
            sys.path.insert(0, _src)
        break
    _p = os.path.dirname(_p)

from horizon_physics.proteins import hqiv_predict_structure, alpha_helix_geometry, backbone_geometry

# Crambin (1CRN) sequence — 46 residues
CRAMBIN_FASTA = """>1CRN crambin
TTCCPSIVARSNFNVCRLPGTPEAIICGDVCDLDCTAKTCFSIICT

"""

def main():
    geom_backbone = backbone_geometry()
    geom_helix = alpha_helix_geometry()
    print("Crambin (46 residues) — HQIV first principles")
    print(f"  Backbone Cα–C: {geom_backbone['Calpha_C']:.4f} Å, C–N: {geom_backbone['C_N']:.4f} Å")
    print(f"  Helix: {geom_helix['residues_per_turn']:.2f} res/turn, pitch {geom_helix['pitch_ang']:.2f} Å")
    pdb = hqiv_predict_structure(CRAMBIN_FASTA)
    n_atoms = pdb.count("ATOM")
    print(f"  PDB: {n_atoms} ATOM lines (backbone N, CA, C, O)")
    # Write to file if desired
    out_path = os.path.join(os.path.dirname(__file__), "crambin_hqiv.pdb")
    with open(out_path, "w") as f:
        f.write(pdb)
    print(f"  Written: {out_path}")
    return pdb


if __name__ == "__main__":
    main()
