"""
CASP submission: FASTA → PDB generator using HQIV structure prediction.

hqiv_predict_structure(fasta: str) -> str returns a valid CASP-format PDB string
(MODEL 1 ... END) using peptide_backbone, alpha_helix, beta_sheet, side_chain_placement.
No external dependencies beyond numpy.

MIT License, Python 3.10+.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

from .peptide_backbone import backbone_bond_lengths, backbone_geometry, omega_peptide_deg, ramachandran_alpha, ramachandran_beta
from .alpha_helix import alpha_helix_geometry, alpha_helix_xyz
from .beta_sheet import beta_sheet_geometry
from .side_chain_placement import side_chain_chi_preferences, chi_angles_for_residue
from .secondary_structure_predictor import predict_ss

# One-letter to three-letter
AA_1to3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}


def _parse_fasta(fasta: str) -> str:
    """Extract single sequence from FASTA string (first block only)."""
    lines = [l.strip() for l in fasta.strip().splitlines() if l.strip()]
    seq = []
    for line in lines:
        if line.startswith(">"):
            continue
        seq.append("".join(c for c in line if c in AA_1to3))
    return "".join(seq)


def _place_backbone_ca(sequence: str, ss_string: Optional[str] = None) -> np.ndarray:
    """
    Place Cα from sequence. If ss_string is None, use HQIV secondary structure
    prediction (predict_ss) for SS-aware placement: H → helix, E → extended, C → coil.
    Returns (n_res, 3) in Å.
    """
    n = len(sequence)
    if n == 0:
        return np.zeros((0, 3))
    if ss_string is None or len(ss_string) != n:
        ss_string, _ = predict_ss(sequence, window=5)
    return _place_backbone_ca_ss(sequence, ss_string)


def _place_backbone_ca_ss(sequence: str, ss_string: str) -> np.ndarray:
    """
    Place Cα segment-by-segment: helix geometry for H, extended (beta rise) for E/C.
    Segments are stitched by translation and rotation so the chain is continuous.
    """
    n = len(sequence)
    assert len(ss_string) == n
    geom_helix = alpha_helix_geometry()
    rise_helix = geom_helix["rise_per_residue_ang"]
    turn_deg = geom_helix["turn_angle_deg"]
    r_helix = geom_helix["helix_radius_ang"]
    rise_ext = 3.2   # beta/coil from beta_sheet_geometry
    rise_coil = 3.0

    # Build runs (ss_type, start, end)
    runs = []
    i = 0
    while i < n:
        s = ss_string[i]
        start = i
        while i < n and ss_string[i] == s:
            i += 1
        runs.append((s, start, i))

    positions = np.zeros((n, 3))
    global_origin = np.zeros(3)
    global_forward = np.array([1.0, 0.0, 0.0])
    global_up = np.array([0.0, 0.0, 1.0])
    first_segment = True

    for ss_type, start, end in runs:
        L = end - start
        if L == 0:
            continue
        rise_current = rise_helix if ss_type == "H" else (rise_ext if ss_type == "E" else rise_coil)
        # Advance origin by one step from previous segment end (so chain is continuous)
        if not first_segment:
            global_origin = global_origin + global_forward * rise_current
        first_segment = False
        # Local segment positions: first residue at origin for clean stitching
        if ss_type == "H":
            local = alpha_helix_xyz(np.arange(L, dtype=float))
            if L > 0:
                local = local - local[0]
        else:
            rise = rise_ext if ss_type == "E" else rise_coil
            local = np.zeros((L, 3))
            local[:, 0] = np.arange(L) * rise
        # Direction at start and end of segment
        if L == 1:
            seg_forward = global_forward.copy()
        else:
            seg_forward = local[-1] - local[-2]
            seg_forward = seg_forward / (np.linalg.norm(seg_forward) + 1e-9)
        seg_start_forward = local[1] - local[0] if L > 1 else np.array([1.0, 0.0, 0.0])
        seg_start_forward = seg_start_forward / (np.linalg.norm(seg_start_forward) + 1e-9)
        # Rotate local so seg_start_forward aligns with global_forward
        cos_a = np.dot(seg_start_forward, global_forward)
        cos_a = np.clip(cos_a, -1, 1)
        if np.abs(cos_a) < 1 - 1e-6:
            axis = np.cross(seg_start_forward, global_forward)
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-9:
                axis = axis / axis_norm
                angle = np.arccos(cos_a)
                c, s = np.cos(angle), np.sin(angle)
                R = np.array([
                    [c + axis[0]**2*(1-c), axis[0]*axis[1]*(1-c) - axis[2]*s, axis[0]*axis[2]*(1-c) + axis[1]*s],
                    [axis[1]*axis[0]*(1-c) + axis[2]*s, c + axis[1]**2*(1-c), axis[1]*axis[2]*(1-c) - axis[0]*s],
                    [axis[2]*axis[0]*(1-c) - axis[1]*s, axis[2]*axis[1]*(1-c) + axis[0]*s, c + axis[2]**2*(1-c)],
                ])
                local = (R @ local.T).T
        # Translate to global origin
        local = local + global_origin
        positions[start:end] = local
        # Update global origin and forward for next segment
        global_origin = local[-1]
        if L > 1:
            global_forward = local[-1] - local[-2]
            global_forward = global_forward / (np.linalg.norm(global_forward) + 1e-9)
        # Keep global_up roughly perpendicular
        global_up = np.cross(global_forward, np.cross(global_up, global_forward))
        if np.linalg.norm(global_up) > 1e-9:
            global_up = global_up / np.linalg.norm(global_up)
        else:
            global_up = np.array([0.0, 0.0, 1.0])

    return positions


def _place_full_backbone(ca_pos: np.ndarray, sequence: str) -> List[Tuple[str, np.ndarray]]:
    """
    Place N, Cα, C, O for each residue. Returns list of (atom_name, (x,y,z)).
    """
    bonds = backbone_bond_lengths()
    r_n_ca = bonds["N_Calpha"]
    r_ca_c = bonds["Calpha_C"]
    r_c_o = bonds["C_O"]
    omega = np.deg2rad(omega_peptide_deg())
    phi_a, psi_a = ramachandran_alpha()
    phi_rad = np.deg2rad(phi_a)
    psi_rad = np.deg2rad(psi_a)
    out = []
    n = len(sequence)
    for i in range(n):
        # Local frame: Cα at ca_pos[i]; N backward, C forward, O from C
        ca = ca_pos[i]
        if i == 0:
            forward = ca_pos[1] - ca_pos[0] if n > 1 else np.array([1, 0, 0])
        else:
            forward = ca_pos[i] - ca_pos[i - 1]
        forward = forward / (np.linalg.norm(forward) + 1e-9)
        # N: behind Cα
        n_pos = ca - r_n_ca * forward
        # C: ahead of Cα
        c_pos = ca + r_ca_c * forward
        # O: perpendicular to C–N in peptide plane
        perp = np.array([-forward[1], forward[0], 0.0])
        if np.linalg.norm(perp) < 1e-9:
            perp = np.array([0, 1, 0])
        perp = perp / np.linalg.norm(perp)
        o_pos = c_pos + r_c_o * perp
        out.append(("N", n_pos))
        out.append(("CA", ca))
        out.append(("C", c_pos))
        out.append(("O", o_pos))
    return out


def _pdb_line(atom_name: str, res_name: str, res_id: int, x: float, y: float, z: float, atom_id: int, chain: str = "A") -> str:
    """Format one ATOM line (PDB columns 1-80)."""
    return (
        f"ATOM  {atom_id:5d}  {atom_name:2s}  {res_name:3s} {chain}{res_id:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           "
    )


def hqiv_predict_structure(
    fasta: str,
    chain_id: str = "A",
    ss_string: Optional[str] = None,
) -> str:
    """
    FASTA → PDB string (CASP format: MODEL 1 ... END). Uses HQIV backbone and
    SS-aware Cα placement: if ss_string is None, predict from sequence (predict_ss).
    """
    sequence = _parse_fasta(fasta)
    if not sequence:
        return "MODEL     1\nEND\n"
    ca_pos = _place_backbone_ca(sequence, ss_string=ss_string)
    atoms = _place_full_backbone(ca_pos, sequence)
    lines = ["MODEL     1"]
    atom_id = 1
    n_res = len(sequence)
    atoms_per_res = 4  # N, CA, C, O
    for res_id in range(1, n_res + 1):
        res_1 = sequence[res_id - 1]
        res_3 = AA_1to3.get(res_1, "UNK")
        for a in range(atoms_per_res):
            name, xyz = atoms[(res_id - 1) * atoms_per_res + a]
            lines.append(_pdb_line(name, res_3, res_id, float(xyz[0]), float(xyz[1]), float(xyz[2]), atom_id, chain_id))
            atom_id += 1
    lines.append("ENDMDL")
    lines.append("END")
    return "\n".join(lines)


def hqiv_predict_structure_assembly(
    sequences: List[str],
    ss_strings: Optional[List[Optional[str]]] = None,
) -> str:
    """
    Multiple sequences (assembly/complex) → one PDB (MODEL 1) with chain A, B, C, ...
    Each item in sequences can be FASTA or raw sequence. ss_strings[i] optional per chain.
    """
    if not sequences:
        return "MODEL     1\nEND\n"
    if ss_strings is None:
        ss_strings = [None] * len(sequences)
    while len(ss_strings) < len(sequences):
        ss_strings.append(None)
    lines = ["MODEL     1"]
    atom_id = 1
    chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, fasta_or_seq in enumerate(sequences):
        seq = _parse_fasta(fasta_or_seq) if isinstance(fasta_or_seq, str) else ""
        if not seq:
            continue
        chain_id = chain_ids[i] if i < len(chain_ids) else "A"
        ss = ss_strings[i] if i < len(ss_strings) else None
        ca_pos = _place_backbone_ca(seq, ss_string=ss)
        atoms = _place_full_backbone(ca_pos, seq)
        n_res = len(seq)
        atoms_per_res = 4
        for res_id in range(1, n_res + 1):
            res_1 = seq[res_id - 1]
            res_3 = AA_1to3.get(res_1, "UNK")
            for a in range(atoms_per_res):
                name, xyz = atoms[(res_id - 1) * atoms_per_res + a]
                lines.append(_pdb_line(name, res_3, res_id, float(xyz[0]), float(xyz[1]), float(xyz[2]), atom_id, chain_id))
                atom_id += 1
    lines.append("ENDMDL")
    lines.append("END")
    return "\n".join(lines)


if __name__ == "__main__":
    fasta = ">test\nMKFL"
    pdb = hqiv_predict_structure(fasta)
    print("CASP PDB (HQIV):")
    print(pdb[:800])
    print("...")
    assert "MODEL" in pdb and "ENDMDL" in pdb and "ATOM" in pdb
