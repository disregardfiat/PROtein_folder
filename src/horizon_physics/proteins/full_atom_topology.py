"""
Full heavy-atom polymer topology for HQIV + OSHoracle.

Exposes :class:`FullHeavyAtomChain` with ``bond_edges`` for ``folding_energy.e_tot_polymer_with_bonds``
and per-residue atom spans for sparse OSH blocks.

**Side-chain heavy atoms** beyond CB are placed for ``G,Y,D,P,E,T,W`` (chignolin) and **A** (Ala is CB-only).
Others: backbone + CB only until templates expand.

Lean alignment: backbone lengths from ``peptide_backbone.backbone_bond_lengths``; aqueous screening via
``energy_kwargs`` ``em_scale`` (see ``hqiv_lean_folding.em_scale_aqueous``) on the OSH caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .casp_submission import _place_full_backbone, AA_1to3
from .full_protein_minimizer import _add_cb
from .peptide_backbone import backbone_bond_lengths
from .side_chain_placement import chi_angles_for_residue


def _u(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return np.asarray(v, dtype=float) / n


def _rot(v: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    nrm = float(np.linalg.norm(axis))
    if nrm < 1e-12:
        return v.copy()
    axis = axis / nrm
    rad = np.deg2rad(float(deg))
    c, s = np.cos(rad), np.sin(rad)
    return v * c + np.cross(axis, v) * s + axis * (np.dot(axis, v) * (1 - c))


def _place_sc_branch(
    root: np.ndarray,
    axis_start: np.ndarray,
    axis_end: np.ndarray,
    ref_point: np.ndarray,
    bond_len: float,
    dihedral_deg: float,
) -> np.ndarray:
    axis = _u(axis_end - axis_start)
    v0 = _u(np.cross(axis, _u(ref_point - axis_start)))
    if float(np.linalg.norm(np.cross(axis, v0))) < 1e-6:
        v0 = _u(np.cross(axis, np.array([0.0, 1.0, 0.0])))
    base = _u(np.cross(v0, axis)) * float(bond_len)
    base = _rot(base, axis, float(dihedral_deg))
    return root + base


def _tyr_ring_coords(cg: np.ndarray, cap_cb: np.ndarray, cap_ca: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    ucb = _u(cg - cap_cb)
    plane_n = _u(np.cross(ucb, cap_ca - cap_cb))
    u1 = _u(np.cross(plane_n, ucb))
    r = 1.40
    cd1 = cg + r * u1
    cd2 = cg + r * _rot(u1, plane_n, 120.0)
    ce1 = cd1 + r * _u(_u(cg - cd2) + _u(cg - cd1))
    ce2 = cd2 + r * _u(_u(cg - cd1) + _u(cg - cd2))
    cz = 0.5 * (ce1 + ce2) + 0.35 * r * plane_n
    oh_d = np.cross(ce1 - cz, ce2 - cz)
    if float(np.linalg.norm(oh_d)) < 1e-9:
        oh_d = plane_n
    oh = cz + 1.36 * _u(oh_d)
    return [("CD1", cd1), ("CD2", cd2), ("CE1", ce1), ("CE2", ce2), ("CZ", cz), ("OH", oh)]


def _trp_side_coords(cg: np.ndarray, cap_cb: np.ndarray, cap_ca: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    ucb = _u(cg - cap_cb)
    n_plane = _u(np.cross(ucb, cap_ca - cap_cb))
    u1 = _u(np.cross(n_plane, ucb))
    r = 1.40
    cd2 = cg + r * u1
    cd1 = cg + r * _rot(u1, n_plane, 52.0)
    ne1 = cd1 + 1.38 * _u(np.cross(_u(cd1 - cg), n_plane))
    ce2 = cd2 + r * _u(np.cross(n_plane, _u(cd2 - cg)))
    cz2 = ne1 + r * _u(_u(ce2 - ne1))
    ce3 = ce2 + r * _u(cg - cd2)
    cz3 = ce3 + r * _u(ce2 - cd2)
    ch2 = 0.5 * (cz3 + cd2) + 0.15 * r * n_plane
    return [
        ("CD1", cd1),
        ("CD2", cd2),
        ("NE1", ne1),
        ("CE2", ce2),
        ("CZ2", cz2),
        ("CE3", ce3),
        ("CZ3", cz3),
        ("CH2", ch2),
    ]


def z_list_from_names(names: List[str]) -> np.ndarray:
    return np.array([_z_from_name(n) for n in names], dtype=np.int32)


def _z_from_name(name: str) -> int:
    s = name.strip().upper()
    if not s:
        return 6
    c0 = s[0]
    if c0 == "O":
        return 8
    if c0 == "N":
        return 7
    if c0 == "S":
        return 16
    return 6


def _append(atoms: List[Tuple[str, np.ndarray]], name: str, xyz: np.ndarray) -> int:
    atoms.append((name, np.asarray(xyz, dtype=float).copy()))
    return len(atoms) - 1


@dataclass
class FullHeavyAtomChain:
    sequence: str
    names: List[str]
    positions: np.ndarray
    z: np.ndarray
    bond_edges: List[Tuple[int, int, float]]
    residue_ranges: List[Tuple[int, int]]
    ca_atom_indices: np.ndarray

    def copy_positions(self) -> np.ndarray:
        return np.asarray(self.positions, dtype=float).copy()


def build_full_heavy_chain(
    sequence: str,
    ca_trace: np.ndarray,
    *,
    include_sidechain_heavy: bool = True,
) -> FullHeavyAtomChain:
    seq = sequence.upper().strip()
    ca = np.asarray(ca_trace, dtype=float)
    bl = backbone_bond_lengths()
    r_nca = float(bl["N_Calpha"])
    r_cac = float(bl["Calpha_C"])
    r_co = float(bl["C_O"])
    r_cn = float(bl["C_N"])
    r_ca_cb = float(bl["Calpha_Cbeta"])

    bb = _place_full_backbone(ca, seq)
    bb_cb = _add_cb(bb, seq) if include_sidechain_heavy else bb

    atoms: List[Tuple[str, np.ndarray]] = []
    edges: List[Tuple[int, int, float]] = []
    res_ranges: List[Tuple[int, int]] = []
    ca_inds: List[int] = []

    src = 0
    prev_i_c: int | None = None
    n_res = len(seq)

    for ri, aa in enumerate(seq):
        start_atom = len(atoms)
        n_xyz = bb_cb[src][1]
        ca_xyz = bb_cb[src + 1][1]
        if aa == "G":
            c_xyz = bb_cb[src + 2][1]
            o_xyz = bb_cb[src + 3][1]
            src += 4
            has_cb = False
        else:
            cb_xyz = bb_cb[src + 2][1]
            c_xyz = bb_cb[src + 3][1]
            o_xyz = bb_cb[src + 4][1]
            src += 5
            has_cb = True

        three = AA_1to3.get(aa, "ALA")
        chi = chi_angles_for_residue(three) if include_sidechain_heavy else {}

        i_n = _append(atoms, "N", n_xyz)
        i_ca = _append(atoms, "CA", ca_xyz)
        ca_inds.append(i_ca)
        edges.append((i_n, i_ca, r_nca))
        if prev_i_c is not None:
            edges.append((prev_i_c, i_n, r_cn))

        chi1 = float(chi.get("chi1_deg", -60.0)) if chi else -60.0
        chi2 = float(chi.get("chi2_deg", 180.0)) if chi else 180.0

        if aa == "G":
            i_c = _append(atoms, "C", c_xyz)
            i_o = _append(atoms, "O", o_xyz)
            edges.append((i_ca, i_c, r_cac))
            edges.append((i_c, i_o, r_co))
            prev_i_c = i_c
        elif not has_cb:
            raise RuntimeError("internal: non-Gly without CB")
        else:
            i_cb = _append(atoms, "CB", cb_xyz)
            edges.append((i_ca, i_cb, r_ca_cb))

            i_c = _append(atoms, "C", c_xyz)
            i_o = _append(atoms, "O", o_xyz)
            edges.append((i_ca, i_c, r_cac))
            edges.append((i_c, i_o, r_co))

            if include_sidechain_heavy and aa in ("D", "E", "Y", "W", "T", "P"):
                if aa == "D":
                    cg = _place_sc_branch(cb_xyz, ca_xyz, cb_xyz, n_xyz, 1.52, chi1)
                    i_cg = _append(atoms, "CG", cg)
                    edges.append((i_cb, i_cg, 1.52))
                    v = _u(cb_xyz - ca_xyz)
                    od1 = _append(
                        atoms,
                        "OD1",
                        cg + 1.25 * _u(v + _u(np.cross(v, cg - cb_xyz))),
                    )
                    od2 = _append(
                        atoms,
                        "OD2",
                        cg + 1.25 * _u(v - _u(np.cross(v, cg - cb_xyz))),
                    )
                    edges.append((i_cg, od1, 1.25))
                    edges.append((i_cg, od2, 1.25))
                elif aa == "E":
                    cg = _place_sc_branch(cb_xyz, ca_xyz, cb_xyz, n_xyz, 1.52, chi1)
                    i_cg = _append(atoms, "CG", cg)
                    edges.append((i_cb, i_cg, 1.52))
                    cd = _place_sc_branch(cg, cb_xyz, cg, ca_xyz, 1.52, chi2)
                    i_cd = _append(atoms, "CD", cd)
                    edges.append((i_cg, i_cd, 1.52))
                    v = _u(cg - cb_xyz)
                    oe1 = _append(
                        atoms,
                        "OE1",
                        cd + 1.23 * _u(v + _u(np.cross(v, cd - cg))),
                    )
                    oe2 = _append(
                        atoms,
                        "OE2",
                        cd + 1.23 * _u(v - _u(np.cross(v, cd - cg))),
                    )
                    edges.append((i_cd, oe1, 1.23))
                    edges.append((i_cd, oe2, 1.23))
                elif aa == "Y":
                    cg = _place_sc_branch(cb_xyz, ca_xyz, cb_xyz, n_xyz, 1.52, chi1)
                    i_cg = _append(atoms, "CG", cg)
                    edges.append((i_cb, i_cg, 1.52))
                    ring = _tyr_ring_coords(cg, cb_xyz, ca_xyz)
                    i_cd1 = _append(atoms, ring[0][0], ring[0][1])
                    i_cd2 = _append(atoms, ring[1][0], ring[1][1])
                    i_ce1 = _append(atoms, ring[2][0], ring[2][1])
                    i_ce2 = _append(atoms, ring[3][0], ring[3][1])
                    i_cz = _append(atoms, ring[4][0], ring[4][1])
                    i_oh = _append(atoms, ring[5][0], ring[5][1])
                    r_ar = 1.40
                    edges.extend(
                        [
                            (i_cg, i_cd1, r_ar),
                            (i_cg, i_cd2, r_ar),
                            (i_cd1, i_ce1, r_ar),
                            (i_cd2, i_ce2, r_ar),
                            (i_ce1, i_cz, r_ar),
                            (i_ce2, i_cz, r_ar),
                            (i_cz, i_oh, 1.36),
                        ]
                    )
                elif aa == "W":
                    cg = _place_sc_branch(cb_xyz, ca_xyz, cb_xyz, n_xyz, 1.52, chi1)
                    i_cg = _append(atoms, "CG", cg)
                    edges.append((i_cb, i_cg, 1.52))
                    tc = _trp_side_coords(cg, cb_xyz, ca_xyz)
                    i_cd1 = _append(atoms, tc[0][0], tc[0][1])
                    i_cd2 = _append(atoms, tc[1][0], tc[1][1])
                    i_ne1 = _append(atoms, tc[2][0], tc[2][1])
                    i_ce2 = _append(atoms, tc[3][0], tc[3][1])
                    i_cz2 = _append(atoms, tc[4][0], tc[4][1])
                    i_ce3 = _append(atoms, tc[5][0], tc[5][1])
                    i_cz3 = _append(atoms, tc[6][0], tc[6][1])
                    i_ch2 = _append(atoms, tc[7][0], tc[7][1])
                    edges.extend(
                        [
                            (i_cg, i_cd1, 1.40),
                            (i_cg, i_cd2, 1.40),
                            (i_cd1, i_ne1, 1.38),
                            (i_ne1, i_cz2, 1.35),
                            (i_cz2, i_ce2, 1.38),
                            (i_cd2, i_ce2, 1.40),
                            (i_ce2, i_ce3, 1.40),
                            (i_ce3, i_cz3, 1.40),
                            (i_cz3, i_ch2, 1.40),
                            (i_cd2, i_cz3, 1.40),
                            (i_ch2, i_cz2, 1.40),
                        ]
                    )
                elif aa == "T":
                    cg2 = _place_sc_branch(cb_xyz, ca_xyz, cb_xyz, n_xyz, 1.52, chi1 + 120.0)
                    og1 = _place_sc_branch(cb_xyz, ca_xyz, cb_xyz, n_xyz, 1.43, chi1 - 110.0)
                    i_c2 = _append(atoms, "CG2", cg2)
                    i_o1 = _append(atoms, "OG1", og1)
                    edges.append((i_cb, i_c2, 1.52))
                    edges.append((i_cb, i_o1, 1.43))
                elif aa == "P":
                    cg = _place_sc_branch(cb_xyz, ca_xyz, cb_xyz, n_xyz, 1.52, chi1)
                    i_cg = _append(atoms, "CG", cg)
                    edges.append((i_cb, i_cg, 1.52))
                    cd = _place_sc_branch(cg, cb_xyz, cg, ca_xyz, 1.52, -40.0)
                    i_cd = _append(atoms, "CD", cd)
                    edges.append((i_cg, i_cd, 1.52))
                    edges.append((i_cd, i_n, 1.47))

            prev_i_c = i_c
        end_atom = len(atoms)
        res_ranges.append((start_atom, end_atom))

    names = [nm for nm, _ in atoms]
    pos = np.array([xyz for _, xyz in atoms], dtype=float)
    z = z_list_from_names(names)
    return FullHeavyAtomChain(
        sequence=seq,
        names=names,
        positions=pos,
        z=z,
        bond_edges=edges,
        residue_ranges=res_ranges,
        ca_atom_indices=np.array(ca_inds, dtype=int),
    )


def full_heavy_chain_energy_budget(
    chain: FullHeavyAtomChain,
    *,
    fast_local_theta: bool = False,
    include_clash: bool = True,
    ca_target: Optional[np.ndarray] = None,
    k_ca_target: float = 0.0,
    em_scale: float = 1.0,
    neighbor_cutoff: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, float]:
    """
    Full-atom energy **budget** (evaluation only): same decomposition as
    :func:`horizon_physics.proteins.folding_energy.full_atom_polymer_energy_budget` on this chain.

    Use this to **score** a heavy-atom layout built from a Cα trace, then feed scalars (e.g.
    ``e_objective_ev``, ``e_horizon_em_pole_magnitude_sum_ev``, ``e_clash_ev``) back into a Cα OSH
    outer loop — without running :func:`horizon_physics.proteins.osh_oracle_full_atom.minimize_full_heavy_with_osh_oracle`
    as the primary mover.

    ``kwargs`` are forwarded to :func:`~.folding_energy.full_atom_polymer_energy_budget` for bond /
    horizon parameters (``r_bond_min``, ``r_horizon``, ``k_horizon``, …).
    """
    from .folding_energy import full_atom_polymer_energy_budget

    return full_atom_polymer_energy_budget(
        chain.positions,
        chain.z,
        chain.bond_edges,
        fast_local_theta=bool(fast_local_theta),
        include_clash=bool(include_clash),
        ca_atom_indices=chain.ca_atom_indices,
        ca_target=ca_target,
        k_ca_target=float(k_ca_target),
        em_scale=float(em_scale),
        neighbor_cutoff=neighbor_cutoff,
        **kwargs,
    )


def chain_to_pdb_line_string(chain: FullHeavyAtomChain, chain_id: str = "A") -> str:
    """Minimal PDB (MODEL 1 … END) from a chain."""
    lines = ["MODEL     1"]
    aid = 1
    for ri, (a, b) in enumerate(chain.residue_ranges):
        res_1 = chain.sequence[ri]
        res_3 = AA_1to3.get(res_1, "UNK")
        for idx in range(a, b):
            nm = chain.names[idx]
            xyz = chain.positions[idx]
            nm4 = f"{nm.strip()[:4]:>4}"
            lines.append(
                f"ATOM  {aid:5d} {nm4} {res_3:3s} {chain_id}{ri + 1:4d}    "
                f"{float(xyz[0]):8.3f}{float(xyz[1]):8.3f}{float(xyz[2]):8.3f}  1.00  0.00"
            )
            aid += 1
    lines.append("ENDMDL")
    lines.append("END")
    return "\n".join(lines)
