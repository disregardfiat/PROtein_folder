"""
Ribosome tunnel EM field and null-space builder.

Given a ribosome PDB and a chosen tunnel axis / PTC origin, this module:
  - extracts atoms near the tunnel and lip,
  - builds a static EMField for that region, and
  - computes a boolean tunnel mask (null search space) from a 1D radius profile
    with constriction + vestibule.

The resulting arrays (field potential, origin/size/res, tunnel_mask) can be saved
once (e.g. as .npz) and reused by extrusion / co-translational folding code.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Iterable, List, Tuple

import numpy as np

from .em_field_pipeline import Atom as EMAtom, EMField


@dataclasses.dataclass
class TunnelProfile:
    """
    Simple 1D radius profile r_allowed(s) along the tunnel axis.

    s is the distance along the axis from the PTC origin (Å).
    """

    r_ptc: float = 6.0
    r_min: float = 4.0
    r_vest: float = 10.0
    s_constriction_start: float = 20.0
    s_constriction_end: float = 40.0
    s_vestibule_start: float = 50.0
    s_vestibule_end: float = 70.0

    def radius(self, s: float) -> float:
        """
        Allowed tunnel radius at position s along axis (Å).
        Piecewise: PTC → constriction → vestibule → lip.
        """
        if s <= self.s_constriction_start:
            return self.r_ptc
        if s <= self.s_constriction_end:
            # Linear ramp down to r_min
            t = (s - self.s_constriction_start) / max(
                self.s_constriction_end - self.s_constriction_start, 1e-6
            )
            return (1.0 - t) * self.r_ptc + t * self.r_min
        if s <= self.s_vestibule_start:
            return self.r_min
        if s <= self.s_vestibule_end:
            # Linear ramp up to vestibule radius
            t = (s - self.s_vestibule_start) / max(
                self.s_vestibule_end - self.s_vestibule_start, 1e-6
            )
            return (1.0 - t) * self.r_min + t * self.r_vest
        return self.r_vest


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return v / n


def _parse_pdb_atoms(path: str) -> List[Tuple[str, str, int, np.ndarray]]:
    """
    Minimal PDB parser: returns (atom_name, resname, resid, xyz) for ATOM/HETATM.
    """
    out: List[Tuple[str, str, int, np.ndarray]] = []
    with open(path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                name = line[12:16].strip()
                resname = line[17:20].strip()
                resid = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue
            out.append((name, resname, resid, np.array([x, y, z], dtype=float)))
    return out


def _parse_cif_atoms(path: str) -> List[Tuple[str, str, int, np.ndarray]]:
    """
    Minimal mmCIF _atom_site parser: returns (atom_name, resname, resid, xyz).
    Supports quoted values (e.g. "O5'") by stripping quotes for atom name.
    """
    out: List[Tuple[str, str, int, np.ndarray]] = []
    idx_x = idx_y = idx_z = idx_atom = idx_comp = idx_seq = -1
    in_loop = False
    col_count = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "loop_":
                in_loop = True
                col_count = 0
                idx_x = idx_y = idx_z = idx_atom = idx_comp = idx_seq = -1
                continue
            if in_loop and line.startswith("_atom_site."):
                col_count += 1
                if "Cartn_x" in line:
                    idx_x = col_count - 1
                elif "Cartn_y" in line:
                    idx_y = col_count - 1
                elif "Cartn_z" in line:
                    idx_z = col_count - 1
                elif "label_atom_id" in line:
                    idx_atom = col_count - 1
                elif "label_comp_id" in line:
                    idx_comp = col_count - 1
                elif "label_seq_id" in line:
                    idx_seq = col_count - 1
                continue
            if in_loop and idx_x >= 0 and (line.startswith("ATOM") or line.startswith("HETATM")):
                parts = line.split()
                if len(parts) <= max(idx_x, idx_y, idx_z, idx_atom, idx_comp, idx_seq):
                    continue
                try:
                    x = float(parts[idx_x])
                    y = float(parts[idx_y])
                    z = float(parts[idx_z])
                except (ValueError, IndexError, TypeError):
                    continue
                name = parts[idx_atom].strip('"')
                resname = parts[idx_comp].strip('"') if idx_comp >= 0 else "UNK"
                try:
                    resid = int(parts[idx_seq]) if idx_seq >= 0 else 0
                except (ValueError, TypeError):
                    resid = 0
                out.append((name, resname, resid, np.array([x, y, z], dtype=float)))
            elif in_loop and line.startswith("_"):
                in_loop = False
    return out


def _load_atoms_from_file(path: str) -> List[Tuple[str, str, int, np.ndarray]]:
    """Dispatch to PDB or CIF parser by extension."""
    path_lower = path.lower()
    if path_lower.endswith(".cif") or path_lower.endswith(".mmcif"):
        return _parse_cif_atoms(path)
    return _parse_pdb_atoms(path)


def select_tunnel_atoms(
    atoms: Iterable[Tuple[str, str, int, np.ndarray]],
    ptc_origin: np.ndarray,
    axis: np.ndarray,
    tunnel_length: float,
    capture_radius: float = 25.0,
    extra_lip: float = 15.0,
) -> List[EMAtom]:
    """
    Select atoms near the tunnel and lip from a full ribosome.

    Keeps atoms whose projection s along axis lies in [0, tunnel_length+extra_lip]
    and whose perpendicular distance to the axis is <= capture_radius.
    """
    ptc_origin = np.asarray(ptc_origin, dtype=float)
    axis = _normalize(axis)
    out: List[EMAtom] = []
    for name, resname, resid, xyz in atoms:
        v = xyz - ptc_origin
        s = float(np.dot(v, axis))
        if s < -5.0 or s > tunnel_length + extra_lip:
            continue
        r_para = s * axis
        r_perp = v - r_para
        d_perp = float(np.linalg.norm(r_perp))
        if d_perp > capture_radius:
            continue
        out.append(EMAtom(name, resname, resid, xyz))
    return out


def build_ribosome_tunnel_field(
    pdb_path: str,
    ptc_origin: np.ndarray,
    axis: np.ndarray,
    tunnel_length: float,
    field_res: float = 1.0,
    padding: float = 15.0,
    capture_radius: float = 25.0,
    extra_lip: float = 15.0,
) -> Tuple[EMField, List[EMAtom]]:
    """
    Build a static EMField for the ribosome tunnel + lip region.

    Parameters
    ----------
    pdb_path : str
        Path to a ribosome PDB file (~250k atoms).
    ptc_origin : (3,) array-like
        Approximate position of the peptidyl-transfer center (Å).
    axis : (3,) array-like
        Tunnel extrusion axis; will be normalized.
    tunnel_length : float
        Length of tunnel segment to consider (Å).
    field_res : float
        EMField lattice resolution (Å).
    padding : float
        Extra padding around selected atoms (Å).
    capture_radius : float
        Radial cutoff from axis when selecting tunnel atoms (Å).
    extra_lip : float
        Additional length beyond tunnel_length to include lip atoms (Å).
    """
    raw_atoms = _load_atoms_from_file(pdb_path)
    sel_atoms = select_tunnel_atoms(
        raw_atoms,
        ptc_origin=ptc_origin,
        axis=axis,
        tunnel_length=tunnel_length,
        capture_radius=capture_radius,
        extra_lip=extra_lip,
    )
    if not sel_atoms:
        raise ValueError("No tunnel atoms selected from PDB; check ptc_origin/axis/tunnel_length.")

    pos = np.array([a.pos for a in sel_atoms])
    lo = np.min(pos, axis=0) - padding
    hi = np.max(pos, axis=0) + padding
    origin = lo
    size = hi - lo
    field = EMField(origin=origin, size=size, res=field_res)
    for a in sel_atoms:
        field.add_atom(a)
    return field, sel_atoms


def compute_tunnel_mask(
    field: EMField,
    ptc_origin: np.ndarray,
    axis: np.ndarray,
    tunnel_length: float,
    profile: TunnelProfile,
) -> np.ndarray:
    """
    Compute a boolean mask over the EMField grid for the tunnel null search space.

    A voxel center is "inside tunnel" if its projection s along axis is in [0, tunnel_length]
    and its perpendicular distance to the axis is <= profile.radius(s).
    """
    ptc_origin = np.asarray(ptc_origin, dtype=float)
    axis = _normalize(axis)
    origin = field.origin
    res = field.res
    shape = field.shape

    mask = np.zeros(shape, dtype=bool)
    # Iterate over grid; this is offline and done once per ribosome.
    for i in range(shape[0]):
        x = origin[0] + i * res
        for j in range(shape[1]):
            y = origin[1] + j * res
            for k in range(shape[2]):
                z = origin[2] + k * res
                p = np.array([x, y, z], dtype=float)
                v = p - ptc_origin
                s = float(np.dot(v, axis))
                if s < -1e-3 or s > tunnel_length + 1e-3:
                    continue
                r_para = s * axis
                r_perp = v - r_para
                d_perp = float(np.linalg.norm(r_perp))
                r_allowed = profile.radius(s)
                if d_perp <= r_allowed:
                    mask[i, j, k] = True
    return mask


def save_tunnel_field_npz(
    out_path: str,
    field: EMField,
    tunnel_mask: np.ndarray,
    ptc_origin: np.ndarray,
    axis: np.ndarray,
    tunnel_length: float,
    profile: TunnelProfile,
) -> None:
    """
    Save a precomputed tunnel EM field and mask as a portable .npz.
    """
    np.savez_compressed(
        out_path,
        potential=field.potential,
        origin=field.origin,
        size=field.size,
        res=np.array(field.res, dtype=float),
        tunnel_mask=tunnel_mask.astype(np.uint8),
        ptc_origin=np.asarray(ptc_origin, dtype=float),
        axis=_normalize(axis),
        tunnel_length=float(tunnel_length),
        profile_dataclass=np.array(
            [
                profile.r_ptc,
                profile.r_min,
                profile.r_vest,
                profile.s_constriction_start,
                profile.s_constriction_end,
                profile.s_vestibule_start,
                profile.s_vestibule_end,
            ],
            dtype=float,
        ),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a static EM field + tunnel mask for a ribosome tunnel region."
    )
    parser.add_argument("pdb", help="Ribosome PDB path.")
    parser.add_argument("out", help="Output .npz path for tunnel field/mask.")
    parser.add_argument(
        "--ptc-origin",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        required=True,
        help="PTC origin in Å (PDB coordinates).",
    )
    parser.add_argument(
        "--axis",
        nargs=3,
        type=float,
        metavar=("AX", "AY", "AZ"),
        default=[0.0, 0.0, 1.0],
        help="Tunnel axis vector (default +Z).",
    )
    parser.add_argument(
        "--tunnel-length",
        type=float,
        required=True,
        help="Tunnel length in Å (distance from PTC along axis).",
    )
    parser.add_argument(
        "--field-res",
        type=float,
        default=1.0,
        help="EM field lattice resolution in Å (default 1.0).",
    )
    parser.add_argument(
        "--capture-radius",
        type=float,
        default=25.0,
        help="Radial cutoff from axis when selecting tunnel atoms (Å).",
    )
    parser.add_argument(
        "--extra-lip",
        type=float,
        default=15.0,
        help="Extra length beyond tunnel_length to include lip atoms (Å).",
    )
    args = parser.parse_args()

    ptc_origin = np.array(args.ptc_origin, dtype=float)
    axis = np.array(args.axis, dtype=float)
    field, _ = build_ribosome_tunnel_field(
        args.pdb,
        ptc_origin=ptc_origin,
        axis=axis,
        tunnel_length=args.tunnel_length,
        field_res=args.field_res,
        capture_radius=args.capture_radius,
        extra_lip=args.extra_lip,
    )
    profile = TunnelProfile()
    tunnel_mask = compute_tunnel_mask(
        field,
        ptc_origin=ptc_origin,
        axis=axis,
        tunnel_length=args.tunnel_length,
        profile=profile,
    )
    save_tunnel_field_npz(
        args.out,
        field=field,
        tunnel_mask=tunnel_mask,
        ptc_origin=ptc_origin,
        axis=axis,
        tunnel_length=args.tunnel_length,
        profile=profile,
    )
    print(f"Wrote tunnel field + mask to {args.out}")

