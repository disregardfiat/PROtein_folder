"""
EM-field pipeline: lattice field, force-following relaxation, helical kinetic groups.

- Atom: each atom as its own class, radius/Θ from pyhqiv.
- EMField: 3D lattice field; build/update via local stencil (no n²).
- Helical segments: kinetic groups (rigid bodies) during relaxation.
- After chain is built: low-energy pockets become docking hot-spots for ligands/domains.

Pipeline: build whole protein at once (SS-aware, helices form naturally) → relax until
convergence. Helices move as rigid groups; geometries grow from the physics.

MIT License. Python 3.10+. Numpy.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple, Sequence

from ._hqiv_base import bond_length_from_theta, theta_for_atom
from .casp_submission import _parse_fasta, _place_backbone_ca, _place_full_backbone, AA_1to3
from .secondary_structure_predictor import predict_ss
from .peptide_backbone import backbone_bond_lengths
from .folding_energy import K_BOND_EM_RELAX

# Z_shell for folding_energy (Cα)
_Z_CA = 6

try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except ImportError:
    cKDTree = None
    _HAS_SCIPY = False


def _helix_runs(ss_string: str) -> List[Tuple[int, int]]:
    """Return [(start, end), ...] for each helical run (H)."""
    runs: List[Tuple[int, int]] = []
    i = 0
    n = len(ss_string)
    while i < n:
        if ss_string[i] == "H":
            start = i
            while i < n and ss_string[i] == "H":
                i += 1
            if i - start >= 2:  # at least 2 residues for a helix
                runs.append((start, i))
        else:
            i += 1
    return runs


def _bonded_neighbors(i: int, n: int) -> List[int]:
    """Bonded neighbors for backbone: N–CA–C–O, C–N (peptide)."""
    out = []
    if i > 0:
        out.append(i - 1)
    if i < n - 1:
        out.append(i + 1)
    if i >= 3 and (i % 4) == 0:  # N atom
        out.append(i - 3)  # C of prev residue
    elif i >= 1 and (i % 4) == 3 and i + 1 < n:  # C atom
        out.append(i + 1)  # N of next residue
    return list(set(out))


def _radius_from_theta(symbol: str, coordination: int = 1) -> float:
    """Effective radius (Å) from pyhqiv Θ. Θ/2 ≈ vdW-like scale."""
    theta = theta_for_atom(symbol, coordination)
    return max(0.5, theta * 0.5)


def equilibrium_bond_length_ang(prev_name: str, curr_name: str, bonds: Optional[dict] = None) -> float:
    """
    Equilibrium bond length (Å) for backbone pair (prev, curr). No constants:
    uses backbone_bond_lengths() from peptide_backbone (pyhqiv Θ-derived).
    """
    if bonds is None:
        bonds = backbone_bond_lengths()
    prev, curr = (prev_name or "C").strip().upper(), (curr_name or "C").strip().upper()
    if prev == "N" and curr == "CA":
        return float(bonds["N_Calpha"])
    if prev == "CA" and curr == "C":
        return float(bonds["Calpha_C"])
    if prev == "C" and curr == "O":
        return float(bonds["C_O"])
    if prev == "C" and curr == "N":
        return float(bonds["C_N"])
    if prev == "CA" and curr == "N":
        return float(bonds["N_Calpha"])
    # Fallback: use Θ from pyhqiv for both atoms
    s_prev = prev[0] if prev else "C"
    s_curr = curr[0] if curr else "C"
    return float(bond_length_from_theta(theta_for_atom(s_prev, 2), theta_for_atom(s_curr, 2), 1.0))


# Coupling label: +, -, ++, -- per atom (coupling point for torsion/coupling DOFs). From pyhqiv when available.
CouplingLabel = Optional[str]  # "+", "-", "++", "--", or None

# Simple partial charges for backbone (N, CA, C, O) — from physics; could move to pyhqiv
_BACKBONE_CHARGE = {"N": -0.3, "CA": 0.0, "C": 0.5, "O": -0.5}


class Atom:
    """
    Backbone atom with position, radius/Θ from pyhqiv, and optional coupling label.
    Allowed angles for bonds come from pyhqiv (discrete_dof / coupling_angle_energy_profile);
    theta_local is the Θ (Å) at this atom for use in allowed-angle profiles.
    coupling_label: "+", "-", "++", or "--" acting as coupling point (from pyhqiv when available).
    """

    def __init__(
        self,
        name: str,
        resname: str,
        resid: int,
        pos: np.ndarray,
        charge: Optional[float] = None,
        radius: Optional[float] = None,
        theta_local: Optional[float] = None,
        coupling_label: CouplingLabel = None,
    ):
        self.name = name
        self.resname = resname
        self.resid = resid
        self.pos = np.array(pos, dtype=float)
        symbol = name[0] if name else "C"
        self.charge = charge if charge is not None else _BACKBONE_CHARGE.get(name, 0.0)
        self.radius = radius if radius is not None else _radius_from_theta(symbol)
        # Θ (Å) at this atom — from pyhqiv theta_for_atom; used for allowed-angle profiles
        self.theta_local = theta_local if theta_local is not None else theta_for_atom(symbol, 2)
        # Coupling point for torsion/coupling DOFs: +, -, ++, -- (from pyhqiv when available)
        self.coupling_label = coupling_label

    def __repr__(self) -> str:
        return f"{self.name}{self.resid}@{self.pos.round(2)}"


def atoms_to_pdb(atoms: List[Atom], chain_id: str = "A") -> str:
    """Convert Atom list to CASP-format PDB string (MODEL 1 ... END)."""
    lines = ["MODEL     1"]
    atom_id = 1
    for a in atoms:
        lines.append(
            f"ATOM  {atom_id:5d}  {a.name:2s}  {a.resname:3s} {chain_id}{a.resid:4d}    "
            f"{float(a.pos[0]):8.3f}{float(a.pos[1]):8.3f}{float(a.pos[2]):8.3f}  1.00  0.00           "
        )
        atom_id += 1
    lines.append("ENDMDL")
    lines.append("END")
    return "\n".join(lines)


class EMField:
    """3D lattice field. Step over voxels to build/update (local stencil, no n²)."""

    def __init__(self, origin: np.ndarray, size: np.ndarray, res: float = 0.8):
        self.res = res
        self.origin = np.array(origin, dtype=float)
        self.size = np.array(size, dtype=float)
        self.shape = tuple(((self.size) / res).astype(int) + 1)
        self.potential = np.zeros(self.shape, dtype=float)

    def world_to_grid(self, pos: np.ndarray) -> Tuple[int, int, int]:
        idx = ((pos - self.origin) / self.res).astype(int)
        return tuple(np.clip(idx, 0, np.array(self.shape) - 1))

    def add_atom(self, atom: Atom, cutoff: float = 8.5) -> None:
        """Rasterize one atom onto the lattice (vectorized stencil)."""
        center = np.array(self.world_to_grid(atom.pos), dtype=int)
        rmax = int(cutoff / self.res)
        d = np.arange(-rmax, rmax + 1, dtype=np.float64) * self.res
        dx, dy, dz = np.meshgrid(d, d, d, indexing="ij")
        r = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-8
        steric = 12.0 * ((atom.radius / r) ** 12 - (atom.radius / r) ** 6)
        elec = 4.0 * (atom.charge / r)
        contrib = steric + elec
        gx = center[0] + np.arange(-rmax, rmax + 1, dtype=int)
        gy = center[1] + np.arange(-rmax, rmax + 1, dtype=int)
        gz = center[2] + np.arange(-rmax, rmax + 1, dtype=int)
        gx, gy, gz = np.meshgrid(gx, gy, gz, indexing="ij")
        mask = (
            (gx >= 0) & (gx < self.shape[0]) &
            (gy >= 0) & (gy < self.shape[1]) &
            (gz >= 0) & (gz < self.shape[2])
        )
        np.add.at(self.potential, (gx[mask], gy[mask], gz[mask]), contrib[mask])

    def force_at(self, pos: np.ndarray) -> np.ndarray:
        """Return -∇potential (move along field lines)."""
        g = self.world_to_grid(pos)
        f = np.zeros(3)
        for ax in range(3):
            p1 = list(g)
            p1[ax] = min(p1[ax] + 1, self.shape[ax] - 1)
            p2 = list(g)
            p2[ax] = max(p2[ax] - 1, 0)
            df = self.potential[tuple(p1)] - self.potential[tuple(p2)]
            f[ax] = -df / (2 * self.res)
        return f

    def forces_at_all(self, positions: np.ndarray) -> np.ndarray:
        """Return (N, 3) forces for all positions (vectorized)."""
        idx = ((positions - self.origin) / self.res).astype(int)
        idx = np.clip(idx, 0, np.array(self.shape) - 1)
        n = positions.shape[0]
        forces = np.zeros((n, 3))
        for ax in range(3):
            idx_p1 = idx.copy()
            idx_p1[:, ax] = np.minimum(idx[:, ax] + 1, self.shape[ax] - 1)
            idx_p2 = idx.copy()
            idx_p2[:, ax] = np.maximum(idx[:, ax] - 1, 0)
            p1 = self.potential[idx_p1[:, 0], idx_p1[:, 1], idx_p1[:, 2]]
            p2 = self.potential[idx_p2[:, 0], idx_p2[:, 1], idx_p2[:, 2]]
            forces[:, ax] = -(p1 - p2) / (2 * self.res)
        return forces

    def expand(self, new_pos: np.ndarray, padding: float = 20.0) -> bool:
        """Grow box + re-zero field. Returns True if expanded."""
        expanded = False
        for i in range(3):
            if new_pos[i] < self.origin[i]:
                self.origin[i] = new_pos[i] - padding
                expanded = True
            if new_pos[i] > self.origin[i] + self.size[i]:
                self.size[i] = new_pos[i] - self.origin[i] + padding
                expanded = True
        if expanded:
            self.shape = tuple(((self.size) / self.res).astype(int) + 1)
            self.potential = np.zeros(self.shape, dtype=float)
        return expanded


class CoTranslationalAssembler:
    """
    EM-field folding with optional co-translational flavour: field updates, rebound, relax.

    Temperature parameter represents the effective thermal environment (e.g. body temperature),
    and can be used to scale stochastic/annealing moves in future refinements.
    """

    def __init__(
        self,
        tunnel_exit: Optional[np.ndarray] = None,
        tunnel_axis: Optional[np.ndarray] = None,
        box_origin: Optional[np.ndarray] = None,
        box_size: Optional[np.ndarray] = None,
        field_res: float = 0.8,
        batch_size: int = 50,
        temperature: float = 310.0,
    ):
        self.tunnel_exit = np.array(tunnel_exit if tunnel_exit is not None else [-40.0, 0.0, 0.0])
        self.tunnel_axis = np.array(tunnel_axis if tunnel_axis is not None else [1.0, 0.0, 0.0])
        self.tunnel_axis = self.tunnel_axis / np.linalg.norm(self.tunnel_axis)
        origin = box_origin if box_origin is not None else np.array([-80.0, -80.0, -80.0])
        size = box_size if box_size is not None else np.array([200.0, 200.0, 200.0])
        self.field = EMField(origin=origin, size=size, res=field_res)
        self.atoms: List[Atom] = []
        self.batch_size = batch_size
        self._sequence = ""
        self._ss_string: str = ""
        # Effective temperature (Kelvin-like scale); used to modulate relaxation / noise.
        self.temperature = float(temperature)
        # Ruler: residue addition rate calibrates step; anchor = lowest |F| for rotations
        self._step_size = 0.08
        self._anchor_idx: Optional[int] = None
        self._target_disp_per_step = 0.15  # Å per step (ruler)

    def _atoms_from_backbone(
        self, backbone: List[Tuple[str, np.ndarray]], sequence: str, start_resid: int = 1
    ) -> List[Atom]:
        """Convert (name, xyz) backbone to Atom list."""
        atoms: List[Atom] = []
        n_res = len(sequence)
        atoms_per_res = 4
        for i in range(n_res):
            res_1 = sequence[i]
            res_3 = AA_1to3.get(res_1, "UNK")
            resid = start_resid + i
            for j in range(atoms_per_res):
                name, xyz = backbone[i * atoms_per_res + j]
                atoms.append(Atom(name, res_3, resid, np.array(xyz)))
        return atoms

    def load_from_backbone_atoms(
        self,
        backbone: List[Tuple[str, np.ndarray]],
        sequence: str,
        ss_string: Optional[str] = None,
    ) -> None:
        """
        Load an existing full backbone (N,CA,C,O per residue) without re-placing Cα.

        Used after co-translational tunnel extrusion: positions stay in the same frame;
        the 3D EM field is rebuilt from these atoms for ``relax_to_convergence`` (no
        COM shift toward ``tunnel_exit`` unlike ``load_from_fast_assembler``).
        """
        seq = _parse_fasta(sequence) if ">" in sequence or "\n" in sequence else sequence
        seq = "".join(c for c in seq.upper() if c in AA_1to3)
        if not seq or len(backbone) != 4 * len(seq):
            self.atoms = []
            self._sequence = ""
            self._ss_string = ""
            return
        self._sequence = seq
        if ss_string is None or len(ss_string) != len(seq):
            ss_string, _ = predict_ss(seq, window=5)
        self._ss_string = ss_string
        self.atoms = self._atoms_from_backbone(backbone, seq)
        self._rebuild_field()

    def load_from_fast_assembler(
        self, sequence: str, ss_string: Optional[str] = None
    ) -> None:
        """
        Build initial chain via fast assembler (Cα placement + backbone).
        Drops entire chain into the box; field is built from scratch.
        """
        seq = _parse_fasta(sequence) if ">" in sequence or "\n" in sequence else sequence
        seq = "".join(c for c in seq.upper() if c in AA_1to3)
        if not seq:
            return
        self._sequence = seq
        if ss_string is None or len(ss_string) != len(seq):
            ss_string, _ = predict_ss(seq, window=5)
        self._ss_string = ss_string
        ca_pos = _place_backbone_ca(seq, ss_string=ss_string)
        backbone = _place_full_backbone(ca_pos, seq)
        # Center chain near origin, shift so tunnel exit is at -X
        all_xyz = np.array([xyz for _, xyz in backbone])
        com = np.mean(all_xyz, axis=0)
        shift = self.tunnel_exit - com + self.tunnel_axis * 10.0
        self.atoms = []
        for i, (name, xyz) in enumerate(backbone):
            res_1 = seq[i // 4]
            res_3 = AA_1to3.get(res_1, "UNK")
            self.atoms.append(Atom(name, res_3, i // 4 + 1, np.array(xyz) + shift))
        self._rebuild_field()

    def _fit_field_to_atoms(self, padding: float = 25.0) -> None:
        """Resize field to tightly fit atoms (fewer voxels = faster rebuild)."""
        if not self.atoms:
            return
        pos = np.array([a.pos for a in self.atoms])
        lo = np.min(pos, axis=0) - padding
        hi = np.max(pos, axis=0) + padding
        self.field.origin = lo
        self.field.size = hi - lo
        self.field.shape = tuple((self.field.size / self.field.res).astype(int) + 1)
        self.field.potential = np.zeros(self.field.shape, dtype=float)

    def _rebuild_field(self, tight_box: bool = True) -> None:
        """Rebuild field from all atoms. tight_box: fit grid to atoms for speed."""
        if tight_box and self.atoms:
            self._fit_field_to_atoms(padding=25.0)
        else:
            self.field.potential.fill(0)
        for a in self.atoms:
            self.field.add_atom(a)

    def _hke_ca_step(
        self,
        step_size: float = 0.15,
        r_bond_min: float = 2.5,
        r_bond_max: float = 6.0,
    ) -> float:
        """
        One HKE step: gradient descent on Cα (horizon + bonds + clash), project bonds,
        reconstruct full backbone, update atoms. Returns max Cα displacement.
        """
        if not self._sequence or not self.atoms:
            return 0.0
        ca_pos = np.array([a.pos for a in self.atoms if a.name == "CA"])
        n_res = len(ca_pos)
        if n_res < 2:
            return 0.0
        z_list = np.full(n_res, _Z_CA, dtype=np.int32)
        from .folding_energy import grad_full
        from .gradient_descent_folding import _project_bonds

        grad = grad_full(
            ca_pos, z_list,
            include_bonds=True, include_horizon=True, include_clash=True,
        )
        g_norm = np.linalg.norm(grad)
        if g_norm < 1e-9:
            return 0.0
        step = step_size / (g_norm + 1e-9)
        ca_new = ca_pos - step * grad
        ca_new = _project_bonds(ca_new, r_min=r_bond_min, r_max=r_bond_max)
        max_disp = float(np.max(np.linalg.norm(ca_new - ca_pos, axis=1)))
        backbone = _place_full_backbone(ca_new, self._sequence)
        for i, (name, xyz) in enumerate(backbone):
            self.atoms[i].pos = np.array(xyz)
        return max_disp

    def _relax_step(
        self,
        step_size: float,
        drag_factor: float = 0.35,
        rebound_tail: int = 9,
        free_fold: bool = False,
        high_potential_first: bool = False,
    ) -> float:
        """
        One relax step: anchor = lowest |F|, drag peers, apply displacements.
        free_fold: no tunnel rebounding (chain can fold freely).
        high_potential_first: scale displacement by |F| so high-force atoms move more.
        Returns max displacement (for ruler / adaptive step).
        """
        n = len(self.atoms)
        positions = np.array([a.pos for a in self.atoms])
        forces = self.field.forces_at_all(positions)
        # Bond constraint: r_eq from pyhqiv (backbone_bond_lengths); spring from K_BOND
        bonds = backbone_bond_lengths()
        k_bond_em = float(K_BOND_EM_RELAX)  # eV/Å² for EM relax step; from pyhqiv when available
        for i, a in enumerate(self.atoms):
            prev_idx = i - 3 if a.name == "N" and i >= 3 else (i - 1 if i > 0 else -1)
            if prev_idx >= 0:
                prev = self.atoms[prev_idx]
                bond_vec = prev.pos - a.pos
                dist = np.linalg.norm(bond_vec)
                r_eq = equilibrium_bond_length_ang(prev.name, a.name, bonds)
                if dist > 1e-9 and dist > r_eq * 1.15:
                    forces[i] += k_bond_em * (dist - r_eq) * bond_vec / dist
            if not free_fold and n - i < rebound_tail:
                forces[i] = forces[i] + 1.2 * self.tunnel_axis

        # Anchor = atom with smallest |F| (most equilibrated); use for rotation pivot
        norms = np.linalg.norm(forces, axis=1)
        norms = np.where(norms < 1e-12, 1e12, norms)
        self._anchor_idx = int(np.argmin(norms))

        # Displacements: power-law scale (1/1 -> 1/4) instead of linear; light effect, doubled by
        # force from both sides of each attraction/repulsion
        disp = np.zeros((n, 3))
        f_norms = np.linalg.norm(forces, axis=1)
        f_max = float(np.max(f_norms)) + 1e-12
        power = 0.25  # 1/4: subtle (1/1->1/4); effect doubled by force from both sides
        for i in range(n):
            f = forces[i]
            nrm = np.linalg.norm(f) + 1e-9
            scale = (f_norms[i] / f_max) ** power if high_potential_first else 1.0
            disp[i] = step_size * scale * f / nrm

        # Drag: each atom pulls its bonded neighbors along
        for i in range(n):
            for j in _bonded_neighbors(i, n):
                disp[j] += drag_factor * disp[i]

        # Helical kinetic groups: helices move as rigid bodies (same translation for all atoms)
        if self._ss_string and len(self._ss_string) == len(self.atoms) // 4:
            for start, end in _helix_runs(self._ss_string):
                idx = [4 * r + a for r in range(start, end) for a in range(4)]
                if idx:
                    mean_disp = np.mean(disp[idx], axis=0)
                    for i in idx:
                        disp[i] = mean_disp

        # Apply
        max_disp = 0.0
        for i, a in enumerate(self.atoms):
            a.pos += disp[i]
            max_disp = max(max_disp, np.linalg.norm(disp[i]))
        return max_disp

    def _rotate_group_around_anchor(
        self, angle_deg: float = 2.0, nn_radius: float = 12.0, axis: Optional[np.ndarray] = None
    ) -> None:
        """
        Rotate NN group around anchor (lowest |F|). Uses anchor as pivot.
        axis: rotation axis; default = tunnel axis.
        """
        if self._anchor_idx is None or len(self.atoms) < 2:
            return
        ax = axis if axis is not None else self.tunnel_axis
        ax = np.array(ax) / (np.linalg.norm(ax) + 1e-9)
        pivot = self.atoms[self._anchor_idx].pos.copy()
        angle = np.deg2rad(angle_deg)
        c, s = np.cos(angle), np.sin(angle)
        R = c * np.eye(3) + (1 - c) * np.outer(ax, ax) + s * np.array(
            [[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]]
        )
        if _HAS_SCIPY and len(self.atoms) > 10:
            pos = np.array([a.pos for a in self.atoms])
            tree = cKDTree(pos)
            nn = tree.query_ball_point(pivot, nn_radius)
        else:
            nn = list(range(len(self.atoms)))
        for i in nn:
            rel = self.atoms[i].pos - pivot
            self.atoms[i].pos = pivot + (R @ rel)

    def add_residue(
        self,
        resname: str,
        atoms_in_res: List[dict],
        relax_steps: int = 12,
        step_size: Optional[float] = None,
        direction: Optional[np.ndarray] = None,
        drag_factor: float = 0.35,
        use_ruler: bool = True,
        alternate_hke: bool = True,
    ) -> None:
        """
        New residue emerges from the tunnel. Rebound, update field, relax along field lines.
        atoms_in_res: [{"name": "N", "offset": [0,0,0], "charge": -0.3}, ...]
        direction: optional growth direction; else from last two Cα or tunnel axis.
        use_ruler: adapt step from residue-addition rate (target disp per step).
        """
        if self.atoms:
            ca_list = [a.pos for a in self.atoms if a.name == "CA"]
            last_ca = ca_list[-1] if ca_list else self.atoms[-1].pos
            if direction is not None:
                direction = np.array(direction) / (np.linalg.norm(direction) + 1e-9)
            elif len(ca_list) >= 2:
                direction = (ca_list[-1] - ca_list[-2]) / (np.linalg.norm(ca_list[-1] - ca_list[-2]) + 1e-9)
            else:
                direction = self.tunnel_axis.copy()
        else:
            last_ca = self.tunnel_exit
            direction = self.tunnel_axis.copy() if direction is None else np.array(direction) / (np.linalg.norm(direction) + 1e-9)

        bonds = backbone_bond_lengths()
        r_ca_ca = 3.8  # approximate Cα–Cα spacing

        new_atoms = []
        for a in atoms_in_res:
            offset = np.array(a.get("offset", [0, 0, 0]))
            pos = last_ca + direction * r_ca_ca + offset
            atom = Atom(
                a["name"],
                resname,
                len(self.atoms) // 4 + 1,
                pos,
                charge=a.get("charge"),
            )
            new_atoms.append(atom)

        self.atoms.extend(new_atoms)

        # Rebound: push newest atoms away from tunnel (no back-folding)
        lip_dist = np.dot(self.tunnel_exit, self.tunnel_axis) + 5.0
        for a in new_atoms[-4:]:
            s = np.dot(a.pos - self.tunnel_exit, self.tunnel_axis)
            if s < 5.0:
                a.pos = self.tunnel_exit + self.tunnel_axis * 5.0 + np.random.uniform(-0.5, 0.5, 3)

        # Grow box if needed
        for a in new_atoms:
            self.field.expand(a.pos)

        # Rebuild field (full for prototype; later local-only)
        self._rebuild_field()

        # Relax: alternate field step with HKE step (both need field rebuild anyway)
        st = step_size if step_size is not None else self._step_size
        for step in range(relax_steps):
            max_d = self._relax_step(st, drag_factor=drag_factor)
            if alternate_hke:
                self._hke_ca_step(step_size=0.12)
            self._rebuild_field()
            if use_ruler and max_d > 1e-9:
                ratio = self._target_disp_per_step / max_d
                st = np.clip(st * ratio, 0.02, 0.2)
            if step > 0 and step % 4 == 3:
                self._rotate_group_around_anchor(angle_deg=1.5, nn_radius=10.0)
        self._step_size = st

    def compactness(self) -> Tuple[np.ndarray, float]:
        """Return (span per axis, end-to-end distance)."""
        if not self.atoms:
            return np.zeros(3), 0.0
        pos = np.array([a.pos for a in self.atoms])
        return np.ptp(pos, axis=0), float(np.linalg.norm(pos[0] - pos[-1]))

    def relax_to_convergence(
        self,
        max_disp_threshold: float = 0.02,
        max_steps: int = 2000,
        min_steps: int = 20,
        step_size: float = 0.1,
        high_potential_first: bool = True,
        drag_factor: float = 0.35,
        rotate_every: int = 4,
        alternate_hke: bool = True,
        field_res_schedule: Optional[Sequence[float]] = None,
    ) -> int:
        """
        Post-assembly relaxation: chain folds freely until atoms barely move.
        No tunnel rebounding. Prioritizes high-potential regions (scale by |F|).

        field_res_schedule: coarse-to-fine lattice resolutions in Å (e.g. [5, 3, 1]).
        Coarse phases are fast (few voxels); fine phase refines the fit.
        Returns total number of steps taken.
        """
        if not self.atoms:
            return 0
        schedule = field_res_schedule if field_res_schedule is not None else (self.field.res,)
        total_steps = 0
        steps_per_phase = max(1, max_steps // len(schedule))
        min_before_exit = min_steps

        for phase_res in schedule:
            self.field.res = float(phase_res)
            st = step_size
            for step in range(steps_per_phase):
                max_d = self._relax_step(
                    st,
                    drag_factor=drag_factor,
                    rebound_tail=0,
                    free_fold=True,
                    high_potential_first=high_potential_first,
                )
                max_hke = self._hke_ca_step(step_size=0.12) if alternate_hke else 0.0
                self._rebuild_field()
                max_d = max(max_d, max_hke)
                total_steps += 1
                if total_steps >= min_before_exit and max_d < max_disp_threshold:
                    return total_steps
                if max_d > 1e-9:
                    st = np.clip(st * 0.98, 0.02, 0.2)
                if rotate_every > 0 and step > 0 and step % rotate_every == rotate_every - 1:
                    self._rotate_group_around_anchor(angle_deg=1.0, nn_radius=12.0)

        return total_steps

    def low_energy_pockets(
        self, threshold_percentile: float = 10.0, min_separation: float = 8.0
    ) -> List[np.ndarray]:
        """
        Find low-energy pockets in the field (potential docking hot-spots).
        Returns list of (3,) positions where potential is below threshold.
        """
        p = self.field.potential
        threshold = np.percentile(p.ravel(), threshold_percentile)
        low_idx = np.where(p < threshold)
        positions = []
        for i, j, k in zip(low_idx[0], low_idx[1], low_idx[2]):
            pos = self.field.origin + np.array([i, j, k]) * self.field.res
            if all(np.linalg.norm(pos - x) >= min_separation for x in positions):
                positions.append(pos)
        return positions[:20]  # Top 20 pockets

    def run_pipeline(
        self,
        sequence: str,
        ss_string: Optional[str] = None,
        batch_size: Optional[int] = None,
        compact_until_converged: bool = True,
        max_disp_threshold: float = 0.02,
        alternate_hke: bool = True,
        field_res_schedule: Optional[Sequence[float]] = None,
        refine: bool = False,
    ) -> List[Atom]:
        """
        Full pipeline: build whole protein at once, let geometries grow naturally.

        Loads full chain from fast assembler (SS-aware: helices form before exit).
        Helical segments are kinetic groups (rigid bodies) during relaxation.
        Post-assembly: relax freely until atoms barely move.
        """
        seq = _parse_fasta(sequence) if ">" in sequence or "\n" in sequence else sequence
        seq = "".join(c for c in seq.upper() if c in AA_1to3)
        if not seq:
            return []
        n = len(seq)

        # Build whole chain at once (no 1-per-pass)
        self.load_from_fast_assembler(seq, ss_string)

        if compact_until_converged:
            if refine and n <= 60:
                min_s, max_s = 50, 800
                res_schedule = field_res_schedule or (5.0, 3.0, 1.0, 0.5)
                thresh = 0.01
            else:
                min_s = 20 if n <= 60 else 20
                max_s = 300 if n <= 60 else 2000
                res_schedule = field_res_schedule or ((5.0, 3.0, 1.0, 0.5) if n <= 60 else None)
                thresh = 0.01 if n <= 60 else max_disp_threshold
            self.relax_to_convergence(
                max_disp_threshold=thresh,
                alternate_hke=alternate_hke,
                min_steps=min_s,
                max_steps=max_s,
                field_res_schedule=res_schedule,
            )

        return self.atoms
