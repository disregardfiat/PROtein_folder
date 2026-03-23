"""
Protein: root of the kinematic tree with grouping strategies and staged minimization.

Supports flexible grouping (per-residue, secondary-structure aware, user-defined domains).
Methods: forward_kinematics() -> (N,3), get_dofs()/set_dofs(), compute_total_energy(),
minimize_hierarchical() (coarse rigid-body -> internal torsions -> Cartesian refinement).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

from . import backend as _be
from .rigid_group import RigidGroup
from .atom import Atom
from . import energy as _energy

xp = _be.np_or_jnp()

# Bond lengths (Å) from peptide_backbone
def _backbone_bonds() -> dict:
    from horizon_physics.proteins.peptide_backbone import backbone_bond_lengths
    return backbone_bond_lengths()


def _deg2rad(x: float) -> float:
    return float(x) * 3.141592653589793 / 180.0


def _rotation_z(angle: float) -> Any:
    c, s = float(__import__("math").cos(angle)), float(__import__("math").sin(angle))
    return xp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=xp.float64)


def _rotation_y(angle: float) -> Any:
    c, s = float(__import__("math").cos(angle)), float(__import__("math").sin(angle))
    return xp.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=xp.float64)


def _local_residue_atoms() -> List[Tuple[str, Any, int]]:
    """Fixed (name, local_xyz, z_shell) for one residue with CA at origin, forward = +x."""
    bonds = _backbone_bonds()
    r_n_ca = bonds["N_Calpha"]
    r_ca_c = bonds["Calpha_C"]
    r_c_o = bonds["C_O"]
    # N behind CA, C ahead, O perpendicular (y)
    n_local = xp.array([-r_n_ca, 0.0, 0.0], dtype=xp.float64)
    ca_local = xp.array([0.0, 0.0, 0.0], dtype=xp.float64)
    c_local = xp.array([r_ca_c, 0.0, 0.0], dtype=xp.float64)
    o_local = xp.array([r_ca_c, r_c_o, 0.0], dtype=xp.float64)
    return [
        ("N", n_local, 7),
        ("CA", ca_local, 6),
        ("C", c_local, 6),
        ("O", o_local, 8),
    ]


def _local_residue_com() -> Any:
    """COM of one residue in local frame (CA at origin)."""
    atoms = _local_residue_atoms()
    local_com = xp.zeros(3, dtype=xp.float64)
    for _, pos, _ in atoms:
        local_com += xp.reshape(xp.asarray(pos, dtype=xp.float64), (3,))
    return local_com / 4.0


def generate_compact_start(
    n_groups: int,
    radius: float = 5.0,
    seed: Optional[int] = None,
) -> Any:
    """
    Generate DOFs so group COMs lie in a compact cloud (ball).
    n_groups: number of kinematic groups (residues or segments).
    Returns dofs array: [t1(3), euler1(3), phi2, psi2, ..., phi_n, psi_n].
    """
    import numpy as np
    if n_groups == 0:
        return xp.array([], dtype=xp.float64)
    rng = np.random.default_rng(seed)
    # Points in ball
    points = rng.standard_normal((n_groups, 3))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    points = points / norms * radius * (rng.uniform(0.3, 1.0, (n_groups, 1)) ** (1.0 / 3.0))
    bonds = _backbone_bonds()
    r_ca_c = bonds["Calpha_C"]
    r_c_n = bonds["C_N"]
    r_n_ca = bonds["N_Calpha"]
    step = r_c_n + r_n_ca
    local_com = _be.to_numpy(_local_residue_com())
    # First group: t1 so COM is at points[0]
    t1 = points[0] - local_com
    euler1 = np.array([0.0, 0.0, 0.0])
    dofs = [float(t1[0]), float(t1[1]), float(t1[2]), float(euler1[0]), float(euler1[1]), float(euler1[2])]
    R_prev = np.eye(3)
    t_prev = t1
    com_prev = t_prev + R_prev @ local_com
    for i in range(1, n_groups):
        direction = points[i] - com_prev
        d_norm = np.linalg.norm(direction)
        if d_norm < 1e-9:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = direction / d_norm
        forward_local = R_prev.T @ direction
        forward_local = forward_local / (np.linalg.norm(forward_local) + 1e-12)
        phi_rad = np.arctan2(forward_local[1], forward_local[0])
        psi_rad = -np.arcsin(np.clip(forward_local[2], -1.0, 1.0))
        phi_deg = np.degrees(phi_rad)
        psi_deg = np.degrees(psi_rad)
        dofs.append(float(phi_deg))
        dofs.append(float(psi_deg))
        R_next, t_next = _place_next_group_from_phi_psi(
            _be.np_or_jnp().array(R_prev), _be.np_or_jnp().array(t_prev), phi_deg, psi_deg
        )
        R_prev = _be.to_numpy(R_next)
        t_prev = _be.to_numpy(t_next)
        com_prev = t_prev + R_prev @ local_com
    return xp.array(dofs, dtype=xp.float64)


def _place_next_group_from_phi_psi(
    prev_R: Any,
    prev_t: Any,
    phi_deg: float,
    psi_deg: float,
) -> Tuple[Any, Any]:
    """Given previous group pose (R, t) and phi, psi in degrees, return (R_next, t_next) for next residue (CA at origin)."""
    bonds = _backbone_bonds()
    r_c_n = bonds["C_N"]
    r_n_ca = bonds["N_Calpha"]
    phi = _deg2rad(phi_deg)
    psi = _deg2rad(psi_deg)
    # Forward direction in prev frame: R_y(psi) @ R_z(phi) @ [1,0,0]
    e = xp.array([1.0, 0.0, 0.0], dtype=xp.float64)
    Ry = _rotation_y(psi)
    Rz = _rotation_z(phi)
    forward_local = Ry @ Rz @ e
    forward_local = forward_local / (xp.linalg.norm(forward_local) + 1e-12)
    # Prev C in prev frame is at (r_ca_c, 0, 0)
    r_ca_c = bonds["Calpha_C"]
    prev_c_local = xp.array([r_ca_c, 0.0, 0.0], dtype=xp.float64)
    prev_c_world = prev_t + prev_R @ prev_c_local
    # Next N = prev_C + r_c_n * forward_world, next CA = next_N + r_n_ca * forward_world
    forward_world = prev_R @ forward_local
    next_ca_world = prev_c_world + (r_c_n + r_n_ca) * forward_world
    R_next_local = Ry @ Rz
    R_next = prev_R @ R_next_local
    return R_next, next_ca_world


class Protein:
    """
    Root of the kinematic tree. Builds residue (or SS/domain) groups and exposes
    forward_kinematics(), get_dofs(), set_dofs(), compute_total_energy(), minimize_hierarchical().
    """

    def __init__(
        self,
        sequence: str,
        ss_string: Optional[str] = None,
        grouping_strategy: str = "residue",
        domain_ranges: Optional[List[Tuple[int, int]]] = None,
    ):
        """
        sequence: one-letter amino acid sequence.
        ss_string: optional secondary structure (H/E/C). If None, predicted.
        grouping_strategy: 'residue' | 'ss' | 'domain' | 'helix_unit'.
        domain_ranges: for strategy 'domain', list of (start, end) residue indices.
        """
        self.sequence = sequence.strip().upper()
        self.n_res = len(self.sequence)
        if ss_string is None or len(ss_string) != self.n_res:
            from horizon_physics.proteins.secondary_structure_predictor import predict_ss
            ss_string, _ = predict_ss(self.sequence, window=5)
        self.ss_string = ss_string
        self.grouping_strategy = grouping_strategy
        self.domain_ranges = domain_ranges or []
        self._root = RigidGroup(name="root")
        self._groups: List[RigidGroup] = []
        self._phi_psi: List[Tuple[float, float]] = []  # per residue (phi, psi) or junction only for helix_unit
        self._segment_phi_psi: List[Tuple[float, float]] = []  # junction (phi, psi) for helix_unit
        self._build_tree()

    def _build_tree(self) -> None:
        """Build kinematic tree from sequence and grouping strategy."""
        if self.n_res == 0:
            return
        local_atoms = _local_residue_atoms()
        if self.grouping_strategy == "residue":
            self._build_per_residue(local_atoms)
        elif self.grouping_strategy == "ss":
            self._build_ss_aware(local_atoms)
        elif self.grouping_strategy == "domain":
            self._build_domain(local_atoms)
        elif self.grouping_strategy == "helix_unit":
            self._build_helix_units(local_atoms)
        else:
            self._build_per_residue(local_atoms)

    def _build_per_residue(self, local_atoms: List[Tuple[str, Any, int]]) -> None:
        """One RigidGroup per residue; DOF: 6 for first, 2 (phi, psi) for rest."""
        from horizon_physics.proteins.peptide_backbone import ramachandran_alpha
        phi_a, psi_a = ramachandran_alpha()
        self._groups = []
        self._phi_psi = []
        for i in range(self.n_res):
            g = RigidGroup(name=f"res{i}")
            for name, pos, z in local_atoms:
                g.children.append((name, pos.copy(), z))
            self._groups.append(g)
            self._phi_psi.append((float(phi_a), float(psi_a)))
        # First group: 6 DOF (identity + place at origin for now; minimizer will set)
        self._root.children = list(self._groups)
        # Chain connectivity: set (R, t) of groups 1..n-1 from phi, psi
        self._apply_chain_poses()

    def _apply_chain_poses(self) -> None:
        """Set (R, t) of each group from first group 6DOF and phi, psi chain."""
        if not self._groups:
            return
        # Group 0 already has R, t from set_dofs or initial
        for i in range(1, self.n_res):
            R_prev, t_prev = self._groups[i - 1].get_pose()
            phi, psi = self._phi_psi[i]
            R_next, t_next = _place_next_group_from_phi_psi(R_prev, t_prev, phi, psi)
            self._groups[i].set_pose(R_next, t_next)

    def _build_ss_aware(self, local_atoms: List[Tuple[str, Any, int]]) -> None:
        """Per-residue groups with initial phi/psi set from SS (H=alpha, E=beta, C=alpha)."""
        from horizon_physics.proteins.peptide_backbone import ramachandran_alpha, ramachandran_beta
        phi_a, psi_a = ramachandran_alpha()
        phi_b, psi_b = ramachandran_beta()
        self._groups = []
        self._phi_psi = []
        for i in range(self.n_res):
            g = RigidGroup(name=f"res{i}")
            for name, pos, z in local_atoms:
                g.children.append((name, pos.copy(), z))
            self._groups.append(g)
            if self.ss_string[i] == "E":
                self._phi_psi.append((float(phi_b), float(psi_b)))
            else:
                self._phi_psi.append((float(phi_a), float(psi_a)))
        self._root.children = list(self._groups)
        self._apply_chain_poses()

    def _build_domain(self, local_atoms: List[Tuple[str, Any, int]]) -> None:
        """Per-residue groups; domain_ranges can be used later for coarse 6DOF per domain."""
        self._build_per_residue(local_atoms)

    def _segment_ranges_from_ss(self) -> List[Tuple[int, int, str]]:
        """Contiguous SS segments: (start, end, 'H'|'E'|'C'). Merge H, merge E; C one-residue segments."""
        out: List[Tuple[int, int, str]] = []
        i = 0
        while i < self.n_res:
            ss = self.ss_string[i]
            j = i
            while j < self.n_res and self.ss_string[j] == ss:
                j += 1
            # Merge only H and E into multi-residue segments; C stays one per residue for flexibility
            if ss == "C":
                for k in range(i, j):
                    out.append((k, k + 1, "C"))
            else:
                out.append((i, j, ss))
            i = j
        return out

    def _build_helix_units(self, local_atoms: List[Tuple[str, Any, int]]) -> None:
        """Tight helices (and strands) as single kinetic units: one RigidGroup per contiguous H or E segment, fixed internal geometry."""
        from horizon_physics.proteins.peptide_backbone import ramachandran_alpha, ramachandran_beta
        phi_a, psi_a = ramachandran_alpha()
        phi_b, psi_b = ramachandran_beta()
        segments = self._segment_ranges_from_ss()
        self._groups = []
        self._segment_phi_psi = []
        for seg_idx, (start, end, ss) in enumerate(segments):
            seg = RigidGroup(name=f"seg{seg_idx}")
            # Ideal phi, psi for this SS
            phi_deg, psi_deg = (float(phi_b), float(psi_b)) if ss == "E" else (float(phi_a), float(psi_a))
            R_prev = xp.eye(3, dtype=xp.float64)
            t_prev = xp.zeros(3, dtype=xp.float64)
            for r in range(start, end):
                res_group = RigidGroup(name=f"res{r}")
                for name, pos, z in local_atoms:
                    res_group.children.append((name, pos.copy(), z))
                res_group.set_pose(R_prev, t_prev)
                seg.children.append(res_group)
                if r < end - 1:
                    R_prev, t_prev = _place_next_group_from_phi_psi(R_prev, t_prev, phi_deg, psi_deg)
            seg.last_pose_local = (R_prev, t_prev)
            self._groups.append(seg)
            if seg_idx > 0:
                self._segment_phi_psi.append((float(phi_a), float(psi_a)))
        self._root.children = list(self._groups)

    def n_groups(self) -> int:
        """Number of kinematic groups (residues or segments). For compact init and DOF size."""
        return len(self._groups)

    def _initial_dofs(self) -> Any:
        """Initial DOF vector: 6 for first group (t=0, euler=0), then phi, psi per remaining residue."""
        if self.n_res == 0:
            return xp.array([], dtype=xp.float64)
        dofs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i in range(1, self.n_res):
            phi, psi = self._phi_psi[i]
            dofs.append(phi)
            dofs.append(psi)
        return xp.array(dofs, dtype=xp.float64)

    def get_dofs(self) -> Any:
        """Vector of all free DOFs: [t_x, t_y, t_z, euler_z, euler_y, euler_x] for first group, then [phi_i, psi_i] for i=1..n-1 (or junctions for helix_unit)."""
        if self.n_res == 0:
            return xp.array([], dtype=xp.float64)
        R0, t0 = self._groups[0].get_pose()
        from .rigid_group import _rotation_to_euler
        euler = _rotation_to_euler(R0)
        dofs = [float(t0[0]), float(t0[1]), float(t0[2]), float(euler[0]), float(euler[1]), float(euler[2])]
        if self.grouping_strategy == "helix_unit":
            for phi, psi in self._segment_phi_psi:
                dofs.append(float(phi))
                dofs.append(float(psi))
        else:
            for i in range(1, self.n_res):
                phi, psi = self._phi_psi[i]
                dofs.append(float(phi))
                dofs.append(float(psi))
        return xp.array(dofs, dtype=xp.float64)

    def get_6dof_per_residue(self) -> Any:
        """Return (n_res, 6) world 6-DOF per residue: [t_x, t_y, t_z, euler_z, euler_y, euler_x] in Å and radians (ZYX)."""
        if self.n_res == 0:
            return xp.zeros((0, 6), dtype=xp.float64)
        if self.grouping_strategy == "helix_unit":
            from .rigid_group import _rotation_to_euler
            rows = []
            for seg in self._groups:
                R_seg, t_seg = seg.get_pose()
                for child in seg.children:
                    if isinstance(child, RigidGroup):
                        R_c, t_c = child.get_pose()
                        R_w = R_seg @ R_c
                        t_w = t_seg + R_seg @ xp.reshape(t_c, (3,))
                        euler = _rotation_to_euler(R_w)
                        rows.append(xp.concatenate([xp.ravel(t_w), xp.ravel(euler)]))
            return xp.stack(rows, axis=0) if rows else xp.zeros((0, 6), dtype=xp.float64)
        rows = [self._groups[i].get_6dof() for i in range(self.n_res)]
        return xp.stack(rows, axis=0)

    def set_dofs(self, dofs: Any) -> None:
        """Update all group poses and phi/psi from DOF vector."""
        dofs = xp.asarray(dofs)
        if self.n_res == 0:
            return
        self._groups[0].set_6dof(dofs[:6])
        if self.grouping_strategy == "helix_unit":
            n_seg = len(self._groups)
            for i in range(1, n_seg):
                last_R, last_t = self._groups[i - 1].get_last_residue_pose()
                phi = float(dofs[6 + 2 * (i - 1)])
                psi = float(dofs[6 + 2 * (i - 1) + 1])
                next_R, next_t = _place_next_group_from_phi_psi(last_R, last_t, phi, psi)
                self._groups[i].set_pose(next_R, next_t)
            for idx in range(len(self._segment_phi_psi)):
                self._segment_phi_psi[idx] = (float(dofs[6 + 2 * idx]), float(dofs[6 + 2 * idx + 1]))
        else:
            for i in range(1, self.n_res):
                self._phi_psi[i] = (float(dofs[6 + 2 * (i - 1)]), float(dofs[6 + 2 * (i - 1) + 1]))
            self._apply_chain_poses()

    def forward_kinematics(self) -> Tuple[Any, Any]:
        """
        Return flat (N, 3) positions and (N,) atom types (z_shell) for full interop with e_tot / grad_full.
        """
        pos, z_list = self._root.collect_positions_and_z()
        return pos, z_list

    def _group_coms_from_positions(self, pos: Any) -> Any:
        """Group COMs from flat positions (n_res*4, 3). Returns (n_res, 3)."""
        import numpy as np
        pos = np.asarray(pos).reshape(-1, 3)
        n_res = self.n_res
        if n_res == 0:
            return np.zeros((0, 3))
        # 4 atoms per residue (N, CA, C, O)
        pos_g = pos.reshape(n_res, 4, 3)
        return np.mean(pos_g, axis=1)

    def _inter_group_attraction(self, pos: Any, scale: float = 1.0, length_scale: float = 8.0) -> float:
        """COM–COM attraction between non-adjacent groups: scale * sum(exp(-dist_com / length_scale))."""
        import numpy as np
        coms = self._group_coms_from_positions(pos)
        n = len(coms)
        if n < 3:
            return 0.0
        e = 0.0
        for i in range(n):
            for j in range(i + 2, n):  # non-adjacent only (j - i > 1)
                d = np.linalg.norm(coms[j] - coms[i])
                e += np.exp(-d / length_scale)
        return float(scale * e)

    def _funnel_penalty(
        self,
        pos: Any,
        funnel_radius: float,
        stiffness: float = 1.0,
        funnel_radius_exit: Optional[float] = None,
    ) -> float:
        """
        Soft cone funnel: axis from first to last group COM; penalize group COMs outside local radius.
        Radius grows linearly along the axis: R(t) = funnel_radius + t * (funnel_radius_exit - funnel_radius),
        with t in [0, 1] from first to last COM. If funnel_radius_exit is None, R(t) = funnel_radius (cylinder).
        E_funnel = stiffness * sum over i of max(0, d_i - R(t_i))^2.
        """
        import numpy as np
        coms = self._group_coms_from_positions(pos)
        n = len(coms)
        if n < 2 or funnel_radius <= 0:
            return 0.0
        # Default cone: exit radius twice narrow end; set funnel_radius_exit=funnel_radius for cylinder
        r_exit = (2.0 * funnel_radius) if funnel_radius_exit is None else funnel_radius_exit
        a = np.asarray(coms[0], dtype=np.float64)
        b = np.asarray(coms[-1], dtype=np.float64)
        v = b - a
        v_sq = float(np.dot(v, v))
        if v_sq < 1e-20:
            # Degenerate axis: treat as point; use max radius for cylinder-like penalty
            r_local = max(funnel_radius, r_exit)
            e = 0.0
            for i in range(n):
                d = np.linalg.norm(coms[i] - a)
                if d > r_local:
                    e += (d - r_local) ** 2
            return float(stiffness * e)
        e = 0.0
        for i in range(n):
            w = np.asarray(coms[i], dtype=np.float64) - a
            t = np.dot(w, v) / v_sq
            t = max(0.0, min(1.0, float(t)))
            r_allowed = funnel_radius + t * (r_exit - funnel_radius)
            q = a + t * v
            d = float(np.linalg.norm(coms[i] - q))
            if d > r_allowed:
                e += (d - r_allowed) ** 2
        return float(stiffness * e)

    def compute_total_energy(
        self,
        include_bonds: bool = True,
        include_horizon: bool = True,
        include_clash: bool = True,
        inter_group_weight: float = 0.0,
        inter_group_length_scale: float = 8.0,
        funnel_radius: Optional[float] = None,
        funnel_stiffness: float = 1.0,
        funnel_radius_exit: Optional[float] = None,
    ) -> float:
        """Reuses existing energy + optional group-level COM–COM attraction + optional funnel (cone) soft bound."""
        pos, z_list = self.forward_kinematics()
        e = float(_energy.energy_total(pos, z_list, include_bonds, include_horizon, include_clash))
        if inter_group_weight > 0:
            e += self._inter_group_attraction(pos, scale=inter_group_weight, length_scale=inter_group_length_scale)
        if funnel_radius is not None and funnel_radius > 0:
            e += self._funnel_penalty(
                pos,
                funnel_radius=funnel_radius,
                stiffness=funnel_stiffness,
                funnel_radius_exit=funnel_radius_exit,
            )
        return e

    def minimize_hierarchical(
        self,
        max_iter_stage1: int = 80,
        max_iter_stage2: int = 120,
        max_iter_stage3: int = 100,
        gtol: float = 1e-5,
        device: Optional[str] = None,
        trajectory_log_path: Optional[str] = None,
        funnel_radius: Optional[float] = None,
        funnel_stiffness: float = 1.0,
        funnel_radius_exit: Optional[float] = None,
        converge_max_disp_ang: Optional[float] = None,
    ) -> Tuple[Any, dict]:
        """
        Staged optimization: (1) coarse rigid-body 6DOF + torsions, (2) internal torsions refinement,
        (3) optional final flat Cartesian refinement. Returns (positions (N,3), info dict).
        When funnel_radius is set, a soft cone confines group COMs in stages 1-2 (off in stage 3).
        When converge_max_disp_ang is set, stage 3 stops when max Cα displacement per step < threshold.
        """
        from .minimize_hierarchical import run_staged_minimization
        return run_staged_minimization(
            self,
            max_iter_stage1=max_iter_stage1,
            max_iter_stage2=max_iter_stage2,
            max_iter_stage3=max_iter_stage3,
            gtol=gtol,
            device=device,
            trajectory_log_path=trajectory_log_path,
            funnel_radius=funnel_radius,
            funnel_stiffness=funnel_stiffness,
            funnel_radius_exit=funnel_radius_exit,
            converge_max_disp_ang=converge_max_disp_ang,
        )


__all__ = [
    "Protein",
    "forward_kinematics",
    "get_dofs",
    "set_dofs",
    "compute_total_energy",
    "minimize_hierarchical",
    "generate_compact_start",
]
