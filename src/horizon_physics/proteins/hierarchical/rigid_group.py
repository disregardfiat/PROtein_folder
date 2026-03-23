"""
RigidGroup: rigid body with 6DOF transform and children (Atoms or nested RigidGroups).

Represents a "multi-system area": 3x3 rotation R + translation t. Maintains
combined energy well (horizon, clash, custom) for the whole group. Groups can be nested.
JAX-friendly: positions collected into arrays for energy/grad.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union

from . import backend as _be
from .atom import Atom

xp = _be.np_or_jnp()

# Child: Atom | RigidGroup | (name: str, local_pos: (3,), z_shell: int) fixed point
ChildType = Union[Atom, "RigidGroup", Tuple[str, Any, int]]


def _euler_to_rotation(euler: Any) -> Any:
    """Euler angles (3,) in radians (ZYX or similar) -> 3x3 rotation matrix."""
    # Use Z-Y-X (yaw-pitch-roll) for uniqueness
    a, b, c = euler[0], euler[1], euler[2]
    ca, sa = xp.cos(a), xp.sin(a)
    cb, sb = xp.cos(b), xp.sin(b)
    cc, sc = xp.cos(c), xp.sin(c)
    return xp.array([
        [ca * cb, ca * sb * sc - sa * cc, ca * sb * cc + sa * sc],
        [sa * cb, sa * sb * sc + ca * cc, sa * sb * cc - ca * sc],
        [-sb, cb * sc, cb * cc],
    ], dtype=xp.float64)


def _rotation_to_euler(R: Any) -> Any:
    """3x3 rotation matrix -> euler angles (3,) ZYX."""
    R = xp.asarray(R)
    sy = xp.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    a = xp.where(singular, xp.zeros_like(sy), xp.arctan2(R[1, 0], R[0, 0]))
    b = xp.where(singular, xp.arctan2(-R[2, 0], sy), xp.arctan2(-R[2, 0], sy))
    c = xp.where(
        singular,
        xp.arctan2(-R[1, 2], R[1, 1]),
        xp.arctan2(R[0, 2], R[1, 2]),
    )
    return xp.array([a, b, c], dtype=xp.float64)


@dataclass
class RigidGroup:
    """
    Rigid body: 6DOF (R 3x3, t 3) and list of children (Atom or RigidGroup).

    Energy is evaluated on all descendant atom positions (horizon + clash + custom).
    Children are in local frame; world pose = (R, t) applied to local.
    last_pose_local: when set, (R, t) of last residue in this group's frame (for segment chains).
    """

    R: Any = field(default_factory=lambda: xp.eye(3, dtype=xp.float64))
    t: Any = field(default_factory=lambda: xp.zeros(3, dtype=xp.float64))
    children: List[ChildType] = field(default_factory=list)
    name: str = ""
    last_pose_local: Optional[Tuple[Any, Any]] = None

    def get_pose(self) -> Tuple[Any, Any]:
        """Return (R (3,3), t (3,))."""
        return (
            xp.asarray(self.R, dtype=xp.float64),
            xp.reshape(xp.asarray(self.t, dtype=xp.float64), (3,)),
        )

    def set_pose(self, R: Any, t: Any) -> None:
        """Set 6DOF from R (3,3) and t (3,)."""
        self.R = xp.asarray(R, dtype=xp.float64)
        self.t = xp.reshape(xp.asarray(t, dtype=xp.float64), (3,))

    def get_6dof(self) -> Any:
        """Return 6DOF as flat array: [t_x, t_y, t_z, euler_z, euler_y, euler_x]."""
        R, t = self.get_pose()
        euler = _rotation_to_euler(R)
        return xp.concatenate([xp.ravel(t), xp.ravel(euler)])

    def set_6dof(self, dof: Any) -> None:
        """Set 6DOF from flat array of length 6."""
        dof = xp.asarray(dof)
        self.t = dof[:3]
        self.R = _euler_to_rotation(dof[3:6])

    def get_last_residue_pose(self) -> Tuple[Any, Any]:
        """If last_pose_local is set, return (R, t) of last residue in world (this group's frame). Else return get_pose()."""
        if self.last_pose_local is None:
            return self.get_pose()
        R_l, t_l = self.last_pose_local
        R_l = xp.asarray(R_l, dtype=xp.float64)
        t_l = xp.reshape(xp.asarray(t_l, dtype=xp.float64), (3,))
        return (self.R @ R_l, self.t + self.R @ t_l)

    def _collect_positions_recursive(
        self,
        parent_R: Any,
        parent_t: Any,
        positions: List[Any],
        z_list: List[int],
    ) -> None:
        """Append world positions and z of all descendant atoms to lists."""
        R, t = self.get_pose()
        world_R = parent_R @ R
        world_t = parent_t + parent_R @ t
        for child in self.children:
            if isinstance(child, Atom):
                pos, _ = child.forward_kinematics(world_R, world_t)
                positions.append(pos)
                z_list.append(child.z_shell)
            elif isinstance(child, RigidGroup):
                child._collect_positions_recursive(world_R, world_t, positions, z_list)
            else:
                # (name, local_pos, z_shell) fixed point
                name, local_pos, z_shell = child[0], child[1], child[2]
                pos = world_t + world_R @ xp.reshape(xp.asarray(local_pos, dtype=xp.float64), (3,))
                positions.append(pos)
                z_list.append(z_shell)

    def collect_positions_and_z(
        self,
        parent_R: Optional[Any] = None,
        parent_t: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """
        Collect all descendant atom positions (N, 3) and z_list (N,) in world frame.

        If parent_R/parent_t are None, use identity/zero (root).
        """
        if parent_R is None:
            parent_R = xp.eye(3, dtype=xp.float64)
        if parent_t is None:
            parent_t = xp.zeros(3, dtype=xp.float64)
        positions: List[Any] = []
        z_list: List[int] = []
        self._collect_positions_recursive(parent_R, parent_t, positions, z_list)
        if not positions:
            return xp.zeros((0, 3), dtype=xp.float64), xp.array([], dtype=xp.int32)
        return xp.stack(positions, axis=0), xp.array(z_list, dtype=xp.int32)
