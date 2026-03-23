"""
Atom: relative 6DOF internal coordinates and forward kinematics.

Each atom stores relative internal coordinates to its parent: bond length,
two bond angles, torsion, and optional rigid offset. Lightweight, JAX-friendly
where possible. Forward kinematics: parent (R, t) -> world position (and optional frame).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from . import backend as _be

xp = _be.np_or_jnp()


def _rotation_x(angle: Any) -> Any:  # noqa: A001
    """Rotation matrix about x-axis (radians)."""
    c, s = xp.cos(angle), xp.sin(angle)
    return xp.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=xp.float64)


def _rotation_y(angle: Any) -> Any:
    """Rotation matrix about y-axis (radians)."""
    c, s = xp.cos(angle), xp.sin(angle)
    return xp.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=xp.float64)


def _rotation_z(angle: Any) -> Any:
    """Rotation matrix about z-axis (radians)."""
    c, s = xp.cos(angle), xp.sin(angle)
    return xp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=xp.float64)


def _local_position(
    bond_length: Any,
    angle1: Any,
    angle2: Any,
    torsion: Any,
    rigid_offset: Optional[Any] = None,
) -> Any:
    """
    Position in parent frame from internal coords.
    Bond along local z; angle1 (x), angle2 (y), torsion (z).
    local = Rz(torsion) @ Ry(angle2) @ Rx(angle1) @ [0, 0, bond_length].
    """
    # Direction from bond length along z after rotations
    ez = xp.array([0.0, 0.0, 1.0], dtype=xp.float64)
    r1 = _rotation_x(angle1)
    r2 = _rotation_y(angle2)
    r3 = _rotation_z(torsion)
    local_vec = r3 @ r2 @ r1 @ (bond_length * ez)
    if rigid_offset is not None:
        local_vec = local_vec + xp.reshape(xp.asarray(rigid_offset, dtype=xp.float64), (3,))
    return local_vec


def _local_frame(
    bond_length: Any,
    angle1: Any,
    angle2: Any,
    torsion: Any,
) -> Any:
    """3x3 rotation matrix of atom frame in parent frame (z along bond)."""
    r1 = _rotation_x(angle1)
    r2 = _rotation_y(angle2)
    r3 = _rotation_z(torsion)
    return r3 @ r2 @ r1


@dataclass
class Atom:
    """
    Atom with relative internal coordinates to parent.

    Internal DOF: bond_length, angle1, angle2, torsion (radians).
    Optional rigid_offset: (3,) fixed offset in parent frame (Å).
    z_shell: Z for HQIV energy (default 6).
    """

    bond_length: float
    angle1: float  # radians
    angle2: float  # radians
    torsion: float  # radians
    rigid_offset: Optional[Tuple[float, float, float]] = None
    z_shell: int = 6
    name: str = ""

    def get_dofs(self) -> Any:
        """Return internal DOF as array [bond_length, angle1, angle2, torsion]."""
        arr = xp.array(
            [self.bond_length, self.angle1, self.angle2, self.torsion],
            dtype=xp.float64,
        )
        return arr

    def set_dofs(self, dofs: Any) -> None:
        """Update internal DOF from array of length 4."""
        dofs = xp.asarray(dofs)
        self.bond_length = float(dofs[0])
        self.angle1 = float(dofs[1])
        self.angle2 = float(dofs[2])
        self.torsion = float(dofs[3])

    def local_position(self) -> Any:
        """Position in parent frame (3,) from current internal coords."""
        off = None
        if self.rigid_offset is not None:
            off = xp.array(self.rigid_offset, dtype=xp.float64)
        return _local_position(
            self.bond_length,
            self.angle1,
            self.angle2,
            self.torsion,
            off,
        )

    def forward_kinematics(self, parent_R: Any, parent_t: Any) -> Tuple[Any, Any]:
        """
        World position and world frame (3x3) given parent pose.

        parent_R: (3,3), parent_t: (3,). Returns (world_pos (3,), world_R (3,3)).
        """
        parent_R = xp.asarray(parent_R, dtype=xp.float64)
        parent_t = xp.reshape(xp.asarray(parent_t, dtype=xp.float64), (3,))
        local_pos = self.local_position()
        world_pos = parent_t + parent_R @ local_pos
        local_R = _local_frame(
            self.bond_length,
            self.angle1,
            self.angle2,
            self.torsion,
        )
        world_R = parent_R @ local_R
        return world_pos, world_R
