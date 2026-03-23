"""
Discrete DOF representation for temperature-aware folding.

This wraps pyhqiv.molecular APIs to provide:
- Per-DOF discrete angle states.
- Energy for each state.
- Neighbor ΔE table for cheap move scoring and locking.

Higher-level folding code can use this to drive a small, temperature-aware path
search instead of continuous optimization over angles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    # pyhqiv is expected to be importable when using discrete DOFs
    from pyhqiv import molecular as _molecular
except ImportError:  # pragma: no cover
    _molecular = None  # type: ignore[assignment]

# Local HQIV helper for Θ per residue
try:
    from .secondary_structure_predictor import theta_eff_residue as _theta_eff_residue
except ImportError:  # pragma: no cover
    _theta_eff_residue = None  # type: ignore[assignment]


@dataclass
class DiscreteDof:
    """
    One torsion / coupling DOF with a discrete set of angle states.

    Attributes
    ----------
    dof_type : str
        "phi", "psi", "omega", "chi", or higher-level group DOF name.
    theta_local : float
        Local diamond size Θ in Å associated with this DOF.
    temperature : float
        Effective temperature in K.
    angles : np.ndarray
        Angle states in radians, shape (n_states,).
    energies : np.ndarray
        Energy in eV at each angle state, shape (n_states,).
    deltaE_neighbors : np.ndarray
        First difference E[i+1] - E[i]; last element is NaN, shape (n_states,).
    state_index : int
        Current state index into `angles` / `energies`.
    locked : bool
        If True, DOF is considered frozen (no moves proposed).
    """

    dof_type: str
    theta_local: float
    temperature: float
    angles: np.ndarray
    energies: np.ndarray
    deltaE_neighbors: np.ndarray
    state_index: int = 0
    locked: bool = False

    @classmethod
    def from_hqiv(
        cls,
        dof_type: str,
        theta_local: float,
        temperature: float,
        *,
        n_states: int = 32,
    ) -> "DiscreteDof":
        """
        Build a DiscreteDof from pyhqiv.molecular.coupling_angle_energy_profile.
        """
        if _molecular is None:
            raise ImportError("pyhqiv.molecular is required for DiscreteDof.from_hqiv()")
        angles, energies, dE = _molecular.coupling_angle_energy_profile(
            dof_type,
            theta_local_ang=float(theta_local),
            temperature=float(temperature),
            n_states=int(n_states),
        )
        return cls(
            dof_type=dof_type,
            theta_local=float(theta_local),
            temperature=float(temperature),
            angles=np.asarray(angles, dtype=float),
            energies=np.asarray(energies, dtype=float),
            deltaE_neighbors=np.asarray(dE, dtype=float),
            state_index=int(np.argmin(energies)),
        )

    @property
    def angle(self) -> float:
        """Current angle in radians."""
        return float(self.angles[self.state_index])

    @property
    def energy(self) -> float:
        """Current energy in eV."""
        return float(self.energies[self.state_index])

    def candidate_moves(self) -> List[Tuple[int, float]]:
        """
        Return a list of (new_state_index, ΔE) moves from current state.

        Only immediate neighbors (state_index ± 1) are considered for now.
        """
        if self.locked:
            return []
        moves: List[Tuple[int, float]] = []
        i = self.state_index
        # left neighbor
        if i - 1 >= 0:
            dE = float(self.energies[i - 1] - self.energies[i])
            moves.append((i - 1, dE))
        # right neighbor
        if i + 1 < len(self.energies):
            dE = float(self.energies[i + 1] - self.energies[i])
            moves.append((i + 1, dE))
        return moves

    def maybe_lock(self, kT_eff: float, barrier_factor: float = 5.0) -> None:
        """
        Lock DOF if all neighbor moves are too expensive relative to kT.

        Parameters
        ----------
        kT_eff : float
            Effective kT in eV.
        barrier_factor : float
            Lock when |ΔE| > barrier_factor * kT for all neighbors.
        """
        if self.locked or kT_eff <= 0.0:
            return
        moves = self.candidate_moves()
        if not moves:
            self.locked = True
            return
        threshold = barrier_factor * kT_eff
        if all(abs(dE) > threshold for _, dE in moves):
            self.locked = True


@dataclass
class DofGroup:
    """
    A kinetic group composed of multiple discrete DOFs (e.g. residue, helix, loop).

    This is a light container: higher-level code decides how group moves translate
    into coordinate updates. Here we just:
    - track DOFs,
    - expose candidate moves with ΔE,
    - provide a simple "all locked" check.
    """

    dofs: List[DiscreteDof] = field(default_factory=list)

    def all_locked(self) -> bool:
        return all(d.locked for d in self.dofs)

    def candidate_moves(self) -> List[Tuple[int, int, float]]:
        """
        Return candidate moves as (dof_index, new_state_index, ΔE).
        """
        moves: List[Tuple[int, int, float]] = []
        for idx, d in enumerate(self.dofs):
            for new_idx, dE in d.candidate_moves():
                moves.append((idx, new_idx, dE))
        return moves


def build_backbone_dofs_for_sequence(
    sequence: str,
    temperature: float,
    *,
    n_states: int = 32,
) -> Tuple[List[DiscreteDof], List[DiscreteDof]]:
    """
    Build discrete φ/ψ DOFs for a backbone sequence at a given temperature.

    Parameters
    ----------
    sequence : str
        One-letter amino-acid sequence.
    temperature : float
        Effective temperature in K.
    n_states : int
        Number of discrete torsion states per DOF (per residue).

    Returns
    -------
    phi_dofs, psi_dofs : list[DiscreteDof], list[DiscreteDof]
        Per-residue φ and ψ DOFs. For terminal residues, callers may choose to
        ignore one of the torsions if desired.
    """
    if _molecular is None:
        raise ImportError("pyhqiv.molecular is required for build_backbone_dofs_for_sequence().")
    if _theta_eff_residue is None:
        raise ImportError("theta_eff_residue is required for build_backbone_dofs_for_sequence().")

    seq = "".join(c for c in sequence.strip().upper() if c.isalpha())
    phi_dofs: List[DiscreteDof] = []
    psi_dofs: List[DiscreteDof] = []
    for aa in seq:
        theta_eff = float(_theta_eff_residue(aa))
        phi_dofs.append(
            DiscreteDof.from_hqiv(
                "phi",
                theta_local=theta_eff,
                temperature=temperature,
                n_states=n_states,
            )
        )
        psi_dofs.append(
            DiscreteDof.from_hqiv(
                "psi",
                theta_local=theta_eff,
                temperature=temperature,
                n_states=n_states,
            )
        )
    return phi_dofs, psi_dofs


