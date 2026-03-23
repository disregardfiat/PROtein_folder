"""
Energy evaluation for HKE: JAX-differentiable path and NumPy path reusing folding_energy.

Enables grad w.r.t. positions (and via chain rule w.r.t. DOFs). Group-level potentials
(horizon, clash, bonds) are combined; existing e_tot / grad_full are reused on NumPy path.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from . import backend as _be

xp = _be.np_or_jnp()
_jax_available = _be.is_jax_available()

# Reuse constants from folding_energy when on NumPy path
if not _jax_available:
    from horizon_physics.proteins.folding_energy import (
        e_tot_ca_with_bonds as _e_tot_ca_with_bonds_np,
        grad_full as _grad_full_np,
    )
else:
    from . import _energy_jax as _jax_energy


def energy_total(
    positions: Any,
    z_list: Any,
    include_bonds: bool = True,
    include_horizon: bool = True,
    include_clash: bool = True,
) -> float:
    """
    Total energy E_tot (informational + damping + bonds + clash).

    Uses existing folding_energy when backend is NumPy; uses JAX implementation when JAX.
    positions: (N, 3), z_list: (N,).
    """
    if _jax_available:
        return float(_jax_energy.energy_total_jax(positions, z_list, include_bonds, include_horizon, include_clash))
    pos_np = _be.to_numpy(positions)
    z_np = _be.to_numpy(z_list)
    return float(_e_tot_ca_with_bonds_np(pos_np, z_np))


def grad_positions(
    positions: Any,
    z_list: Any,
    include_bonds: bool = True,
    include_horizon: bool = True,
    include_clash: bool = True,
) -> Any:
    """Gradient of E_tot w.r.t. positions (N, 3)."""
    if _jax_available:
        return _jax_energy.grad_positions_jax(positions, z_list, include_bonds, include_horizon, include_clash)
    pos_np = _be.to_numpy(positions)
    z_np = _be.to_numpy(z_list)
    return _grad_full_np(
        pos_np,
        z_np,
        include_bonds=include_bonds,
        include_horizon=include_horizon,
        include_clash=include_clash,
    )


# Optional: expose for testing
__all__ = ["energy_total", "grad_positions"]
