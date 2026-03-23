"""
JAX-differentiable energy: e_tot (informational + damping) + bonds + clash + horizon.

No scipy; all ops JIT/grad compatible. Used when JAX backend is active.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import grad, jit

# Match folding_energy / _hqiv_base constants
HBAR_C_EV_ANG = 1973.269804
A_LOC_ANG = 1.0
R_CA_CA_EQ = 3.8
R_BOND_MIN = 2.5
R_BOND_MAX = 6.0
R_CLASH = 2.0
R_HORIZON = 15.0
R_REF = 2.0
K_BOND = 200.0 * HBAR_C_EV_ANG
K_CLASH = 500.0 * HBAR_C_EV_ANG
K_HORIZON = 0.5 * HBAR_C_EV_ANG
LAMBDA_DAMP = 0.1 * HBAR_C_EV_ANG


def _theta_local(z_shell: int, coordination: int = 1) -> float:
    """Diamond size from _hqiv_base."""
    alpha = 0.91
    theta0 = 1.53 * (6 ** alpha) * (2 ** (1 / 3))
    return theta0 * (z_shell ** (-alpha)) / (coordination ** (1 / 3))


def _horizon_scalar(theta_ang: float) -> float:
    if theta_ang <= 0:
        return 0.0
    return 2.0 / theta_ang


def _theta_at_position(positions: jnp.ndarray, i: int, z_shell: int) -> jnp.ndarray:
    """Theta at node i; JAX-friendly (no Python loop over i)."""
    n = positions.shape[0]
    base = _theta_local(z_shell, 2)
    if n < 2:
        return jnp.array(base)
    d = jnp.linalg.norm(positions - positions[i], axis=1)
    d = jnp.where(jnp.arange(n) == i, jnp.inf, d)
    r_min = jnp.min(d)
    return base * jnp.minimum(1.0, r_min / R_REF)


def e_info_sum(positions: jnp.ndarray, z_list: jnp.ndarray) -> jnp.ndarray:
    """Sum_i hbar_c/theta_i using per-atom theta from min distance to neighbor."""
    n = positions.shape[0]
    def theta_i(i):
        d = jnp.linalg.norm(positions - positions[i], axis=1)
        d = jnp.where(jnp.arange(n) == i, jnp.inf, d)
        r_min = jnp.min(d)
        base = _theta_local(int(z_list[i]), 2)
        th = base * jnp.minimum(1.0, r_min / R_REF)
        return jnp.where(th > 0, HBAR_C_EV_ANG / th, 0.0)
    return jnp.sum(jnp.array([theta_i(i) for i in range(n)]))


def e_damp_sum(positions: jnp.ndarray, z_list: jnp.ndarray) -> jnp.ndarray:
    n = positions.shape[0]
    def u_i(i):
        d = jnp.linalg.norm(positions - positions[i], axis=1)
        d = jnp.where(jnp.arange(n) == i, jnp.inf, d)
        r_min = jnp.min(d)
        base = _theta_local(int(z_list[i]), 2)
        th = base * jnp.minimum(1.0, r_min / R_REF)
        phi = jnp.where(th > 0, 2.0 / th, 0.0)
        denom = A_LOC_ANG + phi / 6.0
        return jnp.where(denom > 0, phi / denom, 0.0)
    return jnp.sum(jnp.array([u_i(i) for i in range(n)]))


def e_bonds(positions: jnp.ndarray) -> jnp.ndarray:
    """Bond penalty for consecutive pairs."""
    r = jnp.linalg.norm(positions[1:] - positions[:-1], axis=1)
    r = jnp.where(r < 1e-12, 1.0, r)
    term = jnp.where(
        r < R_BOND_MIN,
        (R_BOND_MIN - r) ** 2,
        jnp.where(r > R_BOND_MAX, (r - R_BOND_MAX) ** 2, 0.1 * (r - R_CA_CA_EQ) ** 2),
    )
    return K_BOND * jnp.sum(term)


def e_clash(positions: jnp.ndarray, window: int = 20) -> jnp.ndarray:
    """Clash penalty for non-bonded pairs within window."""
    n = positions.shape[0]
    window = min(window, n)
    total = 0.0
    for j in range(2, window + 1):
        for i in range(n - j):
            r = jnp.linalg.norm(positions[i + j] - positions[i])
            total = total + jnp.where(r < R_CLASH, K_CLASH * (R_CLASH - r) ** 2, 0.0)
    return total


def e_horizon(positions: jnp.ndarray) -> jnp.ndarray:
    """Horizon repulsion potential: sum over pairs within R_HORIZON; gradient gives horizon force."""
    n = positions.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = positions[j] - positions[i]
            r = jnp.linalg.norm(d)
            theta_ij = _theta_local(6, 2) * jnp.minimum(1.0, r / R_REF)
            phi = jnp.where(theta_ij > 0, 2.0 / theta_ij, 0.0)
            # U_ij = -pot*r so gradient gives force on i = -pot*unit_ij (horizon repulsion)
            pot = jnp.where((r > 1e-9) & (r <= R_HORIZON), K_HORIZON * phi / (theta_ij + 1e-9) * r, 0.0)
            total = total - pot
    return total


def energy_total_jax(
    positions: jnp.ndarray,
    z_list: jnp.ndarray,
    include_bonds: bool = True,
    include_horizon: bool = True,
    include_clash: bool = True,
) -> jnp.ndarray:
    """JAX scalar energy."""
    e = e_info_sum(positions, z_list) + LAMBDA_DAMP * e_damp_sum(positions, z_list)
    if include_bonds:
        e = e + e_bonds(positions)
    if include_clash:
        e = e + e_clash(positions)
    if include_horizon:
        e = e + e_horizon(positions)
    return e


_grad_positions_jax = grad(energy_total_jax, argnums=0)


def grad_positions_jax(
    positions: jnp.ndarray,
    z_list: jnp.ndarray,
    include_bonds: bool = True,
    include_horizon: bool = True,
    include_clash: bool = True,
) -> jnp.ndarray:
    """Gradient of energy w.r.t. positions."""
    return _grad_positions_jax(
        positions,
        z_list,
        include_bonds=include_bonds,
        include_horizon=include_horizon,
        include_clash=include_clash,
    )
