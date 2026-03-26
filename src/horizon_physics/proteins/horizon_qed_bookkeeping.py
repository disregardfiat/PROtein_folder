"""
Horizon QED scalars aligned with Lean ``Hqiv/QuantumOptics/HorizonQED.lean``.

Depends on the same shell ladder as ``OctonionicLightCone`` / ``AuxiliaryField``:
``latticeSimplexCount``, ``T`` / ``T_Pl`` ratios, and ``phi_of_shell`` where relevant.

Use this module for **bookkeeping** (mode counts, ω tags, zero-point sums, JC/Rabi scalars)
when replacing ad hoc ``fa*``-style heuristic factors with Lean-named quantities.

Units
-----
- Energies in **eV** where noted (``k_b_ev_k`` × Kelvin).
- ``omega_shell_si_rad_per_s`` uses ħ in eV·s → rad/s.
- ``field_quantization_prefactor_si`` expects SI: ħ [J·s], ω [rad/s], V [m³], ε₀ [F/m].

Full bosonic Fock space and canonical commutation relations are **not** implemented here
(Lean notes the same scope: finite Pauli / two-level block only).
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from .hqiv_long_range import PHI_TEMPERATURE_COEFF, lattice_simplex_count, phi_of_shell

# k_B [eV/K] — same as ``temperature_path_search.K_B_EV_K`` / gradient Langevin paths
K_B_EV_K = 8.617333262e-5
# ħ [eV·s] (CODATA)
HBAR_EV_S = 6.582119569e-16
# JC ladder–φ tag: Lean ``jcCouplingTag``, ``jcCouplingTag_eq_sqrt_two``
JC_COUPLING_TAG = math.sqrt(2.0)
# Lindblad scalar rate placeholder (≥ 0); set from experiment or upstream theory when wired
LINDBLAD_SCALAR_RATE = 0.0


def shell_spatial_mode_count(m: int) -> int:
    """
    Lean ``shellSpatialModeCount``; equals ``latticeSimplexCount m`` per ``shellSpatialModeCount_eq``.
    """
    return lattice_simplex_count(m)


def dimensionless_omega_shell(m: int) -> float:
    """
    Lean ``dimensionlessOmegaShell``: ``T(m) / T_Pl = 1 / (m + 1)`` (``dimensionlessOmegaShell_eq``).
    """
    if m < 0:
        raise ValueError("shell index m must be nonnegative")
    return 1.0 / float(m + 1)


def omega_shell_si_rad_per_s(
    m: int,
    temperature_k: float,
    *,
    k_b_ev_k: float = K_B_EV_K,
    hbar_ev_s: float = HBAR_EV_S,
) -> float:
    """
    Angular frequency [rad/s] from ``ω_m = (k_B T / ℏ) * (T(m)/T_Pl)`` with ``T(m)/T_Pl = 1/(m+1)``.

    So ``ω_m = (k_B T_K / ħ) / (m + 1)`` at reference temperature ``temperature_k``.
    """
    if hbar_ev_s <= 0:
        raise ValueError("hbar_ev_s must be positive")
    return (float(k_b_ev_k) * float(temperature_k) / hbar_ev_s) * dimensionless_omega_shell(m)


def zero_point_energy_shell_ev(
    m: int,
    temperature_k: float,
    *,
    k_b_ev_k: float = K_B_EV_K,
    hbar_ev_s: float = HBAR_EV_S,
) -> float:
    """
    Lean ``zeroPointEnergyShellSI`` identification: ``(1/2) ħ ω_m`` with ``ħ ω_m = k_B T * T(m)/T_Pl``.

    Yields ``(1/2) k_B T / (m + 1)`` [eV] at ``temperature_k`` (needs ``hbar_ev_s > 0`` for parity checks).
    """
    if hbar_ev_s <= 0:
        raise ValueError("hbar_ev_s must be positive (Lean ``zeroPointEnergyShellSI_eq``)")
    hbar_omega = float(k_b_ev_k) * float(temperature_k) * dimensionless_omega_shell(m)
    return 0.5 * hbar_omega


def default_shell_temperature_k(m: int, reference_temperature_k: float) -> float:
    """``T(m) = T_ref / (m + 1)`` — consistent with ``dimensionless_omega_shell`` as ``T(m)/T_Pl`` tag."""
    if m < 0:
        raise ValueError("shell index m must be nonnegative")
    return float(reference_temperature_k) / float(m + 1)


def truncated_vacuum_zero_point_ev(
    cap_m: int,
    reference_temperature_k: float,
    *,
    k_b_ev_k: float = K_B_EV_K,
    hbar_ev_s: float = HBAR_EV_S,
    shell_temperature_k: Optional[Callable[[int], float]] = None,
    mode_count_fn: Callable[[int], int] = shell_spatial_mode_count,
) -> float:
    """
    Lean ``truncatedVacuumZeroPointSI``: ``∑_{m < M} N_m * (1/2) k_B T(m)`` [eV].

    Default ``T(m) = reference_temperature_k / (m + 1)``. Override via ``shell_temperature_k``.
    Default ``N_m = shellSpatialModeCount m``.
    """
    if cap_m < 0:
        raise ValueError("cap_m must be nonnegative")
    if hbar_ev_s <= 0:
        raise ValueError("hbar_ev_s must be positive")
    total = 0.0
    T_ref = float(reference_temperature_k)
    for m in range(int(cap_m)):
        Nm = int(mode_count_fn(m))
        Tk = (
            float(shell_temperature_k(m))
            if shell_temperature_k is not None
            else default_shell_temperature_k(m, T_ref)
        )
        total += float(Nm) * 0.5 * float(k_b_ev_k) * Tk
    return total


def field_quantization_prefactor_si(
    omega_rad_s: float,
    epsilon0: float,
    volume_m3: float,
    hbar_j_s: float,
) -> float:
    """
    Lean ``fieldQuantizationPrefactorSI``: ``√(ħ ω / (2 ε₀ V))`` in SI base units (J^{1/2}·…).

    All arguments must be strictly positive for the formal ``pos`` lemma analogue.
    """
    if omega_rad_s <= 0 or epsilon0 <= 0 or volume_m3 <= 0 or hbar_j_s <= 0:
        raise ValueError("omega_rad_s, epsilon0, volume_m3, hbar_j_s must be positive")
    return math.sqrt(hbar_j_s * omega_rad_s / (2.0 * epsilon0 * volume_m3))


# Pauli matrices over ℂ (Lean ``sigmaPlus``, ``sigmaMinus``, ``sigmaZ``)
SIGMA_PLUS = np.array([[0.0 + 0.0j, 1.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex128)
SIGMA_MINUS = np.array([[0.0 + 0.0j, 0.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex128)
SIGMA_Z = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]], dtype=np.complex128)


def _commutator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b - b @ a


def rabi_angular_frequency(g: float) -> float:
    """Lean ``rabiAngularFrequency``: ``Ω = 2 g`` (``rabiAngularFrequency_pos`` for ``g > 0``)."""
    return 2.0 * float(g)


def jc_coupling_tag() -> float:
    """Lean ``jcCouplingTag`` (= √2)."""
    return JC_COUPLING_TAG


def lindblad_scalar_rate() -> float:
    """Lean ``lindbladScalarRate`` placeholder (nonnegative)."""
    return float(LINDBLAD_SCALAR_RATE)


if __name__ == "__main__":
    # Sanity: JC algebra matches Lean commutator lemmas (numerical)
    sp, sm, sz = SIGMA_PLUS, SIGMA_MINUS, SIGMA_Z
    assert np.allclose(_commutator(sp, sm), sz)
    assert np.allclose(_commutator(sz, sp), 2.0 * sp)
    assert np.allclose(_commutator(sz, sm), -2.0 * sm)
    m = 2
    print(
        f"m={m}  N_spatial={shell_spatial_mode_count(m)}  "
        f"omega_tilde={dimensionless_omega_shell(m):.4f}  "
        f"phi_of_shell={phi_of_shell(m):.4g}  PHI coeff={PHI_TEMPERATURE_COEFF}"
    )
    T310 = 310.0
    print(f"  omega_si(310K)={omega_shell_si_rad_per_s(m, T310):.4e} rad/s")
    print(f"  ZP/2 shell eV={zero_point_energy_shell_ev(m, T310):.4e}")
    print(f"  truncated ZP M=4: {truncated_vacuum_zero_point_ev(4, T310):.4e} eV")
