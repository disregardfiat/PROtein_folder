"""Parity tests for ``collective_modes_scalars`` vs Lean HQIVCollectiveModes anchors."""

from __future__ import annotations

from .collective_modes_scalars import (
    ANNEAL_TEMPERATURE_KELVIN,
    KINK_RMSD_UPPER_ANGSTROM,
    PDB_9GGO_A_RMSD_UPPER_A,
    collective_relax_helix_torques_dt,
    collective_relax_scalar,
    helix_multipole_kink_budget,
    helix_multipole_torque,
    k_multipole,
)
from .hqiv_long_range import K_hbond


def test_k_multipole_matches_k_hbond():
    for m in (0, 1, 3, 5):
        assert k_multipole(m) == K_hbond(m)


def test_lean_symbolic_anchors():
    assert ANNEAL_TEMPERATURE_KELVIN == 310.0
    assert KINK_RMSD_UPPER_ANGSTROM == 2.0
    assert PDB_9GGO_A_RMSD_UPPER_A == 7.0
    assert 14.0 - PDB_9GGO_A_RMSD_UPPER_A == 7.0
    assert KINK_RMSD_UPPER_ANGSTROM > 0.0


def test_collective_relax_scalar_sign():
    m = 3
    dt = 0.01
    c = 0.5
    s = collective_relax_scalar(m, c, dt)
    assert s == -K_hbond(m) * c * dt


def test_kink_budget_positive_when_delta_positive():
    m = 3
    assert helix_multipole_kink_budget(m, 0.1) > 0.0
    assert helix_multipole_kink_budget(m, 0.0) == 0.0


def test_torque_kink_raises_less_negative_than_straight():
    """Lean ``helixMultipoleTorque_kink_raises_EM_budget``: lower coupling ⇒ less negative torque."""
    m = 3
    phi_lo = lambda _s: 0.2
    phi_hi = lambda _s: 0.9
    t_lo = helix_multipole_torque(m, 1, phi_lo)
    t_hi = helix_multipole_torque(m, 1, phi_hi)
    assert t_lo > t_hi


def test_collective_batch_linear_length():
    m = 3
    phi = lambda s: 1.0
    dt = 0.01
    out = collective_relax_helix_torques_dt(m, [2, 3, 5], phi, dt)
    assert len(out) == 3


if __name__ == "__main__":
    test_k_multipole_matches_k_hbond()
    test_lean_symbolic_anchors()
    test_collective_relax_scalar_sign()
    test_kink_budget_positive_when_delta_positive()
    test_torque_kink_raises_less_negative_than_straight()
    test_collective_batch_linear_length()
    print("collective_modes_scalars tests passed.")
