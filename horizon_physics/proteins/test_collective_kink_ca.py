"""Cα collective kink budget vs HQIV α-helix geometry (Lean ``HQIVCollectiveModes``)."""

from __future__ import annotations

import numpy as np

from .alpha_helix import alpha_helix_xyz
from .collective_modes_scalars import (
    ca_virtual_bond_angle_rad,
    default_collective_kink_theta_ref_rad,
    e_ca_collective_kink_sum,
    grad_e_ca_collective_kink_fd,
)
from .folding_energy import e_tot_ca_with_bonds, grad_full


def test_default_theta_matches_helix_trace():
    ca = alpha_helix_xyz(np.arange(16))
    tr = default_collective_kink_theta_ref_rad()
    i = 8
    th = ca_virtual_bond_angle_rad(ca, i)
    assert abs(th - tr) < 0.02


def test_kink_sum_near_zero_on_ideal_helix():
    ca = alpha_helix_xyz(np.arange(24))
    tr = default_collective_kink_theta_ref_rad()
    e = e_ca_collective_kink_sum(ca, 3, theta_ref_rad=tr)
    assert e < 0.05


def test_extended_chain_higher_kink_than_helix():
    n = 24
    line = np.zeros((n, 3))
    line[:, 0] = np.arange(n) * 3.8
    hel = alpha_helix_xyz(np.arange(n))
    tr = default_collective_kink_theta_ref_rad()
    e_line = e_ca_collective_kink_sum(line, 3, theta_ref_rad=tr)
    e_hel = e_ca_collective_kink_sum(hel, 3, theta_ref_rad=tr)
    assert e_line > e_hel + 1.0


def test_grad_fd_nonzero_on_deformed():
    ca = alpha_helix_xyz(np.arange(10))
    ca[5, 0] += 2.0
    tr = default_collective_kink_theta_ref_rad()
    g = grad_e_ca_collective_kink_fd(ca, 3, theta_ref_rad=tr, eps=2e-3)
    assert float(np.linalg.norm(g)) > 1e-6


def test_e_tot_ca_includes_collective_kink():
    n = 8
    z = np.full(n, 6)
    line = np.zeros((n, 3))
    line[:, 0] = np.arange(n) * 3.8
    e0 = e_tot_ca_with_bonds(line, z, collective_kink_weight=0.0)
    e1 = e_tot_ca_with_bonds(line, z, collective_kink_weight=0.01)
    assert e1 > e0


def test_grad_full_collective_kink():
    n = 6
    z = np.full(n, 6)
    ca = alpha_helix_xyz(np.arange(n))
    ca[3, 1] += 1.0
    g0 = grad_full(ca, z, include_bonds=True, include_horizon=False, include_clash=False)
    g1 = grad_full(
        ca,
        z,
        include_bonds=True,
        include_horizon=False,
        include_clash=False,
        collective_kink_weight=0.05,
        collective_kink_m=3,
    )
    assert float(np.linalg.norm(g1 - g0)) > 1e-9


if __name__ == "__main__":
    test_default_theta_matches_helix_trace()
    test_kink_sum_near_zero_on_ideal_helix()
    test_extended_chain_higher_kink_than_helix()
    test_grad_fd_nonzero_on_deformed()
    test_e_tot_ca_includes_collective_kink()
    test_grad_full_collective_kink()
    print("test_collective_kink_ca passed.")
