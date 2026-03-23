"""Closed-form parity with Lean ``HQIVLongRange`` definitions (no Mathlib)."""

from __future__ import annotations

import math

from .hqiv_long_range import (
    K_hbond,
    R_hbond,
    available_modes_nat,
    h_bond_proxy,
    lattice_simplex_count,
    long_range_valley_with_aqueous_em_scalar,
    phi_of_shell,
    water_dielectric_r_effective,
)


def test_phi_of_shell_phi_temperature_coeff():
    for m in (0, 1, 4, 10):
        assert abs(phi_of_shell(m) - 2.0 * (m + 1)) < 1e-12


def test_lattice_simplex_and_modes():
    for m in (0, 2, 5):
        ls = (m + 2) * (m + 1)
        assert lattice_simplex_count(m) == ls
        assert available_modes_nat(m) == 4 * ls


def test_K_hbond_closed_form():
    for m in (0, 1, 3, 7):
        nu = float(available_modes_nat(m))
        want = 2.0 * (m + 1) / nu
        assert abs(K_hbond(m) - want) < 1e-12


def test_R_hbond():
    for m in (0, 5):
        assert abs(R_hbond(m) - float(m + 1)) < 1e-12


def test_h_bond_proxy_sign_when_cosines_positive():
    m = 3
    th = math.radians(40.0)
    ph = math.radians(30.0)
    d = 5.0
    hb = h_bond_proxy(m, th, ph, d)
    assert hb < 0.0
    k, rh = K_hbond(m), R_hbond(m)
    den = 1.0 + (d / rh) ** 2
    hand = -k * (math.cos(th) * math.cos(ph)) / den
    assert abs(hb - hand) < 1e-12


def test_water_dielectric_r_effective():
    assert abs(water_dielectric_r_effective(10.0, 80.0) - 800.0) < 1e-9


def test_long_range_valley_aqueous_scalar_split():
    m = 2
    em_scaled = 0.37
    hb = h_bond_proxy(m, 0.5, 0.4, 4.0)
    tot = long_range_valley_with_aqueous_em_scalar(em_scaled, m, 0.5, 0.4, 4.0)
    assert abs(tot - (em_scaled + hb)) < 1e-12


if __name__ == "__main__":
    test_phi_of_shell_phi_temperature_coeff()
    test_lattice_simplex_and_modes()
    test_K_hbond_closed_form()
    test_R_hbond()
    test_h_bond_proxy_sign_when_cosines_positive()
    test_water_dielectric_r_effective()
    test_long_range_valley_aqueous_scalar_split()
    print("test_hqiv_long_range_lean_parity passed.")
