"""fast_local_theta matches legacy Θ_i energy; neighbor_cutoff does not widen horizon pairs."""

from __future__ import annotations

import numpy as np

from horizon_physics.proteins import folding_energy as fe


def test_e_tot_informational_fast_matches_legacy():
    rng = np.random.default_rng(0)
    for n in (2, 8, 24, 40):
        pos = rng.normal(size=(n, 3)) * 4.0
        z = np.full(n, 6, dtype=int)
        e0 = fe.e_tot_informational(pos, z, fast_local_theta=False)
        e1 = fe.e_tot_informational(pos, z, fast_local_theta=True)
        assert abs(e0 - e1) < 1e-6 * max(1.0, abs(e0))


def test_e_tot_ca_with_bonds_fast_matches_legacy():
    rng = np.random.default_rng(1)
    n = 32
    pos = rng.normal(size=(n, 3)) * 3.5
    z = np.full(n, 6, dtype=int)
    e0 = fe.e_tot_ca_with_bonds(pos, z, fast_local_theta=False)
    e1 = fe.e_tot_ca_with_bonds(pos, z, fast_local_theta=True)
    assert abs(e0 - e1) < 1e-5 * max(1.0, abs(e0))


def test_horizon_neighbor_cutoff_fewer_pairs():
    rng = np.random.default_rng(2)
    n = 20
    pos = rng.normal(size=(n, 3)) * 5.0
    z = np.full(n, 6, dtype=int)
    poles_w = fe.build_horizon_poles(pos, z, neighbor_cutoff=12.0)
    poles_t = fe.build_horizon_poles(pos, z, neighbor_cutoff=6.0)
    assert len(poles_t) <= len(poles_w)
