"""Tests for Lean-aligned assembly energy bookkeeping (``assembly_energy`` / ``folding_energy``)."""

from __future__ import annotations

import numpy as np

from horizon_physics.proteins.assembly_energy import (
    assembly_fold_energy_tree_sum,
    decompose_ca_fold_energy_scalar_budget,
    linear_polymer_parent_list,
    sum_bond_valley_over_edges,
)
from horizon_physics.proteins.folding_energy import (
    e_atomic_site,
    e_sequential_bond_penalty_ca,
    e_tot,
    e_tot_ca_with_bonds,
)


def test_sum_atomic_sites_equals_e_tot():
    pos = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.5, 0.2, 0.0]], dtype=float)
    z = np.array([6, 6, 6], dtype=int)
    s = sum(e_atomic_site(pos, z, i) for i in range(3))
    assert abs(s - e_tot(pos, z)) < 1e-9


def test_linear_tree_branch_eq_matches_sequential_bonds():
    pos = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0], [11.0, 0.5, 0.0]], dtype=float)
    z = np.full(4, 6, dtype=int)
    par = linear_polymer_parent_list(4)
    tree = assembly_fold_energy_tree_sum(pos, z, par)
    bond = e_sequential_bond_penalty_ca(pos)
    et = e_tot(pos, z)
    assert abs(tree - (et + bond)) < 1e-9


def test_decompose_matches_e_tot_ca_with_bonds():
    pos = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.5, 0.3, 0.0]], dtype=float)
    z = np.full(3, 6, dtype=int)
    d = decompose_ca_fold_energy_scalar_budget(pos, z)
    assert d["branch_eq_linear_ok"] < 1e-9
    assert abs(d["total_ca_with_bonds"] - e_tot_ca_with_bonds(pos, z)) < 1e-9
    assert abs(d["atomic_sum"] + d["bond_sequential"] + d["clash"] - d["total_ca_with_bonds"]) < 1e-9


def test_binary_star_edges_sum():
    """Two leaves attached to one root: same bond sum as explicit edge list."""
    pos = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=float)
    z = np.full(3, 6, dtype=int)
    parent_of = [None, 0, 0]
    tree = assembly_fold_energy_tree_sum(pos, z, parent_of)
    bond_list = sum_bond_valley_over_edges(pos, [(0, 1), (0, 2)])
    et = e_tot(pos, z)
    assert abs(tree - (et + bond_list)) < 1e-9


def test_multi_root_raises():
    pos = np.zeros((2, 3))
    z = np.full(2, 6, dtype=int)
    try:
        assembly_fold_energy_tree_sum(pos, z, [None, None])
    except ValueError:
        return
    raise AssertionError("expected ValueError for two roots")
