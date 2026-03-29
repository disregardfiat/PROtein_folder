"""Tests for Lean-aligned QC soft clash (``ProteinQCRefinement`` Python port)."""

import numpy as np

from horizon_physics.proteins.protein_qc_refinement import (
    grad_qc_soft_clash,
    grad_qc_soft_clash_protein_fixed,
    qc_soft_clash_energy,
    qc_soft_clash_energy_protein_ligand,
)


def test_soft_clash_energy_matches_finite_difference():
    rng = np.random.default_rng(0)
    pos = rng.standard_normal((6, 3)) * 2.0
    sigma = 2.8
    eps = 1e-5
    g = grad_qc_soft_clash(pos, sigma)
    for i in range(6):
        for c in range(3):
            pos_p = pos.copy()
            pos_m = pos.copy()
            pos_p[i, c] += eps
            pos_m[i, c] -= eps
            fd = (qc_soft_clash_energy(pos_p, sigma) - qc_soft_clash_energy(pos_m, sigma)) / (2 * eps)
            assert abs(fd - g[i, c]) < 5e-4


def test_protein_fixed_matches_full_gradient_on_ligand_rows():
    rng = np.random.default_rng(1)
    n_p, n_l = 20, 4
    pp = rng.standard_normal((n_p, 3))
    pl = rng.standard_normal((n_l, 3))
    sigma = 3.0
    comb = np.vstack([pp, pl])
    g_full = grad_qc_soft_clash(comb, sigma)
    g_sub = grad_qc_soft_clash_protein_fixed(pp, pl, sigma)
    np.testing.assert_allclose(g_full[n_p:], g_sub, rtol=1e-9, atol=1e-9)


def test_protein_ligand_energy_nonneg():
    rng = np.random.default_rng(2)
    pp = rng.standard_normal((10, 3))
    pl = rng.standard_normal((3, 3))
    e = qc_soft_clash_energy_protein_ligand(pp, pl, 3.0)
    assert e >= 0.0
