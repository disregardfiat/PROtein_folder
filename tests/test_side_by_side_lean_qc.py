"""Smoke tests for Lean QC side-by-side minimization."""

from __future__ import annotations

import numpy as np

from horizon_physics.proteins.side_by_side_lean_qc import (
    eta_mode_phi_constant,
    lattice_full_mode_energy_ev,
    run_side_by_side_lean_qc,
    site_energy_trace_ev,
)


def test_eta_mode_phi_reference_m4():
    assert abs(eta_mode_phi_constant(4) - 1.0 / 30.0) < 1e-12


def test_site_trace_matches_sum_lattice_energy():
    z = np.array([3, 4, 5], dtype=np.int32)
    s = site_energy_trace_ev(z)
    manual = sum(lattice_full_mode_energy_ev(int(z[i])) for i in range(3))
    assert abs(s - manual) < 1e-9


def test_run_side_by_side_lean_qc_runs():
    rng = np.random.default_rng(0)
    ca = np.cumsum(rng.normal(0.0, 1.0, size=(8, 3)), axis=0)
    ca *= 3.8
    out = run_side_by_side_lean_qc(
        ca,
        z_shell=6,
        traditional_iters=12,
        lean_iters=12,
        traditional_step_size=0.015,
        lean_step_size=0.015,
        ang_mix=0.05,
        ensure_progress=False,
    )
    assert out["n_residues"] == 8
    assert "traditional" in out and "lean_natural_disp" in out
    assert out["site_energy_trace_ev"] > 0.0
    assert np.isfinite(out["traditional"]["final_energy_ev"])
    assert np.isfinite(out["lean_natural_disp"]["final_energy_ev"])
    assert "ProteinFoldingQuantumChemistry.lean" in out["lean_references"]["quantum_chemistry_contract"]
