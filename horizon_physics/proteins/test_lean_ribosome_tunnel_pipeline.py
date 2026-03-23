"""
Tests for Lean ribosome tunnel pipeline and post-extrusion refinement (no runaway loops).

Run: python3 -m horizon_physics.proteins.test_lean_ribosome_tunnel_pipeline
Or: pytest horizon_physics/proteins/test_lean_ribosome_tunnel_pipeline.py -v
"""

from __future__ import annotations

import time

from .full_protein_minimizer import minimize_full_chain
from .lean_ribosome_tunnel_pipeline import fold_lean_ribosome_tunnel


def test_post_extrusion_respects_round_cap_and_floor():
    """Short chains used to use max_disp = 0.5*n/100 with no floor → many outer rounds."""
    seq = "MKFLNDFE"  # 8 residues
    raw = minimize_full_chain(
        seq,
        simulate_ribosome_tunnel=True,
        post_extrusion_refine=True,
        post_extrusion_refine_mode="hke",
        post_extrusion_max_disp_floor=0.25,
        post_extrusion_max_rounds=5,
        fast_pass_steps_per_connection=0,
        min_pass_iter_per_connection=0,
        kappa_dihedral=0.0,
        quick=True,
    )
    info = raw["info"]
    assert "post_extrusion_rounds" in info
    assert info["post_extrusion_rounds"] <= 5
    assert info["post_extrusion_max_disp_threshold"] >= 0.25


def test_lean_pipeline_completes_fast_no_tunnel_hke():
    """No connection-triggered HKE (0 fast-pass, 0 L-BFGS iters per segment); post-extrusion still bounded."""
    t0 = time.perf_counter()
    r = fold_lean_ribosome_tunnel(
        "MKFL",
        kappa_dihedral=0.0,
        quick=True,
        fast_pass_steps_per_connection=0,
        min_pass_iter_per_connection=0,
        post_extrusion_refine_mode="hke",
        post_extrusion_max_rounds=8,
        post_extrusion_max_disp_floor=0.25,
    )
    elapsed = time.perf_counter() - t0
    assert len(r.pdb) > 100
    assert r.raw_result["info"].get("post_extrusion_rounds", 0) <= 8
    assert elapsed < 30.0, f"took {elapsed:.1f}s"


def test_post_extrusion_threshold_not_absurd_for_small_n():
    """Floor prevents sub-0.1 Å thresholds for peptides (would stall refinement)."""
    n = 5
    floor = 0.25
    thr = max(floor, 0.5 * (n / 100.0))
    assert thr == floor


def test_post_extrusion_em_treetorque_anneal_smoke():
    """Anneal requested after EM: valid PDB; anneal metadata if pyhqiv discrete DOFs are available."""
    r = fold_lean_ribosome_tunnel(
        "MKFL",
        kappa_dihedral=0.0,
        quick=True,
        fast_pass_steps_per_connection=0,
        min_pass_iter_per_connection=0,
        post_extrusion_refine_mode="em_treetorque",
        post_extrusion_anneal=True,
        post_extrusion_treetorque_phases=2,
        post_extrusion_treetorque_n_steps=48,
    )
    assert len(r.pdb) > 100
    info = r.raw_result.get("info", {})
    assert info.get("post_extrusion_anneal") is True
    err = str(info.get("treetorque_error") or "")
    if "pyhqiv" in err.lower():
        # Discrete tree-torque / anneal needs pyhqiv.molecular; EM-only path still returns a structure.
        return
    assert info.get("anneal_stages") is not None or info.get("anneal_schedule_k") is not None


if __name__ == "__main__":
    test_post_extrusion_respects_round_cap_and_floor()
    test_lean_pipeline_completes_fast_no_tunnel_hke()
    test_post_extrusion_threshold_not_absurd_for_small_n()
    test_post_extrusion_em_treetorque_anneal_smoke()
    print("All lean ribosome tunnel pipeline tests passed.")
