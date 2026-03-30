from __future__ import annotations

import numpy as np
import pytest

from horizon_physics.proteins.osh_oracle_backbone import minimize_backbone_with_osh_oracle
from horizon_physics.proteins.osh_oracle_folding import (
    REFERENCE_M_HQIV_NATIVE,
    auto_detect_cys_ligation_pairs,
    amplify_low_energy,
    apply_gate_sparse,
    apply_gate_sparse_hqiv_native,
    hqiv_harmonic_flat_index_ell_m0,
    hqiv_pivot_from_shells,
    contact_reflector_indices,
    per_residue_terminus_step_scale,
    compute_tunnel_harmonic_budget_ev,
    current_parameters,
    causal_expand_support,
    detect_flipped_kets,
    detect_flipped_kets_amplitude,
    estimate_natural_harmonic_scale_ca,
    harmonic_temperature_schedule,
    harmonic_tunneled_qaoa_folding,
    qpe_low_energy_subspace,
    minimize_ca_with_osh_oracle,
    minimize_ca_with_osh_oracle_additive_cycles,
    metropolis_accept_with_harmonic,
    _local_rapidity_displacement,
    prune_to_flipped,
    sparse_visible_energy,
    sparse_basis_card,
    wrap_idx,
)
from horizon_physics.proteins.pipeline_interchange import FoldState, make_osh_oracle_stage, run_pipeline


def test_sparse_primitives_follow_lean_shape():
    L = 3
    reg = [(0, 1.0), (2, 0.5)]
    assert sparse_basis_card(L) == 16
    assert wrap_idx(L, 17) == 1
    expd = causal_expand_support(L, reg)
    assert len(expd) == 2 * len(reg)
    out = apply_gate_sparse(L, reg, gate_mix=0.5)
    assert len(out) == 2 * len(reg)


# Quantum gate map visualization (from Grok-generated picture):
# ![HQIV-native sparse pi-phase gate map](assets/images/Uu9Hk.jpg)
# Image credit: Grok (from the provided picture)
def test_hqiv_native_pivot_and_phase_preserves_norm_sq():
    L = 3
    shells = np.array([1, 2, 3], dtype=np.int64)
    pv = hqiv_pivot_from_shells(shells, L, reference_m=REFERENCE_M_HQIV_NATIVE)
    assert pv == (int(np.sum(shells)) + REFERENCE_M_HQIV_NATIVE) % (L + 1)
    fi = hqiv_harmonic_flat_index_ell_m0(L, pv)
    assert 0 <= fi < sparse_basis_card(L)
    reg = [(0, 1.0), (2, 0.5)]
    out = apply_gate_sparse_hqiv_native(L, reg, shells, reference_m=REFERENCE_M_HQIV_NATIVE)
    assert len(out) == 2 * len(reg)
    # π phase on one mode preserves Euclidean norm of the dense intermediate.
    from horizon_physics.proteins.osh_oracle_folding import (
        _hqiv_phase_negate_one_mode,
        dense_of_sparse,
    )

    expd = causal_expand_support(L, reg)
    dense = dense_of_sparse(L, expd)
    fi = hqiv_harmonic_flat_index_ell_m0(L, pv)
    evolved = _hqiv_phase_negate_one_mode(dense, fi)
    assert abs(float(np.sum(dense * dense)) - float(np.sum(evolved * evolved))) < 1e-9


def test_minimize_ca_with_hqiv_native_gate_smoke():
    ca0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.8, 0.2, 0.0],
            [7.7, -0.1, 0.1],
            [11.5, 0.3, -0.2],
            [15.2, 0.0, 0.3],
        ],
        dtype=float,
    )
    ca1, info = minimize_ca_with_osh_oracle(
        ca0,
        n_iter=12,
        step_size=0.02,
        ansatz_depth=3,
        use_hqiv_native_gate=True,
        hqiv_reference_m=REFERENCE_M_HQIV_NATIVE,
    )
    assert ca1.shape == ca0.shape
    assert info.iterations_executed <= info.iterations


def test_detect_and_prune_support():
    before = [(0, 1.0), (1, 1.0), (3, 0.2)]
    after = [(1, 0.8), (2, 0.6), (3, 0.3)]
    flipped = detect_flipped_kets(before, after)
    assert 0 in flipped and 2 in flipped
    pruned = prune_to_flipped(flipped, after)
    idx = [i for i, _ in pruned]
    assert set(idx).issubset(set(flipped))


def test_detect_flipped_kets_amplitude_and_sign():
    before = [(0, 1.0), (1, -0.25), (2, 0.5)]
    after = [(0, -1.0), (1, -0.25), (2, 0.51)]
    flipped = detect_flipped_kets_amplitude(before, after, amp_delta_eps=1e-3, include_sign_flip=True)
    assert 0 in flipped
    assert 2 in flipped


def test_current_parameters_bounds():
    for i in range(8):
        phi_mix, psi_mix = current_parameters(i, 8, 0.6)
        assert 0.0 <= phi_mix <= 1.0
        assert 0.0 <= psi_mix <= 1.0


def test_minimize_backbone_with_osh_oracle_smoke():
    from horizon_physics.proteins.casp_submission import _place_full_backbone

    seq = "ACAG"
    ca = np.array([[i * 3.8, 0.0, 0.0] for i in range(len(seq))], dtype=float)
    bb = _place_full_backbone(ca, seq)
    pos = np.array([xyz for _, xyz in bb], dtype=float)
    pos2, info = minimize_backbone_with_osh_oracle(
        pos,
        n_iter=8,
        step_size=0.02,
        ansatz_depth=2,
        use_energy_reservoir=False,
    )
    assert pos2.shape == pos.shape
    assert info.iterations_executed <= info.iterations
    assert info.last_step_size > 0.0


def test_minimize_ca_with_osh_oracle_smoke():
    ca0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.8, 0.2, 0.0],
            [7.7, -0.1, 0.1],
            [11.5, 0.3, -0.2],
            [15.2, 0.0, 0.3],
        ],
        dtype=float,
    )
    ca1, info = minimize_ca_with_osh_oracle(ca0, n_iter=12, step_size=0.02, ansatz_depth=3)
    assert ca1.shape == ca0.shape
    assert info.iterations == 12
    assert info.iterations_executed <= info.iterations
    assert info.last_step_size > 0.0
    assert info.avg_flipped_count >= 0.0
    assert info.natural_harmonic_scale > 0.0


def test_minimize_ca_with_additive_cycles_smoke():
    ca0 = np.array([[3.8 * i, 0.0, 0.0] for i in range(10)], dtype=float)
    ca1, info, cycles = minimize_ca_with_osh_oracle_additive_cycles(
        ca0,
        n_iter=12,
        max_cycles=2,
        step_size=0.02,
        use_energy_reservoir=True,
        reservoir_init=5.0,
        additive_kick_gain=0.001,
        additive_update_every=1,
    )
    assert ca1.shape == ca0.shape
    assert info.iterations_executed <= info.iterations
    assert len(cycles) >= 1
    assert cycles[0].reservoir_before_ev >= 0.0


def test_harmonic_helpers_and_acceptance():
    ca0 = np.array([[3.8 * i, 0.0, 0.0] for i in range(6)], dtype=float)
    w = estimate_natural_harmonic_scale_ca(ca0, 6, max_dims=12)
    assert w > 0.0
    t0 = harmonic_temperature_schedule(w, 0, 20, initial_energy_abs=1000.0)
    t1 = harmonic_temperature_schedule(w, 19, 20, initial_energy_abs=1000.0)
    assert t0 >= t1
    rng = np.random.default_rng(123)
    assert (
        metropolis_accept_with_harmonic(
            1.0,
            0.9,
            iteration=5,
            n_iter=20,
            omega=w,
            initial_energy_abs=1000.0,
            rng=rng,
        )
        is True
    )


def test_rapidity_displacement_respects_active_subset() -> None:
    """
    Rapidity translation is fully computable; when we prune to a sparse active support,
    the displacement must match the dense computation restricted to that same active set.
    """
    rng = np.random.default_rng(1234)
    n = 8
    ca = rng.normal(size=(n, 3)).astype(float)
    grad = rng.normal(size=(n, 3)).astype(float)

    active_idx = np.array([2, 3, 5], dtype=int)  # all interior indices (1..n-2)
    disp_sparse = _local_rapidity_displacement(
        ca, grad, active_idx, gain=0.25, tangent_weight=0.7, normal_weight=0.3
    )
    disp_dense = _local_rapidity_displacement(
        ca, grad, np.arange(n, dtype=int), gain=0.25, tangent_weight=0.7, normal_weight=0.3
    )

    # In sparse mode, inactive indices must not be written at all.
    for i in range(n):
        if i in set(active_idx.tolist()):
            assert np.allclose(disp_sparse[i], disp_dense[i], atol=1e-12, rtol=0.0)
        else:
            assert np.allclose(disp_sparse[i], np.zeros(3), atol=1e-12, rtol=0.0)


def test_prune_to_flipped_preserves_sparse_energy_when_kept() -> None:
    """
    Mirrors the paper's "Flipped support and pruning" lemma:
    if every active ket in `r` has its index inside `flipped`, pruning must preserve
    the sparse Euclidean energy on listed amplitudes.
    """
    L = 3
    basis = sparse_basis_card(L)

    rng = np.random.default_rng(42)
    # Construct an "after" sparse register with explicitly chosen indices.
    after = [(2, float(rng.normal())), (5, float(rng.normal())), (9, float(rng.normal()))]

    # Case 1: flipped contains all indices in `after` (premise holds).
    flipped_ok = sorted({2, 5, 9, 13, 1})
    local_energy = np.ones((basis,), dtype=float)
    e0 = sparse_visible_energy(after, local_energy)
    pruned_ok = prune_to_flipped(flipped_ok, after)
    e1 = sparse_visible_energy(pruned_ok, local_energy)
    assert pruned_ok == after  # no entries should be removed
    assert np.isclose(e1, e0, rtol=0.0, atol=1e-12)

    # Case 2: flipped misses one index (premise fails) -> energy should drop.
    flipped_bad = sorted({2, 5})
    pruned_bad = prune_to_flipped(flipped_bad, after)
    assert [i for i, _ in pruned_bad] == [2, 5]
    e2 = sparse_visible_energy(pruned_bad, local_energy)
    assert e2 < e0


def test_pipeline_stage_runs():
    seq = "MKFLN"
    ca0 = np.array([[3.8 * i, 0.0, 0.0] for i in range(len(seq))], dtype=float)
    s0 = FoldState(sequence=seq, ca_positions=ca0)
    stage = make_osh_oracle_stage(
        n_iter=8,
        step_size=0.02,
        use_harmonic_metropolis=True,
        harmonic_max_dims=12,
        random_seed=123,
    )
    out = run_pipeline(s0, [stage])
    assert out.ca_positions is not None
    assert len(out.stage_history) == 1
    assert out.stage_history[0]["stage"] == "osh_oracle_sparse_refine"


def test_settle_stop_can_trigger():
    ca0 = np.array([[3.8 * i, 0.0, 0.0] for i in range(8)], dtype=float)
    _, info = minimize_ca_with_osh_oracle(
        ca0,
        n_iter=80,
        step_size=0.01,
        stop_when_settled=True,
        settle_window=8,
        settle_energy_tol=1e-2,
        settle_step_tol=5e-4,
        settle_min_iter=10,
    )
    assert info.iterations_executed <= info.iterations
    assert info.stop_reason in {"settled", "max_iter_reached", "early_break_no_active_support"}
    assert info.reservoir_energy_final_ev >= 0.0


def test_qpe_and_amplification_helpers():
    local_e = np.array([0.1, 1.0, 0.2, 2.0], dtype=float)
    reg = [(0, 0.9), (1, 0.2), (2, 0.7), (3, 0.1)]
    vis = sparse_visible_energy(reg, local_e)
    assert vis > 0.0
    tgt = qpe_low_energy_subspace(reg, local_e, k=2)
    assert len(tgt) == 2
    amp = amplify_low_energy(reg, tgt, gain=2.0, non_target_decay=0.9)
    assert len(amp) == len(reg)


def test_harmonic_tunneled_qaoa_folding_smoke():
    ca0 = np.array([[3.8 * i, 0.0, 0.0] for i in range(10)], dtype=float)
    ca1, info = harmonic_tunneled_qaoa_folding(
        ca0,
        layers=4,
        depth=4,
        qpe_k=3,
        base_step=0.02,
        use_harmonic_metropolis=False,
    )
    assert ca1.shape == ca0.shape
    assert info.layers == 4
    assert info.final_energy_ev >= 0.0
    assert info.natural_harmonic_scale > 0.0
    assert info.reservoir_energy_final_ev >= 0.0


def test_contact_reflector_indices_detects_proximity():
    ca = np.array([[3.8 * float(i), 0.0, 0.0] for i in range(8)], dtype=float)
    ca[0] = ca[5] + np.array([2.0, 0.0, 0.0], dtype=float)
    s = contact_reflector_indices(
        ca, None, min_seq_sep=4, cutoff_ang=8.0, max_reflectors=16, grad_coupling=0.0
    )
    assert 0 in s and 5 in s


def test_per_residue_terminus_step_scale_endpoints():
    s = per_residue_terminus_step_scale(20, boost=1.6, transition_width=4, core_scale=1.0)
    assert s.shape == (20,)
    assert float(s[0]) == pytest.approx(1.6, rel=1e-6)
    assert float(s[19]) == pytest.approx(1.6, rel=1e-6)
    assert float(s[10]) == pytest.approx(1.0, rel=1e-6)


def test_contact_reflector_terminal_score_can_reorder_pairs():
    """Interior pair is geometrically tighter; terminal boost should flip priority."""
    n = 12
    ca = np.zeros((n, 3), dtype=float)
    special = {0, 4, 6, 8}
    for k in range(n):
        if k not in special:
            ca[k] = np.array([23.7 * float(k), 41.3 * float(k * k), 17.1 * float(k**3)], dtype=float)
    # Pair (4,8): seq sep 4, tight geometry.
    ca[4] = np.array([0.0, 0.0, 0.0], dtype=float)
    ca[8] = np.array([3.0, 0.0, 0.0], dtype=float)
    # Pair (0,6): looser geometry but involves N-terminus.
    ca[0] = np.array([10.0, 0.0, 0.0], dtype=float)
    ca[6] = np.array([14.0, 0.0, 0.0], dtype=float)
    s_interior_first = contact_reflector_indices(
        ca,
        None,
        min_seq_sep=4,
        cutoff_ang=8.0,
        max_reflectors=2,
        grad_coupling=0.0,
        contact_terminus_window=0,
        contact_terminus_score_scale=1.0,
    )
    s_term_bias = contact_reflector_indices(
        ca,
        None,
        min_seq_sep=4,
        cutoff_ang=8.0,
        max_reflectors=2,
        grad_coupling=0.0,
        contact_terminus_window=3,
        contact_terminus_score_scale=25.0,
    )
    assert s_interior_first == {4, 8}
    assert s_term_bias == {0, 6}


def test_tunnel_budget_changes_with_contact_reflectors():
    ca = np.array([[3.8 * float(i), 0.0, 0.0] for i in range(8)], dtype=float)
    ca[0] = ca[5] + np.array([2.0, 0.0, 0.0], dtype=float)
    masses = np.ones(8, dtype=float)
    b0 = compute_tunnel_harmonic_budget_ev(ca, masses, [], contact_reflectors=None)
    cref = contact_reflector_indices(ca, None, min_seq_sep=4, cutoff_ang=8.0, max_reflectors=16)
    b1 = compute_tunnel_harmonic_budget_ev(ca, masses, [], contact_reflectors=cref)
    assert b0 >= 0.0 and b1 >= 0.0
    assert abs(b1 - b0) > 1e-6


def test_minimize_contact_reflectors_exposes_counts():
    ca0 = np.array([[3.8 * i, 0.0, 0.0] for i in range(8)], dtype=float)
    _, info = minimize_ca_with_osh_oracle(
        ca0,
        n_iter=6,
        step_size=0.02,
        use_contact_reflectors=True,
        contact_cutoff_ang=15.0,
        omega_refresh_period=2,
    )
    assert info.contact_reflector_count >= 0
    assert info.omega_refresh_count >= 1


def test_ligation_helpers_and_run_path():
    seq = "ACCCCCA"
    ca0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [7.6, 0.0, 0.0],
            [10.5, 2.5, 0.0],
            [7.6, 5.0, 0.0],
            [3.8, 5.0, 0.0],
            [0.0, 5.0, 0.0],
        ],
        dtype=float,
    )
    pairs = auto_detect_cys_ligation_pairs(seq, ca0, max_dist_ang=6.5)
    assert isinstance(pairs, list)
    _, info = minimize_ca_with_osh_oracle(
        ca0,
        sequence=seq,
        auto_detect_cys_ligation=True,
        n_iter=10,
        step_size=0.01,
        use_local_rapidity_translation=True,
        rapidity_gain=0.2,
    )
    assert info.iterations_executed <= info.iterations
