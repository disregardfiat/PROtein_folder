"""
OSHoracle on **variable-size per-residue heavy-atom blocks** (full side chains).

Uses ``folding_energy.e_tot_polymer_with_bonds`` / ``grad_full_polymer`` with a covalent edge list
from :mod:`horizon_physics.proteins.full_atom_topology`. One sparse register site per residue; each
accept moves **all** atoms in that residue's span.

**Coupled refinement (default):** Cα participates in the step and in bond projection; optional
``limit_ca_displacement_ang`` caps ‖ΔCα‖ per iteration vs the previous accepted state so horizon
and side-chain forces can still move the backbone without the blow-ups from unconstrained
projection + large sparse steps.

**Hard pin (optional):** ``freeze_ca_positions=True`` fixes Cα to ``ca_target`` (when
``k_ca_target`` > 0) or to the initial Cα after each projection, and zeros Cα gradients — use only
when you intentionally want side-chain-only motion (e.g. validation against a fixed trace).

Lean / HQIV: same horizon and screening hooks as backbone OSH (``emScale``, pH / ε_r via caller
``energy_kwargs``); see ``hqiv_lean_folding`` and ``lean_ribosome_tunnel_pipeline``.

For **evaluation-only** full-atom budgets (no OSH steps), use
:func:`horizon_physics.proteins.folding_energy.full_atom_polymer_energy_budget` or
:func:`horizon_physics.proteins.full_atom_topology.full_heavy_chain_energy_budget`, then drive
Cα search with :func:`horizon_physics.proteins.osh_oracle_folding.minimize_ca_with_osh_oracle` using
those scalars as outer-loop criteria or rewards.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .folding_energy import (
    K_BOND,
    K_CLASH,
    R_BOND_MAX,
    R_BOND_MIN,
    R_CLASH,
    e_tot_polymer_with_bonds,
    grad_full_polymer,
    project_polymer_covalent_bonds,
)
from .osh_oracle_folding import (
    LigationPairs,
    OSHOracleFoldInfo,
    SparseRegister,
    _inertial_pk_energy,
    apply_ansatz_sparse,
    compute_local_compaction_score,
    compute_tunnel_harmonic_budget_ev,
    contact_reflector_indices,
    current_parameters,
    detect_flipped_kets_amplitude,
    estimate_natural_harmonic_scale_ca,
    fixed_free_first_mode_factor,
    metropolis_accept_with_harmonic,
    per_residue_resonance_multiplier,
    per_residue_terminus_step_scale,
    prune_to_flipped,
    wrap_idx,
)


def _register_variable_residue_blocks(
    grad: np.ndarray,
    residue_ranges: List[Tuple[int, int]],
    *,
    L: int,
    amp_threshold: float,
) -> SparseRegister:
    g = np.asarray(grad, dtype=float)
    reg: SparseRegister = []
    for ri, (a, b) in enumerate(residue_ranges):
        sl = slice(int(a), int(b))
        m = float(np.linalg.norm(g[sl]))
        if m >= float(amp_threshold):
            reg.append((wrap_idx(int(L), int(ri)), m))
    return reg


def _energy_polymer(
    pos: np.ndarray,
    z: np.ndarray,
    bond_edges: List[Tuple[int, int, float]],
    e_kw: Dict[str, Any],
) -> float:
    kw = dict(e_kw)
    flt = bool(kw.pop("fast_local_theta", False))
    ca_idx = kw.pop("ca_atom_indices", None)
    ca_tgt = kw.pop("ca_target", None)
    k_ca_t = float(kw.pop("k_ca_target", 0.0))
    return float(
        e_tot_polymer_with_bonds(
            pos,
            z,
            bond_edges,
            fast_local_theta=flt,
            r_bond_min=float(kw.pop("r_bond_min", R_BOND_MIN)),
            r_bond_max=float(kw.pop("r_bond_max", R_BOND_MAX)),
            k_bond=float(kw.pop("k_bond", K_BOND)),
            r_clash=float(kw.pop("r_clash", R_CLASH)),
            k_clash=float(kw.pop("k_clash", K_CLASH)),
            include_clash=bool(kw.pop("include_clash", True)),
            ca_atom_indices=ca_idx,
            ca_target=ca_tgt,
            k_ca_target=k_ca_t,
        )
    )


def _grad_polymer(
    pos: np.ndarray,
    z: np.ndarray,
    bond_edges: List[Tuple[int, int, float]],
    e_kw: Dict[str, Any],
) -> np.ndarray:
    kw = dict(e_kw)
    kw.pop("fast_local_theta", None)
    ca_idx = kw.get("ca_atom_indices")
    ca_tgt = kw.get("ca_target")
    k_ca_t = float(kw.get("k_ca_target", 0.0))
    horizon_kw = {
        k: kw[k]
        for k in ("r_ref", "r_horizon", "k_horizon", "em_scale", "use_neighbor_list", "neighbor_cutoff")
        if k in kw
    }
    return grad_full_polymer(
        pos,
        z,
        bond_edges,
        include_bonds=bool(kw.get("include_backbone_bonds", True)),
        include_horizon=bool(kw.get("include_horizon", True)),
        include_clash=bool(kw.get("include_clash", True)),
        r_bond_min=float(kw.get("r_bond_min", R_BOND_MIN)),
        r_bond_max=float(kw.get("r_bond_max", R_BOND_MAX)),
        k_bond=float(kw.get("k_bond", K_BOND)),
        r_clash=float(kw.get("r_clash", R_CLASH)),
        k_clash=float(kw.get("k_clash", K_CLASH)),
        ca_atom_indices=ca_idx,
        ca_target=ca_tgt,
        k_ca_target=k_ca_t,
        **horizon_kw,
    )


def minimize_full_heavy_with_osh_oracle(
    pos_init: np.ndarray,
    z: np.ndarray,
    bond_edges: List[Tuple[int, int, float]],
    residue_ranges: List[Tuple[int, int]],
    ca_atom_indices: np.ndarray,
    *,
    z_shell: int = 6,
    n_iter: int = 120,
    step_size: float = 0.03,
    gate_mix: float = 0.55,
    ansatz_depth: int = 2,
    amp_threshold_quantile: float = 0.7,
    flip_amp_delta_eps: float = 1e-6,
    flip_include_sign: bool = True,
    use_harmonic_metropolis: bool = False,
    harmonic_fd_eps: float = 5e-3,
    harmonic_max_dims: int = 72,
    random_seed: Optional[int] = None,
    inertial_pk_weight: float = 0.0,
    inertial_k_potential: float = 1.0,
    inertial_k_kinetic: float = 1.0,
    inertial_velocity_decay: float = 0.9,
    use_energy_reservoir: bool = True,
    reservoir_init: float = 0.0,
    reservoir_gain_scale: float = 1.0,
    strict_descent_budget_mode: bool = True,
    schedule_period: int = 100,
    harmonic_step_anneal: bool = True,
    harmonic_base_temp: float = 1.0,
    harmonic_min_temp: float = 1e-4,
    stop_when_settled: bool = False,
    settle_window: int = 20,
    settle_energy_tol: float = 1e-3,
    settle_step_tol: float = 3e-4,
    settle_min_iter: int = 30,
    use_contact_reflectors: bool = False,
    contact_min_seq_sep: int = 4,
    contact_cutoff_ang: float = 8.0,
    contact_max_reflectors: int = 16,
    contact_grad_coupling: float = 1.0,
    contact_weight_gradient: bool = True,
    contact_score_mode: str = "hard_linear",
    contact_inverse_power: float = 2.0,
    contact_score_min_dist_ang: float = 1.0,
    contact_terminus_window: int = 0,
    contact_terminus_score_scale: float = 1.0,
    use_resonance_multiplier: bool = False,
    resonance_terminus_boost: float = 1.8,
    resonance_core_damping: float = 0.4,
    resonance_transition_width: int = 5,
    resonance_compaction_cutoff_ang: float = 8.0,
    resonance_compaction_min_seq_sep: int = 4,
    tunnel_budget_distance_score_mode: str = "linear",
    tunnel_budget_inverse_power: float = 2.0,
    tunnel_budget_distance_d0_ang: float = 1.0,
    use_end_bias_budget: bool = False,
    end_bias_scale: float = 2.0,
    end_bias_floor: float = 0.1,
    use_mode_shape_participation: bool = False,
    mode_shape_fixed_end: str = "right",
    mode_shape_factor_min: float = 0.5,
    mode_shape_factor_max: float = 1.2,
    omega_refresh_period: int = 0,
    use_terminus_gradient_boost: bool = False,
    terminus_gradient_boost: float = 1.28,
    terminus_gradient_transition_width: int = 8,
    terminus_gradient_core_scale: float = 1.0,
    backbone_projection_passes: int = 3,
    freeze_ca_positions: bool = False,
    limit_ca_displacement_ang: Optional[float] = None,
    energy_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, OSHOracleFoldInfo]:
    pos = np.asarray(pos_init, dtype=float).copy()
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("minimize_full_heavy_with_osh_oracle: pos_init must be (n_atom, 3).")
    n_atom = int(pos.shape[0])
    z_arr = np.asarray(z, dtype=np.int32).reshape(-1)
    if z_arr.shape[0] != n_atom:
        raise ValueError("minimize_full_heavy_with_osh_oracle: z length must match n_atom.")
    n_res = len(residue_ranges)
    if n_res < 2:
        info = OSHOracleFoldInfo(
            iterations=int(max(1, int(n_iter))),
            iterations_executed=0,
            accepted_steps=0,
            final_energy_ev=0.0,
            last_step_size=float(step_size),
            last_flipped_count=0,
            avg_flipped_count=0.0,
            natural_harmonic_scale=1.0,
            metropolis_accepts=0,
            stop_reason="too_short",
            settled=True,
            inertial_energy_final_ev=0.0,
            reservoir_energy_final_ev=float(reservoir_init),
            reservoir_uphill_accepts=0,
            tunnel_harmonic_budget_final_ev=0.0,
            contact_reflector_count=0,
            omega_refresh_count=0,
        )
        return pos, info

    ca_idx = np.asarray(ca_atom_indices, dtype=int).reshape(-1)
    if ca_idx.size != n_res:
        raise ValueError("ca_atom_indices must have one entry per residue.")
    edges = list(bond_edges)
    e_kw = dict(energy_kwargs or {})
    e_kw["ca_atom_indices"] = ca_idx

    ref_ca_frozen: Optional[np.ndarray] = None
    if bool(freeze_ca_positions):
        ct = e_kw.get("ca_target")
        kct = float(e_kw.get("k_ca_target", 0.0))
        if ct is not None and kct > 0.0:
            ref_ca_frozen = np.asarray(ct, dtype=float).reshape(n_res, 3).copy()
        else:
            ref_ca_frozen = pos[ca_idx].copy()
        pos[ca_idx] = ref_ca_frozen

    pairs: LigationPairs = []
    L = max(1, n_res - 1)
    step = float(step_size)
    mass_res = np.zeros(n_res, dtype=float)
    for ri, (a, b) in enumerate(residue_ranges):
        mass_res[ri] = float(np.mean(z_arr[int(a) : int(b)]))
    mass_res = np.maximum(1.0, mass_res)
    mass_atom = np.maximum(1.0, z_arr.astype(float))
    velocity = np.zeros_like(pos)
    prev_pos = pos.copy()
    reservoir = float(max(0.0, reservoir_init))
    reservoir_uphill_accepts = 0
    tunnel_budget_ev_last = 0.0
    omega_refresh_count = 0
    contact_reflector_count_last = 0
    accepted = 0
    metropolis_accepts = 0
    prev_reg: List[Tuple[int, float]] = []
    last_flipped_count = 0
    sum_flipped_count = 0.0
    settled = False
    stop_reason = "max_iter_reached"
    executed = 0
    recent_energy: List[float] = []
    recent_step: List[float] = []
    rng = np.random.default_rng(random_seed)
    ca = pos[ca_idx].copy()
    omega = estimate_natural_harmonic_scale_ca(
        ca,
        int(z_shell),
        energy_kwargs=e_kw,
        ligation_pairs=pairs,
        ligation_r_eq=3.8,
        ligation_r_min=2.5,
        ligation_r_max=6.0,
        ligation_k_bond=60.0,
        fd_eps=float(harmonic_fd_eps),
        max_dims=int(min(harmonic_max_dims, max(24, 3 * n_res))),
    )
    initial_energy_abs = float(abs(_energy_polymer(pos, z_arr, edges, dict(e_kw))))
    n_iter_eff = max(1, int(n_iter))
    sched_n = max(2, int(schedule_period))

    for it in range(n_iter_eff):
        executed = it + 1
        per_w = int(omega_refresh_period)
        if per_w > 0 and it > 0 and (it % per_w) == 0:
            ca = pos[ca_idx].copy()
            omega = estimate_natural_harmonic_scale_ca(
                ca,
                int(z_shell),
                energy_kwargs=e_kw,
                ligation_pairs=pairs,
                ligation_r_eq=3.8,
                ligation_r_min=2.5,
                ligation_r_max=6.0,
                ligation_k_bond=60.0,
                fd_eps=float(harmonic_fd_eps),
                max_dims=int(min(harmonic_max_dims, max(24, 3 * n_res))),
            )
            omega_refresh_count += 1
        e0_base = _energy_polymer(pos, z_arr, edges, dict(e_kw))
        e0_inertial = _inertial_pk_energy(
            pos,
            prev_pos,
            velocity,
            mass_atom,
            k_potential=float(inertial_k_potential),
            k_kinetic=float(inertial_k_kinetic),
        )
        e0 = float(e0_base + float(inertial_pk_weight) * e0_inertial)
        grad = _grad_polymer(pos, z_arr, edges, dict(e_kw))
        if ref_ca_frozen is not None:
            grad[ca_idx] = 0.0
        mags = np.zeros(n_res, dtype=float)
        for ri, (a, b) in enumerate(residue_ranges):
            mags[ri] = float(np.linalg.norm(grad[int(a) : int(b)]))
        ca = pos[ca_idx]
        cref: Optional[Set[int]] = None
        if bool(use_contact_reflectors):
            gm_pass = (
                np.asarray(mags, dtype=float)
                if bool(contact_weight_gradient) and float(contact_grad_coupling) > 0.0
                else None
            )
            cref = contact_reflector_indices(
                ca,
                gm_pass,
                min_seq_sep=int(contact_min_seq_sep),
                cutoff_ang=float(contact_cutoff_ang),
                max_reflectors=int(contact_max_reflectors),
                grad_coupling=float(contact_grad_coupling),
                score_mode=str(contact_score_mode),
                inverse_power=float(contact_inverse_power),
                score_min_dist_ang=float(contact_score_min_dist_ang),
                contact_terminus_window=int(contact_terminus_window),
                contact_terminus_score_scale=float(contact_terminus_score_scale),
            )
            contact_reflector_count_last = len(cref)
        q = float(np.clip(amp_threshold_quantile, 0.0, 1.0))
        thresh = float(np.quantile(mags, q)) if mags.size else 0.0
        reg = _register_variable_residue_blocks(
            grad, residue_ranges, L=L, amp_threshold=thresh
        )
        phase_it = int(it % sched_n)
        phi_mix, psi_mix = current_parameters(phase_it, sched_n, gate_mix)
        reg_after = apply_ansatz_sparse(
            L,
            reg,
            depth=int(ansatz_depth),
            phi_mix=float(phi_mix),
            psi_mix=float(psi_mix),
        )
        flipped = (
            detect_flipped_kets_amplitude(
                prev_reg,
                reg_after,
                amp_delta_eps=float(flip_amp_delta_eps),
                include_sign_flip=bool(flip_include_sign),
            )
            if prev_reg
            else [i for i, _ in reg_after]
        )
        if not flipped:
            flipped = [i for i, _ in reg_after]
        pruned = prune_to_flipped(flipped, reg_after)
        active_res: Set[int] = {int(i % n_res) for i, _ in pruned}
        if not active_res:
            active_res = {int(i % n_res) for i, _ in reg_after}
        active_idx = np.array(sorted(active_res), dtype=int)
        if active_idx.size == 0:
            break

        step_eff = float(step)
        base_temp_dyn = float(harmonic_base_temp)
        if bool(use_resonance_multiplier) and (bool(harmonic_step_anneal) or bool(use_harmonic_metropolis)):
            compaction = compute_local_compaction_score(
                ca,
                cutoff_ang=float(resonance_compaction_cutoff_ang),
                min_seq_sep=int(resonance_compaction_min_seq_sep),
            )
            resonance_mult = per_residue_resonance_multiplier(
                n_res,
                compaction,
                terminus_boost=float(resonance_terminus_boost),
                core_damping=float(resonance_core_damping),
                transition_width=int(resonance_transition_width),
            )
            resonance_active_mean = (
                float(np.mean(resonance_mult[active_idx])) if active_idx.size else 1.0
            )
            base_temp_dyn = float(harmonic_base_temp) * resonance_active_mean

        if bool(harmonic_step_anneal):
            tunnel_budget_ev = compute_tunnel_harmonic_budget_ev(
                ca,
                mass_res,
                pairs,
                active_idx=active_idx,
                contact_reflectors=cref,
                distance_score_mode=tunnel_budget_distance_score_mode,
                inverse_power=float(tunnel_budget_inverse_power),
                distance_d0_ang=float(tunnel_budget_distance_d0_ang),
                use_end_bias_budget=bool(use_end_bias_budget),
                end_bias_scale=float(end_bias_scale),
                end_bias_floor=float(end_bias_floor),
            )
            tunnel_budget_ev_last = float(tunnel_budget_ev)
            norm_budget = float(tunnel_budget_ev / max(1e-8, abs(initial_energy_abs)))
            g_all = float(np.mean(mags) + 1e-12)
            g_act = float(np.mean(mags[active_idx]) if active_idx.size else g_all)
            rigidity = max(1e-6, g_act / g_all)
            state_scale = float((float(base_temp_dyn) * norm_budget) / rigidity)
            mode_mean = 1.0
            if bool(use_mode_shape_participation):
                mode_factor = fixed_free_first_mode_factor(
                    ca,
                    fixed_end=str(mode_shape_fixed_end),
                    factor_min=float(mode_shape_factor_min),
                    factor_max=float(mode_shape_factor_max),
                )
                if active_idx.size:
                    mode_mean = float(np.mean(mode_factor[active_idx]))
            step_eff = float(step * np.sqrt(max(1e-12, state_scale) * mode_mean))

        term_scale = np.ones((n_res,), dtype=float)
        if bool(use_terminus_gradient_boost):
            term_scale = per_residue_terminus_step_scale(
                n_res,
                boost=float(terminus_gradient_boost),
                transition_width=int(terminus_gradient_transition_width),
                core_scale=float(terminus_gradient_core_scale),
            )

        cand = pos.copy()
        for ri in active_idx.tolist():
            a, b = residue_ranges[int(ri)]
            sl = slice(int(a), int(b))
            sc = float(term_scale[int(ri)])
            cand[sl] = pos[sl] - step_eff * sc * grad[sl]

        cand = project_polymer_covalent_bonds(
            cand,
            edges,
            r_min=R_BOND_MIN,
            r_max=R_BOND_MAX,
            passes=int(backbone_projection_passes),
        )
        if ref_ca_frozen is not None:
            cand[ca_idx] = ref_ca_frozen
        elif limit_ca_displacement_ang is not None:
            lim = float(limit_ca_displacement_ang)
            if lim > 0.0:
                ca_prev = pos[ca_idx]
                d = cand[ca_idx] - ca_prev
                m = np.linalg.norm(d, axis=1, keepdims=True)
                m_safe = np.maximum(m, 1e-12)
                scale = np.minimum(1.0, lim / m_safe)
                cand[ca_idx] = ca_prev + d * scale
        cand_velocity = float(np.clip(inertial_velocity_decay, 0.0, 1.0)) * velocity
        for ri in active_idx.tolist():
            a, b = residue_ranges[int(ri)]
            sl = slice(int(a), int(b))
            cand_velocity[sl] += cand[sl] - pos[sl]
        ek = dict(e_kw)
        e1_base = _energy_polymer(cand, z_arr, edges, ek)
        e1_inertial = _inertial_pk_energy(
            cand,
            pos,
            cand_velocity,
            mass_atom,
            k_potential=float(inertial_k_potential),
            k_kinetic=float(inertial_k_kinetic),
        )
        e1 = float(e1_base + float(inertial_pk_weight) * e1_inertial)
        delta_e = float(e1 - e0)
        accept = False
        if bool(strict_descent_budget_mode):
            if delta_e <= 0.0:
                accept = True
            elif bool(use_energy_reservoir):
                uphill_cost = float(max(0.0, delta_e))
                if uphill_cost <= reservoir:
                    accept = True
                    reservoir -= uphill_cost
                    reservoir_uphill_accepts += 1
        else:
            accept = bool(delta_e <= 0.0)
            if (not accept) and bool(use_harmonic_metropolis):
                accept = metropolis_accept_with_harmonic(
                    e0,
                    e1,
                    iteration=int(it),
                    n_iter=int(n_iter_eff),
                    omega=float(omega),
                    initial_energy_abs=float(initial_energy_abs),
                    base_temp=float(base_temp_dyn),
                    min_temp=float(harmonic_min_temp),
                    rng=rng,
                )
                if accept:
                    metropolis_accepts += 1
            if (not accept) and bool(use_energy_reservoir):
                uphill_cost = float(max(0.0, delta_e))
                if uphill_cost <= reservoir:
                    accept = True
                    reservoir -= uphill_cost
                    reservoir_uphill_accepts += 1
        if accept:
            drop = float(max(0.0, -delta_e))
            if bool(use_energy_reservoir) and drop > 0.0:
                reservoir += float(reservoir_gain_scale) * drop
            prev_pos = pos.copy()
            pos = cand
            velocity = cand_velocity
            accepted += 1
            step *= 1.02
        else:
            step *= 0.6
        step = float(np.clip(step, 1e-4, 0.12))
        prev_reg = reg_after
        last_flipped_count = len(flipped)
        sum_flipped_count += float(last_flipped_count)
        recent_energy.append(float(e1 if accept else e0))
        recent_step.append(float(step_eff))
        w = max(2, int(settle_window))
        if len(recent_energy) > w:
            recent_energy = recent_energy[-w:]
            recent_step = recent_step[-w:]
        if (
            bool(stop_when_settled)
            and executed >= int(max(1, settle_min_iter))
            and len(recent_energy) >= w
        ):
            e_span = float(max(recent_energy) - min(recent_energy))
            s_mean = float(np.mean(np.asarray(recent_step, dtype=float)))
            if e_span <= float(settle_energy_tol) and s_mean <= float(settle_step_tol):
                settled = True
                stop_reason = "settled"
                break

    e_final_base = _energy_polymer(pos, z_arr, edges, dict(e_kw))
    e_final_inertial = _inertial_pk_energy(
        pos,
        prev_pos,
        velocity,
        mass_atom,
        k_potential=float(inertial_k_potential),
        k_kinetic=float(inertial_k_kinetic),
    )
    e_final = float(e_final_base + float(inertial_pk_weight) * e_final_inertial)
    if (not settled) and executed < n_iter_eff:
        stop_reason = "early_break_no_active_support"
    info = OSHOracleFoldInfo(
        iterations=int(n_iter_eff),
        iterations_executed=int(executed),
        accepted_steps=int(accepted),
        final_energy_ev=e_final,
        last_step_size=float(step),
        last_flipped_count=int(last_flipped_count),
        avg_flipped_count=float(sum_flipped_count / float(max(1, executed))),
        natural_harmonic_scale=float(omega),
        metropolis_accepts=int(metropolis_accepts),
        stop_reason=str(stop_reason),
        settled=bool(settled),
        inertial_energy_final_ev=float(e_final_inertial),
        reservoir_energy_final_ev=float(reservoir),
        reservoir_uphill_accepts=int(reservoir_uphill_accepts),
        tunnel_harmonic_budget_final_ev=float(tunnel_budget_ev_last),
        contact_reflector_count=int(contact_reflector_count_last),
        omega_refresh_count=int(omega_refresh_count),
    )
    return pos, info
