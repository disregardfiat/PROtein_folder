"""
Force-carrier ensemble helpers.

This module implements the S2-inspired allowed translation envelope and
selection logic used to guide inner translation updates.

Design goals
- Reuse the same envelope structure as Lean `Hqiv/Physics/ForceCarrierWhip.lean`.
- Cheap selection: score candidates with a gradient-linearized proxy + whip
  penalty, then return the best displacement field.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def _envelope_order(p: float) -> int:
    """
    Lean: `envelopeOrder p := max 1 (Int.toNat ⌊p⌋)`.

    We clamp `p` to >= 0 so the exponent is well-defined as an integer power.
    """
    pp = max(0.0, float(p))
    return max(1, int(np.floor(pp)))


def _d_norm_clipped(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """
    Vectorized clipped normalized chain distance in [0,1]:
    `dNorm n i j = min 1 (|i-j| / max 1 (n-1))`.
    """
    if n <= 1:
        return np.zeros_like(i, dtype=float)
    d_raw = np.abs(i - j) / max(1.0, float(n - 1))
    return np.minimum(1.0, d_raw)


def carrier_amplitude(
    *,
    step: float,
    span: float,
    p: float,
    n: int,
    src: int,
    idxs: np.ndarray,
) -> np.ndarray:
    """
    Lean: `carrierAmplitude step span p n src j :=
      step * exp (-(dNorm n src j)/span) * s2Envelope p n src j`.
    """
    step = max(0.0, float(step))
    span = max(1e-9, float(span))
    p = max(0.0, float(p))

    envelope_order = _envelope_order(p)
    d_norm = _d_norm_clipped(n, idxs.astype(float), np.full_like(idxs, float(src), dtype=float))
    decay = np.exp(-(d_norm) / span)

    theta = (np.pi / 2.0) * (1.0 - d_norm)
    # `s2Envelope p = sin(theta) ^ envelopeOrder`
    sin_env = np.sin(theta) ** envelope_order
    sin_env = np.maximum(0.0, sin_env)
    return step * decay * sin_env


def amp_forward(
    *,
    step: float,
    span: float,
    p: float,
    n: int,
    src: int,
    idxs: np.ndarray,
) -> np.ndarray:
    """Lean `ampForward`."""
    return carrier_amplitude(step=step, span=span, p=p, n=n, src=src, idxs=idxs)


def amp_backward(
    *,
    step: float,
    span: float,
    p: float,
    beta: float,
    n: int,
    src: int,
    idxs: np.ndarray,
) -> np.ndarray:
    """Lean `ampBackward` (scaled by beta)."""
    b = float(beta)
    return b * carrier_amplitude(step=step, span=span, p=p, n=n, src=src, idxs=idxs)


def amp_net(
    *,
    step: float,
    span: float,
    p: float,
    beta: float,
    n: int,
    src: int,
    idxs: np.ndarray,
) -> np.ndarray:
    """Lean `ampNet = ampForward - ampBackward`."""
    aF = amp_forward(step=step, span=span, p=p, n=n, src=src, idxs=idxs)
    aB = amp_backward(step=step, span=span, p=p, beta=beta, n=n, src=src, idxs=idxs)
    return aF - aB


def whip_proxy_from_forward_back_scalar(
    dispF_scalar: float,
    dispB_scalar: float,
) -> float:
    """
    Lean `whipProxy dispF dispB = |dispF * dispB|`.
    """
    return float(abs(float(dispF_scalar) * float(dispB_scalar)))


def whip_proxy(
    *,
    step: float,
    span: float,
    p: float,
    beta: float,
    n: int,
    src: int,
) -> float:
    """
    Runtime proxy: map forward/back amplitude fields to the Lean scalar
    `dispF`, `dispB` via a simple mean-of-absolute values reduction.
    """
    idxs = np.arange(n, dtype=int)
    aF = amp_forward(step=step, span=span, p=p, n=n, src=src, idxs=idxs)
    aB = amp_backward(step=step, span=span, p=p, beta=beta, n=n, src=src, idxs=idxs)
    dispF_scalar = float(np.mean(np.abs(aF)))
    dispB_scalar = float(np.mean(np.abs(aB)))
    return whip_proxy_from_forward_back_scalar(dispF_scalar, dispB_scalar)


def build_direction_set_6_axes() -> np.ndarray:
    """(6,3) unit vectors: ±X, ±Y, ±Z."""
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )


def build_em_field_direction_set(
    *,
    positions: np.ndarray,
    grad: np.ndarray,
    base_direction_set: np.ndarray,
    max_extra_vectors: int = 24,
    min_vector_norm: float = 1e-9,
    colinearity_cosine: float = 0.985,
) -> np.ndarray:
    """
    Rebuild candidate translation directions from the current R3 field.

    For each bond (i,i+1), add bond-local EM and tangent vectors:
      em_i = -(grad[i] + grad[i+1]) / 2
      t_i  =  positions[i+1] - positions[i]
    Then merge with base directions after near-colinearity pruning.
    """
    pos = np.asarray(positions, dtype=float)
    g = np.asarray(grad, dtype=float)
    base = np.asarray(base_direction_set, dtype=float).reshape(-1, 3)
    if pos.ndim != 2 or pos.shape[1] != 3 or g.shape != pos.shape:
        return base
    n = int(pos.shape[0])
    if n < 2:
        return base

    candidates: list[np.ndarray] = []
    for i in range(n - 1):
        candidates.append(-0.5 * (g[i] + g[i + 1]))
        candidates.append(pos[i + 1] - pos[i])

    kept: list[np.ndarray] = []
    for v in base:
        nv = float(np.linalg.norm(v))
        if nv > min_vector_norm:
            kept.append(v / nv)
    extra: list[np.ndarray] = []
    cos_thr = float(colinearity_cosine)
    cap = max(0, int(max_extra_vectors))
    for v in candidates:
        nv = float(np.linalg.norm(v))
        if nv <= min_vector_norm:
            continue
        for u in (v / nv, -v / nv):
            if all(abs(float(np.dot(u, q))) < cos_thr for q in kept):
                kept.append(u)
                extra.append(u)
                if len(extra) >= cap:
                    break
        if len(extra) >= cap:
            break

    if not extra:
        return base
    return np.vstack([base, np.asarray(extra, dtype=float)])


def _bond_partition_masses(
    residue_masses: np.ndarray,
    *,
    left_anchor_infinite: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each bond i (between residues i and i+1), compute:
      - left mass  L_i = sum_{k<=i} m_k
      - right mass R_i = sum_{k>i}  m_k
    """
    m = np.asarray(residue_masses, dtype=float).reshape(-1)
    n = int(m.shape[0])
    if n <= 1:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    csum = np.cumsum(m)
    total = float(csum[-1])
    left = csum[:-1].copy()
    right = total - csum[:-1]
    if bool(left_anchor_infinite):
        left[:] = np.inf
    return left, right


def _reduced_mass_profile(
    residue_masses: np.ndarray,
    *,
    left_anchor_infinite: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert bond (L,R) partition masses to a per-residue effective inertial mass
    using reduced mass μ = (L*R)/(L+R), with μ=R when L=∞.
    """
    m = np.asarray(residue_masses, dtype=float).reshape(-1)
    n = int(m.shape[0])
    left_bond, right_bond = _bond_partition_masses(m, left_anchor_infinite=left_anchor_infinite)
    if n == 1:
        return np.asarray([max(1e-9, float(m[0]))], dtype=float), left_bond, right_bond
    mu_bond = np.empty_like(left_bond, dtype=float)
    inf_mask = np.isinf(left_bond)
    mu_bond[inf_mask] = right_bond[inf_mask]
    fin_mask = ~inf_mask
    mu_bond[fin_mask] = (left_bond[fin_mask] * right_bond[fin_mask]) / (
        left_bond[fin_mask] + right_bond[fin_mask] + 1e-12
    )
    mu_res = np.zeros((n,), dtype=float)
    mu_res[0] = float(mu_bond[0])
    mu_res[-1] = float(mu_bond[-1])
    if n > 2:
        mu_res[1:-1] = 0.5 * (mu_bond[:-1] + mu_bond[1:])
    mu_res = np.maximum(mu_res, 1e-9)
    return mu_res, left_bond, right_bond


def choose_best_translation_direction(
    *,
    grad: np.ndarray,
    positions: Optional[np.ndarray] = None,
    step: float,
    span: float,
    p: float,
    beta: float,
    score_lambda: float,
    direction_set: np.ndarray,
    sources: Sequence[int] = (0, -1),
    residue_masses: Optional[np.ndarray] = None,
    left_anchor_infinite: bool = False,
    linear_momentum_state: Optional[np.ndarray] = None,
    barrier_budget_state: Optional[np.ndarray] = None,
    inertial_dt: float = 1.0,
    linear_damping: float = 0.9,
    linear_gain: float = 1.0,
    damping_mode: str = "linear",
    barrier_decay: float = 0.95,
    barrier_build: float = 0.05,
    barrier_relief: float = 0.25,
    barrier_drive_gain: float = 0.0,
    barrier_floor: float = 0.0,
    barrier_trigger_offset: float = 1e-12,
    barrier_kick_gain: float = 0.0,
    directional_budget_base: float = 0.0,
    directional_budget_bend_coeff: float = 0.0,
    directional_budget_torsion_coeff: float = 0.0,
    directional_budget_conjugation_coeff: float = 0.0,
    directional_budget_bend_state: float = 0.0,
    directional_budget_torsion_state: float = 0.0,
    directional_budget_conjugation_state: float = 0.0,
    wave_leak_floor: float = 0.12,
    resonance_state: Optional[float] = None,
    resonance_decay: float = 0.92,
    resonance_gain: float = 0.35,
    resonance_harmonic_weight: float = 0.5,
    resonance_damping_span: float = 0.35,
    resonance_gain_boost: float = 0.5,
    omega_state: Optional[np.ndarray] = None,
    angular_mix: float = 0.0,
    angular_damping: float = 0.9,
    angular_gain: float = 0.2,
) -> Dict[str, object]:
    """
    Choose the best displacement field among a direction set and anchor sources.

    Scoring:
      objective_proxy(δ) = <grad, δ> - score_lambda * whipProxy
    where <grad, δ> is a linearized energy change.
    """
    grad = np.asarray(grad, dtype=float)
    if grad.ndim != 2 or grad.shape[1] != 3:
        raise ValueError("choose_best_translation_direction: grad must be (n,3).")
    n = int(grad.shape[0])
    idxs = np.arange(n, dtype=int)
    lam = float(score_lambda)
    use_torque_whip = positions is not None
    pos_mat = np.asarray(positions, dtype=float) if use_torque_whip else None
    if use_torque_whip:
        if pos_mat.shape != (n, 3):
            raise ValueError("choose_best_translation_direction: positions must be (n,3) matching grad.")

    # Bidirectional terminus transport:
    # Treat the `sources` list as simultaneously emitting forward/back amplitudes,
    # then superpose their resulting `ampNet` fields into a single displacement candidate.
    srcs_norm = [(n - 1) if int(s0) == -1 else int(s0) for s0 in sources]
    srcs_norm = [s for s in srcs_norm if 0 <= s < n]
    if not srcs_norm:
        srcs_norm = [0]

    # Precompute total forward/back amplitudes and their net.
    aF_total = np.zeros((n,), dtype=float)
    aB_total = np.zeros((n,), dtype=float)
    for src in srcs_norm:
        aF_total += amp_forward(step=step, span=span, p=p, n=n, src=src, idxs=idxs)
        aB_total += amp_backward(step=step, span=span, p=p, beta=beta, n=n, src=src, idxs=idxs)
    aNet_total = aF_total - aB_total

    ang_mix = max(0.0, float(angular_mix))
    dt = max(1e-9, float(inertial_dt))
    lin_damp = float(linear_damping)
    if str(damping_mode).strip().lower() == "sqrt":
        lin_damp_eff = np.sqrt(max(0.0, lin_damp))
    else:
        lin_damp_eff = lin_damp
    lin_gain = float(linear_gain)
    ang_damp = float(angular_damping)
    ang_gain = float(angular_gain)
    p0 = (
        np.asarray(linear_momentum_state, dtype=float).reshape(n, 3)
        if linear_momentum_state is not None
        else np.zeros((n, 3), dtype=float)
    )
    mass_vec = (
        np.asarray(residue_masses, dtype=float).reshape(n,)
        if residue_masses is not None
        else np.ones((n,), dtype=float)
    )
    mu_res, left_bond_mass, right_bond_mass = _reduced_mass_profile(
        mass_vec, left_anchor_infinite=bool(left_anchor_infinite)
    )
    b_floor = max(0.0, float(barrier_floor))
    b0 = (
        np.asarray(barrier_budget_state, dtype=float).reshape(n,)
        if barrier_budget_state is not None
        else np.zeros((n,), dtype=float)
    )
    if b_floor > 0.0:
        b0 = np.maximum(b0, b_floor)
    r0 = float(resonance_state) if resonance_state is not None else 0.0
    r0 = float(np.clip(r0, 0.0, 1.0))
    omega0 = np.asarray(omega_state, dtype=float).reshape(3,) if omega_state is not None else np.zeros(3, dtype=float)

    if not use_torque_whip:
        # Lean `whipProxy dispF dispB = |dispF * dispB|` with scalar reductions:
        # dispF := mean |ampForward| and dispB := mean |ampBackward|.
        dispF_scalar = float(np.mean(np.abs(aF_total)))
        dispB_scalar = float(np.mean(np.abs(aB_total)))
        w_total = whip_proxy_from_forward_back_scalar(dispF_scalar, dispB_scalar)
    else:
        # For torque whip we average the terminus-local torque proxy over all anchors.
        anchors = pos_mat[np.array(srcs_norm, dtype=int)]  # (m,3)
        # r_stack: (m,n,3) with r_stack[k,i,:] = pos[i,:] - anchor[k,:]
        r_stack = pos_mat[None, :, :] - anchors[:, None, :]
        # Mean geometry reference used for angular displacement contribution.
        r_mean = np.mean(r_stack, axis=0)  # (n,3)

    best_score = float("inf")
    best_disp = np.zeros_like(grad)
    best_src: int = int(srcs_norm[0])
    best_dir: np.ndarray = direction_set[0]
    best_whip = 0.0
    best_linear_momentum = p0.copy()
    best_barrier_budget = b0.copy()
    best_resonance_state = r0
    best_resonance_measure = 0.0
    best_omega = omega0.copy()

    for d in direction_set:
        d = np.asarray(d, dtype=float).reshape(3,)
        force_trans = aNet_total[:, None] * d[None, :]
        drive = np.linalg.norm(force_trans, axis=1) / (mu_res + 1e-12)
        trig_off = max(0.0, float(barrier_trigger_offset))
        # Lean ProteinNaturalFolding directional budget:
        # budgetEff = base + bendCoeff*bendState + torsionCoeff*torsionState + conjugationCoeff*conjugationState
        # triggerDir = drive / (drive + budgetEff + 1)
        dir_budget = (
            float(directional_budget_base)
            + float(directional_budget_bend_coeff) * float(directional_budget_bend_state)
            + float(directional_budget_torsion_coeff) * float(directional_budget_torsion_state)
            + float(directional_budget_conjugation_coeff) * float(directional_budget_conjugation_state)
        )
        dir_budget = max(0.0, dir_budget)
        trigger = drive / (drive + b0 + dir_budget + trig_off)
        # Ensure some transport always propagates through the chain.
        # This avoids complete local shutoff and lets small updates accumulate.
        leak = float(np.clip(float(wave_leak_floor), 0.0, 1.0))
        trigger_eff = leak + (1.0 - leak) * trigger
        accel_eff = (force_trans / (mu_res[:, None] + 1e-12)) * trigger_eff[:, None]
        overdrive = np.maximum(0.0, drive - b0)
        # Lean ProteinNaturalFolding inertial/barrier update adds a deterministic
        # post-threshold kick: p_next += kickGain * overdrive.
        kick_eff = float(max(0.0, barrier_kick_gain))
        kick_term = kick_eff * overdrive[:, None]
        if str(damping_mode).strip().lower() == "resonant":
            p_norm = np.linalg.norm(p0, axis=1)
            a_norm = np.linalg.norm(accel_eff, axis=1)
            mask = (p_norm > 1e-12) & (a_norm > 1e-12)
            if np.any(mask):
                phase_raw = np.mean(
                    np.sum(p0[mask] * accel_eff[mask], axis=1) / (p_norm[mask] * a_norm[mask] + 1e-12)
                )
            else:
                phase_raw = 0.0
            phase_lock = 0.5 * (float(np.clip(phase_raw, -1.0, 1.0)) + 1.0)
            proj = np.asarray(np.dot(p0, d), dtype=float).reshape(-1)
            proj = proj - float(np.mean(proj))
            if proj.size >= 4:
                spec = np.abs(np.fft.rfft(proj))[1:] ** 2
                harmonic_ratio = float(np.max(spec) / (np.sum(spec) + 1e-12)) if spec.size > 0 else 0.0
            else:
                harmonic_ratio = 0.0
            h_w = float(np.clip(float(resonance_harmonic_weight), 0.0, 1.0))
            resonance_measure = (1.0 - h_w) * phase_lock + h_w * harmonic_ratio
            r_next = float(
                np.clip(float(resonance_decay) * r0 + float(resonance_gain) * resonance_measure, 0.0, 1.0)
            )
            damp_local = float(np.clip(lin_damp_eff * (1.0 - float(resonance_damping_span) * r_next), 0.0, 1.5))
            gain_local = float(max(0.0, lin_gain * (1.0 + float(resonance_gain_boost) * r_next)))
            p_next = damp_local * p0 + gain_local * accel_eff + kick_term
        else:
            resonance_measure = 0.0
            r_next = r0
            p_next = lin_damp_eff * p0 + lin_gain * accel_eff + kick_term
        disp_trans = dt * p_next
        disp = disp_trans
        if use_torque_whip and ang_mix > 0.0:
            # Rotational carrier contribution: convert current angular state to
            # translational displacement around the local anchor geometry.
            disp_rot = dt * np.cross(np.broadcast_to(omega0, r_mean.shape), r_mean)
            disp = disp_trans + ang_mix * disp_rot
        lin = float(np.sum(grad * disp))

        if use_torque_whip:
            tau = np.cross(r_stack, accel_eff[None, :, :])  # (m,n,3)
            # Mean-normalized "whip" per anchor:
            #   whip_k = sum_i ||cross(r_k[i], disp[i])|| / n
            whip_per_anchor = np.sum(np.linalg.norm(tau, axis=2), axis=1) / max(1.0, float(n))
            whip = float(np.mean(whip_per_anchor))
            score = lin - lam * whip
            # Update angular carrier state from mean torque over anchors.
            tau_vec = np.mean(np.sum(tau, axis=1), axis=0) / max(1.0, float(n))
            omega_next = ang_damp * omega0 + ang_gain * tau_vec
        else:
            score = lin - lam * float(w_total)
            whip = float(w_total)
            omega_next = omega0
        underdrive = np.maximum(0.0, b0 - drive)
        b_next = np.maximum(
            b_floor,
            float(barrier_decay) * b0
            + float(barrier_build) * underdrive
            + float(barrier_drive_gain) * drive
            - float(barrier_relief) * overdrive,
        )

        if score < best_score:
            best_score = score
            best_disp = disp
            best_dir = d
            best_whip = float(whip)
            best_linear_momentum = np.asarray(p_next, dtype=float)
            best_barrier_budget = np.asarray(b_next, dtype=float).reshape(n,)
            best_resonance_state = float(r_next)
            best_resonance_measure = float(resonance_measure)
            best_omega = np.asarray(omega_next, dtype=float).reshape(3,)

    return {
        "best_score": float(best_score),
        "best_disp": best_disp,
        "best_source_idx": int(best_src),
        "best_direction": best_dir,
        "best_whip_proxy": float(best_whip),
        "best_linear_momentum_state": best_linear_momentum,
        "best_barrier_budget_state": best_barrier_budget,
        "best_resonance_state": float(best_resonance_state),
        "best_resonance_measure": float(best_resonance_measure),
        "best_omega_state": best_omega,
        "mass_profile": mu_res,
        "bond_left_mass": left_bond_mass,
        "bond_right_mass": right_bond_mass,
    }


def maybe_refresh_em_field_direction_set(
    pos_before: np.ndarray,
    pos_after: np.ndarray,
    grad: np.ndarray,
    base_direction_set: np.ndarray,
    direction_set_active: np.ndarray,
    disp_candidate: Optional[np.ndarray],
    *,
    refresh_on_horizon_crossing: bool,
    refresh_on_horizon_leaving: bool = False,
    horizon_ang: float,
    min_seq_sep: int,
    refresh_on_large_disp: bool,
    large_disp_thresh: float,
    max_extra_vectors: int,
) -> Tuple[np.ndarray, int, int, bool]:
    """
    Optionally rebuild EM-augmented translation directions when either:

    - a nonlocal Cα pair **enters** the horizon radius (default 15 Å),
    - (optional) a pair **leaves** that radius (coupling drops off), or
    - (optional) the raw carrier displacement exceeds ``large_disp_thresh``.

    Returns ``(direction_set_active, n_pairs_entering, n_pairs_leaving, did_refresh)``.
    """
    from .folding_energy import (
        count_nonlocal_pairs_entering_horizon,
        count_nonlocal_pairs_leaving_horizon,
    )

    n_enter = 0
    if bool(refresh_on_horizon_crossing):
        n_enter = int(
            count_nonlocal_pairs_entering_horizon(
                pos_before,
                pos_after,
                r_horizon=float(horizon_ang),
                min_seq_sep=int(min_seq_sep),
            )
        )
    n_leave = 0
    if bool(refresh_on_horizon_leaving):
        n_leave = int(
            count_nonlocal_pairs_leaving_horizon(
                pos_before,
                pos_after,
                r_horizon=float(horizon_ang),
                min_seq_sep=int(min_seq_sep),
            )
        )
    large = False
    if bool(refresh_on_large_disp) and disp_candidate is not None:
        dc = np.asarray(disp_candidate, dtype=float)
        if dc.ndim == 1 and int(dc.size) % 3 == 0 and dc.size > 0:
            dc = dc.reshape(-1, 3)
        if dc.ndim == 2 and dc.shape[1] == 3 and dc.shape[0] > 0:
            large = bool(
                float(np.max(np.linalg.norm(dc, axis=1))) >= float(large_disp_thresh) - 1e-15
            )
    refresh = (
        (bool(refresh_on_horizon_crossing) and n_enter > 0)
        or (bool(refresh_on_horizon_leaving) and n_leave > 0)
        or large
    )
    if not refresh:
        return np.asarray(direction_set_active, dtype=float), n_enter, n_leave, False
    new_ds = build_em_field_direction_set(
        positions=np.asarray(pos_after, dtype=float),
        grad=np.asarray(grad, dtype=float),
        base_direction_set=np.asarray(base_direction_set, dtype=float),
        max_extra_vectors=int(max_extra_vectors),
    )
    return new_ds, n_enter, n_leave, True

