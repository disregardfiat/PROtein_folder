"""
OSHoracle-inspired sparse-support folding stage.

This module mirrors the Lean digital sparse pattern from
`HQIV_LEAN/Hqiv/QuantumComputing/OSHoracle.lean`:

1) causal expansion of sparse support (i -> i and i+1),
2) dense reconstruction and gate-like amplitude evolution,
3) changed-support detection (`detectFlippedKets`),
4) prune/update only the changed support.

Those steps are a VQC-style sparse register on an abstract index set; they do not
inherently require Cα-only coordinates. The concrete minimizer
``minimize_ca_with_osh_oracle`` is wired to one residue per site (Cα trace),
``grad_full`` / ``e_tot_ca_with_bonds``, Cα bond projection, and residue-indexed
contacts. A multi-atom model would reuse the same sparse primitives with a
different state layout (e.g. flat atom list or residue × atom blocks), energy
and gradient callables, and geometry projection appropriate to that graph.

**HQIV-native digital gate** (Lean ``Hqiv/QuantumComputing/OSHoracleHQIVNative.lean`` /
``ProteinFoldingHook.lean``): optional ``apply_gate_sparse_hqiv_native`` uses
``hqivPivotFromShells`` = ``(Σ shells + referenceM) % (L+1)``, ``HarmonicIndex`` slot
``ℓ = pivot mod (L+1)``, ``m = 0``, and a π phase (sign flip) on that mode — same
sparse pipeline cost as ``applyGateSparse`` in Lean.

Heavy-atom backbone (N, CA, C, O) with the same sparse register logic:
``horizon_physics.proteins.osh_oracle_backbone.minimize_backbone_with_osh_oracle``.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .folding_energy import (
    count_nonlocal_pairs_entering_horizon,
    e_tot_ca_with_bonds,
    grad_full,
)
from .gradient_descent_folding import _project_bonds
from .horizon_qed_bookkeeping import phi_of_shell, shell_spatial_mode_count

SparseRegister = List[Tuple[int, float]]
LigationPairs = List[Tuple[int, int]]

# ``grad_full`` accepts horizon / H-bond / FD extras (e.g. ``em_scale``); ``e_tot_ca_with_bonds`` does not.
_E_TOT_CA_KWARG_NAMES = frozenset(
    p
    for p in inspect.signature(e_tot_ca_with_bonds).parameters
    if p not in ("positions", "z_list")
)


def _e_tot_ca_kwargs(e_kw: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in e_kw.items() if k in _E_TOT_CA_KWARG_NAMES}


def sparse_basis_card(L: int) -> int:
    """Lean: sparseBasisCard L = (L + 1)^2."""
    return int((int(L) + 1) ** 2)


def wrap_idx(L: int, i: int) -> int:
    """Lean: wrapIdx (L i) = i % sparseBasisCard L."""
    card = sparse_basis_card(L)
    if card <= 0:
        return 0
    return int(i) % card


def causal_expand_support(L: int, reg: SparseRegister) -> SparseRegister:
    """Lean-style one-step causal support expansion (keep i and i+1)."""
    out: SparseRegister = []
    for idx, amp in reg:
        i = wrap_idx(L, idx)
        j = wrap_idx(L, idx + 1)
        out.append((i, float(amp)))
        out.append((j, float(amp)))
    return out


def dense_of_sparse(L: int, reg: SparseRegister) -> np.ndarray:
    """Dense reconstruction by summing amplitudes on colliding wrapped indices."""
    card = sparse_basis_card(L)
    dense = np.zeros(card, dtype=float)
    for idx, amp in reg:
        dense[wrap_idx(L, idx)] += float(amp)
    return dense


def _gate_evolve_dense(dense: np.ndarray, gate_mix: float) -> np.ndarray:
    """
    A deterministic, local gate surrogate:
    evolved[i] = (1-g)*dense[i] + 0.5*g*(dense[i-1] + dense[i+1]).
    """
    g = float(np.clip(gate_mix, 0.0, 1.0))
    if dense.size == 0:
        return dense
    left = np.roll(dense, 1)
    right = np.roll(dense, -1)
    return (1.0 - g) * dense + 0.5 * g * (left + right)


def apply_gate_sparse(L: int, reg: SparseRegister, gate_mix: float = 0.5) -> SparseRegister:
    """Sparse gate action: expand support then pull amplitudes from evolved dense state."""
    expanded = causal_expand_support(L, reg)
    evolved = _gate_evolve_dense(dense_of_sparse(L, expanded), gate_mix=gate_mix)
    return [(idx, float(evolved[wrap_idx(L, idx)])) for idx, _ in expanded]


def apply_ansatz_sparse(
    L: int,
    reg: SparseRegister,
    *,
    depth: int,
    phi_mix: float,
    psi_mix: float,
) -> SparseRegister:
    """
    VQE-like ansatz stack: alternating φ and ψ-style sparse gate layers.
    """
    out = list(reg)
    d = max(1, int(depth))
    for k in range(d):
        mix = float(phi_mix if (k % 2 == 0) else psi_mix)
        out = apply_gate_sparse(L, out, gate_mix=mix)
    return out


# Lean ``OctonionicLightCone.referenceM`` (proton anchor); must match HQIV-Lean proofs.
REFERENCE_M_HQIV_NATIVE = 4


def hqiv_pivot_from_shells(
    shells: np.ndarray,
    L: int,
    *,
    reference_m: int = REFERENCE_M_HQIV_NATIVE,
) -> int:
    """
    Lean ``hqivPivotFromShells``: ``(shells.sum + referenceM) % (L + 1)``.
    Requires ``len(shells) == L`` (API guard matching ``shells.length = L``).
    """
    sh = np.asarray(shells, dtype=np.int64).reshape(-1)
    if int(sh.shape[0]) != int(L):
        raise ValueError(f"hqiv_pivot_from_shells: len(shells)={sh.shape[0]} != L={L}")
    s = int(np.sum(sh))
    return int((s + int(reference_m)) % (int(L) + 1))


def hqiv_harmonic_flat_index_ell_m0(L: int, pivot: int) -> int:
    """
    Lean ``hqivHarmonicPivot`` with ``m = 0``: ``ℓ = pivot % (L+1)``, linear index ``ℓ² + m``.
    Cumulative mode offset for shell ``ℓ`` is ``ℓ²`` on the ``(L+1)²`` harmonic ladder.
    """
    ell = int(pivot) % (int(L) + 1)
    return int(ell * ell)


def _hqiv_phase_negate_one_mode(dense: np.ndarray, flat_idx: int) -> np.ndarray:
    """Lean ``phaseGate``: negate amplitude on one ``HarmonicIndex`` slot (π phase on reals)."""
    out = np.asarray(dense, dtype=float).copy()
    fi = int(flat_idx)
    if 0 <= fi < int(out.size):
        out[fi] = -float(out[fi])
    return out


def apply_gate_sparse_hqiv_native(
    L: int,
    reg: SparseRegister,
    shells: np.ndarray,
    *,
    reference_m: int = REFERENCE_M_HQIV_NATIVE,
) -> SparseRegister:
    """
    Lean ``hqivNativeOracleSparseStep``: ``applyGateSparse`` with ``hqivNativePhaseGate``.

    Same pipeline as ``apply_gate_sparse`` (causal expand → dense → map) but gate is a
    π phase on the pivot mode from ``hqiv_pivot_from_shells``; preserves ``Σᵢ aᵢ²`` on reals.
    """
    pivot = hqiv_pivot_from_shells(shells, L, reference_m=reference_m)
    flat_idx = hqiv_harmonic_flat_index_ell_m0(L, pivot)
    expanded = causal_expand_support(L, reg)
    dense = dense_of_sparse(L, expanded)
    evolved = _hqiv_phase_negate_one_mode(dense, flat_idx)
    return [(idx, float(evolved[wrap_idx(L, idx)])) for idx, _ in expanded]


def apply_ansatz_sparse_hqiv_native(
    L: int,
    reg: SparseRegister,
    *,
    depth: int,
    shells: np.ndarray,
    reference_m: int = REFERENCE_M_HQIV_NATIVE,
) -> SparseRegister:
    """Stack ``depth`` native sparse gates (identical gate each layer — deterministic)."""
    out = list(reg)
    d = max(1, int(depth))
    for _ in range(d):
        out = apply_gate_sparse_hqiv_native(L, out, shells, reference_m=reference_m)
    return out


def current_parameters(iteration: int, n_iter: int, base_gate_mix: float) -> Tuple[float, float]:
    """
    Iteration-dependent ansatz parameters analogous to `currentParameters state i`.
    """
    t = 0.0 if n_iter <= 1 else float(iteration) / float(max(1, n_iter - 1))
    osc = 0.5 + 0.5 * np.cos(np.pi * t)
    phi_mix = float(np.clip(base_gate_mix * (0.85 + 0.30 * osc), 0.0, 1.0))
    psi_mix = float(np.clip(base_gate_mix * (0.80 + 0.40 * (1.0 - osc)), 0.0, 1.0))
    return phi_mix, psi_mix


def detect_flipped_kets(before: SparseRegister, after: SparseRegister) -> List[int]:
    """Lean analog: indices that changed support between sparse snapshots."""
    b_idx = {int(i) for i, _ in before}
    a_idx = {int(i) for i, _ in after}
    return sorted((b_idx - a_idx) | (a_idx - b_idx))


def detect_flipped_kets_amplitude(
    before: SparseRegister,
    after: SparseRegister,
    *,
    amp_delta_eps: float = 1e-6,
    include_sign_flip: bool = True,
) -> List[int]:
    """
    Enhanced flipped-ket detection: support changes + sign/amplitude flips.
    """
    base = set(detect_flipped_kets(before, after))
    b_map: Dict[int, float] = {}
    a_map: Dict[int, float] = {}
    for i, a in before:
        ii = int(i)
        b_map[ii] = b_map.get(ii, 0.0) + float(a)
    for i, a in after:
        ii = int(i)
        a_map[ii] = a_map.get(ii, 0.0) + float(a)
    for i in set(b_map.keys()) & set(a_map.keys()):
        b = float(b_map[i])
        a = float(a_map[i])
        if include_sign_flip and (b == 0.0) != (a == 0.0):
            base.add(i)
            continue
        if include_sign_flip and b != 0.0 and a != 0.0 and (b > 0) != (a > 0):
            base.add(i)
            continue
        if abs(a - b) >= float(amp_delta_eps):
            base.add(i)
    return sorted(base)


def prune_to_flipped(flipped: Iterable[int], reg: SparseRegister) -> SparseRegister:
    """
    Keep only sparse entries whose index is in ``flipped``.

    In steady state, ``flipped`` is usually far smaller than the expanded register,
    so few residue indices get a coordinate update each iteration. The energy and
    ``grad_full`` are still evaluated on the full chain every step.
    """
    keep = {int(i) for i in flipped}
    return [(i, a) for i, a in reg if int(i) in keep]


def _register_from_gradients(
    grad: np.ndarray,
    *,
    L: int,
    amp_threshold: float,
) -> SparseRegister:
    """
    Build sparse support from gradient magnitudes:
    register entry for residue i if ||grad_i|| >= threshold.
    """
    mags = np.linalg.norm(np.asarray(grad, dtype=float), axis=1)
    reg: SparseRegister = []
    for i, m in enumerate(mags):
        if float(m) >= float(amp_threshold):
            reg.append((wrap_idx(L, i), float(m)))
    return reg


def _normalize_ligation_pairs(
    pairs: Optional[Iterable[Tuple[int, int]]],
    n_res: int,
) -> LigationPairs:
    out: LigationPairs = []
    seen: Set[Tuple[int, int]] = set()
    if not pairs:
        return out
    for i, j in pairs:
        a = int(i)
        b = int(j)
        if a == b:
            continue
        if a < 0 or b < 0 or a >= n_res or b >= n_res:
            continue
        x, y = (a, b) if a < b else (b, a)
        if (x, y) in seen:
            continue
        seen.add((x, y))
        out.append((x, y))
    return out


def auto_detect_cys_ligation_pairs(
    sequence: str,
    ca: np.ndarray,
    *,
    min_seq_sep: int = 2,
    max_dist_ang: float = 6.5,
) -> LigationPairs:
    """
    Auto-detect candidate ligation pairs from cysteines that are spatially close.
    """
    seq = "".join(c for c in str(sequence).strip().upper() if c.isalpha())
    pos = np.asarray(ca, dtype=float)
    n = min(len(seq), int(pos.shape[0]))
    cys = [i for i in range(n) if seq[i] == "C"]
    out: LigationPairs = []
    for a in range(len(cys)):
        i = cys[a]
        for b in range(a + 1, len(cys)):
            j = cys[b]
            if abs(j - i) < int(min_seq_sep):
                continue
            d = float(np.linalg.norm(pos[j] - pos[i]))
            if d <= float(max_dist_ang):
                out.append((i, j))
    return out


def _project_extra_bonds(
    positions: np.ndarray,
    ligation_pairs: LigationPairs,
    *,
    r_min: float,
    r_max: float,
    passes: int = 2,
) -> np.ndarray:
    """
    Soft projection for additional (non-sequential) bond constraints.
    """
    if not ligation_pairs:
        return np.asarray(positions, dtype=float)
    pos = np.asarray(positions, dtype=float).copy()
    for _ in range(max(1, int(passes))):
        for i, j in ligation_pairs:
            d = pos[j] - pos[i]
            r = float(np.linalg.norm(d))
            if r < 1e-12:
                continue
            if r > float(r_max):
                tgt = float(r_max)
            elif r < float(r_min):
                tgt = float(r_min)
            else:
                continue
            u = d / r
            delta = 0.5 * (r - tgt) * u
            pos[i] += delta
            pos[j] -= delta
    return pos


def _extra_bond_energy(
    positions: np.ndarray,
    ligation_pairs: LigationPairs,
    *,
    r_eq: float = 3.8,
    r_min: float = 2.5,
    r_max: float = 6.0,
    k_bond: float = 60.0,
) -> float:
    if not ligation_pairs:
        return 0.0
    pos = np.asarray(positions, dtype=float)
    e = 0.0
    for i, j in ligation_pairs:
        r = float(np.linalg.norm(pos[j] - pos[i]))
        if r < 1e-12:
            continue
        if r < float(r_min):
            e += float(k_bond) * (float(r_min) - r) ** 2
        elif r > float(r_max):
            e += float(k_bond) * (r - float(r_max)) ** 2
        else:
            e += 0.1 * float(k_bond) * (r - float(r_eq)) ** 2
    return float(e)


def _energy_with_ligation(
    positions: np.ndarray,
    z: np.ndarray,
    e_kw: Dict[str, Any],
    ligation_pairs: LigationPairs,
    *,
    ligation_r_eq: float,
    ligation_r_min: float,
    ligation_r_max: float,
    ligation_k_bond: float,
) -> float:
    base = float(e_tot_ca_with_bonds(positions, z, **_e_tot_ca_kwargs(e_kw)))
    if not ligation_pairs:
        return base
    return float(
        base
        + _extra_bond_energy(
            positions,
            ligation_pairs,
            r_eq=float(ligation_r_eq),
            r_min=float(ligation_r_min),
            r_max=float(ligation_r_max),
            k_bond=float(ligation_k_bond),
        )
    )


def _local_rapidity_displacement(
    ca: np.ndarray,
    grad: np.ndarray,
    active_idx: np.ndarray,
    *,
    gain: float = 0.25,
    tangent_weight: float = 0.7,
    normal_weight: float = 0.3,
) -> np.ndarray:
    """
    Local-first Cartesian rapidity-like translation.

    For each active residue, build a local frame from chain geometry and split gradient
    into tangent/normal channels. A signed rapidity scale (tanh of local curvature proxy)
    modulates how much tangent vs normal motion is injected.
    """
    pos = np.asarray(ca, dtype=float)
    g = np.asarray(grad, dtype=float)
    n = int(pos.shape[0])
    out = np.zeros_like(pos)
    if n < 3 or active_idx.size == 0:
        return out
    tw = float(tangent_weight)
    nw = float(normal_weight)
    gn = float(max(0.0, gain))
    for i in active_idx.tolist():
        if i <= 0 or i >= n - 1:
            continue
        t = pos[i + 1] - pos[i - 1]
        nt = float(np.linalg.norm(t))
        if nt < 1e-12:
            continue
        t = t / nt
        # Normal proxy from local turning.
        d1 = pos[i] - pos[i - 1]
        d2 = pos[i + 1] - pos[i]
        nvec = d2 - d1
        nn = float(np.linalg.norm(nvec))
        if nn < 1e-12:
            # fallback orthogonal direction from gradient
            gv = g[i] - float(np.dot(g[i], t)) * t
            nn = float(np.linalg.norm(gv))
            if nn < 1e-12:
                continue
            nvec = gv / nn
        else:
            nvec = nvec / nn
        # Signed local rapidity from tangent-aligned gradient component.
        g_t = float(np.dot(g[i], t))
        g_n = g[i] - g_t * t
        eta = float(np.tanh(g_t / (float(np.linalg.norm(g[i])) + 1e-12)))
        disp = -gn * (tw * eta * g_t * t + nw * (1.0 - abs(eta)) * g_n)
        out[i] = disp
    return out


@dataclass
class OSHOracleFoldInfo:
    iterations: int
    iterations_executed: int
    accepted_steps: int
    final_energy_ev: float
    last_step_size: float
    last_flipped_count: int
    avg_flipped_count: float
    natural_harmonic_scale: float
    metropolis_accepts: int
    stop_reason: str
    settled: bool
    inertial_energy_final_ev: float
    reservoir_energy_final_ev: float
    reservoir_uphill_accepts: int
    tunnel_harmonic_budget_final_ev: float
    contact_reflector_count: int = 0
    omega_refresh_count: int = 0


@dataclass
class QAOAHarmonicFoldInfo:
    layers: int
    accepted_layers: int
    final_energy_ev: float
    final_visible_energy_ev: float
    natural_harmonic_scale: float
    avg_flipped_count: float
    qpe_targets_last: int
    final_step_size: float
    inertial_energy_final_ev: float
    reservoir_energy_final_ev: float
    reservoir_uphill_accepts: int
    tunnel_harmonic_budget_final_ev: float


@dataclass
class AdditiveKickInfo:
    step: int
    applied: bool
    torque_updated: bool
    trigger_reason: str
    kick_norm_mean: float
    kick_norm_max: float
    additive_field_trace_ev: float
    torque_trace_ev: float


@dataclass
class OSHAdditiveCycleInfo:
    cycle_index: int
    reservoir_before_ev: float
    reservoir_after_osh_ev: float
    reservoir_draw_ev: float
    horizon_pairs_entered_count: int
    osh_displacement_mean_ang: float
    osh_displacement_max_ang: float
    osh_energy_delta_ev: float
    osh_info: OSHOracleFoldInfo
    additive_kick: AdditiveKickInfo


def _lattice_full_mode_energy(shell: int) -> float:
    """Lean bridge: `available_modes * (phi_of_shell / 2)`."""
    m = int(max(0, shell))
    return float(shell_spatial_mode_count(m) * (phi_of_shell(m) / 2.0))


def _additive_field_and_torque_kick(
    ca: np.ndarray,
    *,
    shell: int,
    kick_gain: float = 0.004,
    torque_mix: float = 0.3,
    kick_max_norm_ang: float = 0.002,
    cached_torque_diag: Optional[np.ndarray] = None,
    update_torque: bool = True,
) -> Tuple[np.ndarray, np.ndarray, AdditiveKickInfo]:
    """
    Additive mean-field + infrequent torque kick between OSH cycles.

    - Additive field uses a linear sum over all sites with Manhattan-radius proxy.
    - Torque term follows Lean's diagonal proxy `orientationDev * field`.
    """
    pos = np.asarray(ca, dtype=float)
    n = int(pos.shape[0])
    if n < 2:
        z = np.zeros_like(pos)
        info = AdditiveKickInfo(
            step=0,
            applied=False,
            torque_updated=bool(update_torque),
            trigger_reason="too_short",
            kick_norm_mean=0.0,
            kick_norm_max=0.0,
            additive_field_trace_ev=0.0,
            torque_trace_ev=0.0,
        )
        return z, np.zeros((n,), dtype=float), info
    e_shell = _lattice_full_mode_energy(int(shell))
    f_scalar = np.zeros((n,), dtype=float)
    f_vec = np.zeros_like(pos)
    for i in range(n):
        acc = np.zeros((3,), dtype=float)
        s = 0.0
        for j in range(n):
            if j == i:
                continue
            d = pos[i] - pos[j]
            r_proxy = float(np.abs(d[0]) + np.abs(d[1]) + np.abs(d[2]) + 1.0)
            w = float(e_shell / max(1e-8, r_proxy))
            s += w
            nd = float(np.linalg.norm(d))
            if nd > 1e-12:
                acc += w * d / nd
        f_scalar[i] = s
        f_vec[i] = acc
    # Local orientation deviation: |pi - bend angle| on Cα triples.
    orient = np.zeros((n,), dtype=float)
    for i in range(1, n - 1):
        u = pos[i - 1] - pos[i]
        v = pos[i + 1] - pos[i]
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu < 1e-12 or nv < 1e-12:
            continue
        c = float(np.clip(np.dot(u / nu, v / nv), -1.0, 1.0))
        th = float(np.arccos(c))
        orient[i] = abs(np.pi - th)
    if update_torque or cached_torque_diag is None or int(cached_torque_diag.size) != n:
        torque_diag = orient * f_scalar
        torque_updated = True
    else:
        torque_diag = np.asarray(cached_torque_diag, dtype=float).reshape(-1)
        torque_updated = False
    kick = np.zeros_like(pos)
    for i in range(1, n - 1):
        t = pos[i + 1] - pos[i - 1]
        nt = float(np.linalg.norm(t))
        if nt < 1e-12:
            continue
        t = t / nt
        fv = f_vec[i]
        nf = float(np.linalg.norm(fv))
        if nf < 1e-12:
            continue
        fv = fv / nf
        torque_vec = np.cross(t, fv)
        kt = float(np.clip(torque_mix, 0.0, 1.0))
        direction = (1.0 - kt) * fv + kt * float(np.tanh(torque_diag[i])) * torque_vec
        nd = float(np.linalg.norm(direction))
        if nd < 1e-12:
            continue
        kick[i] = -float(max(0.0, kick_gain)) * direction / nd
    # Global safety: clamp per-residue kick norm to avoid large unstable displacements.
    kmax = float(max(0.0, kick_max_norm_ang))
    if kmax > 0.0:
        kn = np.linalg.norm(kick, axis=1)
        for i in range(n):
            if kn[i] > kmax:
                kick[i] *= kmax / max(1e-12, float(kn[i]))
    info = AdditiveKickInfo(
        step=0,
        applied=True,
        torque_updated=bool(torque_updated),
        trigger_reason="applied",
        kick_norm_mean=float(np.mean(np.linalg.norm(kick, axis=1))),
        kick_norm_max=float(np.max(np.linalg.norm(kick, axis=1))),
        additive_field_trace_ev=float(np.sum(f_scalar)),
        torque_trace_ev=float(np.sum(torque_diag)),
    )
    return kick, torque_diag, info


def _inertial_pk_energy(
    positions: np.ndarray,
    prev_positions: np.ndarray,
    velocity: np.ndarray,
    atom_mass_like: np.ndarray,
    *,
    k_potential: float,
    k_kinetic: float,
) -> float:
    """
    Per-atom inertial P/K budget:
      P_i = 0.5 * k_potential * m_i * ||x_i - x_prev_i||^2
      K_i = 0.5 * k_kinetic  * m_i * ||v_i||^2
    """
    dx = np.asarray(positions, dtype=float) - np.asarray(prev_positions, dtype=float)
    v = np.asarray(velocity, dtype=float)
    m = np.asarray(atom_mass_like, dtype=float).reshape(-1, 1)
    p_term = 0.5 * float(k_potential) * float(np.sum(m * (dx * dx)))
    k_term = 0.5 * float(k_kinetic) * float(np.sum(m * (v * v)))
    return float(p_term + k_term)


def _reflector_set(
    n_res: int,
    ligation_pairs: LigationPairs,
    contact_reflectors: Optional[Iterable[int]] = None,
) -> Set[int]:
    """
    Wave reflectors along the chain: termini, ligation endpoints, and optional
    contact-based virtual reflectors (non-bond steric proximity).
    """
    r: Set[int] = {0, max(0, int(n_res) - 1)}
    for i, j in ligation_pairs:
        r.add(int(i))
        r.add(int(j))
    if contact_reflectors is not None:
        n = int(n_res)
        for i in contact_reflectors:
            ii = int(i)
            if 0 <= ii < n:
                r.add(ii)
    return r


def contact_reflector_indices(
    ca: np.ndarray,
    grad_mags: Optional[np.ndarray] = None,
    *,
    min_seq_sep: int = 4,
    cutoff_ang: float = 8.0,
    max_reflectors: int = 16,
    grad_coupling: float = 1.0,
    score_mode: str = "hard_linear",
    inverse_power: float = 2.0,
    score_min_dist_ang: float = 1.0,
    contact_terminus_window: int = 0,
    contact_terminus_score_scale: float = 1.0,
) -> Set[int]:
    """
    Virtual reflector residues from nonlocal Cα–Cα proximity (no covalent bond).

    Pairs with sequence separation >= ``min_seq_sep`` and distance below
    ``cutoff_ang`` (Å) score higher when closer. If ``grad_mags`` is given,
    pairs where both ends have large |∇E| are boosted (repulsion / stiffening proxy).

    Used to transfer harmonic path structure into compacted / crowded regions
    without adding explicit ligation bonds.
    """
    pos = np.asarray(ca, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        return set()
    n = int(pos.shape[0])
    if n < 3:
        return set()
    sep = max(2, int(min_seq_sep))
    cut = float(max(1e-3, cutoff_ang))
    cap = max(2, int(max_reflectors))
    gm = None
    if grad_mags is not None:
        gm = np.asarray(grad_mags, dtype=float).reshape(-1)
        if gm.size != n:
            gm = None
    med_g = float(np.median(gm) + 1e-8) if gm is not None else 1.0
    tw = max(0, int(contact_terminus_window))
    ts_scale = float(contact_terminus_score_scale)
    scored: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + sep, n):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d >= cut:
                continue
            mode = str(score_mode).strip().lower()
            if mode in {"hard_linear", "linear"}:
                # Current behavior: weight by remaining slack to the cutoff.
                geom = max(0.0, cut - d)
            elif mode in {"inverse_power", "inverse_square", "inverse"}:
                # Inverse-power proximity: monotone in distance with ~1/d^p growth.
                p = float(inverse_power)
                md = float(max(1e-6, score_min_dist_ang))
                geom = (cut ** p) / (max(d, md) ** p)
            else:
                raise ValueError(f"Unknown contact score_mode: {score_mode!r}")
            if tw > 0 and ts_scale > 0.0 and (i < tw or i >= n - tw or j < tw or j >= n - tw):
                geom *= ts_scale
            boost = 1.0
            if gm is not None and float(grad_coupling) > 0.0:
                gi = float(gm[i]) / med_g
                gj = float(gm[j]) / med_g
                boost = 1.0 + float(grad_coupling) * gi * gj
            scored.append((geom * boost, i, j))
    scored.sort(key=lambda t: -t[0])
    chosen: Set[int] = set()
    for _, i, j in scored:
        if len(chosen) >= cap:
            break
        if len(chosen) < cap:
            chosen.add(int(i))
        if len(chosen) < cap:
            chosen.add(int(j))
    return chosen


def per_residue_resonance_multiplier(
    n_res: int,
    compaction_score: np.ndarray,
    *,
    terminus_boost: float = 1.8,
    core_damping: float = 0.4,
    transition_width: int = 5,
) -> np.ndarray:
    """
    Per-residue resonance multiplier >1 at free termini, <1 in core.

    Higher compaction_score means "more packed" -> lower resonance multiplier.
    """
    n = int(n_res)
    res = np.ones(n, dtype=float)
    cs = np.asarray(compaction_score, dtype=float).reshape(-1)
    if cs.size != n:
        cs = np.zeros(n, dtype=float)

    # Base damping from compaction: compact core -> damp resonance.
    res[:] = 1.0 - (float(core_damping) * cs)

    # Free termini boost: linear ramp from end to transition_width.
    tw = max(1, int(transition_width))
    for k in range(min(tw, n)):
        ramp = float(k) / float(tw)
        end_val = float(terminus_boost) * (1.0 - ramp) + float(res[k]) * ramp
        res[k] = end_val
        kk = n - 1 - k
        res[kk] = float(terminus_boost) * (1.0 - ramp) + float(res[kk]) * ramp
    return np.clip(res, 0.2, 3.0)


def per_residue_terminus_step_scale(
    n_res: int,
    *,
    boost: float = 1.3,
    transition_width: int = 8,
    core_scale: float = 1.0,
) -> np.ndarray:
    """
    Per-residue multipliers for gradient (and rapidity) steps: ``core_scale`` in the
    chain interior, ramping up to ``boost`` at each terminus.

    Unlike ``per_residue_resonance_multiplier``, this uses **only** sequence position
    (no compaction field), so it directly targets N/C-terminal placement without
    coupling to the harmonic temperature / tunnel-budget path.
    """
    n = int(n_res)
    c = float(core_scale)
    b = float(boost)
    out = np.full(n, c, dtype=float)
    if n < 2:
        return np.clip(out, 0.25, 4.0)
    lo = min(c, b)
    hi = max(c, b)
    tw = max(1, int(transition_width))
    for k in range(min(tw, n)):
        ramp = float(k) / float(tw)
        v = lo + (hi - lo) * (1.0 - ramp)
        out[k] = v
        out[n - 1 - k] = v
    return np.clip(out, 0.25, 4.0)


def compute_local_compaction_score(
    ca: np.ndarray,
    *,
    cutoff_ang: float = 8.0,
    min_seq_sep: int = 4,
) -> np.ndarray:
    """Simple local contact density per residue: fraction of nonlocal CA pairs within cutoff."""
    pos = np.asarray(ca, dtype=float)
    n = int(pos.shape[0])
    score = np.zeros(n, dtype=float)
    if n <= min_seq_sep + 1:
        return score
    cut2 = float(cutoff_ang) ** 2
    sep = int(min_seq_sep)
    for i in range(n):
        count = 0
        for j in range(i + sep, n):
            d2 = float(np.sum((pos[i] - pos[j]) ** 2))
            if d2 < cut2:
                count += 1
        score[i] = float(count) / float(max(1, n - sep))
    return score


def _segment_path_lengths(ca: np.ndarray) -> np.ndarray:
    pos = np.asarray(ca, dtype=float)
    n = int(pos.shape[0])
    if n <= 1:
        return np.zeros((0,), dtype=float)
    return np.linalg.norm(pos[1:] - pos[:-1], axis=1)


def fixed_free_first_mode_factor(
    ca: np.ndarray,
    *,
    fixed_end: str = "right",
    factor_min: float = 0.5,
    factor_max: float = 1.2,
) -> np.ndarray:
    """
    First-mode standing-wave participation for a fixed-free 1D string.

    Boundary conditions:
      - fixed end: displacement = 0
      - free end: spatial derivative = 0

    Mode shape (up to scale): y(s) = cos(pi*s/(2L)) when the LEFT end is free.
    For fixed_end="left", we mirror s -> L - s.

    Returns per-residue factor in [factor_min, factor_max] where the free end
    has ~factor_max and the fixed end has ~factor_min.
    """
    pos = np.asarray(ca, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        return np.ones((int(pos.shape[0]) if pos.ndim == 2 else 0,), dtype=float)
    n = int(pos.shape[0])
    if n < 2:
        return np.ones((n,), dtype=float)

    seg = _segment_path_lengths(pos)
    cumulative = np.zeros((n,), dtype=float)
    if seg.size > 0:
        cumulative[1:] = np.cumsum(seg)
    L = float(cumulative[n - 1])
    if L <= 1e-12:
        return np.ones((n,), dtype=float)

    fe = str(fixed_end).strip().lower()
    s = cumulative
    if fe in {"left", "nterm", "n-terminus"}:
        # Fixed at left -> free at right: mirror.
        s = float(L) - s
    elif fe in {"right", "cterm", "c-terminus"}:
        # Fixed at right -> free at left: keep s as-is.
        pass
    else:
        raise ValueError(f"Unknown fixed_end: {fixed_end!r}")

    # Participation: y(s)^2 with y(s)=cos(pi*s/(2L)).
    y = np.cos((np.pi * s) / (2.0 * L))
    part = y * y  # in [0,1]
    fmin = float(factor_min)
    fmax = float(factor_max)
    if fmax < fmin:
        fmin, fmax = fmax, fmin
    out = fmin + (fmax - fmin) * part
    return np.asarray(out, dtype=float)


def _distance_to_nearest_reflector_along_chain(
    idx: int,
    cumulative: np.ndarray,
    reflectors: Set[int],
) -> float:
    xi = float(cumulative[int(idx)])
    best = float("inf")
    for r in reflectors:
        d = abs(xi - float(cumulative[int(r)]))
        if d < best:
            best = d
    return float(best if np.isfinite(best) else 0.0)


def compute_tunnel_harmonic_budget_ev(
    ca: np.ndarray,
    atom_mass_like: np.ndarray,
    ligation_pairs: LigationPairs,
    active_idx: Optional[np.ndarray] = None,
    contact_reflectors: Optional[Iterable[int]] = None,
    distance_score_mode: str = "linear",
    inverse_power: float = 2.0,
    distance_d0_ang: float = 1.0,
    use_end_bias_budget: bool = False,
    end_bias_scale: float = 2.0,
    end_bias_floor: float = 0.1,
) -> float:
    """
    Harmonic tunnel budget (eV-like model units) from path length to bond reflectors.

    Budget per residue i:
      b_i ~ m_i * distance_along_chain_to_nearest_reflector(i)

    Reflectors: chain termini, ligation endpoints, and optional contact-based indices.
    """
    pos = np.asarray(ca, dtype=float)
    n = int(pos.shape[0])
    if n < 2:
        return 0.0
    masses = np.asarray(atom_mass_like, dtype=float).reshape(-1)
    seg = _segment_path_lengths(pos)
    cumulative = np.zeros((n,), dtype=float)
    if seg.size > 0:
        cumulative[1:] = np.cumsum(seg)
    reflectors = _reflector_set(n, ligation_pairs, contact_reflectors=contact_reflectors)
    total_len = float(cumulative[n - 1]) if n > 0 else 0.0
    if active_idx is None or int(active_idx.size) == 0:
        idx_list = np.arange(n, dtype=int)
    else:
        idx_list = np.asarray(active_idx, dtype=int)
    vals = []
    mode = str(distance_score_mode).strip().lower()
    p = float(inverse_power)
    d0 = float(max(0.0, distance_d0_ang))
    for i in idx_list.tolist():
        ii = int(i)
        d = _distance_to_nearest_reflector_along_chain(ii, cumulative, reflectors)
        m = float(masses[ii])
        base = float(max(0.0, m) * max(0.0, d))
        if bool(use_end_bias_budget) and total_len > 1e-8:
            # Ends-first: multiply the tunneling budget by a factor that is
            # maximal near the chain ends and damped toward the middle.
            dl = float(cumulative[ii])  # distance to left end along chain
            dr = float(max(0.0, total_len - dl))  # distance to right end along chain
            nearest = float(min(dl, dr))
            other = float(max(dl, dr))
            ratio = float(nearest / max(1e-8, other))
            # Additive boost: keep core close to 1.0 while termini can increase mobility.
            end_factor = float(1.0 + float(end_bias_scale) * (1.0 - ratio))
            end_factor = float(np.clip(end_factor, float(end_bias_floor), 1.0 + float(end_bias_scale)))
            base *= end_factor
        if mode in {"linear"}:
            vals.append(base)
        elif mode in {"inverse_power", "inverse", "inverse_square"}:
            # Avoid singularities at d=0 by shifting with d0.
            denom = float(max(1e-6, float(max(0.0, d)) + d0))
            base_inv = float(max(0.0, m) / (denom**p if p != 0.0 else 1.0))
            if bool(use_end_bias_budget) and total_len > 1e-8:
                dl = float(cumulative[ii])
                dr = float(max(0.0, total_len - dl))
                nearest = float(min(dl, dr))
                other = float(max(dl, dr))
                ratio = float(nearest / max(1e-8, other))
                end_factor = float(1.0 + float(end_bias_scale) * (1.0 - ratio))
                end_factor = float(np.clip(end_factor, float(end_bias_floor), 1.0 + float(end_bias_scale)))
                base_inv *= end_factor
            vals.append(base_inv)
        else:
            raise ValueError(f"Unknown distance_score_mode: {distance_score_mode!r}")
    if not vals:
        return 0.0
    return float(np.mean(np.asarray(vals, dtype=float)))


def _local_energy_by_index_from_grad(grad: np.ndarray) -> np.ndarray:
    """
    Build a positive local energy proxy per residue from gradient magnitudes.
    Lower values correspond to locally smoother/low-energy directions.
    """
    mags = np.linalg.norm(np.asarray(grad, dtype=float), axis=1)
    if mags.size == 0:
        return mags
    base = float(np.median(mags) + 1e-8)
    return np.asarray(mags / base, dtype=float)


def sparse_visible_energy(reg: SparseRegister, local_energy: np.ndarray) -> float:
    """
    Sparse state visible energy proxy:
      E_vis = Σ |amp_i|^2 * local_energy[idx_i].
    """
    n = int(local_energy.shape[0])
    if n <= 0:
        return 0.0
    e = 0.0
    for i, a in reg:
        idx = int(i) % n
        e += float(a) * float(a) * float(local_energy[idx])
    return float(e)


def harmonic_tunneling_mixer_strength(
    omega: float,
    visible_energy: float,
    *,
    mixer_floor: float = 0.05,
    mixer_cap: float = 0.95,
) -> float:
    """
    Harmonic-tuned mixer strength:
      strength ~ omega / |E_visible|, clipped to a stable range.
    """
    raw = float(omega) / max(1e-8, abs(float(visible_energy)))
    return float(np.clip(raw, float(mixer_floor), float(mixer_cap)))


def qpe_low_energy_subspace(
    reg: SparseRegister,
    local_energy: np.ndarray,
    *,
    k: int = 4,
) -> List[int]:
    """
    QPE-style low-energy subspace proxy:
    rank sparse indices by score = |amp| / (1 + local_energy[idx]),
    return top-k unique indices.
    """
    n = int(local_energy.shape[0])
    if n <= 0 or not reg:
        return []
    scored: List[Tuple[float, int]] = []
    for i, a in reg:
        idx = int(i) % n
        s = abs(float(a)) / (1.0 + float(local_energy[idx]))
        scored.append((s, idx))
    scored.sort(key=lambda t: t[0], reverse=True)
    out: List[int] = []
    seen: Set[int] = set()
    for _, idx in scored:
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
        if len(out) >= max(1, int(k)):
            break
    return out


def amplify_low_energy(
    reg: SparseRegister,
    targets: Iterable[int],
    *,
    gain: float = 1.6,
    non_target_decay: float = 0.985,
) -> SparseRegister:
    """
    Grover-style amplitude amplification surrogate on target subspace.
    """
    tgt = {int(t) for t in targets}
    if not reg:
        return []
    out: SparseRegister = []
    for i, a in reg:
        aa = float(a)
        if int(i) in tgt:
            aa *= float(gain)
        else:
            aa *= float(non_target_decay)
        out.append((int(i), aa))
    # Global renormalization keeps amplitudes numerically stable.
    norm = float(np.sqrt(sum((a * a) for _, a in out)))
    if norm > 1e-12:
        out = [(i, float(a / norm)) for i, a in out]
    return out


def harmonic_tunneled_qaoa_folding(
    ca_init: np.ndarray,
    *,
    sequence: Optional[str] = None,
    ligation_pairs: Optional[LigationPairs] = None,
    auto_detect_cys_ligation: bool = False,
    ligation_detect_max_dist: float = 6.5,
    ligation_r_eq: float = 3.8,
    ligation_r_min: float = 2.5,
    ligation_r_max: float = 6.0,
    ligation_k_bond: float = 60.0,
    z_shell: int = 6,
    depth: int = 8,
    layers: int = 6,
    qpe_k: int = 4,
    base_step: float = 0.03,
    energy_kwargs: Optional[Dict[str, Any]] = None,
    use_harmonic_metropolis: bool = False,
    random_seed: Optional[int] = None,
    inertial_pk_weight: float = 0.0,
    inertial_k_potential: float = 1.0,
    inertial_k_kinetic: float = 1.0,
    inertial_velocity_decay: float = 0.9,
    use_energy_reservoir: bool = True,
    reservoir_init: float = 0.0,
    reservoir_gain_scale: float = 1.0,
    use_local_rapidity_translation: bool = False,
    rapidity_gain: float = 0.25,
    rapidity_tangent_weight: float = 0.7,
    rapidity_normal_weight: float = 0.3,
    r_min: float = 2.5,
    r_max: float = 6.0,
) -> Tuple[np.ndarray, QAOAHarmonicFoldInfo]:
    """
    Quantum-inspired global explorer:
      cost layer -> harmonic mixer -> QPE-like subspace pick ->
      low-energy amplification -> sparsity prune -> CA update.
    """
    ca = np.asarray(ca_init, dtype=float).copy()
    if ca.ndim != 2 or ca.shape[1] != 3:
        raise ValueError("harmonic_tunneled_qaoa_folding: ca_init must be shape (n, 3).")
    n = int(ca.shape[0])
    if n < 2:
        info = QAOAHarmonicFoldInfo(
            layers=int(layers),
            accepted_layers=0,
            final_energy_ev=0.0,
            final_visible_energy_ev=0.0,
            natural_harmonic_scale=1.0,
            avg_flipped_count=0.0,
            qpe_targets_last=0,
            final_step_size=float(base_step),
            inertial_energy_final_ev=0.0,
            reservoir_energy_final_ev=float(reservoir_init),
            reservoir_uphill_accepts=0,
            tunnel_harmonic_budget_final_ev=0.0,
        )
        return ca, info

    e_kw = dict(energy_kwargs or {})
    z = np.full(n, int(z_shell), dtype=np.int32)
    pairs = _normalize_ligation_pairs(ligation_pairs, n)
    if bool(auto_detect_cys_ligation) and sequence:
        pairs = _normalize_ligation_pairs(
            [*pairs, *auto_detect_cys_ligation_pairs(sequence, ca, max_dist_ang=float(ligation_detect_max_dist))],
            n,
        )
    omega = estimate_natural_harmonic_scale_ca(
        ca,
        int(z_shell),
        energy_kwargs=e_kw,
        ligation_pairs=pairs,
        ligation_r_eq=float(ligation_r_eq),
        ligation_r_min=float(ligation_r_min),
        ligation_r_max=float(ligation_r_max),
        ligation_k_bond=float(ligation_k_bond),
    )
    initial_energy_abs = float(
        abs(
            _energy_with_ligation(
                ca,
                z,
                e_kw,
                pairs,
                ligation_r_eq=float(ligation_r_eq),
                ligation_r_min=float(ligation_r_min),
                ligation_r_max=float(ligation_r_max),
                ligation_k_bond=float(ligation_k_bond),
            )
        )
    )
    L = max(1, n - 1)
    rng = np.random.default_rng(random_seed)
    atom_mass_like = np.full(n, float(max(1, int(z_shell))), dtype=float)
    velocity = np.zeros_like(ca)
    prev_ca = ca.copy()
    reservoir = float(max(0.0, reservoir_init))
    reservoir_uphill_accepts = 0
    tunnel_budget_ev_last = 0.0
    accepted = 0
    prev_reg: SparseRegister = []
    step = float(base_step)
    sum_flip = 0.0
    qpe_targets_last = 0
    vis_e_last = 0.0

    for layer in range(max(1, int(layers))):
        e0_base = _energy_with_ligation(
            ca,
            z,
            e_kw,
            pairs,
            ligation_r_eq=float(ligation_r_eq),
            ligation_r_min=float(ligation_r_min),
            ligation_r_max=float(ligation_r_max),
            ligation_k_bond=float(ligation_k_bond),
        )
        e0_inertial = _inertial_pk_energy(
            ca,
            prev_ca,
            velocity,
            atom_mass_like,
            k_potential=float(inertial_k_potential),
            k_kinetic=float(inertial_k_kinetic),
        )
        e0 = float(e0_base + float(inertial_pk_weight) * e0_inertial)
        grad = grad_full(ca, z, include_bonds=True, include_horizon=True, **e_kw)
        local_e = _local_energy_by_index_from_grad(grad)
        reg = _register_from_gradients(
            grad,
            L=L,
            amp_threshold=float(np.quantile(np.linalg.norm(grad, axis=1), 0.60)),
        )
        if not reg:
            reg = [(i, float(np.linalg.norm(grad[i]))) for i in range(n)]
        # Cost layer (energy operator proxy): favor local low-energy support.
        vis_e = sparse_visible_energy(reg, local_e)
        vis_e_last = float(vis_e)
        cost_mix = float(np.clip(0.15 + 0.5 / (1.0 + vis_e), 0.05, 0.75))
        reg_cost = apply_ansatz_sparse(L, reg, depth=max(1, int(depth // 2)), phi_mix=cost_mix, psi_mix=cost_mix)
        # Mixer layer: harmonic-tuned tunneling strength.
        mix_strength = harmonic_tunneling_mixer_strength(float(omega), float(vis_e))
        reg_mix = apply_ansatz_sparse(
            L,
            reg_cost,
            depth=max(1, int(depth)),
            phi_mix=mix_strength,
            psi_mix=float(np.clip(0.5 * (mix_strength + cost_mix), 0.05, 0.95)),
        )
        # QPE-like low-energy subspace selection + amplification.
        targets = qpe_low_energy_subspace(reg_mix, local_e, k=int(qpe_k))
        qpe_targets_last = len(targets)
        reg_amp = amplify_low_energy(reg_mix, targets, gain=1.8, non_target_decay=0.985)
        # Sparsity pruning.
        flipped = (
            detect_flipped_kets_amplitude(prev_reg, reg_amp, amp_delta_eps=1e-6, include_sign_flip=True)
            if prev_reg
            else [i for i, _ in reg_amp]
        )
        if not flipped:
            flipped = [i for i, _ in reg_amp]
        reg_pruned = prune_to_flipped(flipped, reg_amp)
        prev_reg = reg_amp
        sum_flip += float(len(flipped))

        active_idx = np.array(sorted({int(i % n) for i, _ in reg_pruned}), dtype=int)
        if active_idx.size == 0:
            active_idx = np.arange(n, dtype=int)

        tunnel_budget_ev = compute_tunnel_harmonic_budget_ev(ca, atom_mass_like, pairs, active_idx=active_idx)
        tunnel_budget_ev_last = float(tunnel_budget_ev)
        norm_budget = float(tunnel_budget_ev / max(1e-8, abs(initial_energy_abs)))
        step_eff = float(step * np.sqrt(max(1e-12, norm_budget / max(1e-12, 1.0))))

        cand = ca.copy()
        cand[active_idx] -= step_eff * grad[active_idx]
        if bool(use_local_rapidity_translation):
            disp_r = _local_rapidity_displacement(
                ca,
                grad,
                active_idx,
                gain=float(rapidity_gain),
                tangent_weight=float(rapidity_tangent_weight),
                normal_weight=float(rapidity_normal_weight),
            )
            cand[active_idx] += disp_r[active_idx]
        cand = _project_bonds(cand, r_min=float(r_min), r_max=float(r_max))
        cand = _project_extra_bonds(
            cand,
            pairs,
            r_min=float(ligation_r_min),
            r_max=float(ligation_r_max),
            passes=2,
        )
        cand_velocity = float(np.clip(inertial_velocity_decay, 0.0, 1.0)) * velocity
        cand_velocity[active_idx] += cand[active_idx] - ca[active_idx]
        e1_base = _energy_with_ligation(
            cand,
            z,
            e_kw,
            pairs,
            ligation_r_eq=float(ligation_r_eq),
            ligation_r_min=float(ligation_r_min),
            ligation_r_max=float(ligation_r_max),
            ligation_k_bond=float(ligation_k_bond),
        )
        e1_inertial = _inertial_pk_energy(
            cand,
            ca,
            cand_velocity,
            atom_mass_like,
            k_potential=float(inertial_k_potential),
            k_kinetic=float(inertial_k_kinetic),
        )
        e1 = float(e1_base + float(inertial_pk_weight) * e1_inertial)
        accept = bool(e1 <= e0)
        if (not accept) and bool(use_harmonic_metropolis):
            accept = metropolis_accept_with_harmonic(
                e0,
                e1,
                iteration=int(layer),
                n_iter=int(max(1, layers)),
                omega=float(omega),
                initial_energy_abs=float(initial_energy_abs),
                base_temp=1.0,
                min_temp=1e-4,
                rng=rng,
            )
        if (not accept) and bool(use_energy_reservoir):
            uphill_cost = float(max(0.0, e1 - e0))
            if uphill_cost <= reservoir:
                accept = True
                reservoir -= uphill_cost
                reservoir_uphill_accepts += 1
        if accept:
            drop = float(max(0.0, e0 - e1))
            if bool(use_energy_reservoir) and drop > 0.0:
                reservoir += float(reservoir_gain_scale) * drop
            prev_ca = ca.copy()
            ca = cand
            velocity = cand_velocity
            accepted += 1
            step *= 1.02
        else:
            step *= 0.65
        step = float(np.clip(step, 1e-4, 0.12))

    e_final_base = _energy_with_ligation(
        ca,
        z,
        e_kw,
        pairs,
        ligation_r_eq=float(ligation_r_eq),
        ligation_r_min=float(ligation_r_min),
        ligation_r_max=float(ligation_r_max),
        ligation_k_bond=float(ligation_k_bond),
    )
    e_final_inertial = _inertial_pk_energy(
        ca,
        prev_ca,
        velocity,
        atom_mass_like,
        k_potential=float(inertial_k_potential),
        k_kinetic=float(inertial_k_kinetic),
    )
    e_final = float(e_final_base + float(inertial_pk_weight) * e_final_inertial)
    info = QAOAHarmonicFoldInfo(
        layers=int(max(1, int(layers))),
        accepted_layers=int(accepted),
        final_energy_ev=float(e_final),
        final_visible_energy_ev=float(vis_e_last),
        natural_harmonic_scale=float(omega),
        avg_flipped_count=float(sum_flip / float(max(1, int(layers)))),
        qpe_targets_last=int(qpe_targets_last),
        final_step_size=float(step),
        inertial_energy_final_ev=float(e_final_inertial),
        reservoir_energy_final_ev=float(reservoir),
        reservoir_uphill_accepts=int(reservoir_uphill_accepts),
        tunnel_harmonic_budget_final_ev=float(tunnel_budget_ev_last),
    )
    return ca, info


def estimate_natural_harmonic_scale_ca(
    ca: np.ndarray,
    z_shell: int,
    *,
    energy_kwargs: Optional[Dict[str, Any]] = None,
    ligation_pairs: Optional[LigationPairs] = None,
    ligation_r_eq: float = 3.8,
    ligation_r_min: float = 2.5,
    ligation_r_max: float = 6.0,
    ligation_k_bond: float = 60.0,
    fd_eps: float = 5e-3,
    max_dims: int = 72,
) -> float:
    """
    Estimate a protein-specific harmonic scale ω from local energy curvature.

    Uses a finite-difference approximation to diagonal Hessian entries of
    ``e_tot_ca_with_bonds`` around the current Cα state:
        d²E/dx_i² ≈ [E(x+εe_i) - 2E(x) + E(x-εe_i)] / ε²
    and returns:
        ω ≈ sqrt(mean(abs(diag(H)))).
    """
    pos = np.asarray(ca, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        return 1.0
    n = int(pos.shape[0])
    if n < 2:
        return 1.0
    e_kw = dict(energy_kwargs or {})
    z = np.full(n, int(z_shell), dtype=np.int32)
    pairs = _normalize_ligation_pairs(ligation_pairs, n)
    e0 = _energy_with_ligation(
        pos,
        z,
        e_kw,
        pairs,
        ligation_r_eq=float(ligation_r_eq),
        ligation_r_min=float(ligation_r_min),
        ligation_r_max=float(ligation_r_max),
        ligation_k_bond=float(ligation_k_bond),
    )
    eps = float(max(1e-4, fd_eps))
    dims = n * 3
    step = max(1, int(np.ceil(float(dims) / float(max(1, max_dims)))))
    diag_vals: List[float] = []
    for k in range(0, dims, step):
        i = k // 3
        d = k % 3
        p = pos.copy()
        m = pos.copy()
        p[i, d] += eps
        m[i, d] -= eps
        ep = _energy_with_ligation(
            p,
            z,
            e_kw,
            pairs,
            ligation_r_eq=float(ligation_r_eq),
            ligation_r_min=float(ligation_r_min),
            ligation_r_max=float(ligation_r_max),
            ligation_k_bond=float(ligation_k_bond),
        )
        em = _energy_with_ligation(
            m,
            z,
            e_kw,
            pairs,
            ligation_r_eq=float(ligation_r_eq),
            ligation_r_min=float(ligation_r_min),
            ligation_r_max=float(ligation_r_max),
            ligation_k_bond=float(ligation_k_bond),
        )
        h_ii = (ep - 2.0 * e0 + em) / (eps ** 2)
        diag_vals.append(abs(float(h_ii)))
    if not diag_vals:
        return 1.0
    return float(np.sqrt(max(1e-12, float(np.mean(diag_vals)))))


def harmonic_temperature_schedule(
    omega: float,
    iteration: int,
    n_iter: int,
    *,
    initial_energy_abs: float,
    base_temp: float = 1.0,
    min_temp: float = 1e-4,
) -> float:
    """
    Dimensionless harmonic schedule:
      T(i) = base_temp * (omega / |E0|) * (1 - i/n_iter), clipped at min_temp.
    """
    n = max(1, int(n_iter))
    i = int(np.clip(iteration, 0, n))
    e0 = max(1e-8, float(abs(initial_energy_abs)))
    normalized_omega = float(omega) / e0
    cooling = 1.0 - float(i) / float(n)
    t = float(base_temp) * normalized_omega * cooling
    return float(max(float(min_temp), t))


def metropolis_accept_with_harmonic(
    e0: float,
    e1: float,
    *,
    iteration: int,
    n_iter: int,
    omega: float,
    initial_energy_abs: float,
    base_temp: float = 1.0,
    min_temp: float = 1e-4,
    rng: np.random.Generator,
) -> bool:
    """
    Harmonic-scale Metropolis acceptance:
      accept if downhill, else with exp(-(e1-e0)/T(i)), T(i) from ω.
    """
    if float(e1) <= float(e0):
        return True
    delta_e = float(e1) - float(e0)
    temp = harmonic_temperature_schedule(
        float(omega),
        int(iteration),
        int(n_iter),
        initial_energy_abs=float(initial_energy_abs),
        base_temp=float(base_temp),
        min_temp=float(min_temp),
    )
    prob = float(np.exp(-delta_e / max(1e-12, temp)))
    return bool(rng.random() < prob)


def minimize_ca_with_osh_oracle(
    ca_init: np.ndarray,
    *,
    sequence: Optional[str] = None,
    ligation_pairs: Optional[LigationPairs] = None,
    auto_detect_cys_ligation: bool = False,
    ligation_detect_max_dist: float = 6.5,
    ligation_r_eq: float = 3.8,
    ligation_r_min: float = 2.5,
    ligation_r_max: float = 6.0,
    ligation_k_bond: float = 60.0,
    z_shell: int = 6,
    z_shells: Optional[np.ndarray] = None,
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
    use_local_rapidity_translation: bool = False,
    rapidity_gain: float = 0.25,
    rapidity_tangent_weight: float = 0.7,
    rapidity_normal_weight: float = 0.3,
    stop_when_settled: bool = False,
    settle_window: int = 20,
    settle_energy_tol: float = 1e-3,
    settle_step_tol: float = 3e-4,
    settle_min_iter: int = 30,
    r_min: float = 2.5,
    r_max: float = 6.0,
    use_contact_reflectors: bool = False,
    contact_min_seq_sep: int = 4,
    contact_cutoff_ang: float = 8.0,
    contact_max_reflectors: int = 16,
    contact_grad_coupling: float = 1.0,
    contact_weight_gradient: bool = True,
    contact_score_mode: str = "hard_linear",
    contact_inverse_power: float = 2.0,
    contact_score_min_dist_ang: float = 1.0,
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
    contact_terminus_window: int = 0,
    contact_terminus_score_scale: float = 1.0,
    energy_kwargs: Optional[Dict[str, Any]] = None,
    use_hqiv_native_gate: bool = False,
    hqiv_reference_m: int = REFERENCE_M_HQIV_NATIVE,
) -> Tuple[np.ndarray, OSHOracleFoldInfo]:
    """
    Cα minimization driven by OSHoracle sparse support updates.

    Per iteration:
    - compute full HQIV gradient,
    - build sparse register on high-gradient support,
    - apply causal expand + gate evolve,
    - detect flipped support and update only mapped residues,
    - accept if energy improves (simple backoff otherwise).

    ``use_hqiv_native_gate``: if True, use Lean ``hqivNativePhaseGate`` / ``hqivPivotFromShells``
    (π phase on one harmonic mode; pivot from per-residue ``z`` shells and ``hqiv_reference_m``).
    Sparse cutoff ``L`` is set to ``n`` residues to match ``ProteinFoldingHook`` (else ``max(1,n-1)``).

    Optional ``use_contact_reflectors``: add virtual wave reflectors at nonlocal Cα
    contacts (and optionally gradient-weighted) to reshape harmonic tunnel budgets
    as the chain compacts, without explicit ligation bonds.

    ``omega_refresh_period``: if > 0, re-estimate natural harmonic scale ω every
    that many iterations (mainly affects harmonic Metropolis acceptance).
    """
    ca = np.asarray(ca_init, dtype=float).copy()
    if ca.ndim != 2 or ca.shape[1] != 3:
        raise ValueError("minimize_ca_with_osh_oracle: ca_init must be shape (n, 3).")
    n = int(ca.shape[0])
    if n < 2:
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
        return ca, info

    e_kw = dict(energy_kwargs or {})
    if z_shells is not None:
        zs = np.asarray(z_shells, dtype=np.int32).reshape(-1)
        if int(zs.shape[0]) != n:
            raise ValueError("minimize_ca_with_osh_oracle: z_shells must have length n.")
        z = zs
    else:
        z = np.full(n, int(z_shell), dtype=np.int32)
    pairs = _normalize_ligation_pairs(ligation_pairs, n)
    if bool(auto_detect_cys_ligation) and sequence:
        pairs = _normalize_ligation_pairs(
            [*pairs, *auto_detect_cys_ligation_pairs(sequence, ca, max_dist_ang=float(ligation_detect_max_dist))],
            n,
        )
    L = max(1, n - 1)
    L_sparse = int(n) if bool(use_hqiv_native_gate) else int(L)
    shells_hqiv = np.asarray(z, dtype=np.int64)
    step = float(step_size)
    atom_mass_like = np.maximum(1.0, z.astype(float))
    velocity = np.zeros_like(ca)
    prev_ca = ca.copy()
    reservoir = float(max(0.0, reservoir_init))
    reservoir_uphill_accepts = 0
    tunnel_budget_ev_last = 0.0
    omega_refresh_count = 0
    contact_reflector_count_last = 0
    accepted = 0
    metropolis_accepts = 0
    prev_reg: SparseRegister = []
    last_flipped_count = 0
    sum_flipped_count = 0.0
    settled = False
    stop_reason = "max_iter_reached"
    executed = 0
    recent_energy: List[float] = []
    recent_step: List[float] = []
    rng = np.random.default_rng(random_seed)
    omega = estimate_natural_harmonic_scale_ca(
        ca,
        int(z_shell),
        energy_kwargs=e_kw,
        ligation_pairs=pairs,
        ligation_r_eq=float(ligation_r_eq),
        ligation_r_min=float(ligation_r_min),
        ligation_r_max=float(ligation_r_max),
        ligation_k_bond=float(ligation_k_bond),
        fd_eps=float(harmonic_fd_eps),
        max_dims=int(harmonic_max_dims),
    )
    initial_energy_abs = float(
        abs(
            _energy_with_ligation(
                ca,
                z,
                e_kw,
                pairs,
                ligation_r_eq=float(ligation_r_eq),
                ligation_r_min=float(ligation_r_min),
                ligation_r_max=float(ligation_r_max),
                ligation_k_bond=float(ligation_k_bond),
            )
        )
    )

    n_iter_eff = max(1, int(n_iter))
    sched_n = max(2, int(schedule_period))
    for it in range(n_iter_eff):
        executed = it + 1
        per_w = int(omega_refresh_period)
        if per_w > 0 and it > 0 and (it % per_w) == 0:
            omega = estimate_natural_harmonic_scale_ca(
                ca,
                int(z_shell),
                energy_kwargs=e_kw,
                ligation_pairs=pairs,
                ligation_r_eq=float(ligation_r_eq),
                ligation_r_min=float(ligation_r_min),
                ligation_r_max=float(ligation_r_max),
                ligation_k_bond=float(ligation_k_bond),
                fd_eps=float(harmonic_fd_eps),
                max_dims=int(harmonic_max_dims),
            )
            omega_refresh_count += 1
        e0_base = _energy_with_ligation(
            ca,
            z,
            e_kw,
            pairs,
            ligation_r_eq=float(ligation_r_eq),
            ligation_r_min=float(ligation_r_min),
            ligation_r_max=float(ligation_r_max),
            ligation_k_bond=float(ligation_k_bond),
        )
        e0_inertial = _inertial_pk_energy(
            ca,
            prev_ca,
            velocity,
            atom_mass_like,
            k_potential=float(inertial_k_potential),
            k_kinetic=float(inertial_k_kinetic),
        )
        e0 = float(e0_base + float(inertial_pk_weight) * e0_inertial)
        grad = grad_full(ca, z, include_bonds=True, include_horizon=True, **e_kw)
        mags = np.linalg.norm(grad, axis=1)
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
        reg = _register_from_gradients(grad, L=L_sparse, amp_threshold=thresh)
        # Decouple phase schedule from safety cap: fixed period keeps dynamics invariant
        # when only max-iteration budget changes.
        phase_it = int(it % sched_n)
        phi_mix, psi_mix = current_parameters(phase_it, sched_n, gate_mix)
        if bool(use_hqiv_native_gate):
            reg_after = apply_ansatz_sparse_hqiv_native(
                L_sparse,
                reg,
                depth=int(ansatz_depth),
                shells=shells_hqiv,
                reference_m=int(hqiv_reference_m),
            )
        else:
            reg_after = apply_ansatz_sparse(
                L_sparse,
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
        active_res: Set[int] = {int(i % n) for i, _ in pruned}
        if not active_res:
            active_res = {int(i % n) for i, _ in reg_after}
        active_idx = np.array(sorted(active_res), dtype=int)
        if active_idx.size == 0:
            break

        step_eff = float(step)
        # Resonance multiplier: dampen steps in compact core, boost around free termini.
        # This is a lightweight "virtual impedance" proxy that can help termini wrap
        # without requiring explicit bond/ligation topology changes.
        base_temp_dyn = float(harmonic_base_temp)
        resonance_active_mean = 1.0
        if bool(use_resonance_multiplier) and (bool(harmonic_step_anneal) or bool(use_harmonic_metropolis)):
            compaction = compute_local_compaction_score(
                ca,
                cutoff_ang=float(resonance_compaction_cutoff_ang),
                min_seq_sep=int(resonance_compaction_min_seq_sep),
            )
            resonance_mult = per_residue_resonance_multiplier(
                n,
                compaction,
                terminus_boost=float(resonance_terminus_boost),
                core_damping=float(resonance_core_damping),
                transition_width=int(resonance_transition_width),
            )
            if active_idx.size:
                resonance_active_mean = float(np.mean(resonance_mult[active_idx]))
            else:
                resonance_active_mean = 1.0
            base_temp_dyn = float(harmonic_base_temp) * resonance_active_mean

        if bool(harmonic_step_anneal):
            # Harmonic tunnel budget from path length to nearest bond/contact reflector.
            tunnel_budget_ev = compute_tunnel_harmonic_budget_ev(
                ca,
                atom_mass_like,
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
            g_act = float(np.mean(np.linalg.norm(grad[active_idx], axis=1)) if active_idx.size else g_all)
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

        term_scale = np.ones((n,), dtype=float)
        if bool(use_terminus_gradient_boost):
            term_scale = per_residue_terminus_step_scale(
                n,
                boost=float(terminus_gradient_boost),
                transition_width=int(terminus_gradient_transition_width),
                core_scale=float(terminus_gradient_core_scale),
            )
        ts_act = term_scale[active_idx].reshape(-1, 1)

        cand = ca.copy()
        cand[active_idx] -= step_eff * ts_act * grad[active_idx]
        if bool(use_local_rapidity_translation):
            disp_r = _local_rapidity_displacement(
                ca,
                grad,
                active_idx,
                gain=float(rapidity_gain),
                tangent_weight=float(rapidity_tangent_weight),
                normal_weight=float(rapidity_normal_weight),
            )
            cand[active_idx] += ts_act * disp_r[active_idx]
        cand = _project_bonds(cand, r_min=float(r_min), r_max=float(r_max))
        cand = _project_extra_bonds(
            cand,
            pairs,
            r_min=float(ligation_r_min),
            r_max=float(ligation_r_max),
            passes=2,
        )
        cand_velocity = float(np.clip(inertial_velocity_decay, 0.0, 1.0)) * velocity
        cand_velocity[active_idx] += cand[active_idx] - ca[active_idx]
        e1_base = _energy_with_ligation(
            cand,
            z,
            e_kw,
            pairs,
            ligation_r_eq=float(ligation_r_eq),
            ligation_r_min=float(ligation_r_min),
            ligation_r_max=float(ligation_r_max),
            ligation_k_bond=float(ligation_k_bond),
        )
        e1_inertial = _inertial_pk_energy(
            cand,
            ca,
            cand_velocity,
            atom_mass_like,
            k_potential=float(inertial_k_potential),
            k_kinetic=float(inertial_k_kinetic),
        )
        e1 = float(e1_base + float(inertial_pk_weight) * e1_inertial)
        delta_e = float(e1 - e0)
        accept = False
        if bool(strict_descent_budget_mode):
            # Phase 1: strict descent only.
            if delta_e <= 0.0:
                accept = True
            else:
                # Phase 2: if descent fails, attempt budgeted climb only.
                if bool(use_energy_reservoir):
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
            prev_ca = ca.copy()
            ca = cand
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

    e_final_base = _energy_with_ligation(
        ca,
        z,
        e_kw,
        pairs,
        ligation_r_eq=float(ligation_r_eq),
        ligation_r_min=float(ligation_r_min),
        ligation_r_max=float(ligation_r_max),
        ligation_k_bond=float(ligation_k_bond),
    )
    e_final_inertial = _inertial_pk_energy(
        ca,
        prev_ca,
        velocity,
        atom_mass_like,
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
    return ca, info


def minimize_ca_with_osh_oracle_additive_cycles(
    ca_init: np.ndarray,
    *,
    max_cycles: int = 4,
    additive_update_every: int = 1,
    additive_kick_gain: float = 0.004,
    additive_torque_mix: float = 0.3,
    cycle_kick_stop_tol: float = 1e-4,
    cycle_energy_stop_tol: float = 1e-4,
    em_trigger_every_n_cycles: int = 0,
    em_trigger_on_settled: bool = True,
    em_trigger_disp_mean_ang: float = 0.35,
    em_trigger_energy_rise_ev: float = 1e3,
    em_trigger_reservoir_draw_ev: float = 5.0,
    em_trigger_horizon_enter_count: int = 1,
    em_trigger_horizon_radius_ang: float = 15.0,
    em_trigger_horizon_min_seq_sep: int = 3,
    reservoir_carry_cap_ev: float = 1e6,
    kick_max_norm_ang: float = 0.002,
    kick_accept_max_energy_rise_ev: float = 5e3,
    **osh_kwargs: Any,
) -> Tuple[np.ndarray, OSHOracleFoldInfo, List[OSHAdditiveCycleInfo]]:
    """
    Cyclic refinement: OSHoracle settle -> additive field/torque kick -> repeat.

    Budget bookkeeping:
    - Reservoir is carried between cycles (`reservoir_init <- reservoir_energy_final_ev`).
    - Per-cycle additive field and torque trace terms are reported.
    """
    ca = np.asarray(ca_init, dtype=float).copy()
    if ca.ndim != 2 or ca.shape[1] != 3:
        raise ValueError("minimize_ca_with_osh_oracle_additive_cycles: ca_init must be shape (n,3).")
    n = int(ca.shape[0])
    if n < 2:
        info = OSHOracleFoldInfo(
            iterations=0,
            iterations_executed=0,
            accepted_steps=0,
            final_energy_ev=0.0,
            last_step_size=0.0,
            last_flipped_count=0,
            avg_flipped_count=0.0,
            natural_harmonic_scale=1.0,
            metropolis_accepts=0,
            stop_reason="too_short",
            settled=True,
            inertial_energy_final_ev=0.0,
            reservoir_energy_final_ev=float(osh_kwargs.get("reservoir_init", 0.0)),
            reservoir_uphill_accepts=0,
            tunnel_harmonic_budget_final_ev=0.0,
            contact_reflector_count=0,
            omega_refresh_count=0,
        )
        return ca, info, []

    base_kwargs = dict(osh_kwargs)
    reservoir_carry = float(max(0.0, base_kwargs.get("reservoir_init", 0.0)))
    res_cap = float(max(0.0, reservoir_carry_cap_ev))
    if res_cap > 0.0:
        reservoir_carry = min(reservoir_carry, res_cap)
    torque_cache: Optional[np.ndarray] = None
    prev_energy: Optional[float] = None
    cycle_infos: List[OSHAdditiveCycleInfo] = []
    last_info: Optional[OSHOracleFoldInfo] = None
    n_cycles = max(1, int(max_cycles))
    update_every = max(1, int(additive_update_every))
    z_shell = int(base_kwargs.get("z_shell", 6))
    r_min = float(base_kwargs.get("r_min", 2.5))
    r_max = float(base_kwargs.get("r_max", 6.0))
    for cyc in range(n_cycles):
        ca_before = ca.copy()
        cyc_kwargs = dict(base_kwargs)
        cyc_kwargs["reservoir_init"] = float(reservoir_carry)
        ca, info = minimize_ca_with_osh_oracle(ca, **cyc_kwargs)
        last_info = info
        reservoir_after_osh = float(max(0.0, info.reservoir_energy_final_ev))
        reservoir_draw = float(max(0.0, float(cyc_kwargs["reservoir_init"]) - reservoir_after_osh))
        d_ca = np.linalg.norm(ca - ca_before, axis=1)
        disp_mean = float(np.mean(d_ca)) if d_ca.size else 0.0
        disp_max = float(np.max(d_ca)) if d_ca.size else 0.0
        e_delta = 0.0 if prev_energy is None else float(info.final_energy_ev - prev_energy)
        h_enter = int(
            count_nonlocal_pairs_entering_horizon(
                ca_before,
                ca,
                r_horizon=float(em_trigger_horizon_radius_ang),
                min_seq_sep=int(em_trigger_horizon_min_seq_sep),
            )
        )
        reservoir_carry = reservoir_after_osh
        if res_cap > 0.0:
            reservoir_carry = min(reservoir_carry, res_cap)
        reasons: List[str] = []
        if bool(em_trigger_on_settled) and bool(info.settled):
            reasons.append("settled")
        if disp_mean >= float(em_trigger_disp_mean_ang):
            reasons.append("big_shift")
        if e_delta >= float(em_trigger_energy_rise_ev):
            reasons.append("energy_rise")
        if reservoir_draw >= float(em_trigger_reservoir_draw_ev):
            reasons.append("reservoir_draw")
        if h_enter >= int(max(0, em_trigger_horizon_enter_count)):
            reasons.append("horizon_enter")
        per_n = int(em_trigger_every_n_cycles)
        if per_n > 0 and ((cyc + 1) % per_n) == 0:
            reasons.append("periodic")
        should_kick = len(reasons) > 0
        if should_kick:
            do_update_torque = (cyc % update_every) == 0
            kick, torque_cache, kick_info = _additive_field_and_torque_kick(
                ca,
                shell=z_shell,
                kick_gain=float(additive_kick_gain),
                torque_mix=float(additive_torque_mix),
                kick_max_norm_ang=float(kick_max_norm_ang),
                cached_torque_diag=torque_cache,
                update_torque=bool(do_update_torque),
            )
            kick_info.step = int(cyc)
            kick_info.trigger_reason = ",".join(reasons)
            ca_candidate = _project_bonds(ca + kick, r_min=r_min, r_max=r_max)
            # Safety gate: reject kick if non-finite or causes too-large energy rise.
            if not np.isfinite(ca_candidate).all():
                kick_info.applied = False
                kick_info.trigger_reason = f"{kick_info.trigger_reason},rejected_nonfinite"
            else:
                e_after_kick = float(
                    e_tot_ca_with_bonds(
                        ca_candidate,
                        np.full(n, int(z_shell), dtype=np.int32),
                        **_e_tot_ca_kwargs(
                            dict(base_kwargs.get("energy_kwargs", {}) or {})
                        ),
                    )
                )
                e_before_kick = float(info.final_energy_ev)
                if (not np.isfinite(e_after_kick)) or (
                    e_after_kick - e_before_kick > float(kick_accept_max_energy_rise_ev)
                ):
                    kick_info.applied = False
                    kick_info.trigger_reason = f"{kick_info.trigger_reason},rejected_energy_shock"
                else:
                    ca = ca_candidate
        else:
            kick_info = AdditiveKickInfo(
                step=int(cyc),
                applied=False,
                torque_updated=False,
                trigger_reason="not_triggered",
                kick_norm_mean=0.0,
                kick_norm_max=0.0,
                additive_field_trace_ev=0.0,
                torque_trace_ev=0.0,
            )
        cycle_infos.append(
            OSHAdditiveCycleInfo(
                cycle_index=int(cyc),
                reservoir_before_ev=float(cyc_kwargs["reservoir_init"]),
                reservoir_after_osh_ev=float(reservoir_after_osh),
                reservoir_draw_ev=float(reservoir_draw),
                horizon_pairs_entered_count=int(h_enter),
                osh_displacement_mean_ang=float(disp_mean),
                osh_displacement_max_ang=float(disp_max),
                osh_energy_delta_ev=float(e_delta),
                osh_info=info,
                additive_kick=kick_info,
            )
        )
        energy_converged = (
            prev_energy is not None
            and abs(float(info.final_energy_ev) - float(prev_energy)) <= float(cycle_energy_stop_tol)
        )
        prev_energy = float(info.final_energy_ev)
        if bool(info.settled) and kick_info.kick_norm_mean <= float(cycle_kick_stop_tol) and energy_converged:
            break
    if last_info is None:
        raise RuntimeError("minimize_ca_with_osh_oracle_additive_cycles: no cycles executed")
    return ca, last_info, cycle_infos
