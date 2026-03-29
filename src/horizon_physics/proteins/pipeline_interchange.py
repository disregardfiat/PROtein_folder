"""
Interchangeable folding pipeline primitives.

This module provides a portable fold-state object plus pluggable stage callables so
different algorithms can be composed and compared with a shared data container.

Design goals:
- One state object passed between stages.
- Tunnel-first orchestration helper.
- Easy wrapping of existing minimizer functions (no rewrite of core physics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .casp_submission import _place_full_backbone
from .folding_energy import e_tot_ca_with_bonds
from .full_protein_minimizer import full_chain_to_pdb, minimize_full_chain
from .gradient_descent_folding import _project_bonds
from .grade_folds import load_ca_and_sequence_from_pdb
from .lean_ribosome_tunnel_pipeline import fold_lean_ribosome_tunnel
from .osh_oracle_folding import minimize_ca_with_osh_oracle

BackboneAtoms = List[Tuple[str, np.ndarray]]
StageFn = Callable[["FoldState"], "FoldState"]


@dataclass
class FoldState:
    """Portable structure/state container used by interchangeable folding stages."""

    sequence: str
    ca_positions: Optional[np.ndarray] = None
    backbone_atoms: Optional[BackboneAtoms] = None
    raw_result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    stage_history: List[Dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def from_sequence(sequence: str) -> "FoldState":
        seq = "".join(c for c in str(sequence).strip().upper() if c.isalpha())
        if not seq:
            raise ValueError("FoldState.from_sequence: empty sequence.")
        return FoldState(sequence=seq)

    @staticmethod
    def from_pdb(path: str) -> "FoldState":
        ca, seq = load_ca_and_sequence_from_pdb(path)
        if not seq:
            raise ValueError(f"FoldState.from_pdb: no CA sequence found in {path}.")
        return FoldState(sequence=seq, ca_positions=np.asarray(ca, dtype=float), metadata={"source_pdb": path})

    def to_result_dict(self) -> Dict[str, Any]:
        """Return a result-like dict compatible with ``full_chain_to_pdb``."""
        bb = self.backbone_atoms
        if bb is None:
            if self.ca_positions is None:
                raise ValueError("FoldState.to_result_dict: no backbone_atoms or ca_positions available.")
            bb = _place_full_backbone(np.asarray(self.ca_positions, dtype=float), self.sequence)
        return {
            "sequence": self.sequence,
            "n_res": len(self.sequence),
            "backbone_atoms": bb,
            "include_sidechains": False,
        }

    def to_pdb(self, chain_id: str = "A") -> str:
        return full_chain_to_pdb(self.to_result_dict(), chain_id=chain_id)


def _state_with_result(
    state: FoldState,
    *,
    stage_name: str,
    result: Dict[str, Any],
    stage_config: Optional[Dict[str, Any]] = None,
) -> FoldState:
    ca = np.asarray(result.get("ca_min"), dtype=float) if result.get("ca_min") is not None else state.ca_positions
    bb = result.get("backbone_atoms")
    entry = {
        "stage": stage_name,
        "E_ca_final": result.get("E_ca_final"),
        "E_backbone_final": result.get("E_backbone_final"),
        "info": result.get("info"),
        "config": stage_config or {},
    }
    return FoldState(
        sequence=state.sequence,
        ca_positions=np.asarray(ca, dtype=float) if ca is not None else None,
        backbone_atoms=bb if bb is not None else state.backbone_atoms,
        raw_result=result,
        metadata=dict(state.metadata),
        stage_history=[*state.stage_history, entry],
    )


def _state_with_ca(
    state: FoldState,
    *,
    stage_name: str,
    ca_positions: np.ndarray,
    info: Optional[Dict[str, Any]] = None,
    stage_config: Optional[Dict[str, Any]] = None,
) -> FoldState:
    ca = np.asarray(ca_positions, dtype=float)
    bb = _place_full_backbone(ca, state.sequence)
    entry = {
        "stage": stage_name,
        "E_ca_final": None,
        "E_backbone_final": None,
        "info": info or {},
        "config": stage_config or {},
    }
    return FoldState(
        sequence=state.sequence,
        ca_positions=ca,
        backbone_atoms=bb,
        raw_result=state.raw_result,
        metadata=dict(state.metadata),
        stage_history=[*state.stage_history, entry],
    )


def make_minimize_stage(name: str, **minimize_kwargs: Any) -> StageFn:
    """
    Build a stage around ``minimize_full_chain``.

    The current state's Cα positions are used as ``ca_init`` when shape matches the sequence.
    """

    def _run(state: FoldState) -> FoldState:
        ca_init = None
        if state.ca_positions is not None:
            ca = np.asarray(state.ca_positions, dtype=float)
            if ca.shape == (len(state.sequence), 3):
                ca_init = ca
        kw = dict(minimize_kwargs)
        result = minimize_full_chain(state.sequence, ca_init=ca_init, **kw)
        return _state_with_result(state, stage_name=name, result=result, stage_config=kw)

    return _run


def make_force_carrier_stage(
    name: str = "force_carrier_torque",
    *,
    n_cycles: int = 6,
    translation_step: float = 0.35,
    decay_span: float = 0.25,
    back_propagation_scale: float = 0.35,
    torque_gain_weight: float = 0.0,
    s2_sin_exponent: float = 1.0,
    dt_eff: float = 1.0,
    epsilon_guard: float = 1e-9,
    improve_tol: float = 1e-6,
    r_bond_min: float = 2.5,
    r_bond_max: float = 6.0,
    energy_kwargs: Optional[Dict[str, Any]] = None,
) -> StageFn:
    """
    Build a terminus-translation force-carrier stage with forward + back propagation.

    Each cycle proposes lever-like translations at N/C termini across ±x/±y/±z, propagates
    displacement along the chain with an S2 sinusoidal envelope and exponential attenuation, adds
    a smaller back-propagated amplitude scaled by β, and accepts the best candidate by
    energy/objective.
    """

    dirs = np.array(
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
    e_kw = dict(energy_kwargs or {})
    span = max(1e-6, float(decay_span))
    dt = max(1e-9, float(dt_eff))
    eps = max(1e-12, float(epsilon_guard))
    beta = float(back_propagation_scale)
    lam = float(torque_gain_weight)
    step = max(0.0, float(translation_step))

    # Faithful to Lean: envelopeOrder p := max 1 (Int.toNat (floor p)).
    p = max(0.0, float(s2_sin_exponent))
    envelope_order = max(1, int(np.floor(p)))

    def _d_norm(n: int, i: np.ndarray, j: int) -> np.ndarray:
        d_raw = np.abs(i - float(j)) / max(1.0, float(n - 1))
        return np.minimum(1.0, d_raw)

    def _carrier_amplitude(n: int, src: int, j: np.ndarray) -> np.ndarray:
        """
        Lean-aligned `carrierAmplitude step span p n src j` over a vector of residue indices `j`.
        """
        i = j.astype(float)
        d_norm = _d_norm(n, i, src)
        decay = np.exp(-(d_norm) / span)
        theta = (np.pi / 2.0) * (1.0 - d_norm)
        sin_env = np.sin(theta) ** envelope_order
        sin_env = np.maximum(0.0, sin_env)
        return step * decay * sin_env

    def _amp_forward(n: int, src: int) -> np.ndarray:
        j = np.arange(n, dtype=float)
        return _carrier_amplitude(n, src, j)

    def _mean_speed_proxy_from_abs_disp(abs_disp_scalar: float, dt_: float, eps_: float) -> float:
        # Lean: meanSpeedProxy disp dt ε = |disp| / max ε dt
        return float(abs_disp_scalar) / max(float(eps_), float(dt_))

    def _peak_speed_proxy_from_abs_disp(abs_disp_scalar_peak: float, dt_: float, eps_: float) -> float:
        return float(abs_disp_scalar_peak) / max(float(eps_), float(dt_))

    def _kinetic_proxy(m_eff: float, speed: float) -> float:
        # Lean: kineticProxy mEff speed = (max 0 mEff) * speed^2 / 2
        m_eff = float(m_eff)
        s = max(0.0, float(speed))
        return 0.5 * max(0.0, m_eff) * (s ** 2)

    def _barrier_speed_proxy(delta_E: float, m_eff: float) -> float:
        # Lean: barrierSpeedProxy ΔE mEff ε = sqrt(2 * max 0 ΔE / max ε mEff)
        num = 2.0 * max(0.0, float(delta_E))
        den = max(float(eps), float(m_eff))
        return float(np.sqrt(max(0.0, num / max(1e-12, den))))

    def _whip_proxy_from_amp_scalar(dispF_scalar: float, dispB_scalar: float) -> float:
        # Lean: whipProxy dispF dispB = |dispF * dispB|
        return float(abs(float(dispF_scalar) * float(dispB_scalar)))

    def _objective_energy_minus_whip(E: float, whip: float) -> float:
        # Lean objective: E - λ * whip, with λ >= 0 in guidance theorems.
        return float(E) - lam * float(whip)

    def _run(state: FoldState) -> FoldState:
        if state.ca_positions is None:
            if state.backbone_atoms is not None and len(state.backbone_atoms) == 4 * len(state.sequence):
                ca = np.array([state.backbone_atoms[4 * i + 1][1] for i in range(len(state.sequence))], dtype=float)
            else:
                raise ValueError("force carrier stage needs CA coordinates in state.")
        else:
            ca = np.asarray(state.ca_positions, dtype=float).copy()

        n = ca.shape[0]
        if n < 2:
            return _state_with_ca(
                state,
                stage_name=name,
                ca_positions=ca,
                info={"message": "force carrier skipped (n<2)"},
                stage_config={"n_cycles": n_cycles},
            )

        z = np.full(n, 6, dtype=np.int32)
        accepted = 0
        cycle_metrics: List[Dict[str, float]] = []
        for _ in range(max(0, int(n_cycles))):
            ca_prev = ca
            e_curr = float(e_tot_ca_with_bonds(ca_prev, z, **e_kw))
            best_obj = e_curr
            best_ca = ca_prev
            best_whip = 0.0
            best_source = 0

            improved = False
            for source_idx in (0, n - 1):  # N-terminus, C-terminus
                amp_fwd = _amp_forward(n, source_idx)
                amp_bwd = beta * amp_fwd
                amp_net = amp_fwd - amp_bwd
                dispF_scalar = float(np.mean(np.abs(amp_fwd)))
                dispB_scalar = float(np.mean(np.abs(amp_bwd)))
                whip = _whip_proxy_from_amp_scalar(dispF_scalar, dispB_scalar)

                for dvec in dirs:
                    disp = amp_net[:, None] * dvec[None, :]
                    cand = _project_bonds(ca_prev + disp, r_min=r_bond_min, r_max=r_bond_max)
                    E = float(e_tot_ca_with_bonds(cand, z, **e_kw))
                    obj = _objective_energy_minus_whip(E, whip)
                    if obj + float(improve_tol) < best_obj:
                        best_obj = obj
                        best_ca = cand
                        best_whip = whip
                        best_source = source_idx
                        improved = True

            if not improved:
                break

            e_next = float(e_tot_ca_with_bonds(best_ca, z, **e_kw))
            de = float(e_curr - e_next)
            actual_disp = best_ca - ca_prev
            abs_disp = np.linalg.norm(actual_disp, axis=1)
            abs_disp_scalar = float(np.mean(abs_disp))
            abs_disp_peak = float(np.max(abs_disp))

            m_eff = float(n)
            mean_speed = _mean_speed_proxy_from_abs_disp(abs_disp_scalar, dt, eps)
            peak_speed = _peak_speed_proxy_from_abs_disp(abs_disp_peak, dt, eps)
            kinetic_proxy = _kinetic_proxy(m_eff, mean_speed)
            barrier_speed = _barrier_speed_proxy(de, m_eff)

            # Whip proxy here is amplitude-based (Lean scalar proxy), not projection-adjusted.
            cycle_metrics.append(
                {
                    "delta_energy": de,
                    "mean_speed": mean_speed,
                    "peak_speed": peak_speed,
                    "barrier_speed": barrier_speed,
                    "kinetic_proxy": kinetic_proxy,
                    "inertial_whip_proxy": best_whip,
                    "accepted_source_idx": float(best_source),
                }
            )

            ca = best_ca
            accepted += 1

        avg_mean_speed = float(np.mean([m["mean_speed"] for m in cycle_metrics])) if cycle_metrics else 0.0
        avg_barrier_speed = float(np.mean([m["barrier_speed"] for m in cycle_metrics])) if cycle_metrics else 0.0
        avg_whip = float(np.mean([m["inertial_whip_proxy"] for m in cycle_metrics])) if cycle_metrics else 0.0
        info = {
            "message": "force carrier complete",
            "n_cycles": int(n_cycles),
            "accepted_cycles": int(accepted),
            "translation_step": float(translation_step),
            "decay_span": float(decay_span),
            "back_propagation_scale": float(back_propagation_scale),
            "torque_gain_weight": float(torque_gain_weight),
            "s2_sin_exponent": float(s2_sin_exponent),
            "dt_eff": float(dt_eff),
            "epsilon_guard": float(epsilon_guard),
            "s2_envelope_order": float(envelope_order),
            "avg_mean_speed": avg_mean_speed,
            "avg_barrier_speed": avg_barrier_speed,
            "avg_inertial_whip_proxy": avg_whip,
            "cycle_metrics": cycle_metrics,
        }
        return _state_with_ca(state, stage_name=name, ca_positions=ca, info=info, stage_config=info)

    return _run


def make_lean_tunnel_stage(name: str = "lean_tunnel_fold", **lean_kwargs: Any) -> StageFn:
    """Build a stage around ``fold_lean_ribosome_tunnel``."""

    def _run(state: FoldState) -> FoldState:
        out = fold_lean_ribosome_tunnel(state.sequence, **lean_kwargs)
        raw = out.raw_result
        cfg = dict(lean_kwargs)
        cfg["em_scale"] = out.em_scale
        cfg["epsilon_r"] = out.epsilon_r
        return _state_with_result(state, stage_name=name, result=raw, stage_config=cfg)

    return _run


def make_osh_oracle_stage(
    name: str = "osh_oracle_sparse_refine",
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
    harmonic_step_anneal: bool = True,
    harmonic_base_temp: float = 1.0,
    harmonic_min_temp: float = 1e-4,
    stop_when_settled: bool = False,
    settle_window: int = 20,
    settle_energy_tol: float = 1e-3,
    settle_step_tol: float = 3e-4,
    settle_min_iter: int = 30,
    ligation_pairs: Optional[List[Tuple[int, int]]] = None,
    auto_detect_cys_ligation: bool = False,
    ligation_detect_max_dist: float = 6.5,
    ligation_r_eq: float = 3.8,
    ligation_r_min: float = 2.5,
    ligation_r_max: float = 6.0,
    ligation_k_bond: float = 60.0,
    r_bond_min: float = 2.5,
    r_bond_max: float = 6.0,
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
    omega_refresh_period: int = 0,
    use_mode_shape_participation: bool = False,
    mode_shape_fixed_end: str = "right",
    mode_shape_factor_min: float = 0.5,
    mode_shape_factor_max: float = 1.2,
    use_terminus_gradient_boost: bool = False,
    terminus_gradient_boost: float = 1.28,
    terminus_gradient_transition_width: int = 8,
    terminus_gradient_core_scale: float = 1.0,
    contact_terminus_window: int = 0,
    contact_terminus_score_scale: float = 1.0,
    energy_kwargs: Optional[Dict[str, Any]] = None,
    use_hqiv_native_gate: bool = False,
    hqiv_reference_m: int = 4,
) -> StageFn:
    """
    Build a stage around the OSHoracle-inspired sparse support optimizer.

    Requires Cα coordinates in state (or backbone atoms from which Cα can be extracted).
    """

    e_kw = dict(energy_kwargs or {})

    def _run(state: FoldState) -> FoldState:
        if state.ca_positions is None:
            if state.backbone_atoms is not None and len(state.backbone_atoms) == 4 * len(state.sequence):
                ca0 = np.array([state.backbone_atoms[4 * i + 1][1] for i in range(len(state.sequence))], dtype=float)
            else:
                raise ValueError("OSHoracle stage needs CA coordinates in state.")
        else:
            ca0 = np.asarray(state.ca_positions, dtype=float)

        ca1, info_obj = minimize_ca_with_osh_oracle(
            ca0,
            z_shell=int(z_shell),
            n_iter=int(n_iter),
            step_size=float(step_size),
            gate_mix=float(gate_mix),
            ansatz_depth=int(ansatz_depth),
            amp_threshold_quantile=float(amp_threshold_quantile),
            flip_amp_delta_eps=float(flip_amp_delta_eps),
            flip_include_sign=bool(flip_include_sign),
            use_harmonic_metropolis=bool(use_harmonic_metropolis),
            harmonic_fd_eps=float(harmonic_fd_eps),
            harmonic_max_dims=int(harmonic_max_dims),
            random_seed=random_seed,
            harmonic_step_anneal=bool(harmonic_step_anneal),
            harmonic_base_temp=float(harmonic_base_temp),
            harmonic_min_temp=float(harmonic_min_temp),
            stop_when_settled=bool(stop_when_settled),
            settle_window=int(settle_window),
            settle_energy_tol=float(settle_energy_tol),
            settle_step_tol=float(settle_step_tol),
            settle_min_iter=int(settle_min_iter),
            sequence=state.sequence,
            ligation_pairs=ligation_pairs,
            auto_detect_cys_ligation=bool(auto_detect_cys_ligation),
            ligation_detect_max_dist=float(ligation_detect_max_dist),
            ligation_r_eq=float(ligation_r_eq),
            ligation_r_min=float(ligation_r_min),
            ligation_r_max=float(ligation_r_max),
            ligation_k_bond=float(ligation_k_bond),
            r_min=float(r_bond_min),
            r_max=float(r_bond_max),
            energy_kwargs=e_kw,
            use_contact_reflectors=bool(use_contact_reflectors),
            contact_min_seq_sep=int(contact_min_seq_sep),
            contact_cutoff_ang=float(contact_cutoff_ang),
            contact_max_reflectors=int(contact_max_reflectors),
            contact_grad_coupling=float(contact_grad_coupling),
            contact_weight_gradient=bool(contact_weight_gradient),
            contact_score_mode=str(contact_score_mode),
            contact_inverse_power=float(contact_inverse_power),
            contact_score_min_dist_ang=float(contact_score_min_dist_ang),
            use_resonance_multiplier=bool(use_resonance_multiplier),
            resonance_terminus_boost=float(resonance_terminus_boost),
            resonance_core_damping=float(resonance_core_damping),
            resonance_transition_width=int(resonance_transition_width),
            resonance_compaction_cutoff_ang=float(resonance_compaction_cutoff_ang),
            resonance_compaction_min_seq_sep=int(resonance_compaction_min_seq_sep),
            omega_refresh_period=int(omega_refresh_period),
            use_mode_shape_participation=bool(use_mode_shape_participation),
            mode_shape_fixed_end=str(mode_shape_fixed_end),
            mode_shape_factor_min=float(mode_shape_factor_min),
            mode_shape_factor_max=float(mode_shape_factor_max),
            use_terminus_gradient_boost=bool(use_terminus_gradient_boost),
            terminus_gradient_boost=float(terminus_gradient_boost),
            terminus_gradient_transition_width=int(terminus_gradient_transition_width),
            terminus_gradient_core_scale=float(terminus_gradient_core_scale),
            contact_terminus_window=int(contact_terminus_window),
            contact_terminus_score_scale=float(contact_terminus_score_scale),
            use_hqiv_native_gate=bool(use_hqiv_native_gate),
            hqiv_reference_m=int(hqiv_reference_m),
        )
        info = {
            "message": "OSHoracle sparse refinement complete",
            "accepted_steps": int(info_obj.accepted_steps),
            "iterations": int(info_obj.iterations),
            "final_energy_ev": float(info_obj.final_energy_ev),
            "last_step_size": float(info_obj.last_step_size),
            "last_flipped_count": int(info_obj.last_flipped_count),
            "z_shell": int(z_shell),
            "gate_mix": float(gate_mix),
            "ansatz_depth": int(ansatz_depth),
            "amp_threshold_quantile": float(amp_threshold_quantile),
            "flip_amp_delta_eps": float(flip_amp_delta_eps),
            "flip_include_sign": bool(flip_include_sign),
            "avg_flipped_count": float(info_obj.avg_flipped_count),
            "natural_harmonic_scale": float(info_obj.natural_harmonic_scale),
            "metropolis_accepts": int(info_obj.metropolis_accepts),
            "iterations_executed": int(info_obj.iterations_executed),
            "stop_reason": str(info_obj.stop_reason),
            "settled": bool(info_obj.settled),
            "n_ligation_pairs": int(len(ligation_pairs or [])),
            "auto_detect_cys_ligation": bool(auto_detect_cys_ligation),
            "contact_reflector_count": int(info_obj.contact_reflector_count),
            "omega_refresh_count": int(info_obj.omega_refresh_count),
            "use_resonance_multiplier": bool(use_resonance_multiplier),
            "use_mode_shape_participation": bool(use_mode_shape_participation),
            "mode_shape_fixed_end": str(mode_shape_fixed_end),
            "use_terminus_gradient_boost": bool(use_terminus_gradient_boost),
            "terminus_gradient_boost": float(terminus_gradient_boost),
            "terminus_gradient_transition_width": int(terminus_gradient_transition_width),
            "contact_terminus_window": int(contact_terminus_window),
            "contact_terminus_score_scale": float(contact_terminus_score_scale),
            "use_hqiv_native_gate": bool(use_hqiv_native_gate),
            "hqiv_reference_m": int(hqiv_reference_m),
        }
        cfg = {
            "z_shell": int(z_shell),
            "n_iter": int(n_iter),
            "step_size": float(step_size),
            "gate_mix": float(gate_mix),
            "ansatz_depth": int(ansatz_depth),
            "amp_threshold_quantile": float(amp_threshold_quantile),
            "flip_amp_delta_eps": float(flip_amp_delta_eps),
            "flip_include_sign": bool(flip_include_sign),
            "use_harmonic_metropolis": bool(use_harmonic_metropolis),
            "harmonic_fd_eps": float(harmonic_fd_eps),
            "harmonic_max_dims": int(harmonic_max_dims),
            "random_seed": random_seed,
            "harmonic_step_anneal": bool(harmonic_step_anneal),
            "harmonic_base_temp": float(harmonic_base_temp),
            "harmonic_min_temp": float(harmonic_min_temp),
            "stop_when_settled": bool(stop_when_settled),
            "settle_window": int(settle_window),
            "settle_energy_tol": float(settle_energy_tol),
            "settle_step_tol": float(settle_step_tol),
            "settle_min_iter": int(settle_min_iter),
            "ligation_pairs": ligation_pairs,
            "auto_detect_cys_ligation": bool(auto_detect_cys_ligation),
            "ligation_detect_max_dist": float(ligation_detect_max_dist),
            "ligation_r_eq": float(ligation_r_eq),
            "ligation_r_min": float(ligation_r_min),
            "ligation_r_max": float(ligation_r_max),
            "ligation_k_bond": float(ligation_k_bond),
            "r_bond_min": float(r_bond_min),
            "r_bond_max": float(r_bond_max),
            "energy_kwargs": e_kw,
            "use_contact_reflectors": bool(use_contact_reflectors),
            "contact_min_seq_sep": int(contact_min_seq_sep),
            "contact_cutoff_ang": float(contact_cutoff_ang),
            "contact_max_reflectors": int(contact_max_reflectors),
            "contact_grad_coupling": float(contact_grad_coupling),
            "contact_weight_gradient": bool(contact_weight_gradient),
            "contact_score_mode": str(contact_score_mode),
            "contact_inverse_power": float(contact_inverse_power),
            "contact_score_min_dist_ang": float(contact_score_min_dist_ang),
            "omega_refresh_period": int(omega_refresh_period),
            "use_mode_shape_participation": bool(use_mode_shape_participation),
            "mode_shape_fixed_end": str(mode_shape_fixed_end),
            "mode_shape_factor_min": float(mode_shape_factor_min),
            "mode_shape_factor_max": float(mode_shape_factor_max),
            "use_terminus_gradient_boost": bool(use_terminus_gradient_boost),
            "terminus_gradient_boost": float(terminus_gradient_boost),
            "terminus_gradient_transition_width": int(terminus_gradient_transition_width),
            "terminus_gradient_core_scale": float(terminus_gradient_core_scale),
            "contact_terminus_window": int(contact_terminus_window),
            "contact_terminus_score_scale": float(contact_terminus_score_scale),
            **info,
        }
        return _state_with_ca(state, stage_name=name, ca_positions=ca1, info=info, stage_config=cfg)

    return _run


def run_pipeline(initial: FoldState, stages: Sequence[StageFn]) -> FoldState:
    """Run a sequence of stages, returning the final fold state."""
    state = initial
    for stage in stages:
        state = stage(state)
    return state


def run_tunnel_first_pipeline(
    initial: FoldState,
    *,
    tunnel_kwargs: Optional[Dict[str, Any]] = None,
    refinement_stages: Optional[Sequence[StageFn]] = None,
) -> FoldState:
    """
    Enforce a tunnel-first flow, then execute interchangeable refinement stages.

    This guarantees the residue chain always passes through a tunnel extrusion/fold step
    before additional algorithmic variants are applied.
    """
    tkw = dict(tunnel_kwargs or {})
    tkw["simulate_ribosome_tunnel"] = True
    tunnel_stage = make_minimize_stage("tunnel_extrusion", **tkw)
    stages: List[StageFn] = [tunnel_stage]
    if refinement_stages:
        stages.extend(list(refinement_stages))
    return run_pipeline(initial, stages)

