"""
Post–tunnel-extrusion refinement: 3D EM lattice relaxation + discrete tree-torque.

Replaces the expensive ``_minimize_bonds_fast`` / L-BFGS (HKE) post-extrusion loop while
**keeping** co-translational tunnel geometry from ``co_translational_minimize``.

- **EM:** ``CoTranslationalAssembler.relax_to_convergence`` with ``alternate_hke=False`` so
  steps use the voxel field + soft bond springs only (no ``grad_full`` Cα HKE step).
- **Tree-torque:** ``run_discrete_refinement`` (EM-guided φ/ψ unlock; optional true Metropolis at ``temperature``).
- **Optional:** ``thermal_gradient_relax_ca`` — a few ``grad_full`` steps with kT-scaled noise on Cα.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .casp_submission import _place_full_backbone


def _default_anneal_schedule_k(*, quick: bool, final_temperature_k: float) -> Tuple[float, ...]:
    """Short cool-down for discrete Metropolis (high → ``final_temperature_k``)."""
    tf = float(max(273.15, final_temperature_k))
    if quick:
        return (min(tf + 22.0, 360.0), tf)
    a1 = min(tf + 34.0, 365.0)
    a2 = min(tf + 18.0, 345.0)
    a3 = min(tf + 7.0, 328.0)
    return (a1, a2, a3, tf)


def _run_discrete_anneal(
    seq: str,
    backbone_start: List[Tuple[str, np.ndarray]],
    *,
    schedule_k: Sequence[float],
    n_steps_total: int,
    phases_cap_total: int,
    quick: bool,
    seed: Optional[int],
) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]], List[Dict[str, object]]]:
    """Several Metropolis discrete passes at decreasing T; returns (ca, backbone, stage_infos)."""
    from .temperature_path_search import run_discrete_refinement

    sched = tuple(float(t) for t in schedule_k)
    if len(sched) < 2:
        sched = _default_anneal_schedule_k(
            quick=quick, final_temperature_k=sched[0] if len(sched) == 1 else 310.0
        )

    n_st = len(sched)
    steps_each = max(24, int(n_steps_total) // n_st)
    phases_each = max(2, int(phases_cap_total) // n_st)

    backbone = list(backbone_start)
    stage_infos: List[Dict[str, object]] = []
    ca_out = np.zeros((0, 3))

    for T in sched:
        ref = run_discrete_refinement(
            seq,
            temperature=float(T),
            n_steps=steps_each,
            initial_backbone_atoms=list(backbone),
            run_until_converged=False,
            max_phases_cap=phases_each,
            assembly_mode=False,
            seed=seed,
            use_metropolis=True,
        )
        backbone = list(ref.backbone_atoms)
        ca_out = np.asarray(ref.ca_min, dtype=float)
        stage_infos.append(
            {
                "temperature_k": float(T),
                "n_accept": ref.n_accept,
                "E_ca_final": float(ref.E_ca_final),
                "phases_ran": len(ref.info.get("phases", [])),
            }
        )
    return ca_out, backbone, stage_infos


def _ca_from_assembler_atoms(atoms, n_res: int) -> np.ndarray:
    ca = np.array([a.pos for a in atoms if a.name == "CA"], dtype=float)
    if ca.shape[0] != n_res:
        raise ValueError(
            f"post_tunnel_em_treetorque: expected {n_res} CA atoms, got {ca.shape[0]}"
        )
    return ca


def refine_ca_post_tunnel_em_treetorque(
    ca_min: np.ndarray,
    sequence: str,
    *,
    quick: bool = False,
    temperature: float = 310.0,
    em_max_steps: Optional[int] = None,
    treetorque_phases: int = 8,
    treetorque_n_steps: int = 200,
    discrete_use_metropolis: bool = False,
    discrete_seed: Optional[int] = None,
    langevin_steps: int = 0,
    langevin_noise_fraction: float = 0.2,
    post_extrusion_anneal: bool = False,
    post_extrusion_anneal_schedule_k: Optional[Union[Tuple[float, ...], List[float]]] = None,
    grad_full_extra_kwargs: Optional[Dict[str, object]] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Return (ca_refined, info_dict). On tree-torque failure, returns EM-only CA and notes error.

    ``discrete_use_metropolis``: thermal acceptance on the discrete φ/ψ grid at ``temperature``.
    ``langevin_steps``: if > 0, run ``thermal_gradient_relax_ca`` after discrete refine (uses
    ``grad_full_extra_kwargs`` e.g. ``em_scale`` from Lean water screening).

    ``post_extrusion_anneal``: if True, run a **short Metropolis anneal** (several temperatures,
    high→low) instead of a single discrete pass; ``discrete_use_metropolis`` is ignored for
    those stages (always thermal). Optional ``post_extrusion_anneal_schedule_k`` overrides
    the default schedule; last stage is clamped toward ``temperature``.
    """
    from .em_field_pipeline import CoTranslationalAssembler

    seq = "".join(c for c in sequence.strip().upper() if c.isalpha())
    n = len(seq)
    if n == 0:
        return np.zeros((0, 3)), {"message": "empty sequence"}
    ca0 = np.asarray(ca_min, dtype=float)
    if ca0.shape != (n, 3):
        raise ValueError(f"refine_ca_post_tunnel_em_treetorque: ca shape {ca0.shape} != ({n}, 3)")

    if em_max_steps is None:
        if quick:
            em_max_steps = int(min(140, 35 + 18 * n))
        else:
            em_max_steps = int(min(520, 90 + 14 * n))

    phases_cap = min(4, treetorque_phases) if quick else int(treetorque_phases)
    field_schedule: Tuple[float, ...] = (3.0, 1.0) if quick else (5.0, 3.0, 1.0, 0.5)
    disp_thr = 0.03 if quick else 0.02
    min_steps = min(15, em_max_steps // 4) if quick else min(25, em_max_steps // 5)

    backbone0: List[Tuple[str, np.ndarray]] = _place_full_backbone(ca0, seq)
    asm = CoTranslationalAssembler(temperature=float(temperature))
    asm.load_from_backbone_atoms(backbone0, seq)
    em_steps = asm.relax_to_convergence(
        max_disp_threshold=disp_thr,
        max_steps=int(em_max_steps),
        min_steps=max(5, min_steps),
        alternate_hke=False,
        field_res_schedule=field_schedule,
    )
    ca_em = _ca_from_assembler_atoms(asm.atoms, n)
    backbone_em = _place_full_backbone(ca_em, seq)

    info: Dict[str, object] = {
        "message": "Post-tunnel: EM field relax (no HKE) + tree-torque",
        "em_relax_steps": em_steps,
        "em_max_steps_cap": em_max_steps,
        "treetorque_phases_cap": phases_cap,
        "treetorque_ok": False,
        "discrete_use_metropolis": bool(discrete_use_metropolis),
        "post_extrusion_anneal": bool(post_extrusion_anneal),
    }

    ca_out = ca_em
    T_final = float(temperature)
    try:
        if post_extrusion_anneal:
            sched: Tuple[float, ...]
            if post_extrusion_anneal_schedule_k is not None and len(post_extrusion_anneal_schedule_k) >= 2:
                sched = tuple(float(x) for x in post_extrusion_anneal_schedule_k)
            else:
                sched = _default_anneal_schedule_k(quick=quick, final_temperature_k=T_final)
            ca_out, _bb_anneal, stage_infos = _run_discrete_anneal(
                seq,
                list(backbone_em),
                schedule_k=sched,
                n_steps_total=int(treetorque_n_steps),
                phases_cap_total=phases_cap,
                quick=quick,
                seed=discrete_seed,
            )
            info["treetorque_ok"] = True
            info["anneal_schedule_k"] = list(sched)
            info["anneal_stages"] = stage_infos
            info["treetorque_accept"] = sum(int(s.get("n_accept", 0) or 0) for s in stage_infos)
            info["treetorque_phases_ran"] = sum(int(s.get("phases_ran", 0) or 0) for s in stage_infos)
            T_final = float(sched[-1])
        else:
            from .temperature_path_search import run_discrete_refinement

            ref = run_discrete_refinement(
                seq,
                temperature=float(temperature),
                n_steps=int(treetorque_n_steps),
                initial_backbone_atoms=list(backbone_em),
                run_until_converged=False,
                max_phases_cap=phases_cap,
                assembly_mode=False,
                seed=discrete_seed,
                use_metropolis=bool(discrete_use_metropolis),
            )
            info["treetorque_ok"] = True
            info["treetorque_accept"] = ref.n_accept
            info["treetorque_phases_ran"] = len(ref.info.get("phases", []))
            ca_out = np.asarray(ref.ca_min, dtype=float)
    except Exception as exc:
        info["treetorque_error"] = str(exc)
        ca_out = ca_em

    n_lv = int(langevin_steps)
    if n_lv > 0:
        try:
            from .full_protein_minimizer import _z_list_ca
            from .gradient_descent_folding import thermal_gradient_relax_ca

            z_ca = _z_list_ca(seq)
            lv_steps = min(n_lv, 40) if quick else n_lv
            ca_out, lv_info = thermal_gradient_relax_ca(
                ca_out,
                z_ca,
                n_steps=lv_steps,
                step_size=0.022 if quick else 0.028,
                temperature_k=T_final,
                reference_temperature_k=310.0,
                noise_fraction=float(langevin_noise_fraction),
                grad_full_extra_kwargs=grad_full_extra_kwargs,
                seed=discrete_seed,
            )
            info["thermal_gradient_relax"] = lv_info
        except Exception as exc:
            info["thermal_gradient_relax_error"] = str(exc)

    return ca_out, info
