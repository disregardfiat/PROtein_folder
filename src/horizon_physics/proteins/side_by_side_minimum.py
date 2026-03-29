"""
Side-by-side minimum search on the same C-alpha objective.

Runs two minimizers from the same initial coordinates:
- Traditional math: deterministic gradient descent with backtracking.
- OSHoracle: sparse-support minimization on the same objective.

Outputs comparable final energies, structural agreement, and wall-clock timing.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

from .folding_energy import e_tot_ca_with_bonds, grad_full
from .grade_folds import kabsch_superpose, load_ca_from_pdb
from .gradient_descent_folding import _project_bonds
from .osh_oracle_folding import minimize_ca_with_osh_oracle


@dataclass
class MinimizerResult:
    method: str
    wall_seconds: float
    initial_energy_ev: float
    final_energy_ev: float
    energy_drop_ev: float
    n_steps: int
    stop_reason: str


def _energy(ca: np.ndarray, z: np.ndarray, energy_kwargs: Dict[str, Any]) -> float:
    return float(e_tot_ca_with_bonds(ca, z, **energy_kwargs))


def _rmsd_aligned(a: np.ndarray, b: np.ndarray) -> float:
    ref = np.asarray(a, dtype=float)
    pred = np.asarray(b, dtype=float)
    _, _, pred_aligned = kabsch_superpose(ref, pred)
    delta = ref - pred_aligned
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _traditional_minimize_with_backtracking(
    ca_init: np.ndarray,
    z: np.ndarray,
    *,
    n_iter: int,
    step_size: float,
    energy_kwargs: Dict[str, Any],
) -> tuple[np.ndarray, float, int]:
    ca = np.asarray(ca_init, dtype=float).copy()
    e_curr = _energy(ca, z, energy_kwargs)
    accepted = 0
    n_eff = max(1, int(n_iter))
    base_step = float(step_size)
    for _ in range(n_eff):
        g = grad_full(
            ca,
            z,
            include_bonds=True,
            include_horizon=True,
            include_clash=True,
            **energy_kwargs,
        )
        alpha = base_step
        accepted_this_iter = False
        for _ in range(20):
            cand = _project_bonds(ca - alpha * g, r_min=2.5, r_max=6.0)
            e1 = _energy(cand, z, energy_kwargs)
            if e1 <= e_curr:
                ca = cand
                e_curr = e1
                accepted += 1
                accepted_this_iter = True
                break
            alpha *= 0.5
        if not accepted_this_iter:
            continue
    return ca, e_curr, accepted


def _oracle_minimize(
    ca_init: np.ndarray,
    *,
    z_shell: int,
    n_iter: int,
    step_size: float,
    gate_mix: float,
    quantile: float,
    seed: Optional[int],
    energy_kwargs: Dict[str, Any],
):
    return minimize_ca_with_osh_oracle(
        np.asarray(ca_init, dtype=float).copy(),
        z_shell=int(z_shell),
        n_iter=int(n_iter),
        step_size=float(step_size),
        gate_mix=float(gate_mix),
        amp_threshold_quantile=float(quantile),
        random_seed=seed,
        # Keep objective/path comparable to traditional descent:
        use_harmonic_metropolis=False,
        use_energy_reservoir=False,
        strict_descent_budget_mode=True,
        harmonic_step_anneal=False,
        ansatz_depth=1,
        use_local_rapidity_translation=False,
        use_contact_reflectors=False,
        use_resonance_multiplier=False,
        use_mode_shape_participation=False,
        use_terminus_gradient_boost=False,
        energy_kwargs=energy_kwargs,
    )


def _oracle_minimize_with_step_ladder(
    ca_init: np.ndarray,
    *,
    z_shell: int,
    n_iter: int,
    step_size: float,
    gate_mix: float,
    quantile: float,
    seed: Optional[int],
    energy_kwargs: Dict[str, Any],
    step_factors: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625),
):
    best_ca = np.asarray(ca_init, dtype=float).copy()
    best_info = None
    best_e = float("inf")
    for fac in step_factors:
        ca_try, info_try = _oracle_minimize(
            ca_init,
            z_shell=int(z_shell),
            n_iter=int(n_iter),
            step_size=float(step_size) * float(fac),
            gate_mix=float(gate_mix),
            quantile=float(quantile),
            seed=seed,
            energy_kwargs=energy_kwargs,
        )
        e_try = _energy(ca_try, np.full(ca_try.shape[0], int(z_shell), dtype=np.int32), energy_kwargs)
        if e_try < best_e:
            best_e = e_try
            best_ca = ca_try
            best_info = info_try
        if int(info_try.accepted_steps) > 0:
            return ca_try, info_try, float(fac), best_e
    return best_ca, best_info, float(step_factors[-1]), best_e


def run_side_by_side_minimum(
    ca_init: np.ndarray,
    *,
    z_shell: int = 6,
    energy_kwargs: Optional[Dict[str, Any]] = None,
    traditional_iters: int = 300,
    traditional_step_size: float = 0.02,
    osh_iters: int = 300,
    osh_step_size: float = 0.02,
    osh_gate_mix: float = 0.0,
    osh_quantile: float = 0.0,
    osh_seed: Optional[int] = 123,
    ensure_progress: bool = True,
    max_restarts: int = 4,
    restart_jitter_sigma_ang: float = 0.75,
    same_energy_tol_ev: float = 1e-3,
    same_rmsd_tol_ang: float = 0.05,
) -> Dict[str, Any]:
    ca0 = np.asarray(ca_init, dtype=float).copy()
    if ca0.ndim != 2 or ca0.shape[1] != 3:
        raise ValueError("ca_init must be shape (n, 3)")
    n = int(ca0.shape[0])
    if n < 2:
        raise ValueError("Need at least two residues for minimization.")
    z = np.full(n, int(z_shell), dtype=np.int32)
    e_kw: Dict[str, Any] = dict(energy_kwargs or {})
    e0 = _energy(ca0, z, e_kw)

    rng = np.random.default_rng(osh_seed)

    t0 = time.perf_counter()
    ca_trad = ca0.copy()
    e_trad = e0
    trad_accept = 0
    trad_restart_used = 0
    n_trad = max(1, int(traditional_iters))
    ca_trad, e_trad, trad_accept = _traditional_minimize_with_backtracking(
        ca_trad,
        z,
        n_iter=n_trad,
        step_size=float(traditional_step_size),
        energy_kwargs=e_kw,
    )
    if bool(ensure_progress) and trad_accept == 0:
        for rr in range(max(0, int(max_restarts))):
            jitter = rng.normal(0.0, float(restart_jitter_sigma_ang), size=ca0.shape)
            trial_ca = ca0 + jitter
            trial_ca, trial_e, trial_accept = _traditional_minimize_with_backtracking(
                trial_ca,
                z,
                n_iter=n_trad,
                step_size=float(traditional_step_size),
                energy_kwargs=e_kw,
            )
            if trial_accept > 0 and trial_e <= e_trad:
                ca_trad, e_trad, trad_accept = trial_ca, trial_e, trial_accept
                trad_restart_used = rr + 1
                break
    trad_wall = float(time.perf_counter() - t0)

    traditional = MinimizerResult(
        method="traditional_gradient_descent",
        wall_seconds=trad_wall,
        initial_energy_ev=e0,
        final_energy_ev=e_trad,
        energy_drop_ev=float(e0 - e_trad),
        n_steps=int(n_trad),
        stop_reason=f"accepted_steps={trad_accept}; restart_used={trad_restart_used}",
    )

    t1 = time.perf_counter()
    ca_osh, info_osh, osh_step_factor_used, e_osh_best = _oracle_minimize_with_step_ladder(
        ca0,
        z_shell=int(z_shell),
        n_iter=int(osh_iters),
        step_size=float(osh_step_size),
        gate_mix=float(osh_gate_mix),
        quantile=float(osh_quantile),
        seed=osh_seed,
        energy_kwargs=e_kw,
    )
    osh_restart_used = 0
    if bool(ensure_progress) and int(info_osh.accepted_steps) == 0:
        for rr in range(max(0, int(max_restarts))):
            jitter = rng.normal(0.0, float(restart_jitter_sigma_ang), size=ca0.shape)
            trial_ca, trial_info, trial_fac, trial_e = _oracle_minimize_with_step_ladder(
                ca0 + jitter,
                z_shell=int(z_shell),
                n_iter=int(osh_iters),
                step_size=float(osh_step_size),
                gate_mix=float(osh_gate_mix),
                quantile=float(osh_quantile),
                seed=(None if osh_seed is None else int(osh_seed) + rr + 1),
                energy_kwargs=e_kw,
            )
            if int(trial_info.accepted_steps) > 0 and trial_e <= e_osh_best:
                ca_osh, info_osh = trial_ca, trial_info
                osh_step_factor_used = trial_fac
                e_osh_best = trial_e
                osh_restart_used = rr + 1
                break
    osh_wall = float(time.perf_counter() - t1)
    e_osh = _energy(ca_osh, z, e_kw)

    osh = MinimizerResult(
        method="osh_oracle_sparse",
        wall_seconds=osh_wall,
        initial_energy_ev=e0,
        final_energy_ev=e_osh,
        energy_drop_ev=float(e0 - e_osh),
        n_steps=int(info_osh.iterations_executed),
        stop_reason=(
            f"{str(info_osh.stop_reason)}; accepted_steps={int(info_osh.accepted_steps)}; "
            f"restart_used={osh_restart_used}; step_factor_used={osh_step_factor_used:.5f}"
        ),
    )

    energy_delta = float(abs(e_trad - e_osh))
    final_rmsd = _rmsd_aligned(ca_trad, ca_osh)
    same_minimum = bool(
        energy_delta <= float(same_energy_tol_ev) and final_rmsd <= float(same_rmsd_tol_ang)
    )

    return {
        "n_residues": n,
        "z_shell": int(z_shell),
        "objective": "e_tot_ca_with_bonds + projected Cα bond constraints",
        "energy_kwargs": e_kw,
        "ensure_progress": bool(ensure_progress),
        "max_restarts": int(max_restarts),
        "restart_jitter_sigma_ang": float(restart_jitter_sigma_ang),
        "same_minimum_thresholds": {
            "energy_tol_ev": float(same_energy_tol_ev),
            "rmsd_tol_ang": float(same_rmsd_tol_ang),
        },
        "traditional": asdict(traditional),
        "osh_oracle": asdict(osh),
        "agreement": {
            "final_energy_delta_ev": energy_delta,
            "final_ca_rmsd_ang_aligned": final_rmsd,
            "same_minimum": same_minimum,
        },
    }


def _parse_energy_kwargs(raw: Optional[str]) -> Dict[str, Any]:
    if raw is None:
        return {}
    s = raw.strip()
    if not s:
        return {}
    data = json.loads(s)
    if not isinstance(data, dict):
        raise ValueError("--energy-kwargs-json must decode to a JSON object")
    return dict(data)


def cli_main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run traditional gradient descent and OSHoracle side-by-side on one target, "
            "using the same starting C-alpha coordinates and objective."
        )
    )
    ap.add_argument("--target-pdb", required=True, help="Input PDB path used as initial Cα target.")
    ap.add_argument(
        "--out-json",
        default=None,
        help="Optional path to write JSON results (default: alongside target with .side_by_side_min.json).",
    )
    ap.add_argument("--z-shell", type=int, default=6, help="Shell-mass proxy for all residues.")
    ap.add_argument("--traditional-iters", type=int, default=300)
    ap.add_argument("--traditional-step-size", type=float, default=0.02)
    ap.add_argument("--osh-iters", type=int, default=300)
    ap.add_argument("--osh-step-size", type=float, default=0.02)
    ap.add_argument("--osh-gate-mix", type=float, default=0.0)
    ap.add_argument("--osh-quantile", type=float, default=0.0)
    ap.add_argument("--osh-seed", type=int, default=123)
    ap.add_argument(
        "--no-ensure-progress",
        action="store_true",
        help="Disable auto-restart with jitter when a method accepts zero moves.",
    )
    ap.add_argument(
        "--max-restarts",
        type=int,
        default=4,
        help="Maximum auto-restarts per method when ensure-progress is enabled.",
    )
    ap.add_argument(
        "--restart-jitter-sigma-ang",
        type=float,
        default=0.75,
        help="Gaussian jitter sigma (Å) used for restart attempts.",
    )
    ap.add_argument(
        "--same-energy-tol-ev",
        type=float,
        default=1e-3,
        help="Energy threshold below which both results are treated as same minimum.",
    )
    ap.add_argument(
        "--same-rmsd-tol-ang",
        type=float,
        default=0.05,
        help="Aligned Cα RMSD threshold below which both results are treated as same minimum.",
    )
    ap.add_argument(
        "--energy-kwargs-json",
        default=None,
        help='Optional JSON object forwarded to e_tot_ca_with_bonds/grad_full (example: \'{"em_scale": 0.5}\').',
    )
    ap.add_argument(
        "--init-noise-sigma-ang",
        type=float,
        default=0.0,
        help="Optional Gaussian noise (Å) added to initial target coordinates for stress testing.",
    )
    args = ap.parse_args(argv)

    target_path = os.path.abspath(str(args.target_pdb))
    ca, _ = load_ca_from_pdb(target_path)
    ca0 = np.asarray(ca, dtype=float)
    if float(args.init_noise_sigma_ang) > 0.0:
        rng = np.random.default_rng(int(args.osh_seed))
        ca0 = ca0 + rng.normal(0.0, float(args.init_noise_sigma_ang), size=ca0.shape)

    out_json = args.out_json
    if out_json is None:
        out_json = f"{target_path}.side_by_side_min.json"
    out_json = os.path.abspath(str(out_json))
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    out = run_side_by_side_minimum(
        ca0,
        z_shell=int(args.z_shell),
        energy_kwargs=_parse_energy_kwargs(args.energy_kwargs_json),
        traditional_iters=int(args.traditional_iters),
        traditional_step_size=float(args.traditional_step_size),
        osh_iters=int(args.osh_iters),
        osh_step_size=float(args.osh_step_size),
        osh_gate_mix=float(args.osh_gate_mix),
        osh_quantile=float(args.osh_quantile),
        osh_seed=int(args.osh_seed),
        ensure_progress=not bool(args.no_ensure_progress),
        max_restarts=int(args.max_restarts),
        restart_jitter_sigma_ang=float(args.restart_jitter_sigma_ang),
        same_energy_tol_ev=float(args.same_energy_tol_ev),
        same_rmsd_tol_ang=float(args.same_rmsd_tol_ang),
    )
    out["target_pdb"] = target_path
    out["init_noise_sigma_ang"] = float(args.init_noise_sigma_ang)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    t = out["traditional"]
    q = out["osh_oracle"]
    a = out["agreement"]
    print(f"WROTE {out_json}")
    print(
        "{}  | E_final={:.6f} eV | wall={:.3f}s | steps={}".format(
            str(t["method"]).ljust(27),
            float(t["final_energy_ev"]),
            float(t["wall_seconds"]),
            int(t["n_steps"]),
        )
    )
    print(
        "{}  | E_final={:.6f} eV | wall={:.3f}s | steps={}".format(
            str(q["method"]).ljust(27),
            float(q["final_energy_ev"]),
            float(q["wall_seconds"]),
            int(q["n_steps"]),
        )
    )
    print(
        "agreement          | dE={:.6e} eV | aligned Cα RMSD={:.6f} Å | same_minimum={}".format(
            float(a["final_energy_delta_ev"]),
            float(a["final_ca_rmsd_ang_aligned"]),
            bool(a["same_minimum"]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())

