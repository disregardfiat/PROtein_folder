"""
Side-by-side: plain Cα descent vs Lean **ProteinResearch** folding update laws.

Lean references (HQIV_LEAN ``Hqiv/ProteinResearch/``):

* ``ProteinFoldingQuantumChemistry.lean`` — site block as trace of per-shell
  ``latticeFullModeEnergy`` (additive HQIV site budget) plus separate pair
  physics in Python as ``e_tot_ca_with_bonds`` / ``grad_full``.
* ``AtomEnergyOSHoracleBridge.lean`` — diagonal site energies
  ``available_modes * (φ(m)/2)``; Python OSH bridge uses the same ladder via
  ``shell_spatial_mode_count`` × ``φ/2`` (see ``osh_oracle_folding``).
* ``ProteinNaturalFolding.lean`` / ``ProteinHKEMinimizer.lean`` —
  ``naturalDisp = η_φ(m) · (−step · ∇E) + carrierDisp`` with proved
  ``η_φ`` constancy at ``referenceM`` (default 4).

This module keeps the **same** objective as ``side_by_side_minimum`` but compares:

1. **traditional_gradient_descent** — backtracking on ``−step ∇E`` (bond projection).
2. **lean_natural_disp** — backtracking on ``step * (η · (−∇E) + ang_mix · ω×(r_i−r̄))``,
   matching the coherent scaling plus a deterministic WHIP-style rotational carrier
   (Lean ``rotationalDisp`` when angular mixing is nonzero).

Diagnostics include **site_energy_trace_ev**: ∑_i latticeFullModeEnergy(z_i), which is
constant for fixed shells but documents the QC layer alongside total energy.
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
from .horizon_qed_bookkeeping import shell_spatial_mode_count
from .hqiv_long_range import phi_of_shell

# Lean ``ProteinHKEMinimizer``: referenceM = 4 in HQIV protein scripts.
REFERENCE_M_LEAN = 4


def eta_mode_phi_constant(reference_m: int = REFERENCE_M_LEAN) -> float:
    """Lean ``etaModePhi_constant``: 1 / ((referenceM + 2) * (referenceM + 1))."""
    rm = int(reference_m)
    return 1.0 / float((rm + 2) * (rm + 1))


def lattice_full_mode_energy_ev(shell: int) -> float:
    """
    Per-shell HQIV site zero-point budget aligned with Python OSH bridge
    (``osh_oracle_folding._lattice_full_mode_energy``).
    """
    m = int(max(0, shell))
    return float(shell_spatial_mode_count(m) * (phi_of_shell(m) / 2.0))


def site_energy_trace_ev(z: np.ndarray) -> float:
    """∑_i latticeFullModeEnergy(z_i) — Lean ``trace (atomSiteEnergyMatrix shell)`` analog."""
    z = np.asarray(z, dtype=np.int32).reshape(-1)
    return float(sum(lattice_full_mode_energy_ev(int(z[i])) for i in range(z.shape[0])))


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


def _omega_unit_from_ca(ca: np.ndarray) -> np.ndarray:
    """Unit axis for ω×r carrier: orthogonal to first peptide direction and global z when possible."""
    ca = np.asarray(ca, dtype=float)
    n = int(ca.shape[0])
    if n >= 2:
        e = ca[1] - ca[0]
        ne = float(np.linalg.norm(e))
        if ne > 1e-9:
            e = e / ne
        else:
            e = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        e = np.array([1.0, 0.0, 0.0], dtype=float)
    zax = np.array([0.0, 0.0, 1.0], dtype=float)
    w = np.cross(e, zax)
    nw = float(np.linalg.norm(w))
    if nw < 1e-9:
        return np.array([0.0, 1.0, 0.0], dtype=float)
    return (w / nw).astype(float)


def _rotational_carrier_disp(ca: np.ndarray, omega_unit: np.ndarray) -> np.ndarray:
    """Per residue ω × (r_i − r̄), matching Lean ``rotationalDisp`` up to dt scaling."""
    ca = np.asarray(ca, dtype=float)
    com = np.mean(ca, axis=0)
    out = np.zeros_like(ca)
    for i in range(ca.shape[0]):
        r = ca[i] - com
        out[i] = np.cross(omega_unit, r)
    return out


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


def _lean_natural_minimize_with_backtracking(
    ca_init: np.ndarray,
    z: np.ndarray,
    *,
    n_iter: int,
    step_size: float,
    eta_mode_phi: float,
    ang_mix: float,
    energy_kwargs: Dict[str, Any],
) -> tuple[np.ndarray, float, int]:
    """
    Backtracking along Lean ``naturalDisp`` direction:
    Δx_i ∝ η·(−∇E_i) + ang_mix·(ω×(r_i−r̄)).
    """
    ca = np.asarray(ca_init, dtype=float).copy()
    e_curr = _energy(ca, z, energy_kwargs)
    accepted = 0
    n_eff = max(1, int(n_iter))
    base_step = float(step_size)
    eta = float(eta_mode_phi)
    mix = float(ang_mix)
    for _ in range(n_eff):
        g = grad_full(
            ca,
            z,
            include_bonds=True,
            include_horizon=True,
            include_clash=True,
            **energy_kwargs,
        )
        omega_u = _omega_unit_from_ca(ca)
        whip = _rotational_carrier_disp(ca, omega_u)
        # Combined search direction (Lean naturalDisp structure; scalar step linesearch).
        direction = -eta * g + mix * whip
        alpha = base_step
        accepted_this_iter = False
        for _ in range(20):
            cand = _project_bonds(ca + alpha * direction, r_min=2.5, r_max=6.0)
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


def run_side_by_side_lean_qc(
    ca_init: np.ndarray,
    *,
    z_shell: int = 6,
    energy_kwargs: Optional[Dict[str, Any]] = None,
    traditional_iters: int = 300,
    traditional_step_size: float = 0.02,
    lean_iters: int = 300,
    lean_step_size: float = 0.02,
    reference_m: int = REFERENCE_M_LEAN,
    ang_mix: float = 0.12,
    same_energy_tol_ev: float = 1e-3,
    same_rmsd_tol_ang: float = 0.05,
    ensure_progress: bool = True,
    max_restarts: int = 4,
    restart_jitter_sigma_ang: float = 0.75,
    lean_seed: Optional[int] = 123,
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
    trace0 = site_energy_trace_ev(z)
    eta = eta_mode_phi_constant(int(reference_m))

    rng = np.random.default_rng(lean_seed)

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
            trial_ca, trial_e, trial_accept = _traditional_minimize_with_backtracking(
                ca0 + jitter,
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
    ca_lean = ca0.copy()
    e_lean = e0
    lean_accept = 0
    lean_restart_used = 0
    n_lean = max(1, int(lean_iters))
    ca_lean, e_lean, lean_accept = _lean_natural_minimize_with_backtracking(
        ca_lean,
        z,
        n_iter=n_lean,
        step_size=float(lean_step_size),
        eta_mode_phi=eta,
        ang_mix=float(ang_mix),
        energy_kwargs=e_kw,
    )
    if bool(ensure_progress) and lean_accept == 0:
        for rr in range(max(0, int(max_restarts))):
            seed_rr = None if lean_seed is None else int(lean_seed) + 901 + rr
            rng_rr = np.random.default_rng(seed_rr)
            jitter = rng_rr.normal(0.0, float(restart_jitter_sigma_ang), size=ca0.shape)
            trial_ca, trial_e, trial_accept = _lean_natural_minimize_with_backtracking(
                ca0 + jitter,
                z,
                n_iter=n_lean,
                step_size=float(lean_step_size),
                eta_mode_phi=eta,
                ang_mix=float(ang_mix),
                energy_kwargs=e_kw,
            )
            if trial_accept > 0 and trial_e <= e_lean:
                ca_lean, e_lean, lean_accept = trial_ca, trial_e, trial_accept
                lean_restart_used = rr + 1
                break
    lean_wall = float(time.perf_counter() - t1)
    e_lean = _energy(ca_lean, z, e_kw)

    lean_natural = MinimizerResult(
        method="lean_natural_disp",
        wall_seconds=lean_wall,
        initial_energy_ev=e0,
        final_energy_ev=e_lean,
        energy_drop_ev=float(e0 - e_lean),
        n_steps=int(n_lean),
        stop_reason=(
            f"accepted_steps={lean_accept}; restart_used={lean_restart_used}; "
            f"eta_mode_phi={eta:.8g}; ang_mix={float(ang_mix):.6g}; reference_m={int(reference_m)}"
        ),
    )

    energy_delta = float(abs(e_trad - e_lean))
    final_rmsd = _rmsd_aligned(ca_trad, ca_lean)
    same_minimum = bool(
        energy_delta <= float(same_energy_tol_ev) and final_rmsd <= float(same_rmsd_tol_ang)
    )

    return {
        "n_residues": n,
        "z_shell": int(z_shell),
        "lean_reference_m": int(reference_m),
        "eta_mode_phi": float(eta),
        "ang_mix": float(ang_mix),
        "objective": "e_tot_ca_with_bonds + projected Cα bond constraints",
        "site_energy_trace_ev": float(trace0),
        "lean_references": {
            "quantum_chemistry_contract": "Hqiv/ProteinResearch/ProteinFoldingQuantumChemistry.lean",
            "site_matrix_bridge": "Hqiv/ProteinResearch/AtomEnergyOSHoracleBridge.lean",
            "natural_update": "Hqiv/ProteinResearch/ProteinNaturalFolding.lean",
            "hke_minimizer_spec": "Hqiv/ProteinResearch/ProteinHKEMinimizer.lean",
        },
        "energy_kwargs": e_kw,
        "ensure_progress": bool(ensure_progress),
        "max_restarts": int(max_restarts),
        "restart_jitter_sigma_ang": float(restart_jitter_sigma_ang),
        "same_minimum_thresholds": {
            "energy_tol_ev": float(same_energy_tol_ev),
            "rmsd_tol_ang": float(same_rmsd_tol_ang),
        },
        "traditional": asdict(traditional),
        "lean_natural_disp": asdict(lean_natural),
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
            "Compare traditional Cα gradient descent vs Lean naturalDisp-style updates "
            "on the same objective (ProteinResearch QC contract)."
        )
    )
    ap.add_argument("--target-pdb", required=True, help="Input PDB path used as initial Cα target.")
    ap.add_argument(
        "--out-json",
        default=None,
        help="Optional path to write JSON results (default: alongside target with .side_by_side_lean_qc.json).",
    )
    ap.add_argument("--z-shell", type=int, default=6, help="Shell-mass proxy for all residues.")
    ap.add_argument("--reference-m", type=int, default=REFERENCE_M_LEAN, help="Lean referenceM for η_φ.")
    ap.add_argument("--ang-mix", type=float, default=0.12, help="WHIP rotational carrier weight (Lean angMix).")
    ap.add_argument("--traditional-iters", type=int, default=300)
    ap.add_argument("--traditional-step-size", type=float, default=0.02)
    ap.add_argument("--lean-iters", type=int, default=300)
    ap.add_argument("--lean-step-size", type=float, default=0.02)
    ap.add_argument("--lean-seed", type=int, default=123)
    ap.add_argument(
        "--no-ensure-progress",
        action="store_true",
        help="Disable auto-restart with jitter when a method accepts zero moves.",
    )
    ap.add_argument("--max-restarts", type=int, default=4)
    ap.add_argument("--restart-jitter-sigma-ang", type=float, default=0.75)
    ap.add_argument("--same-energy-tol-ev", type=float, default=1e-3)
    ap.add_argument("--same-rmsd-tol-ang", type=float, default=0.05)
    ap.add_argument(
        "--energy-kwargs-json",
        default=None,
        help='Optional JSON object forwarded to e_tot_ca_with_bonds/grad_full.',
    )
    ap.add_argument(
        "--init-noise-sigma-ang",
        type=float,
        default=0.0,
        help="Optional Gaussian noise (Å) added to initial target coordinates.",
    )
    args = ap.parse_args(argv)

    target_path = os.path.abspath(str(args.target_pdb))
    ca, _ = load_ca_from_pdb(target_path)
    ca0 = np.asarray(ca, dtype=float)
    if float(args.init_noise_sigma_ang) > 0.0:
        rng = np.random.default_rng(int(args.lean_seed))
        ca0 = ca0 + rng.normal(0.0, float(args.init_noise_sigma_ang), size=ca0.shape)

    out_json = args.out_json
    if out_json is None:
        out_json = f"{target_path}.side_by_side_lean_qc.json"
    out_json = os.path.abspath(str(out_json))
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)

    out = run_side_by_side_lean_qc(
        ca0,
        z_shell=int(args.z_shell),
        energy_kwargs=_parse_energy_kwargs(args.energy_kwargs_json),
        traditional_iters=int(args.traditional_iters),
        traditional_step_size=float(args.traditional_step_size),
        lean_iters=int(args.lean_iters),
        lean_step_size=float(args.lean_step_size),
        reference_m=int(args.reference_m),
        ang_mix=float(args.ang_mix),
        same_energy_tol_ev=float(args.same_energy_tol_ev),
        same_rmsd_tol_ang=float(args.same_rmsd_tol_ang),
        ensure_progress=not bool(args.no_ensure_progress),
        max_restarts=int(args.max_restarts),
        restart_jitter_sigma_ang=float(args.restart_jitter_sigma_ang),
        lean_seed=int(args.lean_seed),
    )
    out["target_pdb"] = target_path
    out["init_noise_sigma_ang"] = float(args.init_noise_sigma_ang)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    t = out["traditional"]
    q = out["lean_natural_disp"]
    a = out["agreement"]
    print(f"WROTE {out_json}")
    print(f"site_energy_trace_ev (fixed shells) = {out['site_energy_trace_ev']:.6f}")
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
