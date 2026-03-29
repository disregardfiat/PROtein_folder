#!/usr/bin/env python3
"""
Smoke-test the Lean CASP pipeline (same calls as casp_server default): timing, PDB sanity,
scalar metrics (E_ca, Rg), and optional Cα-RMSD vs a reference PDB.

No Flask required. Run from repo root:
  python3 scripts/validate_casp_lean_pipeline.py
  python3 scripts/validate_casp_lean_pipeline.py --grade-ref /path/to/ref.pdb

Env: VALIDATE_LEAN_REF_PDB — same as --grade-ref if the flag is omitted.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from horizon_physics.proteins.folding_energy import rg_squared  # noqa: E402
from horizon_physics.proteins.grade_folds import ca_rmsd  # noqa: E402
from horizon_physics.proteins.hqiv_lean_folding import PHYSIOLOGICAL_PH  # noqa: E402
from horizon_physics.proteins.lean_ribosome_tunnel_pipeline import fold_lean_ribosome_tunnel  # noqa: E402
from horizon_physics.proteins.ligands import parse_ligands  # noqa: E402


def _lean_kw(quick: bool, *, tunnel: bool) -> dict:
    """Mirror casp_server._lean_fold_env_kwargs; set tunnel=False for fast CI (Cartesian Lean path)."""
    if quick:
        base = {
            "temperature_k": 310.0,
            "ph": float(PHYSIOLOGICAL_PH),
            "kappa_dihedral": 0.01,
            "post_extrusion_max_rounds": 12,
            "fast_pass_steps_per_connection": 2,
            "min_pass_iter_per_connection": 5,
            "hbond_weight": 0.0,
            "hbond_shell_m": 3,
            "ligand_refine_steps": 50,
        }
    else:
        base = {
            "temperature_k": 310.0,
            "ph": float(PHYSIOLOGICAL_PH),
            "kappa_dihedral": 0.01,
            "post_extrusion_max_rounds": 32,
            "fast_pass_steps_per_connection": 5,
            "min_pass_iter_per_connection": 15,
            "hbond_weight": 0.0,
            "hbond_shell_m": 3,
            "ligand_refine_steps": 150,
        }
    if not tunnel:
        base["simulate_ribosome_tunnel"] = False
        base["post_extrusion_refine"] = False
    return base


def _minimal_pdb_sanity(pdb: str) -> tuple[bool, str]:
    """Lightweight check: MODEL, ATOM lines, finite coords."""
    if "MODEL" not in pdb and "ATOM" not in pdb and "HETATM" not in pdb:
        return False, "no MODEL/ATOM/HETATM"
    for line in pdb.splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 54:
            try:
                float(line[30:38])
                float(line[38:46])
                float(line[46:54])
            except ValueError:
                return False, f"bad coords: {line[:60]!r}"
    if pdb.count("\n") < 2:
        return False, "too short"
    return True, ""


def _metrics_from_raw(raw: dict) -> dict:
    """E_tot on Cα / full backbone (eV) and radius of gyration from Cα (Å)."""
    ca = raw.get("ca_min")
    ec = raw.get("E_ca_final")
    ebb = raw.get("E_backbone_final")
    rg: float | None = None
    if ca is not None:
        ca = np.asarray(ca, dtype=float)
        if ca.size > 0 and ca.ndim == 2 and ca.shape[1] == 3:
            rg = float(np.sqrt(max(0.0, rg_squared(ca))))
    return {"E_ca_eV": ec, "E_backbone_eV": ebb, "Rg_A": rg}


def _print_metrics(label: str, raw: dict) -> None:
    m = _metrics_from_raw(raw)
    ec, ebb, rgv = m["E_ca_eV"], m["E_backbone_eV"], m["Rg_A"]
    parts = []
    if ec is not None:
        parts.append(f"E_ca={float(ec):.5g} eV")
    if ebb is not None:
        parts.append(f"E_bb={float(ebb):.5g} eV")
    if rgv is not None:
        parts.append(f"Rg={rgv:.3f} Å")
    if parts:
        print(f"   {label}: " + ", ".join(parts))


def _grade_vs_ref(pdb_str: str, ref_path: str, *, label: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False, encoding="utf-8") as tmp:
        tmp.write(pdb_str)
        pred_path = tmp.name
    try:
        rmsd, _, _, _ = ca_rmsd(
            pred_path,
            ref_path,
            align_by_resid=False,
            trim_to_min_length=True,
        )
        print(f"   {label}: Cα-RMSD vs ref = {rmsd:.3f} Å (order-aligned, trimmed to min length)")
    finally:
        try:
            os.unlink(pred_path)
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate Lean CASP fold pipeline (timing + metrics + optional RMSD).")
    ap.add_argument(
        "--grade-ref",
        type=str,
        default=None,
        metavar="REF.pdb",
        help="Reference PDB for Cα-RMSD after Kabsch (same sequence order; shorter chain is trimmed). "
        "Default: env VALIDATE_LEAN_REF_PDB.",
    )
    args = ap.parse_args()
    ref_env = (os.environ.get("VALIDATE_LEAN_REF_PDB") or "").strip()
    grade_ref = (args.grade_ref or ref_env or "").strip() or None

    seq24 = "ACDEFGHIKLMNPQRSTVWYACDK"
    seq12 = seq24[:12]
    seq18 = seq24[:18]

    t0 = time.perf_counter()
    out1 = fold_lean_ribosome_tunnel(
        seq12, quick=True, include_ligands=False, **_lean_kw(True, tunnel=False)
    )
    print(f"1) fold_lean quick (Cartesian Lean path)  {time.perf_counter() - t0:8.2f}s")
    ok, r = _minimal_pdb_sanity(out1.pdb)
    assert ok, r
    _print_metrics("metrics", out1.raw_result)
    if grade_ref:
        _grade_vs_ref(out1.pdb, grade_ref, label="grade vs ref (step 1)")

    t0 = time.perf_counter()
    out_a = fold_lean_ribosome_tunnel(seq12[:8], quick=True, include_ligands=False, **_lean_kw(True, tunnel=False))
    out_b = fold_lean_ribosome_tunnel(seq12[4:], quick=True, include_ligands=False, **_lean_kw(True, tunnel=False))
    print(f"2) two quick folds (assembly analogue)   {time.perf_counter() - t0:8.2f}s")
    ok, r = _minimal_pdb_sanity(out_a.pdb)
    assert ok, f"A: {r}"
    ok, r = _minimal_pdb_sanity(out_b.pdb)
    assert ok, f"B: {r}"
    _print_metrics("metrics A", out_a.raw_result)
    _print_metrics("metrics B", out_b.raw_result)

    lig_pdb = """HETATM    1  C1  LIG L   1      10.000  0.000  0.000  1.00  0.00           C
HETATM    2  O1  LIG L   1      12.000  0.000  0.000  1.00  0.00           O
"""
    ligs = parse_ligands(lig_pdb)
    t0 = time.perf_counter()
    out3 = fold_lean_ribosome_tunnel(
        seq18,
        quick=True,
        include_ligands=True,
        ligands=ligs,
        ligand_chain_id="L",
        **_lean_kw(True, tunnel=False),
    )
    print(f"3) fold_lean full + ligands (chain L)    {time.perf_counter() - t0:8.2f}s")
    assert "HETATM" in out3.pdb
    ok, r = _minimal_pdb_sanity(out3.pdb)
    assert ok, r
    _print_metrics("metrics", out3.raw_result)

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
