#!/usr/bin/env python3
"""
Validate Lean ``ProteinQCRefinement``-style ligand refinement against published PDB coordinates.

Fetches RCSB structures, loads fixed protein backbone + HETATM ligands, perturbs ligand pose,
runs ``_refine_ligands_6dof`` (default ``lean_qc``), and reports soft-clash energy + ligand RMSD
to the deposited pose after Kabsch alignment.

Usage (repo root):
  PYTHONPATH=src python3 scripts/validate_qc_ligand_published_pdbs.py
  PYTHONPATH=src python3 scripts/validate_qc_ligand_published_pdbs.py --cameo --steps 50
  PYTHONPATH=src python3 scripts/validate_qc_ligand_published_pdbs.py --pdb 1HVR 3FAP --no-fetch /path/to/1HVR.pdb

Env ``HTTP(S)_PROXY`` is respected by urllib.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import tempfile
import urllib.request

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from horizon_physics.proteins.full_protein_minimizer import (  # noqa: E402
    _full_backbone_positions_and_z,
    _refine_ligands_6dof,
)
from horizon_physics.proteins.grade_folds import load_backbone_atoms_ordered_from_pdb  # noqa: E402
from horizon_physics.proteins.cameo_style_benchmark_pdbs import (  # noqa: E402
    CAMEO_STYLE_LIGAND_PDB_IDS,
)
from horizon_physics.proteins.protein_qc_refinement import (  # noqa: E402
    ligand_heavy_atom_rmsd,
    load_ligand_agents_from_pdb_file,
    qc_soft_clash_energy_protein_ligand,
)


RCSB_DOWNLOAD = "https://files.rcsb.org/download/{id}.pdb"

# Published complexes (PDB id, notes). Waters skipped in ligand loader.
DEFAULT_PDB_IDS = ("1HVR", "3FAP")


def _fetch_pdb(pdb_id: str, dest: str) -> None:
    url = RCSB_DOWNLOAD.format(id=pdb_id.upper())
    req = urllib.request.Request(url, headers={"User-Agent": "PROtien-validate-qc-ligand/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
        body = r.read()
    with open(dest, "wb") as f:
        f.write(body)


def _largest_ligand(agents: list) -> list:
    """If multiple HET groups, keep the largest by atom count for a clean benchmark."""
    if len(agents) <= 1:
        return agents
    agents = sorted(agents, key=lambda a: a.n_atoms(), reverse=True)
    return [agents[0]]


def _run_case(
    pdb_path: str,
    *,
    sigma: float,
    w_clash: float,
    steps: int,
    perturb: float,
    mode: str,
) -> dict:
    bb = load_backbone_atoms_ordered_from_pdb(pdb_path, chain_id=None)
    pos_bb, z_bb = _full_backbone_positions_and_z(bb)
    agents = load_ligand_agents_from_pdb_file(pdb_path)
    if not agents:
        return {"ok": False, "reason": "no HETATM ligands parsed"}
    agents = _largest_ligand(list(agents))

    ref_blocks = [a.get_world_positions().copy() for a in agents]

    rng = np.random.default_rng(42)
    for ag in agents:
        t, euler = ag.get_6dof()
        delta = rng.standard_normal(3).astype(np.float64)
        n = float(np.linalg.norm(delta)) or 1.0
        ag.set_6dof(t + float(perturb) * (delta / n), euler + 0.35 * rng.standard_normal(3))

    lig_pos_a = np.concatenate([a.get_world_positions() for a in agents], axis=0)
    clash_before = qc_soft_clash_energy_protein_ligand(pos_bb, lig_pos_a, sigma)

    _refine_ligands_6dof(
        pos_bb,
        z_bb,
        agents,
        max_steps=int(steps),
        step_t=0.12 if mode == "lean_qc" else 0.05,
        step_ang=0.06 if mode == "lean_qc" else 0.02,
        grad_full_kwargs={"em_scale": 1.0, "r_horizon": 15.0},
        ligand_refinement_mode=mode,
        qc_soft_clash_sigma=float(sigma),
        qc_clash_weight=float(w_clash),
    )

    lig_pos_b = np.concatenate([a.get_world_positions() for a in agents], axis=0)
    clash_after = qc_soft_clash_energy_protein_ligand(pos_bb, lig_pos_b, sigma)

    rmsds = []
    for ag, ref in zip(agents, ref_blocks):
        pred = ag.get_world_positions()
        r, _ = ligand_heavy_atom_rmsd(pred, ref)
        rmsds.append(r)

    return {
        "ok": True,
        "n_bb_atoms": int(pos_bb.shape[0]),
        "n_ligand_groups": len(agents),
        "n_ligand_atoms": sum(a.n_atoms() for a in agents),
        "clash_before": float(clash_before),
        "clash_after": float(clash_after),
        "ligand_rmsd_to_pdb_A": float(np.mean(rmsds)) if rmsds else None,
        "mode": mode,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="QC ligand refinement vs published PDBs")
    ap.add_argument(
        "--pdb",
        nargs="*",
        default=None,
        metavar="ID",
        help="PDB ids or local .pdb paths (default: 1HVR 3FAP; use --cameo for CAMEO3D-style batch)",
    )
    ap.add_argument(
        "--cameo",
        action="store_true",
        help=f"CAMEO3D-style ligand batch ({len(CAMEO_STYLE_LIGAND_PDB_IDS)} wwPDB ids from cameo_style_benchmark_pdbs)",
    )
    ap.add_argument("--no-fetch", action="store_true", help="Do not download; pass local paths as ids")
    ap.add_argument("--sigma", type=float, default=3.0, help="Soft clash σ (Å), Lean qcSoftClashEnergy")
    ap.add_argument("--w-clash", type=float, default=1.0, dest="w_clash", help="Weight on clash term")
    ap.add_argument("--steps", type=int, default=80, help="Rigid-body refinement steps")
    ap.add_argument("--perturb", type=float, default=2.5, help="Å translation bump on ligand centroid")
    ap.add_argument("--horizon", action="store_true", help="Use legacy horizon refinement instead of lean_qc")
    args = ap.parse_args()
    mode = "horizon" if args.horizon else "lean_qc"
    if args.pdb:
        pdb_list = list(args.pdb)
    elif args.cameo:
        pdb_list = list(CAMEO_STYLE_LIGAND_PDB_IDS)
    else:
        pdb_list = list(DEFAULT_PDB_IDS)

    print("validate_qc_ligand_published_pdbs (Lean ProteinQCRefinement soft clash + ligand 6-DOF)")
    print("mode =", mode, "sigma =", args.sigma, "steps =", args.steps, "n_pdbs =", len(pdb_list))
    print("-" * 88)

    for pid in pdb_list:
        path = pid
        if not args.no_fetch and len(pid) == 4 and os.path.isfile(pid) is False:
            tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
            path = tmp.name
            tmp.close()
            try:
                print(f"fetch {pid.upper()} …", flush=True)
                _fetch_pdb(pid, path)
            except Exception as e:
                print({"pdb": pid, "ok": False, "error": str(e)})
                try:
                    os.unlink(path)
                except OSError:
                    pass
                continue
        elif not os.path.isfile(path):
            print({"pdb": pid, "ok": False, "error": f"not a file: {path}"})
            continue

        try:
            out = _run_case(
                path,
                sigma=args.sigma,
                w_clash=args.w_clash,
                steps=args.steps,
                perturb=args.perturb,
                mode=mode,
            )
            out["pdb"] = os.path.basename(path) if path else pid
            print(out)
        finally:
            if not args.no_fetch and len(pid) == 4 and path != pid and os.path.isfile(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    print("-" * 88)
    print("done. clash_after < clash_before indicates reduced steric overlap under the QC model.")


if __name__ == "__main__":
    main()
