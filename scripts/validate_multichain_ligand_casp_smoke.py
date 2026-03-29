#!/usr/bin/env python3
"""
Smoke-test multi-chain assembly + ligand folds (same stack as casp_server Lean path):
finite coordinates, optional Cα-RMSD vs merged experimental references.

Run from repo root:
  PYTHONPATH=src python3 scripts/validate_multichain_ligand_casp_smoke.py

Uses quick Lean budgets + HQIV-native OSH with a modest iteration cap unless overridden.
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from horizon_physics.proteins.casp_submission import _place_full_backbone
from horizon_physics.proteins.full_protein_minimizer import (
    full_chain_to_pdb_complex,
    merged_multichain_backbone_atoms,
)
from horizon_physics.proteins.grade_folds import ca_rmsd, load_ca_and_sequence_from_pdb
from horizon_physics.proteins.lean_ribosome_tunnel_pipeline import fold_lean_ribosome_tunnel
from horizon_physics.proteins.ligands import parse_ligands


def _pdb_coord_sanity(pdb: str) -> Tuple[bool, str, float]:
    mx = 0.0
    n = 0
    for line in pdb.splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 54:
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                return False, "bad_float", mx
            if not all(np.isfinite([x, y, z])):
                return False, "non_finite", mx
            mx = max(mx, abs(x), abs(y), abs(z))
            n += 1
    if n == 0:
        return False, "no_atoms", mx
    if mx > 1.0e6:
        return False, f"coord_too_large_{mx:.3g}", mx
    return True, "ok", mx


def _lean_kw() -> Dict[str, Any]:
    """Bounded CASP-like Lean tunnel + post-extrusion + HQIV OSH (short iterations)."""
    return {
        "quick": True,
        "temperature_k": 310.0,
        "kappa_dihedral": 0.01,
        "post_extrusion_refine": True,
        "post_extrusion_refine_mode": "em_treetorque",
        "post_extrusion_treetorque_phases": 2,
        "post_extrusion_treetorque_n_steps": 24,
        "post_extrusion_osh_hqiv_native": True,
        "post_extrusion_osh_n_iter": 36,
        "post_extrusion_osh_step_size": 0.028,
        "post_extrusion_osh_ansatz_depth": 2,
        "fast_pass_steps_per_connection": 2,
        "min_pass_iter_per_connection": 4,
        "hbond_weight": 0.0,
    }


def _raw_to_merged_pdb(
    raw_a: Dict[str, Any],
    raw_b: Dict[str, Any],
) -> str:
    seq_a = str(raw_a["sequence"])
    seq_b = str(raw_b["sequence"])
    merged = merged_multichain_backbone_atoms([(raw_a, "A"), (raw_b, "B")], chain_gap=50.0)
    n_a = 4 * len(seq_a)
    bb_a = merged[:n_a]
    bb_b = merged[n_a:]
    return full_chain_to_pdb_complex(bb_a, bb_b, seq_a, seq_b, chain_id_a="A", chain_id_b="B")


def _pdb_one_chain_only(full_pdb: str, chain: str) -> str:
    """Keep ATOM lines for a single chain (Cα grading helper)."""
    lines: List[str] = []
    for line in full_pdb.splitlines():
        if line.startswith("ATOM ") and len(line) > 21:
            if line[21].strip() == chain.strip():
                lines.append(line)
    return "\n".join(lines) + "\n"


def _ref_merged_pdb(
    path_a: str,
    path_b: str,
    n_a: int,
    n_b: int,
) -> str:
    ca_a, s_a = load_ca_and_sequence_from_pdb(path_a)
    ca_b, s_b = load_ca_and_sequence_from_pdb(path_b)
    ca_a = ca_a[:n_a]
    s_a = s_a[:n_a]
    ca_b = ca_b[:n_b]
    s_b = s_b[:n_b]
    bb_a = _place_full_backbone(ca_a, s_a)
    bb_b = _place_full_backbone(ca_b, s_b)
    raw_a = {
        "backbone_atoms": bb_a,
        "sequence": s_a,
        "n_res": len(s_a),
        "include_sidechains": False,
    }
    raw_b = {
        "backbone_atoms": bb_b,
        "sequence": s_b,
        "n_res": len(s_b),
        "include_sidechains": False,
    }
    return _raw_to_merged_pdb(raw_a, raw_b)


def main() -> int:
    kw = _lean_kw()
    rows: List[Dict[str, Any]] = []

    # --- 1) Two-chain assembly (sequences = truncated 1CRN + 1VII) ---
    p_crn = os.path.join(_REPO, "proteins", "1CRN.pdb")
    p_vii = os.path.join(_REPO, "proteins", "1VII.pdb")
    ca_x, s_x = load_ca_and_sequence_from_pdb(p_crn)
    ca_y, s_y = load_ca_and_sequence_from_pdb(p_vii)
    n_a = min(14, len(s_x))
    n_b = min(12, len(s_y))
    seq_a = s_x[:n_a]
    seq_b = s_y[:n_b]

    out_a = fold_lean_ribosome_tunnel(seq_a, include_ligands=False, **kw)
    out_b = fold_lean_ribosome_tunnel(seq_b, include_ligands=False, **kw)
    pdb_2ch = _raw_to_merged_pdb(dict(out_a.raw_result), dict(out_b.raw_result))
    ok, reason, mx = _pdb_coord_sanity(pdb_2ch)
    rmsd_ab: float | None = None
    if ok:
        ref_2ch = _ref_merged_pdb(p_crn, p_vii, n_a, n_b)
        pred_a = _pdb_one_chain_only(pdb_2ch, "A")
        pred_b = _pdb_one_chain_only(pdb_2ch, "B")
        ref_a = _pdb_one_chain_only(ref_2ch, "A")
        ref_b = _pdb_one_chain_only(ref_2ch, "B")
        paths: List[str] = []
        try:
            for content in (pred_a, ref_a, pred_b, ref_b):
                tf = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False, encoding="utf-8")
                tf.write(content)
                tf.close()
                paths.append(tf.name)
            pa, ra, pb, rb = paths
            rms_a, _, _, _ = ca_rmsd(pa, ra, align_by_resid=False, trim_to_min_length=True)
            rms_b, _, _, _ = ca_rmsd(pb, rb, align_by_resid=False, trim_to_min_length=True)
            rmsd_ab = float((rms_a + rms_b) / 2.0)
        except Exception as ex:
            rmsd_ab = None
            reason = f"grade_{ex}"
        finally:
            for p in paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass
    rows.append(
        {
            "case": "2-chain assembly",
            "seq_lens": f"{n_a}+{n_b}",
            "sanity_ok": ok,
            "sanity": reason,
            "max_abs_coord_A": mx,
            "ca_rmsd_mean_A_B_vs_ref": rmsd_ab,
        }
    )

    # --- 2) Single chain + minimal HETATM ligand (2 atoms) ---
    seq_l = "ACAG"
    het_block = """
HETATM 9901  O   HOH L   1      12.000  10.000   8.000  1.00  0.00           O
HETATM 9902  H1  HOH L   1      12.600  10.000   8.000  1.00  0.00           H
"""
    ligs = parse_ligands(het_block)
    out_l = fold_lean_ribosome_tunnel(
        seq_l,
        include_ligands=bool(ligs),
        ligands=ligs if ligs else None,
        ligand_chain_id="L",
        **kw,
    )
    pdb_l = out_l.pdb
    ok_l, reason_l, mx_l = _pdb_coord_sanity(pdb_l)
    rmsd_l: float | None = None
    # No small-molecule "native"; grade protein Cα vs extended peptide geometry is weak — use 1CRN first 4 if sequence matched
    # Here sequence ACAG ≠ crambin; report protein-only RMSD N/A or skip
    rows.append(
        {
            "case": "1-chain + HETATM ligand",
            "seq_lens": str(len(seq_l)),
            "sanity_ok": ok_l,
            "sanity": reason_l,
            "max_abs_coord_A": mx_l,
            "ca_rmsd_mean_A_B_vs_ref": None,
            "note": "ligand case: sanity only (sequence not matched to a gold PDB)",
        }
    )

    # --- 3) Second multi-chain: very short chains (stress layout) ---
    out_s1 = fold_lean_ribosome_tunnel("ACDEFGHI", include_ligands=False, **kw)
    out_s2 = fold_lean_ribosome_tunnel("KLMNPQRT", include_ligands=False, **kw)
    pdb_short = _raw_to_merged_pdb(dict(out_s1.raw_result), dict(out_s2.raw_result))
    ok_s, reason_s, mx_s = _pdb_coord_sanity(pdb_short)
    rows.append(
        {
            "case": "2-chain short (8+8)",
            "seq_lens": "8+8",
            "sanity_ok": ok_s,
            "sanity": reason_s,
            "max_abs_coord_A": mx_s,
            "ca_rmsd_mean_A_B_vs_ref": None,
            "note": "no experimental ref (ab initio smoke)",
        }
    )

    print("validate_multichain_ligand_casp_smoke (Lean tunnel + EM/TT + HQIV OSH native)")
    print("-" * 88)
    for r in rows:
        print(r)
    print("-" * 88)
    all_ok = all(bool(r.get("sanity_ok")) for r in rows)
    print("ALL_SANITY_OK" if all_ok else "SANITY_FAILURE")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
