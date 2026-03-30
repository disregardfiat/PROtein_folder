#!/usr/bin/env python3
"""
Print Cα radius of gyration for a prediction PDB; optionally Cα-RMSD vs a reference.

Typical use (repo root):

  PYTHONPATH=src python3 scripts/compare_naf_casp_to_experiment.py \\
    artifacts/naf_casp_pull/single_1774827685_3011.pdb

With an experimental structure (same sequence / compatible ordering):

  PYTHONPATH=src python3 scripts/compare_naf_casp_to_experiment.py pred.pdb \\
    --ref-pdb /path/to/experiment.pdb

For multi-chain PDBs, compare each chain separately (mean reported):

  PYTHONPATH=src python3 scripts/compare_naf_casp_to_experiment.py pred.pdb \\
    --ref-pdb ref.pdb --chain-pairs A:A,B:B

If lengths differ only at the tail, the default uses order alignment with trim
(see grade_folds.ca_rmsd trim_to_min_length).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from typing import List, Tuple

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from horizon_physics.proteins.grade_folds import ca_rmsd, load_ca_from_pdb  # noqa: E402


def _chains_with_ca(path: str) -> List[str]:
    chains = set()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("ATOM ") and len(line) > 21:
                if line[12:16].strip() != "CA":
                    continue
                chains.add(line[21].strip())
    return sorted(chains, key=lambda c: (c == "", c))


def _rg_ang(ca: np.ndarray) -> float:
    if ca.size == 0:
        return float("nan")
    c = np.mean(ca, axis=0)
    d = ca - c
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def _write_chain_pdb(path: str, chain: str) -> str:
    want = chain.strip()
    lines: List[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("ATOM ") and len(line) > 21:
                if line[21].strip() == want:
                    lines.append(line)
    tf = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False, encoding="utf-8")
    try:
        tf.writelines(lines)
        tf.write("\n")
        tf.close()
        return tf.name
    except Exception:
        try:
            os.unlink(tf.name)
        except OSError:
            pass
        raise


def _parse_pairs(spec: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Bad pair {part!r}; use PRED:REF e.g. A:A,B:B")
        a, b = part.split(":", 1)
        out.append((a.strip(), b.strip()))
    if not out:
        raise ValueError("No chain pairs parsed.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Rg and optional Cα-RMSD vs reference PDB.")
    ap.add_argument("pred_pdb", help="Predicted / model PDB path")
    ap.add_argument("--ref-pdb", help="Optional reference (e.g. experimental) PDB")
    ap.add_argument(
        "--chain-pairs",
        metavar="SPEC",
        help='Multichain: comma-separated PRED:REF pairs, e.g. "A:A,B:B"',
    )
    ap.add_argument(
        "--align-by-resid",
        action="store_true",
        help="Match Cα by residue number (default: align by order)",
    )
    args = ap.parse_args()
    pred = os.path.abspath(args.pred_pdb)

    chains = _chains_with_ca(pred)
    ca_all, _ = load_ca_from_pdb(pred)
    print(f"pred: {pred}")
    print(f"  chains (Cα): {', '.join('(blank)' if c == '' else c for c in chains) if chains else '(none)'}")
    print(f"  n_Cα (all chains): {len(ca_all)}")
    print(f"  Rg (all Cα): {_rg_ang(ca_all):.3f} Å")
    for c in chains:
        ca_c, _ = load_ca_from_pdb(pred, chain_id=c)
        label = "(blank)" if c == "" else c
        print(f"  Rg chain {label}: {_rg_ang(ca_c):.3f} Å  (n={len(ca_c)})")

    if not args.ref_pdb:
        print("No --ref-pdb; skipping Cα-RMSD.")
        return 0

    ref = os.path.abspath(args.ref_pdb)
    align_by_resid = bool(args.align_by_resid)
    temps: List[str] = []

    def _cleanup() -> None:
        for p in temps:
            try:
                os.unlink(p)
            except OSError:
                pass

    try:
        if args.chain_pairs:
            pairs = _parse_pairs(args.chain_pairs)
            rmsds: List[float] = []
            for pc, rc in pairs:
                pp = _write_chain_pdb(pred, pc)
                rp = _write_chain_pdb(ref, rc)
                temps.extend([pp, rp])
                r, _, _, _ = ca_rmsd(
                    pp,
                    rp,
                    align_by_resid=align_by_resid,
                    trim_to_min_length=True,
                )
                rmsds.append(float(r))
                print(f"  Cα-RMSD pred chain {pc} vs ref chain {rc}: {r:.3f} Å")
            print(f"  mean Cα-RMSD (pairs): {float(np.mean(rmsds)):.3f} Å")
        else:
            r, _, _, _ = ca_rmsd(
                pred,
                ref,
                align_by_resid=align_by_resid,
                trim_to_min_length=True,
            )
            print(f"ref: {ref}")
            print(f"  Cα-RMSD (full PDBs, order/trim): {r:.3f} Å")
    except Exception as e:
        print(f"Error grading: {e}")
        _cleanup()
        return 1
    _cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
