#!/usr/bin/env python3
"""
Grade crambin predictions against experimental gold standard (1CRN).

1CRN is the classic crambin crystal structure (1.50 Å). Sequence differs slightly
from our target (TTCCPSIVARSNFNVCRLPGTPEAIICGDVCDLDCTAKTCFSIICT) but both are
46 residues; we align by residue order for fold comparison.

Usage:
  python -m horizon_physics.proteins.examples.grade_crambin
  python -m horizon_physics.proteins.examples.grade_crambin crambin_minimized_emfield.pdb
"""

from __future__ import annotations

import os
import sys

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
GOLD_STANDARD = os.path.join(EXAMPLES_DIR, "crambin_1CRN.pdb")


def main() -> None:
    pred_name = sys.argv[1] if len(sys.argv) > 1 else "crambin_minimized_emfield.pdb"
    pred_path = os.path.join(EXAMPLES_DIR, pred_name) if not os.path.isabs(pred_name) else pred_name

    if not os.path.isfile(GOLD_STANDARD):
        print("Gold standard not found. Download with:")
        print("  curl -sL https://files.rcsb.org/download/1CRN.pdb -o", GOLD_STANDARD)
        sys.exit(1)

    if not os.path.isfile(pred_path):
        print("Prediction not found:", pred_path)
        sys.exit(1)

    from horizon_physics.proteins.grade_folds import ca_rmsd, load_ca_and_sequence_from_pdb

    # Align by order (both 46 residues; sequence differs slightly)
    rmsd, per_res, pred_aligned, ref_ca = ca_rmsd(pred_path, GOLD_STANDARD, align_by_resid=False)

    ref_seq = load_ca_and_sequence_from_pdb(GOLD_STANDARD)[1]
    target_seq = "TTCCPSIVARSNFNVCRLPGTPEAIICGDVCDLDCTAKTCFSIICT"
    seq_note = " (1CRN seq differs; aligned by order)" if ref_seq != target_seq else ""

    print("Crambin vs gold standard (1CRN, 1.50 Å)")
    print("=" * 50)
    print(f"  Prediction:     {os.path.basename(pred_path)}")
    print(f"  Gold standard:  crambin_1CRN.pdb (PDB 1CRN)")
    print(f"  Cα-RMSD:       {rmsd:.3f} Å{seq_note}")
    if per_res is not None:
        print(f"  Per-residue:   min={float(per_res.min()):.3f} max={float(per_res.max()):.3f} Å")
    print("=" * 50)


if __name__ == "__main__":
    main()
