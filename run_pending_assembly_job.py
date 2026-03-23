#!/usr/bin/env python3
"""
Run the pending 2-chain assembly job (1772419986_4500) through the new pipeline.
Sequences from server pending/1772419986_4500.request.json (146 + 141 res).
Uses Cartesian minimizer per chain to avoid HKE overflow; then placement + complex minimization.
Outputs: chain_a.pdb, chain_b.pdb, complex.pdb and assembly.zip in casp_results/outputs/.
"""

from __future__ import annotations

import os
import sys
import zipfile

# Repo root
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_root, "src"))

# Same sequences as pending job 1772419986_4500 (CAMEO 6aaf9bfc19)
SEQ_A = "VHLTGEEKSGLTALWAKVNVEEIGGEALGRLLVVYPWTQRFFEHFGDLSTADAVMKNPKVKKHGQKVLASFGEGLKHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVVVLARHFGKEFTPELQTAYQKVVAGVANALAHKYH"
SEQ_B = "VLSPADKTNVKAAWAKVGNHAADFGAEALERMFMSFPSTKTYFSHFDLGHNSTQVKGHGKKVADALTKAVGHLDTLPDALSDLSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPGDFTPSVHASLDKFLASVSTVLTSKYR"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "casp_results", "outputs")
JOB_ID = "1772419986_4500"


def main():
    from horizon_physics.proteins import minimize_full_chain, full_chain_to_pdb, full_chain_to_pdb_complex
    from horizon_physics.proteins.assembly_dock import run_two_chain_assembly

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Use long-chain path (n_res>50): collapse + refine
    print("Folding chain A (146 res)...", flush=True)
    result_a = minimize_full_chain(
        SEQ_A,
        max_iter=100,
        long_chain_max_iter=80,
        include_sidechains=False,
    )
    print("Folding chain B (141 res)...", flush=True)
    result_b = minimize_full_chain(
        SEQ_B,
        max_iter=100,
        long_chain_max_iter=80,
        include_sidechains=False,
    )
    print("Placement + complex minimization...", flush=True)
    result_a, result_b, result_complex = run_two_chain_assembly(
        result_a, result_b, max_dock_iter=60
    )
    pdb_a = full_chain_to_pdb(result_a, chain_id="A")
    pdb_b = full_chain_to_pdb(result_b, chain_id="B")
    pdb_complex = full_chain_to_pdb_complex(
        result_complex["backbone_chain_a"],
        result_complex["backbone_chain_b"],
        result_a["sequence"],
        result_b["sequence"],
        chain_id_a="A",
        chain_id_b="B",
    )
    # Write 3 PDBs + ZIP (same as server _move_to_outputs_assembly)
    for name, content in [
        (f"{JOB_ID}_chain_a.pdb", pdb_a),
        (f"{JOB_ID}_chain_b.pdb", pdb_b),
        (f"{JOB_ID}_complex.pdb", pdb_complex),
    ]:
        path = os.path.join(OUTPUT_DIR, name)
        with open(path, "w") as f:
            f.write(content)
        print(f"  Wrote {path}", flush=True)
    zip_path = os.path.join(OUTPUT_DIR, f"{JOB_ID}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("chain_a.pdb", pdb_a)
        zf.writestr("chain_b.pdb", pdb_b)
        zf.writestr("complex.pdb", pdb_complex)
    print(f"  Wrote {zip_path}", flush=True)
    print("Done. Job completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
