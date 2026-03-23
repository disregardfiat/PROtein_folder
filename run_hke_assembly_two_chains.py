#!/usr/bin/env python3
"""
Run the HKE 2-chain pipeline (same as server) on the two proteins from job 1772419986_4500.
Pipeline: HKE-with-funnel per chain (parallel) → place → complex HKE (no funnel) to 0.5 Å per 100 res.
Outputs: chain_a.pdb, chain_b.pdb, complex.pdb and hke_assembly.zip in casp_results/outputs/.
"""

from __future__ import annotations

import os
import sys
import zipfile

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_root, "src"))

# Same sequences as pending job 1772419986_4500 (146 + 141 res)
SEQ_A = "VHLTGEEKSGLTALWAKVNVEEIGGEALGRLLVVYPWTQRFFEHFGDLSTADAVMKNPKVKKHGQKVLASFGEGLKHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVVVLARHFGKEFTPELQTAYQKVVAGVANALAHKYH"
SEQ_B = "VLSPADKTNVKAAWAKVGNHAADFGAEALERMFMSFPSTKTYFSHFDLGHNSTQVKGHGKKVADALTKAVGHLDTLPDALSDLSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPGDFTPSVHASLDKFLASVSTVLTSKYR"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "casp_results", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Params: use fewer iters + looser RMS stop for faster run (still same pipeline)
FUNNEL_RADIUS = 10.0
FUNNEL_RADIUS_EXIT = 20.0
HKE_MAX_ITER = (8, 15, 35)   # was (15, 25, 50) — fewer for speed
CONVERGE_ANG_PER_100 = 1.0   # RMS stop: 1.0 Å per 100 res (looser = stop sooner)
MAX_DOCK_ITER = 600          # cap complex L-BFGS
GTOL = 1e-4                  # looser gradient tol so stages 1–2 converge sooner


def main():
    from horizon_physics.proteins import full_chain_to_pdb, full_chain_to_pdb_complex
    from horizon_physics.proteins.assembly_dock import run_two_chain_assembly_hke

    print("Running HKE 2-chain pipeline (parallel HKE funnel → place → complex HKE)...", flush=True)
    result_a, result_b, result_complex = run_two_chain_assembly_hke(
        SEQ_A,
        SEQ_B,
        funnel_radius=FUNNEL_RADIUS,
        funnel_radius_exit=FUNNEL_RADIUS_EXIT,
        funnel_stiffness=1.0,
        hke_max_iter_s1=HKE_MAX_ITER[0],
        hke_max_iter_s2=HKE_MAX_ITER[1],
        hke_max_iter_s3=HKE_MAX_ITER[2],
        converge_max_disp_per_100_res=CONVERGE_ANG_PER_100,
        max_dock_iter=MAX_DOCK_ITER,
        gtol=GTOL,
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
    for name, content in [
        ("hke_chain_a.pdb", pdb_a),
        ("hke_chain_b.pdb", pdb_b),
        ("hke_complex.pdb", pdb_complex),
    ]:
        path = os.path.join(OUTPUT_DIR, name)
        with open(path, "w") as f:
            f.write(content)
        print(f"  Wrote {path}", flush=True)
    zip_path = os.path.join(OUTPUT_DIR, "hke_assembly.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("chain_a.pdb", pdb_a)
        zf.writestr("chain_b.pdb", pdb_b)
        zf.writestr("complex.pdb", pdb_complex)
    print(f"  Wrote {zip_path}", flush=True)
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
