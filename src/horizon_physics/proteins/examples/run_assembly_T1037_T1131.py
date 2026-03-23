#!/usr/bin/env python3
"""
Run the new 2-chain assembly system on T1037 (chain A) + T1131 (chain B).
Tries HKE (helix_unit, fast params) first; on overflow falls back to Cartesian.
Writes chain_a.pdb, chain_b.pdb, complex.pdb and assembly.zip to this directory.
"""

from __future__ import annotations

import os
import sys
import zipfile

examples_dir = os.path.dirname(os.path.abspath(__file__))
_p = examples_dir
while _p != os.path.dirname(_p):
    if os.path.isfile(os.path.join(_p, "pyproject.toml")):
        _src = os.path.join(_p, "src")
        if _src not in sys.path:
            sys.path.insert(0, _src)
        break
    _p = os.path.dirname(_p)

# Import sequences from the T1 example modules
from horizon_physics.proteins.examples.T1037 import SEQUENCE as SEQ_T1037
from horizon_physics.proteins.examples.T1131 import SEQUENCE as SEQ_T1131


def main():
    from horizon_physics.proteins import full_chain_to_pdb, full_chain_to_pdb_complex, minimize_full_chain
    from horizon_physics.proteins.assembly_dock import run_two_chain_assembly_hke, run_two_chain_assembly

    seq_a, seq_b = SEQ_T1037, SEQ_T1131
    print(f"Chain A (T1037): {len(seq_a)} res", flush=True)
    print(f"Chain B (T1131): {len(seq_b)} res", flush=True)

    assembly = None
    print("Trying HKE 2-chain (helix_unit, fast params)...", flush=True)
    try:
        r_a, r_b, r_c = run_two_chain_assembly_hke(
            seq_a, seq_b,
            funnel_radius=10.0, funnel_radius_exit=20.0, funnel_stiffness=1.0,
            hke_max_iter_s1=8, hke_max_iter_s2=15, hke_max_iter_s3=35,
            converge_max_disp_per_100_res=1.0, max_dock_iter=600,
            gtol=1e-4, grouping_strategy="helix_unit",
        )
        pdb_a = full_chain_to_pdb(r_a, chain_id="A")
        pdb_b = full_chain_to_pdb(r_b, chain_id="B")
        pdb_c = full_chain_to_pdb_complex(
            r_c["backbone_chain_a"], r_c["backbone_chain_b"],
            r_a["sequence"], r_b["sequence"], chain_id_a="A", chain_id_b="B",
        )
        assembly = (pdb_a, pdb_b, pdb_c)
        print("   HKE path OK.", flush=True)
    except Exception as e:
        print(f"   HKE failed ({e}), falling back to Cartesian...", flush=True)
        res_a = minimize_full_chain(seq_a, max_iter=100, long_chain_max_iter=80, include_sidechains=False)
        res_b = minimize_full_chain(seq_b, max_iter=100, long_chain_max_iter=80, include_sidechains=False)
        res_a, res_b, res_c = run_two_chain_assembly(
            res_a, res_b, max_dock_iter=80, converge_max_disp_per_100_res=1.0,
        )
        pdb_a = full_chain_to_pdb(res_a, chain_id="A")
        pdb_b = full_chain_to_pdb(res_b, chain_id="B")
        pdb_c = full_chain_to_pdb_complex(
            res_c["backbone_chain_a"], res_c["backbone_chain_b"],
            res_a["sequence"], res_b["sequence"], chain_id_a="A", chain_id_b="B",
        )
        assembly = (pdb_a, pdb_b, pdb_c)
        print("   Cartesian path OK.", flush=True)

    if assembly is None:
        print("No assembly produced.", flush=True)
        return 1

    pdb_a, pdb_b, pdb_c = assembly
    for name, content in [
        ("T1037_T1131_chain_a.pdb", pdb_a),
        ("T1037_T1131_chain_b.pdb", pdb_b),
        ("T1037_T1131_complex.pdb", pdb_c),
    ]:
        path = os.path.join(examples_dir, name)
        with open(path, "w") as f:
            f.write(content)
        print(f"   Wrote {path}", flush=True)
    zip_path = os.path.join(examples_dir, "T1037_T1131_assembly.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("chain_a.pdb", pdb_a)
        zf.writestr("chain_b.pdb", pdb_b)
        zf.writestr("complex.pdb", pdb_c)
    print(f"   Wrote {zip_path}", flush=True)
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
