#!/usr/bin/env python3
"""
Quick tunnel + discrete φ/ψ + EM-field refinement pipeline on small targets.

This script runs:
  1) Quick Cartesian baseline (no tunnel) via minimize_full_chain(quick=True).
  2) Quick tunnel HKE (no long post-extrusion refine).
  3) Discrete φ/ψ refinement with EM-field unfreezing (run_discrete_refinement).

It writes quick-tunnel and discrete+EM PDBs for inspection and reports:
  - Cα-RMSD Cartesian vs quick-tunnel.
  - Cα-RMSD quick-tunnel vs discrete+EM.

Intended for crambin / insulin fragment scale where the whole loop is very fast.
"""

from __future__ import annotations

import os
import time

from horizon_physics.proteins import (
    minimize_full_chain,
    full_chain_to_pdb,
    ca_rmsd,
)
from horizon_physics.proteins.temperature_path_search import run_discrete_refinement


EXAMPLES = [
    ("crambin", "TTCCPSIVARSNFNVCRLPGTPEAIICGDVCDLDCTAKTCFSIICT"),
    ("insulin_fragment", "FVNQHLCGSHLVEALYLVCGERGFFYTPK"),
]

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))


def run_quick_pipeline(name: str, seq: str, temperature: float = 310.0) -> None:
    print(f"=== {name} quick tunnel + discrete+EM ({len(seq)} residues) ===")
    t0 = time.time()

    # Quick tunnel run without long post-extrusion refine
    tunnel_result = minimize_full_chain(
        seq,
        include_sidechains=False,
        simulate_ribosome_tunnel=True,
        quick=True,
        post_extrusion_refine=False,
        long_chain_max_iter=80,
    )
    tunnel_time = time.time() - t0

    # Quick Cartesian baseline (no tunnel)
    cart_result = minimize_full_chain(
        seq,
        include_sidechains=False,
        simulate_ribosome_tunnel=False,
        quick=True,
        long_chain_max_iter=80,
    )

    cartesian_pdb = full_chain_to_pdb(cart_result)
    tunnel_pdb = full_chain_to_pdb(tunnel_result)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cart_path = os.path.join(tmp, "cart.pdb")
        tunn_path = os.path.join(tmp, "tunnel.pdb")
        with open(cart_path, "w") as f:
            f.write(cartesian_pdb)
        with open(tunn_path, "w") as f:
            f.write(tunnel_pdb)
        rmsd_ct, _, _, _ = ca_rmsd(cart_path, tunn_path, align_by_resid=False)

    print(f"  Tunnel quick Time: {tunnel_time:.1f}s")
    print(f"  Cα-RMSD cart vs quick-tunnel: {rmsd_ct:.3f} Å")

    # Discrete φ/ψ refinement + EM unfreezing from quick tunnel backbone
    backbone = tunnel_result.get("backbone_atoms")
    if not backbone or len(backbone) != 4 * len(seq):
        print("  Skipping discrete+EM: tunnel_result lacks full backbone.")
        return

    refine_res = run_discrete_refinement(
        seq,
        temperature=temperature,
        n_steps=120,
        initial_backbone_atoms=backbone,
        seed=1,
    )
    pdb_result = {
        "backbone_atoms": refine_res.backbone_atoms,
        "sequence": refine_res.sequence,
        "n_res": refine_res.n_res,
        "include_sidechains": False,
    }
    discrete_pdb = full_chain_to_pdb(pdb_result)

    with tempfile.TemporaryDirectory() as tmp2:
        t_path = os.path.join(tmp2, "tunnel.pdb")
        d_path = os.path.join(tmp2, "discrete_em.pdb")
        with open(t_path, "w") as f:
            f.write(tunnel_pdb)
        with open(d_path, "w") as f:
            f.write(discrete_pdb)
        rmsd_td, _, _, _ = ca_rmsd(t_path, d_path, align_by_resid=False)

    # Write PDBs next to other examples for visualization
    out_tunnel = os.path.join(EXAMPLES_DIR, f"{name}_quick_tunnel.pdb")
    out_discrete = os.path.join(EXAMPLES_DIR, f"{name}_quick_discrete_em.pdb")
    with open(out_tunnel, "w") as f:
        f.write(tunnel_pdb)
    with open(out_discrete, "w") as f:
        f.write(discrete_pdb)

    print(
        f"  Discrete+EM: {refine_res.n_accept}/{refine_res.n_steps} accepted, "
        f"E_ca={refine_res.E_ca_final:.4f} eV"
    )
    print(f"  Cα-RMSD quick-tunnel vs discrete+EM: {rmsd_td:.3f} Å")
    print(f"  Wrote {os.path.basename(out_tunnel)}, {os.path.basename(out_discrete)}")
    print(f"  phases: {refine_res.info.get('phases')}")
    print()


def main() -> None:
    for name, seq in EXAMPLES:
        run_quick_pipeline(name, seq)


if __name__ == "__main__":
    main()

