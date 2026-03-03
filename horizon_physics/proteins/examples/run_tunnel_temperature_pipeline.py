#!/usr/bin/env python3
"""
Run tunnel + temperature folding pipeline on example residue chains.

Runs Cartesian and tunnel stages; for crambin and insulin_fragment also runs
discrete φ/ψ refinement (Metropolis) when pyhqiv is available. Writes PDBs
to the examples dir and reports Cα-RMSD; crambin is graded vs 1CRN if present.
"""

from __future__ import annotations

import os
import time

from horizon_physics.proteins import (
    fold_with_tunnel_temperature,
    full_chain_to_pdb,
    ca_rmsd,
)

EXAMPLES = [
    ("crambin", "TTCCPSIVARSNFNVCRLPGTPEAIICGDVCDLDCTAKTCFSIICT"),
    ("insulin_fragment", "FVNQHLCGSHLVEALYLVCGERGFFYTPK"),
    (
        "T1131",
        "FVPEEQYNKDFNFLYDYAVIHNLVMDGFSEEDGQYNWDFAKNPDSSRSDESIAYVKELQKLK"
        "REDAINFGANAWVLNHNIGFDYKTLKNHQFNLTDANENHSFVVEYWNLKNDETGRHTFWDSV"
        "IGEKYGEYLYNADEDTRINGKLKTPYAWVKQILYGIEDAGAPGFSSISA",
    ),
    (
        "T1037",
        "SKINFYTTTIETLETEDQNNTLTTFKVQNVSNASTIFSNGKTYWNFARPSYISNRINTFKNNPGVLRQLLNTSY"
        "GQSSLWAKHLLGEEKNVTGDFVLAGNARESASENRLKSLELSIFNSLQEKDKGAEGNDNGSISIVDQLADKLN"
        "KVLRGGTKNGTSIYSTVTPGDKSTLHEIKIDHFIPETISSFSNGTMIFNDKIVNAFTDHFVSEVNRMKEAYQE"
        "LETLPESKRVVHYHTDARGNVMKDGKLAGNAFKSGHILSELSFDQITQDDNEMLKLYNEDGSPINPKGAVSNE"
        "QKILIKQTINKVLNQRIKENIRYFKDQGLVIDTVNKDGNKGFHFHGLDKSIMSEYTDDIQLTEFDISHVVSD"
        "FTLNSILASIEYTKLFTGDPANYKNMVDFFKRVPATYTN",
    ),
]

# Small targets we run discrete refinement on (when pyhqiv available)
SMALL_TARGETS = {"crambin", "insulin_fragment"}

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
CRAMBIN_GOLD = os.path.join(EXAMPLES_DIR, "crambin_1CRN.pdb")


def _run_discrete_refinement(seq: str, tunnel_result: dict, temperature: float = 310.0, n_steps: int = 150):
    """Run discrete φ/ψ refinement from tunnel backbone; returns (refine_result, pdb_str) or (None, None) if unavailable."""
    try:
        from horizon_physics.proteins.temperature_path_search import run_discrete_refinement
    except ImportError:
        return None, None
    backbone = tunnel_result.get("backbone_atoms")
    if not backbone or len(backbone) != 4 * len(seq):
        return None, None
    res = run_discrete_refinement(
        seq,
        temperature=temperature,
        n_steps=n_steps,
        initial_backbone_atoms=backbone,
        seed=42,
    )
    pdb_result = {
        "backbone_atoms": res.backbone_atoms,
        "sequence": res.sequence,
        "n_res": res.n_res,
        "include_sidechains": False,
    }
    return res, full_chain_to_pdb(pdb_result)


def main() -> None:
    print("Tunnel + temperature folding pipeline on example targets\n")
    for name, seq in EXAMPLES:
        print(f"=== {name} ({len(seq)} residues) ===")
        t0 = time.time()
        result = fold_with_tunnel_temperature(
            seq,
            temperature=310.0,
            return_tunnel_result=(name in SMALL_TARGETS),
        )
        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s")
        if result.ca_rmsd_cart_vs_tunnel is not None:
            print(f"  Cα-RMSD cartesian vs tunnel: {result.ca_rmsd_cart_vs_tunnel:.3f} Å")
        else:
            print("  Cα-RMSD cartesian vs tunnel: n/a")

        # For small targets: discrete refinement and write PDBs
        if name in SMALL_TARGETS:
            tunnel_result = getattr(result, "tunnel_result", None)
            if tunnel_result:
                refine_res, discrete_pdb = _run_discrete_refinement(seq, tunnel_result)
                if refine_res is not None:
                    tunnel_pdb = full_chain_to_pdb(tunnel_result)
                    tunnel_path = os.path.join(EXAMPLES_DIR, f"{name}_tunnel.pdb")
                    discrete_path = os.path.join(EXAMPLES_DIR, f"{name}_discrete.pdb")
                    with open(tunnel_path, "w") as f:
                        f.write(tunnel_pdb)
                    with open(discrete_path, "w") as f:
                        f.write(discrete_pdb)
                    print(f"  Wrote {os.path.basename(tunnel_path)}, {os.path.basename(discrete_path)}")
                    print(f"  Discrete: {refine_res.n_accept}/{refine_res.n_steps} moves accepted, E_ca={refine_res.E_ca_final:.4f} eV")
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmpdir:
                        t_path = os.path.join(tmpdir, "t.pdb")
                        d_path = os.path.join(tmpdir, "d.pdb")
                        with open(t_path, "w") as f:
                            f.write(tunnel_pdb)
                        with open(d_path, "w") as f:
                            f.write(discrete_pdb)
                        rmsd_td, _, _, _ = ca_rmsd(t_path, d_path, align_by_resid=False)
                    print(f"  Cα-RMSD tunnel vs discrete: {rmsd_td:.3f} Å")
                else:
                    print("  Discrete refinement skipped (pyhqiv not available).")

            # Crambin: grade vs 1CRN if gold exists
            if name == "crambin" and os.path.isfile(CRAMBIN_GOLD):
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    pred_path = os.path.join(EXAMPLES_DIR, "crambin_tunnel.pdb")
                    if os.path.isfile(pred_path):
                        rmsd_gold, _, _, _ = ca_rmsd(pred_path, CRAMBIN_GOLD, align_by_resid=False)
                        print(f"  Cα-RMSD vs 1CRN (gold): {rmsd_gold:.3f} Å")
                    pred_d = os.path.join(EXAMPLES_DIR, "crambin_discrete.pdb")
                    if os.path.isfile(pred_d):
                        rmsd_d, _, _, _ = ca_rmsd(pred_d, CRAMBIN_GOLD, align_by_resid=False)
                        print(f"  Cα-RMSD discrete vs 1CRN: {rmsd_d:.3f} Å")
        print(flush=True)


if __name__ == "__main__":
    main()

