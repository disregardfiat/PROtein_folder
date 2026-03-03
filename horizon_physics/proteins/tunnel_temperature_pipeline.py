"""
Tunnel + temperature-aware folding pipeline (wrapper around minimize_full_chain).

Stage 0: Cartesian backbone via minimize_full_chain (simulate_ribosome_tunnel=False).
Stage 1–2: Tunnel + post-extrusion HKE via minimize_full_chain (simulate_ribosome_tunnel=True).
Temperature is passed through so higher-level code can eventually use it in Metropolis/annealing.
This module provides a single entry point and a small result dict suitable for examples/tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .full_protein_minimizer import minimize_full_chain, full_chain_to_pdb
from .grade_folds import ca_rmsd, load_ca_from_pdb


@dataclass
class FoldResult:
    sequence: str
    n_res: int
    temperature: float
    cartesian_pdb: str
    tunnel_pdb: str
    ca_rmsd_cart_vs_tunnel: Optional[float]
    tunnel_result: Optional[Dict[str, Any]] = None  # set when return_tunnel_result=True


def fold_with_tunnel_temperature(
    sequence: str,
    temperature: float = 310.0,
    *,
    include_sidechains: bool = False,
    simulate_ribosome_tunnel: bool = True,
    return_tunnel_result: bool = False,
) -> FoldResult:
    """
    High-level folding pipeline:

    1) Cartesian backbone via minimize_full_chain (no tunnel).
    2) Tunnel + post-extrusion HKE via minimize_full_chain(simulate_ribosome_tunnel=True).

    Returns PDB strings for both and an optional Cα-RMSD between them.
    """
    seq = "".join(c for c in sequence.strip().upper() if c.isalpha())
    if not seq:
        raise ValueError("Empty sequence passed to fold_with_tunnel_temperature.")

    # Stage 0: Cartesian baseline (no tunnel)
    cart_result: Dict[str, object] = minimize_full_chain(
        seq,
        include_sidechains=include_sidechains,
        simulate_ribosome_tunnel=False,
    )
    cartesian_pdb = full_chain_to_pdb(cart_result)

    tunnel_result: Optional[Dict[str, Any]] = None
    if simulate_ribosome_tunnel:
        tunnel_result = minimize_full_chain(
            seq,
            include_sidechains=include_sidechains,
            simulate_ribosome_tunnel=True,
        )
        tunnel_pdb = full_chain_to_pdb(tunnel_result)
    else:
        tunnel_pdb = cartesian_pdb

    # Optional Cα-RMSD (order-only; both are MODEL 1, chain A)
    try:
        from io import StringIO
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            cart_path = os.path.join(tmpdir, "cart.pdb")
            tunn_path = os.path.join(tmpdir, "tunnel.pdb")
            with open(cart_path, "w") as f:
                f.write(cartesian_pdb)
            with open(tunn_path, "w") as f:
                f.write(tunnel_pdb)
            rmsd, _, _, _ = ca_rmsd(cart_path, tunn_path, align_by_resid=False)
            ca_rmsd_cart_vs_tunnel = float(rmsd)
    except Exception:
        ca_rmsd_cart_vs_tunnel = None

    return FoldResult(
        sequence=seq,
        n_res=len(seq),
        temperature=float(temperature),
        cartesian_pdb=cartesian_pdb,
        tunnel_pdb=tunnel_pdb,
        ca_rmsd_cart_vs_tunnel=ca_rmsd_cart_vs_tunnel,
        tunnel_result=tunnel_result if return_tunnel_result else None,
    )

