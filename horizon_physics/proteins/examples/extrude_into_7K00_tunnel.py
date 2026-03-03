#!/usr/bin/env python3
"""
Extrude small proteins (crambin, insulin fragment) into the 7K00 ribosome tunnel field.

Pipeline:
  1) Load precomputed 7K00 tunnel EM field + mask from proteins/7K00_tunnel_field.npz.
  2) Build SS-aware Cα trace and backbone; align to tunnel axis / PTC; snap Cα into
     tunnel_mask (null search space) for 0 <= s <= tunnel_length.
  3) Once the chain has cleared the tunnel (all residues placed), run free
     minimize_full_chain (no tunnel) to relax the extruded geometry.
  4) Run the tree-torque field method (discrete φ/ψ + EM unfreezing) until no free
     moves remain (0/n and EM unfreeze returns False). This is the final refinement.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import numpy as np

from horizon_physics.proteins import minimize_full_chain, full_chain_to_pdb
from horizon_physics.proteins.casp_submission import _place_backbone_ca, _place_full_backbone
from horizon_physics.proteins.co_translational_tunnel import align_chain_to_tunnel
from horizon_physics.proteins.temperature_path_search import run_discrete_refinement


EXAMPLES = [
    ("crambin", "TTCCPSIVARSNFNVCRLPGTPEAIICGDVCDLDCTAKTCFSIICT"),
    ("insulin_fragment", "FVNQHLCGSHLVEALYLVCGERGFFYTPK"),
]

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(EXAMPLES_DIR)))
TUNNEL_FIELD_PATH = os.path.join(REPO_ROOT, "proteins", "7K00_tunnel_field.npz")

# Gold-standard references: proteins/1CRN.pdb (crambin), proteins/4INS.pdb (insulin B fragment)
PROTEINS_DIR = os.path.join(REPO_ROOT, "proteins")
GOLD_REF_CANDIDATES = {
    "crambin": [os.path.join(PROTEINS_DIR, "1CRN.pdb")],
    "insulin_fragment": [os.path.join(PROTEINS_DIR, "4INS.pdb")],
}


def _gold_ref_path(name: str) -> str | None:
    for p in GOLD_REF_CANDIDATES.get(name, []):
        if os.path.isfile(p):
            return p
    return None


def _load_tunnel_field_npz(path: str):
    d = np.load(path, allow_pickle=True)
    potential = d["potential"]
    origin = d["origin"]
    res = float(d["res"])
    tunnel_mask = d["tunnel_mask"].astype(bool)
    ptc_origin = d["ptc_origin"]
    axis = d["axis"]
    tunnel_length = float(d["tunnel_length"])
    return potential, origin, res, tunnel_mask, ptc_origin, axis, tunnel_length


def _snap_ca_into_tunnel_mask(
    ca_pos: np.ndarray,
    origin: np.ndarray,
    res: float,
    tunnel_mask: np.ndarray,
    ptc_origin: np.ndarray,
    axis: np.ndarray,
    tunnel_length: float,
) -> np.ndarray:
    """
    For each Cα with 0 <= s <= tunnel_length along axis, move it to the nearest
    voxel center where tunnel_mask is True at that s-slice.
    For s outside this range, positions are left unchanged.
    """
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    out = np.array(ca_pos, dtype=float)
    shape = tunnel_mask.shape
    for i, p in enumerate(out):
        v = p - ptc_origin
        s = float(np.dot(v, axis))
        if s < -1e-3 or s > tunnel_length + 1e-3:
            continue
        # Compute approximate grid indices
        # Assuming axis ≈ +Z for this tunnel field; general axis handled via projection though k from s.
        # Map s to k via z coordinate of p.
        k = int(round((p[2] - origin[2]) / res))
        if k < 0 or k >= shape[2]:
            continue
        i0 = int(round((p[0] - origin[0]) / res))
        j0 = int(round((p[1] - origin[1]) / res))
        best_idx: Tuple[int, int] | None = None
        best_dist = float("inf")
        # Search a small window around (i0, j0)
        radius = 5
        for di in range(-radius, radius + 1):
            ii = i0 + di
            if ii < 0 or ii >= shape[0]:
                continue
            for dj in range(-radius, radius + 1):
                jj = j0 + dj
                if jj < 0 or jj >= shape[1]:
                    continue
                if not tunnel_mask[ii, jj, k]:
                    continue
                xc = origin[0] + ii * res
                yc = origin[1] + jj * res
                d2 = (xc - p[0]) ** 2 + (yc - p[1]) ** 2
                if d2 < best_dist:
                    best_dist = d2
                    best_idx = (ii, jj)
        if best_idx is not None:
            ii, jj = best_idx
            out[i, 0] = origin[0] + ii * res
            out[i, 1] = origin[1] + jj * res
    return out


def run_extrusion(name: str, seq: str, max_phases_cap: int = 10000) -> None:
    print(f"=== {name} extrusion into 7K00 tunnel ===")
    if not os.path.isfile(TUNNEL_FIELD_PATH):
        print("  Tunnel field npz not found:", TUNNEL_FIELD_PATH)
        return
    potential, origin, res, tunnel_mask, ptc_origin, axis, tunnel_length = _load_tunnel_field_npz(
        TUNNEL_FIELD_PATH
    )

    # Build initial CA and align to tunnel axis / PTC
    ca_init = _place_backbone_ca(seq)
    ca_aligned = align_chain_to_tunnel(ca_init, ptc_origin, axis)

    # Snap Cα into tunnel mask for residues inside the tunnel
    ca_in_tunnel = _snap_ca_into_tunnel_mask(
        ca_aligned,
        origin=origin,
        res=res,
        tunnel_mask=tunnel_mask,
        ptc_origin=ptc_origin,
        axis=axis,
        tunnel_length=tunnel_length,
    )

    # Post-extrusion: free minimize (no tunnel) to relax the snapped chain
    t0 = time.time()
    result = minimize_full_chain(
        seq,
        include_sidechains=False,
        simulate_ribosome_tunnel=False,
        ca_init=ca_in_tunnel,
        quick=True,
        long_chain_max_iter=120,
    )
    elapsed = time.time() - t0
    print(f"  Free minimize_full_chain (post-extrusion): {elapsed:.1f}s")

    # Tree-torque field method until no free moves remain (discrete φ/ψ + EM unfreeze)
    backbone = result.get("backbone_atoms")
    if backbone and len(backbone) == 4 * len(seq):
        t1 = time.time()
        refine = run_discrete_refinement(
            seq,
            temperature=310.0,
            n_steps=200,
            initial_backbone_atoms=backbone,
            run_until_converged=True,
            max_phases_cap=max_phases_cap,
        )
        elapsed_refine = time.time() - t1
        print(
            f"  Tree-torque refinement: {refine.n_accept}/{refine.n_steps} accepts, "
            f"{len(refine.info.get('phases', []))} phases, converged={refine.info.get('converged', False)}, "
            f"{elapsed_refine:.1f}s"
        )
        result_pdb = full_chain_to_pdb({
            "backbone_atoms": refine.backbone_atoms,
            "sequence": refine.sequence,
            "n_res": refine.n_res,
            "include_sidechains": False,
        })
        out_pdb = os.path.join(EXAMPLES_DIR, f"{name}_extruded_7K00_tree_torque.pdb")
        with open(out_pdb, "w") as f:
            f.write(result_pdb)
        print(f"  Wrote {os.path.basename(out_pdb)}")

        # Compare to gold standard if available
        ref_path = _gold_ref_path(name)
        if ref_path:
            from horizon_physics.proteins.grade_folds import ca_rmsd

            ref_chain = "B" if "4INS" in os.path.basename(ref_path) and name == "insulin_fragment" else None
            trim = name == "insulin_fragment"  # pred may be 29 vs ref B 30
            rmsd, per_res, _, _ = ca_rmsd(
                out_pdb, ref_path, align_by_resid=False,
                ref_chain_id=ref_chain, trim_to_min_length=trim,
            )
            ref_label = f"{os.path.basename(ref_path)}" + (f" chain {ref_chain}" if ref_chain else "")
            print(f"  Cα-RMSD vs gold ({ref_label}): {rmsd:.3f} Å")
            if per_res is not None:
                print(f"    per-residue: min={float(per_res.min()):.3f} max={float(per_res.max()):.3f} Å")
        else:
            print("  (No gold reference PDB found for Cα-RMSD)")
    else:
        pdb_str = full_chain_to_pdb(result)
        out_pdb = os.path.join(EXAMPLES_DIR, f"{name}_extruded_7K00_minimized.pdb")
        with open(out_pdb, "w") as f:
            f.write(pdb_str)
        print(f"  Wrote {os.path.basename(out_pdb)} (no discrete refinement)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrude into 7K00 tunnel, minimize, tree-torque refine, compare to gold.")
    parser.add_argument("--max-phases", type=int, default=10000, metavar="N", help="Max tree-torque phases (default 10000, use e.g. 30 for quick run)")
    parser.add_argument("--targets", type=str, default=None, help="Comma-separated subset: crambin,insulin_fragment (default: both)")
    args = parser.parse_args()
    targets = [t.strip() for t in args.targets.split(",")] if args.targets else None
    for name, seq in EXAMPLES:
        if targets is not None and name not in targets:
            continue
        run_extrusion(name, seq, max_phases_cap=args.max_phases)
    if not targets or len(targets) == 0:
        print("Gold refs: crambin → proteins/1CRN.pdb, insulin_fragment → proteins/4INS.pdb chain B")


if __name__ == "__main__":
    main()

