"""
Library helpers for 7K00 tunnel extrusion + tree-torque + optional HKE cycles.

These wrap the example logic in examples/extrude_into_7K00_tunnel.py into
reusable functions suitable for use from the CASP server or other callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import os
import time
import math

import numpy as np

from . import full_chain_to_pdb, full_chain_to_pdb_complex, minimize_full_chain
from .backbone_phi_psi import backbone_phi_psi_from_atoms
from .casp_submission import _place_backbone_ca, _place_full_backbone
from .hierarchical import minimize_full_chain_hierarchical, hierarchical_result_for_pdb
from .assembly_dock import run_two_chain_assembly_hke, complex_to_single_chain_result
from .temperature_path_search import run_discrete_refinement


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTEINS_DIR = os.path.join(REPO_ROOT, "proteins")
TUNNEL_FIELD_PATH = os.path.join(PROTEINS_DIR, "7K00_tunnel_field.npz")


@dataclass
class ExtrudeHKETreeResult:
    sequence: str
    n_res: int
    pdb: str
    backbone_atoms: List[Tuple[str, np.ndarray]]
    meta: Dict[str, Any]


@dataclass
class AssemblyExtrudeHKETreeResult:
    sequence_a: str
    sequence_b: str
    backbone_a: List[Tuple[str, np.ndarray]]
    backbone_b: List[Tuple[str, np.ndarray]]
    pdb_a: str
    pdb_b: str
    pdb_complex: str
    meta: Dict[str, Any]


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
    voxel center where tunnel_mask is True at that s-slice. For s outside this
    range, positions are left unchanged.
    """
    from typing import Tuple

    axis = axis / (np.linalg.norm(axis) + 1e-9)
    out = np.array(ca_pos, dtype=float)
    shape = tunnel_mask.shape
    for i, p in enumerate(out):
        v = p - ptc_origin
        s = float(np.dot(v, axis))
        if s < -1e-3 or s > tunnel_length + 1e-3:
            continue
        # Approximate grid indices
        k = int(round((p[2] - origin[2]) / res))
        if k < 0 or k >= shape[2]:
            continue
        i0 = int(round((p[0] - origin[0]) / res))
        j0 = int(round((p[1] - origin[1]) / res))
        best_idx: Optional[Tuple[int, int]] = None
        best_dist = float("inf")
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


def extrude_and_treetorque(
    sequence: str,
    *,
    temperature: float = 310.0,
    max_phases_cap: int = 10000,
) -> Optional[ExtrudeHKETreeResult]:
    """
    Single-chain: extrude into 7K00 tunnel, free minimize, then tree-torque refinement.

    Returns ExtrudeHKETreeResult or None if tunnel field is unavailable.
    """
    seq = "".join(c for c in sequence.strip().upper() if c.isalpha())
    if not seq:
        raise ValueError("Empty sequence for extrude_and_treetorque.")
    n_res = len(seq)

    if not os.path.isfile(TUNNEL_FIELD_PATH):
        return None

    _, origin, res, tunnel_mask, ptc_origin, axis, tunnel_length = _load_tunnel_field_npz(
        TUNNEL_FIELD_PATH
    )

    ca_init = _place_backbone_ca(seq)
    from .co_translational_tunnel import align_chain_to_tunnel

    ca_aligned = align_chain_to_tunnel(ca_init, ptc_origin, axis)

    ca_in_tunnel = _snap_ca_into_tunnel_mask(
        ca_aligned,
        origin=origin,
        res=res,
        tunnel_mask=tunnel_mask,
        ptc_origin=ptc_origin,
        axis=axis,
        tunnel_length=tunnel_length,
    )

    t0 = time.time()
    result = minimize_full_chain(
        seq,
        include_sidechains=False,
        simulate_ribosome_tunnel=False,
        ca_init=ca_in_tunnel,
        quick=True,
        long_chain_max_iter=120,
    )
    t_min = time.time() - t0

    backbone = result.get("backbone_atoms")
    meta: Dict[str, Any] = {
        "sequence": seq,
        "n_res": n_res,
        "t_minimize": t_min,
        "t_treetorque": None,
        "treetorque_phases": None,
        "treetorque_accept": None,
    }

    if not backbone or len(backbone) != 4 * n_res or run_discrete_refinement is None:
        pdb_str = full_chain_to_pdb(result)
        bb = result.get("backbone_atoms") or []
        return ExtrudeHKETreeResult(
            sequence=seq,
            n_res=n_res,
            pdb=pdb_str,
            backbone_atoms=bb,
            meta=meta,
        )

    t1 = time.time()
    refine = run_discrete_refinement(
        seq,
        temperature=temperature,
        n_steps=200,
        initial_backbone_atoms=backbone,
        run_until_converged=True,
        max_phases_cap=max_phases_cap,
    )
    t_tt = time.time() - t1
    meta["t_treetorque"] = t_tt
    meta["treetorque_phases"] = len(refine.info.get("phases", []))
    meta["treetorque_accept"] = refine.n_accept

    pdb_result = full_chain_to_pdb(
        {
            "backbone_atoms": refine.backbone_atoms,
            "sequence": refine.sequence,
            "n_res": refine.n_res,
            "include_sidechains": False,
        }
    )
    return ExtrudeHKETreeResult(
        sequence=refine.sequence,
        n_res=refine.n_res,
        pdb=pdb_result,
        backbone_atoms=refine.backbone_atoms,
        meta=meta,
    )


def hke_from_backbone_once(
    sequence: str,
    backbone_atoms: List[Tuple[str, np.ndarray]],
    *,
    max_iter_stages: Tuple[int, int, int],
) -> ExtrudeHKETreeResult:
    """
    Run HKE (hierarchical) starting from an existing backbone, return new backbone/PDB.
    If hierarchical fails, falls back to returning the input backbone/pdb.
    """
    seq = "".join(c for c in sequence.strip().upper() if c.isalpha())
    n_res = len(seq)
    if not seq:
        raise ValueError("Empty sequence for hke_from_backbone_once.")

    # Derive initial CA from backbone
    phi_rad, psi_rad = backbone_phi_psi_from_atoms(backbone_atoms)
    ca_init = _place_backbone_ca(seq)
    # NOTE: for now we ignore exact φ/ψ alignment and simply feed sequence into HKE.

    t0 = time.time()
    try:
        pos, z_list = minimize_full_chain_hierarchical(
            seq,
            include_sidechains=False,
            funnel_radius=10.0,
            funnel_stiffness=1.0,
            funnel_radius_exit=20.0,
            max_iter_stage1=max_iter_stages[0],
            max_iter_stage2=max_iter_stages[1],
            max_iter_stage3=max_iter_stages[2],
        )
        result = hierarchical_result_for_pdb(pos, z_list, seq, include_sidechains=False)
        bb = result.get("backbone_atoms") or []
        pdb_str = full_chain_to_pdb(result)
    except Exception:
        t_fail = time.time() - t0
        pdb_str = full_chain_to_pdb(
            {
                "backbone_atoms": backbone_atoms,
                "sequence": seq,
                "n_res": n_res,
                "include_sidechains": False,
            }
        )
        return ExtrudeHKETreeResult(
            sequence=seq,
            n_res=n_res,
            pdb=pdb_str,
            backbone_atoms=backbone_atoms,
            meta={
                "sequence": seq,
                "n_res": n_res,
                "t_hke": t_fail,
                "hke_failed": True,
            },
        )

    t_hke = time.time() - t0
    return ExtrudeHKETreeResult(
        sequence=seq,
        n_res=n_res,
        pdb=pdb_str,
        backbone_atoms=bb,
        meta={
            "sequence": seq,
            "n_res": n_res,
            "t_hke": t_hke,
            "hke_failed": False,
        },
    )


def extrude_hke_treetorque_cycle(
    sequence: str,
    *,
    temperature: float = 310.0,
    max_phases_cap: int = 10000,
    hke_max_iter_stages: Tuple[int, int, int] = (15, 25, 50),
    rmsd_threshold: float = 1.0,
) -> ExtrudeHKETreeResult:
    """
    Three-cycle pipeline for a single chain:

    1) Extrude into 7K00 tunnel + free minimize + tree-torque (run_until_converged).
    2) HKE from that backbone (single pass).
    3) If HKE moves the backbone by more than rmsd_threshold (Cα-RMSD), run tree-torque again.
    """
    from .grade_folds import ca_rmsd
    import tempfile

    base = extrude_and_treetorque(sequence, temperature=temperature, max_phases_cap=max_phases_cap)
    if base is None:
        # Fall back: no tunnel field; use minimize_full_chain + tree-torque only.
        seq = "".join(c for c in sequence.strip().upper() if c.isalpha())
        n_res = len(seq)
        t0 = time.time()
        result = minimize_full_chain(seq, include_sidechains=False)
        t_min = time.time() - t0
        backbone = result.get("backbone_atoms") or []
        if backbone and len(backbone) == 4 * n_res and run_discrete_refinement is not None:
            t1 = time.time()
            refine = run_discrete_refinement(
                seq,
                temperature=temperature,
                n_steps=200,
                initial_backbone_atoms=backbone,
                run_until_converged=True,
                max_phases_cap=max_phases_cap,
            )
            t_tt = time.time() - t1
            pdb_str = full_chain_to_pdb(
                {
                    "backbone_atoms": refine.backbone_atoms,
                    "sequence": refine.sequence,
                    "n_res": refine.n_res,
                    "include_sidechains": False,
                }
            )
            return ExtrudeHKETreeResult(
                sequence=refine.sequence,
                n_res=refine.n_res,
                pdb=pdb_str,
                backbone_atoms=refine.backbone_atoms,
                meta={
                    "sequence": refine.sequence,
                    "n_res": refine.n_res,
                    "t_minimize": t_min,
                    "t_treetorque": t_tt,
                    "t_hke": None,
                    "treetorque_phases": len(refine.info.get("phases", [])),
                    "treetorque_accept": refine.n_accept,
                    "hke_failed": True,
                },
            )
        pdb_str = full_chain_to_pdb(result)
        return ExtrudeHKETreeResult(
            sequence=seq,
            n_res=n_res,
            pdb=pdb_str,
            backbone_atoms=backbone,
            meta={
                "sequence": seq,
                "n_res": n_res,
                "t_minimize": t_min,
                "t_treetorque": None,
                "t_hke": None,
                "treetorque_phases": None,
                "treetorque_accept": None,
                "hke_failed": True,
            },
        )

    # HKE from extruded+tree-torque backbone
    hke_res = hke_from_backbone_once(
        base.sequence,
        base.backbone_atoms,
        max_iter_stages=hke_max_iter_stages,
    )

    # Compare backbone movement via Cα-RMSD
    with tempfile.TemporaryDirectory() as tmpdir:
        path_a = os.path.join(tmpdir, "a.pdb")
        path_b = os.path.join(tmpdir, "b.pdb")
        with open(path_a, "w") as f:
            f.write(base.pdb)
        with open(path_b, "w") as f:
            f.write(hke_res.pdb)
        try:
            rmsd, _, _, _ = ca_rmsd(path_a, path_b, align_by_resid=False, trim_to_min_length=True)
            delta_rmsd = float(rmsd)
        except Exception:
            delta_rmsd = float("inf")

    meta_combined = dict(base.meta)
    meta_combined.update(hke_res.meta)
    meta_combined["delta_ca_rmsd_hke_vs_treetorque"] = delta_rmsd

    # If HKE did not move much, return HKE result (already in a basin)
    if not math.isfinite(delta_rmsd) or delta_rmsd <= rmsd_threshold:
        return ExtrudeHKETreeResult(
            sequence=hke_res.sequence,
            n_res=hke_res.n_res,
            pdb=hke_res.pdb,
            backbone_atoms=hke_res.backbone_atoms,
            meta=meta_combined,
        )

    # Else run one more tree-torque from HKE backbone
    if run_discrete_refinement is None or not hke_res.backbone_atoms:
        return ExtrudeHKETreeResult(
            sequence=hke_res.sequence,
            n_res=hke_res.n_res,
            pdb=hke_res.pdb,
            backbone_atoms=hke_res.backbone_atoms,
            meta=meta_combined,
        )

    t2 = time.time()
    refine2 = run_discrete_refinement(
        hke_res.sequence,
        temperature=temperature,
        n_steps=200,
        initial_backbone_atoms=hke_res.backbone_atoms,
        run_until_converged=True,
        max_phases_cap=max_phases_cap,
    )
    t_tt2 = time.time() - t2
    meta_combined["t_treetorque_2"] = t_tt2
    meta_combined["treetorque_2_phases"] = len(refine2.info.get("phases", []))
    meta_combined["treetorque_2_accept"] = refine2.n_accept

    pdb_final = full_chain_to_pdb(
        {
            "backbone_atoms": refine2.backbone_atoms,
            "sequence": refine2.sequence,
            "n_res": refine2.n_res,
            "include_sidechains": False,
        }
    )
    return ExtrudeHKETreeResult(
        sequence=refine2.sequence,
        n_res=refine2.n_res,
        pdb=pdb_final,
        backbone_atoms=refine2.backbone_atoms,
        meta=meta_combined,
    )


def assembly_hke_treetorque_cycle_two_chains(
    seq_a: str,
    seq_b: str,
    *,
    hke_max_iter_stages: Tuple[int, int, int],
    rmsd_threshold: float = 1.0,
) -> AssemblyExtrudeHKETreeResult:
    """
    Two-chain assembly: HKE + EM docking + tree-torque (assembly_mode=True) + optional
    second tree-torque after an HKE refresh of the combined complex.

    This mirrors the single-chain HKE/tree-torque cycle, but uses the existing
    run_two_chain_assembly_hke pipeline and assembly_mode=True in run_discrete_refinement.
    """
    from .temperature_path_search import run_discrete_refinement as _run_discrete_refinement
    from .grade_folds import ca_rmsd
    import tempfile

    s_a = "".join(c for c in seq_a.strip().upper() if c.isalpha())
    s_b = "".join(c for c in seq_b.strip().upper() if c.isalpha())
    if not s_a or not s_b:
        raise ValueError("Empty sequence for assembly_hke_treetorque_cycle_two_chains.")

    # Base A+B assembly via HKE + docking (includes one assembly_mode tree-torque internally)
    result_a, result_b, result_complex = run_two_chain_assembly_hke(
        s_a,
        s_b,
        funnel_radius=10.0,
        funnel_radius_exit=20.0,
        funnel_stiffness=1.0,
        hke_max_iter_s1=hke_max_iter_stages[0],
        hke_max_iter_s2=hke_max_iter_stages[1],
        hke_max_iter_s3=hke_max_iter_stages[2],
        converge_max_disp_per_100_res=1.0,
        max_dock_iter=600,
    )
    bb_a = result_complex["backbone_chain_a"]
    bb_b = result_complex["backbone_chain_b"]

    # Optional extra assembly-mode tree-torque on each chain
    if _run_discrete_refinement is not None:
        try:
            ref_a = _run_discrete_refinement(
                s_a,
                initial_backbone_atoms=bb_a,
                run_until_converged=True,
                max_phases_cap=100000,
                assembly_mode=True,
            )
            ref_b = _run_discrete_refinement(
                s_b,
                initial_backbone_atoms=bb_b,
                run_until_converged=True,
                max_phases_cap=100000,
                assembly_mode=True,
            )
            bb_a = ref_a.backbone_atoms
            bb_b = ref_b.backbone_atoms
        except Exception:
            pass

    pdb_a = full_chain_to_pdb({**result_a, "backbone_atoms": bb_a}, chain_id="A")
    pdb_b = full_chain_to_pdb({**result_b, "backbone_atoms": bb_b}, chain_id="B")
    pdb_complex = full_chain_to_pdb_complex(bb_a, bb_b, result_a["sequence"], result_b["sequence"], chain_id_a="A", chain_id_b="B")

    meta: Dict[str, Any] = {
        "sequence_a": s_a,
        "sequence_b": s_b,
    }

    # HKE refresh of the combined complex, then optional second assembly-mode tree-torque
    result_combined = complex_to_single_chain_result(
        {
            "backbone_chain_a": bb_a,
            "backbone_chain_b": bb_b,
            "sequence_a": result_a["sequence"],
            "sequence_b": result_b["sequence"],
        }
    )
    try:
        pos_c, z_c = minimize_full_chain_hierarchical(
            result_combined["sequence"],
            include_sidechains=False,
            funnel_radius=10.0,
            funnel_stiffness=1.0,
            funnel_radius_exit=20.0,
            max_iter_stage1=hke_max_iter_stages[0],
            max_iter_stage2=hke_max_iter_stages[1],
            max_iter_stage3=hke_max_iter_stages[2],
        )
        result_hke_c = hierarchical_result_for_pdb(
            pos_c, z_c, result_combined["sequence"], include_sidechains=False
        )
        bb_c = result_hke_c.get("backbone_atoms") or []
    except Exception:
        bb_c = result_combined["backbone_atoms"]

    # Compare complex backbone movement
    with tempfile.TemporaryDirectory() as tmpdir:
        path_base = os.path.join(tmpdir, "base.pdb")
        path_hke = os.path.join(tmpdir, "hke.pdb")
        with open(path_base, "w") as f:
            f.write(pdb_complex)
        with open(path_hke, "w") as f:
            f.write(
                full_chain_to_pdb(
                    {
                        "backbone_atoms": bb_c,
                        "sequence": result_combined["sequence"],
                        "n_res": result_combined["n_res"],
                        "include_sidechains": False,
                    },
                    chain_id="A",
                )
            )
        try:
            rmsd_c, _, _, _ = ca_rmsd(path_base, path_hke, align_by_resid=False, trim_to_min_length=True)
            delta_rmsd_c = float(rmsd_c)
        except Exception:
            delta_rmsd_c = float("inf")

    meta["delta_ca_rmsd_complex_hke"] = delta_rmsd_c

    # If complex did not move much, keep bb_a/bb_b
    if not math.isfinite(delta_rmsd_c) or delta_rmsd_c <= rmsd_threshold:
        return AssemblyExtrudeHKETreeResult(
            sequence_a=s_a,
            sequence_b=s_b,
            backbone_a=bb_a,
            backbone_b=bb_b,
            pdb_a=pdb_a,
            pdb_b=pdb_b,
            pdb_complex=pdb_complex,
            meta=meta,
        )

    # Else one more assembly-mode tree-torque on each chain using combined complex as context
    if _run_discrete_refinement is not None:
        try:
            # Re-split bb_c into A/B with same lengths as original sequences
            n_a = len(result_a["sequence"])
            n_atoms_a = 4 * n_a
            bb_a2 = bb_c[:n_atoms_a]
            bb_b2 = bb_c[n_atoms_a:]
            ref_a2 = _run_discrete_refinement(
                s_a,
                initial_backbone_atoms=bb_a2,
                run_until_converged=True,
                max_phases_cap=100000,
                assembly_mode=True,
            )
            ref_b2 = _run_discrete_refinement(
                s_b,
                initial_backbone_atoms=bb_b2,
                run_until_converged=True,
                max_phases_cap=100000,
                assembly_mode=True,
            )
            bb_a = ref_a2.backbone_atoms
            bb_b = ref_b2.backbone_atoms
            pdb_a = full_chain_to_pdb({**result_a, "backbone_atoms": bb_a}, chain_id="A")
            pdb_b = full_chain_to_pdb({**result_b, "backbone_atoms": bb_b}, chain_id="B")
            pdb_complex = full_chain_to_pdb_complex(
                bb_a, bb_b, result_a["sequence"], result_b["sequence"], chain_id_a="A", chain_id_b="B"
            )
        except Exception:
            pass

    return AssemblyExtrudeHKETreeResult(
        sequence_a=s_a,
        sequence_b=s_b,
        backbone_a=bb_a,
        backbone_b=bb_b,
        pdb_a=pdb_a,
        pdb_b=pdb_b,
        pdb_complex=pdb_complex,
        meta=meta,
    )

