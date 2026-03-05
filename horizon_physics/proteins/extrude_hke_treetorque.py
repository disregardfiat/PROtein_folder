"""
Library helpers for 7K00 tunnel extrusion + tree-torque + optional HKE cycles.

These wrap the example logic in examples/extrude_into_7K00_tunnel.py into
reusable functions suitable for use from the CASP server or other callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import os
import signal
import sys
import time
import math

import numpy as np

from . import full_chain_to_pdb, full_chain_to_pdb_complex, minimize_full_chain
from .backbone_phi_psi import backbone_phi_psi_from_atoms
from .casp_submission import _place_backbone_ca, _place_full_backbone
from .hierarchical import minimize_full_chain_hierarchical, hierarchical_result_for_pdb
from .assembly_dock import run_two_chain_assembly
from .temperature_path_search import run_discrete_refinement


# Repo root: horizon_physics/proteins/extrude_hke_treetorque.py -> horizon_physics -> PROtien
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROTEINS_DIR = os.path.join(_REPO_ROOT, "proteins")
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

    try:
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
    except Exception:
        # e.g. pyhqiv.molecular not available at call time
        pdb_str = full_chain_to_pdb(result)
        return ExtrudeHKETreeResult(
            sequence=seq,
            n_res=n_res,
            pdb=pdb_str,
            backbone_atoms=backbone,
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
    signal_dump_path: Optional[str] = None,
    deadline_sec: Optional[float] = None,
) -> ExtrudeHKETreeResult:
    """
    Three-cycle pipeline for a single chain:

    1) Extrude into 7K00 tunnel + free minimize + tree-torque (run_until_converged).
    2) HKE from that backbone (single pass).
    3) If HKE moves the backbone by more than rmsd_threshold (Cα-RMSD), run tree-torque again.

    If signal_dump_path is set, send SIGUSR1 to the process to write the current stage's
    backbone to that PDB (dump-only, process continues). Use: kill -USR1 <pid>

    If deadline_sec is set (Unix time), before starting each next step we check time.time() >= deadline_sec;
    if over deadline, return current result with meta["timed_out"] = True and stop.
    """
    from .grade_folds import ca_rmsd
    import tempfile

    def _over_deadline() -> bool:
        return deadline_sec is not None and time.time() >= deadline_sec

    # Mutable ref for progress dump on SIGUSR1 (dump-only, no exit)
    _current: Dict[str, Any] = {"backbone_atoms": None, "sequence": None, "n_res": 0, "stage": "init"}

    def _do_signal_dump() -> None:
        if not signal_dump_path or _current["backbone_atoms"] is None:
            return
        bb = _current["backbone_atoms"]
        seq = _current["sequence"] or ""
        if not seq or len(bb) != 4 * _current["n_res"]:
            return
        try:
            pdb_str = full_chain_to_pdb(
                {"backbone_atoms": bb, "sequence": seq, "n_res": _current["n_res"], "include_sidechains": False}
            )
            with open(signal_dump_path, "w") as f:
                f.write(pdb_str)
            sys.stderr.write(f"Signal dump: wrote {_current['stage']} to {signal_dump_path}\n")
        except Exception as e:
            sys.stderr.write(f"Signal dump failed: {e}\n")

    if signal_dump_path and hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, lambda _s, _f: _do_signal_dump())

    base = extrude_and_treetorque(sequence, temperature=temperature, max_phases_cap=max_phases_cap)
    if base is not None:
        _current["backbone_atoms"] = base.backbone_atoms
        _current["sequence"] = base.sequence
        _current["n_res"] = base.n_res
        _current["stage"] = "after_extrude_treetorque"
        if _over_deadline():
            meta = dict(base.meta)
            meta["timed_out"] = True
            return ExtrudeHKETreeResult(
                sequence=base.sequence,
                n_res=base.n_res,
                pdb=base.pdb,
                backbone_atoms=base.backbone_atoms,
                meta=meta,
            )
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
            _current["backbone_atoms"] = refine.backbone_atoms
            _current["sequence"] = refine.sequence
            _current["n_res"] = refine.n_res
            _current["stage"] = "after_fallback_treetorque"
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
        _current["backbone_atoms"] = backbone
        _current["sequence"] = seq
        _current["n_res"] = n_res
        _current["stage"] = "after_fallback_minimize"
        if _over_deadline():
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
                    "timed_out": True,
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
    if _over_deadline():
        meta = dict(base.meta)
        meta["timed_out"] = True
        return ExtrudeHKETreeResult(
            sequence=base.sequence,
            n_res=base.n_res,
            pdb=base.pdb,
            backbone_atoms=base.backbone_atoms,
            meta=meta,
        )
    hke_res = hke_from_backbone_once(
        base.sequence,
        base.backbone_atoms,
        max_iter_stages=hke_max_iter_stages,
    )
    _current["backbone_atoms"] = hke_res.backbone_atoms
    _current["sequence"] = hke_res.sequence
    _current["n_res"] = hke_res.n_res
    _current["stage"] = "after_hke"

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

    if _over_deadline():
        return ExtrudeHKETreeResult(
            sequence=hke_res.sequence,
            n_res=hke_res.n_res,
            pdb=hke_res.pdb,
            backbone_atoms=hke_res.backbone_atoms,
            meta={**meta_combined, "timed_out": True},
        )

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

    if _over_deadline():
        return ExtrudeHKETreeResult(
            sequence=hke_res.sequence,
            n_res=hke_res.n_res,
            pdb=hke_res.pdb,
            backbone_atoms=hke_res.backbone_atoms,
            meta={**meta_combined, "timed_out": True},
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
    _current["backbone_atoms"] = refine2.backbone_atoms
    _current["sequence"] = refine2.sequence
    _current["n_res"] = refine2.n_res
    _current["stage"] = "after_treetorque_2"
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
    temperature: float = 310.0,
    max_phases_cap: int = 100000,
    hke_max_iter_stages: Tuple[int, int, int],
    rmsd_threshold: float = 1.0,
) -> AssemblyExtrudeHKETreeResult:
    """
    Two-chain assembly: run each chain through the single-chain 3-cycle (extrude +
    tree-torque + HKE + optional tree-torque), then add them in reduced form via
    EM-field docking (run_two_chain_assembly), then assembly-mode tree-torque,
    then optional complex re-minimize + second tree-torque if things moved.

    Both chains are folded in their reduced form first; then coupled via the
    most obvious bond site and minimized as a complex.
    """
    from .temperature_path_search import run_discrete_refinement as _run_discrete_refinement
    from .grade_folds import ca_rmsd
    import tempfile

    s_a = "".join(c for c in seq_a.strip().upper() if c.isalpha())
    s_b = "".join(c for c in seq_b.strip().upper() if c.isalpha())
    if not s_a or not s_b:
        raise ValueError("Empty sequence for assembly_hke_treetorque_cycle_two_chains.")

    # 1) Single-chain 3-cycle for each chain (extrude → tree-torque → HKE → tree-torque)
    res_a = extrude_hke_treetorque_cycle(
        seq_a,
        temperature=temperature,
        max_phases_cap=max_phases_cap,
        hke_max_iter_stages=hke_max_iter_stages,
        rmsd_threshold=rmsd_threshold,
    )
    res_b = extrude_hke_treetorque_cycle(
        seq_b,
        temperature=temperature,
        max_phases_cap=max_phases_cap,
        hke_max_iter_stages=hke_max_iter_stages,
        rmsd_threshold=rmsd_threshold,
    )

    # 2) Build result dicts and dock in reduced form (EM field, bond site, minimize complex)
    result_a_dict: Dict[str, Any] = {
        "backbone_atoms": res_a.backbone_atoms,
        "sequence": res_a.sequence,
        "n_res": res_a.n_res,
        "include_sidechains": False,
    }
    result_b_dict: Dict[str, Any] = {
        "backbone_atoms": res_b.backbone_atoms,
        "sequence": res_b.sequence,
        "n_res": res_b.n_res,
        "include_sidechains": False,
    }
    result_a, result_b, result_complex = run_two_chain_assembly(
        result_a_dict,
        result_b_dict,
        max_dock_iter=600,
        converge_max_disp_per_100_res=1.0,
    )
    bb_a = result_complex["backbone_chain_a"]
    bb_b = result_complex["backbone_chain_b"]

    # 3) Assembly-mode tree-torque on each chain (further from COM first)
    if _run_discrete_refinement is not None:
        try:
            ref_a = _run_discrete_refinement(
                s_a,
                initial_backbone_atoms=bb_a,
                run_until_converged=True,
                max_phases_cap=max_phases_cap,
                assembly_mode=True,
            )
            ref_b = _run_discrete_refinement(
                s_b,
                initial_backbone_atoms=bb_b,
                run_until_converged=True,
                max_phases_cap=max_phases_cap,
                assembly_mode=True,
            )
            bb_a = ref_a.backbone_atoms
            bb_b = ref_b.backbone_atoms
        except Exception:
            pass

    pdb_a = full_chain_to_pdb({**result_a, "backbone_atoms": bb_a}, chain_id="A")
    pdb_b = full_chain_to_pdb({**result_b, "backbone_atoms": bb_b}, chain_id="B")
    pdb_complex = full_chain_to_pdb_complex(
        bb_a, bb_b, result_a["sequence"], result_b["sequence"], chain_id_a="A", chain_id_b="B"
    )

    meta: Dict[str, Any] = {
        "sequence_a": s_a,
        "sequence_b": s_b,
    }

    # 4) "HKE" for complex = re-dock and minimize from tree-torque backbones; then optional second tree-torque
    result_tt_a = {
        "backbone_atoms": bb_a,
        "sequence": s_a,
        "n_res": len(s_a),
        "include_sidechains": False,
    }
    result_tt_b = {
        "backbone_atoms": bb_b,
        "sequence": s_b,
        "n_res": len(s_b),
        "include_sidechains": False,
    }
    try:
        _, _, result_complex2 = run_two_chain_assembly(
            result_tt_a,
            result_tt_b,
            max_dock_iter=600,
            converge_max_disp_per_100_res=1.0,
        )
        bb_c_a = result_complex2["backbone_chain_a"]
        bb_c_b = result_complex2["backbone_chain_b"]
    except Exception:
        bb_c_a, bb_c_b = bb_a, bb_b

    # Compare before/after complex re-minimize
    with tempfile.TemporaryDirectory() as tmpdir:
        path_base = os.path.join(tmpdir, "base.pdb")
        path_rem = os.path.join(tmpdir, "rem.pdb")
        with open(path_base, "w") as f:
            f.write(pdb_complex)
        pdb_rem = full_chain_to_pdb_complex(
            bb_c_a, bb_c_b, result_a["sequence"], result_b["sequence"], chain_id_a="A", chain_id_b="B"
        )
        with open(path_rem, "w") as f:
            f.write(pdb_rem)
        try:
            rmsd_c, _, _, _ = ca_rmsd(path_base, path_rem, align_by_resid=False, trim_to_min_length=True)
            delta_rmsd_c = float(rmsd_c)
        except Exception:
            delta_rmsd_c = float("inf")

    meta["delta_ca_rmsd_complex_rem"] = delta_rmsd_c

    if math.isfinite(delta_rmsd_c) and delta_rmsd_c > rmsd_threshold:
        bb_a, bb_b = bb_c_a, bb_c_b
        pdb_a = full_chain_to_pdb({**result_a, "backbone_atoms": bb_a}, chain_id="A")
        pdb_b = full_chain_to_pdb({**result_b, "backbone_atoms": bb_b}, chain_id="B")
        pdb_complex = full_chain_to_pdb_complex(
            bb_a, bb_b, result_a["sequence"], result_b["sequence"], chain_id_a="A", chain_id_b="B"
        )
        # Second assembly-mode tree-torque after complex re-minimize
        if _run_discrete_refinement is not None:
            try:
                ref_a2 = _run_discrete_refinement(
                    s_a,
                    initial_backbone_atoms=bb_a,
                    run_until_converged=True,
                    max_phases_cap=max_phases_cap,
                    assembly_mode=True,
                )
                ref_b2 = _run_discrete_refinement(
                    s_b,
                    initial_backbone_atoms=bb_b,
                    run_until_converged=True,
                    max_phases_cap=max_phases_cap,
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

