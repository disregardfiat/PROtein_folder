#!/usr/bin/env python3
"""
Run the co-translational tunnel method (minimize_full_chain with simulate_ribosome_tunnel=True)
against the same test targets as run_all_pipelines, then score with Cα-RMSD when reference
PDBs are provided.

Usage:
  python -m horizon_physics.proteins.examples.run_tunnel_and_grade
  python -m horizon_physics.proteins.examples.run_tunnel_and_grade --quick
  python -m horizon_physics.proteins.examples.run_tunnel_and_grade --targets crambin,insulin_fragment
  python -m horizon_physics.proteins.examples.run_tunnel_and_grade --ref-dir /path/to/gold
  python -m horizon_physics.proteins.examples.run_tunnel_and_grade --ref-dir . --ref-names crambin:1crn.pdb

Reference PDBs: put reference (gold) PDBs in a directory and pass --ref-dir. By default we look for:
  <ref-dir>/<label>.pdb  (e.g. crambin.pdb, insulin_b_fragment.pdb)
  <ref-dir>/1crn.pdb     (for crambin, if crambin.pdb not found)
Use --ref-names to map target labels to ref files: crambin:1crn.pdb,insulin_b_fragment:2bvr.pdb
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_p = os.path.dirname(os.path.abspath(__file__))
while _p != os.path.dirname(_p):
    if os.path.isfile(os.path.join(_p, "pyproject.toml")):
        _src = os.path.join(_p, "src")
        if _src not in sys.path:
            sys.path.insert(0, _src)
        break
    _p = os.path.dirname(_p)

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))

# Same targets as run_all_pipelines: (name, sequence, short_label for filenames)
TARGETS = [
    ("T1037", "SKINFYTTTIETLETEDQNNTLTTFKVQNVSNASTIFSNGKTYWNFARPSYISNRINTFKNNPGVLRQLLNTSYGQSSLWAKHLLGEEKNVTGDFVLAGNARESASENRLKSLELSIFNSLQEKDKGAEGNDNGSISIVDQLADKLNKVLRGGTKNGTSIYSTVTPGDKSTLHEIKIDHFIPETISSFSNGTMIFNDKIVNAFTDHFVSEVNRMKEAYQELETLPESKRVVHYHTDARGNVMKDGKLAGNAFKSGHILSELSFDQITQDDNEMLKLYNEDGSPINPKGAVSNEQKILIKQTINKVLNQRIKENIRYFKDQGLVIDTVNKDGNKGFHFHGLDKSIMSEYTDDIQLTEFDISHVVSDFTLNSILASIEYTKLFTGDPANYKNMVDFFKRVPATYTN", "T1037_S0A2C3d4"),
    ("T1131", "FVPEEQYNKDFNFLYDYAVIHNLVMDGFSEEDGQYNWDFAKNPDSSRSDESIAYVKELQKLKREDAINFGANAWVLNHNIGFDYKTLKNHQFNLTDANENHSFVVEYWNLKNDETGRHTFWDSVIGEKYGEYLYNADEDTRINGKLKTPYAWVKQILYGIEDAGAPGFSSISA", "T1131_hormaphis_cornu"),
    ("crambin", "TTCCPSIVARSNFNVCRLPGTPEAIICGDVCDLDCTAKTCFSIICT", "crambin"),
    ("insulin_fragment", "FVNQHLCGSHLVEALYLVCGERGFFYTPK", "insulin_b_fragment"),
]

# Default ref PDB names per label (RCSB-style; user can override with --ref-names)
DEFAULT_REF_NAMES = {
    "crambin": ["crambin.pdb", "1crn.pdb"],
    "insulin_b_fragment": ["insulin_b_fragment.pdb", "insulin_fragment.pdb"],
    "T1131_hormaphis_cornu": ["T1131_hormaphis_cornu.pdb"],
    "T1037_S0A2C3d4": ["T1037_S0A2C3d4.pdb"],
}


def run_tunnel(
    name: str,
    sequence: str,
    label: str,
    quick: bool = False,
    trajectory_log_path: str | None = None,
    signal_dump_path: str | None = None,
) -> tuple[str, float, bool, float | None]:
    """Run co-translational tunnel pipeline; write *_minimized_tunnel.pdb. Returns (path, time_s, ok, E_ca_final)."""
    from horizon_physics.proteins import minimize_full_chain, full_chain_to_pdb

    out_path = os.path.join(EXAMPLES_DIR, f"{label}_minimized_tunnel.pdb")
    n_res = len(sequence)
    if quick:
        fast_pass_steps = 2
        min_pass_iter = 5
        include_sidechains = False
    else:
        fast_pass_steps = 5
        min_pass_iter = 15
        include_sidechains = False  # tunnel path is backbone-focused; side chains optional

    t0 = time.time()
    try:
        result = minimize_full_chain(
            sequence,
            include_sidechains=include_sidechains,
            simulate_ribosome_tunnel=True,
            tunnel_length=25.0,
            cone_half_angle_deg=12.0,
            lip_plane_distance=0.0,
            hke_above_tunnel_fraction=0.5,
            trajectory_log_path=trajectory_log_path,
            signal_dump_path=signal_dump_path,
        )
        pdb_str = full_chain_to_pdb(result)
        pdb_str = f"REMARK   {label} co-translational tunnel at {time.strftime('%Y-%m-%d %H:%M:%S')}\n{pdb_str}"
        with open(out_path, "w") as f:
            f.write(pdb_str)
        elapsed = time.time() - t0
        E_ca = result.get("E_ca_final")
        return out_path, elapsed, True, float(E_ca) if E_ca is not None else None
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  Tunnel FAILED: {e}", flush=True)
        return out_path, elapsed, False, None


def find_ref_path(ref_dir: str, label: str, ref_names_map: dict[str, str] | None) -> str | None:
    """Return path to reference PDB for this label, or None if not found."""
    if ref_names_map and label in ref_names_map:
        p = os.path.join(ref_dir, ref_names_map[label])
        return p if os.path.isfile(p) else None
    for cand in DEFAULT_REF_NAMES.get(label, [label + ".pdb"]):
        p = os.path.join(ref_dir, cand)
        if os.path.isfile(p):
            return p
    p = os.path.join(ref_dir, label + ".pdb")
    return p if os.path.isfile(p) else None


def score_prediction(pred_path: str, ref_path: str, align_by_resid: bool = True) -> float | None:
    """Return Cα-RMSD in Å, or None on error."""
    try:
        from horizon_physics.proteins.grade_folds import ca_rmsd
        rmsd_ang, _, _, _ = ca_rmsd(pred_path, ref_path, align_by_resid=align_by_resid)
        return float(rmsd_ang)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run co-translational tunnel on example targets and optionally score vs reference PDBs."
    )
    parser.add_argument("--quick", action="store_true", help="Fewer steps for smoke test")
    parser.add_argument("--targets", type=str, default=None, help="Comma-separated: T1037,T1131,crambin,insulin_fragment (default: all)")
    parser.add_argument("--ref-dir", type=str, default=None, metavar="DIR", help="Directory containing reference PDBs for scoring (e.g. gold/1crn.pdb)")
    parser.add_argument("--ref-names", type=str, default=None, help="Optional mapping label:file,label:file (e.g. crambin:1crn.pdb)")
    parser.add_argument("--score-against", type=str, choices=("cartesian", "hierarchical"), default=None, help="Score tunnel vs existing *_minimized_cartesian.pdb or *_minimized_hierarchical.pdb in examples/ (no ref-dir needed)")
    parser.add_argument("--no-resid", action="store_true", help="Align by residue order instead of residue ID when scoring")
    parser.add_argument("--trajectory-log", type=str, default=None, metavar="PATH", help="Write JSONL trajectory to PATH for live_trajectory_viz (e.g. /tmp/t1037_traj.jsonl)")
    parser.add_argument("--signal-dump", type=str, default=None, metavar="PATH", help="On Ctrl+C (or kill), write current Cα state to this PDB and exit (e.g. /tmp/t1037_dump.pdb)")
    args = parser.parse_args()

    ref_names_map: dict[str, str] = {}
    if args.ref_names:
        for part in args.ref_names.split(","):
            part = part.strip()
            if ":" in part:
                label, fname = part.split(":", 1)
                ref_names_map[label.strip()] = fname.strip()

    if args.targets:
        want = {s.strip().lower() for s in args.targets.split(",")}
        targets_to_run = [t for t in TARGETS if t[0].lower() in want or t[2].lower().replace("_", "") in want]
        if not targets_to_run:
            print("No targets matched. Use: T1037, T1131, crambin, insulin_fragment", file=sys.stderr)
            sys.exit(1)
    else:
        targets_to_run = TARGETS

    print("Co-translational tunnel pipeline (simulate_ribosome_tunnel=True)", flush=True)
    if args.quick:
        print("(quick mode: fewer steps)", flush=True)
    if args.trajectory_log:
        print(f"Trajectory log: {args.trajectory_log} (run: python -m horizon_physics.proteins.examples.live_trajectory_viz {args.trajectory_log})", flush=True)
    print("Outputs: examples/*_minimized_tunnel.pdb\n", flush=True)

    results = []
    for name, sequence, label in targets_to_run:
        n_res = len(sequence)
        print(f"=== {name} ({n_res} residues) ===", flush=True)
        path, elapsed, ok, E_ca = run_tunnel(
            name, sequence, label,
            quick=args.quick,
            trajectory_log_path=args.trajectory_log,
            signal_dump_path=args.signal_dump,
        )
        rmsd_ang: float | None = None
        ref_path: str | None = None
        if ok:
            if args.score_against:
                # Score vs existing pipeline output in examples/
                suffix = "_minimized_cartesian.pdb" if args.score_against == "cartesian" else "_minimized_hierarchical.pdb"
                ref_path = os.path.join(EXAMPLES_DIR, f"{label}{suffix}")
                if os.path.isfile(ref_path):
                    rmsd_ang = score_prediction(path, ref_path, align_by_resid=not args.no_resid)
                    if rmsd_ang is not None:
                        print(f"  Cα-RMSD vs {args.score_against}: {rmsd_ang:.3f} Å", flush=True)
                else:
                    print(f"  No {args.score_against} ref found: {ref_path}", flush=True)
            elif args.ref_dir and os.path.isdir(args.ref_dir):
                ref_path = find_ref_path(args.ref_dir, label, ref_names_map if ref_names_map else None)
                if ref_path:
                    rmsd_ang = score_prediction(path, ref_path, align_by_resid=not args.no_resid)
                    if rmsd_ang is not None:
                        print(f"  Cα-RMSD vs {os.path.basename(ref_path)}: {rmsd_ang:.3f} Å", flush=True)
                    else:
                        print("  Score failed (missing ref or length mismatch)", flush=True)
                else:
                    print("  No reference PDB found in ref-dir", flush=True)
        print(f"  Tunnel: {path}  {elapsed:.1f}s  E_ca={E_ca:.2f} eV" if E_ca is not None else f"  Tunnel: {path}  {elapsed:.1f}s", flush=True)
        print(f"  {'OK' if ok else 'FAIL'}", flush=True)
        results.append((name, label, n_res, path, elapsed, ok, E_ca, rmsd_ang, ref_path))
        print(flush=True)

    # Summary table
    print("=== Summary ===")
    print(f"{'Target':<22} {'n_res':>6} {'Time(s)':>8} {'E_ca(eV)':>10} {'Cα-RMSD(Å)':>12} {'Ref':<20} {'Status':<6}")
    print("-" * 90)
    for name, label, n_res, path, elapsed, ok, E_ca, rmsd_ang, ref_path in results:
        E_str = f"{E_ca:.2f}" if E_ca is not None else "—"
        rmsd_str = f"{rmsd_ang:.3f}" if rmsd_ang is not None else "—"
        ref_str = (os.path.basename(ref_path) if ref_path else "—")[:18]
        status = "OK" if ok else "FAIL"
        print(f"{name:<22} {n_res:>6} {elapsed:>8.1f} {E_str:>10} {rmsd_str:>12} {ref_str:<20} {status:<6}")
    print("-" * 90)

    failed = sum(1 for _ in results if not _[5])
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
