#!/usr/bin/env python3
"""
Run all four examples (T1037, T1131, crambin, insulin fragment) against both pipelines:
  - Cartesian (flat): minimize_full_chain → *_minimized_cartesian.pdb
  - Hierarchical (HKE): minimize_full_chain_hierarchical → *_minimized_hierarchical.pdb

Usage:
  python -m horizon_physics.proteins.examples.run_all_pipelines           # all four, both pipelines
  python -m horizon_physics.proteins.examples.run_all_pipelines --quick    # fewer iters, backbone only
  python -m horizon_physics.proteins.examples.run_all_pipelines --targets crambin,insulin_fragment  # subset
  python -m horizon_physics.proteins.examples.run_all_pipelines --cartesian-only   # or --hierarchical-only
"""

from __future__ import annotations

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

# All four targets: (name, sequence, short_label for filenames)
TARGETS = [
    ("T1037", "SKINFYTTTIETLETEDQNNTLTTFKVQNVSNASTIFSNGKTYWNFARPSYISNRINTFKNNPGVLRQLLNTSYGQSSLWAKHLLGEEKNVTGDFVLAGNARESASENRLKSLELSIFNSLQEKDKGAEGNDNGSISIVDQLADKLNKVLRGGTKNGTSIYSTVTPGDKSTLHEIKIDHFIPETISSFSNGTMIFNDKIVNAFTDHFVSEVNRMKEAYQELETLPESKRVVHYHTDARGNVMKDGKLAGNAFKSGHILSELSFDQITQDDNEMLKLYNEDGSPINPKGAVSNEQKILIKQTINKVLNQRIKENIRYFKDQGLVIDTVNKDGNKGFHFHGLDKSIMSEYTDDIQLTEFDISHVVSDFTLNSILASIEYTKLFTGDPANYKNMVDFFKRVPATYTN", "T1037_S0A2C3d4"),
    ("T1131", "FVPEEQYNKDFNFLYDYAVIHNLVMDGFSEEDGQYNWDFAKNPDSSRSDESIAYVKELQKLKREDAINFGANAWVLNHNIGFDYKTLKNHQFNLTDANENHSFVVEYWNLKNDETGRHTFWDSVIGEKYGEYLYNADEDTRINGKLKTPYAWVKQILYGIEDAGAPGFSSISA", "T1131_hormaphis_cornu"),
    ("crambin", "TTCCPSIVARSNFNVCRLPGTPEAIICGDVCDLDCTAKTCFSIICT", "crambin"),
    ("insulin_fragment", "FVNQHLCGSHLVEALYLVCGERGFFYTPK", "insulin_b_fragment"),
]


def _atom_count_and_n_res(pdb_path: str) -> tuple[int, int]:
    n_atoms = 0
    res_ids = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM "):
                n_atoms += 1
                try:
                    res_ids.add(int(line[22:26]))
                except ValueError:
                    pass
            if line.startswith("ENDMDL"):
                break
    return n_atoms, len(res_ids)


def run_cartesian(name: str, sequence: str, label: str, include_sidechains: bool = True, quick: bool = False, trajectory_log_dir: str | None = None) -> tuple[str, float, bool]:
    """Run flat pipeline; write *_minimized_cartesian.pdb. Returns (path, time_s, ok)."""
    from horizon_physics.proteins import minimize_full_chain, full_chain_to_pdb
    out_path = os.path.join(EXAMPLES_DIR, f"{label}_minimized_cartesian.pdb")
    n_res = len(sequence)
    if quick:
        max_iter = 30 if n_res > 100 else 50
        long_iter = 25 if n_res > 100 else None
        include_sidechains = False
    else:
        max_iter = 80 if n_res > 100 else 150
        long_iter = 60 if n_res > 100 else None
    traj_path = os.path.join(trajectory_log_dir, f"{label}_cartesian_traj.jsonl") if trajectory_log_dir else None
    t0 = time.time()
    try:
        result = minimize_full_chain(
            sequence,
            include_sidechains=include_sidechains,
            max_iter=max_iter,
            long_chain_max_iter=long_iter,
            trajectory_log_path=traj_path,
        )
        pdb_str = full_chain_to_pdb(result)
        pdb_str = f"REMARK   {label} Cartesian pipeline at {time.strftime('%Y-%m-%d %H:%M:%S')}\n{pdb_str}"
        with open(out_path, "w") as f:
            f.write(pdb_str)
        elapsed = time.time() - t0
        return out_path, elapsed, True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  Cartesian FAILED: {e}")
        return out_path, elapsed, False


def run_hierarchical(name: str, sequence: str, label: str, include_sidechains: bool = True, quick: bool = False, trajectory_log_dir: str | None = None) -> tuple[str, float, bool]:
    """Run HKE pipeline; write *_minimized_hierarchical.pdb. Returns (path, time_s, ok)."""
    from horizon_physics.proteins import minimize_full_chain_hierarchical, full_chain_to_pdb
    from horizon_physics.proteins.hierarchical import hierarchical_result_for_pdb
    out_path = os.path.join(EXAMPLES_DIR, f"{label}_minimized_hierarchical.pdb")
    n_res = len(sequence)
    if quick:
        max_s1, max_s2, max_s3 = (10, 15, 20) if n_res > 100 else (15, 20, 30)
        include_sidechains = False
    else:
        max_s1, max_s2, max_s3 = (25, 35, 50) if n_res > 100 else (40, 50, 80)
    traj_path = os.path.join(trajectory_log_dir, f"{label}_hierarchical_traj.jsonl") if trajectory_log_dir else None
    t0 = time.time()
    try:
        pos, z_list = minimize_full_chain_hierarchical(
            sequence,
            include_sidechains=include_sidechains,
            device="cpu",
            grouping_strategy="residue",
            max_iter_stage1=max_s1,
            max_iter_stage2=max_s2,
            max_iter_stage3=max_s3,
            trajectory_log_path=traj_path,
        )
        result = hierarchical_result_for_pdb(pos, z_list, sequence, include_sidechains)
        pdb_str = full_chain_to_pdb(result)
        pdb_str = f"REMARK   {label} Hierarchical pipeline at {time.strftime('%Y-%m-%d %H:%M:%S')}\n{pdb_str}"
        with open(out_path, "w") as f:
            f.write(pdb_str)
        elapsed = time.time() - t0
        return out_path, elapsed, True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  Hierarchical FAILED: {e}")
        return out_path, elapsed, False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run all four examples (T1037, T1131, crambin, insulin) with Cartesian and Hierarchical pipelines.")
    parser.add_argument("--quick", action="store_true", help="Fewer iterations, no sidechains; faster run for CI/smoke test")
    parser.add_argument("--cartesian-only", action="store_true", help="Run only Cartesian pipeline")
    parser.add_argument("--hierarchical-only", action="store_true", help="Run only Hierarchical pipeline")
    parser.add_argument("--targets", type=str, default=None, help="Comma-separated subset: T1037,T1131,crambin,insulin_fragment (default: all four)")
    parser.add_argument("--trajectory-log", "-l", type=str, default=None, metavar="DIR", help="Write JSONL trajectory (one line per step: {\"t\", \"positions\"}) to DIR/<label>_cartesian_traj.jsonl and DIR/<label>_hierarchical_traj.jsonl for Manim/tail -f")
    args = parser.parse_args()
    quick = args.quick
    run_cart = not args.hierarchical_only
    run_hier = not args.cartesian_only
    traj_dir = args.trajectory_log
    if traj_dir and not os.path.isdir(traj_dir):
        os.makedirs(traj_dir, exist_ok=True)
    if args.targets:
        want = {s.strip().lower() for s in args.targets.split(",")}
        targets_to_run = [t for t in TARGETS if t[0].lower() in want or t[2].lower().replace("_", "") in want]
        if not targets_to_run:
            print("No targets matched. Use: T1037, T1131, crambin, insulin_fragment")
            sys.exit(1)
    else:
        targets_to_run = TARGETS

    print("Run examples against Cartesian and Hierarchical pipelines", flush=True)
    if quick:
        print("(quick mode: fewer iters, backbone only)", flush=True)
    print("Outputs: examples/*_minimized_cartesian.pdb and *_minimized_hierarchical.pdb\n", flush=True)
    results = []
    for name, sequence, label in targets_to_run:
        n_res = len(sequence)
        print(f"=== {name} ({n_res} residues) ===", flush=True)
        ok_c, ok_h, path_c, path_h = False, False, "", ""
        if run_cart:
            path_c, time_c, ok_c = run_cartesian(name, sequence, label, quick=quick, trajectory_log_dir=traj_dir)
            print(f"  Cartesian:  {path_c}  {time_c:.1f}s  {'OK' if ok_c else 'FAIL'}", flush=True)
            results.append((name, "cartesian", path_c, time_c, ok_c))
        if run_hier:
            path_h, time_h, ok_h = run_hierarchical(name, sequence, label, quick=quick, trajectory_log_dir=traj_dir)
            print(f"  Hierarchical: {path_h}  {time_h:.1f}s  {'OK' if ok_h else 'FAIL'}", flush=True)
            results.append((name, "hierarchical", path_h, time_h, ok_h))
        # Compare atom/residue counts if both pipelines ran and succeeded
        if run_cart and run_hier and ok_c and ok_h:
            nc, rc = _atom_count_and_n_res(path_c)
            nh, rh = _atom_count_and_n_res(path_h)
            match = nc == nh and rc == rh and rc == n_res
            print(f"  Compare: cartesian {nc} atoms {rc} res | hierarchical {nh} atoms {rh} res  {'match' if match else 'MISMATCH'}")
        print()
    # Summary
    print("=== Summary ===")
    for name, pipeline, path, elapsed, ok in results:
        print(f"  {name} {pipeline}: {'OK' if ok else 'FAIL'}  {elapsed:.1f}s  {os.path.basename(path)}")
    failed = sum(1 for _, _, _, _, ok in results if not ok)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
