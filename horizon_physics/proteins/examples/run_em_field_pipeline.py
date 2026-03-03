#!/usr/bin/env python3
"""
Run EM-field co-translational pipeline on example targets and score vs reference.

Usage:
  python -m horizon_physics.proteins.examples.run_em_field_pipeline
  python -m horizon_physics.proteins.examples.run_em_field_pipeline --targets crambin,insulin_fragment
  python -m horizon_physics.proteins.examples.run_em_field_pipeline --score-against cartesian
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))

TARGETS = [
    ("T1037", "SKINFYTTTIETLETEDQNNTLTTFKVQNVSNASTIFSNGKTYWNFARPSYISNRINTFKNNPGVLRQLLNTSYGQSSLWAKHLLGEEKNVTGDFVLAGNARESASENRLKSLELSIFNSLQEKDKGAEGNDNGSISIVDQLADKLNKVLRGGTKNGTSIYSTVTPGDKSTLHEIKIDHFIPETISSFSNGTMIFNDKIVNAFTDHFVSEVNRMKEAYQELETLPESKRVVHYHTDARGNVMKDGKLAGNAFKSGHILSELSFDQITQDDNEMLKLYNEDGSPINPKGAVSNEQKILIKQTINKVLNQRIKENIRYFKDQGLVIDTVNKDGNKGFHFHGLDKSIMSEYTDDIQLTEFDISHVVSDFTLNSILASIEYTKLFTGDPANYKNMVDFFKRVPATYTN", "T1037_S0A2C3d4"),
    ("T1131", "FVPEEQYNKDFNFLYDYAVIHNLVMDGFSEEDGQYNWDFAKNPDSSRSDESIAYVKELQKLKREDAINFGANAWVLNHNIGFDYKTLKNHQFNLTDANENHSFVVEYWNLKNDETGRHTFWDSVIGEKYGEYLYNADEDTRINGKLKTPYAWVKQILYGIEDAGAPGFSSISA", "T1131_hormaphis_cornu"),
    ("crambin", "TTCCPSIVARSNFNVCRLPGTPEAIICGDVCDLDCTAKTCFSIICT", "crambin"),
    ("insulin_fragment", "FVNQHLCGSHLVEALYLVCGERGFFYTPK", "insulin_b_fragment"),
]


def run_em_field(
    name: str,
    sequence: str,
    label: str,
    batch_size: int = 50,
    compact_until_converged: bool = True,
    alternate_hke: bool = True,
    refine: bool = False,
) -> tuple[str, float, bool, float | None, float | None]:
    """Run EM-field pipeline; write *_minimized_emfield.pdb. Returns (path, time_s, ok, end2end, compactness)."""
    import numpy as np
    from horizon_physics.proteins import CoTranslationalAssembler, atoms_to_pdb

    out_path = os.path.join(EXAMPLES_DIR, f"{label}_minimized_emfield.pdb")
    t0 = time.time()
    try:
        assembler = CoTranslationalAssembler(batch_size=batch_size)
        assembler.run_pipeline(
            sequence,
            compact_until_converged=compact_until_converged,
            alternate_hke=alternate_hke,
            refine=refine,
        )
        spans, end2end = assembler.compactness()
        compactness = float(np.linalg.norm(spans))
        pdb_str = atoms_to_pdb(assembler.atoms)
        pdb_str = f"REMARK   {label} EM-field pipeline at {time.strftime('%Y-%m-%d %H:%M:%S')}\n{pdb_str}"
        with open(out_path, "w") as f:
            f.write(pdb_str)
        elapsed = time.time() - t0
        return out_path, elapsed, True, float(end2end), compactness
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  EM-field FAILED: {e}", flush=True)
        return out_path, elapsed, False, None, None


def score_prediction(pred_path: str, ref_path: str, align_by_resid: bool = True) -> float | None:
    """Return Cα-RMSD in Å, or None on error."""
    try:
        from horizon_physics.proteins.grade_folds import ca_rmsd
        rmsd_ang, _, _, _ = ca_rmsd(pred_path, ref_path, align_by_resid=align_by_resid)
        return float(rmsd_ang)
    except Exception:
        return None


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Run EM-field co-translational pipeline on example targets and score."
    )
    parser.add_argument("--targets", type=str, default=None, help="Comma-separated: T1037,T1131,crambin,insulin_fragment (default: all)")
    parser.add_argument("--batch-size", type=int, default=50, help="Residues from fast assembler before 1-per-pass (default: 50)")
    parser.add_argument("--score-against", type=str, choices=("cartesian", "hierarchical"), default="cartesian",
                        help="Score vs *_minimized_cartesian.pdb or *_minimized_hierarchical.pdb (default: cartesian)")
    parser.add_argument("--no-compact", action="store_true", help="Skip post-assembly relaxation (no free-fold compaction)")
    parser.add_argument("--no-hke", action="store_true", help="Skip alternating HKE steps (field-only relaxation)")
    parser.add_argument("--refine", action="store_true", help="Longer iteration (800 steps), finer res (0.5 Å), tighter convergence (0.01 Å)")
    parser.add_argument("--gold", action="store_true", help="Grade crambin vs experimental 1CRN (download if missing)")
    parser.add_argument("--no-resid", action="store_true", help="Align by residue order when scoring")
    args = parser.parse_args()

    if args.targets:
        want = {s.strip().lower() for s in args.targets.split(",")}
        targets_to_run = [t for t in TARGETS if t[0].lower() in want or t[2].lower().replace("_", "") in want]
        if not targets_to_run:
            print("No targets matched. Use: T1037, T1131, crambin, insulin_fragment", file=sys.stderr)
            sys.exit(1)
    else:
        targets_to_run = TARGETS

    print("EM-field pipeline on example targets")
    print("Outputs: examples/*_minimized_emfield.pdb\n", flush=True)

    results = []
    for name, sequence, label in targets_to_run:
        n_res = len(sequence)
        print(f"=== {name} ({n_res} residues) ===", flush=True)
        path, elapsed, ok, end2end, compact = run_em_field(
            name, sequence, label, batch_size=args.batch_size,
            compact_until_converged=not args.no_compact,
            alternate_hke=not args.no_hke,
            refine=args.refine,
        )
        rmsd_ang: float | None = None
        ref_path: str | None = None
        if ok:
            suffix = f"_minimized_{args.score_against}.pdb"
            ref_path = os.path.join(EXAMPLES_DIR, f"{label}{suffix}")
            if not os.path.isfile(ref_path) and args.score_against == "cartesian":
                # Fallback: score vs tunnel when cartesian ref missing (e.g. insulin)
                tunnel_path = os.path.join(EXAMPLES_DIR, f"{label}_minimized_tunnel.pdb")
                if os.path.isfile(tunnel_path):
                    ref_path = tunnel_path
                    ref_label = "tunnel"
            else:
                ref_label = args.score_against
            if os.path.isfile(ref_path):
                rmsd_ang = score_prediction(path, ref_path, align_by_resid=not args.no_resid)
                if rmsd_ang is not None:
                    print(f"  Cα-RMSD vs {ref_label}: {rmsd_ang:.3f} Å", flush=True)
            else:
                print(f"  No {args.score_against} ref found: {ref_path}", flush=True)
        e2e_str = f"{end2end:.1f}" if end2end is not None else "—"
        compact_str = f"{compact:.1f}" if compact is not None else "—"
        print(f"  EM-field: {path}  {elapsed:.1f}s  E2E={e2e_str} Å  span={compact_str} Å", flush=True)
        if ok and args.gold and label == "crambin":
            gold_path = os.path.join(EXAMPLES_DIR, "crambin_1CRN.pdb")
            if os.path.isfile(gold_path):
                gold_rmsd = score_prediction(path, gold_path, align_by_resid=False)
                if gold_rmsd is not None:
                    print(f"  Cα-RMSD vs 1CRN (gold): {gold_rmsd:.3f} Å", flush=True)
        print(f"  {'OK' if ok else 'FAIL'}", flush=True)
        results.append((name, label, n_res, path, elapsed, ok, end2end, compact, rmsd_ang, ref_path))
        print(flush=True)

    # Summary table
    print("=== Summary ===")
    print(f"{'Target':<22} {'n_res':>6} {'Time(s)':>8} {'E2E(Å)':>8} {'Span(Å)':>8} {'Cα-RMSD(Å)':>12} {'Ref':<25} {'Status':<6}")
    print("-" * 100)
    for name, label, n_res, path, elapsed, ok, end2end, compact, rmsd_ang, ref_path in results:
        e2e_str = f"{end2end:.1f}" if end2end is not None else "—"
        compact_str = f"{compact:.1f}" if compact is not None else "—"
        rmsd_str = f"{rmsd_ang:.3f}" if rmsd_ang is not None else "—"
        ref_str = (os.path.basename(ref_path) if ref_path else "—")[:23]
        status = "OK" if ok else "FAIL"
        print(f"{name:<22} {n_res:>6} {elapsed:>8.1f} {e2e_str:>8} {compact_str:>8} {rmsd_str:>12} {ref_str:<25} {status:<6}")
    print("-" * 100)

    failed = sum(1 for _ in results if not _[5])
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
