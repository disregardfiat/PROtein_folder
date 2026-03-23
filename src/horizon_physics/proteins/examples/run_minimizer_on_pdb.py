#!/usr/bin/env python3
"""
Run the minimizer starting from an existing PDB (troubleshooting / refine a structure).

Loads Cα positions and sequence from the PDB (CA lines; residue names 17-20),
then runs minimize_full_chain(sequence, ca_init=loaded_ca, ...). Optionally use
tunnel mode, signal dump on Ctrl+C, and trajectory log.

Usage:
  python -m horizon_physics.proteins.examples.run_minimizer_on_pdb model.pdb -o refined.pdb
  python -m horizon_physics.proteins.examples.run_minimizer_on_pdb model.pdb -o out.pdb --tunnel --signal-dump /tmp/dump.pdb --trajectory-log /tmp/traj.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
_p = EXAMPLES_DIR
while _p != os.path.dirname(_p):
    if os.path.isfile(os.path.join(_p, "pyproject.toml")):
        _src = os.path.join(_p, "src")
        if _src not in sys.path:
            sys.path.insert(0, _src)
        break
    _p = os.path.dirname(_p)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run minimizer from an existing PDB (load CA + sequence, then minimize)."
    )
    parser.add_argument("input_pdb", help="Input PDB path (CA and sequence are read from it)")
    parser.add_argument("-o", "--output", required=True, help="Output PDB path")
    parser.add_argument("--tunnel", action="store_true", help="Use co-translational tunnel mode")
    parser.add_argument("--signal-dump", type=str, default=None, metavar="PATH", help="On Ctrl+C, write current state to this PDB and exit")
    parser.add_argument("--trajectory-log", type=str, default=None, metavar="PATH", help="Write JSONL trajectory for live_trajectory_viz")
    parser.add_argument("--quick", action="store_true", help="Fewer iterations, faster run")
    parser.add_argument("--no-post-extrusion", action="store_true", dest="no_post_extrusion", help="If --tunnel, skip post-extrusion refinement")
    args = parser.parse_args()

    from horizon_physics.proteins import (
        load_ca_and_sequence_from_pdb,
        minimize_full_chain,
        full_chain_to_pdb,
    )

    if not os.path.isfile(args.input_pdb):
        print(f"Error: not a file: {args.input_pdb}", file=sys.stderr)
        return 1

    ca, sequence = load_ca_and_sequence_from_pdb(args.input_pdb)
    if ca.shape[0] == 0:
        print("Error: no CA atoms found in PDB", file=sys.stderr)
        return 1
    if "X" in sequence:
        print("Warning: unknown residue types in PDB (shown as X); minimizer may not treat them correctly.", file=sys.stderr)

    print(f"Loaded {ca.shape[0]} residues from {args.input_pdb}")
    result = minimize_full_chain(
        sequence,
        ca_init=ca,
        quick=args.quick,
        simulate_ribosome_tunnel=args.tunnel,
        post_extrusion_refine=args.tunnel and not args.no_post_extrusion,
        trajectory_log_path=args.trajectory_log,
        signal_dump_path=args.signal_dump,
    )
    pdb_str = full_chain_to_pdb(result)
    with open(args.output, "w") as f:
        f.write(pdb_str)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
