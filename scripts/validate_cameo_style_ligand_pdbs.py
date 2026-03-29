#!/usr/bin/env python3
"""
Run the QC ligand minimizer against the CAMEO3D-style wwPDB batch (see ``cameo_style_benchmark_pdbs``).

Equivalent to:
  PYTHONPATH=src python3 scripts/validate_qc_ligand_published_pdbs.py --cameo "$@"
"""

from __future__ import annotations

import os
import runpy
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

if __name__ == "__main__":
    script = os.path.join(_REPO, "scripts", "validate_qc_ligand_published_pdbs.py")
    sys.argv = [script, "--cameo"] + sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
