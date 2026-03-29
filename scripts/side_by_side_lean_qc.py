#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from horizon_physics.proteins.side_by_side_lean_qc import cli_main


if __name__ == "__main__":
    raise SystemExit(cli_main())
