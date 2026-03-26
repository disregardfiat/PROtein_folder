#!/usr/bin/env python3
"""Download reference PDBs for ``very_short_fold_targets`` into ``proteins/``."""

from __future__ import annotations

import os
import sys
import urllib.request

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROTEINS = os.path.join(_REPO, "proteins")
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from horizon_physics.proteins.very_short_fold_targets import VERY_SHORT_FOLD_TARGETS  # noqa: E402


def main() -> int:
    os.makedirs(_PROTEINS, exist_ok=True)
    for t in VERY_SHORT_FOLD_TARGETS:
        code = t.pdb_code.upper()
        dest = os.path.join(_PROTEINS, t.reference_pdb)
        if os.path.isfile(dest) and os.path.getsize(dest) > 100:
            print("skip (exists)", dest)
            continue
        url = f"https://files.rcsb.org/download/{code}.pdb"
        print("fetch", url, "->", dest)
        urllib.request.urlretrieve(url, dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
