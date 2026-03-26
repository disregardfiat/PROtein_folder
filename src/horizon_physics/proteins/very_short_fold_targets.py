"""
Ultra-short experimental miniproteins for fast folding / band-finding.

Sequences are taken from the referenced PDB **ATOM** Cα records (single model/chain
where applicable). Use for sweeps before scaling to crambin (~46) and beyond.

See ``scripts/fetch_very_short_reference_pdbs.py`` to download ``reference_pdb``
into ``proteins/`` when missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class VeryShortFoldTarget:
    key: str
    sequence: str
    pdb_code: str
    description: str
    reference_pdb: str
    """Filename under repo ``proteins/`` after fetch (e.g. ``1UAO.pdb``)."""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "sequence": self.sequence,
            "n_res": len(self.sequence),
            "pdb_code": self.pdb_code,
            "description": self.description,
            "reference_pdb": self.reference_pdb,
        }


# Ladder: 10 → 20 → 36 aa (crambin ~46 is next step in-repo with ``1CRN.pdb``).
VERY_SHORT_FOLD_TARGETS: Tuple[VeryShortFoldTarget, ...] = (
    VeryShortFoldTarget(
        key="chignolin_1uao",
        sequence="GYDPETGTWG",
        pdb_code="1UAO",
        description="Chignolin CLN025 (10 aa), crystal.",
        reference_pdb="1UAO.pdb",
    ),
    VeryShortFoldTarget(
        key="trpcage_1l2y",
        sequence="NLYIQWLKDGGPSSGRPPPS",
        pdb_code="1L2Y",
        description="Trp-cage TC5b miniprotein (20 aa), NMR model 1.",
        reference_pdb="1L2Y.pdb",
    ),
    VeryShortFoldTarget(
        key="villin_hp35_1vii",
        sequence="MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
        pdb_code="1VII",
        description="Villin headpiece HP-35 (36 aa); PDB uses standard three-letter codes only.",
        reference_pdb="1VII.pdb",
    ),
)

VERY_SHORT_FOLD_TARGETS_LIST: List[VeryShortFoldTarget] = list(VERY_SHORT_FOLD_TARGETS)


def target_by_key(key: str) -> VeryShortFoldTarget:
    for t in VERY_SHORT_FOLD_TARGETS:
        if t.key == key:
            return t
    raise KeyError(f"Unknown very-short target key: {key!r}")
