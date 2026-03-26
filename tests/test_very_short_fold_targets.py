from __future__ import annotations

import os

from horizon_physics.proteins.very_short_fold_targets import (
    VERY_SHORT_FOLD_TARGETS,
    target_by_key,
)


def test_targets_monotone_length_ladder():
    lengths = [len(t.sequence) for t in VERY_SHORT_FOLD_TARGETS]
    assert lengths == sorted(lengths)
    assert lengths[0] >= 5
    assert all(20 <= L <= 50 for L in lengths[1:])


def test_target_by_key():
    t = target_by_key("chignolin_1uao")
    assert t.sequence == "GYDPETGTWG"
    assert len(t.sequence) == 10


def test_reference_pdbs_fetchable_or_present():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proteins = os.path.join(repo, "proteins")
    for t in VERY_SHORT_FOLD_TARGETS:
        path = os.path.join(proteins, t.reference_pdb)
        assert os.path.isfile(path), f"Run: python scripts/fetch_very_short_reference_pdbs.py — missing {path}"
