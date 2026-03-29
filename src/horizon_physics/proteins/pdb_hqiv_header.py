"""
HQIV CASP-style PDB model header: ``REMARK 999 HQIV-QConSi-<n>-step`` + ``MODEL``.

The tag is exposed as a REMARK so the full string stays human-readable; the ``MODEL`` serial
matches ``<n>``. Override the step with env ``CASP_PDB_QCONSI_STEP`` (default ``1``) or pass
``step=`` where functions support it.
"""

from __future__ import annotations

import os
from typing import List, Optional


def qconsi_step_from_env() -> int:
    raw = (os.environ.get("CASP_PDB_QCONSI_STEP") or "1").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def hqiv_qconsi_model_lines(*, step: Optional[int] = None) -> List[str]:
    """
    Return the opening lines for a single-model PDB (before ATOM/HETATM).

    ``step`` defaults to ``CASP_PDB_QCONSI_STEP`` or ``1``.
    """
    s = max(1, int(step if step is not None else qconsi_step_from_env()))
    return [
        f"REMARK 999 HQIV-QConSi-{s}-step",
        f"MODEL        {s}",
    ]


def hqiv_qconsi_empty_pdb_block(*, step: Optional[int] = None) -> str:
    """No coordinates: REMARK + MODEL + ENDMDL + END."""
    lines = hqiv_qconsi_model_lines(step=step)
    lines.extend(["ENDMDL", "END"])
    return "\n".join(lines) + "\n"
