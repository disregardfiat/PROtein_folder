"""
Surface attachment analysis for protein docking: O(n) exposed-surface detection
and ideal attachment point from horizon φ and vector math.

Given a folded chain (Cα positions), finds where bonding is most likely on the
exposed surface and returns the ideal attachment point and outward direction for
docking. Uses neighbor count for exposure, horizon gradient for outward normal,
and φ for coupling strength.

MIT License. Python 3.10+. Numpy.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from .folding_energy import (
    build_horizon_poles,
    grad_from_poles,
    theta_at_position,
    R_HORIZON,
)
from ._hqiv_base import horizon_scalar

try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except ImportError:
    cKDTree = None
    _HAS_SCIPY = False

# Exposure: atoms with fewer than this many neighbors within exposure_cutoff are "surface"
EXPOSURE_CUTOFF = 8.0   # Å
MAX_NEIGHBORS_EXPOSED = 6
# Top-k exposed atoms to average for attachment point
TOP_K_ATTACHMENT = 12
# Contact distance for ideal placement (Å)
ATTACHMENT_CONTACT_DIST = 7.0


def _neighbor_counts(positions: np.ndarray, cutoff: float = EXPOSURE_CUTOFF) -> np.ndarray:
    """O(n) neighbor count per atom. Returns (n,) array of counts (excluding self)."""
    n = positions.shape[0]
    counts = np.zeros(n, dtype=int)
    if _HAS_SCIPY and n > 1:
        tree = cKDTree(positions)
        for i in range(n):
            idx = tree.query_ball_point(positions[i], cutoff)
            counts[i] = len([j for j in idx if j != i])
    else:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                r = np.linalg.norm(positions[j] - positions[i])
                if r < cutoff:
                    counts[i] += 1
    return counts


def find_attachment_point(
    positions: np.ndarray,
    z_list: np.ndarray | None = None,
    exposure_cutoff: float = EXPOSURE_CUTOFF,
    max_neighbors_exposed: int = MAX_NEIGHBORS_EXPOSED,
    top_k: int = TOP_K_ATTACHMENT,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Find the ideal attachment point and outward direction on a folded chain.

    Uses O(n) surface exposure (neighbor count) and horizon gradient for outward
    normal. Bonding potential = φ × exposure_score (higher = better for coupling).

    Args:
        positions: (n, 3) Cα positions in Å.
        z_list: (n,) z_shell per atom; default all 6.
        exposure_cutoff: Radius for neighbor count (Å).
        max_neighbors_exposed: Atoms with fewer neighbors are "exposed".
        top_k: Number of top exposed atoms to average for centroid/direction.

    Returns:
        (pos, dir_outward, score): Attachment point (centroid of top exposed),
        unit outward direction (negative of horizon gradient), and best score.
        If no exposed atoms, returns (COM, +x, 0.0).
    """
    n = positions.shape[0]
    if n == 0:
        return np.zeros(3), np.array([1.0, 0.0, 0.0]), 0.0
    if z_list is None:
        z_list = np.full(n, 6)
    positions = np.asarray(positions, dtype=float)
    if np.any(np.abs(positions) > 1e6):
        # Sanity: coordinates in Å should be < 1e6; return COM + default dir
        return np.mean(positions, axis=0), np.array([1.0, 0.0, 0.0]), 0.0

    # 1) Exposed surface: neighbor count < max_neighbors_exposed
    counts = _neighbor_counts(positions, cutoff=exposure_cutoff)
    exposed_mask = counts < max_neighbors_exposed
    exposed_indices = np.where(exposed_mask)[0]

    if len(exposed_indices) == 0:
        com = np.mean(positions, axis=0)
        return com, np.array([1.0, 0.0, 0.0]), 0.0

    # 2) Horizon gradient at each atom (points inward from neighbors)
    poles = build_horizon_poles(
        positions, z_list,
        r_horizon=min(R_HORIZON, 12.0),
        use_neighbor_list=True,
    )
    grad = grad_from_poles(poles, n)
    # Outward = -gradient (direction a partner would approach)
    outward = np.asarray(-grad, dtype=float)
    outward = np.nan_to_num(outward, nan=0.0, posinf=0.0, neginf=0.0)
    outward_norms = np.linalg.norm(outward, axis=1, keepdims=True)
    outward_unit = np.zeros_like(outward)
    for i in range(n):
        if outward_norms[i, 0] > 1e-9:
            outward_unit[i] = outward[i] / outward_norms[i, 0]
        else:
            outward_unit[i] = np.array([1.0, 0.0, 0.0])

    # 3) φ at each atom (from theta_at_position)
    phi_arr = np.zeros(n)
    for i in range(n):
        theta_i = theta_at_position(positions, i, int(z_list[i]))
        phi_arr[i] = horizon_scalar(theta_i) if theta_i > 0 else 0.0

    # 4) Bonding score: φ × (max_neighbors - count) — more exposed + higher φ = better
    exposure_score = max_neighbors_exposed - counts
    exposure_score = np.maximum(exposure_score, 0)
    bonding_score = phi_arr * exposure_score

    # 5) Top-k exposed by bonding score
    scores_exposed = bonding_score[exposed_indices]
    order = np.argsort(-scores_exposed)[:top_k]
    top_indices = exposed_indices[order]

    # 6) Centroid and average outward direction
    pos_centroid = np.mean(positions[top_indices], axis=0)
    dir_avg = np.mean(outward_unit[top_indices], axis=0)
    dir_norm = np.linalg.norm(dir_avg)
    if dir_norm < 1e-9:
        dir_avg = np.array([1.0, 0.0, 0.0])
    else:
        dir_avg = dir_avg / dir_norm

    best_score = float(np.mean(bonding_score[top_indices]))
    return pos_centroid, dir_avg, best_score
