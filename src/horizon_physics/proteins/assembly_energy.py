"""
Assembly / tree-shaped energy bookkeeping aligned with Lean HQIV assembly algebra.

Lean references (same formal story for proteins, layered stacks, part graphs):
  - ``Hqiv.Physics.HQIVMolecules`` — ``assembly_foldEnergy_branch_eq``,
    ``assembly_foldEnergy_binary_branch``, list-sum lemmas.
  - ``Hqiv.Physics.HQIVAssembly`` — ``assembly_energy_branch_decomposition`` (name bridge to
    ``foldEnergy`` / ``TorqueTree``).

Python scope (reliable, explicit):
  - **Atomic / site budget:** per-Cα ``e_atomic_site`` sums to ``e_tot`` (informational + damping).
  - **Bond valley edges:** ``bond_valley_em_scalar`` on each tree edge (parent → child root);
    for a **path graph** (linear polymer) this matches ``e_sequential_bond_penalty_ca``.
  - **Clash:** pairwise penalty as in ``e_tot_ca_with_bonds`` — *not* part of the Lean branch
    identity as stated; keep as an extra additive term when comparing to ``e_tot_ca_with_bonds``.

**Horizon (EM vector sum):** ``grad_full`` adds long-range horizon forces without a single scalar
``E_horizon`` in all evaluators. Treat horizon as the documented additive *gradient* correction on
top of bonds + local terms, per HQIVAssembly notes — do not equate this module's scalar budget with
full L-BFGS objective unless you integrate horizon energy separately.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .folding_energy import (
    K_BOND,
    R_BOND_MAX,
    R_BOND_MIN,
    R_CA_CA_EQ,
    bond_valley_em_scalar,
    e_atomic_site,
    e_clash_penalty_ca_windowed,
    e_sequential_bond_penalty_ca,
    e_tot,
    e_tot_ca_with_bonds,
)


def sum_bond_valley_over_edges(
    positions: np.ndarray,
    edges: List[Tuple[int, int]],
    *,
    r_eq: float = R_CA_CA_EQ,
    r_min: float = R_BOND_MIN,
    r_max: float = R_BOND_MAX,
    k_bond: float = K_BOND,
) -> float:
    """
    Σ V_bond along explicit undirected tree (or arbitrary) edges (i, j).

    For each edge, uses distance ‖x_i − x_j‖ and ``bond_valley_em_scalar`` (same as sequential Cα
    bonds). Orientation does not matter.
    """
    s = 0.0
    for i, j in edges:
        d = positions[j] - positions[i]
        r = float(np.linalg.norm(d))
        s += bond_valley_em_scalar(r, r_eq=r_eq, r_min=r_min, r_max=r_max, k_bond=k_bond)
    return float(s)


def linear_polymer_parent_list(n: int) -> List[Optional[int]]:
    """Path graph 0—1—…—(n−1): root 0, parent[i]=i−1 for i>0."""
    if n <= 0:
        return []
    return [None] + [i - 1 for i in range(1, n)]


def _children_from_parent(parent_of: List[Optional[int]]) -> Tuple[Dict[int, List[int]], List[int]]:
    n = len(parent_of)
    ch: Dict[int, List[int]] = {i: [] for i in range(n)}
    roots: List[int] = []
    for i, p in enumerate(parent_of):
        if p is None:
            roots.append(i)
        else:
            if not (0 <= int(p) < n):
                raise ValueError(f"assembly_energy: invalid parent {p} for child {i}")
            ch[int(p)].append(i)
    return ch, roots


def assembly_fold_energy_tree_sum(
    positions: np.ndarray,
    z_list: np.ndarray,
    parent_of: List[Optional[int]],
) -> float:
    """
    Evaluate the **branch recursion** analogue of ``assembly_foldEnergy_branch_eq`` on a rooted tree:

    E(tree) = Σ_i E_atomic(i) + Σ_{(p,c) tree edge} V_bond(p, c),

    where tree edges are parent→child links from ``parent_of``.

    For the linear polymer parent list from ``linear_polymer_parent_list(n)``, this equals
    ``e_tot + e_sequential_bond_penalty_ca`` on the same ``positions``/``z_list`` (see tests).
    """
    _, roots = _children_from_parent(parent_of)
    if len(roots) != 1:
        raise ValueError(
            "assembly_fold_energy_tree_sum: expected exactly one root (one None in parent_of), "
            f"got roots={roots!r}"
        )
    root = roots[0]
    ch, _ = _children_from_parent(parent_of)

    def sub(node: int) -> float:
        acc = e_atomic_site(positions, z_list, node)
        for c in ch[node]:
            d = positions[c] - positions[node]
            r = float(np.linalg.norm(d))
            acc += sub(c) + bond_valley_em_scalar(r)
        return acc

    return float(sub(root))


def decompose_ca_fold_energy_scalar_budget(
    positions: np.ndarray,
    z_list: np.ndarray,
    *,
    parent_of: Optional[List[Optional[int]]] = None,
) -> Dict[str, float]:
    """
    Scalar breakdown for analysis / logging (no horizon scalar here).

    Keys: ``atomic_sum`` (= ``e_tot``), ``bond_sequential`` or ``bond_tree``, ``clash``,
    ``total_ca_with_bonds`` (= ``e_tot_ca_with_bonds``), ``tree_sum_check`` when ``parent_of`` set.

    If ``parent_of`` is None, uses the linear polymer path (same bonds as ``e_tot_ca_with_bonds``).
    """
    n = positions.shape[0]
    if n == 0:
        return {
            "atomic_sum": 0.0,
            "bond_sequential": 0.0,
            "bond_tree_recursive": 0.0,
            "clash": 0.0,
            "total_ca_with_bonds": 0.0,
            "tree_sum_atomic_plus_bonds": 0.0,
            "branch_eq_linear_ok": 0.0,
        }
    par = parent_of if parent_of is not None else linear_polymer_parent_list(n)
    bond_seq = e_sequential_bond_penalty_ca(positions)
    tree_sum = assembly_fold_energy_tree_sum(positions, z_list, par)
    clash = e_clash_penalty_ca_windowed(positions)
    etot = e_tot(positions, z_list)
    full = e_tot_ca_with_bonds(positions, z_list)
    out: Dict[str, float] = {
        "atomic_sum": float(etot),
        "bond_sequential": float(bond_seq),
        "bond_tree_recursive": float(tree_sum - etot),
        "clash": float(clash),
        "total_ca_with_bonds": float(full),
        "tree_sum_atomic_plus_bonds": float(tree_sum),
    }
    # For a path tree, bond_tree_recursive should match bond_sequential
    out["branch_eq_linear_ok"] = float(abs(out["bond_tree_recursive"] - bond_seq))
    return out
