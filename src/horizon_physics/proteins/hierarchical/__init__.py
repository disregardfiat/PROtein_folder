# horizon_physics.proteins.hierarchical: Hierarchical Kinematic Engine (HKE)
#
# Optional parallel architecture for protein folding. Use flat (minimize_full_chain)
# or hierarchical (minimize_full_chain_hierarchical) via a single flag.
#
# JAX primary backend; NumPy fallback when JAX not installed.

from .backend import (
    get_backend,
    is_jax_available,
    np_or_jnp,
    jit_if_available,
    grad_if_available,
    device_put_if_available,
    to_numpy,
)
from .atom import Atom
from .rigid_group import RigidGroup
from .protein import Protein, generate_compact_start
from . import energy
from .minimize_hierarchical import (
    minimize_full_chain_hierarchical,
    run_staged_minimization,
    hierarchical_result_for_pdb,
    relative_6dof_to_world_backbone,
)

__all__ = [
    "get_backend",
    "is_jax_available",
    "np_or_jnp",
    "jit_if_available",
    "grad_if_available",
    "device_put_if_available",
    "to_numpy",
    "Atom",
    "RigidGroup",
    "Protein",
    "generate_compact_start",
    "energy",
    "minimize_full_chain_hierarchical",
    "run_staged_minimization",
    "hierarchical_result_for_pdb",
    "relative_6dof_to_world_backbone",
]
