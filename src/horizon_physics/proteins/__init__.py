# horizon_physics.proteins: HQIV first-principles protein structure and folding.
# MIT License. Python 3.10+. No external dependencies beyond numpy.

from .peptide_backbone import (
    backbone_bond_lengths,
    backbone_geometry,
    omega_peptide_deg,
    ramachandran_alpha,
    ramachandran_beta,
    rational_ramachandran_alpha,
    ramachandran_map,
)
from .alpha_helix import alpha_helix_geometry, alpha_helix_xyz, rational_alpha_parameters

# Alias for README / external use
hqiv_alpha_helix = alpha_helix_geometry
from .beta_sheet import (
    beta_sheet_geometry,
    beta_sheet_parallel_geometry,
    beta_sheet_antiparallel_geometry,
)
from .side_chain_placement import (
    side_chain_chi_preferences,
    chi_angles_for_residue,
    side_chain_placement_geometry,
    AA_LIST,
)
from .folding_energy import (
    e_tot,
    minimize_e_tot,
    small_peptide_energy,
    build_horizon_poles,
    build_bond_poles,
    grad_from_poles,
    full_atom_polymer_energy_budget,
)
from .casp_submission import hqiv_predict_structure, hqiv_predict_structure_assembly
from .gradient_descent_folding import (
    minimize_e_tot_lbfgs,
    rational_alpha_parameters as rational_alpha_parameters_folding,
)
from .secondary_structure_predictor import (
    predict_ss,
    predict_ss_with_angles,
    theta_eff_residue,
    preferred_basin_phi_psi,
)
from .full_protein_minimizer import minimize_full_chain, full_chain_to_pdb, full_chain_to_pdb_complex, pack_sidechains
from .grade_folds import ca_rmsd, load_ca_from_pdb, load_ca_and_sequence_from_pdb, kabsch_superpose
from .surface_attachment import find_attachment_point
from .em_field_pipeline import Atom, EMField, CoTranslationalAssembler, atoms_to_pdb
from .tunnel_temperature_pipeline import fold_with_tunnel_temperature
from .lean_ribosome_tunnel_pipeline import fold_lean_ribosome_tunnel, LeanTunnelFoldResult
from .pipeline_interchange import (
    FoldState,
    make_minimize_stage,
    make_force_carrier_stage,
    make_lean_tunnel_stage,
    make_osh_oracle_stage,
    make_full_heavy_osh_stage,
    run_pipeline,
    run_tunnel_first_pipeline,
)
from .osh_oracle_folding import (
    SparseRegister,
    sparse_basis_card,
    wrap_idx,
    causal_expand_support,
    dense_of_sparse,
    apply_gate_sparse,
    apply_ansatz_sparse,
    current_parameters,
    detect_flipped_kets,
    detect_flipped_kets_amplitude,
    estimate_natural_harmonic_scale_ca,
    harmonic_temperature_schedule,
    metropolis_accept_with_harmonic,
    harmonic_tunneling_mixer_strength,
    qpe_low_energy_subspace,
    amplify_low_energy,
    sparse_visible_energy,
    auto_detect_cys_ligation_pairs,
    QAOAHarmonicFoldInfo,
    harmonic_tunneled_qaoa_folding,
    prune_to_flipped,
    OSHOracleFoldInfo,
    minimize_ca_with_osh_oracle,
    contact_reflector_indices,
    compute_tunnel_harmonic_budget_ev,
)
from .osh_oracle_backbone import minimize_backbone_with_osh_oracle
from .osh_oracle_full_atom import minimize_full_heavy_with_osh_oracle
from .full_atom_topology import (
    FullHeavyAtomChain,
    build_full_heavy_chain,
    chain_to_pdb_line_string,
    full_heavy_chain_energy_budget,
)
from .very_short_fold_targets import (
    VERY_SHORT_FOLD_TARGETS,
    VERY_SHORT_FOLD_TARGETS_LIST,
    VeryShortFoldTarget,
    target_by_key,
)
from .ligands import (
    LigandAgent,
    ligand_summary,
    parse_ligands,
    parse_ligands_from_file,
    z_shell_for_element,
)
from .hqiv_long_range import (
    K_hbond,
    R_hbond,
    h_bond_proxy,
    phi_of_shell,
    available_modes_nat,
    total_h_bond_proxy_energy_ca,
)

# Optional hierarchical kinematic engine (parallel path)
try:
    from .hierarchical import minimize_full_chain_hierarchical
except ImportError:
    minimize_full_chain_hierarchical = None  # type: ignore[misc, assignment]

# Optional grading (trajectory + gold → metrics for AI/ML; see grading/README.md)
try:
    from .grading import grade_trajectory, load_trajectory_frames, grade_prediction
except ImportError:
    grade_trajectory = None  # type: ignore[misc, assignment]
    load_trajectory_frames = None  # type: ignore[misc, assignment]
    grade_prediction = None  # type: ignore[misc, assignment]

try:
    from importlib.metadata import version as _distribution_version

    __version__ = _distribution_version("protein-folder")
except Exception:
    __version__ = "0.1.0"

__all__ = [
    "__version__",
    "hqiv_alpha_helix",
    "backbone_bond_lengths",
    "backbone_geometry",
    "omega_peptide_deg",
    "ramachandran_alpha",
    "ramachandran_beta",
    "rational_ramachandran_alpha",
    "ramachandran_map",
    "alpha_helix_geometry",
    "rational_alpha_parameters",
    "alpha_helix_xyz",
    "beta_sheet_geometry",
    "beta_sheet_parallel_geometry",
    "beta_sheet_antiparallel_geometry",
    "side_chain_chi_preferences",
    "chi_angles_for_residue",
    "side_chain_placement_geometry",
    "AA_LIST",
    "e_tot",
    "minimize_e_tot",
    "small_peptide_energy",
    "build_horizon_poles",
    "build_bond_poles",
    "grad_from_poles",
    "full_atom_polymer_energy_budget",
    "hqiv_predict_structure",
    "hqiv_predict_structure_assembly",
    "minimize_e_tot_lbfgs",
    "rational_alpha_parameters_folding",
    "predict_ss",
    "predict_ss_with_angles",
    "theta_eff_residue",
    "preferred_basin_phi_psi",
    "minimize_full_chain",
    "full_chain_to_pdb",
    "full_chain_to_pdb_complex",
    "pack_sidechains",
    "ca_rmsd",
    "load_ca_from_pdb",
    "load_ca_and_sequence_from_pdb",
    "kabsch_superpose",
    "find_attachment_point",
    "Atom",
    "EMField",
    "CoTranslationalAssembler",
    "atoms_to_pdb",
    "fold_with_tunnel_temperature",
    "fold_lean_ribosome_tunnel",
    "LeanTunnelFoldResult",
    "FoldState",
    "make_minimize_stage",
    "make_force_carrier_stage",
    "make_lean_tunnel_stage",
    "make_osh_oracle_stage",
    "make_full_heavy_osh_stage",
    "run_pipeline",
    "run_tunnel_first_pipeline",
    "SparseRegister",
    "sparse_basis_card",
    "wrap_idx",
    "causal_expand_support",
    "dense_of_sparse",
    "apply_gate_sparse",
    "apply_ansatz_sparse",
    "current_parameters",
    "detect_flipped_kets",
    "detect_flipped_kets_amplitude",
    "estimate_natural_harmonic_scale_ca",
    "harmonic_temperature_schedule",
    "metropolis_accept_with_harmonic",
    "harmonic_tunneling_mixer_strength",
    "qpe_low_energy_subspace",
    "amplify_low_energy",
    "sparse_visible_energy",
    "auto_detect_cys_ligation_pairs",
    "QAOAHarmonicFoldInfo",
    "harmonic_tunneled_qaoa_folding",
    "prune_to_flipped",
    "OSHOracleFoldInfo",
    "minimize_ca_with_osh_oracle",
    "minimize_backbone_with_osh_oracle",
    "minimize_full_heavy_with_osh_oracle",
    "FullHeavyAtomChain",
    "build_full_heavy_chain",
    "chain_to_pdb_line_string",
    "full_heavy_chain_energy_budget",
    "VERY_SHORT_FOLD_TARGETS",
    "VERY_SHORT_FOLD_TARGETS_LIST",
    "VeryShortFoldTarget",
    "target_by_key",
    "LigandAgent",
    "ligand_summary",
    "parse_ligands",
    "parse_ligands_from_file",
    "z_shell_for_element",
    "K_hbond",
    "R_hbond",
    "h_bond_proxy",
    "phi_of_shell",
    "available_modes_nat",
    "total_h_bond_proxy_energy_ca",
    "minimize_full_chain_hierarchical",
    "grade_trajectory",
    "load_trajectory_frames",
    "grade_prediction",
]
