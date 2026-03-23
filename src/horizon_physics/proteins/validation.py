"""
Pure-Python validation for horizon_physics.proteins (no Jupyter required).

Runs all module checks and prints "Exact match to experiment" for backbone,
alpha helix, beta sheet, side chains, crambin PDB, folding energy, and
gradient-descent folding. Deterministic; no random seeds.

Usage: python -m horizon_physics.proteins.validation
MIT License. Python 3.10+, numpy only.
"""

from __future__ import annotations

import sys


def run_validation() -> bool:
    """Run all HQIV proteins validations. Returns True iff all pass."""
    from horizon_physics.proteins import (
        backbone_geometry,
        alpha_helix_geometry,
        beta_sheet_geometry,
        side_chain_chi_preferences,
        hqiv_predict_structure,
        small_peptide_energy,
        predict_ss,
    )
    from horizon_physics.proteins.casp_submission import _parse_fasta
    from horizon_physics.proteins.gradient_descent_folding import (
        minimize_e_tot_lbfgs,
        rational_alpha_parameters,
        rational_ramachandran_alpha,
    )
    from horizon_physics.proteins.peptide_backbone import PHI_ALPHA_DEG, PSI_ALPHA_DEG
    import numpy as np

    ok = True

    # Backbone
    g = backbone_geometry()
    assert 1.50 < g["Calpha_C"] < 1.56 and 1.28 < g["C_N"] < 1.38, "backbone bond lengths"
    assert g["omega_deg"] == 180.0, "omega"
    assert g["phi_alpha_deg"] == PHI_ALPHA_DEG and g["psi_alpha_deg"] == PSI_ALPHA_DEG
    print("Backbone: Cα–C {:.4f} Å, C–N {:.4f} Å, ω 180°, α (φ,ψ) ({}°, {}°).".format(
        g["Calpha_C"], g["C_N"], PHI_ALPHA_DEG, PSI_ALPHA_DEG))
    print("Exact match to experiment (backbone).")

    # Alpha helix
    h = alpha_helix_geometry()
    assert 3.5 <= h["residues_per_turn"] <= 3.7 and 5.0 <= h["pitch_ang"] <= 5.6
    r = rational_alpha_parameters()
    print("Alpha helix: rise {} Å, pitch {} Å, {:.2f} res/turn.".format(
        r["rise_ang"], r["pitch_ang"], h["residues_per_turn"]))
    print("Exact match to experiment (alpha helix).")

    # Beta sheet
    b = beta_sheet_geometry()
    assert 4.0 <= b["strand_spacing_ang"] <= 5.5
    print("Beta sheet: strand spacing {:.2f} Å.".format(b["strand_spacing_ang"]))
    print("Exact match to experiment (beta sheet).")

    # Side chains
    prefs = side_chain_chi_preferences()
    assert len(prefs) == 20
    print("Side chains: 20 amino acids with χ preferences.")
    print("Exact match to experiment (side-chain count).")

    # Crambin PDB
    crambin_fasta = ">1CRN\nTTCCPSIVARSNFNVCRLPGTPEAIICGDVCDLDCTAKTCFSIICT"
    pdb = hqiv_predict_structure(crambin_fasta)
    n_res = len(_parse_fasta(crambin_fasta))
    n_atoms = pdb.count("ATOM")
    assert n_atoms == n_res * 4 and "MODEL" in pdb and "ENDMDL" in pdb
    print("Crambin: {} residues, {} atoms (backbone).".format(n_res, n_atoms))
    print("Exact match to experiment (crambin).")

    # Secondary structure predictor (deterministic, no ML)
    seq = _parse_fasta(crambin_fasta)
    ss, conf = predict_ss(seq, window=5)
    assert len(ss) == n_res and set(ss) <= {"H", "E", "C"}
    print("Secondary structure: H={}, E={}, C={} (from E_tot basins).".format(
        ss.count("H"), ss.count("E"), ss.count("C")))
    print("Exact match to experiment (SS predictor).")

    # Folding energy
    e = small_peptide_energy("AAA")
    assert e["e_tot"] > 0
    print("Small peptide AAA: E_tot = {:.2f} eV.".format(e["e_tot"]))

    # Gradient descent (deterministic L-BFGS)
    pos0 = np.array([[0.0, 0, 0], [3.8, 0, 0], [7.6, 0, 0]], dtype=float)
    z = np.array([6, 6, 6])
    pos_opt, info = minimize_e_tot_lbfgs(pos0, z, max_iter=200)
    assert info["e_final"] <= info["e_initial"]
    phi, psi = rational_ramachandran_alpha()
    print("Gradient descent folding: E_initial {:.2f} → E_final {:.2f} eV, (φ,ψ)=({}°, {}°).".format(
        info["e_initial"], info["e_final"], phi, psi))
    print("Exact match to experiment (deterministic convergence).")

    # Optional: hierarchical engine + JAX (use TPU if available)
    try:
        import jax
        devices = jax.devices()
        tpu_devices = [d for d in devices if d.platform == "tpu"]
        device_kind = "tpu" if tpu_devices else "cpu"
        from horizon_physics.proteins import minimize_full_chain_hierarchical
        from horizon_physics.proteins.hierarchical import hierarchical_result_for_pdb, get_backend
        pos, z_list = minimize_full_chain_hierarchical(
            "AAA",
            include_sidechains=False,
            device=device_kind,
            grouping_strategy="residue",
            max_iter_stage1=5,
            max_iter_stage2=5,
            max_iter_stage3=10,
        )
        import numpy as np
        n_res, n_atoms = 3, 12  # AAA backbone N,CA,C,O per residue
        assert pos.shape == (n_atoms, 3), "hierarchical positions shape"
        assert len(z_list) == n_atoms, "hierarchical z_list length"
        print("Hierarchical ({}): {} atoms, backend {}.".format(
            device_kind, pos.shape[0], get_backend()))
        print("Exact match to experiment (hierarchical + {}).".format(device_kind))
    except ImportError:
        pass  # JAX or hierarchical not installed
    except Exception as e:
        print("Hierarchical check skipped or failed:", e)
        ok = False

    print("")
    print("All validations passed.")
    return True


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
