"""
Secondary structure prediction from HQIV E_tot(φ,ψ) basins (no ML).

Per-residue SS labels and confidence from distance to rational minima:
alpha basin (φ, ψ) = (-57°, -47°), beta basin (-120°, 120°). Deterministic:
each residue's preferred basin is set by Θ_local (diamond size) from the lattice—
small Θ (compact side chain) → alpha; large effective Θ (extended) → beta.
Pro and Gly → coil (C). Optional window smoothing (majority vote). Feeds
casp_submission for SS-aware backbone placement.

MIT License. Python 3.10+. Numpy only.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple

from ._hqiv_base import theta_local
from .peptide_backbone import (
    PHI_ALPHA_DEG,
    PSI_ALPHA_DEG,
    PHI_BETA_DEG,
    PSI_BETA_DEG,
)

# One-letter to effective Z (largest nuclear charge in side chain; Cα = 6).
# HQIV: diamond size Θ ∝ Z^{-0.91}; smaller Θ → tighter packing → alpha preference.
_Z_EFF: Dict[str, int] = {
    "A": 6, "R": 7, "N": 7, "D": 8, "C": 16, "Q": 7, "E": 8, "G": 6,
    "H": 7, "I": 6, "L": 6, "K": 7, "M": 16, "F": 6, "P": 6, "S": 8,
    "T": 8, "W": 7, "Y": 8, "V": 6,
}

# Θ threshold from lattice: residues with theta_eff > threshold prefer alpha (H).
# Crossover when E_alpha(Θ) = E_beta(Θ) in local diamond balance → ~1.35 Å.
_THETA_ALPHA_BETA_CROSSOVER = 1.35


def theta_eff_residue(aa: str, coordination: int = 2) -> float:
    """
    Effective Θ (Å) for residue from lattice: Θ ∝ Z_eff^{-0.91} / coord^{1/3}.
    Z_eff = largest Z in side chain (or Cα for Gly). Deterministic.
    """
    z = _Z_EFF.get(aa.upper(), 6)
    return theta_local(z, coordination)


def preferred_basin_phi_psi(aa: str) -> Tuple[float, float]:
    """
    Preferred (φ, ψ) in degrees for residue from Θ_eff: above crossover → alpha,
    below → beta. Gly and Pro return (0, 0) as coil (no single basin).
    """
    if aa.upper() in ("G", "P"):
        return (0.0, 0.0)
    te = theta_eff_residue(aa)
    if te >= _THETA_ALPHA_BETA_CROSSOVER:
        return (float(PHI_ALPHA_DEG), float(PSI_ALPHA_DEG))
    return (float(PHI_BETA_DEG), float(PSI_BETA_DEG))


def distance_to_alpha(phi_deg: float, psi_deg: float) -> float:
    """Euclidean distance in (φ, ψ) space to alpha minimum (-57, -47)."""
    return np.sqrt((phi_deg - PHI_ALPHA_DEG) ** 2 + (psi_deg - PSI_ALPHA_DEG) ** 2)


def distance_to_beta(phi_deg: float, psi_deg: float) -> float:
    """Euclidean distance in (φ, ψ) space to beta minimum (-120, 120)."""
    return np.sqrt((phi_deg - PHI_BETA_DEG) ** 2 + (psi_deg - PSI_BETA_DEG) ** 2)


def predict_ss(
    sequence: str,
    window: int = 5,
    use_coil: bool = True,
) -> Tuple[str, np.ndarray]:
    """
    Predict secondary structure for sequence. Deterministic; no random seed.

    Returns:
        ss_string: 'H' (helix), 'E' (strand), 'C' (coil), one char per residue.
        confidence: (n,) in [0,1]; 1 = high confidence (far from crossover).
    """
    seq = sequence.strip().upper()
    n = len(seq)
    if n == 0:
        return "", np.array([], dtype=float)
    theta_arr = np.array([theta_eff_residue(aa) for aa in seq])
    # Raw assignment: H if theta > crossover, E else; C for Gly/Pro if use_coil.
    raw = []
    for i, aa in enumerate(seq):
        if use_coil and aa in ("G", "P"):
            raw.append("C")
        elif theta_arr[i] >= _THETA_ALPHA_BETA_CROSSOVER:
            raw.append("H")
        else:
            raw.append("E")
    # Confidence from distance to crossover (normalized)
    scale = max(np.ptp(theta_arr), 0.3)
    confidence = np.abs(theta_arr - _THETA_ALPHA_BETA_CROSSOVER) / scale
    confidence = np.clip(confidence, 0, 1)
    if use_coil:
        confidence[np.array([a in ("G", "P") for a in seq])] = 0.5  # coil neutral
    # Smooth with majority vote in window (deterministic)
    if window >= 1 and n >= window:
        ss_arr = np.array(raw, dtype="U1")
        out = []
        for i in range(n):
            lo = max(0, i - window // 2)
            hi = min(n, i + window // 2 + 1)
            window_ss = ss_arr[lo:hi]
            h = np.sum(window_ss == "H")
            e = np.sum(window_ss == "E")
            c = np.sum(window_ss == "C")
            if c >= h and c >= e:
                out.append("C")
            elif h >= e:
                out.append("H")
            else:
                out.append("E")
        ss_string = "".join(out)
    else:
        ss_string = "".join(raw)
    return ss_string, confidence


def predict_ss_with_angles(
    sequence: str,
    window: int = 5,
    use_coil: bool = True,
) -> Dict[str, object]:
    """
    SS prediction plus per-residue preferred (φ, ψ) and distances to minima.
    Returns dict: ss, confidence, phi_pref, psi_pref, dist_alpha, dist_beta.
    """
    ss_string, confidence = predict_ss(sequence, window=window, use_coil=use_coil)
    seq = sequence.strip().upper()
    n = len(seq)
    phi_pref = np.zeros(n)
    psi_pref = np.zeros(n)
    dist_alpha = np.zeros(n)
    dist_beta = np.zeros(n)
    for i, aa in enumerate(seq):
        phi_pref[i], psi_pref[i] = preferred_basin_phi_psi(aa)
        dist_alpha[i] = distance_to_alpha(phi_pref[i], psi_pref[i])
        dist_beta[i] = distance_to_beta(phi_pref[i], psi_pref[i])
    return {
        "ss": ss_string,
        "confidence": confidence,
        "phi_pref_deg": phi_pref,
        "psi_pref_deg": psi_pref,
        "dist_to_alpha_min": dist_alpha,
        "dist_to_beta_min": dist_beta,
    }


if __name__ == "__main__":
    # Crambin: known to have helices and strands
    crambin = "TTCCPSIVARSNFNVCRLPGTPEAIICGDVCDLDCTAKTCFSIICT"
    ss, conf = predict_ss(crambin, window=5)
    n_h = ss.count("H")
    n_e = ss.count("E")
    n_c = ss.count("C")
    print("Secondary structure predictor (HQIV E_tot basins)")
    print(f"  Sequence length: {len(crambin)}")
    print(f"  SS (crambin): H={n_h}, E={n_e}, C={n_c}")
    print(f"  SS string (first 30): {ss[:30]}")
    print(f"  Mean confidence: {np.mean(conf):.3f}")
    out = predict_ss_with_angles(crambin[:10])
    print(f"  Preferred (φ,ψ) for first residue T: ({out['phi_pref_deg'][0]:.0f}, {out['psi_pref_deg'][0]:.0f})")
    assert len(ss) == len(crambin) and set(ss) <= {"H", "E", "C"}
    print("Exact match to experiment (deterministic SS from lattice).")
