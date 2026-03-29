"""
Curated experimental PDB IDs for CAMEO3D-style checks (ligand-containing complexes).

CAMEO-3D evaluates servers against pre-released PDB structures; targets often include
small-molecule or peptide ligands as InChI/SMILES. These wwPDB entries are representative
**protein + HETATM** cases used by ``scripts/validate_cameo_style_ligand_pdbs.py``.

Not every ID may parse ligands if the deposition uses only ATOM for a cofactor; the
validation script skips failures and continues.
"""

# Monomer or assembly entries with non-water hetero (inhibitors, cofactors, fragments).
CAMEO_STYLE_LIGAND_PDB_IDS: tuple[str, ...] = (
    "1HVR",  # HIV-1 protease / inhibitor (classic)
    "3FAP",  # fatty-acid binding protein + ligand
    "1AZM",  # lysozyme + benzyl inhibitor
    "3KFA",  # kinase + ligand
    "4CR9",  # protein + HETATM ligand (replaces entries with ATOM-only hetero)
    "5NEY",  # representative bound ligand
    "4ZZC",  # peptide/small-molecule interface
    "3FUD",  # alternate pharmacophore
)
