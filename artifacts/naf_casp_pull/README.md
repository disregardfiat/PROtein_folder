# CASP outputs pulled from production (naf)

Source: `/home/ubuntu/protein_folder/casp_results/outputs/` on **naf** (jobs after queue clean, email `sjettingerjr@gmail.com`).

| File | Job | Request type |
|------|-----|----------------|
| `single_1774827685_3011.pdb` | `1774827685_3011` | Single chain 15-mer |
| `multichain_1774827686_8438.pdb` | `1774827686_8438` | Two chains |
| `multichain_ligand_1774827688_4801.pdb` | `1774827688_4801` | Two chains + LIG HETATM |

These sequences are **synthetic API smoke tests**, not natural proteins. There is **no unique experimental PDB** whose sequence matches the prediction end-to-end, so **Cα-RMSD vs “the” experiment is undefined** unless you pick a reference for a specific benchmark (same sequence as a released structure).

Use `scripts/compare_naf_casp_to_experiment.py` for Rg / optional RMSD when you supply a reference PDB.
