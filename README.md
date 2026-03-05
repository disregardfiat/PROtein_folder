# PROtein — HQIV Protein Folding

Physically Rotating by Order: First-principles protein structure prediction from the **Horizon-centric Quantum Information and Vacuum (HQIV)** framework. No empirical force fields, no PDB statistics.
HQIV => https://zenodo.org/records/18794890

## Quick start

```bash
pip install numpy  # Python 3.10+
python -c "
from horizon_physics.proteins import minimize_full_chain, full_chain_to_pdb
result = minimize_full_chain('MKFLNDR', include_sidechains=True)
print(full_chain_to_pdb(result))
"
```

### Optional: JAX and TPU

For the hierarchical minimizer and TPU/GPU acceleration:

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -r requirements-tpu.txt   # jax, jaxlib
```

On a **Google Cloud TPU VM**, install TPU support so JAX sees the device:

```bash
pip install "jax[tpu]"
# or if libtpu download fails: pip install libtpu -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Co-translational ribosome tunnel (optional)

Simulate the ribosome exit tunnel: null search cone, plane at the lip, **binary-tree** segment schedule (to damp vibrations), fast-pass spaghetti (rigid group + bell-end), and a short HKE min pass per segment. With `post_extrusion_refine=True` (default), the full HKE collapse/refine is run repeatedly after extrusion until no Cα moves more than 0.5 Å per 100 residues (adaptive).

```bash
python -c "
from horizon_physics.proteins import minimize_full_chain, full_chain_to_pdb
result = minimize_full_chain(
    'MKFLNDR',
    simulate_ribosome_tunnel=True,
    tunnel_length=25.0,
    cone_half_angle_deg=12.0,
)
print(full_chain_to_pdb(result))
"
```

### Live trajectory visualizer

When running the minimizer (or tunnel pipeline) with a trajectory log, you can watch Cα positions update in real time in a separate process:

```bash
# Terminal 1: run tunnel with trajectory log (writes JSONL)
python -m horizon_physics.proteins.examples.run_tunnel_and_grade --targets T1037 --trajectory-log /tmp/t1037_traj.jsonl

# Terminal 2: live 3D view (tails the log; requires matplotlib)
python -m horizon_physics.proteins.examples.live_trajectory_viz /tmp/t1037_traj.jsonl
```

The visualizer runs in a separate process and tails the JSONL file, so it does not affect minimizer performance.

## Package

- **`horizon_physics/proteins/`** — Full HQIV protein folding pipeline
- See [horizon_physics/proteins/README.md](horizon_physics/proteins/README.md) for details

## License

**License note (March 2026)**  
This project is released under the MIT License **with an explicit Government Use Restriction**.  
Governments worldwide may **not** use, fork, or run this software without first purchasing a paid seat from me (disregardfiat). Seat availability and pricing are at my sole discretion.  
Individuals, companies, universities, and non-profits continue to enjoy full MIT rights.