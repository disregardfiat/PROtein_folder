# Installation & quick start

## PyPI (recommended)

Python **3.10+**.

```bash
pip install "protein-folder[full]"
```

Extras:

| Extra | Purpose |
|-------|---------|
| `(none)` | Core: `numpy`, `scipy`, `pyhqiv` |
| `full` | Adds `matplotlib` for plotting / examples |
| `tpu` | JAX wheels (`jax`, `jaxlib`) for hierarchical / device runs |
| `grading` | `pandas` for trajectory grading exports |
| `demo` | `streamlit`, `flask`, `gunicorn` for local UI and CASP-style server |

```bash
pip install "protein-folder[full,tpu,grading,demo]"
```

Editable install from a git clone:

```bash
git clone https://github.com/disregardfiat/protein_folder.git
cd protein_folder
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Docker

**CPU**

```bash
docker build -t protein-folder:cpu .
docker run --rm protein-folder:cpu
```

**JAX stack (see `Dockerfile.tpu` notes for Cloud TPU VMs)**

```bash
docker build -f Dockerfile.tpu -t protein-folder:tpu .
```

On Google Cloud TPU VMs, follow [JAX TPU installation](https://jax.readthedocs.io/en/latest/installation.html#google-cloud-tpu) and then `pip install "protein-folder[tpu,full]"` on the VM image if you prefer not to use the Dockerfile.

## Minimal usage

```python
from horizon_physics.proteins import minimize_full_chain, full_chain_to_pdb

result = minimize_full_chain("MKFLNDR", include_sidechains=True)
print(full_chain_to_pdb(result))
```

Tunnel mode:

```python
result = minimize_full_chain(
    "MKFLNDR",
    simulate_ribosome_tunnel=True,
    tunnel_length=25.0,
    cone_half_angle_deg=12.0,
)
```

## Government-use note

Install metadata and this site document the **MIT + government-use restriction**. See [Government use & license](government-use.md).
