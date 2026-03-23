# Live demo

## Streamlit (local)

Install demo extras:

```bash
pip install "protein-folder[demo]"
```

Launch:

```bash
protein-folder-streamlit
```

This runs [`demo_streamlit_ui.py`](https://github.com/disregardfiat/protein_folder/blob/main/src/horizon_physics/proteins/demo_streamlit_ui.py): paste a short sequence, optionally enable the **Lean ribosome tunnel** path, and download a PDB text snapshot.

!!! note "Performance"
    Long sequences are CPU-intensive. The demo is meant for **local exploration**, not batch CASP production.

## Flask / gunicorn (CASP server)

The production-oriented HTTP server is [`casp_server.py`](https://github.com/disregardfiat/protein_folder/blob/main/casp_server.py) at the repository root (POST FASTA → PDB, optional email). Typical deployment:

```bash
gunicorn -w 1 -b 127.0.0.1:8050 casp_server:app
```

Requires the repo layout (or `PYTHONPATH` including the repo root **and** `src/`) so both `casp_server` and `horizon_physics` resolve. Prefer `pip install -e .` on the server.

See `deploy/setup_casp_server.sh` for an example systemd unit.
