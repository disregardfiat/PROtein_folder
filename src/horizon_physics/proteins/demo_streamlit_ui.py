"""
Streamlit UI for local HQIV folding demos (short sequences; physics-first minimizer).

Run via: protein-folder-streamlit
  or:    python -m streamlit run demo_streamlit_ui.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="PROtein / HQIV — Live demo", layout="wide")

st.title("PROtein — HQIV folding demo")
st.markdown(
    "First-principles Cα + optional sidechains (**no ML / no empirical force fields**). "
    "Theory and Lean proofs: [hqiv-lean](https://github.com/disregardfiat/hqiv-lean). "
    "Docs: [protein_folder site](https://disregardfiat.github.io/protein_folder/) (when deployed)."
)

with st.sidebar:
    st.header("Options")
    tunnel = st.checkbox("Co-translational tunnel (Lean pipeline)", value=False)
    sidechains = st.checkbox("Include sidechains", value=True)
    seq = st.text_area("Sequence (1-letter)", value="MKFLNDR", height=120)
    go = st.button("Fold", type="primary")

st.caption(
    "Government-use restriction applies — see GOVERNMENT_USE.md. "
    "Long sequences can take substantial CPU time."
)

if go and seq.strip():
    from horizon_physics.proteins import full_chain_to_pdb, minimize_full_chain

    clean = "".join(c for c in seq.upper().strip() if c.isalpha())
    if not clean:
        st.error("No amino acid letters in input.")
    else:
        with st.spinner("Running minimizer…"):
            try:
                result = minimize_full_chain(
                    clean,
                    include_sidechains=sidechains,
                    simulate_ribosome_tunnel=tunnel,
                )
                pdb = full_chain_to_pdb(result)
            except Exception as e:
                st.exception(e)
            else:
                st.success(f"Done — {len(clean)} residues")
                st.download_button("Download PDB", pdb, file_name="hqiv_fold.pdb")
                st.code(pdb[:8000] + ("\n…" if len(pdb) > 8000 else ""), language="text")
