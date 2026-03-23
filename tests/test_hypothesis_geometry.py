"""Property-style checks for geometry / energy invariants (Hypothesis)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from horizon_physics.proteins import backbone_geometry
from horizon_physics.proteins import folding_energy as fe


@settings(deadline=None, max_examples=25)
@given(
    n=st.integers(min_value=2, max_value=24),
    data=st.data(),
)
def test_e_tot_informational_finite(n: int, data: st.DataObject) -> None:
    pos = data.draw(
        arrays(
            shape=(n, 3),
            dtype=np.float64,
            elements=st.floats(width=64, min_value=-8.0, max_value=8.0, allow_nan=False),
        )
    )
    z = np.full(n, 6, dtype=int)
    e = fe.e_tot_informational(pos, z, fast_local_theta=True)
    assert np.isfinite(e)


@settings(deadline=None, max_examples=15)
@given(
    n=st.integers(min_value=2, max_value=16),
    data=st.data(),
)
def test_horizon_pairs_monotone_in_cutoff(n: int, data: st.DataObject) -> None:
    pos = data.draw(
        arrays(
            shape=(n, 3),
            dtype=np.float64,
            elements=st.floats(width=64, min_value=-6.0, max_value=6.0, allow_nan=False),
        )
    )
    z = np.full(n, 6, dtype=int)
    poles_w = fe.build_horizon_poles(pos, z, neighbor_cutoff=14.0)
    poles_t = fe.build_horizon_poles(pos, z, neighbor_cutoff=6.0)
    assert len(poles_t) <= len(poles_w)


def test_backbone_geometry_experimental_window() -> None:
    g = backbone_geometry()
    assert 1.50 < g["Calpha_C"] < 1.56
    assert 1.28 < g["C_N"] < 1.38
    assert g["omega_deg"] == pytest.approx(180.0)
