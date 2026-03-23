"""
Unit tests for co-translational tunnel: cone constraint, plane null-rotation,
group rotation fast-pass, connection-triggered min pass.

Run: python -m horizon_physics.proteins.test_co_translational_tunnel
Or: pytest horizon_physics/proteins/test_co_translational_tunnel.py -v
"""

from __future__ import annotations

import numpy as np

from .co_translational_tunnel import (
    DEFAULT_TUNNEL_AXIS,
    inside_tunnel_mask,
    past_lip_mask,
    cone_constraint_mask_gradient,
    plane_lip_null_backward_gradient,
    apply_cone_and_plane_masking,
    rigid_body_gradient_for_group,
    bell_end_indices,
    align_chain_to_tunnel,
    run_masked_lbfgs_pass,
    co_translational_minimize,
)


def test_cone_constraint():
    """(a) Cone constraint: gradient component that would push Cα outside cone is zeroed."""
    ptc = np.zeros(3)
    axis = np.array([0.0, 0.0, 1.0])
    tunnel_length = 25.0
    cone_half_deg = 12.0
    # Residue at s=10 along axis, at cone boundary: r_perp = 10*tan(12°)
    s = 10.0
    r_max = s * np.tan(np.deg2rad(cone_half_deg))
    pos = np.array([[0.0, r_max, s]])  # on boundary
    grad = np.array([[0.0, 1.0, 0.0]])  # outward radial
    cone_constraint_mask_gradient(grad, pos, ptc, axis, cone_half_deg, tunnel_length)
    # Outward radial component should be zeroed
    assert np.abs(grad[0, 1]) < 1e-9, "outward radial gradient should be zeroed at cone boundary"
    # Residue inside cone: gradient unchanged
    pos_inside = np.array([[0.0, 0.5 * r_max, s]])
    grad_inside = np.array([[1.0, 2.0, 3.0]])
    cone_constraint_mask_gradient(grad_inside, pos_inside, ptc, axis, cone_half_deg, tunnel_length)
    np.testing.assert_allclose(grad_inside, [[1.0, 2.0, 3.0]], err_msg="gradient inside cone unchanged")


def test_plane_null_rotation():
    """(b) Plane at lip: gradient component that would move residue back across lip is nullified."""
    ptc = np.zeros(3)
    axis = np.array([0.0, 0.0, 1.0])
    lip_distance = 25.0
    # Residue past lip (s=30), gradient pulling toward PTC (-axis)
    pos = np.array([[0.0, 0.0, 30.0]])
    grad = np.array([[0.0, 0.0, -1.0]])  # backward
    plane_lip_null_backward_gradient(grad, pos, ptc, axis, lip_distance)
    np.testing.assert_allclose(grad[0], [0.0, 0.0, 0.0], atol=1e-9, err_msg="backward component nullified")
    # Forward gradient unchanged
    pos2 = np.array([[0.0, 0.0, 30.0]])
    grad2 = np.array([[0.0, 0.0, 1.0]])
    plane_lip_null_backward_gradient(grad2, pos2, ptc, axis, lip_distance)
    np.testing.assert_allclose(grad2[0], [0.0, 0.0, 1.0], err_msg="forward gradient unchanged")


def test_group_rotation_fast_pass():
    """(c) Rigid-body gradient: group of points gets single 6-DOF gradient (translation + rotation)."""
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    grad = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    rigid_body_gradient_for_group(positions, grad, [0, 1, 2])
    # grad_rigid[i] = F + omega x r_i with F = sum(original grad) = (3,0,0); sum over group = n*F = (9,0,0)
    np.testing.assert_allclose(grad.sum(axis=0), [9.0, 0.0, 0.0], atol=1e-6, err_msg="rigid group sum = n*F")
    # All points get same translation F; rotation adds different component per point
    np.testing.assert_allclose(grad[0], grad[1], atol=1e-5)
    np.testing.assert_allclose(grad[1], grad[2], atol=1e-5)


def test_connection_triggered_min_pass():
    """(d) Connection-triggered min pass: run_masked_lbfgs_pass decreases energy with cone/plane active."""
    from .folding_energy import e_tot_ca_with_bonds, grad_full

    n = 5
    # Short chain along +Z inside tunnel
    pos0 = np.zeros((n, 3))
    pos0[:, 2] = np.arange(n) * 3.8
    z_list = np.full(n, 6)
    ptc = np.zeros(3)
    axis = np.array([0.0, 0.0, 1.0])
    tunnel_length = 25.0
    cone_half_deg = 12.0
    lip_plane_distance = 0.0

    def grad_func(p, z):
        return grad_full(p, z, include_bonds=True, include_horizon=True, include_clash=True)

    from .gradient_descent_folding import _project_bonds

    pos_final, n_iter = run_masked_lbfgs_pass(
        pos0, z_list, ptc, axis, tunnel_length, cone_half_deg, lip_plane_distance,
        max_iter=20, grad_func=grad_func, project_bonds=_project_bonds,
    )
    e0 = e_tot_ca_with_bonds(pos0, z_list)
    e_final = e_tot_ca_with_bonds(pos_final, z_list)
    assert e_final <= e0 + 0.01, "masked L-BFGS pass should not increase energy"
    assert n_iter >= 1, "at least one step"


def test_inside_tunnel_and_past_lip_masks():
    """Helper: inside_tunnel_mask and past_lip_mask classify correctly."""
    ptc = np.zeros(3)
    axis = np.array([0.0, 0.0, 1.0])
    positions = np.array([[0, 0, 0], [0, 0, 10], [0, 0, 25], [0, 0, 30]])
    inside = inside_tunnel_mask(positions, ptc, axis, tunnel_length=25.0)
    assert inside[0] and inside[1] and inside[2] and not inside[3]
    past_lip = past_lip_mask(positions, ptc, axis, lip_distance=25.0)
    assert not past_lip[0] and not past_lip[1] and not past_lip[2] and past_lip[3]


def test_bell_end_indices():
    """Bell-end indices are last n_bell residues."""
    assert bell_end_indices(5, 2) == [3, 4]
    assert bell_end_indices(3, 2) == [1, 2]
    assert bell_end_indices(1, 2) == [0]


def test_align_chain_to_tunnel():
    """Chain aligns so residue 0 at PTC and first bond along axis."""
    ca = np.array([[1.0, 2.0, 3.0], [4.0, 2.0, 3.0], [7.0, 2.0, 3.0]])  # along X
    ptc = np.zeros(3)
    axis = np.array([0.0, 0.0, 1.0])
    out = align_chain_to_tunnel(ca, ptc, axis)
    np.testing.assert_allclose(out[0], ptc, atol=1e-9, err_msg="first residue at PTC")
    bond = out[1] - out[0]
    bond = bond / (np.linalg.norm(bond) + 1e-12)
    np.testing.assert_allclose(bond, axis, atol=1e-5, err_msg="first bond along axis")


def test_co_translational_minimize_short_chain():
    """Full co-translational minimize on a short chain (edge case: n < tunnel_length)."""
    from .folding_energy import grad_full
    from .gradient_descent_folding import _project_bonds

    seq = "MKF"
    n = 3
    ca_init = np.zeros((n, 3))
    ca_init[:, 0] = np.arange(n) * 3.8
    z_list = np.full(n, 6)
    ptc = np.zeros(3)
    axis = DEFAULT_TUNNEL_AXIS.copy()
    ca_min, info = co_translational_minimize(
        ca_init, z_list, ptc, axis,
        tunnel_length=25.0, cone_half_angle_deg=12.0, lip_plane_distance=0.0,
        grad_func=lambda p, z: grad_full(p, z, include_bonds=True, include_horizon=True, include_clash=True),
        project_bonds=_project_bonds,
        n_bell=2,
        fast_pass_steps_per_connection=2,
        min_pass_iter_per_connection=5,
    )
    assert ca_min.shape == (n, 3)
    assert info["success"]
    assert "e_final" in info


def test_co_translational_minimize_tunnel_thermal_counts_steps():
    """Per-segment thermal relax runs after masked L-BFGS; stats aggregate total steps."""
    from .folding_energy import grad_full
    from .gradient_descent_folding import _project_bonds

    seq = "MKF"
    n = 3
    ca_init = np.zeros((n, 3))
    ca_init[:, 0] = np.arange(n) * 3.8
    z_list = np.full(n, 6)
    ptc = np.zeros(3)
    axis = DEFAULT_TUNNEL_AXIS.copy()
    ca_min, info = co_translational_minimize(
        ca_init,
        z_list,
        ptc,
        axis,
        tunnel_length=25.0,
        cone_half_angle_deg=12.0,
        lip_plane_distance=0.0,
        grad_func=lambda p, z: grad_full(
            p, z, include_bonds=True, include_horizon=True, include_clash=True
        ),
        project_bonds=_project_bonds,
        n_bell=2,
        fast_pass_steps_per_connection=1,
        min_pass_iter_per_connection=2,
        tunnel_thermal_gradient_steps=4,
        tunnel_thermal_temperature_k=310.0,
        tunnel_thermal_seed=42,
        tunnel_thermal_quick_cap=False,
    )
    assert ca_min.shape == (n, 3)
    assert int(info.get("tunnel_thermal_total_steps", 0)) > 0
    assert int(info.get("tunnel_thermal_segments", 0)) >= 1


def test_minimize_full_chain_backward_compat():
    """Default simulate_ribosome_tunnel=False: behavior unchanged (no tunnel)."""
    from .full_protein_minimizer import minimize_full_chain

    result = minimize_full_chain("MKFL", max_iter=20, quick=True)
    assert result["n_res"] == 4
    assert result["ca_min"].shape == (4, 3)
    assert "info" in result


def test_minimize_full_chain_tunnel_mode():
    """With simulate_ribosome_tunnel=True: runs co-translational path and returns valid result."""
    from .full_protein_minimizer import minimize_full_chain

    result = minimize_full_chain(
        "MKFL",
        max_iter=50,
        quick=True,
        simulate_ribosome_tunnel=True,
        tunnel_length=25.0,
        cone_half_angle_deg=12.0,
        lip_plane_distance=0.0,
    )
    assert result["n_res"] == 4
    assert result["ca_min"].shape == (4, 3)
    msg = result["info"].get("message", "").lower()
    assert (
        "co-translational" in msg
        or "tunnel" in msg
        or "post-tunnel" in msg
        or result["info"].get("post_extrusion_refine_mode") == "em_treetorque"
    )


def test_post_extrusion_refine():
    """With post_extrusion_refine=True (default): message includes post-extrusion; with False, no refine stage."""
    from .full_protein_minimizer import minimize_full_chain

    result_refine = minimize_full_chain(
        "MKFL",
        max_iter=50,
        quick=True,
        simulate_ribosome_tunnel=True,
        post_extrusion_refine=True,
        tunnel_length=25.0,
        cone_half_angle_deg=12.0,
    )
    msg_r = result_refine["info"].get("message", "").lower()
    assert "post-extrusion" in msg_r or "post-tunnel" in msg_r or "tree-torque" in msg_r
    assert result_refine["ca_min"].shape == (4, 3)

    result_no_refine = minimize_full_chain(
        "MKFL",
        max_iter=50,
        quick=True,
        simulate_ribosome_tunnel=True,
        post_extrusion_refine=False,
        tunnel_length=25.0,
        cone_half_angle_deg=12.0,
    )
    msg_nr = result_no_refine["info"].get("message", "").lower()
    assert "co-translational" in msg_nr or "tunnel" in msg_nr
    assert result_no_refine["ca_min"].shape == (4, 3)


if __name__ == "__main__":
    test_cone_constraint()
    test_plane_null_rotation()
    test_group_rotation_fast_pass()
    test_connection_triggered_min_pass()
    test_inside_tunnel_and_past_lip_masks()
    test_bell_end_indices()
    test_align_chain_to_tunnel()
    test_co_translational_minimize_short_chain()
    test_co_translational_minimize_tunnel_thermal_counts_steps()
    test_minimize_full_chain_backward_compat()
    test_minimize_full_chain_tunnel_mode()
    test_post_extrusion_refine()
    print("All tests passed.")
