"""
Unit tests for solutal buoyancy and mixed convection in vertical cathode channels.
"""

from models.solutal_convection import (
    SolutalChannelParams,
    solve_solutal_mixed_convection,
)


def test_solutal_density_depletion_and_grashof():
    """Verify that Fe2+ depletion generates significant buoyancy and high Grashof number."""
    params = SolutalChannelParams(cathode_height_m=1.0)
    res = solve_solutal_mixed_convection(
        forced_velocity_m_s=0.10,
        flow_direction="upward",
        bulk_fe2_mol_L=1.5,
        surface_fe2_mol_L=0.3,
        params=params,
    )

    # 1.2 M Fe2+ depletion gives ~25-80 kg/m3 density drop
    assert 20.0 <= res.density_depletion_kg_m3 <= 90.0
    assert res.grashof_number_Gr_H > 1e10  # Height-scale Grashof (~4e11)
    assert res.grashof_number_Gr_dh > 1e4  # Gap-scale Grashof (~8.9e4)
    assert res.buoyancy_velocity_m_s > 0.05
    assert not res.is_flow_reversal_threat
    # Effective Nernst layer must stay physical (tens of microns). Regression guard:
    # re-referencing the H-based natural-convection Sh to the gap must NOT collapse
    # the boundary layer to ~1 um (which would imply unrealistically fast transport).
    assert 10.0 <= res.effective_boundary_layer_um <= 300.0


def test_downward_flow_reversal_threat():
    """Verify that slow downward forced flow triggers opposing solutal flow reversal."""
    params = SolutalChannelParams(cathode_height_m=1.0)

    # Slow downward flow (0.02 m/s < u_crit ~ 0.15 m/s)
    res_slow_down = solve_solutal_mixed_convection(
        forced_velocity_m_s=0.02,
        flow_direction="downward",
        bulk_fe2_mol_L=1.5,
        surface_fe2_mol_L=0.3,
        params=params,
    )
    assert res_slow_down.is_flow_reversal_threat
    assert res_slow_down.forced_velocity_m_s < res_slow_down.critical_antireversal_velocity_m_s

    # Fast downward flow (0.50 m/s > u_crit) overcomes solutal buoyancy
    res_fast_down = solve_solutal_mixed_convection(
        forced_velocity_m_s=0.50,
        flow_direction="downward",
        bulk_fe2_mol_L=1.5,
        surface_fe2_mol_L=0.3,
        params=params,
    )
    assert not res_fast_down.is_flow_reversal_threat
