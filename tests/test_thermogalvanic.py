"""
Unit tests for thermogalvanic Seebeck potential and vertical current maldistribution.
"""

import pytest
from models.thermogalvanic import (
    ThermogalvanicParams,
    get_thermogalvanic_equilibrium_potential,
    solve_vertical_current_maldistribution,
)


def test_thermogalvanic_equilibrium_potentials():
    """Verify that equilibrium potentials shift with temperature matching Seebeck coefficients."""
    params = ThermogalvanicParams()

    # Fe reaction: S_Fe ≈ +1.2 mV/K (increases with T)
    e_fe_25 = get_thermogalvanic_equilibrium_potential(25.0, "fe", params=params)
    e_fe_60 = get_thermogalvanic_equilibrium_potential(60.0, "fe", params=params)

    assert e_fe_60 > e_fe_25
    assert pytest.approx(e_fe_60 - e_fe_25, abs=1e-3) == params.s_fe_V_K * 35.0

    # OER reaction: S_OER ≈ -1.36 mV/K (decreases with T)
    e_oer_25 = get_thermogalvanic_equilibrium_potential(25.0, "oer", params=params)
    e_oer_60 = get_thermogalvanic_equilibrium_potential(60.0, "oer", params=params)

    assert e_oer_60 < e_oer_25


def test_solve_vertical_current_maldistribution_flow_trends():
    """Verify fluid velocity and current trends match physical non-isothermal cooling."""
    params = ThermogalvanicParams()

    # Compare slow (0.02 m/s) vs fast (0.2 m/s) flow at 150 mA/cm² average
    res_slow = solve_vertical_current_maldistribution(
        j_avg_mA_cm2=150.0,
        v_cell_V=3.2,
        t_inlet_C=60.0,
        flow_velocity_m_s=0.02,
        params=params,
    )
    res_fast = solve_vertical_current_maldistribution(
        j_avg_mA_cm2=150.0,
        v_cell_V=3.2,
        t_inlet_C=60.0,
        flow_velocity_m_s=0.2,
        params=params,
    )

    # 1. Faster flow carries heat away, reducing the total temperature rise along the channel
    assert res_fast.total_thermal_rise_C < res_slow.total_thermal_rise_C

    # 2. Faster flow and lower thermal gradients lead to a more uniform current distribution (lower MDI)
    assert res_fast.maldistribution_index < res_slow.maldistribution_index
    assert res_fast.maldistribution_index >= 0.0


def test_current_density_impact_on_maldistribution():
    """Verify higher nominal currents increase thermal gradients and vertical maldistribution."""
    params = ThermogalvanicParams()

    res_low_j = solve_vertical_current_maldistribution(
        j_avg_mA_cm2=50.0,
        v_cell_V=2.5,
        t_inlet_C=60.0,
        flow_velocity_m_s=0.1,
        params=params,
    )
    res_high_j = solve_vertical_current_maldistribution(
        j_avg_mA_cm2=300.0,
        v_cell_V=4.0,
        t_inlet_C=60.0,
        flow_velocity_m_s=0.1,
        params=params,
    )

    assert res_high_j.total_thermal_rise_C > res_low_j.total_thermal_rise_C
    assert res_high_j.maldistribution_index > res_low_j.maldistribution_index
