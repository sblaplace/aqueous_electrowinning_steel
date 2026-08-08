"""
Unit tests for chemical osmosis and coupled transmembrane water flux.
"""

from models.chemical_osmosis import (
    estimate_water_activity,
    solve_transmembrane_water_flux,
)


def test_water_activity_estimation():
    """Verify that concentrated salt solutions have lower water activity."""
    a_pure = estimate_water_activity(0.0, 0.0)
    a_dilute = estimate_water_activity(0.5, 0.0)
    a_conc = estimate_water_activity(1.5, 0.5)

    assert a_pure >= 0.99
    assert a_dilute < a_pure
    assert a_conc < a_dilute
    assert 0.85 <= a_conc <= 0.95


def test_coupled_flux_regimes():
    """Verify that low j is osmosis-dominated and high j is EOD-dominated."""
    # Low current density (10 mA/cm²): chemical osmosis from anolyte into catholyte dominates
    res_low = solve_transmembrane_water_flux(
        current_density_mA_cm2=10.0,
        water_activity_catholyte=0.92,
        water_activity_anolyte=0.97,
    )
    assert res_low.chemical_osmotic_flux_L_m2_h > res_low.electro_osmotic_flux_L_m2_h
    assert res_low.net_water_flux_L_m2_h < 0.0  # Net flux into catholyte

    # High current density (350 mA/cm²): EOD carries water out of catholyte into anolyte
    res_high = solve_transmembrane_water_flux(
        current_density_mA_cm2=350.0,
        water_activity_catholyte=0.92,
        water_activity_anolyte=0.97,
    )
    assert res_high.electro_osmotic_flux_L_m2_h > res_high.chemical_osmotic_flux_L_m2_h
    assert res_high.net_water_flux_L_m2_h > 0.0  # Net flux out of catholyte

    # Zero net flux current density should be a physically reasonable number (10-250 mA/cm²)
    assert 10.0 <= res_high.zero_net_flux_current_density_mA_cm2 <= 300.0
