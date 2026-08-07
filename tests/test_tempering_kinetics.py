"""
Unit tests for 4-stage tempering kinetics and LSW Ostwald ripening model.
"""

import pytest
from models.tempering_kinetics import (
    SteelMicrostructureSpec,
    lsw_coarsening_rate_constant,
    simulate_tempering_kinetics,
)


def test_tempering_stages_progression():
    """Verify that tempering temperature progresses through stages 1 to 4."""
    spec = SteelMicrostructureSpec(carbon_wt_percent=0.40)

    res_150 = simulate_tempering_kinetics(spec, temperature_C=150.0, time_hours=1.0)
    assert "Stage 1" in res_150.tempering_stage
    assert not res_150.tme_embrittlement_risk

    res_300 = simulate_tempering_kinetics(spec, temperature_C=300.0, time_hours=1.0)
    assert "Stage 3" in res_300.tempering_stage
    assert res_300.tme_embrittlement_risk  # Tempered Martensite Embrittlement

    res_550 = simulate_tempering_kinetics(spec, temperature_C=550.0, time_hours=2.0)
    assert "Stage 4" in res_550.tempering_stage
    assert not res_550.tme_embrittlement_risk


def test_lsw_ostwald_ripening_coarsening():
    """Verify that Stage 4 spheroidization coarsens carbides and softens matrix."""
    spec = SteelMicrostructureSpec(carbon_wt_percent=0.40)

    res_short = simulate_tempering_kinetics(spec, temperature_C=600.0, time_hours=0.5)
    res_long = simulate_tempering_kinetics(spec, temperature_C=600.0, time_hours=8.0)

    # Longer annealing increases mean carbide radius via LSW
    assert res_long.mean_carbide_radius_nm > res_short.mean_carbide_radius_nm
    # Larger spacing reduces Orowan yield increment
    assert res_long.orowan_yield_increment_MPa < res_short.orowan_yield_increment_MPa
    # Coarser carbides restore ductile impact energy
    assert res_long.estimated_charpy_energy_J > 30.0
