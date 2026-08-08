"""
Unit tests for electrochemical carbon co-deposition (Round 5, A1).
"""

import pytest

from models.carbon_electrodeposition import (
    CarbonElectrodepositionParams,
    additive_carbon_wt_percent,
    carbon_limiting_current_density_A_m2,
    carbon_partial_current_density_A_m2,
    deposit_carbon_wt_percent,
    steel_grade_for_carbon,
)


def test_limiting_current_scales_with_concentration_and_temp():
    """Higher dissolved-carbon concentration -> higher transport-limited j_C."""
    j_lo = carbon_limiting_current_density_A_m2(0.01, 25.0)
    j_hi = carbon_limiting_current_density_A_m2(0.5, 25.0)
    assert j_hi > j_lo
    assert j_lo > 0.0


def test_zero_carbon_gives_zero_deposit_c():
    """No dissolved carbon -> no electrodeposited carbon (no additive)."""
    res = deposit_carbon_wt_percent(3000.0, 0.0, 60.0, 0.4)
    assert res["c_wt_percent"] == pytest.approx(0.0)
    assert res["j_c_A_m2"] == 0.0


def test_c_wt_percent_rises_with_dissolved_carbon():
    """More dissolved carbon -> higher deposit C wt%."""
    lo = deposit_carbon_wt_percent(3000.0, 0.02, 60.0, 0.4)
    hi = deposit_carbon_wt_percent(3000.0, 0.5, 60.0, 0.4)
    assert hi["c_wt_percent"] > lo["c_wt_percent"]


def test_c_wt_percent_falls_as_fe_current_rises():
    """At fixed carbon supply, more Fe current dilutes the carbon."""
    dilute = deposit_carbon_wt_percent(30000.0, 0.1, 60.0, 0.4)
    conc = deposit_carbon_wt_percent(3000.0, 0.1, 60.0, 0.4)
    assert conc["c_wt_percent"] > dilute["c_wt_percent"]


def test_grade_routing_monotonic():
    """Higher C wt% routes to higher carbon AISI grades."""
    assert steel_grade_for_carbon(0.03).startswith("AISI 1005")
    assert steel_grade_for_carbon(0.10).startswith("AISI 1018")
    assert steel_grade_for_carbon(0.30).startswith("AISI 1045")
    assert steel_grade_for_carbon(0.60).startswith("AISI 1095")


def test_additive_carbon_increases_with_loading_and_age():
    """Additive carbon contribution rises with additive loading and aging."""
    assert additive_carbon_wt_percent(4.0) > additive_carbon_wt_percent(2.0)
    assert additive_carbon_wt_percent(2.0, aged_fraction=0.0) == pytest.approx(0.0)


def test_carbon_vector_default_is_formate():
    """Default carbon vector uses n_C=3 (formate) and formate diffusivity."""
    assert CarbonElectrodepositionParams().n_C == 3
