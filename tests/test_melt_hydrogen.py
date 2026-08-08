"""
Unit tests for melt hydrogen / white-spot flake risk (Round 5, B2).
"""

import pytest

from models.melt_hydrogen import (
    MeltHydrogenParams,
    melt_hydrogen_budget,
    required_bakeout_C_H_ppm,
    sieverts_H_ppm_liquid,
)


def test_sieverts_liquid_solubility_positive_and_temperature_dependent():
    """Liquid-Fe H solubility is positive and rises with temperature."""
    s_1500 = sieverts_H_ppm_liquid(1500.0)
    s_1650 = sieverts_H_ppm_liquid(1650.0)
    assert s_1500 > 0.0
    assert s_1650 > s_1500  # hotter -> more H dissolves


def test_zero_deposit_h_gives_zero_risk():
    """A hydrogen-free charge has no flake risk."""
    res = melt_hydrogen_budget(c_h_deposit_ppm=0.0)
    assert res["h_in_melt_ppm"] == pytest.approx(0.0)
    assert res["flake_risk_index"] == pytest.approx(0.0)
    assert res["needs_bake_or_degas"] is False


def test_higher_deposit_h_raises_risk():
    """More H in the deposit -> higher melt H and flake risk."""
    lo = melt_hydrogen_budget(c_h_deposit_ppm=1.0)
    hi = melt_hydrogen_budget(c_h_deposit_ppm=20.0)
    assert hi["excess_h_ppm"] > lo["excess_h_ppm"]
    assert hi["flake_risk_index"] > lo["flake_risk_index"]


def test_risk_capped_at_one():
    """Flake-risk index stays within [0, 1]."""
    res = melt_hydrogen_budget(c_h_deposit_ppm=1e6)
    assert 0.0 <= res["flake_risk_index"] <= 1.0


def test_required_bakeout_lowers_allowable_h():
    """The allowable deposit-H spec is positive and scales with charge fraction."""
    spec = required_bakeout_C_H_ppm(25.0)
    assert spec > 0.0
    # More electrowon in the charge -> tighter (lower) allowable deposit H.
    spec_full = required_bakeout_C_H_ppm(25.0, charge_fraction=1.0)
    spec_half = required_bakeout_C_H_ppm(25.0, charge_fraction=0.5)
    assert spec_full < spec_half


def test_charge_transfer_fraction_lowers_risk():
    """Lower transfer fraction (more H off-gasses) -> lower risk."""
    p = MeltHydrogenParams(charge_h_transfer_fraction=0.1)
    res = melt_hydrogen_budget(c_h_deposit_ppm=20.0, params=p)
    assert res["flake_risk_index"] < melt_hydrogen_budget(20.0)["flake_risk_index"]
