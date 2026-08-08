"""
Unit tests for tramp Cu/Sn surface hot-shortness (Round 5, E1).
"""

import pytest

from models.hot_shortness import (
    HotShortnessParams,
    cu_sn_surface_enrichment,
    effective_liquidus_C,
    hot_shortness_risk,
)


def test_surface_enrichment_positive_and_monotonic():
    """Higher bulk Cu/Sn -> higher surface enrichment."""
    lo = cu_sn_surface_enrichment(0.05, 0.01)
    hi = cu_sn_surface_enrichment(0.30, 0.05)
    assert hi["cu_surface_wt"] > lo["cu_surface_wt"]
    assert hi["sn_surface_wt"] > lo["sn_surface_wt"]
    assert lo["cu_surface_wt"] > 0.0


def test_sn_depresses_liquidus():
    """Surface Sn lowers the effective Cu-rich liquidus."""
    low = effective_liquidus_C(0.0)
    high = effective_liquidus_C(2.0)
    assert high < low


def test_risk_rises_with_tramp_content():
    """Higher Cu/Sn -> higher hot-shortness risk."""
    lo = hot_shortness_risk(0.02, 0.005)
    hi = hot_shortness_risk(0.40, 0.10)
    assert hi["risk_index"] > lo["risk_index"]
    assert 0.0 <= hi["risk_index"] <= 1.0


def test_risk_rises_with_roll_temperature():
    """Rolling hotter (above the liquidus) increases risk."""
    cool = hot_shortness_risk(0.02, 0.005, roll_temperature_C=1050.0)
    hot = hot_shortness_risk(0.02, 0.005, roll_temperature_C=1250.0)
    assert hot["risk_index"] > cool["risk_index"]


def test_deep_draw_eligibility():
    """Low residual Cu+Sn is eligible for deep-drawing; high is not."""
    p = HotShortnessParams(allowable_residual_deep_draw_wt=0.10)
    clean = hot_shortness_risk(0.03, 0.01, params=p)
    dirty = hot_shortness_risk(0.30, 0.10, params=p)
    assert clean["eligible_deep_draw"] is True
    assert dirty["eligible_deep_draw"] is False


def test_rolling_below_ceiling_flag():
    """Flag reflects whether the requested roll temp is safe."""
    p = HotShortnessParams()
    res = hot_shortness_risk(0.05, 0.01, roll_temperature_C=1150.0, params=p)
    assert isinstance(res["rolling_below_ceiling"], bool)
    assert res["roll_temp_ceiling_C"] > 0.0
