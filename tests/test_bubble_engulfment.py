"""
Unit tests for H₂ bubble engulfment -> deposit porosity (Round 5, B3).
"""

import pytest

from models.bubble_engulfment import (
    BubbleEngulfmentParams,
    bubble_capture_porosity_fraction,
    bubble_detachment_radius_m,
    deposit_advance_velocity_m_s,
)


def test_detachment_radius_positive_and_reasonable():
    """Detachment radius is positive and in the tens-to-hundreds of microns."""
    r = bubble_detachment_radius_m()
    assert r > 0.0
    assert 5.0 < r * 1e6 < 1000.0  # µm


def test_deposit_advance_velocity_scales_with_j():
    """Higher iron current -> faster deposit-front advance."""
    v_lo = deposit_advance_velocity_m_s(1000.0)
    v_hi = deposit_advance_velocity_m_s(10000.0)
    assert v_hi > v_lo
    assert v_lo > 0.0


def test_porosity_rises_with_her():
    """More HER -> more H₂ bubbles -> higher porosity."""
    lo = bubble_capture_porosity_fraction(3000.0, her_efficiency=0.02)
    hi = bubble_capture_porosity_fraction(3000.0, her_efficiency=0.3)
    assert hi["porosity_fraction"] > lo["porosity_fraction"]


def test_no_her_gives_no_porosity():
    """No HER -> no H₂ bubbles -> zero porosity."""
    res = bubble_capture_porosity_fraction(3000.0, her_efficiency=0.0)
    assert res["porosity_fraction"] == pytest.approx(0.0)
    assert res["pinhole_blister_flag"] is False


def test_porosity_capped_at_one():
    """Porosity fraction stays in [0, 1]."""
    res = bubble_capture_porosity_fraction(1e6, her_efficiency=0.9)
    assert 0.0 <= res["porosity_fraction"] <= 1.0


def test_high_her_triggers_blister_flag():
    """Sufficiently high HER-driven porosity trips the pinhole/blister flag."""
    p = BubbleEngulfmentParams(bubble_coverage_ref=0.5, capture_fraction_ref=0.9)
    res = bubble_capture_porosity_fraction(30000.0, her_efficiency=0.5, params=p)
    assert res["pinhole_blister_flag"] is True
