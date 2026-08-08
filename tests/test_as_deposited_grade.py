"""
Unit tests for as-deposited Fe-C -> steel grade router (Round 5, A2).
"""

import pytest

from models.as_deposited_grade import (
    as_deposited_grade,
    phase_fractions,
)


def test_hypoeutectoid_slow_cooling_ferrite_pearlite():
    """Low-C slow-cooled steel is ferrite + pearlite (no martensite)."""
    ph = phase_fractions(0.15, cooling="slow")
    assert ph["martensite"] == 0.0
    assert ph["ferrite"] > 0.0
    assert ph["pearlite"] > 0.0
    assert abs((ph["ferrite"] + ph["pearlite"]) - 1.0) < 1e-6


def test_fast_quench_gives_martensite():
    """Fast cooling produces martensite."""
    ph = phase_fractions(0.4, cooling="fast")
    assert ph["martensite"] > 0.5
    assert ph["ferrite"] + ph["martensite"] == pytest.approx(1.0, abs=1e-6)


def test_grade_routing_monotonic():
    """Higher C -> higher-carbon AISI grade."""
    assert "1005" in as_deposited_grade(0.03)["grade"]
    assert "1045" in as_deposited_grade(0.40)["grade"]


def test_quenched_harder_than_annealed():
    """Martensitic (fast-cooled) layer is harder than ferritic/pearlitic."""
    annealed = as_deposited_grade(0.40, cooling="slow")
    quenched = as_deposited_grade(0.40, cooling="fast")
    assert quenched["hardness_HV_proxy"] > annealed["hardness_HV_proxy"]
    assert quenched["yield_strength_MPa_proxy"] > annealed["yield_strength_MPa_proxy"]


def test_deep_draw_eligibility():
    """Low S/P is deep-draw eligible; high S/P is not."""
    clean = as_deposited_grade(0.05, s_wt_percent=0.01, p_wt_percent=0.01)
    dirty = as_deposited_grade(0.05, s_wt_percent=0.05, p_wt_percent=0.03)
    assert clean["deep_draw_eligible"] is True
    assert dirty["deep_draw_eligible"] is False
