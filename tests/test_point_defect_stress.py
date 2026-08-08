"""
Unit tests for non-equilibrium point-defect intrinsic stress (Round 5, C2).
"""

import pytest

from models.point_defect_stress import (
    PointDefectStressParams,
    defect_injection_stress_MPa,
)


def test_stress_rises_with_overpotential():
    """Higher overpotential -> more defect injection -> higher stress."""
    lo = defect_injection_stress_MPa(0.1, 60.0, 900.0)
    hi = defect_injection_stress_MPa(0.5, 60.0, 900.0)
    assert hi["steady_stress_MPa"] > lo["steady_stress_MPa"]
    assert hi["net_stress_MPa"] > lo["net_stress_MPa"]


def test_stress_builds_with_time():
    """Stress relaxes/accumulates with deposition time toward steady state."""
    early = defect_injection_stress_MPa(0.3, 60.0, 10.0)
    late = defect_injection_stress_MPa(0.3, 60.0, 3600.0)
    assert late["net_stress_MPa"] > early["net_stress_MPa"]
    assert early["fractional_relaxation"] <= 1.0


def test_additive_increases_defect_stress():
    """Brightener coverage increases defect incorporation -> higher stress."""
    plain = defect_injection_stress_MPa(0.3, 60.0, 900.0, additive_coverage_fraction=0.0)
    with_add = defect_injection_stress_MPa(0.3, 60.0, 900.0, additive_coverage_fraction=0.7)
    assert with_add["steady_stress_MPa"] > plain["steady_stress_MPa"]


def test_zero_overpotential_zero_stress():
    """No overpotential -> no defect injection -> zero stress."""
    res = defect_injection_stress_MPa(0.0, 60.0, 900.0)
    assert res["steady_stress_MPa"] == pytest.approx(0.0)
    assert res["net_stress_MPa"] == pytest.approx(0.0)


def test_higher_temperature_reduces_time_constant():
    """Higher temperature shortens the anneal time constant."""
    cool = defect_injection_stress_MPa(0.3, 40.0, 900.0)
    warm = defect_injection_stress_MPa(0.3, 80.0, 900.0)
    assert warm["time_constant_s"] < cool["time_constant_s"]
