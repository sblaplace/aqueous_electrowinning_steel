"""
Unit tests for recombination-poison control of absorbed hydrogen (Round 5, B1).
"""

import pytest

from models.recombination_poison import (
    PoisonParams,
    absorption_promotion_factor,
    poison_coverage_fraction,
    poisoned_absorption_fraction,
    sulfide_from_ppm,
)


def test_clean_surface_no_promotion():
    """Zero poison concentration must give zero coverage and promotion factor 1.0."""
    coverages = poison_coverage_fraction({}, temperature_C=60.0, cathodic_overpotential_V=0.2)
    assert all(v == 0.0 for v in coverages.values())
    assert absorption_promotion_factor(coverages) == pytest.approx(1.0)


def test_coverage_rises_with_concentration():
    """Higher sulfide concentration -> higher sulfide coverage."""
    lo = poison_coverage_fraction({"sulfide": 1e-5}, 60.0, 0.2)
    hi = poison_coverage_fraction({"sulfide": 1e-2}, 60.0, 0.2)
    assert hi["sulfide"] > lo["sulfide"]
    assert 0.0 <= lo["sulfide"] <= 1.0
    assert 0.0 <= hi["sulfide"] <= 1.0


def test_coverage_rises_with_overpotential():
    """More negative cathodic overpotential -> higher poison coverage."""
    a = poison_coverage_fraction({"sulfide": 1e-3}, 60.0, 0.05)
    b = poison_coverage_fraction({"sulfide": 1e-3}, 60.0, 0.4)
    assert b["sulfide"] > a["sulfide"]


def test_promotion_factor_multiplies_and_caps():
    """Promotion factor >= 1 and capped at max_promotion_factor."""
    params = PoisonParams()
    coverages = {"sulfide": 1.0, "arsenic": 1.0}
    factor = absorption_promotion_factor(coverages, params)
    assert factor >= 1.0
    assert factor <= params.max_promotion_factor


def test_poisoned_fraction_raises_absorbed_h():
    """Applying the promotion factor must raise the base absorption fraction."""
    base = 0.05
    res = poisoned_absorption_fraction(base, {"sulfide": 1e-3}, 60.0, 0.3)
    assert res["promotion_factor"] > 1.0
    assert res["poisoned_absorption_fraction"] > base
    assert res["poisoned_absorption_fraction"] <= 1.0


def test_arsenic_stronger_than_sulfide_relative_promotion():
    """As is a stronger promoter per unit coverage than S (screening ranking)."""
    theta_s = poison_coverage_fraction({"sulfide": 1e-3}, 60.0, 0.3)
    theta_as = poison_coverage_fraction({"arsenic": 1e-3}, 60.0, 0.3)
    f_s = absorption_promotion_factor(theta_s)
    f_as = absorption_promotion_factor(theta_as)
    assert f_as > f_s


def test_sulfide_from_ppm_monotonic():
    """Higher feedstock S ppm -> higher bath sulfide estimate."""
    assert sulfide_from_ppm(600.0) > sulfide_from_ppm(300.0)
    assert sulfide_from_ppm(0.0) == 0.0
