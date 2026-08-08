"""Tests for the operating-temperature Pourbaix boundary lines.

The legacy `models/pourbaix.py` keeps every standard potential E0 at its 25 °C
value and only scales the Nernst *slope* with T.  `models/pourbaix_at_T.py`
adds the standard-entropy E0(T) drift, which moves the Fe couples and the
hydrogen line in *different* directions.  These tests pin the monotonic
direction of every boundary so the central lever -- the gap between Fe2+
deposition and HER -- cannot drift silently.
"""

import pytest

from models.pourbaix import (
    E0_FE2_FE,
    E0_FE3_FE2,
    E0_FEOH2_FE,
    E0_FEOH3_FE2,
    E0_FEOH3_FEOH2,
    E0_HFEO2_FE,
)
from models.pourbaix_at_T import PourbaixAtT, e0_at_T, DS_FE2_FE, DS_FE3_FE2

T25, T60, T90 = 298.15, 333.15, 363.15


def e_pH(line, T, pH=0.0):
    """Helper: evaluate a boundary line method at a fixed T/pH."""
    p = PourbaixAtT(activity=1.0, temperature_C=T - 273.15)
    return float(getattr(p, line)(pH))


class TestLegacyAnchorsRecovered:
    """At 25 °C the at-T module must return the legacy module's E0 exactly."""

    def test_e0_function_recovers_reference(self):
        assert e0_at_T(E0_FE2_FE, DS_FE2_FE, 2, T25) == pytest.approx(E0_FE2_FE)
        assert e0_at_T(E0_FE3_FE2, DS_FE3_FE2, 1, T25) == pytest.approx(E0_FE3_FE2)

    def test_25C_lines_match_legacy_intercepts(self):
        p = PourbaixAtT(activity=1.0, temperature_C=25.0)
        assert p.E0_Fe2_Fe() == pytest.approx(E0_FE2_FE)
        assert p.E0_Fe3_Fe2() == pytest.approx(E0_FE3_FE2)
        assert p.E0_FeOH2_Fe() == pytest.approx(E0_FEOH2_FE)
        assert p.E0_FeOH3_Fe2() == pytest.approx(E0_FEOH3_FE2)
        assert p.E0_FeOH3_FeOH2() == pytest.approx(E0_FEOH3_FEOH2)
        assert p.E0_HFeO2_Fe() == pytest.approx(E0_HFEO2_FE)


class TestMonotonicDirections:
    """Each boundary line must shift monotonically with T in its known sense."""

    @pytest.mark.parametrize("line", [
        "E_Fe2_Fe",        # Fe2+/Fe deposition: shifts UP (more positive)
        "E_Fe3_Fe2",       # Fe3+/Fe2+: shifts UP
        "E_FeOH2_Fe",      # Fe(OH)2/Fe: shifts UP
        "E_FeOH3_FeOH2",   # Fe(OH)3/Fe(OH)2: shifts UP
        "E_HFeO2_Fe",      # HFeO2-/Fe: shifts UP
    ])
    def test_line_shifts_up_with_temperature(self, line):
        pH = 0.0
        v25, v60, v90 = (e_pH(line, T, pH) for T in (T25, T60, T90))
        assert v25 < v60 < v90, f"{line} should rise with T: {v25} {v60} {v90}"

    def test_feoh3_fe2_shifts_down_with_temperature(self):
        pH = 0.0
        v25, v60, v90 = (e_pH("E_FeOH3_Fe2", T, pH) for T in (T25, T60, T90))
        assert v25 > v60 > v90

    def test_her_line_steepens_down_with_temperature(self):
        """H+/H2 line: SHE=0 V at every T, but the slope steepens, so at pH>0
        the line sits lower as T rises."""
        pH = 2.0
        h25, h60, h90 = (e_pH("E_HER", T, pH) for T in (T25, T60, T90))
        assert h25 > h60 > h90

    def test_oer_line_drifts_down_with_temperature(self):
        pH = 2.0
        o25, o60, o90 = (e_pH("E_OER", T, pH) for T in (T25, T60, T90))
        assert o25 > o60 > o90

    def test_fe_and_her_move_in_opposite_directions(self):
        """The distinguishing feature of the at-T diagram: the Fe deposition
        lines move *up* while the HER line moves *down*."""
        fe_shift = e_pH("E_Fe2_Fe", T60) - e_pH("E_Fe2_Fe", T25)
        her_shift = e_pH("E_HER", T60, pH=2.0) - e_pH("E_HER", T25, pH=2.0)
        assert fe_shift > 0 > her_shift


class TestWaterWindowConvention:
    def test_her_passes_through_zero_at_every_temperature(self):
        for T in (T25, T60, T90):
            assert e_pH("E_HER", T, pH=0.0) == pytest.approx(0.0, abs=1e-9)

    def test_nernst_slope_at_25C_is_negative_59_mv_per_pH(self):
        # 2H+/2e- couple: -59.16 mV/pH at 298.15 K, unchanged from legacy.
        dE = e_pH("E_FeOH2_Fe", T25, pH=14.0) - e_pH("E_FeOH2_Fe", T25, pH=0.0)
        assert dE / 14.0 == pytest.approx(-0.05916, abs=1e-4)


class TestCentralLever:
    """The gap between Fe2+ deposition and HER -- the program's lever -- must
    narrow as T rises (Fe line up, HER line down both shrink it)."""

    def test_her_margin_narrows_with_temperature(self):
        pH = 2.0  # acid regime -> Fe2+/Fe deposition branch
        margins = {
            T: PourbaixAtT(activity=1.0, temperature_C=T - 273.15).her_margin(pH)
            for T in (T25, T60, T90)
        }
        assert margins[T25] > margins[T60] > margins[T90]
        # 25 -> 60 C narrows by a few tens of mV, well within screening scale.
        assert 0.02 < margins[T25] - margins[T60] < 0.10

    def test_her_margin_positive_at_every_pH_and_T(self):
        for T in (T25, T60, T90):
            p = PourbaixAtT(activity=1.0, temperature_C=T - 273.15)
            for pH in (0, 2, 4, 7, 10, 12, 14):
                assert p.her_margin(pH) > 0

    def test_activity_dependence_preserved(self):
        dilute = PourbaixAtT(activity=1e-6).E_Fe2_Fe(2.0)
        conc = PourbaixAtT(activity=1.0).E_Fe2_Fe(2.0)
        assert dilute < conc


class TestVerticalBoundary:
    def test_fe2_precipitation_pH_shifts_left_with_temperature(self):
        """Fe(OH)2 solubility rises with T, so the Fe2+/Fe(OH)2 hydrolysis pH
        moves to a lower value (precipitation starts earlier) as T rises."""
        left_to_right = {
            T: PourbaixAtT(activity=1.0, temperature_C=T - 273.15).pH_Fe2_FeOH2
            for T in (T25, T60, T90)
        }
        assert left_to_right[T25] > left_to_right[T60] > left_to_right[T90]


def test_sweep_is_small_and_smoke():
    """boundary_sweep returns a JSON-ish dict per temperature (a smoke check
    that the runnable path stays wired)."""
    from models.pourbaix_at_T import boundary_sweep
    s = boundary_sweep(60.0, pH=(2.0,))
    assert isinstance(s, dict)
    assert s["temperature_C"] == 60.0
    assert s["E0_Fe2_Fe"] < s["E0_Fe3_Fe2"]