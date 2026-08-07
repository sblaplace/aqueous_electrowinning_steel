"""Tests for the S/P/Mn/Si/B bath-impurity co-deposition and AISI routing."""

import numpy as np
import pytest

from models.bath_impurity_codeposition import (
    BathImpurityCoDeposition,
    BathImpurityKinetics,
    SCREENING_FLAG,
    langmuir_coverage,
    route_steel_grade,
)


# ── Langmuir helper ────────────────────────────────────────────────────────


def test_langmuir_coverage_bounds_and_monotonic():
    assert langmuir_coverage(0.0, 0.02) == 0.0
    low = langmuir_coverage(10.0, 0.02)
    high = langmuir_coverage(1000.0, 0.02)
    assert 0.0 <= low <= high < 1.0
    # Large KC → coverage → 1 (asymptotically)
    assert langmuir_coverage(1e6, 0.02) == pytest.approx(1.0, abs=1e-4)


# ── Deposit composition ────────────────────────────────────────────────────


def test_deposit_composition_keys_and_sum():
    r = BathImpurityCoDeposition().deposit_composition(100.0)
    for k in ["fe_wt_percent", "mn_wt_percent", "s_wt_percent", "p_wt_percent",
              "si_wt_percent", "b_wt_percent"]:
        assert k in r
    total = (r["fe_wt_percent"] + r["mn_wt_percent"] + r["s_wt_percent"]
             + r["p_wt_percent"] + r["si_wt_percent"] + r["b_wt_percent"])
    assert total == pytest.approx(100.0, abs=1e-6)
    assert r["flag"] == SCREENING_FLAG


def test_deposit_fe_is_mostly_iron():
    r = BathImpurityCoDeposition().deposit_composition(100.0)
    assert r["fe_wt_percent"] > 99.0


def test_pure_bath_gives_negligible_impurities():
    model = BathImpurityCoDeposition(mn_ppm=0.0, s_ppm=0.0, p_ppm=0.0,
                                     si_ppm=0.0, b_ppm=0.0)
    r = model.deposit_composition(100.0)
    assert r["fe_wt_percent"] == pytest.approx(100.0, abs=1e-9)


def test_mn_stays_negligible_as_less_noble():
    # Mn²⁺ (E° ≈ −1.19 V) is far less noble than Fe (E° ≈ −0.45 V), so under
    # normal Fe-potential deposition it does not co-deposit — the same
    # less-noble behaviour the parent module models for Zn.  Uptake stays
    # negligible even at high bath Mn.
    lo = BathImpurityCoDeposition(mn_ppm=10).deposit_composition(100.0)["mn_in_ppm"]
    hi = BathImpurityCoDeposition(mn_ppm=1000).deposit_composition(100.0)["mn_in_ppm"]
    assert lo < 1e6 * 1e-6  # trace-level (≤ ~ppm) both ways
    assert hi <= 1.0


def test_s_uptake_increases_with_bath_concentration():
    lo = BathImpurityCoDeposition(s_ppm=5).deposit_composition(100.0)["s_in_ppm"]
    hi = BathImpurityCoDeposition(s_ppm=500).deposit_composition(100.0)["s_in_ppm"]
    assert hi > lo


def test_custom_kinetics_overrides_defaults():
    custom = BathImpurityKinetics(mn_i0=1.0e-5, k_s=0.05)
    lo = BathImpurityCoDeposition(s_ppm=100).deposit_composition(100.0)["s_in_ppm"]
    hi = BathImpurityCoDeposition(s_ppm=100, custom_kinetics=custom).deposit_composition(100.0)["s_in_ppm"]
    assert hi > lo


def test_ppm_wt_consistency():
    r = BathImpurityCoDeposition().deposit_composition(100.0)
    for name in ["mn", "s", "p", "si", "b"]:
        assert r[f"{name}_in_ppm"] == pytest.approx(
            r[f"{name}_wt_percent"] * 1e4, abs=1e-9)


def test_all_waveforms_and_bath_types_run():
    for bath in ["sulfate", "chloride"]:
        r = BathImpurityCoDeposition(bath_type=bath).deposit_composition(100.0)
        assert np.isfinite(r["total_impurity_wt"])
        assert r["bath_type"] == bath


# ── AISI routing ───────────────────────────────────────────────────────────


def test_route_low_sulfur_p_deep_drawing():
    r = route_steel_grade(c_wt_percent=0.05, mn_wt_percent=0.2,
                          p_wt_percent=0.01, s_wt_percent=0.01)
    assert r["category"] == "deep_drawing"
    assert r["flag"] == SCREENING_FLAG


def test_route_high_sulfur_resulfurized():
    r = route_steel_grade(c_wt_percent=0.10, mn_wt_percent=0.5,
                          p_wt_percent=0.02, s_wt_percent=0.08)
    assert r["category"] == "resulfurized"
    assert r["grade"] == "resulfurized"


def test_route_1005_vs_1018():
    r1005 = route_steel_grade(c_wt_percent=0.04, mn_wt_percent=0.2,
                              p_wt_percent=0.03, s_wt_percent=0.03)
    assert r1005["grade"] == "AISI_1005"
    r1018 = route_steel_grade(c_wt_percent=0.18, mn_wt_percent=0.7,
                              p_wt_percent=0.03, s_wt_percent=0.03)
    assert r1018["grade"] == "AISI_1018"


def test_route_out_of_spec():
    r = route_steel_grade(c_wt_percent=0.5, mn_wt_percent=1.0,
                          p_wt_percent=0.03, s_wt_percent=0.03)
    assert r["grade"] == "out-of-spec-10xx"
