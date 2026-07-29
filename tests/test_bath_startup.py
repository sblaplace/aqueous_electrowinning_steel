"""Tests for bath startup kinetics — Fe²⁺ air oxidation and ascorbic acid stabilization."""

import pytest
import numpy as np
from models.bath_startup import (
    BathParams,
    g_per_L_to_mol_L,
    dissolved_o2_saturation_mol_L,
    fe2_oxidation_rate,
    aa_reduction_rate,
    aa_autoxidation_rate,
    simulate_bath,
    recommend_ascorbic_loading,
    ascorbic_consumption_summary,
    sensitivity_ph,
    sensitivity_temperature,
    M_AA,
    ASCORBIC_PKA1,
)


# ── Test 1: Fe³⁺/Fe²⁺ ratio rises faster in open beaker than covered ────────


def test_fe3_ratio_open_vs_covered():
    """Uncovered beaker (high SA/V) oxidizes Fe²⁺ faster than covered (low SA/V).

    At pH 3, 25°C the oxidation is significant enough to see a clear difference.
    """
    t_end = 12.0

    open_params = BathParams(fe2_0=1.0, pH=3.0, sa_v_ratio=2.0)
    covered_params = BathParams(fe2_0=1.0, pH=3.0, sa_v_ratio=0.1)

    res_open = simulate_bath(open_params, t_end_hr=t_end)
    res_covered = simulate_bath(covered_params, t_end_hr=t_end)

    # Open beaker should reach higher Fe³⁺ ratio at end
    assert res_open.fe3_ratio[-1] > res_covered.fe3_ratio[-1], \
        "Open beaker should have higher Fe³⁺/Fe²⁺ than covered"

    # Both should show monotonic Fe³⁺ increase (no AA = no regeneration)
    assert res_open.fe3[-1] > res_open.fe3[0], "Fe³⁺ should increase without AA"
    assert res_covered.fe3[-1] > res_covered.fe3[0], "Fe³⁺ should increase without AA"


# ── Test 2: Ascorbic acid consumption rate increases with temperature ────────


def test_aa_consumption_increases_with_temperature():
    """Higher temperature → faster kinetics → more AA consumed."""
    temps = [25.0, 40.0, 60.0]
    consumptions = []

    for T in temps:
        p = BathParams(fe2_0=1.0, aa_0=g_per_L_to_mol_L(1.0),
                       pH=2.0, T_C=T, sa_v_ratio=1.0)
        summary = ascorbic_consumption_summary(p, t_end_hr=12.0)
        consumptions.append(summary["aa_consumed_g_L"])

    for i in range(1, len(consumptions)):
        assert consumptions[i] > consumptions[i - 1], \
            f"AA consumption should increase with T: {consumptions}"


# ── Test 3: AA consumption increases with [Fe²⁺] ────────────────────────────


def test_aa_consumption_increases_with_fe2():
    """Higher [Fe²⁺] → more oxidation → more AA consumed.

    Use pH 3.0 and sa_v=2 to stress the system so the differences are visible.
    """
    fe2_levels = [0.5, 1.0, 2.0]
    consumptions = []

    for fe2_0 in fe2_levels:
        p = BathParams(fe2_0=fe2_0, aa_0=g_per_L_to_mol_L(1.0),
                       pH=3.0, T_C=40.0, sa_v_ratio=2.0)
        summary = ascorbic_consumption_summary(p, t_end_hr=12.0)
        consumptions.append(summary["aa_consumed_g_L"])

    for i in range(1, len(consumptions)):
        assert consumptions[i] > consumptions[i - 1], \
            f"AA consumption should increase with [Fe²⁺]: {consumptions}"


# ── Test 4: AA consumption increases with pH (faster oxidation) ─────────────


def test_aa_consumption_increases_with_pH():
    """Higher pH → much faster Fe²⁺ oxidation (OH⁻² dependence) → more AA consumed."""
    p_low = BathParams(fe2_0=1.0, aa_0=g_per_L_to_mol_L(1.0),
                       pH=2.0, T_C=25.0, sa_v_ratio=1.0)
    p_high = BathParams(fe2_0=1.0, aa_0=g_per_L_to_mol_L(1.0),
                        pH=3.0, T_C=25.0, sa_v_ratio=1.0)

    sum_low = ascorbic_consumption_summary(p_low, t_end_hr=12.0)
    sum_high = ascorbic_consumption_summary(p_high, t_end_hr=12.0)

    assert sum_high["aa_consumed_g_L"] > sum_low["aa_consumed_g_L"], \
        f"Higher pH should consume more AA: pH2={sum_low['aa_consumed_g_L']:.4f}, " \
        f"pH3={sum_high['aa_consumed_g_L']:.4f}"


# ── Test 5: Recommended ascorbic acid loading for 24h stability ──────────────


def test_recommend_ascorbic_loading_24h():
    """recommend_ascorbic_loading returns a loading that keeps Fe³⁺/Fe²⁺ < 5% for 24h.

    At pH 3.0, 40°C, open beaker the oxidation is significant.
    """
    rec = recommend_ascorbic_loading(pH=3.0, T_C=40.0, fe2_0=1.0,
                                     sa_v_ratio=2.0, target_hr=24.0)
    assert 0.0 < rec <= 50.0, f"Recommended loading unreasonable: {rec} g/L"

    # Verify it actually works
    p = BathParams(fe2_0=1.0, aa_0=g_per_L_to_mol_L(rec),
                   pH=3.0, T_C=40.0, sa_v_ratio=2.0)
    res = simulate_bath(p, t_end_hr=24.1)
    assert res.time_to_threshold_hr is None or res.time_to_threshold_hr > 24.0, \
        f"Recommended loading {rec:.2f} g/L should keep bath stable for 24h, " \
        f"but threshold reached at {res.time_to_threshold_hr:.1f}h"


# ── Test 6: pH sensitivity — pH 3.0 vs 3.5 changes the answer ───────────────


def test_ph_sensitivity_changes_result():
    """pH 2.5 should require more AA than pH 2.0 (10× oxidation rate, 100× OH⁻²).

    At T=50°C, sa_v=2 the system is stressed. pH 2.5 crosses the 5% threshold
    in 24h; pH 2.0 stays under 1%. So recommend returns ~0 for pH 2.0 and
    significant AA for pH 2.5.
    """
    rec_20 = recommend_ascorbic_loading(pH=2.0, T_C=50.0, fe2_0=1.0,
                                        sa_v_ratio=2.0, target_hr=24.0)
    rec_25 = recommend_ascorbic_loading(pH=2.5, T_C=50.0, fe2_0=1.0,
                                        sa_v_ratio=2.0, target_hr=24.0)

    # pH 2.0: bath stable → rec ≈ 0; pH 2.5: bath crosses 5% → rec > 0
    assert rec_25 > rec_20, \
        f"pH 2.5 should need more AA than pH 2.0: {rec_25:.2f} vs {rec_20:.2f} g/L"

    # The difference should be substantial (not just rounding noise)
    assert rec_25 > 1.0, \
        f"pH 2.5 at 50°C should need meaningful AA: {rec_25:.2f} g/L"


# ── Test 7: AA extends time-to-threshold dramatically ───────────────────────


def test_aa_extends_threshold_time():
    """Adding AA should extend time-to-threshold.

    Use pH 3.0, 40°C, sa_v=2 where oxidation is fast enough to reach
    threshold in <48h without AA.
    """
    no_aa = BathParams(fe2_0=1.0, aa_0=0.0, pH=3.0, T_C=40.0, sa_v_ratio=2.0)
    with_aa = BathParams(fe2_0=1.0, aa_0=g_per_L_to_mol_L(2.0),
                         pH=3.0, T_C=40.0, sa_v_ratio=2.0)

    res_no = simulate_bath(no_aa, t_end_hr=48.0)
    res_with = simulate_bath(with_aa, t_end_hr=48.0)

    assert res_no.time_to_threshold_hr is not None, \
        "Without AA, threshold should be reached within 48h at pH 3, 40°C"

    # With AA, threshold should be reached much later (or not at all)
    if res_with.time_to_threshold_hr is not None:
        assert res_with.time_to_threshold_hr > res_no.time_to_threshold_hr, \
            "AA should extend time-to-threshold"


# ── Test 8: Fe mass balance — total Fe conserved ────────────────────────────


def test_fe_mass_balance():
    """[Fe²⁺] + [Fe³⁺] should remain constant (Fe not created or destroyed)."""
    params = BathParams(fe2_0=1.0, aa_0=g_per_L_to_mol_L(1.0),
                        pH=2.0, T_C=25.0, sa_v_ratio=1.0)
    res = simulate_bath(params, t_end_hr=24.0)

    total_fe = res.fe2 + res.fe3
    initial_fe = params.fe2_0 + params.fe3_0

    np.testing.assert_allclose(total_fe, initial_fe, rtol=1e-5,
                                err_msg="Total Fe (Fe²⁺+Fe³⁺) should be conserved")


# ── Test 9: Dissolved O₂ saturation decreases with temperature ──────────────


def test_o2_saturation_decreases_with_temperature():
    """Gas solubility decreases with T — O₂ saturation should follow."""
    o2_25 = dissolved_o2_saturation_mol_L(25.0)
    o2_60 = dissolved_o2_saturation_mol_L(60.0)

    assert o2_25 > o2_60, "O₂ saturation should decrease with temperature"
    # Sanity: 25°C O₂ sat should be ~2.6e-4 M (Weiss 1970)
    assert 1e-4 < o2_25 < 1e-3, f"O₂ saturation at 25°C unreasonable: {o2_25}"


# ── Test 10: g_per_L_to_mol_L conversion ────────────────────────────────────


def test_g_per_L_conversion():
    """1 g/L ascorbic acid ≈ 5.68 mmol/L (MW = 176.12 g/mol)."""
    mol = g_per_L_to_mol_L(1.0)
    expected = 1.0 / 176.12
    assert abs(mol - expected) < 1e-10, f"Conversion mismatch: {mol} vs {expected}"
    assert abs(mol - 5.68e-3) < 1e-4, f"1 g/L AA should be ~5.68 mM, got {mol*1000:.2f} mM"
    assert abs(g_per_L_to_mol_L(0.0)) < 1e-15
    # Linear scaling
    assert abs(g_per_L_to_mol_L(5.0) - 5 * mol) < 1e-10


# ── Test 11: Oxidation rate scaling checks ───────────────────────────────────


def test_fe2_oxidation_rate_scaling():
    """Individual rate function: check temperature and pH scaling."""
    r_25 = fe2_oxidation_rate(1.0, 2.5e-4, 2.0, 25.0, 1e-4, 50000.0)
    r_60 = fe2_oxidation_rate(1.0, 2.5e-4, 2.0, 60.0, 1e-4, 50000.0)
    r_ph3 = fe2_oxidation_rate(1.0, 2.5e-4, 3.0, 25.0, 1e-4, 50000.0)

    # Temperature: ~3× faster at 60°C vs 25°C for Ea=50 kJ/mol
    assert r_60 > r_25, "Rate should increase with temperature"
    assert r_60 / r_25 > 2.0, f"T scaling too weak: {r_60/r_25:.1f}×"

    # pH: 0→[OH⁻]×10 at pH 3 vs pH 2 → rate ×100
    ratio_ph = r_ph3 / r_25
    assert abs(ratio_ph - 100.0) < 1.0, f"pH scaling should be 100×, got {ratio_ph:.1f}×"
