"""Tests for rinse_carryover — bath liquor becomes melt-shop sulfur (V6 §1.3)."""

import math

import pytest

from models.anchors import get_anchor
from models.electrochemistry import RHO_FE
from models.rinse_carryover import (
    MELT_SCREENING_S_WT_PCT,
    M_B,
    M_NA,
    M_S,
    SCREENING_FLAG,
    BathLiquor,
    ProductForm,
    RinseTrain,
    cascade_dilution,
    carryover_liquor_kg_per_t,
    default_charge_s_wt_pct,
    divided_cell_liquor,
    evaluate_rinse,
    foil_web,
    model_scope,
    powder_column,
    stage1_concentration_ratio,
    stages_to_meet,
    withdrawal_film_um,
)


# ── counter-current cascade: exact mass balances ───────────────────────

def test_cascade_dilution_is_the_exact_counterflow_series():
    assert cascade_dilution(0, 5.0) == 1.0                 # unrinsed
    assert cascade_dilution(1, 5.0) == pytest.approx(1.0 / 6.0)
    for n in (2, 3, 4):
        r = 5.0
        series = sum(r ** k for k in range(n + 1))
        assert cascade_dilution(n, r) == pytest.approx(1.0 / series, rel=1e-12)
        closed = (r - 1.0) / (r ** (n + 1) - 1.0)
        assert cascade_dilution(n, r) == pytest.approx(closed, rel=1e-12)


def test_cascade_handles_r_equal_one_and_monotone_in_r():
    assert cascade_dilution(3, 1.0) == pytest.approx(1.0 / 4.0)
    assert (cascade_dilution(3, 2.0)
            > cascade_dilution(3, 5.0)
            > cascade_dilution(3, 20.0))


def test_stage1_strength_is_between_product_and_bath():
    c1 = stage1_concentration_ratio(3, 5.0)
    assert cascade_dilution(3, 5.0) < c1 < 1.0
    # n=1 counterflow degenerates to a single mixed tank: c1 = c0/(1+r)
    assert stage1_concentration_ratio(1, 5.0) == pytest.approx(1.0 / 6.0)
    assert stage1_concentration_ratio(0, 5.0) == 1.0


def test_counterflow_uses_one_water_stream_for_all_stages():
    """Water demand is r × drag-out regardless of stage count — the point."""
    for n in (1, 2, 3):
        rep = evaluate_rinse(product=powder_column(),
                             rinse=RinseTrain(n_stages=n, r_ratio=5.0))
        drag_out = rep.m_liquor_kg_t / 1.2    # kg → L at ρ=1.2 kg/L
        assert rep.water_m3_t == pytest.approx(5.0 * drag_out / 1000.0)


# ── liquor species stoichiometry ───────────────────────────────────────

def test_species_concentrations_are_stoichiometric():
    b = BathLiquor(fe2_M=1.0, na2so4_M=0.5, h3bo3_M=0.4, pH=2.0).resolved()
    assert b.sulfate_M() == pytest.approx(1.0 + 0.5 + 0.01)
    sp = b.species_g_per_L()
    assert sp["S"] == pytest.approx(b.sulfate_M() * M_S)
    assert sp["Na"] == pytest.approx(2.0 * 0.5 * M_NA)
    assert sp["B"] == pytest.approx(0.4 * M_B)
    # sodium channel is zero for the BATH_SPEC liquor (no Na₂SO₄)
    assert BathLiquor(fe2_M=1.0, na2so4_M=0.0).resolved().species_g_per_L()["Na"] == 0.0


# ── carryover mechanisms ───────────────────────────────────────────────

def test_landau_levich_film_law_is_self_consistent():
    film = withdrawal_film_um(0.25)
    l_c = math.sqrt(film["surface_tension_N_m"] / (1200.0 * 9.80665))
    h_expected = 0.94 * l_c * film["capillary_number"] ** (2.0 / 3.0)
    assert film["film_um"] == pytest.approx(h_expected * 1.0e6, rel=1e-6)
    assert film["capillary_length_mm"] == pytest.approx(l_c * 1.0e3)
    # faster withdrawal → thicker film, sub-linearly (Ca^(2/3))
    fast = withdrawal_film_um(1.0)
    assert 1.5 < fast["film_um"] / film["film_um"] < 4.0


def test_powder_cake_liquor_is_exact():
    f = get_anchor("POWDER_CAKE_LIQUOR_FRAC").value
    carry = carryover_liquor_kg_per_t(powder_column())
    assert carry["m_liquor_kg_t"] == pytest.approx(1000.0 * f / (1.0 - f))


def test_web_liquor_closes_on_film_area_and_retention():
    carry = carryover_liquor_kg_per_t(foil_web())
    t_um = carry["foil_thickness_um"]
    area = (1000.0 / RHO_FE) / (t_um * 1.0e-6)
    assert carry["web_area_m2_per_t"] == pytest.approx(area)
    expected = (area * 2.0 * carry["landau_levich_film_um"] * 1.0e-6
                * carry["liquor_density_kg_m3"] * carry["drain_retention"])
    assert carry["m_liquor_kg_t"] == pytest.approx(expected, rel=1e-9)


# ── evaluations: the steel-grade plumbing decision ─────────────────────

def test_unrinsed_product_is_unmeltable():
    # kilograms of S per tonne — the V6 §1.3 story
    powder = evaluate_rinse(product=powder_column(), rinse=RinseTrain(n_stages=0))
    web = evaluate_rinse(product=foil_web(), rinse=RinseTrain(n_stages=0))
    for rep in (powder, web):
        assert rep.dilution_factor == 1.0
        assert rep.charge_s_wt_pct > 0.10            # wt%: catastrophic
        assert rep.verdict == "fails"


def test_stage_counts_cross_the_grade_splits_in_order():
    web = evaluate_rinse(product=foil_web(), rinse=RinseTrain(n_stages=1))
    assert web.charge_s_wt_pct > web.deep_draw_split_wt_pct   # fails at n=1
    web2 = evaluate_rinse(product=foil_web(), rinse=RinseTrain(n_stages=2))
    assert web2.charge_s_wt_pct < web.charge_s_wt_pct
    web3 = evaluate_rinse(product=foil_web(), rinse=RinseTrain(n_stages=3))
    assert web3.charge_s_wt_pct <= MELT_SCREENING_S_WT_PCT
    # budgets are internally consistent
    assert web3.charge_s_wt_pct == pytest.approx(
        web3.budgets["S"]["g_per_t"] / 1.0e4)
    assert web3.residual_salt_g_t == pytest.approx(
        sum(v["g_per_t"] for v in web3.budgets.values()))


def test_stages_to_meet_is_minimal_and_correct():
    n = stages_to_meet(MELT_SCREENING_S_WT_PCT, product=foil_web())
    assert n is not None and n > 0
    meets = evaluate_rinse(product=foil_web(), rinse=RinseTrain(n_stages=n))
    misses = evaluate_rinse(product=foil_web(), rinse=RinseTrain(n_stages=n - 1))
    assert meets.charge_s_wt_pct <= MELT_SCREENING_S_WT_PCT
    assert misses.charge_s_wt_pct > MELT_SCREENING_S_WT_PCT


def test_conductivity_endpoint_is_dilution_scaled_and_gated():
    rep = evaluate_rinse(product=powder_column(), rinse=RinseTrain(n_stages=3))
    sigma = get_anchor("BATH_CONDUCTIVITY_MS_CM").value * 1000.0
    assert rep.final_rinse_conductivity_uS_cm == pytest.approx(
        sigma * rep.dilution_factor)
    assert rep.endpoint_ok == (
        rep.final_rinse_conductivity_uS_cm
        <= get_anchor("RINSE_ENDPOINT_US_CM").value)
    # at the anchored defaults the endpoint demands one more stage
    n4 = evaluate_rinse(product=powder_column(), rinse=RinseTrain(n_stages=4))
    assert n4.endpoint_ok
    assert n4.verdict == "rinse-qualified"


def test_tank_return_vs_effluent_flag():
    ret = evaluate_rinse(product=powder_column(), rinse=RinseTrain(n_stages=3))
    assert ret.tank_return_salt_kg_t > 0.0
    assert ret.effluent_salt_kg_t == 0.0
    # flipping the flag reroutes the identical salt load to the effluent
    dump = evaluate_rinse(product=powder_column(),
                          rinse=RinseTrain(n_stages=3,
                                           return_first_stage_to_tank=False))
    assert dump.effluent_salt_kg_t == pytest.approx(ret.tank_return_salt_kg_t)
    assert dump.tank_return_salt_kg_t == 0.0


def test_divided_cell_liquor_adds_the_sodium_channel():
    rep = evaluate_rinse(product=powder_column(), bath=divided_cell_liquor())
    assert rep.budgets["Na"]["g_per_t"] > 1.0
    base = evaluate_rinse(product=powder_column())
    assert rep.charge_s_wt_pct > base.charge_s_wt_pct     # more sulfate


def test_boron_channel_tracked_not_gated():
    rep = evaluate_rinse(product=foil_web())
    assert rep.budgets["B"]["g_per_t"] > 0.0
    heavy = evaluate_rinse(product=foil_web(), bath=BathLiquor(h3bo3_M=0.8))
    assert heavy.budgets["B"]["g_per_t"] == pytest.approx(
        2.0 * rep.budgets["B"]["g_per_t"])
    assert any("boron" in r for r in rep.reasons)


# ── live feeds: melt_balance + the ladder gate ─────────────────────────

def test_charge_s_feeds_melt_balance_live():
    from models.melt_balance import ChargeState, _resolved_state

    live = default_charge_s_wt_pct("briquette")
    assert live <= MELT_SCREENING_S_WT_PCT
    resolved = _resolved_state(ChargeState())
    assert resolved["s_wt_pct"] == pytest.approx(live)


def test_product_form_mapping():
    assert (default_charge_s_wt_pct("foil")
            != default_charge_s_wt_pct("briquette"))
    assert default_charge_s_wt_pct("briquette") == pytest.approx(
        evaluate_rinse(product=powder_column()).charge_s_wt_pct)


def test_melt_balance_snapshot_is_stable_with_live_s():
    """Live S must not overturn the melt verdict (S was never binding)."""
    from models.melt_balance import evaluate_charge

    v = evaluate_charge(route="eaf")
    assert v.verdict == "conditional"
    assert v.s_wt_pct == pytest.approx(default_charge_s_wt_pct("briquette"))


def test_ladder_gate_flipped_to_modelled():
    from models.product_ladder import gate_status

    status = gate_status("g_rinse")
    assert status["exists"] is True
    assert status["state"] == "modelled (L1)"
    assert status["flag"] == SCREENING_FLAG


def test_model_scope_and_anchor_registry():
    scope = model_scope()
    assert scope["screening_flag"] == SCREENING_FLAG
    assert any("melt_balance" in s for s in scope["live_consumers"])
    for key in (
        "BATH_SURFACE_TENSION_N_M", "DRAIN_RETENTION_FRAC",
        "POWDER_CAKE_LIQUOR_FRAC", "RINSE_RATIO", "RINSE_STAGES",
        "PRODUCT_FOIL_THICKNESS_UM", "WEB_SPEED_M_S",
        "BATH_CONDUCTIVITY_MS_CM", "RINSE_ENDPOINT_US_CM",
        "RINSE_BATH_FE2_M", "RINSE_BATH_NA2SO4_M", "RINSE_BATH_H3BO3_M",
    ):
        a = get_anchor(key)
        assert a.ref and a.notes


def test_validation():
    with pytest.raises(ValueError):
        ProductForm("ingot")
    with pytest.raises(ValueError):
        cascade_dilution(-1, 5.0)
