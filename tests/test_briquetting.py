"""Tests for briquetting — densification & the shippable-product spec (V6 §1.4)."""

import math
import sys

import pytest

from models.anchors import get_anchor
from models.briquetting import (
    SCREENING_FLAG,
    BriquettingLine,
    E_FE_GPA,
    FeedForm,
    HeckelLaw,
    as_deposited_yield_MPa,
    bridging_min_outlet_m,
    cold_crush_strength_N,
    column_powder,
    default_shipped_spec,
    drum_flake,
    ejection_fraction_pct,
    evaluate_line,
    fines_fraction_wt_pct,
    from_architecture,
    green_strength_MPa,
    model_scope,
    press_work_kWh_per_t,
    rathole_critical_outlet_m,
    residual_o_wt_pct,
    shipped_fines_fraction,
    shippable_spec_block,
    springback_pct,
)
from models.electrochemistry import RHO_FE


def _aval(key: str) -> float:
    return float(get_anchor(key).value)


# ── feed form & Heckel law closures ─────────────────────────────────────

def test_feed_form_and_architecture_mapping():
    assert from_architecture("rotating_cylinder").kind == "powder"
    assert from_architecture("drum_and_strip").kind == "flake"
    # foil web is born dense — no densification line (intentional raise)
    with pytest.raises(ValueError):
        from_architecture("plate_and_frame")
    with pytest.raises(ValueError):
        FeedForm("foil")


def test_heckel_a_is_pinned_to_tap_density():
    law = HeckelLaw.from_yield(400.0)
    d_tap = _aval("TAP_DENSITY_POWDER_REL")
    assert law.a == pytest.approx(math.log(1.0 / (1.0 - d_tap)))
    # exact rearrangement closure: P = 0 recovers the tapped fill state
    assert law.relative_density(0.0) == pytest.approx(d_tap)
    assert law.pressure_MPa_for(d_tap) == pytest.approx(0.0)
    # round-trip both ways
    for p in (1.0, 200.0, 550.0):
        d = law.relative_density(p)
        assert law.pressure_MPa_for(d) == pytest.approx(p, rel=1e-9)


def test_heckel_yield_pressure_is_3_sigma_y_over_friable():
    law = HeckelLaw.from_yield(400.0, friable_factor=1.0)
    assert law.heckel_yield_pressure_MPa() == pytest.approx(3.0 * 400.0)
    law2 = HeckelLaw.from_yield(400.0)  # anchored friable factor
    assert law2.heckel_yield_pressure_MPa() == pytest.approx(
        3.0 * 400.0 / _aval("HECKEL_FRIABLE_FACTOR"))


def test_density_ordering_is_physical():
    p = 400.0
    soft = HeckelLaw.from_yield(200.0)
    hard = HeckelLaw.from_yield(800.0)
    assert hard.relative_density(p) < soft.relative_density(p) < 1.0
    # friable factor raises density at fixed P — the V6 inverted design goal
    fri = HeckelLaw.from_yield(400.0, friable_factor=2.0)
    stiff = HeckelLaw.from_yield(400.0, friable_factor=1.0)
    assert fri.relative_density(p) > stiff.relative_density(p)
    # hot pressing (softened σ_y) dominates the same press force
    hot = HeckelLaw.from_yield(400.0, hot=True)
    cold = HeckelLaw.from_yield(400.0)
    assert hot.relative_density(p) > cold.relative_density(p)
    # density is bounded below 1 for any finite pressure
    assert cold.relative_density(5000.0) < 1.0


# ── post-press state ────────────────────────────────────────────────────

def test_green_strength_law_exact_and_sinter_multiplier():
    d = 0.75
    g = green_strength_MPa(d)
    assert g == pytest.approx(
        _aval("GREEN_STRENGTH_PRE_MPA") * math.exp(_aval("GREEN_STRENGTH_B") * d))
    assert green_strength_MPa(d, sintered=True) == pytest.approx(
        g * _aval("SINTER_STRENGTH_FACTOR"))
    assert green_strength_MPa(0.9) > g > green_strength_MPa(0.5)
    with pytest.raises(ValueError):
        green_strength_MPa(0.0)


def test_springback_closure_and_scaling():
    d, p = 0.8, 550.0
    e_eff = E_FE_GPA * d ** _aval("SPRINGBACK_MODULUS_EXP")
    assert springback_pct(p, d) == pytest.approx(p / (e_eff * 1000.0) * 100.0)
    # higher pressure → more stored elastic strain; denser compact is
    # stiffer → smaller strain at fixed pressure
    assert springback_pct(700.0, d) > springback_pct(p, d)
    assert springback_pct(p, 0.95) < springback_pct(p, 0.7)
    assert ejection_fraction_pct() == pytest.approx(
        _aval("DIE_WALL_MU") * _aval("RADIAL_STRESS_FRACTION") * 100.0)


def test_press_work_bookkeeping():
    law = HeckelLaw.from_yield(400.0)
    d0 = law.tap_density_rel
    with pytest.raises(ValueError):
        press_work_kWh_per_t(law, d0)  # no compaction, no work
    d1, d2 = 0.75, 0.95
    w1 = press_work_kWh_per_t(law, d1)
    w2 = press_work_kWh_per_t(law, d2)
    assert w1["work_ideal_J_kg"] > 0.0
    assert w2["work_ideal_J_kg"] > w1["work_ideal_J_kg"]
    # units: J/kg → kWh/t
    assert w1["energy_ideal_kWh_per_t"] == pytest.approx(
        w1["work_ideal_J_kg"] * 1000.0 / 3.6e6)
    assert w1["energy_delivered_kWh_per_t"] == pytest.approx(
        w1["energy_ideal_kWh_per_t"] / _aval("PRESS_HYDRAULIC_ETA"))
    # trapezoid result converges (same integral at 4× resolution)
    w1_fine = press_work_kWh_per_t(law, d1, n_steps=32768)
    assert w1_fine["work_ideal_J_kg"] == pytest.approx(
        w1["work_ideal_J_kg"], rel=1e-4)
    # hot pressing is *less* work to a higher density — the HBI lesson
    hot = HeckelLaw.from_yield(400.0, hot=True)
    w_hot = press_work_kWh_per_t(hot, law.relative_density(550.0))
    assert w_hot["energy_ideal_kWh_per_t"] > 0.0


def test_fines_ratio_form_anchored_at_reference_strength():
    ref_s = _aval("GREEN_STRENGTH_REF_MPA")
    assert fines_fraction_wt_pct(ref_s) == pytest.approx(_aval("FINES_REF_PCT"))
    assert fines_fraction_wt_pct(2.0 * ref_s) < _aval("FINES_REF_PCT")
    assert fines_fraction_wt_pct(0.5 * ref_s) > _aval("FINES_REF_PCT")
    with pytest.raises(ValueError):
        fines_fraction_wt_pct(0.0)


def test_cold_crush_is_strength_times_face():
    s = 20.0
    size = _aval("BRIQUETTE_SIZE_MM")
    assert cold_crush_strength_N(s) == pytest.approx(s * size ** 2)


# ── densification order & residual oxygen ───────────────────────────────

def test_residual_o_branches_carry_their_sources():
    from models.product_oxidation import postharvest_o_pickup_wt_pct

    pass_first = residual_o_wt_pct("passivate_first", column_powder())
    assert pass_first["residual_o_wt_pct"] == pytest.approx(
        postharvest_o_pickup_wt_pct("powder"), rel=1e-9)
    assert "live" in pass_first["source"]
    sint_first = residual_o_wt_pct("sinter_first", column_powder())
    assert sint_first["residual_o_wt_pct"] == pytest.approx(
        _aval("SINTER_RESIDUAL_O_WT_PCT"))
    # flake's thinner web carries less passivation O than the powder briquette
    flake_o = residual_o_wt_pct("passivate_first", drum_flake())
    assert flake_o["residual_o_wt_pct"] < pass_first["residual_o_wt_pct"]
    with pytest.raises(ValueError):
        residual_o_wt_pct("anneal_first")


# ── line verdicts: the module's headline contrasts ──────────────────────

def test_hot_press_clears_where_cold_press_is_conditional():
    hot = evaluate_line(BriquettingLine(form=column_powder(), hot=True))
    cold = evaluate_line(BriquettingLine(form=column_powder()))
    assert hot.verdict == "shippable-spec"
    assert cold.verdict == "conditional"
    assert cold.relative_density < _aval("DENSITY_REL_SPEC") <= hot.relative_density
    assert hot.cold_crush_strength_N > cold.cold_crush_strength_N
    assert hot.fines_wt_pct < cold.fines_wt_pct
    assert any("density" in r for r in cold.reasons)
    # softening factor is what buys the margin at the same press force
    assert hot.sigma_y_effective_MPa == pytest.approx(
        cold.sigma_y_effective_MPa * _aval("HOT_PRESS_SIGMA_SOFTEN"))


def test_sinter_first_fixes_strength_not_density():
    sp = evaluate_line(BriquettingLine(form=column_powder(),
                                       sinter_first=True))
    cold = evaluate_line(BriquettingLine(form=column_powder()))
    assert sp.relative_density == pytest.approx(cold.relative_density)
    assert sp.green_or_sintered_strength_MPa == pytest.approx(
        cold.green_or_sintered_strength_MPa * _aval("SINTER_STRENGTH_FACTOR"))
    assert sp.fines_wt_pct <= _aval("FINES_SPEC_PCT")  # strength fixed
    assert sp.sinter_kWh_per_t == pytest.approx(_aval("SINTER_KWH_PER_T"))
    assert sp.residual_o_wt_pct == pytest.approx(_aval("SINTER_RESIDUAL_O_WT_PCT"))


def test_flake_is_handicapped_by_fill_state():
    p_f = evaluate_line(BriquettingLine(form=drum_flake()))
    p_p = evaluate_line(BriquettingLine(form=column_powder()))
    assert p_f.relative_density < p_p.relative_density
    assert p_f.fines_wt_pct > p_p.fines_wt_pct


def test_sigma_y_override_rederives_the_whole_line():
    soft = evaluate_line(BriquettingLine(sigma_y_MPa=150.0))
    hard = evaluate_line(BriquettingLine(sigma_y_MPa=900.0))
    assert soft.relative_density > hard.relative_density
    assert soft.verdict in ("shippable-spec", "conditional")
    assert BriquettingLine().resolved_press_MPa() == pytest.approx(
        _aval("PRESS_DESIGN_MPA"))


# ── live feeds ──────────────────────────────────────────────────────────

def test_yield_feed_is_live_with_anchor_fallback(monkeypatch):
    feed = as_deposited_yield_MPa()
    assert "live" in feed["source"]
    assert feed["sigma_y_MPa"] > 0.0
    monkeypatch.setitem(sys.modules, "models.mechanical_properties", None)
    fallback = as_deposited_yield_MPa()
    assert "fallback" in fallback["source"]
    assert fallback["sigma_y_MPa"] == pytest.approx(
        _aval("AS_DEPOSITED_SIGMA_Y_MPA"))


def test_residual_o_falls_back_when_product_oxidation_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "models.product_oxidation", None)
    o = residual_o_wt_pct("passivate_first", column_powder())
    assert "fallback" in o["source"]
    assert o["residual_o_wt_pct"] == pytest.approx(
        _aval("POSTHARVEST_O_PICKUP_WT_PCT"))


# ── hopper / flow screens ───────────────────────────────────────────────

def test_rathole_screen_bookkeeping():
    rh = rathole_critical_outlet_m(column_powder())
    rho_b = _aval("TAP_DENSITY_POWDER_REL") * RHO_FE
    assert rh["brathole_m"] == pytest.approx(
        _aval("JENIKE_H_THETA")
        * (_aval("UNCONFINED_YIELD_PA") + _aval("MAGNETIC_COHESION_PA"))
        / (rho_b * 9.80665))
    # flake packs lighter → larger critical outlet
    assert (rathole_critical_outlet_m(drum_flake())["brathole_m"]
            > rh["brathole_m"])
    # briquette flows on the bridging rule, not the rathole screen
    br = bridging_min_outlet_m()
    assert br["min_outlet_m"] == pytest.approx(
        _aval("BRIDGING_RULE_MULTIPLE") * _aval("BRIQUETTE_SIZE_MM") / 1000.0)


# ── exports & integration ───────────────────────────────────────────────

def test_shipped_fines_fraction_and_export_block():
    spec = default_shipped_spec()
    frac = shipped_fines_fraction()
    assert frac == pytest.approx(spec.fines_wt_pct / 100.0, rel=1e-12)
    assert 0.0 < frac < 0.05
    block = shippable_spec_block()
    assert block["verdict"] == spec.verdict
    assert block["flag"] == SCREENING_FLAG
    assert "feedstock_logistics.py" in block["consumers"]


def test_melt_balance_consumes_shipped_fines_live():
    from models.melt_balance import evaluate_charge

    v = evaluate_charge()  # ChargeState() defaults resolve live
    assert v.fines_fraction == pytest.approx(shipped_fines_fraction(),
                                             rel=1e-9)


def test_model_scope_and_gate_probe():
    scope = model_scope()
    for key in ("screening_flag", "live_derivations", "live_consumers",
                "exact", "screening_proxies_anchored", "out_of_scope"):
        assert key in scope
    assert scope["screening_flag"] == SCREENING_FLAG

    from models.product_ladder import gate_status

    g = gate_status("g_briquet")
    assert g["exists"] is True
    assert g["state"] == "modelled (L1)"
