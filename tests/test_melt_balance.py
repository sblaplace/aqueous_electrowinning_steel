"""Tests for melt_balance — the melt-shop remelt verdict (V6 §1.5)."""

import pytest

from models.anchors import get_anchor
from models.melt_balance import (
    CO_PER_O,
    C_PER_O,
    ChargeState,
    compare_baselines,
    comparison_table,
    evaluate_charge,
    model_scope,
    route_params,
    to_dict_full,
)


# ── stoichiometry and mass closure ──────────────────────────────────────

def test_stoichiometry_is_exact():
    st = ChargeState(o_wt_pct=1.0, s_wt_pct=0.0, fines_fraction=0.0)
    v = evaluate_charge(st, "eaf")
    r = get_anchor("EAF_OXIDE_RECOVERY_FRAC").value
    o_kg = v.o_kg_t
    assert o_kg == pytest.approx(10.0)           # 1.0 wt% of a tonne
    assert v.carbon_required_kg_t == pytest.approx(r * o_kg * C_PER_O)
    assert v.co_boil_kg_t == pytest.approx(r * o_kg * CO_PER_O)
    # CO conserves C + O
    assert v.co_boil_kg_t == pytest.approx(v.carbon_required_kg_t + r * o_kg)


def test_iron_ledger_closes():
    st = ChargeState(o_wt_pct=1.2, s_wt_pct=0.02, fines_fraction=0.05)
    feo_per_fe = (55.845 + 16.00) / 55.845
    for route in ("eaf", "induction"):
        v = evaluate_charge(st, route)
        # closure: metal + slag Fe + dust Fe = iron charged
        assert (v.fe_to_metal_kg_t + v.fe_to_slag_kg_t + v.fe_to_dust_kg_t
                ) == pytest.approx(v.fe_in_charge_kg_t, rel=1e-9)
        # slag carries the unrecovered FeO, lime, and gangue
        assert v.slag_total_kg_t == pytest.approx(
            v.fe_to_slag_kg_t * feo_per_fe + v.lime_kg_t + v.gangue_kg_t,
            rel=1e-9,
        )


def test_zero_oxygen_means_no_boil_and_metal_yield_above_gate():
    st = ChargeState(o_wt_pct=0.0, s_wt_pct=0.0, fines_fraction=0.0, c_h_ppm=0.0)
    v = evaluate_charge(st, "eaf")
    assert v.co_boil_kg_t == 0.0
    assert v.carbon_required_kg_t == 0.0
    assert v.fe_yield_pct == pytest.approx(100.0)
    assert v.verdict == "qualified"


# ── route contrast ───────────────────────────────────────────────────────

def test_route_contrast_is_physical():
    """EAF wins on oxide recovery (slag); induction wins on fines (dust)."""
    st = ChargeState(o_wt_pct=1.0, fines_fraction=0.05)
    e = evaluate_charge(st, "eaf")
    i = evaluate_charge(st, "induction")
    assert i.slag_total_kg_t >= e.slag_total_kg_t
    assert i.fe_to_slag_kg_t >= e.fe_to_slag_kg_t
    assert i.dust_kg_t <= e.dust_kg_t  # gentle off-gas, less carryover
    assert any("induction" in reason for reason in i.reasons)
    # with no fines, EAF's better oxide recovery must dominate
    st_nf = ChargeState(o_wt_pct=1.0, fines_fraction=0.0)
    assert (evaluate_charge(st_nf, "eaf").fe_yield_pct
            >= evaluate_charge(st_nf, "induction").fe_yield_pct)
    # yield comparison otherwise is charge-dependent — the model must
    # report it, not hide it behind a route ordering
    assert i.fe_yield_pct != pytest.approx(0.0)


def test_route_error_on_unknown():
    with pytest.raises(ValueError):
        route_params("open_hearth")


# ── monotonicities ───────────────────────────────────────────────────────

def test_higher_oxygen_lowers_yield_and_raises_boil():
    lo = evaluate_charge(ChargeState(o_wt_pct=0.3), "eaf")
    hi = evaluate_charge(ChargeState(o_wt_pct=2.5), "eaf")
    assert hi.fe_yield_pct < lo.fe_yield_pct
    assert hi.co_boil_kg_t > lo.co_boil_kg_t
    assert hi.thermal_penalty_kWh_t > lo.thermal_penalty_kWh_t
    assert hi.slag_total_kg_t > lo.slag_total_kg_t


def test_higher_fines_means_more_dust_and_lower_yield():
    lo = evaluate_charge(ChargeState(fines_fraction=0.01), "eaf")
    hi = evaluate_charge(ChargeState(fines_fraction=0.10), "eaf")
    assert hi.dust_kg_t > lo.dust_kg_t
    assert hi.fe_yield_pct < lo.fe_yield_pct


def test_dirty_charge_can_fail_the_yield_gate():
    dirty = ChargeState(o_wt_pct=3.5, s_wt_pct=0.03, fines_fraction=0.15)
    v = evaluate_charge(dirty, "induction")
    assert v.verdict in ("conditional", "fails")
    assert any("yield" in r for r in v.reasons)


# ── live derivation links ────────────────────────────────────────────────

def test_default_charge_o_comes_live_from_oxygen_engine():
    from models.oxygen_in_iron import OxygenInIronModel

    live_ppm = float(OxygenInIronModel().predict()["o_ppm"])
    expected = live_ppm / 1.0e4 + get_anchor("POSTHARVEST_O_PICKUP_WT_PCT").value
    v = evaluate_charge()  # default state resolves live
    assert v.o_wt_pct == pytest.approx(expected, rel=1e-9)


def test_h_verdict_is_live_melt_hydrogen_call():
    v = evaluate_charge(ChargeState(c_h_ppm=240.0), "eaf")
    assert "flake_risk_index" in v.h_budget
    assert v.h_budget["needs_bake_or_degas"] in (True, False)
    # any ridge in reasons mentions the M-name of the upstream model
    assert any("melt_hydrogen" in r for r in v.reasons) or not (
        v.h_budget["needs_bake_or_degas"]
    )


def test_default_verdict_shape_and_reasons():
    v = evaluate_charge()
    assert v.verdict in ("qualified", "conditional", "fails")
    assert isinstance(v.reasons, list)
    # defaults: well-rinsed briquetted flake at reference operating point —
    # yield must clear the scrap band on the EAF route
    assert v.fe_yield_pct >= get_anchor("SCRAP_YIELD_PCT").value - 2.0


# ── baselines & reports ─────────────────────────────────────────────────

def test_baselines_present_and_consistent():
    b = compare_baselines()
    assert set(b) == {"electrowon", "no1_scrap", "dri_hbi"}
    assert b["electrowon"]["fe_yield_pct"] > 0
    assert b["no1_scrap"]["fe_yield_pct"] == pytest.approx(
        get_anchor("SCRAP_YIELD_PCT").value
    )
    # DRI runs the same engine with its own O anchor
    assert b["dri_hbi"]["o_wt_pct"] == pytest.approx(
        get_anchor("DRI_O_WT_PCT").value
    )


def test_full_dict_and_table_render():
    payload = to_dict_full()
    assert payload["screening_flag"].startswith("unvalidated")
    assert set(payload) >= {"charge_state", "eaf", "induction", "baselines"}
    tbl = comparison_table()
    for token in ("eaf", "induction", "yield", "verdict"):
        assert token in tbl


def test_scope_declaration_honest():
    scope = model_scope()
    assert scope["screening_flag"].startswith("unvalidated")
    assert "exact" in scope and "screening_proxies_anchored" in scope
