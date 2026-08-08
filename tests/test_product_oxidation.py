"""Tests for product_oxidation — post-harvest O budget & Semenov (V6 §1.2)."""

import pytest

from models.anchors import get_anchor
from models.electrochemistry import RHO_FE
from models.product_oxidation import (
    SCREENING_FLAG,
    BedState,
    column_powder,
    design_passivation,
    drum_flake,
    from_architecture,
    model_scope,
    mol_o_per_nm_m2,
    o_wt_pct_from_film,
    oxidation_rate_mol_m2_s,
    passive_film_after_air_nm,
    passivated_briquette,
    postharvest_o_pickup_wt_pct,
    pyrophoricity_class,
    semenov_critical_T,
    self_heat_transient,
    specific_surface_area_m2_kg,
    ProductState,
)


# ── geometry: specific surface bookkeeping is exact ────────────────────

def test_specific_surface_closures():
    powder = specific_surface_area_m2_kg(column_powder())
    d50 = get_anchor("POWDER_D50_UM").value
    rough = get_anchor("POWDER_ROUGHNESS_FACTOR").value
    assert powder["s_area_m2_kg"] == pytest.approx(
        6.0 / (RHO_FE * d50 * 1.0e-6) * rough)

    flake = specific_surface_area_m2_kg(drum_flake())
    t_um = get_anchor("PRODUCT_FOIL_THICKNESS_UM").value
    assert flake["s_area_m2_kg"] == pytest.approx(
        2.0 / (RHO_FE * t_um * 1.0e-6))
    # denser forms carry less area — the whole safety story in one ratio
    assert flake["s_area_m2_kg"] < powder["s_area_m2_kg"]


def test_architecture_mapping_is_the_v6_feed():
    assert from_architecture("rotating_cylinder").kind == "powder"
    assert from_architecture("drum_and_strip").kind == "flake"
    assert from_architecture("plate_and_frame").kind == "foil"
    with pytest.raises(ValueError):
        from_architecture("fluidized_bed")
    with pytest.raises(ValueError):
        ProductState("ingot")


# ── film & rate laws ───────────────────────────────────────────────────

def test_film_to_oxygen_stoichiometry_is_exact():
    per_nm = mol_o_per_nm_m2()
    expected = (get_anchor("OXIDE_DENSITY_KG_M3").value * 1.0e-9
                * get_anchor("OXIDE_O_MASS_FRAC").value / 0.016)
    assert per_nm == pytest.approx(expected)
    o = o_wt_pct_from_film(3.0, 10.0)
    assert o == pytest.approx(3.0 * per_nm * 0.016 * 10.0 * 100.0)


def test_rate_law_is_inverse_film_linear_po2_arrhenius():
    r3 = oxidation_rate_mol_m2_s(60.0, 3.0)
    assert oxidation_rate_mol_m2_s(60.0, 1.0) == pytest.approx(3.0 * r3)
    assert oxidation_rate_mol_m2_s(60.0, 3.0, pO2_frac=0.105) == pytest.approx(
        0.5 * r3)
    assert oxidation_rate_mol_m2_s(80.0, 3.0) > r3
    with pytest.raises(ValueError):
        oxidation_rate_mol_m2_s(60.0, 0.0)


def test_film_branches_have_the_right_asymptotes():
    x_lim = get_anchor("PASSIV_FILM_LIM_NM").value
    seed = 1.0
    # short RT exposure: log branch active, approaches x_lim
    early = passive_film_after_air_nm(get_anchor("PASSIV_TAU_S").value,
                                      T_C=25.0, seed_film_nm=seed)
    assert seed < early < x_lim
    sat = passive_film_after_air_nm(100.0 * 3600.0, T_C=25.0,
                                    seed_film_nm=seed)
    assert sat == pytest.approx(x_lim, rel=1e-6)
    # hot, long: parabolic branch outruns the log ceiling
    hot = passive_film_after_air_nm(4.0 * 3600.0, T_C=150.0,
                                    seed_film_nm=seed)
    assert hot > x_lim


# ── Semenov thermal-runaway criterion ─────────────────────────────────

def test_tcrit_ordering_is_area_driven():
    coarse = semenov_critical_T(ProductState("powder", d50_um=200.0))
    default = semenov_critical_T(column_powder())
    fine = semenov_critical_T(ProductState("powder", d50_um=20.0))
    assert (coarse["t_crit_ambient_C"] > default["t_crit_ambient_C"]
            > fine["t_crit_ambient_C"])
    # default passivated powder hot-dries sub-critical, but only just —
    # the "slim margin" PM phenomenology the rate anchor was built from
    assert default["dryer_subcritical"]
    assert 0.0 < default["dryer_margin_K"] < 30.0
    # the fault case: fine powder in a hot dryer is super-critical
    assert not fine["dryer_subcritical"]


def test_tcrit_responds_to_loss_channel_and_film():
    base = semenov_critical_T(column_powder())
    lossy = semenov_critical_T(column_powder(), bed=BedState(h_W_m2K=60.0))
    assert lossy["t_crit_ambient_C"] > base["t_crit_ambient_C"]
    thick_film = semenov_critical_T(column_powder(), film_nm=6.0)
    assert thick_film["t_crit_ambient_C"] > base["t_crit_ambient_C"]
    assert base["loss_coeff_W_kgK"] == pytest.approx(
        get_anchor("DRYER_H_W_M2K").value
        / get_anchor("TRAY_BED_DEPTH_M").value
        / get_anchor("POWDER_BULK_DENSITY_KG_M3").value)


def test_self_heat_transient_stable_vs_runaway():
    stable = self_heat_transient(drum_flake())
    assert stable["runaway"] is False
    assert stable["t_peak_C"] < stable["t_amb_C"] + 50.0
    fault = self_heat_transient(ProductState("powder", d50_um=20.0),
                                film_nm=1.0)
    assert fault["runaway"] is True
    assert fault["runaway_after_s"] is not None
    assert fault["t_peak_C"] == pytest.approx(fault["t_amb_C"] + 50.0)


# ── pyrophoricity class + the product spec ─────────────────────────────

def test_pyrophoricity_classification():
    boundary = get_anchor("COMBUSTIBLE_DUST_D50_UM").value
    fine = pyrophoricity_class(ProductState("powder", d50_um=boundary - 10.0))
    assert fine["class"].startswith("combustible-dust")
    coarse = pyrophoricity_class(ProductState("powder", d50_um=boundary + 60.0))
    assert coarse["class"].startswith("coarse")
    web = pyrophoricity_class(drum_flake())
    assert "web" in web["class"]


def test_passivation_design_makes_the_buyable_article():
    proto = design_passivation(passivated_briquette())
    assert proto.spec_ok
    assert proto.verdict == "spec-qualified"
    assert proto.achieved_film_nm >= proto.target_film_nm
    assert proto.o_pickup_wt_pct <= get_anchor("PRODUCT_PASSIV_O_MAX_WT_PCT").value
    assert proto.semenov["dryer_subcritical"]
    assert any("protocol" in r for r in proto.reasons)


# ── live melt feed + ladder gate ───────────────────────────────────────

def test_postharvest_pickup_band_and_form_ordering():
    briq = postharvest_o_pickup_wt_pct("briquette")
    flake = postharvest_o_pickup_wt_pct("flake")
    foil = postharvest_o_pickup_wt_pct("foil")
    assert 0.0 < briq < get_anchor("POSTHARVEST_O_PICKUP_WT_PCT").value
    assert 0.0 < flake < briq and 0.0 < foil <= briq
    # the passivated article is a ppm-to-sub-0.1 wt% matter
    assert briq < 0.15


def test_melt_balance_oxygen_leg_is_live():
    from models.melt_balance import ChargeState, evaluate_charge
    from models.oxygen_in_iron import OxygenInIronModel

    live_ppm = float(OxygenInIronModel().predict()["o_ppm"])
    v = evaluate_charge()  # default briquette
    assert v.o_wt_pct == pytest.approx(
        live_ppm / 1.0e4 + postharvest_o_pickup_wt_pct("briquette"), rel=1e-9)
    assert v.verdict in ("qualified", "conditional", "fails")
    # product form flows through
    v_foil = evaluate_charge(ChargeState(product_form="foil"))
    assert v_foil.o_wt_pct < v.o_wt_pct


def test_ladder_gate_flipped_to_modelled():
    from models.product_ladder import gate_status

    status = gate_status("g_product_ox")
    assert status["exists"] is True
    assert status["state"] == "modelled (L1)"
    assert status["flag"] == SCREENING_FLAG


def test_model_scope_and_anchor_registry():
    scope = model_scope()
    assert scope["screening_flag"] == SCREENING_FLAG
    assert any("melt_balance" in s for s in scope["live_consumers"])
    assert any("SPECULATIVE" in s
               for s in scope["screening_proxies_anchored"])
    for key in (
        "PASSIV_FILM_LIM_NM", "PASSIV_TAU_S", "OXIDE_O_MASS_FRAC",
        "OXIDE_DENSITY_KG_M3", "POWDER_D50_UM", "POWDER_ROUGHNESS_FACTOR",
        "OX_RATE_REF_MOL_M2_S", "OX_EA_KJ_MOL", "OX_HEAT_KJ_MOL_O",
        "DRYER_H_W_M2K", "TRAY_BED_DEPTH_M", "POWDER_BULK_DENSITY_KG_M3",
        "PASSIV_PO2_FRAC", "PASSIV_PROTOCOL_T_C", "DRYER_AIR_T_C",
        "PRODUCT_PASSIV_O_MAX_WT_PCT", "STORAGE_HOURS",
        "COMBUSTIBLE_DUST_D50_UM",
    ):
        a = get_anchor(key)
        assert a.ref and a.notes
