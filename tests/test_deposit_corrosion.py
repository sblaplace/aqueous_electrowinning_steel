"""Tests for deposit_corrosion — idle corrosion & ferric etch (V6 §1.1)."""

import pandas as pd
import pytest

from models.anchors import get_anchor
from models.deposit_corrosion import (
    IdleBathState,
    SCREENING_FLAG,
    corrosion_current,
    evaluate_idle,
    ferric_etch_flux,
    mass_loss_over_idle,
    model_scope,
    predicted_idle_terms,
)
from models.electrochemistry import FARADAY, M_FE, RHO_FE

JCORR_REF = get_anchor("FE_ACID_JCORR_REF_UA_CM2").value


# ── mixed-potential solver: pinning and monotone structure ─────────────

def test_reference_state_self_pinning():
    """At the anchored reference state the solver reproduces the anchor."""
    r = corrosion_current(pH=2.0, T_C=25.0,
                          o2_fraction_of_sat=0.0, theta_additive=0.0)
    assert r["j_corr_uA_cm2"] == pytest.approx(JCORR_REF, rel=1e-9)
    assert r["e_corr_vs_ref_mV"] == pytest.approx(0.0, abs=1e-6)
    assert r["flag"] == SCREENING_FLAG


def test_higher_pH_lowers_jcorr():
    js = [corrosion_current(pH=pH, T_C=25.0,
                            o2_fraction_of_sat=0.0,
                            theta_additive=0.0)["j_corr_uA_cm2"]
          for pH in (2.0, 3.0, 4.0)]
    assert js[0] > js[1] > js[2]
    # one pH decade cuts the HER-pinned j_corr by ~5–6× at b_a=40/b_c=120
    assert 3.0 < js[0] / js[1] < 10.0


def test_oxygen_and_temperature_and_additive_structure():
    base = corrosion_current(pH=2.0, T_C=25.0,
                             o2_fraction_of_sat=0.0, theta_additive=0.0)
    aerated = corrosion_current(pH=2.0, T_C=25.0,
                                o2_fraction_of_sat=1.0, theta_additive=0.0)
    hot = corrosion_current(pH=2.0, T_C=60.0,
                            o2_fraction_of_sat=0.0, theta_additive=0.0)
    inhibited = corrosion_current(pH=2.0, T_C=25.0,
                                  o2_fraction_of_sat=0.0, theta_additive=0.9)
    assert aerated["j_corr_uA_cm2"] > 4.0 * base["j_corr_uA_cm2"]
    assert aerated["j_o2_lim_uA_cm2"] > 0.0
    assert hot["j_corr_uA_cm2"] > base["j_corr_uA_cm2"]        # Arrhenius
    assert inhibited["j_corr_uA_cm2"] < base["j_corr_uA_cm2"]  # blocking
    # cathodic partition is consistent
    for r in (base, aerated):
        assert r["j_corr_uA_cm2"] == pytest.approx(
            r["j_her_uA_cm2"] + r["j_o2_lim_uA_cm2"], rel=1e-3)


# ── ferric etch law ────────────────────────────────────────────────────

def test_etch_is_half_order_in_ferric_and_zero_at_zero():
    k1 = ferric_etch_flux(0.05)
    k4 = ferric_etch_flux(0.20)
    assert k1 > 0.0
    assert k4 / k1 == pytest.approx(2.0, rel=1e-9)   # half-order law
    assert ferric_etch_flux(0.0) == 0.0


def test_etch_is_acid_catalysed_and_strongly_temperature_dependent():
    k_ref = ferric_etch_flux(0.05, pH=2.0, T_C=25.0)
    assert ferric_etch_flux(0.05, pH=1.0, T_C=25.0) > k_ref
    assert ferric_etch_flux(0.05, pH=3.0, T_C=25.0) < k_ref
    assert ferric_etch_flux(0.05, pH=2.0, T_C=60.0) > 3.0 * k_ref  # Ea ~ 50 kJ/mol
    # anchored reference state reproduces the anchor exactly
    assert k_ref == pytest.approx(get_anchor("FE3_ETCH_K_REF_MOL_M2_S").value)


# ── idle integration ───────────────────────────────────────────────────

def test_stirred_loss_is_linear_and_charge_exact():
    st = IdleBathState(a_Fe3_bulk_M=0.02, mixing="stirred", area_m2=2.0)
    l8 = mass_loss_over_idle(8.0 * 3600.0, state=st)
    l16 = mass_loss_over_idle(16.0 * 3600.0, state=st)
    assert l16["um_lost"] == pytest.approx(2.0 * l8["um_lost"], rel=1e-9)
    # area and charge bookkeeping
    assert l8["fe_mol"] == pytest.approx(l8["fe_dissolved_mol_m2"] * 2.0)
    assert l8["charge_equivalent_C"] == pytest.approx(2.0 * FARADAY * l8["fe_mol"])
    # µm conversion is the iron-density identity
    assert l8["um_lost"] == pytest.approx(
        l8["fe_dissolved_mol_m2"] * M_FE / RHO_FE * 1.0e6, rel=1e-9)


def test_stagnant_etch_is_supply_limited_and_sublinear():
    st = IdleBathState(a_Fe3_bulk_M=0.05, o2_fraction_of_sat=0.0,
                       theta_additive=0.95, T_C=25.0)
    stag16 = mass_loss_over_idle(16.0 * 3600.0, state=st)
    stag64 = mass_loss_over_idle(64.0 * 3600.0, state=st)
    # √t supply makes 4× time cost less than 4× loss
    assert stag64["um_lost"] < 4.0 * stag16["um_lost"]
    import dataclasses

    stir64 = mass_loss_over_idle(64.0 * 3600.0,
                                 state=dataclasses.replace(st, mixing="stirred"))
    assert stir64["um_from_etch"] > stag64["um_from_etch"]  # stagnant depletes


def test_thickness_cap():
    st = IdleBathState(a_Fe3_bulk_M=0.05, mixing="stirred", T_C=25.0)
    loss = mass_loss_over_idle(64.0 * 3600.0, state=st,
                               deposit_thickness_um=0.5)
    assert loss["capped_by_thickness"] is True
    assert loss["um_lost"] == pytest.approx(0.5, rel=1e-3)
    assert loss["t_to_consume_h"] < 64.0
    uncapped = mass_loss_over_idle(64.0 * 3600.0, state=st)
    assert uncapped["capped_by_thickness"] is False
    assert uncapped["um_lost"] > 0.5


def test_verdict_bands_are_reachable():
    negligible = evaluate_idle(
        t_idle_h=8.0,
        state=IdleBathState(theta_additive=0.99, o2_fraction_of_sat=0.0,
                            a_Fe3_bulk_M=0.0, T_C=25.0),
    )
    assert negligible.verdict == "negligible"
    night = evaluate_idle(t_idle_h=8.0)          # anchored defaults
    assert night.verdict == "ledger_term"
    weekend = evaluate_idle(t_idle_h=64.0)
    assert weekend.verdict == "material"
    assert weekend.loss["um_lost"] > night.loss["um_lost"]
    for v in (negligible, night, weekend):
        assert v.reasons


# ── run_record wiring (predicted, not measured, ledger terms) ──────────

def _manifest(idle_hours=8.0, area_cm2=100.0):
    return {
        "setup": {
            "idle": {"hours": idle_hours},
            "cathode": {"area_cm2": area_cm2},
        }
    }


def test_predicted_idle_terms_from_manifest():
    terms = predicted_idle_terms(_manifest(idle_hours=8.0, area_cm2=100.0))
    assert terms is not None
    assert terms["fe_mol"] > 0.0
    assert terms["charge_C"] == pytest.approx(2.0 * FARADAY * terms["fe_mol"])
    assert terms["screening_flag"] == SCREENING_FLAG
    # area conversion: 100 cm² = 1e-2 m² → 100× less loss than the 1 m² default
    big = predicted_idle_terms(_manifest(idle_hours=8.0, area_cm2=10000.0))
    assert big["fe_mol"] == pytest.approx(terms["fe_mol"] * 100.0, rel=1e-9)


def test_predicted_idle_terms_absent_without_declaration():
    assert predicted_idle_terms({}) is None
    assert predicted_idle_terms({"setup": {}}) is None
    assert predicted_idle_terms({"setup": {"idle": {"hours": 0.0}}}) is None
    assert predicted_idle_terms(_manifest(idle_hours=-5.0)) is None
    assert predicted_idle_terms({"setup": {"idle": {"hours": "n/a"}}}) is None


def _derived_and_closed_ledgers():
    """A complete run: iron ledger closes, charge ledger has unresolved C."""
    from models.plating_data import PlatingDerived
    from models.run_record import compute_ledgers

    derived = PlatingDerived(
        charge_C=100000.0,
        duration_s=7200.0,
        mean_cathodic_current_A=100000.0 / 7200.0,
        mean_voltage_V=3.0,
        energy_Wh=100000.0 * 3.0 / 3600.0,
        current_density_mA_cm2=138.9,
        faradaic_efficiency=27.5 / (100000.0 * 55.845 / (2.0 * FARADAY)),
        faradaic_efficiency_percent=95.0,
        theoretical_fe_mass_g=100000.0 * 55.845 / (2.0 * FARADAY),
        net_deposit_mass_g=27.5,
    )
    bath_batch = {
        "composition": {"fe2_g_L": 50.0, "volume_mL": 1000.0},
        "analysis": {"fe2_measured_g_L": 22.058, "solids_fe_mol": 0.0,
                     "other_fe_mol": 0.0},
    }
    characterization = pd.DataFrame(
        {"analyte": ["Fe"], "unit": ["wt%"], "technique": ["ICP"],
         "value": [100.0]}
    )
    ledgers = compute_ledgers(
        derived, bath_batch=bath_batch, characterization=characterization
    )
    return derived, bath_batch, characterization, ledgers


def test_run_record_ledgers_accept_predicted_idle_terms():
    from models.run_record import compute_ledgers

    derived, bath_batch, characterization, base = _derived_and_closed_ledgers()
    assert base["iron"]["status"] == "closed"
    assert "predicted_idle_redissolution_charge_C" not in base["charge"]
    assert "predicted_idle_transfer_to_bath_fe_mol" not in base["iron"]

    terms = predicted_idle_terms(_manifest())
    ledgers = compute_ledgers(
        derived, bath_batch=bath_batch, characterization=characterization,
        predicted_idle=terms,
    )
    # the mol closure is untouched: idle redissolution conserves bath+deposit Fe
    assert ledgers["iron"]["unaccounted_fe_mol"] == pytest.approx(
        base["iron"]["unaccounted_fe_mol"])
    # the predicted term annotates the charge-ledger residual instead
    charge = ledgers["charge"]
    assert charge["predicted_idle_redissolution_charge_C"] == pytest.approx(
        terms["charge_C"])
    assert charge["unresolved_charge_after_predicted_idle_C"] == pytest.approx(
        base["charge"]["unresolved_charge_C"] - terms["charge_C"])
    iron = ledgers["iron"]
    assert iron["predicted_idle_transfer_to_bath_fe_mol"] == pytest.approx(
        terms["fe_mol"])
    assert base["charge"]["unresolved_charge_C"] is not None


# ── closed_loop campaign wiring ────────────────────────────────────────

def test_campaign_idle_accounting_bookkeeps_nights_and_weekends():
    from models.closed_loop import (
        CampaignCalendar,
        PhaseIVOperatingPoint,
        campaign_idle_accounting,
    )

    op = PhaseIVOperatingPoint(current_density_mA_cm2=300.0, anode_area_m2=1.0)
    five_day = campaign_idle_accounting(
        operating=op, calendar=CampaignCalendar(days=30, hours_on_per_day=16.0,
                                                off_days_per_week=2.0))
    assert five_day["production_fe_mol"] > 0.0
    assert five_day["idle_fe_mol"] > 0.0
    # screening sanity: idle loss is a bookkeeping leak, not a process sink
    assert five_day["idle_loss_pct_of_production"] < 1.0
    assert five_day["apparent_fe_bias_pp"] < 0.0
    # adding weekend segments strictly costs more metal
    seven_day = campaign_idle_accounting(
        operating=op, calendar=CampaignCalendar(days=30, hours_on_per_day=16.0,
                                                off_days_per_week=0.0))
    assert seven_day["idle_fe_mol"] < five_day["idle_fe_mol"]
    # channel split sums to the total
    assert (five_day["idle_from_acid_g"] + five_day["idle_from_etch_g"]
            ) == pytest.approx(five_day["idle_fe_g"])
    assert five_day["screening_flag"] == SCREENING_FLAG


# ── ladder rederivation + scope + anchors ──────────────────────────────

def test_ladder_gate_flipped_to_modelled():
    from models.product_ladder import gate_status

    status = gate_status("g_oc_corrosion")
    assert status["exists"] is True
    assert status["state"].startswith("modelled")
    assert status["flag"] == SCREENING_FLAG


def test_model_scope_declares_live_and_anchored_parts():
    scope = model_scope()
    assert scope["screening_flag"] == SCREENING_FLAG
    assert any("fe3_shuttle" in s for s in scope["live_derivations"])
    assert any("SPECULATIVE" in s for s in scope["screening_proxies_anchored"])
    for key in ("live_derivations", "screening_proxies_anchored",
                "out_of_scope", "exact"):
        assert scope[key]


def test_all_module_anchors_registered():
    for key in (
        "FE_ACID_JCORR_REF_UA_CM2", "FE_ACID_ANODIC_TAFEL_MV_DEC",
        "FE_ACID_HER_TAFEL_MV_DEC", "FE_CORR_EA_KJ_MOL",
        "O2_DIFFUSIVITY_25C_M2_S", "DIFFUSION_EA_KJ_MOL",
        "DIFFUSION_LAYER_IDLE_M", "FE3_ETCH_K_REF_MOL_M2_S",
        "FE3_ETCH_REF_M", "FE3_ETCH_H_ORDER", "FE3_ETCH_EA_KJ_MOL",
        "ADDITIVE_BLOCKING_COVERAGE", "IDLE_BATH_FE3_M",
        "O2_FRACTION_SAT_IDLE",
    ):
        a = get_anchor(key)  # KeyError is the intentional failure mode
        assert a.ref and a.notes
