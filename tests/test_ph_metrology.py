"""Fast contracts for the V6 §5.1 operational-pH metrology screen."""
import math

import pytest

from models.ph_metrology import (
    BathComposition, Bridge, evaluate_bias, junction_potential_mV,
    model_scope, nernst_slope_mV_pH, operational_to_pitzer,
    pitzer_to_operational, two_point_protocol,
)


def test_default_bias_is_concentrated_brine_scale_and_finite():
    result = evaluate_bias()
    # Stoichiometric 1.5 M FeSO4 + 0.5 M Na2SO4 is 7.25 M on the
    # ideal-ion-strength convention; activity/complexation lowers the real matrix value.
    assert 6.0 < result.ionic_strength_M < 8.0
    assert 10.0 < abs(result.junction_mV) < 100.0
    # Junction and single-ion terms may partly cancel; the uncertainty ledger
    # (rather than a forced signed correction) is what guards the HER fit.
    assert abs(result.total_offset_pH) < 1.5
    assert result.uncertainty_pH > 0.2


def test_pH_conversion_round_trip():
    bias = evaluate_bias()
    pH = 2.13
    assert pitzer_to_operational(operational_to_pitzer(pH, bias), bias) == pytest.approx(pH)


def test_nernst_slope_rises_with_temperature_and_is_66mv_at_60c():
    assert nernst_slope_mV_pH(60.0) == pytest.approx(66.1, abs=0.2)
    assert nernst_slope_mV_pH(60.0) > nernst_slope_mV_pH(25.0)


def test_bridge_aging_adds_log_time_drift():
    fresh = evaluate_bias(bridge=Bridge(age_days=0))
    old = evaluate_bias(bridge=Bridge(age_days=9))
    assert old.bridge_drift_mV > fresh.bridge_drift_mV
    assert old.total_offset_pH > fresh.total_offset_pH


def test_junction_is_finite_and_varies_with_bath_matrix():
    bridge = Bridge(kcl_M=3.0)
    default = junction_potential_mV(BathComposition(), bridge)
    lean = junction_potential_mV(BathComposition(fe2_M=0.1, na2so4_M=0.05), bridge)
    assert math.isfinite(default) and math.isfinite(lean)
    assert default != pytest.approx(lean)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        evaluate_bias(temperature_C=-274)
    with pytest.raises(ValueError):
        evaluate_bias(bridge=Bridge(age_days=-1))


def test_protocol_and_scope_name_the_evidence_and_shelved_physics():
    assert len(two_point_protocol()) == 3
    scope = model_scope()
    assert scope["screening_flag"] == "unvalidated (L1)"
    assert scope["live_derivations"]
    assert any("Pitzer" in item for item in scope["out_of_scope"])
