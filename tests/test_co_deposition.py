"""Unit tests for the Phase III co-deposition model."""
from __future__ import annotations

import numpy as np
import pytest

from models.co_deposition import (
    GuglielmiCarbonIncorporation,
    AnomalousFeNiKinetics,
    PhaseIIICoDeposition,
    build_phase3_model,
)


class TestAnomalousFeNiKinetics:
    def test_default_initialization(self):
        kin = AnomalousFeNiKinetics()
        assert kin.bath_fe_M == 0.5
        assert kin.bath_ni_M == 0.5
        assert kin.mechanism == "hydroxide_suppression"

    def test_surface_pH_rises_with_current(self):
        kin = AnomalousFeNiKinetics(pH=3.5, buffer_capacity_M=0.05)
        pH_low = kin.surface_pH(10.0)
        pH_high = kin.surface_pH(300.0)
        assert pH_high > pH_low

    def test_alloy_composition_runs(self):
        kin = AnomalousFeNiKinetics()
        res = kin.alloy_composition(100.0)
        assert "fe_wt_percent" in res
        assert "ni_wt_percent" in res
        assert "is_anomalous" in res

    def test_efficiency_sweep_shape(self):
        kin = AnomalousFeNiKinetics()
        js, ce, fe, anom = kin.efficiency_sweep([10.0, 50.0, 100.0, 200.0])
        assert len(js) == 4
        assert len(ce) == 4


class TestGuglielmiCarbonIncorporation:
    def test_default_initialization(self):
        gc = GuglielmiCarbonIncorporation()
        assert gc.particle_conc_g_L == 1.0
        assert gc.particle_size_um == 1.5

    def test_loose_coverage_in_range(self):
        gc = GuglielmiCarbonIncorporation(particle_conc_g_L=2.0)
        sigma = gc.loose_adsorption_coverage_sigma(100.0)
        assert 0.0 <= sigma < 1.0

    def test_carbon_content_positive(self):
        gc = GuglielmiCarbonIncorporation(particle_conc_g_L=5.0)
        w_c = gc.carbon_content_wt_percent(150.0)
        assert w_c >= 0.0

    def test_incorporation_result_has_keys(self):
        gc = GuglielmiCarbonIncorporation()
        res = gc.carbon_incorporation_result(100.0)
        assert "predicted_carbon_wt_percent" in res
        assert "loose_adsorption_coverage_sigma" in res
        assert "surface_blocking_factor" in res


class TestPhaseIIICoDeposition:
    def test_default_model_string_output(self):
        model = build_phase3_model()
        s = str(model)
        assert "Phase III" in s
        assert "Anomalous" in s or "NORMAL" in s

    def test_run_at_current_structure(self):
        model = build_phase3_model()
        res = model.run_at_current(150.0)
        assert "alloy_kinetics" in res
        assert "carbon_incorporation" in res
        assert "integrated_metrics" in res
        assert isinstance(res["alloy_kinetics"]["is_anomalous"], bool)

    def test_sweep_returns_arrays(self):
        model = build_phase3_model()
        sweep = model.run_sweep([10.0, 100.0, 200.0])
        assert len(sweep["j_mA_cm2"]) == 3
        assert len(sweep["fe_wt_percent"]) == 3
        assert len(sweep["carbon_wt_percent"]) == 3

    def test_anomalous_mechanism_produces_different_result(self):
        m1 = build_phase3_model(mechanism_fe_ni="hydroxide_suppression")
        m2 = build_phase3_model(mechanism_fe_ni="intermediate_adsorption")
        r1 = m1.run_at_current(200.0)
        r2 = m2.run_at_current(200.0)
        # The two mechanisms should give different predictions
        assert r1["alloy_kinetics"]["fe_wt_percent"] != r2["alloy_kinetics"]["fe_wt_percent"]


class TestModuleImports:
    def test_import_from_init(self):
        from models import PhaseIIICoDeposition, GuglielmiCarbonIncorporation, AnomalousFeNiKinetics
        assert PhaseIIICoDeposition is not None
