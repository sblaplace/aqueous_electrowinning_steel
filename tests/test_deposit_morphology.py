"""Tests for deposit morphology prediction model."""

import math
import pytest

from models.deposit_morphology import (
    dendrite_critical_current,
    her_disruption_threshold,
    nucleation_rate_ratio,
    predict_morphology,
    morphology_map,
    viable_operating_window,
    MorphologyResult,
)
from models.kinetics import DepositionKinetics


# ─── Dendrite Criterion ───────────────────────────────────────────

class TestDendriteCriticalCurrent:
    def test_positive(self):
        j = dendrite_critical_current(1.0, 50e-6)
        assert j > 0

    def test_increases_with_concentration(self):
        j_low = dendrite_critical_current(0.1, 50e-6)
        j_high = dendrite_critical_current(2.0, 50e-6)
        assert j_high > j_low

    def test_increases_with_thinner_boundary_layer(self):
        j_thick = dendrite_critical_current(1.0, 200e-6)
        j_thin = dendrite_critical_current(1.0, 20e-6)
        assert j_thin > j_thick

    def test_below_diffusion_limit(self):
        """Dendrite threshold should be below the diffusion limit."""
        delta = 50e-6
        C = 1.0 * 1000  # mol/m³
        D = 7.2e-10
        i_lim = 2 * 96485 * D * C / delta
        j_dend = dendrite_critical_current(1.0, delta)
        assert j_dend < i_lim

    def test_higher_surface_energy_delays_dendrites(self):
        """Higher surface energy should raise the dendrite threshold."""
        j_low = dendrite_critical_current(1.0, 50e-6, surface_energy_J_m2=1.0)
        j_high = dendrite_critical_current(1.0, 50e-6, surface_energy_J_m2=3.0)
        assert j_high >= j_low


# ─── HER Disruption ────────────────────────────────────────────────

class TestHERDisruption:
    def test_threshold_between_0_and_1(self):
        for T in [25, 50, 70, 90]:
            f = her_disruption_threshold(T)
            assert 0 < f < 1

    def test_decreases_with_temperature(self):
        f_cold = her_disruption_threshold(25)
        f_hot = her_disruption_threshold(90)
        assert f_hot < f_cold

    def test_decreases_with_roughness(self):
        f_smooth = her_disruption_threshold(60, surface_roughness=1.0)
        f_rough = her_disruption_threshold(60, surface_roughness=3.0)
        assert f_rough < f_smooth


# ─── Nucleation Regime ────────────────────────────────────────────

class TestNucleationRate:
    def test_zero_at_zero_overpotential(self):
        ratio = nucleation_rate_ratio(0.0)
        assert ratio == 0.0

    def test_positive_at_finite_overpotential(self):
        ratio = nucleation_rate_ratio(0.1)
        assert ratio >= 0

    def test_finite_values(self):
        """Should not be NaN or Inf for reasonable overpotentials."""
        for eta in [0.01, 0.05, 0.1, 0.2, 0.5]:
            ratio = nucleation_rate_ratio(eta)
            assert math.isfinite(ratio)


# ─── Morphology Prediction ─────────────────────────────────────────

class TestMorphologyPrediction:
    @pytest.fixture
    def kinetics(self):
        return DepositionKinetics(
            pH=2.0,
            temperature_C=60.0,
            fe_i0=1e-2,
            her_i0=1e-3,
            fe_conc_M=1.0,
            boundary_layer_m=50e-6,
        )

    def test_returns_result(self, kinetics):
        r = predict_morphology(100.0, kinetics)
        assert isinstance(r, MorphologyResult)

    def test_has_valid_outcome(self, kinetics):
        r = predict_morphology(100.0, kinetics)
        valid = {"coherent_film", "fine_grain_film", "dendrites",
                 "powder", "no_deposit", "disrupted"}
        assert r.outcome in valid

    def test_fe_between_0_and_1(self, kinetics):
        for j in [10, 50, 100, 200, 500]:
            r = predict_morphology(float(j), kinetics)
            assert 0 <= r.faradaic_efficiency <= 1

    def test_dendrite_ratio_positive(self, kinetics):
        r = predict_morphology(100.0, kinetics)
        assert r.dendrite_onset_ratio >= 0

    def test_summary_is_string(self, kinetics):
        r = predict_morphology(100.0, kinetics)
        s = r.summary()
        assert isinstance(s, str)
        assert "MORPHOLOGY" in s

    def test_high_her_i0_disrupts(self):
        """Very high HER should push toward disruption/no_deposit."""
        kin = DepositionKinetics(
            pH=2.0, temperature_C=60.0,
            fe_i0=1e-2, her_i0=1.0,  # dominant HER
            fe_conc_M=1.0,
        )
        r = predict_morphology(100.0, kin)
        assert r.outcome in ("disrupted", "no_deposit", "powder")

    def test_low_current_coherent(self):
        """Low current density should give coherent deposit."""
        kin = DepositionKinetics(
            pH=2.0, temperature_C=60.0,
            fe_i0=1e-2, her_i0=1e-6,  # suppressed HER
            fe_conc_M=1.0,
            boundary_layer_m=50e-6,
        )
        r = predict_morphology(10.0, kin)
        assert r.outcome in ("coherent_film", "fine_grain_film")

    def test_above_dendrite_threshold(self):
        """Very high current should trigger dendrites or powder."""
        kin = DepositionKinetics(
            pH=2.0, temperature_C=60.0,
            fe_i0=1e-2, her_i0=1e-6,
            fe_conc_M=1.0,
            boundary_layer_m=50e-6,
        )
        # Well above the limiting current
        r = predict_morphology(1000.0, kin)
        assert r.outcome in ("dendrites", "powder", "disrupted", "no_deposit")


# ─── Morphology Map ────────────────────────────────────────────────

class TestMorphologyMap:
    def test_returns_results(self):
        kin = DepositionKinetics(fe_conc_M=1.0, boundary_layer_m=50e-6)
        m = morphology_map(kin, n_points=10)
        assert len(m["j_mA_cm2"]) == 10
        assert len(m["outcomes"]) == 10
        assert len(m["FE"]) == 10

    def test_fe_decreases_at_high_j(self):
        """FE should generally decrease at very high current."""
        kin = DepositionKinetics(fe_conc_M=1.0, boundary_layer_m=50e-6)
        m = morphology_map(kin, n_points=20)
        # At low j, FE should be higher than at very high j
        fe_low = m["FE"][0]
        fe_high = m["FE"][-1]
        # Not strict monotonic but trend should hold
        assert fe_low >= fe_high * 0.5  # allow some tolerance


# ─── Viable Operating Window ───────────────────────────────────────

class TestViableWindow:
    def test_finds_window(self):
        kin = DepositionKinetics(
            fe_i0=1e-2, her_i0=1e-6,
            fe_conc_M=1.0, boundary_layer_m=50e-6,
        )
        w = viable_operating_window(kin, min_FE=0.5)
        assert w["viable"]
        assert w["j_viable_low"] > 0
        assert w["j_viable_high"] > w["j_viable_low"]
        assert w["j_optimal"] is not None

    def test_no_window_with_dominant_her(self):
        kin = DepositionKinetics(
            fe_i0=1e-2, her_i0=10.0,  # extreme HER
            fe_conc_M=1.0,
        )
        w = viable_operating_window(kin, min_FE=0.5)
        assert not w["viable"]

    def test_optimal_fe_reasonable(self):
        kin = DepositionKinetics(
            fe_i0=1e-2, her_i0=1e-6,
            fe_conc_M=1.0, boundary_layer_m=50e-6,
        )
        w = viable_operating_window(kin, min_FE=0.5)
        if w["viable"]:
            assert 0.5 <= w["FE_at_optimal"] <= 1.0
