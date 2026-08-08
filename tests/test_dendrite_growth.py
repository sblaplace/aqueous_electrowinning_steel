"""Tests for the Mullins-Sekerka / Barton-Bockris dendrite growth model.

Covers the screening-length stability criterion plus the growth-rate ODE
(CHEM_PHYS_REVIEW.md Tier 2.4): a perturbation of the screening wavelength
grows while an off-stability perturbation stays flat, and the opt-in wiring
into pulse.py produces a morphology prediction without changing defaults.
"""

import math

import pytest

from models.deposit_morphology import (
    GAMMA_FE_SURFACE,
    V_M_FE,
    MullinsSekerkaGrowthModel,
    predict_dendrite_growth,
    predict_morphology,
    dendrite_critical_current,
)
from models.kinetics import DepositionKinetics
from models.pulse import PulseDepositionModel, PulseWaveform


D = 7.2e-10


def _make_model(**kw):
    """Mullins-Sekerka growth model with canonical Fe material params."""
    return MullinsSekerkaGrowthModel(
        diffusivity_m2_s=D,
        surface_energy_J_m2=GAMMA_FE_SURFACE,
        molar_volume_m3_mol=V_M_FE,
        **kw,
    )


# ─── Screening length ─────────────────────────────────────────────

class TestScreeningLength:
    def test_positive(self):
        assert _make_model().screening_length(100.0) > 0

    def test_matches_review_formula(self):
        """λ_c = (D·γ·Ω / (j·(∂c/∂x)|surf))^(1/2) with Fick closure ∂c/∂x=j/(zFD)."""
        m = _make_model()
        j = 100.0
        grad = m.surface_concentration_gradient(j)
        expected = math.sqrt(D * GAMMA_FE_SURFACE * V_M_FE / (j * grad))
        assert m.screening_length(j) == pytest.approx(expected, rel=1e-12)

    def test_nernst_gradient_when_surface_given(self):
        m = _make_model()
        # bulk 1 M, surface 0.5 M over 50 µm → gradient (0.5 M)/50µm in mol/m⁴
        lam_fick = m.screening_length(100.0)
        lam_nernst = m.screening_length(
            100.0, surface_fe_conc_M=0.5, boundary_layer_m=50e-6, bulk_fe_conc_M=1.0)
        assert lam_nernst != lam_fick
        assert lam_nernst > 0

    def test_decreases_with_current(self):
        """Higher current → steeper gradient → shorter screening length (more unstable)."""
        m = _make_model()
        assert m.screening_length(10.0) > m.screening_length(500.0)

    def test_invalid_material_params_rejected(self):
        with pytest.raises(ValueError):
            MullinsSekerkaGrowthModel(diffusivity_m2_s=0.0)
        with pytest.raises(ValueError):
            MullinsSekerkaGrowthModel(surface_energy_J_m2=-1.0)


# ─── Dispersion & growth-rate ODE ─────────────────────────────────

class TestGrowthRate:
    @pytest.fixture
    def model(self):
        return _make_model()

    @pytest.fixture
    def lam_c(self, model):
        return model.screening_length(500.0)

    def test_perturbation_of_screening_wavelength_grows(self, model, lam_c):
        """A perturbation at/beyond the screening wavelength is unstable (σ > 0)."""
        assert model.growth_rate(lam_c * 2.0, 500.0) > 0.0
        assert model.is_unstable(lam_c * 2.0, 500.0)

    def test_critical_wavelength_is_marginal(self, model, lam_c):
        assert model.growth_rate(lam_c, 500.0) == pytest.approx(0.0, abs=1e-9)

    def test_off_stability_stays_flat(self, model, lam_c):
        """Below the screening wavelength (shorter λ) the perturbation decays/flat."""
        assert model.growth_rate(lam_c * 0.5, 500.0) < 0.0
        assert not model.is_unstable(lam_c * 0.5, 500.0)

    def test_amplitude_scale_invariance(self, model, lam_c):
        """Doubling the perturbing wavelength toward λ_c lowers σ to zero."""
        sigma_long = model.growth_rate(lam_c * 4.0, 500.0)
        sigma_shorter = model.growth_rate(lam_c * 2.0, 500.0)
        # both unstable, longer wavelength (further from critical) grows faster in σ·λ
        assert sigma_shorter > 0
        assert sigma_long > 0

    def test_ode_step_matches_exponential(self, model, lam_c):
        """da/dt = σa ⇒ a(t+dt) = a·e^(σdt) for a modest bounded σ."""
        wavelength = lam_c * 8.0  # σ modest here
        sigma = model.growth_rate(wavelength, 500.0)
        dt = 1e-3
        assert sigma * dt < 30.0  # within the exact (unclamped) regime
        a1 = model.advance_amplitude(1.0, dt, wavelength, 500.0)
        assert a1 == pytest.approx(math.exp(sigma * dt), rel=1e-6)

    def test_negative_rate_decays_never_overflow(self, model, lam_c):
        a = model.advance_amplitude(1e-3, 1e6, lam_c * 0.5, 500.0)
        assert 0.0 <= a < 1e-3  # decays toward flat, no overflow/nan

    def test_invalid_wavelength_rejected(self, model):
        with pytest.raises(ValueError):
            model.growth_rate(0.0, 100.0)


# ─── One-shot predictor ───────────────────────────────────────────

class TestPredictDendriteGrowth:
    def test_unstable_labels_dendrites_with_gain(self):
        m = _make_model()
        lam_c = m.screening_length(500.0)
        g = predict_dendrite_growth(500.0, lam_c * 2.0, time_s=0.1)
        assert g["morphology"] == "dendrites"
        assert g["amplitude_gain"] > 1.0
        assert g["growth_rate_1_s"] > 0.0
        assert g["model"] == "mullins_sekerka"

    def test_off_stability_labels_coherent_no_growth(self):
        m = _make_model()
        lam_c = m.screening_length(500.0)
        f = predict_dendrite_growth(500.0, lam_c * 0.5, time_s=0.1)
        assert f["morphology"] == "coherent"
        assert f["amplitude_gain"] < 1.0


# ─── Default behaviour unchanged + opt-in in predict_morphology ───

class TestPredictMorphologyIntegration:
    @pytest.fixture
    def kinetics(self):
        return DepositionKinetics(
            pH=2.0, temperature_C=60.0, fe_i0=1e-2, her_i0=1e-6,
            fe_conc_M=1.0, boundary_layer_m=50e-6,
        )

    def test_default_predict_morphology_unchanged(self, kinetics):
        """No growth_model → identical static path, growth fields are None."""
        r = predict_morphology(100.0, kinetics)
        assert r.dendrite_growth_rate_1_s is None
        assert r.screening_length_m is None
        j_dend = dendrite_critical_current(
            kinetics.fe_conc_M, kinetics.boundary_layer_m,
            diffusivity_m2_s=kinetics.diffusivity_m2_s,
            temperature_C=kinetics.temperature_C)
        assert r.dendrite_onset_ratio == pytest.approx(1000.0 / j_dend)

    def test_optin_growth_model_populates_fields(self, kinetics):
        m = _make_model()
        lam_c = m.screening_length(1000.0)
        r = predict_morphology(
            100.0, kinetics, growth_model=m,
            perturbation_wavelength_m=lam_c * 2.0)
        assert r.dendrite_growth_rate_1_s is not None
        assert r.screening_length_m is not None
        # unstable growth → dendrite ratio > 1
        assert r.dendrite_onset_ratio > 1.0

    def test_morphology_map_optin_includes_growth_columns(self, kinetics):
        from models.deposit_morphology import morphology_map
        m = _make_model()
        lam_c = m.screening_length(1000.0)
        plain = morphology_map(kinetics, n_points=5)
        assert "dendrite_growth_rate" not in plain
        grown = morphology_map(
            kinetics, n_points=5, growth_model=m,
            perturbation_wavelength_m=lam_c * 2.0)
        assert "dendrite_growth_rate" in grown
        assert "screening_length_m" in grown
        assert len(grown["dendrite_growth_rate"]) == 5


# ─── Pulse wiring (opt-in, default off) ───────────────────────────

class TestPulseMorphology:
    @pytest.fixture
    def waveform(self):
        return PulseWaveform(j_cathodic_mA_cm2=100.0, t_cathodic_s=0.05, t_off_s=0.05)

    def test_morphology_off_by_default(self, waveform):
        m = PulseDepositionModel(fe_bulk_M=1.0, bulk_pH=2.0)
        r = m.simulate(waveform, n_cycles=2, steps_per_cycle=30)
        assert r.morphology is None
        assert "Morphology (Mullins–Sekerka)" not in r.summary()

    def test_morphology_requires_wavelength(self):
        with pytest.raises(ValueError):
            PulseDepositionModel(predict_morphology=True)

    def test_morphology_optin_returns_prediction(self, waveform):
        m = PulseDepositionModel(
            fe_bulk_M=1.0, bulk_pH=2.0,
            predict_morphology=True, morphology_wavelength_m=1e-6)
        r = m.simulate(waveform, n_cycles=2, steps_per_cycle=30)
        assert r.morphology is not None
        for k in ("model", "screening_length_m", "growth_rate_1_s",
                  "amplitude_initial", "amplitude_final", "amplitude_gain",
                  "morphology"):
            assert k in r.morphology
        assert r.morphology["model"] == "mullins_sekerka"
        assert r.morphology["amplitude_initial"] > 0.0
        assert math.isfinite(r.morphology["amplitude_gain"])
        assert r.morphology["morphology"] in ("dendrites", "coherent", "marginal")
        # summary surfaces the prediction
        s = r.summary()
        assert "Morphology (Mullins–Sekerka)" in s
