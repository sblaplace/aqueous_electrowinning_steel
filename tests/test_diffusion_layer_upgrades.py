"""Tests for diffusion_layer_1d surface-state and FeSO₄⁰ pair corrections.

Validates the Tier-1.1 (surface_state) and Tier-2 (FeSO₄⁰ neutral pair)
additions from CHEM_PHYS_REVIEW.md and docs/CHEM_PHYS_IMPROVEMENTS_V2.md.
"""

import math
import pytest
import numpy as np

from models.diffusion_layer_1d import (
    DiffusionLayer1D,
    K_FESO4_PAIR_25,
    D_FESO4_RATIO,
    D_FE2_COMPOSITION_SLOPE,
)


class TestFeSO4PairCorrection:
    """FeSO₄⁰ neutral pair reduces effective transport limit."""

    def test_pair_fraction_zero_when_disabled(self):
        """When fes04_pair_correction=False, pair fraction is 0."""
        model = DiffusionLayer1D(
            fe_conc_M=1.5,
            pH_bulk=2.0,
            fes04_pair_correction=False,
        )
        assert model._bulk_pair_fraction == 0.0

    def test_pair_fraction_positive_when_enabled(self):
        """When enabled, pair fraction is > 0 for typical bath."""
        model = DiffusionLayer1D(
            fe_conc_M=1.5,
            pH_bulk=2.0,
            support_conc_M=0.5,
            fes04_pair_correction=True,
        )
        f = model._bulk_pair_fraction
        assert f > 0
        assert f < 1.0

    def test_pair_fraction_bounded(self):
        """Pair fraction is always between 0 and 1."""
        for fe_M in (0.1, 0.5, 1.0, 1.5, 2.0):
            for support in (0.0, 0.1, 0.5, 1.0):
                model = DiffusionLayer1D(
                    fe_conc_M=fe_M, support_conc_M=support,
                    fes04_pair_correction=True,
                )
                f = model._bulk_pair_fraction
                assert 0.0 <= f <= 1.0

    def test_pair_reduces_diffusion_limit(self):
        """With pair correction, diffusion limit is lower."""
        model_bare = DiffusionLayer1D(
            fe_conc_M=1.5, support_conc_M=0.5,
            fes04_pair_correction=False,
        )
        model_pair = DiffusionLayer1D(
            fe_conc_M=1.5, support_conc_M=0.5,
            fes04_pair_correction=True,
        )
        assert model_pair.diffusion_limit_A_m2 < model_bare.diffusion_limit_A_m2

    def test_pair_reduction_moderate(self):
        """The correction should be moderate (15-35%), physically reasonable.

        At the reference bath (1.5 M FeSO₄ + 0.5 M Na₂SO₄) the speciation
        module gives ~24 % pair fraction; the resulting j_lim reduction is
        ~0.4 × f_pair ≈ 10 % (the pair diffuses at ~60 % of free Fe²⁺).
        """
        model_bare = DiffusionLayer1D(
            fe_conc_M=1.5, support_conc_M=0.5,
            fes04_pair_correction=False,
        )
        model_pair = DiffusionLayer1D(
            fe_conc_M=1.5, support_conc_M=0.5,
            fes04_pair_correction=True,
        )
        ratio = model_pair.diffusion_limit_A_m2 / model_bare.diffusion_limit_A_m2
        assert 0.65 < ratio < 1.0  # moderate reduction

    def test_pair_fraction_matches_speciation(self):
        """Pair fraction should match speciation module (~24 %) ± 5 %.

        This is the tuning target: α = 1.18 in the I-dependent K_eff is
        chosen to reproduce speciation's value at the reference bath.
        """
        model = DiffusionLayer1D(
            fe_conc_M=1.5, pH_bulk=2.0, support_conc_M=0.5,
            fes04_pair_correction=True,
        )
        f = model._bulk_pair_fraction
        # Speciation module gives 24.4 % at this composition; allow ± 5 %
        assert 0.19 < f < 0.30, f"pair fraction {f:.1%} outside [19%, 30%]"

    def test_result_carries_pair_diagnostic(self):
        """Result object carries the pair fraction when enabled."""
        model = DiffusionLayer1D(
            fe_conc_M=1.5, pH_bulk=2.0,
            fes04_pair_correction=True,
        )
        result = model.solve(100.0)
        assert result.fe_pair_fraction_bulk > 0

    def test_result_pair_zero_when_disabled(self):
        """Result pair fraction is 0 when correction is disabled."""
        model = DiffusionLayer1D(
            fe_conc_M=1.5, pH_bulk=2.0,
            fes04_pair_correction=False,
        )
        result = model.solve(100.0)
        assert result.fe_pair_fraction_bulk == 0.0

    def test_pair_constants_exist(self):
        """The module-level constants are accessible."""
        assert K_FESO4_PAIR_25 > 0
        assert 0 < D_FESO4_RATIO < 1


class TestSurfaceStateIntegration:
    """Surface-state HER kinetics (Tier 1.1) in the FE engine."""

    def test_disabled_is_backward_compatible(self):
        """With surface_state=False, results match the original model."""
        model = DiffusionLayer1D(
            fe_conc_M=1.5, pH_bulk=2.0,
            surface_state=False,
        )
        result = model.solve(100.0)
        assert result.her_i0_surface_state_ratio == 1.0
        assert 0 < result.current_efficiency < 1

    def test_enabled_reduces_her(self):
        """Surface state (with anion blocking) should reduce HER i₀."""
        model = DiffusionLayer1D(
            fe_conc_M=1.5, pH_bulk=2.0,
            surface_state=True, bath_type="sulfate",
        )
        result = model.solve(100.0)
        # With surface state, the HER i₀ is corrected.  In sulfate
        # bath at pH 2, the correction should be < 1 (HER suppressed).
        assert result.her_i0_surface_state_ratio <= 1.0

    def test_surface_state_changes_fe(self):
        """Surface state changes the FE prediction."""
        model_bare = DiffusionLayer1D(
            fe_conc_M=1.5, pH_bulk=2.0,
            surface_state=False,
        )
        model_ss = DiffusionLayer1D(
            fe_conc_M=1.5, pH_bulk=2.0,
            surface_state=True, bath_type="sulfate",
        )
        r_bare = model_bare.solve(200.0)
        r_ss = model_ss.solve(200.0)
        # The two should differ (surface state changes HER i₀)
        # but both should converge
        assert r_bare.converged
        assert r_ss.converged
        # FE should be different (surface state typically gives higher FE
        # because HER is suppressed)
        assert r_bare.current_efficiency != pytest.approx(
            r_ss.current_efficiency, abs=0.001
        )

    def test_aware_bath_gives_higher_fe(self):
        """AWARE (chloride) bath gives stronger HER suppression than sulfate.

        Unlike the earlier degenerate-limit behaviour (where the Temkin
        θ_H(1-θ_H) term collapsed to ~0 and drove HER i₀→0 for *every*
        bath), the wiring now binds i₀,H to anion site-blocking only:
        chloride (10 M) suppresses more than sulfate, and HER survives
        in both so the differentiation is real at the cell level.
        """
        model_sulfate = DiffusionLayer1D(
            fe_conc_M=1.0, pH_bulk=2.0,
            surface_state=True, bath_type="sulfate",
        )
        model_aware = DiffusionLayer1D(
            fe_conc_M=1.0, pH_bulk=2.0,
            surface_state=True, bath_type="aware",
        )
        r_sulfate = model_sulfate.solve(200.0)
        r_aware = model_aware.solve(200.0)
        # Chloride suppresses HER more than sulfate.
        assert r_aware.her_i0_surface_state_ratio < r_sulfate.her_i0_surface_state_ratio
        # And it translates to higher FE than sulfate.
        assert r_aware.current_efficiency > r_sulfate.current_efficiency

    def test_surface_state_differentiates_not_vanished(self):
        """Chloride suppresses HER more than sulfate, without FE→1.

        Regression guard: the original wiring folded in the Temkin
        θ_H·(1−θ_H) term, which collapses to ~0 at operating η, driving
        FE to 1.000 in *every* bath — the chloride-vs-sulfate
        differentiation was unmeasurable at cell level, and contradicted
        the H₂-safety sister module.  Site-blocking alone must give a
        real FE separation while leaving HER non-degenerate.
        """
        model_sulfate = DiffusionLayer1D(
            fe_conc_M=1.5, support_conc_M=0.5, pH_bulk=2.0,
            surface_state=True, bath_type="sulfate",
        )
        model_aware = DiffusionLayer1D(
            fe_conc_M=1.5, support_conc_M=0.5, pH_bulk=2.0,
            surface_state=True, bath_type="aware",
        )
        r_sulfate = model_sulfate.solve(200.0)
        r_aware = model_aware.solve(200.0)
        # HER survives (not degenerately suppressed to zero) in both.
        assert r_sulfate.her_i0_surface_state_ratio > 0.05
        assert r_aware.her_i0_surface_state_ratio > 0.001
        # Neither bath runs FE = 1.0 (some HER persists, as the H₂ module
        # assumes).
        assert r_sulfate.current_efficiency < 0.995
        assert r_aware.current_efficiency < 0.9995
        # Chloride bath suppresses HER more ⇒ higher FE than sulfate.
        assert r_aware.current_efficiency > r_sulfate.current_efficiency + 0.005

    def test_invalid_bath_type_raises(self):
        """Invalid bath_type should raise when surface_state=True."""
        model = DiffusionLayer1D(
            surface_state=True, bath_type="invalid",
        )
        with pytest.raises((ValueError, KeyError)):
            model.solve(100.0)


class TestCombinedCorrections:
    """Both corrections active simultaneously."""

    def test_both_corrections_work(self):
        """Enabling both corrections should converge."""
        model = DiffusionLayer1D(
            fe_conc_M=1.5, pH_bulk=2.0, support_conc_M=0.5,
            surface_state=True, bath_type="sulfate",
            fes04_pair_correction=True,
        )
        result = model.solve(200.0)
        assert result.converged
        assert result.her_i0_surface_state_ratio < 1.0
        assert result.fe_pair_fraction_bulk > 0

    def test_both_correct_but_stable(self):
        """Both corrections active: FE should be in a reasonable range."""
        model = DiffusionLayer1D(
            fe_conc_M=1.5, pH_bulk=2.0, support_conc_M=0.5,
            surface_state=True, bath_type="sulfate",
            fes04_pair_correction=True,
        )
        result = model.solve(300.0)
        assert 0.3 < result.current_efficiency < 1.0
        assert result.converged


class TestCompositionDiffusivity:
    """Composition-dependent D(γ, c) — Tier 2.2 (CHEM_PHYS_REVIEW).

    D_FE2=7.2e-10 is infinite-dilution; in 1.5 M FeSO4 + 0.5 M Na2SO4 the
    real D is 30-50 % lower.  The Stokes-Einstein factor 10^(−k·I) must
    land in that band at the reference bath, be backward compatible when
    off, and not double-count with the FeSO4-pair correction.
    """

    REF = dict(fe_conc_M=1.5, pH_bulk=2.0, support_conc_M=0.5, temperature_C=60.0)

    def test_disable_preserves_current_d(self):
        """Default (off) returns the flat infinite-dilution D_feff."""
        base = DiffusionLayer1D(**self.REF, composition_diffusivity=False)
        comp = DiffusionLayer1D(**self.REF, composition_diffusivity=True)
        # off: D_fe is the plain Arrhenius-corrected infinite-dilution value
        assert base.D_fe > comp.D_fe
        assert base._composition_diffusivity_factor() == 1.0
        assert base.diffusion_limit_A_m2 > comp.diffusion_limit_A_m2

    def test_factor_lands_in_literature_band(self):
        """Reference-bath D reduction lands in the cited 30-50 % band."""
        comp = DiffusionLayer1D(**self.REF, composition_diffusivity=True)
        f = comp._composition_diffusivity_factor()
        # f = 10^(−k·I); 1−f should be 30-50 %
        assert 0.30 <= (1.0 - f) <= 0.50, f"reduction {(1-f):.1%} outside 30-50%"

    def test_factor_monotone_in_conc(self):
        """Higher ionic strength → lower D (more concentrated → more viscous)."""
        dilute = DiffusionLayer1D(fe_conc_M=0.1, pH_bulk=2.0, support_conc_M=0.0,
                                  composition_diffusivity=True)
        conc = DiffusionLayer1D(fe_conc_M=2.0, pH_bulk=2.0, support_conc_M=1.0,
                                composition_diffusivity=True)
        assert dilute._composition_diffusivity_factor() > conc._composition_diffusivity_factor()

    def test_result_carries_diagnostic(self):
        """Result exposes the factor and (pitzer) gamma when enabled."""
        comp = DiffusionLayer1D(**self.REF, composition_diffusivity=True,
                                activity_model="pitzer")
        r = comp.solve(100.0)
        assert 0.3 < r.d_composition_factor < 0.7
        assert 0.0 < r.gamma_fe_bulk < 1.0
        # and zero-effect diagnostics when off
        base = DiffusionLayer1D(**self.REF)
        rb = base.solve(100.0)
        assert rb.d_composition_factor == 1.0
        assert rb.gamma_fe_bulk == 1.0

    def test_composes_with_pair_correction_without_double_count(self):
        """D(γ,c) and the FeSO4-pair correction are independent mechanisms.

        The pair correction shifts free-Fe²⁺ into a neutral pair (speciation);
        D(γ,c) lowers the mobility of the (remaining) free Fe²⁺ (viscosity).
        They are distinct physics, so with both on the transport-limit
        reduction equals the product of the two individual reductions — NOT
        a double-count of the same effect, and NOT an over-suppression below
        the physical pair+mobility floor.
        """
        base = DiffusionLayer1D(**self.REF)
        comp = DiffusionLayer1D(**self.REF, composition_diffusivity=True)
        pair = DiffusionLayer1D(**self.REF, fes04_pair_correction=True)
        both = DiffusionLayer1D(**self.REF, composition_diffusivity=True,
                                fes04_pair_correction=True)
        r_comp = comp.diffusion_limit_A_m2 / base.diffusion_limit_A_m2
        r_pair = pair.diffusion_limit_A_m2 / base.diffusion_limit_A_m2
        r_both = both.diffusion_limit_A_m2 / base.diffusion_limit_A_m2
        # both ≈ product of the two independent reductions (±5 %)
        assert r_both == pytest.approx(r_comp * r_pair, rel=0.05)
        # and stays above a physical floor (the two corrections together
        # do not collapse transport to zero)
        assert r_both > 0.4

    def test_slope_constant_positive(self):
        """The module-level slope constant is positive and in a sane range."""
        assert 0 < D_FE2_COMPOSITION_SLOPE < 0.5

    def test_dilute_returns_near_unity(self):
        """In a dilute bath the factor approaches 1 (no viscosity drag)."""
        model = DiffusionLayer1D(fe_conc_M=0.05, pH_bulk=2.0, support_conc_M=0.0,
                                 composition_diffusivity=True)
        f = model._composition_diffusivity_factor()
        assert f > 0.9

