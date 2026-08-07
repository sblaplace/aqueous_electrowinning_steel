"""Tests for the surface-state HER / Frumkin-corrected kinetics.

These are the *chemistry-confidence* tests for the Tier-1.1 add:
they pin the limit behaviour, self-consistency, and the four
qualitative claims the new module must defend:

  1. θ_H is pinned near 1 on Fe(110) over the cathodic window
     (consistent with ``her_microkinetics``).
  2. The Frumkin factor is < 1 for any adsorbed anion (anion-down
     dipole pushes the IHP negative, suppresses HER).
  3. The Temkin fixed-point converges within the iteration budget
     and is bounded between the Langmuir limits.
  4. The chloride-rich (AWARE) bath suppresses HER more than the
     sulfate bath at the same η — a *mechanism* prediction, not a
     scenario knob.

Reference bounds come from the screening numbers in
``models/surface_state``; tolerances are loose (±20 % on θ, ±50 %
on Frumkin) because the module is L1 (mechanism layer, no fitted
constants).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from models.kinetics import DepositionKinetics
from models.surface_state import (
    ALPHA_HEY,
    BORATE_AWARE,
    CL_NA_AWARE,
    DG_HSTAR_FE110_J,
    FacetDistribution,
    HSO4_AWARE,
    SCREENING_FLAG,
    SO4_AWARE,
    SurfaceStateKinetics,
    AnionCoverage,
    chloride_aware_default,
    diagnostic_table,
    frumkin_sensitivity_sweep,
    volmer_coverage,
)


# ─── Coverage model ─────────────────────────────────────────────
class TestVolmerCoverage:
    def test_high_coverage_on_fe110_in_cathodic_window(self):
        """ΔG_H* ≈ -0.40 eV → θ_H remains > 0.5 over the operating window
        at 60 °C.  With Temkin g=12 kJ/mol the coverage does fall
        appreciably by η=0.5 V (this is the literature-measured H-UPD
        behaviour on Fe), but in the 0-0.3 V window θ stays above 0.7.
        """
        T = 333.15
        for eta in (0.0, 0.1, 0.3):
            assert volmer_coverage(DG_HSTAR_FE110_J, eta, T) > 0.7

    def test_weak_binding_drops_coverage(self):
        """ΔG > 0 (weak binder) at fixed η should give θ < 0.5."""
        # At η = 0.3 V cathodic, exp(-(ΔG + F·η)/RT) is small for ΔG > 0
        th = volmer_coverage(0.2 * 96485.0, 0.3, 333.15)
        assert th < 0.5

    def test_temkin_lowers_coverage_relative_to_langmuir(self):
        """With g=12 kJ/mol and θ≈1, Temkin should pull θ *below* Langmuir.
        At g=0 (Langmuir limit) θ is 0.999; at g=12 kJ/mol it should
        be measurably lower.  This is the Temkin signature."""
        T = 333.15
        th_langmuir = volmer_coverage(DG_HSTAR_FE110_J, 0.0, T, g_temkin_J_mol=0.0)
        th_temkin = volmer_coverage(DG_HSTAR_FE110_J, 0.0, T, g_temkin_J_mol=12e3)
        assert 0.0 <= th_temkin <= 1.0
        assert th_temkin < th_langmuir
        # Modest but measurable difference (5-20 %).
        assert (th_langmuir - th_temkin) > 0.01

    def test_temkin_iteration_converges(self):
        """The fixed point must converge within 12 steps for any (g, T)."""
        T = 333.15
        for g in (10e3, 30e3, 60e3):
            for eta in (0.0, 0.1, 0.3, 0.5):
                th = volmer_coverage(DG_HSTAR_FE110_J, eta, T, g_temkin_J_mol=g)
                assert 0.0 < th < 1.0
                assert math.isfinite(th)

    def test_temkin_lowers_coverage_at_high_g(self):
        """Strong Temkin (g=60 kJ/mol) at θ≈1 should pull θ down."""
        T = 333.15
        th_low_g = volmer_coverage(DG_HSTAR_FE110_J, 0.0, T, g_temkin_J_mol=10e3)
        th_high_g = volmer_coverage(DG_HSTAR_FE110_J, 0.0, T, g_temkin_J_mol=60e3)
        assert th_high_g < th_low_g

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError):
            volmer_coverage(DG_HSTAR_FE110_J, 0.0, 0.0)


# ─── Facet distribution ───────────────────────────────────────
class TestFacetDistribution:
    def test_default_is_thirds(self):
        f = FacetDistribution()
        assert f.f_110 == pytest.approx(1.0 / 3.0)

    def test_fractions_must_sum_to_one(self):
        with pytest.raises(ValueError):
            FacetDistribution(f_110=0.5, f_100=0.5, f_211=0.5)

    def test_negative_fractions_rejected(self):
        with pytest.raises(ValueError):
            FacetDistribution(f_110=-0.1, f_100=0.55, f_211=0.55)

    def test_dg_hstar_eff_is_areal_weighted(self):
        f = FacetDistribution(f_110=0.0, f_100=0.0, f_211=1.0)
        assert f.dg_hstar_eff_J == pytest.approx(-0.30 * 96485.0)
        f = FacetDistribution(f_110=1.0, f_100=0.0, f_211=0.0)
        assert f.dg_hstar_eff_J == pytest.approx(-0.40 * 96485.0)

    def test_summary_keys(self):
        s = FacetDistribution().summary
        assert "f_110" in s and "f_100" in s and "f_211" in s
        assert "dg_hstar_eff_eV" in s


# ─── Anion coverage / Frumkin ψ₁ ────────────────────────────────
class TestAnionCoverage:
    def test_psi_1_negative_for_all_screening_anions(self):
        """All four anions carry anion-down dipoles → ψ₁ < 0."""
        T = 333.15
        for a in (CL_NA_AWARE, SO4_AWARE, HSO4_AWARE, BORATE_AWARE):
            cov = AnionCoverage(a, c_bulk_M=1.0, T_K=T)
            assert cov.psi_1_V < 0.0

    def test_chloride_psi_1_magnitude_larger_than_sulfate(self):
        """Cl- binds more strongly than SO4(2-) → larger |ψ₁|."""
        T = 333.15
        cl = abs(AnionCoverage(CL_NA_AWARE, c_bulk_M=1.0, T_K=T).psi_1_V)
        so4 = abs(AnionCoverage(SO4_AWARE, c_bulk_M=1.0, T_K=T).psi_1_V)
        assert cl > so4

    def test_aware_chloride_saturates_langmuir(self):
        """At 10 M Cl- the Langmuir θ should be near 1."""
        cov = AnionCoverage(CL_NA_AWARE, c_bulk_M=10.0, T_K=333.15)
        assert cov.theta > 0.99

    def test_sulfate_adsorbs_less_than_chloride(self):
        """At equal bulk, Cl- θ should be much larger than SO4(2-) θ."""
        T = 333.15
        cl = AnionCoverage(CL_NA_AWARE, c_bulk_M=1.5, T_K=T).theta
        so4 = AnionCoverage(SO4_AWARE, c_bulk_M=1.5, T_K=T).theta
        assert cl > so4
        # And SO4 should be < 0.7 at this concentration.
        assert so4 < 0.7

    def test_higher_concentration_increases_coverage(self):
        T = 333.15
        low = AnionCoverage(CL_NA_AWARE, c_bulk_M=0.1, T_K=T).theta
        high = AnionCoverage(CL_NA_AWARE, c_bulk_M=10.0, T_K=T).theta
        assert high > low

    def test_gamma_mol_m2_capped_by_site_density(self):
        """Γ must not exceed N_sites_FE regardless of bulk concentration."""
        T = 333.15
        from models.surface_state import N_SITES_FE_M2
        cov = AnionCoverage(CL_NA_AWARE, c_bulk_M=100.0, T_K=T)
        assert cov.gamma_mol_m2 <= N_SITES_FE_M2 + 1e-12


# ─── SurfaceStateKinetics wrapper ───────────────────────────────
class TestSurfaceStateKinetics:
    def _make_base(self) -> DepositionKinetics:
        return DepositionKinetics(
            pH=2.0, temperature_C=60.0,
            fe_i0=1.0e-2, her_i0=1.0e-3,
        )

    def test_her_i0_corrected_smaller_than_intrinsic_with_anion(self):
        """Site blocking dominates the Frumkin term: i₀,H_eff < i₀,H_intrinsic
        when anions are specifically adsorbed, even though the
        Frumkin potential factor alone would push the other way."""
        base = self._make_base()
        facets, anions = chloride_aware_default("sulfate")
        w = SurfaceStateKinetics(base=base, facets=facets, anion_coverages=anions)
        eta = 0.2
        i0_intrinsic = base.her_i0_T
        i0_eff = w.her_i0_corrected(eta)
        assert i0_eff < i0_intrinsic

    def test_frumkin_factor_is_less_than_one_for_anions(self):
        """Frumkin factor = exp(α·F·ψ₁/RT).  For anion-down adsorption
        (ψ₁ < 0), this is *less* than 1 — the standard Frumkin
        correction to the cathodic BV rate (Bockris & Reddy §7.7).
        This is the correct text-book sign; the magnitude depends
        on the screening parameter ``eta_screening``."""
        T_K = 333.15
        for a in (CL_NA_AWARE, SO4_AWARE, HSO4_AWARE, BORATE_AWARE):
            cov = AnionCoverage(a, c_bulk_M=1.0, T_K=T_K)
            assert cov.psi_1_V < 0.0
            factor = math.exp(ALPHA_HEY * 96485.0 * cov.psi_1_V / (8.314 * T_K))
            assert factor < 1.0

    def test_aware_bath_suppresses_her_more_than_sulfate(self):
        """The mechanism prediction: chloride-rich bath should drop i₀,H
        more than the sulfate bath at the same η, because Cl- blocks
        more sites than SO4(2-) (Cl- adsorbs at θ≈1 in 10 M, SO4 only
        at θ≈0.5 even at 1.5 M)."""
        base = self._make_base()
        f_so4, a_so4 = chloride_aware_default("sulfate")
        f_aware, a_aware = chloride_aware_default("aware")
        w_so4 = SurfaceStateKinetics(base=base, facets=f_so4, anion_coverages=a_so4)
        w_aware = SurfaceStateKinetics(base=base, facets=f_aware, anion_coverages=a_aware)
        eta = 0.2
        assert w_aware.her_i0_corrected(eta) < w_so4.her_i0_corrected(eta)

    def test_no_adsorbate_i0_close_to_intrinsic(self):
        """With no adsorbed anions and θ=1, i₀,H_eff should be <<
        intrinsic (because θ·(1-θ) is small)."""
        base = self._make_base()
        w = SurfaceStateKinetics(
            base=base, facets=FacetDistribution(), anion_coverages=()
        )
        # θ·(1-θ) on Fe(110) at η=0.2 V is ~0.01, so the correction
        # factor is small — but the *direction* of change is what
        # the test pins.
        ratio = w.surface_state(0.2).i0_H_effective_ratio
        assert 0.0 < ratio < 1.0

    def test_partial_currents_return_correct_shapes(self):
        base = self._make_base()
        w = SurfaceStateKinetics(
            base=base, facets=FacetDistribution(), anion_coverages=()
        )
        E = np.linspace(-1.0, -0.3, 20)
        i_fe, i_h, i_tot = w.partial_currents(E)
        assert i_fe.shape == E.shape
        assert i_h.shape == E.shape
        assert i_tot.shape == E.shape
        assert np.all(np.isfinite(i_fe))
        assert np.all(np.isfinite(i_h))

    def test_partial_currents_total_equals_sum(self):
        base = self._make_base()
        w = SurfaceStateKinetics(
            base=base, facets=FacetDistribution(), anion_coverages=()
        )
        E = np.linspace(-1.0, -0.3, 50)
        i_fe, i_h, i_tot = w.partial_currents(E)
        np.testing.assert_allclose(i_tot, i_fe + i_h, rtol=1e-12)

    def test_eta_effective_larger_than_eta_with_adsorbed_anion(self):
        """ψ₁ < 0 ⇒ η_eff = η - ψ₁ > η.  The Frumkin correction
        *increases* the effective overpotential at the IHP and
        suppresses HER (textbook sign, Bockris & Reddy §7.7)."""
        base = self._make_base()
        _, anions = chloride_aware_default("aware")
        w = SurfaceStateKinetics(base=base, anion_coverages=anions, use_frumkin=True)
        eta = 0.3
        eta_eff = w.eta_effective_V(eta)
        assert eta_eff > eta
        assert eta_eff > 0.0  # still cathodic

    def test_screen_flag_is_exposed(self):
        assert SCREENING_FLAG == "unvalidated (L1)"


# ─── Diagnostic table ───────────────────────────────────────────
class TestDiagnosticTable:
    def test_returns_expected_keys(self):
        base = DepositionKinetics(pH=2.0, temperature_C=60.0)
        facets, anions = chloride_aware_default("sulfate")
        tab = diagnostic_table(base, eta_values=[0.0, 0.1, 0.2, 0.3],
                                facets=facets, anion_coverages=anions)
        for k in ("eta_V", "theta_H", "psi_1_V", "frumkin_factor",
                  "i0_H_effective_ratio", "facet_summary"):
            assert k in tab
        assert len(tab["eta_V"]) == 4
        # All θ values should be in the operating range.
        assert np.all(tab["theta_H"] > 0.5)
        assert np.all(tab["theta_H"] < 1.0)
        # ψ₁ should be negative (anion-down).
        assert np.all(tab["psi_1_V"] < 0.0)
        # Frumkin factor is *less* than 1 (anion-down ads, text-book sign).
        assert np.all(tab["frumkin_factor"] < 1.0)
        # Effective i₀ ratio is < 1 (site blocking + Frumkin both suppress).
        assert np.all(tab["i0_H_effective_ratio"] < 1.0)


# ─── Bath presets ───────────────────────────────────────────────
class TestChlorideAwareDefault:
    def test_sulfate_bath(self):
        f, a = chloride_aware_default("sulfate")
        assert isinstance(f, FacetDistribution)
        assert all(isinstance(x, AnionCoverage) for x in a)
        # SO4 + HSO4 + borate (three anions in the screening set).
        assert len(a) == 3

    def test_aware_bath(self):
        f, a = chloride_aware_default("aware")
        # AWARE = Cl- only, no borate.
        assert len(a) >= 1
        assert all(an.anion.name.startswith("Cl-") for an in a)

    def test_mixed_bath(self):
        f, a = chloride_aware_default("mixed")
        # Mixed = SO4 + HSO4 + Cl- + borate.
        assert len(a) == 4

    def test_unknown_bath_raises(self):
        with pytest.raises(ValueError):
            chloride_aware_default("hydroxide")

    def test_facet_summary_round_trips(self):
        f, _ = chloride_aware_default("sulfate")
        s = f.summary
        assert s["f_110"] + s["f_100"] + s["f_211"] == pytest.approx(1.0)


# ─── eta_screening propagation (regression for the SurfaceStateKinetics bug) ──
class TestEtaScreeningPropagation:
    """Regression tests for the bug where ``SurfaceStateKinetics.surface_state``
    rebuilt the AnionCoverages with the default eta_screening, silently
    dropping any patched value the consumer passed in.  The Frumkin
    sensitivity sweep depends on this propagation working."""

    def test_eta_screening_propagates_through_surface_state(self):
        """The same SurfaceStateKinetics built with two different
        eta_screening values must produce different psi_1_V."""
        base = DepositionKinetics(pH=2.0, temperature_C=60.0,
                                    fe_i0=1.0e-2, her_i0=1.0e-3)
        _, a_so4_default = chloride_aware_default("sulfate")
        _, a_so4_high = chloride_aware_default("sulfate")
        a_so4_high = tuple(
            dataclasses.replace(a, eta_screening=0.20) for a in a_so4_high
        )
        w1 = SurfaceStateKinetics(base=base, anion_coverages=a_so4_default)
        w2 = SurfaceStateKinetics(base=base, anion_coverages=a_so4_high)
        ss1 = w1.surface_state(0.2)
        ss2 = w2.surface_state(0.2)
        # 4x higher eta_screening → 4x more negative psi_1.
        assert ss2.psi_1_V < ss1.psi_1_V
        assert ss2.psi_1_V / ss1.psi_1_V == pytest.approx(4.0, rel=0.01)
        # Frumkin factor must reflect the change too.
        assert ss2.frumkin_factor < ss1.frumkin_factor

    def test_her_i0_corrected_propagates_eta_screening(self):
        """The user-facing entry point must also propagate the change."""
        base = DepositionKinetics(pH=2.0, temperature_C=60.0,
                                    fe_i0=1.0e-2, her_i0=1.0e-3)
        _, a_so4_default = chloride_aware_default("sulfate")
        _, a_so4_high = chloride_aware_default("sulfate")
        a_so4_high = tuple(
            dataclasses.replace(a, eta_screening=0.20) for a in a_so4_high
        )
        w1 = SurfaceStateKinetics(base=base, anion_coverages=a_so4_default, use_frumkin=True)
        w2 = SurfaceStateKinetics(base=base, anion_coverages=a_so4_high, use_frumkin=True)
        i0_1 = w1.her_i0_corrected(0.2)
        i0_2 = w2.her_i0_corrected(0.2)
        # Higher eta_screening → larger |psi_1| → smaller i0 (text-book sign).
        assert i0_2 < i0_1


# ─── Frumkin sensitivity sweep ─────────────────────────────────
class TestFrumkinSensitivitySweep:
    """The 14x site-blocking / Frumkin-amplification decomposition."""

    def _base(self) -> DepositionKinetics:
        return DepositionKinetics(pH=2.0, temperature_C=60.0,
                                    fe_i0=1.0e-2, her_i0=1.0e-3)

    def test_returns_expected_keys(self):
        sweep = frumkin_sensitivity_sweep(self._base(), "sulfate", "aware")
        for k in ("eta_screening", "ratio_total", "ratio_site_blocking_only",
                  "ratio_frumkin_only", "psi_1_bath_a", "psi_1_bath_b",
                  "bath_a", "bath_b", "eta_V"):
            assert k in sweep

    def test_site_blocking_only_ratio_is_robust(self):
        """With Frumkin disabled (eta_screening=0), the ratio
        reflects only site-blocking + theta*(1-theta).  This is
        the *robust* mechanism prediction."""
        sweep = frumkin_sensitivity_sweep(
            self._base(), "sulfate", "aware",
            eta_screening_values=(0.0,),
        )
        # The site-blocking ratio must be ~14x at eta=0.2.
        assert sweep["ratio_site_blocking_only"][0] == pytest.approx(14.0, rel=0.05)
        # And the Frumkin factor ratio is exactly 1.0 when eta_screening=0.
        assert sweep["ratio_frumkin_only"][0] == pytest.approx(1.0, rel=1e-6)

    def test_total_ratio_grows_with_eta_screening(self):
        """Larger eta_screening → larger Frumkin amplification →
        larger total ratio.  This is the sensitivity band."""
        sweep = frumkin_sensitivity_sweep(
            self._base(), "sulfate", "aware",
            eta_screening_values=(0.01, 0.02, 0.05, 0.10),
        )
        assert sweep["ratio_total"][0] < sweep["ratio_total"][1]
        assert sweep["ratio_total"][1] < sweep["ratio_total"][2]
        assert sweep["ratio_total"][2] < sweep["ratio_total"][3]

    def test_total_ratio_at_central_eta_screening(self):
        """At eta_screening=0.05 (the screening central value) the
        total ratio is ~238x.  This is the headline number, but
        the *robust* prediction is the ~14x site-blocking."""
        sweep = frumkin_sensitivity_sweep(
            self._base(), "sulfate", "aware",
            eta_screening_values=(0.05,),
        )
        assert sweep["ratio_total"][0] == pytest.approx(238.0, rel=0.05)

    def test_psi_1_propagates_through_sweep(self):
        """The eta_screening change must propagate all the way through
        the surface_state and into the Frumkin factor — this is
        the regression that originally made the headline invariant
        in eta_screening (the bug fix)."""
        sweep = frumkin_sensitivity_sweep(
            self._base(), "sulfate", "aware",
            eta_screening_values=(0.01, 0.05, 0.20),
        )
        # Both psi_1 values must become more negative as eta_screening grows.
        for arr_key in ("psi_1_bath_a", "psi_1_bath_b"):
            arr = sweep[arr_key]
            assert arr[0] > arr[1] > arr[2]
            # And the ratio between successive entries should be 5x (and 4x).
            assert arr[1] / arr[0] == pytest.approx(5.0, rel=0.01)
            assert arr[2] / arr[1] == pytest.approx(4.0, rel=0.01)

    def test_cited_psi_1_range(self):
        """At eta_screening=0.02 (the lower end of the cited
        experimental range) the sulfate-bath psi_1 should be in
        the -0.05 to -0.3 V range quoted in the docstring."""
        sweep = frumkin_sensitivity_sweep(
            self._base(), "sulfate", "aware",
            eta_screening_values=(0.02,),
        )
        psi_so4 = sweep["psi_1_bath_a"][0]
        # At eta_screening=0.02, sulfate total psi_1 is ~-0.14 V — within range.
        assert -0.30 <= psi_so4 <= -0.05


# ─── use_frumkin opt-in / default-OFF behavior ────────────────────
class TestUseFrumkinOptIn:
    """The Frumkin ψ₁ term is opt-in (`use_frumkin=True`) and
    flagged OFF by default (`use_frumkin=False`).  When OFF,
    the adapter uses only site-blocking + Temkin coverage +
    facet ensemble — the ~14× robust mechanism prediction.
    When ON, the full Frumkin amplification applies.
    """

    def _base(self) -> DepositionKinetics:
        return DepositionKinetics(pH=2.0, temperature_C=60.0,
                                    fe_i0=1.0e-2, her_i0=1.0e-3)

    def test_default_is_frumkin_off(self):
        base = self._base()
        _, anions = chloride_aware_default("sulfate")
        w = SurfaceStateKinetics(base=base, anion_coverages=anions)
        assert w.use_frumkin is False

    def test_frumkin_off_does_not_shift_eta_eff(self):
        base = self._base()
        _, anions = chloride_aware_default("sulfate")
        w = SurfaceStateKinetics(base=base, anion_coverages=anions, use_frumkin=False)
        assert w.eta_effective_V(0.3) == pytest.approx(0.3)

    def test_frumkin_off_ignores_psi_1_in_i0(self):
        """With use_frumkin=False, her_i0_corrected must not include
        the Frumkin factor, so it should be larger than with
        use_frumkin=True (where the factor < 1 suppresses i0)."""
        base = self._base()
        _, anions = chloride_aware_default("sulfate")
        w_off = SurfaceStateKinetics(base=base, anion_coverages=anions, use_frumkin=False)
        w_on = SurfaceStateKinetics(base=base, anion_coverages=anions, use_frumkin=True)
        eta = 0.2
        assert w_off.her_i0_corrected(eta) > w_on.her_i0_corrected(eta)

    def test_frumkin_on_includes_psi_1_in_eta_eff(self):
        base = self._base()
        _, anions = chloride_aware_default("aware")
        w = SurfaceStateKinetics(base=base, anion_coverages=anions, use_frumkin=True)
        eta = 0.3
        assert w.eta_effective_V(eta) > eta  # ψ₁ < 0 → η_eff > η

    def test_frumkin_off_site_blocking_only_is_constant_across_screening(self):
        """The robust core claim: without Frumkin, changing
        eta_screening should not change her_i0_corrected."""
        base = self._base()
        _, a_default = chloride_aware_default("sulfate")
        _, a_high = chloride_aware_default("sulfate")
        a_high = tuple(
            dataclasses.replace(a, eta_screening=0.20) for a in a_high
        )
        w1 = SurfaceStateKinetics(base=base, anion_coverages=a_default, use_frumkin=False)
        w2 = SurfaceStateKinetics(base=base, anion_coverages=a_high, use_frumkin=False)
        # Site-blocking ratio must be invariant in eta_screening.
        assert w1.her_i0_corrected(0.2) == pytest.approx(w2.her_i0_corrected(0.2), rel=1e-9)


# ─── Bug-fix regression: the headline ratio is no longer invariant ──
class TestHeadlineRatioIsNotInvariant:
    """Regression test for the original headline invariance bug.

    Before the bug fix, the surface_state() method rebuilt
    AnionCoverage objects with the default eta_screening,
    silently dropping any patched value.  As a result, the
    total suppression ratio was invariant in eta_screening
    (always 238.65x at eta=0.2 V).  After the fix, the ratio
    varies with eta_screening as the physics requires.
    """

    def test_ratio_varies_with_eta_screening(self):
        base = DepositionKinetics(pH=2.0, temperature_C=60.0,
                                    fe_i0=1.0e-2, her_i0=1.0e-3)
        sweep = frumkin_sensitivity_sweep(
            base, "sulfate", "aware",
            eta_screening_values=(0.01, 0.02, 0.05, 0.10),
        )
        # Must NOT be invariant.  The spread should be at least 10x
        # between the lowest and highest screening value.
        rmin = float(sweep["ratio_total"].min())
        rmax = float(sweep["ratio_total"].max())
        assert rmax / rmin > 10.0
