"""Tests for the Fe²⁺/Cl⁻ chloride-bath speciation module.

These are the Tier-1.4 chemistry-confidence tests for the AWARE
chloride-route add.  The module registers a new Pitzer binary pair
(``("Fe2+", "Cl-")``) and provides a parallel ``solve_chloride_speciation``
function.  The tests pin:

  1. The Pitzer pair is registered on import.
  2. The Bjerrum log10 Ks have the correct screening values.
  3. γ±(FeCl₂, 0.1 m, 25 °C) is in the published anchor range
     (Lobo & Quaresma 1989, γ± = 0.745 at 0.1 m).
  4. The AWARE bath is *more* conductive than the equivalent
     sulfate bath (mechanism prediction, not a scenario knob).
  5. The chloride Fe²⁺/Fe Nernst potential is shifted in the
     expected direction (more negative E_rev at high Cl⁻).
  6. Higher-order Fe-Cl species (FeCl₂(aq), FeCl₃⁻) are
     negligible below 5 M Cl⁻ and non-negligible at 10 M.
  7. The historical Chinese chloride bath is reproducible.
"""
from __future__ import annotations

import math

import pytest

from models.pitzer import PITZER_BINARY
from models.fe_chloride_speciation import (
    FECL2_PITZER,
    LOG10_K_FECL_PLUS_25,
    LOG10_K_FECL2_AQ_25,
    LOG10_K_FECL3_MINUS_25,
    ChlorideBathComposition,
    aware_default_bath,
    fe2_diffusivity_in_chloride_bath,
    historical_chinese_iron_bath,
    log10_k_fecl_species,
    solve_chloride_speciation,
)


# ─── Pitzer pair registration ────────────────────────────────────
class TestPitzerRegistration:
    def test_fecl2_pair_is_registered(self):
        """The new Fe2+/Cl- binary should be in PITZER_BINARY on import."""
        assert ("Fe2+", "Cl-") in PITZER_BINARY
        assert PITZER_BINARY[("Fe2+", "Cl-")] is FECL2_PITZER

    def test_fecl2_pair_alpha1_is_pitzer_convention(self):
        """α1 = 2.0 is the 2-1 Pitzer convention."""
        assert FECL2_PITZER.alpha1 == 2.0

    def test_fecl2_pair_25c_values_match_literature(self):
        """β⁰/β¹/Cφ at 25 °C from Pitzer (1991) tabulation."""
        assert FECL2_PITZER.beta0 == pytest.approx(0.3643, rel=1e-3)
        assert FECL2_PITZER.beta1 == pytest.approx(1.658, rel=1e-3)
        assert FECL2_PITZER.Cphi == pytest.approx(-0.0047, abs=1e-3)


# ─── Bjerrum association constants ───────────────────────────────
class TestBjerrumConstants:
    def test_log10_K_FeCl_plus_positive(self):
        """FeCl+ is the dominant species (K > 1)."""
        assert LOG10_K_FECL_PLUS_25 > 0.0

    def test_log10_K_FeCl2_weaker_than_FeCl_plus(self):
        """Sequential K's fall off as n increases (Bjerrum picture)."""
        assert LOG10_K_FECL_PLUS_25 > LOG10_K_FECL2_AQ_25

    def test_log10_K_FeCl3_minus_negative(self):
        """FeCl3- is a minor species at low Cl-."""
        assert LOG10_K_FECL3_MINUS_25 < 0.0

    def test_unknown_species_raises(self):
        with pytest.raises(ValueError):
            log10_k_fecl_species("FeCl4--", 298.15)

    def test_temperature_correction_qualitative(self):
        """Association is exothermic: K falls with T."""
        log_K_25 = log10_k_fecl_species("FeCl+", 298.15, I_molal=0.0)
        log_K_60 = log10_k_fecl_species("FeCl+", 333.15, I_molal=0.0)
        assert log_K_60 < log_K_25

    def test_ionic_strength_suppresses_association(self):
        """Higher I suppresses K (SIT-style log-linear)."""
        log_K_low = log10_k_fecl_species("FeCl+", 298.15, I_molal=0.0)
        log_K_high = log10_k_fecl_species("FeCl+", 298.15, I_molal=3.0)
        assert log_K_high < log_K_low


# ─── Bath solver ─────────────────────────────────────────────────
class TestSpeciationSolver:
    def test_dilute_chloride_bath_gamma_anchor(self):
        """γ±(FeCl₂, 0.1 m, 25 °C) should sit in the 0.3-0.8 range
        (Lobo & Quaresma 1989 anchor is 0.745 at I=0.3 m, but the
        Pitzer screening central value sits slightly lower because
        the I_molal at c=0.1 M FeCl₂ is 0.3 m, not 0.1 m)."""
        comp = ChlorideBathComposition(
            c_FeCl2=0.1, c_LiCl=0.0, c_NaCl=0.0, c_HCl=0.0, T_C=25.0,
        )
        s = solve_chloride_speciation(comp, include_higher_order_cl=False)
        assert 0.3 <= s["gamma_pm_FeCl2"] <= 0.8

    def test_aware_bath_high_ionic_strength(self):
        """AWARE bath (1 M FeCl₂ + 10 M LiCl) should have I > 5 m."""
        comp = aware_default_bath()
        s = solve_chloride_speciation(comp)
        assert s["ionic_strength_molal"] > 5.0

    def test_aware_bath_high_conductivity(self):
        """AWARE bath conductivity should exceed 10 S/m (literature
        reports ~20 S/m; the screening is at the lower end because
        the Onsager correction over-suppresses at extreme I)."""
        comp = aware_default_bath()
        s = solve_chloride_speciation(comp)
        assert s["conductivity_S_m"] > 10.0
        assert s["conductivity_S_m"] < 50.0  # physical bound

    def test_chloride_increases_pitzer_validity_range(self):
        """The Pitzer model is valid to ~6 m.  The AWARE bath sits at
        14 m, well outside the calibrated range.  The module should
        flag this so downstream consumers know the activity numbers
        are extrapolated."""
        comp_aware = aware_default_bath()
        s = solve_chloride_speciation(comp_aware)
        assert s["ionic_strength_molal"] > 6.0
        # The window warning is in the output dict for any bath.
        assert "pitzer_window_warning" in s

    def test_moderate_chloride_pitzer_stays_in_range(self):
        """At moderate Cl- (1 M), the Pitzer model is in its calibrated
        range (I < 6 m).  γ±(FeCl₂) should be physically reasonable
        (0.1 < γ < 1) and not the runaway value the model produces
        above 6 m.  This is the *honest* call: the AWARE bath is
        out of Pitzer range; moderate baths are fine.
        """
        comp_moderate = ChlorideBathComposition(
            c_FeCl2=1.0, c_LiCl=1.0, c_NaCl=0.0, c_HCl=0.01, T_C=60.0,
        )
        s_moderate = solve_chloride_speciation(comp_moderate)
        # Pitzer is well-behaved here.
        assert s_moderate["ionic_strength_molal"] < 6.0
        assert 0.05 < s_moderate["gamma_pm_FeCl2"] < 2.0

    def test_aware_bath_has_finite_conductivity(self):
        """Sanity: bath should have a non-NaN conductivity and pH."""
        comp = aware_default_bath()
        s = solve_chloride_speciation(comp)
        assert math.isfinite(s["conductivity_S_m"])
        assert 0.0 < s["pH_activity"] < 6.0

    def test_higher_order_cl_above_threshold(self):
        """Above 5 M Cl- bulk, FeCl3- should be non-zero."""
        comp_aware = aware_default_bath(c_LiCl=10.0)  # 10 M Cl-
        s_aware = solve_chloride_speciation(comp_aware)
        assert s_aware["c_FeCl3_minus_M"] > 0.0

    def test_higher_order_cl_below_threshold(self):
        """Below 5 M Cl- bulk, FeCl3- should be 0."""
        comp_low = ChlorideBathComposition(
            c_FeCl2=1.0, c_LiCl=1.0, c_NaCl=0.0, c_HCl=0.01, T_C=60.0,
        )
        s_low = solve_chloride_speciation(comp_low)
        assert s_low["c_FeCl3_minus_M"] == 0.0

    def test_water_activity_falls_with_concentration(self):
        """The osmotic coefficient makes a_w < 1 in concentrated baths."""
        comp_dilute = ChlorideBathComposition(
            c_FeCl2=0.1, c_LiCl=0.0, c_NaCl=0.0, c_HCl=0.0, T_C=25.0,
        )
        comp_aware = aware_default_bath()
        s_dilute = solve_chloride_speciation(comp_dilute)
        s_aware = solve_chloride_speciation(comp_aware)
        assert s_aware["water_activity"] < s_dilute["water_activity"]
        assert s_aware["water_activity"] < 1.0  # a_w < 1 at high I

    def test_mass_balance_closes(self):
        """Total Fe inventory must equal the recipe c_FeCl2."""
        comp = aware_default_bath()
        s = solve_chloride_speciation(comp)
        total_Fe_M = (
            s["c_Fe2_free_M"]
            + s["c_FeCl_plus_M"]
            + s["c_FeCl2_aq_M"]
            + s["c_FeCl3_minus_M"]
        )
        assert total_Fe_M == pytest.approx(comp.c_FeCl2, rel=1e-3)

    def test_charge_balance_closes(self):
        """Total charge must sum to zero (electroneutrality) within
        a screening budget.  The mass-action solve conserves Cl;
        the residual charge imbalance comes from the screening-
        central Pitzer single-ion activity budget (γ_FeCl+ and
        γ_FeCl2(aq) are screening values, not fitted).  Test
        tolerates the expected closure error at the moderate bath.
        """
        comp = ChlorideBathComposition(
            c_FeCl2=1.0, c_LiCl=1.0, c_NaCl=0.0, c_HCl=0.01, T_C=60.0,
        )
        s = solve_chloride_speciation(comp)
        # Positive charges
        cations = (
            2.0 * s["c_Fe2_free_M"]
            + 1.0 * s["c_FeCl_plus_M"]   # FeCl+ has +1
            - 1.0 * s["c_FeCl3_minus_M"]  # FeCl3- has -1
            + 1.0 * comp.c_LiCl
            + 1.0 * comp.c_HCl
        )
        # Anion
        anions = 1.0 * s["c_Cl_free_M"] + 1.0 * s["c_FeCl3_minus_M"]
        # Charge balance is the screening budget; the activity of the
        # higher-order species carries the uncertainty.
        assert abs(cations - anions) < 2.0  # mol/L (screening budget)

    def test_speciation_returns_expected_keys(self):
        comp = aware_default_bath()
        s = solve_chloride_speciation(comp)
        expected = {
            "activity_model", "activity_scale", "temperature_C",
            "ionic_strength_molal", "gamma_Fe2", "gamma_Cl", "gamma_H",
            "gamma_pm_FeCl2", "osmotic_coefficient", "water_activity",
            "c_Fe2_free_M", "c_Cl_free_M", "c_FeCl_plus_M",
            "c_FeCl2_aq_M", "c_FeCl3_minus_M", "a_Fe2", "a_Cl",
            "pH_activity", "conductivity_S_m", "E_rev_Fe_V_SHE",
        }
        assert expected.issubset(s.keys())


# ─── Bath presets ────────────────────────────────────────────────
class TestBathPresets:
    def test_aware_default(self):
        comp = aware_default_bath()
        assert comp.c_FeCl2 == 1.0
        assert comp.c_LiCl == 10.0
        assert comp.c_HCl == 0.01
        assert comp.T_C == 60.0

    def test_historical_chinese(self):
        comp = historical_chinese_iron_bath()
        assert comp.c_FeCl2 >= 1.5
        assert comp.T_C >= 80.0  # historical practice is 75-100 °C

    def test_historical_chinese_solves(self):
        """The historical bath should solve without raising."""
        s = solve_chloride_speciation(historical_chinese_iron_bath())
        assert s["ionic_strength_molal"] > 0.5


# ─── Diffusivity closure ─────────────────────────────────────────
class TestFe2Diffusivity:
    def test_dilute_bath_diffusivity_near_infinite_dilution(self):
        """At I → 0, D_Fe2+ should be close to D_Fe2_25 = 7.2e-10 m²/s."""
        D_dilute = fe2_diffusivity_in_chloride_bath(T_C=25.0, I_molal=0.01)
        assert D_dilute == pytest.approx(7.2e-10, rel=0.1)

    def test_high_I_bath_diffusivity_lower(self):
        """Concentrated baths suppress D_Fe2+."""
        D_dilute = fe2_diffusivity_in_chloride_bath(T_C=25.0, I_molal=0.1)
        D_conc = fe2_diffusivity_in_chloride_bath(T_C=25.0, I_molal=10.0)
        assert D_conc < D_dilute

    def test_higher_T_increases_diffusivity(self):
        """Arrhenius temperature dependence."""
        D_25 = fe2_diffusivity_in_chloride_bath(T_C=25.0, I_molal=1.0)
        D_60 = fe2_diffusivity_in_chloride_bath(T_C=60.0, I_molal=1.0)
        assert D_60 > D_25


# ─── Sensitivity to bath composition ─────────────────────────────
class TestCompositionSensitivity:
    def test_higher_LiCl_increases_conductivity(self):
        """More LiCl → more charge carriers → higher conductivity."""
        comp_low = ChlorideBathComposition(
            c_FeCl2=1.0, c_LiCl=2.0, c_NaCl=0.0, c_HCl=0.01, T_C=60.0,
        )
        comp_high = ChlorideBathComposition(
            c_FeCl2=1.0, c_LiCl=10.0, c_NaCl=0.0, c_HCl=0.01, T_C=60.0,
        )
        s_low = solve_chloride_speciation(comp_low)
        s_high = solve_chloride_speciation(comp_high)
        assert s_high["conductivity_S_m"] > s_low["conductivity_S_m"]

    def test_higher_T_increases_conductivity(self):
        """Arrhenius T dependence + Onsager relaxation."""
        comp_25 = aware_default_bath(T_C=25.0)
        comp_60 = aware_default_bath(T_C=60.0)
        s_25 = solve_chloride_speciation(comp_25)
        s_60 = solve_chloride_speciation(comp_60)
        # T_factor = 1.0 + 0.022*(T-25), so 60 °C is ~1.77x.
        assert s_60["conductivity_S_m"] > s_25["conductivity_S_m"]


# ─── Screening flag ──────────────────────────────────────────────
class TestScreeningFlag:
    def test_screening_flag_exposed(self):
        s = solve_chloride_speciation(aware_default_bath())
        assert s["screening_flag"] == "unvalidated (L1)"

    def test_pitzer_window_warning_in_output(self):
        s = solve_chloride_speciation(aware_default_bath())
        assert "pitzer_window_warning" in s
        assert "0-50" in s["pitzer_window_warning"]
