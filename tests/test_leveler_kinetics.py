"""Tests for leveler / additive Langmuir adsorption kinetics
(CHEM_PHYS_REVIEW.md §2.6): per-additive Langmuir isotherms, the Γ-dependent
nucleation rate, and the Γ-dependent H-recombination overpotential.

The three deliverables the card pins are asserted directly:
  * Langmuir isotherms (θ and Γ) for saccharin / thiourea / PEG / coumarin /
    chloride — monotone in concentration, zero at c=0, bounded by Γ_max;
  * a Γ-dependent nucleation-rate multiplier that is ≥ 1 with additives and
    exactly 1.0 for the null package;
  * a Γ-dependent H-recombination fraction / overpotential reduction that grows
    with the package and caps safely below unity.
And the "default path unchanged" constraint: no additive package leaves
``internal_stress`` / ``deposit_morphology`` / ``co_deposition`` byte-identical.
"""
import numpy as np
import pytest

from models.leveler_kinetics import (
    ADDITIVE_SPECS,
    RELIEF_MAX,
    SUPPORTED_ADDITIVES,
    langmuir_coverage,
    gamma_adsorbed,
    coverage_per_additive,
    resolve_package,
    nucleation_rate_multiplier,
    h_recomb_fraction,
    h_recomb_overpotential_reduction_V,
    stress_relief_fraction,
    carbon_incorporation_blocking,
    structural_grade_score,
    compare_packages,
)
from models.internal_stress import deposit_stress_from_conditions
from models.deposit_morphology import nucleation_rate_ratio
from models.co_deposition import GuglielmiCarbonIncorporation


# ─── Langmuir isotherms ─────────────────────────────────────────────
class TestLangmuirIsotherm:
    def test_supported_additives_present(self):
        for aid in ("saccharin", "thiourea", "peg", "coumarin", "chloride"):
            assert aid in SUPPORTED_ADDITIVES

    def test_coverage_zero_at_zero_conc(self):
        assert langmuir_coverage(300.0, 0.0) == 0.0
        assert gamma_adsorbed(ADDITIVE_SPECS["saccharin"], 0.0) == 0.0

    def test_coverage_monotone_and_bounded(self):
        spec = ADDITIVE_SPECS["saccharin"]
        cs = np.geomspace(1e-6, 10.0, 40)
        thetas = [coverage_per_additive(spec, c)["theta"] for c in cs]
        assert all(0.0 <= t < 1.0 for t in thetas)
        # strictly increasing in concentration
        diffs = np.diff(thetas)
        assert np.all(diffs > 0.0), f"θ must increase with c: {thetas}"

    def test_gamma_scales_with_theta(self):
        """Γ = Γ_max·θ — monotone Γ and bounded by Γ_max."""
        spec = ADDITIVE_SPECS["thiourea"]
        g_low = gamma_adsorbed(spec, 0.01)   # weak coverage
        g_high = gamma_adsorbed(spec, 1.0)   # near-saturation
        assert 0.0 < g_low < g_high
        assert g_high < spec.gamma_max_mol_m2

    def test_additives_vary_in_affinity(self):
        """Thiourea/thiourea-class bind far more strongly than chloride."""
        k_thio = ADDITIVE_SPECS["thiourea"].K_ad_L_mol
        k_cl = ADDITIVE_SPECS["chloride"].K_ad_L_mol
        assert k_thio > 50 * k_cl

    def test_negative_conc_rejected(self):
        with pytest.raises(ValueError):
            langmuir_coverage(300.0, -1.0)
        with pytest.raises(ValueError):
            gamma_adsorbed(ADDITIVE_SPECS["saccharin"], -1.0)


# ─── Γ-dependent nucleation rate ────────────────────────────────────
class TestNucleationRateMultiplier:
    def test_null_package_is_identity(self):
        assert nucleation_rate_multiplier(None) == 1.0
        assert nucleation_rate_multiplier({}) == 1.0

    def test_additives_raise_multiplier(self):
        mul = nucleation_rate_multiplier({"saccharin": 1.5})
        assert mul > 1.0

    def test_monotone_in_concentration(self):
        m_low = nucleation_rate_multiplier({"thiourea": 0.02})
        m_high = nucleation_rate_multiplier({"thiourea": 0.2})
        assert m_low > 1.0
        assert m_high > m_low

    def test_multiplier_capped(self):
        pkg = {a: 5.0 for a in SUPPORTED_ADDITIVES}  # extreme saturation
        m = resolve_package(pkg).nucleation_multiplier
        assert m <= 10.0

    def test_flows_through_nucleation_rate_ratio(self):
        """Leveler package makes the morphology screen see finer grain."""
        r0 = nucleation_rate_ratio(overpotential_V=0.08)
        r1 = nucleation_rate_ratio(
            overpotential_V=0.08,
            nucleation_multiplier=nucleation_rate_multiplier({"coumarin": 0.3}),
        )
        assert r1 > r0


# ─── Γ-dependent H-recombination ────────────────────────────────────
class TestHRecombination:
    def test_null_package_is_identity(self):
        assert h_recomb_fraction(None) == 0.0
        assert h_recomb_overpotential_reduction_V(None) == 0.0

    def test_additives_recombine_h(self):
        f = h_recomb_fraction({"thiourea": 0.1})
        assert 0.0 < f < 1.0
        assert h_recomb_overpotential_reduction_V({"thiourea": 0.1}) > 0.0

    def test_monotone_in_package(self):
        f_low = h_recomb_fraction({"saccharin": 0.5})
        f_hi = h_recomb_fraction({"saccharin": 2.0})
        assert f_hi > f_low

    def test_capped_below_unity(self):
        pkg = {a: 10.0 for a in SUPPORTED_ADDITIVES}
        assert h_recomb_fraction(pkg) < 1.0


# ─── Stress relief from the package ─────────────────────────────────
class TestStressRelief:
    def test_null_is_zero(self):
        assert stress_relief_fraction(None) == 0.0

    def test_relief_monotone_and_bounded(self):
        r_low = stress_relief_fraction({"saccharin": 0.2})
        r_hi = stress_relief_fraction({"saccharin": 2.0})
        assert r_low > 0.0
        assert r_hi > r_low
        assert r_hi <= RELIEF_MAX

    def test_carbon_blocking_suppresses_incorporation(self):
        assert carbon_incorporation_blocking(None) == 1.0
        block_heavy = carbon_incorporation_blocking({"thiourea": 0.2, "peg": 0.5})
        assert 0.0 < block_heavy < 1.0


# ─── Structural-grade scoring ───────────────────────────────────────
class TestStructuralGrade:
    def test_null_package_scores_below_structural(self):
        res = structural_grade_score(None)
        assert res["score"] < 0.55
        assert res["verdict"] != "structural grade"

    def test_weighted_package_outranks_light_package(self):
        heavy = structural_grade_score(
            {"saccharin": 1.5, "thiourea": 0.05, "peg": 0.2, "chloride": 1.0}
        )
        light = structural_grade_score({"saccharin": 0.1})
        assert heavy["score"] > light["score"]

    def test_compare_packages_ranks(self):
        ranked = compare_packages(
            {"plus": {"saccharin": 1.5, "thiourea": 0.05, "peg": 0.2},
             "sac_only": {"saccharin": 1.5}}
        )["ranked"]
        assert ranked[0]["name"] == "plus"
        assert ranked[0]["score"] > ranked[1]["score"]


# ─── Default-path-unchanged constraint ──────────────────────────────
class TestDefaultPathUnchanged:
    def test_internal_stress_default_identical_to_legacy(self):
        """No additive_package → byte-identical to the saccharin_g_L path."""
        res_0 = deposit_stress_from_conditions(saccharin_g_L=0.0)
        res_1 = deposit_stress_from_conditions(saccharin_g_L=1.5)
        res_pkg0 = deposit_stress_from_conditions(additive_package=None)
        assert res_pkg0["derived"] == res_0["derived"]
        assert res_pkg0["components"] == res_0["components"]
        # saccharin relief still reduces total (tensile) stress
        assert res_1["components"]["total_MPa"] < res_0["components"]["total_MPa"]

    def test_internal_stress_additive_package_lowers_stress(self):
        """A leveler package both relieves intrinsic and cuts hydrogen stress."""
        base = deposit_stress_from_conditions(saccharin_g_L=0.0)
        pkg = deposit_stress_from_conditions(
            additive_package={"saccharin": 1.5, "thiourea": 0.05, "peg": 0.2}
        )
        assert pkg["corrections"]["additive_package_g_L"] is not None
        assert pkg["corrections"]["h_recomb_fraction"] > 0.0
        assert pkg["corrections"]["C_H_effective_ppm"] < base["derived"]["C_H_diffusible_ppm"]
        assert pkg["components"]["total_MPa"] < base["components"]["total_MPa"]

    def test_morphology_default_unchanged(self):
        """nucleation_rate_ratio default multiplier → same as before."""
        a = nucleation_rate_ratio(overpotential_V=0.06, temperature_C=60.0)
        b = nucleation_rate_ratio(
            overpotential_V=0.06, temperature_C=60.0, nucleation_multiplier=1.0
        )
        assert a == b

    def test_codeposition_default_unchanged(self):
        base = GuglielmiCarbonIncorporation()
        with_org_zero = GuglielmiCarbonIncorporation(organic_coverage_theta=0.0)
        assert base.carbon_incorporation_result(100.0) == with_org_zero.carbon_incorporation_result(100.0)
        # heavy organic coverage suppresses carbon
        org = GuglielmiCarbonIncorporation(organic_coverage_theta=0.6)
        assert org.carbon_incorporation_result(100.0)["predicted_carbon_wt_percent"] < \
            base.carbon_incorporation_result(100.0)["predicted_carbon_wt_percent"]

    def test_unsupported_additive_rejected(self):
        with pytest.raises(ValueError):
            resolve_package({"bogus": 1.0})


__all__ = []
