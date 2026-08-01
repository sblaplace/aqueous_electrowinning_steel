"""Tests for the deposit adhesion / peel model.

The model's job is to make the gating unknown named in
``cell_architecture.py`` computable: does electrodeposited iron peel from a
drum?  These tests pin the fracture-mechanics arithmetic against closed-form
values, check the physical monotonicities, and guard the specific modelling
decisions that keep the screen honest — thickness-confined plastic
dissipation, the cohesive-failure branch that stops a strong interface being
read as good adhesion, and the conductivity veto that rejects a release
coating on physics rather than by omission.
"""

import math

import numpy as np
import pytest

from models.adhesion_peel import (
    ALPHA_FE_PER_K,
    ALPHA_TI_PER_K,
    E_FE_GPA,
    GAMMA_FE_J_M2,
    HOFFMAN_DELTA_M,
    MAX_WINDER_TENSION_N_PER_M,
    NU_FE,
    PLASTIC_ZONE_THICKNESS_UM,
    SUBSTRATES,
    PeelConditions,
    SubstrateSpec,
    amplification_robustness,
    biaxial_modulus_Pa,
    comparison_table,
    conditions_from_deposition,
    confined_plastic_amplification,
    coupon_test_protocol,
    critical_thickness_um,
    dupre_work_of_adhesion,
    energy_release_rate,
    evaluate_peel,
    film_tearing_energy_J_m2,
    foil_route_verdict,
    girifalco_good_work_of_adhesion,
    grain_size_sweep,
    hoffman_intrinsic_stress_MPa,
    hydrogen_stress_MPa,
    hydrogen_sweep,
    hydrogen_toughness_knockdown,
    interfacial_toughness,
    model_scope,
    peel_force_per_width,
    plane_strain_modulus_Pa,
    residual_stress,
    screen_substrates,
    stoney_stress_MPa,
    stoney_validity,
    thermal_mismatch_stress_MPa,
    thickness_sweep,
    web_stress_MPa,
)


# ═══════════════════════════════════════════════════════════════════
#  Elastic helpers
# ═══════════════════════════════════════════════════════════════════

class TestElasticHelpers:
    def test_plane_strain_modulus_matches_definition(self):
        assert plane_strain_modulus_Pa(200.0, 0.3) == pytest.approx(
            200e9 / (1 - 0.09)
        )

    def test_biaxial_modulus_matches_definition(self):
        assert biaxial_modulus_Pa(200.0, 0.3) == pytest.approx(200e9 / 0.7)

    def test_biaxial_exceeds_plane_strain(self):
        """E/(1−ν) > E/(1−ν²) for any 0 < ν < 0.5."""
        assert biaxial_modulus_Pa() > plane_strain_modulus_Pa()

    def test_rejects_nonphysical_poisson_ratio(self):
        with pytest.raises(ValueError):
            plane_strain_modulus_Pa(200.0, 0.7)

    def test_rejects_nonpositive_modulus(self):
        with pytest.raises(ValueError):
            biaxial_modulus_Pa(0.0)


# ═══════════════════════════════════════════════════════════════════
#  Residual stress
# ═══════════════════════════════════════════════════════════════════

class TestResidualStress:
    def test_hoffman_matches_closed_form(self):
        d_um = 0.5
        expected = biaxial_modulus_Pa() * HOFFMAN_DELTA_M / (d_um * 1e-6) / 1e6
        assert hoffman_intrinsic_stress_MPa(d_um) == pytest.approx(expected)

    def test_hoffman_scales_inversely_with_grain_size(self):
        """σ ∝ 1/d — halving the grain doubles the stress."""
        a = hoffman_intrinsic_stress_MPa(1.0)
        b = hoffman_intrinsic_stress_MPa(0.5)
        assert b == pytest.approx(2.0 * a)

    def test_hoffman_is_tensile(self):
        assert hoffman_intrinsic_stress_MPa(0.5) > 0

    def test_hoffman_magnitude_is_physically_plausible(self):
        """Sub-micron electrodeposits report a few hundred MPa tensile."""
        assert 100.0 < hoffman_intrinsic_stress_MPa(0.1) < 1000.0
        assert 10.0 < hoffman_intrinsic_stress_MPa(1.0) < 200.0

    def test_hoffman_rejects_nonpositive_grain(self):
        with pytest.raises(ValueError):
            hoffman_intrinsic_stress_MPa(0.0)

    def test_hydrogen_stress_zero_without_hydrogen(self):
        assert hydrogen_stress_MPa(0.0) == pytest.approx(0.0)

    def test_hydrogen_stress_is_linear_and_tensile(self):
        a = hydrogen_stress_MPa(5.0)
        b = hydrogen_stress_MPa(10.0)
        assert a > 0
        assert b == pytest.approx(2.0 * a)

    def test_hydrogen_stress_scales_with_effused_fraction(self):
        full = hydrogen_stress_MPa(10.0, fraction_effused=1.0)
        half = hydrogen_stress_MPa(10.0, fraction_effused=0.5)
        assert half == pytest.approx(0.5 * full)

    def test_hydrogen_stress_rejects_bad_fraction(self):
        with pytest.raises(ValueError):
            hydrogen_stress_MPa(1.0, fraction_effused=1.5)

    def test_thermal_mismatch_sign_follows_expansion_difference(self):
        """Fe on Ti (α_Fe > α_Ti) is tensile on cooling; on Cu it is not."""
        on_ti = thermal_mismatch_stress_MPa(
            35.0, ALPHA_FE_PER_K, ALPHA_TI_PER_K
        )
        on_cu = thermal_mismatch_stress_MPa(35.0, ALPHA_FE_PER_K, 16.5e-6)
        assert on_ti > 0
        assert on_cu < 0

    def test_thermal_mismatch_vanishes_for_matched_expansion(self):
        assert thermal_mismatch_stress_MPa(
            50.0, ALPHA_FE_PER_K, ALPHA_FE_PER_K
        ) == pytest.approx(0.0)

    def test_residual_stress_sums_its_parts(self):
        r = residual_stress(grain_size_um=0.5, C_H_ppm=5.0)
        assert r["total_MPa"] == pytest.approx(
            r["intrinsic_MPa"] + r["hydrogen_MPa"] + r["thermal_MPa"]
        )

    def test_residual_stress_identifies_dominant_mechanism(self):
        """At coarse grain and high H, hydrogen must dominate."""
        r = residual_stress(grain_size_um=5.0, C_H_ppm=200.0)
        assert r["dominant_mechanism"] == "hydrogen"

    def test_fine_grain_makes_intrinsic_dominant(self):
        r = residual_stress(grain_size_um=0.05, C_H_ppm=0.0)
        assert r["dominant_mechanism"] == "intrinsic"

    def test_residual_stress_reports_sign(self):
        r = residual_stress(grain_size_um=0.2)
        assert r["sign"] == "tensile"


class TestStoney:
    def test_stoney_matches_closed_form(self):
        kappa, hs, hf = 0.5, 100e-6, 5e-6
        expected = 116e9 * hs ** 2 * kappa / (6 * (1 - 0.32) * hf) / 1e6
        assert stoney_stress_MPa(kappa, hs, hf) == pytest.approx(expected)

    def test_stoney_is_linear_in_curvature(self):
        a = stoney_stress_MPa(0.2, 100e-6, 5e-6)
        b = stoney_stress_MPa(0.4, 100e-6, 5e-6)
        assert b == pytest.approx(2.0 * a)

    def test_stoney_inverse_of_forward_model(self):
        """Stoney and the forward stress estimate must be mutually consistent.

        Pick a stress, invert Stoney for the curvature it implies, feed that
        curvature back in, and recover the stress.
        """
        sigma = 250.0
        hs, hf = 100e-6, 5e-6
        kappa = sigma * 1e6 * 6 * (1 - 0.32) * hf / (116e9 * hs ** 2)
        assert stoney_stress_MPa(kappa, hs, hf) == pytest.approx(sigma)

    def test_stoney_validity_flags_thick_films(self):
        assert stoney_validity(100e-6, 5e-6)["valid"]
        assert not stoney_validity(100e-6, 50e-6)["valid"]

    def test_stoney_rejects_nonpositive_thickness(self):
        with pytest.raises(ValueError):
            stoney_stress_MPa(0.1, 0.0, 5e-6)


# ═══════════════════════════════════════════════════════════════════
#  Driving force
# ═══════════════════════════════════════════════════════════════════

class TestEnergyReleaseRate:
    def test_matches_hutchinson_suo_closed_form(self):
        sigma, h = 300.0, 25.0
        expected = (1 - NU_FE) * (sigma * 1e6) ** 2 * (h * 1e-6) / (E_FE_GPA * 1e9)
        assert energy_release_rate(sigma, h) == pytest.approx(expected)

    def test_linear_in_thickness(self):
        """The design lever: doubling h doubles the driving force."""
        a = energy_release_rate(300.0, 10.0)
        b = energy_release_rate(300.0, 20.0)
        assert b == pytest.approx(2.0 * a)

    def test_quadratic_in_stress(self):
        a = energy_release_rate(100.0, 25.0)
        b = energy_release_rate(200.0, 25.0)
        assert b == pytest.approx(4.0 * a)

    def test_sign_of_stress_does_not_matter(self):
        """Compressive films store energy too (buckle-delamination)."""
        assert energy_release_rate(-300.0, 25.0) == pytest.approx(
            energy_release_rate(300.0, 25.0)
        )

    def test_rejects_nonpositive_thickness(self):
        with pytest.raises(ValueError):
            energy_release_rate(300.0, 0.0)


class TestCriticalThickness:
    def test_is_the_inverse_of_energy_release_rate(self):
        """At h = h_c the driving force must equal the toughness."""
        sigma, gamma = 250.0, 3.0
        h_c = critical_thickness_um(sigma, gamma)
        assert energy_release_rate(sigma, h_c) == pytest.approx(gamma)

    def test_falls_with_stress(self):
        assert critical_thickness_um(400.0, 3.0) < critical_thickness_um(200.0, 3.0)

    def test_rises_with_toughness(self):
        assert critical_thickness_um(300.0, 6.0) > critical_thickness_um(300.0, 3.0)

    def test_infinite_at_zero_stress(self):
        assert math.isinf(critical_thickness_um(0.0, 3.0))

    def test_rejects_nonpositive_toughness(self):
        with pytest.raises(ValueError):
            critical_thickness_um(300.0, 0.0)


# ═══════════════════════════════════════════════════════════════════
#  Interfacial toughness
# ═══════════════════════════════════════════════════════════════════

class TestWorkOfAdhesion:
    def test_dupre_matches_definition(self):
        assert dupre_work_of_adhesion(2.4, 0.45, 2.55) == pytest.approx(0.30)

    def test_dupre_clipped_at_zero(self):
        """A negative work of adhesion is unphysical, not a repulsion."""
        assert dupre_work_of_adhesion(1.0, 1.0, 5.0) == 0.0

    def test_girifalco_geometric_mean(self):
        assert girifalco_good_work_of_adhesion(2.4, 0.45) == pytest.approx(
            2.0 * math.sqrt(2.4 * 0.45)
        )

    def test_metallic_couples_bond_more_strongly_than_passive_oxides(self):
        """The ordering the whole screen depends on."""
        metallic = dupre_work_of_adhesion(
            GAMMA_FE_J_M2,
            SUBSTRATES["ti_bare_etched"].surface_energy_J_m2,
            SUBSTRATES["ti_bare_etched"].interface_energy_J_m2,
        )
        passive = dupre_work_of_adhesion(
            GAMMA_FE_J_M2,
            SUBSTRATES["ti_passive_tio2"].surface_energy_J_m2,
            SUBSTRATES["ti_passive_tio2"].interface_energy_J_m2,
        )
        assert metallic > passive


class TestHydrogenKnockdown:
    def test_unity_without_hydrogen(self):
        assert hydrogen_toughness_knockdown(0.0) == pytest.approx(1.0)

    def test_monotonically_decreasing(self):
        vals = [hydrogen_toughness_knockdown(c) for c in (0.0, 1.0, 10.0, 100.0)]
        assert all(a >= b for a, b in zip(vals, vals[1:]))

    def test_respects_the_floor(self):
        assert hydrogen_toughness_knockdown(1e6) == pytest.approx(0.15)

    def test_varies_across_the_range_iron_deposits_occupy(self):
        """A floor reached at 1 ppm would make hydrogen a no-op."""
        assert (
            hydrogen_toughness_knockdown(1.0)
            > hydrogen_toughness_knockdown(10.0)
            > hydrogen_toughness_knockdown(50.0)
        )

    def test_never_exceeds_unity(self):
        assert hydrogen_toughness_knockdown(0.001) <= 1.0

    def test_rejects_negative_hydrogen(self):
        with pytest.raises(ValueError):
            hydrogen_toughness_knockdown(-1.0)


class TestConfinedPlasticity:
    def test_saturates_at_the_plastic_zone_thickness(self):
        assert confined_plastic_amplification(
            25.0, PLASTIC_ZONE_THICKNESS_UM
        ) == pytest.approx(25.0)

    def test_never_exceeds_the_unconfined_value(self):
        assert confined_plastic_amplification(
            25.0, 10 * PLASTIC_ZONE_THICKNESS_UM
        ) == pytest.approx(25.0)

    def test_thin_films_collapse_toward_work_of_adhesion(self):
        """A vanishingly thin film has no room for a plastic zone."""
        assert confined_plastic_amplification(25.0, 1e-6) == pytest.approx(
            1.0, abs=1e-3
        )

    def test_increases_with_thickness(self):
        a = confined_plastic_amplification(25.0, 5.0)
        b = confined_plastic_amplification(25.0, 25.0)
        assert 1.0 < a < b

    def test_rejects_amplification_below_one(self):
        with pytest.raises(ValueError):
            confined_plastic_amplification(0.5, 25.0)


class TestInterfacialToughness:
    def test_exceeds_the_thermodynamic_floor(self):
        t = interfacial_toughness(SUBSTRATES["ti_passive_tio2"])
        assert t["toughness_J_m2"] > t["work_of_adhesion_J_m2"]

    def test_hydrogen_reduces_toughness(self):
        dry = interfacial_toughness(SUBSTRATES["ti_passive_tio2"], C_H_ppm=0.0)
        wet = interfacial_toughness(SUBSTRATES["ti_passive_tio2"], C_H_ppm=50.0)
        assert wet["toughness_J_m2"] < dry["toughness_J_m2"]

    def test_roughness_raises_toughness(self):
        smooth = SUBSTRATES["ti_passive_tio2"]
        rough = SubstrateSpec(
            **{**smooth.__dict__, "id": "rough", "roughness_Ra_um": 3.0}
        )
        assert (
            interfacial_toughness(rough)["toughness_J_m2"]
            > interfacial_toughness(smooth)["toughness_J_m2"]
        )

    def test_thicker_films_are_harder_to_peel(self):
        thin = interfacial_toughness(
            SUBSTRATES["ti_passive_tio2"], thickness_um=2.0
        )
        thick = interfacial_toughness(
            SUBSTRATES["ti_passive_tio2"], thickness_um=100.0
        )
        assert thick["toughness_J_m2"] > thin["toughness_J_m2"]

    def test_amplification_override_is_respected(self):
        base = interfacial_toughness(
            SUBSTRATES["ti_passive_tio2"], thickness_um=200.0
        )
        hi = interfacial_toughness(
            SUBSTRATES["ti_passive_tio2"], thickness_um=200.0,
            amplification_override=200.0,
        )
        assert hi["toughness_J_m2"] > base["toughness_J_m2"]

    def test_metallic_substrates_are_far_tougher(self):
        """The screen is useless if it cannot separate these."""
        passive = interfacial_toughness(SUBSTRATES["ti_passive_tio2"])
        metallic = interfacial_toughness(SUBSTRATES["ti_bare_etched"])
        assert metallic["toughness_J_m2"] > 10 * passive["toughness_J_m2"]


# ═══════════════════════════════════════════════════════════════════
#  Peel mechanics
# ═══════════════════════════════════════════════════════════════════

class TestPeelMechanics:
    def test_ninety_degree_peel_equals_toughness(self):
        """At θ = 90°, (1 − cos θ) = 1 and P/b = Γ numerically."""
        assert peel_force_per_width(5.0, 90.0) == pytest.approx(5.0)

    def test_one_eighty_degree_peel_halves_the_force(self):
        assert peel_force_per_width(5.0, 180.0) == pytest.approx(2.5)

    def test_shallow_angles_need_much_more_force(self):
        assert peel_force_per_width(5.0, 10.0) > peel_force_per_width(5.0, 90.0)

    def test_residual_energy_assists_the_peel(self):
        assert peel_force_per_width(5.0, 90.0, residual_G_J_m2=3.0) == pytest.approx(2.0)

    def test_residual_energy_exceeding_toughness_gives_zero_force(self):
        assert peel_force_per_width(5.0, 90.0, residual_G_J_m2=9.0) == 0.0

    def test_rejects_bad_angle(self):
        with pytest.raises(ValueError):
            peel_force_per_width(5.0, 0.0)

    def test_web_stress_matches_definition(self):
        assert web_stress_MPa(100.0, 25.0) == pytest.approx(100.0 / 25e-6 / 1e6)

    def test_web_stress_rises_as_foil_thins(self):
        """Why thin foil tears: the same force acts on less cross-section."""
        assert web_stress_MPa(50.0, 5.0) > web_stress_MPa(50.0, 50.0)

    def test_film_tearing_energy_scales_with_thickness(self):
        a = film_tearing_energy_J_m2(400.0, 0.05, 10.0)
        b = film_tearing_energy_J_m2(400.0, 0.05, 20.0)
        assert b == pytest.approx(2.0 * a)

    def test_film_tearing_energy_falls_with_embrittlement(self):
        ductile = film_tearing_energy_J_m2(400.0, 0.20, 25.0)
        brittle = film_tearing_energy_J_m2(400.0, 0.01, 25.0)
        assert brittle < ductile

    def test_film_tearing_rejects_bad_elongation(self):
        with pytest.raises(ValueError):
            film_tearing_energy_J_m2(400.0, 1.5, 25.0)


# ═══════════════════════════════════════════════════════════════════
#  Conditions
# ═══════════════════════════════════════════════════════════════════

class TestPeelConditions:
    def test_defaults_match_the_drum_target_thickness(self):
        """cell_architecture drum_and_strip targets 25 µm foil."""
        assert PeelConditions().thickness_um == pytest.approx(25.0)

    def test_rejects_nonpositive_thickness(self):
        with pytest.raises(ValueError):
            PeelConditions(thickness_um=0.0)

    def test_rejects_bad_safety_factor(self):
        with pytest.raises(ValueError):
            PeelConditions(web_stress_safety_factor=1.5)

    def test_rejects_bad_elongation(self):
        with pytest.raises(ValueError):
            PeelConditions(foil_elongation_fraction=0.0)


# ═══════════════════════════════════════════════════════════════════
#  Evaluation and classification
# ═══════════════════════════════════════════════════════════════════

class TestEvaluatePeel:
    def test_returns_a_known_outcome(self):
        r = evaluate_peel(SUBSTRATES["ti_passive_tio2"])
        assert r.outcome in {
            "bonded_no_release", "cohesive_failure_in_film",
            "tears_before_peel", "marginal_peel", "clean_peel",
            "spontaneous_delamination",
        }

    def test_always_gives_a_reason(self):
        for s in SUBSTRATES.values():
            assert evaluate_peel(s).reasons

    def test_self_release_ratio_is_G_over_gamma(self):
        r = evaluate_peel(SUBSTRATES["ti_passive_tio2"])
        assert r.self_release_ratio == pytest.approx(
            r.driving_force_J_m2 / r.toughness_J_m2
        )

    def test_metallic_control_does_not_peel(self):
        """Iron bonds to copper. A screen that says otherwise is broken."""
        r = evaluate_peel(SUBSTRATES["copper_substrate"])
        assert not r.peelable

    def test_passive_oxide_beats_bare_metal(self):
        """The physical claim underneath the whole drum concept."""
        passive = evaluate_peel(SUBSTRATES["ti_passive_tio2"])
        bare = evaluate_peel(SUBSTRATES["ti_bare_etched"])
        assert passive.peel_force_N_per_m < bare.peel_force_N_per_m

    def test_insulating_substrate_is_flagged_even_when_it_releases(self):
        r = evaluate_peel(SUBSTRATES["ptfe_release_coating"])
        assert not r.conductive
        assert any("insulating" in x for x in r.reasons)

    def test_very_thick_deposits_self_release(self):
        r = evaluate_peel(
            SUBSTRATES["ti_passive_tio2"], PeelConditions(thickness_um=2000.0)
        )
        assert r.outcome == "spontaneous_delamination"
        assert r.good_for_flake_harvest

    def test_high_hydrogen_drives_release(self):
        low = evaluate_peel(
            SUBSTRATES["ti_passive_tio2"], PeelConditions(C_H_ppm=0.1)
        )
        high = evaluate_peel(
            SUBSTRATES["ti_passive_tio2"], PeelConditions(C_H_ppm=200.0)
        )
        assert high.self_release_ratio > low.self_release_ratio

    def test_strongly_bonded_interface_is_rejected_not_praised(self):
        """A huge amplification must not read as 'good adhesion, fine'."""
        r = evaluate_peel(
            SUBSTRATES["ti_passive_tio2"], amplification_override=200.0
        )
        assert r.outcome in ("bonded_no_release", "cohesive_failure_in_film",
                             "tears_before_peel", "marginal_peel", "clean_peel")

    def test_cohesive_failure_when_interface_outruns_the_film(self):
        """A weak, brittle foil on a tough interface must fail cohesively."""
        weak_foil = PeelConditions(
            thickness_um=3.0,
            foil_yield_strength_MPa=50.0,
            foil_elongation_fraction=0.001,
            max_winder_tension_N_per_m=1e9,
        )
        r = evaluate_peel(SUBSTRATES["ti_bare_etched"], weak_foil)
        assert r.outcome in ("cohesive_failure_in_film", "tears_before_peel")

    def test_tear_before_peel_for_a_weak_web(self):
        cond = PeelConditions(
            thickness_um=1.0,
            foil_yield_strength_MPa=5.0,
            foil_elongation_fraction=0.5,
        )
        r = evaluate_peel(SUBSTRATES["stainless_316_passive"], cond)
        assert r.outcome in ("tears_before_peel", "bonded_no_release")

    def test_to_dict_is_json_safe(self):
        import json

        for s in SUBSTRATES.values():
            json.dumps(evaluate_peel(s).to_dict())

    def test_summary_is_a_nonempty_string(self):
        assert len(evaluate_peel(SUBSTRATES["ti_passive_tio2"]).summary()) > 50


# ═══════════════════════════════════════════════════════════════════
#  Substrate library
# ═══════════════════════════════════════════════════════════════════

class TestSubstrateLibrary:
    def test_every_substrate_declares_provenance_and_evidence(self):
        for s in SUBSTRATES.values():
            assert s.provenance
            assert s.evidence_level in ("commercial", "pilot", "lab", "concept")

    def test_every_substrate_declares_limitations(self):
        for s in SUBSTRATES.values():
            assert s.limitations

    def test_ids_are_self_consistent(self):
        for key, s in SUBSTRATES.items():
            assert key == s.id

    def test_metallic_bonding_implies_higher_amplification(self):
        metallic = [s for s in SUBSTRATES.values() if s.bonding == "metallic"]
        passive = [s for s in SUBSTRATES.values() if s.bonding == "oxide_passive"]
        assert min(s.plastic_amplification for s in metallic) > max(
            s.plastic_amplification for s in passive
        )

    def test_library_contains_a_negative_control(self):
        """Without a known-adherent control the screen cannot be falsified."""
        assert any(s.bonding == "metallic" for s in SUBSTRATES.values())

    def test_library_contains_a_nonconductive_case(self):
        assert any(not s.electrically_conductive for s in SUBSTRATES.values())

    def test_spec_rejects_amplification_below_one(self):
        with pytest.raises(ValueError):
            SubstrateSpec(
                **{**SUBSTRATES["ti_passive_tio2"].__dict__,
                   "plastic_amplification": 0.5}
            )


# ═══════════════════════════════════════════════════════════════════
#  Screen, sweeps, verdict
# ═══════════════════════════════════════════════════════════════════

class TestScreen:
    def test_screen_covers_every_substrate(self):
        assert len(screen_substrates()) == len(SUBSTRATES)

    def test_conductive_substrates_rank_ahead_of_insulating_ones(self):
        results = screen_substrates()
        first_insulating = next(
            i for i, r in enumerate(results) if not r.conductive
        )
        assert all(r.conductive for r in results[:first_insulating])

    def test_screen_discriminates_between_substrates(self):
        """If every surface gets the same verdict the model says nothing."""
        outcomes = {r.outcome for r in screen_substrates()}
        assert len(outcomes) >= 3

    def test_comparison_table_lists_all_rows(self):
        table = comparison_table(screen_substrates())
        for s in SUBSTRATES.values():
            assert s.name[:20] in table


class TestSweeps:
    def test_thickness_sweep_driving_force_is_monotonic(self):
        sw = thickness_sweep(
            SUBSTRATES["ti_passive_tio2"], np.array([5.0, 25.0, 100.0])
        )
        G = sw["driving_force_J_m2"]
        assert G[0] < G[1] < G[2]

    def test_thickness_sweep_finds_a_bounded_window(self):
        sw = thickness_sweep(SUBSTRATES["ti_passive_tio2"])
        assert sw["viable_thickness_max_um"] is not None
        assert sw["viable_thickness_max_um"] < max(sw["thickness_um"])

    def test_thick_enough_always_self_releases(self):
        sw = thickness_sweep(
            SUBSTRATES["ti_passive_tio2"], np.array([10.0, 5000.0])
        )
        assert sw["outcome"][-1] == "spontaneous_delamination"

    def test_hydrogen_sweep_toughness_decreases(self):
        sw = hydrogen_sweep(
            SUBSTRATES["ti_passive_tio2"], np.array([0.1, 5.0, 40.0])
        )
        t = sw["toughness_J_m2"]
        assert t[0] > t[1] > t[2]

    def test_hydrogen_sweep_release_ratio_increases(self):
        sw = hydrogen_sweep(
            SUBSTRATES["ti_passive_tio2"], np.array([0.1, 10.0, 100.0])
        )
        r = sw["self_release_ratio"]
        assert r[0] < r[1] < r[2]

    def test_grain_refinement_raises_stress_and_release(self):
        """The conflict with mechanical_properties, quantified."""
        sw = grain_size_sweep(
            SUBSTRATES["ti_passive_tio2"], np.array([0.1, 1.0, 5.0])
        )
        assert sw["residual_stress_MPa"][0] > sw["residual_stress_MPa"][-1]
        assert sw["self_release_ratio"][0] > sw["self_release_ratio"][-1]

    def test_amplification_robustness_fractions_sum_to_one(self):
        rob = amplification_robustness(SUBSTRATES["ti_passive_tio2"])
        assert sum(rob["outcome_fractions"].values()) == pytest.approx(1.0)

    def test_amplification_robustness_reports_a_dominant_outcome(self):
        rob = amplification_robustness(SUBSTRATES["ti_passive_tio2"])
        assert rob["dominant_outcome"] in rob["outcome_fractions"]
        assert isinstance(rob["verdict_robust"], bool)

    def test_copper_never_self_releases_at_any_amplification(self):
        """The negative control must never look like a release surface.

        Its verdict is legitimately *not* robust — at a low enough plastic
        amplification even copper peels — and the model is right to say so.
        What must hold at every amplification is that iron on copper never
        falls off by itself.
        """
        rob = amplification_robustness(SUBSTRATES["copper_substrate"])
        assert "spontaneous_delamination" not in rob["outcome_fractions"]

    def test_copper_is_bonded_at_and_above_the_library_estimate(self):
        cu = SUBSTRATES["copper_substrate"]
        for phi in (cu.plastic_amplification, 2 * cu.plastic_amplification):
            r = evaluate_peel(cu, amplification_override=phi)
            assert not r.peelable


class TestFoilRouteVerdict:
    def test_returns_a_known_verdict(self):
        v = foil_route_verdict()
        assert v["verdict"] in (
            "proceed", "proceed_with_coupon_test",
            "pivot_to_flake_harvest", "no_go",
        )

    def test_verdict_carries_an_interpretation(self):
        assert len(foil_route_verdict()["interpretation"]) > 50

    def test_verdict_is_json_safe(self):
        import json

        json.dumps(foil_route_verdict())

    def test_thick_deposits_redirect_to_flake_harvest(self):
        """The Option-A fallback the program already names as primary."""
        v = foil_route_verdict(PeelConditions(thickness_um=3000.0))
        assert v["verdict"] in ("pivot_to_flake_harvest",
                                "proceed_with_coupon_test")
        assert v["outcome"] == "spontaneous_delamination"

    def test_alternatives_exclude_the_reference_substrate(self):
        v = foil_route_verdict()
        assert "ti_passive_tio2" not in v["peelable_alternatives"]


# ═══════════════════════════════════════════════════════════════════
#  Integration with the deposition models
# ═══════════════════════════════════════════════════════════════════

class TestConditionsFromDeposition:
    def test_thickness_follows_faradays_law(self):
        """0.85 FE at 100 mA/cm² for 900 s ≈ 28 µm of iron."""
        d = conditions_from_deposition(
            j_mA_cm2=100.0, current_efficiency_percent=85.0,
            deposition_time_s=900.0,
        )
        assert 20.0 < d["derived"]["thickness_um"] < 40.0

    def test_thickness_scales_with_time(self):
        a = conditions_from_deposition(deposition_time_s=900.0)
        b = conditions_from_deposition(deposition_time_s=1800.0)
        assert b["derived"]["thickness_um"] == pytest.approx(
            2.0 * a["derived"]["thickness_um"]
        )

    def test_lower_efficiency_means_more_hydrogen(self):
        good = conditions_from_deposition(current_efficiency_percent=95.0)
        bad = conditions_from_deposition(current_efficiency_percent=40.0)
        assert (
            bad["derived"]["C_H_diffusible_ppm"]
            > good["derived"]["C_H_diffusible_ppm"]
        )

    def test_pulse_refines_grain_relative_to_dc(self):
        dc = conditions_from_deposition(waveform="dc")
        pre = conditions_from_deposition(
            waveform="pre", j_peak_mA_cm2=300.0, duty_cycle=0.33
        )
        assert pre["derived"]["grain_size_um"] < dc["derived"]["grain_size_um"]

    def test_conditions_feed_straight_into_evaluate_peel(self):
        d = conditions_from_deposition()
        r = evaluate_peel(SUBSTRATES["ti_passive_tio2"], d["conditions"])
        assert r.outcome

    def test_sources_are_named_for_traceability(self):
        d = conditions_from_deposition()
        assert "mechanical_properties" in d["sources"]["grain_size"]
        assert "hydrogen_embrittlement" in d["sources"]["hydrogen"]

    def test_rejects_bad_efficiency(self):
        with pytest.raises(ValueError):
            conditions_from_deposition(current_efficiency_percent=0.0)

    def test_rejects_nonpositive_current(self):
        with pytest.raises(ValueError):
            conditions_from_deposition(j_mA_cm2=0.0)


# ═══════════════════════════════════════════════════════════════════
#  Protocol and scope
# ═══════════════════════════════════════════════════════════════════

class TestCouponProtocol:
    def test_only_conductive_coupons_are_specified(self):
        p = coupon_test_protocol()
        ids = {c["substrate"] for c in p["coupons"]}
        assert "ptfe_release_coating" not in ids

    def test_includes_a_negative_control(self):
        roles = {c["role"] for c in coupon_test_protocol()["coupons"]}
        assert any("control" in r for r in roles)

    def test_every_measurement_names_what_it_replaces(self):
        for m in coupon_test_protocol()["measurements"]:
            assert m["replaces_in_model"]

    def test_decision_rules_can_kill_and_confirm(self):
        rules = coupon_test_protocol()["decision_rules"]
        assert "kills_foil_branch" in rules
        assert "confirms_foil_branch" in rules
        assert "redirects_to_flake" in rules

    def test_cost_total_matches_its_line_items(self):
        cost = coupon_test_protocol()["estimated_cost_usd"]
        lines = {k: v for k, v in cost.items() if k != "total"}
        assert cost["total"] == pytest.approx(sum(lines.values()))

    def test_protocol_is_json_safe(self):
        import json

        json.dumps(coupon_test_protocol())


class TestModelScope:
    def test_declares_no_wet_lab_data(self):
        assert "no wet-lab" in model_scope()["provenance"].lower()

    def test_names_its_key_uncertainty(self):
        assert "amplification" in model_scope()["key_uncertainty"]

    def test_lists_what_it_does_not_compute(self):
        assert len(model_scope()["does_not_compute"]) >= 4

    def test_lists_required_calibration(self):
        assert len(model_scope()["calibration_required"]) >= 4


# ═══════════════════════════════════════════════════════════════════
#  Cross-model consistency
# ═══════════════════════════════════════════════════════════════════

class TestCrossModelConsistency:
    def test_iron_surface_energy_matches_deposit_morphology(self):
        from models.deposit_morphology import GAMMA_FE_SURFACE

        assert GAMMA_FE_J_M2 == pytest.approx(GAMMA_FE_SURFACE)

    def test_iron_density_matches_electrochemistry(self):
        from models.adhesion_peel import RHO_FE as RHO_LOCAL
        from models.electrochemistry import RHO_FE as RHO_SHARED

        assert RHO_LOCAL == pytest.approx(RHO_SHARED)

    def test_default_thickness_matches_the_drum_architecture_target(self):
        from models.cell_architecture import ARCHITECTURES

        assert PeelConditions().thickness_um == pytest.approx(
            ARCHITECTURES["drum_and_strip"].target_deposit_thickness_um
        )

    def test_this_module_answers_what_cell_architecture_declined_to(self):
        """The two scope contracts must stay complementary, not overlapping."""
        from models.cell_architecture import model_scope as arch_scope

        declined = " ".join(arch_scope()["does_not_compute"]).lower()
        assert "adhesion" in declined or "peel" in declined
        computed = " ".join(model_scope()["computes"]).lower()
        assert "peel" in computed

    def test_winder_ceiling_is_a_plausible_web_tension(self):
        """A ceiling far off foil-line practice would rig every verdict."""
        assert 50.0 <= MAX_WINDER_TENSION_N_PER_M <= 2000.0
