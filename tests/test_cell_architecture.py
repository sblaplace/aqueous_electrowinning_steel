"""Tests for the cell architecture screening model.

The model's job is to make kill criterion #3 computable: given a reactor
architecture, what areal productivity does it reach and what may the cell
cost per m²?  These tests pin the dimensionless-group arithmetic against
analytic values, check the physical monotonicities, and guard the specific
modelling decisions that make the comparison honest (footprint ceilings for
3-D electrodes, harvest downtime that scales with plating rate).
"""

import math

import numpy as np
import pytest

from models.cell_architecture import (
    ARCHITECTURES,
    IRON_PRODUCT_VALUE_PER_T,
    ZINC_TANKHOUSE,
    ArchitectureSpec,
    OperatingConditions,
    areal_productivity_t_m2_yr,
    capital_charge_per_t_fe,
    capital_recovery_factor,
    comparison_table,
    compare_architectures,
    concentration_sweep,
    deposition_rate_um_hr,
    evaluate_architecture,
    kill_criterion_assessment,
    limiting_current_from_km,
    mass_transfer_coefficient,
    max_affordable_cost_per_m2,
    model_scope,
    reynolds_number,
    schmidt_number,
    sherwood_number,
    velocity_sweep,
    zinc_tankhouse_productivity,
)
from models.electrochemistry import FARADAY, M_FE, Z_FE


# ═══════════════════════════════════════════════════════════════════
#  Dimensionless groups
# ═══════════════════════════════════════════════════════════════════

class TestDimensionlessGroups:
    def test_schmidt_matches_definition(self):
        assert schmidt_number(1e-6, 1e-9) == pytest.approx(1000.0)

    def test_schmidt_for_iron_sulfate_is_order_1000(self):
        """Sc = ν/D for Fe²⁺ in water is ~1400."""
        Sc = schmidt_number()
        assert 500 < Sc < 5000

    def test_schmidt_rejects_zero_diffusivity(self):
        with pytest.raises(ValueError):
            schmidt_number(1e-6, 0.0)

    def test_reynolds_matches_definition(self):
        assert reynolds_number(2.0, 0.05, 1e-6) == pytest.approx(1.0e5)

    def test_reynolds_uses_absolute_velocity(self):
        assert reynolds_number(-1.0, 0.1, 1e-6) == reynolds_number(1.0, 0.1, 1e-6)

    def test_reynolds_rejects_zero_viscosity(self):
        with pytest.raises(ValueError):
            reynolds_number(1.0, 0.1, 0.0)

    def test_sherwood_power_law(self):
        # 0.023 · 10000^0.8 · 1000^0.33
        expected = 0.023 * (10000 ** 0.8) * (1000 ** 0.33)
        assert sherwood_number(10000, 1000, 0.023, 0.8, 0.33) == pytest.approx(expected)

    def test_sherwood_additive_term_is_the_stagnant_limit(self):
        """Ranz-Marshall reduces to Sh = 2 for a sphere at rest."""
        assert sherwood_number(0.0, 1000, 0.6, 0.5, 0.33, additive=2.0) == pytest.approx(2.0)

    def test_sherwood_rejects_negative_reynolds(self):
        with pytest.raises(ValueError):
            sherwood_number(-1.0, 1000, 0.023, 0.8, 0.33)

    def test_mass_transfer_coefficient_definition(self):
        assert mass_transfer_coefficient(100.0, 0.01, 1e-9) == pytest.approx(1e-5)

    def test_mass_transfer_coefficient_rejects_zero_length(self):
        with pytest.raises(ValueError):
            mass_transfer_coefficient(100.0, 0.0)


class TestLimitingCurrent:
    def test_matches_z_f_km_c(self):
        """i_lim = zFk_mC, with C converted mol/L → mol/m³."""
        k_m, C = 1e-5, 1.0
        expected = Z_FE * FARADAY * k_m * C * 1000.0
        assert limiting_current_from_km(k_m, C) == pytest.approx(expected)

    def test_linear_in_concentration(self):
        a = limiting_current_from_km(1e-5, 1.0)
        b = limiting_current_from_km(1e-5, 2.0)
        assert b == pytest.approx(2.0 * a)

    def test_zero_concentration_gives_zero_current(self):
        assert limiting_current_from_km(1e-5, 0.0) == 0.0


# ═══════════════════════════════════════════════════════════════════
#  Productivity arithmetic
# ═══════════════════════════════════════════════════════════════════

class TestArealProductivity:
    def test_reproduces_program_doc_figure(self):
        """docs/RESEARCH_PROGRAM.md: 100 mA/cm² at 85% FE ≈ 7.8 t/(m²·yr)."""
        p = areal_productivity_t_m2_yr(1000.0, 0.85, capacity_factor=1.0)
        assert p == pytest.approx(7.8, abs=0.3)

    def test_matches_faraday_law_directly(self):
        j, fe = 500.0, 0.9
        expected = j * fe * M_FE / (Z_FE * FARADAY) * 3600.0 * 8760.0 / 1000.0
        assert areal_productivity_t_m2_yr(j, fe) == pytest.approx(expected)

    def test_linear_in_current_density(self):
        a = areal_productivity_t_m2_yr(500.0)
        b = areal_productivity_t_m2_yr(1000.0)
        assert b == pytest.approx(2.0 * a)

    def test_capacity_factor_scales_output(self):
        full = areal_productivity_t_m2_yr(1000.0, 0.85, capacity_factor=1.0)
        half = areal_productivity_t_m2_yr(1000.0, 0.85, capacity_factor=0.5)
        assert half == pytest.approx(0.5 * full)

    def test_zero_current_gives_zero(self):
        assert areal_productivity_t_m2_yr(0.0) == 0.0

    def test_rejects_negative_current(self):
        with pytest.raises(ValueError):
            areal_productivity_t_m2_yr(-1.0)

    def test_deposition_rate_is_positive_and_scales(self):
        r1 = deposition_rate_um_hr(1000.0)
        r2 = deposition_rate_um_hr(2000.0)
        assert r1 > 0 and r2 == pytest.approx(2.0 * r1)

    def test_deposition_rate_magnitude_is_plausible(self):
        """1000 A/m² at 85% FE ≈ 110 µm/hr for iron."""
        assert 50 < deposition_rate_um_hr(1000.0, 0.85) < 200


class TestCapitalRecoveryFactor:
    def test_known_value(self):
        """8% over 25 years ≈ 0.0937."""
        assert capital_recovery_factor(0.08, 25) == pytest.approx(0.0937, abs=1e-4)

    def test_zero_discount_is_straight_line(self):
        assert capital_recovery_factor(0.0, 20) == pytest.approx(0.05)

    def test_increases_with_discount_rate(self):
        assert capital_recovery_factor(0.12, 25) > capital_recovery_factor(0.05, 25)

    def test_decreases_with_lifetime(self):
        assert capital_recovery_factor(0.08, 40) < capital_recovery_factor(0.08, 10)

    def test_rejects_zero_lifetime(self):
        with pytest.raises(ValueError):
            capital_recovery_factor(0.08, 0)


# ═══════════════════════════════════════════════════════════════════
#  Architecture registry integrity
# ═══════════════════════════════════════════════════════════════════

class TestRegistry:
    def test_all_named_candidates_present(self):
        """The four architectures the program doc asks to paper-study, plus
        the plate-and-frame baseline."""
        for key in (
            "plate_and_frame",
            "rotating_cylinder",
            "drum_and_strip",
            "moving_belt",
            "fluidized_bed",
        ):
            assert key in ARCHITECTURES

    def test_ids_match_keys(self):
        for key, spec in ARCHITECTURES.items():
            assert spec.id == key

    def test_costs_and_geometry_are_positive(self):
        for spec in ARCHITECTURES.values():
            assert spec.direct_cost_per_m2 > 0
            assert spec.characteristic_length_m > 0
            assert spec.default_velocity_m_s > 0
            assert spec.active_area_ratio > 0
            assert spec.flow_enhancement_factor >= 1.0

    def test_every_architecture_declares_limitations(self):
        """No architecture may be presented without its known weaknesses."""
        for spec in ARCHITECTURES.values():
            assert spec.limitations, f"{spec.id} has no declared limitations"
            assert spec.notes

    def test_evidence_levels_are_valid(self):
        valid = {"commercial", "pilot", "lab", "concept"}
        for spec in ARCHITECTURES.values():
            assert spec.evidence_level in valid

    def test_moving_belt_is_marked_concept(self):
        """It has no iron demonstration; the model must not imply otherwise."""
        assert ARCHITECTURES["moving_belt"].evidence_level == "concept"

    def test_plate_and_frame_matches_technoeconomic_defaults(self):
        """The baseline must not silently diverge from CAPEXModel."""
        from models.technoeconomic import CAPEXModel

        capex = CAPEXModel()
        spec = ARCHITECTURES["plate_and_frame"]
        assert spec.electrode_cost_per_m2 == capex.electrode_cost_per_m2
        assert spec.separator_cost_per_m2 == capex.membrane_separator_cost_per_m2
        assert spec.hardware_cost_per_m2 == capex.cell_hardware_cost_per_m2

    def test_cost_breakdown_sums_to_direct_total(self):
        for spec in ARCHITECTURES.values():
            b = spec.cost_breakdown()
            parts = (
                b["electrodes_per_m2"]
                + b["separator_per_m2"]
                + b["hardware_per_m2"]
                + b["harvesting_per_m2"]
            )
            assert parts == pytest.approx(b["direct_total_per_m2"])

    def test_continuous_flag_agrees_with_harvest_mode(self):
        for spec in ARCHITECTURES.values():
            assert spec.is_continuous == (spec.harvest_mode == "continuous")

    def test_continuous_architectures_have_no_harvest_downtime(self):
        for spec in ARCHITECTURES.values():
            if spec.is_continuous:
                assert spec.harvest_downtime_hr == 0.0


# ═══════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════

class TestEvaluateArchitecture:
    def test_returns_finite_positive_metrics(self):
        for spec in ARCHITECTURES.values():
            r = evaluate_architecture(spec)
            assert r.transport_limit_A_m2 > 0
            assert r.j_operating_A_m2 > 0
            assert r.areal_productivity_t_m2_yr > 0
            assert math.isfinite(r.capital_charge_per_t_fe)
            assert r.capital_charge_per_t_fe > 0

    def test_operating_point_respects_both_ceilings(self):
        for spec in ARCHITECTURES.values():
            r = evaluate_architecture(spec)
            assert r.j_operating_A_m2 <= spec.max_practical_j_A_m2 + 1e-6
            if spec.max_footprint_current_A_m2 is not None:
                assert r.j_installed_A_m2 <= spec.max_footprint_current_A_m2 + 1e-6

    def test_limited_by_is_reported_and_valid(self):
        valid = {"transport", "practical_ceiling", "footprint_ceiling"}
        for spec in ARCHITECTURES.values():
            assert evaluate_architecture(spec).limited_by in valid

    def test_fluidized_bed_is_footprint_limited(self):
        """The bed's binding constraint is potential distribution through the
        bed depth, not film transport. If this ever flips to 'transport' the
        area accounting has broken."""
        r = evaluate_architecture(ARCHITECTURES["fluidized_bed"])
        assert r.limited_by == "footprint_ceiling"
        assert r.j_installed_A_m2 == pytest.approx(1000.0)

    def test_fluidized_bed_footprint_current_is_physical(self):
        """Without the footprint cap the 600x area ratio implies ~18 kA/m²,
        which no real bed achieves."""
        r = evaluate_architecture(ARCHITECTURES["fluidized_bed"])
        assert r.j_installed_A_m2 < 2000.0

    def test_drum_operating_range_matches_industry(self):
        """Cu-foil drums run 30-120 A/dm² = 300-1200 mA/cm² = 3-12 kA/m²."""
        r = evaluate_architecture(ARCHITECTURES["drum_and_strip"])
        assert 2000.0 <= r.j_operating_A_m2 <= 12000.0

    def test_rotating_cylinder_beats_plate_on_transport(self):
        """Turbulent RCE transport is the documented reason to consider it."""
        rce = evaluate_architecture(ARCHITECTURES["rotating_cylinder"])
        plate = evaluate_architecture(ARCHITECTURES["plate_and_frame"])
        assert rce.transport_limit_A_m2 > plate.transport_limit_A_m2

    def test_velocity_increases_transport_limit(self):
        spec = ARCHITECTURES["plate_and_frame"]
        slow = evaluate_architecture(spec, velocity_m_s=0.05)
        fast = evaluate_architecture(spec, velocity_m_s=0.5)
        assert fast.transport_limit_A_m2 > slow.transport_limit_A_m2

    def test_concentration_increases_transport_limit(self):
        spec = ARCHITECTURES["plate_and_frame"]
        lean = evaluate_architecture(spec, OperatingConditions(fe_conc_M=0.5))
        rich = evaluate_architecture(spec, OperatingConditions(fe_conc_M=2.0))
        assert rich.transport_limit_A_m2 > lean.transport_limit_A_m2

    def test_flow_enhancement_raises_transport_limit(self):
        base = ARCHITECTURES["plate_and_frame"]
        enhanced = ArchitectureSpec(**{**base.__dict__, "flow_enhancement_factor": 4.0})
        r0 = evaluate_architecture(base)
        r1 = evaluate_architecture(enhanced)
        assert r1.transport_limit_A_m2 == pytest.approx(4.0 * r0.transport_limit_A_m2)

    def test_active_area_ratio_scales_footprint_current(self):
        r = evaluate_architecture(ARCHITECTURES["drum_and_strip"])
        spec = ARCHITECTURES["drum_and_strip"]
        assert r.j_installed_A_m2 == pytest.approx(
            r.j_operating_A_m2 * spec.active_area_ratio
        )

    def test_to_dict_is_json_serializable(self):
        import json

        for spec in ARCHITECTURES.values():
            json.dumps(evaluate_architecture(spec).to_dict())

    def test_summary_mentions_the_architecture(self):
        r = evaluate_architecture(ARCHITECTURES["rotating_cylinder"])
        assert "Rotating cylinder" in r.summary()
        assert "capital charge" in r.summary()


class TestHarvestCycle:
    def test_batch_capacity_factor_below_base_availability(self):
        spec = ARCHITECTURES["plate_and_frame"]
        r = evaluate_architecture(spec)
        assert r.capacity_factor < spec.base_availability

    def test_continuous_keeps_full_base_availability(self):
        for spec in ARCHITECTURES.values():
            if spec.is_continuous:
                r = evaluate_architecture(spec)
                assert r.capacity_factor == pytest.approx(spec.base_availability)

    def test_batch_reports_a_plating_cycle(self):
        r = evaluate_architecture(ARCHITECTURES["plate_and_frame"])
        assert r.plating_cycle_hr is not None and r.plating_cycle_hr > 0

    def test_continuous_reports_no_cycle(self):
        r = evaluate_architecture(ARCHITECTURES["rotating_cylinder"])
        assert r.plating_cycle_hr is None

    def test_faster_plating_costs_a_batch_cell_more_uptime(self):
        """The central argument for continuous harvesting: raising current
        density shortens the plating cycle, so downtime eats a larger share
        of the calendar."""
        spec = ARCHITECTURES["plate_and_frame"]
        slow = evaluate_architecture(spec, velocity_m_s=0.02)
        fast = evaluate_architecture(spec, velocity_m_s=1.0)
        assert fast.j_operating_A_m2 > slow.j_operating_A_m2
        assert fast.capacity_factor < slow.capacity_factor

    def test_thicker_target_deposit_improves_duty_cycle(self):
        base = ARCHITECTURES["plate_and_frame"]
        thick = ArchitectureSpec(
            **{**base.__dict__, "target_deposit_thickness_um": 5000.0}
        )
        assert (
            evaluate_architecture(thick).capacity_factor
            > evaluate_architecture(base).capacity_factor
        )


# ═══════════════════════════════════════════════════════════════════
#  Comparison and kill criterion
# ═══════════════════════════════════════════════════════════════════

class TestComparison:
    def test_returns_every_architecture(self):
        assert len(compare_architectures()) == len(ARCHITECTURES)

    def test_sorted_by_capital_charge(self):
        charges = [r.capital_charge_per_t_fe for r in compare_architectures()]
        assert charges == sorted(charges)

    def test_subset_selection(self):
        res = compare_architectures(architecture_ids=["plate_and_frame"])
        assert len(res) == 1 and res[0].architecture_id == "plate_and_frame"

    def test_table_renders_all_rows(self):
        table = comparison_table(compare_architectures())
        for spec in ARCHITECTURES.values():
            assert spec.name[:20] in table

    def test_table_distinguishes_active_and_footprint_current(self):
        table = comparison_table(compare_architectures())
        assert "j act." in table and "j ftpt." in table


class TestKillCriterion:
    def test_threshold_formula(self):
        """$/m²_max = budget × productivity / CRF."""
        crf = capital_recovery_factor(0.08, 25)
        assert max_affordable_cost_per_m2(10.0, 50.0, 0.08, 25) == pytest.approx(
            50.0 * 10.0 / crf
        )

    def test_threshold_rises_with_productivity(self):
        assert max_affordable_cost_per_m2(20.0, 50.0) > max_affordable_cost_per_m2(5.0, 50.0)

    def test_threshold_rises_with_budget(self):
        assert max_affordable_cost_per_m2(10.0, 100.0) > max_affordable_cost_per_m2(10.0, 50.0)

    def test_zero_productivity_affords_nothing(self):
        assert max_affordable_cost_per_m2(0.0, 50.0) == 0.0

    def test_threshold_and_charge_are_mutually_inverse(self):
        """A cell priced exactly at the threshold must land on the budget."""
        productivity, budget = 12.0, 45.0
        threshold = max_affordable_cost_per_m2(productivity, budget)
        assert capital_charge_per_t_fe(threshold, productivity) == pytest.approx(budget)

    def test_capital_charge_infinite_without_production(self):
        assert math.isinf(capital_charge_per_t_fe(1000.0, 0.0))

    def test_assessment_covers_all_architectures(self):
        a = kill_criterion_assessment()
        assert len(a["architectures"]) == len(ARCHITECTURES)

    def test_assessment_verdicts_are_consistent(self):
        for v in kill_criterion_assessment()["architectures"]:
            assert v["passes"] == (
                v["installed_cost_per_m2"] <= v["max_affordable_cost_per_m2"]
            )
            assert v["verdict"] == ("within budget" if v["passes"] else "exceeds budget")

    def test_headroom_sign_matches_verdict(self):
        for v in kill_criterion_assessment()["architectures"]:
            assert (v["headroom_per_m2"] >= 0) == v["passes"]

    def test_tighter_budget_never_passes_more_architectures(self):
        loose = kill_criterion_assessment(capital_charge_budget_per_t_fe=200.0)
        tight = kill_criterion_assessment(capital_charge_budget_per_t_fe=5.0)
        assert tight["n_passing"] <= loose["n_passing"]

    def test_assessment_is_json_serializable(self):
        import json

        json.dumps(kill_criterion_assessment())

    def test_best_is_the_lowest_capital_charge(self):
        a = kill_criterion_assessment()
        charges = [v["capital_charge_per_t_fe"] for v in a["architectures"]]
        assert a["best"]["capital_charge_per_t_fe"] == min(charges)


class TestZincBenchmark:
    def test_iron_equivalent_productivity_is_plausible(self):
        """500 A/m² making iron instead of zinc: ~4 t/(m²·yr)."""
        p = zinc_tankhouse_productivity()
        assert 2.0 < p < 6.0

    def test_benchmark_constants_present(self):
        for key in (
            "current_density_A_m2",
            "capex_per_annual_tonne_low",
            "product_value_per_t_low",
        ):
            assert key in ZINC_TANKHOUSE

    def test_iron_is_worth_less_than_zinc(self):
        """The whole difficulty in one assertion: same machine, cheaper product."""
        assert IRON_PRODUCT_VALUE_PER_T["high"] < ZINC_TANKHOUSE["product_value_per_t_low"]

    def test_assessment_reports_productivity_ratio_vs_zinc(self):
        for v in kill_criterion_assessment()["architectures"]:
            assert v["productivity_vs_zinc"] is not None
            assert v["productivity_vs_zinc"] > 0

    def test_continuous_harvesting_reaches_the_5x_target(self):
        """The program's stated requirement is ~5x zinc areal productivity.
        At least one continuous architecture must clear it, or the whole
        architecture thesis fails."""
        a = kill_criterion_assessment()
        ratios = [
            v["productivity_vs_zinc"]
            for v in a["architectures"]
            if ARCHITECTURES[v["architecture_id"]].is_continuous
        ]
        assert max(ratios) >= 5.0


# ═══════════════════════════════════════════════════════════════════
#  Sweeps
# ═══════════════════════════════════════════════════════════════════

class TestSweeps:
    def test_velocity_sweep_shape(self):
        s = velocity_sweep(ARCHITECTURES["plate_and_frame"], np.array([0.05, 0.1, 0.2]))
        assert len(s["velocity_m_s"]) == 3
        assert len(s["areal_productivity_t_m2_yr"]) == 3

    def test_velocity_sweep_transport_limit_is_monotonic(self):
        s = velocity_sweep(ARCHITECTURES["plate_and_frame"], np.linspace(0.02, 1.0, 12))
        assert np.all(np.diff(s["transport_limit_A_m2"]) > 0)

    def test_velocity_sweep_saturates_at_practical_ceiling(self):
        """Past some speed, pumping harder stops buying current."""
        spec = ARCHITECTURES["rotating_cylinder"]
        s = velocity_sweep(spec, np.logspace(-1, 1.5, 25))
        assert "practical_ceiling" in s["limited_by"]
        assert max(s["j_operating_A_m2"]) <= spec.max_practical_j_A_m2 + 1e-6

    def test_concentration_sweep_is_monotonic_in_transport(self):
        s = concentration_sweep(ARCHITECTURES["plate_and_frame"], np.linspace(0.2, 2.0, 10))
        assert np.all(np.diff(s["transport_limit_A_m2"]) > 0)

    def test_concentration_sweep_lowers_unit_capital_charge(self):
        """More iron in solution → more output per m² → cheaper capital per tonne."""
        s = concentration_sweep(ARCHITECTURES["plate_and_frame"], np.array([0.25, 2.5]))
        assert s["capital_charge_per_t_fe"][1] < s["capital_charge_per_t_fe"][0]

    def test_sweeps_are_json_serializable(self):
        import json

        json.dumps(velocity_sweep(ARCHITECTURES["moving_belt"], np.array([0.1, 0.2])))
        json.dumps(concentration_sweep(ARCHITECTURES["moving_belt"], np.array([0.5, 1.5])))


# ═══════════════════════════════════════════════════════════════════
#  Scope declaration
# ═══════════════════════════════════════════════════════════════════

class TestModelScope:
    def test_declares_no_wet_lab_data(self):
        scope = model_scope()
        assert "no wet-lab" in scope["provenance"].lower()

    def test_disclaims_faradaic_efficiency_and_voltage(self):
        text = " ".join(model_scope()["does_not_compute"]).lower()
        assert "faradaic efficiency" in text
        assert "cell voltage" in text

    def test_disclaims_adhesion_the_gating_drum_unknown(self):
        text = " ".join(model_scope()["does_not_compute"]).lower()
        assert "adhesion" in text or "peelab" in text

    def test_lists_calibration_requirements(self):
        assert len(model_scope()["calibration_required"]) >= 3

    def test_drum_limitations_flag_peelability_risk(self):
        text = " ".join(ARCHITECTURES["drum_and_strip"].limitations).lower()
        assert "peel" in text
