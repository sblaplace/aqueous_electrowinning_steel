"""Tests for the literature-anchored operating scenarios.

The scenario set is quoted directly in ``docs/PROGRAM_SUMMARY.md`` as the
program's decision-grade position.  These tests keep the code and that table
from drifting apart, and check that the cell-voltage decomposition behaves
like an electrochemical cell rather than a lookup table.
"""

import pytest

from models.electrochemistry import specific_energy_kWh_per_t
from models.scenarios import (
    ALL_SCENARIOS,
    AWARE_ACIDIC,
    CONSERVATIVE_ALKALINE,
    FUTURE_TARGET,
    OPTIMIZED_ALKALINE,
    Scenario,
)


class TestScenarioSet:
    def test_four_scenarios_defined(self):
        assert len(ALL_SCENARIOS) == 4

    def test_names_are_unique(self):
        names = [s.name for s in ALL_SCENARIOS]
        assert len(set(names)) == len(names)

    def test_all_carry_references(self):
        """Every scenario must say where its numbers come from."""
        for s in ALL_SCENARIOS:
            assert s.references.strip()

    def test_operating_parameters_are_physical(self):
        for s in ALL_SCENARIOS:
            assert s.current_density_mA_cm2 > 0
            assert 0.0 < s.current_efficiency <= 1.0
            assert 0.0 < s.temperature_C < 150.0
            assert s.electricity_price_kWh > 0

    def test_electrolyte_types_recognized(self):
        for s in ALL_SCENARIOS:
            assert s.electrolyte_type in {"alkaline", "acidic_anion_rich", "acidic"}


class TestCellVoltage:
    def test_all_voltages_positive_and_plausible(self):
        for s in ALL_SCENARIOS:
            assert 0.5 < s.V_cell < 5.0

    def test_decomposition_sums_correctly(self):
        """V_cell = |E_anode − E_cathode| + η_c + η_a + IR."""
        for s in ALL_SCENARIOS:
            expected = (
                abs(s._effective_anode_eq - s.E_cathode_eq)
                + s.eta_cathode
                + s._effective_eta_anode
                + s.ir_drop
            )
            assert s.V_cell == pytest.approx(expected)

    def test_matches_program_summary_table(self):
        """docs/PROGRAM_SUMMARY.md quotes these to three decimals."""
        assert OPTIMIZED_ALKALINE.V_cell == pytest.approx(1.418, abs=0.002)
        assert AWARE_ACIDIC.V_cell == pytest.approx(2.485, abs=0.002)
        assert FUTURE_TARGET.V_cell == pytest.approx(2.441, abs=0.002)

    def test_more_ir_drop_raises_voltage(self):
        base = CONSERVATIVE_ALKALINE
        worse = Scenario(**{**base.__dict__, "ir_drop": base.ir_drop + 0.5})
        assert worse.V_cell == pytest.approx(base.V_cell + 0.5)

    def test_anode_summary_present_only_with_a_model(self):
        for s in ALL_SCENARIOS:
            summary = s.anode_summary
            assert isinstance(summary, dict)
            if s.anode is None:
                assert summary == {}


class TestSpecificEnergy:
    def test_matches_program_summary_energies(self):
        """E = 959.9 × V/FE kWh/t Fe, as quoted in PROGRAM_SUMMARY.md."""
        for scenario, expected in (
            (OPTIMIZED_ALKALINE, 1464),
            (AWARE_ACIDIC, 2410),
            (FUTURE_TARGET, 2415),
        ):
            e = specific_energy_kWh_per_t(scenario.V_cell, scenario.current_efficiency)
            assert e == pytest.approx(expected, rel=0.01)

    def test_all_scenarios_beat_dri_h2_on_dc_energy(self):
        """The headline claim: DC electrolysis energy under ~3,300 kWh/t."""
        for s in ALL_SCENARIOS:
            e = specific_energy_kWh_per_t(s.V_cell, s.current_efficiency)
            assert e < 3300.0

    def test_energy_rises_when_efficiency_falls(self):
        base = OPTIMIZED_ALKALINE
        poor = Scenario(**{**base.__dict__, "current_efficiency": 0.60})
        assert specific_energy_kWh_per_t(
            poor.V_cell, poor.current_efficiency
        ) > specific_energy_kWh_per_t(base.V_cell, base.current_efficiency)

    def test_kill_criterion_threshold_arithmetic(self):
        """4,000 kWh/t at FE = 70% corresponds to V_cell ≈ 2.92 V."""
        assert specific_energy_kWh_per_t(2.92, 0.70) == pytest.approx(4000, rel=0.01)


class TestScenarioOrdering:
    def test_aware_runs_the_highest_current_density(self):
        assert AWARE_ACIDIC.current_density_mA_cm2 == max(
            s.current_density_mA_cm2 for s in ALL_SCENARIOS
        )

    def test_aware_claims_the_highest_efficiency(self):
        assert AWARE_ACIDIC.current_efficiency == max(
            s.current_efficiency for s in ALL_SCENARIOS
        )

    def test_optimized_alkaline_has_the_lowest_voltage(self):
        """Alkaline's thermodynamic advantage is the reason to consider it."""
        assert OPTIMIZED_ALKALINE.V_cell == min(s.V_cell for s in ALL_SCENARIOS)

    def test_future_target_is_more_aggressive_than_conservative(self):
        assert (
            FUTURE_TARGET.current_density_mA_cm2
            > CONSERVATIVE_ALKALINE.current_density_mA_cm2
        )
        assert FUTURE_TARGET.current_efficiency > CONSERVATIVE_ALKALINE.current_efficiency

    def test_future_target_assumes_cheaper_capital_and_power(self):
        assert FUTURE_TARGET.capex_modifier <= CONSERVATIVE_ALKALINE.capex_modifier
        assert FUTURE_TARGET.electricity_price_kWh <= CONSERVATIVE_ALKALINE.electricity_price_kWh
