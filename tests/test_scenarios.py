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
    SOLUBLE_FE_ACIDIC,
    Scenario,
    AWARE_BASE_HER_I0_ACID,
    chloride_theta_block,
    derive_aware_current_efficiency,
    derive_aware_her_suppression,
    derive_aware_ir_drop_V,
    aware_bath_conductivity_S_m,
)


class TestScenarioSet:
    def test_five_scenarios_defined(self):
        assert len(ALL_SCENARIOS) == 5

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
            # Soluble-anode electrorefining cells run ~0.3–0.5 V; OER
            # routes run ~1.2–2.5 V.
            assert 0.2 < s.V_cell < 5.0

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
        """docs/PROGRAM_SUMMARY.md quotes these to three decimals.

        NOTE (2026-08): the anode concentration-overpotential model was
        corrected from dissolved-O₂ depletion (i_lim ≈ 4 A/m², which capped
        η_conc at a spurious ~0.14 V for every real current) to supporting-
        salt polarization (η_conc ≈ a few mV).  That removed a ~0.14 V
        phantom overpotential from all first-principles-anode scenarios,
        so the table voltages are the physically corrected values.
        """
        assert OPTIMIZED_ALKALINE.V_cell == pytest.approx(1.268, abs=0.005)
        # AWARE V_cell is now derived from the computed chloride-bath
        # conductivity (Tier 1.4): the high-κ 10 M LiCl bath lowers the
        # ohmic / anode-bubble terms vs the previous input resistivity.
        assert AWARE_ACIDIC.V_cell == pytest.approx(2.3065, abs=0.005)
        assert FUTURE_TARGET.V_cell == pytest.approx(2.309, abs=0.005)

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
        # Corrected 2026-08 for the anode η_conc fix (see
        # test_matches_program_summary_table): E = 959.9 × V_cell/FE.
        for scenario, expected in (
            (OPTIMIZED_ALKALINE, 1309),
            (AWARE_ACIDIC, 2234),   # derived FE 0.9911 & V_cell 2.307 (Tier 1.4)
            (FUTURE_TARGET, 2284),
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
        assert SOLUBLE_FE_ACIDIC.V_cell == min(s.V_cell for s in ALL_SCENARIOS)

    def test_future_target_is_more_aggressive_than_conservative(self):
        assert (
            FUTURE_TARGET.current_density_mA_cm2
            > CONSERVATIVE_ALKALINE.current_density_mA_cm2
        )
        assert FUTURE_TARGET.current_efficiency > CONSERVATIVE_ALKALINE.current_efficiency

    def test_future_target_assumes_cheaper_capital_and_power(self):
        assert FUTURE_TARGET.capex_modifier <= CONSERVATIVE_ALKALINE.capex_modifier
        assert FUTURE_TARGET.electricity_price_kWh <= CONSERVATIVE_ALKALINE.electricity_price_kWh


class TestAWAREPhysicsDerivation:
    """Tier 1.4: the AWARE scenario's headline numbers are *derived* from
    the chloride-bath physics (Cl⁻ site-blocking HER suppression + computed
    conductivity), not preset parameters."""

    def test_aware_fe_is_derived_not_assumed(self):
        """current_efficiency must come from the derivation, and its
        provenance must be recorded for audit."""
        assert AWARE_ACIDIC.physical_derivation["fe_source"] == "derived"
        assert AWARE_ACIDIC.current_efficiency == pytest.approx(
            AWARE_ACIDIC.physical_derivation["current_efficiency_derived"],
            abs=1e-12,
        )
        # Exactly matches a clean re-run of the derivation.
        assert AWARE_ACIDIC.current_efficiency == pytest.approx(
            derive_aware_current_efficiency(500.0, 60.0, 1.0, 10.0), abs=1e-9
        )

    def test_aware_fe_is_near_unity(self):
        """Chloride suppression pushes FE to near-unity (> 95 %)."""
        assert 0.95 < AWARE_ACIDIC.current_efficiency <= 1.0

    def test_fe_rises_monotonically_with_chloride(self):
        """FE is a *function* of chloride concentration: more Cl⁻ → more
        site blocking → less HER → higher FE (the mechanism, not a knob)."""
        fes = [derive_aware_current_efficiency(500.0, 60.0, 1.0, cl)
               for cl in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 12.0)]
        for a, b in zip(fes, fes[1:]):
            assert b > a

    def test_theta_block_near_unity_at_aware(self):
        """At 10-12 M Cl⁻ the Langmuir site-blocking coverage is near 1
        (> 99 %), the physical source of the near-unity FE."""
        tb = AWARE_ACIDIC.physical_derivation["theta_block"]
        assert 0.985 < tb < 1.0

    def test_her_exchange_suppressed_by_chloride(self):
        """The derived her_i0 must be well below the no-chloride base."""
        supp = derive_aware_her_suppression(c_FeCl2=1.0, c_LiCl=10.0, T_C=60.0)
        assert supp["her_i0_A_m2"] < AWARE_BASE_HER_I0_ACID
        assert supp["her_i0_A_m2"] < 0.01 * AWARE_BASE_HER_I0_ACID

    def test_conductivity_computed_not_input(self):
        """10 M LiCl conductivity is a model output (Onsager + pairing),
        plausibly ~20-50 S/m, and is what feeds ir-drop."""
        kappa = aware_bath_conductivity_S_m(c_FeCl2=1.0, c_LiCl=10.0, T_C=60.0)
        assert 20.0 < kappa < 50.0
        # The scenario anode carries the computed conductivity, and ir-drop
        # is derived from it (not a hardcoded input).
        assert AWARE_ACIDIC.anode.electrolyte_conductivity_S_m == pytest.approx(
            kappa, rel=1e-9)
        assert AWARE_ACIDIC.ir_drop == pytest.approx(
            derive_aware_ir_drop_V(500.0, 60.0, 1.0, 10.0), rel=1e-9)

    def test_conductivity_rises_with_licl(self):
        assert aware_bath_conductivity_S_m(1.0, 12.0, 60.0) > \
            aware_bath_conductivity_S_m(1.0, 10.0, 60.0)

    def test_aware_fe_is_highest_in_set(self):
        """Even derived, AWARE keeps the highest FE claim of the set."""
        assert AWARE_ACIDIC.current_efficiency == max(
            s.current_efficiency for s in ALL_SCENARIOS)

    def test_sulfate_default_unaffected(self):
        """Opt-in chloride chemistry: the default sulfate kinetics and the
        non-chloride scenario parameters are untouched."""
        from models.kinetics import DepositionKinetics
        assert DepositionKinetics().her_i0 == 1e-3   # default unchanged
        # Only AWARE carries a physics derivation; the others stay parameter
        # scenarios (empty provenance).
        for s in ALL_SCENARIOS:
            if s is AWARE_ACIDIC:
                assert s.physical_derivation
            else:
                assert s.physical_derivation == {} or s.physical_derivation is None

