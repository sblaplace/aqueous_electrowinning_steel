"""Tests for the techno-economic model.

``technoeconomic.py`` is load-bearing: every kill criterion and scenario
comparison in the program routes through LCOFe.  It previously had no direct
test coverage.  These tests pin the Faraday and discounting arithmetic against
closed-form values and check that the cost model responds to its drivers in
the direction physics requires.
"""

import math

import pytest

from models.electrochemistry import (
    FARADAY,
    M_FE,
    Z_FE,
    specific_energy_kWh_per_t,
)
from models.technoeconomic import (
    BENCHMARK_COSTS,
    CAPEXModel,
    ElectrolyzerParams,
    LevelizedCost,
    OPEXModel,
    compare_routes,
    sensitivity_analysis,
)


@pytest.fixture
def params():
    return ElectrolyzerParams()


@pytest.fixture
def priced(params):
    """Base-case CAPEX/OPEX/LCOFe bundle at the default 10 stacks."""
    capex = CAPEXModel().estimate(params, n_stacks=10)
    opex = OPEXModel().estimate(params, capex["Total CAPEX ($)"], n_stacks=10)
    lcofe = LevelizedCost().calculate(
        capex["Total CAPEX ($)"],
        opex["Total OPEX ($/yr)"],
        capex["Annual capacity (t/yr)"],
    )
    return capex, opex, lcofe


# ═══════════════════════════════════════════════════════════════════
#  Electrolyzer parameters
# ═══════════════════════════════════════════════════════════════════

class TestElectrolyzerParams:
    def test_total_current_from_density_and_area(self):
        """100 mA/cm² = 1000 A/m²; over 1 m² that is 1000 A."""
        p = ElectrolyzerParams(current_density_mA_cm2=100.0, electrode_area_m2=1.0)
        assert p.total_current_A() == pytest.approx(1000.0)

    def test_stack_voltage_is_series_sum(self):
        p = ElectrolyzerParams(cell_voltage=2.5, n_cells=100)
        assert p.stack_voltage_V() == pytest.approx(250.0)

    def test_stack_power_is_current_times_voltage(self, params):
        expected = params.total_current_A() * params.stack_voltage_V() / 1000.0
        assert params.stack_power_kW() == pytest.approx(expected)

    def test_production_rate_obeys_faraday(self):
        """ṁ = I·FE·M/(zF) per cell, summed over the stack."""
        p = ElectrolyzerParams(
            current_density_mA_cm2=100.0,
            electrode_area_m2=1.0,
            current_efficiency=0.90,
            n_cells=10,
        )
        per_cell = 1000.0 * 0.90 * M_FE / (Z_FE * FARADAY) * 3600.0
        assert p.production_rate_kg_hr() == pytest.approx(per_cell * 10, rel=1e-6)

    def test_annual_production_scales_with_hours(self, params):
        a = params.production_rate_t_yr(operating_hours=8000.0)
        b = params.production_rate_t_yr(operating_hours=4000.0)
        assert a == pytest.approx(2.0 * b)

    def test_zero_efficiency_produces_nothing(self):
        p = ElectrolyzerParams(current_efficiency=0.0)
        assert p.production_rate_kg_hr() == 0.0


# ═══════════════════════════════════════════════════════════════════
#  CAPEX
# ═══════════════════════════════════════════════════════════════════

class TestCAPEXModel:
    def test_reports_all_expected_lines(self, priced):
        capex, _, _ = priced
        for key in (
            "Electrodes ($)",
            "Membranes/separators ($)",
            "Cell hardware ($)",
            "Stack subtotal ($)",
            "Rectifiers ($)",
            "Total CAPEX ($)",
            "Annual capacity (t/yr)",
        ):
            assert key in capex

    def test_total_is_positive_and_finite(self, priced):
        capex, _, _ = priced
        assert capex["Total CAPEX ($)"] > 0
        assert math.isfinite(capex["Total CAPEX ($)"])

    def test_area_counts_both_electrode_faces(self, params):
        """area_per_stack = area × n_cells × 2."""
        capex = CAPEXModel().estimate(params, n_stacks=1)
        area = params.electrode_area_m2 * params.n_cells * 2
        assert capex["Electrodes ($)"] == pytest.approx(150.0 * area)

    def test_electrode_cost_scales_with_unit_price(self, params):
        cheap = CAPEXModel(electrode_cost_per_m2=100.0).estimate(params, 5)
        dear = CAPEXModel(electrode_cost_per_m2=200.0).estimate(params, 5)
        assert dear["Electrodes ($)"] == pytest.approx(2.0 * cheap["Electrodes ($)"])

    def test_more_stacks_cost_more_and_produce_more(self, params):
        small = CAPEXModel().estimate(params, n_stacks=5)
        large = CAPEXModel().estimate(params, n_stacks=20)
        assert large["Total CAPEX ($)"] > small["Total CAPEX ($)"]
        assert large["Annual capacity (t/yr)"] > small["Annual capacity (t/yr)"]

    def test_millions_field_is_consistent(self, priced):
        capex, _, _ = priced
        assert capex["Total CAPEX (M$)"] == pytest.approx(
            capex["Total CAPEX ($)"] / 1e6, abs=0.01
        )

    def test_total_exceeds_direct_costs(self, priced):
        """Indirects and contingency must add to, never subtract from, direct."""
        capex, _, _ = priced
        direct = capex["Stack subtotal ($)"] + capex["BOP subtotal ($)"]
        assert capex["Total CAPEX ($)"] > direct


# ═══════════════════════════════════════════════════════════════════
#  OPEX
# ═══════════════════════════════════════════════════════════════════

class TestOPEXModel:
    def test_reports_all_expected_lines(self, priced):
        _, opex, _ = priced
        for key in (
            "Electricity ($/yr)",
            "Variable OPEX ($/yr)",
            "Fixed OPEX ($/yr)",
            "Total OPEX ($/yr)",
            "Specific energy (kWh/t Fe)",
        ):
            assert key in opex

    def test_specific_energy_matches_the_governing_formula(self, params, priced):
        """E = 959.9 × V/FE kWh/t Fe — the number the whole program turns on."""
        _, opex, _ = priced
        expected = specific_energy_kWh_per_t(
            params.cell_voltage, params.current_efficiency
        )
        assert opex["Specific energy (kWh/t Fe)"] == pytest.approx(expected, abs=1.0)
        analytic = 959.9 * params.cell_voltage / params.current_efficiency
        assert opex["Specific energy (kWh/t Fe)"] == pytest.approx(analytic, rel=0.01)

    def test_electricity_cost_scales_with_price(self, params, priced):
        capex, _, _ = priced
        cheap = OPEXModel(electricity_price_kWh=0.02).estimate(
            params, capex["Total CAPEX ($)"], 10
        )
        dear = OPEXModel(electricity_price_kWh=0.08).estimate(
            params, capex["Total CAPEX ($)"], 10
        )
        # Reported values are rounded to whole dollars, so allow $1 of slack.
        assert dear["Electricity ($/yr)"] == pytest.approx(
            4.0 * cheap["Electricity ($/yr)"], abs=4.0
        )

    def test_grinding_is_a_subset_of_electricity(self, priced):
        _, opex, _ = priced
        assert 0 < opex["  of which grinding ($/yr)"] < opex["Electricity ($/yr)"]

    def test_lower_efficiency_raises_specific_energy(self, params, priced):
        capex, _, _ = priced
        good = ElectrolyzerParams(**{**params.__dict__, "current_efficiency": 0.95})
        poor = ElectrolyzerParams(**{**params.__dict__, "current_efficiency": 0.60})
        o_good = OPEXModel().estimate(good, capex["Total CAPEX ($)"], 10)
        o_poor = OPEXModel().estimate(poor, capex["Total CAPEX ($)"], 10)
        assert o_poor["Specific energy (kWh/t Fe)"] > o_good["Specific energy (kWh/t Fe)"]

    def test_total_is_variable_plus_fixed(self, priced):
        _, opex, _ = priced
        assert opex["Total OPEX ($/yr)"] == pytest.approx(
            opex["Variable OPEX ($/yr)"] + opex["Fixed OPEX ($/yr)"], rel=1e-6
        )

    def test_maintenance_scales_with_capex(self, params):
        low = OPEXModel().estimate(params, 1e6, 10)
        high = OPEXModel().estimate(params, 1e8, 10)
        assert high["Maintenance ($/yr)"] > low["Maintenance ($/yr)"]


# ═══════════════════════════════════════════════════════════════════
#  Levelized cost
# ═══════════════════════════════════════════════════════════════════

class TestLevelizedCost:
    def test_crf_matches_closed_form(self):
        """LCOFe = (CRF·CAPEX + OPEX)/production, CRF = r(1+r)^n/((1+r)^n−1).

        LCOFe is reported rounded to whole dollars, so the identity is checked
        to $1; the capital recovery factor itself is checked at full precision.
        """
        lc = LevelizedCost(plant_lifetime_yr=25, discount_rate=0.08)
        capex, opex, prod = 1e8, 1e7, 50000.0
        r, n = 0.08, 25
        crf = r * (1 + r) ** n / ((1 + r) ** n - 1)
        result = lc.calculate(capex, opex, prod)
        assert result["Capital recovery factor"] == pytest.approx(crf, abs=5e-5)
        expected = (crf * capex + opex) / prod
        assert result["LCOFe ($/t Fe)"] == pytest.approx(expected, abs=1.0)

    def test_higher_discount_rate_raises_lcofe(self):
        cheap = LevelizedCost(discount_rate=0.04).calculate(1e8, 1e7, 50000.0)
        dear = LevelizedCost(discount_rate=0.15).calculate(1e8, 1e7, 50000.0)
        assert dear["LCOFe ($/t Fe)"] > cheap["LCOFe ($/t Fe)"]

    def test_longer_life_lowers_lcofe(self):
        short = LevelizedCost(plant_lifetime_yr=10).calculate(1e8, 1e7, 50000.0)
        long = LevelizedCost(plant_lifetime_yr=40).calculate(1e8, 1e7, 50000.0)
        assert long["LCOFe ($/t Fe)"] < short["LCOFe ($/t Fe)"]

    def test_more_production_lowers_unit_cost(self):
        lc = LevelizedCost()
        assert (
            lc.calculate(1e8, 1e7, 100000.0)["LCOFe ($/t Fe)"]
            < lc.calculate(1e8, 1e7, 50000.0)["LCOFe ($/t Fe)"]
        )

    def test_base_case_lcofe_is_positive_and_finite(self, priced):
        _, _, lcofe = priced
        v = lcofe["LCOFe ($/t Fe)"]
        assert v > 0 and math.isfinite(v)


# ═══════════════════════════════════════════════════════════════════
#  Sensitivity and benchmarking
# ═══════════════════════════════════════════════════════════════════

class TestSensitivityAnalysis:
    def test_includes_base_and_swept_parameters(self, params):
        s = sensitivity_analysis(params, n_stacks=5)
        assert "base" in s
        assert any("Current efficiency" in k for k in s)
        assert any("Cell voltage" in k for k in s)
        assert any("Electricity price" in k for k in s)

    def test_higher_cell_voltage_costs_more(self, params):
        s = sensitivity_analysis(params, n_stacks=5)
        assert s["Cell voltage (V)_high"]["LCOFe"] > s["Cell voltage (V)_low"]["LCOFe"]

    def test_higher_current_efficiency_costs_less(self, params):
        s = sensitivity_analysis(params, n_stacks=5)
        assert s["Current efficiency_high"]["LCOFe"] < s["Current efficiency_low"]["LCOFe"]

    def test_cheaper_electricity_costs_less(self, params):
        s = sensitivity_analysis(params, n_stacks=5)
        assert (
            s["Electricity price_low"]["LCOFe"] < s["Electricity price_high"]["LCOFe"]
        )

    def test_all_lcofe_values_finite(self, params):
        for v in sensitivity_analysis(params, n_stacks=5).values():
            assert math.isfinite(v["LCOFe"])


class TestRouteComparison:
    def test_covers_all_benchmarks_plus_our_route(self):
        c = compare_routes(400.0)
        for route in BENCHMARK_COSTS:
            assert route in c
        assert "Aqueous Electrowinning" in c

    def test_carbon_price_penalizes_bf_bof_most(self):
        """BF-BOF carries 1.8 t CO₂/t Fe; a carbon price should hit it hardest."""
        low = compare_routes(400.0, carbon_price_tCO2=0.0)
        high = compare_routes(400.0, carbon_price_tCO2=200.0)
        bf_delta = (
            high["BF-BOF"]["Adjusted cost ($/t Fe)"]
            - low["BF-BOF"]["Adjusted cost ($/t Fe)"]
        )
        aq_delta = (
            high["Aqueous Electrowinning"]["Adjusted cost ($/t Fe)"]
            - low["Aqueous Electrowinning"]["Adjusted cost ($/t Fe)"]
        )
        assert bf_delta > aq_delta

    def test_zero_carbon_price_leaves_base_cost_unchanged(self):
        c = compare_routes(400.0, carbon_price_tCO2=0.0)
        assert c["BF-BOF"]["Adjusted cost ($/t Fe)"] == pytest.approx(
            BENCHMARK_COSTS["BF-BOF"]["mid"]
        )

    def test_benchmark_ranges_are_ordered(self):
        for data in BENCHMARK_COSTS.values():
            assert data["low"] < data["mid"] < data["high"]

    def test_our_route_reports_the_supplied_lcofe(self):
        c = compare_routes(377.0)
        assert c["Aqueous Electrowinning"]["Base cost ($/t Fe)"] == 377.0
