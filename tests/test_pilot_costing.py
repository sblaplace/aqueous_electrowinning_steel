"""Tests for pilot-plant CAPEX/OPEX model."""

import pytest
from models.pilot_costing import (
    six_tenths_scale,
    estimate_capex,
    capex_at_all_scales,
    capex_by_category,
    capex_sensitivity_tornado,
    PilotOPEXModel,
    PID_EQUIPMENT,
    SCALE_PILOT,
    equipment_table,
)


class TestSixTenthsScale:
    """Six-tenths rule scaling."""

    def test_identity_at_reference(self):
        """Cost is unchanged at the reference scale."""
        assert six_tenths_scale(10_000, 10, 10) == pytest.approx(10_000)

    def test_monotonic_increase(self):
        """Larger scale → higher cost."""
        costs = [six_tenths_scale(10_000, 10, s) for s in [1, 10, 100]]
        assert costs[0] < costs[1] < costs[2]

    def test_exponent_effect(self):
        """Higher exponent → steeper scaling."""
        cost_low = six_tenths_scale(10_000, 10, 100, exponent=0.4)
        cost_high = six_tenths_scale(10_000, 10, 100, exponent=0.8)
        assert cost_high > cost_low

    def test_invalid_scale_raises(self):
        with pytest.raises(ValueError):
            six_tenths_scale(10_000, 0, 10)
        with pytest.raises(ValueError):
            six_tenths_scale(10_000, 10, -1)


class TestCAPEXEstimate:
    """CAPEX estimation at various scales."""

    def test_pilot_total_positive(self):
        result = estimate_capex(SCALE_PILOT)
        assert result.total_capex > 0

    def test_all_equipment_positive(self):
        result = estimate_capex(SCALE_PILOT)
        for tag, cost in result.equipment.items():
            assert cost > 0, f"Equipment {tag} has non-positive cost: {cost}"

    def test_pilot_total_in_range(self):
        """Pilot CAPEX should be 100k–1M range."""
        result = estimate_capex(SCALE_PILOT)
        assert 100_000 <= result.total_capex <= 1_000_000

    def test_piping_is_30pct_of_equipment(self):
        result = estimate_capex(SCALE_PILOT)
        assert result.piping_structural == pytest.approx(0.30 * result.subtotal_equipment)

    def test_components_sum_to_total(self):
        result = estimate_capex(SCALE_PILOT)
        summed = result.subtotal_equipment + result.piping_structural + result.engineering + result.contingency
        assert summed == pytest.approx(result.total_capex, rel=1e-6)

    def test_all_scales_monotonic(self):
        """CAPEX increases monotonically with scale."""
        scales = capex_at_all_scales()
        assert scales["lab"].total_capex < scales["pilot"].total_capex
        assert scales["pilot"].total_capex < scales["production"].total_capex


class TestCAPEXCategories:
    """Category aggregation."""

    def test_categories_cover_all_equipment(self):
        result = estimate_capex(SCALE_PILOT)
        cats = capex_by_category(result)
        cat_sum = sum(cats.values())
        assert cat_sum == pytest.approx(result.subtotal_equipment, rel=1e-6)

    def test_cell_is_largest_category(self):
        result = estimate_capex(SCALE_PILOT)
        cats = capex_by_category(result)
        assert cats["cell"] == max(cats.values())


class TestOPEX:
    """OPEX model tests."""

    def test_opex_positive(self):
        capex = estimate_capex(SCALE_PILOT)
        opex = PilotOPEXModel()
        result = opex.estimate(SCALE_PILOT, capex)
        assert result["Total OPEX ($/yr)"] > 0

    def test_opex_components_sum(self):
        capex = estimate_capex(SCALE_PILOT)
        opex = PilotOPEXModel()
        result = opex.estimate(SCALE_PILOT, capex)
        assert result["Variable OPEX ($/yr)"] + result["Consumables ($/yr)"] + result["Fixed OPEX ($/yr)"] == \
               pytest.approx(result["Total OPEX ($/yr)"], rel=1e-3)

    def test_specific_opex_reasonable(self):
        """Specific OPEX should be $1–100/kg Fe (pilot scale is expensive per kg)."""
        capex = estimate_capex(SCALE_PILOT)
        opex = PilotOPEXModel()
        result = opex.estimate(SCALE_PILOT, capex)
        assert 1.0 <= result["Specific OPEX ($/kg Fe)"] <= 100.0

    def test_gas_cost_positive(self):
        opex = PilotOPEXModel()
        assert opex.gas_cost_per_batch() > 0

    def test_furnace_electricity_positive(self):
        opex = PilotOPEXModel()
        assert opex.furnace_electricity_per_batch() > 0

    def test_instrument_maintenance_5pct(self):
        """Instrument maintenance = 5% of instrument CAPEX."""
        capex = estimate_capex(SCALE_PILOT)
        opex = PilotOPEXModel()
        result = opex.estimate(SCALE_PILOT, capex)
        inst_capex = capex.equipment.get("INST", 0)
        assert result["Instrument maintenance ($/yr)"] == pytest.approx(0.05 * inst_capex)


class TestSensitivity:
    """Sensitivity / tornado analysis."""

    def test_tornado_has_base_and_params(self):
        sens = capex_sensitivity_tornado()
        assert "base" in sens
        assert "cell_current_low" in sens
        assert "production_scale_high" in sens

    def test_production_scale_low_is_cheaper(self):
        sens = capex_sensitivity_tornado()
        assert sens["production_scale_low"]["CAPEX"] < sens["base"]["CAPEX"]

    def test_production_scale_high_is_costlier(self):
        sens = capex_sensitivity_tornado()
        assert sens["production_scale_high"]["CAPEX"] > sens["base"]["CAPEX"]


class TestEquipmentTable:
    """Equipment list helpers."""

    def test_table_length(self):
        table = equipment_table()
        assert len(table) == len(PID_EQUIPMENT)

    def test_all_tags_present(self):
        table = equipment_table()
        tags = {e["tag"] for e in table}
        expected = {e.tag for e in PID_EQUIPMENT}
        assert tags == expected
