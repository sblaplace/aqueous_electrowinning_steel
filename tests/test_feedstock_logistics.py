"""Tests for supply chain model — materials, recycling, siting."""

import pytest
from models.feedstock_logistics import (
    DesignPoint,
    material_balance,
    electrolyte_recycling,
    site_score,
    compare_locations,
    electricity_sensitivity,
    CANDIDATE_LOCATIONS,
    LocationData,
    LocationWeight,
)


class TestDesignPoint:
    """DesignPoint operating parameters."""

    def test_production_rate_positive(self):
        dp = DesignPoint()
        assert dp.production_rate_kg_hr() > 0

    def test_energy_per_kg_reasonable(self):
        dp = DesignPoint()
        energy = dp.energy_kWh_per_kg()
        # Typical aqueous electrowinning: 3-10 kWh/kg
        assert 1.0 < energy < 20.0

    def test_production_scales_with_cells(self):
        dp10 = DesignPoint(n_cells=10)
        dp20 = DesignPoint(n_cells=20)
        assert dp20.production_rate_kg_hr() == pytest.approx(2 * dp10.production_rate_kg_hr())


class TestMaterialBalance:
    """Feedstock material balance."""

    def test_all_materials_present(self):
        dp = DesignPoint(carburize_with_BaCO3=True)
        bal = material_balance(dp, 100.0)
        names = [it.name for it in bal.items]
        assert "Ferrous sulfate" in names
        assert "Sulfuric acid" in names
        assert "Sodium hydroxide" in names
        assert "Barium carbonate" in names
        assert "Carbon black" in names

    def test_feso4_stoichiometry(self):
        dp = DesignPoint()
        bal = material_balance(dp, 100.0)
        feso4 = next(it for it in bal.items if "Ferrous" in it.name)
        # FeSO4·7H2O / Fe = 278.01 / 55.845 ≈ 4.98
        assert feso4.consumption_kg_per_day == pytest.approx(100.0 * 278.01 / 55.845, rel=1e-3)

    def test_fractions_sum_to_one(self):
        dp = DesignPoint()
        bal = material_balance(dp, 50.0)
        total_frac = sum(it.fraction_of_total for it in bal.items)
        assert total_frac == pytest.approx(1.0, rel=1e-6)

    def test_annual_cost_positive(self):
        dp = DesignPoint()
        bal = material_balance(dp, 100.0)
        assert bal.total_annual_feedstock_cost > 0
        assert bal.specific_feedstock_cost_per_kg > 0

    def test_ni_co_deposition_adds_nickel_sulfate(self):
        dp_no_ni = DesignPoint(Ni2_mol_L=0.0)
        dp_ni = DesignPoint(Ni2_mol_L=0.1)
        bal_no = material_balance(dp_no_ni, 100.0)
        bal_ni = material_balance(dp_ni, 100.0)
        names_no = [it.name for it in bal_no.items]
        names_ni = [it.name for it in bal_ni.items]
        assert "Nickel sulfate" not in names_no
        assert "Nickel sulfate" in names_ni
        assert bal_ni.total_annual_feedstock_cost > bal_no.total_annual_feedstock_cost

    def test_higher_rate_increases_cost(self):
        dp = DesignPoint()
        bal_low = material_balance(dp, 50.0)
        bal_high = material_balance(dp, 200.0)
        assert bal_high.total_annual_feedstock_cost > bal_low.total_annual_feedstock_cost

    def test_no_baco3_when_disabled(self):
        dp = DesignPoint(carburize_with_BaCO3=False)
        bal = material_balance(dp, 100.0)
        names = [it.name for it in bal.items]
        assert "Barium carbonate" not in names


class TestElectrolyteRecycling:
    """Electrolyte recycling economics."""

    def test_fe2_depletion_positive(self):
        dp = DesignPoint()
        bal = material_balance(dp, 100.0)
        rec = electrolyte_recycling(bal)
        assert rec.fe2_depletion_rate_kg_day > 0

    def test_makeup_higher_than_depletion(self):
        """FeSO4·7H2O is heavier than Fe."""
        dp = DesignPoint()
        bal = material_balance(dp, 100.0)
        rec = electrolyte_recycling(bal)
        assert rec.fe2_makeup_rate_kg_day > rec.fe2_depletion_rate_kg_day

    def test_purge_fraction_small(self):
        """Purge should be a small fraction of total volume."""
        dp = DesignPoint()
        bal = material_balance(dp, 10.0)
        rec = electrolyte_recycling(bal)
        assert 0 < rec.purge_fraction_per_day < 0.10  # < 10% per day

    def test_recycling_cost_positive(self):
        dp = DesignPoint()
        bal = material_balance(dp, 100.0)
        rec = electrolyte_recycling(bal)
        assert rec.total_recycling_cost_per_kg_fe > 0


class TestSiteScoring:
    """Location siting analysis."""

    def test_site_score_bounded(self):
        loc = CANDIDATE_LOCATIONS[0]
        s = site_score(loc)
        assert 0 <= s.total_score <= 1.0

    def test_cheap_electricity_scores_higher(self):
        cheap = LocationData("Cheap", "X", 0.02, 0.5, 2.0, 0.4,
                             50, 50, 0.7, 30.0, 0.95, 0.80)
        expensive = LocationData("Expensive", "X", 0.25, 0.5, 2.0, 0.4,
                                 50, 50, 0.7, 30.0, 0.95, 0.80)
        s_cheap = site_score(cheap)
        s_exp = site_score(expensive)
        assert s_cheap.total_score > s_exp.total_score

    def test_compare_locations_ranking(self):
        dp = DesignPoint()
        ranking = compare_locations(CANDIDATE_LOCATIONS, dp, 100.0)
        assert len(ranking.locations) == len(CANDIDATE_LOCATIONS)
        # Scores should be in descending order
        scores = [loc.total_score for loc in ranking.locations]
        assert scores == sorted(scores, reverse=True)

    def test_compare_locations_electricity_cost_filled(self):
        dp = DesignPoint()
        ranking = compare_locations(CANDIDATE_LOCATIONS, dp, 100.0)
        for loc in ranking.locations:
            assert loc.annual_electricity_cost > 0

    def test_custom_weights_change_ranking(self):
        dp = DesignPoint()
        w_default = LocationWeight()
        w_labor = LocationWeight(labor_cost=0.50, electricity_cost=0.10,
                                 renewable_fraction=0.05, water_availability=0.05,
                                 feedstock_proximity=0.10, regulatory=0.05,
                                 grid_reliability=0.10, logistics=0.05)
        r1 = compare_locations(CANDIDATE_LOCATIONS, dp, 100.0, w_default)
        r2 = compare_locations(CANDIDATE_LOCATIONS, dp, 100.0, w_labor)
        # At least the top location should differ or scores should change
        assert r1.locations[0].total_score != r2.locations[0].total_score or \
               r1.locations[-1].total_score != r2.locations[-1].total_score

    def test_five_or_more_locations(self):
        assert len(CANDIDATE_LOCATIONS) >= 5


class TestElectricitySensitivity:
    """Sensitivity to electricity price."""

    def test_sensitivity_result_shape(self):
        dp = DesignPoint()
        result = electricity_sensitivity(dp, 100.0)
        assert len(result.electricity_prices) == 15
        assert len(result.total_costs) == 15

    def test_total_cost_increases_with_electricity(self):
        dp = DesignPoint()
        result = electricity_sensitivity(dp, 100.0)
        assert result.total_costs[-1] > result.total_costs[0]

    def test_feedstock_constant_across_prices(self):
        dp = DesignPoint()
        result = electricity_sensitivity(dp, 100.0)
        # Feedstock cost doesn't depend on electricity price
        assert result.feedstock_costs[0] == pytest.approx(result.feedstock_costs[-1])
