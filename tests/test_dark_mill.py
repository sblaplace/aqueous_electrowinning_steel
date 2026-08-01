"""Tests for the dark mill site-sizing digital twin."""

import pytest
from models.dark_mill import (
    SiteDefinition, GridSpec, ClimateSpec, StackDesign, SiteReport,
    size_dark_mill, run_site, run_all_sites, comparison_table,
    EXAMPLE_SITES, FEEDSTOCKS,
)


# ─── Site definition tests ────────────────────────────────────────

class TestSiteDefinition:
    def test_feedstock_lookup(self):
        site = SiteDefinition(name="test", feedstock_key="pickle_liquor")
        assert site.feedstock.name.startswith("Spent pickle")
        assert site.feedstock.already_dissolved is True
        assert site.feedstock.cost_per_t_feedstock < 0  # negative cost

    def test_invalid_feedstock_raises(self):
        site = SiteDefinition(name="test", feedstock_key="nonexistent")
        with pytest.raises(KeyError):
            _ = site.feedstock

    def test_all_example_sites_have_valid_feedstocks(self):
        for key, site in EXAMPLE_SITES.items():
            assert site.feedstock_key in FEEDSTOCKS, f"{key}: bad feedstock_key"


# ─── Sizing tests ─────────────────────────────────────────────────

class TestSizeDarkMill:
    @pytest.fixture
    def basic_report(self):
        site = SiteDefinition(
            name="test",
            feedstock_key="pickle_liquor",
            grid=GridSpec(electricity_price_kWh=0.04, max_power_MW=10.0),
            climate=ClimateSpec(ambient_temp_C=25.0),
            target_capacity_t_Fe_yr=1000.0,
        )
        return size_dark_mill(site)

    def test_report_has_all_fields(self, basic_report):
        assert isinstance(basic_report, SiteReport)
        assert basic_report.stack_design is not None
        assert basic_report.mass_balance is not None
        assert basic_report.capex is not None
        assert basic_report.opex is not None
        assert basic_report.lcofe is not None
        assert basic_report.go_no_go is not None

    def test_stack_sized_for_target(self, basic_report):
        sd = basic_report.stack_design
        assert sd.n_stacks >= 1
        assert sd.cells_per_stack >= 1
        assert sd.annual_production_t() > 0

    def test_production_meets_target(self, basic_report):
        target = 1000.0
        actual = basic_report.stack_design.annual_production_t()
        # Should be at least target (may exceed due to integer stack rounding)
        assert actual >= target

    def test_power_is_positive(self, basic_report):
        assert basic_report.stack_design.total_power_kW > 0

    def test_mass_balance_consistency(self, basic_report):
        mb = basic_report.mass_balance
        assert mb.iron_production_t_yr > 0
        assert mb.feedstock_consumed_t_yr > 0
        assert mb.total_energy_MWh_yr > 0
        assert mb.water_consumption_m3_yr > 0

    def test_lcofe_positive(self, basic_report):
        assert basic_report.lcofe["LCOFe ($/t Fe)"] > 0

    def test_go_nogo_has_criteria(self, basic_report):
        gng = basic_report.go_no_go
        assert "Grid capacity" in gng
        assert "LCOFe vs DRI-H2" in gng
        assert "Site footprint" in gng
        assert "Operating temperature" in gng
        assert "Energy kill criterion" in gng
        for criterion, result in gng.items():
            assert "pass" in result
            assert "detail" in result

    def test_negative_feedstock_gives_revenue(self):
        site = SiteDefinition(
            name="test",
            feedstock_key="pickle_liquor",  # negative cost
            target_capacity_t_Fe_yr=1000.0,
        )
        report = size_dark_mill(site)
        if "Feedstock revenue ($/t Fe)" in report.lcofe:
            assert report.lcofe["Feedstock revenue ($/t Fe)"] > 0
            assert report.lcofe["LCOFe ($/t Fe) adjusted"] < report.lcofe["LCOFe ($/t Fe)"]

    def test_grid_capacity_fail(self):
        site = SiteDefinition(
            name="test",
            feedstock_key="high_grade_ore",
            grid=GridSpec(max_power_MW=0.001),  # tiny grid
            target_capacity_t_Fe_yr=10000.0,
        )
        report = size_dark_mill(site)
        assert not report.go_no_go["Grid capacity"]["pass"]

    def test_footprint_fail(self):
        site = SiteDefinition(
            name="test",
            feedstock_key="pickle_liquor",
            target_capacity_t_Fe_yr=100000.0,  # huge plant
            available_area_m2=10.0,  # tiny site
        )
        report = size_dark_mill(site)
        assert report.go_no_go["Site footprint"]["pass"] is False

    def test_summary_is_string(self, basic_report):
        s = basic_report.summary()
        assert isinstance(s, str)
        assert "DARK MILL" in s
        assert "GO" in s or "NO-GO" in s


# ─── Example sites tests ──────────────────────────────────────────

class TestExampleSites:
    def test_all_sites_run(self):
        reports = run_all_sites()
        assert len(reports) == len(EXAMPLE_SITES)
        for key, report in reports.items():
            assert isinstance(report, SiteReport)
            assert report.stack_design.annual_production_t() > 0

    def test_comparison_table(self):
        reports = run_all_sites()
        table = comparison_table(reports)
        assert isinstance(table, str)
        assert "Site" in table
        assert "LCOFe" in table

    @pytest.mark.parametrize("site_key", list(EXAMPLE_SITES.keys()))
    def test_individual_site(self, site_key):
        report = run_site(site_key)
        assert report.stack_design.annual_production_t() > 0
        assert report.lcofe["LCOFe ($/t Fe)"] > 0

    def test_pickle_liquor_cheapest_feedstock(self):
        """Pickle liquor (negative cost, no grinding) should be the cheapest site."""
        reports = run_all_sites()
        pickle_report = reports["pickle_liquor_us_midwest"]
        ore_report = reports["wind_farm_ore"]
        # Pickle liquor has negative feedstock cost, ore doesn't
        assert pickle_report.mass_balance.feedstock_cost_per_t_Fe < ore_report.mass_balance.feedstock_cost_per_t_Fe

    def test_wind_farm_lowest_co2(self):
        """Wind farm with 95% renewable should have lowest scope 2 CO2."""
        reports = run_all_sites()
        wind = reports["wind_farm_ore"]
        midwest = reports["pickle_liquor_us_midwest"]
        wind_co2 = wind.mass_balance.scope2_CO2_t_yr / wind.mass_balance.iron_production_t_yr
        midwest_co2 = midwest.mass_balance.scope2_CO2_t_yr / midwest.mass_balance.iron_production_t_yr
        assert wind_co2 < midwest_co2


# ─── Stack design property tests ──────────────────────────────────

class TestStackDesign:
    def test_total_area_uses_both_faces(self):
        sd = StackDesign(
            n_stacks=1, cells_per_stack=10, electrode_area_m2=1.0,
            current_density_mA_cm2=100, current_efficiency=0.9,
            cell_voltage_V=2.5, temperature_C=50,
        )
        # 1 m² × 10 cells × 2 faces = 20 m²
        assert sd.total_electrode_area_m2 == 20.0

    def test_stack_voltage_scales_with_cells(self):
        sd = StackDesign(
            n_stacks=1, cells_per_stack=20, electrode_area_m2=0.5,
            current_density_mA_cm2=100, current_efficiency=0.9,
            cell_voltage_V=2.5, temperature_C=50,
        )
        assert sd.stack_voltage_V == 50.0  # 20 × 2.5

    def test_production_scales_linearly(self):
        sd1 = StackDesign(
            n_stacks=1, cells_per_stack=10, electrode_area_m2=0.5,
            current_density_mA_cm2=100, current_efficiency=0.9,
            cell_voltage_V=2.5, temperature_C=50,
        )
        sd2 = StackDesign(
            n_stacks=2, cells_per_stack=10, electrode_area_m2=0.5,
            current_density_mA_cm2=100, current_efficiency=0.9,
            cell_voltage_V=2.5, temperature_C=50,
        )
        assert abs(sd2.production_rate_kg_hr - 2 * sd1.production_rate_kg_hr) < 0.01
