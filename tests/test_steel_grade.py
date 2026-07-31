"""Tests for steel grade routing and post-processing integration."""

import pytest
from models.steel_grade import (
    SteelGradeSpec, STEEL_GRADES, select_route, size_post_processing,
    CarburizationSizing, CoDepositionSizing, PostProcessingResult,
)
from models.dark_mill import (
    SiteDefinition, GridSpec, ClimateSpec, size_dark_mill, EXAMPLE_SITES,
)


# ─── Grade Definitions ────────────────────────────────────────────

class TestSteelGrades:
    def test_all_grades_have_valid_composition(self):
        for key, grade in STEEL_GRADES.items():
            assert 0 <= grade.c_wt_percent_target <= 2.14, f"{key}: C out of range"
            assert grade.c_wt_percent_tolerance > 0, f"{key}: tolerance must be positive"

    def test_pure_iron_is_low_carbon(self):
        g = STEEL_GRADES["pure_iron"]
        assert g.c_wt_percent_target < 0.05
        assert g.category == "pure"

    def test_alloy_grades_have_alloying_elements(self):
        g = STEEL_GRADES["AISI_8620"]
        assert g.is_alloy
        assert g.ni_wt_percent > 0
        assert g.cr_wt_percent > 0

    def test_plain_carbon_grades_are_plain_carbon(self):
        for key in ["AISI_1008", "AISI_1018", "AISI_1040", "AISI_1080", "AISI_1095"]:
            g = STEEL_GRADES[key]
            assert g.is_plain_carbon, f"{key} should be plain carbon"

    def test_case_hardened_flag(self):
        assert STEEL_GRADES["AISI_1018_case"].case_hardened
        assert STEEL_GRADES["AISI_8620"].case_hardened
        assert not STEEL_GRADES["AISI_1040"].case_hardened


# ─── Route Selection ──────────────────────────────────────────────

class TestRouteSelection:
    def test_pure_iron_is_none(self):
        assert select_route(STEEL_GRADES["pure_iron"]) == "none"

    def test_plain_carbon_is_carburize(self):
        for key in ["AISI_1008", "AISI_1018", "AISI_1040", "AISI_1080"]:
            route = select_route(STEEL_GRADES[key])
            assert route == "carburize", f"{key} should route to carburize"

    def test_alloy_is_codeposit(self):
        route = select_route(STEEL_GRADES["AISI_8620"])
        assert route == "codeposit"

    def test_case_hardened_is_carburize(self):
        route = select_route(STEEL_GRADES["AISI_1018_case"])
        assert route == "carburize"


# ─── Post-Processing Sizing ───────────────────────────────────────

class TestPostProcessingSizing:
    def test_pure_iron_no_processing(self):
        result = size_post_processing(STEEL_GRADES["pure_iron"])
        assert result.route == "none"
        assert result.total_energy_kWh_per_t == 0
        assert result.additional_capex_fraction == 0

    def test_carburization_sizing(self):
        result = size_post_processing(STEEL_GRADES["AISI_1040"])
        assert result.route == "carburize"
        assert result.carburization is not None
        c = result.carburization
        assert c.temperature_C > 0
        assert c.duration_hr > 0
        assert c.surface_carbon_wt_percent > 0
        assert c.furnace_length_mm > 0
        assert c.furnace_power_kW > 0
        assert c.energy_kWh_per_t_Fe > 0
        assert c.batch_time_hr > 0

    def test_codeposition_sizing(self):
        result = size_post_processing(STEEL_GRADES["AISI_8620"])
        assert result.route == "codeposit"
        assert result.codeposition is not None
        c = result.codeposition
        assert c.carbon_loading_g_L > 0
        assert c.ni_loading_M > 0  # alloy grade needs Ni
        assert c.particle_size_um > 0

    def test_high_carbon_needs_more_energy(self):
        # Higher carbon targets require more total energy (heating + soak)
        low = size_post_processing(STEEL_GRADES["AISI_1018"])
        high = size_post_processing(STEEL_GRADES["AISI_1080"])
        assert high.total_energy_kWh_per_t > 0
        assert low.total_energy_kWh_per_t > 0

    def test_case_hardened_has_case_depth(self):
        result = size_post_processing(STEEL_GRADES["AISI_1018_case"])
        assert result.carburization.case_depth_um > 0

    def test_through_hardening_no_case_depth(self):
        result = size_post_processing(STEEL_GRADES["AISI_1040"])
        assert result.carburization.case_depth_um == 0

    def test_explicit_route_override(self):
        # Force codeposit for a plain carbon grade
        result = size_post_processing(STEEL_GRADES["AISI_1040"], route="codeposit")
        assert result.route == "codeposit"
        assert result.codeposition is not None

    def test_summary_is_string(self):
        result = size_post_processing(STEEL_GRADES["AISI_1040"])
        s = result.summary()
        assert isinstance(s, str)
        assert "POST-PROCESSING" in s
        assert "AISI 1040" in s

    def test_energy_positive_for_all_grades(self):
        for key, grade in STEEL_GRADES.items():
            result = size_post_processing(grade)
            assert result.total_energy_kWh_per_t >= 0, f"{key}: negative energy"


# ─── Dark Mill Integration ────────────────────────────────────────

class TestDarkMillWithGrade:
    def test_default_site_is_pure_iron(self):
        site = SiteDefinition(name="test", feedstock_key="pickle_liquor")
        assert site.effective_grade.name == "Pure Iron (EW grade)"
        assert site.target_grade is None

    def test_site_with_grade(self):
        site = SiteDefinition(
            name="test",
            feedstock_key="pickle_liquor",
            target_grade=STEEL_GRADES["AISI_1040"],
        )
        assert site.effective_grade.c_wt_percent_target == 0.40

    def test_site_with_explicit_route(self):
        site = SiteDefinition(
            name="test",
            feedstock_key="pickle_liquor",
            target_grade=STEEL_GRADES["AISI_1040"],
            post_processing_route="codeposit",
        )
        assert site.post_processing_route == "codeposit"

    def test_size_dark_mill_with_grade(self):
        site = SiteDefinition(
            name="test",
            feedstock_key="pickle_liquor",
            target_grade=STEEL_GRADES["AISI_1040"],
        )
        report = size_dark_mill(site)
        assert report.post_processing is not None
        assert report.post_processing.route == "carburize"
        assert report.mass_balance.product_grade == "AISI 1040 — Medium carbon"
        assert report.mass_balance.post_processing_energy_MWh_yr > 0

    def test_size_dark_mill_pure_iron(self):
        site = SiteDefinition(name="test", feedstock_key="pickle_liquor")
        report = size_dark_mill(site)
        assert report.post_processing is not None
        assert report.post_processing.route == "none"
        assert report.mass_balance.post_processing_energy_MWh_yr == 0

    def test_alloy_grade_routes_to_codeposit(self):
        site = SiteDefinition(
            name="test",
            feedstock_key="pickle_liquor",
            target_grade=STEEL_GRADES["AISI_8620"],
        )
        report = size_dark_mill(site)
        assert report.post_processing.route == "codeposit"
        assert report.post_processing.codeposition is not None
        assert report.post_processing.codeposition.ni_loading_M > 0

    def test_summary_includes_post_processing(self):
        site = SiteDefinition(
            name="test",
            feedstock_key="pickle_liquor",
            target_grade=STEEL_GRADES["AISI_1040"],
        )
        report = size_dark_mill(site)
        s = report.summary()
        assert "POST-PROCESSING" in s
        assert "carburize" in s

    def test_capex_increases_with_post_processing(self):
        site_pure = SiteDefinition(name="test", feedstock_key="pickle_liquor")
        site_grade = SiteDefinition(
            name="test",
            feedstock_key="pickle_liquor",
            target_grade=STEEL_GRADES["AISI_1040"],
        )
        r_pure = size_dark_mill(site_pure)
        r_grade = size_dark_mill(site_grade)
        assert r_grade.capex["Total CAPEX ($)"] > r_pure.capex["Total CAPEX ($)"]

    def test_energy_increases_with_carburization(self):
        site_pure = SiteDefinition(name="test", feedstock_key="pickle_liquor")
        site_grade = SiteDefinition(
            name="test",
            feedstock_key="pickle_liquor",
            target_grade=STEEL_GRADES["AISI_1040"],
        )
        r_pure = size_dark_mill(site_pure)
        r_grade = size_dark_mill(site_grade)
        # Total energy should be higher with carburization
        assert r_grade.mass_balance.total_energy_MWh_yr > r_pure.mass_balance.total_energy_MWh_yr

    def test_all_example_sites_work_with_grade(self):
        for site_key in EXAMPLE_SITES:
            site = EXAMPLE_SITES[site_key]
            site_with_grade = SiteDefinition(
                name=site.name,
                feedstock_key=site.feedstock_key,
                grid=site.grid,
                climate=site.climate,
                target_grade=STEEL_GRADES["AISI_1040"],
            )
            report = size_dark_mill(site_with_grade)
            assert report.post_processing.route == "carburize"
            assert report.stack_design.annual_production_t() > 0


# ─── CAD Config with Post-Processing ─────────────────────────────

class TestDarkMillConfigWithPostProcessing:
    def test_default_config_no_post_processing(self):
        from models.cad.dark_mill_config import DarkMillConfig
        cfg = DarkMillConfig()
        assert cfg.post_processing_route == "none"

    def test_carburize_config_has_furnace(self):
        from models.cad.dark_mill_config import DarkMillConfig
        cfg = DarkMillConfig(post_processing_route="carburize")
        assert cfg.furnace_length > 0
        assert cfg.furnace_width > 0
        assert cfg.furnace_height > 0
        assert cfg.quench_tank_length > 0

    def test_codeposit_config_has_particle_system(self):
        from models.cad.dark_mill_config import DarkMillConfig
        cfg = DarkMillConfig(post_processing_route="codeposit")
        assert cfg.particle_hopper_diameter > 0
        assert cfg.ultrasonic_bath_length > 0

    def test_from_sizing_with_post_processing(self):
        from models.cad.dark_mill_config import DarkMillConfig
        sizing = {
            "stack_design": {"n_stacks": 5, "cells_per_stack": 15},
            "post_processing": {"route": "carburize", "carburization": {
                "furnace_length_mm": 2000.0,
                "furnace_width_mm": 1500.0,
                "furnace_height_mm": 1800.0,
            }},
        }
        cfg = DarkMillConfig.from_sizing(sizing)
        assert cfg.n_stacks == 5
        assert cfg.post_processing_route == "carburize"
        assert cfg.furnace_length == 2000.0

    def test_transportability_with_furnace(self):
        from models.cad.dark_mill_config import DarkMillConfig, check_transportability
        cfg = DarkMillConfig(post_processing_route="carburize")
        t = check_transportability(cfg)
        # Furnace is inside the enclosure, so transportability unchanged
        assert t["fits_trailer"] is True
