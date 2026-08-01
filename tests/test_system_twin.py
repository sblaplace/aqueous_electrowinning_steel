"""Tests for the whole-system twin — process + crate + site composition.

Verifies:
- ClimateSpec and SiteDefinition extended with structural/env fields
- crate verdict feeds dark_mill go/no-go
- environmental limits in operating_twin trigger storm-mode safe-state
- system_twin driver emits per-layer credibility vector, stability verdict,
  required ballast/mounting, environmental safe-state action
- reuses the 3 legacy dark-mill site scenarios
All numbers screening-grade L0 until validated.
"""

import pytest

from models.dark_mill import ClimateSpec, SiteDefinition, EXAMPLE_SITES, size_dark_mill
from models.system_twin import (
    CredibilityVector,
    SystemTwinReport,
    evaluate_system_twin,
    evaluate_all_sites,
    LEGACY_THREE,
)
from models.crate import CrateConfig, CrateSpec, WindLoad, GroundSpec, EnvironmentalLoads
from models.operating_twin import TwinConfig, SensorSnapshot, OperatingTwin


# ─── ClimateSpec / SiteDefinition extensions ────────────────────────

class TestSiteDefinitionExtensions:
    def test_climate_has_structural_fields(self):
        cs = ClimateSpec()
        assert hasattr(cs, "wind_gust_m_s")
        assert hasattr(cs, "wind_terrain")
        assert hasattr(cs, "rainfall_intensity_mm_hr")
        assert hasattr(cs, "snow_load_kPa")
        assert hasattr(cs, "freeze_depth_m")

    def test_site_has_ground_fields(self):
        sd = SiteDefinition(name="test", feedstock_key="pickle_liquor")
        assert hasattr(sd, "soil_bearing_kPa")
        assert hasattr(sd, "seismic_coefficient")
        assert hasattr(sd, "flood_depth_m")
        assert hasattr(sd, "sealing_class")
        assert hasattr(sd, "ground_friction_mu")

    def test_example_sites_have_realistic_structural_values(self):
        for key, site in EXAMPLE_SITES.items():
            assert site.climate.wind_gust_m_s > 0
            assert site.climate.wind_terrain in {"open", "suburban", "urban"}
            assert site.soil_bearing_kPa >= 10
            # flood depth non-negative
            assert site.flood_depth_m >= 0

    def test_site_to_crate_mapping(self):
        from models.dark_mill import site_to_crate_config
        site = EXAMPLE_SITES["pickle_liquor_us_midwest"]
        cfg = site_to_crate_config(site)
        assert cfg.wind.gust_m_s == site.climate.wind_gust_m_s
        assert cfg.ground.p_allow_kPa == site.soil_bearing_kPa
        assert cfg.ground.flood_depth_m == site.flood_depth_m


# ─── Crate fed into dark_mill go/no-go ─────────────────────────────

class TestCrateFeedsDarkMill:
    def test_dark_mill_report_has_crate_verdict(self):
        site = EXAMPLE_SITES["pickle_liquor_us_midwest"]
        report = size_dark_mill(site)
        assert report.crate_verdict is not None
        assert hasattr(report.crate_verdict, "fs_overturn")
        assert hasattr(report.crate_verdict, "min_ballast_kg")
        assert hasattr(report.crate_verdict, "mounting_spec")

    def test_go_no_go_includes_crate_criteria(self):
        site = EXAMPLE_SITES["pickle_liquor_us_midwest"]
        report = size_dark_mill(site)
        gng = report.go_no_go
        # Must include crate checks
        assert "Crate overturning" in gng
        assert "Crate bearing" in gng
        assert "Crate sliding" in gng
        assert "Crate ingress" in gng
        for k in ["Crate overturning", "Crate bearing", "Crate sliding", "Crate ingress"]:
            assert "pass" in gng[k]
            assert "detail" in gng[k]

    def test_soft_ground_fails_bearing(self):
        site = SiteDefinition(
            name="soft ground test",
            feedstock_key="pickle_liquor",
            soil_bearing_kPa=0.5,  # absurdly soft → bearing fail
        )
        report = size_dark_mill(site)
        assert not report.go_no_go["Crate bearing"]["pass"]

    def test_high_wind_needs_ballast(self):
        # wind_farm_ore has 55 m/s gust
        site = EXAMPLE_SITES["wind_farm_ore"]
        report = size_dark_mill(site)
        # broadside may still pass but min_ballast >0 for end-on worst
        assert report.crate_verdict is not None
        # system twin evaluates end-on worst
        sys_report = evaluate_system_twin("wind_farm_ore")
        assert sys_report.required_ballast_kg > 0
        assert "ballast" in sys_report.mounting_spec.lower() or "anchor" in sys_report.mounting_spec.lower()


# ─── Environmental limits in operating_twin ────────────────────────

class TestOperatingTwinEnvironmental:
    def _base_config(self, **env_limits):
        return TwinConfig(
            cell_id="TEST-ENV-001",
            max_current_A=10.0,
            max_current_density_mA_cm2=200.0,
            max_voltage_V=5.0,
            min_temperature_C=0.0,
            max_temperature_C=80.0,
            min_fe2_M=0.2,
            max_fe2_M=2.0,
            min_pH=0.5,
            max_pH=5.0,
            target_current_A=5.0,
            **env_limits,
        )

    def _base_snapshot(self, **env_obs):
        return SensorSnapshot(
            timestamp_s=0.0,
            current_A=1.0,
            voltage_V=2.5,
            temperature_C=25.0,
            pH=2.0,
            fe2_M=1.0,
            cathode_area_cm2=100.0,
            source_run_id="env-test",
            **env_obs,
        )

    def test_high_wind_triggers_trip(self):
        cfg = self._base_config(max_wind_gust_m_s=40.0)
        twin = OperatingTwin(cfg)
        snap = self._base_snapshot(wind_gust_m_s=55.0)
        state = twin.update(snap)
        assert "high_wind" in state.trip_reasons

    def test_flood_triggers_trip(self):
        cfg = self._base_config(max_flood_depth_m=0.1)
        twin = OperatingTwin(cfg)
        snap = self._base_snapshot(flood_depth_m=0.2)
        state = twin.update(snap)
        assert "flood" in state.trip_reasons

    def test_ingress_triggers_trip(self):
        cfg = self._base_config()
        twin = OperatingTwin(cfg)
        snap = self._base_snapshot(ingress_detected=True)
        state = twin.update(snap)
        assert "ingress" in state.trip_reasons

    def test_environmental_safe_state_action(self):
        cfg = self._base_config(max_wind_gust_m_s=40.0, max_flood_depth_m=0.1)
        twin = OperatingTwin(cfg)
        # normal
        snap_ok = self._base_snapshot(wind_gust_m_s=20.0, flood_depth_m=0.0)
        assert twin.environmental_safe_state(snap_ok) == "normal_operation"
        # high wind
        snap_wind = self._base_snapshot(wind_gust_m_s=50.0)
        assert "high_wind" in twin.environmental_safe_state(snap_wind) or "storm_mode" in twin.environmental_safe_state(snap_wind)
        # flood
        snap_flood = self._base_snapshot(flood_depth_m=0.3)
        assert "flood" in twin.environmental_safe_state(snap_flood)

    def test_no_env_limits_no_trip(self):
        # Without env limits set, env observation should not trip
        cfg = self._base_config()
        twin = OperatingTwin(cfg)
        snap = self._base_snapshot(wind_gust_m_s=100.0, flood_depth_m=1.0, rain_intensity_mm_hr=200.0)
        # Only ingress should trip when no limits; wind/flood only trip if limit set
        # So here only ingress would cause trip, but we didn't set ingress
        state = twin.update(snap)
        # No trip because no limits and no ingress
        assert state.mode.value != "tripped" or len(state.trip_reasons) == 0


# ─── System twin driver ────────────────────────────────────────────

class TestSystemTwinDriver:
    def test_credibility_vector_is_L0(self):
        cred = CredibilityVector.screening()
        assert cred.process_level == 0
        assert cred.crate_level == 0
        assert cred.site_level == 0
        assert "L0" in cred.label()

    def test_evaluate_single_site(self):
        report = evaluate_system_twin("pickle_liquor_us_midwest")
        assert isinstance(report, SystemTwinReport)
        assert report.site_key == "pickle_liquor_us_midwest"
        assert report.credibility.process_level == 0
        assert report.credibility.crate_level == 0
        assert report.credibility.site_level == 0
        # Has required outputs
        assert hasattr(report, "required_ballast_kg")
        assert hasattr(report, "mounting_spec")
        assert hasattr(report, "environmental_safe_state")
        assert hasattr(report, "combined_stable")
        assert isinstance(report.go_no_go, dict)

    def test_per_layer_credibility_vector_present(self):
        report = evaluate_system_twin("pickle_liquor_us_midwest")
        d = report.to_dict()
        assert "credibility_vector" in d
        assert d["credibility_vector"]["process"] == 0
        assert d["credibility_vector"]["crate"] == 0
        assert d["credibility_vector"]["site"] == 0
        assert "process L0 / crate L0 / site L0" in d["credibility_label"]

    def test_combined_stability_verdict(self):
        # Midwest should be stabilisable (broadside stable)
        midwest = evaluate_system_twin("pickle_liquor_us_midwest")
        assert isinstance(midwest.combined_stable, bool)
        # Windy site needs ballast but is stabilisable
        windy = evaluate_system_twin("wind_farm_ore")
        assert windy.required_ballast_kg > 0
        # After our definition, windy combined_stable should be True (stabilisable)
        assert windy.combined_stable is True

    def test_required_ballast_and_mounting(self):
        windy = evaluate_system_twin("wind_farm_ore")
        assert windy.required_ballast_kg >= 0
        assert isinstance(windy.mounting_spec, str)
        assert len(windy.mounting_spec) > 0

    def test_environmental_safe_state(self):
        midwest = evaluate_system_twin("pickle_liquor_us_midwest")
        assert midwest.environmental_safe_state == "normal_operation"
        windy = evaluate_system_twin("wind_farm_ore")
        assert "storm_mode" in windy.environmental_safe_state or "high_wind" in windy.environmental_safe_state

    def test_reuses_three_legacy_sites(self):
        assert len(LEGACY_THREE) == 3
        for key in LEGACY_THREE:
            assert key in EXAMPLE_SITES
            report = evaluate_system_twin(key)
            assert isinstance(report, SystemTwinReport)
            # Must have crates
            assert report.crate_verdict is not None

    def test_evaluate_all_sites(self):
        all_reports = evaluate_all_sites()
        assert len(all_reports) == len(EXAMPLE_SITES)
        for k, r in all_reports.items():
            assert isinstance(r, SystemTwinReport)
            assert r.to_dict()["site_key"] == k

    def test_summary_contains_key_info(self):
        report = evaluate_system_twin("pickle_liquor_us_midwest")
        s = report.summary()
        assert "Credibility vector" in s
        assert "process L0 / crate L0 / site L0" in s
        assert "Required ballast" in s
        assert "Mounting" in s
        assert "Safe-state action" in s or "safe-state" in s.lower()
        assert "Combined stable" in s or "COMBINED STABILITY" in s

    def test_system_twin_report_json_serializable(self):
        report = evaluate_system_twin("red_mud_alumina_refinery")
        d = report.to_dict()
        # Should be JSON serializable without custom default (apart from numpy)
        import json, pathlib
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
