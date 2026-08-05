"""Tests for the L3 site-layer design module (`models/site_layer.py`).

These verify the four design blocks against a `SiteDefinition` + the D4 crate
verdict: foundation/ballast, flood & drainage, terrain wind exposure, and site
layout/access.  Screening-grade by construction; tests pin the *logic*, not an
authoritative civil/geotechnical answer.
"""
from __future__ import annotations


from models.site_layer import (
    SiteLayer,
    SiteLayerVerdict,
    access_power,
    _water_need,
)
from models.dark_mill import EXAMPLE_SITES, evaluate_crate_for_site
from models.crate import CrateVerdict


def layer_for(key="copperas_tio2_plant"):
    site = EXAMPLE_SITES[key]
    return SiteLayer().evaluate(site, crate_verdict=evaluate_crate_for_site(site))


def test_all_example_sites_produce_a_verdict():
    for key in EXAMPLE_SITES:
        v = layer_for(key)
        assert isinstance(v, SiteLayerVerdict)
        assert isinstance(v.will_sink(), bool)
        assert isinstance(v.will_flood(), bool)
        assert isinstance(v.will_get_wind_exposed(), bool)


def test_default_verdict_flat_is_json_serialisable():
    v = layer_for("copperas_tio2_plant")
    d = v.flat()
    assert set(d) >= {"will_sink", "will_flood", "will_get_wind_exposed",
                      "foundation", "drainage", "wind", "layout"}
    assert isinstance(d["foundation"]["pad_bearing_kPa"], float)


class TestFoundation:
    def test_pad_is_larger_than_crate_footprint(self):
        v = layer_for()
        from models.crate import CrateSpec
        crab = CrateSpec()
        assert v.foundation.pad_length_m > crab.length_m
        assert v.foundation.pad_width_m > crab.width_m

    def test_bearing_well_under_allowable_on_compacted_pad(self):
        v = layer_for()
        assert v.foundation.bearing_ok
        assert v.foundation.pad_bearing_kPa < 50.0

    def test_frost_depth_lengthens_footer(self):
        # frost is read from the site climate
        site = EXAMPLE_SITES["pickle_liquor_us_midwest"]  # freeze_depth_m=0.9
        v = SiteLayer().evaluate(site, crate_verdict=evaluate_crate_for_site(site))
        assert v.foundation.footer_depth_m >= 0.5 + 0.9 - 1e-9

    def test_foundation_type_set(self):
        v = layer_for()
        assert v.foundation.foundation_type in ("compacted gravel pad + concrete footing",
                                                "concrete slab")


class TestDrainage:
    def test_clear_site_does_not_flood(self):
        v = layer_for("copperas_tio2_plant")  # flood_depth_m=0
        assert v.drainage.flood_clearance_m > 0
        assert not v.will_flood()

    def test_flooded_site_flags(self):
        site = EXAMPLE_SITES["red_mud_alumina_refinery"]
        # raise flood depth above freeboard to force a flood verdict
        from dataclasses import replace
        site = replace(site, flood_depth_m=1.0)
        v = SiteLayer().evaluate(site, crate_verdict=evaluate_crate_for_site(site))
        assert v.will_flood()
        assert v.drainage.drainage_verdict == "high"
        assert "voussoirs" in v.drainage.spec

    def test_marginal_freeboard_is_medium(self):
        # red_mud_alumina_refinery: flood 0.20, freeboard 0.30 → 0.10 clear
        v = layer_for("red_mud_alumina_refinery")
        assert v.drainage.drainage_verdict == "medium"


class TestWindExposure:
    def test_open_terrain_gust_pressure_grows_with_gust(self):
        v_g = layer_for("wind_farm_ore")      # 55 m/s open
        v_c = layer_for("copperas_tio2_plant")  # 38 m/s suburban
        assert v_g.wind.gust_pressure_Pa > v_c.wind.gust_pressure_Pa

    def test_wind_exposed_flag_when_gust_high(self):
        v = layer_for("wind_farm_ore")  # 55 m/s
        assert v.wind.design_gust_m_s < 55.0  # height/roughness derate applied
        assert v.wind.gust_pressure_Pa > 0

    def test_terrain_multiplier_bounds(self):
        v_open = layer_for("acid_mine_drainage_appalachia")   # open
        v_urb = layer_for("pickle_liquor_us_midwest")         # suburban
        assert v_open.wind.terrain_multiplier >= v_urb.wind.terrain_multiplier


class TestLayoutAccess:
    def test_copperas_beachhead_layout_fits(self):
        v = layer_for("copperas_tio2_plant")
        assert v.layout.feedstock_ok
        assert v.layout.power_ok
        assert v.layout.water_ok
        assert v.layout.product_ok
        assert v.layout.layout_fit

    def test_scarce_water_blocks_layout(self):
        v = layer_for("wind_farm_ore")  # water_availability="scarce"
        assert not v.layout.water_ok
        assert "water" in v.layout.spec

    def test_power_demand_scales_with_capacity(self):
        small = access_power(500)
        large = access_power(100000)
        assert large > small
        assert large > 0

    def test_power_headroom_gate(self):
        # a huge target on a small source must fail power access
        site = EXAMPLE_SITES["copperas_tio2_plant"]
        from dataclasses import replace
        big = replace(site, target_capacity_t_Fe_yr=600_000.0)
        v = SiteLayer().evaluate(big, crate_verdict=evaluate_crate_for_site(big))
        assert not v.layout.power_ok


class TestWaterNeed:
    def test_water_need_grows_with_scale(self):
        assert _water_need(1000) < _water_need(100000)
        assert _water_need(1000) > 0


class TestCompositionWithCrateVerdict:
    def test_accepts_crate_verdict_object(self):
        site = EXAMPLE_SITES["pickle_liquor_us_midwest"]
        cv = evaluate_crate_for_site(site)
        assert isinstance(cv, CrateVerdict)
        v = SiteLayer().evaluate(site, crate_verdict=cv)
        assert v.foundation.ballast_kg >= 0
