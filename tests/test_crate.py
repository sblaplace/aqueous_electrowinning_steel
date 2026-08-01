"""Tests for the crate structural/environmental model (whole-system twin, L2)."""
from __future__ import annotations

import math

import pytest

from models.crate import (
    Crate,
    CrateConfig,
    CrateSpec,
    EnvironmentalLoads,
    GroundSpec,
    WindLoad,
)


def eval_(crate=None, wind=None, ground=None, env=None, ballast_kg=0.0):
    return Crate().evaluate(CrateConfig(
        crate=crate or CrateSpec(),
        wind=wind or WindLoad(),
        ground=ground or GroundSpec(),
        env=env or EnvironmentalLoads(),
        ballast_kg=ballast_kg,
    ))


class TestStabilityBasics:
    def test_default_broadside_is_stable(self):
        # A low, long container on its long axis resists broadside wind easily.
        v = eval_()
        assert v.stable
        assert v.fs_overturn > 5.0
        assert v.fs_bearing > 1.0

    def test_overturn_moment_and_restoring_positive(self):
        v = eval_()
        assert v.overturning_moment_Nm > 0
        assert v.restoring_moment_Nm > v.overturning_moment_Nm

    def test_end_on_wind_is_more_limiting_than_broadside(self):
        # End-on wind projects the long face and pivots about the short edge.
        broad = eval_()
        end = eval_(wind=WindLoad(direction="end"))
        assert end.fs_overturn < broad.fs_overturn


class TestBallast:
    def test_high_wind_end_on_needs_ballast(self):
        v = eval_(wind=WindLoad(direction="end", gust_m_s=55.0))
        assert v.min_ballast_kg > 0
        assert v.fs_overturn < 1.0  # unstable without ballast

    def test_ballast_raises_fs_overturn(self):
        low = eval_(wind=WindLoad(direction="end", gust_m_s=55.0))
        high = eval_(wind=WindLoad(direction="end", gust_m_s=55.0), ballast_kg=low.min_ballast_kg * 1.2)
        assert high.fs_overturn > low.fs_overturn
        assert high.fs_overturn >= 1.5

    def test_min_ballast_meets_target(self):
        cfg = CrateConfig(CrateSpec(), WindLoad(direction="end", gust_m_s=55.0))
        m = Crate()
        need = m.evaluate(cfg).min_ballast_kg
        got = m.evaluate(CrateConfig(cfg.crate, cfg.wind, cfg.ground, cfg.env, ballast_kg=need * 1.05))
        assert got.fs_overturn >= 0.999 * cfg.target_fs_overturn


class TestSliding:
    def test_sliding_limit_may_bind_after_ballast(self):
        # Ballast fixes overturn but sliding depends on friction + anchors.
        v = eval_(wind=WindLoad(direction="end", gust_m_s=55.0), ballast_kg=7000.0)
        assert v.fs_slide < v.fs_overturn or "tie-down" in v.mounting_spec

    def test_anchored_increases_slide_margin(self):
        plain = eval_(wind=WindLoad(direction="end", gust_m_s=55.0), ballast_kg=7000.0)
        anchored = eval_(
            wind=WindLoad(direction="end", gust_m_s=55.0),
            ground=GroundSpec(anchored=True), ballast_kg=7000.0)
        assert anchored.fs_slide > plain.fs_slide


class TestBearing:
    def test_fails_on_soft_ground(self):
        v = eval_(ground=GroundSpec(p_allow_kPa=0.5))  # absurdly soft
        assert not v.checks["bearing"].ok

    def test_container_bearing_is_light(self):
        v = eval_()
        assert v.net_bearing_kPa < 10.0  # footprint pressure well under 100 kPa


class TestIngress:
    def test_low_ingress_when_sealed_and_drained(self):
        v = eval_(env=EnvironmentalLoads(rain_intensity_mm_hr=10.0, sealing_class="sealed"))
        assert v.ingress_risk == "low"

    def test_high_ingress_when_flood_and_industrial(self):
        v = eval_(env=EnvironmentalLoads(
            rain_intensity_mm_hr=120.0, sealing_class="industrial"),
            ground=GroundSpec(drainable=False, flood_depth_m=0.5))
        assert v.ingress_risk == "high"


class TestMountingSpec:
    def test_mounting_spec_recommends_action_when_unstable(self):
        v = eval_(wind=WindLoad(direction="end", gust_m_s=55.0))
        assert not v.stable
        assert "ballast" in v.mounting_spec

    def test_default_needs_no_ballast(self):
        v = eval_()
        assert "no ballast required" in v.mounting_spec


class TestOutput:
    def test_to_dict_is_serialisable(self):
        d = eval_().to_dict()
        assert isinstance(d, dict)
        assert "fs_overturn" in d and "ingress_risk" in d and "min_ballast_kg" in d

    def test_seismic_load_raises_lateral_force(self):
        # low wind so the seismic lateral load dominates the max()
        no_seis = eval_(wind=WindLoad(direction="end", gust_m_s=10.0))
        seis = eval_(wind=WindLoad(direction="end", gust_m_s=10.0),
                     env=EnvironmentalLoads(seismic_base_coefficient=0.5))
        assert seis.fs_slide < no_seis.fs_slide
        assert seis.overturning_moment_Nm > no_seis.overturning_moment_Nm
