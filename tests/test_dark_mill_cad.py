"""Tests for the dark mill 3D CAD model."""

import pytest

from models.cad.dark_mill_config import (
    DarkMillConfig,
    check_transportability,
    check_rainwater,
    check_maintenance_access,
)


# ─── Config tests ─────────────────────────────────────────────────

class TestDarkMillConfig:
    def test_defaults(self):
        cfg = DarkMillConfig()
        assert cfg.enclosure_length > 0
        assert cfg.enclosure_width > 0
        assert cfg.enclosure_height > 0
        assert cfg.n_stacks > 0

    def test_stack_height(self):
        cfg = DarkMillConfig(cells_per_stack=20, cell_gap=40, stack_base_height=200, stack_frame_width=50)
        expected = 200 + 20 * 40 + 50
        assert cfg.stack_height == expected

    def test_stacks_per_row(self):
        cfg = DarkMillConfig()
        assert cfg.n_stacks_per_row >= 1
        assert cfg.n_stacks_per_row <= cfg.n_stacks

    def test_from_sizing(self):
        sizing = {"stack_design": {"n_stacks": 5, "cells_per_stack": 15}}
        cfg = DarkMillConfig.from_sizing(sizing)
        assert cfg.n_stacks == 5
        assert cfg.cells_per_stack == 15

    def test_to_dict(self):
        cfg = DarkMillConfig()
        d = cfg.to_dict()
        assert "enclosure_length" in d
        assert isinstance(d["enclosure_length"], float)

    def test_inner_dimensions(self):
        cfg = DarkMillConfig()
        assert cfg.enclosure_inner_length() < cfg.enclosure_length
        assert cfg.enclosure_inner_width() < cfg.enclosure_width
        assert cfg.enclosure_inner_height() < cfg.enclosure_height


# ─── Transportability tests ───────────────────────────────────────

class TestTransportability:
    def test_fits_standard_trailer(self):
        cfg = DarkMillConfig()
        t = check_transportability(cfg)
        assert t["fits_width"] is True
        assert t["fits_height"] is True
        assert t["fits_trailer"] is True

    def test_oversized_enclosure_fails(self):
        cfg = DarkMillConfig(enclosure_width=3000.0, enclosure_height=5000.0)
        t = check_transportability(cfg)
        assert t["fits_width"] is False
        assert t["fits_height"] is False
        assert t["fits_trailer"] is False

    def test_weight_estimate_positive(self):
        cfg = DarkMillConfig()
        t = check_transportability(cfg)
        assert t["weight_estimate_kg"] > 0

    def test_fits_20ft_container(self):
        cfg = DarkMillConfig()
        t = check_transportability(cfg)
        assert t["fits_length_20ft"] is True


# ─── Rainwater tests ──────────────────────────────────────────────

class TestRainwater:
    def test_roof_slope_positive(self):
        cfg = DarkMillConfig()
        r = check_rainwater(cfg)
        assert r["roof_slope_deg"] > 0
        assert r["roof_height_diff_mm"] > 0

    def test_runoff_direction_specified(self):
        cfg = DarkMillConfig()
        r = check_rainwater(cfg)
        assert "low side" in r["runoff_direction"].lower()

    def test_puddle_risk_zones_listed(self):
        cfg = DarkMillConfig()
        r = check_rainwater(cfg)
        assert len(r["puddle_risk_zones"]) > 0


# ─── Maintenance access tests ─────────────────────────────────────

class TestMaintenanceAccess:
    def test_aisle_width_positive(self):
        cfg = DarkMillConfig()
        a = check_maintenance_access(cfg)
        assert a["center_aisle_width_mm"] > 0

    def test_forklift_pockets(self):
        cfg = DarkMillConfig()
        a = check_maintenance_access(cfg)
        assert a["forklift_pockets"] is True


# ─── Geometry tests (require CadQuery) ────────────────────────────

class TestGeometry:
    @pytest.fixture
    def cfg(self):
        return DarkMillConfig()

    def test_build_frame(self):
        try:
            from models.cad.dark_mill_cad import build_frame
        except ImportError:
            pytest.skip("CadQuery not available")
        cfg = DarkMillConfig()
        frame = build_frame(cfg)
        bb = frame.val().BoundingBox()
        assert bb.xlen > 0
        assert bb.ylen > 0
        assert bb.zlen > 0

    def test_build_floor(self):
        try:
            from models.cad.dark_mill_cad import build_floor
        except ImportError:
            pytest.skip("CadQuery not available")
        cfg = DarkMillConfig()
        floor = build_floor(cfg)
        bb = floor.val().BoundingBox()
        assert bb.xlen > 0

    def test_build_cell_stacks(self):
        try:
            from models.cad.dark_mill_cad import build_cell_stacks
        except ImportError:
            pytest.skip("CadQuery not available")
        cfg = DarkMillConfig(n_stacks=3, cells_per_stack=10)
        stacks = build_cell_stacks(cfg)
        bb = stacks.val().BoundingBox()
        assert bb.xlen > 0

    def test_build_dark_mill(self):
        try:
            from models.cad.dark_mill_cad import build_dark_mill
        except ImportError:
            pytest.skip("CadQuery not available")
        cfg = DarkMillConfig(n_stacks=2, cells_per_stack=5)
        model = build_dark_mill(cfg)
        bb = model.val().BoundingBox()
        assert bb.xlen > 0
        assert bb.ylen > 0
        assert bb.zlen > 0

    def test_full_assembly_builds(self):
        try:
            from models.cad.dark_mill_cad import build_dark_mill_assembly
        except ImportError:
            pytest.skip("CadQuery not available")
        cfg = DarkMillConfig(n_stacks=2, cells_per_stack=5)
        assy = build_dark_mill_assembly(cfg)
        assert assy is not None
