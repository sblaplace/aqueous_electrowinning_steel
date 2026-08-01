"""Tests for the physics-coupled process model (digital twin measurement model)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from models.twin_physics import CellProcessModel, ProcessPrediction, default_process_model


def _small_model(tmp_path) -> CellProcessModel:
    """Fast-to-build surrogate on a coarse grid."""
    return CellProcessModel(
        j_grid=(100.0, 150.0),
        T_grid=(50.0, 70.0),
        fe2_grid=(0.8, 1.2),
        cache_path=str(tmp_path / "tiny_cache.json"),
    )


class TestBuildAndPredict:
    def test_predict_returns_process_prediction(self, tmp_path):
        m = _small_model(tmp_path)
        p = m.predict(125.0, 60.0, 1.0)
        assert isinstance(p, ProcessPrediction)

    def test_fe_within_physical_bounds(self, tmp_path):
        m = _small_model(tmp_path)
        for j in (100.0, 125.0, 150.0):
            for fe2 in (0.8, 1.0, 1.2):
                p = m.predict(j, 60.0, fe2)
                assert 0.0 <= p.current_efficiency <= 1.0
                assert 0.0 <= p.fe_percent <= 100.0

    def test_v_cell_positive_and_reasonable(self, tmp_path):
        m = _small_model(tmp_path)
        p = m.predict(125.0, 60.0, 1.0)
        assert p.v_cell_V > 0.5
        assert p.v_cell_V < 20.0

    def test_deposit_rate_positive(self, tmp_path):
        m = _small_model(tmp_path)
        p = m.predict(100.0, 50.0, 0.8)
        assert p.deposit_rate_um_hr > 0.0

    def test_surface_pH_near_bulk(self, tmp_path):
        # Nernst-Planck keeps surface pH near bulk in a buffered divided cell
        m = _small_model(tmp_path)
        p = m.predict(125.0, 60.0, 1.0)
        assert p.surface_pH >= 0.0
        assert abs(p.surface_pH - 2.0) < 1.0

    def test_faraday_cross_check(self, tmp_path):
        # deposit rate should scale with j*FE: doubling j roughly doubles rate
        m = _small_model(tmp_path)
        p1 = m.predict(100.0, 60.0, 1.0)
        p2 = m.predict(150.0, 60.0, 1.0)
        ratio = p2.deposit_rate_um_hr / p1.deposit_rate_um_hr
        assert 1.2 < ratio < 2.4, f"deposit rate should scale ~with j, got {ratio}"


class TestInterpolation:
    def test_interpolates_between_grid_points(self, tmp_path):
        m = _small_model(tmp_path)
        # exactly on a grid node
        on_node = m.predict(100.0, 50.0, 0.8)
        # between nodes
        between = m.predict(115.0, 55.0, 0.9)
        assert math.isfinite(between.v_cell_V)
        assert 0.0 <= between.current_efficiency <= 1.0

    def test_online_speed(self, tmp_path):
        m = _small_model(tmp_path)
        import time as _t
        t = _t.perf_counter()
        for _ in range(2000):
            m.predict(125.0, 60.0, 1.0)
        dt = _t.perf_counter() - t
        # surrogate must be far faster than a full physics solve (~0.3 s each)
        assert dt < 5.0, f"surrogate too slow for online EKF: {dt:.2f}s for 2000 predicts"


class TestCache:
    def test_cache_round_trip(self, tmp_path):
        cache = tmp_path / "cache.json"
        m1 = CellProcessModel(
            j_grid=(100.0, 150.0), T_grid=(50.0, 70.0), fe2_grid=(0.8, 1.2),
            cache_path=str(cache))
        p1 = m1.predict(125.0, 60.0, 1.0)
        # reload from cache (no rebuild)
        m2 = CellProcessModel(
            j_grid=(100.0, 150.0), T_grid=(50.0, 70.0), fe2_grid=(0.8, 1.2),
            cache_path=str(cache))
        p2 = m2.predict(125.0, 60.0, 1.0)
        assert p1.current_efficiency == pytest.approx(p2.current_efficiency, abs=1e-9)
        assert p1.v_cell_V == pytest.approx(p2.v_cell_V, abs=1e-9)


class TestNominal:
    def test_nominal_is_reasonable(self, tmp_path):
        m = _small_model(tmp_path)
        n = m.nominal
        assert "temperature_C" in n and "j_avg_mA_cm2" in n and "cell_voltage_V" in n
        assert 0 < n["cell_voltage_V"] < 20.0
