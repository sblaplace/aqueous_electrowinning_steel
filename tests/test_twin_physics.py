"""Tests for the physics-coupled process model (digital twin measurement model)."""
from __future__ import annotations

import math

import pytest

from models.twin_physics import CellProcessModel, ProcessPrediction


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
        assert math.isfinite(on_node.v_cell_V)
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


class TestValidityGuard:
    """The surrogate must not silently extrapolate nonsense outside its grid.

    The interpolators extrapolate linearly (bounds_error=False, fill_value=None),
    which once returned physically impossible values (negative deposit rate,
    ~30 V cell) during transients.  predict() flags out-of-grid queries and
    clamps the physical outputs to the calibrated envelope.
    """

    def test_in_bounds_is_flagged_false(self, tmp_path):
        m = _small_model(tmp_path)
        assert m.in_bounds(125.0, 60.0, 1.0)
        assert m.predict(125.0, 60.0, 1.0).extrapolated is False
        assert not m.in_bounds(800.0, 60.0, 1.0)
        assert not m.in_bounds(125.0, 5.0, 1.0)
        assert not m.in_bounds(125.0, 60.0, 5.0)

    def test_oob_query_is_flagged_and_clamped(self, tmp_path):
        m = _small_model(tmp_path)
        p = m.predict(800.0, 5.0, 0.01)  # far outside every axis
        assert p.extrapolated is True
        # Physical impossibility clamps: no negative growth, no absurd cell V.
        assert p.deposit_rate_um_hr >= 0.0
        assert m.dep_map.min() <= p.deposit_rate_um_hr <= m.dep_map.max()
        assert m.vcell_map.min() <= p.v_cell_V <= m.vcell_map.max()

    def test_deposit_rate_never_negative(self, tmp_path):
        m = _small_model(tmp_path)
        # Sweep a wide OOB neighbourhood; no query may return negative growth.
        for j, T, fe2 in ((800.0, 60.0, 1.0), (0.0, 60.0, 1.0),
                          (125.0, 5.0, 1.0), (125.0, 100.0, 1.0),
                          (125.0, 60.0, 0.0), (300.0, 100.0, 3.0)):
            assert m.predict(j, T, fe2).deposit_rate_um_hr >= 0.0, (j, T, fe2)

    def test_in_bounds_predict_preserved(self, tmp_path):
        # Guard is a pure no-op for in-grid queries (backward compatible).
        m = _small_model(tmp_path)
        assert m.predict(125.0, 60.0, 1.0).extrapolated is False

    def test_grid_bounds_reports_validity_envelope(self, tmp_path):
        m = _small_model(tmp_path)
        b = m.grid_bounds
        assert b["j_mA_cm2"] == (100.0, 150.0)
        assert b["temperature_C"] == (50.0, 70.0)
        assert b["fe2_M"] == (0.8, 1.2)
