"""
Tests for the design space explorer — >=6 test cases covering:
1. explore_design_space with 2-D grid returns DesignSpaceResult
2. Confidence values in [0, 1] range
3. Interpolation predict function works
4. Best point matches argmax of confidence_values
5. robust_optimum returns RobustOptimum with design margins
6. pareto_front_robust returns non-dominated front
7. Pareto front is properly non-dominated (no point dominates another)
8. _make_design_point correctly maps sweep names
9. _estimate_cost returns positive values
10. DesignSpaceResult.summary_dict structure
11. ParetoFront.summary_dict structure
12. RobustOptimum.summary_dict structure
"""

from __future__ import annotations

import numpy as np
import pytest

from models.uncertainty.design_space import (
    DesignSpaceResult,
    ParetoFront,
    ParetoPoint,
    RobustOptimum,
    _estimate_cost,
    _extract_pareto,
    _make_design_point,
    explore_design_space,
    pareto_front_robust,
    robust_optimum,
)
from models.uncertainty.monte_carlo import DEFAULT_DESIGN_POINT
from models.uncertainty.specification import SPECS_A36, SPECS_ELECTROWINNING


# ---------------------------------------------------------------------------
# Shared fixtures — tiny grid for fast tests
# ---------------------------------------------------------------------------

TINY_RANGES = {
    "j_avg": (100.0, 200.0),
    "T_bath": (40.0, 70.0),
}

MICRO_RANGES = {
    "j_avg": (120.0, 180.0),
}


@pytest.fixture(scope="module")
def ds_result_2d():
    """2-D grid exploration with MC=50 per point (fast)."""
    return explore_design_space(
        ranges=TINY_RANGES,
        specs=SPECS_A36,
        n_grid=3,
        mc_samples=50,
        seed=42,
        n_jobs=1,
        spec_set_name="test_A36",
    )


@pytest.fixture(scope="module")
def ds_result_1d():
    """1-D grid exploration (very fast)."""
    return explore_design_space(
        ranges=MICRO_RANGES,
        specs=SPECS_A36,
        n_grid=4,
        mc_samples=30,
        seed=42,
        n_jobs=1,
    )


# ---------------------------------------------------------------------------
# Test 1: explore_design_space returns DesignSpaceResult
# ---------------------------------------------------------------------------

def test_explore_returns_result(ds_result_2d):
    assert isinstance(ds_result_2d, DesignSpaceResult)
    assert ds_result_2d.n_grid == 3
    assert ds_result_2d.mc_samples_per_point == 50
    assert ds_result_2d.param_names == ["T_bath", "j_avg"]  # sorted
    assert len(ds_result_2d.grid_points) == 9
    assert len(ds_result_2d.confidence_values) == 9
    assert len(ds_result_2d.mc_results) == 9
    assert ds_result_2d.spec_set_name == "test_A36"
    assert ds_result_2d.elapsed_seconds > 0


# ---------------------------------------------------------------------------
# Test 2: Confidence values in [0, 1]
# ---------------------------------------------------------------------------

def test_confidence_bounded(ds_result_2d):
    vals = ds_result_2d.confidence_values
    assert np.all(vals >= 0.0), f"Negative confidence: {vals}"
    assert np.all(vals <= 1.0), f"Confidence > 1: {vals}"


# ---------------------------------------------------------------------------
# Test 3: Interpolation predict works
# ---------------------------------------------------------------------------

def test_interpolation_predict(ds_result_2d):
    # Should be able to predict at the grid points
    pts = ds_result_2d.grid_points
    pred = ds_result_2d.predict(pts)
    assert len(pred) == len(pts)
    assert np.all(np.isfinite(pred))

    # If all confidences are identical (common with very small MC), skip correlation
    vals = ds_result_2d.confidence_values
    if np.std(vals) < 1e-10:
        # Constant surface — interpolation should return approximately constant
        assert np.std(pred) < 0.5
    else:
        corr = np.corrcoef(vals, pred)[0, 1]
        assert corr > 0.3, f"Interpolation correlation too low: {corr:.3f}"


# ---------------------------------------------------------------------------
# Test 4: Best point matches argmax
# ---------------------------------------------------------------------------

def test_best_point_matches_argmax(ds_result_2d):
    best = ds_result_2d.best_point
    idx = int(np.argmax(ds_result_2d.confidence_values))
    for i, name in enumerate(ds_result_2d.param_names):
        assert abs(best[name] - ds_result_2d.grid_points[idx, i]) < 1e-10
    assert abs(ds_result_2d.max_confidence - ds_result_2d.confidence_values[idx]) < 1e-10


# ---------------------------------------------------------------------------
# Test 5: robust_optimum returns valid RobustOptimum
# ---------------------------------------------------------------------------

def test_robust_optimum_result():
    result = robust_optimum(
        ranges=MICRO_RANGES,
        specs=SPECS_A36,
        n_calls=15,         # very few for speed
        target=0.95,
        mc_samples=30,
        seed=42,
        n_jobs=1,
        spec_set_name="test_opt",
    )

    assert isinstance(result, RobustOptimum)
    assert "j_avg" in result.optimum_point
    assert 0.0 <= result.optimum_confidence <= 1.0
    assert result.n_calls == 15
    assert result.target == 0.95
    assert result.elapsed_seconds > 0
    assert isinstance(result.achieved_target, bool)
    assert len(result.all_evaluations) > 0
    assert len(result.all_confidences) > 0
    assert result.spec_set_name == "test_opt"

    # Design margins should be present
    assert len(result.design_margins) > 0
    for param, margin in result.design_margins.items():
        assert "optimum" in margin
        assert "lo_90" in margin
        assert "hi_90" in margin


# ---------------------------------------------------------------------------
# Test 6: pareto_front_robust returns non-dominated front
# ---------------------------------------------------------------------------

def test_pareto_front_result():
    result = pareto_front_robust(
        objectives=["confidence", "cost", "energy"],
        ranges=MICRO_RANGES,
        specs=SPECS_A36,
        n_grid=3,
        mc_samples=30,
        seed=42,
        n_jobs=1,
        spec_set_name="test_pareto",
    )

    assert isinstance(result, ParetoFront)
    assert len(result.points) > 0
    assert result.n_evaluated == 3
    assert result.elapsed_seconds > 0
    assert result.spec_set_name == "test_pareto"

    # Each point should be a ParetoPoint
    for p in result.points:
        assert isinstance(p, ParetoPoint)
        assert 0.0 <= p.confidence <= 1.0
        assert p.cost > 0
        assert p.energy > 0


# ---------------------------------------------------------------------------
# Test 7: Pareto front is non-dominated
# ---------------------------------------------------------------------------

def test_pareto_non_dominated():
    """Verify no point on the front dominates another."""
    candidates = [
        ParetoPoint(point={"x": 1.0}, confidence=0.8, cost=2.0, energy=5.0),
        ParetoPoint(point={"x": 2.0}, confidence=0.9, cost=1.5, energy=6.0),
        ParetoPoint(point={"x": 3.0}, confidence=0.7, cost=3.0, energy=4.0),  # dominated
        ParetoPoint(point={"x": 4.0}, confidence=0.95, cost=1.0, energy=5.5),
    ]
    front, dominated = _extract_pareto(candidates)

    assert dominated == 1  # point 3 is dominated by point 4
    assert len(front) == 3

    # Verify non-domination
    for i, p in enumerate(front):
        for j, q in enumerate(front):
            if i == j:
                continue
            # q should NOT dominate p
            q_better_conf = q.confidence >= p.confidence
            q_better_cost = q.cost <= p.cost
            q_better_energy = q.energy <= p.energy
            if q_better_conf and q_better_cost and q_better_energy:
                # At least one must be equal (not strictly better)
                assert (q.confidence == p.confidence and
                        q.cost == p.cost and
                        q.energy == p.energy), \
                    f"Point {q.point} dominates {p.point} on front"


# ---------------------------------------------------------------------------
# Test 8: _make_design_point maps sweep names correctly
# ---------------------------------------------------------------------------

def test_make_design_point():
    base = dict(DEFAULT_DESIGN_POINT)
    dp = _make_design_point(base, {
        "j_avg": 200.0,
        "T_bath": 55.0,
        "pH": 3.0,
        "Ni_conc": 0.8,
        "carburizing_T": 920.0,
        "tempering_T": 250.0,
        "duty": 0.3,
    })

    assert dp["j_avg_mA_cm2"] == 200.0
    assert dp["temperature_C"] == 55.0
    assert dp["pH"] == 3.0
    assert dp["bath_ni_M"] == 0.8
    assert dp["carburizing_temperature_C"] == 920.0
    assert dp["tempering_temperature_C"] == 250.0
    assert dp["duty_cycle"] == 0.3
    # j_peak should be j_avg / duty = 200 / 0.3
    assert abs(dp["j_peak_mA_cm2"] - 200.0 / 0.3) < 1e-6


# ---------------------------------------------------------------------------
# Test 9: _estimate_cost returns positive values
# ---------------------------------------------------------------------------

def test_estimate_cost():
    base = dict(DEFAULT_DESIGN_POINT)
    cost_default = _estimate_cost(base)
    assert cost_default > 0

    # Higher current density → higher cost
    dp_high_j = dict(base)
    dp_high_j["j_avg_mA_cm2"] = 300.0
    assert _estimate_cost(dp_high_j) > cost_default

    # Higher carburizing temp → higher cost
    dp_high_t = dict(base)
    dp_high_t["carburizing_temperature_C"] = 950.0
    assert _estimate_cost(dp_high_t) > cost_default


# ---------------------------------------------------------------------------
# Test 10: DesignSpaceResult.summary_dict structure
# ---------------------------------------------------------------------------

def test_design_space_summary_dict(ds_result_2d):
    d = ds_result_2d.summary_dict()
    assert "param_names" in d
    assert "ranges" in d
    assert "n_grid" in d
    assert "mc_samples_per_point" in d
    assert "max_confidence" in d
    assert "best_point" in d
    assert "elapsed_seconds" in d
    assert "grid_confidences" in d
    assert len(d["grid_confidences"]) == 9
    assert isinstance(d["max_confidence"], float)


# ---------------------------------------------------------------------------
# Test 11: ParetoFront.summary_dict structure
# ---------------------------------------------------------------------------

def test_pareto_summary_dict():
    front = ParetoFront(
        points=[
            ParetoPoint(point={"x": 1.0}, confidence=0.9, cost=1.5, energy=5.0),
            ParetoPoint(point={"x": 2.0}, confidence=0.8, cost=1.0, energy=4.0),
        ],
        dominated_count=3,
        n_evaluated=5,
        elapsed_seconds=10.0,
    )
    d = front.summary_dict()
    assert d["n_front_points"] == 2
    assert d["dominated_count"] == 3
    assert d["n_evaluated"] == 5
    assert len(d["front"]) == 2
    assert "confidence" in d["front"][0]
    assert "cost" in d["front"][0]
    assert "energy" in d["front"][0]


# ---------------------------------------------------------------------------
# Test 12: RobustOptimum.summary_dict structure
# ---------------------------------------------------------------------------

def test_robust_optimum_summary_dict():
    opt = RobustOptimum(
        optimum_point={"j_avg": 150.0},
        optimum_confidence=0.92,
        achieved_target=False,
        all_evaluations=np.array([[150.0]]),
        all_confidences=np.array([0.92]),
        n_calls=10,
        target=0.95,
        elapsed_seconds=5.0,
        design_margins={"j_avg": {
            "optimum": 150.0, "lo_90": 120.0, "hi_90": 200.0,
            "margin_lo_pct": 20.0, "margin_hi_pct": 30.0,
        }},
    )
    d = opt.summary_dict()
    assert d["optimum_confidence"] == 0.92
    assert d["achieved_target"] is False
    assert d["target"] == 0.95
    assert "j_avg" in d["design_margins"]
    assert d["design_margins"]["j_avg"]["lo_90"] == 120.0
