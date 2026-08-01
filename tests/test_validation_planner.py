"""
Tests for the validation experiment planner — >=6 test cases covering:
1. Experiment catalog has >=7 types
2. plan_validation_experiments returns ranked plan
3. Gain per dollar ordering is correct
4. Sequential planner re-ranks after completion
5. Uncertainty trajectory shows convergence
6. Cost budget constraint works
7. All experiments constrain at least one registry parameter
8. Trajectory monotonically decreasing variance
"""

from __future__ import annotations

import pytest

from models.uncertainty.parameter_registry import REGISTRY
from models.uncertainty.validation_planner import (
    ValidationPlan,
    UncertaintyTrajectory,
    experiment_catalog,
    plan_validation_experiments,
    sequential_planner,
    uncertainty_reduction_trajectory,
    _experiment_information_gain,
    _param_variance,
)


# ---------------------------------------------------------------------------
# Test 1: Catalog completeness
# ---------------------------------------------------------------------------

def test_catalog_covers_at_least_7_types():
    """Catalog must include >=7 experiment types."""
    catalog = experiment_catalog()
    assert len(catalog) >= 7, f"Catalog has {len(catalog)} types, need >= 7"
    expected_names = {
        "foil_test", "tensile_test", "ebsd_grain_size", "vickers_hardness",
        "o2_probe_calibration", "icp_oes_composition", "tempering_curve",
    }
    assert expected_names.issubset(set(catalog.keys())), (
        f"Missing experiments: {expected_names - set(catalog.keys())}"
    )


# ---------------------------------------------------------------------------
# Test 2: Plan returns valid structure
# ---------------------------------------------------------------------------

def test_plan_returns_validation_plan():
    """plan_validation_experiments returns a well-formed ValidationPlan."""
    plan = plan_validation_experiments(REGISTRY, budget=7)
    assert isinstance(plan, ValidationPlan)
    assert len(plan.experiments) > 0
    assert plan.total_cost_usd > 0
    assert plan.total_duration_hours > 0
    assert len(plan.gain_per_dollar) == len(plan.experiments)


# ---------------------------------------------------------------------------
# Test 3: Experiments ordered by gain per dollar
# ---------------------------------------------------------------------------

def test_experiments_ordered_by_gain_per_dollar():
    """Plan experiments should be sorted descending by gain/$."""
    plan = plan_validation_experiments(REGISTRY, budget=7)
    for i in range(len(plan.gain_per_dollar) - 1):
        assert plan.gain_per_dollar[i] >= plan.gain_per_dollar[i + 1] - 1e-12, (
            f"Experiment {i} gain/${plan.gain_per_dollar[i]:.4e} < "
            f"{i+1} gain/${plan.gain_per_dollar[i+1]:.4e}"
        )


# ---------------------------------------------------------------------------
# Test 4: Sequential planner re-ranks
# ---------------------------------------------------------------------------

def test_sequential_planner_removes_completed():
    """After completing an experiment, sequential planner excludes it."""
    plan_initial = plan_validation_experiments(REGISTRY, budget=7)
    first_exp = plan_initial.experiments[0].name

    remaining = sequential_planner(REGISTRY, completed=[first_exp])
    remaining_names = [e.name for e in remaining.experiments]
    assert first_exp not in remaining_names, (
        f"Completed experiment '{first_exp}' still in remaining plan"
    )
    # Should have one fewer
    assert len(remaining.experiments) == len(plan_initial.experiments) - 1


# ---------------------------------------------------------------------------
# Test 5: Trajectory shows convergence
# ---------------------------------------------------------------------------

def test_trajectory_shows_convergence():
    """Uncertainty trajectory should decrease monotonically."""
    plan = plan_validation_experiments(REGISTRY, budget=7)
    traj = uncertainty_reduction_trajectory(REGISTRY, plan)

    assert isinstance(traj, UncertaintyTrajectory)
    assert len(traj.experiment_names) == len(plan.experiments)
    assert traj.remaining_variance_frac[0] < 1.0  # first experiment reduces

    # Monotonic decrease (non-increasing)
    for i in range(1, len(traj.remaining_variance_frac)):
        assert traj.remaining_variance_frac[i] <= traj.remaining_variance_frac[i - 1] + 1e-10, (
            f"Variance increased at step {i}: "
            f"{traj.remaining_variance_frac[i-1]:.4f} -> {traj.remaining_variance_frac[i]:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 6: Cost budget constraint
# ---------------------------------------------------------------------------

def test_cost_budget_limits_selection():
    """A tight cost budget should exclude expensive experiments."""
    # $300 budget can only afford the cheapest experiments
    plan = plan_validation_experiments(REGISTRY, budget=7, cost_budget_usd=300.0)
    assert plan.total_cost_usd <= 300.0 + 1e-6
    assert len(plan.experiments) >= 1, "Should fit at least one experiment under $300"


# ---------------------------------------------------------------------------
# Test 7: All experiments constrain real registry params
# ---------------------------------------------------------------------------

def test_all_experiments_constrain_registry_params():
    """Every experiment's constrained_params should exist in REGISTRY."""
    catalog = experiment_catalog()
    registry_keys = set(REGISTRY.keys())
    for name, exp in catalog.items():
        for param in exp.constrained_params:
            assert param in registry_keys, (
                f"Experiment '{name}' constrains '{param}' which is not in REGISTRY"
            )


# ---------------------------------------------------------------------------
# Test 8: Trajectory final variance is significantly reduced
# ---------------------------------------------------------------------------

def test_trajectory_final_variance_below_50_pct():
    """After all 7 experiments, remaining variance should be <50%."""
    plan = plan_validation_experiments(REGISTRY, budget=7)
    traj = uncertainty_reduction_trajectory(REGISTRY, plan)
    final_pct = traj.remaining_variance_frac[-1] * 100
    assert final_pct < 50.0, (
        f"Final remaining variance {final_pct:.1f}% is too high, expected <50%"
    )


# ---------------------------------------------------------------------------
# Test 9: Information gain function
# ---------------------------------------------------------------------------

def test_experiment_information_gain_positive():
    """Every catalog experiment should have positive information gain."""
    catalog = experiment_catalog()
    for name, exp in catalog.items():
        gain = _experiment_information_gain(exp, REGISTRY)
        assert gain > 0, f"Experiment '{name}' has zero information gain"


# ---------------------------------------------------------------------------
# Test 10: Param variance helper
# ---------------------------------------------------------------------------

def test_param_variance_positive():
    """Parameters with positive std should have positive variance."""
    p = {"std": 5.0, "bounds": (0.0, 10.0)}
    assert _param_variance(p) == 25.0

    # Uniform fallback
    p2 = {"std": 0.0, "bounds": (0.0, 12.0)}
    assert _param_variance(p2) == pytest.approx(144.0 / 12.0, rel=1e-10)
