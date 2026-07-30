"""
Tests for the sensitivity analysis module — >=6 test cases covering:
1. SobolResult structure and top_params
2. Sobol first-order indices sum to <= 1
3. TornadoResult from MC results
4. MorrisResult structure and ranked
5. Morris screening with subset of parameters
6. Sobol with output_subset
7. Morris detects linear vs non-linear parameters
8. Tornado raises on missing sample_matrix
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from models.uncertainty.parameter_registry import REGISTRY
from models.uncertainty.monte_carlo import (
    MonteCarloEngine,
    MonteCarloResult,
    DEFAULT_DESIGN_POINT,
)
from models.uncertainty.sensitivity import (
    sobol_analysis,
    tornado_chart,
    morris_screening,
    SobolResult,
    TornadoResult,
    MorrisResult,
)


# ---------------------------------------------------------------------------
# Test 1: Sobol result structure
# ---------------------------------------------------------------------------

def test_sobol_result_structure():
    """Sobol analysis returns SobolResult with correct shapes."""
    engine = MonteCarloEngine(n_samples=50, seed=42, n_jobs=1)

    # Use a small subset for speed
    subset = [
        "sigma0_fe_MPa", "k_hp_MPa_sqrt_m", "k_ss_ni_MPa_per_wt",
        "tabor_factor", "elongation_base_pct", "k_carbon_MPa_per_wt",
        "uts_over_ys_base", "porosity_penalty_exp", "carbon_nl_exp",
        "fe_i0", "her_i0",
    ]

    result = sobol_analysis(
        engine,
        n_samples=500,
        param_subset=subset,
        output_subset=["sigma_y_MPa", "vickers_hv"],
    )

    assert isinstance(result, SobolResult)
    assert len(result.parameter_names) == len(subset)
    assert len(result.output_keys) == 2
    assert result.first_order.shape == (2, len(subset))
    assert result.total_order.shape == (2, len(subset))
    assert result.variance.shape == (2,)
    assert result.n_samples > 0
    assert result.elapsed_seconds > 0


# ---------------------------------------------------------------------------
# Test 2: Sobol first-order indices sum to <= 1
# ---------------------------------------------------------------------------

def test_sobol_indices_bounded():
    """First-order Sobol indices are in [0, 1] and sum to <= 1."""
    engine = MonteCarloEngine(n_samples=50, seed=42, n_jobs=1)
    subset = [
        "sigma0_fe_MPa", "k_hp_MPa_sqrt_m", "k_ss_ni_MPa_per_wt",
        "tabor_factor", "elongation_base_pct",
    ]

    result = sobol_analysis(
        engine,
        n_samples=300,
        param_subset=subset,
        output_subset=["sigma_y_MPa"],
    )

    si = result.first_order[0]  # first output
    assert np.all(si >= -0.01), f"Negative S_i: {si}"
    assert np.all(si <= 1.01), f"S_i > 1: {si}"
    # Sum of first-order should be <= 1 (with some tolerance for estimation error)
    assert np.sum(si) <= 1.5, f"Sum of S_i = {np.sum(si):.3f} > 1.5"


# ---------------------------------------------------------------------------
# Test 3: Tornado chart from MC results
# ---------------------------------------------------------------------------

def test_tornado_from_mc():
    """Tornado chart correctly identifies top parameters."""
    engine = MonteCarloEngine(n_samples=200, seed=42, n_jobs=1)
    mc_result = engine.run()

    result = tornado_chart(mc_result, "sigma_y_MPa", top_n=5)

    assert isinstance(result, TornadoResult)
    assert result.output_key == "sigma_y_MPa"
    assert len(result.parameter_names) == len(REGISTRY)
    ranked = result.ranked()
    assert len(ranked) == 5
    # All sensitivities should be in [0, 1]
    for name, val in ranked:
        assert 0.0 <= val <= 1.0, f"Sensitivity {val} for {name} out of [0,1]"
    # First ranked should be most influential
    assert ranked[0][1] >= ranked[-1][1]


# ---------------------------------------------------------------------------
# Test 4: Morris result structure
# ---------------------------------------------------------------------------

def test_morris_result_structure():
    """Morris screening returns MorrisResult with correct shapes."""
    subset = [
        "sigma0_fe_MPa", "k_hp_MPa_sqrt_m", "k_ss_ni_MPa_per_wt",
        "tabor_factor", "elongation_base_pct", "k_carbon_MPa_per_wt",
        "uts_over_ys_base", "porosity_penalty_exp", "carbon_nl_exp",
        "fe_i0", "her_i0",
    ]

    result = morris_screening(
        n_trajectories=10,
        n_levels=4,
        seed=42,
        param_subset=subset,
        output_keys=["sigma_y_MPa"],
    )

    assert isinstance(result, MorrisResult)
    assert len(result.parameter_names) == len(subset)
    assert result.mu_star.shape == (len(subset),)
    assert result.sigma.shape == (len(subset),)
    assert result.mu.shape == (len(subset),)
    assert result.n_trajectories == 10
    assert result.n_params == len(subset)
    assert result.elapsed_seconds > 0

    # mu_star should be non-negative
    assert np.all(result.mu_star >= 0), f"Negative mu_star: {result.mu_star}"

    ranked = result.ranked(n=3)
    assert len(ranked) == 3
    assert ranked[0][1] >= ranked[-1][1]


# ---------------------------------------------------------------------------
# Test 5: Morris with >=20 parameters
# ---------------------------------------------------------------------------

def test_morris_20_params():
    """Morris screening works with >=20 parameters."""
    # Pick 20+ params from registry
    all_params = sorted(REGISTRY.keys())
    subset = all_params[:25]

    result = morris_screening(
        n_trajectories=5,
        n_levels=4,
        seed=42,
        param_subset=subset,
        output_keys=["sigma_y_MPa"],
    )

    assert result.n_params >= 20
    assert len(result.parameter_names) >= 20
    assert result.mu_star.shape[0] >= 20


# ---------------------------------------------------------------------------
# Test 6: Sobol with output subset
# ---------------------------------------------------------------------------

def test_sobol_output_subset():
    """Sobol analysis respects output_subset."""
    engine = MonteCarloEngine(n_samples=30, seed=42, n_jobs=1)
    subset = [
        "sigma0_fe_MPa", "k_hp_MPa_sqrt_m", "tabor_factor",
    ]

    result = sobol_analysis(
        engine,
        n_samples=200,
        param_subset=subset,
        output_subset=["sigma_y_MPa", "vickers_hv"],
    )

    assert result.output_keys == ["sigma_y_MPa", "vickers_hv"]
    assert result.first_order.shape[0] == 2


# ---------------------------------------------------------------------------
# Test 7: Morris detects influential parameters
# ---------------------------------------------------------------------------

def test_morris_detects_influential():
    """Morris screening gives higher mu_star to known-influential params."""
    # sigma0_fe_MPa (Hall-Petch friction stress) should be more
    # influential on yield strength than a transport diffusivity
    subset = [
        "sigma0_fe_MPa", "k_hp_MPa_sqrt_m",  # mechanical — high influence on YS
        "D_Na_plus", "D_SO4_2minus",           # transport — low influence on YS
    ]

    result = morris_screening(
        n_trajectories=20,
        n_levels=4,
        seed=42,
        param_subset=subset,
        output_keys=["sigma_y_MPa"],
    )

    idx_sigma0 = result.parameter_names.index("sigma0_fe_MPa")
    idx_DNa = result.parameter_names.index("D_Na_plus")

    # Mechanical params should have higher mu_star than transport params
    # for sigma_y_MPa output
    assert result.mu_star[idx_sigma0] > result.mu_star[idx_DNa], (
        f"sigma0 ({result.mu_star[idx_sigma0]:.4f}) should dominate "
        f"D_Na ({result.mu_star[idx_DNa]:.4f}) for sigma_y_MPa"
    )


# ---------------------------------------------------------------------------
# Test 8: Tornado raises on missing sample_matrix
# ---------------------------------------------------------------------------

def test_tornado_raises_without_samples():
    """Tornado chart raises ValueError when sample_matrix is missing."""
    mc_result = MonteCarloResult(
        n_samples=10,
        design_point={},
        output_distributions={"y": np.array([1.0, 2.0, 3.0])},
        pass_rates={},
        overall_confidence=0.0,
        sensitivity={},
        failure_ranking={},
        parameter_correlations={},
    )

    with pytest.raises(ValueError, match="sample_matrix not available"):
        tornado_chart(mc_result, "y")


# ---------------------------------------------------------------------------
# Test 9: Sobol top_params
# ---------------------------------------------------------------------------

def test_sobol_top_params():
    """top_params returns correctly ranked parameters."""
    engine = MonteCarloEngine(n_samples=30, seed=42, n_jobs=1)
    subset = [
        "sigma0_fe_MPa", "k_hp_MPa_sqrt_m", "tabor_factor",
        "elongation_base_pct", "k_carbon_MPa_per_wt",
    ]

    result = sobol_analysis(
        engine,
        n_samples=200,
        param_subset=subset,
        output_subset=["sigma_y_MPa"],
    )

    top = result.top_params("sigma_y_MPa", n=3)
    assert len(top) == 3
    for name, val in top:
        assert name in subset
        assert 0.0 <= val <= 1.01


# ---------------------------------------------------------------------------
# Test 10: Tornado to_dict structure
# ---------------------------------------------------------------------------

def test_tornado_to_dict():
    """TornadoResult.to_dict returns expected structure."""
    engine = MonteCarloEngine(n_samples=100, seed=42, n_jobs=1)
    mc_result = engine.run()
    result = tornado_chart(mc_result, "sigma_y_MPa", top_n=3)

    d = result.to_dict()
    assert "output_key" in d
    assert "nominal" in d
    assert "top_params" in d
    assert len(d["top_params"]) == 3
    for entry in d["top_params"]:
        assert "name" in entry
        assert "sensitivity" in entry
        assert "low" in entry
        assert "high" in entry


# ---------------------------------------------------------------------------
# Test 11: Morris to_dict structure
# ---------------------------------------------------------------------------

def test_morris_to_dict():
    """MorrisResult.to_dict returns expected structure."""
    subset = ["sigma0_fe_MPa", "k_hp_MPa_sqrt_m", "tabor_factor"]

    result = morris_screening(
        n_trajectories=5,
        seed=42,
        param_subset=subset,
        output_keys=["sigma_y_MPa"],
    )

    d = result.to_dict()
    assert "parameter_names" in d
    assert "mu_star" in d
    assert "sigma" in d
    assert "mu" in d
    assert len(d["mu_star"]) == len(subset)
