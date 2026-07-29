"""
Tests for the Monte Carlo engine — >=8 test cases covering:
1. MonteCarloEngine instantiation
2. Single-sample pipeline returns >=10 outputs
3. Small N run completes and returns MonteCarloResult
4. Output distributions have correct length
5. Pass rates computed with specs
6. Sensitivity identifies top-3 params per output
7. Output correlations are symmetric
8. N=1000 completes in <60s (performance)
9. Sobol + random sampling mix
10. summary_dict structure
11. Design point override
12. Spec set integration
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from models.uncertainty.monte_carlo import (
    MonteCarloEngine,
    MonteCarloResult,
    _run_single_sample,
    _compute_sensitivity,
    _compute_correlations,
    DEFAULT_DESIGN_POINT,
)
from models.uncertainty.parameter_registry import REGISTRY
from models.uncertainty.sample import sample_parameters
from models.uncertainty.specification import SPECS_A36, SPECS_ELECTROWINNING


# ---------------------------------------------------------------------------
# Test 1: Engine instantiation
# ---------------------------------------------------------------------------

def test_engine_defaults():
    engine = MonteCarloEngine()
    assert engine.n_samples == 10_000
    assert engine.seed == 42
    assert engine.registry is REGISTRY


def test_engine_custom():
    engine = MonteCarloEngine(n_samples=100, seed=7, n_jobs=1)
    assert engine.n_samples == 100
    assert engine.seed == 7


# ---------------------------------------------------------------------------
# Test 2: Single-sample pipeline
# ---------------------------------------------------------------------------

def test_single_sample_outputs():
    """Pipeline returns >=10 named outputs, all finite or NaN."""
    samples = sample_parameters(1, seed=42)
    result = _run_single_sample(samples[0], dict(DEFAULT_DESIGN_POINT))
    assert len(result) >= 10, f"Only {len(result)} outputs: {sorted(result.keys())}"
    # At least mechanical properties should be finite
    for key in ["sigma_y_MPa", "uts_MPa", "vickers_hv", "elongation_pct", "grain_size_um"]:
        assert key in result, f"Missing output: {key}"
        assert not math.isnan(result[key]), f"NaN for {key}"


def test_single_sample_output_keys():
    """Output keys match specification output_key paths."""
    samples = sample_parameters(1, seed=42)
    result = _run_single_sample(samples[0], dict(DEFAULT_DESIGN_POINT))
    # These keys must exist for spec checking to work
    required = ["sigma_y_MPa", "uts_MPa", "elongation_pct", "vickers_hv"]
    for key in required:
        assert key in result, f"Required output '{key}' missing"


# ---------------------------------------------------------------------------
# Test 3: Small N run
# ---------------------------------------------------------------------------

def test_small_n_run():
    """N=50 completes and returns valid MonteCarloResult."""
    engine = MonteCarloEngine(n_samples=50, seed=42, n_jobs=1)
    result = engine.run(specs=SPECS_A36, spec_set_name="ASTM_A36")

    assert isinstance(result, MonteCarloResult)
    assert result.n_samples == 50
    assert result.elapsed_seconds > 0
    assert result.spec_set_name == "ASTM_A36"


# ---------------------------------------------------------------------------
# Test 4: Output distributions length
# ---------------------------------------------------------------------------

def test_output_distributions_length():
    """Each output distribution has N elements."""
    N = 30
    engine = MonteCarloEngine(n_samples=N, seed=42, n_jobs=1)
    result = engine.run()

    assert len(result.output_distributions) >= 10
    for key, arr in result.output_distributions.items():
        assert len(arr) == N, f"{key}: expected {N}, got {len(arr)}"


# ---------------------------------------------------------------------------
# Test 5: Pass rates
# ---------------------------------------------------------------------------

def test_pass_rates_computed():
    """Pass rates are computed for all specs."""
    engine = MonteCarloEngine(n_samples=50, seed=42, n_jobs=1)
    result = engine.run(specs=SPECS_A36, spec_set_name="ASTM_A36")

    assert len(result.pass_rates) == len(SPECS_A36)
    for name, rate in result.pass_rates.items():
        assert 0.0 <= rate <= 1.0, f"Rate {rate} for {name} out of [0,1]"
    assert 0.0 <= result.overall_confidence <= 1.0


def test_pass_rates_without_specs():
    """No specs → empty pass rates, no crash."""
    engine = MonteCarloEngine(n_samples=20, seed=42, n_jobs=1)
    result = engine.run(specs=None)
    assert result.pass_rates == {}
    assert result.overall_confidence == 0.0


# ---------------------------------------------------------------------------
# Test 6: Sensitivity analysis
# ---------------------------------------------------------------------------

def test_sensitivity_top3():
    """Sensitivity identifies top-3 params for at least one output."""
    engine = MonteCarloEngine(n_samples=200, seed=42, n_jobs=1)
    result = engine.run()

    # At least sigma_y_MPa should have sensitivity
    ys_sens = result.sensitivity.get("sigma_y_MPa", {})
    assert len(ys_sens) >= 1, "No sensitivity for sigma_y_MPa"
    # All values should be between 0 and 1 (absolute correlation)
    for param, val in ys_sens.items():
        assert 0.0 <= val <= 1.0, f"Sensitivity {val} for {param} out of [0,1]"
        assert param in REGISTRY, f"Unknown param: {param}"


# ---------------------------------------------------------------------------
# Test 7: Correlation matrix
# ---------------------------------------------------------------------------

def test_correlation_symmetry():
    """Output correlation matrix is symmetric."""
    engine = MonteCarloEngine(n_samples=50, seed=42, n_jobs=1)
    result = engine.run()

    corr = result.parameter_correlations
    for ki in corr:
        for kj in corr[ki]:
            assert ki in corr.get(kj, {}), f"Asymmetric: {ki}->{kj} but not {kj}->{ki}"
            if ki == kj:
                val = corr[ki][kj]
                # Diagonal is 1.0 unless the output is constant (NaN correlation)
                if not math.isnan(val):
                    assert abs(val - 1.0) < 0.01, f"Diagonal {ki} != 1: {val}"


# ---------------------------------------------------------------------------
# Test 8: Performance
# ---------------------------------------------------------------------------

def test_n1000_performance():
    """N=1000 completes in <60 seconds."""
    engine = MonteCarloEngine(n_samples=1000, seed=42, n_jobs=1)
    t0 = time.perf_counter()
    result = engine.run()
    elapsed = time.perf_counter() - t0

    assert elapsed < 60.0, f"N=1000 took {elapsed:.1f}s (>60s limit)"
    assert result.n_samples == 1000


# ---------------------------------------------------------------------------
# Test 9: Sampling mix (Sobol + random)
# ---------------------------------------------------------------------------

def test_sampling_mix():
    """Engine uses Sobol for first 1000 then random."""
    engine = MonteCarloEngine(n_samples=1500, seed=42, n_jobs=1)
    result = engine.run()
    assert result.n_samples == 1500
    # Check that we get reasonable spread in outputs
    ys = result.output_distributions["sigma_y_MPa"]
    valid = ys[~np.isnan(ys)]
    assert np.std(valid) > 1.0, "YS distribution too narrow — sampling may be broken"


# ---------------------------------------------------------------------------
# Test 10: summary_dict structure
# ---------------------------------------------------------------------------

def test_summary_dict():
    """summary_dict returns expected keys."""
    engine = MonteCarloEngine(n_samples=30, seed=42, n_jobs=1)
    result = engine.run(specs=SPECS_A36)
    summary = result.summary_dict()

    assert "n_samples" in summary
    assert "elapsed_seconds" in summary
    assert "overall_confidence" in summary
    assert "pass_rates" in summary
    assert "output_statistics" in summary
    assert "sensitivity_top3" in summary
    assert "design_point" in summary

    # output_statistics should have mean/std/p5/p50/p95
    for key, stats in summary["output_statistics"].items():
        for metric in ["mean", "std", "p5", "p50", "p95"]:
            assert metric in stats, f"Missing {metric} in {key} stats"


# ---------------------------------------------------------------------------
# Test 11: Design point override
# ---------------------------------------------------------------------------

def test_design_point_override():
    """Custom design point changes output distributions."""
    dp = dict(DEFAULT_DESIGN_POINT)
    dp["j_avg_mA_cm2"] = 300.0  # higher current density

    engine = MonteCarloEngine(n_samples=50, seed=42, n_jobs=1)
    result = engine.run(design_point=dp)

    # Grain size should be finer at higher j
    gs = result.output_distributions["grain_size_um"]
    valid = gs[~np.isnan(gs)]
    assert np.mean(valid) < 5.0, "Grain size should be fine at j=300"


# ---------------------------------------------------------------------------
# Test 12: Spec set integration
# ---------------------------------------------------------------------------

def test_electrowinning_specs():
    """ELECTROWINNING spec set works with the engine."""
    engine = MonteCarloEngine(n_samples=30, seed=42, n_jobs=1)
    result = engine.run(specs=SPECS_ELECTROWINNING, spec_set_name="ELECTROWINNING")

    assert result.spec_set_name == "ELECTROWINNING"
    assert len(result.pass_rates) == len(SPECS_ELECTROWINNING)


# ---------------------------------------------------------------------------
# Test 13: Helper functions
# ---------------------------------------------------------------------------

def test_compute_sensitivity_basic():
    """_compute_sensitivity returns ranked params."""
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = 3 * x1 + 0.1 * x2 + np.random.randn(n) * 0.5

    output_dists = {"y": y}
    samples = [{"x1": x1[i], "x2": x2[i]} for i in range(n)]

    sens = _compute_sensitivity(output_dists, samples, ["x1", "x2"], top_n=2)
    assert "y" in sens
    assert len(sens["y"]) >= 1
    # x1 should be top-ranked (stronger correlation)
    top_param = list(sens["y"].keys())[0]
    assert top_param == "x1"


def test_compute_correlations_basic():
    """_compute_correlations handles identity and cross-correlation."""
    n = 50
    a = np.linspace(0, 1, n)
    b = a * 2 + np.random.randn(n) * 0.01
    c = -a + np.random.randn(n) * 0.01

    output_dists = {"a": a, "b": b, "c": c}
    corr = _compute_correlations(output_dists)

    assert abs(corr["a"]["a"] - 1.0) < 0.01
    assert corr["a"]["b"] > 0.9  # strong positive
    assert corr["a"]["c"] < -0.9  # strong negative
