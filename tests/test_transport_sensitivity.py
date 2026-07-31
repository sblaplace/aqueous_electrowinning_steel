"""
Tests for the Sobol global-sensitivity module over the 1D diffusion-layer
FE engine (``models/transport_sensitivity.py``).

Heavy ODE evaluation is avoided in most tests: the pure sampling/transform/
index math is checked against closed-form additive and interaction models,
and the orchestration pipeline is exercised with a cheap synthetic evaluator.
A single ODE test locks the ``fast_mode``-vs-tight equivalence the screening
evaluation relies on.
"""

import numpy as np

from models.transport_sensitivity import (
    OUTPUT_KEYS,
    _sobol_from_arrays,
    _transform,
    parameter_space,
    saltelli_matrices,
)
from models.diffusion_layer_1d import DiffusionLayer1D


# --------------------------------------------------------------------------
# Parameter space
# --------------------------------------------------------------------------

def test_parameter_space_has_ten_unique_levers():
    specs = parameter_space()
    assert len(specs) == 10
    names = [s.name for s in specs]
    assert len(set(names)) == len(names)
    # The 10 levers must be a superset of the key physics the model exposes.
    for key in ("j_mA_cm2", "fe_conc_M", "pH_bulk", "temperature_C",
                "delta_m", "her_i0", "fe_i0", "her_tafel_V"):
        assert key in names
    for s in specs:
        assert s.bounds[0] <= s.bounds[1]
        assert s.transform in ("linear", "log")
        assert s.advice and s.lever


# --------------------------------------------------------------------------
# Sampling / transform
# --------------------------------------------------------------------------

def test_saltelli_matrices_structure():
    n, d = 16, 10
    a, b, ab = saltelli_matrices(n, d, seed=0)
    assert a.shape == (n, d) and b.shape == (n, d)
    assert len(ab) == d
    for i, abi in enumerate(ab):
        assert abi.shape == (n, d)
        assert np.allclose(abi[:, i], a[:, i])       # column i from A
        # every other column matches B
        assert np.allclose(np.delete(abi, i, axis=1), np.delete(b, i, axis=1))
    # A and B are independent draws (not identical)
    assert not np.allclose(a, b)


def test_transform_linear_bounds():
    spec = parameter_space()[2]  # pH_bulk, linear [0.5, 4.0]
    assert spec.name == "pH_bulk"
    lo, hi = spec.bounds
    u = np.array([0.0, 0.5, 1.0])
    vals = _transform(u, spec)
    assert np.allclose(vals, [lo, (lo + hi) / 2.0, hi])


def test_transform_log_bounds():
    spec = parameter_space()[0]  # j_mA_cm2, log [20, 300]
    assert spec.name == "j_mA_cm2"
    lo, hi = spec.bounds
    vals = _transform(np.array([0.0, 1.0]), spec)
    assert np.isclose(vals[0], lo)
    assert np.isclose(vals[1], hi)
    # geometric (not arithmetic) midpoint in log space
    mid = _transform(np.array([0.5]), spec)[0]
    assert np.isclose(mid, np.sqrt(lo * hi))


# --------------------------------------------------------------------------
# Sobol index math on closed-form models
# --------------------------------------------------------------------------

def _sobol_for_model(func, n=2048, d=2, seed=3):
    a, b, ab = saltelli_matrices(n, d, seed=seed)
    y_a = np.array([func(row) for row in a])
    y_b = np.array([func(row) for row in b])
    y_ab = [np.array([func(row) for row in abi]) for abi in ab]
    return _sobol_from_arrays(y_a, y_b, y_ab)


def _additive(u):
    # y = 3*x0 + 1*x1, x uniform on [0,1): S1_0 = 9/10, S1_1 = 1/10.
    return 3.0 * u[0] + 1.0 * u[1]


def _interacting(u):
    # y = x0 + 2*x0*x1 has interaction => ST > S1 for both.
    return u[0] + 2.0 * u[0] * u[1]


def test_additive_linear_model_indices():
    s1, st = _sobol_for_model(_additive)
    assert np.isclose(s1[0], 0.9, atol=0.02)
    assert np.isclose(s1[1], 0.1, atol=0.02)
    # additive => total order == first order
    assert np.allclose(st, s1, atol=0.03)


def test_interaction_model_st_and_s1():
    s1, st = _sobol_for_model(_interacting)
    for i in range(2):
        assert st[i] >= s1[i] - 1e-9
        assert 0.0 <= s1[i] <= 1.0 and 0.0 <= st[i] <= 1.0
    assert st[0] > s1[0] + 0.05


def _constant(u):
    return 5.0


def test_indices_are_zero_for_constant_output():
    s1, st = _sobol_for_model(_constant)
    assert np.all(np.isnan(s1)) and np.all(np.isnan(st))


# --------------------------------------------------------------------------
# Orchestration pipeline (synthetic evaluator, no ODE)
# --------------------------------------------------------------------------

def test_run_analysis_pipeline_structure(monkeypatch):
    def fake_eval(point):
        # deterministic synthetic outputs depending on the first two levers
        return {
            "FE_pct": 60.0 + 30.0 * point["j_mA_cm2"] / 300.0,
            "V_cell_V": 2.0 + 0.5 * point["fe_conc_M"],
            "surface_pH": 2.5,
        }

    monkeypatch.setattr("models.transport_sensitivity._safe_eval", fake_eval)

    from models.transport_sensitivity import run_analysis

    result = run_analysis(n_samples=8, seed=1, n_workers=1)
    assert result.n_samples == 8
    assert result.n_failed == 0
    assert len(result.outputs) == len(OUTPUT_KEYS)
    fe = result.fe_output
    assert fe.output == "FE_pct"
    # With FE driven only by j, j_mA_cm2 must rank first.
    assert fe.rank_by_st[0] == "j_mA_cm2"
    assert len(result.recommendations) == 5
    assert result.recommendations[0]["parameter"] == "j_mA_cm2"

    summary = result.summary_dict()
    assert summary["n_evaluated"] == 8
    # summary must be JSON-serializable
    import json
    json.dumps(summary)


def test_run_analysis_handles_failures(monkeypatch):
    def failing_eval(point):
        if point["j_mA_cm2"] > 270.0:  # only extreme top of j-range fails
            raise RuntimeError("synthetic failure")
        return {
            "FE_pct": 50.0 + 40.0 * point["fe_conc_M"] / 2.0,
            "V_cell_V": 2.0 + 0.5 * point["fe_conc_M"],
            "surface_pH": 2.5 + 0.2 * point["fe_conc_M"],
        }

    # Patch the raw evaluator: _safe_eval wraps it and turns raises into NaN.
    monkeypatch.setattr("models.transport_sensitivity._eval_point", failing_eval)

    from models.transport_sensitivity import run_analysis

    result = run_analysis(n_samples=8, seed=1, n_workers=1)
    assert result.n_failed > 0
    assert result.n_evaluated > 0
    # dropped rows are filtered; surviving rows still yield finite indices
    for out in result.outputs:
        assert np.all(np.isfinite(out.s1))
        assert np.all(np.isfinite(out.st))


# --------------------------------------------------------------------------
# ODE-level: fast_mode equivalence (one operating point)
# --------------------------------------------------------------------------

def test_fast_mode_matches_tight_solver():
    kw = dict(fe_conc_M=0.8, pH_bulk=1.5, temperature_C=40.0,
              delta_m=40e-6, buffer_conc_M=0.2)
    tight = DiffusionLayer1D(**kw).solve(100.0)
    fast = DiffusionLayer1D(**kw, fast_mode=True).solve(100.0)
    assert abs(tight.fe_percent - fast.fe_percent) < 0.1
    assert abs(tight.V_cell - fast.V_cell) < 1e-3
