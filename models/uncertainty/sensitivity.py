"""
Sensitivity analysis for the aqueous electrowinning model chain.

Methods
-------
1. **Sobol indices** (Saltelli 2010): first-order S_i = V[E(Y|X_i)] / V(Y)
   and total-order S_Ti = 1 - V[E(Y|~X_i)] / V(Y).  Uses the Jansen (1999)
   estimator for total-order indices.
2. **Tornado charts**: rank parameters by conditional output spread
   from Monte Carlo results — for each parameter, samples are binned
   by that parameter's quantile, and the conditional output range
   determines the bar length.
3. **Morris screening** (elementary effects): OAT trajectories for
   high-dimensional screening.  mu_star = mean |EE| measures
   importance; sigma measures non-linearity / interactions.

All methods work with the MonteCarloEngine pipeline for model evaluation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .parameter_registry import Parameter, REGISTRY
from .sample import _transform
from .monte_carlo import (
    MonteCarloEngine,
    MonteCarloResult,
    _run_single_sample,
    DEFAULT_DESIGN_POINT,
)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class SobolResult:
    """First-order and total-order Sobol sensitivity indices.

    Attributes
    ----------
    parameter_names : list of str
        Ordered parameter names (columns of the index matrices).
    output_keys : list of str
        Ordered output names (rows of the index matrices).
    first_order : np.ndarray, shape (n_outputs, n_params)
        First-order Sobol indices S_i.
    total_order : np.ndarray, shape (n_outputs, n_params)
        Total-order Sobol indices S_Ti.
    variance : np.ndarray, shape (n_outputs,)
        Total output variance V(Y) for each output.
    n_samples : int
        Total model evaluations performed.
    elapsed_seconds : float
    """

    parameter_names: List[str]
    output_keys: List[str]
    first_order: np.ndarray      # (n_outputs, n_params)
    total_order: np.ndarray      # (n_outputs, n_params)
    variance: np.ndarray         # (n_outputs,)
    n_samples: int
    elapsed_seconds: float = 0.0

    def top_params(self, output_key: str, n: int = 5) -> List[Tuple[str, float]]:
        """Return top-n parameters by first-order index for an output."""
        idx = self.output_keys.index(output_key)
        si = self.first_order[idx]
        order = np.argsort(si)[::-1][:n]
        return [(self.parameter_names[i], float(si[i])) for i in order]

    def to_dict(self) -> Dict[str, Any]:
        """Machine-readable summary."""
        return {
            "parameter_names": self.parameter_names,
            "output_keys": self.output_keys,
            "first_order": self.first_order.tolist(),
            "total_order": self.total_order.tolist(),
            "variance": self.variance.tolist(),
            "n_samples": self.n_samples,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


@dataclass
class TornadoResult:
    """Tornado chart data: parameter rankings by sensitivity.

    Attributes
    ----------
    output_key : str
        Which output was analyzed.
    parameter_names : list of str
        Ordered parameter names.
    sensitivities : np.ndarray
        Sensitivity measure for each parameter (higher = more influence).
    low_values : np.ndarray
        Output when parameter is at its low condition.
    high_values : np.ndarray
        Output when parameter is at its high condition.
    nominal : float
        Output at nominal parameter values.
    top_n : int
        How many top parameters to return.
    """

    output_key: str
    parameter_names: List[str]
    sensitivities: np.ndarray
    low_values: np.ndarray
    high_values: np.ndarray
    nominal: float
    top_n: int = 10

    def ranked(self) -> List[Tuple[str, float]]:
        """Return parameters ranked by sensitivity (highest first)."""
        order = np.argsort(self.sensitivities)[::-1][:self.top_n]
        return [(self.parameter_names[i], float(self.sensitivities[i])) for i in order]

    def to_dict(self) -> Dict[str, Any]:
        order = np.argsort(self.sensitivities)[::-1][:self.top_n]
        return {
            "output_key": self.output_key,
            "nominal": round(self.nominal, 4),
            "top_params": [
                {
                    "name": self.parameter_names[i],
                    "sensitivity": round(float(self.sensitivities[i]), 6),
                    "low": round(float(self.low_values[i]), 4),
                    "high": round(float(self.high_values[i]), 4),
                }
                for i in order
            ],
        }


@dataclass
class MorrisResult:
    """Morris screening results: elementary effects for each parameter.

    Attributes
    ----------
    parameter_names : list of str
    mu_star : np.ndarray
        Mean of |elementary effects| — overall importance measure.
    sigma : np.ndarray
        Standard deviation of elementary effects — non-linearity /
        interaction indicator.
    mu : np.ndarray
        Signed mean of elementary effects — direction of influence.
    n_trajectories : int
    n_params : int
    elapsed_seconds : float
    """

    parameter_names: List[str]
    mu_star: np.ndarray
    sigma: np.ndarray
    mu: np.ndarray
    n_trajectories: int
    n_params: int
    elapsed_seconds: float = 0.0

    def ranked(self, n: int = 10) -> List[Tuple[str, float]]:
        """Return top-n parameters by mu_star (most influential)."""
        order = np.argsort(self.mu_star)[::-1][:n]
        return [(self.parameter_names[i], float(self.mu_star[i])) for i in order]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter_names": self.parameter_names,
            "mu_star": self.mu_star.tolist(),
            "sigma": self.sigma.tolist(),
            "mu": self.mu.tolist(),
            "n_trajectories": self.n_trajectories,
            "n_params": self.n_params,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


# ---------------------------------------------------------------------------
# Sobol analysis (Saltelli 2010 / Jansen 1999 estimator)
# ---------------------------------------------------------------------------

def sobol_analysis(
    engine: MonteCarloEngine,
    design_point: Optional[Dict[str, Any]] = None,
    specs: Optional[Sequence] = None,
    n_samples: int = 10000,
    param_subset: Optional[List[str]] = None,
    output_subset: Optional[List[str]] = None,
) -> SobolResult:
    """Compute Sobol first-order and total-order sensitivity indices.

    Uses the Saltelli (2010) sampling strategy with the Jansen (1999)
    estimator for total-order indices.

    The total sample budget is *n_samples*; the base sample size is
    ``N_base = n_samples // (D + 2)`` where D is the number of
    parameters.  Total evaluations = ``(D + 2) * N_base``.

    Parameters
    ----------
    engine : MonteCarloEngine
        Engine instance (provides registry and seed).
    design_point : dict, optional
        Operating conditions held fixed.  Defaults to the engine's
        default design point.
    specs : sequence, optional
        Unused directly — indices are computed on raw outputs.
    n_samples : int
        Total sample budget (default 10 000).
    param_subset : list of str, optional
        Parameter names to analyse.  Defaults to all in registry.
    output_subset : list of str, optional
        Output keys to analyse.  Defaults to all outputs produced by
        the pipeline.

    Returns
    -------
    SobolResult
    """
    t0 = time.perf_counter()
    dp = design_point or dict(DEFAULT_DESIGN_POINT)
    registry = engine.registry

    # Select parameters
    if param_subset:
        param_names = [p for p in param_subset if p in registry]
    else:
        param_names = sorted(registry.keys())
    D = len(param_names)
    if D == 0:
        raise ValueError("No parameters to analyse")

    # Base sample size: (D+2) * N_base <= n_samples
    N_base = max(10, n_samples // (D + 2))

    # Generate two independent sample matrices A and B
    rng_a = np.random.default_rng(engine.seed)
    rng_b = np.random.default_rng(engine.seed + 1000)

    A_unit = rng_a.random((N_base, D))
    B_unit = rng_b.random((N_base, D))

    # Transform to parameter space
    A = np.zeros((N_base, D))
    B = np.zeros((N_base, D))
    for j, name in enumerate(param_names):
        p = registry[name]
        for i in range(N_base):
            A[i, j] = _transform(float(A_unit[i, j]), p)
            B[i, j] = _transform(float(B_unit[i, j]), p)

    # ── Evaluate the model ──────────────────────────────────────────

    # Fill in default (mean) values for non-analyzed parameters so the
    # pipeline always receives a complete sample dict.
    default_values = {name: float(registry[name].mean) for name in registry}

    def _eval_row(row: np.ndarray) -> Dict[str, float]:
        """Evaluate the pipeline for one parameter vector."""
        sample = dict(default_values)
        for j in range(D):
            sample[param_names[j]] = float(row[j])
        return _run_single_sample(sample, dp)

    # Evaluate at A and B
    f_a_list = [_eval_row(A[i]) for i in range(N_base)]
    f_b_list = [_eval_row(B[i]) for i in range(N_base)]

    # Determine output keys
    all_output_keys = sorted(f_a_list[0].keys()) if f_a_list else []
    if output_subset:
        output_keys = [k for k in output_subset if k in all_output_keys]
    else:
        output_keys = all_output_keys
    n_outputs = len(output_keys)

    # Collect f(A) and f(B) as arrays
    f_a_arr = np.full((N_base, n_outputs), np.nan)
    f_b_arr = np.full((N_base, n_outputs), np.nan)
    for i in range(N_base):
        for k, key in enumerate(output_keys):
            f_a_arr[i, k] = f_a_list[i].get(key, np.nan)
            f_b_arr[i, k] = f_b_list[i].get(key, np.nan)

    # Evaluate A_B^(i) for each parameter — column i from B, rest from A
    f_ab = np.full((D, N_base, n_outputs), np.nan)
    for j in range(D):
        ab = A.copy()
        ab[:, j] = B[:, j]
        for i in range(N_base):
            result = _eval_row(ab[i])
            for k, key in enumerate(output_keys):
                f_ab[j, i, k] = result.get(key, np.nan)

    # ── Compute Sobol indices (Jansen 1999) ─────────────────────────

    first_order = np.zeros((n_outputs, D))
    total_order = np.zeros((n_outputs, D))
    variance = np.zeros(n_outputs)

    for k in range(n_outputs):
        fa = f_a_arr[:, k]
        fb = f_b_arr[:, k]

        # Filter NaN samples
        valid = ~(np.isnan(fa) | np.isnan(fb))
        for j in range(D):
            valid = valid & ~np.isnan(f_ab[j, :, k])

        if valid.sum() < 10:
            continue

        fa_v = fa[valid]
        fb_v = fb[valid]
        n_valid = len(fa_v)

        # Total variance (from both matrices for stability)
        all_vals = np.concatenate([fa_v, fb_v])
        var_y = float(np.var(all_vals))
        variance[k] = var_y

        if var_y < 1e-30:
            continue

        for j in range(D):
            fab_v = f_ab[j, valid, k]

            # First-order (Saltelli 2010):
            #   S_i = (1/N) * sum( f(B) * (f(A_B^i) - f(A)) ) / V(Y)
            numerator_s = float(np.sum(fb_v * (fab_v - fa_v)))
            first_order[k, j] = numerator_s / (n_valid * var_y)

            # Total-order (Jansen 1999):
            #   S_Ti = 1 - (1/(2N)) * sum( (f(A) - f(A_B^i))^2 ) / V(Y)
            numerator_st = float(np.sum((fa_v - fab_v) ** 2))
            total_order[k, j] = 1.0 - numerator_st / (2.0 * n_valid * var_y)

    # Clip to physically meaningful [0, 1]
    first_order = np.clip(first_order, 0.0, 1.0)
    total_order = np.clip(total_order, 0.0, 1.0)

    elapsed = time.perf_counter() - t0
    total_evals = (D + 2) * N_base

    return SobolResult(
        parameter_names=param_names,
        output_keys=output_keys,
        first_order=first_order,
        total_order=total_order,
        variance=variance,
        n_samples=total_evals,
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Tornado chart (from Monte Carlo results)
# ---------------------------------------------------------------------------

def tornado_chart(
    mc_result: MonteCarloResult,
    output_key: str,
    top_n: int = 10,
    n_bins: int = 10,
) -> TornadoResult:
    """Compute tornado chart data from Monte Carlo results.

    For each parameter, samples are sorted by that parameter's value
    and split into quantile bins.  The conditional output range (high
    bin mean – low bin mean) measures the parameter's influence on the
    output.

    Parameters
    ----------
    mc_result : MonteCarloResult
        MC results with ``sample_matrix`` and ``parameter_names``.
    output_key : str
        Which output to analyse.
    top_n : int
        Number of top parameters to include in the tornado.
    n_bins : int
        Number of quantile bins (default 10 → 10th/90th percentile).

    Returns
    -------
    TornadoResult

    Raises
    ------
    KeyError
        If *output_key* is not in the MC output distributions.
    ValueError
        If ``sample_matrix`` is not available (MC was run without it).
    """
    if output_key not in mc_result.output_distributions:
        raise KeyError(f"Output '{output_key}' not in MC results")
    if mc_result.sample_matrix is None or len(mc_result.parameter_names) == 0:
        raise ValueError(
            "sample_matrix not available — re-run MonteCarloEngine.run() "
            "to store raw samples"
        )

    out_vals = mc_result.output_distributions[output_key]
    param_names = mc_result.parameter_names
    sample_matrix = mc_result.sample_matrix
    n_params = len(param_names)

    # Drop samples with NaN output
    valid_mask = ~np.isnan(out_vals)
    out_clean = out_vals[valid_mask]
    mat_clean = sample_matrix[valid_mask]
    n_clean = len(out_clean)

    if n_clean < 2 * n_bins:
        raise ValueError(
            f"Need >= {2 * n_bins} valid samples for {n_bins}-bin tornado, "
            f"got {n_clean}"
        )

    # Nominal = mean output
    nominal = float(np.mean(out_clean))

    # For each parameter, bin by quantiles and compute conditional output
    sensitivities = np.zeros(n_params)
    low_values = np.full(n_params, nominal)
    high_values = np.full(n_params, nominal)

    low_q = 1.0 / n_bins       # e.g. 0.1 for 10 bins
    high_q = 1.0 - low_q       # e.g. 0.9

    for j in range(n_params):
        col = mat_clean[:, j]
        if np.std(col) < 1e-30:
            continue

        # Split into low / high halves at median
        low_mask = col <= np.percentile(col, low_q * 100)
        high_mask = col >= np.percentile(col, high_q * 100)

        if low_mask.sum() < 2 or high_mask.sum() < 2:
            continue

        mean_low = float(np.mean(out_clean[low_mask]))
        mean_high = float(np.mean(out_clean[high_mask]))

        low_values[j] = mean_low
        high_values[j] = mean_high
        sensitivities[j] = abs(mean_high - mean_low)

    # Normalise to [0, 1]
    max_sens = np.max(sensitivities) if np.max(sensitivities) > 0 else 1.0
    sensitivities = sensitivities / max_sens

    return TornadoResult(
        output_key=output_key,
        parameter_names=param_names,
        sensitivities=sensitivities,
        low_values=low_values,
        high_values=high_values,
        nominal=nominal,
        top_n=min(top_n, n_params),
    )


# ---------------------------------------------------------------------------
# Morris screening (elementary effects)
# ---------------------------------------------------------------------------

def morris_screening(
    design_point: Optional[Dict[str, Any]] = None,
    registry: Optional[Dict[str, Parameter]] = None,
    n_trajectories: int = 50,
    n_levels: int = 4,
    seed: int = 42,
    param_subset: Optional[List[str]] = None,
    output_keys: Optional[List[str]] = None,
) -> MorrisResult:
    """Morris screening: elementary effects for high-dimensional spaces.

    Each trajectory evaluates the model at D+1 points, moving one
    parameter at a time by a fixed step Δ.  The elementary effect for
    parameter *i* is::

        EE_i = (Y(X + Δ·e_i) - Y(X)) / Δ

    where Δ = 1/(n_levels - 1) in unit [0, 1] space.

    Parameters
    ----------
    design_point : dict, optional
        Operating conditions held fixed.
    registry : dict, optional
        Parameter registry.  Defaults to REGISTRY.
    n_trajectories : int
        Number of random trajectories (default 50).
    n_levels : int
        Grid levels (default 4 → Δ = 1/3).
    seed : int
        Random seed.
    param_subset : list of str, optional
        Subset of parameter names to screen.  Defaults to all.
    output_keys : list of str, optional
        Which outputs to track.  Defaults to all produced by the
        pipeline.

    Returns
    -------
    MorrisResult
    """
    t0 = time.perf_counter()
    dp = design_point or dict(DEFAULT_DESIGN_POINT)
    reg = registry or REGISTRY

    if param_subset:
        param_names = [p for p in param_subset if p in reg]
    else:
        param_names = sorted(reg.keys())

    D = len(param_names)
    if D == 0:
        raise ValueError("No parameters to screen")

    rng = np.random.default_rng(seed)
    delta = 1.0 / (n_levels - 1)

    # Probe output keys from a dummy evaluation
    sample0 = {name: float(reg[name].mean) for name in param_names}
    out0 = _run_single_sample(sample0, dp)
    all_output_keys = sorted(out0.keys())
    if output_keys:
        tracked_outputs = [k for k in output_keys if k in all_output_keys]
    else:
        tracked_outputs = all_output_keys
    n_outputs = len(tracked_outputs)

    # Elementary effects storage: ee[trajectory, param, output]
    ee = np.full((n_trajectories, D, n_outputs), np.nan)

    for t in range(n_trajectories):
        # Random starting point on the grid
        x_unit = rng.choice(np.linspace(0, 1, n_levels), size=D)
        # Ensure we have room to step in either direction
        x_unit = np.clip(x_unit, delta, 1.0 - delta)

        # Evaluate at starting point
        sample = {}
        for j, name in enumerate(param_names):
            p = reg[name]
            sample[name] = _transform(float(x_unit[j]), p)
        y_prev = _run_single_sample(sample, dp)

        # Random permutation of parameters
        perm = rng.permutation(D)

        for step, j in enumerate(perm):
            # Perturb parameter j by ±delta
            direction = rng.choice([-1, 1])
            x_unit_new = x_unit.copy()
            x_unit_new[j] = np.clip(x_unit[j] + direction * delta, 0, 1)
            actual_delta = x_unit_new[j] - x_unit[j]

            if abs(actual_delta) < 1e-12:
                # Can't step — skip
                continue

            # Build new sample
            sample_new = {}
            for k_idx, name in enumerate(param_names):
                p = reg[name]
                sample_new[name] = _transform(float(x_unit_new[k_idx]), p)
            y_new = _run_single_sample(sample_new, dp)

            # Record elementary effects
            for k_out, key in enumerate(tracked_outputs):
                v_prev = y_prev.get(key, np.nan)
                v_new = y_new.get(key, np.nan)
                if not (math.isnan(v_prev) or math.isnan(v_new)):
                    ee[t, j, k_out] = (v_new - v_prev) / actual_delta

            # Advance
            x_unit = x_unit_new
            y_prev = y_new

    # Aggregate across trajectories and outputs
    mu = np.zeros(D)
    mu_star = np.zeros(D)
    sigma = np.zeros(D)

    for j in range(D):
        # Flatten across all tracked outputs and trajectories
        all_ee = ee[:, j, :].flatten()
        valid = all_ee[~np.isnan(all_ee)]
        if len(valid) > 0:
            mu[j] = float(np.mean(valid))
            mu_star[j] = float(np.mean(np.abs(valid)))
            sigma[j] = float(np.std(valid))

    elapsed = time.perf_counter() - t0

    return MorrisResult(
        parameter_names=param_names,
        mu_star=mu_star,
        sigma=sigma,
        mu=mu,
        n_trajectories=n_trajectories,
        n_params=D,
        elapsed_seconds=elapsed,
    )
