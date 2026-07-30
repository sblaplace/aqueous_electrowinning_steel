"""
Design space explorer — find robust operating regions.

Provides:
* :func:`explore_design_space` — grid search + Monte Carlo for P(all specs met)
* :func:`robust_optimum` — Bayesian optimization targeting P >= 0.95
* :func:`pareto_front_robust` — multi-objective Pareto front

Architecture
------------
1. Grid search over operating-parameter ranges (coarse mesh).
2. Monte Carlo at each grid point for P(all specs met).
3. Scipy/RBF interpolation for continuous confidence surface.
4. Bayesian optimization (scikit-optimize) for the robust optimum.

Each design-point evaluation runs the full Monte Carlo engine with
parameter uncertainty, so the confidence surface reflects *propagated
uncertainty* — not just a deterministic prediction.
"""

from __future__ import annotations

import itertools
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .monte_carlo import (
    DEFAULT_DESIGN_POINT,
    MonteCarloEngine,
    MonteCarloResult,
)
from .parameter_registry import REGISTRY
from .specification import Specification


# --------------------------------------------------------------------------- 
# Result containers
# --------------------------------------------------------------------------- 

@dataclass
class DesignSpaceResult:
    """Result of a grid-based design-space exploration."""

    param_names: List[str]            # names of swept operating parameters
    ranges: Dict[str, Tuple[float, float]]  # {param: (lo, hi)}
    grid_points: np.ndarray           # (N, D) grid coordinates
    confidence_values: np.ndarray     # (N,) P(all specs met) at each point
    interpolation_func: Any = None    # callable(pts) -> confidence (RBF)
    n_grid: int = 0
    mc_samples_per_point: int = 0
    elapsed_seconds: float = 0.0
    spec_set_name: str = ""

    # Per-grid-point MC detail
    mc_results: List[MonteCarloResult] = field(default_factory=list)

    @property
    def max_confidence(self) -> float:
        """Best confidence found on the grid."""
        return float(np.nanmax(self.confidence_values)) if len(self.confidence_values) > 0 else 0.0

    @property
    def best_point(self) -> Dict[str, float]:
        """Operating conditions at the grid point with highest confidence."""
        idx = int(np.nanargmax(self.confidence_values))
        return {name: float(self.grid_points[idx, i])
                for i, name in enumerate(self.param_names)}

    def predict(self, points: np.ndarray) -> np.ndarray:
        """Interpolate confidence at arbitrary points."""
        if self.interpolation_func is None:
            raise RuntimeError("No interpolation function; run explore_design_space first")
        return self.interpolation_func(points)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "param_names": self.param_names,
            "ranges": {k: list(v) for k, v in self.ranges.items()},
            "n_grid": self.n_grid,
            "mc_samples_per_point": self.mc_samples_per_point,
            "max_confidence": round(self.max_confidence, 4),
            "best_point": {k: round(v, 4) for k, v in self.best_point.items()},
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "spec_set_name": self.spec_set_name,
            "grid_confidences": [round(float(c), 4) for c in self.confidence_values],
        }


@dataclass
class RobustOptimum:
    """Result of Bayesian optimization for robust operating conditions."""

    optimum_point: Dict[str, float]
    optimum_confidence: float
    achieved_target: bool
    all_evaluations: np.ndarray       # (N, D) tested points
    all_confidences: np.ndarray       # (N,) confidence at each
    n_calls: int = 0
    target: float = 0.95
    elapsed_seconds: float = 0.0
    spec_set_name: str = ""
    design_margins: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "optimum_point": {k: round(v, 4) for k, v in self.optimum_point.items()},
            "optimum_confidence": round(self.optimum_confidence, 4),
            "achieved_target": self.achieved_target,
            "n_calls": self.n_calls,
            "target": self.target,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "spec_set_name": self.spec_set_name,
            "design_margins": self.design_margins,
        }


@dataclass
class ParetoPoint:
    """A single point on the Pareto front."""
    point: Dict[str, float]
    confidence: float
    cost: float       # relative cost metric
    energy: float     # specific energy kWh/kg


@dataclass
class ParetoFront:
    """Pareto front: confidence vs cost vs energy."""

    points: List[ParetoPoint]
    dominated_count: int = 0
    n_evaluated: int = 0
    elapsed_seconds: float = 0.0
    spec_set_name: str = ""

    @property
    def best_confidence_point(self) -> ParetoPoint:
        return max(self.points, key=lambda p: p.confidence)

    @property
    def best_cost_point(self) -> ParetoPoint:
        return min(self.points, key=lambda p: p.cost)

    @property
    def best_energy_point(self) -> ParetoPoint:
        return min(self.points, key=lambda p: p.energy)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "n_front_points": len(self.points),
            "dominated_count": self.dominated_count,
            "n_evaluated": self.n_evaluated,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "spec_set_name": self.spec_set_name,
            "front": [
                {
                    "point": {k: round(v, 4) for k, v in p.point.items()},
                    "confidence": round(p.confidence, 4),
                    "cost": round(p.cost, 4),
                    "energy": round(p.energy, 4),
                }
                for p in self.points
            ],
        }


# --------------------------------------------------------------------------- 
# Design-point update helper
# --------------------------------------------------------------------------- 

def _make_design_point(
    base: Dict[str, Any],
    param_values: Dict[str, float],
) -> Dict[str, Any]:
    """Create a new design point by overriding base with swept values.

    Maps short sweep names to the design-point keys expected by the
    Monte Carlo engine.
    """
    dp = dict(base)
    for name, val in param_values.items():
        if name == "j_avg":
            dp["j_avg_mA_cm2"] = val
            # Scale peak current from average
            dp["j_peak_mA_cm2"] = val / max(dp.get("duty_cycle", 0.5), 0.01)
        elif name == "duty":
            dp["duty_cycle"] = val
            dp["j_peak_mA_cm2"] = dp.get("j_avg_mA_cm2", 150.0) / max(val, 0.01)
        elif name == "frequency":
            # Store frequency for reference (not used by current MC pipeline
            # but recorded for design-point completeness)
            dp["pulse_frequency_Hz"] = val
        elif name == "T_bath":
            dp["temperature_C"] = val
        elif name == "pH":
            dp["pH"] = val
        elif name == "Ni_conc":
            dp["bath_ni_M"] = val
        elif name == "carburizing_T":
            dp["carburizing_temperature_C"] = val
        elif name == "tempering_T":
            dp["tempering_temperature_C"] = val
        else:
            # Pass through as-is
            dp[name] = val
    return dp


# --------------------------------------------------------------------------- 
# Core: grid-based exploration
# --------------------------------------------------------------------------- 

def explore_design_space(
    ranges: Dict[str, Tuple[float, float]],
    specs: Sequence[Specification],
    registry: Optional[Dict[str, Any]] = None,
    n_grid: int = 10,
    mc_samples: int = 1000,
    seed: int = 42,
    n_jobs: int = -1,
    base_design_point: Optional[Dict[str, Any]] = None,
    spec_set_name: str = "",
) -> DesignSpaceResult:
    """Grid search over operating parameters with MC at each grid point.

    Parameters
    ----------
    ranges : dict
        ``{param_name: (lo, hi)}`` for operating parameters to sweep.
        Supported sweep names: ``j_avg``, ``duty``, ``frequency``,
        ``T_bath``, ``pH``, ``Ni_conc``, ``carburizing_T``, ``tempering_T``.
    specs : sequence of Specification
        Design requirements to check against.
    registry : dict, optional
        Parameter registry for MC sampling.
    n_grid : int
        Grid points per dimension (coarse: 5-10 recommended).
    mc_samples : int
        Monte Carlo samples per grid point.
    seed : int
        Random seed.
    n_jobs : int
        Parallel workers for MC (-1 = all cores).
    base_design_point : dict, optional
        Starting design point.  Defaults to ``DEFAULT_DESIGN_POINT``.
    spec_set_name : str
        Label.

    Returns
    -------
    DesignSpaceResult
    """
    t0 = time.perf_counter()
    base_dp = base_design_point or dict(DEFAULT_DESIGN_POINT)
    reg = registry or REGISTRY

    param_names = sorted(ranges.keys())
    D = len(param_names)

    # Build 1-D grids
    grids_1d = []
    for name in param_names:
        lo, hi = ranges[name]
        grids_1d.append(np.linspace(lo, hi, n_grid))

    # Full factorial mesh
    mesh = np.array(list(itertools.product(*grids_1d)))  # (n_grid**D, D)
    n_points = mesh.shape[0]

    grid_points = mesh
    confidence_values = np.zeros(n_points)
    mc_results_list: List[MonteCarloResult] = []

    engine = MonteCarloEngine(
        n_samples=mc_samples, seed=seed, n_jobs=n_jobs, registry=reg,
    )

    for i in range(n_points):
        param_dict = {name: float(mesh[i, j]) for j, name in enumerate(param_names)}
        dp = _make_design_point(base_dp, param_dict)

        mc_result = engine.run(design_point=dp, specs=specs, spec_set_name=spec_set_name)
        confidence_values[i] = mc_result.overall_confidence
        mc_results_list.append(mc_result)

    # Build interpolation function
    interp_func = _build_interpolator(grid_points, confidence_values, D)

    elapsed = time.perf_counter() - t0

    return DesignSpaceResult(
        param_names=param_names,
        ranges=ranges,
        grid_points=grid_points,
        confidence_values=confidence_values,
        interpolation_func=interp_func,
        n_grid=n_grid,
        mc_samples_per_point=mc_samples,
        elapsed_seconds=elapsed,
        spec_set_name=spec_set_name,
        mc_results=mc_results_list,
    )


def _build_interpolator(
    points: np.ndarray,
    values: np.ndarray,
    D: int,
) -> Any:
    """Build an RBF interpolator for the confidence surface."""
    from scipy.interpolate import RBFInterpolator

    # RBFInterpolator expects (N, D) and (N, M)
    # Use thin-plate spline for smooth surfaces
    try:
        rbf = RBFInterpolator(
            points, values.reshape(-1, 1),
            kernel="thin_plate_spline",
            smoothing=1.0,
        )

        def interp_fn(pts: np.ndarray) -> np.ndarray:
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
            return rbf(pts).ravel()

        return interp_fn
    except Exception:
        # Fallback: nearest-neighbour
        from scipy.interpolate import NearestNDInterpolator
        nn = NearestNDInterpolator(points, values)

        def interp_fn(pts: np.ndarray) -> np.ndarray:
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
            return nn(pts)

        return interp_fn


# --------------------------------------------------------------------------- 
# Bayesian optimization for robust optimum
# --------------------------------------------------------------------------- 

def robust_optimum(
    ranges: Dict[str, Tuple[float, float]],
    specs: Sequence[Specification],
    registry: Optional[Dict[str, Any]] = None,
    n_calls: int = 200,
    target: float = 0.95,
    mc_samples: int = 500,
    seed: int = 42,
    n_jobs: int = -1,
    base_design_point: Optional[Dict[str, Any]] = None,
    spec_set_name: str = "",
) -> RobustOptimum:
    """Bayesian optimization to find operating conditions with P >= target.

    Uses scikit-optimize's Gaussian-process surrogate with expected
    improvement to efficiently locate the confidence maximum.

    Parameters
    ----------
    ranges : dict
        Operating parameter ranges (same keys as :func:`explore_design_space`).
    specs : sequence of Specification
        Design requirements.
    n_calls : int
        Total evaluations (including initial random points).
    target : float
        Target confidence (P(all specs met)).
    mc_samples : int
        MC samples per evaluation (smaller for speed during optimization).

    Returns
    -------
    RobustOptimum
    """
    t0 = time.perf_counter()
    from skopt import gp_minimize
    from skopt.space import Real

    base_dp = base_design_point or dict(DEFAULT_DESIGN_POINT)
    reg = registry or REGISTRY

    param_names = sorted(ranges.keys())
    D = len(param_names)

    # Build search space
    space = [Real(ranges[n][0], ranges[n][1], name=n) for n in param_names]

    # Track all evaluations
    all_points: List[np.ndarray] = []
    all_confidences: List[float] = []

    engine = MonteCarloEngine(
        n_samples=mc_samples, seed=seed, n_jobs=n_jobs, registry=reg,
    )

    def objective(x: List[float]) -> float:
        """Minimize negative confidence (i.e. maximize confidence)."""
        param_dict = {name: float(val) for name, val in zip(param_names, x)}
        dp = _make_design_point(base_dp, param_dict)
        mc_result = engine.run(design_point=dp, specs=specs, spec_set_name=spec_set_name)
        conf = mc_result.overall_confidence

        all_points.append(np.array(x))
        all_confidences.append(conf)

        return -(conf)  # minimize negative confidence

    # Run GP optimization
    n_init = min(20, n_calls // 3)
    result = gp_minimize(
        objective,
        dimensions=space,
        n_calls=n_calls,
        n_initial_points=n_init,
        random_state=seed,
        acq_func="EI",
        noise="gaussian",
    )

    # Extract optimum
    opt_x = result.x
    opt_conf = -result.fun

    optimum_point = {name: float(val) for name, val in zip(param_names, opt_x)}
    achieved = bool(opt_conf >= target)

    # Compute design margins around the optimum
    margins = _compute_design_margins(
        optimum_point, ranges, specs, engine, base_dp, reg, spec_set_name,
    )

    elapsed = time.perf_counter() - t0

    return RobustOptimum(
        optimum_point=optimum_point,
        optimum_confidence=opt_conf,
        achieved_target=achieved,
        all_evaluations=np.array(all_points),
        all_confidences=np.array(all_confidences),
        n_calls=n_calls,
        target=target,
        elapsed_seconds=elapsed,
        spec_set_name=spec_set_name,
        design_margins=margins,
    )


def _compute_design_margins(
    optimum: Dict[str, float],
    ranges: Dict[str, Tuple[float, float]],
    specs: Sequence[Specification],
    engine: MonteCarloEngine,
    base_dp: Dict[str, Any],
    registry: Dict[str, Any],
    spec_set_name: str,
    margin_steps: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Estimate how far each operating parameter can deviate from optimum
    before confidence drops below 90%.

    Returns dict: ``{param: {"optimum": val, "lo_90": val, "hi_90": val,
    "margin_lo_pct": val, "margin_hi_pct": val}}``.
    """
    margins: Dict[str, Dict[str, float]] = {}
    threshold = 0.90

    for name, (lo, hi) in ranges.items():
        opt_val = optimum[name]
        sweep = np.linspace(lo, hi, margin_steps)
        confs = []

        for v in sweep:
            params = dict(optimum)
            params[name] = v
            dp = _make_design_point(base_dp, params)
            mc = engine.run(
                design_point=dp, specs=specs, spec_set_name=spec_set_name,
            )
            confs.append(mc.overall_confidence)

        # Find range where confidence >= threshold
        above = [v for v, c in zip(sweep, confs) if c >= threshold]
        if above:
            lo_90 = min(above)
            hi_90 = max(above)
        else:
            lo_90 = opt_val
            hi_90 = opt_val

        range_span = hi - lo if hi > lo else 1.0
        margins[name] = {
            "optimum": round(opt_val, 4),
            "lo_90": round(lo_90, 4),
            "hi_90": round(hi_90, 4),
            "margin_lo_pct": round((opt_val - lo_90) / range_span * 100, 1),
            "margin_hi_pct": round((hi_90 - opt_val) / range_span * 100, 1),
        }

    return margins


# --------------------------------------------------------------------------- 
# Pareto front: confidence vs cost vs energy
# --------------------------------------------------------------------------- 

def pareto_front_robust(
    objectives: List[str],
    ranges: Dict[str, Tuple[float, float]],
    specs: Sequence[Specification],
    registry: Optional[Dict[str, Any]] = None,
    n_grid: int = 8,
    mc_samples: int = 500,
    seed: int = 42,
    n_jobs: int = -1,
    base_design_point: Optional[Dict[str, Any]] = None,
    spec_set_name: str = "",
) -> ParetoFront:
    """Compute a Pareto front over confidence, cost, and energy.

    Parameters
    ----------
    objectives : list of str
        Objective names (currently supported: ``"confidence"``,
        ``"cost"``, ``"energy"``).
    ranges : dict
        Operating parameter ranges.
    specs : sequence of Specification
        Design requirements.
    n_grid : int
        Grid points per dimension.
    mc_samples : int
        MC samples per grid point.

    Returns
    -------
    ParetoFront
    """
    t0 = time.perf_counter()
    base_dp = base_design_point or dict(DEFAULT_DESIGN_POINT)
    reg = registry or REGISTRY

    param_names = sorted(ranges.keys())

    # Build grid
    grids_1d = [np.linspace(ranges[n][0], ranges[n][1], n_grid) for n in param_names]
    mesh = np.array(list(itertools.product(*grids_1d)))
    n_points = mesh.shape[0]

    engine = MonteCarloEngine(
        n_samples=mc_samples, seed=seed, n_jobs=n_jobs, registry=reg,
    )

    candidates: List[ParetoPoint] = []

    for i in range(n_points):
        param_dict = {name: float(mesh[i, j]) for j, name in enumerate(param_names)}
        dp = _make_design_point(base_dp, param_dict)

        mc_result = engine.run(design_point=dp, specs=specs, spec_set_name=spec_set_name)
        conf = mc_result.overall_confidence

        # Cost proxy: relative cost based on operating conditions
        # Higher current density, longer carburizing, higher temperature = more cost
        cost = _estimate_cost(dp)

        # Energy from MC output
        energy_arr = mc_result.output_distributions.get("specific_energy_kWh_per_kg")
        if energy_arr is not None:
            valid = energy_arr[~np.isnan(energy_arr)]
            energy = float(np.median(valid)) if len(valid) > 0 else 8.0
        else:
            energy = 8.0

        candidates.append(ParetoPoint(
            point=param_dict,
            confidence=conf,
            cost=cost,
            energy=energy,
        ))

    # Extract Pareto front (non-dominated points)
    front, dominated = _extract_pareto(candidates)

    elapsed = time.perf_counter() - t0

    return ParetoFront(
        points=front,
        dominated_count=dominated,
        n_evaluated=n_points,
        elapsed_seconds=elapsed,
        spec_set_name=spec_set_name,
    )


def _estimate_cost(dp: Dict[str, Any]) -> float:
    """Relative cost proxy for operating conditions.

    Normalised to ~1.0 at default design point.  Higher = more expensive.
    """
    # Current density cost (power)
    j = dp.get("j_avg_mA_cm2", 150.0)
    j_cost = (j / 150.0) ** 0.5

    # Carburizing temperature/time cost
    T_carb = dp.get("carburizing_temperature_C", 900.0)
    t_carb = dp.get("carburizing_duration_hr", 4.0)
    carb_cost = (T_carb / 900.0) * (t_carb / 4.0) ** 0.3

    # Tempering cost
    T_temp = dp.get("tempering_temperature_C", 200.0)
    temp_cost = T_temp / 200.0

    # Bath concentration cost (Ni)
    ni = dp.get("bath_ni_M", 0.5)
    ni_cost = ni / 0.5

    # Cell voltage effect
    V = dp.get("cell_voltage_V", 2.5)
    volt_cost = V / 2.5

    return j_cost * carb_cost * temp_cost * ni_cost * volt_cost


def _extract_pareto(
    candidates: List[ParetoPoint],
) -> Tuple[List[ParetoPoint], int]:
    """Extract non-dominated (Pareto-optimal) points.

    Minimises cost and energy, maximises confidence.
    """
    n = len(candidates)
    dominated_mask = [False] * n

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is better-or-equal in all objectives
            # and strictly better in at least one
            ci, cj = candidates[i], candidates[j]
            cj_better_conf = cj.confidence >= ci.confidence
            cj_better_cost = cj.cost <= ci.cost
            cj_better_energy = cj.energy <= ci.energy

            if cj_better_conf and cj_better_cost and cj_better_energy:
                if (cj.confidence > ci.confidence or
                        cj.cost < ci.cost or
                        cj.energy < ci.energy):
                    dominated_mask[i] = True
                    break

    front = [c for c, dom in zip(candidates, dominated_mask) if not dom]
    dominated = sum(dominated_mask)

    # Sort front by confidence descending
    front.sort(key=lambda p: p.confidence, reverse=True)

    return front, dominated


# --------------------------------------------------------------------------- 
# Confidence surface convenience (2-D slice)
# --------------------------------------------------------------------------- 

def confidence_surface_2d(
    param_x: str,
    param_y: str,
    ranges: Dict[str, Tuple[float, float]],
    specs: Sequence[Specification],
    registry: Optional[Dict[str, Any]] = None,
    n_grid: int = 10,
    mc_samples: int = 500,
    seed: int = 42,
    n_jobs: int = -1,
    base_design_point: Optional[Dict[str, Any]] = None,
    spec_set_name: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a 2-D confidence surface for two operating parameters.

    Returns (X, Y, Z) suitable for contour plotting.
    """
    result = explore_design_space(
        ranges={param_x: ranges[param_x], param_y: ranges[param_y]},
        specs=specs,
        registry=registry,
        n_grid=n_grid,
        mc_samples=mc_samples,
        seed=seed,
        n_jobs=n_jobs,
        base_design_point=base_design_point,
        spec_set_name=spec_set_name,
    )

    x_vals = np.linspace(ranges[param_x][0], ranges[param_x][1], n_grid)
    y_vals = np.linspace(ranges[param_y][0], ranges[param_y][1], n_grid)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = result.confidence_values.reshape(n_grid, n_grid)

    return X, Y, Z
