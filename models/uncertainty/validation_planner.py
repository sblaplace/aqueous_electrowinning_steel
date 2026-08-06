"""
Validation experiment planner — DOE to maximally reduce uncertainty.

Designs the minimum set of experiments that reduces total output uncertainty
below the threshold needed for design qualification.  Experiments are ranked
by expected information gain per dollar, and a sequential planner updates
the recommendation after each experiment completes.

Experiment catalog (8 types):
  1. foil_test           — constrains D0, Q (diffusivity)
  2. tensile_test        — constrains sigma0, k_HP, k_SS (Hall-Petch + SS)
  3. ebsd_grain_size     — constrains k_HP, grain model params
  4. vickers_hardness    — constrains tabor_factor, Maynier coefficients
  5. o2_probe_calibration — constrains K_B, K_CH4 (carbon potential)
  6. icp_oes_composition — constrains k_SS_NI, anomalous Ni parameters
  7. tempering_curve     — constrains k_softening, C_HJ
  8. polarization_temperature_series — constrains the electrochemical
     kinetics block (i0, Tafel slopes, Arrhenius Ea); added 2026-08 when the
     Arrhenius exchange-current activation energies entered the registry —
     before that, no catalog experiment touched any kinetics parameter and
     the (J/mol-scale) Ea variances swamped the planner's possible
     reduction.

References:
  * Montgomery, D.C. (2020). Design and Analysis of Experiments, 10th ed.
  * Fedorov, V.V. (1972). Theory of Optimal Experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any



# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    """A single validation experiment type in the catalog."""

    name: str
    cost_usd: float
    duration_hours: float
    constrained_params: List[str]
    description: str
    equipment: str = ""
    priority: int = 0  # higher = more critical (set by planner)


@dataclass
class ValidationPlan:
    """An ordered list of experiments to run, with expected gains."""

    experiments: List[Experiment]
    expected_variance_reduction: float  # total fractional reduction
    total_cost_usd: float
    total_duration_hours: float
    gain_per_dollar: List[float]  # per-experiment marginal gain/$


@dataclass
class UncertaintyTrajectory:
    """Cumulative uncertainty reduction as experiments are completed."""

    experiment_names: List[str]
    cumulative_cost_usd: List[float]
    remaining_variance_frac: List[float]  # fraction of original variance
    marginal_reduction: List[float]  # per-step reduction
    params_constrained: List[List[str]]  # which params each step constrains


# ---------------------------------------------------------------------------
# Experiment catalog
# ---------------------------------------------------------------------------

def experiment_catalog() -> Dict[str, Experiment]:
    """Return the full catalog of 7+ validation experiment types.

    Each experiment has a cost, duration, and list of registry parameters
    it helps constrain.
    """
    catalog = {
        "foil_test": Experiment(
            name="foil_test",
            cost_usd=200.0,
            duration_hours=8.0,
            constrained_params=["D0_ferrite_m2_s", "Q_ferrite_kJ_mol",
                                "D0_austenite_m2_s", "Q_austenite_kJ_mol"],
            description=(
                "Carbon foil weight-gain test at 900°C — measures effective "
                "diffusivity and activation energy via mass uptake kinetics."
            ),
            equipment="tube furnace, analytical balance, Ar/CH4 atmosphere",
        ),
        "tensile_test": Experiment(
            name="tensile_test",
            cost_usd=350.0,
            duration_hours=12.0,
            constrained_params=["sigma0_fe_MPa", "k_hp_MPa_sqrt_m",
                                "k_ss_ni_MPa_per_wt"],
            description=(
                "ASTM E8 tensile test on electrodeposited coupons — directly "
                "measures yield strength and UTS for Hall-Petch calibration."
            ),
            equipment="universal testing machine, extensometer, specimen cutter",
        ),
        "ebsd_grain_size": Experiment(
            name="ebsd_grain_size",
            cost_usd=500.0,
            duration_hours=16.0,
            constrained_params=["k_hp_MPa_sqrt_m", "grain_d0_dc_ref_um",
                                "grain_j_exponent", "grain_pe_factor_base"],
            description=(
                "EBSD mapping of grain-size distribution — constrains the "
                "Hall-Petch slope and grain-size estimation model."
            ),
            equipment="SEM with EBSD detector, sample polishing, ion milling",
        ),
        "vickers_hardness": Experiment(
            name="vickers_hardness",
            cost_usd=150.0,
            duration_hours=4.0,
            constrained_params=["tabor_factor", "HV_base_Maynier",
                                "HV_per_C_wt_Maynier"],
            description=(
                "Vickers micro-hardness mapping (HV0.3 to HV1) across "
                "cross-section — calibrates Tabor and Maynier correlations."
            ),
            equipment="Vickers micro-hardness tester, optical microscope",
        ),
        "o2_probe_calibration": Experiment(
            name="o2_probe_calibration",
            cost_usd=800.0,
            duration_hours=24.0,
            constrained_params=["dG_boudouard_intercept", "dG_boudouard_slope",
                                "dG_ch4_intercept", "dG_ch4_slope"],
            description=(
                "Oxygen probe calibration against known CO/CO2/CH4 mixtures "
                "at carburizing temperatures — constrains carbon potential "
                "thermodynamic correlations."
            ),
            equipment="zirconia O2 probe, gas mixing manifold, furnace",
        ),
        "icp_oes_composition": Experiment(
            name="icp_oes_composition",
            cost_usd=250.0,
            duration_hours=6.0,
            constrained_params=["k_ss_ni_MPa_per_wt", "ss_ni_exp",
                                "ss_ni_sat_wt"],
            description=(
                "ICP-OES elemental analysis of Ni content in deposits — "
                "constrains solid-solution strengthening model parameters."
            ),
            equipment="ICP-OES spectrometer, acid digestion setup",
        ),
        "tempering_curve": Experiment(
            name="tempering_curve",
            cost_usd=300.0,
            duration_hours=20.0,
            constrained_params=["k_softening", "C_HJ", "softening_floor",
                                "KM_alpha_K_inv"],
            description=(
                "Tempering kinetics: HV vs temperature and time series — "
                "calibrates Hollomon-Jaffe and softening parameters."
            ),
            equipment="muffle furnace, Vickers tester, temperature controller",
        ),
        "polarization_temperature_series": Experiment(
            name="polarization_temperature_series",
            cost_usd=800.0,
            duration_hours=24.0,
            constrained_params=["fe_i0", "her_i0", "fe_tafel_V", "her_tafel_V",
                                "fe_i0_Ea_J_mol", "her_i0_Ea_J_mol"],
            description=(
                "Potentiostatic polarization curves (LSV/RDE) on Fe at "
                "several temperatures (~25-70 °C) — the Arrhenius regression "
                "of the exchange currents is exactly the experiment that "
                "pins the apparent activation energies and Tafel slopes of "
                "the Fe-deposition and HER branches."
            ),
            equipment="potentiostat, rotating-disk electrode, thermostated cell jacket",
        ),
    }
    return catalog


# ---------------------------------------------------------------------------
# Information-gain model
# ---------------------------------------------------------------------------

def _param_variance(param: Dict[str, Any]) -> float:
    """Approximate prior variance for a registry parameter.

    Uses std^2 directly if available; otherwise derives from bounds for
    a uniform approximation.
    """
    std = param.get("std", 0.0)
    if std > 0:
        return std ** 2
    lo, hi = param.get("bounds", (0.0, 1.0))
    return ((hi - lo) ** 2) / 12.0


def _experiment_information_gain(
    experiment: Experiment,
    registry: Dict[str, Any],
    sensitivity: Optional[Dict[str, Dict[str, float]]] = None,
    reduction_factor: float = 0.70,
) -> float:
    """Estimate the expected variance reduction from one experiment.

    For each parameter the experiment constrains, we estimate the fraction
    of total output variance attributable to it (via sensitivity weights
    if available, else equal share), then multiply by a reduction factor
    representing how much the experiment tightens that parameter's posterior.

    Parameters
    ----------
    experiment : Experiment
    registry : dict
        The REGISTRY dict of Parameter objects.
    sensitivity : dict, optional
        Maps output_name -> {param_name: Sobol_index}. If provided, we
        weight each parameter by its contribution to output variance.
    reduction_factor : float
        Fractional variance reduction per constrained parameter (0–1).
        0.70 means the experiment is expected to reduce that parameter's
        contribution to output variance by 70%.

    Returns
    -------
    float
        Expected total output variance reduction (in arbitrary units).
    """
    # Build sensitivity weights: average Sobol index across outputs
    param_weight: Dict[str, float] = {}
    if sensitivity:
        all_params: set = set()
        for out_sobol in sensitivity.values():
            all_params.update(out_sobol.keys())
        for p in all_params:
            vals = [out_sobol.get(p, 0.0) for out_sobol in sensitivity.values()]
            param_weight[p] = sum(vals) / max(len(vals), 1)
    else:
        # Equal weight fallback
        for p in experiment.constrained_params:
            param_weight[p] = 1.0 / max(len(experiment.constrained_params), 1)

    # Compute total registry variance for normalization
    total_variance = sum(
        _param_variance({"std": p.std, "bounds": p.bounds})
        for p in registry.values()
    )
    if total_variance <= 0:
        total_variance = 1.0

    total_gain = 0.0
    for pname in experiment.constrained_params:
        if pname in registry:
            weight = param_weight.get(pname, 0.01)
            # Scale by parameter's fraction of total variance
            p = registry[pname]
            p_var = _param_variance({"std": p.std, "bounds": p.bounds})
            frac = p_var / max(total_variance, 1e-30)
            total_gain += weight * reduction_factor * frac

    return total_gain


# ---------------------------------------------------------------------------
# Core planning functions
# ---------------------------------------------------------------------------

def plan_validation_experiments(
    registry: Dict[str, Any],
    sensitivity: Optional[Dict[str, Dict[str, float]]] = None,
    budget: int = 7,
    cost_budget_usd: float = float("inf"),
) -> ValidationPlan:
    """Select the top-``budget`` experiments ranked by information gain per dollar.

    Parameters
    ----------
    registry : dict
        The central parameter REGISTRY.
    sensitivity : dict, optional
        Sobol sensitivity indices per output.
    budget : int
        Maximum number of experiments.
    cost_budget_usd : float
        Maximum total cost.

    Returns
    -------
    ValidationPlan
    """
    catalog = experiment_catalog()

    scored: List[Tuple[float, Experiment]] = []
    for exp in catalog.values():
        gain = _experiment_information_gain(exp, registry, sensitivity)
        gain_per_dollar = gain / max(exp.cost_usd, 1.0)
        scored.append((gain_per_dollar, exp))

    # Sort descending by gain/$
    scored.sort(key=lambda x: x[0], reverse=True)

    selected: List[Experiment] = []
    gains: List[float] = []
    total_cost = 0.0
    total_gain = 0.0

    for gpd, exp in scored:
        if len(selected) >= budget:
            break
        if total_cost + exp.cost_usd > cost_budget_usd:
            continue
        selected.append(exp)
        gains.append(gpd)
        total_cost += exp.cost_usd
        total_gain += _experiment_information_gain(exp, registry, sensitivity)

    total_duration = sum(e.duration_hours for e in selected)
    total_variance = sum(
        _param_variance({"std": p.std, "bounds": p.bounds})
        for p in registry.values()
    )
    expected_reduction = total_gain  # in normalized units

    return ValidationPlan(
        experiments=selected,
        expected_variance_reduction=expected_reduction,
        total_cost_usd=total_cost,
        total_duration_hours=total_duration,
        gain_per_dollar=gains,
    )


def sequential_planner(
    registry: Dict[str, Any],
    completed: List[str],
    remaining_budget_usd: float = float("inf"),
    sensitivity: Optional[Dict[str, Dict[str, float]]] = None,
) -> ValidationPlan:
    """Sequential planner that updates recommendations after each experiment.

    Removes completed experiments from the catalog, then re-ranks the
    remaining by (updated) information gain per dollar.  The key insight:
    once an experiment constrains its parameters, subsequent experiments
    targeting the same parameters get a reduced gain estimate.

    Parameters
    ----------
    registry : dict
    completed : list of str
        Names of already-completed experiment types.
    remaining_budget_usd : float
    sensitivity : dict, optional

    Returns
    -------
    ValidationPlan
        Remaining experiments, re-ranked.
    """
    catalog = experiment_catalog()
    # Remove completed
    remaining = {k: v for k, v in catalog.items() if k not in completed}

    # Reduce gain for params already constrained by completed experiments
    already_constrained: set = set()
    for cname in completed:
        if cname in catalog:
            already_constrained.update(catalog[cname].constrained_params)

    scored: List[Tuple[float, Experiment]] = []
    for exp in remaining.values():
        # Compute gain with reduced factor for already-constrained params
        overlap = set(exp.constrained_params) & already_constrained
        novel_params = [p for p in exp.constrained_params if p not in already_constrained]

        # Novel params get full reduction, overlapping get diminished
        gain_novel = 0.0
        gain_overlap = 0.0
        param_weight: Dict[str, float] = {}
        if sensitivity:
            all_params: set = set()
            for out_sobol in sensitivity.values():
                all_params.update(out_sobol.keys())
            for p in all_params:
                vals = [out_sobol.get(p, 0.0) for out_sobol in sensitivity.values()]
                param_weight[p] = sum(vals) / max(len(vals), 1)
        else:
            for p in exp.constrained_params:
                param_weight[p] = 1.0 / max(len(exp.constrained_params), 1)

        # Compute total variance for normalization
        total_var = sum(
            _param_variance({"std": p.std, "bounds": p.bounds})
            for p in registry.values()
        )
        if total_var <= 0:
            total_var = 1.0

        for p in novel_params:
            if p in registry:
                p_obj = registry[p]
                p_frac = _param_variance({"std": p_obj.std, "bounds": p_obj.bounds}) / total_var
                gain_novel += param_weight.get(p, 0.01) * 0.70 * p_frac
        for p in overlap:
            if p in registry:
                p_obj = registry[p]
                p_frac = _param_variance({"std": p_obj.std, "bounds": p_obj.bounds}) / total_var
                # Diminished gain: experiment refines an already-constrained param
                gain_overlap += param_weight.get(p, 0.01) * 0.15 * p_frac

        total_gain = gain_novel + gain_overlap
        gpd = total_gain / max(exp.cost_usd, 1.0)
        scored.append((gpd, exp))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected: List[Experiment] = []
    gains: List[float] = []
    total_cost = 0.0

    for gpd, exp in scored:
        if total_cost + exp.cost_usd > remaining_budget_usd:
            continue
        selected.append(exp)
        gains.append(gpd)
        total_cost += exp.cost_usd

    total_duration = sum(e.duration_hours for e in selected)
    total_gain = sum(
        _experiment_information_gain(e, registry, sensitivity)
        for e in selected
    )

    return ValidationPlan(
        experiments=selected,
        expected_variance_reduction=total_gain,
        total_cost_usd=total_cost,
        total_duration_hours=total_duration,
        gain_per_dollar=gains,
    )


def uncertainty_reduction_trajectory(
    registry: Dict[str, Any],
    plan: ValidationPlan,
    sensitivity: Optional[Dict[str, Dict[str, float]]] = None,
) -> UncertaintyTrajectory:
    """Compute the cumulative uncertainty reduction trajectory for a plan.

    Models each experiment as reducing variance of its constrained parameters
    by ~70% (first exposure) and 15% (refinement of already-constrained).

    Returns
    -------
    UncertaintyTrajectory
    """
    total_variance = sum(
        _param_variance({"std": p.std, "bounds": p.bounds})
        for p in registry.values()
    )
    if total_variance <= 0:
        total_variance = 1.0

    # Track per-parameter remaining variance fraction
    param_remaining: Dict[str, float] = {
        name: 1.0 for name in registry
    }

    experiment_names: List[str] = []
    cumulative_cost: List[float] = []
    remaining_frac: List[float] = []
    marginal_red: List[List[str]] = []
    params_constrained_list: List[List[str]] = []

    cumulative = 0.0
    already_done: set = set()

    for exp in plan.experiments:
        cumulative += exp.cost_usd

        # Determine which params are novel vs already constrained
        novel = [p for p in exp.constrained_params if p not in already_done]
        refined = [p for p in exp.constrained_params if p in already_done]

        # Reduce variance
        for p in novel:
            if p in param_remaining:
                param_remaining[p] *= 0.30  # 70% reduction
        for p in refined:
            if p in param_remaining:
                param_remaining[p] *= 0.85  # 15% refinement

        already_done.update(exp.constrained_params)

        # Compute total remaining variance
        remaining_var = sum(
            param_remaining.get(name, 1.0)
            * _param_variance({"std": p.std, "bounds": p.bounds})
            for name, p in registry.items()
        )
        frac = remaining_var / total_variance

        experiment_names.append(exp.name)
        cumulative_cost.append(cumulative)
        remaining_frac.append(frac)
        params_constrained_list.append(exp.constrained_params)
        marginal_red.append(novel + refined)

    return UncertaintyTrajectory(
        experiment_names=experiment_names,
        cumulative_cost_usd=cumulative_cost,
        remaining_variance_frac=remaining_frac,
        marginal_reduction=[0.0] + [
            remaining_frac[i - 1] - remaining_frac[i]
            for i in range(1, len(remaining_frac))
        ],
        params_constrained=params_constrained_list,
    )
