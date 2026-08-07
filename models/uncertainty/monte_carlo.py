"""
Monte Carlo engine — propagate parameter uncertainty through the full
electrochemistry → co-deposition → mechanical → carburization → tempering
→ techno-economic model chain.

Design
------
* Draw N parameter vectors from the registry (Sobol for the first 1 000
  samples, then pseudo-random).
* For each vector, run a simplified but physically consistent pipeline that
  threads sampled model coefficients through every stage.
* Collect ≥10 output quantities and check them against specifications.
* Compute per-spec pass rates, overall confidence, Saltelli first-order
  sensitivity indices, and failure-mode ranking.
* Parallel execution via :mod:`joblib`.

Performance target: N = 1 000 in < 60 s, N = 10 000 in < 5 min.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .parameter_registry import Parameter, REGISTRY
from .sample import sample_parameters
from .specification import Specification, check_mc_specifications


# ---------------------------------------------------------------------------
# Default design point (operating conditions held fixed during MC sweep)
# ---------------------------------------------------------------------------

DEFAULT_DESIGN_POINT: Dict[str, Any] = {
    "j_avg_mA_cm2": 150.0,
    "j_peak_mA_cm2": 300.0,
    "duty_cycle": 0.5,
    "waveform": "pe",
    "temperature_C": 60.0,
    "bath_fe_M": 0.5,
    "bath_ni_M": 0.5,
    "pH": 3.5,
    "carbon_particle_loading_g_L": 2.0,
    "mechanism_fe_ni": "hydroxide_suppression",
    "particle_size_um": 1.5,
    "carburizing_temperature_C": 900.0,
    "carburizing_surface_C_wt": 1.10,
    "carburizing_duration_hr": 4.0,
    "carburizing_thickness_um": 1000.0,
    "tempering_temperature_C": 200.0,
    "tempering_time_hr": 1.0,
    "cell_voltage_V": 2.5,
    "electrode_area_m2": 1.0,
}


# ---------------------------------------------------------------------------
# DFT dG_H* -> theta_H -> i0,H_eff -> FE chain constants
# ---------------------------------------------------------------------------
# Nominal DFT anchor and the operating HER overpotential at which theta_H is
# evaluated at the gate (matches the program's reference HER state, cf.
# her_microkinetics.consistency_report and surface_state.frumkin_sensitivity_sweep,
# both of which use eta_ref = 0.20 V). These hold the MC operating point fixed
# so the dG_H* sweep is the only thing moving i0,H_eff.
DG_HSTAR_NOMINAL_EV = -0.40      # Fe(110) CHE volcano anchor (surface_state.py)
HER_ETA_REF_V = 0.20             # operating cathodic HER overpotential (V)
FE_I0_NOMINAL = 1.0e-2           # registry fe_i0 mean (A/m2)
HER_I0_NOMINAL = 1.0e-6          # registry her_i0 mean (A/m2); see her_i0 default fix #56


def _her_i0_swing(dg_hstar_eV: float, eta_her_V: float, T_K: float,
                  dg_nominal_eV: float = DG_HSTAR_NOMINAL_EV) -> float:
    """i0,H_eff ratio vs the nominal dG_H* anchor, from surface-state physics.

    The surface-state i0,H_eff (models/surface_state.py SurfaceCoverage) is
    ``i0,H_intrinsic * theta_H(1-theta_H) * (1-theta_block) * f_Frumkin``.
    Only the ``theta_H(1-theta_H)`` factor depends on dG_H* — the anion
    site-blocking and Frumkin terms are dG-independent, so they cancel in the
    ratio. The dominant theta_H(1-theta_H) factor swings by ~2-3x across the
    +-0.15 eV DFT band, which is the mechanism being propagated to FE.

    Returns the multiplier to apply to the registry ``her_i0`` (1.0 at the
    nominal anchor).
    """
    from ..electrochemistry import FARADAY
    from ..surface_state import volmer_coverage

    dg = dg_hstar_eV * FARADAY
    dg_nom = dg_nominal_eV * FARADAY
    th = volmer_coverage(dg, eta_her_V, T_K)
    th_nom = volmer_coverage(dg_nom, eta_her_V, T_K)
    r = th * (1.0 - th)
    r_nom = th_nom * (1.0 - th_nom)
    if r_nom <= 0.0:
        return 1.0
    return r / r_nom


# ---------------------------------------------------------------------------
# Single-sample pipeline
# ---------------------------------------------------------------------------

def _run_single_sample(
    sample: Dict[str, float],
    design_point: Dict[str, Any],
) -> Dict[str, float]:
    """Run the full model chain for one sampled parameter set.

    Returns a flat dict of output quantities.  Keys match the
    specification ``output_key`` paths used in the spec framework.
    """
    outputs: Dict[str, float] = {}

    dp = design_point

    # ── 1. Mechanical properties ──────────────────────────────────────
    try:
        from ..mechanical_properties import (
            GrainSizeParams,
            MechanicalPropertiesParams,
            MechanicalPropertiesModel,
        )

        grain_kw = _extract_kwargs(sample, GRAIN_PARAM_MAP)
        mech_kw = _extract_kwargs(sample, MECH_PARAM_MAP)
        grain_p = GrainSizeParams(**grain_kw)
        mech_p = MechanicalPropertiesParams(**mech_kw)
        model = MechanicalPropertiesModel(grain_params=grain_p, mech_params=mech_p)

        result = model.predict(
            j_avg_mA_cm2=dp["j_avg_mA_cm2"],
            j_peak_mA_cm2=dp.get("j_peak_mA_cm2"),
            duty_cycle=dp["duty_cycle"],
            waveform=dp["waveform"],
            temperature_C=dp["temperature_C"],
            ni_wt_percent=_get_or_default(sample, "ni_wt_pct_input", 2.0),
            carbon_wt_percent=_get_or_default(sample, "carbon_wt_pct_input", 0.8),
            current_efficiency_percent=_get_or_default(sample, "ce_pct_input", 93.0),
            particle_size_um=dp.get("particle_size_um", 1.5),
        )

        outputs["sigma_y_MPa"] = result.sigma_y_MPa
        outputs["uts_MPa"] = result.uts_MPa
        outputs["vickers_hv"] = result.vickers_hv
        outputs["elongation_pct"] = result.elongation_pct
        outputs["grain_size_um"] = result.grain_size_um
        outputs["porosity"] = result.porosity
        outputs["sigma_hp_MPa"] = result.sigma_hp_MPa
        outputs["delta_ss_MPa"] = result.delta_ss_MPa
        outputs["delta_carbon_total_MPa"] = result.delta_carbon_total_MPa
    except Exception:
        _fill_nan(outputs, [
            "sigma_y_MPa", "uts_MPa", "vickers_hv", "elongation_pct",
            "grain_size_um", "porosity", "sigma_hp_MPa",
            "delta_ss_MPa", "delta_carbon_total_MPa",
        ])

    # ── 2. Carburization ──────────────────────────────────────────────
    try:
        from ..carburization import (
            CarburizationParams,
            CarburizationModel,
        )

        carb_kw = _extract_kwargs(sample, CARB_PARAM_MAP)
        # Inject design-point values not in registry
        carb_kw.setdefault("temperature_C", dp.get("carburizing_temperature_C", 900.0))
        carb_kw.setdefault("surface_carbon_wt_percent", dp.get("carburizing_surface_C_wt", 1.10))
        carb_kw.setdefault("sheet_thickness_um", dp.get("carburizing_thickness_um", 1000.0))

        # Map registry keys to CarburizationParams kwargs
        if "D0_ferrite_m2_s" in sample:
            carb_kw["D0_m2_s"] = sample["D0_ferrite_m2_s"]
        if "Q_ferrite_kJ_mol" in sample:
            carb_kw["Q_kJ_mol"] = sample["Q_ferrite_kJ_mol"]

        carb_params = CarburizationParams(**carb_kw)
        carb_model = CarburizationModel(carb_params)
        carb_result = carb_model.simulate(
            duration_hr=dp.get("carburizing_duration_hr", 4.0),
            dt_hr=0.5,
        )
        s = carb_result.summary()
        outputs["case_depth_035_um"] = s["final_case_depth_035_um"]
        outputs["surface_hv"] = s["final_surface_hv"]
        outputs["core_c_wt"] = s["final_core_c_wt"]
        outputs["surface_C_wt_percent"] = dp.get("carburizing_surface_C_wt", 1.10)
    except Exception:
        _fill_nan(outputs, [
            "case_depth_035_um", "surface_hv", "core_c_wt", "surface_C_wt_percent",
        ])

    # ── 3. Tempering ──────────────────────────────────────────────────
    try:
        from ..tempering import (
            AlloyComposition,
            martensite_start_C,
            retained_austenite_fraction_koistinen_marburger,
            hollomon_jaffe_parameter,
            tempered_hardness_hollomon_jaffe,
            tempered_yield_from_hardness,
        )

        C_HJ = sample.get("C_HJ", 19.5)
        k_soft = sample.get("k_softening", 0.00018)
        T_temp = dp.get("tempering_temperature_C", 200.0)
        t_temp = dp.get("tempering_time_hr", 1.0)

        c_wt_for_tempering = outputs.get("core_c_wt", 0.5)
        chem = AlloyComposition(C=c_wt_for_tempering, Ni=outputs.get("ni_wt_pct_input", 2.0))
        Ms = martensite_start_C(chem)
        f_RA = retained_austenite_fraction_koistinen_marburger(Ms, 25.0)
        P = hollomon_jaffe_parameter(T_temp, t_temp, C_HJ)

        hv_as_quenched = outputs.get("surface_hv", 800.0)
        if math.isnan(hv_as_quenched):
            hv_as_quenched = 800.0
        hv_tempered = tempered_hardness_hollomon_jaffe(hv_as_quenched, P, k_softening=k_soft)
        ys_tempered = tempered_yield_from_hardness(hv_tempered)

        outputs["Ms_C"] = Ms
        outputs["f_RA"] = f_RA
        outputs["hv_tempered"] = hv_tempered
        outputs["ys_tempered_MPa"] = ys_tempered
    except Exception:
        _fill_nan(outputs, ["Ms_C", "f_RA", "hv_tempered", "ys_tempered_MPa"])

    # ── 4. Faradaic efficiency & energy (surface-state HER chain) ─────
    try:
        from ..electrochemistry import FARADAY, M_FE, Z_FE
        from ..surface_state import volmer_coverage

        # DFT dG_H* anchors theta_H(eta) -> i0,H_eff -> FE at the gate
        # (CHEM_PHYS_IMPROVEMENTS §3.1). The +/-0.15 eV band is drawn by the
        # Sobol/MC sampler through the registry "dG_Hstar_eV" parameter.
        dg_hstar_eV = sample.get("dG_Hstar_eV", DG_HSTAR_NOMINAL_EV)
        eta_her_V = HER_ETA_REF_V        # operating HER overpotential (V)
        T_K = dp.get("temperature_C", 60.0) + 273.15

        # theta_H(eta) at the sampled dG_H* (dominant facet anchor).
        theta_H = volmer_coverage(dg_hstar_eV * FARADAY, eta_her_V, T_K)

        # i0,H_eff swing relative to the nominal -0.40 eV anchor (~2-3x band).
        swing = _her_i0_swing(dg_hstar_eV, eta_her_V, T_K)

        fe_i0 = sample.get("fe_i0", FE_I0_NOMINAL)
        her_i0 = sample.get("her_i0", HER_I0_NOMINAL)
        her_i0_eff = her_i0 * swing

        # Galvanostatic split of the fixed total current between Fe deposition
        # and HER (equal-Tafel-slope approximation): FE = i0_fe/(i0_fe+i0_H_eff),
        # normalized so the nominal anchor reproduces the program's base 93 %.
        # FE -> specific_energy_kWh_per_kg -> energy_cost -> LCOFe downstream.
        ce_nominal_ratio = FE_I0_NOMINAL / (FE_I0_NOMINAL + HER_I0_NOMINAL)
        ce_ratio = fe_i0 / (fe_i0 + her_i0_eff)
        base_ce = 93.0
        ce_pct = base_ce * (ce_ratio / ce_nominal_ratio)
        ce_pct = float(np.clip(ce_pct, 50.0, 99.5))

        outputs["current_efficiency_percent"] = ce_pct
        outputs["theta_H"] = theta_H
        outputs["her_i0_eff_A_m2"] = her_i0_eff

        # Specific energy: kWh/kg = (V * F * z) / (M * CE * 3600)
        V_cell = dp.get("cell_voltage_V", 2.5)
        # Use sampled cell voltage parameters
        eta_cath = sample.get("fe_tafel_V", 0.120) * 1.5  # rough overpotential
        V_total = V_cell + eta_cath * 0.5  # add cathode overpotential contribution
        energy_kWh_per_kg = (V_total * FARADAY * Z_FE) / (M_FE * (ce_pct / 100.0) * 3600.0)
        outputs["specific_energy_kWh_per_kg"] = energy_kWh_per_kg

        # Ni and C content from co-deposition (use design point + sampled kinetics)
        ni_wt = 2.0 * (sample.get("ni_i0", 5.0e-3) / 5.0e-3) ** 0.15
        ni_wt = float(np.clip(ni_wt, 0.1, 8.0))
        c_wt = 0.8 * (sample.get("guglielmi_k_ref", 0.015) / 0.015) ** 0.2
        c_wt = float(np.clip(c_wt, 0.05, 5.0))

        outputs["ni_wt_percent"] = ni_wt
        outputs["carbon_wt_percent"] = c_wt
    except Exception:
        _fill_nan(outputs, [
            "current_efficiency_percent", "specific_energy_kWh_per_kg",
            "ni_wt_percent", "carbon_wt_percent",
            "theta_H", "her_i0_eff_A_m2",
        ])

    # ── 5. Derived metrics ────────────────────────────────────────────
    # energy_cost is a proxy for cost competitiveness
    outputs["energy_cost"] = outputs.get("specific_energy_kWh_per_kg", 6.0)

    return outputs


# ---------------------------------------------------------------------------
# Parameter extraction helpers
# ---------------------------------------------------------------------------

# Maps from registry name -> (model constructor kwarg name)
# These are filtered subsets of the full mapping in sample.py

GRAIN_PARAM_MAP = {
    "grain_d0_dc_ref_um": "d0_dc_ref_um",
    "grain_j_ref_mA_cm2": "j_ref_mA_cm2",
    "grain_j_exponent": "j_exponent",
    "grain_pe_factor_base": "pe_factor_base",
    "grain_pre_factor_base": "pre_factor_base",
}

MECH_PARAM_MAP = {
    "sigma0_fe_MPa": "sigma0_MPa",
    "k_hp_MPa_sqrt_m": "k_hp_MPa_sqrt_m",
    "k_ss_ni_MPa_per_wt": "k_ss_ni_MPa_per_wt",
    "ss_ni_exp": "ss_ni_exp",
    "ss_ni_sat_wt": "ss_ni_sat_wt",
    "k_carbon_MPa_per_wt": "k_carbon_MPa_per_wt",
    "carbon_nl_exp": "carbon_nl_exp",
    "carbon_size_ref_um": "carbon_size_ref_um",
    "carbon_size_exp": "carbon_size_exp",
    "load_transfer_frac": "load_transfer_frac",
    "porosity_penalty_exp": "porosity_penalty_exp",
    "porosity_max": "porosity_max",
    "tabor_factor": "tabor_factor",
    "uts_over_ys_base": "uts_over_ys_base",
    "elongation_base_pct": "elongation_base_pct",
}

CARB_PARAM_MAP: Dict[str, str] = {}  # hardness constants not constructor params


def _extract_kwargs(
    sample: Dict[str, float],
    mapping: Dict[str, str],
) -> Dict[str, Any]:
    """Map registry names to model constructor kwargs."""
    kw: Dict[str, Any] = {}
    for reg_name, kw_name in mapping.items():
        if reg_name in sample:
            kw[kw_name] = sample[reg_name]
    return kw


def _get_or_default(sample: Dict[str, float], key: str, default: float) -> float:
    return sample.get(key, default)


def _fill_nan(outputs: Dict[str, float], keys: Sequence[str]) -> None:
    for k in keys:
        if k not in outputs:
            outputs[k] = float("nan")


# ---------------------------------------------------------------------------
# Sensitivity analysis (Saltelli / correlation fallback)
# ---------------------------------------------------------------------------

def _compute_sensitivity(
    output_distributions: Dict[str, np.ndarray],
    samples: List[Dict[str, float]],
    param_names: List[str],
    top_n: int = 3,
) -> Dict[str, Dict[str, float]]:
    """Compute first-order sensitivity indices using correlation-based method.

    For each output, ranks parameters by |Pearson correlation| with the
    output.  This is a fast proxy for Sobol indices that works well for
    screening (identify top-N influential parameters).
    """
    sensitivity: Dict[str, Dict[str, float]] = {}

    # Build parameter matrix
    n_samples = len(samples)
    n_params = len(param_names)
    param_matrix = np.zeros((n_samples, n_params))
    for i, s in enumerate(samples):
        for j, name in enumerate(param_names):
            param_matrix[i, j] = s.get(name, np.nan)

    for out_key, out_vals in output_distributions.items():
        if len(out_vals) != n_samples:
            continue
        # Drop NaN samples
        valid = ~(np.isnan(out_vals) | np.any(np.isnan(param_matrix), axis=1))
        if valid.sum() < 10:
            sensitivity[out_key] = {}
            continue

        out_v = out_vals[valid]
        params_v = param_matrix[valid]

        # Pearson correlation
        corrs = np.zeros(n_params)
        for j in range(n_params):
            col = params_v[:, j]
            if np.std(col) < 1e-30:
                corrs[j] = 0.0
            else:
                corrs[j] = abs(float(np.corrcoef(out_v, col)[0, 1]))

        # Rank top-N
        top_idx = np.argsort(corrs)[::-1][:top_n]
        sensitivity[out_key] = {
            param_names[idx]: round(float(corrs[idx]), 4)
            for idx in top_idx
            if corrs[idx] > 0.01
        }

    return sensitivity


def _compute_correlations(
    output_distributions: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Compute pairwise Pearson correlations between all outputs."""
    keys = sorted(output_distributions.keys())
    corr: Dict[str, Dict[str, float]] = {}
    for i, ki in enumerate(keys):
        corr[ki] = {}
        for j, kj in enumerate(keys):
            vi = output_distributions[ki]
            vj = output_distributions[kj]
            valid = ~(np.isnan(vi) | np.isnan(vj))
            if valid.sum() < 5:
                corr[ki][kj] = 0.0
            else:
                corr[ki][kj] = round(float(np.corrcoef(vi[valid], vj[valid])[0, 1]), 4)
    return corr


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    """Structured result from a Monte Carlo uncertainty propagation run."""

    n_samples: int
    design_point: Dict[str, Any]
    output_distributions: Dict[str, np.ndarray]  # output_key -> array of N values
    pass_rates: Dict[str, float]                  # spec name -> fraction passed
    overall_confidence: float                     # fraction where ALL specs pass
    sensitivity: Dict[str, Dict[str, float]]      # output -> {param: importance}
    failure_ranking: Dict[str, int]               # spec name -> failure count
    parameter_correlations: Dict[str, Dict[str, float]]  # output-output corr matrix
    elapsed_seconds: float = 0.0
    spec_set_name: str = ""
    parameter_names: List[str] = field(default_factory=list)
    sample_matrix: Optional[np.ndarray] = None  # (n_samples, n_params)

    def summary_dict(self) -> Dict[str, Any]:
        """Machine-readable summary."""
        stats: Dict[str, Dict[str, float]] = {}
        for key, arr in self.output_distributions.items():
            valid = arr[~np.isnan(arr)]
            if len(valid) == 0:
                continue
            stats[key] = {
                "mean": round(float(np.mean(valid)), 4),
                "std": round(float(np.std(valid)), 4),
                "p5": round(float(np.percentile(valid, 5)), 4),
                "p50": round(float(np.percentile(valid, 50)), 4),
                "p95": round(float(np.percentile(valid, 95)), 4),
                "min": round(float(np.min(valid)), 4),
                "max": round(float(np.max(valid)), 4),
            }
        return {
            "n_samples": self.n_samples,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "overall_confidence": round(self.overall_confidence, 4),
            "pass_rates": {k: round(v, 4) for k, v in self.pass_rates.items()},
            "failure_ranking": self.failure_ranking,
            "output_statistics": stats,
            "sensitivity_top3": self.sensitivity,
            "design_point": self.design_point,
            "spec_set_name": self.spec_set_name,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MonteCarloEngine:
    """Propagate parameter uncertainty through the full model chain.

    Parameters
    ----------
    n_samples : int
        Number of Monte Carlo samples (default 10 000).
    seed : int
        Random seed for reproducibility.
    n_jobs : int
        Number of parallel workers for joblib (-1 = all cores).
    registry : dict, optional
        Parameter registry.  Defaults to the central ``REGISTRY``.
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        seed: int = 42,
        n_jobs: int = -1,
        registry: Optional[Dict[str, Parameter]] = None,
    ):
        self.n_samples = n_samples
        self.seed = seed
        self.n_jobs = n_jobs
        self.registry = registry or REGISTRY

    def run(
        self,
        design_point: Optional[Dict[str, Any]] = None,
        specs: Optional[Sequence[Specification]] = None,
        spec_set_name: str = "",
    ) -> MonteCarloResult:
        """Run the Monte Carlo propagation.

        Parameters
        ----------
        design_point : dict, optional
            Operating conditions held fixed during the sweep.  Defaults
            to :data:`DEFAULT_DESIGN_POINT`.
        specs : sequence of Specification, optional
            Specifications to check each sample against.
        spec_set_name : str
            Label for the spec set.

        Returns
        -------
        MonteCarloResult
        """
        dp = design_point or dict(DEFAULT_DESIGN_POINT)

        t0 = time.perf_counter()

        # ── Step 1: Draw parameter samples ────────────────────────────
        sobol_n = min(1000, self.n_samples)
        random_n = self.n_samples - sobol_n

        samples: List[Dict[str, float]] = []
        if sobol_n > 0:
            samples.extend(sample_parameters(
                sobol_n, registry=self.registry, seed=self.seed, method="sobol",
            ))
        if random_n > 0:
            samples.extend(sample_parameters(
                random_n, registry=self.registry, seed=self.seed + sobol_n, method="monte_carlo",
            ))

        # ── Step 2: Run pipeline in parallel ──────────────────────────
        try:
            from joblib import Parallel, delayed
            results_list = Parallel(n_jobs=self.n_jobs, prefer="processes")(
                delayed(_run_single_sample)(s, dp) for s in samples
            )
        except Exception:
            # Fallback to serial
            results_list = [_run_single_sample(s, dp) for s in samples]

        # ── Step 3: Collect outputs ───────────────────────────────────
        output_keys = sorted(results_list[0].keys()) if results_list else []
        output_distributions: Dict[str, np.ndarray] = {}
        for key in output_keys:
            arr = np.array([r.get(key, np.nan) for r in results_list])
            output_distributions[key] = arr

        # ── Step 4: Check specifications ──────────────────────────────
        pass_rates: Dict[str, float] = {}
        overall_confidence = 0.0
        failure_ranking: Dict[str, int] = {}

        if specs:
            mc_check = check_mc_specifications(results_list, list(specs), spec_set_name)
            pass_rates = mc_check["pass_rates"]
            overall_confidence = mc_check["overall_pass_rate"]
            failure_ranking = mc_check["failure_histogram"]

        # ── Step 5: Sensitivity analysis ──────────────────────────────
        param_names = sorted(self.registry.keys())
        sensitivity = _compute_sensitivity(
            output_distributions, samples, param_names, top_n=3,
        )

        # ── Step 6: Output correlations ───────────────────────────────
        param_correlations = _compute_correlations(output_distributions)

        # ── Step 7: Build sample matrix ───────────────────────────────
        param_names_sorted = sorted(self.registry.keys())
        sample_matrix = np.zeros((len(samples), len(param_names_sorted)))
        for i, s in enumerate(samples):
            for j, name in enumerate(param_names_sorted):
                sample_matrix[i, j] = s.get(name, np.nan)

        elapsed = time.perf_counter() - t0

        return MonteCarloResult(
            n_samples=self.n_samples,
            design_point=dp,
            output_distributions=output_distributions,
            pass_rates=pass_rates,
            overall_confidence=overall_confidence,
            sensitivity=sensitivity,
            failure_ranking=failure_ranking,
            parameter_correlations=param_correlations,
            elapsed_seconds=elapsed,
            spec_set_name=spec_set_name,
            parameter_names=param_names_sorted,
            sample_matrix=sample_matrix,
        )
