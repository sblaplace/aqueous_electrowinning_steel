"""Calibration of Fe-deposition/HER screening kinetics against Phase-I measurements.

The fit is deliberately an *apparent total-cathodic-current* calibration.  An
LSV alone does not uniquely identify Fe deposition and HER; use independent
Faradaic-efficiency, gas, RDE, or composition measurements before treating the
separate branches as mechanistic constants.  The report keeps that limitation
and the source run identifiers beside every fitted parameter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .campaign import validate_manifest
from .eis import exchange_current_from_rct, fit_randles_spectrum, load_spectrum
from .kinetics import DepositionKinetics
from .experimental_data import load_measurements


@dataclass(frozen=True)
class PolarizationFit:
    """Parameters and diagnostics from a bounded total-current LSV fit."""

    fe_i0_A_m2: float
    her_i0_A_m2: float
    fe_tafel_V_dec: float
    her_tafel_V_dec: float
    boundary_layer_m: float
    rmse_log10_current: float
    n_points: int
    converged: bool
    parameter_std_log10: dict[str, float | None]
    assumptions: tuple[str, ...]


def _standard_errors(jacobian: np.ndarray, residual: np.ndarray, names: Sequence[str]) -> dict[str, float | None]:
    """Approximate standard errors in log10-parameter space from the Jacobian."""
    if len(residual) <= len(names):
        return {name: None for name in names}
    try:
        covariance = np.linalg.pinv(jacobian.T @ jacobian) * np.sum(residual ** 2) / (len(residual) - len(names))
        values = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        return {name: float(value) for name, value in zip(names, values)}
    except np.linalg.LinAlgError:
        return {name: None for name in names}


def fit_total_cathodic_polarization(
    data: pd.DataFrame,
    *,
    pH: float,
    temperature_C: float,
    fe_conc_M: float,
    reference_to_she_V: float,
    fe_E_eq_V_vs_she: float = -0.440,
    potential_min_V_vs_ref: float | None = None,
    potential_max_V_vs_ref: float | None = None,
    initial_fe_i0_A_m2: float = 1e-2,
    initial_her_i0_A_m2: float = 1e-3,
    initial_fe_tafel_V_dec: float = 0.12,
    initial_her_tafel_V_dec: float = 0.14,
    initial_boundary_layer_m: float = 5e-5,
) -> PolarizationFit:
    """Fit the sum of Fe and HER cathodic currents to a mapped LSV.

    ``reference_to_she_V`` is added to measured potentials to obtain V vs SHE
    (for example the temperature- and electrolyte-appropriate reference
    conversion).  It must be documented for each real dataset.
    """
    needed = {"potential_V_vs_ref", "current_density_A_m2"}
    missing = needed - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    frame = data.copy()
    if potential_min_V_vs_ref is not None:
        frame = frame[frame["potential_V_vs_ref"] >= potential_min_V_vs_ref]
    if potential_max_V_vs_ref is not None:
        frame = frame[frame["potential_V_vs_ref"] <= potential_max_V_vs_ref]
    # Canonical data use negative cathodic current.  Positive points contain no
    # usable cathodic kinetic signal and are excluded rather than silently flipped.
    frame = frame[frame["current_density_A_m2"] < 0].copy()
    E_she = frame["potential_V_vs_ref"].to_numpy(float) + reference_to_she_V
    observed = -frame["current_density_A_m2"].to_numpy(float)
    valid = np.isfinite(E_she) & np.isfinite(observed) & (observed > 0)
    E_she, observed = E_she[valid], observed[valid]
    if len(observed) < 10:
        raise ValueError("Polarization calibration requires at least 10 finite cathodic points")
    if min(pH, temperature_C, fe_conc_M) < 0 or fe_conc_M <= 0:
        raise ValueError("pH, temperature_C, and fe_conc_M must be physically positive/nonnegative")

    # log(i0), tafel slopes, and log(delta): bounded to physically plausible
    # screening ranges, avoiding unconstrained fits that masquerade as knowledge.
    x0 = np.log10([initial_fe_i0_A_m2, initial_her_i0_A_m2, initial_boundary_layer_m])
    x0 = np.array([x0[0], x0[1], initial_fe_tafel_V_dec, initial_her_tafel_V_dec, x0[2]])
    lower = np.array([-12.0, -12.0, 0.02, 0.02, -7.0])
    upper = np.array([5.0, 5.0, 0.50, 0.50, -2.0])

    def model(parameters: np.ndarray) -> np.ndarray:
        kinetic = DepositionKinetics(
            pH=pH, temperature_C=temperature_C, fe_i0=10 ** parameters[0], her_i0=10 ** parameters[1],
            fe_tafel_V=parameters[2], her_tafel_V=parameters[3], fe_conc_M=fe_conc_M,
            boundary_layer_m=10 ** parameters[4], fe_E_eq=fe_E_eq_V_vs_she,
        )
        return np.asarray(kinetic.partial_currents(E_she)[2], dtype=float)

    def residual(parameters: np.ndarray) -> np.ndarray:
        return np.log10(np.maximum(model(parameters), 1e-30)) - np.log10(observed)

    result = least_squares(residual, x0=np.clip(x0, lower, upper), bounds=(lower, upper), max_nfev=20_000)
    p = result.x
    std = _standard_errors(result.jac, result.fun, ("fe_i0", "her_i0", "fe_tafel", "her_tafel", "boundary_layer"))
    return PolarizationFit(
        fe_i0_A_m2=float(10 ** p[0]), her_i0_A_m2=float(10 ** p[1]),
        fe_tafel_V_dec=float(p[2]), her_tafel_V_dec=float(p[3]), boundary_layer_m=float(10 ** p[4]),
        rmse_log10_current=float(np.sqrt(np.mean(result.fun ** 2))), n_points=int(len(observed)),
        converged=bool(result.success), parameter_std_log10=std,
        assumptions=(
            "Fits total cathodic current, not independently measured Fe and HER partial currents.",
            "Potential conversion to SHE is supplied by the caller and must be temperature/electrolyte appropriate.",
            "Use independent FE, gas, RDE, or deposit-composition measurements to identify separate branches.",
        ),
    )


def calibration_ready_runs(manifest_path: str | Path) -> pd.DataFrame:
    """Return complete, linked campaign rows that passed the manifest QA check."""
    report = validate_manifest(manifest_path)
    ready = {item["run_id"] for item in report["runs"] if item["ready_for_analysis"]}
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    return frame[frame["run_id"].isin(ready)].copy()


def calibrate_lsv_run(
    manifest_path: str | Path, run_id: str, **fit_kwargs: float,
) -> dict:
    """Load one QA-ready LSV manifest entry, fit it, and return a traceable report."""
    manifest = Path(manifest_path)
    runs = calibration_ready_runs(manifest)
    selected = runs[(runs["run_id"] == run_id) & (runs["technique"].str.upper().isin(["LSV", "CV"]))]
    if selected.empty:
        raise ValueError(f"Run {run_id!r} is not a QA-ready LSV/CV entry")
    if len(selected) > 1:
        raise ValueError(f"Run {run_id!r} occurs more than once in the ready manifest")
    row = selected.iloc[0]
    data = load_measurements(manifest.parent / row["processed_file"])
    fit = fit_total_cathodic_polarization(data, **fit_kwargs)
    return {"run_id": run_id, "processed_file": row["processed_file"], "polarization_fit": asdict(fit)}


def fit_eis_exchange_current(path: str | Path, *, include_warburg: bool = True, n_electrons: int = 2) -> dict:
    """Fit an EIS spectrum and report the near-equilibrium implied current density.

    The result is explicitly a consistency value, not a branch-specific Fe i0,
    unless the measurement is near equilibrium and competing reactions are absent.
    """
    spectrum = load_spectrum(path)
    freq = spectrum["frequency_hz"].to_numpy(float)
    z = spectrum["z_real_ohm"].to_numpy(float) + 1j * spectrum["z_imag_ohm"].to_numpy(float)
    fit = fit_randles_spectrum(freq, z, include_warburg=include_warburg)
    area = float(spectrum["working_electrode_area_cm2"].iloc[0]) if "working_electrode_area_cm2" in spectrum else None
    temperature_K = float(spectrum["temperature_C"].iloc[0]) + 273.15 if "temperature_C" in spectrum else 298.15
    current_A = exchange_current_from_rct(fit.rct_ohm, n_electrons=n_electrons, temperature_K=temperature_K)
    return {
        "fit": asdict(fit), "exchange_current_A_from_rct": current_A,
        "exchange_current_density_A_m2_from_rct": None if area is None else current_A / (area * 1e-4),
        "assumption": "Valid as an exchange-current estimate only near equilibrium; at biased Fe/HER electrodes Rct is combined faradaic conductance.",
    }
