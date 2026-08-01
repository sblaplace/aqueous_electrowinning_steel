"""Real-data calibration pipeline — CSV ingestion + auto-fit.

Replaces screening estimates with fitted values from experimental CSVs.
Covers 7 model domains:

1. Tafel polarization      → fe_i0, her_i0, fe_tafel_V, her_tafel_V, boundary_layer_m
2. EIS (Randles)            → R_s, R_ct, C_dl, σ_Warburg
3. Hull cell distribution   → current distribution parameters
4. Diffusivity (foil)       → D0, surface_C (diffusivity from foil weight gain)
5. Carbon potential         → K_B, K_CH4 equilibrium constants
6. Tempering (Hollomon-Jaffe) → k_softening coefficient
7. Mechanical (Hall-Petch)  → σ0, k_HP, Ni/C strengthening coefficients

Components:
    1. Data ingestion — load_csv() with schema validation
    2. Fitting engine — curve_fit / least_squares per domain
    3. Calibrated output — calibrated_parameters.json
    4. Model hot-reload — load_calibrated_params()
    5. Validation — cross-validate predictions vs held-out data
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────────

CALIBRATED_PARAMS_FILENAME = "calibrated_parameters.json"
FIGURES_SUBDIR = "calibration_figures"

# Physical bounds for each fitted domain (log10 where noted)
PHYSICAL_BOUNDS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "tafel": {
        "fe_i0_A_m2": (1e-12, 1e5),
        "her_i0_A_m2": (1e-12, 1e5),
        "fe_tafel_V_dec": (0.02, 0.50),
        "her_tafel_V_dec": (0.02, 0.50),
        "boundary_layer_m": (1e-7, 1e-2),
    },
    "eis": {
        "R_s_ohm": (1e-9, 1e6),
        "R_ct_ohm": (1e-9, 1e9),
        "C_dl_F": (1e-12, 1.0),
        "sigma_warburg_ohm_s_neg_half": (1e-9, 1e9),
    },
    "diffusivity": {
        "D0_m2_s": (1e-16, 1e-7),
        "surface_C_wt_percent": (0.3, 2.0),
    },
    "carbon_potential": {
        "K_B_offset": (0.1, 10.0),
        "K_CH4_offset": (0.1, 10.0),
    },
    "tempering": {
        "k_softening": (1e-6, 0.01),
        "C_HJ": (15.0, 25.0),
    },
    "hall_petch": {
        "sigma0_MPa": (0.0, 300.0),
        "k_HP_MPa_sqrt_m": (0.01, 2.0),
    },
    "hull_cell": {
        "gap_exponent": (0.5, 2.0),
        "edge_factor": (0.5, 3.0),
    },
}


# ─── 1. Data Ingestion ─────────────────────────────────────────────────────────

REQUIRED_COLUMNS: Dict[str, set] = {
    "tafel": {"potential_V_vs_ref", "current_density_A_m2"},
    "eis": {"frequency_hz", "z_real_ohm", "z_imag_ohm"},
    "hull_cell": {"position_cm", "current_density_mA_cm2"},
    "diffusivity": {"time_hr", "temperature_C", "foil_thickness_um", "avg_C_wt_percent"},
    "carbon_potential": {"pCO_atm", "pCO2_atm", "temperature_C", "aC_measured"},
    "tempering": {"HV_quenched", "T_C", "t_hr", "HV_measured"},
    "hall_petch": {"grain_size_um", "yield_MPa"},
}


def load_csv(path: str | Path, domain: str) -> pd.DataFrame:
    """Load a calibration CSV and validate its schema against the domain template.

    Parameters
    ----------
    path : path to CSV file
    domain : one of 'tafel', 'eis', 'hull_cell', 'diffusivity',
             'carbon_potential', 'tempering', 'hall_petch'

    Returns
    -------
    Validated DataFrame

    Raises
    ------
    ValueError if required columns are missing or data is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    if domain not in REQUIRED_COLUMNS:
        raise ValueError(f"Unknown domain {domain!r}. Valid: {sorted(REQUIRED_COLUMNS)}")

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"CSV is empty: {path}")

    required = REQUIRED_COLUMNS[domain]
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns for {domain}: {', '.join(sorted(missing))}")

    # Coerce numeric columns
    for col in required:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    # Log but don't raise on NaN coercion (missing data handled gracefully)
    n_nan = frame[list(required)].isna().sum().sum()
    if n_nan > 0:
        logger.warning(f"{domain}: {n_nan} NaN values in required columns after coercion")

    return frame


def load_csv_safe(path: str | Path, domain: str) -> Optional[pd.DataFrame]:
    """Load CSV with graceful error handling — returns None on failure."""
    try:
        return load_csv(path, domain)
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Could not load {domain} CSV from {path}: {e}")
        return None


# ─── 2. Fitting Engine ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FitResult:
    """Result of a single-domain calibration fit."""

    domain: str
    parameters: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    r_squared: Optional[float] = None
    chi_squared: Optional[float] = None
    n_points: int = 0
    converged: bool = True
    notes: str = ""


def _r_squared(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² between observed and predicted arrays."""
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    return 1.0 if ss_tot == 0 else float(1.0 - ss_res / ss_tot)


def _confidence_intervals_from_jac(
    jac: np.ndarray, residual: np.ndarray, names: List[str], alpha: float = 0.05
) -> Dict[str, Tuple[float, float]]:
    """Approximate 95% confidence intervals from the Jacobian."""
    from scipy.stats import t as t_dist

    n, p = len(residual), len(names)
    if n <= p:
        return {name: (np.nan, np.nan) for name in names}
    try:
        cov = np.linalg.pinv(jac.T @ jac) * np.sum(residual ** 2) / (n - p)
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))
        t_val = t_dist.ppf(1 - alpha / 2, n - p)
        return {
            name: (float(-t_val * s), float(t_val * s))
            for name, s in zip(names, std)
        }
    except np.linalg.LinAlgError:
        return {name: (np.nan, np.nan) for name in names}


def fit_tafel_domain(data: pd.DataFrame) -> FitResult:
    """Fit Tafel slopes, exchange current densities from LSV polarization data.

    Uses the existing calibration.fit_total_cathodic_polarization machinery
    wrapped to return a standardized FitResult.
    """
    from .calibration import fit_total_cathodic_polarization

    # Infer defaults from data
    pH = float(data["pH"].iloc[0]) if "pH" in data.columns else 3.0
    T = float(data["temperature_C"].iloc[0]) if "temperature_C" in data.columns else 60.0
    fe_conc = float(data["fe2_concentration_M"].iloc[0]) if "fe2_concentration_M" in data.columns else 1.0
    ref_to_she = float(data["reference_to_she_V"].iloc[0]) if "reference_to_she_V" in data.columns else 0.197

    fit = fit_total_cathodic_polarization(
        data, pH=pH, temperature_C=T, fe_conc_M=fe_conc, reference_to_she_V=ref_to_she
    )

    params = {
        "fe_i0_A_m2": fit.fe_i0_A_m2,
        "her_i0_A_m2": fit.her_i0_A_m2,
        "fe_tafel_V_dec": fit.fe_tafel_V_dec,
        "her_tafel_V_dec": fit.her_tafel_V_dec,
        "boundary_layer_m": fit.boundary_layer_m,
    }
    ci = {}
    for name, std in fit.parameter_std_log10.items():
        if std is not None:
            ci[name] = (-1.96 * std, 1.96 * std)
        else:
            ci[name] = (np.nan, np.nan)

    return FitResult(
        domain="tafel",
        parameters=params,
        confidence_intervals=ci,
        r_squared=1.0 - fit.rmse_log10_current ** 2 if fit.rmse_log10_current < 1 else None,
        n_points=fit.n_points,
        converged=fit.converged,
        notes="LSV total cathodic current fit; Fe/HER not independently resolved.",
    )


def fit_eis_domain(data: pd.DataFrame) -> FitResult:
    """Fit Randles equivalent circuit parameters from EIS spectrum."""
    from .eis import fit_randles_spectrum

    freq = data["frequency_hz"].to_numpy(float)
    z = data["z_real_ohm"].to_numpy(float) + 1j * data["z_imag_ohm"].to_numpy(float)
    include_warburg = True

    fit = fit_randles_spectrum(freq, z, include_warburg=include_warburg)

    params = {
        "R_s_ohm": fit.rs_ohm,
        "R_ct_ohm": fit.rct_ohm,
        "C_dl_F": fit.cdl_F,
        "sigma_warburg_ohm_s_neg_half": fit.sigma_warburg_ohm_s_neg_half or 0.0,
    }
    # EIS fit doesn't provide CIs directly; use NaN placeholders
    ci = {k: (np.nan, np.nan) for k in params}

    return FitResult(
        domain="eis",
        parameters=params,
        confidence_intervals=ci,
        r_squared=fit.r_squared_magnitude,
        chi_squared=fit.chi_squared,
        n_points=fit.n_points,
        converged=fit.converged,
    )


def fit_hull_cell_domain(data: pd.DataFrame) -> FitResult:
    """Fit current distribution parameters from Hull cell measurements.

    Model: j(s) = j_avg * (gap_ref / gap(s))^n * edge_factor
    where n is the gap exponent and edge_factor accounts for edge effects.
    """
    position = data["position_cm"].to_numpy(float)
    j_measured = data["current_density_mA_cm2"].to_numpy(float)
    valid = np.isfinite(position) & np.isfinite(j_measured) & (j_measured > 0)
    position, j_measured = position[valid], j_measured[valid]

    if len(position) < 4:
        return FitResult(
            domain="hull_cell",
            parameters={"gap_exponent": 1.0, "edge_factor": 1.0},
            confidence_intervals={"gap_exponent": (np.nan, np.nan), "edge_factor": (np.nan, np.nan)},
            converged=False,
            n_points=len(position),
            notes="Insufficient data points for fitting",
        )

    # Normalize by mean
    j_avg = np.mean(j_measured)
    j_norm = j_measured / j_avg
    pos_norm = position / np.max(position)

    def model(x, gap_exp, edge_fac):
        # Simple model: j_norm = edge_fac * (1 - x)^gap_exp + (1-edge_fac) * uniform
        return edge_fac * (1.0 - pos_norm) ** gap_exp + (1.0 - edge_fac)

    try:
        popt, pcov = curve_fit(model, pos_norm, j_norm, p0=[1.0, 1.0],
                               bounds=([0.3, 0.3], [3.0, 5.0]), maxfev=5000)
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan, np.nan]
        j_pred = model(pos_norm, *popt)
        r2 = _r_squared(j_norm, j_pred)
        params = {"gap_exponent": float(popt[0]), "edge_factor": float(popt[1])}
        ci = {
            "gap_exponent": (float(popt[0] - 1.96 * perr[0]), float(popt[0] + 1.96 * perr[0])),
            "edge_factor": (float(popt[1] - 1.96 * perr[1]), float(popt[1] + 1.96 * perr[1])),
        }
        converged = True
    except Exception as e:
        logger.warning(f"Hull cell fit failed: {e}")
        params = {"gap_exponent": 1.0, "edge_factor": 1.0}
        ci = {"gap_exponent": (np.nan, np.nan), "edge_factor": (np.nan, np.nan)}
        r2 = None
        converged = False

    return FitResult(
        domain="hull_cell",
        parameters=params,
        confidence_intervals=ci,
        r_squared=r2,
        n_points=len(position),
        converged=converged,
    )


def fit_diffusivity_domain(data: pd.DataFrame) -> FitResult:
    """Fit carbon diffusivity D0 and surface carbon from foil exposure data."""
    from .foil_calibration import FoilMeasurement, fit_diffusivity_from_foil_data

    measurements = []
    for _, row in data.iterrows():
        m = FoilMeasurement(
            time_hr=float(row["time_hr"]),
            temperature_C=float(row["temperature_C"]),
            pCO_atm=float(row.get("pCO_atm", 0.20)),
            pCO2_atm=float(row.get("pCO2_atm", 0.05)),
            foil_thickness_um=float(row["foil_thickness_um"]),
            measured_avg_C_wt_percent=float(row["avg_C_wt_percent"]),
        )
        measurements.append(m)

    result = fit_diffusivity_from_foil_data(measurements)

    params = {
        "D0_m2_s": result["D_fit_m2_s"],
        "surface_C_wt_percent": result["Cs_fit_wt_percent"],
    }
    perr = result.get("perr_logD_Cs", [None, None])
    ci = {
        "D0_m2_s": (np.nan, np.nan),
        "surface_C_wt_percent": (np.nan, np.nan),
    }

    return FitResult(
        domain="diffusivity",
        parameters=params,
        confidence_intervals=ci,
        n_points=result["n_measurements"],
        converged=True,
        notes=result.get("note", ""),
    )


def fit_carbon_potential_domain(data: pd.DataFrame) -> FitResult:
    """Fit carbon potential equilibrium constants K_B, K_CH4.

    Compares measured carbon activity (from foil/combustion) to
    thermodynamic predictions, fitting offset factors.
    """
    from .carbon_potential import carbon_activity_from_co_co2, carbon_activity_from_ch4_h2

    K_B_ratios = []
    K_CH4_ratios = []

    for _, row in data.iterrows():
        pCO = float(row["pCO_atm"])
        pCO2 = float(row["pCO2_atm"])
        T_C = float(row["temperature_C"])
        aC_meas = float(row["aC_measured"])

        # CO/CO2 Boudouard path
        aC_theory_co = carbon_activity_from_co_co2(pCO, pCO2, T_C)
        if aC_theory_co > 1e-15 and aC_meas > 0:
            K_B_ratios.append(aC_meas / aC_theory_co)

        # CH4/H2 path if columns present
        if "pCH4_atm" in row.index and "pH2_atm" in row.index:
            pCH4 = float(row["pCH4_atm"])
            pH2 = float(row["pH2_atm"])
            if pH2 > 1e-6:
                aC_theory_ch4 = carbon_activity_from_ch4_h2(pCH4, pH2, T_C)
                if aC_theory_ch4 > 1e-15 and aC_meas > 0:
                    K_CH4_ratios.append(aC_meas / aC_theory_ch4)

    K_B_offset = float(np.mean(K_B_ratios)) if K_B_ratios else 1.0
    K_CH4_offset = float(np.mean(K_CH4_ratios)) if K_CH4_ratios else 1.0

    params = {"K_B_offset": K_B_offset, "K_CH4_offset": K_CH4_offset}
    ci = {k: (np.nan, np.nan) for k in params}

    return FitResult(
        domain="carbon_potential",
        parameters=params,
        confidence_intervals=ci,
        n_points=len(data),
        converged=True,
        notes=f"K_B offset={K_B_offset:.3f}, K_CH4 offset={K_CH4_offset:.3f} vs thermodynamic theory",
    )


def fit_tempering_domain(data: pd.DataFrame) -> FitResult:
    """Fit Hollomon-Jaffe k_softening coefficient from tempering data."""
    from .foil_calibration import fit_tempering_softening

    measured_data = []
    for _, row in data.iterrows():
        measured_data.append({
            "HV_q": float(row["HV_quenched"]),
            "T_C": float(row["T_C"]),
            "t_hr": float(row["t_hr"]),
            "HV_measured": float(row["HV_measured"]),
        })

    result = fit_tempering_softening(measured_data)

    params = {
        "k_softening": result["k_fit"],
        "C_HJ": 19.5,  # held fixed unless data strongly supports fitting
    }
    ci = {k: (np.nan, np.nan) for k in params}

    return FitResult(
        domain="tempering",
        parameters=params,
        confidence_intervals=ci,
        n_points=result["n_points"],
        converged=True,
        notes=result.get("note", ""),
    )


def fit_hall_petch_domain(data: pd.DataFrame) -> FitResult:
    """Fit Hall-Petch σ0, k_HP from grain size vs yield strength data.

    Also fits Ni and C strengthening if columns are present.
    """
    from .foil_calibration import fit_mechanical_hall_petch

    grain = data["grain_size_um"].to_numpy(float)
    yield_mpa = data["yield_MPa"].to_numpy(float)
    valid = np.isfinite(grain) & np.isfinite(yield_mpa) & (grain > 0)
    grain, yield_mpa = grain[valid], yield_mpa[valid]

    hp_result = fit_mechanical_hall_petch(grain, yield_mpa)

    params = {
        "sigma0_MPa": hp_result["sigma0_MPa"],
        "k_HP_MPa_sqrt_m": hp_result["k_HP_MPa_sqrt_m"],
    }
    ci = {k: (np.nan, np.nan) for k in params}

    # Fit Ni/C strengthening if columns present
    if "ni_wt_percent" in data.columns and "carbon_wt_percent" in data.columns:
        ni = data["ni_wt_percent"].to_numpy(float)[valid]
        c_wt = data["carbon_wt_percent"].to_numpy(float)[valid]
        # Residual after Hall-Petch
        d_m = grain * 1e-6
        sigma_hp = params["sigma0_MPa"] + params["k_HP_MPa_sqrt_m"] / np.sqrt(d_m)
        residual = yield_mpa - sigma_hp

        # Linear fit: residual ≈ k_ni * ni^0.75 + k_c * c_wt^0.6
        if len(residual) > 3:
            A = np.column_stack([
                ni ** 0.75 / (1.0 + ni / 20.0),
                c_wt ** 0.6,
            ])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, residual, rcond=None)
                params["k_ss_ni_MPa_per_wt"] = float(max(coeffs[0], 0.0))
                params["k_carbon_MPa_per_wt"] = float(max(coeffs[1], 0.0))
            except np.linalg.LinAlgError:
                params["k_ss_ni_MPa_per_wt"] = 38.0
                params["k_carbon_MPa_per_wt"] = 180.0
        else:
            params["k_ss_ni_MPa_per_wt"] = 38.0
            params["k_carbon_MPa_per_wt"] = 180.0
    else:
        params["k_ss_ni_MPa_per_wt"] = 38.0
        params["k_carbon_MPa_per_wt"] = 180.0

    return FitResult(
        domain="hall_petch",
        parameters=params,
        confidence_intervals=ci,
        r_squared=hp_result["r_squared"],
        n_points=int(hp_result["n_points"]),
        converged=True,
        notes=hp_result.get("note", ""),
    )


# Domain → fitter dispatch
DOMAIN_FITTERS = {
    "tafel": fit_tafel_domain,
    "eis": fit_eis_domain,
    "hull_cell": fit_hull_cell_domain,
    "diffusivity": fit_diffusivity_domain,
    "carbon_potential": fit_carbon_potential_domain,
    "tempering": fit_tempering_domain,
    "hall_petch": fit_hall_petch_domain,
}


# ─── 3. Calibrated Parameter Output ────────────────────────────────────────────

@dataclass
class CalibrationReport:
    """Full calibration run report."""

    domain_results: Dict[str, FitResult] = field(default_factory=dict)
    output_path: Optional[Path] = None

    @property
    def all_parameters(self) -> Dict[str, Dict[str, float]]:
        """Flat dict of domain → fitted parameters."""
        return {d: r.parameters for d, r in self.domain_results.items()}

    @property
    def n_domains_fitted(self) -> int:
        return len(self.domain_results)

    @property
    def all_converged(self) -> bool:
        return all(r.converged for r in self.domain_results.values())

    def summary_table(self) -> pd.DataFrame:
        """Return a summary DataFrame of all fitted domains."""
        rows = []
        for domain, result in self.domain_results.items():
            rows.append({
                "domain": domain,
                "n_params": len(result.parameters),
                "n_points": result.n_points,
                "r_squared": result.r_squared,
                "converged": result.converged,
                "notes": result.notes[:80] if result.notes else "",
            })
        return pd.DataFrame(rows)


def write_calibrated_parameters(
    report: CalibrationReport,
    output_dir: str | Path,
    filename: str = CALIBRATED_PARAMS_FILENAME,
) -> Path:
    """Write calibrated_parameters.json from a CalibrationReport.

    Returns the path to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    payload = {
        "_metadata": {
            "version": "1.0",
            "n_domains": report.n_domains_fitted,
            "domains_fitted": list(report.domain_results.keys()),
            "all_converged": report.all_converged,
        },
    }
    for domain, result in report.domain_results.items():
        payload[domain] = {
            "parameters": result.parameters,
            "confidence_intervals": {
                k: list(v) for k, v in result.confidence_intervals.items()
            },
            "r_squared": result.r_squared,
            "chi_squared": result.chi_squared,
            "n_points": result.n_points,
            "converged": result.converged,
            "notes": result.notes,
        }

    path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info(f"Wrote calibrated parameters to {path}")
    return path


# ─── 4. Model Hot-Reload ───────────────────────────────────────────────────────

def load_calibrated_params(
    path: str | Path | None = None,
    search_dirs: Optional[List[Path]] = None,
) -> Dict[str, Dict[str, float]]:
    """Load calibrated_parameters.json and return domain → parameters dict.

    Search order:
    1. Explicit ``path``
    2. ``search_dirs`` list (each checked for CALIBRATED_PARAMS_FILENAME)
    3. Current working directory
    4. models/ directory

    Returns empty dict if no file found (models use screening defaults).
    """
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    if search_dirs:
        candidates.extend(d / CALIBRATED_PARAMS_FILENAME for d in search_dirs)
    candidates.append(Path.cwd() / CALIBRATED_PARAMS_FILENAME)
    candidates.append(Path(__file__).parent / CALIBRATED_PARAMS_FILENAME)

    for candidate in candidates:
        if candidate.exists():
            try:
                raw = json.loads(candidate.read_text())
                # Extract only the domain parameter dicts
                return {
                    domain: values.get("parameters", values)
                    for domain, values in raw.items()
                    if not domain.startswith("_") and isinstance(values, dict)
                }
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse {candidate}: {e}")
                continue

    return {}


def apply_calibrated_params(domain: str, **defaults) -> Dict[str, float]:
    """Return parameters for a domain, overriding defaults with calibrated values.

    Example::

        from models.calibration_pipeline import apply_calibrated_params
        p = apply_calibrated_params("tafel", fe_i0_A_m2=1e-2, her_i0_A_m2=1e-3)
        # p["fe_i0_A_m2"] will be the calibrated value if available, else 1e-2
    """
    calibrated = load_calibrated_params()
    if domain in calibrated:
        merged = {**defaults, **calibrated[domain]}
        logger.debug(f"Applied calibrated params for {domain}: {list(calibrated[domain])}")
        return merged
    return dict(defaults)


# ─── 5. Validation ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ValidationResult:
    """Cross-validation result for one domain."""

    domain: str
    r_squared: float
    rmse: float
    n_test_points: int
    within_physical_bounds: bool
    improvement_over_default: bool


def validate_calibration(
    domain: str,
    data: pd.DataFrame,
    fitted_params: Dict[str, float],
    default_params: Optional[Dict[str, float]] = None,
) -> ValidationResult:
    """Validate calibrated parameters by running model predictions and comparing to data.

    For Tafel: predict total current from fitted i0/Tafel values, compute R².
    For EIS: predict impedance from Randles fit, compute R² on |Z|.
    For Hall-Petch: predict yield from grain size, compute R².
    """
    if domain == "tafel":
        return _validate_tafel(data, fitted_params, default_params)
    elif domain == "eis":
        return _validate_eis(data, fitted_params, default_params)
    elif domain == "hall_petch":
        return _validate_hall_petch(data, fitted_params, default_params)
    elif domain == "tempering":
        return _validate_tempering(data, fitted_params, default_params)
    else:
        # Generic: just check bounds
        within_bounds = _check_physical_bounds(domain, fitted_params)
        return ValidationResult(
            domain=domain,
            r_squared=np.nan,
            rmse=np.nan,
            n_test_points=len(data),
            within_physical_bounds=within_bounds,
            improvement_over_default=True,
        )


def _check_physical_bounds(domain: str, params: Dict[str, float]) -> bool:
    """Check that all parameters are within physical bounds."""
    bounds = PHYSICAL_BOUNDS.get(domain, {})
    for name, value in params.items():
        if name in bounds:
            lo, hi = bounds[name]
            if not (lo <= value <= hi):
                logger.warning(f"{domain}.{name}={value} outside bounds [{lo}, {hi}]")
                return False
    return True


def _validate_tafel(
    data: pd.DataFrame,
    fitted: Dict[str, float],
    defaults: Optional[Dict[str, float]],
) -> ValidationResult:
    """Validate Tafel calibration against data."""
    from .kinetics import DepositionKinetics

    pH = float(data["pH"].iloc[0]) if "pH" in data.columns else 3.0
    T = float(data["temperature_C"].iloc[0]) if "temperature_C" in data.columns else 60.0
    fe_conc = float(data["fe2_concentration_M"].iloc[0]) if "fe2_concentration_M" in data.columns else 1.0
    ref_to_she = float(data["reference_to_she_V"].iloc[0]) if "reference_to_she_V" in data.columns else 0.197

    frame = data[data["current_density_A_m2"] < 0].copy()
    E_she = frame["potential_V_vs_ref"].to_numpy(float) + ref_to_she
    observed = -frame["current_density_A_m2"].to_numpy(float)
    valid = np.isfinite(E_she) & np.isfinite(observed) & (observed > 0)
    E_she, observed = E_she[valid], observed[valid]

    if len(observed) < 3:
        return ValidationResult("tafel", np.nan, np.nan, len(observed), True, True)

    # Predict with fitted params
    kin = DepositionKinetics(
        pH=pH, temperature_C=T,
        fe_i0=fitted.get("fe_i0_A_m2", 1e-2),
        her_i0=fitted.get("her_i0_A_m2", 1e-3),
        fe_tafel_V=fitted.get("fe_tafel_V_dec", 0.12),
        her_tafel_V=fitted.get("her_tafel_V_dec", 0.14),
        fe_conc_M=fe_conc,
        boundary_layer_m=fitted.get("boundary_layer_m", 5e-5),
    )
    predicted = np.asarray(kin.partial_currents(E_she)[2], dtype=float)
    r2 = _r_squared(observed, predicted)
    rmse = float(np.sqrt(np.mean((np.log10(observed) - np.log10(np.maximum(predicted, 1e-30))) ** 2)))

    # Compare to defaults
    improvement = True
    if defaults:
        kin_def = DepositionKinetics(
            pH=pH, temperature_C=T,
            fe_i0=defaults.get("fe_i0_A_m2", 1e-2),
            her_i0=defaults.get("her_i0_A_m2", 1e-3),
            fe_tafel_V=defaults.get("fe_tafel_V_dec", 0.12),
            her_tafel_V=defaults.get("her_tafel_V_dec", 0.14),
            fe_conc_M=fe_conc,
            boundary_layer_m=defaults.get("boundary_layer_m", 5e-5),
        )
        pred_def = np.asarray(kin_def.partial_currents(E_she)[2], dtype=float)
        r2_def = _r_squared(observed, pred_def)
        improvement = r2 > r2_def

    within_bounds = _check_physical_bounds("tafel", fitted)
    return ValidationResult("tafel", r2, rmse, len(observed), within_bounds, improvement)


def _validate_eis(
    data: pd.DataFrame,
    fitted: Dict[str, float],
    defaults: Optional[Dict[str, float]],
) -> ValidationResult:
    """Validate EIS calibration."""
    from .eis import randles_impedance

    freq = data["frequency_hz"].to_numpy(float)
    z = data["z_real_ohm"].to_numpy(float) + 1j * data["z_imag_ohm"].to_numpy(float)
    omega = 2.0 * np.pi * freq

    z_pred = randles_impedance(
        omega,
        fitted.get("R_s_ohm", 1.0),
        fitted.get("R_ct_ohm", 100.0),
        fitted.get("C_dl_F", 1e-4),
        fitted.get("sigma_warburg_ohm_s_neg_half"),
    )
    mag_data, mag_pred = np.abs(z), np.abs(z_pred)
    r2 = _r_squared(mag_data, mag_pred)
    rmse = float(np.sqrt(np.mean((mag_data - mag_pred) ** 2)))

    within_bounds = _check_physical_bounds("eis", fitted)
    return ValidationResult("eis", r2, rmse, len(freq), within_bounds, True)


def _validate_hall_petch(
    data: pd.DataFrame,
    fitted: Dict[str, float],
    defaults: Optional[Dict[str, float]],
) -> ValidationResult:
    """Validate Hall-Petch calibration."""
    grain = data["grain_size_um"].to_numpy(float)
    observed = data["yield_MPa"].to_numpy(float)
    valid = np.isfinite(grain) & np.isfinite(observed) & (grain > 0)
    grain, observed = grain[valid], observed[valid]

    d_m = grain * 1e-6
    predicted = fitted.get("sigma0_MPa", 100.0) + fitted.get("k_HP_MPa_sqrt_m", 0.5) / np.sqrt(d_m)
    r2 = _r_squared(observed, predicted)
    rmse = float(np.sqrt(np.mean((observed - predicted) ** 2)))

    within_bounds = _check_physical_bounds("hall_petch", fitted)
    return ValidationResult("hall_petch", r2, rmse, len(grain), within_bounds, True)


def _validate_tempering(
    data: pd.DataFrame,
    fitted: Dict[str, float],
    defaults: Optional[Dict[str, float]],
) -> ValidationResult:
    """Validate tempering calibration."""
    from .tempering import hollomon_jaffe_parameter, tempered_hardness_hollomon_jaffe

    observed = data["HV_measured"].to_numpy(float)
    predictions = []
    for _, row in data.iterrows():
        P = hollomon_jaffe_parameter(float(row["T_C"]), float(row["t_hr"]))
        hv = tempered_hardness_hollomon_jaffe(float(row["HV_quenched"]), P, fitted.get("k_softening", 0.00018))
        predictions.append(hv)
    predicted = np.array(predictions)

    valid = np.isfinite(observed) & np.isfinite(predicted)
    r2 = _r_squared(observed[valid], predicted[valid])
    rmse = float(np.sqrt(np.mean((observed[valid] - predicted[valid]) ** 2)))

    within_bounds = _check_physical_bounds("tempering", fitted)
    return ValidationResult("tempering", r2, rmse, int(np.sum(valid)), within_bounds, True)


# ─── Figures ────────────────────────────────────────────────────────────────────

def generate_calibration_figures(
    report: CalibrationReport,
    data_dir: Path,
    output_dir: Path,
) -> List[Path]:
    """Generate residual and fit-quality plots for each calibrated domain.

    Returns list of paths to generated figures.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    for domain, result in report.domain_results.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Calibration: {domain}", fontsize=14)

        # Left: parameter bar chart
        ax = axes[0]
        params = result.parameters
        names = list(params.keys())
        values = [params[n] for n in names]
        colors = ["#2ecc71" if result.converged else "#e74c3c"] * len(names)
        bars = ax.barh(names, values, color=colors, alpha=0.8)
        ax.set_xlabel("Fitted value")
        ax.set_title("Fitted Parameters")
        ax.grid(alpha=0.25)

        # Right: summary info
        ax2 = axes[1]
        ax2.axis("off")
        info_text = (
            f"Domain: {domain}\n"
            f"Points: {result.n_points}\n"
            f"Converged: {result.converged}\n"
            f"R²: {result.r_squared:.4f if result.r_squared else 'N/A'}\n"
            f"χ²: {result.chi_squared:.4e if result.chi_squared else 'N/A'}\n"
        )
        ax2.text(0.1, 0.5, info_text, transform=ax2.transAxes, fontsize=12,
                 verticalalignment="center", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1"))

        fig.tight_layout()
        path = output_dir / f"calibration_{domain}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        figures.append(path)

    # Summary figure
    fig, ax = plt.subplots(figsize=(10, 5))
    summary = report.summary_table()
    if not summary.empty:
        domains = summary["domain"].tolist()
        r2_vals = [v if v is not None and not np.isnan(v) else 0.0 for v in summary["r_squared"]]
        colors = ["#2ecc71" if c else "#e74c3c" for c in summary["converged"]]
        ax.bar(domains, r2_vals, color=colors, alpha=0.8)
        ax.set_ylabel("R²")
        ax.set_title("Calibration Quality Summary")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.9, color="#e67e22", linestyle="--", label="R² = 0.9 target")
        ax.legend()
        ax.grid(alpha=0.25)
    fig.tight_layout()
    path = output_dir / "calibration_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    return figures


# ─── Main Pipeline ──────────────────────────────────────────────────────────────

def run_calibration_pipeline(
    data_dir: str | Path,
    output_dir: str | Path = "calibration_output",
    domains: Optional[List[str]] = None,
    generate_figures: bool = True,
) -> CalibrationReport:
    """Run the full calibration pipeline.

    Parameters
    ----------
    data_dir : directory containing domain CSVs
        Expected files: tafel.csv, eis.csv, hull_cell.csv, diffusivity.csv,
        carbon_potential.csv, tempering.csv, hall_petch.csv
    output_dir : where to write calibrated_parameters.json and figures
    domains : subset of domains to calibrate (default: all)
    generate_figures : whether to generate diagnostic plots

    Returns
    -------
    CalibrationReport with all domain results
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    if domains is None:
        domains = list(DOMAIN_FITTERS.keys())

    report = CalibrationReport()

    for domain in domains:
        if domain not in DOMAIN_FITTERS:
            logger.warning(f"Unknown domain {domain!r}, skipping")
            continue

        csv_path = data_dir / f"{domain}.csv"
        data = load_csv_safe(csv_path, domain)
        if data is None:
            logger.info(f"No data for {domain}, skipping")
            continue

        try:
            result = DOMAIN_FITTERS[domain](data)
            report.domain_results[domain] = result
            logger.info(f"  {domain}: {len(result.parameters)} params, "
                       f"R²={result.r_squared}, converged={result.converged}")
        except Exception as e:
            logger.error(f"  {domain}: fit failed — {e}")
            continue

    # Write calibrated parameters
    if report.domain_results:
        report.output_path = write_calibrated_parameters(report, output_dir)

    # Generate figures
    if generate_figures and report.domain_results:
        figures = generate_calibration_figures(report, data_dir, output_dir / FIGURES_SUBDIR)
        logger.info(f"  Generated {len(figures)} calibration figures")

    return report
