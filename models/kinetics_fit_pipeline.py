"""
Bridge module connecting raw RDE/LSV and EIS measurements to the existing
calibration machinery (models.calibration, models.calibration_pipeline).
"""

from typing import Union, Optional
from pathlib import Path
import numpy as np
import pandas as pd

from .calibration import fit_total_cathodic_polarization, PolarizationFit
from .kinetics import DepositionKinetics

# Explicit column-name contract for instrument exports
RAW_LSV_MAPPING = {
    "Voltage_V": "potential_V_vs_ref",
    "Current_A": "current_A",
    "Area_cm2": "working_electrode_area_cm2",
    "pH": "pH",
    "Temp_C": "temperature_C",
    "Fe_M": "fe2_concentration_M",
    "Ref_V": "reference_to_she_V",
}

RAW_EIS_MAPPING = {
    "Frequency_Hz": "frequency_hz",
    "Z_real_Ohm": "z_real_ohm",
    "Z_imag_Ohm": "z_imag_ohm",
}


def polarization_from_export(path: Union[str, Path]) -> pd.DataFrame:
    """
    Adapter to convert a raw LSV/RDE export into the fitter's expected format.
    Requires at minimum Voltage_V, Current_A, and Area_cm2.
    """
    df = pd.read_csv(path)
    required_raw = {"Voltage_V", "Current_A", "Area_cm2"}
    missing = required_raw - set(df.columns)
    if missing:
        raise ValueError(f"Raw export is missing required columns: {sorted(missing)}")

    renamed = {}
    for raw_col, canonical_col in RAW_LSV_MAPPING.items():
        if raw_col in df.columns:
            renamed[raw_col] = canonical_col

    df = df.rename(columns=renamed)

    # Compute area-normalized current density (A/m^2)
    df["current_density_A_m2"] = df["current_A"] / (df["working_electrode_area_cm2"] * 1e-4)

    return df


def eis_from_export(path: Union[str, Path]) -> pd.DataFrame:
    """
    Adapter to convert a raw EIS export into the canonical EIS format.
    Requires at minimum Frequency_Hz, Z_real_Ohm, and Z_imag_Ohm.
    """
    df = pd.read_csv(path)
    required_raw = {"Frequency_Hz", "Z_real_Ohm", "Z_imag_Ohm"}
    missing = required_raw - set(df.columns)
    if missing:
        raise ValueError(f"Raw export is missing required columns: {sorted(missing)}")

    renamed = {}
    for raw_col, canonical_col in RAW_EIS_MAPPING.items():
        if raw_col in df.columns:
            renamed[raw_col] = canonical_col

    df = df.rename(columns=renamed)
    return df


def fit_kinetics_from_data(
    data_or_path: Union[str, Path, pd.DataFrame],
    *,
    pH: Optional[float] = None,
    temperature_C: Optional[float] = None,
    fe_conc_M: Optional[float] = None,
    reference_to_she_V: Optional[float] = None,
) -> PolarizationFit:
    """
    Thin, typed wrapper that runs fit_total_cathodic_polarization on the adapted
    frame, surfacing her_tafel_V_dec, fe_tafel_V_dec, i0s, boundary_layer_m,
    converged, and parameter_std_log10.
    Missing kwargs are inferred from DataFrame columns if present, otherwise defaults
    to 3.0, 60.0, 1.0, 0.197.
    """
    if isinstance(data_or_path, (str, Path)):
        df = polarization_from_export(data_or_path)
    else:
        df = data_or_path.copy()

    # Mirror inference logic from fit_tafel_domain
    final_pH = pH if pH is not None else (float(df["pH"].iloc[0]) if "pH" in df.columns else 3.0)
    final_T = (
        temperature_C
        if temperature_C is not None
        else (float(df["temperature_C"].iloc[0]) if "temperature_C" in df.columns else 60.0)
    )
    final_fe = (
        fe_conc_M
        if fe_conc_M is not None
        else (
            float(df["fe2_concentration_M"].iloc[0]) if "fe2_concentration_M" in df.columns else 1.0
        )
    )
    final_ref = (
        reference_to_she_V
        if reference_to_she_V is not None
        else (
            float(df["reference_to_she_V"].iloc[0]) if "reference_to_she_V" in df.columns else 0.197
        )
    )

    return fit_total_cathodic_polarization(
        df, pH=final_pH, temperature_C=final_T, fe_conc_M=final_fe, reference_to_she_V=final_ref
    )


def make_synthetic_polarization(
    kinetics: DepositionKinetics, *, potentials: np.ndarray, noise_sigma: float = 0
) -> pd.DataFrame:
    """
    Generate a round-trip fixture: evaluate DepositionKinetics.partial_currents,
    sum Fe+HER, optional Gaussian noise, and output as potential_V_vs_ref and
    current_density_A_m2 (cathodic-negative) along with metadata columns.
    """
    # Assuming reference_to_she_V = 0.0 for the synthetic frame
    ref_to_she_V = 0.0
    E_she = potentials + ref_to_she_V

    # Kinetics model outputs magnitude of cathodic current components
    _, _, i_tot = kinetics.partial_currents(E_she)

    # Canonical format expects negative values for cathodic current
    j_A_m2 = -i_tot

    if noise_sigma > 0:
        j_A_m2 += np.random.normal(0, noise_sigma, size=j_A_m2.shape)

    df = pd.DataFrame(
        {
            "potential_V_vs_ref": potentials,
            "current_density_A_m2": j_A_m2,
            "pH": kinetics.pH,
            "temperature_C": kinetics.temperature_C,
            "fe2_concentration_M": kinetics.fe_conc_M,
            "reference_to_she_V": ref_to_she_V,
        }
    )
    return df


def self_test(seed: Union[int, None] = None) -> dict:
    """
    The inversion harness:
      (a) pick known true kinetics
      (b) generate no-noise synthetic polarization
      (c) fit it
      (d) return recovered params + relative error vs known truth
      (e) run a noisy case to confirm the fit converged and stays within a wider tolerance
    """
    if seed is not None:
        np.random.seed(seed)

    true_params = {
        "fe_i0": 2.5e-2,
        "her_i0": 5.0e-4,
        "fe_tafel_V": 0.115,
        "her_tafel_V": 0.135,
        "boundary_layer_m": 4.5e-5,
    }

    kinetics = DepositionKinetics(pH=2.5, temperature_C=50.0, fe_conc_M=1.2, **true_params)

    # Span -0.5 V to -1.2 V vs REF to ensure both branches have strong signal
    potentials = np.linspace(-0.5, -1.2, 50)

    df_clean = make_synthetic_polarization(kinetics, potentials=potentials, noise_sigma=0.0)
    fit_clean = fit_kinetics_from_data(
        df_clean,
        pH=kinetics.pH,
        temperature_C=kinetics.temperature_C,
        fe_conc_M=kinetics.fe_conc_M,
        reference_to_she_V=0.0,
    )

    recovered_clean = {
        "fe_i0": fit_clean.fe_i0_A_m2,
        "her_i0": fit_clean.her_i0_A_m2,
        "fe_tafel_V": fit_clean.fe_tafel_V_dec,
        "her_tafel_V": fit_clean.her_tafel_V_dec,
        "boundary_layer_m": fit_clean.boundary_layer_m,
    }

    error_clean = {}
    for k in true_params:
        if k.endswith("i0"):
            # Exchange currents use log-error
            error_clean[k] = abs(np.log10(recovered_clean[k]) - np.log10(true_params[k]))
        else:
            # Linear relative error
            error_clean[k] = abs(recovered_clean[k] - true_params[k]) / true_params[k]

    # Noisy case: add 0.5 A/m^2 Gaussian noise
    df_noisy = make_synthetic_polarization(kinetics, potentials=potentials, noise_sigma=0.5)
    fit_noisy = fit_kinetics_from_data(
        df_noisy,
        pH=kinetics.pH,
        temperature_C=kinetics.temperature_C,
        fe_conc_M=kinetics.fe_conc_M,
        reference_to_she_V=0.0,
    )

    recovered_noisy = {
        "fe_i0": fit_noisy.fe_i0_A_m2,
        "her_i0": fit_noisy.her_i0_A_m2,
        "fe_tafel_V": fit_noisy.fe_tafel_V_dec,
        "her_tafel_V": fit_noisy.her_tafel_V_dec,
        "boundary_layer_m": fit_noisy.boundary_layer_m,
    }

    error_noisy = {}
    for k in true_params:
        if k.endswith("i0"):
            error_noisy[k] = abs(np.log10(recovered_noisy[k]) - np.log10(true_params[k]))
        else:
            error_noisy[k] = abs(recovered_noisy[k] - true_params[k]) / true_params[k]

    return {
        "true_params": true_params,
        "clean": {
            "recovered": recovered_clean,
            "error": error_clean,
            "converged": fit_clean.converged,
        },
        "noisy": {
            "recovered": recovered_noisy,
            "error": error_noisy,
            "converged": fit_noisy.converged,
        },
    }


def main():
    print("Running self-test (synthetic round-trip inversion)...")
    results = self_test(seed=42)

    def print_case(title, case_data, tol_pct, tol_log):
        print(f"\n--- {title} ---")
        print(
            f"{'Parameter':<18} {'True Value':<15} {'Recovered':<15} {'Error':<12} {'Verdict':<10}"
        )
        print("-" * 75)
        for k in results["true_params"]:
            true_val = results["true_params"][k]
            rec_val = case_data["recovered"][k]
            err = case_data["error"][k]

            if k.endswith("i0"):
                err_str = f"{err:.3f} dec"
                verdict = "PASS" if err < tol_log else "FAIL"
            else:
                err_str = f"{err * 100:.2f} %"
                verdict = "PASS" if (err * 100) < tol_pct else "FAIL"

            print(f"{k:<18} {true_val:<15.4e} {rec_val:<15.4e} {err_str:<12} {verdict:<10}")
        print(f"Converged: {case_data['converged']}")

    print_case("No-Noise Case", results["clean"], tol_pct=5.0, tol_log=0.1)
    # Relaxed tolerance for noisy case: 25% for linear, 1.0 dec for log
    print_case("Noisy Case (0.5 A/m^2 sigma)", results["noisy"], tol_pct=25.0, tol_log=1.0)


if __name__ == "__main__":
    main()
