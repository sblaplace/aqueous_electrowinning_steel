#!/usr/bin/env python3
"""
Voltammetry Analysis Driver: Synthetic CV generation + Phase I analysis and Tafel fitting.

Generates:
    experiments/data/synthetic_voltammetry.csv
    docs/figures/voltammetry_analysis.png
    docs/figures/tafel_analysis.png
    experiments/data/voltammetry_report.json

Usage:
    python -m models.run_voltammetry
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.experimental_data import load_measurements, summarize_run
from models.voltammetry import (
    segment_sweeps,
    baseline_correct,
    scan_rate_V_s,
    extrema,
)
from models.tafel import fit_tafel

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "docs" / "figures"
DATA_DIR = ROOT / "experiments" / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})


def generate_synthetic_cv(
    filepath: Path,
    scan_rate_V_s: float = 0.05,
    num_cycles: int = 2,
    R_u_ohm: float = 12.0,
    noise_level_A: float = 5e-6,
) -> pd.DataFrame:
    """Generate and save a realistic synthetic CV dataset of iron/HER electrochemistry."""
    np.random.seed(42)
    dt = 0.1  # time step, s
    E_start = -0.2
    E_vertex = -0.9
    v = scan_rate_V_s
    segment_duration = abs(E_vertex - E_start) / v
    t_segment = np.arange(0, segment_duration + dt, dt)

    # 1. Base potential sweeps
    E_forward = E_start - v * t_segment
    E_reverse = E_vertex + v * t_segment

    timestamps = []
    E_actuals = []
    cycles = []
    segments = []

    t_offset = 0.0
    for c in range(1, num_cycles + 1):
        # Forward segment (cathodic)
        timestamps.extend(t_offset + t_segment)
        E_actuals.extend(E_forward)
        cycles.extend([c] * len(E_forward))
        segments.extend(["forward"] * len(E_forward))
        t_offset += segment_duration

        # Reverse segment (anodic)
        timestamps.extend(t_offset + t_segment)
        E_actuals.extend(E_reverse)
        cycles.extend([c] * len(E_reverse))
        segments.extend(["reverse"] * len(E_reverse))
        t_offset += segment_duration

    timestamps = np.array(timestamps)
    E_actuals = np.array(E_actuals)
    cycles = np.array(cycles)
    segments = np.array(segments)

    # 2. Electrochemical parameters
    A_cm2 = 1.0
    # HER parameters (vs Ag/AgCl reference)
    E_eq_HER = -0.37
    i0_HER = 1.2e-6  # exchange current in A for 1 cm2
    b_HER = 0.125   # Tafel slope in V/dec

    # Iron parameters (vs Ag/AgCl reference)
    E_eq_Fe = -0.63
    i0_Fe = 1.8e-5  # exchange current in A for 1 cm2
    b_Fe = 0.082    # Tafel slope in V/dec
    i_lim_Fe = -0.018  # diffusion limit, A

    # Double layer capacitance
    C_dl = 40e-6  # F

    # 3. Calculate currents
    currents = []
    for E, seg in zip(E_actuals, segments):
        # HER current (cathodic, always negative)
        eta_h = E_eq_HER - E
        i_her = -i0_HER * 10.0 ** (np.clip(eta_h, 0.0, None) / b_HER)

        # Iron deposition (cathodic, active negative of E_eq_Fe)
        if E <= E_eq_Fe:
            eta_fe = E_eq_Fe - E
            i_fe_kin = -i0_Fe * 10.0 ** (eta_fe / b_Fe)
            i_fe = 1.0 / (1.0 / i_fe_kin + 1.0 / i_lim_Fe)
        else:
            i_fe = 0.0

        # Capacitive current
        dE_dt = -v if seg == "forward" else v
        i_cap = C_dl * dE_dt

        i_tot = i_her + i_fe + i_cap
        currents.append(i_tot)

    currents = np.array(currents)

    # 4. Apply uncompensated Ohmic resistance (iR drop)
    # E_applied = E_actual + iR
    E_applied = E_actuals + currents * R_u_ohm

    # 5. Add random experimental measurement noise
    noise = np.random.normal(0, noise_level_A, len(currents))
    currents_noisy = currents + noise

    # 6. Build the DataFrame
    df = pd.DataFrame({
        "timestamp_s": np.round(timestamps, 4),
        "potential_V_vs_ref": E_applied,
        "current_A": currents_noisy,
        "working_electrode_area_cm2": A_cm2,
        "cycle": cycles,
        "segment": segments,
        "temperature_C": 60.0,
        "pH": 3.0,
        "fe2_concentration_M": 1.0,
        "electrolyte_id": "FE-SO4-SYNTHETIC",
        "reference_electrode": "Ag/AgCl",
        "notes": f"Synthetic CV with Fe/HER, Ru={R_u_ohm} ohm, C_dl={C_dl*1e6} uF",
    })

    df.to_csv(filepath, index=False)
    return df


def main() -> None:
    print("=" * 70)
    print("VOLTAMMETRY ANALYSIS — SYNTHETIC DATA GENERATION & Tafel FITTING")
    print("=" * 70)

    # 1. Generate Synthetic CV
    csv_path = DATA_DIR / "synthetic_voltammetry.csv"
    print(f"\nGenerating synthetic voltammetry data to: {csv_path}...")
    raw_df = generate_synthetic_cv(csv_path, scan_rate_V_s=0.05, num_cycles=2)
    print(f"  Generated {len(raw_df)} data points across 2 CV cycles.")

    # 2. Load and validate using models.experimental_data
    print("\nLoading and validating via experimental_data module...")
    df = load_measurements(csv_path)
    summary = summarize_run(df)
    print(f"  - n_points: {summary['n_points']}")
    print(f"  - duration_s: {summary['duration_s']:.1f} s")
    print(f"  - potential range: {summary['potential_min_V']:.3f} to {summary['potential_max_V']:.3f} V vs Ag/AgCl")
    print(f"  - mean current density: {summary['current_density_mean_mA_cm2']:.2f} mA/cm²")

    # Estimate scan rate
    sr = scan_rate_V_s(df)
    print(f"  ✅ Estimated Scan Rate: {sr*1000:.1f} mV/s (Expected: 50.0 mV/s)")

    # 3. Separate sweeps and perform baseline correction
    print("\nSegmenting sweeps and applying baseline correction...")
    sweeps = segment_sweeps(df)
    
    # Cycle 2 forward is index 2 (Cycle 1 FWD=0, C1 REV=1, C2 FWD=2, C2 REV=3)
    c2_forward = sweeps[2]
    
    # Automatically find the current at Fe onset (-0.63 V vs Ag/AgCl)
    idx_onset = (c2_forward["potential_V_vs_ref"] - (-0.63)).abs().idxmin()
    onset_current_A = c2_forward.loc[idx_onset]["current_A"]
    print(f"  Detected baseline onset current at -0.63 V: {onset_current_A*1e3:.3f} mA")
    
    c2_forward_corrected = baseline_correct(c2_forward, baseline_current_A=onset_current_A)

    # 4. Perform Tafel Fitting
    print("\nPerforming Tafel fitting...")
    # A. Fit HER in the low-overpotential region (where iron hasn't deposited yet)
    # E_eq_HER is -0.37 V vs Ag/AgCl. Let's fit between -0.55 V and -0.45 V.
    her_fit = fit_tafel(
        c2_forward,
        potential_min_V=-0.55,
        potential_max_V=-0.45,
        equilibrium_potential_V=-0.37,
        current_column="current_A",
    )
    print(f"  ✅ HER Tafel Fit (-0.55 to -0.45 V vs Ag/AgCl):")
    print(f"    - Tafel Slope: {her_fit.slope_V_decade*1000:.1f} mV/decade")
    print(f"    - Extrapolated Exchange Current: {her_fit.exchange_current_A*1e6:.3f} µA")
    print(f"    - R²: {her_fit.r_squared:.5f}")

    # B. Fit Iron deposition in the kinetic region (after baseline correction for HER background)
    # E_eq_Fe is -0.63 V vs Ag/AgCl. Let's fit between -0.73 V and -0.67 V.
    fe_fit = fit_tafel(
        c2_forward_corrected,
        potential_min_V=-0.73,
        potential_max_V=-0.67,
        equilibrium_potential_V=-0.63,
        current_column="current_corrected_A",
    )
    print(f"  ✅ Fe Tafel Fit (-0.73 to -0.67 V vs Ag/AgCl, corrected):")
    print(f"    - Tafel Slope: {fe_fit.slope_V_decade*1000:.1f} mV/decade")
    print(f"    - Extrapolated Exchange Current: {fe_fit.exchange_current_A*1e6:.3f} µA")
    print(f"    - R²: {fe_fit.r_squared:.5f}")

    # Extrema analysis
    ext = extrema(c2_forward_corrected)
    print(f"  ✅ Peak Extrema (Cycle 2 Forward):")
    print(f"    - Cathodic Peak Potential: {ext['cathodic_peak_potential_V']:.3f} V vs Ag/AgCl")
    print(f"    - Cathodic Peak Current: {ext['cathodic_peak_current_A']*1000:.2f} mA")

    # 5. Generate Figures
    print("\nGenerating Voltammetry and Tafel figures...")
    
    # Figure 1: Voltammetry Analysis
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Left: Complete Cyclic Voltammetry
    ax = axes[0]
    for i, sweep in enumerate(sweeps):
        cyc = sweep["cycle"].iloc[0]
        seg = sweep["segment"].iloc[0]
        label = f"Cycle {cyc} {seg}" if i < 4 else None
        color = "blue" if cyc == 1 else "red"
        style = "-" if seg == "forward" else "--"
        ax.plot(sweep["potential_V_vs_ref"], sweep["current_density_mA_cm2"], 
                color=color, linestyle=style, label=label, alpha=0.8)
    
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.axvline(-0.63, color="green", lw=1.0, ls="--", label="Fe/Fe²⁺ E$_{eq}$")
    ax.axvline(-0.37, color="orange", lw=1.0, ls="--", label="HER E$_{eq}$")
    ax.set_xlabel("Potential (V vs Ag/AgCl)")
    ax.set_ylabel("Current Density (mA/cm²)")
    ax.set_title("Full Cyclic Voltammetry (2 Cycles, 50 mV/s)", fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    # Right: Baseline subtraction visualization
    ax = axes[1]
    c2f = c2_forward.copy()
    
    ax.plot(c2f["potential_V_vs_ref"], c2f["current_A"] * 1000.0, "k-", label="Total Measured Current")
    ax.axhline(onset_current_A * 1000.0, color="r", lw=1.2, ls="--", label="Onset HER Baseline")
    ax.plot(c2_forward_corrected["potential_V_vs_ref"], c2_forward_corrected["current_corrected_A"] * 1000.0, 
            "g-.", label="Corrected Fe Current")
    
    ax.set_xlabel("Potential (V vs Ag/AgCl)")
    ax.set_ylabel("Current (mA)")
    ax.set_title("HER Baseline Subtraction (Cycle 2 Forward)", fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1_path = FIG_DIR / "voltammetry_analysis.png"
    plt.savefig(fig1_path, dpi=180)
    plt.close()
    print(f"  ✅ Saved: {fig1_path}")

    # Figure 2: Tafel Analysis
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Left: HER Tafel Plot
    ax = axes[0]
    # Extract data in HER fitting region
    her_sel = c2_forward[c2_forward["potential_V_vs_ref"].between(-0.55, -0.45)]
    eta_h = -0.37 - her_sel["potential_V_vs_ref"].to_numpy()
    log_i_h = np.log10(np.abs(her_sel["current_A"].to_numpy()))
    
    # Fitted line
    eta_fit_h = np.linspace(0.08, 0.20, 100)
    log_i_fit_h = (1.0 / her_fit.slope_V_decade) * eta_fit_h + her_fit.intercept_log10_A
    
    ax.scatter(eta_h, log_i_h, color="orange", facecolors="none", edgecolors="orange", label="Synthetic Data")
    ax.plot(eta_fit_h, log_i_fit_h, "r-", label="Tafel Fit")
    ax.set_xlabel("HER Overpotential η$_H$ (V)")
    ax.set_ylabel("log$_{10}$(|Current (A)|)")
    ax.set_title("HER Tafel Plot & Fit (Slope = {:.1f} mV/dec)".format(her_fit.slope_V_decade * 1000.0), fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.12, log_i_h.min() + 0.2, f"i$_0$ = {her_fit.exchange_current_A*1e6:.2f} µA\nR² = {her_fit.r_squared:.5f}", 
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=9)

    # Right: Fe Tafel Plot
    ax = axes[1]
    # Extract data in Fe fitting region
    fe_sel = c2_forward_corrected[c2_forward_corrected["potential_V_vs_ref"].between(-0.73, -0.67)]
    eta_fe = -0.63 - fe_sel["potential_V_vs_ref"].to_numpy()
    log_i_fe = np.log10(np.abs(fe_sel["current_corrected_A"].to_numpy()))
    
    # Fitted line
    eta_fit_fe = np.linspace(0.03, 0.12, 100)
    log_i_fit_fe = (1.0 / fe_fit.slope_V_decade) * eta_fit_fe + fe_fit.intercept_log10_A
    
    ax.scatter(eta_fe, log_i_fe, color="green", facecolors="none", edgecolors="green", label="Baseline-Corrected Fe")
    ax.plot(eta_fit_fe, log_i_fit_fe, "r-", label="Tafel Fit")
    ax.set_xlabel("Fe Overpotential η$_{Fe}$ (V)")
    ax.set_ylabel("log$_{10}$(|Corrected Current (A)|)")
    ax.set_title("Fe Tafel Plot & Fit (Slope = {:.1f} mV/dec)".format(fe_fit.slope_V_decade * 1000.0), fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.06, log_i_fe.min() + 0.2, f"i$_0$ = {fe_fit.exchange_current_A*1e6:.2f} µA\nR² = {fe_fit.r_squared:.5f}", 
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=9)

    plt.tight_layout()
    fig2_path = FIG_DIR / "tafel_analysis.png"
    plt.savefig(fig2_path, dpi=180)
    plt.close()
    print(f"  ✅ Saved: {fig2_path}")

    # 6. Save Voltammetry Report
    report = {
        "dataset_summary": summary,
        "derived_properties": {
            "estimated_scan_rate_V_s": sr,
        },
        "extrema_analysis": {
            "cathodic_peak_potential_V_vs_ref": ext["cathodic_peak_potential_V"],
            "cathodic_peak_current_A": ext["cathodic_peak_current_A"],
            "anodic_peak_potential_V_vs_ref": ext["anodic_peak_potential_V"],
            "anodic_peak_current_A": ext["anodic_peak_current_A"],
        },
        "tafel_fits": {
            "her": {
                "slope_V_decade": her_fit.slope_V_decade,
                "intercept_log10_A": her_fit.intercept_log10_A,
                "exchange_current_A": her_fit.exchange_current_A,
                "r_squared": her_fit.r_squared,
                "n_points": her_fit.n_points,
                "potential_min_V": her_fit.potential_min_V,
                "potential_max_V": her_fit.potential_max_V,
            },
            "iron": {
                "slope_V_decade": fe_fit.slope_V_decade,
                "intercept_log10_A": fe_fit.intercept_log10_A,
                "exchange_current_A": fe_fit.exchange_current_A,
                "r_squared": fe_fit.r_squared,
                "n_points": fe_fit.n_points,
                "potential_min_V": fe_fit.potential_min_V,
                "potential_max_V": fe_fit.potential_max_V,
            }
        }
    }

    report_path = DATA_DIR / "voltammetry_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"  ✅ Saved: {report_path}")
    print("\n✅ Voltammetry analysis driver complete!")


if __name__ == "__main__":
    main()
