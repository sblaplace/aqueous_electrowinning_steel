#!/usr/bin/env python3
"""
EIS Analysis Driver: Synthetic impedance spectrum generation + Randles fitting.

Generates:
    experiments/data/synthetic_eis.csv
    docs/figures/eis_nyquist.png
    docs/figures/eis_bode.png
    experiments/data/eis_report.json

Usage:
    python -m models.run_eis
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.eis import (
    load_spectrum,
    summarize_spectrum,
    randles_impedance,
    fit_randles_spectrum,
    exchange_current_from_rct,
    synthetic_randles_spectrum,
    nyquist_plot,
    bode_plot,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "docs" / "figures"
DATA_DIR = ROOT / "experiments" / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})

# Physical parameters for the synthetic Fe-deposition cathode
# (1 cm² electrode, 1 M FeSO₄, pH 3, 60 °C, biased at -0.70 V vs Ag/AgCl)
RS_OHM = 8.0          # electrolyte + leads
RCT_OHM = 12.0        # combined Fe deposition + HER charge-transfer resistance
CDL_F = 50e-6         # double-layer capacitance
SIGMA_WAR = 3.0       # Warburg coefficient, ohm s^-1/2 (Fe2+ diffusion)
FREQ_MIN_HZ = 0.01
FREQ_MAX_HZ = 1.0e5
AREA_CM2 = 1.0
TEMPERATURE_K = 333.15


def main() -> None:
    print("=" * 70)
    print("EIS ANALYSIS — SYNTHETIC SPECTRUM GENERATION & RANDLES FITTING")
    print("=" * 70)

    # 1. Generate the synthetic spectrum (relative-noise corrupted)
    csv_path = DATA_DIR / "synthetic_eis.csv"
    print(f"\nGenerating synthetic EIS data to: {csv_path}...")
    spectrum = synthetic_randles_spectrum(
        RS_OHM, RCT_OHM, CDL_F, SIGMA_WAR,
        FREQ_MIN_HZ, FREQ_MAX_HZ, points_per_decade=12,
        noise_rel=0.005, area_cm2=AREA_CM2, dc_bias_V_vs_ref=-0.70,
    )
    spectrum.to_csv(csv_path, index=False)
    print(f"  Generated {len(spectrum)} frequency points "
          f"({FREQ_MAX_HZ:.0e} -> {FREQ_MIN_HZ:.0e} Hz) with 0.5% relative noise.")

    # 2. Load and validate
    print("\nLoading and validating via eis module...")
    data = load_spectrum(csv_path)
    summary = summarize_spectrum(data)
    print(f"  - n_points: {summary['n_points']} over {summary['decades']:.1f} decades")
    print(f"  - high-frequency intercept: {summary['high_freq_real_ohm']:.2f} Ω")
    print(f"  - semicircle top: {summary['semicircle_top_freq_hz']:.1f} Hz")

    freq = data["frequency_hz"].to_numpy(float)
    z = data["z_real_ohm"].to_numpy(float) + 1j * data["z_imag_ohm"].to_numpy(float)

    # 3. Fit Randles with and without the Warburg diffusion element
    print("\nFitting Randles equivalent circuits...")
    fits = {}
    for tag, warburg in (("randles", False), ("randles_warburg", True)):
        fit = fit_randles_spectrum(freq, z, include_warburg=warburg)
        fits[tag] = fit
        label = "Randles + Warburg" if warburg else "Randles (no Warburg)"
        print(f"  ✅ {label}:")
        print(f"    - Rs = {fit.rs_ohm:.3f} Ω (true {RS_OHM})")
        print(f"    - Rct = {fit.rct_ohm:.3f} Ω (true {RCT_OHM})")
        print(f"    - Cdl = {fit.cdl_F*1e6:.2f} µF (true {CDL_F*1e6:.0f})")
        if warburg:
            print(f"    - σ_W = {fit.sigma_warburg_ohm_s_neg_half:.3f} Ω·s^(-1/2) "
                  f"(true {SIGMA_WAR})")
        print(f"    - χ² = {fit.chi_squared:.4f}, R²(|Z|) = {fit.r_squared_magnitude:.6f}")

    improvement = fits["randles"].chi_squared / fits["randles_warburg"].chi_squared
    print(f"  ✅ Adding the Warburg element reduces χ² by a factor of {improvement:.1f}"
          " — diffusion control at low frequency is real, not noise.")

    # 4. Derive exchange current from Rct (n = 2 for Fe2+ + 2e- -> Fe)
    best = fits["randles_warburg"]
    i0_A = exchange_current_from_rct(best.rct_ohm,
                                     n_electrons=2, temperature_K=TEMPERATURE_K)
    print("\nExchange current from Rct (linearized Butler–Volmer, 60 °C):")
    print(f"  ✅ i₀ = {i0_A*1e3:.3f} mA for {AREA_CM2} cm² "
          f"({i0_A*1e3/AREA_CM2:.3f} mA/cm² combined Fe + HER)")

    # 5. Figures
    print("\nGenerating Nyquist and Bode figures...")
    omega = 2.0 * np.pi * freq

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    nyquist_plot(z, ax=axes[0])
    axes[0].set_title("Measured spectrum (synthetic)\n"
                      "-0.70 V vs Ag/AgCl, 1 M FeSO₄, pH 3, 60 °C", fontweight="bold")
    z_model_no_w = randles_impedance(omega, fits["randles"].rs_ohm,
                                     fits["randles"].rct_ohm, fits["randles"].cdl_F)
    nyquist_plot(z, z_model_no_w, ax=axes[1])
    z_model_w = randles_impedance(omega, best.rs_ohm, best.rct_ohm, best.cdl_F,
                                  best.sigma_warburg_ohm_s_neg_half)
    axes[1].plot(z_model_w.real, -z_model_w.imag, "-", linewidth=1.5,
                 label="Randles + Warburg fit")
    axes[1].set_title("Equivalent-circuit fits", fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[0].relim()
    axes[0].autoscale_view()
    plt.tight_layout()
    nyquist_path = FIG_DIR / "eis_nyquist.png"
    plt.savefig(nyquist_path, dpi=180)
    plt.close()
    print(f"  ✅ Saved: {nyquist_path}")

    fig, axes = plt.subplots(2, 1, figsize=(8, 8.5), sharex=True)
    bode_plot(freq, z, z_model_w, axes=axes)
    axes[0].set_title("Bode plot with Randles + Warburg fit", fontweight="bold")
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    bode_path = FIG_DIR / "eis_bode.png"
    plt.savefig(bode_path, dpi=180)
    plt.close()
    print(f"  ✅ Saved: {bode_path}")

    # 6. Report
    report = {
        "dataset_summary": summary,
        "true_parameters": {
            "rs_ohm": RS_OHM, "rct_ohm": RCT_OHM, "cdl_F": CDL_F,
            "sigma_warburg_ohm_s_neg_half": SIGMA_WAR,
        },
        "fits": {
            tag: {
                "rs_ohm": f.rs_ohm, "rct_ohm": f.rct_ohm, "cdl_F": f.cdl_F,
                "sigma_warburg_ohm_s_neg_half": f.sigma_warburg_ohm_s_neg_half,
                "chi_squared": f.chi_squared,
                "r_squared_magnitude": f.r_squared_magnitude,
                "n_points": f.n_points, "converged": f.converged,
            } for tag, f in fits.items()
        },
        "warburg_chi_squared_improvement_factor": improvement,
        "derived_properties": {
            "exchange_current_A_from_rct": i0_A,
            "exchange_current_density_mA_cm2": i0_A / AREA_CM2 * 1e3,
            "n_electrons": 2,
            "temperature_K": TEMPERATURE_K,
            "dc_bias_V_vs_Ag_AgCl": -0.70,
        },
    }
    report_path = DATA_DIR / "eis_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"  ✅ Saved: {report_path}")
    print("\n✅ EIS analysis driver complete!")


if __name__ == "__main__":
    main()
