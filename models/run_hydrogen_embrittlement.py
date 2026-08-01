"""
Driver for hydrogen embrittlement screening — models H uptake during
electrodeposition, trap-modified diffusion, HE susceptibility, bake-out
protocol, and integration with mechanical/carburization outputs.

Generates:
* docs/figures/he_h_uptake.png           — H uptake vs j, T, pH
* docs/figures/he_diffusivity.png        — D_eff vs grain size, temperature
* docs/figures/he_index_sweep.png        — I_HE vs strength and H content
* docs/figures/he_bakeout.png            — bake-out time vs temperature/thickness
* experiments/data/hydrogen_embrittlement_report.json
* experiments/data/synthetic_h_uptake.csv

Usage:
    python -m models.run_hydrogen_embrittlement
    python -m models.run_hydrogen_embrittlement --current-density 150 --pH 3.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.hydrogen_embrittlement import (
    HydrogenEmbrittlementModel,
    effective_diffusivity_m2_s,
    he_susceptibility_index,
    bakeout_time_hr,
    synthetic_h_uptake_data,
)
from models.mechanical_properties import build_mechanical_model_from_phase3_result
from models.co_deposition import build_phase3_model

FIG_DIR = ROOT / "docs" / "figures"
DATA_DIR = ROOT / "experiments" / "data"


def main(argv=None, current_density=None, pH=None, thickness=None, bakeout_temp=None):
    """
    Main driver — accepts either CLI args (argv) or direct keyword overrides.
    """
    if current_density is None and argv is None:
        parser = argparse.ArgumentParser(description="Hydrogen embrittlement screening driver")
        parser.add_argument("--current-density", type=float, default=100.0, help="j (mA/cm²)")
        parser.add_argument("--pH", type=float, default=3.5, help="Bath pH")
        parser.add_argument("--thickness", type=float, default=1000.0, help="Deposit thickness (µm)")
        parser.add_argument("--bakeout-temp", type=float, default=170.0, help="Bake-out T (°C)")
        parsed = parser.parse_args(argv)
        args = parsed
    else:
        class _NS: pass
        args = _NS()
        args.current_density = current_density if current_density is not None else 100.0
        args.pH = pH if pH is not None else 3.5
        args.thickness = thickness if thickness is not None else 1000.0
        args.bakeout_temp = bakeout_temp if bakeout_temp is not None else 170.0
        if argv is not None:
            parser = argparse.ArgumentParser(description="Hydrogen embrittlement screening driver")
            parser.add_argument("--current-density", type=float, default=args.current_density)
            parser.add_argument("--pH", type=float, default=args.pH)
            parser.add_argument("--thickness", type=float, default=args.thickness)
            parser.add_argument("--bakeout-temp", type=float, default=args.bakeout_temp)
            parsed = parser.parse_args(argv)
            args = parsed

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("HYDROGEN EMBRITTLEMENT — SCREENING MODEL")
    print("=" * 72)

    model = HydrogenEmbrittlementModel()

    # ── Main prediction ──────────────────────────────────────────────────
    result = model.predict(
        current_density_mA_cm2=args.current_density,
        deposition_time_hr=2.0,
        bath_pH=args.pH,
        sigma_y_MPa=450.0,
        grain_size_um=2.0,
        carbon_wt_percent=0.5,
        deposit_thickness_um=args.thickness,
        bakeout_temperature_C=args.bakeout_temp,
    )
    s = result.summary()
    print("\nBaseline prediction:")
    for k, v in s.items():
        if k != "flags":
            print(f"  {k}: {v}")
    print(f"  flags: {s['flags']}")

    # ── Synthetic H uptake data ──────────────────────────────────────────
    syn = synthetic_h_uptake_data(deposition_time_hr=2.0)

    # ── Integration with mechanical properties ───────────────────────────
    print("\nIntegration with mechanical properties model:")
    p3 = build_phase3_model(
        bath_fe_M=0.6, bath_ni_M=0.4, pH=3.5, temperature_C=60.0,
        carbon_particle_loading_g_L=2.5, mechanism_fe_ni="mixed_metal_intermediate",
    )
    p3_at_100 = p3.run_at_current(100.0)
    mech_result = build_mechanical_model_from_phase3_result(
        p3_at_100, j_avg_mA_cm2=100, j_peak_mA_cm2=200, duty_cycle=0.5,
        waveform="pe", temperature_C=60.0,
    )
    mech_summary = mech_result.summary()

    he_integrated = model.predict_with_integration(
        mechanical_result=mech_summary,
        current_density_mA_cm2=args.current_density,
        deposition_time_hr=2.0,
        bath_pH=args.pH,
        deposit_thickness_um=args.thickness,
        bakeout_temperature_C=args.bakeout_temp,
    )
    isum = he_integrated.summary()
    print(f"  YS={isum['sigma_y_MPa']:.0f} MPa, C_H={isum['C_H_diffusible_ppm']:.3f} ppm, "
          f"I_HE={isum['I_HE']:.2f}, risk={isum['risk_level']}")

    # ── Save CSV ─────────────────────────────────────────────────────────
    rows = []
    for j, H in zip(syn["j_mA_cm2"], syn["H_vs_j_ppm"]):
        rows.append({"sweep": "current_density", "x_var": float(j), "C_H_ppm": float(H)})
    for T, H in zip(syn["T_C"], syn["H_vs_T_ppm"]):
        rows.append({"sweep": "temperature", "x_var": float(T), "C_H_ppm": float(H)})
    for pH, H in zip(syn["pH"], syn["H_vs_pH_ppm"]):
        rows.append({"sweep": "pH", "x_var": float(pH), "C_H_ppm": float(H)})

    df = pd.DataFrame(rows)
    csv_path = DATA_DIR / "synthetic_h_uptake.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  ✅ Saved CSV: {csv_path} ({len(df)} rows)")

    # ── Figure 1: H uptake vs j, T, pH ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(syn["j_mA_cm2"], syn["H_vs_j_ppm"], color="#e41a1c", marker="o", ms=3)
    ax.set(xlabel="Current density (mA/cm²)", ylabel="C_H diffusible (ppm)",
           title="H uptake vs j")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(syn["T_C"], syn["H_vs_T_ppm"], color="#1874b4", marker="s", ms=3)
    ax.set(xlabel="Bath temperature (°C)", ylabel="C_H diffusible (ppm)",
           title="H uptake vs temperature")
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(syn["pH"], syn["H_vs_pH_ppm"], color="#4daf4a", marker="^", ms=3)
    ax.set(xlabel="Bath pH", ylabel="C_H diffusible (ppm)",
           title="H uptake vs pH")
    ax.grid(alpha=0.25)

    fig.suptitle("Hydrogen uptake from electrodeposition (screening)", fontweight="bold")
    fig.tight_layout()
    uptake_fig = FIG_DIR / "he_h_uptake.png"
    fig.savefig(uptake_fig, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {uptake_fig}")

    # ── Figure 2: D_eff vs grain size and temperature ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # D_eff vs grain size
    gs_range = np.logspace(-1, 1, 40)
    d_eff_gs = []
    d_lat_gs = []
    for gs in gs_range:
        de, dl, _ = effective_diffusivity_m2_s(25.0, grain_size_um=float(gs), carbon_wt_percent=0.5)
        d_eff_gs.append(de)
        d_lat_gs.append(dl)

    ax = axes[0]
    ax.loglog(gs_range, d_lat_gs, color="#777777", ls="--", label="D_lattice")
    ax.loglog(gs_range, d_eff_gs, color="#e41a1c", label="D_eff (traps)")
    ax.set(xlabel="Grain size (µm)", ylabel="D (m²/s)", title="Effective H diffusivity vs grain size")
    ax.grid(alpha=0.25, which="both")
    ax.legend()

    # D_eff vs temperature
    T_range = np.linspace(0, 100, 50)
    d_eff_T = []
    d_lat_T = []
    for T in T_range:
        de, dl, _ = effective_diffusivity_m2_s(float(T), grain_size_um=2.0, carbon_wt_percent=0.5)
        d_eff_T.append(de)
        d_lat_T.append(dl)

    ax = axes[1]
    ax.semilogy(T_range, d_lat_T, color="#777777", ls="--", label="D_lattice")
    ax.semilogy(T_range, d_eff_T, color="#1874b4", label="D_eff (traps)")
    ax.set(xlabel="Temperature (°C)", ylabel="D (m²/s)", title="H diffusivity vs temperature")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.suptitle("H diffusion in electrodeposited Fe (screening)", fontweight="bold")
    fig.tight_layout()
    diff_fig = FIG_DIR / "he_diffusivity.png"
    fig.savefig(diff_fig, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {diff_fig}")

    # ── Figure 3: HE index sweep — strength × H content ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # I_HE vs yield strength
    sigma_range = np.linspace(200, 1200, 40)
    he_vs_sigma_005 = [he_susceptibility_index(s, 0.05)["I_HE"] for s in sigma_range]
    he_vs_sigma_01 = [he_susceptibility_index(s, 0.1)["I_HE"] for s in sigma_range]
    he_vs_sigma_1 = [he_susceptibility_index(s, 1.0)["I_HE"] for s in sigma_range]
    he_vs_sigma_5 = [he_susceptibility_index(s, 5.0)["I_HE"] for s in sigma_range]

    ax = axes[0]
    ax.semilogy(sigma_range, he_vs_sigma_005, color="#4daf4a", label="0.05 ppm H")
    ax.semilogy(sigma_range, he_vs_sigma_01, color="#377eb8", label="0.1 ppm H")
    ax.semilogy(sigma_range, he_vs_sigma_1, color="#e41a1c", label="1.0 ppm H")
    ax.semilogy(sigma_range, he_vs_sigma_5, color="#984ea3", label="5.0 ppm H")
    ax.axhline(5.0, color="#777777", ls=":", alpha=0.5)
    ax.axhline(20.0, color="#333333", ls=":", alpha=0.5)
    ax.set(xlabel="Yield strength (MPa)", ylabel="I_HE", title="HE index vs strength")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    # I_HE vs H content
    H_range = np.logspace(-2, 1, 40)
    he_vs_H_low = [he_susceptibility_index(300, h)["I_HE"] for h in H_range]
    he_vs_H_mid = [he_susceptibility_index(600, h)["I_HE"] for h in H_range]
    he_vs_H_high = [he_susceptibility_index(1000, h)["I_HE"] for h in H_range]

    ax = axes[1]
    ax.loglog(H_range, he_vs_H_low, color="#4daf4a", label="σ=300 MPa")
    ax.loglog(H_range, he_vs_H_mid, color="#e41a1c", label="σ=600 MPa")
    ax.loglog(H_range, he_vs_H_high, color="#984ea3", label="σ=1000 MPa")
    ax.axvline(0.1, color="#777777", ls=":", alpha=0.5, label="0.1 ppm critical")
    ax.set(xlabel="Diffusible H (ppm)", ylabel="I_HE", title="HE index vs H content")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)

    fig.suptitle("Hydrogen embrittlement susceptibility index (Troiano-type)", fontweight="bold")
    fig.tight_layout()
    index_fig = FIG_DIR / "he_index_sweep.png"
    fig.savefig(index_fig, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {index_fig}")

    # ── Figure 4: Bake-out time vs temperature and thickness ─────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Bake-out schedule for different thicknesses
    ax = axes[0]
    thicknesses = [500, 1000, 2000, 3000]
    T_bake = np.linspace(100, 300, 30)
    for L in thicknesses:
        times = []
        for T in T_bake:
            res = bakeout_time_hr(
                deposit_thickness_um=L, initial_C_H_ppm=0.5,
                target_C_H_ppm=0.1, temperature_C=float(T),
            )
            t = res["bakeout_time_hr"]
            times.append(t if t < 1000 else np.nan)
        ax.semilogy(T_bake, times, label=f"L={L} µm")

    ax.set(xlabel="Bake-out temperature (°C)", ylabel="Time (hr)",
           title="Bake-out time vs temperature (C₀=0.5 ppm → 0.1 ppm)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    ax.set_ylim(0.01, 1000)

    # Bake-out time vs initial H content
    ax = axes[1]
    H_init_range = np.logspace(-1, 1, 30)
    bake_temps = [120, 170, 200, 250]
    for T in bake_temps:
        times = []
        for H0 in H_init_range:
            res = bakeout_time_hr(
                deposit_thickness_um=1000, initial_C_H_ppm=float(H0),
                target_C_H_ppm=0.1, temperature_C=T,
            )
            times.append(res["bakeout_time_hr"])
        ax.loglog(H_init_range, times, label=f"{T}°C")

    ax.set(xlabel="Initial H content (ppm)", ylabel="Bake-out time (hr)",
           title="Bake-out vs initial H (L=1000 µm)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)

    fig.suptitle("Bake-out protocol optimizer (screening)", fontweight="bold")
    fig.tight_layout()
    bakeout_fig = FIG_DIR / "he_bakeout.png"
    fig.savefig(bakeout_fig, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {bakeout_fig}")

    # ── JSON Report ──────────────────────────────────────────────────────
    report = {
        "title": "Hydrogen embrittlement screening model",
        "date": __import__("datetime").datetime.now().isoformat(),
        "model": "Troiano-type HE index + Fickian H diffusion with trap retardation + Faraday H uptake + bake-out optimizer",
        "baseline": result.summary(),
        "integration_with_mechanical": {
            "mechanical_summary": mech_summary,
            "he_result": isum,
            "spatial_risk": he_integrated.spatial_he_risk,
        },
        "parameters": {
            "D0_alpha_m2_s": 7.3e-8,
            "Q_alpha_kJ_mol": 4.6,
            "D0_gamma_m2_s": 5.7e-7,
            "Q_gamma_kJ_mol": 40.0,
            "trap_binding_dislocation_kJ_mol": 26.0,
            "trap_binding_gb_kJ_mol": 20.0,
            "trap_binding_carbide_kJ_mol": 11.0,
            "N_lattice_m3": 8.46e28,
            "critical_H_ppm": 0.1,
            "sigma_critical_MPa": 800,
        },
        "figures": [str(uptake_fig), str(diff_fig), str(index_fig), str(bakeout_fig)],
        "csv": str(csv_path),
        "model_notes": [
            "H diffusion: D = D0 exp(-Q/RT), α-Fe D0=7.3e-8 m²/s Q=4.6 kJ/mol, γ-Fe D0=5.7e-7 Q=40 kJ/mol",
            "Trap model: D_eff = D_lattice / (1 + Σ N_t K_t / N_L), K_t = exp(E_trap/RT)",
            "Trap types: dislocations (26 kJ/mol), grain boundaries (20 kJ/mol), carbides (11 kJ/mol)",
            "H uptake: Faraday law with HER efficiency + pH and temperature corrections",
            "HE index: I_HE = (σ/σ_ref)^γσ × (C_H/C_H_ref)^γH × (T_ref/T)^γT (Troiano-type)",
            "Bake-out: Fourier slab desorption with D_eff, leading-term solver + verification",
            "Integration: accepts MechanicalPropertiesResult + CarburizationResult for spatial HE risk",
            "All coefficients are screening assumptions; calibrate with thermal desorption (TDS) and slow strain rate tests",
        ],
        "calibration_note": (
            "All coefficients are screening literature values. "
            "Requires calibration with: (1) thermal desorption spectroscopy for trap energies, "
            "(2) permeation tests for D_eff, (3) slow strain rate testing (SSRT) for I_HE validation, "
            "(4) ASTM F519 for bake-out verification."
        ),
        "limits": [
            "No H-induced crack propagation model (threshold stress intensity K_IH)",
            "No HIC / SSC / SOHIC failure mode distinction",
            "Trap occupancy not dynamically modeled (Oriani equilibrium assumed)",
            "No electrochemical H entry potential (Volmer-Heyrovsky kinetics simplified)",
            "Spatial resolution limited to surface / case / core zones",
            "Bake-out assumes uniform initial H distribution (not profile-aware)",
        ],
    }
    report_path = DATA_DIR / "hydrogen_embrittlement_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  ✅ Saved report: {report_path}")
    print("\n✅ Hydrogen embrittlement driver complete!")

    return report


if __name__ == "__main__":
    main()
