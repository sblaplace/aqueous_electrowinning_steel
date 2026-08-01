#!/usr/bin/env python3
"""Phase III driver: co-deposition screening — anomalous Fe–Ni + Guglielmi carbon.

This driver produces clearly labelled synthetic examples; it does not
represent wet-lab deposition, microscopy, or verified analytical data.

Generates:
    experiments/data/synthetic_co_deposition.csv
    experiments/data/co_deposition_report.json
    docs/figures/co_deposition_anomalous_kinetics.png
    docs/figures/co_deposition_guglielmi_incorporation.png
    docs/figures/co_deposition_phase3_combined.png

Usage:
    python -m models.run_co_deposition
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.co_deposition import (
    PhaseIIICoDeposition,
    build_phase3_model,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "docs" / "figures"
DATA_DIR = ROOT / "experiments" / "data"

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})


# -------------------------------------------------------------------
# Synthetic data generation
# -------------------------------------------------------------------

def generate_synthetic_co_deposition_data(
    filepath: Path,
    mechanism_fe_ni: str = "hydroxide_suppression",
) -> pd.DataFrame:
    """Generate a synthetic Phase III co-deposition screening dataset."""
    model = build_phase3_model(
        bath_fe_M=0.6,
        bath_ni_M=0.4,
        pH=3.5,
        temperature_C=60.0,
        carbon_particle_loading_g_L=2.5,
        mechanism_fe_ni=mechanism_fe_ni,
        particle_size_um=1.2,
        zeta_potential_mV=-20.0,
        agitation_flow_rate_L_min=3.0,
    )
    j_values = np.linspace(10.0, 300.0, 25)
    rows = []
    for j in j_values:
        res = model.run_at_current(float(j))
        rows.append({
            "run_id": f"SYNTHETIC-PHASE3-{mechanism_fe_ni}-001",
            "current_density_mA_cm2": float(j),
            "potential_V_vs_SHE": res["alloy_kinetics"]["E_op_V_vs_SHE"],
            "pH_bulk": 3.5,
            "temperature_C": 60.0,
            "bath_fe_M": 0.6,
            "bath_ni_M": 0.4,
            "carbon_loading_g_L": 2.5,
            "fe_wt_percent_predicted": res["alloy_kinetics"]["fe_wt_percent"],
            "ni_wt_percent_predicted": res["alloy_kinetics"]["ni_wt_percent"],
            "current_efficiency_percent_predicted": res["alloy_kinetics"]["current_efficiency_percent"],
            "carbon_wt_percent_predicted": res["carbon_incorporation"]["predicted_carbon_wt_percent"],
            "surface_blocking_factor": res["carbon_incorporation"]["surface_blocking_factor"],
            "is_anomalous_predicted": res["alloy_kinetics"]["is_anomalous"],
            "mechanism_fe_ni": mechanism_fe_ni,
            "notes": (
                f"Synthetic Phase III screening; mechanism={mechanism_fe_ni}. "
                "Not wet-lab data. Verify by SEM-EDS and combustion analysis."
            ),
        })
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    return df, model


# -------------------------------------------------------------------
# Figure generation
# -------------------------------------------------------------------

def plot_anomalous_kinetics(
    model: PhaseIIICoDeposition,
    j_range: np.ndarray,
    mechanism_label: str,
    save_path: Path,
) -> None:
    """Plot alloy composition vs. current density for the anomalous model."""
    alloy_sweep = model.run_sweep(j_range)
    js = np.array(alloy_sweep["j_mA_cm2"])
    fe_pct = np.array(alloy_sweep["fe_wt_percent"])
    ni_pct = np.array(alloy_sweep["ni_wt_percent"])
    anomalous = np.array(alloy_sweep["is_anomalous"], dtype=bool)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # Left: composition vs current density
    ax = axes[0]
    ax.plot(js, fe_pct, color="#1874b4", linewidth=2.5, label="Fe (predicted)", marker="o", markersize=3)
    ax.plot(js, ni_pct, color="#d95f02", linewidth=2.5, label="Ni (predicted)", marker="s", markersize=3)
    # Shade anomalous region
    anomalous_js = js[anomalous]
    if len(anomalous_js) > 0:
        ax.axvspan(
            anomalous_js.min(), anomalous_js.max(),
            alpha=0.10, color="#e31a1c", label="Anomalous regime",
        )
    ax.axhline(60.0, color="#777777", linestyle=":", linewidth=1, label="Bath Fe ref (~60 wt%)")
    ax.set(
        title=f"Anomalous Fe–Ni co-deposition ({mechanism_label})",
        xlabel="Current density (mA/cm²)",
        ylabel="Alloy composition (wt%)",
        xlim=(0, max(js) * 1.05),
        ylim=(0, 105),
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="upper right")

    # Right: current efficiency and deposition rate
    ax = axes[1]
    ce_pct = np.array(alloy_sweep["current_efficiency_percent"])
    rate_um_hr = np.array(alloy_sweep["deposition_rate_um_hr"])
    color_ce = "#1a9641"
    color_rate = "#a6cee3"
    ax_right = ax.twinx()
    line1 = ax.plot(js, ce_pct, color=color_ce, linewidth=2.5, marker="o", markersize=3,
                    label="Metal CE (%)")
    line2 = ax_right.plot(js, rate_um_hr, color=color_rate, linewidth=2.5, marker="s", markersize=3,
                         label="Deposition rate (µm/hr)")
    ax.set(xlabel="Current density (mA/cm²)", ylabel="Current efficiency (%)", title="Efficiency & rate")
    ax_right.set_ylabel("Deposition rate (µm/hr)")
    ax.set_ylim(0, 105)
    ax_right.set_ylim(0, max(rate_um_hr) * 1.15 if max(rate_um_hr) > 0 else 50)
    ax.grid(alpha=0.25)
    lines_combined = line1 + line2
    labels_combined = [l.get_label() for l in lines_combined]
    ax.legend(lines_combined, labels_combined, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_guglielmi_incorporation(
    model: PhaseIIICoDeposition,
    j_range: np.ndarray,
    save_path: Path,
) -> None:
    """Plot Guglielmi model predictions for carbon incorporation."""
    carbon_results = []
    for j in j_range:
        carbon_results.append(model.run_at_current(float(j))["carbon_incorporation"])
    sigma_vals = [r["loose_adsorption_sigma"] for r in carbon_results]
    w_c_vals = [r["predicted_carbon_wt_percent"] for r in carbon_results]
    blocking_vals = [r["surface_blocking_factor"] for r in carbon_results]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))

    # Loose adsorption coverage
    ax = axes[0]
    ax.plot(j_range, sigma_vals, color="#4daf4a", linewidth=2.5, marker="o", markersize=4)
    ax.fill_between(j_range, sigma_vals, alpha=0.2, color="#4daf4a")
    ax.set(
        title="Loose adsorption coverage (σ)",
        xlabel="Current density (mA/cm²)", ylabel="σ (fraction, 0–1)",
        ylim=(0, 1.05),
    )
    ax.grid(alpha=0.25)

    # Predicted carbon content
    ax = axes[1]
    ax.plot(j_range, w_c_vals, color="#e41a1c", linewidth=2.5, marker="s", markersize=4)
    ax.fill_between(j_range, w_c_vals, alpha=0.15, color="#e41a1c")
    ax.set(
        title="Predicted carbon content",
        xlabel="Current density (mA/cm²)", ylabel="C (wt%)",
        ylim=(0, max(w_c_vals) * 1.15 if max(w_c_vals) > 0 else 10),
    )
    ax.axhline(5.0, color="#777777", linestyle=":", linewidth=1, label="Typical exp. max (~5%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Surface blocking factor
    ax = axes[2]
    ax.plot(j_range, blocking_vals, color="#984ea3", linewidth=2.5, marker="^", markersize=4)
    ax.set(
        title="Metal surface blocking", xlabel="Current density (mA/cm²)",
        ylabel="Blocking factor",
        ylim=(0.5, 1.05),
    )
    ax.grid(alpha=0.25)
    # Annotate the mechanism
    ax.annotate(
        "σ increases with j →\nmore surface blocked",
        xy=(max(j_range) * 0.7, blocking_vals[int(len(j_range) * 0.7)]),
        fontsize=8, bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6"),
    )

    fig.suptitle(
        "Guglielmi two-step carbon incorporation (synthetic screening)", fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_combined_phase3(
    model: PhaseIIICoDeposition,
    j_range: np.ndarray,
    save_path: Path,
) -> None:
    """Combined overview figure for Phase III co-deposition."""
    sweep = model.run_sweep(j_range)
    js = np.array(sweep["j_mA_cm2"])
    fe_pct = np.array(sweep["fe_wt_percent"])
    carbon_pct = np.array(sweep["carbon_wt_percent"])
    ce_pct = np.array(sweep["adjusted_ce_percent"])
    rate_um_hr = np.array(sweep["deposition_rate_um_hr"])
    anomalous = np.array(sweep["is_anomalous"], dtype=bool)

    fig = plt.figure(figsize=(13.0, 9.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.32, wspace=0.28)

    # Top-left: alloy composition
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(js, fe_pct, color="#1874b4", linewidth=2.5, label="Fe (wt%)", marker="o", markersize=3)
    ax.plot(js, 100.0 - fe_pct, color="#d95f02", linewidth=2.5, label="Ni (wt%)", marker="s", markersize=3)
    if np.any(anomalous):
        ax.axvspan(js[anomalous][0], js[anomalous][-1], alpha=0.08, color="#e31a1c", label="Anomalous region")
    ax.set(title="Alloy composition (Fe–Ni)", xlabel="j (mA/cm²)", ylabel="wt%",
           xlim=(0, max(js) * 1.05), ylim=(0, 105))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # Top-right: carbon incorporation + blocking
    ax = fig.add_subplot(gs[0, 1])
    ax_left = ax
    ax_right = ax.twinx()
    line1 = ax_left.plot(js, carbon_pct, color="#e41a1c", linewidth=2.5, marker="o", markersize=3,
                         label="C (wt%) — left axis")
    blocking_vals = [model.run_at_current(float(j))["carbon_incorporation"]["surface_blocking_factor"]
                      for j in js]
    line2 = ax_right.plot(js, blocking_vals, color="#984ea3", linewidth=2.0, linestyle="--",
                          marker="^", markersize=3, label="Blocking — right axis")
    ax_left.set(title="Carbon incorporation & surface blocking",
                xlabel="j (mA/cm²)", ylabel="C (wt%)", ylim=(0, max(carbon_pct) * 1.15))
    ax_right.set_ylabel("Blocking factor")
    ax_right.set_ylim(0.5, 1.05)
    lines_comb = line1 + line2
    labels_comb = [l.get_label() for l in lines_comb]
    ax_left.legend(lines_comb, labels_comb, fontsize=8, loc="upper right")
    ax_left.grid(alpha=0.25)

    # Middle-left: current efficiency
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(js, ce_pct, color="#1a9641", linewidth=2.5, marker="o", markersize=3, label="Metal CE")
    ax.fill_between(js, ce_pct, alpha=0.15, color="#1a9641")
    ax.set(title="Metal current efficiency", xlabel="j (mA/cm²)", ylabel="CE (%)",
           ylim=(0, 105), xlim=(0, max(js) * 1.05))
    ax.grid(alpha=0.25)

    # Middle-right: deposition rate
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(js, rate_um_hr, color="#377eb8", linewidth=2.5, marker="s", markersize=3, label="Rate")
    ax.fill_between(js, rate_um_hr, alpha=0.1, color="#377eb8")
    ax.set(title="Deposition rate", xlabel="j (mA/cm²)", ylabel="Rate (µm/hr)",
           xlim=(0, max(js) * 1.05))
    ax.grid(alpha=0.25)

    # Bottom: combined diagnostic table / note
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    note_text = (
        f"Phase III Integrated Screening — Synthetic Example\n"
        f"Mechanism (Fe–Ni): {model.mechanism_fe_ni}  |  Mechanism (C): {model.mechanism_carbon}\n"
        f"Bath: Fe = {model.bath_fe_M} M, Ni = {model.bath_ni_M} M, C = {model.carbon_particle_loading_g_L} g/L\n"
        f"Temperature = {model.temperature_C} °C, pH = {model.pH}\n"
        f"Anomalous regime detected: {'YES' if np.any(anomalous) else 'NO'}  |  "
        f"Max C content (predicted): {max(carbon_pct):.2f} wt%  |  "
        f"Min metal CE (predicted): {min(ce_pct):.1f} %"
    )
    ax.text(
        0.5, 0.5, note_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="center",
        horizontalalignment="center",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f7f7f7", edgecolor="#777777", alpha=0.95),
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


# -------------------------------------------------------------------
# Main driver
# -------------------------------------------------------------------

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("PHASE III — CO-DEPOSITION: ANOMALOUS Fe–Ni + GUGLIELMI CARBON")
    print("=" * 70)

    mechanisms_to_test = [
        "hydroxide_suppression",
        "intermediate_adsorption",
        "mixed_metal_intermediate",
    ]
    j_range = np.linspace(10.0, 300.0, 25)

    for mech in mechanisms_to_test:
        label = mech.replace("_", " ").title()
        print(f"\n▶ Mechanism: {label} ({mech})")

        # Generate synthetic dataset
        csv_path = DATA_DIR / f"synthetic_co_deposition_{mech}.csv"
        df, model = generate_synthetic_co_deposition_data(csv_path, mechanism_fe_ni=mech)
        print(f"  Synthetic dataset saved: {csv_path}")
        print(f"  Rows: {len(df)}  |  Columns: {list(df.columns)}")

        # Generate figures
        kin_path = FIG_DIR / f"co_deposition_anomalous_kinetics_{mech}.png"
        gug_path = FIG_DIR / f"co_deposition_guglielmi_incorporation_{mech}.png"
        comb_path = FIG_DIR / f"co_deposition_phase3_combined_{mech}.png"

        plot_anomalous_kinetics(model, j_range, label, kin_path)
        plot_guglielmi_incorporation(model, j_range, gug_path)
        plot_combined_phase3(model, j_range, comb_path)
        print(f"  Figures saved: {kin_path.name}, {gug_path.name}, {comb_path.name}")

        # Print representative point summary
        res_100 = model.run_at_current(100.0)
        print("  @ 100 mA/cm²:")
        print(f"    Alloy: Fe = {res_100['alloy_kinetics']['fe_wt_percent']} wt%, "
              f"Ni = {res_100['alloy_kinetics']['ni_wt_percent']} wt%")
        print(f"    Anomalous: {'YES' if res_100['alloy_kinetics']['is_anomalous'] else 'NO'}")
        print(f"    Carbon (predicted): {res_100['carbon_incorporation']['predicted_carbon_wt_percent']} wt%")
        print(f"    Metal CE: {res_100['alloy_kinetics']['current_efficiency_percent']}%  |  "
              f"Adjusted CE: {res_100['integrated_metrics']['adjusted_overall_current_efficiency_percent']}%")

    # Single representative JSON report (using the first mechanism as exemplar)
    exemplar_model = build_phase3_model(mechanism_fe_ni="hydroxide_suppression")
    report = {
        "project": "Aqueous Electrowinning — Phase III Co-Deposition",
        "date": "2026-07-29",
        "model_version": "1.0",
        "scope": (
            "Synthetic screening of anomalous Fe–Ni kinetics (three mechanistic variants) "
            "and Guglielmi carbon-particle incorporation. Not wet-lab data."
        ),
        "references": [
            "Guglielmi, N. (1972). J. Electrochem. Soc., 119(8), 1009.",
            "Matlosz, M. (1993). J. Electrochem. Soc., 140(8), 2275.",
            "Dahms, H. & Croll, I.M. (1975). J. Electrochem. Soc., 122(8), 1117.",
            "Zhuang, J. et al. (2022). Mixed-metal intermediate mechanism.",
        ],
        "synthetic_examples": {
            "csv_files": [f"synthetic_co_deposition_{m}.csv" for m in mechanisms_to_test],
            "figure_files": [
                f.name for f in FIG_DIR.glob("co_deposition_*.png")
            ],
        },
        "key_predictions_at_100_mA_cm2": exemplar_model.run_at_current(100.0),
        "method_notes": [
            "Alloy predictions: Butler–Volmer + inhibition model; mechanism selectable.",
            "Carbon predictions: Guglielmi two-step (Langmuir loose + electrochemical strong).",
            "Actual incorporation requires combustion analysis / EDS verification.",
            "Synthetic data should be replaced with validated instrument exports.",
        ],
    }
    report_path = DATA_DIR / "co_deposition_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("\n✅ Phase III co-deposition driver complete.")
    print(f"  Report: {report_path}")
    print(f"  Figures: {len(list(FIG_DIR.glob('co_deposition_*.png')))} generated")


if __name__ == "__main__":
    main()
