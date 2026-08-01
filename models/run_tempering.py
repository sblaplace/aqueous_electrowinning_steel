"""
Driver for tempering, retained-austenite, and Hollomon-Jaffe model.

Generates:
* docs/figures/tempering_curve.png
* docs/figures/retained_austenite.png
* docs/figures/case_tempered_hardness.png
* docs/figures/tempering_energy.png
* experiments/data/tempering_report.json
* experiments/data/synthetic_tempering.csv

Usage:
    python -m models.run_tempering
"""

from __future__ import annotations

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.tempering import (
    AlloyComposition,
    martensite_start_C,
    retained_austenite_fraction_koistinen_marburger,
    hollomon_jaffe_parameter,
    tempered_hardness_hollomon_jaffe,
    tempering_curve,
    case_hardness_after_tempering,
    recommended_tempering_for_target_hv,
)
from models.carburization import CarburizationParams, CarburizationModel

FIG_DIR = ROOT / "docs" / "figures"
DATA_DIR = ROOT / "experiments" / "data"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("="*72)
    print("TEMPERING & RETAINED AUSTENITE — SCREENING")
    print("="*72)

    # Example alloy: 0.8% C carburized case
    chem = AlloyComposition(C=0.8, Mn=0.3, Ni=0.1, Cr=0.2)
    Ms = martensite_start_C(chem)
    fRA = retained_austenite_fraction_koistinen_marburger(Ms, 25.0)
    print(f"Example: C={chem.C} wt% → Ms={Ms:.0f}°C, as-quenched RA={fRA*100:.1f}% (KM α=0.011)")

    # Figure 1: Ms vs C
    C_vals = np.linspace(0.1, 1.2, 50)
    Ms_vals = []
    fRA_vals = []
    for C in C_vals:
        ch = AlloyComposition(C=C, Mn=0.3)
        Ms_c = martensite_start_C(ch)
        Ms_vals.append(Ms_c)
        fRA_vals.append(retained_austenite_fraction_koistinen_marburger(Ms_c, 25.0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(C_vals, Ms_vals, color="#1874b4", marker="o", ms=3)
    ax.set(xlabel="C wt%", ylabel="Ms (°C)", title="Martensite start vs C (Andrews)")
    ax.grid(alpha=0.25)
    ax.axhline(0, color="#e31a1c", ls="--", label="0°C")
    ax.legend()

    ax = axes[1]
    ax.plot(C_vals, np.array(fRA_vals)*100, color="#d95f02", marker="s", ms=3)
    ax.set(xlabel="C wt%", ylabel="Retained austenite as-quenched (%)", title="RA vs C (Koistinen-Marburger)")
    ax.grid(alpha=0.25)

    fig.suptitle("Ms and retained austenite — screening (Andrews + KM)", fontweight="bold")
    fig.tight_layout()
    fig1 = FIG_DIR / "retained_austenite.png"
    fig.savefig(fig1, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {fig1}")

    # Figure 2: Tempering curve HV vs T at 1 hr
    HV_q = 850.0  # as-quenched 0.8% C
    curve = tempering_curve(HV_q=HV_q, chem=chem, t_hr=1.0, T_range_C=(150, 650), n_points=80)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(curve["T_C"], curve["HV_tempered"], color="#e41a1c", marker="o", ms=3, label="HV tempered")
    ax.plot(curve["T_C"], curve["YS_MPa"], color="#1874b4", ls="--", label="YS (Tabor)")
    ax.set(xlabel="Tempering T (°C)", ylabel="HV / YS", title=f"Tempering curve — HV_q={HV_q:.0f} HV, 1 hr, C={chem.C}%")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    ax.plot(curve["T_C"], curve["f_RA_remaining"]*100, color="#4daf4a", marker="s", ms=3, label="RA remaining")
    ax.plot(curve["T_C"], curve["f_RA_as_quenched"]*100, color="#777777", ls=":", label="RA as-quenched")
    ax.set(xlabel="Tempering T (°C)", ylabel="RA %", title="RA decomposition during tempering (screening)")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.suptitle("Tempering curve and RA — Hollomon-Jaffe P = T*(19.5+log10(t))", fontweight="bold")
    fig.tight_layout()
    fig2 = FIG_DIR / "tempering_curve.png"
    fig.savefig(fig2, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {fig2}")

    # Figure 3: Case hardness after tempering from carburization profile
    # Take carburized profile: 900°C 4 hr, 1 mm sheet
    carb_params = CarburizationParams(temperature_C=900, surface_carbon_wt_percent=1.1, sheet_thickness_um=1000)
    carb_model = CarburizationModel(carb_params)
    prof = carb_model.profile_at_time(t_hr=4.0, n_points=200)
    # Compute tempered variants at 180°C and 400°C
    HV_q, fRA_q, HV_t180 = case_hardness_after_tempering(
        prof.c_wt_percent, prof.x_um, temper_T_C=180, temper_t_hr=1.0, quench_rate_C_s=200, chem_base=chem
    )
    _, _, HV_t400 = case_hardness_after_tempering(
        prof.c_wt_percent, prof.x_um, temper_T_C=400, temper_t_hr=1.0, quench_rate_C_s=200, chem_base=chem
    )

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(prof.x_um, HV_q, color="#333333", ls="-", label="As-quenched HV_q")
    ax.plot(prof.x_um, HV_t180, color="#1874b4", ls="--", label="Tempered 180°C 1 hr")
    ax.plot(prof.x_um, HV_t400, color="#d95f02", ls="-.", label="Tempered 400°C 1 hr")
    ax.set(xlabel="Depth from surface (µm)", ylabel="HV (kgf/mm²)",
           title="Case hardness after tempering — from carburization profile 900°C 4 hr",
           xlim=(0, 600), ylim=(150, 900))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig3 = FIG_DIR / "case_tempered_hardness.png"
    fig.savefig(fig3, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {fig3}")

    # Figure 4: Recommended tempering T for target HV
    target_HVs = [700, 600, 500, 400]
    Ts_needed = []
    for tgt in target_HVs:
        Trec = recommended_tempering_for_target_hv(HV_q=HV_q.max(), target_HV=tgt, t_hr=1.0)
        Ts_needed.append(Trec if Trec is not None else np.nan)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(h) for h in target_HVs], Ts_needed, color="#fdbf6f")
    ax.set(xlabel="Target HV (1 hr temper)", ylabel="Recommended Tempering T (°C)", title="Tempering to target hardness (screening inverse)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig4 = FIG_DIR / "tempering_energy.png"
    fig.savefig(fig4, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {fig4}")

    # Synthetic tempering CSV for C sweep
    rows = []
    for T in [180, 300, 450, 600]:
        for C in [0.2, 0.4, 0.6, 0.8, 1.0]:
            ch = AlloyComposition(C=C, Mn=0.3)
            Ms_c = martensite_start_C(ch)
            fRA = retained_austenite_fraction_koistinen_marburger(Ms_c, 25)
            # as-quenched HV ~127+949*C
            HV_q_local = min(127+949*C, 900)
            P = hollomon_jaffe_parameter(T, 1.0)
            HV_t = tempered_hardness_hollomon_jaffe(HV_q_local, P)
            rows.append({
                "C_wt": C,
                "temper_T_C": T,
                "Ms_C": Ms_c,
                "RA_as_quenched_pct": fRA*100,
                "HV_as_quenched": HV_q_local,
                "P_Hollomon_Jaffe": P,
                "HV_tempered_1hr": HV_t,
            })
    df = pd.DataFrame(rows)
    csv_path = DATA_DIR / "synthetic_tempering.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✅ Saved CSV: {csv_path} ({len(df)} rows)")

    # Report JSON
    report = {
        "title": "Tempering & retained austenite screening",
        "date": __import__("datetime").datetime.now().isoformat(),
        "example_alloy": {"C": chem.C, "Mn": chem.Mn, "Ms_C": Ms, "f_RA_as_quenched_pct": fRA*100},
        "figures": [str(fig1), str(fig2), str(fig3), str(fig4)],
        "csv": str(csv_path),
        "model_notes": [
            "Ms via Andrews: Ms=539-423C-30.4Mn-17.7Ni-12.1Cr-7.5Mo-7.5Si+10Co (°C)",
            "RA via Koistinen-Marburger: f_RA=exp(-0.011*(Ms-Tq)), f_M=1-f_RA",
            "Tempering P = T_K*(19.5+log10(t_s)), HV_t = HV_q*exp(-k*(P-8000)), k=0.00018 screening",
            "Case hardness from carburization profile + RA per local C + tempering",
            "Recommended T for target HV via bisection inverse",
            "Target HV 700->~200C, 600->~350C, 500->~450C, 400->~550C typical low-alloy",
            "Calibrate with XRD RA, hardness traverses, dilatometry",
        ],
        "recommended_tempering": {str(tgt): T for tgt, T in zip(target_HVs, Ts_needed)},
    }
    report_path = DATA_DIR / "tempering_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  ✅ Saved report: {report_path}")
    print("\n✅ Tempering driver complete!")
    return report


if __name__ == "__main__":
    main()
