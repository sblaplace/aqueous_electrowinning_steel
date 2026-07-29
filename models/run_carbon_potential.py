"""
Driver for carbon potential / activity model (gas carburizing atmosphere).

Generates:
* docs/figures/carbon_potential_map.png
* docs/figures/carbon_potential_dewpoint.png
* docs/figures/carbon_potential_Acm.png
* experiments/data/carbon_potential_report.json
* experiments/data/synthetic_carbon_potential.csv

Usage:
    python -m models.run_carbon_potential
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

from models.carbon_potential import (
    carbon_potential_summary,
    carbon_wt_from_activity,
    carbon_activity_from_co_co2,
    austenite_max_carbon_wt_percent,
)

FIG_DIR = ROOT / "docs" / "figures"
DATA_DIR = ROOT / "experiments" / "data"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("="*72)
    print("CARBON POTENTIAL — GAS CARBURIZING ATMOSPHERE SCREENING")
    print("="*72)

    # Example conditions
    T_C = 930.0
    pCO = 0.20
    pCO2_range = np.logspace(-4, -2, 50)  # 0.01% to 1%
    pH2 = 0.40
    pCH4 = 0.02

    rows = []
    for pCO2 in pCO2_range:
        summ = carbon_potential_summary(T_C=T_C, pCO=pCO, pCO2=pCO2, pCH4=pCH4, pH2=pH2, dew_point_C=-10.0)
        rows.append({
            "pCO_atm": pCO,
            "pCO2_atm": pCO2,
            "pCO2_ppm": pCO2*1e6,
            "aC_CO_CO2": summ["aC_from_CO_CO2"],
            "C_wt_CO_CO2": summ["C_wt_from_CO_CO2"],
            "pO2_atm": summ["pO2_atm"],
            "log10_pO2": summ["log10_pO2"],
            "aC_CH4_H2": summ.get("aC_from_CH4_H2", np.nan),
            "C_wt_CH4_H2": summ.get("C_wt_from_CH4_H2", np.nan),
        })

    df = pd.DataFrame(rows)
    csv_path = DATA_DIR / "synthetic_carbon_potential.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✅ Saved CSV: {csv_path} ({len(df)} rows)")

    # Figure 1: aC and C wt vs pCO2 (CO/CO2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.loglog(df["pCO2_atm"], df["aC_CO_CO2"], color="#1874b4", marker="o", ms=3)
    ax.axhline(1.0, color="#e31a1c", ls="--", label="graphite aC=1")
    ax.set(xlabel="pCO2 (atm)", ylabel="aC (graphite ref)", title=f"Carbon activity vs pCO2 — T={T_C}°C, pCO={pCO}")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    ax.semilogx(df["pCO2_atm"], df["C_wt_CO_CO2"], color="#d95f02", marker="s", ms=3)
    Cmax = austenite_max_carbon_wt_percent(T_C)
    ax.axhline(Cmax, color="#777777", ls=":", label=f"Acm max {Cmax:.2f} wt%")
    ax.set(xlabel="pCO2 (atm)", ylabel="Equil. C wt% in austenite (screening)", title="Surface C vs pCO2")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.suptitle("Carbon potential map — CO/CO2 Boudouard (screening)", fontweight="bold")
    fig.tight_layout()
    fig1 = FIG_DIR / "carbon_potential_map.png"
    fig.savefig(fig1, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {fig1}")

    # Figure 2: dew point vs aC
    dew_points = np.linspace(-30, 20, 50)
    aC_dew = []
    C_dew = []
    for dp in dew_points:
        summ = carbon_potential_summary(T_C=T_C, pCO=pCO, pCO2=0.001, pH2=pH2, dew_point_C=dp)
        aC_dew.append(summ.get("aC_from_dewpoint", np.nan))
        C_dew.append(summ.get("C_wt_from_dewpoint", np.nan))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax2 = ax.twinx()
    l1 = ax.plot(dew_points, aC_dew, color="#1874b4", marker="o", ms=3, label="aC dewpoint")
    l2 = ax2.plot(dew_points, C_dew, color="#e41a1c", marker="s", ms=3, label="C wt% dewpoint")
    ax.set(xlabel="Dew point (°C)", ylabel="aC (graphite ref)", title=f"Carbon potential vs dew point — T={T_C}°C")
    ax2.set_ylabel("C wt%")
    ax.grid(alpha=0.25)
    # combine legends
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig2 = FIG_DIR / "carbon_potential_dewpoint.png"
    fig.savefig(fig2, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {fig2}")

    # Figure 3: Acm line and C_wt from activity
    temps = np.linspace(700, 1100, 100)
    Cmax_vals = [austenite_max_carbon_wt_percent(T) for T in temps]
    # aC=1 and aC=0.8 lines
    aC_1 = 1.0
    aC_08 = 0.8
    C_at_a1 = [carbon_wt_from_activity(aC_1, T) for T in temps]
    C_at_a08 = [carbon_wt_from_activity(aC_08, T) for T in temps]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(temps, Cmax_vals, color="#777777", ls="--", label="Acm max C (solubility)")
    ax.plot(temps, C_at_a1, color="#e31a1c", label="C wt% @ aC=1.0")
    ax.plot(temps, C_at_a08, color="#1874b4", label="C wt% @ aC=0.8")
    ax.set(xlabel="Temperature (°C)", ylabel="C wt% in austenite", title="Acm solubility & equilibrium C vs T (screening)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig3 = FIG_DIR / "carbon_potential_Acm.png"
    fig.savefig(fig3, dpi=180)
    plt.close(fig)
    print(f"  ✅ Saved: {fig3}")

    # Report JSON
    example_summary = carbon_potential_summary(T_C=930, pCO=0.20, pCO2=0.001, pCH4=0.02, pH2=0.40, dew_point_C=-5.0)
    report = {
        "title": "Carbon potential screening — gas carburizing atmosphere",
        "date": __import__("datetime").datetime.now().isoformat(),
        "example_T_C": T_C,
        "example_summary_930C": example_summary,
        "figures": [str(fig1), str(fig2), str(fig3)],
        "csv": str(csv_path),
        "model_notes": [
            "Boudouard: ΔG°=170700-174.5T J/mol, K=exp(-ΔG/RT), aC=K*pCO^2/pCO2",
            "CH4 cracking: ΔG°=90000-109T J/mol, aC=K*pCH4/pH2^2",
            "O2 from CO+0.5O2→CO2 ΔG°=-282800+86.8T J/mol, pO2=(pCO2/(pCO*K))^2",
            "Dew point → pH2O via Magnus, then via WGS K to pCO2 then aC",
            "C wt from aC via empirical a(C)= (C/1.1)*exp(0.9(C-1.1)) at 900°C a=1→1.1wt%",
            "Acm linear 0.76% at 727°C to 2.14% at 1147°C",
            "Calibrate with O2 probe and foil measurements; not full CALPHAD",
        ],
    }
    report_path = DATA_DIR / "carbon_potential_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  ✅ Saved report: {report_path}")
    print("\n✅ Carbon potential driver complete!")
    return report


if __name__ == "__main__":
    main()
