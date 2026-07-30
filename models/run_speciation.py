"""Driver script to run speciation & activity model, emit report, and plot figures."""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from .speciation import SolutionComposition, solve_speciation, speciation_temperature_sweep


def main():
    print("=== Running Concentrated Electrolyte Speciation & Activity Model ===")
    
    comp = SolutionComposition(c_FeSO4=1.0, c_Na2SO4=0.5, c_H2SO4=0.01, c_H3BO3=0.4, T_C=50.0)
    single_res = solve_speciation(comp)
    
    print(f"Single Point (50°C, 1.0M FeSO4, pH {single_res['pH_activity']:.2f}):")
    print(f"  Ionic Strength: {single_res['ionic_strength_M']:.3f} M")
    print(f"  Free [Fe²⁺]: {single_res['c_Fe2_free_M']:.3f} M (gamma_Fe2 = {single_res['gamma_Fe2']:.3f})")
    print(f"  FeSO4(aq) Pair: {single_res['c_FeSO4_pair_M']:.3f} M ({single_res['fe2_pair_percentage']:.1f}% paired)")
    print(f"  Conductivity: {single_res['conductivity_S_m']:.2f} S/m")
    print(f"  Nernst E_rev(Fe): {single_res['E_rev_Fe_V_SHE']:.3f} V vs SHE")
    print(f"  Fe(OH)2 Precip pH: {single_res['pH_precip_Fe_OH2']:.2f}")
    
    # Temperature sweep
    sweep_res = speciation_temperature_sweep(comp, T_min=20.0, T_max=85.0, num=14)
    
    # Prepare outputs
    os.makedirs("experiments/data", exist_ok=True)
    os.makedirs("docs/figures", exist_ok=True)
    
    report_data = {
        "base_case": single_res,
        "temperature_sweep": {k: v.tolist() for k, v in sweep_res.items()}
    }
    
    with open("experiments/data/speciation_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    print("Saved experiments/data/speciation_report.json")
    
    # Plot figures
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 9))
    
    temps = sweep_res["temperature_C"]
    
    # Ax1: Free [Fe2+] and FeSO4 pair vs Temperature
    ax1.plot(temps, sweep_res["c_Fe2_free_M"], 'b-o', label="Free [Fe²⁺] (M)")
    ax1.plot(temps, sweep_res["c_FeSO4_pair_M"], 'r--s', label="FeSO₄⁰ Pair (M)")
    ax1.set_xlabel("Temperature (°C)")
    ax1.set_ylabel("Concentration (M)")
    ax1.set_title("Iron Speciation vs Temperature")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Ax2: Ionic strength and activity coefficients
    ax2.plot(temps, sweep_res["ionic_strength_M"], 'k-d', label="Ionic Strength I (M)")
    ax2.plot(temps, sweep_res["gamma_Fe2"], 'g--^', label="γ(Fe²⁺)")
    ax2.set_xlabel("Temperature (°C)")
    ax2.set_ylabel("Value")
    ax2.set_title("Ionic Strength & Activity Coefficients")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Ax3: Electrical conductivity
    ax3.plot(temps, sweep_res["conductivity_S_m"], 'm-s')
    ax3.set_xlabel("Temperature (°C)")
    ax3.set_ylabel("Conductivity κ (S/m)")
    ax3.set_title("Solution Electrical Conductivity")
    ax3.grid(True, alpha=0.3)
    
    # Ax4: Fe(OH)2 precipitation threshold pH
    ax4.plot(temps, sweep_res["pH_precip_Fe_OH2"], 'c-o')
    ax4.set_xlabel("Temperature (°C)")
    ax4.set_ylabel("Precipitation pH")
    ax4.set_title("Fe(OH)₂ Precipitation Threshold")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = "docs/figures/speciation_profiles.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
