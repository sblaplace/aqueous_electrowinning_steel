"""Driver script to generate batch recipes and full-factorial DOE run sheets."""

import json
import os
import matplotlib.pyplot as plt
import pandas as pd

from .experimental_matrix import ChemicalRecipe, calculate_batch_recipe, generate_factorial_doe


def main():
    print("=== Running Experimental Recipe & Factorial DOE Matrix Generator ===")
    
    # 1. Chemical batch recipe for 1.0 L cell
    rec = ChemicalRecipe(c_FeSO4_M=1.0, c_Na2SO4_M=0.5, c_H3BO3_M=0.4, c_ascorbic_g_L=2.0, target_pH=2.5, volume_L=1.0)
    batch_info = calculate_batch_recipe(rec)
    
    print("1.0 L Electrowinning Bath Batch Recipe:")
    print(f"  FeSO4·7H2O: {batch_info['FeSO4_7H2O_g']:.2f} g")
    print(f"  Na2SO4:     {batch_info['Na2SO4_g']:.2f} g")
    print(f"  H3BO3:      {batch_info['H3BO3_g']:.2f} g")
    print(f"  Ascorbic:   {batch_info['ascorbic_acid_g']:.2f} g")
    print(f"  Est. 98% H2SO4: {batch_info['est_H2SO4_98pct_mL']:.3f} mL")
    
    # 2. Factorial DOE Matrix (3x2x3 = 18 runs)
    doe_df = generate_factorial_doe(
        j_levels=[100.0, 250.0, 400.0],
        pH_levels=[2.0, 3.0],
        T_levels=[35.0, 50.0, 65.0],
        area_cm2=10.0,
        t_run_hr=2.0
    )
    
    print("\nFactorial DOE Matrix Sample (First 5 Runs):")
    print(doe_df[["run_id", "j_mA_cm2", "pH_bulk", "T_C", "predicted_FE_pct", "m_fe_expected_g", "thickness_um", "pass_status"]].head())
    
    os.makedirs("experiments/data", exist_ok=True)
    os.makedirs("docs/figures", exist_ok=True)
    
    # Save CSV and JSON
    csv_path = "experiments/data/factorial_doe_matrix.csv"
    doe_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    
    report_data = {
        "batch_recipe": batch_info,
        "doe_total_runs": len(doe_df),
        "doe_pass_runs": int(doe_df["pass_status"].sum()),
    }
    
    with open("experiments/data/experimental_matrix_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    print("Saved experiments/data/experimental_matrix_report.json")
    
    # Plot DOE Summary Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ['g' if p else 'r' for p in doe_df["pass_status"]]
    ax.bar(doe_df["run_id"], doe_df["predicted_FE_pct"], color=colors, alpha=0.7, edgecolor='k')
    ax.set_xlabel("Run ID")
    ax.set_ylabel("Predicted Faradaic Efficiency (%)")
    ax.set_title("Factorial DOE Matrix — Predicted FE and Pass/Fail Filter (Green = Pass)")
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = "docs/figures/doe_matrix_summary.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
