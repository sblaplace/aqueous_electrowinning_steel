"""Driver script to evaluate pre-lab operating window and generate phase maps."""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

from .operating_window import evaluate_operating_point, map_2d_operating_window


def main():
    print("=== Running Pre-Lab Operating Window Classifier & Phase Mapper ===")

    # Evaluate a baseline operating point
    base_res = evaluate_operating_point(
        j_mA_cm2=200.0,
        pH_bulk=2.5,
        T_C=50.0,
        c_Fe2_M=1.0,
        delta_um=100.0
    )

    print("Base Point (j=200 mA/cm2, pH=2.5, T=50°C):")
    print(f"  Pass Status: {base_res['is_pass']} (Status Code: {base_res['status_code']})")
    print(f"  FE: {base_res['FE']*100:.1f}%")
    print(f"  V_cell: {base_res['V_cell']:.2f} V")
    print(f"  Specific Energy: {base_res['specific_energy_kWh_t']:.0f} kWh/t Fe")
    print(f"  Surface pH: {base_res['pH_surface']:.2f} (Precip limit: {base_res['pH_precip']:.2f})")

    # 2D Operating Window Map: Current Density vs pH
    j_vals = np.linspace(50.0, 500.0, 15)
    pH_vals = np.linspace(1.2, 4.0, 15)

    map_res = map_2d_operating_window(
        param_x_name="j_mA_cm2",
        x_vals=j_vals,
        param_y_name="pH_bulk",
        y_vals=pH_vals,
        fixed_params={"T_C": 50.0, "c_Fe2_M": 1.0, "delta_um": 100.0, "j_mA_cm2": 200.0, "pH_bulk": 2.5}
    )

    print(f"2D Operating Grid (j vs pH): {map_res['pass_fraction']*100:.1f}% of operating space is viable (Golden Window)")

    # Save JSON report
    os.makedirs("experiments/data", exist_ok=True)
    os.makedirs("docs/figures", exist_ok=True)

    report_data = {
        "base_point": base_res,
        "pass_fraction_pct": map_res["pass_fraction"] * 100.0,
        "x_vals_j_mA_cm2": j_vals.tolist(),
        "y_vals_pH": pH_vals.tolist(),
        "pass_mask": map_res["pass_mask"].tolist(),
        "status_grid": map_res["status_grid"].tolist(),
    }

    with open("experiments/data/operating_window_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    print("Saved experiments/data/operating_window_report.json")

    # Plot 2D Phase Map
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    X, Y = np.meshgrid(j_vals, pH_vals)

    # Plot 1: Pass/Fail Mask (Golden Window)
    c1 = ax1.pcolormesh(X, Y, map_res["pass_mask"].astype(int), cmap='RdYlGn', vmin=0, vmax=1, shading='auto')
    ax1.set_xlabel("Current Density j (mA/cm²)")
    ax1.set_ylabel("Bulk pH")
    ax1.set_title("Pre-Lab Viable Operating Window (Green = PASS)")
    ax1.grid(True, alpha=0.3, color='k', linestyle=':')

    # Plot 2: FE Contour Map
    c2 = ax2.contourf(X, Y, map_res["FE_grid"] * 100.0, levels=10, cmap='viridis')
    fig.colorbar(c2, ax=ax2, label="Faradaic Efficiency (%)")
    ax2.set_xlabel("Current Density j (mA/cm²)")
    ax2.set_ylabel("Bulk pH")
    ax2.set_title("Faradaic Efficiency Contour Map")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = "docs/figures/operating_window_map.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
