"""Driver script to simulate dynamic cell heat balance, emit report, and plot figures."""

import json
import os
import matplotlib.pyplot as plt

from .thermal_balance import CellThermalParams, simulate_thermal_transient


def main():
    print("=== Running Cell Heat Balance & Thermal Management Model ===")

    # 1. Uncooled benchtop cell (10 A, 2.5 V, 2 L)
    p_uncooled = CellThermalParams(
        V_cell=2.5,
        current_A=10.0,
        volume_L=2.0,
        cooling_active=False
    )
    res_uncooled = simulate_thermal_transient(p_uncooled, t_end_hr=4.0)

    # 2. Actively cooled cell with jacket (15 °C inlet)
    p_cooled = CellThermalParams(
        V_cell=2.5,
        current_A=10.0,
        volume_L=2.0,
        cooling_active=True,
        T_cool_in_C=15.0,
        UA_jacket_W_K=12.0
    )
    res_cooled = simulate_thermal_transient(p_cooled, t_end_hr=4.0)

    print(f"Uncooled Steady-State Temperature: {res_uncooled['T_ss_C']:.1f} °C (Max: {res_uncooled['T_max_C']:.1f} °C)")
    print(f"Heat Generation Power: {res_uncooled['heat_gen_power_W']:.1f} W")
    print(f"Cooling Duty Required for 50 °C Hold: {res_uncooled['cooling_duty_50C_W']:.1f} W")
    print(f"Actively Cooled Steady-State Temperature: {res_cooled['T_ss_C']:.1f} °C")

    # Save JSON report
    os.makedirs("experiments/data", exist_ok=True)
    os.makedirs("docs/figures", exist_ok=True)

    report_data = {
        "uncooled": {
            "T_ss_C": res_uncooled["T_ss_C"],
            "T_max_C": res_uncooled["T_max_C"],
            "heat_gen_power_W": res_uncooled["heat_gen_power_W"],
            "cooling_duty_50C_W": res_uncooled["cooling_duty_50C_W"],
        },
        "cooled": {
            "T_ss_C": res_cooled["T_ss_C"],
            "T_max_C": res_cooled["T_max_C"],
        }
    }

    with open("experiments/data/thermal_balance_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    print("Saved experiments/data/thermal_balance_report.json")

    # Plot thermal transient profiles
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    t_hr = res_uncooled["time_hr"]
    ax1.plot(t_hr, res_uncooled["temperature_C"], 'r-', linewidth=2, label="Uncooled Cell")
    ax1.plot(t_hr, res_cooled["temperature_C"], 'b--', linewidth=2, label="Jacketed Cooled Cell")
    ax1.axhline(50.0, color='g', linestyle=':', label="Target Hold (50 °C)")
    ax1.axhline(75.0, color='k', linestyle='--', label="Membrane Limit (75 °C)")
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Electrolyte Temperature (°C)")
    ax1.set_title("Transient Electrolyte Temperature Profile")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Heat loss breakdown for uncooled cell
    ax2.plot(t_hr, res_uncooled["Q_gen_W"], 'k-', label="Joule/Overpotential Gen")
    ax2.plot(t_hr, res_uncooled["Q_amb_W"], 'g--', label="Ambient Convective Loss")
    ax2.plot(t_hr, res_uncooled["Q_evap_W"], 'c-.', label="Evaporative Cooling")
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Heat Flow Rate (W)")
    ax2.set_title("Heat Generation & Loss Breakdown")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    fig_path = "docs/figures/thermal_balance_profiles.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
