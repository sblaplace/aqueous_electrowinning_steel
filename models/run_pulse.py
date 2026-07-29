"""Driver script for pulse and pulse-reverse electrodeposition simulations.

Generates diagnostic plots and JSON summary comparing DC, Pulsed (PE), and
Pulse-Reverse (PRE) electrodeposition parameters.
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from models.pulse import (
    PulseDepositionModel,
    PulseWaveform,
    compare_dc_vs_pulse,
)

FIGURES_DIR = Path("docs/figures")
DATA_DIR = Path("experiments/data")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Running Pulse and Pulse-Reverse Electrodeposition Simulations...")

    # 1. Base simulation case: 10 Hz, 50% duty cycle, 100 mA/cm2 peak current
    j_peak = 100.0  # mA/cm2
    freq = 10.0     # Hz
    duty = 0.5

    comparison = compare_dc_vs_pulse(
        j_peak_mA_cm2=j_peak,
        duty_cycle=duty,
        frequency_Hz=freq,
        n_cycles=10,
        fe_bulk_M=1.0,
        bulk_pH=2.0,
    )

    pe_res = comparison["pulsed"]
    pre_res = comparison["pulse_reverse"]
    dc_peak_res = comparison["dc_peak"]
    dc_avg_res = comparison["dc_avg"]

    # --- Plot 1: Transient Profiles over Pulse Cycles ---
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)

    # Convert time to milliseconds for readability
    t_ms = pe_res.time_s * 1000.0

    # Panel 0: Applied current density
    axes[0].plot(t_ms, pe_res.applied_current_A_m2 / 10.0, "b-", label="Pulsed (PE)", linewidth=1.5)
    axes[0].plot(t_ms, pre_res.applied_current_A_m2 / 10.0, "r--", label="Pulse-Reverse (PRE)", linewidth=1.5)
    axes[0].set_ylabel("Applied Current\n(mA/cm²)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[0].set_title("Pulse & Pulse-Reverse Transient Electrodeposition Dynamics (10 Hz, pH 2.0)")

    # Panel 1: Surface Fe2+ Concentration
    axes[1].plot(t_ms, pe_res.surface_fe_M, "b-", label="Pulsed (PE)")
    axes[1].plot(t_ms, pre_res.surface_fe_M, "r--", label="Pulse-Reverse (PRE)")
    axes[1].plot(t_ms, dc_peak_res.surface_fe_M, "k:", label="DC Peak (100 mA/cm²)")
    axes[1].set_ylabel("Surface Fe²⁺\n(M)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right")

    # Panel 2: Surface pH
    axes[2].plot(t_ms, pe_res.surface_pH, "b-", label="Pulsed (PE)")
    axes[2].plot(t_ms, pre_res.surface_pH, "r--", label="Pulse-Reverse (PRE)")
    axes[2].plot(t_ms, dc_peak_res.surface_pH, "k:", label="DC Peak (100 mA/cm²)")
    axes[2].set_ylabel("Surface pH")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    # Panel 3: Instantaneous Current Efficiency
    axes[3].plot(t_ms, pe_res.instant_efficiency * 100.0, "b-", label="Pulsed (PE)")
    axes[3].plot(t_ms, pre_res.instant_efficiency * 100.0, "r--", label="Pulse-Reverse (PRE)")
    axes[3].set_ylabel("Instant Fe Efficiency\n(%)")
    axes[3].set_xlabel("Time (ms)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="lower right")

    plt.tight_layout()
    transient_fig_path = FIGURES_DIR / "pulse_reverse_transient.png"
    plt.savefig(transient_fig_path, dpi=200)
    plt.close()
    print(f"Saved transient dynamics plot to {transient_fig_path}")

    # --- Plot 2: DC vs Pulsed vs PRE Comparison Bar Chart & Frequency Sweep ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Subplot 1: Comparison of current efficiencies
    modes = ["DC Peak\n(100 mA/cm²)", "DC Avg\n(50 mA/cm²)", "Pulsed (PE)\n(10 Hz)", "Pulse-Reverse\n(10 Hz)"]
    efficiencies = [
        dc_peak_res.cycle_avg_efficiency * 100.0,
        dc_avg_res.cycle_avg_efficiency * 100.0,
        pe_res.cycle_avg_efficiency * 100.0,
        pre_res.cycle_avg_efficiency * 100.0,
    ]
    colors = ["#7f7f7f", "#2ca02c", "#1f77b4", "#d62728"]

    bars = ax1.bar(modes, efficiencies, color=colors, width=0.55)
    ax1.set_ylabel("Current Efficiency (%)")
    ax1.set_title("Current Efficiency Comparison")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", alpha=0.3)

    for bar, eff in zip(bars, efficiencies):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, eff + 1.5, f"{eff:.1f}%", ha="center", va="bottom", fontsize=9)

    # Subplot 2: Frequency Sweep (1 Hz to 100 Hz)
    freqs = np.logspace(0, 2, 10)  # 1 Hz to 100 Hz
    pe_effs = []
    pre_effs = []
    pe_max_phs = []

    model = PulseDepositionModel(fe_bulk_M=1.0, bulk_pH=2.0)

    for f in freqs:
        t_c = 1.0 / f
        wf_pe = PulseWaveform(j_cathodic_mA_cm2=100.0, t_cathodic_s=t_c * 0.5, t_off_s=t_c * 0.5)
        wf_pre = PulseWaveform(
            j_cathodic_mA_cm2=100.0,
            t_cathodic_s=t_c * 0.5,
            j_anodic_mA_cm2=-20.0,
            t_anodic_s=t_c * 0.1,
            t_off_s=t_c * 0.4,
        )
        r_pe = model.simulate(wf_pe, n_cycles=10)
        r_pre = model.simulate(wf_pre, n_cycles=10)

        pe_effs.append(r_pe.cycle_avg_efficiency * 100.0)
        pre_effs.append(r_pre.cycle_avg_efficiency * 100.0)
        pe_max_phs.append(r_pe.max_surface_pH)

    ax2.semilogx(freqs, pe_effs, "o-b", label="Pulsed (PE)")
    ax2.semilogx(freqs, pre_effs, "s--r", label="Pulse-Reverse (PRE)")
    ax2.set_xlabel("Pulse Frequency (Hz)")
    ax2.set_ylabel("Cycle-Averaged Efficiency (%)")
    ax2.set_title("Frequency Dependence (50% Duty Cycle)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    comparison_fig_path = FIGURES_DIR / "dc_vs_pulse_comparison.png"
    plt.savefig(comparison_fig_path, dpi=200)
    plt.close()
    print(f"Saved comparison plot to {comparison_fig_path}")

    # Write JSON summary report
    report = {
        "simulation_parameters": {
            "j_peak_mA_cm2": j_peak,
            "frequency_Hz": freq,
            "duty_cycle": duty,
            "fe_bulk_M": 1.0,
            "bulk_pH": 2.0,
        },
        "results": {
            "dc_peak": dc_peak_res.summary(),
            "dc_avg": dc_avg_res.summary(),
            "pulsed": pe_res.summary(),
            "pulse_reverse": pre_res.summary(),
        },
    }

    report_json_path = DATA_DIR / "pulse_reverse_report.json"
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved JSON report to {report_json_path}\n")

    print("Summary of Simulation Results:")
    for mode, res in report["results"].items():
        print(f"\n--- {mode.upper()} ---")
        for k, v in res.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
