"""
Driver for process control simulation.

Generates:
* docs/figures/control_step_response.png   — step response for all 8 loops
* docs/figures/control_disturbance.png     — disturbance rejection comparison
* docs/figures/control_tuning_sensitivity.png — Kp sensitivity sweep
* docs/figures/control_loop_summary.png    — summary table/heatmap
* experiments/data/process_control_report.json

Usage:
    python -m models.run_process_control
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.process_control import (
    default_loops,
    simulate_loop,
    tuning_sensitivity,
    loop_summary_table,
)

FIG_DIR = ROOT / "docs" / "figures"
DATA_DIR = ROOT / "experiments" / "data"

LOOP_COLORS = {
    "electrolyte_temp": "#1f77b4",
    "electrolyte_ph": "#ff7f0e",
    "cell_current": "#2ca02c",
    "recirc_flow": "#d62728",
    "carburize_temp": "#9467bd",
    "carbon_potential": "#8c564b",
    "quench_timing": "#e377c2",
    "tempering_temp": "#7f7f7f",
}


def run_step_response(loops: dict, duration: float = 1200.0, dt: float = 0.5) -> dict:
    """Simulate all loops with a setpoint step and return results."""
    results = {}
    for name, cfg in loops.items():
        results[name] = simulate_loop(name, cfg, duration, dt,
                                      setpoint_step_pct=10.0,
                                      disturbance_time_s=99999.0)  # no disturbance
    return results


def run_disturbance(loops: dict, duration: float = 800.0, dt: float = 0.5) -> dict:
    """Simulate all loops with disturbance injection at t=400s."""
    results = {}
    for name, cfg in loops.items():
        results[name] = simulate_loop(name, cfg, duration, dt,
                                      setpoint_step_pct=0.0,
                                      disturbance_time_s=400.0)
    return results


def plot_step_response(results: dict, output: Path) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, (name, res) in enumerate(results.items()):
        ax = axes[i]
        color = LOOP_COLORS.get(name, "#333")
        t = res.time_s

        ax.plot(t, res.setpoint, "k--", lw=1.0, label="SP")
        ax.plot(t, res.pv, color=color, lw=1.2, label="PV")
        ax2 = ax.twinx()
        ax2.plot(t, res.mv, color=color, alpha=0.3, lw=0.8, label="MV")
        ax2.set_ylabel("MV", fontsize=7, alpha=0.6)
        ax2.tick_params(labelsize=6)

        ax.set_title(f"{res.tag}\n{res.loop_name}", fontsize=8, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel("PV", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.text(0.02, 0.95,
                f"OS={res.overshoot_pct:.1f}%  ts={res.settling_time_s:.0f}s\nSS err={res.steady_state_error:.3f}",
                transform=ax.transAxes, fontsize=6, va="top",
                bbox=dict(fc="white", ec="none", alpha=0.8))
        if i == 0:
            ax.legend(fontsize=6, loc="lower right")

    fig.suptitle("Control Loop Step Response — 10% Setpoint Change",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def plot_disturbance(results: dict, output: Path) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, (name, res) in enumerate(results.items()):
        ax = axes[i]
        color = LOOP_COLORS.get(name, "#333")
        t = res.time_s

        ax.plot(t, res.setpoint, "k--", lw=1.0, label="SP")
        ax.plot(t, res.pv, color=color, lw=1.2, label="PV")
        ax.axvline(x=400.0, color="red", ls=":", lw=0.8, alpha=0.5, label="disturbance")

        ax.set_title(f"{res.tag}\n{res.loop_name}", fontsize=8, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel("PV", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.text(0.02, 0.95,
                f"IAE={res.iae:.2f}\nSS err={res.steady_state_error:.4f}",
                transform=ax.transAxes, fontsize=6, va="top",
                bbox=dict(fc="white", ec="none", alpha=0.8))
        if i == 0:
            ax.legend(fontsize=6, loc="lower right")

    fig.suptitle("Disturbance Rejection — Step Disturbance at t=400s",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def plot_tuning_sensitivity(loops: dict, output: Path) -> Path:
    """Sensitivity analysis for Kp on the 3 most impactful loops."""
    selected = ["electrolyte_temp", "carburize_temp", "tempering_temp"]
    fig, axes = plt.subplots(len(selected), 3, figsize=(15, 4 * len(selected)))

    for row, name in enumerate(selected):
        cfg = loops[name]
        sens = tuning_sensitivity(name, cfg, param_name="Kp",
                                  scale_factors=np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]))

        for col, (metric, label) in enumerate([
            ("iae", "IAE"),
            ("overshoot_pct", "Overshoot (%)"),
            ("settling_time_s", "Settling Time (s)"),
        ]):
            ax = axes[row, col]
            ax.plot(sens["scale_factors"], sens[metric], "o-", color=LOOP_COLORS.get(name, "#333"), lw=1.5)
            ax.axvline(x=1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
            ax.set_xlabel("Kp scale factor", fontsize=7)
            ax.set_ylabel(label, fontsize=7)
            ax.set_title(f"{name} — {label}", fontsize=8)
            ax.tick_params(labelsize=6)

    fig.suptitle("Tuning Sensitivity — Kp Sweep (baseline × factor)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def plot_loop_summary(summary: list, results: dict, output: Path) -> Path:
    """Summary dashboard: table + key metrics bar chart."""
    fig, (ax_table, ax_bar) = plt.subplots(1, 2, figsize=(18, 8),
                                           gridspec_kw={"width_ratios": [1.2, 1]})

    # Table
    ax_table.axis("off")
    col_labels = ["Loop", "Tag", "Type", "Kp", "Ki", "Kd", "SP", "MV range"]
    table_data = []
    for row in summary:
        if row["type"] == "cascade":
            table_data.append([
                row["loop"], row["tag"], "cascade",
                f'{row["outer_Kp"]}/{row["inner_Kp"]}',
                f'{row["outer_Ki"]}/{row["inner_Ki"]}',
                f'{row["outer_Kd"]}/{row["inner_Kd"]}',
                str(row["setpoint"]),
                row["MV_range"],
            ])
        elif row["type"] == "open-loop":
            table_data.append([
                row["loop"], row["tag"], "open-loop", "—", "—", "—",
                row["setpoint"], row["MV_range"],
            ])
        else:
            table_data.append([
                row["loop"], row["tag"], "PID",
                f'{row["Kp"]:.1f}', f'{row["Ki"]:.3f}', f'{row["Kd"]:.3f}',
                f'{row["setpoint"]:.1f}', row["MV_range"],
            ])

    table = ax_table.table(cellText=table_data, colLabels=col_labels,
                           loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.4)
    ax_table.set_title("Control Loop Summary", fontsize=10, fontweight="bold", pad=20)

    # Bar chart of key metrics
    names = []
    iae_vals = []
    os_vals = []
    for name, res in results.items():
        if name == "quench_timing":
            continue
        names.append(res.tag)
        iae_vals.append(res.iae)
        os_vals.append(res.overshoot_pct)

    x = np.arange(len(names))
    width = 0.35
    ax_bar.bar(x - width / 2, iae_vals, width, label="IAE", color="#1f77b4", alpha=0.8)
    ax_bar.bar(x + width / 2, os_vals, width, label="Overshoot %", color="#ff7f0e", alpha=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax_bar.set_ylabel("Value", fontsize=8)
    ax_bar.set_title("Loop Performance Metrics", fontsize=10, fontweight="bold")
    ax_bar.legend(fontsize=7)
    ax_bar.tick_params(labelsize=7)

    fig.suptitle("Process Control — Loop Summary & Performance",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PROCESS CONTROL SIMULATION — PID LOOPS FOR THE P&ID")
    print("=" * 72)

    loops = default_loops()
    print(f"  Loaded {len(loops)} control loops")

    # Step response
    print("\n  Step response simulation...")
    step_results = run_step_response(loops)
    p1 = plot_step_response(step_results, FIG_DIR / "control_step_response.png")
    print(f"  ✅ Saved: {p1}")

    # Disturbance rejection
    print("\n  Disturbance rejection simulation...")
    dist_results = run_disturbance(loops)
    p2 = plot_disturbance(dist_results, FIG_DIR / "control_disturbance.png")
    print(f"  ✅ Saved: {p2}")

    # Tuning sensitivity
    print("\n  Tuning sensitivity analysis...")
    p3 = plot_tuning_sensitivity(loops, FIG_DIR / "control_tuning_sensitivity.png")
    print(f"  ✅ Saved: {p3}")

    # Summary
    print("\n  Loop summary dashboard...")
    summary = loop_summary_table()
    p4 = plot_loop_summary(summary, step_results, FIG_DIR / "control_loop_summary.png")
    print(f"  ✅ Saved: {p4}")

    # Build report
    report = {
        "title": "Process control model — PID loops for the P&ID",
        "date": datetime.now().isoformat(),
        "figures": [str(p1), str(p2), str(p3), str(p4)],
        "n_loops": len(loops),
        "loops": [],
    }
    for name, cfg in loops.items():
        res = step_results[name]
        entry = {
            "name": name,
            "tag": cfg.get("tag", ""),
            "description": cfg["description"],
            "type": "cascade" if cfg.get("cascade") else cfg["plant"].get("type", "PID"),
            "setpoint": float(res.setpoint[-1]) if len(res.setpoint) > 0 else None,
            "settling_time_s": res.settling_time_s,
            "overshoot_pct": res.overshoot_pct,
            "steady_state_error": res.steady_state_error,
            "iae": res.iae,
        }
        if cfg.get("cascade"):
            entry["outer_Kp"] = cfg["outer_pid"].Kp
            entry["inner_Kp"] = cfg["inner_pid"].Kp
        elif cfg["plant"].get("type") != "lookup":
            entry["Kp"] = cfg["pid"].Kp
            entry["Ki"] = cfg["pid"].Ki
            entry["Kd"] = cfg["pid"].Kd
        report["loops"].append(entry)

    report["summary_table"] = summary

    report_path = DATA_DIR / "process_control_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  ✅ Saved report: {report_path}")

    # Print summary
    print("\n  Loop Performance Summary:")
    print(f"  {'Loop':<22} {'Tag':<10} {'Type':<10} {'SP':>8} {'ts(s)':>8} {'OS%':>6} {'SS err':>8} {'IAE':>8}")
    print("  " + "-" * 82)
    for name, cfg in loops.items():
        res = step_results[name]
        tp = "cascade" if cfg.get("cascade") else cfg["plant"].get("type", "PID")
        sp_val = f"{res.setpoint[-1]:.1f}" if len(res.setpoint) > 0 else "N/A"
        print(f"  {name:<22} {res.tag:<10} {tp:<10} {sp_val:>8} "
              f"{res.settling_time_s:>8.1f} {res.overshoot_pct:>6.1f} "
              f"{res.steady_state_error:>8.4f} {res.iae:>8.1f}")

    print("\n✅ Process control driver complete!")


if __name__ == "__main__":
    main()
