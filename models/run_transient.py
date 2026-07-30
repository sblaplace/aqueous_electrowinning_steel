"""
Driver for transient process model — startup, shutdown, upsets.

Generates:
* outputs/startup_sequence.png
* outputs/shutdown_sequence.png
* outputs/upset_power_loss.png
* outputs/upset_ph_excursion.png
* experiments/data/transient_report.json

Usage:
    python -m models.run_transient
    python -m models.run_transient --startup-duration 180 --dt 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.transient import (
    TransientConfig,
    UpsetType,
    simulate_startup,
    simulate_shutdown,
    simulate_upset,
    recovery_time,
    damage_assessment,
)

FIG_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "experiments" / "data"


def _plot_transient(result, fig_path: Path, title: str) -> Path:
    """Multi-panel plot of a transient result."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    t = result.time_min

    # Electrolyte temperature
    ax = axes[0, 0]
    ax.plot(t, result.electrolyte_temp_C, color="#e41a1c", lw=1.5)
    ax.axhline(60, color="#777", ls=":", label="operating")
    ax.axhline(85, color="#e41a1c", ls="--", alpha=0.5, label="max safe")
    ax.set_ylabel("Electrolyte T (°C)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

    # pH
    ax = axes[0, 1]
    ax.plot(t, result.electrolyte_pH, color="#377eb8", lw=1.5)
    ax.axhline(2.5, color="#777", ls=":", label="target pH")
    ax.axhline(3.8, color="#e41a1c", ls="--", alpha=0.5, label="Fe(OH)₃ threshold")
    ax.set_ylabel("Electrolyte pH")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

    # Current density
    ax = axes[1, 0]
    ax.plot(t, result.current_density_mA_cm2, color="#4daf4a", lw=1.5)
    ax.set_ylabel("Current density (mA/cm²)")
    ax.grid(alpha=0.2)

    # Furnace temp
    ax = axes[1, 1]
    ax.plot(t, result.furnace_temp_C, color="#ff7f00", lw=1.5)
    ax.axhline(900, color="#777", ls=":", label="target")
    ax.set_ylabel("Furnace T (°C)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

    # Deposit quality + CE
    ax = axes[2, 0]
    ax.plot(t, result.deposit_quality, color="#984ea3", lw=1.5, label="quality")
    ax.plot(t, result.ce_fraction, color="#a65628", lw=1.2, ls="--", label="CE")
    ax.axhline(0.7, color="#e41a1c", ls=":", alpha=0.5, label="quality threshold")
    ax.set_ylabel("Quality / CE (fraction)")
    ax.set_xlabel("Time (min)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

    # O2 probe + quench
    ax = axes[2, 1]
    o2_log = np.clip(result.o2_probe_ppm, 1, None)
    ax.semilogy(t, o2_log, color="#e41a1c", lw=1.2, label="O₂ probe (ppm)")
    ax.axhline(50, color="#4daf4a", ls=":", label="target O₂")
    ax2 = ax.twinx()
    ax2.plot(t, result.quench_temp_C, color="#377eb8", lw=1.2, ls="--", label="quench T")
    ax2.set_ylabel("Quench T (°C)", color="#377eb8")
    ax.set_ylabel("O₂ probe (ppm)")
    ax.set_xlabel("Time (min)")
    ax.legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.2)

    fig.suptitle(title, fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    return fig_path


def main(
    argv=None,
    startup_duration=None,
    shutdown_duration=None,
    upset_duration=None,
    dt=None,
):
    """Main driver — CLI or programmatic entry point."""
    if startup_duration is None and argv is None:
        parser = argparse.ArgumentParser(description="Transient process model driver")
        parser.add_argument("--startup-duration", type=float, default=120.0)
        parser.add_argument("--shutdown-duration", type=float, default=90.0)
        parser.add_argument("--upset-duration", type=float, default=60.0)
        parser.add_argument("--dt", type=float, default=1.0)
        parsed = parser.parse_args(argv)
        startup_duration = parsed.startup_duration
        shutdown_duration = parsed.shutdown_duration
        upset_duration = parsed.upset_duration
        dt = parsed.dt
    else:
        startup_duration = startup_duration or 120.0
        shutdown_duration = shutdown_duration or 90.0
        upset_duration = upset_duration or 60.0
        dt = dt or 1.0

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("TRANSIENT PROCESS MODEL — STARTUP / SHUTDOWN / UPSETS")
    print("=" * 72)

    # ── Startup ──
    cfg_startup = TransientConfig(startup_duration_min=startup_duration)
    startup = simulate_startup(config=cfg_startup, dt=dt)
    fig1 = _plot_transient(startup, FIG_DIR / "startup_sequence.png",
                           "Plant Startup — Cold to Operating")
    print(f"\n✅ Startup: {startup.summary()}")
    print(f"   Figure: {fig1}")

    # ── Shutdown ──
    cfg_shutdown = TransientConfig(shutdown_duration_min=shutdown_duration)
    shutdown = simulate_shutdown(config=cfg_shutdown, dt=dt)
    fig2 = _plot_transient(shutdown, FIG_DIR / "shutdown_sequence.png",
                           "Controlled Shutdown — Operating to Cold")
    print(f"\n✅ Shutdown: {shutdown.summary()}")
    print(f"   Figure: {fig2}")

    # ── Upset: Power loss ──
    cfg_upset = TransientConfig(upset_duration_min=upset_duration)
    power_upset = simulate_upset(
        config=cfg_upset,
        upset_type=UpsetType.POWER_INTERRUPTION,
        duration=upset_duration,
        dt=dt,
    )
    fig3 = _plot_transient(power_upset, FIG_DIR / "upset_power_loss.png",
                           "Upset: Complete Power Interruption")
    dmg_power = damage_assessment(power_upset)
    print(f"\n✅ Power loss upset: {dmg_power}")
    print(f"   Figure: {fig3}")

    # ── Upset: pH excursion ──
    ph_upset = simulate_upset(
        config=cfg_upset,
        upset_type=UpsetType.PH_EXCURSION,
        duration=upset_duration,
        dt=dt,
    )
    fig4 = _plot_transient(ph_upset, FIG_DIR / "upset_ph_excursion.png",
                           "Upset: pH Excursion (Pump Failure)")
    dmg_ph = damage_assessment(ph_upset)
    print(f"\n✅ pH excursion: {dmg_ph}")
    print(f"   Figure: {fig4}")

    # ── Remaining upsets for report ──
    upset_results = {}
    for utype in UpsetType:
        if utype in (UpsetType.POWER_INTERRUPTION, UpsetType.PH_EXCURSION):
            continue
        result = simulate_upset(config=cfg_upset, upset_type=utype,
                                duration=upset_duration, dt=dt)
        dmg = damage_assessment(result)
        upset_results[utype.value] = dmg
        print(f"\n✅ {utype.value}: {dmg}")

    # ── Report ──
    report = {
        "title": "Transient process model — screening",
        "startup": startup.summary(),
        "shutdown": shutdown.summary(),
        "upsets": {
            "power_interruption": dmg_power,
            "ph_excursion": dmg_ph,
            **upset_results,
        },
        "figures": [str(fig1), str(fig2), str(fig3), str(fig4)],
        "model_notes": [
            "First-order thermal lags (exponential approach to setpoint)",
            "Current ramp: linear ramp over configurable time",
            "Quality: 1.0 = perfect, degrades under pH/temp/current faults",
            "CE: Butler-Volmer screening with pH and temperature penalties",
            "O2 probe: 15-min time constant for atmosphere change",
            "6 upset scenarios: power, pH, temperature, gas, current, rectifier",
            "Screening model — not for PLC logic or real-time control",
        ],
    }
    report_path = DATA_DIR / "transient_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  ✅ Report: {report_path}")
    print("\n✅ Transient driver complete!")

    return report


if __name__ == "__main__":
    main()
