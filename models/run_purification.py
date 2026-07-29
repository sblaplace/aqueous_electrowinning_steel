"""Generate synthetic purification example: figure + report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .purification import (
    CementationModel,
    PurificationFeedstock,
    PurificationModel,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "data"
FIGURES = ROOT / "docs" / "figures"


def build_example() -> PurificationModel:
    """Return a reproducible screening case (not wet-lab data)."""
    feed = PurificationFeedstock(
        cu_M=5.0e-4,     # ~320 ppm relative to Fe
        ni_M=3.0e-4,
        zn_M=2.0e-4,
        fe2_M=1.0,
        fe3_M=0.05,
        pH=2.0,
        temperature_C=50.0,
        volume_L=1000.0,
    )
    return PurificationModel(feedstock=feed)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    model = build_example()
    result = model.simulate()
    summary = result.summary()

    # ── Cementation kinetics figure ───────────────────────────────────
    cem_model = CementationModel(feedstock=model.feedstock)
    cem = cem_model.simulate(duration_hr=8.0, dt_hr=0.05)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    # (0,0) Cementation kinetics
    ax = axes[0, 0]
    ax.plot(cem.time_hr, cem.cu_M * 1e6, label="Cu²⁺", linewidth=2)
    ax.plot(cem.time_hr, cem.ni_M * 1e6, label="Ni²⁺", linewidth=2)
    ax.plot(cem.time_hr, cem.zn_M * 1e6, label="Zn²⁺", linewidth=2)
    ax.set(xlabel="Time (hr)", ylabel="Concentration (µmol/L)",
           title="Cementation kinetics on Fe powder")
    ax.legend()
    ax.set_yscale("log")

    # (0,1) Removal fraction vs temperature
    ax = axes[0, 1]
    temps = np.linspace(20, 80, 20)
    cu_fracs = []
    for T in temps:
        f = PurificationFeedstock(temperature_C=T)
        r = CementationModel(feedstock=f).simulate(4.0)
        cu_fracs.append(r.removal_fractions()["cu"])
    ax.plot(temps, cu_fracs, color="tab:blue", linewidth=2)
    ax.set(xlabel="Temperature (°C)", ylabel="Cu removal fraction",
           title="Cementation: Cu removal vs temperature (4 hr)")
    ax.axhline(0.99, color="tab:red", linestyle="--", label="99% target")
    ax.legend()

    # (1,0) Stage-by-stage removal
    ax = axes[1, 0]
    stages = ["Feed", "After\nCementation", "After\nHydrolysis", "After\nElectrowinning", "After\nIX"]
    cu_levels = [
        result.cu_initial_M * 1e6,
        result.cementation.cu_M[-1] * 1e6,
        (result.cementation.cu_M[-1] - result.hydrolysis.cu_removed_M) * 1e6,
        result.electrowinning.cu_remaining_M * 1e6,
        result.cu_final_M * 1e6,
    ]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
    bars = ax.bar(stages, cu_levels, color=colors, edgecolor="black")
    ax.axhline(0.01 * model.feedstock.fe2_M * 1e6 * 63.546 / 55.845,
               color="tab:red", linestyle="--", label="0.01 wt% limit")
    ax.set(ylabel="Cu²⁺ (µmol/L)", title="Stage-by-stage Cu removal")
    ax.legend()
    ax.set_yscale("log")

    # (1,1) Cost breakdown
    ax = axes[1, 1]
    cost_labels = list(result.stage_costs.keys())
    cost_values = list(result.stage_costs.values())
    ax.barh(cost_labels, cost_values, color="tab:gray", edgecolor="black")
    ax.set(xlabel="Cost (USD)", title="Purification cost breakdown")
    ax.invert_yaxis()

    fig.suptitle("Synthetic feedstock purification screen — not experimental data")
    fig.savefig(FIGURES / "purification_efficiency.png", dpi=180)
    plt.close(fig)

    # ── Report ────────────────────────────────────────────────────────
    report = {
        "provenance": "Synthetic purification screening output; not wet-lab data.",
        "model_scope": (
            "Four-stage screening model: cementation (first-order), hydrolysis "
            "(Fe³⁺ precipitation + co-removal), selective electrowinning "
            "(Butler-Volmer), ion exchange (capacity model). "
            "Calibrate rate constants, co-precipitation fractions and resin "
            "isotherms before design use."
        ),
        "summary": summary,
        "cementation_removal_fractions": result.cementation.removal_fractions(),
        "acceptance": {
            "cu_below_0_01_wt_pct": bool(result.cu_meets_spec()),
            "cost_per_tonne_computed": bool(result.total_cost_per_t_fe > 0),
        },
    }
    report_path = DATA / "purification_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nWrote {FIGURES.relative_to(ROOT)}/purification_efficiency.png")
    print(f"Wrote {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
