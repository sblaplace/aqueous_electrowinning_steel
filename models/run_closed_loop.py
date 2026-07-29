"""Generate the synthetic Phase IV closed-loop and durability example."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .anode import AnodeKinetics, NICO_SPINEL
from .closed_loop import (
    AnodeDurabilityParams,
    ClosedLoopParams,
    PhaseIVClosedLoop,
    PhaseIVOperatingPoint,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "data"
FIGURES = ROOT / "docs" / "figures"


def build_example() -> PhaseIVClosedLoop:
    """Return a reproducible alkaline screening case (not wet-lab data)."""
    anode = AnodeKinetics(
        material=NICO_SPINEL,
        electrolyte_type="alkaline",
        pH=14.0,
        electrolyte_resistivity_ohm_m2=2.0e-4,
    )
    loop = ClosedLoopParams(
        volume_L=1000.0,
        feed_flow_L_hr=20.0,
        purge_flow_L_hr=20.0,
        fe_feed_M=1.25,
        ligand_feed_M=1.50,
        fe_initial_M=1.0,
        ligand_initial_M=1.30,
        impurity_feed_M=2e-4,
        impurity_limit_M=0.01,
    )
    durability = AnodeDurabilityParams(
        coating_loading_g_m2=12.0,
        base_wear_mg_per_kAh=0.35,
    )
    operating = PhaseIVOperatingPoint(
        current_density_mA_cm2=100.0,
        anode_area_m2=1.0,
        current_efficiency=0.95,
    )
    return PhaseIVClosedLoop(anode, loop, durability, operating)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    model = build_example()
    result = model.simulate(duration_hr=4000.0, dt_hr=2.0)
    metrics = model.process_metrics(result)

    frame = pd.DataFrame(result.as_columns())
    frame["quality_flags"] = [";".join(x) for x in result.flags]
    csv_path = DATA / "synthetic_closed_loop.csv"
    frame.to_csv(csv_path, index=False)

    report = {
        "provenance": "Synthetic Phase IV screening output; not wet-lab data.",
        "model_scope": (
            "Constant-volume ideal CSTR plus empirical charge-throughput anode wear. "
            "Calibrate wear, precipitation, speciation and impurity parameters before design use."
        ),
        "summary": result.summary(),
        "process_metrics": metrics,
    }
    report_path = DATA / "closed_loop_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(result.time_hr, result.fe_M, label="Fe")
    axes[0, 0].plot(result.time_hr, result.ligand_M, label="Ligand")
    axes[0, 0].set(ylabel="Concentration (mol/L)", title="Closed-loop inventory")
    axes[0, 0].legend()
    axes[0, 1].plot(result.time_hr, 100 * result.coating_remaining_fraction)
    axes[0, 1].axhline(20, color="tab:red", linestyle="--", label="EOL criterion")
    axes[0, 1].set(ylabel="Coating remaining (%)", title="Anode durability")
    axes[0, 1].legend()
    axes[1, 0].plot(result.time_hr, result.cell_voltage_V)
    axes[1, 0].set(xlabel="Time (hr)", ylabel="Cell voltage (V)", title="Voltage drift")
    axes[1, 1].plot(result.time_hr, result.impurity_M * 1000)
    axes[1, 1].axhline(model.loop.impurity_limit_M * 1000, color="tab:red", linestyle="--")
    axes[1, 1].set(xlabel="Time (hr)", ylabel="Impurity (mmol/L)", title="Impurity buildup")
    fig.suptitle("Synthetic Phase IV closed-loop screen — not experimental data")
    fig.savefig(FIGURES / "closed_loop_durability.png", dpi=180)
    plt.close(fig)

    print(json.dumps(report, indent=2))
    print(f"Wrote {csv_path.relative_to(ROOT)}, {report_path.relative_to(ROOT)}, and figure")


if __name__ == "__main__":
    main()
